"""N32.6.03: 2-seed exploration replication of joint Q1+Q2+Q3 ModalityEdgeHop (<= 460 CPU).

Pre-registered: plans/reference/DESIGN-N32-6-03_20261125.md (D-20261125-548).
Replicates N32.6.02 (seed 20261120) on two unread exploration seeds (20261121, 20261122)
with identical architecture, joint schedule, and zero confirm-seed access.

Usage:
  python scripts/training/n32_6_03_r1_joint_rep.py --smoke
  python scripts/training/n32_6_03_r1_joint_rep.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from scripts.training.n31_guard import preflight  # noqa: E402
from scripts.training.n32_5_01_q3_gru_learnability import (  # noqa: E402
    CapExceeded,
    NonFiniteLoss,
    int_seed,
)
from scripts.training.n32_6_01_r1_edge_hop import (  # noqa: E402
    GATE_MIN,
    GATE_POS,
    KINDS,
    LR,
    Q3_CKPTS,
    WD,
    _lr_at,
    build_data,
    evaluate_kind,
    evaluate_mixed_equivalence,
    make_column,
    prep,
)
from scripts.training.n32_6_02_r1_joint_diag import (  # noqa: E402
    JOINT_BATCH,
    LOSS_WEIGHTS,
    RT_Q3_REF,
    SMOKE_STEPS,
    STEPS,
)
from taiji.modality_edge_hop import ARCH_VERSION  # noqa: E402

CARD = "N32.6.03"
REP_SEEDS: tuple[int, ...] = (20261121, 20261122)
CAP_CPU = 460.0
PROJECTION_MARGIN = 1.08
Q3_2K_GATE = 0.75

PRIOR_SEED_20261120: dict[str, float] = {
    "q1": 0.9959999918937683,
    "q2": 0.9872499704360962,
    "q3": 0.9777500033378601,
    "q3_2k": 0.8494166731834412,
}

SMOKE_REPORT = ROOT / "reports" / "n32_6_03_smoke.json"
FULL_REPORT = ROOT / "reports" / "n32_6_03_joint_rep.json"


def train_joint_seed(
    seed: int,
    data: dict[str, Any],
    deadline: float,
    steps: int = STEPS,
    *,
    skip_eval: bool = False,
) -> dict[str, Any]:
    """Train one task-agnostic ModalityEdgeHop jointly on Q1+Q2+Q3 for one seed."""
    model = make_column(seed, "S_joint")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gens = {
        k: torch.Generator().manual_seed(int_seed(CARD, seed, "S_joint", k, "batches"))
        for k in KINDS
    }
    ckpts: dict[str, dict[str, float]] = {}
    t0 = time.process_time()
    eval_cpu = 0.0

    model.train()
    for i in range(steps):
        if time.process_time() > deadline:
            raise CapExceeded(f"S_joint seed={seed} step {i}")
        lr = _lr_at(i, steps)
        for g in opt.param_groups:
            g["lr"] = lr

        loss: torch.Tensor | None = None
        for k in KINDS:
            dk = data["train"][k]
            idx = torch.randint(0, dk["x"].shape[0], (JOINT_BATCH[k],), generator=gens[k])
            logits = model(prep(dk["x"][idx], data), dk["sm"][idx], dk["qm"][idx])
            term = LOSS_WEIGHTS[k] * F.cross_entropy(logits, dk["y"][idx])
            loss = term if loss is None else loss + term
        assert loss is not None
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"S_joint seed={seed} step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step_num = i + 1
        if not skip_eval and step_num in Q3_CKPTS:
            e0 = time.process_time()
            ckpts[str(step_num)] = {k: evaluate_kind(model, data, "val", k)["acc"] for k in KINDS}
            eval_cpu += time.process_time() - e0
            model.train()

    total_cpu = time.process_time() - t0
    train_loop_cpu = max(total_cpu - eval_cpu, 0.0)
    if skip_eval:
        return {"train_loop_cpu": train_loop_cpu, "cpu": round(total_cpu, 2)}

    e1 = time.process_time()
    final_eval = {k: evaluate_kind(model, data, "val", k, detail=True) for k in KINDS}
    mixed = evaluate_mixed_equivalence(model, data)
    eval_cpu += time.process_time() - e1

    final_accs = {k: float(final_eval[k]["acc"]) for k in KINDS}
    min_acc = min(final_accs.values())
    q3_2k = float(ckpts["2000"]["q3"])
    q3_vs_rt = {
        str(c): round(float(ckpts[str(c)]["q3"]) - float(RT_Q3_REF[str(c)]), 6) for c in Q3_CKPTS
    }
    return {
        "seed": seed,
        "arm": "S_joint",
        "arch_version": ARCH_VERSION,
        "n_params": model.n_params(),
        "ckpts": ckpts,
        "q3_2k": q3_2k,
        "q3_vs_rt_ref_diffs": q3_vs_rt,
        "q3_vs_rt_ref_mean": round(sum(q3_vs_rt.values()) / len(q3_vs_rt), 6),
        "final": final_eval,
        "final_accs": final_accs,
        "min_acc": round(min_acc, 6),
        "mixed_equivalence": mixed,
        "pass_min": bool(mixed["exact"] and min_acc >= GATE_MIN),
        "pass_pos": bool(mixed["exact"] and min_acc >= GATE_POS and q3_2k >= Q3_2K_GATE),
        "train_loop_cpu": round(train_loop_cpu, 3),
        "eval_cpu": round(eval_cpu, 3),
        "cpu": round(time.process_time() - t0, 2),
    }


def smoke(data: dict[str, Any], started: float, encode_cpu: float) -> dict[str, Any]:
    """Run short smoke timing pass and project total 2-seed CPU cost."""
    deadline = started + CAP_CPU
    res = train_joint_seed(REP_SEEDS[0], data, deadline, steps=SMOKE_STEPS, skip_eval=True)
    probe = make_column(REP_SEEDS[0], "S_joint")
    e0 = time.process_time()
    for k in KINDS:
        evaluate_kind(probe, data, "val", k, detail=True)
    eval_pass_cpu = time.process_time() - e0

    per_step = res["train_loop_cpu"] / SMOKE_STEPS
    proj_train = per_step * STEPS
    proj_eval = eval_pass_cpu * (len(Q3_CKPTS) + 1.5)
    proj_2seeds = round(
        encode_cpu + len(REP_SEEDS) * (proj_train + proj_eval) * PROJECTION_MARGIN, 2
    )
    smoke_cpu = round(time.process_time() - started, 2)
    planned = "rep_2seeds" if smoke_cpu + proj_2seeds <= CAP_CPU else "budget_insufficient"
    return {
        "encode_cpu": encode_cpu,
        "smoke_cpu": smoke_cpu,
        "per_step_cpu": per_step,
        "eval_pass_cpu": round(eval_pass_cpu, 3),
        "proj_train_cpu_per_seed": round(proj_train, 2),
        "projected_2seeds": proj_2seeds,
        "margin": PROJECTION_MARGIN,
        "cap_cpu": CAP_CPU,
        "planned": planned,
    }


def _aggregate_3seeds(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine N32.6.02 seed 20261120 with the two N32.6.03 replication seeds."""
    rows: dict[str, dict[str, float]] = {
        "20261120": dict(PRIOR_SEED_20261120),
    }
    for seed_str, res in per_seed.items():
        rows[seed_str] = {
            "q1": float(res["final_accs"]["q1"]),
            "q2": float(res["final_accs"]["q2"]),
            "q3": float(res["final_accs"]["q3"]),
            "q3_2k": float(res["q3_2k"]),
        }
    summary: dict[str, Any] = {"seeds": list(rows.keys()), "per_seed_accs": rows}
    for key in ("q1", "q2", "q3", "q3_2k"):
        vals = [row[key] for row in rows.values()]
        summary[key] = {
            "min": round(min(vals), 6),
            "mean": round(sum(vals) / len(vals), 6),
            "max": round(max(vals), 6),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design = preflight(CARD, list(REP_SEEDS), record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    data = build_data()
    encode_cpu = round(time.process_time() - started, 2)

    report: dict[str, Any] = {
        "schema": "taiji-n32-6-03-v1",
        "card": CARD,
        "arch_version": ARCH_VERSION,
        "rep_seeds": list(REP_SEEDS),
        "prior_seed_20261120": PRIOR_SEED_20261120,
        "steps": STEPS,
        "joint_batch": JOINT_BATCH,
        "loss_weights": LOSS_WEIGHTS,
        "ckpts": list(Q3_CKPTS),
        "design": [design],
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "data": {
            k: {
                "train_rows": int(data["train"][k]["y"].numel()),
                "val_rows": int(data["val"][k]["y"].numel()),
            }
            for k in KINDS
        }
        | {"q3_digests": data["q3_digests"]},
        "gates": {"min": GATE_MIN, "pos": GATE_POS, "q3_2k": Q3_2K_GATE},
        "encode_cpu": encode_cpu,
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "exploratory": True},
    }

    status = 0
    if args.smoke:
        report["projection"] = smoke(data, started, encode_cpu)
        out = SMOKE_REPORT
        if report["projection"]["planned"] == "budget_insufficient":
            status = 3
    else:
        smoke_doc = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
        proj = smoke_doc["projection"]
        if proj["planned"] != "rep_2seeds":
            report.update({"exit": "budget_insufficient", "planned": proj["planned"]})
            status = 3
        else:
            budget = CAP_CPU - float(proj["smoke_cpu"])
            deadline = started + budget
            per_seed: dict[str, dict[str, Any]] = {}
            try:
                for seed in REP_SEEDS:
                    per_seed[str(seed)] = train_joint_seed(seed, data, deadline, steps=STEPS)
                report["planned"] = "rep_2seeds"
                report["per_seed"] = per_seed
                report["all_3seeds"] = _aggregate_3seeds(per_seed)
                if all(r["pass_pos"] for r in per_seed.values()):
                    report["exit"] = "joint_3seed_positive"
                elif all(r["pass_min"] for r in per_seed.values()):
                    report["exit"] = "joint_3seed_partial"
                else:
                    report["exit"] = "joint_3seed_unstable"
            except CapExceeded as exc:
                report.update(
                    {
                        "exit": "engineering_failure",
                        "reason": f"cap exceeded: {exc}",
                        "per_seed": per_seed,
                    }
                )
            except NonFiniteLoss as exc:
                report.update(
                    {
                        "exit": "engineering_failure",
                        "reason": f"non-finite: {exc}",
                        "per_seed": per_seed,
                    }
                )
            if report.get("exit") in ("engineering_failure", "budget_insufficient"):
                status = 3
        out = FULL_REPORT

    report["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        k: report.get(k) for k in ("exit", "all_3seeds", "cost") if report.get(k) is not None
    }
    if "per_seed" in report:
        summary["per_seed"] = {
            s: {
                "final_accs": r["final_accs"],
                "q3_2k": r["q3_2k"],
                "q3_vs_rt_ref_mean": r["q3_vs_rt_ref_mean"],
                "mixed_exact": r["mixed_equivalence"]["exact"],
                "cpu": r["cpu"],
            }
            for s, r in report["per_seed"].items()
        }
    if args.smoke:
        summary["projection"] = report["projection"]
    print(json.dumps(summary, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

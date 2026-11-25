"""N32.6.02: Joint interleaved Q1+Q2+Q3 diagnostic on task-agnostic ModalityEdgeHop (<= 300 CPU).

Pre-registered: plans/reference/DESIGN-N32-6-02_20261125.md (D-20261125-546).
Isolates whether N32.6.01 S's 0.655 plateau came from sequential Q1 lock-in + replay
or from shared single-column capacity limits.

Usage:
  python scripts/training/n32_6_02_r1_joint_diag.py --smoke
  python scripts/training/n32_6_02_r1_joint_diag.py
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
    SCREEN_SEED,
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
from taiji.modality_edge_hop import ARCH_VERSION  # noqa: E402

CARD = "N32.6.02"
STEPS = 6000
JOINT_BATCH = {"q3": 256, "q2": 128, "q1": 64}
LOSS_WEIGHTS = {"q3": 1.0, "q2": 0.5, "q1": 0.25}
SMOKE_STEPS = 40
CAP_CPU = 300.0
PROJECTION_MARGIN = 1.10

# Reference checkpoints from N32.6.01 cold-start RT column on Q3 (report-only)
RT_Q3_REF = {
    "1000": 0.6004999876022339,
    "2000": 0.6004999876022339,
    "4000": 0.8205000162124634,
    "6000": 0.95333331823349,
}

REPORTS = ROOT / "reports"
SMOKE_REPORT = REPORTS / "n32_6_02_smoke.json"
FULL_REPORT = REPORTS / "n32_6_02_joint_diag.json"


def train_joint(
    seed: int,
    data: dict[str, Any],
    deadline: float,
    steps: int = STEPS,
    *,
    skip_eval: bool = False,
) -> dict[str, Any]:
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
            raise CapExceeded(f"S_joint step {i}")
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
            raise NonFiniteLoss(f"S_joint step {i}")
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
    if not mixed["exact"]:
        exit_label = "engineering_failure"
    elif min_acc >= GATE_POS:
        exit_label = "joint_capacity_sufficient"
    elif min_acc >= GATE_MIN:
        exit_label = "joint_partial"
    else:
        exit_label = "joint_interference_persists"

    q3_vs_rt = {
        str(c): round(float(ckpts[str(c)]["q3"]) - float(RT_Q3_REF[str(c)]), 6) for c in Q3_CKPTS
    }
    return {
        "arm": "S_joint",
        "arch_version": ARCH_VERSION,
        "n_params": model.n_params(),
        "ckpts": ckpts,
        "q3_vs_rt_ref_diffs": q3_vs_rt,
        "q3_vs_rt_ref_mean": round(sum(q3_vs_rt.values()) / len(q3_vs_rt), 6),
        "final": final_eval,
        "final_accs": final_accs,
        "min_acc": round(min_acc, 6),
        "mixed_equivalence": mixed,
        "exit": exit_label,
        "train_loop_cpu": round(train_loop_cpu, 3),
        "eval_cpu": round(eval_cpu, 3),
        "cpu": round(time.process_time() - t0, 2),
    }


def smoke(data: dict[str, Any], started: float, encode_cpu: float) -> dict[str, Any]:
    deadline = started + CAP_CPU
    res = train_joint(SCREEN_SEED, data, deadline, steps=SMOKE_STEPS, skip_eval=True)
    probe = make_column(SCREEN_SEED, "S_joint")
    e0 = time.process_time()
    for k in KINDS:
        evaluate_kind(probe, data, "val", k, detail=True)
    eval_pass_cpu = time.process_time() - e0

    per_step = res["train_loop_cpu"] / SMOKE_STEPS
    proj_train = per_step * STEPS
    proj_eval = eval_pass_cpu * (len(Q3_CKPTS) + 1.5)
    proj_1 = round(encode_cpu + (proj_train + proj_eval) * PROJECTION_MARGIN, 2)
    smoke_cpu = round(time.process_time() - started, 2)
    planned = "joint_1seed" if smoke_cpu + proj_1 <= CAP_CPU else "budget_insufficient"
    return {
        "encode_cpu": encode_cpu,
        "smoke_cpu": smoke_cpu,
        "per_step_cpu": per_step,
        "eval_pass_cpu": round(eval_pass_cpu, 3),
        "proj_train_cpu": round(proj_train, 2),
        "projected_1seed": proj_1,
        "margin": PROJECTION_MARGIN,
        "cap_cpu": CAP_CPU,
        "planned": planned,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design = preflight(CARD, [SCREEN_SEED], record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    data = build_data()
    encode_cpu = round(time.process_time() - started, 2)

    report: dict[str, Any] = {
        "schema": "taiji-n32-6-02-v1",
        "card": CARD,
        "arch_version": ARCH_VERSION,
        "seed": SCREEN_SEED,
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
        "gates": {"min": GATE_MIN, "pos": GATE_POS},
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
        if proj["planned"] != "joint_1seed":
            report.update({"exit": "budget_insufficient", "planned": proj["planned"]})
            status = 3
        else:
            budget = CAP_CPU - float(proj["smoke_cpu"])
            deadline = started + budget
            try:
                res = train_joint(SCREEN_SEED, data, deadline, steps=STEPS)
                report.update({"planned": "joint_1seed", **res})
            except CapExceeded as exc:
                report.update({"exit": "engineering_failure", "reason": f"cap exceeded: {exc}"})
            except NonFiniteLoss as exc:
                report.update({"exit": "engineering_failure", "reason": f"non-finite: {exc}"})
            if report.get("exit") in ("engineering_failure", "budget_insufficient"):
                status = 3
        out = FULL_REPORT

    report["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        k: report.get(k)
        for k in ("exit", "final_accs", "q3_vs_rt_ref_diffs", "q3_vs_rt_ref_mean", "cost")
    }
    if args.smoke:
        summary["projection"] = report["projection"]
    print(json.dumps(summary, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

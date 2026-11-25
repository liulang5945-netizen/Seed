"""N32.6.90: formal confirmation of joint task-agnostic S_joint on the sealed v2 confirm split.

Pre-registered: plans/reference/DESIGN-N32-6-90_20261125.md (D-20261125-550).
Runs the frozen N32.6.02/03 configuration on 3 unread confirmation seeds
(20261401, 20261402, 20261403) and evaluates on the sealed confirm worlds
(7,000-7,999), which are opened here for the first and only time.

  C1  confirm min(Q1,Q2,Q3) >= 0.90 per seed
  C2  mixed-batch equivalence exact on confirm
  C3  Q3@2k (val checkpoint) >= 0.75

Usage:
  python scripts/training/n32_6_90_confirm.py --smoke
  python scripts/training/n32_6_90_confirm.py
"""

from __future__ import annotations

import argparse
import hashlib
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
    EXPECTED_DIGESTS,
    CapExceeded,
    NonFiniteLoss,
    tensor_digest,
)
from scripts.training.n32_6_01_r1_edge_hop import (  # noqa: E402
    GATE_MIN,
    GATE_POS,
    KINDS,
    LR,
    Q3_CKPTS,
    WD,
    _lr_at,
    encode_all_kinds_fast,
    evaluate_kind,
    evaluate_mixed_equivalence,
    extract_modality_masks,
    make_column,
    prep,
    union_moments_dim,
)
from scripts.training.n32_6_02_r1_joint_diag import (  # noqa: E402
    JOINT_BATCH,
    LOSS_WEIGHTS,
    SMOKE_STEPS,
    STEPS,
)
from taiji import native_composition_v2 as nc2  # noqa: E402
from taiji.modality_edge_hop import ARCH_VERSION  # noqa: E402

CARD = "N32.6.90"
AUG_COPIES = 3
CONFIRM_SEEDS: tuple[int, ...] = (20261401, 20261402, 20261403)
CAP_CPU = 700.0
PROJECTION_MARGIN = 1.10
Q3_2K_GATE = 0.75

SMOKE_REPORT = ROOT / "reports" / "n32_6_90_smoke.json"
FULL_REPORT = ROOT / "reports" / "n32_6_90_confirm.json"


def _tensor_sha(x: torch.Tensor, y: torch.Tensor) -> str:
    return hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()


def build_data() -> dict[str, Any]:
    """Train/val as in N32.6.01 (digest-checked) + sealed confirm split (opened once here)."""
    train_raw = encode_all_kinds_fast(nc2.load_scenes("train"), AUG_COPIES)
    val_raw = encode_all_kinds_fast(nc2.load_scenes("val"), 0)
    confirm_scenes = nc2.load_scenes("confirm", confirmation_card=CARD)
    confirm_raw = encode_all_kinds_fast(confirm_scenes, 0)

    q3_train_dig = tensor_digest(*train_raw["q3"])
    q3_val_dig = tensor_digest(*val_raw["q3"])
    if q3_train_dig != EXPECTED_DIGESTS["train"]:
        raise AssertionError(f"Q3 train digest mismatch: {q3_train_dig}")
    if q3_val_dig != EXPECTED_DIGESTS["val"]:
        raise AssertionError(f"Q3 val digest mismatch: {q3_val_dig}")

    mu, sd = union_moments_dim([train_raw[k][0] for k in KINDS])
    out: dict[str, Any] = {
        "mu": mu,
        "sd": sd,
        "q3_digests": {"train": q3_train_dig, "val": q3_val_dig},
        "confirm_scenes": len(confirm_scenes),
    }
    out["confirm_digests"] = {k: _tensor_sha(*confirm_raw[k]) for k in KINDS}
    for split, raw in (("train", train_raw), ("val", val_raw), ("confirm", confirm_raw)):
        out[split] = {}
        for k in KINDS:
            x, y = raw[k]
            sm, qm = extract_modality_masks(x)
            out[split][k] = {"x": x, "sm": sm, "qm": qm, "y": y}
    return out


def train_joint_seed(
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
        k: torch.Generator().manual_seed(
            int(hashlib.sha256(f"{CARD}:{seed}:S_joint:{k}:batches".encode()).hexdigest()[:16], 16)
        )
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
    val_final = {k: evaluate_kind(model, data, "val", k, detail=True) for k in KINDS}
    confirm_final = {k: evaluate_kind(model, data, "confirm", k, detail=True) for k in KINDS}
    confirm_view = {"mu": data["mu"], "sd": data["sd"], "val": data["confirm"]}
    mixed = evaluate_mixed_equivalence(model, confirm_view)
    eval_cpu += time.process_time() - e1

    confirm_accs = {k: float(confirm_final[k]["acc"]) for k in KINDS}
    min_acc = min(confirm_accs.values())
    q3_2k = float(ckpts["2000"]["q3"])
    return {
        "seed": seed,
        "arm": "S_joint",
        "arch_version": ARCH_VERSION,
        "n_params": model.n_params(),
        "ckpts_val": ckpts,
        "q3_2k_val": q3_2k,
        "val_final": val_final,
        "confirm_final": confirm_final,
        "confirm_accs": confirm_accs,
        "confirm_min_acc": round(min_acc, 6),
        "confirm_entity_mean": {k: confirm_final[k]["entity_mean_acc"] for k in KINDS},
        "mixed_equivalence_confirm": mixed,
        "pass_c1": bool(min_acc >= GATE_POS),
        "pass_c2": bool(mixed["exact"]),
        "pass_c3": bool(q3_2k >= Q3_2K_GATE),
        "pass_all": bool(min_acc >= GATE_POS and mixed["exact"] and q3_2k >= Q3_2K_GATE),
        "pass_min": bool(min_acc >= GATE_MIN),
        "train_loop_cpu": round(train_loop_cpu, 3),
        "eval_cpu": round(eval_cpu, 3),
        "cpu": round(time.process_time() - t0, 2),
    }


def smoke(data: dict[str, Any], started: float, encode_cpu: float) -> dict[str, Any]:
    deadline = started + CAP_CPU
    res = train_joint_seed(CONFIRM_SEEDS[0], data, deadline, steps=SMOKE_STEPS, skip_eval=True)
    probe = make_column(CONFIRM_SEEDS[0], "S_joint")
    e0 = time.process_time()
    for k in KINDS:
        evaluate_kind(probe, data, "confirm", k, detail=True)
    eval_pass_cpu = time.process_time() - e0
    per_step = res["train_loop_cpu"] / SMOKE_STEPS
    proj_train = per_step * STEPS
    proj_eval = eval_pass_cpu * (len(Q3_CKPTS) + 2.0)
    proj_3seeds = round(
        encode_cpu + len(CONFIRM_SEEDS) * (proj_train + proj_eval) * PROJECTION_MARGIN, 2
    )
    smoke_cpu = round(time.process_time() - started, 2)
    planned = "confirm_3seeds" if smoke_cpu + proj_3seeds <= CAP_CPU else "budget_insufficient"
    return {
        "encode_cpu": encode_cpu,
        "smoke_cpu": smoke_cpu,
        "per_step_cpu": per_step,
        "eval_pass_cpu": round(eval_pass_cpu, 3),
        "proj_train_cpu_per_seed": round(proj_train, 2),
        "projected_3seeds": proj_3seeds,
        "margin": PROJECTION_MARGIN,
        "cap_cpu": CAP_CPU,
        "planned": planned,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # smoke must not consume the one-time read; the full run records it
    design = preflight(CARD, list(CONFIRM_SEEDS), record=not args.smoke).name
    torch.set_num_threads(1)
    started = time.process_time()
    data = build_data()
    encode_cpu = round(time.process_time() - started, 2)

    report: dict[str, Any] = {
        "schema": "taiji-n32-6-90-v1",
        "card": CARD,
        "arch_version": ARCH_VERSION,
        "confirm_seeds": list(CONFIRM_SEEDS),
        "steps": STEPS,
        "joint_batch": JOINT_BATCH,
        "loss_weights": LOSS_WEIGHTS,
        "ckpts": list(Q3_CKPTS),
        "design": design,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "data": {
            "train_rows": {k: int(data["train"][k]["y"].numel()) for k in KINDS},
            "val_rows": {k: int(data["val"][k]["y"].numel()) for k in KINDS},
            "confirm_rows": {k: int(data["confirm"][k]["y"].numel()) for k in KINDS},
            "confirm_scenes": data["confirm_scenes"],
            "q3_digests": data["q3_digests"],
            "confirm_digests": data["confirm_digests"],
        },
        "gates": {
            "c1_min": GATE_POS,
            "c2_mixed_exact": True,
            "c3_q3_2k": Q3_2K_GATE,
            "min_floor": GATE_MIN,
        },
        "encode_cpu": encode_cpu,
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": len(CONFIRM_SEEDS),
            "exploratory": False,
        },
    }

    status = 0
    if args.smoke:
        report["projection"] = smoke(data, started, encode_cpu)
        out = SMOKE_REPORT
        if report["projection"]["planned"] == "budget_insufficient":
            status = 3
    else:
        proj = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))["projection"]
        if proj["planned"] != "confirm_3seeds":
            report.update({"exit": "budget_insufficient", "planned": proj["planned"]})
            status = 3
        else:
            budget = CAP_CPU - float(proj["smoke_cpu"])
            deadline = started + budget
            per_seed: dict[str, dict[str, Any]] = {}
            try:
                for seed in CONFIRM_SEEDS:
                    per_seed[str(seed)] = train_joint_seed(seed, data, deadline, steps=STEPS)
                report["planned"] = "confirm_3seeds"
                report["per_seed"] = per_seed
                accs = {s: r["confirm_accs"] for s, r in per_seed.items()}
                report["confirm_summary"] = {
                    "min_per_seed": {s: r["confirm_min_acc"] for s, r in per_seed.items()},
                    "accs": accs,
                    "pass_all_seeds": all(r["pass_all"] for r in per_seed.values()),
                }
                if all(r["pass_all"] for r in per_seed.values()):
                    report["exit"] = "confirm_positive"
                elif all(r["pass_min"] for r in per_seed.values()):
                    report["exit"] = "confirm_partial"
                else:
                    report["exit"] = "confirm_unstable"
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
        k: report.get(k) for k in ("exit", "confirm_summary", "cost") if report.get(k) is not None
    }
    if "per_seed" in report:
        summary["per_seed"] = {
            s: {
                "confirm_accs": r["confirm_accs"],
                "confirm_min_acc": r["confirm_min_acc"],
                "c1": r["pass_c1"],
                "c2": r["pass_c2"],
                "c3": r["pass_c3"],
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

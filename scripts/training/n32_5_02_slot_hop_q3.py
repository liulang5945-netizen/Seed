"""N32.5.02: SlotHopGRU Q3 two-hop learnability benchmark (Route 1, <= 350 CPU).

Pre-registered: plans/reference/DESIGN-N32-5-02_20261125.md (D-20261125-536).
Reuses the exact bit-verified Q3 pipeline from N32.5.01 (QF order, standardised
inputs, name-relabel augmentation x3, SHA256 digest checks) and compares
SlotHopGRU(d_model=48, n_heads=4, n_hops=2) apples-to-apples against N32.5.01's
plain G3_gru96_q3 (val 0.7039, entity mean 0.4009).

Usage:
  python scripts/training/n32_5_02_slot_hop_q3.py --smoke
  python scripts/training/n32_5_02_slot_hop_q3.py
"""

from __future__ import annotations

import argparse
import json
import math
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
    AUG_COPIES,
    BATCH,
    EVAL_EVERY,
    GATE_GRU,
    Q3_LEN,
    SCREEN_SEED,
    SEEDS,
    STEPS,
    CapExceeded,
    NonFiniteLoss,
    build_data,
    evaluate,
    int_seed,
    prep,
)
from taiji import composition_encoding as ce  # noqa: E402
from taiji.native_composition import NAMES  # noqa: E402
from taiji.slot_memory_gru import ARCH_VERSION, SlotHopGRU  # noqa: E402

CARD = "N32.5.02"
ARM = "SlotHopGRU_q3"
D_MODEL = 48
N_HEADS = 4
N_HOPS = 2
FFN_MULT = 2
LR = 2e-3
WD = 1e-4
WARMUP_STEPS = 200
SMOKE_STEPS = 100
CAP_CPU = 350.0
PROJECTION_MARGIN = 1.10

REPORTS = ROOT / "reports"
SMOKE_REPORT = REPORTS / "n32_5_02_smoke.json"
FULL_REPORT = REPORTS / "n32_5_02_learnability.json"


def make_model(seed: int) -> SlotHopGRU:
    return SlotHopGRU(
        len(ce.ANSWERS),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_hops=N_HOPS,
        ffn_mult=FFN_MULT,
        seed=int_seed(CARD, seed, ARM, "init"),
    )


def _lr_at(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LR * float(step + 1) / float(WARMUP_STEPS)
    progress = float(step - WARMUP_STEPS) / float(max(total_steps - WARMUP_STEPS, 1))
    return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _entity_mean(val_detail: dict[str, Any]) -> float:
    per_class = val_detail["per_class"]
    vals = [float(per_class[name]) for name in NAMES if name in per_class]
    return sum(vals) / len(vals) if vals else 0.0


def train_seed(
    seed: int, data: dict[str, Any], deadline: float, steps: int = STEPS
) -> dict[str, Any]:
    model = make_model(seed)
    gen = torch.Generator().manual_seed(int_seed(CARD, seed, ARM, "batches"))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n = data["train_x"].shape[0]
    curve: list[list[float]] = []

    t0 = time.process_time()
    eval_cpu_total = 0.0
    model.train()
    for i in range(steps):
        if time.process_time() > deadline:
            raise CapExceeded(f"{ARM} seed {seed} step {i}")
        lr = _lr_at(i, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        loss = F.cross_entropy(model(prep(data["train_x"][idx], data)), data["train_y"][idx])
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"{ARM} seed {seed} step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == steps:
            e0 = time.process_time()
            acc = evaluate(model, data, data["val_x"], data["val_y"])["acc"]
            eval_cpu_total += time.process_time() - e0
            curve.append([i + 1, acc])
            model.train()

    e1 = time.process_time()
    val_detail = evaluate(model, data, data["val_x"], data["val_y"], detail=True)
    probe = data["probe_idx"]
    probe_res = evaluate(model, data, data["train_x"][probe], data["train_y"][probe])
    eval_cpu_total += time.process_time() - e1

    total_cpu = time.process_time() - t0
    train_loop_cpu = max(total_cpu - eval_cpu_total, 0.0)
    return {
        "arm": ARM,
        "arch_version": ARCH_VERSION,
        "kind": "q3",
        "seed": seed,
        "steps": steps,
        "presentations": steps * BATCH,
        "val": val_detail,
        "entity_mean_acc": round(_entity_mean(val_detail), 6),
        "train_probe_acc": probe_res["acc"],
        "curve": curve,
        "n_params": model.n_params(),
        "cpu": round(total_cpu, 2),
        "train_loop_cpu": round(train_loop_cpu, 3),
        "eval_cpu_total": round(eval_cpu_total, 3),
    }


def smoke(data: dict[str, Any], started: float, encode_cpu: float) -> dict[str, Any]:
    deadline = started + CAP_CPU
    row = train_seed(SCREEN_SEED, data, deadline, steps=SMOKE_STEPS)
    model = make_model(SCREEN_SEED)
    e0 = time.process_time()
    evaluate(model, data, data["val_x"], data["val_y"], detail=True)
    eval_cpu = time.process_time() - e0
    per_step_cpu = row["train_loop_cpu"] / SMOKE_STEPS

    smoke_cpu = time.process_time() - started
    n_evals = STEPS // EVAL_EVERY + 2
    per_seed = (per_step_cpu * STEPS + n_evals * eval_cpu) * PROJECTION_MARGIN
    proj_1 = round(encode_cpu + per_seed, 2)
    proj_3 = round(encode_cpu + 3.0 * per_seed, 2)

    if smoke_cpu + proj_3 <= CAP_CPU:
        planned = "all_3seeds"
    elif smoke_cpu + proj_1 <= CAP_CPU:
        planned = "screen_only"
    else:
        planned = "budget_insufficient"

    return {
        "encode_cpu": encode_cpu,
        "per_step_cpu": per_step_cpu,
        "eval_cpu": eval_cpu,
        "smoke_cpu": round(smoke_cpu, 2),
        "projected_1seed": proj_1,
        "projected_3seeds": proj_3,
        "margin": PROJECTION_MARGIN,
        "cap_cpu": CAP_CPU,
        "planned": planned,
    }


def full(data: dict[str, Any], started: float, budget: float) -> dict[str, Any]:
    proj = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))["projection"]
    planned = str(proj["planned"])
    if planned == "budget_insufficient" or planned not in ("all_3seeds", "screen_only"):
        return {
            "planned": planned,
            "runs": [],
            "exit": "budget_insufficient",
            "reason": f"smoke projection does not fit CAP_CPU={CAP_CPU}",
        }

    deadline = started + budget
    out: dict[str, Any] = {"planned": planned, "runs": []}
    screen = train_seed(SCREEN_SEED, data, deadline, steps=STEPS)
    out["runs"].append(screen)
    out["screen_acc"] = screen["val"]["acc"]
    out["screen_entity_mean_acc"] = screen["entity_mean_acc"]
    if screen["val"]["acc"] < GATE_GRU:
        out["exit"] = "q3_not_learnable"
        out["accs"] = [screen["val"]["acc"]]
        return out

    if planned == "screen_only":
        out["exit"] = "screen_passed_needs_rep_budget"
        out["accs"] = [screen["val"]["acc"]]
        return out

    reps = [train_seed(s, data, deadline, steps=STEPS) for s in SEEDS[1:]]
    out["runs"].extend(reps)
    accs = [screen["val"]["acc"], *[r["val"]["acc"] for r in reps]]
    out["accs"] = accs
    out["exit"] = "learnable" if min(accs) >= GATE_GRU else "learnable_unstable"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    designs = {preflight(CARD, [s], record=False).name for s in SEEDS}
    torch.set_num_threads(1)
    started = time.process_time()
    data = build_data()
    encode_cpu = round(time.process_time() - started, 2)

    report: dict[str, Any] = {
        "schema": "taiji-n32-5-02-v1",
        "card": CARD,
        "arch_version": ARCH_VERSION,
        "model_cfg": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_hops": N_HOPS,
            "ffn_mult": FFN_MULT,
        },
        "design": sorted(designs),
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "data": {
            "q3": {
                "train_rows": int(data["train_y"].numel()),
                "val_rows": int(data["val_y"].numel()),
                "aug_copies": AUG_COPIES,
                "digests": data["digests"],
                "val_majority": data["majority"],
            }
        },
        "length": Q3_LEN,
        "gates": {"q3": GATE_GRU},
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
        budget = CAP_CPU - float(smoke_doc["projection"]["smoke_cpu"])
        try:
            report.update(full(data, started, budget))
        except CapExceeded as exc:
            report.update({"exit": "engineering_failure", "reason": f"cap exceeded: {exc}"})
        except NonFiniteLoss as exc:
            report.update({"exit": "engineering_failure", "reason": f"non-finite: {exc}"})
        out = FULL_REPORT
        if report.get("exit") in ("engineering_failure", "budget_insufficient"):
            status = 3

    report["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {
        k: report.get(k) for k in ("exit", "planned", "accs", "screen_entity_mean_acc", "cost")
    }
    if args.smoke:
        summary["projection"] = report["projection"]
    print(json.dumps(summary, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

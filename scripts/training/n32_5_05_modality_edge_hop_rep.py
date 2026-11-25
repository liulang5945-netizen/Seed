"""N32.5.05: 2-seed replication (20261121, 20261122) of ModalityEdgeHop on Q3 (<= 500 CPU).

Pre-registered: plans/reference/DESIGN-N32-5-05_20261125.md (D-20261125-542).
Reuses the exact bit-verified Q3 dataset, hyperparameters, and model configuration
from N32.5.04 (seed 20261120 = 0.9899).

Usage:
  python scripts/training/n32_5_05_modality_edge_hop_rep.py --smoke
  python scripts/training/n32_5_05_modality_edge_hop_rep.py
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
    AUG_COPIES,
    BATCH,
    EVAL_EVERY,
    GATE_GRU,
    Q3_LEN,
    CapExceeded,
    NonFiniteLoss,
    int_seed,
)
from scripts.training.n32_5_04_modality_edge_hop_q3 import (  # noqa: E402
    ARM,
    D_EDGE,
    D_MODEL,
    LR,
    N_EDGE_LAYERS,
    N_HOPS,
    SMOKE_STEPS,
    STEPS,
    WD,
    _entity_mean,
    _lr_at,
    build_data,
    evaluate_edge,
    prep,
)
from taiji import composition_encoding as ce  # noqa: E402
from taiji.modality_edge_hop import ARCH_VERSION, ModalityEdgeHop  # noqa: E402

CARD = "N32.5.05"
REP_SEEDS = (20261121, 20261122)
CAP_CPU = 500.0
PROJECTION_MARGIN = 1.10
PRIOR_SCREEN_ACC = 0.9899166822433472

REPORTS = ROOT / "reports"
SMOKE_REPORT = REPORTS / "n32_5_05_smoke.json"
FULL_REPORT = REPORTS / "n32_5_05_learnability.json"


def make_model(seed: int) -> ModalityEdgeHop:
    return ModalityEdgeHop(
        len(ce.ANSWERS),
        d_model=D_MODEL,
        d_edge=D_EDGE,
        n_edge_layers=N_EDGE_LAYERS,
        n_hops=N_HOPS,
        seed=int_seed(CARD, seed, ARM, "init"),
    )


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
        logits = model(
            prep(data["train_x"][idx], data),
            data["train_sm"][idx],
            data["train_qm"][idx],
        )
        loss = F.cross_entropy(logits, data["train_y"][idx])
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"{ARM} seed {seed} step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == steps:
            e0 = time.process_time()
            acc = evaluate_edge(
                model,
                data,
                data["val_x"],
                data["val_sm"],
                data["val_qm"],
                data["val_y"],
            )["acc"]
            eval_cpu_total += time.process_time() - e0
            curve.append([i + 1, acc])
            model.train()

    e1 = time.process_time()
    val_detail = evaluate_edge(
        model,
        data,
        data["val_x"],
        data["val_sm"],
        data["val_qm"],
        data["val_y"],
        detail=True,
    )
    probe = data["probe_idx"]
    probe_res = evaluate_edge(
        model,
        data,
        data["train_x"][probe],
        data["train_sm"][probe],
        data["train_qm"][probe],
        data["train_y"][probe],
    )
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
    row = train_seed(REP_SEEDS[0], data, deadline, steps=SMOKE_STEPS)
    model = make_model(REP_SEEDS[0])
    e0 = time.process_time()
    evaluate_edge(
        model,
        data,
        data["val_x"],
        data["val_sm"],
        data["val_qm"],
        data["val_y"],
        detail=True,
    )
    eval_cpu = time.process_time() - e0
    per_step_cpu = row["train_loop_cpu"] / SMOKE_STEPS

    smoke_cpu = time.process_time() - started
    n_evals = STEPS // EVAL_EVERY + 2
    per_seed = (per_step_cpu * STEPS + n_evals * eval_cpu) * PROJECTION_MARGIN
    proj_2 = round(encode_cpu + 2.0 * per_seed, 2)
    planned = "rep_2seeds" if smoke_cpu + proj_2 <= CAP_CPU else "budget_insufficient"
    return {
        "encode_cpu": encode_cpu,
        "per_step_cpu": per_step_cpu,
        "eval_cpu": eval_cpu,
        "smoke_cpu": round(smoke_cpu, 2),
        "projected_2seeds": proj_2,
        "margin": PROJECTION_MARGIN,
        "cap_cpu": CAP_CPU,
        "planned": planned,
    }


def full(data: dict[str, Any], started: float, budget: float) -> dict[str, Any]:
    proj = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))["projection"]
    planned = str(proj["planned"])
    if planned != "rep_2seeds":
        return {
            "planned": planned,
            "runs": [],
            "exit": "budget_insufficient",
            "reason": f"smoke projection does not fit CAP_CPU={CAP_CPU}",
        }

    deadline = started + budget
    runs = [train_seed(s, data, deadline, steps=STEPS) for s in REP_SEEDS]
    accs = [float(r["val"]["acc"]) for r in runs]
    entity_accs = [float(r["entity_mean_acc"]) for r in runs]
    exit_label = "learnable" if min(accs) >= GATE_GRU else "learnable_unstable"
    return {
        "planned": planned,
        "runs": runs,
        "rep_accs": accs,
        "rep_entity_mean_accs": entity_accs,
        "combined_3seed_accs": [PRIOR_SCREEN_ACC, *accs],
        "exit": exit_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    designs = {preflight(CARD, [s], record=False).name for s in REP_SEEDS}
    torch.set_num_threads(1)
    started = time.process_time()
    data = build_data()
    encode_cpu = round(time.process_time() - started, 2)

    report: dict[str, Any] = {
        "schema": "taiji-n32-5-05-v1",
        "card": CARD,
        "arm": ARM,
        "arch_version": ARCH_VERSION,
        "model_cfg": {
            "d_model": D_MODEL,
            "d_edge": D_EDGE,
            "n_edge_layers": N_EDGE_LAYERS,
            "n_hops": N_HOPS,
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
        k: report.get(k)
        for k in (
            "exit",
            "planned",
            "rep_accs",
            "rep_entity_mean_accs",
            "combined_3seed_accs",
            "cost",
        )
    }
    if args.smoke:
        summary["projection"] = report["projection"]
    print(json.dumps(summary, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

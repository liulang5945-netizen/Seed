"""10.7 verification: one bounded training run, measured on states it never saw.

Six of the twelve samples train; the other six come from three different states and
are the only face that decides. Two learning rates are both published, no cherry-pick.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_living_fragment_pricing_r028 as pricing  # noqa: E402
import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji import native_development as development  # noqa: E402
from taiji.goal_conditioned_training import accuracy, samples, save, train  # noqa: E402

STEPS = 32
LR_GRID = (1e-5, 1e-4)
RETENTION_GATE = 0.030
TRAIN_STATES = 3
CLOSEOUT_CARD = "reports/taiji_core_09_closeout_20261101.json"
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200
OUTPUT_DIR = "output/taiji_core_10_7"


def split_samples() -> tuple[list[dict], list[dict]]:
    rows = samples()
    per_state = len(rows) // 6
    cut = TRAIN_STATES * per_state
    return rows[:cut], rows[cut:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    card = json.loads((ROOT / CLOSEOUT_CARD).read_text(encoding="utf-8"))
    expected_identity = card["identity"]["parent_model_digest"]

    parent, subject, provider = prior.build_subject()
    identity_ok = prior.parent_digest(parent) == expected_identity
    train_rows, held_rows = split_samples()

    baseline = accuracy(subject, held_rows, provider)
    results = []
    for lr in LR_GRID:
        guard_start = time.process_time()
        candidate = copy.deepcopy(subject)
        retention_pre = development.evaluate_career_nll(candidate, pricing.RETENTION, provider)
        outcome = train(candidate, train_rows, lr=lr, steps=STEPS, provider=provider)
        retention_post = development.evaluate_career_nll(candidate, pricing.RETENTION, provider)
        held = accuracy(candidate, held_rows, provider)
        path = ROOT / OUTPUT_DIR / f"trained_lr{lr:g}.pt"
        save(candidate, path)
        results.append(
            {
                "lr": lr,
                "steps": STEPS,
                "first_loss": outcome["losses"][0],
                "last_loss": outcome["losses"][-1],
                "train_cpu_seconds": outcome["cpu_seconds"],
                "retention_drift": abs(retention_post - retention_pre),
                "held_out_hits": held["hits"],
                "held_out_total": held["total"],
                "held_out": held["rows"],
                "weights_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "weights_bytes": path.stat().st_size,
                "wall_seconds": time.process_time() - guard_start,
            }
        )
        print(f"lr {lr:g} done cpu={time.process_time() - total_cpu:.1f}", flush=True)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    passing = [
        row
        for row in results
        if row["held_out_hits"] >= 5 and row["retention_drift"] <= RETENTION_GATE
    ]
    breaching = [row["lr"] for row in results if row["retention_drift"] > RETENTION_GATE]
    best_hits = max(row["held_out_hits"] for row in results)
    learnable_state = (
        "train_learned"
        if any(row["last_loss"] < row["first_loss"] for row in results)
        else "train_flat"
    )

    if not identity_ok:
        status = "candidate_identity_mismatch"
    elif cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif passing:
        status = "training_learned_selection"
    elif best_hits >= 5:
        status = "retention_breach"
    elif best_hits == 4:
        status = "training_partial"
    else:
        status = "training_below_chance_plus"

    checks = [
        {
            "id": "candidate_identity_mismatch",
            "passed": identity_ok,
            "observed": [prior.parent_digest(parent), expected_identity],
        },
        {
            "id": "held_out_states_never_trained",
            "passed": len(held_rows) == 6,
            "observed": len(held_rows),
        },
        {
            "id": "training_learned_selection",
            "passed": bool(passing),
            "observed": [
                {"lr": row["lr"], "hits": row["held_out_hits"], "drift": row["retention_drift"]}
                for row in results
            ],
        },
        {"id": "retention_within_gate", "passed": not breaching, "observed": breaching},
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]

    report = {
        "schema": "taiji-goal-training-10-7-v1",
        "step": "10.7",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "steps": STEPS,
            "lr_grid": list(LR_GRID),
            "train_samples": len(train_rows),
            "held_out_samples": len(held_rows),
            "held_out_states": TRAIN_STATES,
            "retention_gate": RETENTION_GATE,
            "both_lrs_published": True,
        },
        "parent_model_digest": prior.parent_digest(parent),
        "untrained_held_out": {
            "hits": baseline["hits"],
            "total": baseline["total"],
            "rows": baseline["rows"],
        },
        "runs": results,
        "training_direction": learnable_state,
        "weights_published": False,
        "default_model_replaced": False,
        "model_involved": True,
        "capability_claimed": False,
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "Twelve synthetic samples, six held-out from three unseen states. A pass means the bounded "
            "run learned this toy selection on this fixture; it is not natural-language task capability, "
            "does not satisfy P2 on its own, and authorises no further training."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/living_core.py",
                "taiji/native_goal_world.py",
                "taiji/action_verdict_scorer.py",
                "taiji/goal_conditioned_training.py",
                "scripts/training/verify_taiji_goal_training_10_7.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "untrained": [baseline["hits"], baseline["total"]],
                "runs": [
                    {
                        "lr": row["lr"],
                        "hits": row["held_out_hits"],
                        "drift": round(row["retention_drift"], 6),
                        "first_loss": round(row["first_loss"], 4),
                        "last_loss": round(row["last_loss"], 4),
                    }
                    for row in results
                ],
                "cpu_seconds": round(cpu_seconds, 1),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

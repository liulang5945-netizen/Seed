"""10.5 verification: does the contrast interface actually read the goal?

Two goals over one shared initial state, each with one legal correct action. The
gate is the contrast gap against a same-goal null control; the direction is reported
but not judged, because with two candidates the sign decides it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.goal_conditioned_scorer import choose, contrast_rows, margin  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES, build_fixtures, legal_moves  # noqa: E402

SEED = 20261101
PARTS = 5
UNITS_PER_GROUP = 3
STEPS = 8
LR = 1e-5
GAP_FLOOR = 1e-3
CLOSEOUT_CARD = "reports/taiji_core_09_closeout_20261101.json"
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200
GOAL_A = "把蓝色杯子放到窗台"
GOAL_B = "把红色印章放到窗台"


def build_subject():
    return prior.build_subject()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.5 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()

    card = json.loads((ROOT / CLOSEOUT_CARD).read_text(encoding="utf-8"))
    expected_identity = card["identity"]["parent_model_digest"]
    parent, subject, provider = build_subject()
    identity_ok = prior.parent_digest(parent) == expected_identity

    state = build_fixtures()[0].initial
    moves = legal_moves(state, occupy=True)
    rows = contrast_rows(subject, GOAL_A, GOAL_B, moves, provider=provider)
    rows_repeat = contrast_rows(subject, GOAL_A, GOAL_B, moves, provider=provider)
    null_rows = contrast_rows(subject, GOAL_A, GOAL_A, moves, provider=provider)

    cup = (OBJECTS[0], PLACES[1])
    seal = (OBJECTS[1], PLACES[1])
    gap = margin(rows, cup, seal)
    null_gap = margin(null_rows, cup, seal)
    deterministic = all(
        abs(left["contrast"] - right["contrast"]) <= 1e-12
        for left, right in zip(rows, rows_repeat, strict=True)
    )
    selection = choose(subject, GOAL_A, GOAL_B, moves, provider=provider)

    separable = gap["gap"] >= GAP_FLOOR
    null_exact = all(abs(row["contrast"]) <= 1e-12 for row in null_rows)
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall

    if not identity_ok:
        status = "candidate_identity_mismatch"
    elif not deterministic:
        status = "nondeterministic_selection"
    elif not null_exact:
        status = "nonzero_null"
    elif not separable:
        status = "contrast_below_floor"
    else:
        status = "goal_conditioned_selection"

    checks = [
        {
            "id": "candidate_identity_mismatch",
            "passed": identity_ok,
            "observed": [prior.parent_digest(parent), expected_identity],
        },
        {"id": "nondeterministic_selection", "passed": deterministic, "observed": deterministic},
        {
            "id": "nonzero_null",
            "passed": null_exact,
            "observed": [row["contrast"] for row in null_rows],
        },
        {"id": "contrast_below_floor", "passed": separable, "observed": gap},
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-goal-conditioned-10-5-v1",
        "step": "10.5",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "goal_a": GOAL_A,
            "goal_b": GOAL_B,
            "gap_floor": GAP_FLOOR,
            "initial": [list(pair) for pair in state.where],
            "legal_moves": [list(move) for move in moves],
            "direction_judged": False,
        },
        "parent_model_digest": prior.parent_digest(parent),
        "contrast": [{**row, "move": list(row["move"])} for row in rows],
        "null_control": [{**row, "move": list(row["move"])} for row in null_rows],
        "gap": gap,
        "null_gap": null_gap,
        "selection_under_goal_a": {
            "action": list(selection["action"]),
            "contrast": selection["contrast"],
        },
        "model_involved": True,
        "capability_claimed": False,
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "Interface probe only. With two candidate actions the winning action is decided by the "
            "sign, so only the contrast gap is judged. Passing does not show correct selection, task "
            "capability, or that P2 is satisfied."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/living_core.py",
                "taiji/native_goal_world.py",
                "taiji/subject_action_scorer.py",
                "taiji/goal_conditioned_scorer.py",
                "scripts/training/verify_taiji_goal_conditioned_10_5.py",
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
                "gap": round(gap["gap"], 9),
                "contrasts": {str(row["move"]): round(row["contrast"], 9) for row in rows},
                "identity_ok": identity_ok,
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

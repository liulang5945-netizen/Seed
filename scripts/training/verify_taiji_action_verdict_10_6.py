"""10.6 verification: does putting the action inside the prefix give action-specific signal?

One state, one goal, two legal actions. The gate is whether the yes/no verdict gap
between the correct and the other action clears the declared floor, plus an
answer-token distinguishability check.
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
from taiji.action_verdict_scorer import YES, answer_nll, score_actions  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES, build_fixtures, legal_moves  # noqa: E402

GAP_FLOOR = 1e-3
ANSWER_FLOOR = 1e-6
CLOSEOUT_CARD = "reports/taiji_core_09_closeout_20261101.json"
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200
GOAL = "把蓝色杯子放到窗台"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.6 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    card = json.loads((ROOT / CLOSEOUT_CARD).read_text(encoding="utf-8"))
    expected_identity = card["identity"]["parent_model_digest"]

    parent, subject, provider = prior.build_subject()
    identity_ok = prior.parent_digest(parent) == expected_identity

    state = build_fixtures()[0].initial
    moves = legal_moves(state, occupy=True)
    rows = score_actions(subject, state, GOAL, moves, provider=provider)
    rows_repeat = score_actions(subject, state, GOAL, moves, provider=provider)
    deterministic = all(
        abs(left["verdict"] - right["verdict"]) <= 1e-12
        for left, right in zip(rows, rows_repeat, strict=True)
    )
    sample = rows[0]["prefix"]
    repeat_zero = abs(
        answer_nll(subject, sample, YES, provider) - answer_nll(subject, sample, YES, provider)
    )

    cup = (OBJECTS[0], PLACES[1])
    seal = (OBJECTS[1], PLACES[1])
    by_move = {tuple(row["move"]): row for row in rows}
    gap = abs(by_move[cup]["verdict"] - by_move[seal]["verdict"])
    correct_higher = by_move[cup]["verdict"] > by_move[seal]["verdict"]
    answers_separable = max(row["answer_gap"] for row in rows) > ANSWER_FLOOR

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if not identity_ok:
        status = "candidate_identity_mismatch"
    elif not deterministic or repeat_zero > 1e-12:
        status = "nondeterministic_selection"
    elif not answers_separable:
        status = "answer_tokens_indistinguishable"
    elif gap < GAP_FLOOR:
        status = "action_gap_below_floor"
    else:
        status = "action_specific_signal"

    checks = [
        {
            "id": "candidate_identity_mismatch",
            "passed": identity_ok,
            "observed": [prior.parent_digest(parent), expected_identity],
        },
        {
            "id": "nondeterministic_selection",
            "passed": deterministic and repeat_zero <= 1e-12,
            "observed": [deterministic, repeat_zero],
        },
        {
            "id": "answer_tokens_indistinguishable",
            "passed": answers_separable,
            "observed": [row["answer_gap"] for row in rows],
        },
        {"id": "action_gap_below_floor", "passed": gap >= GAP_FLOOR, "observed": gap},
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-action-verdict-10-6-v1",
        "step": "10.6",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "goal": GOAL,
            "gap_floor": GAP_FLOOR,
            "answer_floor": ANSWER_FLOOR,
            "construction": "prefix = state + goal + action sentence; answers yes/no",
            "direction_judged": False,
        },
        "parent_model_digest": prior.parent_digest(parent),
        "rows": [{**row, "move": list(row["move"])} for row in rows],
        "gap": gap,
        "correct_action_higher": correct_higher,
        "repeat_zero": repeat_zero,
        "model_involved": True,
        "capability_claimed": False,
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "Construction probe only. A separable gap means this construction gives the subject an "
            "action-specific signal here; it is not correct selection, not task capability and not P2."
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
                "scripts/training/verify_taiji_action_verdict_10_6.py",
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
                "gap": round(gap, 9),
                "correct_higher": correct_higher,
                "verdicts": {str(row["move"]): round(row["verdict"], 9) for row in rows},
                "answer_gaps": [round(row["answer_gap"], 9) for row in rows],
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

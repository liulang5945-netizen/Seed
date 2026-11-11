"""10.15: is the wide fixture (4 objects x 5 places) fair and big enough?

Three machine checks, no model:

* 480 samples over 120 states, enough for >1 constraint per readout parameter;
* the goal names no entity and the glossary resolves it uniquely;
* a character-overlap matcher cannot beat chance across 20 actions.
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

from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECTS,
    PLACES,
    action_sentence,
    resolve,
    samples,
    states,
)

BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240
COPY_SHARE = 0.10


def bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}


def overlap(first: str, second: str) -> float:
    left, right = bigrams(first), bigrams(second)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.15 card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    rows = samples()
    moves = [(obj, place) for obj in OBJECTS for place in PLACES]
    entity_names = [*OBJECTS, *PLACES]

    leaks = 0
    resolved = 0
    copy_hits = 0
    shared_with_correct = 0
    for row in rows:
        goal = row["goal"]
        if any(name in goal for name in entity_names):
            leaks += 1
        if resolve(goal) == row["correct"]:
            resolved += 1
        best = max(
            ((overlap(goal, action_sentence(o, p)), (o, p)) for o, p in moves),
            key=lambda item: (item[0], item[1]),
        )
        copy_hits += int(best[1] == row["correct"])
        correct_sentence = action_sentence(*row["correct"])
        if bigrams(goal) & bigrams(correct_sentence):
            shared_with_correct += 1

    states_count = len(states())
    constraints = len(rows) * len(moves)
    share = copy_hits / len(rows)
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif leaks == 0 and resolved == len(rows) and shared_with_correct == 0 and share <= COPY_SHARE:
        status = "wide_fixture_ready"
    else:
        status = "wide_fixture_leaks"

    report = {
        "schema": "taiji-wide-fixture-10-15-v1",
        "step": "10.15",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "objects": len(OBJECTS),
            "places": len(PLACES),
            "states": states_count,
            "samples": len(rows),
            "actions_per_state": len(moves),
            "logit_constraints": constraints,
            "copy_share_allowed": COPY_SHARE,
            "chance_top1": 1 / len(moves),
            "training": "none",
        },
        "state_count": states_count,
        "sample_count": len(rows),
        "entity_leaks": leaks,
        "resolved_by_glossary": resolved,
        "copy_top1": copy_hits,
        "goals_sharing_bigram_with_correct": shared_with_correct,
        "copy_share": share,
        "model_involved": False,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Fixture fairness and size only. 480 samples make a readout identifiable in "
            "principle; it says nothing about whether this candidate can learn them."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_world_wide.py",
                "scripts/training/verify_taiji_wide_fixture_10_15.py",
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
                "states": states_count,
                "samples": len(rows),
                "constraints": constraints,
                "entity_leaks": leaks,
                "resolved": resolved,
                "copy_hits": copy_hits,
                "copy_share": round(share, 4),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

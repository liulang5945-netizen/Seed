"""10.9: check that attribute-described goals kill the copy shortcut and stay solvable.

Three properties, all machine-checkable without a model:

* the goal names no entity (no object or place name appears in it);
* the glossary resolves the goal to exactly one (object, place) pair;
* a character-overlap matcher between the goal and the action sentences cannot win.

If all three hold, the fixture is *fair*: the answer is determined, but only through
the glossary, not by copying words.
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

from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.native_goal_attributes import (  # noqa: E402
    GLOSSARY,
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
    action_sentence,
    describe_goal,
    resolve,
)
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

BUDGET_CPU_SECONDS = 60
BUDGET_WALL_SECONDS = 120
COPY_ALLOWED = 2


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
        raise FileExistsError("10.9 card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    rows = samples()
    per_state = len(rows) // 6
    states = [rows[index * per_state] for index in range(6)]
    moves = [(obj, place) for obj in OBJECTS for place in PLACES]
    entity_names = [*OBJECTS, *PLACES]

    leaks = 0
    resolved = 0
    copy_hits = 0
    detail = []
    for state in states:
        occupied = {position for _, position in state["state"].where}
        free = next(place for place in PLACES if place not in occupied)
        for obj in OBJECTS:
            goal = describe_goal(obj, free)
            named = [name for name in entity_names if name in goal]
            if named:
                leaks += 1
            resolved_obj, resolved_place = resolve(goal)
            correct = resolved_obj == obj and resolved_place == free
            resolved += int(correct)
            scored = sorted(
                ((overlap(goal, action_sentence(o, p)), (o, p)) for o, p in moves),
                key=lambda item: (-item[0], item[1]),
            )
            pick = scored[0][1]
            hit = pick == (obj, free)
            copy_hits += int(hit)
            detail.append(
                {
                    "goal": goal,
                    "correct_move": [obj, free],
                    "named_entities": named,
                    "resolved": [resolved_obj, resolved_place],
                    "copy_pick": list(pick),
                    "copy_hit": hit,
                    "top_overlap": scored[0][0],
                }
            )

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif leaks == 0 and resolved == len(detail) and copy_hits <= COPY_ALLOWED:
        status = "attribute_fixture_fair"
    else:
        status = "attribute_fixture_leaks"

    report = {
        "schema": "taiji-goal-attributes-10-9-v1",
        "step": "10.9",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "goals": len(detail),
            "actions_per_goal": len(moves),
            "copy_allowed": COPY_ALLOWED,
            "chance_top1": 1 / len(moves),
            "descriptions": {
                **{obj: text for obj, text in OBJECT_DESCRIPTIONS.items()},
                **{place: text for place, text in PLACE_DESCRIPTIONS.items()},
            },
            "glossary_entries": len(GLOSSARY),
            "training": "none",
        },
        "entity_leaks": leaks,
        "resolved_by_glossary": resolved,
        "copy_top1": copy_hits,
        "rows": detail,
        "model_involved": False,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Fixture fairness only. A fair fixture makes a later capability reading "
            "interpretable; it is not one. The glossary is data a subject would have to be "
            "given; this step does not connect it to any model."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_attributes.py",
                "scripts/training/verify_taiji_goal_attributes_10_9.py",
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
                "goals": len(detail),
                "entity_leaks": leaks,
                "resolved": resolved,
                "copy_top1": copy_hits,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

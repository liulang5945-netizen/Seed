"""10.8: does rewording the goal remove the copy shortcut? Measure it with a matcher.

The 10.7 fixture writes the goal as the correct action sentence verbatim. The new
styles in ``taiji/goal_paraphrase.py`` drop the template but keep the names. If a
character-overlap matcher still solves the six-way task, rewording alone changes
nothing and the fixture needs entity descriptions, not synonyms.

Pure string work: no model, no parameters, no training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.goal_paraphrase import STYLES, action_sentence, paraphrase_goal  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

BUDGET_CPU_SECONDS = 60
BUDGET_WALL_SECONDS = 120
COPY_STILL = 32
COPY_BROKEN = 12


def bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}


def overlap(first: str, second: str) -> float:
    left, right = bigrams(first), bigrams(second)
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def named_entities(goal: str) -> tuple[list[str], list[str]]:
    return (
        [obj for obj in OBJECTS if obj in goal],
        [place for place in PLACES if place in goal],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.8 card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    rows = samples()
    per_state = len(rows) // 6
    states = [rows[index * per_state] for index in range(6)]
    moves = [(obj, place) for obj in OBJECTS for place in PLACES]
    sentences = Counter(action_sentence(o, p) for o, p in moves)

    identical = 0
    indeterminate = 0
    per_style = {}
    rows_out = []
    for style in STYLES:
        hits = 0
        total = 0
        for state in states:
            free = next(
                place
                for place in PLACES
                if place not in {position for _, position in state["state"].where}
            )
            for obj in OBJECTS:
                goal = paraphrase_goal(obj, free, style)
                if goal in sentences:
                    identical += 1
                objects, places = named_entities(goal)
                if len(objects) != 1 or len(places) != 1:
                    indeterminate += 1
                scored = sorted(
                    ((overlap(goal, action_sentence(o, p)), (o, p)) for o, p in moves),
                    key=lambda item: (-item[0], item[1]),
                )
                pick = scored[0][1]
                hit = pick == (obj, free)
                hits += int(hit)
                total += 1
                rows_out.append(
                    {
                        "style": style,
                        "goal": goal,
                        "correct_move": [obj, free],
                        "picked": list(pick),
                        "hit": hit,
                        "top_overlap": scored[0][0],
                        "second_overlap": scored[1][0],
                    }
                )
        per_style[style] = {"hits": hits, "total": total}

    hits_total = sum(entry["hits"] for entry in per_style.values())
    total = sum(entry["total"] for entry in per_style.values())
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif hits_total >= COPY_STILL:
        status = "paraphrase_still_copyable"
    elif hits_total <= COPY_BROKEN:
        status = "paraphrase_breaks_copy"
    else:
        status = "paraphrase_mixed"

    report = {
        "schema": "taiji-goal-paraphrase-10-8-v1",
        "step": "10.8",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "styles": list(STYLES),
            "goals": total,
            "actions": len(moves),
            "matcher": "character-bigram Jaccard between goal and action sentence",
            "still_copyable": COPY_STILL,
            "breaks_copy": COPY_BROKEN,
            "chance_top1": 1 / len(moves),
            "training": "none",
        },
        "copy_top1": hits_total,
        "per_style": per_style,
        "identical_to_action_sentence": identical,
        "indeterminate_goals": indeterminate,
        "rows": rows_out,
        "model_involved": False,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "String-level check only: it says whether a trivial overlap matcher still wins "
            "after rewording. It is not a model reading and does not by itself make the "
            "fixture adequate for a capability claim."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/goal_paraphrase.py",
                "scripts/training/verify_taiji_goal_paraphrase_10_8.py",
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
                "copy_top1": hits_total,
                "total": total,
                "per_style": per_style,
                "identical": identical,
                "indeterminate": indeterminate,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""10.10: can the untouched subject use a glossary? Pairwise reading on the fair fixture.

10.9 built goals that name no entity and a glossary that resolves them. The subject has
never been trained on anything like this, so the honest question is whether giving it the
glossary changes its pairwise choice at all:

* same goal, same two moves, prompt = glossary + goal;
* arm A gets the **real** glossary, arm B gets a **reversed** one;
* no parameter changes anywhere.

If both arms sit at chance, the glossary is not being read and no trained head on top of
these features can be expected to use it either.
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

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.native_goal_attributes import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
    describe_goal,
)
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402
from taiji.subject_action_scorer import score_actions  # noqa: E402

BUDGET_CPU_SECONDS = 900
BUDGET_WALL_SECONDS = 1800
PASS_HITS = 5
ABSENT_HITS = 2


def glossary_text(reverse: bool) -> str:
    object_items = list(OBJECT_DESCRIPTIONS.items())
    place_items = list(PLACE_DESCRIPTIONS.items())
    if reverse:
        object_items = list(reversed(object_items))
        place_items = list(reversed(place_items))
    parts = [f"{text}是{entity}" for entity, text in object_items]
    parts += [f"{text}是{entity}" for entity, text in place_items]
    return "；".join(parts) + "。"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.10 card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)

    rows = samples()
    per_state = len(rows) // 6
    states = [rows[index * per_state] for index in range(6)]
    tasks = []
    for index, state in enumerate(states):
        occupied = {position for _, position in state["state"].where}
        free = next(place for place in PLACES if place not in occupied)
        obj = OBJECTS[index % len(OBJECTS)]
        tasks.append(
            {
                "state": state["state"],
                "obj": obj,
                "free": free,
                "goal": describe_goal(obj, free),
                "other": next((o, p) for o in OBJECTS for p in PLACES if (o, p) != (obj, free)),
            }
        )

    results = {}
    for name, reverse in (("real", False), ("reversed", True)):
        prefix = glossary_text(reverse)
        hits = 0
        detail = []
        for task in tasks:
            moves = [(task["obj"], task["free"]), task["other"]]
            scored = score_actions(subject, prefix + task["goal"], moves, provider=provider)
            best = min(scored, key=lambda row: (row["nll"], row["move"]))
            hit = best["move"] == (task["obj"], task["free"])
            hits += int(hit)
            detail.append(
                {
                    "goal": task["goal"],
                    "correct": [task["obj"], task["free"]],
                    "other": list(task["other"]),
                    "picked": list(best["move"]),
                    "hit": hit,
                    "nll_correct": next(
                        row["nll"] for row in scored if row["move"] == (task["obj"], task["free"])
                    ),
                    "nll_other": next(row["nll"] for row in scored if row["move"] == task["other"]),
                }
            )
        results[name] = {"hits": hits, "total": len(tasks), "rows": detail}

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif results["real"]["hits"] >= PASS_HITS and results["reversed"]["hits"] <= ABSENT_HITS:
        status = "glossary_used_untrained"
    elif results["real"]["hits"] <= ABSENT_HITS:
        status = "glossary_not_used"
    else:
        status = "glossary_inconclusive"

    report = {
        "schema": "taiji-glossary-10-10-v1",
        "step": "10.10",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "goals": len(tasks),
            "actions_per_goal": 2,
            "arms": ["real glossary", "reversed glossary"],
            "pass_hits": PASS_HITS,
            "absent_hits": ABSENT_HITS,
            "chance_hits": 0.5,
            "training": "none",
        },
        "parent_model_digest": digest,
        "results": results,
        "model_involved": True,
        "capability_claimed": status == "glossary_used_untrained",
        "training_admitted": False,
        "limits": (
            "Untrained pairwise reading only: six goals, two moves each, glossary prepended "
            "to the prompt. It says whether the existing subject reads a glossary at all; it "
            "is not six-way selection, not task capability and not P2."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_attributes.py",
                "taiji/subject_action_scorer.py",
                "scripts/training/probe_glossary_use_10_10.py",
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
                "real_hits": results["real"]["hits"],
                "reversed_hits": results["reversed"]["hits"],
                "total": len(tasks),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

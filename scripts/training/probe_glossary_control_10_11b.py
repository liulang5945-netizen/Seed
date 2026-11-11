"""10.11b: the control 10.11 is missing -- same tasks, reversed glossary in records.

10.11 measured 6/6 pairwise with the glossary in records and no prompt leakage. A
number that clean needs a null before it means anything: keep the records channel and
the prompt identical, and only reverse the descriptor->entity mapping. If the reversed
arm also wins, the effect is not reading the mapping.
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
from taiji.constrained_learning import career_nll  # noqa: E402
from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.goal_paraphrase import action_sentence  # noqa: E402
from taiji.living_core import RawSource  # noqa: E402
from taiji.native_goal_attributes import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
    describe_goal,
)
from taiji.native_goal_glossary import glossary_records, journal_with_records  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.11b card is write-once")

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
        tasks.append({"obj": obj, "free": free, "goal": describe_goal(obj, free)})

    moves = [(obj, place) for obj in OBJECTS for place in PLACES]

    def wrong_records():
        # A real permutation, not a reordering: each descriptor points at another entity.
        object_items = list(OBJECT_DESCRIPTIONS.items())
        place_items = list(PLACE_DESCRIPTIONS.items())
        lines = [
            f"{text}是{object_items[(index + 1) % len(object_items)][0]}。"
            for index, (_, text) in enumerate(object_items)
        ]
        lines += [
            f"{text}是{place_items[(index + 1) % len(place_items)][0]}。"
            for index, (_, text) in enumerate(place_items)
        ]
        return tuple(
            RawSource(source_id=f"wrong-{index}", raw=line.encode())
            for index, line in enumerate(lines)
        )

    arms = {}
    for name, build in (("real", glossary_records), ("reversed", wrong_records)):
        records = build()
        hits = 0
        detail = []
        for task in tasks:
            correct = (task["obj"], task["free"])
            other = next((o, p) for o, p in moves if (o, p) != correct)
            scores = {}
            for candidate in (correct, other):
                journal = journal_with_records(
                    task["goal"],
                    action_sentence(*candidate),
                    f"c-{candidate[0]}-{candidate[1]}",
                    records,
                )
                scores[candidate] = float(career_nll(subject, journal, provider).detach())
            hit = scores[correct] < scores[other]
            hits += int(hit)
            detail.append(
                {
                    "goal": task["goal"],
                    "correct": list(correct),
                    "other": list(other),
                    "hit": hit,
                    "nll_correct": scores[correct],
                    "nll_other": scores[other],
                }
            )
        arms[name] = {"hits": hits, "total": len(tasks), "rows": detail}

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif arms["reversed"]["hits"] >= 5:
        status = "control_also_wins"
    elif arms["real"]["hits"] - arms["reversed"]["hits"] >= 3:
        status = "records_mapping_used"
    else:
        status = "records_mapping_unresolved"

    report = {
        "schema": "taiji-glossary-control-10-11b-v1",
        "step": "10.11b",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "goals": len(tasks),
            "arms": ["real records", "reversed records"],
            "decision": "control_also_wins if reversed >=5; mapping_used if real - reversed >=3",
            "training": "none",
        },
        "parent_model_digest": digest,
        "arms": arms,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Null control only: it decides whether 10.11's 6/6 reflects the mapping or "
            "something generic about records. Nothing is trained."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_glossary.py",
                "scripts/training/probe_glossary_control_10_11b.py",
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
                "real_hits": arms["real"]["hits"],
                "reversed_hits": arms["reversed"]["hits"],
                "total": len(tasks),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

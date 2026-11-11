"""10.11: is the glossary-on-records channel live, and does the prompt stay copy-proof?

The training the owner authorised must not be spent on a fixture that leaks. This step
checks two things without training:

* the prompt is still entity-free, so a single-hop overlap matcher cannot win;
* feeding the glossary through ``records`` actually changes the subject's numbers --
  otherwise the channel is inert and any trained head would be reading nothing.
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
from taiji.native_goal_attributes import describe_goal  # noqa: E402
from taiji.native_goal_glossary import glossary_records, journal_with_records  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200
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
        raise FileExistsError("10.11 card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    records = glossary_records()

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
    copy_hits = 0
    differences = 0
    comparisons = 0
    hits = 0
    detail = []
    for task in tasks:
        scored = sorted(
            ((overlap(task["goal"], action_sentence(o, p)), (o, p)) for o, p in moves),
            key=lambda item: (-item[0], item[1]),
        )
        copy_hits += int(scored[0][1] == (task["obj"], task["free"]))

        correct = (task["obj"], task["free"])
        other = next((o, p) for o, p in moves if (o, p) != correct)
        with_records = {}
        without_records = {}
        for candidate in (correct, other):
            sentence = action_sentence(*candidate)
            journal = journal_with_records(
                task["goal"], sentence, f"g-{candidate[0]}-{candidate[1]}", records
            )
            plain = journal_with_records(
                task["goal"], sentence, f"g-{candidate[0]}-{candidate[1]}", ()
            )
            with_records[candidate] = float(career_nll(subject, journal, provider).detach())
            without_records[candidate] = float(career_nll(subject, plain, provider).detach())
            comparisons += 1
            if with_records[candidate] != without_records[candidate]:
                differences += 1
        hit = with_records[correct] < with_records[other]
        hits += int(hit)
        detail.append(
            {
                "goal": task["goal"],
                "correct": list(correct),
                "other": list(other),
                "hit_with_records": hit,
                "nll_with": [with_records[correct], with_records[other]],
                "nll_without": [without_records[correct], without_records[other]],
            }
        )

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif copy_hits > COPY_ALLOWED:
        status = "fairness_regressed"
    elif differences == 0:
        status = "records_channel_inert"
    else:
        status = "records_channel_live"

    report = {
        "schema": "taiji-glossary-records-10-11-v1",
        "step": "10.11",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "goals": len(tasks),
            "records": len(records),
            "copy_allowed": COPY_ALLOWED,
            "prompt": "goal only, entity-free",
            "records_channel": "CareerJournal.records via dataclasses.replace",
            "training": "none",
        },
        "parent_model_digest": digest,
        "copy_top1": copy_hits,
        "records_comparisons": comparisons,
        "records_differences": differences,
        "pairwise_hits_with_records": hits,
        "rows": detail,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Channel check only: it proves the glossary reaches the subject through records "
            "and that the prompt alone is copy-proof. The pairwise hits are a floor reading, "
            "not a capability: nothing is trained and no parameter moves."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_glossary.py",
                "taiji/native_goal_attributes.py",
                "scripts/training/verify_taiji_glossary_records_10_11.py",
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
                "copy_top1": copy_hits,
                "records_differences": differences,
                "records_comparisons": comparisons,
                "pairwise_hits": hits,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

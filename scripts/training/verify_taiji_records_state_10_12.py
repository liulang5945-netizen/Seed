"""10.12: do the glossary records reach the subject's recurrent state at all?

10.11b showed the untrained subject answers 6/6 whether the glossary maps descriptors
correctly or by a permutation. Before spending the authorised training run, check the
necessary condition: with the same goal, does the *state* differ between the real and
the permuted glossary? If the state is identical, records never reach any feature a
head could be trained on, and the run would be empty.

No parameter changes; only forward passes on the goal prompt.
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
from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.living_core import RawSource  # noqa: E402
from taiji.native_goal_attributes import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
    describe_goal,
)
from taiji.native_goal_glossary import glossary_records, journal_with_records  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402
from taiji.state_migration import run_career  # noqa: E402

BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
RELATIVE_FLOOR = 1e-9


def permuted_records() -> tuple[RawSource, ...]:
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
        RawSource(source_id=f"permuted-{index}", raw=line.encode())
        for index, line in enumerate(lines)
    )


def relative_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    scale = float(first.norm()) + float(second.norm())
    if scale == 0.0:
        return 0.0
    return 2.0 * float((first - second).norm()) / scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.12 card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)

    rows = samples()
    per_state = len(rows) // 6
    states = [rows[index * per_state] for index in range(6)]
    real = glossary_records()
    wrong = permuted_records()

    differences = 0
    record_effects = 0
    detail = []
    with torch.no_grad():
        for index, state in enumerate(states):
            occupied = {position for _, position in state["state"].where}
            free = next(place for place in PLACES if place not in occupied)
            obj = OBJECTS[index % len(OBJECTS)]
            goal = describe_goal(obj, free)
            state_h = {}
            for name, records in (("plain", ()), ("real", real), ("wrong", wrong)):
                journal = journal_with_records(goal, "x", f"g{index}-{name}", records)
                _, steps = run_career(
                    subject, journal, growth_gate_provider=provider, retain_outputs=True
                )
                last_patch = max(
                    position for position, (_, byte) in enumerate(steps) if byte is None
                )
                state_h[name] = steps[last_patch][0].state.h.detach()
            same_as_wrong = bool(torch.equal(state_h["real"], state_h["wrong"]))
            differences += int(not same_as_wrong)
            record_effects += int(not torch.equal(state_h["real"], state_h["plain"]))
            detail.append(
                {
                    "goal": goal,
                    "state_differs_when_mapping_changes": not same_as_wrong,
                    "records_change_state": not torch.equal(state_h["real"], state_h["plain"]),
                    "relative_distance_real_vs_wrong": relative_distance(
                        state_h["real"], state_h["wrong"]
                    ),
                    "relative_distance_real_vs_plain": relative_distance(
                        state_h["real"], state_h["plain"]
                    ),
                }
            )

    distances = sorted(row["relative_distance_real_vs_wrong"] for row in detail)
    median = distances[len(distances) // 2]
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif differences == len(detail) and median > RELATIVE_FLOOR:
        status = "records_reach_state"
    elif differences == 0:
        status = "records_do_not_reach_state"
    else:
        status = "records_partially_reach_state"

    report = {
        "schema": "taiji-records-state-10-12-v1",
        "step": "10.12",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "tasks": len(detail),
            "comparison": "goal prompt only; real vs permuted glossary in records",
            "relative_floor": RELATIVE_FLOOR,
            "training": "none",
        },
        "parent_model_digest": digest,
        "mapping_differences": differences,
        "records_change_state": record_effects,
        "median_relative_distance": median,
        "rows": detail,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Necessary-condition check only: if the state moves when the mapping changes, a "
            "head on that state could in principle read it. It does not show that any head "
            "does, and nothing is trained."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_glossary.py",
                "scripts/training/verify_taiji_records_state_10_12.py",
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
                "mapping_differences": differences,
                "tasks": len(detail),
                "median_relative_distance": round(median, 9),
                "records_change_state": record_effects,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

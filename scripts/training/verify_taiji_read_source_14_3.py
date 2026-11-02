"""14.3: does the subject's own read get *which record to read* from the goal?

Pre-registered in plans/reference/SPEC-TAIJI-CORE_READ_SOURCE_14_3_20261102.md. Read-only:
no optimizer, no gradients, no parameter changes, no weight written. The face is the 10.21
composition fixture's dev worlds -- the only leak-free selection fixture in the repo, since
14.2 showed the 10.7 face writes the answer into the prompt.

Three stages, each separately falsifiable, each reported per attention group because
``EpisodicRead`` runs one softmax per group (averaging groups would hide "one group selects,
three average"):

* D1 query side   - changing the descriptor changes the query and the attention;
* D2 selection    - the attention puts most mass on the record that descriptor owns;
* D3 value reach  - with the table cut to that one record, the value path carries the entity
                    it maps to (controls swap only the entity, or only the descriptor).

The ruler's lower bound is a machine anchor rather than a hand-made distribution: with no
records at all the read must return exactly ``EpisodicRead.null_value``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest, load_candidate  # noqa: E402
from taiji.composition_read_probe import descriptor_read  # noqa: E402
from taiji.living_core import RawSource  # noqa: E402
from taiji.native_composition_fixture import build_split  # noqa: E402
from taiji.slot_composition_head import read_slot, relative_distance  # noqa: E402
from taiji.state_migration import resume_subject  # noqa: E402

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
DEV_WORLDS = 4
BUDGET_CPU_SECONDS = 90
BUDGET_WALL_SECONDS = 240
RESERVE_CPU_SECONDS = 30
PRESENT_BAR = 0.40
ABSENT_BAR = 0.10
D3_REACH_BAR = 0.50
D3_UNREACH_BAR = 0.20
D1_FLOOR = 1e-3
STOP_CPU = BUDGET_CPU_SECONDS - RESERVE_CPU_SECONDS

STATUS_MEANING = {
    "query_side_dead": "D1 fails: the descriptor moves neither the query nor the attention. "
    "Wiring or encoding, not selection.",
    "pathway_unreachable": "D3 fails: the value side cannot carry entity identity, so no "
    "readout head, sample size or training dose can rescue this read.",
    "conditional_reads_present": "D2 clears the 0.40 bar read-only. That contradicts the "
    "recorded 0.1125 of trained 10.21.4, so the instrument is suspect first.",
    "selection_bottleneck": "D1 and D3 hold while D2 fails: the query carries information and "
    "the value path carries the entity, so the missing piece is selection itself.",
    "not_resolved": "the stages do not line up into one declared shape, or a reading has no "
    "dynamic range.",
    "budget_stop": "the declared CPU or wall allowance was exceeded.",
    "engineering_failure": "an exception escaped the measurement.",
}


def world_descriptors(world) -> list[tuple[str, str]]:
    return [
        *zip(world.object_descriptions, world.object_names, strict=True),
        *zip(world.place_descriptions, world.place_names, strict=True),
    ]


def rotate(pairs):
    names = [name for _, name in pairs]
    return [
        (description, names[(index + 1) % len(names)])
        for index, (description, _) in enumerate(pairs)
    ]


def lines(mapping) -> tuple[RawSource, ...]:
    return tuple(
        RawSource(source_id=f"line-{index}", raw=f"{description}是{name}。".encode())
        for index, (description, name) in enumerate(mapping)
    )


def group_entropy(weights: torch.Tensor, positions: int) -> float:
    """One group's softmax entropy against its own uniform ceiling, log(positions)."""
    positive = weights[weights > 0]
    entropy = float(-(positive * positive.log()).sum())
    return entropy / math.log(positions) if positions > 1 else 0.0


def measure(model, state, groups: int, readings: dict) -> None:
    cells, d1_rows, d2_rows, d3_rows, null_rows = [], [], [], [], []
    for built in build_split("dev", DEV_WORLDS):
        pairs = world_descriptors(built)
        target_of = {description: index for index, (description, _) in enumerate(pairs)}
        conditions = {
            "real": lines(pairs),
            "rotated": lines(rotate(pairs)),
            "reversed": lines(pairs[::-1]),
        }
        slots: dict[tuple, tuple] = {}
        for condition, records in conditions.items():
            positions = sum(len(record.raw) for record in records) + 1
            for description, _name in pairs:
                reading = descriptor_read(model, state, description, records)
                slot, attention = read_slot(model, state, description, records)
                slots[(condition, description)] = (
                    slot.detach().reshape(groups, -1),
                    attention.detach().reshape(groups, -1),
                )
                reading.update(
                    {"descriptor": description, "condition": condition, "world": built.world}
                )
                cells.append(reading)
                if condition == "real":
                    d2_rows.append(
                        {
                            "world": built.world,
                            "descriptor": description,
                            "group_hits": [
                                bool(row["top1_is_target"]) for row in reading["groups"]
                            ],
                            "null_mass": [float(row["null_mass"]) for row in reading["groups"]],
                            "normalised_entropy": [
                                group_entropy(slots[(condition, description)][1][group], positions)
                                for group in range(groups)
                            ],
                        }
                    )
        for description, name in pairs:
            own = slots[("real", description)]
            for other, _ in pairs:
                if other == description:
                    continue
                rival = slots[("real", other)]
                d1_rows.append(
                    {
                        "kind": "other_descriptor",
                        "descriptor": description,
                        "slot_relative_distance": relative_distance(own[0], rival[0]),
                        "attention_l1": float((own[1] - rival[1]).abs().sum()),
                    }
                )
            rotated = slots[("rotated", description)]
            d1_rows.append(
                {
                    "kind": "same_descriptor_rotated_mapping",
                    "descriptor": description,
                    "slot_relative_distance": relative_distance(own[0], rotated[0]),
                    "attention_l1": float((own[1] - rotated[1]).abs().sum()),
                }
            )
            index = target_of[description]
            other_description, other_name = pairs[(index + 1) % len(pairs)]
            single = lines([(description, name)])
            slot_own, _ = read_slot(model, state, description, single)
            slot_entity, _ = read_slot(
                model, state, description, lines([(description, other_name)])
            )
            slot_rival, _ = read_slot(
                model, state, other_description, lines([(other_description, name)])
            )
            reading = descriptor_read(model, state, description, single)
            d3_rows.append(
                {
                    "world": built.world,
                    "descriptor": description,
                    "entity_only_distance": relative_distance(slot_own, slot_entity),
                    "descriptor_only_distance": relative_distance(slot_own, slot_rival),
                    "record_mass": [float(row["target_mass"]) for row in reading["groups"]],
                    "null_mass": [float(row["null_mass"]) for row in reading["groups"]],
                }
            )
        slot_null, _ = read_slot(model, state, pairs[0][0], lines([]))
        null_anchor = model.episodic.null_value.detach()
        null_rows.append(
            {
                "world": built.world,
                "max_abs_difference_from_null_value": float(
                    (slot_null.detach() - null_anchor.expand_as(slot_null.detach())).abs().max()
                ),
            }
        )

    per_group = [
        sum(row["group_hits"][group] for row in d2_rows) / len(d2_rows) for group in range(groups)
    ]
    readings.update(
        {
            "cells": len(cells),
            "d1_rows": len(d1_rows),
            "d1_median_slot_distance": statistics.median(
                [row["slot_relative_distance"] for row in d1_rows]
            ),
            "d1_min_slot_distance": min(row["slot_relative_distance"] for row in d1_rows),
            "d1_median_attention_l1": statistics.median([row["attention_l1"] for row in d1_rows]),
            "d1_mapping_rotation_slot_distance": statistics.median(
                [
                    row["slot_relative_distance"]
                    for row in d1_rows
                    if row["kind"] == "same_descriptor_rotated_mapping"
                ]
            ),
            "d2_top1_by_group": per_group,
            "d2_mean_over_groups": sum(per_group) / groups,
            "d2_entropy_by_group": [
                statistics.median([row["normalised_entropy"][group] for row in d2_rows])
                for group in range(groups)
            ],
            "d2_null_mass_median_by_group": [
                statistics.median([row["null_mass"][group] for row in d2_rows])
                for group in range(groups)
            ],
            "d3_median_entity_only": statistics.median(
                [row["entity_only_distance"] for row in d3_rows]
            ),
            "d3_median_descriptor_only": statistics.median(
                [row["descriptor_only_distance"] for row in d3_rows]
            ),
            "d3_record_mass_above_null_share": sum(
                1
                for row in d3_rows
                for group in range(groups)
                if row["record_mass"][group] > row["null_mass"][group]
            )
            / (len(d3_rows) * groups),
            "null_anchor_max_abs_difference": max(
                row["max_abs_difference_from_null_value"] for row in null_rows
            ),
            "d1_examples": d1_rows[:4],
            "d2_examples": d2_rows[:4],
            "d3_examples": d3_rows[:4],
            "null_anchor_rows": null_rows,
        }
    )


def verdict(readings: dict) -> tuple[str, str]:
    d1 = (
        readings["d1_median_slot_distance"] > D1_FLOOR
        and readings["d1_median_attention_l1"] > D1_FLOOR
    )
    hits = readings["d2_top1_by_group"]
    d2_present = min(hits) >= PRESENT_BAR
    d2_absent = max(hits) <= ABSENT_BAR
    d3_reachable = (
        readings["d3_median_entity_only"] >= D3_REACH_BAR
        and readings["d3_median_descriptor_only"] >= D3_REACH_BAR
        and readings["d3_record_mass_above_null_share"] > 0.5
    )
    d3_unreachable = readings["d3_median_entity_only"] <= D3_UNREACH_BAR
    if not d1:
        return "query_side_dead", STATUS_MEANING["query_side_dead"]
    if d3_unreachable:
        return "pathway_unreachable", STATUS_MEANING["pathway_unreachable"]
    if d2_present:
        return "conditional_reads_present", STATUS_MEANING["conditional_reads_present"]
    if d2_absent and d3_reachable:
        return "selection_bottleneck", STATUS_MEANING["selection_bottleneck"]
    return (
        "not_resolved",
        STATUS_MEANING["not_resolved"]
        + f" (d1={d1}, d2_top1_by_group={hits}, d2_absent={d2_absent}, d3_reachable={d3_reachable})",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("14.3 card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    readings: dict = {}
    manifest: dict = {}
    groups = 0
    status, note = "not_resolved", ""
    try:
        bundle = ROOT / BUNDLE
        manifest = build_manifest(ROOT, bundle)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
            )
            _loaded, payload = load_candidate(ROOT, manifest_path, bundle, "frozen_eval")
        model, state, _optimizer = resume_subject(payload)
        groups = model.config.groups
        measure(model, state, groups, readings)
        readings["cpu_per_read_estimate"] = (time.process_time() - total_cpu) / max(
            readings["cells"], 1
        )
        status, note = verdict(readings)
    except Exception as error:  # engineering_failure is a declared exit
        status, note = "engineering_failure", f"{type(error).__name__}: {error}"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status, note = "budget_stop", STATUS_MEANING["budget_stop"]
    elif cpu_seconds > STOP_CPU:
        note = f"{note} [stop judgement fired at {cpu_seconds:.1f}s of {BUDGET_CPU_SECONDS}s]"

    report = {
        "schema": "taiji-read-source-14-3-v1",
        "step": "14.3",
        "date": "2026-11-02",
        "supersedes": "reports/taiji_core_read_source_14_3_20261102_v1_engineering_failure.json",
        "status": status,
        "note": note,
        "candidate_id": manifest.get("candidate_id"),
        "attention_groups": groups,
        "dev_worlds_used": DEV_WORLDS,
        "bars": {
            "d1_floor": D1_FLOOR,
            "d2_present": PRESENT_BAR,
            "d2_absent": ABSENT_BAR,
            "d3_reach": D3_REACH_BAR,
            "d3_unreachable": D3_UNREACH_BAR,
        },
        "readings": readings,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "reserve_cpu_seconds": RESERVE_CPU_SECONDS,
        "model_involved": True,
        "parameters_changed": False,
        "optimizer_calls": 0,
        "capability_claimed": False,
        "training_admitted": False,
        "p1_claimed": False,
        "p2_claimed": False,
        "status_meanings": STATUS_MEANING,
        "limits": (
            "Read-only mechanism reading on one synthetic fixture. It produces no output bytes, "
            "so it says nothing about P1 (Chinese expression); it trains nothing, so it says "
            "nothing about P2. The 10.7 face is excluded because 14.1/14.2 showed the goal "
            "equals the correct action sentence there."
        ),
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/composition_read_probe.py",
                "taiji/slot_composition_head.py",
                "taiji/native_composition_fixture.py",
                "taiji/candidate_manifest.py",
                "scripts/training/verify_taiji_read_source_14_3.py",
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
                "cells": readings.get("cells"),
                "d1_slot": readings.get("d1_median_slot_distance"),
                "d1_attention_l1": readings.get("d1_median_attention_l1"),
                "d2_top1_by_group": readings.get("d2_top1_by_group"),
                "d2_entropy_by_group": readings.get("d2_entropy_by_group"),
                "d3_entity_only": readings.get("d3_median_entity_only"),
                "d3_descriptor_only": readings.get("d3_median_descriptor_only"),
                "d3_mass_above_null": readings.get("d3_record_mass_above_null_share"),
                "null_anchor": readings.get("null_anchor_max_abs_difference"),
                "cpu_per_read": readings.get("cpu_per_read_estimate"),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

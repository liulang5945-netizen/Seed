"""10.21a: do the two slots move when the glossary mapping changes?

The authorised first step of 10.21: implement the slot read (done, reusing the model's
own ``EpisodicRead``) and check the necessary condition before any training.

* same descriptor, real vs permuted glossary -> the slot must differ;
* two different descriptors -> the slot must differ more (the descriptor has to matter).

No parameter changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest, load_candidate  # noqa: E402
from taiji.living_core import RawSource  # noqa: E402
from taiji.native_goal_attributes import OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS  # noqa: E402
from taiji.slot_composition_head import read_slot, relative_distance  # noqa: E402
from taiji.state_migration import resume_subject  # noqa: E402

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
MAPPING_FLOOR = 1e-6


def glossary(permuted: bool) -> tuple[RawSource, ...]:
    items = [
        *[(text, entity) for entity, text in OBJECT_DESCRIPTIONS.items()],
        *[(text, entity) for entity, text in PLACE_DESCRIPTIONS.items()],
    ]
    if permuted:
        entities = [entity for _, entity in items]
        items = [
            (text, entities[(index + 1) % len(entities)]) for index, (text, _) in enumerate(items)
        ]
    return tuple(
        RawSource(source_id=f"g{index}", raw=f"{text}是{entity}。".encode())
        for index, (text, entity) in enumerate(items)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.21a card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    bundle = ROOT / BUNDLE
    manifest = build_manifest(ROOT, bundle)
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _manifest, payload = load_candidate(ROOT, manifest_path, bundle, "frozen_eval")
    model, state, _optimizer = resume_subject(payload)
    real = glossary(False)
    permuted = glossary(True)

    rows = []
    moved = 0
    descriptors = [*OBJECT_DESCRIPTIONS.values(), *PLACE_DESCRIPTIONS.values()]
    slots_real = {}
    for descriptor in descriptors:
        slot_real, _ = read_slot(model, state, descriptor, real)
        slot_perm, _ = read_slot(model, state, descriptor, permuted)
        distance = relative_distance(slot_real, slot_perm)
        moved += int(distance > MAPPING_FLOOR)
        slots_real[descriptor] = slot_real
        rows.append(
            {
                "descriptor": descriptor,
                "distance_real_vs_permuted": distance,
                "moved": distance > MAPPING_FLOOR,
            }
        )

    first, second = descriptors[0], descriptors[1]
    across = relative_distance(slots_real[first], slots_real[second])
    within_max = max(row["distance_real_vs_permuted"] for row in rows)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif moved == len(descriptors) and across > within_max:
        status = "slots_respond_to_mapping"
    elif moved == 0:
        status = "slots_ignore_mapping"
    else:
        status = "slots_partially_respond"

    report = {
        "schema": "taiji-slot-channel-10-21a-v1",
        "step": "10.21a",
        "date": "2026-11-01",
        "status": status,
        "candidate_id": manifest["candidate_id"],
        "descriptors": len(descriptors),
        "moved_by_mapping": moved,
        "max_mapping_distance": within_max,
        "descriptor_contrast": across,
        "mapping_floor": MAPPING_FLOOR,
        "rows": rows,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Channel check only: the slots are the model's own episodic read, and the test "
            "asks whether the glossary mapping moves them. It does not train, does not score "
            "actions, and does not show composition works."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/slot_composition_head.py",
                "taiji/candidate_manifest.py",
                "scripts/training/verify_taiji_slot_channel_10_21a.py",
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
                "moved": moved,
                "of": len(descriptors),
                "max_mapping_distance": within_max,
                "descriptor_contrast": across,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

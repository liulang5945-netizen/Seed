"""First-A fixture v2: every split draws three-body worlds.

The v1 fixture (`first_a_support_22_5`) draws train worlds with 1-2 bodies and
every fit world with 3, so a train/fit difference conflates "learned to read" with
"generalises to an unseen body count".  D-20261117-47 recorded that as an
unseparated confound; this module removes it by holding the body count fixed at 3
for all splits.  The label, eligibility and oracle logic are identical to v1, and
the seed derivation is shared, so a world index keeps its seed and only the body
count changes.

The v1 module stays untouched: historical runs keep reproducing bit for bit.
"""

from __future__ import annotations

import hashlib

import torch

from scripts.training.first_a_support_22_5 import (
    GOAL,
    LABELS,
    RGB_COLORS,
    seed_for,
)
from taiji.u2_history import value_digest
from taiji.u2_task_episode import NAMES, locate_public, position_word
from taiji.u2_task_world import PhysicalWorld

BODIES = 3
FIXTURE_ID = "taiji-first-A-v2-3body"


def generate_examples(partition, count):
    if partition not in {"train", "fit", "price"} or count % 4:
        raise ValueError("unsupported_first_A_partition")
    examples = []
    for index in range(count):
        label = LABELS[index % 4]
        for attempt in range(256):
            seed = seed_for(partition, index, attempt)
            world = PhysicalWorld.reset(seed, {"body_count": BODIES})
            result = world.elapse("public-scene")
            bodies = world.snapshot()["bodies"]
            names = [NAMES[RGB_COLORS.index(tuple(body["color"]))] for body in bodies]
            if label == "未知":
                absent = [name for name in NAMES if name not in names]
                entity = absent[seed % len(absent)]
            else:
                eligible = [
                    i
                    for i, body in enumerate(bodies)
                    if position_word(body["position"][0]) == label
                    and min(abs(body["position"][0] + 1 / 3), abs(body["position"][0] - 1 / 3))
                    >= 0.1
                ]
                if not eligible:
                    continue
                entity = names[eligible[seed % len(eligible)]]
            estimated = locate_public(result["raw_media"]["rgb"], entity)
            oracle = "未知" if estimated is None else position_word(estimated)
            if oracle != label:
                continue
            public = {
                "goal": GOAL,
                "request": f"{entity}现在在哪里？".encode(),
                "observations": result["observations"],
                "coordinates": result["coordinates"],
                "outcome": result["execution_outcome"],
                "action": torch.zeros(15),
                "t_start": result["t_start"],
                "t_end": result["t_end"],
            }
            examples.append(
                {
                    "id": f"{partition}-{index}",
                    "public": public,
                    "target": label.encode(),
                    "audit": {
                        "seed": seed,
                        "attempt": attempt,
                        "objects": BODIES,
                        "fixture": FIXTURE_ID,
                        "world_digest": value_digest(world.snapshot()),
                        "oracle": oracle,
                    },
                }
            )
            break
        else:
            raise ValueError("first_A_v2_fixture_generation_exhausted")
    return examples


def fixture_digest(partition: str, count: int) -> str:
    """Stable digest of a split's public content, for the run receipt."""
    rows = []
    for example in generate_examples(partition, count):
        rows.append(
            {
                "id": example["id"],
                "request": example["public"]["request"].hex(),
                "target": example["target"].hex(),
                "world": example["audit"]["world_digest"],
            }
        )
    return hashlib.sha256(repr(sorted(rows, key=lambda row: row["id"])).encode()).hexdigest()

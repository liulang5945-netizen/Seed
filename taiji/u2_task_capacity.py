"""22.4.2.4 D capacity-pressure fixture: 6 entity observations, 2 public frames.

The world contract allows at most three visible bodies per world, so D uses two
three-body sub-worlds to expose six entity observations (capacity 4). Each
sub-world runs the registered 48-round course and releases two public frames; the
legal public oracle produces current/previous checks, with the two sub-worlds
interleaved for 12 checks per seed. The fixture also has an occlusion arm and a
same-initial-appearance/different-dynamics pair. No model is loaded and no
optimizer runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .u2_task_contract import request_text
from .u2_task_episode import (
    NAMES,
    PALETTE,
    PhysicalWorld,
    locate_public,
    position_word,
    public_oracle,
    structured_world,
)

CAPACITY = 4
POSITIONS = [(-0.6, 0.0), (0.0, 0.0), (0.6, 0.0)]
ALTERED_MASS = 2.0
ALTERED_DRAG = 2.0
ROUNDS = 48


def _course_action(index: int) -> torch.Tensor:
    if index < 16:
        target, force = -0.6, -0.1
    elif index < 30:
        target, force = 0.0, -0.1
    else:
        target, force = 0.6, 0.1
    action = torch.zeros(15, dtype=torch.float64)
    action[0] = 1
    action[9] = target
    action[10] = 0.0
    action[11] = force
    action[13] = 0.15
    action[14] = 0.04
    return action


def _snapshot_positions(snapshot: dict[str, Any], names: list[str]) -> dict[str, str]:
    return {
        name: position_word(snapshot["bodies"][index]["position"][0])
        for index, name in enumerate(names)
    }


def _occlude(image: torch.Tensor, entity: str) -> torch.Tensor:
    pixels = image.detach().cpu().numpy().astype(np.float64) / 255
    distances = np.linalg.norm(pixels[..., None, :] - PALETTE, axis=-1)
    mask = torch.from_numpy(distances.argmin(-1) == NAMES.index(entity))
    out = image.clone()
    out[mask] = 0
    return out


def _run_world(seed: int, mass: float, drag: float) -> dict[str, Any]:
    world, names = structured_world(seed, POSITIONS)
    state = world.snapshot()
    for body in state["bodies"]:
        body["mass"] = mass
        body["drag"] = drag
    world = PhysicalWorld.restore(state)
    identity = world.snapshot()["world_id"]
    first = world.elapse("capacity-first")
    first_snapshot = world.snapshot()
    for index in range(ROUNDS - 1):
        result = world.step(f"capacity-{index}", _course_action(index))
    return {
        "world_id": identity,
        "names": names,
        "first_snapshot": first_snapshot,
        "last_snapshot": world.snapshot(),
        "images": [first["raw_media"]["rgb"], result["raw_media"]["rgb"]],
        "first_frame": first["raw_media"]["rgb"],
        "last_frame": result["raw_media"]["rgb"],
    }


def _checks(run: dict[str, Any], occlude_entity: str | None) -> list[dict[str, Any]]:
    images = list(run["images"])
    if occlude_entity is not None:
        images[1] = _occlude(images[1], occlude_entity)
    first = _snapshot_positions(run["first_snapshot"], run["names"])
    last = _snapshot_positions(run["last_snapshot"], run["names"])
    checks = []
    for name in run["names"]:
        first_visible = locate_public(images[0], name)
        last_visible = locate_public(images[1], name)
        for temporal in ("current", "previous"):
            if temporal == "current":
                visible = last_visible
                expected = "未知" if visible is None else last[name]
            else:
                visible = first_visible
                expected = "未知" if visible is None else first[name]
            answer = public_oracle(request_text(name, temporal), images).decode()
            checks.append(
                {
                    "entity": name,
                    "temporal": temporal,
                    "expected": expected,
                    "answer": answer,
                }
            )
    return checks


def _interleave(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for left, right in zip(first, second, strict=True):
        merged.extend((left, right))
    return merged


def _ownership(seed: int) -> dict[str, Any]:
    world, names = structured_world(seed, POSITIONS)
    masses = world.contract["physics"]["masses"]
    drags = world.contract["physics"]["drag_rates"]
    if ALTERED_MASS not in masses or ALTERED_DRAG not in drags:
        raise ValueError("capacity_altered_material_not_registered")
    return {"names": names, "masses": masses, "drags": drags}


def capacity_arms(seed: int) -> dict[str, dict[str, Any]]:
    """Build standard, occlusion and dynamics arms from shared baseline rollouts."""
    ownership = _ownership(seed)
    baseline_a = _run_world(seed, 1.0, 1.0)
    baseline_b = _run_world(seed + 500, 1.0, 1.0)
    standard_checks = _interleave(_checks(baseline_a, None), _checks(baseline_b, None))
    occluded = baseline_b["names"][0]
    occlusion_checks = _interleave(_checks(baseline_a, None), _checks(baseline_b, occluded))
    altered = _run_world(seed, ALTERED_MASS, ALTERED_DRAG)
    base = {
        "seed": seed,
        "capacity": CAPACITY,
        "entities": 6,
        "worlds": 2,
        "world_body_limit": 3,
        "frames": 2,
        "rounds": ROUNDS,
        "facts_per_frame": 6,
        "observed_facts": 12,
        "pressure_ratio": 12 / CAPACITY,
        "names_a": baseline_a["names"],
        "names_b": baseline_b["names"],
        "registered_names": ownership["names"],
    }
    return {
        "standard": {
            **base,
            "arm": "standard",
            "checks": standard_checks,
        },
        "occlusion": {
            **base,
            "arm": "occlusion",
            "occluded_entity": occluded,
            "checks": occlusion_checks,
        },
        "dynamics": {
            **base,
            "arm": "dynamics",
            "altered_mass": ALTERED_MASS,
            "altered_drag": ALTERED_DRAG,
            "baseline_checks": _checks(baseline_a, None),
            "altered_checks": _checks(altered, None),
            "first_frames_equal": bool(
                np.array_equal(baseline_a["first_frame"].numpy(), altered["first_frame"].numpy())
            ),
            "last_answers_differ": any(
                left["answer"] != right["answer"]
                for left, right in zip(
                    _checks(baseline_a, None), _checks(altered, None), strict=True
                )
            ),
        },
    }


def capacity_fixture(seed: int, arm: str = "standard") -> dict[str, Any]:
    if arm not in {"standard", "occlusion", "dynamics"}:
        raise ValueError("unknown_capacity_arm")
    return capacity_arms(seed)[arm]

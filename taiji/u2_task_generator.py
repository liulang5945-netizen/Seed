"""22.4.2.6 task-family generator entry, shape schema and miniature recovery.

The generator bundles the frozen task case with a world/executor/model raw
observation packet. It does not train or forward the model. A/B/C/D/J share one
envelope schema, the same feedback-rights table, and a capsule that can be saved,
loaded and rebuilt bitwise from seed plus family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .u2_history import assert_values_equal, atomic_load, atomic_save, value_digest
from .u2_task_capacity import (
    POSITIONS as CAPACITY_POSITIONS,
)
from .u2_task_capacity import (
    ROUNDS as CAPACITY_ROUNDS,
)
from .u2_task_capacity import (
    _checks as capacity_checks,
)
from .u2_task_capacity import (
    _course_action as capacity_course_action,
)
from .u2_task_capacity import (
    _interleave as capacity_interleave,
)
from .u2_task_contract import Case, Fact, Request, request_text, validate_case
from .u2_task_episode import (
    NAMES,
    PALETTE,
    acquisition_fixture,
    joint_fixture,
    locate_public,
    position_word,
    structured_world,
    temporal_fixture,
)
from .u2_task_world import PhysicalWorld

SCHEMA = "taiji-u2-task-sample-v1"
CAPSULE_SCHEMA = "taiji-u2-task-sample-capsule-v1"
FAMILIES = ("A", "B", "C", "D", "J")
GOAL = "记住可核验观测的位置，并按时间要求回答。".encode()
FEEDBACK_RIGHTS = {
    "observation": ("perception", "online", "development"),
    "train_label": ("development",),
    "online_correction": ("online",),
    "help_reply": ("online",),
    "score_only": ("scoring",),
}
SOURCE_FILES = (
    "u2_task_generator.py",
    "u2_task_world.py",
    "u2_task_sensors.py",
    "u2_task_episode.py",
    "u2_task_capacity.py",
    "u2_task_contract.py",
)


def source_pins() -> dict[str, str]:
    return {
        name: hashlib.sha256(
            (Path(__file__).parent / name).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
        for name in SOURCE_FILES
    }


def _zero_action() -> torch.Tensor:
    action = torch.zeros(15, dtype=torch.float64)
    action[0] = 1
    return action


def _tensor_info(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }


def _byte_info(value: bytes) -> dict[str, Any]:
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def _observation(result: dict[str, Any]) -> dict[str, Any]:
    observations = result.get("observations") or {}
    coordinates = result.get("coordinates") or {}
    if set(observations) != {"rgb", "audio", "proprio"}:
        raise ValueError("generator_missing_model_modality")
    if set(coordinates) != {"rgb", "audio", "proprio"}:
        raise ValueError("generator_missing_sensor_coordinates")
    return {
        "observations": observations,
        "coordinates": coordinates,
        "outcome": result["execution_outcome"],
        "world_after": result["world_after"],
    }


def _temporal_initial_world(seed: int, initial_x: float) -> tuple[PhysicalWorld, str]:
    world = PhysicalWorld.reset(seed, {"body_count": 1})
    state = world.snapshot()
    state["world_id"] = hashlib.sha256(
        f"{state['world_id']}|fixture|{initial_x}".encode()
    ).hexdigest()[:32]
    body = state["bodies"][0]
    body.update(
        id=state["world_id"] + ":body",
        position=[initial_x, 0.0],
        velocity=[0.0, 0.0],
        mass=1.0,
        drag=1.0,
        radius=0.18,
    )
    entity = NAMES[[list(row) for row in (PALETTE * 255).astype(int)].index(body["color"])]
    return PhysicalWorld.restore(state), entity


@dataclass
class Sample:
    family: str
    seed: int
    envelope: dict[str, Any]
    initial_world: dict[str, Any]
    world: dict[str, Any]
    observations: dict[str, torch.Tensor]
    coordinates: dict[str, torch.Tensor]
    outcome: torch.Tensor
    task: dict[str, Any]
    case: Case | None = None
    costs: dict[str, float] = field(default_factory=dict)


def _envelope(
    family: str,
    seed: int,
    initial_world: dict[str, Any],
    world: dict[str, Any],
    observation: dict[str, Any],
    task: dict[str, Any],
    *,
    case: Case | None,
    requests: list[bytes],
    actions: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    model_inputs = [_byte_info(item) for item in case.model_inputs] if case else []
    envelope = {
        "schema": SCHEMA,
        "family": family,
        "seed": seed,
        "world": {
            "world_id": world["world_id"],
            "version": world["version"],
            "tick_index": world["tick_index"],
            "body_count": len(world["bodies"]),
            "modalities": list(world["modalities"]),
            "snapshot_sha256": value_digest(world),
            "initial_world_sha256": value_digest(initial_world),
        },
        "executor": {
            "schema": "taiji-u2-physical-executor-v1",
            "effect_count": world["version"],
            "persistent": False,
        },
        "model_observation": {
            "observations": {
                key: _tensor_info(value) for key, value in observation["observations"].items()
            },
            "coordinates": {
                key: _tensor_info(value) for key, value in observation["coordinates"].items()
            },
            "outcome": _tensor_info(observation["outcome"]),
        },
        "legal_content": {
            "goal_raw": GOAL.hex(),
            "requests": [_byte_info(item) for item in requests],
            "model_inputs": model_inputs,
            "observations": {
                key: _tensor_info(value) for key, value in observation["observations"].items()
            },
            "coordinates": {
                key: _tensor_info(value) for key, value in observation["coordinates"].items()
            },
            "outcome": _tensor_info(observation["outcome"]),
            "action": _tensor_info(actions[-1]) if actions else None,
        },
        "feedback_rights": FEEDBACK_RIGHTS,
        "task": task,
        "provenance": {
            "source_pins": source_pins(),
            "generator": "22.4.2.6",
        },
    }
    envelope["envelope_digest"] = value_digest(
        {key: value for key, value in envelope.items() if key != "envelope_digest"}
    )
    return envelope


def _sample_ab(seed: int, family: str) -> Sample:
    temporal = "current" if family == "A" else "previous"
    fixture = temporal_fixture(seed, -0.55, temporal)
    world, entity = _temporal_initial_world(seed, -0.55)
    initial_snapshot = world.snapshot()
    if fixture.case.request.entity != entity:
        raise ValueError("generator_temporal_entity_mismatch")
    first = world.elapse("first-visible")
    if not torch.equal(first["raw_media"]["rgb"], fixture.images[0]):
        raise ValueError("generator_temporal_first_frame_mismatch")
    result = None
    actions = []
    for index, command in enumerate(fixture.commands):
        actions.append(command)
        result = world.step(f"generator-{index}", command)
    if result is None or not torch.equal(result["raw_media"]["rgb"], fixture.images[1]):
        raise ValueError("generator_temporal_last_frame_mismatch")
    if result["world_after"] != fixture.case.world_version:
        raise ValueError("generator_temporal_world_version_mismatch")
    encode_started = time.process_time()
    observation = _observation(result)
    encoding_cpu = round(time.process_time() - encode_started, 4)
    task = {
        "case": {
            "goal_raw": fixture.case.goal_raw.hex(),
            "request_raw": fixture.case.request.raw.hex(),
            "request_id": fixture.case.request.request_id,
            "entity": fixture.case.request.entity,
            "temporal": fixture.case.request.temporal,
            "target": fixture.case.target.decode(),
            "facts": [
                {
                    "evidence_id": fact.evidence_id,
                    "entity": fact.entity,
                    "value": fact.value,
                    "version": fact.version,
                    "source": fact.source,
                }
                for fact in fixture.case.facts
            ],
        },
        "initial_x": -0.55,
    }
    envelope = _envelope(
        family,
        seed,
        initial_snapshot,
        world.snapshot(),
        observation,
        task,
        case=fixture.case,
        requests=[fixture.case.request.raw],
        actions=actions,
    )
    return Sample(
        family,
        seed,
        envelope,
        initial_snapshot,
        world.snapshot(),
        observation["observations"],
        observation["coordinates"],
        observation["outcome"],
        task,
        fixture.case,
        costs={"encoding_cpu": encoding_cpu},
    )


def _sample_c(seed: int) -> Sample:
    initial_x = -0.55
    fixture = acquisition_fixture(seed, initial_x, True)
    world, names = structured_world(seed, [(initial_x, 0.0)])
    initial_snapshot = world.snapshot()
    prior = world.elapse("prior")
    inspect = PhysicalWorld.restore(world.snapshot())
    action = torch.zeros(15, dtype=torch.float64)
    action[2], action[13], action[14] = 1, 0.1, 0.04
    seen = inspect.step("inspect", action)
    local = locate_public(seen["raw_media"]["rgb"], names[0])
    answer = "未知" if local is None else position_word(0.925 * local)
    if answer != fixture["answers"]["inspect"]:
        raise ValueError("generator_c_inspect_answer_mismatch")
    request = Request(
        f"request-c-{seed}-{initial_x}-True",
        1,
        seen["world_after"],
        request_text(names[0], "current"),
        names[0],
        "current",
        ("inspect",),
    )
    fact = Fact("inspect", names[0], fixture["truth"], seen["world_after"], "sensor")
    case = Case(
        GOAL,
        1,
        seen["world_after"],
        request,
        (fact,),
        (GOAL, request.raw, seen["raw_media"]["rgb"].numpy().tobytes()),
        fixture["truth"].encode(),
    )
    validate_case(case)
    encode_started = time.process_time()
    observation = _observation(seen)
    encoding_cpu = round(time.process_time() - encode_started, 4)
    task = {
        "known": True,
        "prior_frame": _tensor_info(prior["raw_media"]["rgb"]),
        "choice": fixture["best"],
        "answers": fixture["answers"],
        "utilities": fixture["utilities"],
        "truth": fixture["truth"],
    }
    envelope = _envelope(
        "C",
        seed,
        initial_snapshot,
        inspect.snapshot(),
        observation,
        task,
        case=case,
        requests=[request.raw],
        actions=[action],
    )
    return Sample(
        "C",
        seed,
        envelope,
        initial_snapshot,
        inspect.snapshot(),
        observation["observations"],
        observation["coordinates"],
        observation["outcome"],
        task,
        case,
        costs={"encoding_cpu": encoding_cpu},
    )


def _capacity_baseline(seed: int) -> dict[str, Any]:
    world, names = structured_world(seed, CAPACITY_POSITIONS)
    state = world.snapshot()
    for body in state["bodies"]:
        body["mass"] = 1.0
        body["drag"] = 1.0
    world = PhysicalWorld.restore(state)
    first = world.elapse("capacity-first")
    first_snapshot = world.snapshot()
    result = None
    for index in range(CAPACITY_ROUNDS - 1):
        result = world.step(f"capacity-{index}", capacity_course_action(index))
    assert result is not None
    return {
        "world": world,
        "names": names,
        "first_snapshot": first_snapshot,
        "last_snapshot": world.snapshot(),
        "images": [first["raw_media"]["rgb"], result["raw_media"]["rgb"]],
        "result": result,
    }


def _sample_d(seed: int) -> Sample:
    initial_world, _ = structured_world(seed, CAPACITY_POSITIONS)
    initial_snapshot = initial_world.snapshot()
    baseline_a = _capacity_baseline(seed)
    baseline_b = _capacity_baseline(seed + 500)
    checks = capacity_interleave(
        capacity_checks(baseline_a, None), capacity_checks(baseline_b, None)
    )
    encode_started = time.process_time()
    observation = _observation(baseline_a["result"])
    encoding_cpu = round(time.process_time() - encode_started, 4)
    requests = [request_text(item["entity"], item["temporal"]) for item in checks]
    task = {
        "capacity": 4,
        "entities": 6,
        "worlds": 2,
        "rounds": CAPACITY_ROUNDS,
        "observed_facts": 12,
        "pressure_ratio": 3.0,
        "checks": checks,
        "names_a": baseline_a["names"],
        "names_b": baseline_b["names"],
    }
    actions = [capacity_course_action(index) for index in range(CAPACITY_ROUNDS - 1)]
    envelope = _envelope(
        "D",
        seed,
        initial_snapshot,
        baseline_a["world"].snapshot(),
        observation,
        task,
        case=None,
        requests=requests,
        actions=actions,
    )
    return Sample(
        "D",
        seed,
        envelope,
        initial_snapshot,
        baseline_a["world"].snapshot(),
        observation["observations"],
        observation["coordinates"],
        observation["outcome"],
        task,
        costs={"encoding_cpu": encoding_cpu},
    )


def _sample_j(seed: int) -> Sample:
    initial_world, _ = structured_world(seed, [(-0.55, -0.55), (0.0, 0.0), (0.55, 0.55)])
    initial_snapshot = initial_world.snapshot()
    fixture = joint_fixture(seed)
    world = PhysicalWorld.restore(fixture["final_snapshot"])
    result = world.step("generator-final", _zero_action())
    encode_started = time.process_time()
    observation = _observation(result)
    encoding_cpu = round(time.process_time() - encode_started, 4)
    requests = [request_text(item["entity"], item["temporal"]) for item in fixture["checks"]]
    task = {
        "sessions": fixture["sessions"],
        "rounds": fixture["rounds"],
        "checks": fixture["checks"],
        "final_world_version": fixture["final_snapshot"]["version"],
    }
    envelope = _envelope(
        "J",
        seed,
        initial_snapshot,
        world.snapshot(),
        observation,
        task,
        case=None,
        requests=requests,
        actions=[_zero_action()],
    )
    return Sample(
        "J",
        seed,
        envelope,
        initial_snapshot,
        world.snapshot(),
        observation["observations"],
        observation["coordinates"],
        observation["outcome"],
        task,
        costs={"encoding_cpu": encoding_cpu},
    )


def generate(family: str, seed: int) -> Sample:
    if family not in FAMILIES:
        raise ValueError("unsupported_task_family")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("invalid_generator_seed")
    started_cpu, started_wall = time.process_time(), time.monotonic()
    if family in {"A", "B"}:
        sample = _sample_ab(seed, family)
    elif family == "C":
        sample = _sample_c(seed)
    elif family == "D":
        sample = _sample_d(seed)
    else:
        sample = _sample_j(seed)
    sample.costs.update(
        {
            "generation_cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        }
    )
    return sample


def capsule_payload(sample: Sample) -> dict[str, Any]:
    return {
        "schema": CAPSULE_SCHEMA,
        "source_pins": source_pins(),
        "family": sample.family,
        "seed": sample.seed,
        "envelope_digest": sample.envelope["envelope_digest"],
        "envelope": sample.envelope,
        "initial_world": sample.initial_world,
        "world": sample.world,
        "observations": sample.observations,
        "coordinates": sample.coordinates,
        "outcome": sample.outcome,
        "task": sample.task,
    }


def save_capsule(sample: Sample, path: str | Path) -> int:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    atomic_save(capsule_payload(sample), path)
    return path.stat().st_size


def load_capsule(path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = atomic_load(path)
    if (
        payload.get("schema") != CAPSULE_SCHEMA
        or payload.get("source_pins") != source_pins()
        or payload.get("family") not in FAMILIES
        or type(payload.get("seed")) is not int
    ):
        raise ValueError("task_sample_capsule_identity_mismatch")
    envelope = payload["envelope"]
    if payload["family"] != envelope["family"] or payload["seed"] != envelope["seed"]:
        raise ValueError("task_sample_capsule_family_or_seed_mismatch")
    recomputed = value_digest(
        {key: value for key, value in envelope.items() if key != "envelope_digest"}
    )
    if recomputed != envelope["envelope_digest"] or recomputed != payload["envelope_digest"]:
        raise ValueError("task_sample_capsule_envelope_mismatch")
    if value_digest(payload["initial_world"]) != envelope["world"]["initial_world_sha256"]:
        raise ValueError("task_sample_capsule_initial_world_mismatch")
    world = PhysicalWorld.restore(payload["world"])
    if value_digest(world.snapshot()) != envelope["world"]["snapshot_sha256"]:
        raise ValueError("task_sample_capsule_world_mismatch")
    for key in ("rgb", "audio", "proprio"):
        if (
            _tensor_info(payload["observations"][key])
            != envelope["model_observation"]["observations"][key]
        ):
            raise ValueError("task_sample_capsule_observation_mismatch")
        if (
            _tensor_info(payload["coordinates"][key])
            != envelope["model_observation"]["coordinates"][key]
        ):
            raise ValueError("task_sample_capsule_coordinate_mismatch")
    if _tensor_info(payload["outcome"]) != envelope["model_observation"]["outcome"]:
        raise ValueError("task_sample_capsule_outcome_mismatch")
    assert_values_equal(payload["world"], sample_world(payload), "capsule.world")
    return payload


def sample_world(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = PhysicalWorld.restore(payload["world"]).snapshot()
    return snapshot


def rebuild(sample: Sample) -> dict[str, Any]:
    fresh = generate(sample.family, sample.seed)
    if fresh.envelope["envelope_digest"] != sample.envelope["envelope_digest"]:
        raise ValueError("task_sample_rebuild_envelope_mismatch")
    for key in ("rgb", "audio", "proprio"):
        if (
            _tensor_info(fresh.observations[key])
            != sample.envelope["model_observation"]["observations"][key]
        ):
            raise ValueError("task_sample_rebuild_observation_mismatch")
    if value_digest(fresh.world) != value_digest(sample.world):
        raise ValueError("task_sample_rebuild_world_mismatch")
    return {"equal": True, "envelope_digest": fresh.envelope["envelope_digest"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one U2 task-family sample capsule")
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    sample = generate(args.family, args.seed)
    size = save_capsule(sample, args.out)
    print(
        json.dumps(
            {
                "family": sample.family,
                "seed": sample.seed,
                "envelope_digest": sample.envelope["envelope_digest"],
                "capsule_bytes": size,
                "generation_cpu": sample.costs["generation_cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

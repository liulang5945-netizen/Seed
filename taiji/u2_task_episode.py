"""Public-observation temporal fixtures and an independently scored legal oracle."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import torch

from .u2_history import value_digest
from .u2_task_contract import Case, Fact, Request, request_text, validate_case
from .u2_task_world import PhysicalWorld

NAMES = ("红色圆片", "蓝色圆片", "绿色圆片", "黄色圆片", "紫色圆片", "青色圆片")
PALETTE = (
    np.array(
        [
            [220, 45, 45],
            [45, 90, 220],
            [45, 190, 80],
            [230, 190, 40],
            [170, 60, 200],
            [40, 190, 195],
        ]
    )
    / 255
)
COURSE_X = (-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7) + (0.7,) * 8


def position_word(x):
    return "左边" if x < -1 / 3 else "右边" if x > 1 / 3 else "中间"


def locate_public(image, entity):
    """This baseline uses only public RGB and a declared color-name prior."""
    if entity not in NAMES or image is None:
        return None
    pixels = image.detach().cpu().numpy().astype(np.float64) / 255
    distances = np.linalg.norm(pixels[..., None, :] - PALETTE, axis=-1)
    index = NAMES.index(entity)
    weights = np.maximum(0, 1 - distances[..., index] / 0.35) * (distances.argmin(-1) == index)
    if weights.sum() < 0.25:
        return None
    x = 2 * (np.arange(16) + 0.5) / 16 - 1
    return float((weights * x[None, :]).sum() / weights.sum())


def public_oracle(raw_request, images):
    text = raw_request.decode("utf-8")
    if text.endswith("的位置时间未明确。"):
        return "未知".encode()
    for suffix, offset in (("现在在哪里？", -1), ("上一次在哪里？", -2)):
        if text.endswith(suffix):
            entity = text[: -len(suffix)]
            if len(images) < -offset:
                return "未知".encode()
            x = locate_public(images[offset], entity)
            return "未知".encode() if x is None else position_word(x).encode()
    raise ValueError("unsupported_public_oracle_request")


@dataclass(frozen=True)
class TemporalFixture:
    case: Case
    images: tuple[torch.Tensor, ...]
    commands: tuple[torch.Tensor, ...]
    blank_input: bytes
    environment_provenance: dict


def with_intent(fixture, temporal):
    from .u2_task_contract import oracle_answer

    request = replace(
        fixture.case.request,
        request_id=fixture.case.request.request_id.rsplit("-", 1)[0] + "-" + temporal,
        temporal=temporal,
        raw=request_text(fixture.case.request.entity, temporal),
    )
    case = replace(
        fixture.case,
        request=request,
        target=oracle_answer(request, fixture.case.facts),
        model_inputs=(fixture.case.goal_raw, request.raw, *fixture.case.model_inputs[2:]),
    )
    validate_case(case)
    return replace(fixture, case=case, blank_input=case.goal_raw + b"\x00" + request.raw)


def initial_current(fixture):
    first = fixture.case.facts[0]
    request = replace(
        fixture.case.request,
        request_id=fixture.case.request.request_id + "-initial",
        temporal="current",
        raw=request_text(first.entity, "current"),
        world_version=first.version,
        evidence_ids=(first.evidence_id,),
    )
    case = replace(
        fixture.case,
        request=request,
        world_version=first.version,
        facts=(first,),
        target=first.value.encode(),
        model_inputs=(fixture.case.goal_raw, request.raw, fixture.images[0].numpy().tobytes()),
    )
    validate_case(case)
    return replace(
        fixture,
        case=case,
        images=(fixture.images[0],),
        commands=(),
        blank_input=case.goal_raw + b"\x00" + request.raw,
    )


def temporal_fixture(seed, initial_x, temporal="previous"):
    """Counterfactual genesis and a fixed open-loop course, with two visible frames."""
    if initial_x not in (-0.55, 0.55):
        raise ValueError("unregistered_initial_axis")
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
    world = PhysicalWorld.restore(state)
    initial_digest = value_digest(state)
    first = world.elapse("first-visible")
    images = [first["raw_media"]["rgb"]]
    facts = [
        Fact(
            "first-visible",
            entity,
            position_word(world.snapshot()["bodies"][0]["position"][0]),
            first["world_after"],
            "sensor",
        )
    ]
    commands = []
    for index, x in enumerate(COURSE_X):
        action = torch.zeros(15, dtype=torch.float64)
        action[0] = 1
        action[9] = x
        action[11] = 0.8
        action[13] = 0.6
        action[14] = 0.04
        result = world.step(f"course-{index}", action)
        commands.append(action)
    images.append(result["raw_media"]["rgb"])
    facts.append(
        Fact(
            "last-visible",
            entity,
            position_word(world.snapshot()["bodies"][0]["position"][0]),
            result["world_after"],
            "sensor",
        )
    )
    request = Request(
        f"request-{seed}-{initial_x}-{temporal}",
        1,
        result["world_after"],
        request_text(entity, temporal),
        entity,
        temporal,
        tuple(f.evidence_id for f in facts),
    )
    target = (
        (facts[-2].value if temporal == "previous" else facts[-1].value)
        if temporal != "unknown"
        else "未知"
    )
    goal = "记住可核验观测的位置，并按时间要求回答。".encode()
    # Protocol ids, private facts, seeds and the initial-state axis are excluded.
    visible = tuple(image.numpy().tobytes() for image in images)
    actions = tuple(command.numpy().tobytes() for command in commands)
    inputs = (goal, request.raw, *visible, *actions)
    case = Case(goal, 1, result["world_after"], request, tuple(facts), inputs, target.encode())
    validate_case(case)
    blank = goal + b"\x00" + request.raw
    return TemporalFixture(
        case,
        tuple(images),
        tuple(commands),
        blank,
        {
            "construction": "counterfactual genesis snapshot",
            "initial_snapshot": initial_digest,
            "course": "fixed open-loop, identical commands for both initial positions",
            "sensor_schedule": "first and last RGB visible; intermediate media withheld by public plan",
            "query_current_view": "no newly released sensor frame",
            "world_version": result["world_after"],
        },
    )


def structured_world(seed, positions):
    """Declared counterfactual genesis for multi-role and acquisition fixtures."""
    world = PhysicalWorld.reset(seed, {"body_count": len(positions)})
    state = world.snapshot()
    state["world_id"] = hashlib.sha256((state["world_id"] + repr(positions)).encode()).hexdigest()[
        :32
    ]
    names = []
    for i, (body, position) in enumerate(zip(state["bodies"], positions, strict=True)):
        body.update(
            id=state["world_id"] + f":body-{i}",
            position=list(position),
            velocity=[0.0, 0.0],
            mass=1.0,
            drag=1.0,
            radius=0.18,
        )
        colors = [list(row) for row in (PALETTE * 255).astype(int)]
        names.append(NAMES[colors.index(body["color"])])
    return PhysicalWorld.restore(state), names


def acquisition_fixture(seed, initial_x, known):
    world, names = structured_world(seed, [(initial_x, 0.0)])
    prior = world.elapse("prior")
    images = (prior["raw_media"]["rgb"],) if known else ()
    inspect = PhysicalWorld.restore(world.snapshot())
    action = torch.zeros(15, dtype=torch.float64)
    action[2], action[13], action[14] = 1, 0.1, 0.04
    seen = inspect.step("inspect", action)
    local = locate_public(seen["raw_media"]["rgb"], names[0])
    inspect_answer = "未知" if local is None else position_word(0.925 * local)
    request = request_text(names[0], "current")
    wait_answer = public_oracle(request, images).decode()
    truth = position_word(initial_x)
    utilities = {
        "inspect": float(inspect_answer == truth) - 0.01,
        "wait": float(wait_answer == truth) - 0.001,
    }
    return {
        "request": request,
        "known": known,
        "truth": truth,
        "answers": {"inspect": inspect_answer, "wait": wait_answer},
        "utilities": utilities,
        "best": max(utilities, key=lambda name: utilities[name]),
        "source": "public history and calibrated fresh inspect RGB; constant neutral execution Outcome",
    }


def multirole_fixture(seed, permutation):
    world, names = structured_world(seed, [(x, 0.0) for x in permutation])
    result = world.elapse("roles")
    image = result["raw_media"]["rgb"]
    responses = {
        name: position_word(x) if (x := locate_public(image, name)) is not None else "未知"
        for name in names
    }
    truth = {
        name: position_word(world.snapshot()["bodies"][i]["position"][0])
        for i, name in enumerate(names)
    }
    return {
        "names": names,
        "image": image,
        "responses": responses,
        "truth": truth,
        "query_removed_ceiling": max(Counter(truth.values()).values()) / len(truth),
    }


def joint_fixture(seed):
    """One physical world across three 16-round sessions with public fact versions."""
    world, names = structured_world(seed, [(-0.55, -0.55), (0.0, 0.0), (0.55, 0.55)])
    images = []
    facts = []
    checks = []
    identity = world.snapshot()["world_id"]
    for round_index in range(48):
        if round_index < 32:
            sign = 1 if round_index < 16 else -1
            action = torch.zeros(15, dtype=torch.float64)
            action[0] = 1
            action[9] = sign * COURSE_X[round_index % 16]
            action[10] = -0.55
            action[11], action[13], action[14] = sign * 0.8, 0.6, 0.04
            result = world.step(f"joint-{round_index}", action)
        else:
            result = world.elapse(f"joint-{round_index}")
        if round_index in (0, 15, 31):
            images.append(result["raw_media"]["rgb"])
            for i, name in enumerate(names):
                facts.append(
                    Fact(
                        f"{round_index}-{i}",
                        name,
                        position_word(world.snapshot()["bodies"][i]["position"][0]),
                        result["world_after"],
                        "sensor",
                    )
                )
        if round_index in (15, 31, 47):
            for name in names:
                history = [f for f in facts if f.entity == name]
                for temporal in ("current", "previous"):
                    expected = history[-1 if temporal == "current" else -2].value
                    if temporal == "current":
                        expected = position_word(
                            world.snapshot()["bodies"][names.index(name)]["position"][0]
                        )
                    answer = public_oracle(request_text(name, temporal), images).decode()
                    checks.append(
                        {
                            "round": round_index + 1,
                            "entity": name,
                            "temporal": temporal,
                            "expected": expected,
                            "answer": answer,
                        }
                    )
    assert world.snapshot()["world_id"] == identity
    return {
        "world_id": identity,
        "sessions": 3,
        "rounds": 48,
        "checks": checks,
        "facts": facts,
        "final_snapshot": world.snapshot(),
        "qualification": "continuous environment and public oracle; model learning and joint agency pending",
    }

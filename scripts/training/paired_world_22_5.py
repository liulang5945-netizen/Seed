"""Counterfactual public worlds with paired questions and audited partial-input ceilings."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import torch

from scripts.training.first_a_support_22_5 import GOAL, LABELS, RGB_COLORS
from taiji.u2_history import value_digest
from taiji.u2_task_episode import NAMES, locate_public, position_word
from taiji.u2_task_world import PhysicalWorld


def family_seed(partition, position):
    raw = f"paired-world-v1|{partition}|{position}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % 2**63


def public_image(public):
    pixels = public["observations"]["rgb"].reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4)
    return (pixels.reshape(16, 16, 3) * 255).round().clamp(0, 255).to(torch.uint8)


def build_partition(partition):
    if partition not in {"train", "dev"}:
        raise ValueError("paired_partition_not_registered")
    worlds, episodes = {}, []
    for position in range(3):
        seed = family_seed(partition, position)
        original = PhysicalWorld.reset(seed, {"body_count": 1}).snapshot()
        generator = torch.Generator().manual_seed(seed)
        center = (-0.62, 0.0, 0.62)[position]
        x = center + float(torch.rand((), generator=generator)) * 0.12 - 0.06
        y = float(torch.rand((), generator=generator)) * 0.5 - 0.25
        for visible in range(6):
            genesis = PhysicalWorld.restore(original).snapshot()
            world_id = hashlib.sha256(
                f"paired-genesis|{partition}|{position}|{visible}|{seed}".encode()
            ).hexdigest()[:32]
            genesis["world_id"] = world_id
            genesis["bodies"][0].update(
                id=f"{world_id}:0",
                color=list(RGB_COLORS[visible]),
                position=[x, y],
                velocity=[0.0, 0.0],
                radius=0.18,
                mass=1.0,
                drag=1.0,
            )
            world = PhysicalWorld.restore(genesis)
            result = world.elapse("paired-observation")
            public = {
                "goal": GOAL,
                "observations": result["observations"],
                "coordinates": result["coordinates"],
                "outcome": result["execution_outcome"],
                "action": torch.zeros(15),
                "t_start": result["t_start"],
                "t_end": result["t_end"],
            }
            expected = ("左边", "中间", "右边")[position]
            image = public_image(public)
            estimated = locate_public(image, NAMES[visible])
            if estimated is None or position_word(estimated) != expected:
                raise ValueError("paired_public_oracle_not_identifiable")
            worlds[world_id] = {
                "public": public,
                "audit": {
                    "family": f"{partition}-{position}",
                    "seed": seed,
                    "visible": visible,
                    "position": position,
                    "genesis": genesis,
                    "world_digest": value_digest(world.snapshot()),
                },
            }
            for absent in range(6):
                if absent == visible:
                    continue
                if locate_public(image, NAMES[absent]) is not None:
                    raise ValueError("paired_absent_entity_visible")
                for reverse in (False, True):
                    names = [NAMES[visible], NAMES[absent]]
                    targets = [expected.encode(), "未知".encode()]
                    if reverse:
                        names.reverse()
                        targets.reverse()
                    identity = hashlib.sha256(
                        f"paired-episode|{world_id}|{absent}|{reverse}".encode()
                    ).hexdigest()[:32]
                    episodes.append(
                        {
                            "id": identity,
                            "world_id": world_id,
                            "requests": tuple(f"{name}现在在哪里？".encode() for name in names),
                            "targets": tuple(targets),
                            "audit": {
                                "visible": visible,
                                "absent": absent,
                                "position": position,
                                "reverse": reverse,
                            },
                        }
                    )
    return {"worlds": worlds, "episodes": episodes}


def public_example(partition, episode):
    world = partition["worlds"][episode["world_id"]]
    if set(world["public"]) != {
        "goal",
        "observations",
        "coordinates",
        "outcome",
        "action",
        "t_start",
        "t_end",
    }:
        raise ValueError("unregistered_paired_public_fields")
    return {"id": episode["id"], "public": {**world["public"], "request": episode["requests"][0]}}


def _ceiling(groups):
    total = sum(sum(row.values()) for row in groups.values())
    return {
        "correct": sum(max(row.values()) for row in groups.values()),
        "denominator": total,
        "accuracy": sum(max(row.values()) for row in groups.values()) / total,
        "input_classes": len(groups),
    }


def audit_partition(partition):
    groups: dict[str, dict] = {
        name: defaultdict(Counter)
        for name in (
            "question",
            "question_and_turn",
            "question_history",
            "public_without_rgb_and_question_history",
            "public_without_questions_and_turn",
        )
    }
    support: defaultdict[str, Counter[str]] = defaultdict(Counter)
    oracle_correct = 0
    non_rgb_families = defaultdict(set)
    for world in partition["worlds"].values():
        public = world["public"]
        redacted = {
            **public,
            "observations": {
                key: value for key, value in public["observations"].items() if key != "rgb"
            },
        }
        non_rgb_families[world["audit"]["family"]].add(value_digest(redacted))
    if any(len(values) != 1 for values in non_rgb_families.values()):
        raise ValueError("paired_non_rgb_family_leak")
    for episode in partition["episodes"]:
        public = partition["worlds"][episode["world_id"]]["public"]
        image = public_image(public)
        redacted = {
            **public,
            "observations": {
                key: value for key, value in public["observations"].items() if key != "rgb"
            },
        }
        for turn, (request, target) in enumerate(
            zip(episode["requests"], episode["targets"], strict=True)
        ):
            question, label = request.decode(), target.decode()
            entity = question.removesuffix("现在在哪里？")
            x = locate_public(image, entity)
            answer = "未知" if x is None else position_word(x)
            oracle_correct += answer == label
            support[question][label] += 1
            keys = {
                "question": request,
                "question_and_turn": (request, turn),
                "question_history": episode["requests"][: turn + 1],
                "public_without_rgb_and_question_history": (
                    value_digest(redacted),
                    episode["requests"][: turn + 1],
                ),
                "public_without_questions_and_turn": (value_digest(public), turn),
            }
            for name, key in keys.items():
                groups[name][key][label] += 1
    if len(support) != 6 or any(set(row) != set(LABELS) for row in support.values()):
        raise ValueError("paired_incomplete_entity_answer_support")
    return {
        "worlds": len(partition["worlds"]),
        "episodes": len(partition["episodes"]),
        "rounds": sum(sum(row.values()) for row in support.values()),
        "oracle_correct": oracle_correct,
        "support": dict(support),
        "partial_input_ceilings": {name: _ceiling(value) for name, value in groups.items()},
        "non_rgb_equivalent_family_count": len(non_rgb_families),
    }

"""23 composition fixture: Chinese relational questions over three-body worlds.

Question families (all answered from the symbolic world state, oracle-readable):
  Q1 single-hop position: "{X}现在在哪里？"
  Q2 single-hop relation: "{X}的{d}边是谁？"
  Q3 two-hop ordered relation: "{X}{d1}边物体的{d2}边是谁？"
  Q4 source/time wording: "根据{source}，{X}在{time}时在哪里？"

Partitions hold out wording groups, relation directions and two-hop orders so that
train never shares a (template_group, direction, pair_order) triple with fit/check/final.
This module is data-only; it does not train anything and reads no check/final labels
beyond computing sealed digests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from taiji.u2_task_episode import NAMES, position_word
from taiji.u2_task_world import PhysicalWorld

RGB_COLORS = (
    (220, 45, 45),
    (45, 90, 220),
    (45, 190, 80),
    (230, 190, 40),
    (170, 60, 200),
    (40, 190, 195),
)

FIXTURE_ID = "taiji-23-composition-v1"
TEMPLATES = {
    "q1": ("{x}现在在哪里？", "{x}位于何处？", "{x}在什么位置？"),
    "q2": ("{x}的{d}边是谁？", "{x}的{d}侧是谁？", "谁在{x}的{d}边？"),
    "q3": ("{x}{d1}边物体的{d2}边是谁？", "{x}{d1}侧那位物体的{d2}侧是谁？"),
    "q4": (
        "根据{source}，{x}在{time}时在哪里？",
        "依照{source}，{x}在{time}时位于何处？",
    ),
}
DIRECTIONS = ("左边", "右边", "前边", "后边")
SOURCES = ("画面", "之前的记录")
TIMES = ("开始时", "结束时")
POSITION_LABELS = ("左边", "中间", "右边", "未知")
PARTITION_SIZES = {"train": 512, "fit": 128, "check": 128, "final": 128}
PARTITION_OFFSETS = {"train": 0, "fit": 512, "check": 640, "final": 768}
# Held-out axes: train only sees {left,right} directions and one order per pair.
DIRECTION_TRAIN = ("左边", "右边")
DIRECTION_HELDOUT = ("前边", "后边")
PAIR_TRAIN = (("左边", "右边"), ("右边", "左边"))
PAIR_HELDOUT = (("前边", "后边"), ("后边", "前边"), ("左边", "前边"), ("后边", "右边"))


def _body_axis(body: dict[str, Any]) -> tuple[float, float]:
    position = body["position"]
    return float(position[0]), float(position[1])


def _relation(source: str, direction: str, bodies: list[dict[str, Any]]) -> str | None:
    xy = {body["name"]: _body_axis(body) for body in bodies}
    if source not in xy:
        return None
    sx, sy = xy[source]
    candidates: list[tuple[float, str]] = []
    for name, (x, y) in xy.items():
        if name == source:
            continue
        if direction == "左边" and x < sx:
            candidates.append((sx - x, name))
        elif direction == "右边" and x > sx:
            candidates.append((x - sx, name))
        elif direction == "前边" and y > sy:
            candidates.append((y - sy, name))
        elif direction == "后边" and y < sy:
            candidates.append((sy - y, name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _relative_position(source: str, bodies: list[dict[str, Any]]) -> str:
    xy = {body["name"]: _body_axis(body) for body in bodies}
    if source not in xy:
        return "未知"
    x = xy[source][0]
    return cast(str, position_word(x))


def world_scene(seed: int) -> dict[str, Any]:
    world = PhysicalWorld.reset(seed, {"body_count": 3})
    result = world.elapse("public-scene")
    bodies = world.snapshot()["bodies"]
    scene = []
    for body in bodies:
        colour = RGB_COLORS.index(tuple(body["color"]))
        scene.append({"name": NAMES[colour], "position": body["position"]})
    if not scene:
        raise ValueError("23_composition_world_empty")
    return {
        "seed": seed,
        "bodies": scene,
        "t_start": result["t_start"],
        "t_end": result["t_end"],
        "world_digest": hashlib.sha256(
            json.dumps(scene, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }


def _questions(seed: int, partition: str, scene: dict[str, Any]) -> list[dict[str, Any]]:
    bodies = scene["bodies"]
    names = [body["name"] for body in bodies]
    index = seed % len(names)
    entity = names[index]
    questions: list[dict[str, Any]] = []
    # Q1 appears in train only (base position ability); evaluation partitions use Q2/Q3/Q4.
    if partition == "train":
        wording = TEMPLATES["q1"][PARTITION_OFFSETS[partition] % len(TEMPLATES["q1"])]
        questions.append(
            {
                "kind": "q1",
                "text": wording.format(x=entity),
                "answer": _relative_position(entity, bodies),
                "axes": {
                    "template_group": PARTITION_OFFSETS[partition] % len(TEMPLATES["q1"]),
                    "direction": None,
                    "pair_order": None,
                },
            }
        )
    # Q2: direction axis held out between train and the other partitions.
    directions = DIRECTION_TRAIN if partition == "train" else DIRECTION_HELDOUT
    direction = directions[seed % len(directions)]
    target = _relation(entity, direction, bodies)
    answer = target if target is not None else "未知"
    questions.append(
        {
            "kind": "q2",
            "text": TEMPLATES["q2"][PARTITION_OFFSETS[partition] % len(TEMPLATES["q2"])].format(
                x=entity, d=direction
            ),
            "answer": answer,
            "axes": {
                "template_group": PARTITION_OFFSETS[partition] % len(TEMPLATES["q2"]),
                "direction": direction,
                "pair_order": None,
            },
        }
    )
    # Q3: pair-order axis held out.
    pairs = PAIR_TRAIN if partition == "train" else PAIR_HELDOUT
    d1, d2 = pairs[seed % len(pairs)]
    first = _relation(entity, d1, bodies)
    second = _relation(first, d2, bodies) if first is not None else None
    questions.append(
        {
            "kind": "q3",
            "text": TEMPLATES["q3"][PARTITION_OFFSETS[partition] % len(TEMPLATES["q3"])].format(
                x=entity, d1=d1, d2=d2
            ),
            "answer": second if second is not None else "未知",
            "axes": {
                "template_group": PARTITION_OFFSETS[partition] % len(TEMPLATES["q3"]),
                "direction": d1,
                "pair_order": (d1, d2),
            },
        }
    )
    # Q4: source/time wording appears in evaluation partitions only.
    if partition != "train":
        source = SOURCES[(seed + PARTITION_OFFSETS[partition]) % len(SOURCES)]
        time = TIMES[(seed + PARTITION_OFFSETS[partition]) % len(TIMES)]
        questions.append(
            {
                "kind": "q4",
                "text": TEMPLATES["q4"][PARTITION_OFFSETS[partition] % len(TEMPLATES["q4"])].format(
                    source=source, x=entity, time=time
                ),
                "answer": _relative_position(entity, bodies),
                "axes": {
                    "template_group": PARTITION_OFFSETS[partition] % len(TEMPLATES["q4"]),
                    "direction": None,
                    "pair_order": None,
                },
            }
        )
    return questions


def build_partition(partition: str) -> list[dict[str, Any]]:
    size = PARTITION_SIZES[partition]
    offset = PARTITION_OFFSETS[partition]
    rows = []
    for index in range(size):
        seed = 23_0000 + offset + index
        scene = world_scene(seed)
        for question in _questions(seed, partition, scene):
            rows.append(
                {
                    "id": f"{partition}-{index}-{question['kind']}",
                    "seed": seed,
                    "partition": partition,
                    "kind": question["kind"],
                    "text": question["text"],
                    "answer": question["answer"],
                    "axes": question["axes"],
                    "world_digest": scene["world_digest"],
                }
            )
    return rows


def partition_digest(partition: str) -> str:
    rows = build_partition(partition)
    payload = json.dumps(
        [
            {
                "id": row["id"],
                "text": row["text"],
                "answer": row["answer"],
                "axes": row["axes"],
                "world": row["world_digest"],
            }
            for row in rows
        ],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def oracle_check() -> dict[str, Any]:
    rows = build_partition("fit")
    answers = {row["kind"]: row["answer"] for row in rows}
    return {
        "rows": len(rows),
        "answer_space": sorted({row["answer"] for row in rows})[:12],
        "kinds": answers,
    }

"""A wider world for the sample-size question: 4 objects, 5 places, 480 goals.

The 2x3 fixture only has six states, so six training samples exhausted it. This world
keeps the same shape (one free place, goal names no entity, glossary bridges) but has
``5 * 4 * 3 * 2 = 120`` states and four goals per state -- 480 samples, enough for a
1:1 parameter/constraint ratio on a small readout.

Nothing here trains or reads the model; it is the data side only.
"""

from __future__ import annotations

import itertools

OBJECTS = ("蓝色杯子", "红色印章", "绿色瓶子", "黄色球")
PLACES = ("木桌", "窗台", "左盒", "书架", "地毯")

OBJECT_DESCRIPTIONS = {
    "蓝色杯子": "圆口的器皿",
    "红色印章": "带纹路的方块",
    "绿色瓶子": "细长的容器",
    "黄色球": "圆滚滚的东西",
}
PLACE_DESCRIPTIONS = {
    "木桌": "四方木台",
    "窗台": "靠窗的台面",
    "左盒": "手边的匣子",
    "书架": "靠墙的层架",
    "地毯": "脚边的毡子",
}

GLOSSARY = {
    **{text: entity for entity, text in OBJECT_DESCRIPTIONS.items()},
    **{text: entity for entity, text in PLACE_DESCRIPTIONS.items()},
}


def states() -> list[tuple[tuple[str, str], ...]]:
    """Every arrangement of four objects into five distinct places."""
    return [
        tuple(zip(OBJECTS, placement, strict=True))
        for placement in itertools.permutations(PLACES, len(OBJECTS))
    ]


def free_place(state) -> str:
    taken = {place for _, place in state}
    return next(place for place in PLACES if place not in taken)


def state_text(state) -> str:
    return "，".join(f"{obj}在{place}" for obj, place in state) + "。"


def describe_goal(obj: str, place: str) -> str:
    return f"让{OBJECT_DESCRIPTIONS[obj]}到{PLACE_DESCRIPTIONS[place]}。"


def action_sentence(obj: str, place: str) -> str:
    return f"把{obj}放到{place}"


def samples() -> list[dict]:
    rows = []
    for state in states():
        free = free_place(state)
        for obj in OBJECTS:
            rows.append(
                {
                    "state": state,
                    "goal": describe_goal(obj, free),
                    "correct": (obj, free),
                }
            )
    return rows


def resolve(goal: str) -> tuple[str | None, str | None]:
    objects = [text for text in OBJECT_DESCRIPTIONS.values() if text in goal]
    places = [text for text in PLACE_DESCRIPTIONS.values() if text in goal]
    if len(objects) != 1 or len(places) != 1:
        return None, None
    return GLOSSARY[objects[0]], GLOSSARY[places[0]]


def glossary_records() -> tuple[tuple[str, str], ...]:
    return tuple((f"{text}是{entity}。", text) for text, entity in GLOSSARY.items())

"""Goals that describe entities by attribute, plus the glossary that resolves them.

10.8 showed rewording keeps the copy shortcut because the goal still names the object
and the place. Here the goal never contains either name:

    让圆口的器皿到靠窗的台面。

The mapping descriptor -> entity lives in ``GLOSSARY`` and is meant to be handed to a
subject as data (records or prompt), not to be guessed from the action sentence.
"""

from __future__ import annotations

from taiji.native_goal_world import OBJECTS, PLACES

OBJECT_DESCRIPTIONS = {
    OBJECTS[0]: "圆口的器皿",
    OBJECTS[1]: "带纹路的方块",
}
PLACE_DESCRIPTIONS = {
    PLACES[0]: "四方木台",
    PLACES[1]: "靠窗的台面",
    PLACES[2]: "左边的匣子",
}

GLOSSARY = {
    **{text: entity for entity, text in OBJECT_DESCRIPTIONS.items()},
    **{text: entity for entity, text in PLACE_DESCRIPTIONS.items()},
}


def describe_goal(obj: str, place: str) -> str:
    return f"让{OBJECT_DESCRIPTIONS[obj]}到{PLACE_DESCRIPTIONS[place]}。"


def action_sentence(obj: str, place: str) -> str:
    return f"把{obj}放到{place}"


def resolve(goal: str) -> tuple[str | None, str | None]:
    objects = [text for text in OBJECT_DESCRIPTIONS.values() if text in goal]
    places = [text for text in PLACE_DESCRIPTIONS.values() if text in goal]
    if len(objects) != 1 or len(places) != 1:
        return None, None
    return GLOSSARY[objects[0]], GLOSSARY[places[0]]

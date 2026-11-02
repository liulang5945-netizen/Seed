"""Contract guard for 10.9: descriptions name no entity, the glossary resolves them."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji.native_goal_attributes import (  # noqa: E402
    GLOSSARY,
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
    describe_goal,
    resolve,
)
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

ENTITY_NAMES = [*OBJECTS, *PLACES]


def test_descriptions_do_not_name_their_entity():
    for entity, text in {**OBJECT_DESCRIPTIONS, **PLACE_DESCRIPTIONS}.items():
        for name in ENTITY_NAMES:
            assert name not in text, (entity, text, name)


def test_goals_name_no_entity_and_resolve_uniquely():
    for obj in OBJECTS:
        for place in PLACES:
            goal = describe_goal(obj, place)
            for name in ENTITY_NAMES:
                assert name not in goal, (goal, name)
            assert resolve(goal) == (obj, place)


def test_glossary_is_one_to_one():
    assert len(GLOSSARY) == len(OBJECT_DESCRIPTIONS) + len(PLACE_DESCRIPTIONS)
    assert len(set(GLOSSARY.values())) == len(GLOSSARY)

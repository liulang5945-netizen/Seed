"""Contract guard for 10.15: the wide world is big, entity-free and resolvable."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    OBJECTS,
    PLACE_DESCRIPTIONS,
    PLACES,
    free_place,
    resolve,
    samples,
    states,
)

ENTITY_NAMES = [*OBJECTS, *PLACES]


def test_world_size():
    assert len(OBJECTS) == 4 and len(PLACES) == 5
    assert len(states()) == 120
    assert len(samples()) == 480


def test_goals_are_entity_free_and_resolve():
    for row in samples():
        goal = row["goal"]
        for name in ENTITY_NAMES:
            assert name not in goal, (goal, name)
        assert resolve(goal) == row["correct"]


def test_descriptions_do_not_name_their_entity():
    for entity, text in {**OBJECT_DESCRIPTIONS, **PLACE_DESCRIPTIONS}.items():
        for name in ENTITY_NAMES:
            assert name not in text, (entity, text, name)


def test_free_place_is_unique():
    for state in states():
        place = free_place(state)
        assert place in PLACES
        assert all(place != position for _, position in state)

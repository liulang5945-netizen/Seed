"""10.2 oracles: the binder must bind unambiguous goals and refuse the rest."""

from __future__ import annotations

import ast
import inspect

from taiji.goal_binding import Binding, bind
from taiji.native_goal_world import OBJECTS, PLACES, build_fixtures, step

STATE = build_fixtures()[0].initial


def test_paraphrases_bind_to_the_same_conditions():
    first = bind("把蓝色杯子放到窗台", STATE)
    second = bind("蓝色杯子应该出现在窗台上", STATE)
    third = bind("蓝色杯子要在窗台", STATE)
    assert first.conditions == second.conditions == third.conditions
    assert first.conditions == ((OBJECTS[0], PLACES[1], 1),)


def test_negation_keeps_its_polarity():
    binding = bind("红色印章不要留在左盒", STATE)
    assert binding.abstain is False
    assert binding.conditions == ((OBJECTS[1], PLACES[2], -1),)
    assert binding.satisfied(STATE) is False, "the seal is in the box, so the negation fails"
    moved = step(STATE, (OBJECTS[1], PLACES[0]))
    assert binding.satisfied(moved) is True


def test_conjunction_keeps_both_clauses():
    binding = bind("把蓝色杯子放到窗台，并且红色印章不要留在左盒", STATE)
    assert binding.abstain is False
    assert set(binding.conditions) == {
        (OBJECTS[0], PLACES[1], 1),
        (OBJECTS[1], PLACES[2], -1),
    }


def test_missing_referent_abstains_instead_of_guessing():
    binding = bind("把它放到窗台", STATE)
    assert binding.abstain is True
    assert binding.reason == "missing_referent"
    assert binding.conditions == ()
    assert binding.unknowns == ("object",)


def test_ambiguity_abstains_instead_of_picking_one():
    binding = bind("把蓝色杯子放到窗台或木桌", STATE)
    assert binding.abstain is True
    assert binding.reason == "ambiguous"
    assert binding.conditions == ()


def test_binder_never_references_goal_or_score_identifiers():
    import taiji.goal_binding as module

    tree = ast.parse(inspect.getsource(module))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    seen = names | attributes
    for forbidden in ("Goal", "gold", "terminal_state", "reward"):
        assert forbidden not in seen, forbidden


def test_binding_satisfied_requires_all_conditions():
    binding = Binding(((OBJECTS[0], PLACES[1], 1), (OBJECTS[1], PLACES[2], -1)))
    assert binding.satisfied(STATE) is False
    both = step(step(STATE, (OBJECTS[0], PLACES[1])), (OBJECTS[1], PLACES[0]))
    assert binding.satisfied(both) is True
    assert Binding((), abstain=True, reason="ambiguous").satisfied(STATE) is False

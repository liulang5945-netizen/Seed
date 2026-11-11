"""10.4 oracles: the subject scores legal actions itself and selects the lowest NLL."""

from __future__ import annotations

import ast
import inspect

import torch

from taiji.goal_binding import bind
from taiji.living_core import LivingCoreConfig, LivingCoreReference
from taiji.native_goal_world import build_fixtures, legal_moves
from taiji.subject_action_scorer import render, score_actions, select_action

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)


def subject() -> LivingCoreReference:
    torch.manual_seed(20261101)
    return LivingCoreReference(CONFIG).double()


def test_render_uses_the_declared_chinese_form():
    assert render(("蓝色杯子", "窗台")) == "把蓝色杯子放到窗台"


def test_scoring_covers_every_legal_move_only():
    fixture = build_fixtures()[0]
    moves = legal_moves(fixture.initial, occupy=True)
    rows = score_actions(subject(), fixture.goal.text, moves)
    assert len(rows) == len(moves) == 2
    assert [tuple(row["move"]) for row in rows] == list(moves)
    assert all(row["nll"] == row["nll"] for row in rows), "NLLs must be finite"


def test_selection_is_the_lowest_nll_row_and_is_deterministic():
    fixture = build_fixtures()[0]
    moves = legal_moves(fixture.initial, occupy=True)
    first = select_action(subject(), fixture.goal.text, moves)
    second = select_action(subject(), fixture.goal.text, moves)
    assert first["action"] == second["action"]
    assert first["nll"] == min(row["nll"] for row in first["table"])
    assert first["scored"] == len(moves)


def test_scorer_never_references_terminal_scoring():
    import taiji.subject_action_scorer as module

    tree = ast.parse(inspect.getsource(module))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for forbidden in ("score", "state_score", "Goal", "satisfied"):
        assert forbidden not in (names | attributes), forbidden


def test_binding_is_used_only_for_the_goal_text():
    fixture = build_fixtures()[0]
    binding = bind(fixture.goal.text, fixture.initial)
    assert binding.conditions, "the fixture must bind"
    moves = legal_moves(fixture.initial, occupy=True)
    result = select_action(subject(), fixture.goal.text, moves)
    assert result["sentence"] == render(result["action"])

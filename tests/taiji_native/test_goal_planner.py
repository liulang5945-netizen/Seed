"""10.3 oracles: outcomes stay separated, the prediction column can fail, scoring ignores text."""

from __future__ import annotations

import ast
import inspect

from taiji.goal_binding import bind
from taiji.goal_planner import (
    DEPTH,
    execute,
    plan,
    run_arm,
    state_score,
    transition,
)
from taiji.native_goal_world import build_fixtures, optimal

FIXTURES = build_fixtures()


def binding_for(index: int = 0):
    fixture = FIXTURES[index]
    return fixture, bind(fixture.goal.text, fixture.initial)


def test_planner_keeps_predicted_imagined_and_actual_apart():
    fixture, binding = binding_for(0)
    intent = plan(binding, fixture.initial)
    assert intent.ledger["predicted"] is not None
    assert intent.ledger["actual"] is None, "actual must stay empty until execution"
    assert len(intent.imagined) <= DEPTH
    assert intent.ledger["imagined"] == [list(item.where) for item in intent.imagined]
    actual, ledger = execute(intent, fixture.initial)
    assert ledger["actual"] == list(actual.where)
    assert ledger["predicted"] == list(intent.predicted.where)
    assert ledger["predicted_matches_actual"] is True


def test_a_perturbed_model_makes_the_prediction_column_fail():
    fixture = binding_for(0)[0]
    binding = bind(fixture.goal.text, fixture.initial)

    def wrong(state, move):
        return transition(state, move, perturbed=True)

    intent = plan(binding, fixture.initial, model=wrong)
    _actual, ledger = execute(intent, fixture.initial)
    assert ledger["predicted_matches_actual"] is False


def test_planner_reaches_the_exhaustive_optimum_on_every_fixture():
    for fixture in FIXTURES:
        binding = bind(fixture.goal.text, fixture.initial)
        optimum = optimal(fixture.initial, fixture.goal)["score"]
        result = run_arm(binding, fixture.initial, mode="planner")
        assert abs(result["score"] - optimum) <= 1e-12, fixture.fixture_id


def test_scoring_takes_only_state_and_binding():
    signature = inspect.signature(state_score)
    assert list(signature.parameters) == ["state", "binding"]
    fixture, binding = binding_for(0)
    assert state_score(fixture.initial, binding) == state_score(fixture.initial, binding)


def test_explanation_is_recorded_but_never_an_input_to_scoring():
    import taiji.goal_planner as module

    source = inspect.getsource(module)
    tree = ast.parse(source)
    scored_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "state_score"
    ]
    assert scored_calls, "the planner must actually score states"
    for call in scored_calls:
        text_args = [
            keyword
            for keyword in call.keywords
            if keyword.arg in ("explanation", "text", "goal_text")
        ]
        assert not text_args, "scoring must not accept free text"
    fixture, binding = binding_for(0)
    intent = plan(binding, fixture.initial)
    assert isinstance(intent.explanation, str) and intent.explanation

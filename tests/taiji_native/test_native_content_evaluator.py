"""Fixed-denominator scoring contracts for R5."""

from __future__ import annotations

from taiji.native_content_evaluator import (
    ExpectedTurnV1,
    MechanismCaseV1,
    TurnPredictionV1,
    failed_gates,
    score_run,
)


def test_fixed_denominator_counts_missing_and_parse_failure():
    expected = [
        ExpectedTurnV1("e", 2, "current_query", b"blue"),
        ExpectedTurnV1("e", 4, "current_query", b"blue"),
        ExpectedTurnV1("e", 6, "previous_query", b"red"),
        ExpectedTurnV1("e", 7, "unknown_query", b"unknown"),
    ]
    predictions = {
        ("e", 2): TurnPredictionV1("e", 2, b"blue", selected_value="blue"),
        ("e", 4): TurnPredictionV1("e", 4, b"green", selected_value="green"),
        ("e", 6): TurnPredictionV1("e", 6, b"", parse_ok=False),
    }
    score = score_run(expected, predictions)
    assert score["scored_turns"] == 4
    assert score["exact"] == 1
    assert score["parse_failures"] == 1
    assert score["missing_predictions"] == 1
    assert score["answer_exact"] == 0.25
    assert score["state_turns"] == 2
    assert score["state_exact"] == 0.5
    assert score["by_category"]["current_query"] == {"turns": 2, "exact": 1, "parse_failures": 0}
    assert score["by_category"]["previous_query"]["parse_failures"] == 1
    assert score["by_category"]["unknown_query"]["turns"] == 1


def test_cut_and_rescue_require_correctness():
    expected = b"blue"
    baseline = TurnPredictionV1("e", 2, b"blue")
    case = MechanismCaseV1(
        episode_id="e",
        turn_index=2,
        expected=expected,
        baseline=baseline,
        cut=TurnPredictionV1("e", 2, b"red"),
        rescue=TurnPredictionV1("e", 2, b"blue"),
        relevant_swap=TurnPredictionV1("e", 2, b"green"),
        relevant_expected=b"green",
        unrelated_swap=TurnPredictionV1("e", 2, b"blue"),
    )
    wrong_baseline = MechanismCaseV1(
        episode_id="e",
        turn_index=4,
        expected=expected,
        baseline=TurnPredictionV1("e", 4, b"red"),
        cut=TurnPredictionV1("e", 4, b"red"),
        rescue=TurnPredictionV1("e", 4, b"blue"),
    )
    score = score_run([], {}, [case, wrong_baseline])
    assert score["cut_cases"] == 1
    assert score["cut_drop"] == 1.0
    assert score["rescue_cases"] == 1
    assert score["rescue_gain"] == 1.0
    assert score["relevant_swap_exact"] == 1.0
    assert score["unrelated_swap_keep"] == 1.0


def test_failed_gates_reports_below_floor_metrics():
    score = {"answer_exact": 0.25, "current_exact": 0.5, "state_exact": 1.0}
    failures = failed_gates(
        score, required={"answer_exact": 0.80, "current_exact": 0.85, "state_exact": 0.90}
    )
    assert len(failures) == 2
    assert any("answer_exact" in item for item in failures)
    assert any("current_exact" in item for item in failures)

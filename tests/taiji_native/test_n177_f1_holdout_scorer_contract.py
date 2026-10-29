"""F1 评分器的纯函数判据册：归一化、交换、family macro 的分母、三差值与门的可假性。"""

from __future__ import annotations

import json

import pytest

from scripts.training.evaluate_taiji_n1_plan177_f_f1_holdout import (
    ARMS,
    GATES,
    METRICS,
    VERDICT_PRECEDENCE,
    assert_runs_complete,
    contrasts,
    exchanged,
    family_macro,
    gate_report,
    normalize,
    split_by_coverage,
    verdict,
)


def test_normalize_strips_stop_marker_and_trailing_punctuation() -> None:
    assert normalize("红色<|end|>".encode()) == "红色"
    assert normalize("蓝色。".encode()) == "蓝色"
    assert normalize(" 绿色！ ".encode()) == "绿色"


def test_exchanged_moves_to_another_legal_value() -> None:
    assert exchanged("红色") != "红色"
    assert exchanged(exchanged("红色")) != "红色"


def _row(family: str, **overrides) -> dict:
    row = {"family_id": family, **{key: 0.5 for key in METRICS}}
    row.update({"turns": 4, "query_turns": 1})
    row.update(overrides)
    return row


def test_family_macro_ignores_episodes_without_a_denominator() -> None:
    rows = [
        _row("f1", value_exact=0.0, cut_drop=None),
        _row("f1", value_exact=1.0, cut_drop=0.5),
        _row("f2", value_exact=0.5, cut_drop=0.5),
    ]
    macro = family_macro(rows)
    assert macro["value_exact"] == pytest.approx((0.5 + 0.5) / 2)
    assert macro["cut_drop"] == pytest.approx(0.5)


def test_contrasts_are_the_three_frozen_differences() -> None:
    macro = {}
    for seed in (0, 1, 2):
        macro[f"main_credit_on__seed_{seed}"] = {"answer_exact": 0.60}
        macro[f"copy_only__seed_{seed}"] = {"answer_exact": 0.40}
        macro[f"no_state__seed_{seed}"] = {"answer_exact": 0.30}
        macro[f"main_credit_stop__seed_{seed}"] = {"answer_exact": 0.45}
    scores = contrasts(macro, (0, 1, 2))
    assert scores["main_minus_copy_only"] == pytest.approx(0.20)
    assert scores["main_minus_no_state"] == pytest.approx(0.30)
    assert scores["credit_gap"] == pytest.approx(0.15)


def test_gate_report_can_fail_and_fails_closed_on_missing_values() -> None:
    passing = gate_report({key: 1.0 for key in GATES})
    assert all(item["passed"] for item in passing.values())
    failing = gate_report({key: value - 1e-9 for key, value in GATES.items()})
    assert not any(item["passed"] for item in failing.values())
    missing = gate_report({})
    assert all(item["passed"] is False for item in missing.values())


def test_holdout_refuses_before_every_run_is_complete(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing"):
        assert_runs_complete(tmp_path)
    for arm_name in ARMS:
        run_root = tmp_path / f"{arm_name}__seed_0"
        run_root.mkdir(parents=True)
        (run_root / "run_state.json").write_text(
            json.dumps({"status": "running"}), encoding="utf-8", newline=chr(10)
        )
    with pytest.raises(ValueError, match="running"):
        assert_runs_complete(tmp_path, seeds=(0,))
    for arm_name in ARMS:
        run_root = tmp_path / f"{arm_name}__seed_0"
        (run_root / "run_state.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8", newline=chr(10)
        )
    assert_runs_complete(tmp_path, seeds=(0,))


def _passing_scores() -> dict:
    scores = {key: 1.0 for key in GATES}
    scores["copy_only_median"] = 0.0
    scores["no_state_median"] = 0.0
    return scores


def test_verdict_passes_only_with_retention() -> None:
    assert verdict(_passing_scores(), 1.0)["verdict"] == "PASS_LEARNED_VALUE"
    blocked = verdict(_passing_scores(), None)
    assert blocked["verdict"] == "INCONCLUSIVE"
    assert blocked["pass_blocked_by"] == ["old_retention_not_run"]


def test_copy_only_that_matches_main_blocks_the_pass() -> None:
    scores = _passing_scores()
    scores["copy_only_median"] = 1.0
    assert verdict(scores, 1.0)["verdict"] == "FAIL_COPY_ONLY"


def test_state_effect_failure_covers_gap_and_swap() -> None:
    gaps = _passing_scores()
    gaps["main_minus_no_state"] = 0.0
    assert verdict(gaps, 1.0)["verdict"] == "FAIL_NO_STATE_EFFECT"
    swaps = _passing_scores()
    swaps["relevant_swap_delta"] = 0.0
    assert verdict(swaps, 1.0)["verdict"] == "FAIL_NO_STATE_EFFECT"


def test_credit_gap_failure_is_learning_failure() -> None:
    scores = _passing_scores()
    scores["credit_gap"] = 0.0
    assert verdict(scores, 1.0)["verdict"] == "FAIL_NO_LEARNING"


def test_below_main_gate_is_a_generalization_failure_not_a_pass() -> None:
    scores = _passing_scores()
    scores["answer_exact"] = 0.60
    assert verdict(scores, 1.0)["verdict"] == "FAIL_NO_GENERALIZATION"


def test_precedence_reports_the_first_code_and_all_reasons() -> None:
    scores = _passing_scores()
    scores["copy_only_median"] = 1.0
    scores["credit_gap"] = 0.0
    decision = verdict(scores, 1.0)
    assert decision["verdict"] == VERDICT_PRECEDENCE[0]
    assert decision["reasons"][:2] == ["FAIL_COPY_ONLY", "FAIL_NO_LEARNING"]


def _coverage_row(family: str, covered: bool, value: float) -> dict:
    row = {"family_id": family, **{key: value for key in METRICS}}
    row.update({"turns": 4, "query_turns": 1, "entity_covered": covered})
    return row


def test_coverage_split_keeps_the_two_groups_apart() -> None:
    rows = [
        _coverage_row("f1", True, 1.0),
        _coverage_row("f2", True, 0.8),
        _coverage_row("f3", False, 0.0),
        _coverage_row("f4", False, 0.2),
    ]
    split = split_by_coverage(rows)
    assert split["covered_episodes"] == 2
    assert split["uncovered_episodes"] == 2
    assert split["covered"]["answer_exact"] == pytest.approx(0.9)
    assert split["uncovered"]["answer_exact"] == pytest.approx(0.1)


def test_coverage_split_reports_empty_side_as_empty() -> None:
    split = split_by_coverage([_coverage_row("f1", True, 1.0)])
    assert split["uncovered"] == {}
    assert split["uncovered_episodes"] == 0

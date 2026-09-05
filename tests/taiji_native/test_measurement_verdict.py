"""R0.5: versioned absolute/retention/incremental verdicts.

The legacy gate conflates capability existence, retention and incremental
gain into one pass/fail (F03).  These tests pin the three questions
separately, and prove both directions of the R0.5 gate: a true regression
must be caught, while a saturated-but-retained child must not be counted as
new learning (nor as a failure).
"""

from __future__ import annotations

import pytest

from taiji.measurement_verdict import (
    MEASUREMENT_VERDICT_FORMAT,
    MEASUREMENT_VERDICT_VERSION,
    CapabilityVerdict,
    judge_absolute,
    judge_incremental,
    judge_retention,
    verdict_report,
)


def test_absolute_judgement_uses_fixed_threshold_independent_of_parent() -> None:
    passed = judge_absolute(
        ability_id="b1_sequence_prediction",
        metric="bpb",
        value=4.5,
        threshold=6.5,
        direction="lower_is_better",
    )
    assert passed.judgement == "passed"
    assert passed.kind == "absolute"

    failed = judge_absolute(
        ability_id="b1_sequence_prediction",
        metric="bpb",
        value=7.0,
        threshold=6.5,
        direction="lower_is_better",
    )
    assert failed.judgement == "failed"

    higher = judge_absolute(
        ability_id="b2_delayed_memory",
        metric="accuracy",
        value=0.6,
        threshold=0.5,
        direction="higher_is_better",
    )
    assert higher.judgement == "passed"


def test_retention_judgement_catches_true_regression() -> None:
    verdict = judge_retention(
        ability_id="b1_sequence_prediction",
        metric="bpb",
        child_value=7.5,
        parent_value=6.0,
        tolerance=0.25,
        direction="lower_is_better",
    )
    assert verdict.judgement == "failed"
    assert verdict.threshold == pytest.approx(6.25)


def test_retention_judgement_passes_saturation_without_gain() -> None:
    verdict = judge_retention(
        ability_id="b2_delayed_memory",
        metric="accuracy",
        child_value=0.62,
        parent_value=0.60,
        tolerance=0.05,
        direction="higher_is_better",
    )
    assert verdict.judgement == "passed"

    equal = judge_retention(
        ability_id="b4_goal_action",
        metric="success_rate",
        child_value=0.55,
        parent_value=0.55,
        tolerance=0.05,
        direction="higher_is_better",
    )
    assert equal.judgement == "passed"


def test_retention_requires_both_values_for_a_verdict() -> None:
    missing = judge_retention(
        ability_id="b3_world",
        metric="transition_error",
        child_value=0.4,
        parent_value=None,
        tolerance=0.1,
        direction="lower_is_better",
    )
    assert missing.judgement == "not_applicable"


def test_incremental_requires_pre_registered_delta_when_demanded() -> None:
    met = judge_incremental(
        ability_id="b1_sequence_prediction",
        metric="bpb",
        child_value=4.2,
        baseline_value=4.8,
        target_delta=0.5,
        direction="lower_is_better",
        require_delta=True,
    )
    assert met.judgement == "passed"
    assert met.threshold == pytest.approx(4.3)

    unmet = judge_incremental(
        ability_id="b1_sequence_prediction",
        metric="bpb",
        child_value=4.6,
        baseline_value=4.8,
        target_delta=0.5,
        direction="lower_is_better",
        require_delta=True,
    )
    assert unmet.judgement == "failed"


def test_incremental_allows_saturation_when_no_gain_is_demanded() -> None:
    verdict = judge_incremental(
        ability_id="b4_goal_action",
        metric="success_rate",
        child_value=0.55,
        baseline_value=0.55,
        target_delta=0.1,
        direction="higher_is_better",
        require_delta=False,
    )
    assert verdict.judgement == "passed"
    assert verdict.threshold is None


def test_verdict_payload_is_versioned_and_sorted() -> None:
    verdict = judge_absolute(
        ability_id="b2_delayed_memory",
        metric="accuracy",
        value=0.9,
        threshold=0.5,
        direction="higher_is_better",
    )
    report = verdict_report({"b2_delayed_memory": verdict})

    assert report["format"] == MEASUREMENT_VERDICT_FORMAT
    assert report["version"] == MEASUREMENT_VERDICT_VERSION
    assert report["verdicts"][0]["ability_id"] == "b2_delayed_memory"
    assert report["verdicts"][0]["judgement"] == "passed"

    with pytest.raises(ValueError, match="unsupported verdict kind"):
        CapabilityVerdict(
            ability_id="b1",
            kind="capability",
            judgement="passed",
            metric="bpb",
            value=1.0,
            threshold=1.0,
            comparator=">=",
        )

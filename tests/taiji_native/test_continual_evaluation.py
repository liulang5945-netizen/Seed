from __future__ import annotations

import pytest

from taiji.continual_evaluation import (
    ContinualCourseManifest,
    ContinualEvaluationContractError,
    ContinualEvaluationSnapshot,
    ContinualScorecard,
    CoursePhase,
    CrossDomainAggregationError,
    MetricObservation,
    MetricSpec,
)


def _manifest() -> ContinualCourseManifest:
    return ContinualCourseManifest(
        course_id="course-s",
        phases=(
            CoursePhase(
                phase_id="s-1",
                source_digest="source-1",
                dataset_digest="dataset-1",
                train_budget_bytes=64,
                holdout_budget_bytes=16,
                course_seed=11,
                order=1,
            ),
        ),
    )


def _snapshot(
    *,
    checkpoint: str = "child",
    metric_name: str = "bpb",
    domain: str = "c3",
    parent_delta: float | None = 0.0005,
    comparison_delta: float | None = None,
    parent: str | None = "parent",
) -> ContinualEvaluationSnapshot:
    return ContinualEvaluationSnapshot(
        checkpoint_digest=checkpoint,
        parent_checkpoint_digest=parent,
        phase_id="s-1",
        owner_graph_digest="owner-v2",
        read_only_input_digest="eval-input-v1",
        observations=(
            MetricObservation(
                metric_name=metric_name,
                domain_id=domain,
                checkpoint_digest=checkpoint,
                absolute_value=4.2,
                parent_delta=parent_delta,
                parent_checkpoint_digest=parent,
                comparison_delta=comparison_delta,
                comparison_id="arm-b" if comparison_delta is not None else None,
                owner_id="byte-predictive-readout",
                read_only_input_digest="eval-input-v1",
            ),
        ),
    )


def test_metric_direction_normalizes_lower_is_better_delta() -> None:
    spec = MetricSpec(name="bpb", direction="lower", unit="bpb")
    scorecard = ContinualScorecard(
        metric_specs=(spec,),
        snapshots=(_snapshot(parent_delta=0.25),),
    )

    result = scorecard.parent_retention("bpb", "c3")

    assert result["average_parent_gain"] == pytest.approx(-0.25)
    assert result["average_forgetting"] == pytest.approx(0.25)


def test_absolute_value_cannot_be_promoted_to_forgetting_delta() -> None:
    spec = MetricSpec(
        name="active_a_retention_bpb",
        direction="lower",
        unit="bpb",
        baseline_kind="absolute",
    )
    scorecard = ContinualScorecard(
        metric_specs=(spec,),
        snapshots=(_snapshot(metric_name="active_a_retention_bpb", parent_delta=None),),
    )

    result = scorecard.parent_retention("active_a_retention_bpb", "c3")

    assert result["missing_parent_baseline_count"] == 1
    assert result["non_inferiority"] is False
    assert scorecard.can_promote() is False


def test_missing_parent_baseline_fails_closed() -> None:
    spec = MetricSpec(name="bpb", direction="lower", unit="bpb")
    scorecard = ContinualScorecard(
        metric_specs=(spec,),
        snapshots=(_snapshot(parent_delta=None, parent="parent"),),
    )

    result = scorecard.parent_retention("bpb", "c3")

    assert result["valid_parent_delta_count"] == 0
    assert result["non_inferiority"] is False
    assert scorecard.can_promote() is False


def test_cross_domain_raw_bpb_must_not_be_aggregated() -> None:
    spec = MetricSpec(name="bpb", direction="lower", unit="bpb")
    other = _snapshot(checkpoint="child-2", domain="c2", parent_delta=-0.1)
    scorecard = ContinualScorecard(
        metric_specs=(spec,),
        snapshots=(_snapshot(), other),
    )

    with pytest.raises(CrossDomainAggregationError):
        scorecard.parent_retention("bpb")

    assert scorecard.parent_retention("bpb", "c2")["sample_count"] == 1


def test_course_phase_is_evaluator_only() -> None:
    manifest = _manifest()

    assert manifest.model_context("s-1") == {}
    assert "phase_id" not in manifest.model_context("s-1")
    restored = ContinualCourseManifest.from_payload(manifest.to_payload())
    assert restored.content_digest == manifest.content_digest


def test_scorecard_round_trip_preserves_content_digest() -> None:
    scorecard = ContinualScorecard(
        metric_specs=(MetricSpec(name="bpb", direction="lower", unit="bpb"),),
        snapshots=(_snapshot(),),
        epsilon=0.001,
    )

    restored = ContinualScorecard.from_payload(scorecard.to_payload())

    assert restored.to_payload() == scorecard.to_payload()
    assert restored.content_digest == scorecard.content_digest


def test_calibrated_non_inferiority_reports_exact_zero_separately() -> None:
    spec = MetricSpec(name="bpb", direction="lower", unit="bpb")
    scorecard = ContinualScorecard(
        metric_specs=(spec,),
        snapshots=(_snapshot(parent_delta=0.0005),),
        epsilon=0.001,
    )

    result = scorecard.parent_retention("bpb", "c3")

    assert result["exact_zero_non_degradation"] is False
    assert result["non_inferiority"] is True


def test_parent_delta_requires_a_parent_checkpoint() -> None:
    with pytest.raises(ContinualEvaluationContractError):
        MetricObservation(
            metric_name="bpb",
            domain_id="c3",
            checkpoint_digest="child",
            absolute_value=4.2,
            parent_delta=0.1,
            owner_id="readout",
            read_only_input_digest="eval-input-v1",
        )

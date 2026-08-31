from __future__ import annotations

import pytest

from taiji import (
    InteractionGroupOnlineAdmission,
    InteractionGroupOutcomeFeedback,
    InteractionStructuralBridge,
    InteractionStructuralBridgeConfig,
    InteractionStructuralPressure,
    Outcome,
    StructuralRuntimeObservation,
)

SOURCE_DIGEST = "a" * 64
PARENT_DIGEST = "b" * 64


def _feedback(
    feedback_id: str,
    candidate_id: str,
    members: tuple[str, str],
    outcome_id: str,
) -> InteractionGroupOutcomeFeedback:
    return InteractionGroupOutcomeFeedback(
        feedback_id=feedback_id,
        candidate_id=candidate_id,
        member_ids=members,
        source_trace_digest=SOURCE_DIGEST,
        checkpoint_revision=7,
        parent_checkpoint_digest=PARENT_DIGEST,
        outcome=Outcome(
            outcome_id,
            reward=1.0,
            success=True,
            terminal=True,
            provenance="native-workbench",
            tick=4,
        ),
        outcome_id=outcome_id,
        event_ids=(f"{outcome_id}:event",),
        realized_interaction=0.6,
        contribution=1.0,
        resource_cost=0.8,
    )


def _admission(feedback: InteractionGroupOutcomeFeedback) -> InteractionGroupOnlineAdmission:
    return InteractionGroupOnlineAdmission(
        feedback_id=feedback.feedback_id,
        candidate_id=feedback.candidate_id,
        member_ids=feedback.member_ids,
        outcome_id=feedback.outcome_id,
        event_ids=feedback.event_ids,
        parent_checkpoint_digest=PARENT_DIGEST,
        post_learner_checkpoint_digest="c" * 64,
        status="applied",
        reason="native_terminal_outcome_admitted",
    )


def _independent_observations() -> tuple[StructuralRuntimeObservation, ...]:
    return (
        StructuralRuntimeObservation(
            network_id="network",
            region_id="region",
            tick=3,
            usage=0.4,
            resource_pressure=0.2,
            prediction_error=0.1,
            learning_gain=0.2,
            holdout_transfer=0.8,
            evidence_id="holdout:evidence",
            task_slice_id="holdout-task",
            partition="holdout",
        ),
        StructuralRuntimeObservation(
            network_id="network",
            region_id="region",
            tick=4,
            usage=0.4,
            resource_pressure=0.2,
            prediction_error=0.1,
            learning_gain=0.2,
            holdout_transfer=0.8,
            evidence_id="retention:evidence",
            task_slice_id="retention-task",
            partition="retention",
        ),
    )


def test_online_structural_bridge_requires_admitted_repeated_feedback() -> None:
    first = _feedback("1" * 64, "candidate-one", ("member-a", "member-b"), "outcome-one")
    second = _feedback("2" * 64, "candidate-two", ("member-a", "member-c"), "outcome-two")
    bridge = InteractionStructuralBridge(
        InteractionStructuralBridgeConfig(
            network_id="network",
            region_id="region",
        )
    )

    pressure = bridge.project(
        (first, second),
        (_admission(first), _admission(second)),
        _independent_observations(),
    )

    assert isinstance(pressure, InteractionStructuralPressure)
    assert pressure.mean_interaction == pytest.approx(0.6)
    assert pressure.projection.train_window_count == 2
    assert pressure.projection.holdout_window_count == 1
    assert pressure.projection.retention_window_count == 1
    assert f"online-feedback:{first.feedback_id}" in pressure.projection.evidence_ids
    assert f"online-feedback:{second.feedback_id}" in pressure.projection.evidence_ids
    assert InteractionStructuralPressure.from_payload(pressure.to_payload()) == pressure


def test_online_structural_bridge_rejects_failed_or_insufficient_evidence() -> None:
    first = _feedback("3" * 64, "candidate-three", ("member-a", "member-b"), "outcome-three")
    second = _feedback("4" * 64, "candidate-four", ("member-a", "member-c"), "outcome-four")
    bridge = InteractionStructuralBridge(
        InteractionStructuralBridgeConfig(network_id="network", region_id="region")
    )

    with pytest.raises(ValueError, match="repeated online feedback"):
        bridge.project((first,), (_admission(first),), _independent_observations())
    rejected = InteractionGroupOnlineAdmission(
        feedback_id=second.feedback_id,
        candidate_id=second.candidate_id,
        member_ids=second.member_ids,
        outcome_id=second.outcome_id,
        event_ids=second.event_ids,
        parent_checkpoint_digest=PARENT_DIGEST,
        post_learner_checkpoint_digest="c" * 64,
        status="rejected",
        reason="outcome_unsuccessful",
    )
    with pytest.raises(ValueError, match="applied feedback"):
        bridge.project(
            (first, second),
            (_admission(first), rejected),
            _independent_observations(),
        )

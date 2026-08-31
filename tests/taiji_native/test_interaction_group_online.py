from __future__ import annotations

import pytest

from taiji import (
    InteractionGroupMemberEvidence,
    InteractionGroupOnlineLearner,
    InteractionGroupOutcomeFeedback,
    InteractionGroupRecord,
    InteractionGroupTransferLearner,
    InteractionTraceEpisode,
    InteractionTraceEvent,
    Outcome,
)

SOURCE_DIGEST = "a" * 64


def _learner() -> InteractionGroupTransferLearner:
    learner = InteractionGroupTransferLearner(
        ridge=0.1,
        minimum_utility=-10.0,
        maximum_uncertainty=10.0,
    )
    learner.observe_members(
        InteractionGroupMemberEvidence(
            member_id=member,
            source_trace_digest=SOURCE_DIGEST,
            checkpoint_revision=7,
            contribution=0.5 if member == "member-a" else 0.4,
            recovery_effect=0.0,
            resource_cost=0.5,
            observations=2,
            context_count=1,
        )
        for member in ("member-a", "member-b", "member-c")
    )
    learner.observe_records(
        (
            InteractionGroupRecord(
                group_id="observed-ab",
                member_ids=("member-a", "member-b"),
                source_trace_digest=SOURCE_DIGEST,
                checkpoint_revision=7,
                contribution=1.0,
                interaction=0.8,
                uncertainty=0.0,
                resource_cost=1.0,
                owner_policy="test",
            ),
        )
    )
    return learner


def _episode(outcome: Outcome, *, suffix: str = "one") -> InteractionTraceEpisode:
    return InteractionTraceEpisode(
        episode_id=f"online-{suffix}",
        checkpoint_revision=7,
        outcome_id=f"online-{suffix}:outcome",
        events=(
            InteractionTraceEvent(
                event_id=f"online-{suffix}:event:a",
                owner_id="member-a",
                episode_id=f"online-{suffix}",
                checkpoint_revision=7,
                outcome_id=f"online-{suffix}:outcome",
                resource_cost=0.5,
            ),
            InteractionTraceEvent(
                event_id=f"online-{suffix}:event:c",
                owner_id="member-c",
                episode_id=f"online-{suffix}",
                checkpoint_revision=7,
                outcome_id=f"online-{suffix}:outcome",
                resource_cost=0.5,
            ),
        ),
        outcome=outcome.reward,
        context_id="online-context",
    )


def _feedback(
    controller: InteractionGroupOnlineLearner,
    *,
    parent_digest: str,
    outcome: Outcome,
    suffix: str,
) -> InteractionGroupOutcomeFeedback:
    selected = controller.select((("member-a", "member-c"),))
    assert selected is not None
    return InteractionGroupOutcomeFeedback.from_episode(
        candidate=selected[1],
        parent_checkpoint_digest=parent_digest,
        episode=_episode(outcome, suffix=suffix),
        outcome=outcome,
        realized_interaction=0.7,
        contribution=1.0,
    )


def test_online_outcome_admission_checkpoint_and_restart() -> None:
    controller = InteractionGroupOnlineLearner(_learner(), maximum_resource_cost=2.0)
    parent = controller.checkpoint()
    outcome = Outcome(
        "online-intent",
        reward=1.0,
        success=True,
        terminal=True,
        provenance="native-workbench",
        tick=1,
    )
    feedback = _feedback(
        controller,
        parent_digest=str(parent["checkpoint_digest"]),
        outcome=outcome,
        suffix="admit",
    )

    admission = controller.apply_feedback(feedback)

    assert admission.status == "applied"
    assert admission.reason == "native_terminal_outcome_admitted"
    assert len(controller.learner.observed_records) == 2
    assert controller.learner.candidate(("member-a", "member-c")) is None
    restored = InteractionGroupOnlineLearner.from_checkpoint(controller.checkpoint())
    assert restored.applied_feedback_ids == (feedback.feedback_id,)
    assert restored.learner.candidate(("member-a", "member-c")) is None


def test_online_feedback_rejects_without_mutating_and_rollback_blocks_replay() -> None:
    controller = InteractionGroupOnlineLearner(_learner(), maximum_resource_cost=2.0)
    parent = controller.checkpoint()
    successful = Outcome(
        "online-success",
        reward=1.0,
        success=True,
        terminal=True,
        provenance="native-workbench",
        tick=1,
    )
    feedback = _feedback(
        controller,
        parent_digest=str(parent["checkpoint_digest"]),
        outcome=successful,
        suffix="rollback",
    )
    assert controller.apply_feedback(feedback).status == "applied"

    rolled_back = controller.rollback_to(parent, feedback_id=feedback.feedback_id)

    assert rolled_back.status == "rolled_back"
    assert len(controller.learner.observed_records) == 1
    assert feedback.candidate_id in controller.blocked_candidate_ids

    parent_after_rollback = controller.checkpoint()
    failed = Outcome(
        "online-failure",
        reward=-1.0,
        success=False,
        terminal=True,
        provenance="native-workbench",
        tick=2,
    )
    candidate_after_rollback = controller.learner.candidate(
        ("member-a", "member-c"), allow_observed=False
    )
    assert candidate_after_rollback is not None
    rejected_feedback = InteractionGroupOutcomeFeedback.from_episode(
        candidate=candidate_after_rollback,
        parent_checkpoint_digest=str(parent_after_rollback["checkpoint_digest"]),
        episode=_episode(failed, suffix="reject"),
        outcome=failed,
        realized_interaction=0.7,
        contribution=1.0,
    )
    rejected = controller.apply_feedback(rejected_feedback)
    assert rejected.status == "rejected"
    assert rejected.reason == "candidate_blocked_after_prior_rejection_or_rollback"
    assert len(controller.learner.observed_records) == 1


def test_online_feedback_rejects_holdout_partition_and_tampered_checkpoint() -> None:
    controller = InteractionGroupOnlineLearner(_learner())
    parent = controller.checkpoint()
    selected = controller.select((("member-a", "member-c"),))
    assert selected is not None
    outcome = Outcome("holdout-intent", reward=1.0, success=True, terminal=True, tick=1)
    with pytest.raises(ValueError, match="source_split"):
        InteractionGroupOutcomeFeedback(
            feedback_id="b" * 64,
            candidate_id=selected[1].group_id,
            member_ids=selected[1].member_ids,
            source_trace_digest=SOURCE_DIGEST,
            checkpoint_revision=7,
            parent_checkpoint_digest=str(parent["checkpoint_digest"]),
            outcome=outcome,
            outcome_id="holdout-outcome",
            event_ids=("holdout-event",),
            realized_interaction=1.0,
            contribution=1.0,
            resource_cost=1.0,
            source_split="holdout",
        )

    tampered = dict(parent)
    tampered["minimum_interaction"] = 0.25
    with pytest.raises(ValueError, match="checkpoint digest"):
        InteractionGroupOnlineLearner.from_checkpoint(tampered)

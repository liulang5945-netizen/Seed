"""R5A-S1 tests for native consolidation, lesions, and continuation."""

from __future__ import annotations

import pytest
import torch

from taiji import (
    ExternalDescriptionArtifact,
    ExternalDescriptionTombstoneCandidate,
    GroundedFeatureExample,
    InternalizationCausalGate,
    InternalizationLearningReport,
    InternalizationLongitudinalReport,
    InternalizationStabilityGate,
    InternalizationStabilityTrial,
    InternalizedFeatureLearner,
    content_digest,
)


def _example(name: str, features: tuple[float, ...], reward: float) -> GroundedFeatureExample:
    return GroundedFeatureExample(
        example_id=f"example:{name}",
        evidence_id=f"evidence:{name}",
        outcome_id=f"outcome:{name}",
        affordance_id=f"affordance:{name}",
        action_kind="grounded-action",
        grounding=torch.tensor(features, dtype=torch.float32),
        capability_snapshot_digest="capability-sha256:s1",
        parent_checkpoint_id="checkpoint:s1-parent",
        feature_payload_digest=content_digest({"features": features}),
        reward_terms=(("outcome", reward),),
        provenance=(
            ("affordance", f"affordance:{name}"),
            ("evidence", f"evidence:{name}"),
            ("grounding", f"grounding:{name}"),
        ),
        target_reward=reward,
    )


def test_native_consolidation_improves_holdout_and_reports_lesions() -> None:
    train = (_example("left", (1.0, 0.0), 0.8), _example("right", (0.0, 1.0), 0.4))
    holdout = (_example("blend", (0.5, 0.5), 0.6),)
    learner = InternalizedFeatureLearner(feature_dim=2, learning_rate=0.5)

    report = learner.consolidate(
        train,
        holdout_examples=holdout,
        retention_examples=train,
        passes=8,
    )

    assert report.passed is True
    assert report.holdout_gain > 0.0
    assert report.holdout_internalized_lesion_loss > report.holdout_loss_after
    assert report.holdout_grounding_lesion_loss > report.holdout_loss_after
    assert report.fit_updates == len(train) * 8
    assert learner.online_updates == 0


def test_holdout_and_lesion_measurements_are_read_only() -> None:
    train = (_example("left", (1.0, 0.0), 0.8),)
    holdout = (_example("right", (0.0, 1.0), 0.4),)
    learner = InternalizedFeatureLearner(feature_dim=2)
    learner.consolidate(train, holdout_examples=holdout, retention_examples=train)
    before = content_digest(learner.checkpoint())

    learner.mean_squared_error(holdout, internalized_enabled=False)
    learner.mean_squared_error(holdout, grounding_enabled=False)

    assert content_digest(learner.checkpoint()) == before


def test_failed_trial_leaves_parent_state_unchanged() -> None:
    valid = _example("valid", (1.0, 0.0), 0.5)
    invalid = _example("invalid", (0.0, 1.0), 2.0)
    learner = InternalizedFeatureLearner(feature_dim=2)
    before = content_digest(learner.checkpoint())

    with pytest.raises(ValueError, match="outside learner bounds"):
        learner.consolidate(
            (invalid,),
            holdout_examples=(valid,),
            retention_examples=(valid,),
        )

    assert content_digest(learner.checkpoint()) == before
    assert learner.fit_updates == 0


def test_train_and_holdout_must_be_disjoint() -> None:
    shared = _example("shared", (1.0, 0.0), 0.5)
    learner = InternalizedFeatureLearner(feature_dim=2)

    with pytest.raises(ValueError, match="disjoint"):
        learner.consolidate(
            (shared,),
            holdout_examples=(shared,),
            retention_examples=(shared,),
        )


def test_train_only_pairwise_preference_is_checkpointed_and_excludes_holdout() -> None:
    low = _example("low", (1.0, 0.0), 0.2)
    high = _example("high", (0.0, 1.0), 0.8)
    holdout = _example("holdout", (0.5, 0.5), 0.5)
    learner = InternalizedFeatureLearner(
        feature_dim=2,
        learning_rate=0.5,
        pairwise_margin=0.9,
    )

    report = learner.consolidate(
        (low, high),
        holdout_examples=(holdout,),
        retention_examples=(low, high),
        ranking_pairs=((high, low),),
        passes=12,
    )

    assert report.ranking_updates > 0
    assert learner.score(high) > learner.score(low)
    restored = InternalizedFeatureLearner.from_checkpoint(learner.checkpoint())
    assert restored.pairwise_margin == learner.pairwise_margin
    assert restored.ranking_updates == learner.ranking_updates
    with pytest.raises(ValueError, match="holdout"):
        learner.consolidate(
            (low, high),
            holdout_examples=(holdout,),
            retention_examples=(low, high),
            ranking_pairs=((holdout, low),),
        )


def test_checkpoint_roundtrip_preserves_lineage_and_continuation_counters() -> None:
    train = (_example("left", (1.0, 0.0), 0.8),)
    holdout = (_example("right", (0.0, 1.0), 0.4),)
    learner = InternalizedFeatureLearner(feature_dim=2)
    report = learner.consolidate(
        train,
        holdout_examples=holdout,
        retention_examples=train,
        replay_digest="replay-sha256:s1",
    )
    checkpoint = learner.checkpoint()
    restored = InternalizedFeatureLearner.from_checkpoint(checkpoint)

    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)
    assert restored.parent_checkpoint_digest == report.parent_checkpoint_digest
    assert restored.lineage == (report.parent_checkpoint_digest,)
    restored.online_update(train[0])
    assert restored.online_updates == 1
    assert restored.fit_updates == learner.fit_updates
    assert restored.replay_digest == "replay-sha256:s1"


def test_cross_seed_stability_gate_requires_replication_and_deletion_review() -> None:
    artifact = ExternalDescriptionArtifact(
        artifact_id="description:s1-stability",
        content_digest=content_digest({"description": "opaque"}),
    )

    def run_trial(prefix: str, seed: int, task_slice: str) -> InternalizationStabilityTrial:
        learning = InternalizationLearningReport(
            parent_checkpoint_digest=f"parent:{prefix}",
            child_checkpoint_digest=f"child:{prefix}",
            replay_digest=f"replay:{prefix}",
            train_examples=2,
            holdout_examples=1,
            train_loss_before=1.0,
            train_loss_after=0.2,
            holdout_loss_before=0.8,
            holdout_loss_after=0.6,
            holdout_internalized_lesion_loss=1.0,
            holdout_grounding_lesion_loss=1.0,
            retention_loss_before=0.2,
            retention_loss_after=0.21,
            fit_updates=16,
            ranking_updates=8,
            online_updates=0,
            lineage_depth=1,
        )
        causal_gate = InternalizationCausalGate(
            external_sufficiency=True,
            internalization_necessity=True,
            grounding_necessity=True,
            checkpoint_recoverable=True,
            old_task_retention=True,
        )
        example_ids = (f"example:{prefix}:a", f"example:{prefix}:b")
        report = InternalizationLongitudinalReport(
            learning=learning,
            external_description_quality=1.0,
            external_removed_selection_quality=1.0,
            internalized_lesion_selection_quality=0.0,
            grounding_lesion_selection_quality=0.0,
            retention_selection_quality=1.0,
            restored_selection_quality=1.0,
            checkpoint_digest=f"checkpoint:{prefix}",
            checkpoint_recoverable=True,
            causal_gate=causal_gate,
            lifecycle_statuses=tuple((example_id, "internalized") for example_id in example_ids),
            tombstone_candidate=ExternalDescriptionTombstoneCandidate(
                artifact_id=artifact.artifact_id,
                artifact_content_digest=artifact.content_digest,
                example_ids=example_ids,
                checkpoint_digest=f"checkpoint:{prefix}",
                causal_gate=causal_gate,
                manifest_revision=artifact.manifest_revision,
            ),
        )
        return InternalizationStabilityTrial(
            trial_id=f"trial:{prefix}",
            seed=seed,
            task_slice=task_slice,
            report=report,
        )

    trials = (
        run_trial("seed-a", 11, "workspace-read-v1"),
        run_trial("seed-b", 29, "workspace-read-v2"),
    )
    stability = InternalizationStabilityGate(artifact).evaluate(trials)

    assert stability.passed is True
    assert stability.unique_seeds == (11, 29)
    assert stability.unique_task_slices == ("workspace-read-v1", "workspace-read-v2")
    assert stability.minimum_holdout_gain >= 0.05
    assert stability.minimum_internalization_drop >= 0.5
    assert stability.minimum_grounding_drop >= 0.5
    assert stability.independent_deletion_review.passed is True
    assert stability.independent_deletion_review.to_payload()["disposition"] == (
        "review_only_no_physical_deletion"
    )
    assert stability.evidence_digest == content_digest(stability.to_payload())
    checkpoint = InternalizationStabilityGate(artifact).checkpoint(trials)
    assert checkpoint["format"] == "taiji-internalization-stability-v1"
    assert "path" not in checkpoint

    single_trial = InternalizationStabilityGate(artifact).evaluate((trials[0],))
    assert single_trial.passed is False
    assert single_trial.stability_passed is False


def test_independent_deletion_review_rejects_artifact_mismatch() -> None:
    source = ExternalDescriptionArtifact(
        artifact_id="description:source",
        content_digest=content_digest({"description": "source"}),
    )
    wrong_artifact = ExternalDescriptionArtifact(
        artifact_id="description:wrong",
        content_digest=content_digest({"description": "wrong"}),
    )
    learning = InternalizationLearningReport(
        parent_checkpoint_digest="parent:review",
        child_checkpoint_digest="child:review",
        replay_digest="replay:review",
        train_examples=2,
        holdout_examples=1,
        train_loss_before=1.0,
        train_loss_after=0.2,
        holdout_loss_before=0.8,
        holdout_loss_after=0.6,
        holdout_internalized_lesion_loss=1.0,
        holdout_grounding_lesion_loss=1.0,
        retention_loss_before=0.2,
        retention_loss_after=0.21,
        fit_updates=16,
        ranking_updates=8,
        online_updates=0,
        lineage_depth=1,
    )
    causal_gate = InternalizationCausalGate(True, True, True, True, True)
    example_ids = ("example:review:a", "example:review:b")
    report = InternalizationLongitudinalReport(
        learning=learning,
        external_description_quality=1.0,
        external_removed_selection_quality=1.0,
        internalized_lesion_selection_quality=0.0,
        grounding_lesion_selection_quality=0.0,
        retention_selection_quality=1.0,
        restored_selection_quality=1.0,
        checkpoint_digest="checkpoint:review",
        checkpoint_recoverable=True,
        causal_gate=causal_gate,
        lifecycle_statuses=tuple((example_id, "internalized") for example_id in example_ids),
        tombstone_candidate=ExternalDescriptionTombstoneCandidate(
            artifact_id=source.artifact_id,
            artifact_content_digest=source.content_digest,
            example_ids=example_ids,
            checkpoint_digest="checkpoint:review",
            causal_gate=causal_gate,
            manifest_revision=source.manifest_revision,
        ),
    )
    trial = InternalizationStabilityTrial(
        trial_id="trial:artifact-mismatch",
        seed=1,
        task_slice="review",
        report=report,
    )

    review = (
        InternalizationStabilityGate(wrong_artifact).evaluate((trial,)).independent_deletion_review
    )
    assert review.passed is False
    assert any("artifact_binding" in failure for failure in review.failures)

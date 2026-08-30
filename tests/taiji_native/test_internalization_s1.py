"""R5A-S1 tests for native consolidation, lesions, and continuation."""

from __future__ import annotations

import pytest
import torch

from taiji import (
    GroundedFeatureExample,
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

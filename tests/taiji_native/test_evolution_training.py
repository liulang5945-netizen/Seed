from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from taiji.evolution_experience import EvolutionExperience
from taiji.evolution_training import (
    EVOLUTION_REPLAY_TIERS,
    BoundedReplaySelection,
    NativeEvolutionTrainer,
    select_bounded_replay,
)
from taiji.internalization import content_digest


def _experience(
    experience_id: str,
    *,
    partition: str,
    capability_id: str,
    success: bool = True,
    reward: float | None = None,
    correction: bool = False,
) -> EvolutionExperience:
    components = {} if reward is None else {"quality": reward}
    return EvolutionExperience(
        experience_id=experience_id,
        source_kind="workbench",
        source_id="seed.workbench.route",
        source_version="1",
        source_digest=content_digest({"source": "seed.workbench.route", "version": "1"}),
        parent_checkpoint_digest="a" * 64,
        partition=partition,
        status="success" if success else "error",
        success=success,
        input_digest=content_digest({"request": capability_id}),
        capability_id=capability_id,
        capability_snapshot_id="b" * 64,
        reward_components=components,
        result_digest=content_digest({"result": experience_id}),
        user_correction_digest=(
            content_digest({"correction": experience_id}) if correction else ""
        ),
    )


def test_native_route_credit_admits_train_only_update_and_roundtrips() -> None:
    trainer = NativeEvolutionTrainer(feature_dim=64)
    train = (
        _experience("train-1", partition="train", capability_id="editor.open", reward=1.0),
        _experience("train-2", partition="train", capability_id="editor.open", reward=1.0),
    )
    holdout = (
        _experience("holdout-1", partition="holdout", capability_id="editor.open", reward=1.0),
    )
    retention = (
        _experience("retention-1", partition="retention", capability_id="mcp.list", reward=0.0),
    )

    before = trainer.checkpoint()
    report = trainer.consolidate(
        train,
        holdout_experiences=holdout,
        retention_experiences=retention,
    )

    assert report.admitted is True
    assert report.rolled_back is False
    assert report.native_holdout_loss < report.frozen_holdout_loss
    assert report.replay_only_holdout_loss == pytest.approx(report.frozen_holdout_loss)
    assert trainer.revision == 1
    assert trainer.consumed_experience_ids == ("train-1", "train-2")

    restored = NativeEvolutionTrainer.from_checkpoint(trainer.checkpoint())
    assert content_digest(restored.checkpoint()) == content_digest(trainer.checkpoint())
    assert restored.learner.score(restored.example(holdout[0])) == pytest.approx(
        trainer.learner.score(trainer.example(holdout[0]))
    )
    assert before["checkpoint_digest"] != trainer.checkpoint()["checkpoint_digest"]


def test_native_route_credit_never_uses_holdout_or_outcome_fields_as_features() -> None:
    trainer = NativeEvolutionTrainer(feature_dim=64)
    positive = _experience(
        "train-positive", partition="train", capability_id="editor.open", reward=1.0
    )
    negative = _experience(
        "train-negative", partition="train", capability_id="editor.open", reward=-1.0, success=False
    )
    different_route = _experience(
        "train-mcp", partition="train", capability_id="mcp.list", reward=1.0
    )

    assert torch.equal(trainer.encoder.encode(positive), trainer.encoder.encode(negative))
    assert not torch.equal(trainer.encoder.encode(positive), trainer.encoder.encode(different_route))


def test_native_route_credit_rejects_partition_overlap_and_tampered_checkpoint() -> None:
    trainer = NativeEvolutionTrainer(feature_dim=16)
    train = _experience("same", partition="train", capability_id="editor.open")
    holdout = _experience("same", partition="holdout", capability_id="editor.open")
    with pytest.raises(ValueError, match="partitions must be disjoint"):
        trainer.consolidate(
            (train,),
            holdout_experiences=(holdout,),
            retention_experiences=(_experience("retention", partition="retention", capability_id="mcp.list"),),
        )

    tampered = deepcopy(trainer.checkpoint())
    tampered["revision"] = 9
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        NativeEvolutionTrainer.from_checkpoint(tampered)


def _replay_pool() -> tuple[EvolutionExperience, ...]:
    corrections = tuple(
        _experience(
            f"train-correction-{index}",
            partition="train",
            capability_id="editor.open",
            reward=-0.5,
            success=False,
            correction=True,
        )
        for index in range(2)
    )
    failures = tuple(
        _experience(
            f"train-failure-{index}",
            partition="train",
            capability_id="mcp.list",
            reward=-1.0,
            success=False,
        )
        for index in range(3)
    )
    successes = tuple(
        _experience(
            f"train-success-{index}",
            partition="train",
            capability_id="editor.open",
            reward=1.0,
        )
        for index in range(20)
    )
    return corrections + failures + successes


def test_bounded_replay_keeps_correction_and_failure_evidence_under_capacity() -> None:
    pool = _replay_pool()
    selection = select_bounded_replay(pool, capacity=8)

    assert isinstance(selection, BoundedReplaySelection)
    assert len(selection.selected_experience_ids) == 8
    assert selection.capacity == 8
    assert selection.considered_experiences == len(pool)

    selected = set(selection.selected_experience_ids)
    assert {item.experience_id for item in pool if item.user_correction_digest} <= selected
    assert {
        item.experience_id
        for item in pool
        if not item.success and not item.user_correction_digest
    } <= selected
    assert any(item.experience_id in selected for item in pool if item.success)
    assert selection.tier_counts["correction"] == 2
    assert selection.tier_counts["failure"] == 3
    assert selection.tier_counts["success"] == 3


def test_bounded_replay_never_evicts_retained_tiers_when_capacity_is_tight() -> None:
    pool = _replay_pool()
    selection = select_bounded_replay(pool, capacity=5)

    assert len(selection.selected_experience_ids) == 5
    assert selection.tier_counts["correction"] == 2
    assert selection.tier_counts["failure"] == 3
    assert selection.tier_counts["success"] == 0

    with pytest.raises(ValueError, match="capacity cannot drop retained evidence"):
        select_bounded_replay(pool, capacity=4)


def test_bounded_replay_selection_is_content_addressed_and_reproducible() -> None:
    pool = _replay_pool()
    first = select_bounded_replay(pool, capacity=8)
    second = select_bounded_replay(tuple(reversed(pool)), capacity=8)

    assert first.selected_experience_ids == second.selected_experience_ids
    assert first.selection_digest == second.selection_digest

    restored = BoundedReplaySelection.from_payload(first.to_payload())
    assert restored.selection_digest == first.selection_digest
    assert restored.selected_experience_ids == first.selected_experience_ids

    tampered = deepcopy(first.to_payload())
    tampered["selected_experience_ids"] = list(tampered["selected_experience_ids"])[:-1]
    with pytest.raises(ValueError, match="selection digest mismatch"):
        BoundedReplaySelection.from_payload(tampered)


def test_bounded_replay_rejects_non_train_partitions_and_unknown_capacity() -> None:
    assert EVOLUTION_REPLAY_TIERS == ("correction", "failure", "success")

    holdout = _experience("holdout-1", partition="holdout", capability_id="editor.open")
    with pytest.raises(ValueError, match="different partition"):
        select_bounded_replay((holdout,), capacity=4)

    with pytest.raises(ValueError, match="capacity must be positive"):
        select_bounded_replay(_replay_pool(), capacity=0)

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from taiji.evolution_experience import EvolutionExperience
from taiji.evolution_training import NativeEvolutionTrainer
from taiji.internalization import content_digest


def _experience(
    experience_id: str,
    *,
    partition: str,
    capability_id: str,
    success: bool = True,
    reward: float | None = None,
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

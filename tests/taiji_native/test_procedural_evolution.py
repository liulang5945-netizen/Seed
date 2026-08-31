from __future__ import annotations

from copy import deepcopy

import pytest

from taiji.evolution_experience import EvolutionExperience
from taiji.internalization import content_digest
from taiji.procedural_evolution import NativeProceduralMemoryTrainer
from taiji.procedural_memory import ProceduralMemoryLearner


def _experience(
    experience_id: str,
    *,
    partition: str,
    capability_id: str,
    success: bool = True,
) -> EvolutionExperience:
    return EvolutionExperience(
        experience_id=experience_id,
        source_kind="workbench",
        source_id="seed.workbench.procedure",
        source_version="1",
        source_digest=content_digest({"source": "seed.workbench.procedure", "version": "1"}),
        parent_checkpoint_digest="a" * 64,
        partition=partition,
        status="success" if success else "error",
        success=success,
        capability_id=capability_id,
        capability_snapshot_id="b" * 64,
        episode_id=f"episode-{experience_id}",
        tick=1,
        result_digest=content_digest({"result": experience_id}),
    )


def test_procedural_intake_discovers_actions_and_preserves_retention() -> None:
    trainer = NativeProceduralMemoryTrainer(cue_dim=64, epochs=120)
    train = (
        _experience("train-editor", partition="train", capability_id="editor.open"),
        _experience("train-mcp", partition="train", capability_id="mcp.list"),
    )
    holdout = (_experience("holdout-mcp", partition="holdout", capability_id="mcp.list"),)
    retention = (
        _experience("retention-editor", partition="retention", capability_id="editor.open"),
    )
    report = trainer.consolidate(
        train,
        holdout_experiences=holdout,
        retention_experiences=retention,
    )

    assert report.admitted is True
    assert report.native_holdout_accuracy > report.frozen_holdout_accuracy
    assert report.replay_only_holdout_accuracy == report.frozen_holdout_accuracy
    assert report.native_retention_accuracy >= report.frozen_retention_accuracy
    assert trainer.learner.action_kinds == ("editor.open", "mcp.list")
    assert trainer.predict(holdout[0]) == "mcp.list"
    assert trainer.predict(retention[0]) == "editor.open"
    restored = NativeProceduralMemoryTrainer.from_checkpoint(trainer.checkpoint())
    assert content_digest(restored.checkpoint()) == content_digest(trainer.checkpoint())


def test_failed_train_experience_is_excluded_and_not_consumed() -> None:
    trainer = NativeProceduralMemoryTrainer(cue_dim=32, epochs=80)
    train = (
        _experience("train-good", partition="train", capability_id="editor.open"),
        _experience("train-mcp", partition="train", capability_id="mcp.list"),
        _experience(
            "train-failed", partition="train", capability_id="mcp.list", success=False
        ),
    )
    holdout = (_experience("holdout-mcp", partition="holdout", capability_id="mcp.list"),)
    retention = (_experience("retention-editor", partition="retention", capability_id="editor.open"),)
    report = trainer.consolidate(
        train,
        holdout_experiences=holdout,
        retention_experiences=retention,
    )

    assert report.admitted is True
    assert report.excluded_experience_ids == ("train-failed",)
    assert "train-failed" not in trainer.consumed_experience_ids
    assert trainer.consumed_experience_ids == ("train-good", "train-mcp")


def test_procedural_action_readout_expands_without_erasing_old_weights() -> None:
    learner = ProceduralMemoryLearner(cue_dim=4)
    learner.prepare(("editor.open",))
    assert learner.readout is not None
    learner.readout.weight[0].fill_(0.75)
    learner.readout.bias[0] = 0.25
    learner.prepare(("editor.open", "mcp.list"))

    assert learner.action_kinds == ("editor.open", "mcp.list")
    assert learner.readout is not None
    assert float(learner.readout.weight[0, 0]) == pytest.approx(0.75)
    assert float(learner.readout.bias[0]) == pytest.approx(0.25)
    assert learner.readout.weight[1].abs().sum().item() == 0.0


def test_procedural_checkpoint_tamper_and_partition_overlap_fail_closed() -> None:
    trainer = NativeProceduralMemoryTrainer(cue_dim=32)
    train = _experience("same", partition="train", capability_id="editor.open")
    holdout = _experience("same", partition="holdout", capability_id="editor.open")
    with pytest.raises(ValueError, match="partitions must be disjoint"):
        trainer.consolidate(
            (train,),
            holdout_experiences=(holdout,),
            retention_experiences=(
                _experience("retention", partition="retention", capability_id="editor.open"),
            ),
        )
    tampered = deepcopy(trainer.checkpoint())
    tampered["revision"] = 7
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        NativeProceduralMemoryTrainer.from_checkpoint(tampered)

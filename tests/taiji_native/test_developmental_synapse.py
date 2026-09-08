from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.training.migrate_taiji_f1_checkpoint import migrate_checkpoint
from taiji import (
    DevelopmentalSynapseBundle,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24,),
        synapse_fan_in=6,
        motor_fan_in=12,
        memory_units=24,
        memory_fan_in=6,
        memory_meta_dim=16,
        memory_readout_fan_in=12,
        seed=71,
    )


def test_f1_checkpoint_migrates_old_weights_to_slow_and_zero_fast() -> None:
    model = Taiji(_config(), episode_id="r1-source")
    model.learn_bytes(b"abba-caba-abba", epochs=2, learn_fabric=False)
    source_checkpoint = model.checkpoint()

    bundle = DevelopmentalSynapseBundle.from_taiji_checkpoint(source_checkpoint)
    context_bank = bundle.bank("predictive_context.recurrent")
    readout_bank = bundle.bank("predictive_readout.synapses")

    assert bundle.fast_is_zero
    assert torch.equal(
        context_bank.slow_weight,
        source_checkpoint["predictive_context"]["recurrent"]["edge_weight"],
    )
    assert torch.equal(
        readout_bank.slow_weight,
        source_checkpoint["predictive_readout"]["synapses"]["edge_weight"],
    )
    assert context_bank.effective_parameter_count == model.predictive_context.recurrent.edge_count
    assert readout_bank.effective_parameter_count == model.predictive_readout.synapses.edge_count
    assert bundle.state_scalar_count > bundle.effective_parameter_count


def test_mounted_r1_overlay_preserves_score_generation_and_fresh_restore() -> None:
    source = Taiji(_config(), episode_id="r1-equivalence")
    source.learn_bytes(b"abba-caba-abba", epochs=2, learn_fabric=False)
    checkpoint = source.checkpoint()
    baseline = Taiji.from_checkpoint(deepcopy(checkpoint))
    migrated = Taiji.from_checkpoint(deepcopy(checkpoint))

    migration = migrated.migrate_f1_to_developmental_synapses()

    assert migration["fast_is_zero"] is True
    assert migration["source_checkpoint_digest"] == content_digest(checkpoint)
    assert migrated.developmental_f1_enabled
    assert migrated.parameter_count() == baseline.parameter_count()

    baseline_score = baseline.score_bytes(b"caba-abba")
    migrated_score = migrated.score_bytes(b"caba-abba")
    assert migrated_score == baseline_score
    assert migrated.generate(b"caba", 12) == baseline.generate(b"caba", 12)

    migrated_checkpoint = migrated.checkpoint()
    restored = Taiji.from_checkpoint(deepcopy(migrated_checkpoint))
    assert restored.developmental_f1_enabled
    assert content_digest(restored.checkpoint()) == content_digest(migrated_checkpoint)
    assert restored.score_bytes(b"caba-abba") == baseline_score


def test_r1_overlay_is_read_only_until_r2() -> None:
    model = Taiji(_config(), episode_id="r1-read-only")
    model.migrate_f1_to_developmental_synapses()

    with pytest.raises(RuntimeError, match="read-only until R2"):
        model.observe(97, learn=True, readout="predictive")

    model.unmount_developmental_f1()
    step = model.observe(97, learn=True, readout="predictive")
    assert step.observed_symbol == 97


def test_r1_disk_migration_gate_preserves_source_and_fresh_restore() -> None:
    root = Path("output")
    source_path = root / "taiji_r1_disk_migration_source_test.pt"
    output_path = root / "taiji_r1_disk_migration_output_test.pt"
    try:
        source = Taiji(_config(), episode_id="r1-disk")
        source.learn_bytes(b"abba-caba", epochs=1, learn_fabric=False)
        torch.save(source.checkpoint(), source_path)

        report = migrate_checkpoint(source_path, output_path)

        assert report["status"] == "passed"
        assert report["source_unchanged"] is True
        assert report["fresh_restore_matches"] is True
        assert report["fast_is_zero"] is True
        assert output_path.is_file()
    finally:
        source_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def test_r2_fast_wake_keeps_parent_owners_immutable_and_persists_replay() -> None:
    model = Taiji(_config(), episode_id="r2-fast")
    model.learn_bytes(b"abba-caba", epochs=1, learn_fabric=False)
    model.migrate_f1_to_developmental_synapses()
    parent = deepcopy(model.checkpoint())
    parent_readout_digest = content_digest(parent["predictive_readout"])
    parent_context_digest = content_digest(parent["predictive_context"])

    model.set_developmental_f1_learning_mode("fast")
    model.learn_bytes(b"caba-abba", epochs=1, learn_fabric=False)

    assert model.developmental_f1_bundle is not None
    assert model.developmental_f1_bundle.fast_is_zero is False
    assert model.developmental_f1_replay_count > 0
    assert content_digest(model.predictive_readout.to_payload()) == parent_readout_digest
    assert content_digest(model.predictive_context.to_payload()) == parent_context_digest

    checkpoint = model.checkpoint()
    assert Taiji.DEVELOPMENTAL_F1_REPLAY_KEY in checkpoint
    restored = Taiji.from_checkpoint(deepcopy(checkpoint))
    assert restored.developmental_f1_learning_mode == "read_only"
    assert restored.developmental_f1_replay_count == model.developmental_f1_replay_count
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)

    with pytest.raises(RuntimeError, match="read-only until R2"):
        restored.observe(97, learn=True, readout="predictive")


def test_r2_slow_only_and_fast_replay_consolidation_are_distinct() -> None:
    slow = Taiji(_config(), episode_id="r2-slow")
    slow.learn_bytes(b"abba-caba", epochs=1, learn_fabric=False)
    slow.migrate_f1_to_developmental_synapses()
    slow_parent = deepcopy(slow.checkpoint())
    slow.set_developmental_f1_learning_mode("slow")
    slow.learn_bytes(b"caba-abba", epochs=1, learn_fabric=False)
    assert slow.developmental_f1_bundle is not None
    assert slow.developmental_f1_bundle.fast_is_zero
    assert content_digest(slow.checkpoint()[Taiji.DEVELOPMENTAL_F1_KEY]) != content_digest(
        slow_parent[Taiji.DEVELOPMENTAL_F1_KEY]
    )

    candidate = Taiji.from_checkpoint(deepcopy(slow_parent))
    candidate.set_developmental_f1_learning_mode("fast_slow")
    candidate.learn_bytes(b"caba-abba", epochs=1, learn_fabric=False)
    assert candidate.developmental_f1_bundle is not None
    assert candidate.developmental_f1_bundle.fast_is_zero is False
    before_replay = content_digest(candidate.checkpoint()[Taiji.DEVELOPMENTAL_F1_KEY])
    replay = candidate.replay_developmental_f1(consolidate=True, clear_replay=True)
    assert replay["events"] > 0
    assert replay["changed"] is True
    assert candidate.developmental_f1_bundle.fast_is_zero
    assert content_digest(candidate.checkpoint()[Taiji.DEVELOPMENTAL_F1_KEY]) != before_replay

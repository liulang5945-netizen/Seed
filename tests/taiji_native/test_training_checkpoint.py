from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from seed_platform.evolution_ledger import EvolutionExperienceLedger
from seed_platform.training_checkpoint import NativeTrainingCheckpoint
from taiji import Taiji, TaijiConfig
from taiji.evolution_experience import EvolutionCorpusArtifact, EvolutionExperience
from taiji.internalization import content_digest


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(8,),
        synapse_fan_in=2,
        motor_fan_in=4,
        memory_units=16,
        memory_fan_in=2,
        memory_readout_fan_in=2,
        memory_meta_dim=4,
        memory_time_dim=2,
        memory_episode_dim=2,
        lateral_fan_in=2,
        concept_capacity=8,
        seed=17,
    )


def _ledger() -> EvolutionExperienceLedger:
    ledger = EvolutionExperienceLedger()
    corpus = EvolutionCorpusArtifact(
        corpus_id="fixture:training",
        source_kind="skill_artifact",
        source_id="skill.training",
        source_version="1",
        source_digest=content_digest({"source": "skill.training"}),
        unit_kind="knowledge",
        content={"title": "training fixture"},
    )
    ledger.add_corpus(corpus)
    ledger.admit_corpus(corpus.artifact_digest, admission_revision="training-admission")
    ledger.append(
        EvolutionExperience(
            experience_id="training-episode-1",
            source_kind="skill",
            source_id="skill.training",
            source_version="1",
            source_digest=corpus.source_digest,
            parent_checkpoint_digest="a" * 64,
            partition="train",
            status="success",
            success=True,
            episode_id="episode-1",
            tick=1,
            result_digest=content_digest({"status": "ok"}),
            reward_components={"quality": 1.0},
        )
    )
    return ledger


def test_training_checkpoint_roundtrips_native_model_and_ledger() -> None:
    model = Taiji(_config())
    ledger = _ledger()
    record = NativeTrainingCheckpoint.create(
        model,
        ledger,
        checkpoint_kind="parent",
        checkpoint_id="parent-1",
        learner_state={"updates": 0, "owner": "native"},
        resource_ledger={"cpu_ms": 0.0, "checkpoint_bytes": 1.0},
    )
    payload = record.to_payload()
    restored_record = NativeTrainingCheckpoint.from_payload(payload)
    assert restored_record.checkpoint_digest == record.checkpoint_digest
    assert restored_record.dataset_digest == record.dataset_digest
    assert restored_record.ledger_cursor == record.ledger_cursor

    target = Taiji(_config(), episode_id="restore-target")
    restored_ledger = restored_record.restore_into(target, ledger)
    assert content_digest(target.checkpoint()) == record.model_digest
    assert restored_ledger.checkpoint()["checkpoint_digest"] == record.ledger_checkpoint["checkpoint_digest"]
    assert torch.equal(torch.random.get_rng_state(), record.random_state)


def test_training_checkpoint_rejects_tamper_and_dataset_drift_before_restore() -> None:
    model = Taiji(_config())
    ledger = _ledger()
    record = NativeTrainingCheckpoint.create(
        model,
        ledger,
        checkpoint_kind="trial",
        checkpoint_id="trial-1",
        parent_checkpoint_digest="b" * 64,
        learner_state={"updates": 1},
    )
    tampered = deepcopy(record.to_payload())
    tampered["dataset_digest"] = "c" * 64
    with pytest.raises(ValueError, match="training checkpoint digest mismatch"):
        NativeTrainingCheckpoint.from_payload(tampered)

    drifted = EvolutionExperienceLedger.from_checkpoint(ledger.checkpoint())
    drifted.append(
        EvolutionExperience(
            experience_id="training-episode-2",
            source_kind="skill",
            source_id="skill.training",
            source_version="1",
            source_digest="d" * 64,
            parent_checkpoint_digest="a" * 64,
            partition="train",
            status="success",
            success=True,
            episode_id="episode-2",
            tick=2,
            result_digest=content_digest({"status": "ok-2"}),
        )
    )
    target = Taiji(_config(), episode_id="drift-target")
    before = content_digest(target.checkpoint())
    with pytest.raises(ValueError, match="training checkpoint ledger drift"):
        record.restore_into(target, drifted)
    assert content_digest(target.checkpoint()) == before


def test_non_parent_checkpoint_requires_lineage() -> None:
    model = Taiji(_config())
    with pytest.raises(ValueError, match="requires parent_checkpoint_digest"):
        NativeTrainingCheckpoint.create(
            model,
            _ledger(),
            checkpoint_kind="admitted",
            checkpoint_id="admitted-without-parent",
        )

from __future__ import annotations

import json

import pytest
import torch

from scripts.training.eval_taiji_foundation_baseline import (
    _evaluate_loaded_b1,
    _indexed_checkpoint_paths,
    _reuse_b1_measurement,
)
from taiji import Taiji, TaijiConfig
from taiji.foundation_evaluation import FoundationManifest
from taiji.foundation_tasks import SequencePredictionCorpus
from taiji.foundation_training import FoundationTrainingDataset
from taiji.internalization import content_digest


def _config(seed: int) -> TaijiConfig:
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
        seed=seed,
    )


def _joint_child(path, *, seed: int = 11, fabric_learning: bool = False) -> None:
    child = Taiji(_config(seed), episode_id="child")
    parent = Taiji(_config(seed), episode_id="parent")
    payload = {
        "format": "taiji-native-joint-training-v1",
        "version": 4,
        "training_phases": ["sequence"],
        "sequence_fabric_learning": fabric_learning,
        "sequence_predictive_context_mode": "private-plastic-temporal-v1",
        "dataset_digest": "dataset",
        "protected_dataset_digest": "protected",
        "model": child.checkpoint(),
        "parent_model": parent.checkpoint(),
    }
    payload["checkpoint_digest"] = content_digest(payload)
    torch.save(payload, path)


def test_indexed_checkpoint_paths_require_one_path_per_seed(tmp_path) -> None:
    checkpoint = tmp_path / "child.pt"
    checkpoint.write_bytes(b"checkpoint")

    result = _indexed_checkpoint_paths(
        [["11", str(checkpoint)], ["29", str(checkpoint)]],
        seeds=(11, 29),
    )

    assert result == {11: checkpoint, 29: checkpoint}
    with pytest.raises(ValueError, match="exactly one"):
        _indexed_checkpoint_paths([["11", str(checkpoint)]], seeds=(11, 29))


def test_loaded_b1_uses_child_and_parent_without_holdout_writes(tmp_path) -> None:
    checkpoint = tmp_path / "child.pt"
    _joint_child(checkpoint)
    corpus = SequencePredictionCorpus(
        train=b"abcd" * 4,
        holdout=b"bcda" * 2,
        retention=b"cdab" * 2,
    )

    measurement = _evaluate_loaded_b1({11: checkpoint}, corpus)

    assert measurement.ability_id == "b1_sequence_prediction"
    assert measurement.holdout_updates == 0
    assert set(measurement.baseline_metrics) == {
        "random",
        "frozen_parent",
        "simple_rule",
        "hash_only",
    }
    assert any("trained_child_checkpoint_evaluation=true" in item for item in measurement.evidence)


def test_loaded_b1_rejects_fabric_plastic_child(tmp_path) -> None:
    checkpoint = tmp_path / "child.pt"
    _joint_child(checkpoint, fabric_learning=True)
    corpus = SequencePredictionCorpus(train=b"abcd", holdout=b"bcda", retention=b"cdab")

    with pytest.raises(ValueError, match="sequence_fabric_learning=false"):
        _evaluate_loaded_b1({11: checkpoint}, corpus)


def test_reused_b1_report_requires_checkpoint_provenance(tmp_path) -> None:
    manifest = FoundationManifest.load("plans/manifests/taiji_foundation_baseline_v1.json")
    checkpoint = tmp_path / "child.pt"
    _joint_child(checkpoint)
    dataset = FoundationTrainingDataset(
        train=b"a",
        holdout=b"b",
        retention=b"c",
        source_files=(("phase-b.jsonl", "source-digest"),),
        partition_seed=11,
        profile="foundation",
        excluded_dataset_digest="phase-a-digest",
    )
    report = tmp_path / "b1.json"
    report.write_text(
        json.dumps(
            {
                "format": "taiji-foundation-evaluation-v1",
                "manifest_digest": manifest.digest,
                "checkpoint_gate_status": "passed",
                "checkpoint_evaluation": {
                    "mode": True,
                    "checkpoints": [{"seed": 11, "path": str(checkpoint)}],
                    "b1_dataset_digests": {"11": dataset.digest},
                    "b1_protected_dataset_digests": {"11": "phase-a-digest"},
                },
                "measurements": [
                    {
                        "ability_id": "b1_sequence_prediction",
                        "status": "passed",
                        "primary_metric": "bits_per_byte",
                        "metric_direction": "lower_is_better",
                        "metric_value": 1.0,
                        "baseline_metrics": {"simple_rule": 2.0},
                        "sample_counts": {},
                        "holdout_updates": 0,
                        "evidence": ["trained_child_checkpoint_evaluation=true"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    measurement = _reuse_b1_measurement(
        report,
        manifest=manifest,
        checkpoints={11: checkpoint},
        datasets={11: dataset},
    )

    assert measurement.status == "passed"

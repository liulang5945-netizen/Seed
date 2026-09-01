from __future__ import annotations

from taiji import TaijiConfig
from taiji.foundation_tasks import SequencePredictionCorpus, SequencePredictionTask


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


def test_sequence_task_returns_a_real_measurement_and_no_holdout_mutation() -> None:
    corpus = SequencePredictionCorpus(
        train=(b"alpha-beta-" * 12),
        holdout=(b"alpha-gamma-" * 5),
        retention=(b"alpha-delta-" * 5),
    )

    measurement = SequencePredictionTask(
        _config(11), seeds=(11, 29, 47), epochs=1
    ).evaluate(corpus)

    assert measurement.ability_id == "b1_sequence_prediction"
    assert measurement.status in {"passed", "failed"}
    assert measurement.metric_value is not None
    assert measurement.sample_counts == {
        "train": len(corpus.train),
        "holdout": len(corpus.holdout),
        "retention": len(corpus.retention),
    }
    assert measurement.holdout_updates == 0
    assert all(kind in measurement.baseline_metrics for kind in (
        "random",
        "frozen_parent",
        "simple_rule",
        "hash_only",
    ))
    assert any("seed_metrics" in item for item in measurement.evidence)

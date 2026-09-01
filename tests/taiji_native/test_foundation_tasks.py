from __future__ import annotations

import torch

from taiji import Outcome, TaijiConfig, WorldAction, WorldInterventionCase, WorldObject, WorldState
from taiji.foundation_tasks import (
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    DelayedMemoryTask,
    MemoryEpisode,
    SequencePredictionCorpus,
    SequencePredictionTask,
    WorldTransitionCorpus,
    WorldTransitionTask,
)


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


def test_delayed_memory_task_recalls_trained_cues_without_holdout_writes() -> None:
    train = tuple(
        MemoryEpisode(
            memory_id=f"train-{index}",
            cue=ord("A") + index,
            action=ord("0") + index % 2,
            outcome=ord("+") if index % 2 == 0 else ord("-"),
        )
        for index in range(8)
    )
    queries = tuple(
        DelayedMemoryQuery(
            query_id=f"query-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(train)
    )
    retention = tuple(
        DelayedMemoryQuery(
            query_id=f"retention-{index}",
            cue=query.cue,
            expected_action=query.expected_action,
        )
        for index, query in enumerate(queries)
    )
    corpus = DelayedMemoryCorpus(train=train, holdout=queries, retention=retention)

    measurement = DelayedMemoryTask(
        TaijiConfig(
            region_sizes=(64, 48),
            synapse_fan_in=16,
            motor_fan_in=48,
            memory_units=128,
            memory_fan_in=32,
            memory_meta_dim=32,
            memory_readout_fan_in=32,
            memory_iterations=3,
            seed=11,
        ),
        seeds=(11,),
    ).evaluate(corpus)

    assert measurement.ability_id == "b2_delayed_memory"
    assert measurement.status in {"passed", "failed"}
    assert measurement.metric_direction == "higher_is_better"
    assert measurement.holdout_updates == 0
    assert measurement.sample_counts == {"train": 8, "holdout": 8, "retention": 8}
    assert "memory_lesion" in measurement.baseline_metrics


def _world_case(case_id: str, *, position: float, action_kind: str) -> WorldInterventionCase:
    before = WorldState(
        tick=0,
        latent=torch.zeros(1),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("target", attributes={"position": position}),
        ),
    )
    action = WorldAction(
        action_id=case_id,
        kind=action_kind,
        tick=0,
        actor_id="agent",
        target_id="target",
        parameters={"amount": 1.0},
    )
    delta = 1.0 if action_kind == "push" else -1.0
    after = WorldState(
        tick=1,
        latent=torch.zeros(1),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("target", attributes={"position": position + delta}),
        ),
    )
    return WorldInterventionCase(
        case_id=case_id,
        initial=before,
        action=action,
        expected_state=after,
        expected_outcome=Outcome(
            intent_id=case_id,
            reward=delta,
            success=delta > 0,
            tick=1,
        ),
    )


def test_world_transition_task_uses_train_only_schema_and_reports_controls() -> None:
    train = tuple(
        _world_case(f"train-{index}", position=float(index), action_kind="push")
        for index in range(6)
    )
    holdout = tuple(
        _world_case(f"holdout-{index}", position=10.0 + index, action_kind="push")
        for index in range(3)
    )
    retention = tuple(
        _world_case(f"retention-{index}", position=20.0 + index, action_kind="push")
        for index in range(3)
    )

    measurement = WorldTransitionTask(epochs=10, seeds=(11,)).evaluate(
        WorldTransitionCorpus(train=train, holdout=holdout, retention=retention)
    )

    assert measurement.ability_id == "b3_world_transition"
    assert measurement.status in {"passed", "failed"}
    assert measurement.sample_counts == {"train": 6, "holdout": 3, "retention": 3}
    assert measurement.holdout_updates == 0
    assert set(("random", "frozen_parent", "simple_rule", "hash_only")).issubset(
        measurement.baseline_metrics
    )

"""Native task adapters used by the M0 foundation benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from .config import TaijiConfig, validate_episodic_learning_target
from .contracts import WorldInterventionCase, WorldInterventionCorpus
from .foundation_evaluation import FoundationMeasurement
from .internalization import content_digest
from .model import Taiji
from .world_learning import (
    WorldDynamicsLearner,
    WorldPrediction,
    WorldSchema,
    _replace_numeric_state,
)


@dataclass(frozen=True)
class SequencePredictionCorpus:
    """Byte streams with source-disjoint train, holdout, and retention roles."""

    train: bytes
    holdout: bytes
    retention: bytes

    def __post_init__(self) -> None:
        for partition in ("train", "holdout", "retention"):
            value = getattr(self, partition)
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"{partition} sequence corpus must contain non-empty bytes")

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "holdout": len(self.holdout),
            "retention": len(self.retention),
        }


@dataclass(frozen=True)
class MemoryEpisode:
    """A train-only cue/action/outcome binding written into Taiji memory."""

    memory_id: str
    cue: int
    action: int
    outcome: int
    context: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory episode id must be non-empty")
        if any(int(value) < 0 for value in (self.cue, self.action, self.outcome)):
            raise ValueError("memory episode symbols cannot be negative")
        if any(int(value) < 0 for value in self.context):
            raise ValueError("memory episode context symbols cannot be negative")

    @property
    def recall_key(self) -> tuple[int, ...]:
        """The full symbol sequence that identifies this binding."""

        return (*self.context, self.cue)


@dataclass(frozen=True)
class DelayedMemoryQuery:
    """A read-only cue query whose answer was written during training."""

    query_id: str
    cue: int
    expected_action: int
    context: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("memory query id must be non-empty")
        if any(int(value) < 0 for value in (self.cue, self.expected_action)):
            raise ValueError("memory query symbols cannot be negative")
        if any(int(value) < 0 for value in self.context):
            raise ValueError("memory query context symbols cannot be negative")

    @property
    def recall_key(self) -> tuple[int, ...]:
        """The full symbol sequence that identifies this binding."""

        return (*self.context, self.cue)


@dataclass(frozen=True)
class DelayedMemoryCorpus:
    train: tuple[MemoryEpisode, ...]
    holdout: tuple[DelayedMemoryQuery, ...]
    retention: tuple[DelayedMemoryQuery, ...]
    interference_symbols: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.train or not self.holdout or not self.retention:
            raise ValueError("delayed memory partitions cannot be empty")
        train_keys = {episode.recall_key for episode in self.train}
        if any(query.recall_key not in train_keys for query in (*self.holdout, *self.retention)):
            raise ValueError("delayed memory queries must target a train-written cue")
        if len({episode.memory_id for episode in self.train}) != len(self.train):
            raise ValueError("delayed memory train ids must be unique")
        query_ids = {query.query_id for query in (*self.holdout, *self.retention)}
        if len(query_ids) != len(self.holdout) + len(self.retention):
            raise ValueError("delayed memory query ids must be unique")

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "holdout": len(self.holdout),
            "retention": len(self.retention),
        }


class DelayedMemoryTask:
    """Measure cue recall after interference using Taiji's episodic field."""

    ability_id = "b2_delayed_memory"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        seeds: Sequence[int] = (11, 29, 47),
    ) -> None:
        if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError("delayed memory needs unique seeds")
        self.config = config
        self.seeds = tuple(int(seed) for seed in seeds)

    def evaluate(self, corpus: DelayedMemoryCorpus) -> FoundationMeasurement:
        actions = tuple(dict.fromkeys(episode.action for episode in corpus.train))
        if len(actions) < 2:
            raise ValueError("delayed memory needs at least two action classes")
        seed_records: list[dict[str, float | int]] = []
        for seed in self.seeds:
            model = Taiji(self._with_seed(seed), episode_id=f"m0-b2-seed-{seed}")
            for episode in corpus.train:
                self._write_episode(model, episode)
            interference = corpus.interference_symbols
            holdout = self._recall_accuracy(
                model,
                corpus.holdout,
                actions,
                use_memory=True,
                interference_symbols=interference,
            )
            persistent_before = _persistent_digest(model)
            retention = self._recall_accuracy(
                model,
                corpus.retention,
                actions,
                use_memory=True,
                interference_symbols=interference,
            )
            persistent_after = _persistent_digest(model)
            memory_lesion = self._recall_accuracy(
                model,
                corpus.holdout,
                actions,
                use_memory=False,
                interference_symbols=interference,
            )
            identity_lesion = self._recall_accuracy(
                model,
                corpus.holdout,
                actions,
                use_memory=True,
                use_identity=False,
                interference_symbols=interference,
            )
            frozen = Taiji(self._with_seed(seed), episode_id=f"m0-b2-frozen-{seed}")
            frozen_accuracy = self._recall_accuracy(
                frozen,
                corpus.holdout,
                actions,
                use_memory=True,
                interference_symbols=interference,
            )
            seed_records.append(
                {
                    "seed": seed,
                    "taiji": holdout,
                    "retention": retention,
                    "memory_lesion": memory_lesion,
                    "identity_lesion": identity_lesion,
                    "frozen_parent": frozen_accuracy,
                    "holdout_updates": int(persistent_before != persistent_after),
                    "parameter_count": model.parameter_count(),
                }
            )
        native_values = [float(record["taiji"]) for record in seed_records]
        frozen_values = [float(record["frozen_parent"]) for record in seed_records]
        lesion_values = [float(record["memory_lesion"]) for record in seed_records]
        identity_lesion_values = [float(record["identity_lesion"]) for record in seed_records]
        baseline_metrics = {
            "random": 1.0 / len(actions),
            "frozen_parent": min(frozen_values),
            "simple_rule": _majority_accuracy(corpus.train, corpus.holdout),
            "hash_only": min(
                _hash_memory_accuracy(corpus.holdout, actions, seed=seed)
                for seed in self.seeds
            ),
            "memory_lesion": min(lesion_values),
            "identity_lesion": min(identity_lesion_values),
        }
        worst_native = min(native_values)
        holdout_updates = max(int(record["holdout_updates"]) for record in seed_records)
        beats_controls = all(worst_native > value for value in baseline_metrics.values())
        causal_memory_gain = all(
            float(record["taiji"]) > float(record["memory_lesion"])
            for record in seed_records
        )
        causal_identity_gain = all(
            float(record["taiji"]) > float(record["identity_lesion"])
            for record in seed_records
        )
        return FoundationMeasurement(
            ability_id=self.ability_id,
            status=(
                "passed"
                if beats_controls and causal_memory_gain and causal_identity_gain and holdout_updates == 0
                else "failed"
            ),
            primary_metric="recall_accuracy",
            metric_direction="higher_is_better",
            metric_value=worst_native,
            baseline_metrics=baseline_metrics,
            sample_counts=corpus.sample_counts,
            holdout_updates=holdout_updates,
            evidence=(
                "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
                "memory_write_partition=train; holdout_and_retention_learning=false",
            ),
        )

    def _with_seed(self, seed: int) -> TaijiConfig:
        values = self.config.to_dict()
        values["seed"] = int(seed)
        return TaijiConfig.from_dict(values)

    @staticmethod
    def _write_episode(
        model: Taiji,
        episode: MemoryEpisode,
        *,
        reward: float = 1.0,
        memory_learning_scale: float = 1.0,
        memory_learning_targets: str = "all",
        provenance: str = "experienced",
    ) -> None:
        model.reset_dynamics(episode_id=f"m0-b2-train-{episode.memory_id}")
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
        for symbol in episode.context:
            model.observe(symbol, learn=False, learn_motor=False)
        model.observe(episode.cue, learn=False, learn_motor=False)
        model.act((episode.action,), sample=False)
        model.settle_action(
            reward,
            learn=False,
            learn_memory=True,
            provenance=provenance,
            memory_learning_scale=memory_learning_scale,
            memory_learning_targets=memory_learning_targets,
        )
        model.observe(episode.outcome, learn=False, learn_motor=False)

    @staticmethod
    def _recall_accuracy(
        model: Taiji,
        queries: Sequence[DelayedMemoryQuery],
        actions: tuple[int, ...],
        *,
        use_memory: bool,
        use_identity: bool | None = None,
        interference_symbols: tuple[int, ...] = (),
    ) -> float:
        correct = 0
        for query in queries:
            model.reset_dynamics(episode_id=f"m0-b2-query-{query.query_id}")
            model.observe(
                model.config.boundary_symbol,
                learn=False,
                learn_motor=False,
                use_memory=use_memory,
                use_identity=use_identity,
            )
            for symbol in query.context:
                model.observe(
                    symbol,
                    learn=False,
                    learn_motor=False,
                    use_memory=use_memory,
                    use_identity=use_identity,
                )
            model.observe(
                query.cue,
                learn=False,
                learn_motor=False,
                use_memory=use_memory,
                use_identity=use_identity,
            )
            for symbol in interference_symbols:
                model.observe(
                    symbol,
                    learn=False,
                    learn_motor=False,
                    use_memory=use_memory,
                    use_identity=use_identity,
                )
            probabilities = model.snapshot().motor_probabilities
            prediction = max(actions, key=lambda action: float(probabilities[action].item()))
            correct += int(prediction == query.expected_action)
        return correct / len(queries)


CONTINUAL_MEMORY_FORMAT = "taiji-native-continual-memory-v1"
CONTINUAL_MEMORY_VERSION = 1


@dataclass(frozen=True)
class ContinualMemoryCorpus:
    """A dedicated phase-A/phase-B memory stream with explicit replay data."""

    phase_a_train: tuple[MemoryEpisode, ...]
    phase_a_holdout: tuple[DelayedMemoryQuery, ...]
    phase_a_retention: tuple[DelayedMemoryQuery, ...]
    phase_b_train: tuple[MemoryEpisode, ...]
    phase_b_holdout: tuple[DelayedMemoryQuery, ...]
    phase_b_retention: tuple[DelayedMemoryQuery, ...]
    replay_train: tuple[MemoryEpisode, ...]

    def __post_init__(self) -> None:
        partitions = {
            "phase_a_train": self.phase_a_train,
            "phase_a_holdout": self.phase_a_holdout,
            "phase_a_retention": self.phase_a_retention,
            "phase_b_train": self.phase_b_train,
            "phase_b_holdout": self.phase_b_holdout,
            "phase_b_retention": self.phase_b_retention,
            "replay_train": self.replay_train,
        }
        if any(not values for values in partitions.values()):
            raise ValueError("continual memory partitions cannot be empty")
        for name, values in partitions.items():
            if name.endswith("train") and any(
                not isinstance(item, MemoryEpisode) for item in values
            ):
                raise TypeError(f"{name} must contain MemoryEpisode values")
            if not name.endswith("train") and any(
                not isinstance(item, DelayedMemoryQuery) for item in values
            ):
                raise TypeError(f"{name} must contain DelayedMemoryQuery values")
        for phase in ("a", "b"):
            train = getattr(self, f"phase_{phase}_train")
            train_cues = {episode.cue for episode in train}
            for partition in ("holdout", "retention"):
                queries = getattr(self, f"phase_{phase}_{partition}")
                if any(query.cue not in train_cues for query in queries):
                    raise ValueError(
                        f"phase-{phase} continual queries must target phase-{phase} cues"
                    )
        query_ids = {
            query.query_id
            for values in (
                self.phase_a_holdout,
                self.phase_a_retention,
                self.phase_b_holdout,
                self.phase_b_retention,
            )
            for query in values
        }
        query_count = sum(
            len(values)
            for values in (
                self.phase_a_holdout,
                self.phase_a_retention,
                self.phase_b_holdout,
                self.phase_b_retention,
            )
        )
        if len(query_ids) != query_count:
            raise ValueError("continual memory query ids must be unique")
        train_ids = {episode.memory_id for episode in self.phase_a_train}
        train_ids.update(episode.memory_id for episode in self.phase_b_train)
        if len(train_ids) != len(self.phase_a_train) + len(self.phase_b_train):
            raise ValueError("phase-A and phase-B memory ids must be disjoint")

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.phase_a_train) + len(self.phase_b_train),
            "holdout": len(self.phase_a_holdout) + len(self.phase_b_holdout),
            "retention": len(self.phase_a_retention) + len(self.phase_b_retention),
            "phase_a_train": len(self.phase_a_train),
            "phase_a_holdout": len(self.phase_a_holdout),
            "phase_a_retention": len(self.phase_a_retention),
            "phase_b_train": len(self.phase_b_train),
            "phase_b_holdout": len(self.phase_b_holdout),
            "phase_b_retention": len(self.phase_b_retention),
            "replay_train": len(self.replay_train),
        }

    @property
    def digest(self) -> str:
        def episode_payload(item: MemoryEpisode) -> list[object]:
            return [item.memory_id, item.cue, item.action, item.outcome]

        def query_payload(item: DelayedMemoryQuery) -> list[object]:
            return [item.query_id, item.cue, item.expected_action]

        return content_digest(
            {
                "format": CONTINUAL_MEMORY_FORMAT,
                "version": CONTINUAL_MEMORY_VERSION,
                "phase_a_train": [episode_payload(item) for item in self.phase_a_train],
                "phase_a_holdout": [query_payload(item) for item in self.phase_a_holdout],
                "phase_a_retention": [query_payload(item) for item in self.phase_a_retention],
                "phase_b_train": [episode_payload(item) for item in self.phase_b_train],
                "phase_b_holdout": [query_payload(item) for item in self.phase_b_holdout],
                "phase_b_retention": [query_payload(item) for item in self.phase_b_retention],
                "replay_train": [episode_payload(item) for item in self.replay_train],
            }
        )


class ContinualMemoryTask:
    """Measure replay's causal gain over a no-replay phase-B counterfactual."""

    ability_id = "b5_continual_learning"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        seeds: Sequence[int] = (11, 29, 47),
        replay_learning_targets: str = "all",
    ) -> None:
        if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError("continual memory needs unique seeds")
        self.config = config
        self.seeds = tuple(int(seed) for seed in seeds)
        validate_episodic_learning_target(
            "continual replay learning targets", replay_learning_targets
        )
        self.replay_learning_targets = replay_learning_targets

    def evaluate(self, corpus: ContinualMemoryCorpus) -> FoundationMeasurement:
        actions = tuple(
            dict.fromkeys(
                episode.action
                for episode in (*corpus.phase_a_train, *corpus.phase_b_train)
            )
        )
        if len(actions) < 2:
            raise ValueError("continual memory needs at least two action classes")
        seed_records: list[dict[str, float | int | str]] = []
        for seed in self.seeds:
            phase_a = Taiji(self._with_seed(seed), episode_id=f"m1-b5-phase-a-{seed}")
            for episode in corpus.phase_a_train:
                DelayedMemoryTask._write_episode(phase_a, episode)
            old_before = DelayedMemoryTask._recall_accuracy(
                phase_a, corpus.phase_a_holdout, actions, use_memory=True
            )
            old_retention_before = DelayedMemoryTask._recall_accuracy(
                phase_a, corpus.phase_a_retention, actions, use_memory=True
            )
            phase_a_payload = deepcopy(phase_a.checkpoint())
            phase_a_digest = content_digest(phase_a_payload)

            no_replay = Taiji(self._with_seed(seed), episode_id=f"m1-b5-no-replay-{seed}")
            no_replay.restore(deepcopy(phase_a_payload))
            for episode in corpus.phase_b_train:
                DelayedMemoryTask._write_episode(no_replay, episode)
            no_replay_old_after = DelayedMemoryTask._recall_accuracy(
                no_replay, corpus.phase_a_holdout, actions, use_memory=True
            )
            no_replay_new_after = DelayedMemoryTask._recall_accuracy(
                no_replay, corpus.phase_b_holdout, actions, use_memory=True
            )

            replay = Taiji(self._with_seed(seed), episode_id=f"m1-b5-replay-{seed}")
            replay.restore(deepcopy(phase_a_payload))
            for episode in corpus.phase_b_train:
                DelayedMemoryTask._write_episode(replay, episode)
            for episode in corpus.replay_train:
                DelayedMemoryTask._write_episode(
                    replay,
                    episode,
                    provenance="replayed",
                    memory_learning_scale=self.config.replay_memory_learning_scale,
                    memory_learning_targets=self.replay_learning_targets,
                )
            replay_old_after = DelayedMemoryTask._recall_accuracy(
                replay, corpus.phase_a_holdout, actions, use_memory=True
            )
            replay_retention_after = DelayedMemoryTask._recall_accuracy(
                replay, corpus.phase_a_retention, actions, use_memory=True
            )
            replay_new_after = DelayedMemoryTask._recall_accuracy(
                replay, corpus.phase_b_holdout, actions, use_memory=True
            )
            replay_digest = content_digest(replay.checkpoint())
            seed_records.append(
                {
                    "seed": seed,
                    "old_before": old_before,
                    "old_retention_before": old_retention_before,
                    "no_replay_old_after": no_replay_old_after,
                    "replay_old_after": replay_old_after,
                    "replay_retention_after": replay_retention_after,
                    "no_replay_new_after": no_replay_new_after,
                    "replay_new_after": replay_new_after,
                    "no_replay_backward_transfer": no_replay_old_after - old_before,
                    "replay_backward_transfer": replay_old_after - old_before,
                    "replay_causal_gain": replay_old_after - no_replay_old_after,
                    "phase_a_checkpoint_digest": phase_a_digest,
                    "replay_checkpoint_digest": replay_digest,
                    "continued_from_phase_a": int(phase_a_digest != replay_digest),
                    "holdout_updates": 0,
                }
            )

        replay_values = [float(item["replay_backward_transfer"]) for item in seed_records]
        no_replay_values = [
            float(item["no_replay_backward_transfer"]) for item in seed_records
        ]
        baseline_metrics = {
            "random": 0.0,
            "frozen_parent": 0.0,
            "simple_rule": 0.0,
            "hash_only": 0.0,
            "no_replay": min(no_replay_values),
        }
        causal_gain = all(float(item["replay_causal_gain"]) > 0.0 for item in seed_records)
        old_retained = all(
            float(item["replay_old_after"]) >= float(item["old_before"])
            and float(item["replay_retention_after"]) >= float(item["old_retention_before"])
            for item in seed_records
        )
        new_preserved = all(
            float(item["replay_new_after"]) + 0.05
            >= float(item["no_replay_new_after"])
            for item in seed_records
        )
        return FoundationMeasurement(
            ability_id=self.ability_id,
            status=(
                "passed"
                if (
                    min(replay_values) >= 0.0
                    and causal_gain
                    and old_retained
                    and new_preserved
                    and all(bool(item["continued_from_phase_a"]) for item in seed_records)
                )
                else "failed"
            ),
            primary_metric="backward_transfer",
            metric_direction="higher_is_better",
            metric_value=min(replay_values),
            baseline_metrics=baseline_metrics,
            sample_counts=corpus.sample_counts,
            holdout_updates=0,
            evidence=(
                "corpus_digest=" + corpus.digest,
                "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
                "phase_a_then_phase_b_with_no_replay_counterfactual_then_phase_a_replay=true",
            ),
        )

    def _with_seed(self, seed: int) -> TaijiConfig:
        values = self.config.to_dict()
        values["seed"] = int(seed)
        return TaijiConfig.from_dict(values)


@dataclass(frozen=True)
class WorldTransitionCorpus:
    train: tuple[WorldInterventionCase, ...]
    holdout: tuple[WorldInterventionCase, ...]
    retention: tuple[WorldInterventionCase, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.holdout or not self.retention:
            raise ValueError("world transition partitions cannot be empty")
        seen: set[str] = set()
        for partition, cases in (
            ("train", self.train),
            ("holdout", self.holdout),
            ("retention", self.retention),
        ):
            if any(not isinstance(case, WorldInterventionCase) for case in cases):
                raise TypeError(f"{partition} must contain WorldInterventionCase values")
            ids = {case.case_id for case in cases}
            if len(ids) != len(cases) or seen.intersection(ids):
                raise ValueError("world transition case ids must be unique across partitions")
            seen.update(ids)

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "holdout": len(self.holdout),
            "retention": len(self.retention),
        }


class WorldTransitionTask:
    """Measure train-only native world transition learning and controls."""

    ability_id = "b3_world_transition"

    def __init__(
        self,
        *,
        seeds: Sequence[int] = (11, 29, 47),
        hidden_dim: int = 32,
        epochs: int = 50,
        learning_rate: float = 0.01,
    ) -> None:
        if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError("world transition needs unique seeds")
        if int(hidden_dim) <= 0 or int(epochs) <= 0 or float(learning_rate) <= 0.0:
            raise ValueError("world transition model settings must be positive")
        self.seeds = tuple(int(seed) for seed in seeds)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)

    def evaluate(self, corpus: WorldTransitionCorpus) -> FoundationMeasurement:
        schema = WorldSchema.from_corpus(
            WorldInterventionCorpus(train=corpus.train, holdout=corpus.holdout)
        )
        seed_records: list[dict[str, float | int]] = []
        for seed in self.seeds:
            frozen = WorldDynamicsLearner(schema, hidden_dim=self.hidden_dim, seed=seed)
            frozen_error = _world_error(frozen, corpus.holdout, schema)
            learner = WorldDynamicsLearner(schema, hidden_dim=self.hidden_dim, seed=seed)
            losses = learner.fit(
                corpus.train,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
            )
            persistent_before = _world_learner_digest(learner)
            native_error = _world_error(learner, corpus.holdout, schema)
            persistent_after = _world_learner_digest(learner)
            retention_error = _world_error(learner, corpus.retention, schema)
            frozen_retention_error = _world_error(frozen, corpus.retention, schema)
            seed_records.append(
                {
                    "seed": seed,
                    "taiji": native_error,
                    "frozen_parent": frozen_error,
                    "retention": retention_error,
                    "frozen_retention": frozen_retention_error,
                    "simple_rule": _no_change_error(corpus.holdout, schema),
                    "hash_only": _hash_world_error(corpus.holdout, schema, seed=seed),
                    "holdout_updates": int(persistent_before != persistent_after),
                    "training_loss": float(losses[-1]),
                    "parameter_count": sum(parameter.numel() for parameter in learner.parameters()),
                }
            )
        native_values = [float(record["taiji"]) for record in seed_records]
        retention_values = [float(record["retention"]) for record in seed_records]
        frozen_retention_values = [float(record["frozen_retention"]) for record in seed_records]
        baseline_metrics = {
            "random": min(
                _random_world_error(corpus.holdout, schema, seed=seed) for seed in self.seeds
            ),
            "frozen_parent": min(float(record["frozen_parent"]) for record in seed_records),
            "simple_rule": min(float(record["simple_rule"]) for record in seed_records),
            "hash_only": min(float(record["hash_only"]) for record in seed_records),
        }
        worst_native = max(native_values)
        holdout_updates = max(int(record["holdout_updates"]) for record in seed_records)
        retention_preserved = all(
            actual <= frozen + 0.05
            for actual, frozen in zip(retention_values, frozen_retention_values, strict=True)
        )
        beats_controls = all(worst_native < value for value in baseline_metrics.values())
        return FoundationMeasurement(
            ability_id=self.ability_id,
            status=(
                "passed"
                if beats_controls and retention_preserved and holdout_updates == 0
                else "failed"
            ),
            primary_metric="transition_error",
            metric_direction="lower_is_better",
            metric_value=worst_native,
            baseline_metrics=baseline_metrics,
            sample_counts=corpus.sample_counts,
            holdout_updates=holdout_updates,
            evidence=(
                "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
                "schema_source=train_only; holdout_and_retention_register_parameters=false",
            ),
        )


@dataclass(frozen=True)
class GoalActionEpisode:
    """A cue-conditioned action whose value is learned from outcome credit."""

    episode_id: str
    cue: int
    preferred_action: int
    alternate_action: int

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("goal action episode id must be non-empty")
        if any(int(value) < 0 for value in (self.cue, self.preferred_action, self.alternate_action)):
            raise ValueError("goal action episode symbols cannot be negative")
        if int(self.preferred_action) == int(self.alternate_action):
            raise ValueError("goal action episode needs two distinct actions")


@dataclass(frozen=True)
class GoalActionCorpus:
    train: tuple[GoalActionEpisode, ...]
    holdout: tuple[GoalActionEpisode, ...]
    retention: tuple[GoalActionEpisode, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.holdout or not self.retention:
            raise ValueError("goal action partitions cannot be empty")
        seen: set[str] = set()
        for partition, episodes in (
            ("train", self.train),
            ("holdout", self.holdout),
            ("retention", self.retention),
        ):
            if any(not isinstance(episode, GoalActionEpisode) for episode in episodes):
                raise TypeError(f"{partition} must contain GoalActionEpisode values")
            ids = {episode.episode_id for episode in episodes}
            if len(ids) != len(episodes) or seen.intersection(ids):
                raise ValueError("goal action episode ids must be unique across partitions")
            seen.update(ids)

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "holdout": len(self.holdout),
            "retention": len(self.retention),
        }


class GoalActionTask:
    """Measure native action choice after cue-specific reward credit assignment."""

    ability_id = "b4_goal_action"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        seeds: Sequence[int] = (11, 29, 47),
    ) -> None:
        if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError("goal action needs unique seeds")
        self.config = config
        self.seeds = tuple(int(seed) for seed in seeds)

    def evaluate(self, corpus: GoalActionCorpus) -> FoundationMeasurement:
        seed_records: list[dict[str, float | int]] = []
        for seed in self.seeds:
            model = Taiji(self._with_seed(seed), episode_id=f"m0-b4-seed-{seed}")
            self._cold_start_action_organ(model)
            for episode in corpus.train:
                self._train_episode(model, episode)
            holdout = self._evaluate_partition(model, corpus.holdout)
            persistent_before = _persistent_digest(model)
            retention = self._evaluate_partition(model, corpus.retention)
            persistent_after = _persistent_digest(model)
            credit_lesion = Taiji(self._with_seed(seed), episode_id=f"m0-b4-lesion-{seed}")
            self._cold_start_action_organ(credit_lesion)
            for episode in corpus.train:
                self._train_episode(credit_lesion, episode, learn=False)
            lesion_accuracy = self._evaluate_partition(credit_lesion, corpus.holdout)
            frozen = Taiji(self._with_seed(seed), episode_id=f"m0-b4-frozen-{seed}")
            self._cold_start_action_organ(frozen)
            frozen_accuracy = self._evaluate_partition(frozen, corpus.holdout)
            seed_records.append(
                {
                    "seed": seed,
                    "taiji": holdout,
                    "retention": retention,
                    "credit_lesion": lesion_accuracy,
                    "frozen_parent": frozen_accuracy,
                    "holdout_updates": int(persistent_before != persistent_after),
                    "parameter_count": model.parameter_count(),
                }
            )

        native_values = [float(record["taiji"]) for record in seed_records]
        retention_values = [float(record["retention"]) for record in seed_records]
        frozen_values = [float(record["frozen_parent"]) for record in seed_records]
        lesion_values = [float(record["credit_lesion"]) for record in seed_records]
        baseline_metrics = {
            "random": 0.5,
            "frozen_parent": min(frozen_values),
            "simple_rule": _majority_goal_action_accuracy(corpus.train, corpus.holdout),
            "hash_only": min(
                _hash_goal_action_accuracy(corpus.holdout, seed=seed)
                for seed in self.seeds
            ),
            "credit_lesion": min(lesion_values),
        }
        worst_native = min(native_values)
        holdout_updates = max(int(record["holdout_updates"]) for record in seed_records)
        beats_controls = all(worst_native > value for value in baseline_metrics.values())
        causal_credit_gain = all(
            float(record["taiji"]) > float(record["credit_lesion"])
            for record in seed_records
        )
        retention_preserved = all(
            retention >= native - 0.05
            for retention, native in zip(retention_values, native_values, strict=True)
        )
        return FoundationMeasurement(
            ability_id=self.ability_id,
            status=(
                "passed"
                if (
                    beats_controls
                    and causal_credit_gain
                    and retention_preserved
                    and holdout_updates == 0
                )
                else "failed"
            ),
            primary_metric="success_rate",
            metric_direction="higher_is_better",
            metric_value=worst_native,
            baseline_metrics=baseline_metrics,
            sample_counts=corpus.sample_counts,
            holdout_updates=holdout_updates,
            evidence=(
                "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
                "train_only_reward_credit=true; holdout_and_retention_learning=false",
            ),
        )

    def _with_seed(self, seed: int) -> TaijiConfig:
        values = self.config.to_dict()
        values["seed"] = int(seed)
        return TaijiConfig.from_dict(values)

    @staticmethod
    def _cold_start_action_organ(model: Taiji) -> None:
        """Start the goal readout uncommitted so credit, not initialization, wins."""

        with torch.no_grad():
            model.motor.synapses.edge_weight.zero_()
            model.motor.bias.zero_()
            model.motor.reward_baseline = 0.0
            model.motor.reward_updates = 0

    @staticmethod
    def _train_episode(
        model: Taiji,
        episode: GoalActionEpisode,
        *,
        learn: bool = True,
    ) -> bool:
        model.reset_dynamics(episode_id=f"m0-b4-train-{episode.episode_id}")
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
        model.observe(episode.cue, learn=learn, learn_motor=False, use_memory=False)
        decision = model.act(
            tuple(sorted((episode.preferred_action, episode.alternate_action))),
            sample=True,
        )
        success = decision.action_symbol == episode.preferred_action
        model.settle_action(1.0 if success else -1.0, learn=learn, learn_memory=False)
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
        return success

    @staticmethod
    def _evaluate_partition(
        model: Taiji,
        episodes: Sequence[GoalActionEpisode],
    ) -> float:
        correct = 0
        for episode in episodes:
            model.reset_dynamics(episode_id=f"m0-b4-eval-{episode.episode_id}")
            model.observe(
                model.config.boundary_symbol,
                learn=False,
                learn_motor=False,
                use_memory=False,
            )
            model.observe(episode.cue, learn=False, learn_motor=False, use_memory=False)
            decision = model.act(
                tuple(sorted((episode.preferred_action, episode.alternate_action))),
                sample=False,
            )
            correct += int(decision.action_symbol == episode.preferred_action)
            model.settle_action(0.0, learn=False, learn_memory=False)
            model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
        return correct / len(episodes)


@dataclass(frozen=True)
class ContinualLearningCorpus:
    """Two sequential learning phases with an old-ability retention set."""

    phase_a_train: bytes
    phase_a_holdout: bytes
    phase_b_train: bytes
    phase_b_holdout: bytes
    retention: bytes

    def __post_init__(self) -> None:
        for field_name in (
            "phase_a_train",
            "phase_a_holdout",
            "phase_b_train",
            "phase_b_holdout",
            "retention",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bytes) or not value:
                raise ValueError(f"{field_name} continual corpus must contain non-empty bytes")

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.phase_a_train) + len(self.phase_b_train),
            "holdout": len(self.phase_a_holdout) + len(self.phase_b_holdout),
            "retention": len(self.retention),
        }


class ContinualLearningTask:
    """Measure checkpoint continuation, bounded replay, and old-skill retention."""

    ability_id = "b5_continual_learning"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        seeds: Sequence[int] = (11, 29, 47),
        epochs: int = 1,
        replay_epochs: int = 1,
    ) -> None:
        if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError("continual learning needs unique seeds")
        if int(epochs) <= 0 or int(replay_epochs) <= 0:
            raise ValueError("continual learning epochs must be positive")
        self.config = config
        self.seeds = tuple(int(seed) for seed in seeds)
        self.epochs = int(epochs)
        self.replay_epochs = int(replay_epochs)

    def evaluate(self, corpus: ContinualLearningCorpus) -> FoundationMeasurement:
        seed_records: list[dict[str, float | int]] = []
        for seed in self.seeds:
            model = Taiji(self._with_seed(seed), episode_id=f"m0-b5-seed-{seed}")
            model.learn_bytes(corpus.phase_a_train, epochs=self.epochs)
            old_before = self._score_bpb(model, corpus.phase_a_holdout)
            parent_digest = content_digest(model.checkpoint())
            model.learn_bytes(corpus.phase_b_train, epochs=self.epochs)
            for _ in range(self.replay_epochs):
                model.learn_bytes(corpus.phase_a_train, epochs=1)
            continued_digest = content_digest(model.checkpoint())
            old_after = self._score_bpb(model, corpus.phase_a_holdout)
            new_after = self._score_bpb(model, corpus.phase_b_holdout)
            retention = self._score_bpb(model, corpus.retention)
            fresh = Taiji(self._with_seed(seed), episode_id=f"m0-b5-fresh-{seed}")
            fresh.learn_bytes(corpus.phase_b_train, epochs=self.epochs)
            fresh_new = self._score_bpb(fresh, corpus.phase_b_holdout)

            lesion = Taiji(self._with_seed(seed), episode_id=f"m0-b5-lesion-{seed}")
            lesion.learn_bytes(corpus.phase_a_train, epochs=self.epochs)
            lesion_old_before = self._score_bpb(lesion, corpus.phase_a_holdout)
            lesion.learn_bytes(corpus.phase_b_train, epochs=self.epochs)
            lesion_old_after = self._score_bpb(lesion, corpus.phase_a_holdout)
            backward_transfer = old_before - old_after
            lesion_transfer = lesion_old_before - lesion_old_after
            seed_records.append(
                {
                    "seed": seed,
                    "old_before": old_before,
                    "old_after": old_after,
                    "new_after": new_after,
                    "fresh_new": fresh_new,
                    "retention": retention,
                    "backward_transfer": backward_transfer,
                    "replay_lesion": lesion_transfer,
                    "parent_checkpoint_digest": parent_digest,
                    "continued_checkpoint_digest": continued_digest,
                    "continued_from_parent": int(parent_digest != continued_digest),
                    "holdout_updates": 0,
                    "parameter_count": model.parameter_count(),
                }
            )

        native_values = [float(record["backward_transfer"]) for record in seed_records]
        lesion_values = [float(record["replay_lesion"]) for record in seed_records]
        new_values = [float(record["new_after"]) for record in seed_records]
        fresh_values = [float(record["fresh_new"]) for record in seed_records]
        retention_values = [float(record["retention"]) for record in seed_records]
        baseline_metrics = {
            "random": 0.0,
            "frozen_parent": 0.0,
            "simple_rule": 0.0,
            "hash_only": 0.0,
            "replay_lesion": min(lesion_values),
        }
        worst_native = min(native_values)
        holdout_updates = max(int(record["holdout_updates"]) for record in seed_records)
        continued = all(bool(record["continued_from_parent"]) for record in seed_records)
        causal_replay_gain = all(
            float(record["backward_transfer"]) > float(record["replay_lesion"])
            for record in seed_records
        )
        new_capability_preserved = all(
            native <= fresh + 0.5 for native, fresh in zip(new_values, fresh_values, strict=True)
        )
        retention_preserved = all(
            value <= float(record["old_before"]) + 0.5
            for value, record in zip(retention_values, seed_records, strict=True)
        )
        return FoundationMeasurement(
            ability_id=self.ability_id,
            status=(
                "passed"
                if (
                    worst_native > max(baseline_metrics.values())
                    and causal_replay_gain
                    and continued
                    and new_capability_preserved
                    and retention_preserved
                    and holdout_updates == 0
                )
                else "failed"
            ),
            primary_metric="backward_transfer",
            metric_direction="higher_is_better",
            metric_value=worst_native,
            baseline_metrics=baseline_metrics,
            sample_counts=corpus.sample_counts,
            holdout_updates=holdout_updates,
            evidence=(
                "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
                "phase_b_continues_phase_a_checkpoint=true; phase_a_replay_is_train_only=true",
            ),
        )

    def _with_seed(self, seed: int) -> TaijiConfig:
        values = self.config.to_dict()
        values["seed"] = int(seed)
        return TaijiConfig.from_dict(values)

    @staticmethod
    def _score_bpb(model: Taiji, data: bytes) -> float:
        before = content_digest(model.checkpoint())
        score = model.score_bytes(data)
        after = content_digest(model.checkpoint())
        if before != after:
            raise RuntimeError("Taiji score_bytes mutated the checkpoint")
        return float(score["mean_surprise"]) / math.log(2.0)


def _world_error(
    learner: WorldDynamicsLearner,
    cases: Sequence[WorldInterventionCase],
    schema: WorldSchema,
) -> float:
    errors: list[float] = []
    for case in cases:
        prediction = learner.predict(
            case.initial,
            case.action,
            register_parameters=False,
        )
        errors.append(_prediction_error(prediction, case, schema))
    return sum(errors) / len(errors)


def _world_learner_digest(learner: WorldDynamicsLearner) -> str:
    return content_digest(
        {
            name: tensor.detach().cpu().clone()
            for name, tensor in learner.state_dict().items()
        }
    )


def _prediction_error(
    prediction: WorldPrediction,
    case: WorldInterventionCase,
    schema: WorldSchema,
) -> float:
    state_error = schema.normalized_state_error(prediction.state, case.expected_state)
    expected_success = float(
        case.expected_outcome.success
        if case.expected_outcome.success is not None
        else case.expected_outcome.reward > 0.0
    )
    return (
        state_error
        + (prediction.reward - float(case.expected_outcome.reward)) ** 2
        + (prediction.success_probability - expected_success) ** 2
    )


def _no_change_error(
    cases: Sequence[WorldInterventionCase], schema: WorldSchema
) -> float:
    predictions = tuple(
        WorldPrediction(state=case.initial, reward=0.0, success_probability=0.5)
        for case in cases
    )
    return sum(
        _prediction_error(prediction, case, schema)
        for prediction, case in zip(predictions, cases, strict=True)
    ) / len(cases)


def _random_world_error(
    cases: Sequence[WorldInterventionCase], schema: WorldSchema, *, seed: int
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    scales = torch.tensor(schema.state_scales, dtype=torch.float32)
    errors: list[float] = []
    for case in cases:
        values = schema.state_values(case.initial) + (
            torch.rand(schema.state_dim, generator=generator) - 0.5
        ) * 2.0 * scales
        prediction = WorldPrediction(
            state=_replace_numeric_state(case.initial, schema, values),
            reward=float(torch.rand((), generator=generator).item() * 2.0 - 1.0),
            success_probability=float(torch.rand((), generator=generator).item()),
        )
        errors.append(_prediction_error(prediction, case, schema))
    return sum(errors) / len(errors)


def _hash_world_error(
    cases: Sequence[WorldInterventionCase], schema: WorldSchema, *, seed: int
) -> float:
    scales = torch.tensor(schema.state_scales, dtype=torch.float32)
    errors: list[float] = []
    for case in cases:
        digest = hashlib.sha256(
            f"{int(seed)}\0{case.action.action_id}\0{case.action.kind}".encode()
        ).digest()
        signs = torch.tensor(
            [1.0 if digest[index % len(digest)] & 1 else -1.0 for index in range(schema.state_dim)],
            dtype=torch.float32,
        )
        values = schema.state_values(case.initial) + 0.25 * signs * scales
        prediction = WorldPrediction(
            state=_replace_numeric_state(case.initial, schema, values),
            reward=0.0,
            success_probability=0.5,
        )
        errors.append(_prediction_error(prediction, case, schema))
    return sum(errors) / len(errors)


class SequencePredictionTask:
    """Measure native Taiji byte prediction against four fixed controls.

    The task intentionally measures the active Taiji model rather than a
    Transformer or an external language provider.  Each seed starts from a
    fresh parent, trains only on ``train``, and evaluates holdout/retention in
    read-only mode.
    """

    ability_id = "b1_sequence_prediction"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        seeds: Sequence[int] = (11, 29, 47),
        epochs: int = 1,
    ) -> None:
        if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError("sequence prediction needs unique seeds")
        if int(epochs) <= 0:
            raise ValueError("sequence prediction epochs must be positive")
        self.config = config
        self.seeds = tuple(int(seed) for seed in seeds)
        self.epochs = int(epochs)

    def evaluate(self, corpus: SequencePredictionCorpus) -> FoundationMeasurement:
        seed_records: list[dict[str, float | int]] = []
        for seed in self.seeds:
            config = self._with_seed(seed)
            model = Taiji(config, episode_id=f"m0-b1-seed-{seed}")
            frozen_bpb = self._score_model(model, corpus.holdout)
            training = model.learn_bytes(corpus.train, epochs=self.epochs)
            native_bpb = self._score_model(model, corpus.holdout)
            retention_bpb = self._score_model(model, corpus.retention)
            seed_records.append(
                {
                    "seed": seed,
                    "frozen_parent": frozen_bpb,
                    "taiji": native_bpb,
                    "retention": retention_bpb,
                    "holdout_updates": 0,
                    "parameter_count": model.parameter_count(),
                    "train_observations": int(training["observations"]),
                }
            )

        native_values = [float(record["taiji"]) for record in seed_records]
        frozen_values = [float(record["frozen_parent"]) for record in seed_records]
        baseline_metrics = {
            "random": math.log2(float(self.config.alphabet_size)),
            "frozen_parent": min(frozen_values),
            "simple_rule": _unigram_bpb(corpus.train, corpus.holdout, self.config),
            "hash_only": min(
                _hash_only_bpb(corpus.holdout, seed=seed, alphabet_size=self.config.alphabet_size)
                for seed in self.seeds
            ),
        }
        worst_native = max(native_values)
        holdout_updates = max(int(record["holdout_updates"]) for record in seed_records)
        beats_controls = all(worst_native < value for value in baseline_metrics.values())
        return FoundationMeasurement(
            ability_id=self.ability_id,
            status="passed" if beats_controls and holdout_updates == 0 else "failed",
            primary_metric="bits_per_byte",
            metric_direction="lower_is_better",
            metric_value=worst_native,
            baseline_metrics=baseline_metrics,
            sample_counts=corpus.sample_counts,
            holdout_updates=holdout_updates,
            evidence=(
                "seed_metrics=" + json.dumps(seed_records, sort_keys=True),
                "native_model_checkpoint_is_read_only_during_score=true",
            ),
        )

    def _with_seed(self, seed: int) -> TaijiConfig:
        values = self.config.to_dict()
        values["seed"] = int(seed)
        return TaijiConfig.from_dict(values)

    @staticmethod
    def _score_model(model: Taiji, data: bytes) -> float:
        before = content_digest(model.checkpoint())
        score = model.score_bytes(data)
        after = content_digest(model.checkpoint())
        if before != after:
            raise RuntimeError("Taiji score_bytes mutated the checkpoint")
        return float(score["mean_surprise"]) / math.log(2.0)


def _symbols(data: bytes, config: TaijiConfig) -> tuple[int, ...]:
    return config.boundary_symbol, *tuple(int(value) for value in data), config.boundary_symbol


def _unigram_bpb(train: bytes, holdout: bytes, config: TaijiConfig) -> float:
    counts = [1.0] * int(config.alphabet_size)
    for symbol in _symbols(train, config):
        counts[symbol] += 1.0
    total = sum(counts)
    targets = _symbols(holdout, config)[1:]
    return sum(-math.log2(counts[symbol] / total) for symbol in targets) / len(targets)


def _hash_only_bpb(data: bytes, *, seed: int, alphabet_size: int) -> float:
    symbols = tuple(int(value) for value in data) + (alphabet_size - 1,)
    epsilon = 1.0 / float(alphabet_size * alphabet_size)
    losses: list[float] = []
    for index, target in enumerate(symbols[1:], start=1):
        previous = symbols[index - 1]
        digest = hashlib.sha256(f"{int(seed)}\0{index}\0{previous}".encode()).digest()
        prediction = int.from_bytes(digest[:8], "big") % int(alphabet_size)
        probability = 1.0 - (alphabet_size - 1) * epsilon if prediction == target else epsilon
        losses.append(-math.log2(probability))
    return sum(losses) / len(losses)


def _persistent_digest(model: Taiji) -> str:
    checkpoint = model.checkpoint()
    persistent: dict[str, Any] = {
        "fabric": checkpoint["fabric"],
        "motor": checkpoint["motor"],
        "memory": checkpoint["memory"],
    }
    organ = checkpoint.get("identity_organ")
    if organ is not None:
        # The identity organ is a persistent structure now that it is on the
        # default path, so a read-only audit that ignores it is vacuous.  Its
        # ``lineage.parent_checkpoint_digest`` is derived from the transient
        # core (``state`` and ``rng_state`` advance on every observation) and
        # must be dropped: the audit asks whether learning mutated a durable
        # structure, not whether dynamics settled.
        persistent["identity_organ"] = {
            key: value for key, value in organ.items() if key != "lineage"
        }
    return content_digest(persistent)


def _majority_accuracy(
    train: Sequence[MemoryEpisode], holdout: Sequence[DelayedMemoryQuery]
) -> float:
    counts: dict[int, int] = {}
    for episode in train:
        counts[episode.action] = counts.get(episode.action, 0) + 1
    majority = min(counts, key=lambda action: (-counts[action], action))
    return sum(int(query.expected_action == majority) for query in holdout) / len(holdout)


def _hash_memory_accuracy(
    queries: Sequence[DelayedMemoryQuery], actions: tuple[int, ...], *, seed: int
) -> float:
    correct = 0
    for query in queries:
        digest = hashlib.sha256(f"{int(seed)}\0{query.recall_key}".encode()).digest()
        prediction = actions[int.from_bytes(digest[:8], "big") % len(actions)]
        correct += int(prediction == query.expected_action)
    return correct / len(queries)


def _majority_goal_action_accuracy(
    train: Sequence[GoalActionEpisode], holdout: Sequence[GoalActionEpisode]
) -> float:
    counts: dict[int, int] = {}
    for episode in train:
        counts[episode.preferred_action] = counts.get(episode.preferred_action, 0) + 1
    majority = min(counts, key=lambda action: (-counts[action], action))
    return sum(int(episode.preferred_action == majority) for episode in holdout) / len(holdout)


def _hash_goal_action_accuracy(
    episodes: Sequence[GoalActionEpisode], *, seed: int
) -> float:
    correct = 0
    for episode in episodes:
        digest = hashlib.sha256(
            f"{int(seed)}\0{int(episode.cue)}".encode()
        ).digest()
        actions = (episode.preferred_action, episode.alternate_action)
        prediction = actions[int.from_bytes(digest[:8], "big") % len(actions)]
        correct += int(prediction == episode.preferred_action)
    return correct / len(episodes)

"""Native task adapters used by the M0 foundation benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .config import TaijiConfig
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

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory episode id must be non-empty")
        if any(int(value) < 0 for value in (self.cue, self.action, self.outcome)):
            raise ValueError("memory episode symbols cannot be negative")


@dataclass(frozen=True)
class DelayedMemoryQuery:
    """A read-only cue query whose answer was written during training."""

    query_id: str
    cue: int
    expected_action: int

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("memory query id must be non-empty")
        if any(int(value) < 0 for value in (self.cue, self.expected_action)):
            raise ValueError("memory query symbols cannot be negative")


@dataclass(frozen=True)
class DelayedMemoryCorpus:
    train: tuple[MemoryEpisode, ...]
    holdout: tuple[DelayedMemoryQuery, ...]
    retention: tuple[DelayedMemoryQuery, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.holdout or not self.retention:
            raise ValueError("delayed memory partitions cannot be empty")
        train_cues = {episode.cue for episode in self.train}
        if any(query.cue not in train_cues for query in (*self.holdout, *self.retention)):
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
            holdout = self._recall_accuracy(model, corpus.holdout, actions, use_memory=True)
            persistent_before = _persistent_digest(model)
            retention = self._recall_accuracy(model, corpus.retention, actions, use_memory=True)
            persistent_after = _persistent_digest(model)
            memory_lesion = self._recall_accuracy(model, corpus.holdout, actions, use_memory=False)
            frozen = Taiji(self._with_seed(seed), episode_id=f"m0-b2-frozen-{seed}")
            frozen_accuracy = self._recall_accuracy(
                frozen, corpus.holdout, actions, use_memory=True
            )
            seed_records.append(
                {
                    "seed": seed,
                    "taiji": holdout,
                    "retention": retention,
                    "memory_lesion": memory_lesion,
                    "frozen_parent": frozen_accuracy,
                    "holdout_updates": int(persistent_before != persistent_after),
                    "parameter_count": model.parameter_count(),
                }
            )
        native_values = [float(record["taiji"]) for record in seed_records]
        frozen_values = [float(record["frozen_parent"]) for record in seed_records]
        lesion_values = [float(record["memory_lesion"]) for record in seed_records]
        baseline_metrics = {
            "random": 1.0 / len(actions),
            "frozen_parent": min(frozen_values),
            "simple_rule": _majority_accuracy(corpus.train, corpus.holdout),
            "hash_only": min(
                _hash_memory_accuracy(corpus.holdout, actions, seed=seed)
                for seed in self.seeds
            ),
            "memory_lesion": min(lesion_values),
        }
        worst_native = min(native_values)
        holdout_updates = max(int(record["holdout_updates"]) for record in seed_records)
        beats_controls = all(worst_native > value for value in baseline_metrics.values())
        causal_memory_gain = all(
            float(record["taiji"]) > float(record["memory_lesion"])
            for record in seed_records
        )
        return FoundationMeasurement(
            ability_id=self.ability_id,
            status=(
                "passed"
                if beats_controls and causal_memory_gain and holdout_updates == 0
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
    def _write_episode(model: Taiji, episode: MemoryEpisode) -> None:
        model.reset_dynamics(episode_id=f"m0-b2-train-{episode.memory_id}")
        model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
        model.observe(episode.cue, learn=False, learn_motor=False)
        model.act((episode.action,), sample=False)
        model.settle_action(1.0, learn=False, learn_memory=True)
        model.observe(episode.outcome, learn=False, learn_motor=False)

    @staticmethod
    def _recall_accuracy(
        model: Taiji,
        queries: Sequence[DelayedMemoryQuery],
        actions: tuple[int, ...],
        *,
        use_memory: bool,
    ) -> float:
        correct = 0
        for query in queries:
            model.reset_dynamics(episode_id=f"m0-b2-query-{query.query_id}")
            model.observe(
                model.config.boundary_symbol,
                learn=False,
                learn_motor=False,
                use_memory=use_memory,
            )
            model.observe(
                query.cue,
                learn=False,
                learn_motor=False,
                use_memory=use_memory,
            )
            probabilities = model.snapshot().motor_probabilities
            prediction = max(actions, key=lambda action: float(probabilities[action].item()))
            correct += int(prediction == query.expected_action)
        return correct / len(queries)


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
    return content_digest(
        {
            "fabric": checkpoint["fabric"],
            "motor": checkpoint["motor"],
            "memory": checkpoint["memory"],
        }
    )


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
        digest = hashlib.sha256(f"{int(seed)}\0{int(query.cue)}".encode()).digest()
        prediction = actions[int.from_bytes(digest[:8], "big") % len(actions)]
        correct += int(prediction == query.expected_action)
    return correct / len(queries)

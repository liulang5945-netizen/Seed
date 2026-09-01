"""Native task adapters used by the M0 foundation benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from .config import TaijiConfig
from .foundation_evaluation import FoundationMeasurement
from .internalization import content_digest
from .model import Taiji


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

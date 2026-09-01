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

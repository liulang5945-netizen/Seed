"""A1 evaluation contract for Taiji learned perception."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .config import TaijiConfig
from .perception import LearnedPerception

SymbolSequence = tuple[int, ...]


@dataclass(frozen=True)
class PerceptionCorpus:
    """Disjoint A1 data roles; no answer mapping is stored in the evaluator."""

    train: tuple[SymbolSequence, ...]
    unseen_composition: tuple[SymbolSequence, ...]
    boundary_perturbed: tuple[SymbolSequence, ...]
    random_chunk: tuple[SymbolSequence, ...]

    def __post_init__(self) -> None:
        for name in ("train", "unseen_composition", "boundary_perturbed", "random_chunk"):
            sequences = getattr(self, name)
            if not sequences:
                raise ValueError(f"A1 {name} corpus cannot be empty")
            if any(not sequence for sequence in sequences):
                raise ValueError(f"A1 {name} corpus cannot contain empty sequences")
            if any(any(int(symbol) < 0 for symbol in sequence) for sequence in sequences):
                raise ValueError(f"A1 {name} corpus contains a negative symbol")


@dataclass(frozen=True)
class A1EvaluationConfig:
    ridge: float = 1e-3
    minimum_generalization_gain: float = 0.0
    maximum_cross_seed_std: float = 0.25
    minimum_boundary_rate_delta: float = 0.01
    minimum_random_chunk_drop: float = 0.005
    predictive_epochs: int = 3
    predictive_learning_rate: float = 0.01
    predictive_temperature: float = 0.15
    seeds: tuple[int, ...] = (11, 29, 47)

    def __post_init__(self) -> None:
        if self.ridge <= 0.0:
            raise ValueError("A1 ridge must be positive")
        if self.maximum_cross_seed_std < 0.0:
            raise ValueError("A1 maximum_cross_seed_std cannot be negative")
        if self.minimum_boundary_rate_delta < 0.0:
            raise ValueError("A1 minimum_boundary_rate_delta cannot be negative")
        if self.minimum_random_chunk_drop < 0.0:
            raise ValueError("A1 minimum_random_chunk_drop cannot be negative")
        if self.predictive_epochs <= 0:
            raise ValueError("A1 predictive_epochs must be positive")
        if self.predictive_learning_rate <= 0.0:
            raise ValueError("A1 predictive_learning_rate must be positive")
        if self.predictive_temperature <= 0.0:
            raise ValueError("A1 predictive_temperature must be positive")
        if not self.seeds:
            raise ValueError("A1 requires at least one evaluation seed")


class PerceptionEvaluator:
    """Evaluate migration and controls without changing the tested model."""

    def __init__(
        self,
        config: TaijiConfig,
        *,
        evaluation: A1EvaluationConfig | None = None,
    ) -> None:
        self.config = config
        self.evaluation = evaluation or A1EvaluationConfig()

    def evaluate(self, corpus: PerceptionCorpus) -> dict[str, Any]:
        """Return a reproducible report with learned and byte-only controls."""

        seed_reports = [
            self._evaluate_seed(corpus, seed=int(seed)) for seed in self.evaluation.seeds
        ]
        holdout_scores = torch.tensor(
            [report["unseen_composition"]["learned_accuracy"] for report in seed_reports],
            dtype=torch.float64,
        )
        mean_score = float(holdout_scores.mean())
        seed_std = float(holdout_scores.std(unbiased=False))
        primary = seed_reports[0]
        gain = float(primary["unseen_composition"]["generalization_gain"])
        variable_duration = primary["assembly"]["unique_durations"] > 1
        boundary_rate_delta = float(
            primary["boundary_perturbed"]["boundary_rate"]
            - primary["assembly"]["boundary_rate_unseen"]
        )
        random_chunk_drop = float(
            primary["unseen_composition"]["learned_accuracy"]
            - primary["random_chunk_control"]["learned_accuracy"]
        )
        gate_passed = bool(
            variable_duration
            and gain >= float(self.evaluation.minimum_generalization_gain)
            and seed_std <= float(self.evaluation.maximum_cross_seed_std)
            and boundary_rate_delta >= float(self.evaluation.minimum_boundary_rate_delta)
            and random_chunk_drop >= float(self.evaluation.minimum_random_chunk_drop)
        )
        return {
            "contract": "taiji-a1-perception-v1",
            "gate": "A1",
            "gate_passed": gate_passed,
            "config": {
                "ridge": self.evaluation.ridge,
                "minimum_generalization_gain": self.evaluation.minimum_generalization_gain,
                "maximum_cross_seed_std": self.evaluation.maximum_cross_seed_std,
                "minimum_boundary_rate_delta": self.evaluation.minimum_boundary_rate_delta,
                "minimum_random_chunk_drop": self.evaluation.minimum_random_chunk_drop,
                "predictive_epochs": self.evaluation.predictive_epochs,
                "predictive_learning_rate": self.evaluation.predictive_learning_rate,
                "predictive_temperature": self.evaluation.predictive_temperature,
                "seeds": list(self.evaluation.seeds),
            },
            "primary": primary,
            "cross_seed": {
                "learned_accuracy_mean": mean_score,
                "learned_accuracy_std": seed_std,
                "runs": seed_reports,
            },
            "diagnostics": {
                "boundary_rate_delta": boundary_rate_delta,
                "random_chunk_drop": random_chunk_drop,
            },
            "failure_policy": (
                "A1 remains open until learned assembly beats byte-only on unseen composition, "
                "responds to boundary perturbation, degrades under random chunk lesion, retains "
                "variable duration, and stays within the cross-seed variance budget."
            ),
        }

    def _evaluate_seed(self, corpus: PerceptionCorpus, *, seed: int) -> dict[str, Any]:
        config = self.config if int(seed) == int(self.config.seed) else self._with_seed(seed)
        model = LearnedPerception(config)
        training_loss = model.fit_predictive(
            corpus.train,
            epochs=self.evaluation.predictive_epochs,
            learning_rate=self.evaluation.predictive_learning_rate,
            temperature=self.evaluation.predictive_temperature,
        )
        train = self._collect(model, corpus.train, learn=False, label="train")
        unseen = self._collect(model, corpus.unseen_composition, learn=False, label="unseen")
        boundary = self._collect(model, corpus.boundary_perturbed, learn=False, label="boundary")
        random_chunk = self._collect(model, corpus.random_chunk, learn=False, label="random")

        learned_probe = self._fit_probe(train["features"], train["targets"])
        byte_probe = self._fit_probe(train["byte_features"], train["targets"])
        unseen_learned = self._accuracy(learned_probe, unseen["features"], unseen["targets"])
        unseen_byte = self._accuracy(byte_probe, unseen["byte_features"], unseen["targets"])
        return {
            "seed": int(seed),
            "predictive_training": {
                "final_loss": training_loss[-1],
                "loss_curve": training_loss,
            },
            "unseen_composition": {
                "learned_accuracy": unseen_learned,
                "byte_only_accuracy": unseen_byte,
                "generalization_gain": unseen_learned - unseen_byte,
            },
            "boundary_perturbed": {
                "mean_boundary_score": boundary["mean_boundary_score"],
                "boundary_rate": boundary["boundary_rate"],
                "score_delta_from_unseen": (
                    boundary["mean_boundary_score"] - unseen["mean_boundary_score"]
                ),
            },
            "random_chunk_control": {
                "learned_accuracy": self._accuracy(
                    learned_probe, random_chunk["features"], random_chunk["targets"]
                ),
                "byte_only_accuracy": self._accuracy(
                    byte_probe, random_chunk["byte_features"], random_chunk["targets"]
                ),
            },
            "assembly": {
                "mean_duration_train": train["mean_duration"],
                "mean_duration_unseen": unseen["mean_duration"],
                "unique_durations": train["unique_durations"],
                "boundary_rate_unseen": unseen["boundary_rate"],
            },
        }

    def _with_seed(self, seed: int) -> TaijiConfig:
        values = self.config.to_dict()
        values["seed"] = int(seed)
        return TaijiConfig.from_dict(values)

    def _collect(
        self,
        model: LearnedPerception,
        sequences: Sequence[SymbolSequence],
        *,
        learn: bool,
        label: str,
    ) -> dict[str, Any]:
        features: list[torch.Tensor] = []
        byte_features: list[torch.Tensor] = []
        targets: list[int] = []
        durations: list[int] = []
        boundary_scores: list[float] = []
        boundaries: list[bool] = []
        for sequence_index, sequence in enumerate(sequences):
            model.reset_dynamics()
            for tick, raw_symbol in enumerate(sequence):
                symbol = int(raw_symbol)
                if not 0 <= symbol < self.config.alphabet_size:
                    raise ValueError("A1 corpus symbol is outside the configured alphabet")
                event = model.observe(
                    symbol,
                    tick=tick,
                    stream_id=f"a1:{label}:{sequence_index}",
                    learn=learn,
                )
                if tick + 1 >= len(sequence):
                    continue
                features.append(event.features.detach().cpu())
                byte_features.append(
                    torch.nn.functional.one_hot(
                        torch.tensor(symbol), num_classes=self.config.alphabet_size
                    ).to(dtype=torch.float32)
                )
                targets.append(int(sequence[tick + 1]))
                durations.append(int(event.duration))
                boundary_scores.append(float(event.boundary_score))
                boundaries.append(bool(event.boundary))
        if not features:
            raise ValueError(f"A1 {label} corpus produced no probe samples")
        return {
            "features": torch.stack(features).to(dtype=torch.float64),
            "byte_features": torch.stack(byte_features).to(dtype=torch.float64),
            "targets": torch.tensor(targets, dtype=torch.long),
            "mean_duration": float(sum(durations) / len(durations)),
            "unique_durations": len(set(durations)),
            "mean_boundary_score": float(sum(boundary_scores) / len(boundary_scores)),
            "boundary_rate": float(sum(boundaries) / len(boundaries)),
        }

    def _fit_probe(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
            raise ValueError("A1 probe inputs have incompatible shapes")
        outputs = torch.nn.functional.one_hot(targets, num_classes=self.config.alphabet_size).to(
            dtype=torch.float64
        )
        gram = features.T @ features
        ridge = float(self.evaluation.ridge) * torch.eye(features.shape[1], dtype=torch.float64)
        return torch.linalg.solve(gram + ridge, features.T @ outputs)

    @staticmethod
    def _accuracy(probe: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> float:
        predictions = (features @ probe).argmax(dim=1)
        return float((predictions == targets).to(dtype=torch.float64).mean())

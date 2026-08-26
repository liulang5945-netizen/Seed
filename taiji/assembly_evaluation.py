"""Evaluation of compositional assembly binding for Taiji A1."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from .assembly_relations import AssemblyRelationCorpus, AssemblyRelationExample
from .config import TaijiConfig
from .perception import LearnedPerception


@dataclass(frozen=True)
class AssemblyRelationEvaluationConfig:
    ridge: float = 1e-3
    training_epochs: int = 3
    training_learning_rate: float = 0.01
    training_temperature: float = 0.15
    assembly_prediction_weight: float = 0.5
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.2
    minimum_slot_generalization_gain: float = 0.0
    minimum_boundary_consistency: float = 0.7
    minimum_random_binding_drop: float = 0.05
    maximum_cross_seed_std: float = 0.25
    seeds: tuple[int, ...] = (11, 29, 47)

    def __post_init__(self) -> None:
        if self.ridge <= 0.0:
            raise ValueError("assembly relation ridge must be positive")
        if self.training_epochs <= 0:
            raise ValueError("assembly relation training_epochs must be positive")
        if self.training_learning_rate <= 0.0:
            raise ValueError("assembly relation training_learning_rate must be positive")
        if self.training_temperature <= 0.0:
            raise ValueError("assembly relation training_temperature must be positive")
        if self.assembly_prediction_weight < 0.0:
            raise ValueError("assembly relation assembly_prediction_weight cannot be negative")
        if self.contrastive_weight < 0.0:
            raise ValueError("assembly relation contrastive_weight cannot be negative")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("assembly relation contrastive_temperature must be positive")
        if self.minimum_boundary_consistency < -1.0 or self.minimum_boundary_consistency > 1.0:
            raise ValueError("assembly relation minimum_boundary_consistency must be in [-1, 1]")
        if self.minimum_random_binding_drop < 0.0:
            raise ValueError("assembly relation minimum_random_binding_drop cannot be negative")
        if self.maximum_cross_seed_std < 0.0:
            raise ValueError("assembly relation maximum_cross_seed_std cannot be negative")
        if not self.seeds:
            raise ValueError("assembly relation requires at least one evaluation seed")


class AssemblyRelationEvaluator:
    """Evaluate slot binding without exposing relation metadata to the model."""

    def __init__(
        self,
        config: TaijiConfig,
        *,
        evaluation: AssemblyRelationEvaluationConfig | None = None,
    ) -> None:
        self.config = config
        self.evaluation = evaluation or AssemblyRelationEvaluationConfig()

    def evaluate(self, corpus: AssemblyRelationCorpus) -> dict[str, Any]:
        runs = [self._evaluate_seed(corpus, int(seed)) for seed in self.evaluation.seeds]
        slot_scores = torch.tensor(
            [run["slot_binding"]["pair_exact_accuracy_learned"] for run in runs],
            dtype=torch.float64,
        )
        primary = runs[0]
        slot_gains = [float(run["slot_binding"]["pair_exact_generalization_gain"]) for run in runs]
        boundary_scores = [float(run["consistency"]["boundary_consistency"]) for run in runs]
        random_drops = [float(run["controls"]["random_binding_drop"]) for run in runs]
        slot_gain = min(slot_gains)
        boundary_consistency = min(boundary_scores)
        random_drop = min(random_drops)
        cross_seed_std = float(slot_scores.std(unbiased=False))
        gate_passed = bool(
            slot_gain >= float(self.evaluation.minimum_slot_generalization_gain)
            and boundary_consistency >= float(self.evaluation.minimum_boundary_consistency)
            and random_drop >= float(self.evaluation.minimum_random_binding_drop)
            and cross_seed_std <= float(self.evaluation.maximum_cross_seed_std)
        )
        return {
            "contract": "taiji-a1-assembly-relation-v1",
            "gate": "A1",
            "gate_passed": gate_passed,
            "config": {
                "ridge": self.evaluation.ridge,
                "training_epochs": self.evaluation.training_epochs,
                "training_learning_rate": self.evaluation.training_learning_rate,
                "training_temperature": self.evaluation.training_temperature,
                "assembly_prediction_weight": self.evaluation.assembly_prediction_weight,
                "contrastive_weight": self.evaluation.contrastive_weight,
                "contrastive_temperature": self.evaluation.contrastive_temperature,
                "minimum_slot_generalization_gain": (
                    self.evaluation.minimum_slot_generalization_gain
                ),
                "minimum_boundary_consistency": self.evaluation.minimum_boundary_consistency,
                "minimum_random_binding_drop": self.evaluation.minimum_random_binding_drop,
                "maximum_cross_seed_std": self.evaluation.maximum_cross_seed_std,
                "seeds": list(self.evaluation.seeds),
            },
            "primary": primary,
            "cross_seed": {
                "slot_pair_exact_accuracy_mean": float(slot_scores.mean()),
                "slot_pair_exact_accuracy_std": cross_seed_std,
                "slot_generalization_gain_min": slot_gain,
                "boundary_consistency_min": boundary_consistency,
                "random_binding_drop_min": random_drop,
                "random_binding_drop_mean": float(sum(random_drops) / len(random_drops)),
                "runs": runs,
            },
            "failure_policy": (
                "A1 remains open until unseen ordered-pair slot binding beats the fair byte-bag "
                "control, boundary perturbation preserves the relation, random chunk lesion "
                "breaks that relation, and cross-seed variance stays within budget."
            ),
        }

    def _evaluate_seed(self, corpus: AssemblyRelationCorpus, seed: int) -> dict[str, Any]:
        config = self.config if seed == int(self.config.seed) else self._with_seed(seed)
        model = LearnedPerception(config)
        training_loss = model.fit_predictive(
            [example.sequence for example in corpus.train],
            epochs=self.evaluation.training_epochs,
            learning_rate=self.evaluation.training_learning_rate,
            temperature=self.evaluation.training_temperature,
            assembly_prediction_weight=self.evaluation.assembly_prediction_weight,
            contrastive_weight=self.evaluation.contrastive_weight,
            contrastive_temperature=self.evaluation.contrastive_temperature,
        )
        train = self._collect(model, corpus.train)
        unseen = self._collect(model, corpus.unseen_composition)
        boundary = self._collect(model, corpus.boundary_perturbed)
        random_chunk = self._collect(model, corpus.random_chunk)

        learned_left = self._fit_probe(train["features"], train["left_targets"])
        learned_right = self._fit_probe(train["features"], train["right_targets"])
        byte_left = self._fit_probe(train["byte_features"], train["left_targets"])
        byte_right = self._fit_probe(train["byte_features"], train["right_targets"])
        learned_unseen = self._slot_metrics(learned_left, learned_right, unseen)
        byte_unseen = self._slot_metrics(byte_left, byte_right, unseen, byte=True)
        learned_boundary = self._slot_metrics(learned_left, learned_right, boundary)
        learned_random = self._slot_metrics(learned_left, learned_right, random_chunk)
        random_binding_drop = (
            learned_unseen["pair_exact_accuracy"] - learned_random["pair_exact_accuracy"]
        )
        return {
            "seed": seed,
            "predictive_training": {
                "final_loss": training_loss[-1],
                "loss_curve": training_loss,
            },
            "slot_binding": {
                "left_accuracy_learned": learned_unseen["left_accuracy"],
                "right_accuracy_learned": learned_unseen["right_accuracy"],
                "pair_exact_accuracy_learned": learned_unseen["pair_exact_accuracy"],
                "pair_exact_accuracy_byte_bag": byte_unseen["pair_exact_accuracy"],
                "pair_exact_generalization_gain": (
                    learned_unseen["pair_exact_accuracy"] - byte_unseen["pair_exact_accuracy"]
                ),
            },
            "controls": {
                "boundary_pair_exact_accuracy": learned_boundary["pair_exact_accuracy"],
                "random_chunk_pair_exact_accuracy": learned_random["pair_exact_accuracy"],
                "random_binding_drop": random_binding_drop,
            },
            "consistency": {
                "boundary_consistency": self._cosine_mean(unseen["features"], boundary["features"]),
                "random_consistency": self._cosine_mean(
                    unseen["features"], random_chunk["features"]
                ),
                "random_consistency_drop": (
                    self._cosine_mean(unseen["features"], boundary["features"])
                    - self._cosine_mean(unseen["features"], random_chunk["features"])
                ),
            },
        }

    def _with_seed(self, seed: int) -> TaijiConfig:
        values = self.config.to_dict()
        values["seed"] = int(seed)
        return TaijiConfig.from_dict(values)

    def _collect(
        self,
        model: LearnedPerception,
        examples: Sequence[AssemblyRelationExample],
    ) -> dict[str, torch.Tensor]:
        features: list[torch.Tensor] = []
        byte_features: list[torch.Tensor] = []
        left_targets: list[int] = []
        right_targets: list[int] = []
        for index, example in enumerate(examples):
            model.reset_dynamics()
            events = [
                model.observe(
                    int(symbol),
                    tick=tick,
                    stream_id=f"a1-relation:{example.perturbation}:{index}",
                    learn=False,
                )
                for tick, symbol in enumerate(example.sequence)
            ]
            features.append(torch.stack([event.features for event in events]).mean(dim=0).cpu())
            byte_features.append(
                torch.nn.functional.one_hot(
                    torch.tensor(example.sequence), num_classes=self.config.alphabet_size
                )
                .to(dtype=torch.float32)
                .mean(dim=0)
            )
            left_targets.append(int(example.left_atom))
            right_targets.append(int(example.right_atom))
        return {
            "features": torch.stack(features).to(dtype=torch.float64),
            "byte_features": torch.stack(byte_features).to(dtype=torch.float64),
            "left_targets": torch.tensor(left_targets, dtype=torch.long),
            "right_targets": torch.tensor(right_targets, dtype=torch.long),
        }

    def _fit_probe(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = torch.nn.functional.one_hot(targets, num_classes=self.config.alphabet_size).to(
            dtype=torch.float64
        )
        gram = features.T @ features
        ridge = float(self.evaluation.ridge) * torch.eye(features.shape[1], dtype=torch.float64)
        return cast(torch.Tensor, torch.linalg.solve(gram + ridge, features.T @ outputs))

    @staticmethod
    def _slot_metrics(
        left_probe: torch.Tensor,
        right_probe: torch.Tensor,
        corpus: dict[str, torch.Tensor],
        *,
        byte: bool = False,
    ) -> dict[str, float]:
        features = corpus["byte_features"] if byte else corpus["features"]
        left = (features @ left_probe).argmax(dim=1)
        right = (features @ right_probe).argmax(dim=1)
        left_accuracy = (left == corpus["left_targets"]).to(dtype=torch.float64).mean()
        right_accuracy = (right == corpus["right_targets"]).to(dtype=torch.float64).mean()
        pair_accuracy = (
            ((left == corpus["left_targets"]) & (right == corpus["right_targets"]))
            .to(dtype=torch.float64)
            .mean()
        )
        return {
            "left_accuracy": float(left_accuracy),
            "right_accuracy": float(right_accuracy),
            "pair_exact_accuracy": float(pair_accuracy),
        }

    @staticmethod
    def _cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(
            torch.nn.functional.cosine_similarity(left, right, dim=1).to(dtype=torch.float64).mean()
        )

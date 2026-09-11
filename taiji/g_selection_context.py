"""Context-aware 22-parameter G selection learner for the P4.7 capacity test.

The model is ``nn.Linear(21, 1, bias=True)``: 12 candidate-feature weights
+ 9 candidate-set context-feature weights + 1 bias.  The P4.7 inherited
initialisation copies the 12 candidate weights and the bias verbatim from
a 13-parameter ``GSelectionLearner`` parent and zero-initialises the 9
context weights, so at birth (zero fit) the score on any candidate equals
the parent's score exactly—the context dimensions contribute zero.  This
is the zero-impact-at-birth capacity expansion operator whose existence
P4.1's context-lesion reference proved.

Context features are a pure function of the candidate set (P4.1
contract), so ``select`` remains self-contained.  The learner is a native
shadow learner: it never modifies the parent checkpoint, never touches
K1/K2 workers, and never admits topology growth or promotion.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .g_selection import (
    G_SELECTION_FEATURE_NAMES,
    GSelectionCandidate,
    GSelectionCandidateSet,
)
from .g_selection_learning import (
    GSelectionDecision,
    _digest,
    _finite,
    _positive,
    _unit,
)
from .internalization import content_digest
from .local_learning import apply_linear_delta, freeze_parameters, mean_squared_error_delta

CONTEXT_FEATURE_NAMES = (
    "candidate_count_norm",
    "proposal_fraction",
    "safe_fraction",
    "reobserve_fraction",
    "joint_score_mean",
    "joint_score_std",
    "joint_score_max",
    "candidate_rank_norm",
    "candidate_joint_minus_mean",
)
CONTEXT_G_LEARNER_FORMAT = "taiji-context-g-selection-learner-v1"
CONTEXT_G_LEARNER_VERSION = 1
TOTAL_FEATURE_NAMES = (*G_SELECTION_FEATURE_NAMES, *CONTEXT_FEATURE_NAMES)
_ROLE_PRIORITY = {"abstain": 2, "reobserve": 1, "proposal": 0}


def context_features(candidate_set: GSelectionCandidateSet) -> dict[str, tuple[float, ...]]:
    """Compute the 9-dimensional context feature vector per candidate.

    P4.1 contract: 7 set-level features shared by every candidate in the
    set plus 2 per-candidate features (rank norm, joint score minus set
    mean).
    """
    candidates = tuple(candidate_set.candidates)
    if len(candidates) < 2:
        raise ValueError("context features require at least two candidates")
    role_counts = {
        role: sum(item.candidate_role == role for item in candidates)
        for role in ("proposal", "abstain", "reobserve")
    }
    joint_scores = tuple(float(item.joint_score) for item in candidates)
    mean = sum(joint_scores) / len(joint_scores)
    variance = sum((value - mean) ** 2 for value in joint_scores) / len(joint_scores)
    maximum = max(joint_scores)
    ranking = sorted(
        candidates,
        key=lambda item: (
            -item.joint_score,
            -_ROLE_PRIORITY[item.candidate_role],
            item.candidate_id,
        ),
    )
    rank_by_id = {item.candidate_id: index for index, item in enumerate(ranking)}
    set_features = (
        min(1.0, len(candidates) / 12.0),
        role_counts["proposal"] / len(candidates),
        (role_counts["abstain"] + role_counts["reobserve"]) / len(candidates),
        role_counts["reobserve"] / len(candidates),
        mean,
        variance**0.5,
        maximum,
    )
    return {
        item.candidate_id: (
            *set_features,
            1.0 - rank_by_id[item.candidate_id] / max(1, len(candidates) - 1),
            item.joint_score - mean,
        )
        for item in candidates
    }


def augmented_feature_vector(
    candidate: GSelectionCandidate,
    context_vector: Sequence[float],
) -> tuple[float, ...]:
    """Concatenate the 12-dim candidate features with the 9-dim context."""
    return (*candidate.feature_vector, *tuple(float(value) for value in context_vector))


class ContextGSelectionLearner:
    """A 22-parameter G learner with inherited candidate weights.

    Candidate weights (12) and bias (1) are copied from a
    ``GSelectionLearner`` parent; context weights (9) start at zero so
    the birth behaviour matches the parent exactly.
    """

    def __init__(
        self,
        *,
        parent_manifest_digest: str,
        k_checkpoint_digests: Mapping[str, str],
        candidate_weight: Sequence[float],
        candidate_bias: float,
        learning_rate: float = 0.25,
        confidence_floor: float = 0.55,
        selection_margin: float = 0.05,
        device: torch.device | str = "cpu",
    ) -> None:
        self.parent_manifest_digest = _digest(
            str(parent_manifest_digest), "parent_manifest_digest"
        )
        digests = {
            str(key): _digest(str(value), f"k_checkpoint_digests[{key}]")
            for key, value in k_checkpoint_digests.items()
        }
        if tuple(sorted(digests)) != ("k1", "k2"):
            raise ValueError("Context G learner requires exactly k1 and k2 checkpoint digests")
        self.k_checkpoint_digests = digests
        self.learning_rate = _positive(learning_rate, "Context G learning_rate")
        self.confidence_floor = _unit(confidence_floor, "Context G confidence_floor")
        self.selection_margin = _finite(selection_margin, "Context G selection_margin")
        if self.selection_margin < 0.0:
            raise ValueError("Context G selection_margin cannot be negative")
        self.device = torch.device(device)
        self.model = nn.Linear(
            len(TOTAL_FEATURE_NAMES), 1, bias=True, device=self.device
        )
        weights = tuple(_finite(value, "candidate weight") for value in candidate_weight)
        if len(weights) != len(G_SELECTION_FEATURE_NAMES):
            raise ValueError("candidate weight length mismatch")
        bias = _finite(candidate_bias, "candidate bias")
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()
            self.model.weight[0, : len(G_SELECTION_FEATURE_NAMES)] = torch.tensor(
                weights, dtype=torch.float32, device=self.device
            )
            self.model.bias.fill_(bias)
        freeze_parameters(self.model)
        self.training_steps = 0
        self.revision = 0
        self.last_train_digest = ""

    @classmethod
    def from_parent_learner(
        cls,
        parent: Any,
        *,
        device: torch.device | str = "cpu",
    ) -> ContextGSelectionLearner:
        """Inherit candidate weights and bias from a 13-parameter parent."""
        if int(parent.parameter_count) != len(G_SELECTION_FEATURE_NAMES) + 1:
            raise ValueError(
                "Context G inheritance requires the 13-parameter GSelectionLearner parent"
            )
        with torch.no_grad():
            candidate_weight = [
                float(value)
                for value in parent.model.weight.detach().cpu().reshape(-1)
            ]
            candidate_bias = float(parent.model.bias.detach().cpu().reshape(()))
        return cls(
            parent_manifest_digest=parent.parent_manifest_digest,
            k_checkpoint_digests=parent.k_checkpoint_digests,
            candidate_weight=candidate_weight,
            candidate_bias=candidate_bias,
            learning_rate=parent.learning_rate,
            confidence_floor=parent.confidence_floor,
            selection_margin=parent.selection_margin,
            device=device,
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.model.parameters())

    @property
    def model_state_digest(self) -> str:
        return content_digest(
            {
                "weight": self.model.weight.detach().cpu(),
                "bias": self.model.bias.detach().cpu(),
            }
        )

    def assert_lineage(
        self,
        *,
        parent_manifest_digest: str,
        k_checkpoint_digests: Mapping[str, str],
    ) -> None:
        expected_parent = _digest(
            str(parent_manifest_digest), "expected parent_manifest_digest"
        )
        expected_k = {
            str(key): _digest(str(value), f"expected k_checkpoint_digests[{key}]")
            for key, value in k_checkpoint_digests.items()
        }
        if self.parent_manifest_digest != expected_parent:
            raise ValueError("Context G learner parent manifest lineage mismatch")
        if self.k_checkpoint_digests != expected_k:
            raise ValueError("Context G learner K checkpoint lineage mismatch")

    def _features(
        self, candidate: GSelectionCandidate, context_vector: Sequence[float]
    ) -> torch.Tensor:
        if not isinstance(candidate, GSelectionCandidate):
            raise TypeError("Context G learner features require a GSelectionCandidate")
        vector = augmented_feature_vector(candidate, context_vector)
        if len(vector) != len(TOTAL_FEATURE_NAMES):
            raise ValueError("Context G learner feature manifest length drifted")
        features = torch.tensor(vector, dtype=torch.float32, device=self.device).reshape(1, -1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Context G learner augmented features must be finite")
        return features

    def score(
        self, candidate: GSelectionCandidate, context_vector: Sequence[float]
    ) -> float:
        with torch.no_grad():
            return float(
                self.model(self._features(candidate, context_vector)).reshape(()).item()
            )

    @staticmethod
    def _safe_candidate(candidates: Sequence[GSelectionCandidate]) -> GSelectionCandidate:
        safe = [
            candidate
            for candidate in candidates
            if candidate.candidate_role in {"abstain", "reobserve"}
        ]
        if not safe:
            raise ValueError("G candidate set has no safe abstain or reobserve candidate")
        return max(
            safe,
            key=lambda candidate: (
                1 if candidate.candidate_role == "abstain" else 0,
                candidate.confidence,
                candidate.candidate_id,
            ),
        )

    def select(self, candidate_set: GSelectionCandidateSet) -> GSelectionDecision:
        if not isinstance(candidate_set, GSelectionCandidateSet):
            raise TypeError("Context G learner select requires a GSelectionCandidateSet")
        context = context_features(candidate_set)
        scored = [
            (candidate, self.score(candidate, context[candidate.candidate_id]))
            for candidate in candidate_set.candidates
        ]
        scored.sort(
            key=lambda item: (
                -item[1],
                -_ROLE_PRIORITY[item[0].candidate_role],
                item[0].candidate_id,
            )
        )
        selected, selected_score = scored[0]
        safe = self._safe_candidate(candidate_set.candidates)
        safe_score = self.score(safe, context[safe.candidate_id])
        if selected.candidate_role == "proposal":
            proposal_unsafe = (
                selected.goal is None
                or selected.content_plan is None
                or selected.confidence < self.confidence_floor
                or selected_score <= safe_score + self.selection_margin
            )
            if proposal_unsafe:
                selected = safe
        elif selected_score <= safe_score + self.selection_margin:
            selected = safe
        if selected.candidate_role == "proposal":
            status = "selected"
        elif selected.candidate_role == "reobserve":
            status = "reobserve"
        else:
            status = "abstained"
        scores = tuple(
            sorted(
                ((candidate.candidate_id, score) for candidate, score in scored),
                key=lambda item: item[0],
            )
        )
        return GSelectionDecision.create(
            candidate_set=candidate_set,
            selected=selected,
            selection_status=status,
            candidate_scores=scores,
        )

    def functional_fit(
        self,
        parent_learner: Any,
        records: Sequence[Mapping[str, Any]],
        constraint_features: Sequence[Sequence[Sequence[float]]],
        *,
        epochs: int = 1,
        learning_rate: float = 0.15,
        order_seed: int = 0,
        functional_weight: float = 1.0,
        constraint_digest: str = "",
    ) -> dict[str, Any]:
        """Functional parent-preserving fit (P4.6 protocol, 21-dim inputs).

        Each step applies the task delta on one train record, then a
        constraint delta matching the parent's 13-parameter score on the
        candidate-only part of one constraint-cohort record.  The
        constraint cohort's behavior target/utility is never read.
        """
        items = tuple(record["candidate_set"] for record in records)
        if not items or any(item.split != "train" for item in items):
            raise ValueError("Context G functional fit requires non-empty train records")
        if not constraint_features:
            raise ValueError(
                "Context G functional fit requires an independent constraint cohort"
            )
        if functional_weight <= 0.0:
            raise ValueError("Context G functional weight must be positive")
        dataset_digest = content_digest(
            {
                "candidate_set_digests": [item.candidate_set_digest for item in items],
                "constraint_digest": constraint_digest,
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "order_seed": int(order_seed),
                "functional_weight": float(functional_weight),
            }
        )
        task_loss_sum = 0.0
        constraint_loss_sum = 0.0
        constraint_steps = 0
        for epoch in range(int(epochs)):
            order = list(range(len(items)))
            random.Random(int(order_seed) + epoch).shuffle(order)
            for step, index in enumerate(order):
                item = items[index]
                context = context_features(item)
                task_inputs = torch.tensor(
                    [
                        augmented_feature_vector(candidate, context[candidate.candidate_id])
                        for candidate in item.candidates
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                task_targets = torch.zeros(
                    (len(item.candidates), 1), dtype=torch.float32, device=self.device
                )
                target_index = next(
                    position
                    for position, candidate in enumerate(item.candidates)
                    if candidate.candidate_id == item.target_candidate_id
                )
                task_targets[target_index, 0] = 1.0
                task_predictions = self.model(task_inputs)
                task_loss_sum += float(
                    torch.mean((task_predictions - task_targets) ** 2).item()
                )
                apply_linear_delta(
                    self.model,
                    task_inputs,
                    mean_squared_error_delta(task_predictions, task_targets),
                    float(learning_rate),
                )
                group = constraint_features[
                    (epoch * len(order) + step) % len(constraint_features)
                ]
                constraint_inputs = torch.tensor(
                    group, dtype=torch.float32, device=self.device
                )
                candidate_only = constraint_inputs[:, : len(G_SELECTION_FEATURE_NAMES)]
                with torch.no_grad():
                    teacher = parent_learner.model(candidate_only).detach()
                student = self.model(constraint_inputs)
                constraint_loss_sum += float(torch.mean((student - teacher) ** 2).item())
                apply_linear_delta(
                    self.model,
                    constraint_inputs,
                    mean_squared_error_delta(student, teacher) * float(functional_weight),
                    float(learning_rate),
                )
                constraint_steps += 1
                self.training_steps += 2
        self.revision += 1
        self.last_train_digest = dataset_digest
        return {
            "dataset_digest": dataset_digest,
            "candidate_sets": len(items),
            "constraint_groups": len(constraint_features),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "order_seed": int(order_seed),
            "functional_weight": float(functional_weight),
            "training_steps": self.training_steps,
            "revision": self.revision,
            "task_loss_mean": task_loss_sum / (len(items) * int(epochs)),
            "constraint_loss_mean": (
                constraint_loss_sum / constraint_steps if constraint_steps else 0.0
            ),
            "constraint_steps": constraint_steps,
        }

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": CONTEXT_G_LEARNER_FORMAT,
            "version": CONTEXT_G_LEARNER_VERSION,
            "candidate_feature_names": list(G_SELECTION_FEATURE_NAMES),
            "context_feature_names": list(CONTEXT_FEATURE_NAMES),
            "parent_manifest_digest": self.parent_manifest_digest,
            "k_checkpoint_digests": dict(self.k_checkpoint_digests),
            "learning_rate": self.learning_rate,
            "confidence_floor": self.confidence_floor,
            "selection_margin": self.selection_margin,
            "parameter_count": self.parameter_count,
            "weight": self.model.weight.detach().cpu().clone(),
            "bias": self.model.bias.detach().cpu().clone(),
            "model_state_digest": self.model_state_digest,
            "training_steps": self.training_steps,
            "revision": self.revision,
            "last_train_digest": self.last_train_digest,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> ContextGSelectionLearner:
        if payload.get("format") != CONTEXT_G_LEARNER_FORMAT:
            raise ValueError("unsupported Context G learner checkpoint format")
        if int(payload.get("version", -1)) != CONTEXT_G_LEARNER_VERSION:
            raise ValueError("unsupported Context G learner checkpoint version")
        expected_digest = str(payload["checkpoint_digest"])
        unsigned = {
            key: value for key, value in payload.items() if key != "checkpoint_digest"
        }
        if content_digest(unsigned) != expected_digest:
            raise ValueError("Context G learner checkpoint digest mismatch")
        if tuple(payload.get("candidate_feature_names", ())) != G_SELECTION_FEATURE_NAMES:
            raise ValueError("Context G learner candidate feature manifest drifted")
        if tuple(payload.get("context_feature_names", ())) != CONTEXT_FEATURE_NAMES:
            raise ValueError("Context G learner context feature manifest drifted")
        weight = payload["weight"]
        bias = payload["bias"]
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise TypeError("Context G learner checkpoint weights must be tensors")
        if tuple(weight.shape) != (1, len(TOTAL_FEATURE_NAMES)) or tuple(bias.shape) != (1,):
            raise ValueError("Context G learner checkpoint tensor shapes drifted")
        if content_digest({"weight": weight, "bias": bias}) != str(
            payload["model_state_digest"]
        ):
            raise ValueError("Context G learner model state digest mismatch")
        learner = cls(
            parent_manifest_digest=str(payload["parent_manifest_digest"]),
            k_checkpoint_digests=dict(payload["k_checkpoint_digests"]),
            candidate_weight=[
                float(value) for value in weight[0, : len(G_SELECTION_FEATURE_NAMES)]
            ],
            candidate_bias=float(bias.reshape(())),
            learning_rate=float(payload["learning_rate"]),
            confidence_floor=float(payload["confidence_floor"]),
            selection_margin=float(payload["selection_margin"]),
            device=device,
        )
        if int(payload["parameter_count"]) != learner.parameter_count:
            raise ValueError("Context G learner parameter count drifted")
        with torch.no_grad():
            learner.model.weight.copy_(
                weight.detach().to(device=learner.device, dtype=torch.float32)
            )
            learner.model.bias.copy_(
                bias.detach().to(device=learner.device, dtype=torch.float32)
            )
        learner.training_steps = int(payload.get("training_steps", 0))
        learner.revision = int(payload.get("revision", 0))
        learner.last_train_digest = str(payload.get("last_train_digest", ""))
        freeze_parameters(learner.model)
        return learner


__all__ = [
    "CONTEXT_FEATURE_NAMES",
    "CONTEXT_G_LEARNER_FORMAT",
    "CONTEXT_G_LEARNER_VERSION",
    "TOTAL_FEATURE_NAMES",
    "ContextGSelectionLearner",
    "augmented_feature_vector",
    "context_features",
]

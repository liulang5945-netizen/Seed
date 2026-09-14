"""Native incremental learning for the Taiji G selection organ.

P3.3 deliberately starts with a small, inspectable learner.  K produces a
content-addressed candidate set; G learns only a scalar score for each
candidate feature vector.  The K checkpoints are lineage inputs, never
trainable parameters, and the learner has no provider, text, or Transformer
dependency.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .g_selection import G_SELECTION_FEATURE_NAMES, GSelectionCandidate, GSelectionCandidateSet
from .internalization import content_digest
from .local_learning import apply_linear_delta, freeze_parameters, mean_squared_error_delta

G_SELECTION_LEARNER_FORMAT = "taiji-g-selection-learner-v1"
G_SELECTION_LEARNER_VERSION = 1
G_SELECTION_SELECTION_STATUSES = ("selected", "abstained", "reobserve")
G_SELECTION_CANDIDATE_ROLES = ("proposal", "abstain", "reobserve")


def _text(value: str, name: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _digest(value: str, name: str) -> str:
    value = _text(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _unit(value: float, name: str) -> float:
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return value


def _candidate_scores(value: Sequence[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    result = tuple(
        (str(candidate_id), _finite(score, "candidate score")) for candidate_id, score in value
    )
    if not result:
        raise ValueError("candidate scores cannot be empty")
    if len({candidate_id for candidate_id, _score in result}) != len(result):
        raise ValueError("candidate scores must have unique candidate ids")
    if tuple(sorted(candidate_id for candidate_id, _score in result)) != tuple(
        candidate_id for candidate_id, _score in result
    ):
        raise ValueError("candidate scores must be sorted by candidate id")
    return result


@dataclass(frozen=True)
class GSelectionDecision:
    """A label-free runtime selection result emitted by G."""

    candidate_set_digest: str
    inference_digest: str
    selected_candidate_id: str
    selected_candidate_digest: str
    selected_candidate_role: str
    selection_status: str
    selected_goal_id: str | None
    selected_content_id: str | None
    candidate_scores: tuple[tuple[str, float], ...]
    confidence: float
    ambiguity: float
    external_target_used: bool
    decision_digest: str
    format: str = G_SELECTION_LEARNER_FORMAT
    version: int = G_SELECTION_LEARNER_VERSION

    def __post_init__(self) -> None:
        if self.format != G_SELECTION_LEARNER_FORMAT:
            raise ValueError("unsupported G selection decision format")
        if int(self.version) != G_SELECTION_LEARNER_VERSION:
            raise ValueError("unsupported G selection decision version")
        candidate_set_digest = _digest(self.candidate_set_digest, "candidate_set_digest")
        inference_digest = _digest(self.inference_digest, "inference_digest")
        selected_id = _text(self.selected_candidate_id, "selected_candidate_id")
        selected_digest = _digest(self.selected_candidate_digest, "selected_candidate_digest")
        role = _text(self.selected_candidate_role, "selected_candidate_role")
        if role not in G_SELECTION_CANDIDATE_ROLES:
            raise ValueError("unsupported selected candidate role")
        status = _text(self.selection_status, "selection_status")
        if status not in G_SELECTION_SELECTION_STATUSES:
            raise ValueError("unsupported G selection status")
        scores = _candidate_scores(self.candidate_scores)
        if selected_id not in {candidate_id for candidate_id, _score in scores}:
            raise ValueError("selected candidate is missing from candidate scores")
        if status == "selected" and (
            role != "proposal" or not self.selected_goal_id or not self.selected_content_id
        ):
            raise ValueError("selected G decision must contain a proposal goal and content")
        if status == "abstained" and role != "abstain":
            raise ValueError("abstained G decision must select an abstain candidate")
        if status == "reobserve" and role != "reobserve":
            raise ValueError("reobserve G decision must select a reobserve candidate")
        goal_id = (
            None
            if self.selected_goal_id is None
            else _text(self.selected_goal_id, "selected_goal_id")
        )
        content_id = (
            None
            if self.selected_content_id is None
            else _text(self.selected_content_id, "selected_content_id")
        )
        if status != "selected" and (goal_id is not None or content_id is not None):
            raise ValueError("safe G decisions cannot emit a goal or content")
        confidence = _unit(self.confidence, "decision confidence")
        ambiguity = _unit(self.ambiguity, "decision ambiguity")
        if bool(self.external_target_used):
            raise ValueError("runtime G decisions cannot use external targets")
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "candidate_set_digest": candidate_set_digest,
            "inference_digest": inference_digest,
            "selected_candidate_id": selected_id,
            "selected_candidate_digest": selected_digest,
            "selected_candidate_role": role,
            "selection_status": status,
            "selected_goal_id": goal_id,
            "selected_content_id": content_id,
            "candidate_scores": [[candidate_id, score] for candidate_id, score in scores],
            "confidence": confidence,
            "ambiguity": ambiguity,
            "external_target_used": False,
        }
        if content_digest(unsigned) != _digest(self.decision_digest, "decision_digest"):
            raise ValueError("G selection decision digest mismatch")
        object.__setattr__(self, "candidate_set_digest", candidate_set_digest)
        object.__setattr__(self, "inference_digest", inference_digest)
        object.__setattr__(self, "selected_candidate_id", selected_id)
        object.__setattr__(self, "selected_candidate_digest", selected_digest)
        object.__setattr__(self, "selected_candidate_role", role)
        object.__setattr__(self, "selection_status", status)
        object.__setattr__(self, "selected_goal_id", goal_id)
        object.__setattr__(self, "selected_content_id", content_id)
        object.__setattr__(self, "candidate_scores", scores)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "ambiguity", ambiguity)
        object.__setattr__(self, "external_target_used", False)
        object.__setattr__(
            self, "decision_digest", _digest(self.decision_digest, "decision_digest")
        )

    @classmethod
    def create(
        cls,
        *,
        candidate_set: GSelectionCandidateSet,
        selected: GSelectionCandidate,
        selection_status: str,
        candidate_scores: Sequence[tuple[str, float]],
    ) -> GSelectionDecision:
        if not isinstance(candidate_set, GSelectionCandidateSet):
            raise TypeError("G decision requires a GSelectionCandidateSet")
        if not isinstance(selected, GSelectionCandidate):
            raise TypeError("G decision requires a GSelectionCandidate")
        selected_goal_id = None if selected.goal is None else selected.goal.goal_id
        selected_content_id = (
            None if selected.content_plan is None else selected.content_plan.content_id
        )
        if selection_status != "selected":
            selected_goal_id = None
            selected_content_id = None
        normalized_scores = tuple(
            sorted(
                ((str(candidate_id), float(score)) for candidate_id, score in candidate_scores),
                key=lambda item: item[0],
            )
        )
        unsigned = {
            "format": G_SELECTION_LEARNER_FORMAT,
            "version": G_SELECTION_LEARNER_VERSION,
            "candidate_set_digest": candidate_set.candidate_set_digest,
            "inference_digest": candidate_set.inference_digest,
            "selected_candidate_id": selected.candidate_id,
            "selected_candidate_digest": selected.candidate_digest,
            "selected_candidate_role": selected.candidate_role,
            "selection_status": str(selection_status),
            "selected_goal_id": selected_goal_id,
            "selected_content_id": selected_content_id,
            "candidate_scores": [
                [candidate_id, score] for candidate_id, score in normalized_scores
            ],
            "confidence": selected.confidence,
            "ambiguity": selected.ambiguity,
            "external_target_used": False,
        }
        return cls.from_payload({**unsigned, "decision_digest": content_digest(unsigned)})

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "candidate_set_digest": self.candidate_set_digest,
            "inference_digest": self.inference_digest,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selected_candidate_role": self.selected_candidate_role,
            "selection_status": self.selection_status,
            "selected_goal_id": self.selected_goal_id,
            "selected_content_id": self.selected_content_id,
            "candidate_scores": [
                [candidate_id, score] for candidate_id, score in self.candidate_scores
            ],
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "external_target_used": self.external_target_used,
            "decision_digest": self.decision_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GSelectionDecision:
        return cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            candidate_set_digest=str(payload["candidate_set_digest"]),
            inference_digest=str(payload["inference_digest"]),
            selected_candidate_id=str(payload["selected_candidate_id"]),
            selected_candidate_digest=str(payload["selected_candidate_digest"]),
            selected_candidate_role=str(payload["selected_candidate_role"]),
            selection_status=str(payload["selection_status"]),
            selected_goal_id=(
                None
                if payload.get("selected_goal_id") is None
                else str(payload["selected_goal_id"])
            ),
            selected_content_id=(
                None
                if payload.get("selected_content_id") is None
                else str(payload["selected_content_id"])
            ),
            candidate_scores=tuple(
                (str(item[0]), float(item[1])) for item in payload.get("candidate_scores", ())
            ),
            confidence=float(payload["confidence"]),
            ambiguity=float(payload["ambiguity"]),
            external_target_used=bool(payload.get("external_target_used", False)),
            decision_digest=str(payload["decision_digest"]),
        )


class GSelectionLearner:
    """A frozen-lineage native linear scorer that learns only G parameters."""

    def __init__(
        self,
        *,
        parent_manifest_digest: str,
        k_checkpoint_digests: Mapping[str, str],
        learning_rate: float = 0.25,
        confidence_floor: float = 0.55,
        selection_margin: float = 0.05,
        device: torch.device | str = "cpu",
    ) -> None:
        self.parent_manifest_digest = _digest(parent_manifest_digest, "parent_manifest_digest")
        digests = {
            str(key): _digest(str(value), f"k_checkpoint_digests[{key}]")
            for key, value in k_checkpoint_digests.items()
        }
        if tuple(sorted(digests)) != ("k1", "k2"):
            raise ValueError("G learner requires exactly k1 and k2 checkpoint digests")
        self.k_checkpoint_digests = digests
        self.learning_rate = _positive(learning_rate, "G learning_rate")
        self.confidence_floor = _unit(confidence_floor, "G confidence_floor")
        self.selection_margin = _finite(selection_margin, "G selection_margin")
        if self.selection_margin < 0.0:
            raise ValueError("G selection_margin cannot be negative")
        self.device = torch.device(device)
        self.model = nn.Linear(len(G_SELECTION_FEATURE_NAMES), 1, bias=True, device=self.device)
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()
            self.model.weight[0, 2] = 1.0
        freeze_parameters(self.model)
        self.training_steps = 0
        self.revision = 0
        self.last_train_digest = ""

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
        expected_parent = _digest(parent_manifest_digest, "expected parent_manifest_digest")
        expected_k = {
            str(key): _digest(str(value), f"expected k_checkpoint_digests[{key}]")
            for key, value in k_checkpoint_digests.items()
        }
        if self.parent_manifest_digest != expected_parent:
            raise ValueError("G learner parent manifest lineage mismatch")
        if self.k_checkpoint_digests != expected_k:
            raise ValueError("G learner K checkpoint lineage mismatch")

    def _features(self, candidate: GSelectionCandidate) -> torch.Tensor:
        if not isinstance(candidate, GSelectionCandidate):
            raise TypeError("G learner features require a GSelectionCandidate")
        features = torch.tensor(
            candidate.feature_vector, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        if features.shape[1] != len(G_SELECTION_FEATURE_NAMES):
            raise ValueError("G learner feature manifest length drifted")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("G learner candidate features must be finite")
        return features

    def score(self, candidate: GSelectionCandidate) -> float:
        with torch.no_grad():
            return float(self.model(self._features(candidate)).reshape(()).item())

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
            raise TypeError("G learner select requires a GSelectionCandidateSet")
        scored = [(candidate, self.score(candidate)) for candidate in candidate_set.candidates]
        role_priority = {"abstain": 2, "reobserve": 1, "proposal": 0}
        scored.sort(
            key=lambda item: (
                -item[1],
                -role_priority[item[0].candidate_role],
                item[0].candidate_id,
            )
        )
        selected, selected_score = scored[0]
        safe = self._safe_candidate(candidate_set.candidates)
        safe_score = self.score(safe)
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

    @staticmethod
    def _training_items(
        candidate_sets: Iterable[GSelectionCandidateSet],
    ) -> tuple[GSelectionCandidateSet, ...]:
        items = tuple(candidate_sets)
        if not items:
            raise ValueError("G training requires at least one candidate set")
        if any(not isinstance(item, GSelectionCandidateSet) for item in items):
            raise TypeError("G training requires GSelectionCandidateSet values")
        if any(item.split != "train" for item in items):
            raise ValueError("G training accepts train candidate sets only")
        if len({item.candidate_set_digest for item in items}) != len(items):
            raise ValueError("G training candidate sets must be unique")
        for item in items:
            target = item.target_candidate()
            if item.target_kind == "pair" and (
                target.candidate_role != "proposal"
                or target.goal is None
                or target.content_plan is None
            ):
                raise ValueError("pair G target must be a complete proposal")
            if item.target_kind == "abstain" and target.candidate_role != "abstain":
                raise ValueError("abstain G target must be an abstain candidate")
            if item.target_kind == "reobserve" and target.candidate_role != "reobserve":
                raise ValueError("reobserve G target must be a reobserve candidate")
        return tuple(sorted(items, key=lambda item: item.candidate_set_digest))

    def fit(
        self,
        candidate_sets: Iterable[GSelectionCandidateSet],
        *,
        epochs: int = 1,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        items = self._training_items(candidate_sets)
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError("G training epochs must be positive")
        rate = (
            self.learning_rate
            if learning_rate is None
            else _positive(learning_rate, "G fit learning_rate")
        )
        dataset_digest = content_digest(
            {
                "candidate_set_digests": [item.candidate_set_digest for item in items],
                "epochs": epochs,
                "learning_rate": rate,
            }
        )
        for _epoch in range(epochs):
            for item in items:
                inputs = torch.tensor(
                    [candidate.feature_vector for candidate in item.candidates],
                    dtype=torch.float32,
                    device=self.device,
                )
                targets = torch.zeros(
                    (len(item.candidates), 1), dtype=torch.float32, device=self.device
                )
                target_index = next(
                    index
                    for index, candidate in enumerate(item.candidates)
                    if candidate.candidate_id == item.target_candidate_id
                )
                targets[target_index, 0] = 1.0
                predictions = self.model(inputs)
                error = mean_squared_error_delta(predictions, targets)
                apply_linear_delta(self.model, inputs, error, rate)
                self.training_steps += 1
        self.revision += 1
        self.last_train_digest = dataset_digest
        return {
            "dataset_digest": dataset_digest,
            "candidate_sets": len(items),
            "epochs": epochs,
            "learning_rate": rate,
            "training_steps": self.training_steps,
            "revision": self.revision,
        }

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": G_SELECTION_LEARNER_FORMAT,
            "version": G_SELECTION_LEARNER_VERSION,
            "feature_names": list(G_SELECTION_FEATURE_NAMES),
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
    ) -> GSelectionLearner:
        if payload.get("format") != G_SELECTION_LEARNER_FORMAT:
            raise ValueError("unsupported G learner checkpoint format")
        if int(payload.get("version", -1)) != G_SELECTION_LEARNER_VERSION:
            raise ValueError("unsupported G learner checkpoint version")
        expected_digest = _digest(str(payload["checkpoint_digest"]), "checkpoint_digest")
        unsigned = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        if content_digest(unsigned) != expected_digest:
            raise ValueError("G learner checkpoint digest mismatch")
        if tuple(payload.get("feature_names", ())) != G_SELECTION_FEATURE_NAMES:
            raise ValueError("G learner feature manifest drifted")
        learner = cls(
            parent_manifest_digest=str(payload["parent_manifest_digest"]),
            k_checkpoint_digests=dict(payload["k_checkpoint_digests"]),
            learning_rate=float(payload["learning_rate"]),
            confidence_floor=float(payload["confidence_floor"]),
            selection_margin=float(payload["selection_margin"]),
            device=device,
        )
        if int(payload["parameter_count"]) != learner.parameter_count:
            raise ValueError("G learner parameter count drifted")
        weight = payload["weight"]
        bias = payload["bias"]
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise TypeError("G learner checkpoint weights must be tensors")
        if tuple(weight.shape) != tuple(learner.model.weight.shape) or tuple(bias.shape) != tuple(
            learner.model.bias.shape
        ):
            raise ValueError("G learner checkpoint tensor shapes drifted")
        if content_digest({"weight": weight, "bias": bias}) != str(payload["model_state_digest"]):
            raise ValueError("G learner model state digest mismatch")
        with torch.no_grad():
            learner.model.weight.copy_(
                weight.detach().to(device=learner.device, dtype=torch.float32)
            )
            learner.model.bias.copy_(bias.detach().to(device=learner.device, dtype=torch.float32))
        learner.training_steps = int(payload.get("training_steps", 0))
        learner.revision = int(payload.get("revision", 0))
        learner.last_train_digest = str(payload.get("last_train_digest", ""))
        freeze_parameters(learner.model)
        return learner


__all__ = [
    "G_SELECTION_CANDIDATE_ROLES",
    "G_SELECTION_LEARNER_FORMAT",
    "G_SELECTION_LEARNER_VERSION",
    "G_SELECTION_SELECTION_STATUSES",
    "GSelectionDecision",
    "GSelectionLearner",
]

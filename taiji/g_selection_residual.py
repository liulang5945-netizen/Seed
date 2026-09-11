"""Residual G selection learner for the P4.8 representation redesign.

Architecture: a frozen parent head (13 parameters copied bit-exact from a
``GSelectionLearner`` parent, never trainable) plus a zero-initialised
delta head (13 parameters, the only trainable layer).  The decision score
is ``parent_head(x) + delta_head(x)``, so at birth the learner reproduces
the parent's decisions exactly—by construction, with zero numerical
deviation.

The preservation constraint is the preregistered margin-preservation
hinge (P4.8 §3.1): the delta may move scores freely as long as no
parent decision margin erodes below its frozen-parent value.  The hinge
has finite support (zero gradient once margins are intact) and its birth
loss is zero by construction, unlike the scalar-MSE teacher constraint
whose gradient never saturates.  The reference margins are always
available from the frozen parent head—the birth state IS the parent
state.

The learner is a native shadow learner: it never modifies the parent
checkpoint, never touches K1/K2 workers, and never admits topology
growth or promotion.
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

RESIDUAL_G_LEARNER_FORMAT = "taiji-residual-g-selection-learner-v1"
RESIDUAL_G_LEARNER_VERSION = 1
_ROLE_PRIORITY = {"abstain": 2, "reobserve": 1, "proposal": 0}


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


def _apply_selection_rule(
    scored: Sequence[tuple[GSelectionCandidate, float]],
    *,
    confidence_floor: float,
    selection_margin: float,
) -> tuple[GSelectionCandidate, str]:
    """Replicate the frozen G selection rule over arbitrary scores."""
    ordered = sorted(
        scored,
        key=lambda item: (
            -item[1],
            -_ROLE_PRIORITY[item[0].candidate_role],
            item[0].candidate_id,
        ),
    )
    selected, selected_score = ordered[0]
    safe = _safe_candidate([candidate for candidate, _score in scored])
    safe_score = next(score for candidate, score in scored if candidate is safe)
    if selected.candidate_role == "proposal":
        proposal_unsafe = (
            selected.goal is None
            or selected.content_plan is None
            or selected.confidence < confidence_floor
            or selected_score <= safe_score + selection_margin
        )
        if proposal_unsafe:
            selected = safe
    elif selected_score <= safe_score + selection_margin:
        selected = safe
    if selected.candidate_role == "proposal":
        status = "selected"
    elif selected.candidate_role == "reobserve":
        status = "reobserve"
    else:
        status = "abstained"
    return selected, status


class ResidualGSelectionLearner:
    """Frozen parent head + trainable zero-initialised delta head."""

    def __init__(
        self,
        *,
        parent_manifest_digest: str,
        k_checkpoint_digests: Mapping[str, str],
        parent_weight: Sequence[float],
        parent_bias: float,
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
            raise ValueError("Residual G learner requires exactly k1 and k2 checkpoint digests")
        self.k_checkpoint_digests = digests
        self.learning_rate = _positive(learning_rate, "Residual G learning_rate")
        self.confidence_floor = _unit(confidence_floor, "Residual G confidence_floor")
        self.selection_margin = _finite(selection_margin, "Residual G selection_margin")
        if self.selection_margin < 0.0:
            raise ValueError("Residual G selection_margin cannot be negative")
        self.device = torch.device(device)
        self.parent_head = nn.Linear(
            len(G_SELECTION_FEATURE_NAMES), 1, bias=True, device=self.device
        )
        self.delta_head = nn.Linear(
            len(G_SELECTION_FEATURE_NAMES), 1, bias=True, device=self.device
        )
        weights = tuple(_finite(value, "parent weight") for value in parent_weight)
        if len(weights) != len(G_SELECTION_FEATURE_NAMES):
            raise ValueError("parent weight length mismatch")
        bias = _finite(parent_bias, "parent bias")
        with torch.no_grad():
            self.parent_head.weight.copy_(
                torch.tensor(weights, dtype=torch.float32, device=self.device).reshape(1, -1)
            )
            self.parent_head.bias.fill_(bias)
            self.delta_head.weight.zero_()
            self.delta_head.bias.zero_()
        freeze_parameters(self.parent_head)
        freeze_parameters(self.delta_head)
        self.training_steps = 0
        self.revision = 0
        self.last_train_digest = ""

    @classmethod
    def from_parent_learner(
        cls,
        parent: Any,
        *,
        device: torch.device | str = "cpu",
    ) -> ResidualGSelectionLearner:
        """Inherit the frozen parent head from a 13-parameter parent."""
        if int(parent.parameter_count) != len(G_SELECTION_FEATURE_NAMES) + 1:
            raise ValueError(
                "Residual G inheritance requires the 13-parameter GSelectionLearner parent"
            )
        with torch.no_grad():
            parent_weight = [
                float(value) for value in parent.model.weight.detach().cpu().reshape(-1)
            ]
            parent_bias = float(parent.model.bias.detach().cpu().reshape(()))
        return cls(
            parent_manifest_digest=parent.parent_manifest_digest,
            k_checkpoint_digests=parent.k_checkpoint_digests,
            parent_weight=parent_weight,
            parent_bias=parent_bias,
            learning_rate=parent.learning_rate,
            confidence_floor=parent.confidence_floor,
            selection_margin=parent.selection_margin,
            device=device,
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for head in (self.parent_head, self.delta_head) for parameter in head.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.delta_head.parameters())

    @property
    def frozen_parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parent_head.parameters())

    @property
    def parent_head_state_digest(self) -> str:
        return content_digest(
            {
                "weight": self.parent_head.weight.detach().cpu(),
                "bias": self.parent_head.bias.detach().cpu(),
            }
        )

    @property
    def model_state_digest(self) -> str:
        return content_digest(
            {
                "parent_weight": self.parent_head.weight.detach().cpu(),
                "parent_bias": self.parent_head.bias.detach().cpu(),
                "delta_weight": self.delta_head.weight.detach().cpu(),
                "delta_bias": self.delta_head.bias.detach().cpu(),
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
            raise ValueError("Residual G learner parent manifest lineage mismatch")
        if self.k_checkpoint_digests != expected_k:
            raise ValueError("Residual G learner K checkpoint lineage mismatch")

    def _inputs(self, candidates: Sequence[GSelectionCandidate]) -> torch.Tensor:
        matrix = torch.tensor(
            [candidate.feature_vector for candidate in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        if matrix.shape[1] != len(G_SELECTION_FEATURE_NAMES):
            raise ValueError("Residual G learner feature manifest length drifted")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("Residual G learner candidate features must be finite")
        return matrix

    def total_score(self, candidate: GSelectionCandidate) -> float:
        if not isinstance(candidate, GSelectionCandidate):
            raise TypeError("Residual G learner scoring requires a GSelectionCandidate")
        with torch.no_grad():
            x = self._inputs([candidate])
            return float((self.parent_head(x) + self.delta_head(x)).reshape(()).item())

    def select(self, candidate_set: GSelectionCandidateSet) -> GSelectionDecision:
        if not isinstance(candidate_set, GSelectionCandidateSet):
            raise TypeError("Residual G learner select requires a GSelectionCandidateSet")
        candidates = tuple(candidate_set.candidates)
        with torch.no_grad():
            scores = (self.parent_head(self._inputs(candidates)) + self.delta_head(self._inputs(candidates))).reshape(-1)
        scored = [(candidate, float(scores[index])) for index, candidate in enumerate(candidates)]
        selected, status = _apply_selection_rule(
            scored,
            confidence_floor=self.confidence_floor,
            selection_margin=self.selection_margin,
        )
        ordered_scores = tuple(
            sorted(((candidate, score) for candidate, score in scored), key=lambda item: item[0].candidate_id)
        )
        return GSelectionDecision.create(
            candidate_set=candidate_set,
            selected=selected,
            selection_status=status,
            candidate_scores=tuple(
                (candidate.candidate_id, score) for candidate, score in ordered_scores
            ),
        )

    def _parent_decision(self, candidate_set: GSelectionCandidateSet) -> tuple[str, str]:
        """The frozen parent head's decision under the same selection rule."""
        candidates = tuple(candidate_set.candidates)
        with torch.no_grad():
            scores = self.parent_head(self._inputs(candidates)).reshape(-1)
        scored = [(candidate, float(scores[index])) for index, candidate in enumerate(candidates)]
        selected, status = _apply_selection_rule(
            scored,
            confidence_floor=self.confidence_floor,
            selection_margin=self.selection_margin,
        )
        return selected.candidate_id, status

    def invariant_hinge(
        self, candidate_set: GSelectionCandidateSet
    ) -> tuple[float, torch.Tensor]:
        """Margin-preservation hinge (P4.8 §3.1): loss and per-candidate error.

        Zero by construction at birth (delta == 0 implies every current
        margin equals its frozen-parent value).  The error entries are
        ``dL/ds(c_i)`` ready for ``apply_linear_delta`` on the delta head.
        """
        if not isinstance(candidate_set, GSelectionCandidateSet):
            raise TypeError("Residual G hinge requires a GSelectionCandidateSet")
        candidates = tuple(candidate_set.candidates)
        inputs = self._inputs(candidates)
        with torch.no_grad():
            parent_scores = self.parent_head(inputs).reshape(-1)
            total = (parent_scores + self.delta_head(inputs).reshape(-1)).clone()
        index_by_id = {candidate.candidate_id: index for index, candidate in enumerate(candidates)}
        parent_selected_id, parent_status = self._parent_decision(candidate_set)
        error = torch.zeros((len(candidates), 1), dtype=torch.float32, device=self.device)
        loss = 0.0
        if parent_status == "selected":
            pi = index_by_id[parent_selected_id]
            others = [index for index in range(len(candidates)) if index != pi]
            parent_gap = float(parent_scores[pi]) - max(float(parent_scores[i]) for i in others)
            current_gap = float(total[pi]) - max(float(total[i]) for i in others)
            if current_gap < parent_gap:
                loss += parent_gap - current_gap
                error[pi, 0] -= 1.0
                competitor = max(others, key=lambda index: float(total[index]))
                error[competitor, 0] += 1.0
            safe_index = index_by_id[_safe_candidate(candidates).candidate_id]
            parent_safe_gap = float(parent_scores[pi]) - float(parent_scores[safe_index])
            current_safe_gap = float(total[pi]) - float(total[safe_index])
            if current_safe_gap < parent_safe_gap:
                loss += parent_safe_gap - current_safe_gap
                error[pi, 0] -= 1.0
                error[safe_index, 0] += 1.0
        else:
            si = index_by_id[parent_selected_id]
            for index, candidate in enumerate(candidates):
                if candidate.candidate_role != "proposal":
                    continue
                parent_encroachment = float(parent_scores[index]) - float(parent_scores[si])
                current_encroachment = float(total[index]) - float(total[si])
                if current_encroachment > parent_encroachment:
                    loss += current_encroachment - parent_encroachment
                    error[index, 0] += 1.0
                    error[si, 0] -= 1.0
        return loss, error

    def invariant_fit(
        self,
        records: Sequence[Mapping[str, Any]],
        constraint_sets: Sequence[GSelectionCandidateSet],
        *,
        epochs: int = 1,
        learning_rate: float = 0.15,
        order_seed: int = 0,
        constraint_digest: str = "",
    ) -> dict[str, Any]:
        """Interleaved task fit + decision-invariant hinge, delta head only.

        The task step fits the total score to one-hot targets; the
        constraint step applies the margin-preservation hinge error to the
        delta head.  The frozen parent head is never updated.
        """
        items = tuple(record["candidate_set"] for record in records)
        if not items or any(item.split != "train" for item in items):
            raise ValueError("Residual G fit requires non-empty train records")
        if not constraint_sets:
            raise ValueError("Residual G fit requires an independent constraint cohort")
        if len({item.candidate_set_digest for item in items}) != len(items):
            raise ValueError("Residual G fit candidate sets must be unique")
        dataset_digest = content_digest(
            {
                "candidate_set_digests": [item.candidate_set_digest for item in items],
                "constraint_digest": constraint_digest,
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "order_seed": int(order_seed),
            }
        )
        task_loss_sum = 0.0
        hinge_loss_sum = 0.0
        hinge_active_steps = 0
        constraint_steps = 0
        for epoch in range(int(epochs)):
            order = list(range(len(items)))
            random.Random(int(order_seed) + epoch).shuffle(order)
            for step, index in enumerate(order):
                item = items[index]
                inputs = self._inputs(item.candidates)
                targets = torch.zeros(
                    (len(item.candidates), 1), dtype=torch.float32, device=self.device
                )
                target_index = next(
                    position
                    for position, candidate in enumerate(item.candidates)
                    if candidate.candidate_id == item.target_candidate_id
                )
                targets[target_index, 0] = 1.0
                with torch.no_grad():
                    predictions = self.parent_head(inputs) + self.delta_head(inputs)
                task_loss_sum += float(torch.mean((predictions - targets) ** 2).item())
                apply_linear_delta(
                    self.delta_head,
                    inputs,
                    mean_squared_error_delta(predictions, targets),
                    float(learning_rate),
                )
                self.training_steps += 1
                group = constraint_sets[(epoch * len(order) + step) % len(constraint_sets)]
                hinge_loss, hinge_error = self.invariant_hinge(group)
                hinge_loss_sum += hinge_loss
                hinge_active_steps += int(hinge_loss > 0.0)
                apply_linear_delta(
                    self.delta_head,
                    self._inputs(tuple(group.candidates)),
                    hinge_error,
                    float(learning_rate),
                )
                constraint_steps += 1
                self.training_steps += 1
        self.revision += 1
        self.last_train_digest = dataset_digest
        return {
            "dataset_digest": dataset_digest,
            "candidate_sets": len(items),
            "constraint_groups": len(constraint_sets),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "order_seed": int(order_seed),
            "training_steps": self.training_steps,
            "revision": self.revision,
            "task_loss_mean": task_loss_sum / (len(items) * int(epochs)),
            "hinge_loss_mean": (
                hinge_loss_sum / constraint_steps if constraint_steps else 0.0
            ),
            "hinge_active_steps": hinge_active_steps,
            "constraint_steps": constraint_steps,
        }

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": RESIDUAL_G_LEARNER_FORMAT,
            "version": RESIDUAL_G_LEARNER_VERSION,
            "feature_names": list(G_SELECTION_FEATURE_NAMES),
            "parent_manifest_digest": self.parent_manifest_digest,
            "k_checkpoint_digests": dict(self.k_checkpoint_digests),
            "learning_rate": self.learning_rate,
            "confidence_floor": self.confidence_floor,
            "selection_margin": self.selection_margin,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "frozen_parameter_count": self.frozen_parameter_count,
            "parent_weight": self.parent_head.weight.detach().cpu().clone(),
            "parent_bias": self.parent_head.bias.detach().cpu().clone(),
            "delta_weight": self.delta_head.weight.detach().cpu().clone(),
            "delta_bias": self.delta_head.bias.detach().cpu().clone(),
            "parent_head_state_digest": self.parent_head_state_digest,
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
    ) -> ResidualGSelectionLearner:
        if payload.get("format") != RESIDUAL_G_LEARNER_FORMAT:
            raise ValueError("unsupported Residual G learner checkpoint format")
        if int(payload.get("version", -1)) != RESIDUAL_G_LEARNER_VERSION:
            raise ValueError("unsupported Residual G learner checkpoint version")
        expected_digest = str(payload["checkpoint_digest"])
        unsigned = {
            key: value for key, value in payload.items() if key != "checkpoint_digest"
        }
        if content_digest(unsigned) != expected_digest:
            raise ValueError("Residual G learner checkpoint digest mismatch")
        if tuple(payload.get("feature_names", ())) != G_SELECTION_FEATURE_NAMES:
            raise ValueError("Residual G learner feature manifest drifted")
        for key in ("parent_weight", "parent_bias", "delta_weight", "delta_bias"):
            if not isinstance(payload.get(key), torch.Tensor):
                raise TypeError("Residual G learner checkpoint weights must be tensors")
        parent_weight = payload["parent_weight"]
        delta_weight = payload["delta_weight"]
        if tuple(parent_weight.shape) != (1, len(G_SELECTION_FEATURE_NAMES)) or tuple(
            delta_weight.shape
        ) != (1, len(G_SELECTION_FEATURE_NAMES)):
            raise ValueError("Residual G learner checkpoint tensor shapes drifted")
        if content_digest(
            {
                "parent_weight": parent_weight,
                "parent_bias": payload["parent_bias"],
                "delta_weight": delta_weight,
                "delta_bias": payload["delta_bias"],
            }
        ) != str(payload["model_state_digest"]):
            raise ValueError("Residual G learner model state digest mismatch")
        if content_digest(
            {"weight": parent_weight, "bias": payload["parent_bias"]}
        ) != str(payload["parent_head_state_digest"]):
            raise ValueError("Residual G learner frozen parent head digest mismatch")
        learner = cls(
            parent_manifest_digest=str(payload["parent_manifest_digest"]),
            k_checkpoint_digests=dict(payload["k_checkpoint_digests"]),
            parent_weight=[float(value) for value in parent_weight.reshape(-1)],
            parent_bias=float(payload["parent_bias"].reshape(())),
            learning_rate=float(payload["learning_rate"]),
            confidence_floor=float(payload["confidence_floor"]),
            selection_margin=float(payload["selection_margin"]),
            device=device,
        )
        if int(payload["parameter_count"]) != learner.parameter_count:
            raise ValueError("Residual G learner parameter count drifted")
        with torch.no_grad():
            learner.delta_head.weight.copy_(
                delta_weight.detach().to(device=learner.device, dtype=torch.float32)
            )
            learner.delta_head.bias.copy_(
                payload["delta_bias"].detach().to(device=learner.device, dtype=torch.float32)
            )
        learner.training_steps = int(payload.get("training_steps", 0))
        learner.revision = int(payload.get("revision", 0))
        learner.last_train_digest = str(payload.get("last_train_digest", ""))
        freeze_parameters(learner.parent_head)
        freeze_parameters(learner.delta_head)
        return learner


__all__ = [
    "RESIDUAL_G_LEARNER_FORMAT",
    "RESIDUAL_G_LEARNER_VERSION",
    "ResidualGSelectionLearner",
]

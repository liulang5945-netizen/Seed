"""Extended G selection learner for the P4.9 feature factorization test.

The P4.9 probe measured that the joint constraint system (new-task
targets win AND parent decisions preserved) is infeasible over the base
12-dimensional candidate features but exactly feasible over a 16-
dimensional extended space with four frozen-parent-relative margin
features.  This learner is the extended representation: a single
``nn.Linear(16, 1, bias=True)`` head whose base-12 weights and bias are
inherited bit-exact from a 13-parameter ``GSelectionLearner`` parent and
whose four factorization dimensions start at zero, so the birth score
equals the parent's score exactly.

The four parent-relative features (``parent_argmax_margin``,
``parent_safe_margin``, ``parent_rank_norm``, ``is_parent_pick``) are
computed from an internal frozen copy of the parent head.  That copy is
digest-guarded and never trained, so the extended features are
non-drifting environment constants throughout adaptation.  The
preservation constraint is the canonical margin-preservation hinge with
the frozen parent's decisions and margins as reference.

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
from .g_selection_residual import (
    _apply_selection_rule,
    _safe_candidate,
    margin_preservation_hinge,
)
from .internalization import content_digest
from .local_learning import apply_linear_delta, freeze_parameters, mean_squared_error_delta

EXTENDED_FEATURE_NAMES = (
    "parent_argmax_margin",
    "parent_safe_margin",
    "parent_rank_norm",
    "is_parent_pick",
)
EXTENDED_G_LEARNER_FORMAT = "taiji-extended-g-selection-learner-v1"
EXTENDED_G_LEARNER_VERSION = 1
_TOTAL_FEATURE_NAMES = (*G_SELECTION_FEATURE_NAMES, *EXTENDED_FEATURE_NAMES)
_ROLE_PRIORITY = {"abstain": 2, "reobserve": 1, "proposal": 0}


class ExtendedGSelectionLearner:
    """16-feature G head with inherited base weights and zero-init
    factorization dimensions."""

    def __init__(
        self,
        *,
        parent_manifest_digest: str,
        k_checkpoint_digests: Mapping[str, str],
        base_weight: Sequence[float],
        base_bias: float,
        learning_rate: float = 0.25,
        confidence_floor: float = 0.55,
        selection_margin: float = 0.05,
        device: torch.device | str = "cpu",
    ) -> None:
        self.parent_manifest_digest = _digest(str(parent_manifest_digest), "parent_manifest_digest")
        digests = {
            str(key): _digest(str(value), f"k_checkpoint_digests[{key}]")
            for key, value in k_checkpoint_digests.items()
        }
        if tuple(sorted(digests)) != ("k1", "k2"):
            raise ValueError("Extended G learner requires exactly k1 and k2 checkpoint digests")
        self.k_checkpoint_digests = digests
        self.learning_rate = _positive(learning_rate, "Extended G learning_rate")
        self.confidence_floor = _unit(confidence_floor, "Extended G confidence_floor")
        self.selection_margin = _finite(selection_margin, "Extended G selection_margin")
        if self.selection_margin < 0.0:
            raise ValueError("Extended G selection_margin cannot be negative")
        self.device = torch.device(device)
        # Frozen feature source: a bit-exact copy of the parent head.  It
        # computes the non-drifting parent-relative features; it is never
        # trained and its digest is part of the checkpoint contract.
        self.feature_source = nn.Linear(
            len(G_SELECTION_FEATURE_NAMES), 1, bias=True, device=self.device
        )
        self.head = nn.Linear(len(_TOTAL_FEATURE_NAMES), 1, bias=True, device=self.device)
        weights = tuple(_finite(value, "base weight") for value in base_weight)
        if len(weights) != len(G_SELECTION_FEATURE_NAMES):
            raise ValueError("base weight length mismatch")
        bias = _finite(base_bias, "base bias")
        with torch.no_grad():
            self.feature_source.weight.copy_(
                torch.tensor(weights, dtype=torch.float32, device=self.device).reshape(1, -1)
            )
            self.feature_source.bias.fill_(bias)
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[0, : len(G_SELECTION_FEATURE_NAMES)] = self.feature_source.weight[0]
            self.head.bias[0] = self.feature_source.bias[0]
        freeze_parameters(self.feature_source)
        freeze_parameters(self.head)
        self.training_steps = 0
        self.revision = 0
        self.last_train_digest = ""

    @classmethod
    def from_parent_learner(
        cls,
        parent: Any,
        *,
        device: torch.device | str = "cpu",
    ) -> ExtendedGSelectionLearner:
        """Inherit base weights/bias from a 13-parameter parent."""
        if int(parent.parameter_count) != len(G_SELECTION_FEATURE_NAMES) + 1:
            raise ValueError(
                "Extended G inheritance requires the 13-parameter GSelectionLearner parent"
            )
        with torch.no_grad():
            base_weight = [float(value) for value in parent.model.weight.detach().cpu().reshape(-1)]
            base_bias = float(parent.model.bias.detach().cpu().reshape(()))
        return cls(
            parent_manifest_digest=parent.parent_manifest_digest,
            k_checkpoint_digests=parent.k_checkpoint_digests,
            base_weight=base_weight,
            base_bias=base_bias,
            learning_rate=parent.learning_rate,
            confidence_floor=parent.confidence_floor,
            selection_margin=parent.selection_margin,
            device=device,
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.head.parameters())

    @property
    def feature_source_state_digest(self) -> str:
        return content_digest(
            {
                "weight": self.feature_source.weight.detach().cpu(),
                "bias": self.feature_source.bias.detach().cpu(),
            }
        )

    @property
    def model_state_digest(self) -> str:
        return content_digest(
            {
                "weight": self.head.weight.detach().cpu(),
                "bias": self.head.bias.detach().cpu(),
            }
        )

    def assert_lineage(
        self,
        *,
        parent_manifest_digest: str,
        k_checkpoint_digests: Mapping[str, str],
    ) -> None:
        expected_parent = _digest(str(parent_manifest_digest), "expected parent_manifest_digest")
        expected_k = {
            str(key): _digest(str(value), f"expected k_checkpoint_digests[{key}]")
            for key, value in k_checkpoint_digests.items()
        }
        if self.parent_manifest_digest != expected_parent:
            raise ValueError("Extended G learner parent manifest lineage mismatch")
        if self.k_checkpoint_digests != expected_k:
            raise ValueError("Extended G learner K checkpoint lineage mismatch")

    def _base_inputs(self, candidates: Sequence[GSelectionCandidate]) -> torch.Tensor:
        matrix = torch.tensor(
            [candidate.feature_vector for candidate in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        if matrix.shape[1] != len(G_SELECTION_FEATURE_NAMES):
            raise ValueError("Extended G learner base feature manifest drifted")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("Extended G learner candidate features must be finite")
        return matrix

    def relative_features(
        self, candidate_set: GSelectionCandidateSet
    ) -> dict[str, tuple[float, ...]]:
        """Non-drifting per-candidate features: 12 base + 4 parent-relative."""
        if not isinstance(candidate_set, GSelectionCandidateSet):
            raise TypeError("Extended G features require a GSelectionCandidateSet")
        candidates = tuple(candidate_set.candidates)
        with torch.no_grad():
            parent_scores_t = self.feature_source(self._base_inputs(candidates)).reshape(-1)
        parent_scores = {
            candidate.candidate_id: float(parent_scores_t[index])
            for index, candidate in enumerate(candidates)
        }
        safe = _safe_candidate(candidates)
        ordered = sorted(
            candidates,
            key=lambda item: (
                -parent_scores[item.candidate_id],
                -_ROLE_PRIORITY[item.candidate_role],
                item.candidate_id,
            ),
        )
        rank_by_id = {candidate.candidate_id: index for index, candidate in enumerate(ordered)}
        pick_id = ordered[0].candidate_id
        features: dict[str, tuple[float, ...]] = {}
        for candidate in candidates:
            cid = candidate.candidate_id
            others = [item for item in candidates if item.candidate_id != cid]
            features[cid] = (
                *candidate.feature_vector,
                parent_scores[cid] - max(parent_scores[item.candidate_id] for item in others),
                parent_scores[cid] - parent_scores[safe.candidate_id],
                1.0 - rank_by_id[cid] / max(1, len(candidates) - 1),
                1.0 if cid == pick_id else 0.0,
            )
        return features

    def total_scores(self, candidate_set: GSelectionCandidateSet) -> dict[str, float]:
        if not isinstance(candidate_set, GSelectionCandidateSet):
            raise TypeError("Extended G scoring requires a GSelectionCandidateSet")
        features = self.relative_features(candidate_set)
        candidates = tuple(candidate_set.candidates)
        matrix = torch.tensor(
            [features[candidate.candidate_id] for candidate in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("Extended G learner augmented features must be finite")
        with torch.no_grad():
            scores = self.head(matrix).reshape(-1)
        return {
            candidate.candidate_id: float(scores[index])
            for index, candidate in enumerate(candidates)
        }

    def select(self, candidate_set: GSelectionCandidateSet) -> GSelectionDecision:
        if not isinstance(candidate_set, GSelectionCandidateSet):
            raise TypeError("Extended G learner select requires a GSelectionCandidateSet")
        scores = self.total_scores(candidate_set)
        scored = [
            (candidate, scores[candidate.candidate_id]) for candidate in candidate_set.candidates
        ]
        selected, status = _apply_selection_rule(
            scored,
            confidence_floor=self.confidence_floor,
            selection_margin=self.selection_margin,
        )
        ordered_scores = tuple(sorted(scored, key=lambda item: item[0].candidate_id))
        return GSelectionDecision.create(
            candidate_set=candidate_set,
            selected=selected,
            selection_status=status,
            candidate_scores=tuple(
                (candidate.candidate_id, score) for candidate, score in ordered_scores
            ),
        )

    def _reference_decision(self, candidate_set: GSelectionCandidateSet) -> tuple[str, str]:
        """The frozen feature source's decision under the same rule."""
        candidates = tuple(candidate_set.candidates)
        with torch.no_grad():
            scores = self.feature_source(self._base_inputs(candidates)).reshape(-1)
        scored = [(candidate, float(scores[index])) for index, candidate in enumerate(candidates)]
        selected, status = _apply_selection_rule(
            scored,
            confidence_floor=self.confidence_floor,
            selection_margin=self.selection_margin,
        )
        return selected.candidate_id, status

    def invariant_hinge(
        self, candidate_set: GSelectionCandidateSet
    ) -> tuple[float, dict[str, float]]:
        """Margin-preservation hinge against the frozen feature source."""
        if not isinstance(candidate_set, GSelectionCandidateSet):
            raise TypeError("Extended G hinge requires a GSelectionCandidateSet")
        reference_selected_id, reference_status = self._reference_decision(candidate_set)
        with torch.no_grad():
            reference_matrix = self._base_inputs(tuple(candidate_set.candidates))
            reference_scores_t = self.feature_source(reference_matrix).reshape(-1)
        reference_scores = {
            candidate.candidate_id: float(reference_scores_t[index])
            for index, candidate in enumerate(candidate_set.candidates)
        }
        return margin_preservation_hinge(
            candidate_set,
            current_scores=self.total_scores(candidate_set),
            reference_scores=reference_scores,
            reference_selected_id=reference_selected_id,
            reference_status=reference_status,
            selection_margin=self.selection_margin,
        )

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
        """Interleaved task fit + margin-preservation hinge on the head."""
        items = tuple(record["candidate_set"] for record in records)
        if not items or any(item.split != "train" for item in items):
            raise ValueError("Extended G fit requires non-empty train records")
        if not constraint_sets:
            raise ValueError("Extended G fit requires an independent constraint cohort")
        if len({item.candidate_set_digest for item in items}) != len(items):
            raise ValueError("Extended G fit candidate sets must be unique")
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
            for _step, index in enumerate(order):
                item = items[index]
                candidates = tuple(item.candidates)
                features = self.relative_features(item)
                inputs = torch.tensor(
                    [features[candidate.candidate_id] for candidate in candidates],
                    dtype=torch.float32,
                    device=self.device,
                )
                if not bool(torch.isfinite(inputs).all()):
                    raise ValueError("Extended G fit features must be finite")
                targets = torch.zeros((len(candidates), 1), dtype=torch.float32, device=self.device)
                target_index = next(
                    position
                    for position, candidate in enumerate(candidates)
                    if candidate.candidate_id == item.target_candidate_id
                )
                targets[target_index, 0] = 1.0
                with torch.no_grad():
                    predictions = self.head(inputs)
                task_loss_sum += float(torch.mean((predictions - targets) ** 2).item())
                apply_linear_delta(
                    self.head,
                    inputs,
                    mean_squared_error_delta(predictions, targets),
                    float(learning_rate),
                )
                self.training_steps += 1
                group = constraint_sets[(epoch * len(order) + _step) % len(constraint_sets)]
                hinge_loss, error_by_id = self.invariant_hinge(group)
                hinge_loss_sum += hinge_loss
                hinge_active_steps += int(hinge_loss > 0.0)
                group_features = self.relative_features(group)
                group_inputs = torch.tensor(
                    [group_features[candidate.candidate_id] for candidate in group.candidates],
                    dtype=torch.float32,
                    device=self.device,
                )
                hinge_error = torch.tensor(
                    [[error_by_id[candidate.candidate_id]] for candidate in group.candidates],
                    dtype=torch.float32,
                    device=self.device,
                )
                apply_linear_delta(self.head, group_inputs, hinge_error, float(learning_rate))
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
            "hinge_loss_mean": (hinge_loss_sum / constraint_steps if constraint_steps else 0.0),
            "hinge_active_steps": hinge_active_steps,
            "constraint_steps": constraint_steps,
        }

    def apply_projected_weights(self, weight: Sequence[float], *, projection_digest: str) -> None:
        """Apply solver-projected weights to the head (P4.11 mechanism).

        The projection acts on the 16 weight dimensions only; the bias is
        untouched by construction (all constraints are score differences).
        The feature source is never touched.
        """
        projected = tuple(_finite(value, "projected weight") for value in weight)
        if len(projected) != len(_TOTAL_FEATURE_NAMES):
            raise ValueError("projected weight length mismatch")
        with torch.no_grad():
            self.head.weight.copy_(
                torch.tensor(projected, dtype=torch.float32, device=self.device).reshape(1, -1)
            )
        self.revision += 1
        self.last_train_digest = projection_digest

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": EXTENDED_G_LEARNER_FORMAT,
            "version": EXTENDED_G_LEARNER_VERSION,
            "base_feature_names": list(G_SELECTION_FEATURE_NAMES),
            "extended_feature_names": list(EXTENDED_FEATURE_NAMES),
            "parent_manifest_digest": self.parent_manifest_digest,
            "k_checkpoint_digests": dict(self.k_checkpoint_digests),
            "learning_rate": self.learning_rate,
            "confidence_floor": self.confidence_floor,
            "selection_margin": self.selection_margin,
            "parameter_count": self.parameter_count,
            "weight": self.head.weight.detach().cpu().clone(),
            "bias": self.head.bias.detach().cpu().clone(),
            "feature_source_weight": self.feature_source.weight.detach().cpu().clone(),
            "feature_source_bias": self.feature_source.bias.detach().cpu().clone(),
            "feature_source_state_digest": self.feature_source_state_digest,
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
    ) -> ExtendedGSelectionLearner:
        if payload.get("format") != EXTENDED_G_LEARNER_FORMAT:
            raise ValueError("unsupported Extended G learner checkpoint format")
        if int(payload.get("version", -1)) != EXTENDED_G_LEARNER_VERSION:
            raise ValueError("unsupported Extended G learner checkpoint version")
        expected_digest = str(payload["checkpoint_digest"])
        unsigned = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        if content_digest(unsigned) != expected_digest:
            raise ValueError("Extended G learner checkpoint digest mismatch")
        if tuple(payload.get("base_feature_names", ())) != G_SELECTION_FEATURE_NAMES:
            raise ValueError("Extended G learner base feature manifest drifted")
        if tuple(payload.get("extended_feature_names", ())) != EXTENDED_FEATURE_NAMES:
            raise ValueError("Extended G learner extended feature manifest drifted")
        weight = payload["weight"]
        bias = payload["bias"]
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise TypeError("Extended G learner checkpoint weights must be tensors")
        feature_source_weight = payload.get("feature_source_weight")
        feature_source_bias = payload.get("feature_source_bias")
        if not isinstance(feature_source_weight, torch.Tensor) or not isinstance(
            feature_source_bias, torch.Tensor
        ):
            raise TypeError("Extended G learner feature source must be stored tensors")
        if tuple(weight.shape) != (1, len(_TOTAL_FEATURE_NAMES)) or tuple(bias.shape) != (1,):
            raise ValueError("Extended G learner checkpoint tensor shapes drifted")
        if tuple(feature_source_weight.shape) != (1, len(G_SELECTION_FEATURE_NAMES)) or tuple(
            feature_source_bias.shape
        ) != (1,):
            raise ValueError("Extended G learner feature source shapes drifted")
        if content_digest({"weight": weight, "bias": bias}) != str(payload["model_state_digest"]):
            raise ValueError("Extended G learner model state digest mismatch")
        if content_digest(
            {
                "weight": feature_source_weight,
                "bias": feature_source_bias,
            }
        ) != str(payload["feature_source_state_digest"]):
            raise ValueError("Extended G learner frozen feature source digest mismatch")
        learner = cls(
            parent_manifest_digest=str(payload["parent_manifest_digest"]),
            k_checkpoint_digests=dict(payload["k_checkpoint_digests"]),
            base_weight=[float(value) for value in feature_source_weight.reshape(-1)],
            base_bias=float(feature_source_bias.reshape(())),
            learning_rate=float(payload["learning_rate"]),
            confidence_floor=float(payload["confidence_floor"]),
            selection_margin=float(payload["selection_margin"]),
            device=device,
        )
        if int(payload["parameter_count"]) != learner.parameter_count:
            raise ValueError("Extended G learner parameter count drifted")
        # The feature source is restored from its own stored tensors and
        # verified against the recorded digest; the head's base dims may
        # drift freely during adaptation without affecting the
        # non-drifting parent-relative features.
        with torch.no_grad():
            learner.head.weight.copy_(
                weight.detach().to(device=learner.device, dtype=torch.float32)
            )
            learner.head.bias.copy_(bias.detach().to(device=learner.device, dtype=torch.float32))
        learner.training_steps = int(payload.get("training_steps", 0))
        learner.revision = int(payload.get("revision", 0))
        learner.last_train_digest = str(payload.get("last_train_digest", ""))
        freeze_parameters(learner.feature_source)
        freeze_parameters(learner.head)
        return learner


__all__ = [
    "EXTENDED_FEATURE_NAMES",
    "EXTENDED_G_LEARNER_FORMAT",
    "EXTENDED_G_LEARNER_VERSION",
    "ExtendedGSelectionLearner",
]

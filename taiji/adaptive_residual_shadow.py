"""Independent shadow materialization for Taiji R4 residual candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .adaptive_residual_bridge import (
    ADAPTIVE_RESIDUAL_BRIDGE_FORMAT,
    ADAPTIVE_RESIDUAL_BRIDGE_VERSION,
)
from .adaptive_residual_candidate import AdaptiveResidualGrowthCandidate
from .config import TaijiConfig
from .internalization import content_digest
from .neuron_region import AdaptiveNeuronRegion
from .sparse import SparseSynapses, bound_norm

ADAPTIVE_RESIDUAL_SHADOW_FORMAT = "taiji-adaptive-residual-shadow-v1"
ADAPTIVE_RESIDUAL_SHADOW_VERSION = 1


class AdaptiveResidualShadow:
    """A gate-closed, independently materialized residual population.

    The parent bridge remains untouched.  Existing region rows/state are
    copied byte-for-byte; one candidate row is appended through the native
    ``AdaptiveNeuronRegion`` topology operation.  A sparse projection keeps
    the parent direct-output geometry exact while providing a zero-weight edge
    through which the new unit can later learn during shadow training.
    """

    def __init__(
        self,
        config: TaijiConfig,
        *,
        candidate: AdaptiveResidualGrowthCandidate,
        region: AdaptiveNeuronRegion,
        output_projection: SparseSynapses,
        parent_bridge_digest: str,
        residual_gain: float,
        gate_projection: SparseSynapses | None = None,
        birth_anchor_unit_id: str | None = None,
        birth_anchor_unit_ids: tuple[str, ...] | None = None,
        birth_anchor_weights: tuple[float, ...] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.candidate = candidate
        self.region = region
        self.output_projection = output_projection
        self.gate_projection = gate_projection
        self.parent_bridge_digest = str(parent_bridge_digest).strip()
        self.residual_gain = float(residual_gain)
        self._validate_gain(self.residual_gain)
        if not self.parent_bridge_digest:
            raise ValueError("adaptive residual shadow parent bridge digest must not be empty")
        if self.region.region_id != candidate.bridge_id:
            raise ValueError("adaptive residual shadow region does not match candidate")
        if candidate.unit_id not in self.region.unit_ids:
            raise ValueError("adaptive residual shadow is missing the candidate unit")
        normalized_anchor = (
            "" if birth_anchor_unit_id is None else str(birth_anchor_unit_id).strip()
        )
        normalized_anchors = (
            tuple(str(item).strip() for item in birth_anchor_unit_ids)
            if birth_anchor_unit_ids is not None
            else ((normalized_anchor,) if normalized_anchor else ())
        )
        if any(not item for item in normalized_anchors):
            raise ValueError("adaptive residual shadow birth anchors must not be empty")
        if len(set(normalized_anchors)) != len(normalized_anchors):
            raise ValueError("adaptive residual shadow birth anchors must be unique")
        if len(normalized_anchors) > 2:
            raise ValueError("adaptive residual shadow supports at most two birth anchors")
        for anchor in normalized_anchors:
            anchor_index = self.region.unit_index(anchor)
            if anchor_index >= int(candidate.parent_unit_count):
                raise ValueError("adaptive residual shadow birth anchor must be a parent unit")
        normalized_weights = (
            tuple(float(value) for value in birth_anchor_weights)
            if birth_anchor_weights is not None
            else ((1.0,) if normalized_anchors else ())
        )
        if len(normalized_weights) != len(normalized_anchors):
            raise ValueError("adaptive residual shadow birth anchor weights do not match anchors")
        if normalized_weights and (
            any(not math.isfinite(value) or value <= 0.0 for value in normalized_weights)
            or not math.isclose(sum(normalized_weights), 1.0, rel_tol=1e-5, abs_tol=1e-5)
        ):
            raise ValueError("adaptive residual shadow birth anchor weights must be normalized")
        self._birth_anchor_unit_ids = normalized_anchors
        self._birth_anchor_weights = normalized_weights
        if self.region.unit_count != candidate.proposed_unit_count:
            raise ValueError("adaptive residual shadow unit count does not match candidate")
        if output_projection.out_features != self.output_dim:
            raise ValueError("adaptive residual shadow projection output dimension mismatch")
        if output_projection.in_features != self.unit_count:
            raise ValueError("adaptive residual shadow projection input dimension mismatch")
        if gate_projection is not None:
            if gate_projection.out_features != 1:
                raise ValueError("adaptive residual shadow gate must have one output")
            if gate_projection.in_features not in {
                int(candidate.parent_unit_count),
                2 * int(candidate.parent_unit_count),
                2 * int(candidate.parent_unit_count) + int(self.config.motor_context_dim),
            }:
                raise ValueError("adaptive residual shadow gate input dimension mismatch")
        gate_input_dim = (
            int(candidate.parent_unit_count)
            if gate_projection is None
            else int(gate_projection.in_features)
        )
        self._gate_input_dim = gate_input_dim
        self._gate = 0.0
        self._lesioned = False
        self._last_input = torch.zeros(self.output_dim, device=self.device)
        self._last_activity = torch.zeros(self.unit_count, device=self.device)
        self._last_gate_input = torch.zeros(gate_input_dim, device=self.device)
        self._last_parent_context = torch.zeros(self.output_dim, device=self.device)
        self._last_counterfactual_parent_probabilities = torch.zeros(
            self.config.alphabet_size,
            device=self.device,
        )
        self._counterfactual_parent_ready = False
        self._last_candidate_residual = torch.zeros(self.output_dim, device=self.device)
        self._last_candidate_activity = 0.0
        self._last_candidate_eligibility_norm = 0.0
        self._last_parent_residual_norm = 0.0
        self._last_candidate_residual_norm = 0.0
        self._last_candidate_credit_norm = 0.0
        self._last_candidate_projection_update_norm = 0.0
        self._last_candidate_utility = 0.0
        self._last_counterfactual_utility = 0.0
        self._counterfactual_utility_ready = False
        self._candidate_gate_bias = torch.zeros((), device=self.device)
        self._candidate_utility_baseline = 0.0
        self._last_candidate_gate = 1.0 if gate_projection is None else 0.5

    @staticmethod
    def _validate_gain(value: float) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("adaptive residual shadow residual_gain must be finite and positive")

    @property
    def bridge_id(self) -> str:
        return self.candidate.bridge_id

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def birth_anchor_unit_id(self) -> str | None:
        return self._birth_anchor_unit_ids[0] if self._birth_anchor_unit_ids else None

    @property
    def birth_anchor_unit_ids(self) -> tuple[str, ...]:
        return self._birth_anchor_unit_ids

    @property
    def birth_anchor_weights(self) -> tuple[float, ...]:
        return self._birth_anchor_weights

    @property
    def output_dim(self) -> int:
        return int(self.config.motor_context_dim)

    @property
    def unit_count(self) -> int:
        return self.region.unit_count

    @property
    def gate(self) -> float:
        return self._gate

    @property
    def lesioned(self) -> bool:
        return self._lesioned

    @property
    def edge_count(self) -> int:
        return self.region.edge_count + self.output_projection.edge_count + (
            0 if self.gate_projection is None else self.gate_projection.edge_count
        )

    @property
    def candidate_activity(self) -> float:
        return self._last_candidate_activity

    @property
    def candidate_residual_norm(self) -> float:
        return self._last_candidate_residual_norm

    @property
    def candidate_eligibility_norm(self) -> float:
        return self._last_candidate_eligibility_norm

    @property
    def parent_residual_norm(self) -> float:
        return self._last_parent_residual_norm

    @property
    def candidate_credit_norm(self) -> float:
        return self._last_candidate_credit_norm

    @property
    def candidate_projection_update_norm(self) -> float:
        return self._last_candidate_projection_update_norm

    @property
    def candidate_utility(self) -> float:
        return self._last_candidate_utility

    @property
    def candidate_counterfactual_utility(self) -> float:
        return self._last_counterfactual_utility

    @property
    def candidate_gate(self) -> float:
        return self._last_candidate_gate

    @property
    def diagnostics(self) -> dict[str, float]:
        return {
            "candidate_activity": self.candidate_activity,
            "candidate_eligibility_norm": self.candidate_eligibility_norm,
            "parent_residual_norm": self.parent_residual_norm,
            "candidate_residual_norm": self.candidate_residual_norm,
            "candidate_credit_norm": self.candidate_credit_norm,
            "candidate_projection_update_norm": self.candidate_projection_update_norm,
            "candidate_utility": self.candidate_utility,
            "candidate_counterfactual_utility": self.candidate_counterfactual_utility,
            "candidate_gate": self.candidate_gate,
        }

    @property
    def last_activity(self) -> torch.Tensor:
        return self._last_activity.detach().clone()

    @property
    def last_parent_context(self) -> torch.Tensor:
        """Return the same context with the candidate residual removed."""

        return self._last_parent_context.detach().clone()

    @property
    def counterfactual_parent_probabilities(self) -> torch.Tensor | None:
        """Return the readout surface saved for the most recent candidate output."""

        if not self._counterfactual_parent_ready:
            return None
        return self._last_counterfactual_parent_probabilities.detach().clone()

    @torch.no_grad()
    def set_gate(self, gate: float) -> None:
        value = float(gate)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("adaptive residual shadow gate must be finite and in [0, 1]")
        self._gate = value

    @torch.no_grad()
    def record_counterfactual_parent_probabilities(
        self,
        probabilities: torch.Tensor,
    ) -> None:
        """Save the exact candidate-off readout for the next causal target.

        The model records this immediately after the candidate-on readout while
        the same decoder weights and episodic evidence are still live.  The
        next ``observe`` call supplies the actual target symbol and converts
        the two probabilities into an exact loss delta before any learning
        step mutates the decoder or shadow.
        """

        if probabilities.shape != (self.config.alphabet_size,):
            raise ValueError("adaptive residual shadow counterfactual probability dimension mismatch")
        values = probabilities.detach().to(self.device)
        if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
            raise ValueError("adaptive residual shadow counterfactual probabilities must be finite and non-negative")
        self._last_counterfactual_parent_probabilities.copy_(values)
        self._counterfactual_parent_ready = True

    @torch.no_grad()
    def record_counterfactual_utility(self, value: float) -> None:
        """Record candidate-off loss minus candidate-on loss for one target."""

        utility = float(value)
        if not math.isfinite(utility):
            raise ValueError("adaptive residual shadow counterfactual utility must be finite")
        self._last_counterfactual_utility = utility
        self._counterfactual_utility_ready = True

    @torch.no_grad()
    def lesion(self) -> None:
        self._lesioned = True
        self.reset_dynamics()

    @torch.no_grad()
    def unlesion(self) -> None:
        self._lesioned = False

    @torch.no_grad()
    def lesion_candidate(self) -> None:
        """Remove only the appended unit's output contacts for causal ablation."""

        candidate_index = self.region.unit_index(self.candidate.unit_id)
        self.output_projection.edge_weight[self.output_projection.pre_index == candidate_index] = 0.0

    @staticmethod
    @torch.no_grad()
    def _copy_birth_anchor(
        region: AdaptiveNeuronRegion,
        *,
        source_index: int,
        candidate_index: int,
    ) -> None:
        """Give a pressure-born unit a stable local substrate to specialize."""

        region.incoming.pre_index[candidate_index].copy_(
            region.incoming.pre_index[source_index]
        )
        region.incoming.edge_weight[candidate_index].copy_(
            region.incoming.edge_weight[source_index]
        )
        if region.recurrent is not None:
            region.recurrent.pre_index[candidate_index].copy_(
                region.recurrent.pre_index[source_index]
            )
            region.recurrent.edge_weight[candidate_index].copy_(
                region.recurrent.edge_weight[source_index]
            )
        region.threshold[candidate_index] = region.threshold[source_index]

    @staticmethod
    @torch.no_grad()
    def _mix_birth_anchors(
        region: AdaptiveNeuronRegion,
        *,
        anchor_indices: tuple[int, ...],
        anchor_weights: tuple[float, ...],
        candidate_index: int,
    ) -> None:
        """Mix two parent rows into one bounded sparse candidate row."""

        def mix_row(
            pre_index: torch.Tensor,
            edge_weight: torch.Tensor,
            *,
            input_dim: int,
            exclude_index: int | None = None,
        ) -> None:
            dense = torch.zeros(input_dim, device=edge_weight.device, dtype=edge_weight.dtype)
            for anchor_index, anchor_weight in zip(anchor_indices, anchor_weights):
                dense.scatter_add_(
                    0,
                    pre_index[anchor_index].to(torch.long),
                    edge_weight[anchor_index] * float(anchor_weight),
                )
            ranking = dense.abs()
            if exclude_index is not None:
                ranking[exclude_index] = -1.0
            selected = torch.topk(
                ranking,
                k=edge_weight.shape[1],
                largest=True,
                sorted=True,
            ).indices
            pre_index[candidate_index].copy_(selected.to(pre_index.dtype))
            edge_weight[candidate_index].copy_(dense[selected])
            target_squared_norm = torch.zeros(
                (),
                device=edge_weight.device,
                dtype=edge_weight.dtype,
            )
            for anchor_index, anchor_weight in zip(anchor_indices, anchor_weights):
                target_squared_norm = target_squared_norm + float(anchor_weight) * (
                    edge_weight[anchor_index].pow(2).sum()
                )
            candidate_squared_norm = edge_weight[candidate_index].pow(2).sum()
            if float(target_squared_norm.item()) > 1e-12 and float(
                candidate_squared_norm.item()
            ) > 1e-12:
                edge_weight[candidate_index].copy_(
                    edge_weight[candidate_index]
                    * torch.sqrt(target_squared_norm / candidate_squared_norm)
                )

        mix_row(
            region.incoming.pre_index,
            region.incoming.edge_weight,
            input_dim=region.input_dim,
        )
        if region.recurrent is not None:
            mix_row(
                region.recurrent.pre_index,
                region.recurrent.edge_weight,
                input_dim=region.unit_count,
                exclude_index=candidate_index,
            )
        threshold = torch.zeros((), device=region.device, dtype=region.threshold.dtype)
        for anchor_index, anchor_weight in zip(anchor_indices, anchor_weights):
            threshold = threshold + region.threshold[anchor_index] * float(anchor_weight)
        region.threshold[candidate_index] = threshold

    @torch.no_grad()
    def reset_dynamics(self) -> None:
        self.region.membrane.zero_()
        self.region.activity.zero_()
        self.region.trace.zero_()
        self._last_input.zero_()
        self._last_activity.zero_()
        self._last_gate_input.zero_()
        self._last_parent_context.zero_()
        self._last_counterfactual_parent_probabilities.zero_()
        self._counterfactual_parent_ready = False
        self._last_candidate_residual.zero_()
        self._last_candidate_activity = 0.0
        self._last_candidate_eligibility_norm = 0.0
        self._last_parent_residual_norm = 0.0
        self._last_candidate_residual_norm = 0.0
        self._last_candidate_credit_norm = 0.0
        self._last_candidate_projection_update_norm = 0.0
        self._last_candidate_utility = 0.0
        self._last_counterfactual_utility = 0.0
        self._counterfactual_utility_ready = False
        self._last_candidate_gate = 1.0 if self.gate_projection is None else 0.5

    @torch.no_grad()
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape != (self.output_dim,):
            raise ValueError("adaptive residual shadow context dimension mismatch")
        if self._gate == 0.0 or self._lesioned:
            self._last_parent_context.copy_(context.detach().to(self.device))
            self._counterfactual_parent_ready = False
            return context.clone()
        input_context = context.detach().to(self.device).clone()
        activity = self.region.step(input_context)
        self._last_input.copy_(input_context)
        self._last_activity.copy_(activity)
        candidate_index = self.region.unit_index(self.candidate.unit_id)
        candidate_eligibility = self.region.trace[candidate_index]
        candidate_edges = self.output_projection.pre_index == candidate_index
        candidate_residual = (
            self.output_projection.edge_weight * candidate_edges.to(self.output_projection.edge_weight.dtype)
        ).sum(dim=1) * activity[candidate_index]
        residual = self.output_projection.forward(activity)
        candidate_gate = 1.0
        if self.gate_projection is not None:
            parent_count = int(self.candidate.parent_unit_count)
            parent_trace = self.region.trace[:parent_count]
            if self.gate_projection.in_features == parent_count:
                gate_input = parent_trace
            elif self.gate_projection.in_features == 2 * parent_count:
                gate_input = torch.cat((activity[:parent_count], parent_trace), dim=0)
            else:
                gate_input = torch.cat(
                    (input_context, activity[:parent_count], parent_trace),
                    dim=0,
                )
            self._last_gate_input.copy_(gate_input)
            gate_logit = self._candidate_gate_bias + self.gate_projection.forward(gate_input)[0]
            candidate_gate = float(torch.sigmoid(gate_logit).item())
        self._last_candidate_gate = candidate_gate
        gated_candidate_residual = candidate_residual * candidate_gate
        parent_residual = residual - candidate_residual
        residual = parent_residual + gated_candidate_residual
        self._last_candidate_residual.copy_(gated_candidate_residual)
        self._last_candidate_activity = float(abs(activity[candidate_index]).item())
        self._last_candidate_eligibility_norm = float(abs(candidate_eligibility).item())
        self._last_candidate_residual_norm = float(candidate_residual.norm().item())
        self._last_parent_residual_norm = float(parent_residual.norm().item())
        self._last_parent_context.copy_(
            bound_norm(
                input_context + float(self._gate) * float(self.residual_gain) * parent_residual,
                float(self.config.motor_context_norm),
            )
        )
        return bound_norm(
            input_context + float(self._gate) * float(self.residual_gain) * residual,
            float(self.config.motor_context_norm),
        )

    @torch.no_grad()
    def learn(
        self,
        postsynaptic_error: torch.Tensor,
        *,
        freeze_parent: bool = True,
    ) -> None:
        if postsynaptic_error.shape != (self.output_dim,):
            raise ValueError("adaptive residual shadow credit dimension mismatch")
        if not isinstance(freeze_parent, bool):
            raise TypeError("adaptive residual shadow freeze_parent must be a bool")
        if self._gate == 0.0 or self._lesioned:
            return
        parent_unit_count = int(self.candidate.parent_unit_count)
        parent_incoming = (
            self.region.incoming.edge_weight[:parent_unit_count].detach().clone()
            if freeze_parent
            else None
        )
        parent_recurrent = (
            None
            if self.region.recurrent is None or not freeze_parent
            else self.region.recurrent.edge_weight[:parent_unit_count].detach().clone()
        )
        parent_threshold = (
            self.region.threshold[:parent_unit_count].detach().clone()
            if freeze_parent
            else None
        )
        candidate_index = self.region.unit_index(self.candidate.unit_id)
        parent_projection_mask = self.output_projection.pre_index != candidate_index
        parent_projection = (
            self.output_projection.edge_weight.detach().clone()
            if freeze_parent
            else None
        )
        region_error = self.output_projection.backproject(postsynaptic_error)
        self._last_candidate_credit_norm = float(abs(region_error[candidate_index]).item())
        self._last_candidate_utility = float(
            torch.dot(
                self._last_candidate_residual
                * float(self._gate)
                * float(self.residual_gain),
                postsynaptic_error.to(self.device),
            ).item()
        )
        if self.gate_projection is not None:
            if self._counterfactual_utility_ready:
                # The exact counterfactual is already a loss advantage for
                # this deterministic continuous gate.  Subtracting a global
                # EMA baseline would erase the signed direction that should
                # open or close the candidate in this context.
                gate_utility = self._last_counterfactual_utility
                utility_advantage = float(gate_utility)
            else:
                gate_utility = self._last_candidate_utility
                utility_advantage = float(
                    gate_utility - self._candidate_utility_baseline
                )
            self._candidate_utility_baseline = (
                0.9 * self._candidate_utility_baseline
                + 0.1 * gate_utility
            )
            gate_error = torch.tensor(
                [utility_advantage * float(self._gate) * float(self.residual_gain)],
                device=self.device,
            )
            self.gate_projection.local_update(
                gate_error,
                self._last_gate_input,
                learning_rate=float(self.config.predictive_context_learning_rate),
                weight_decay=float(self.config.synapse_decay),
            )
            self._candidate_gate_bias.add_(
                float(self.config.predictive_context_learning_rate) * utility_advantage
            ).clamp_(-4.0, 4.0)
            self._counterfactual_utility_ready = False
        candidate_projection_before = self.output_projection.edge_weight[
            ~parent_projection_mask
        ].detach().clone()
        projection_trace = self._last_activity.detach().clone()
        # The mature parent keeps its instantaneous trace.  The appended
        # candidate uses its own causal eligibility trace so a brief burst of
        # activity can teach the new projection across subsequent silent
        # ticks; this is the only credit boundary changed by R4 revision.
        projection_trace[candidate_index] = self.region.trace[candidate_index]
        self.output_projection.local_update(
            postsynaptic_error * float(self._gate) * float(self.residual_gain),
            projection_trace,
            learning_rate=float(self.config.predictive_context_learning_rate),
            weight_decay=float(self.config.synapse_decay),
        )
        self.region.learn(
            self._last_input,
            region_error * float(self._gate) * float(self.residual_gain),
        )
        # R4 shadow training is deliberately candidate-only.  The copied
        # parent substrate is a frozen control; only the appended row and the
        # new projection contacts may absorb the causal error.
        if freeze_parent:
            assert parent_incoming is not None
            assert parent_threshold is not None
            assert parent_projection is not None
            self.region.incoming.edge_weight[:parent_unit_count].copy_(parent_incoming)
            if self.region.recurrent is not None and parent_recurrent is not None:
                self.region.recurrent.edge_weight[:parent_unit_count].copy_(parent_recurrent)
            self.region.threshold[:parent_unit_count].copy_(parent_threshold)
            self.output_projection.edge_weight[parent_projection_mask] = parent_projection[
                parent_projection_mask
            ]
        self._last_candidate_projection_update_norm = float(
            (
                self.output_projection.edge_weight[~parent_projection_mask]
                - candidate_projection_before
            ).norm()
            .item()
        )

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [*self.region.parameter_tensors(), self.output_projection.edge_weight]
        if self.gate_projection is not None:
            tensors.extend((self.gate_projection.edge_weight, self._candidate_gate_bias))
        return tuple(tensors)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "format": ADAPTIVE_RESIDUAL_SHADOW_FORMAT,
            "version": ADAPTIVE_RESIDUAL_SHADOW_VERSION,
            "candidate": self.candidate.to_payload(),
            "parent_bridge_digest": self.parent_bridge_digest,
            "config_digest": content_digest(self.config.to_dict()),
            "gate": self._gate,
            "lesioned": self._lesioned,
            "residual_gain": self.residual_gain,
            "region": self.region.to_payload(),
            "output_projection": self.output_projection.to_payload(),
            "last_input": self._last_input.detach().cpu().clone(),
            "last_activity": self._last_activity.detach().cpu().clone(),
            "last_parent_context": self._last_parent_context.detach().cpu().clone(),
            "counterfactual_parent_probabilities": (
                self._last_counterfactual_parent_probabilities.detach().cpu().clone()
            ),
            "counterfactual_parent_ready": self._counterfactual_parent_ready,
            "counterfactual_utility_ready": self._counterfactual_utility_ready,
            "diagnostics": dict(self.diagnostics),
        }
        if self.gate_projection is not None:
            payload["last_gate_input"] = self._last_gate_input.detach().cpu().clone()
            payload["gate_projection"] = self.gate_projection.to_payload()
            payload["candidate_gate_bias"] = self._candidate_gate_bias.detach().cpu().clone()
            payload["candidate_utility_baseline"] = self._candidate_utility_baseline
        if self._birth_anchor_unit_ids:
            payload["birth_anchor_unit_id"] = self._birth_anchor_unit_ids[0]
            if len(self._birth_anchor_unit_ids) > 1:
                payload["birth_anchor_unit_ids"] = list(self._birth_anchor_unit_ids)
                payload["birth_anchor_weights"] = list(self._birth_anchor_weights)
        return {**payload, "shadow_digest": content_digest(payload)}

    @classmethod
    def from_parent_bridge(
        cls,
        config: TaijiConfig,
        parent_bridge_payload: Mapping[str, Any],
        candidate: AdaptiveResidualGrowthCandidate,
        *,
        birth_mode: str = "random",
        device: torch.device | str = "cpu",
    ) -> AdaptiveResidualShadow:
        if parent_bridge_payload.get("format") != ADAPTIVE_RESIDUAL_BRIDGE_FORMAT:
            raise ValueError("adaptive residual shadow parent bridge format is invalid")
        if int(parent_bridge_payload.get("version", -1)) != ADAPTIVE_RESIDUAL_BRIDGE_VERSION:
            raise ValueError("adaptive residual shadow parent bridge version is invalid")
        region_payload = parent_bridge_payload.get("region")
        if not isinstance(region_payload, Mapping):
            raise ValueError("adaptive residual shadow parent region payload is invalid")
        if str(region_payload.get("region_id")) != candidate.bridge_id:
            raise ValueError("adaptive residual shadow parent region does not match candidate")
        parent_unit_count = len(region_payload["unit_ids"])
        if parent_unit_count != candidate.parent_unit_count:
            raise ValueError("adaptive residual shadow parent unit count drifted")
        if birth_mode not in {"random", "pressure_anchor", "pressure_mixture"}:
            raise ValueError("unsupported adaptive residual shadow birth mode")
        if int(parent_bridge_payload["region"].get("input_dim", -1)) != config.motor_context_dim:
            raise ValueError("adaptive residual shadow parent input dimension mismatch")
        if candidate.unit_id in {str(item) for item in region_payload["unit_ids"]}:
            raise ValueError("adaptive residual shadow candidate unit already exists")
        constructor_generator = torch.Generator(device="cpu").manual_seed(0)
        region = AdaptiveNeuronRegion.from_payload(
            region_payload,
            generator=constructor_generator,
            device=device,
        )
        growth_seed = int(candidate.candidate_digest[:16], 16) % (2**63 - 1)
        growth_generator = torch.Generator(device="cpu").manual_seed(growth_seed)
        region.apply_topology_proposal(candidate.proposal, generator=growth_generator)
        if region.unit_ids[-1] != candidate.unit_id:
            raise ValueError("adaptive residual shadow candidate was not appended")
        birth_anchor_unit_ids: tuple[str, ...] = ()
        birth_anchor_weights: tuple[float, ...] = ()
        birth_anchor_indices: tuple[int, ...] = ()
        if birth_mode in {"pressure_anchor", "pressure_mixture"}:
            parent_activity = region.activity[: candidate.parent_unit_count].abs()
            parent_eligibility = region.trace[: candidate.parent_unit_count].abs()
            anchor_score = parent_activity + parent_eligibility
            candidate_index = region.unit_index(candidate.unit_id)
            anchor_count = 1 if birth_mode == "pressure_anchor" else min(2, candidate.parent_unit_count)
            anchor_indices = tuple(
                int(index)
                for index in torch.topk(
                    anchor_score,
                    k=anchor_count,
                    largest=True,
                    sorted=True,
                ).indices.tolist()
            )
            birth_anchor_indices = anchor_indices
            selected_scores = anchor_score[list(anchor_indices)]
            if float(selected_scores.sum().item()) <= 1e-8:
                anchor_weights = torch.full_like(
                    selected_scores,
                    1.0 / max(1, anchor_count),
                )
            else:
                anchor_weights = selected_scores / selected_scores.sum()
            birth_anchor_unit_ids = tuple(region.unit_ids[index] for index in anchor_indices)
            birth_anchor_weights = tuple(float(value) for value in anchor_weights.tolist())
            if birth_mode == "pressure_anchor":
                cls._copy_birth_anchor(
                    region,
                    source_index=anchor_indices[0],
                    candidate_index=candidate_index,
                )
            else:
                cls._mix_birth_anchors(
                    region,
                    anchor_indices=anchor_indices,
                    anchor_weights=birth_anchor_weights,
                    candidate_index=candidate_index,
                )
        projection = cls._new_identity_projection(
            config,
            unit_count=region.unit_count,
            candidate_index=region.unit_index(candidate.unit_id),
            generator=growth_generator,
            device=device,
        )
        if birth_anchor_indices:
            cls._seed_birth_output_projection(
                projection,
                anchor_indices=birth_anchor_indices,
                anchor_weights=birth_anchor_weights,
                candidate_index=region.unit_index(candidate.unit_id),
            )
        gate_projection = cls._new_gate_projection(
            config,
            input_dim=(
                int(config.motor_context_dim)
                + 2 * int(candidate.parent_unit_count)
            ),
            generator=growth_generator,
            device=device,
        )
        shadow = cls(
            config,
            candidate=candidate,
            region=region,
            output_projection=projection,
            parent_bridge_digest=content_digest(parent_bridge_payload),
            residual_gain=float(parent_bridge_payload["residual_gain"]),
            gate_projection=gate_projection,
            birth_anchor_unit_ids=birth_anchor_unit_ids,
            birth_anchor_weights=birth_anchor_weights,
            device=device,
        )
        last_input = parent_bridge_payload["last_input"].detach().to(shadow.device).clone()
        last_activity = parent_bridge_payload["last_activity"].detach().to(shadow.device).clone()
        if last_input.shape != (shadow.output_dim,):
            raise ValueError("adaptive residual shadow parent last_input shape mismatch")
        if last_activity.shape != (candidate.parent_unit_count,):
            raise ValueError("adaptive residual shadow parent last_activity shape mismatch")
        shadow._last_input.copy_(last_input)
        shadow._last_activity[: candidate.parent_unit_count].copy_(last_activity)
        return shadow

    @staticmethod
    def _new_gate_projection(
        config: TaijiConfig,
        *,
        input_dim: int,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> SparseSynapses:
        projection = SparseSynapses(
            1,
            int(input_dim),
            min(int(input_dim), max(1, int(config.predictive_context_fan_in))),
            generator=generator,
            init_scale=float(config.weight_init_scale),
            max_weight_norm=float(config.max_weight_norm),
            device=device,
        )
        projection.edge_weight.zero_()
        return projection

    @staticmethod
    def _new_identity_projection(
        config: TaijiConfig,
        *,
        unit_count: int,
        candidate_index: int,
        generator: torch.Generator,
        device: torch.device | str,
    ) -> SparseSynapses:
        fan_in = min(unit_count, max(2, int(config.predictive_context_fan_in)))
        projection = SparseSynapses(
            int(config.motor_context_dim),
            unit_count,
            fan_in,
            generator=generator,
            init_scale=float(config.weight_init_scale),
            max_weight_norm=float(config.max_weight_norm),
            device=device,
        )
        supports: list[list[int]] = []
        for output_index in range(int(config.motor_context_dim)):
            row = [output_index, candidate_index]
            for input_index in range(unit_count):
                if len(row) >= projection.row_fan_in:
                    break
                if input_index not in row:
                    row.append(input_index)
            supports.append(row[: projection.row_fan_in])
        projection.pre_index = torch.tensor(supports, dtype=torch.int32, device=projection.device)
        projection.edge_weight.zero_()
        for output_index in range(int(config.motor_context_dim)):
            identity_positions = projection.pre_index[output_index] == output_index
            projection.edge_weight[output_index, identity_positions] = 1.0
        return projection

    @staticmethod
    @torch.no_grad()
    def _seed_birth_output_projection(
        projection: SparseSynapses,
        *,
        anchor_indices: tuple[int, ...],
        anchor_weights: tuple[float, ...],
        candidate_index: int,
    ) -> None:
        """Align a pressure-born unit's output mapping with its anchors.

        The R3 bridge exposes its region through an identity unit→context
        mapping.  A pressure-born candidate that inherits only dendritic and
        recurrent rows would still enter the readout with a zero axon.  Seed
        the candidate edge at each anchor's identity output row so a mixture
        birth starts as the same functional mixture before local credit
        refines it.  Random births retain the zero candidate output edge.
        """

        for anchor_index, anchor_weight in zip(anchor_indices, anchor_weights):
            if not 0 <= int(anchor_index) < projection.out_features:
                continue
            candidate_positions = projection.pre_index[int(anchor_index)] == candidate_index
            if bool(candidate_positions.any()):
                projection.edge_weight[int(anchor_index), candidate_positions] = float(
                    anchor_weight
                )
        candidate_mask = projection.pre_index == candidate_index
        candidate_weights = projection.edge_weight[candidate_mask]
        if candidate_weights.numel() == 0:
            return
        target_squared_norm = torch.as_tensor(
            sum(float(value) for value in anchor_weights),
            device=projection.device,
            dtype=projection.edge_weight.dtype,
        )
        candidate_squared_norm = candidate_weights.pow(2).sum()
        if float(candidate_squared_norm.item()) > 1e-12:
            scale = torch.sqrt(target_squared_norm / candidate_squared_norm)
            projection.edge_weight[candidate_mask] = candidate_weights * scale

    @classmethod
    def from_checkpoint(
        cls,
        config: TaijiConfig,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> AdaptiveResidualShadow:
        if payload.get("format") != ADAPTIVE_RESIDUAL_SHADOW_FORMAT:
            raise ValueError("unsupported adaptive residual shadow format")
        if int(payload.get("version", -1)) != ADAPTIVE_RESIDUAL_SHADOW_VERSION:
            raise ValueError("unsupported adaptive residual shadow version")
        expected = content_digest(
            {key: value for key, value in payload.items() if key != "shadow_digest"}
        )
        if str(payload.get("shadow_digest", "")) != expected:
            raise ValueError("adaptive residual shadow digest mismatch")
        if payload.get("config_digest") != content_digest(config.to_dict()):
            raise ValueError("adaptive residual shadow configuration does not match")
        candidate_payload = payload.get("candidate")
        if not isinstance(candidate_payload, Mapping):
            raise ValueError("adaptive residual shadow candidate payload is invalid")
        candidate = AdaptiveResidualGrowthCandidate.from_payload(candidate_payload)
        region_payload = payload.get("region")
        projection_payload = payload.get("output_projection")
        if not isinstance(region_payload, Mapping) or not isinstance(projection_payload, Mapping):
            raise ValueError("adaptive residual shadow state payload is invalid")
        constructor_generator = torch.Generator(device="cpu").manual_seed(0)
        region = AdaptiveNeuronRegion.from_payload(
            region_payload,
            generator=constructor_generator,
            device=device,
        )
        projection = SparseSynapses(
            int(projection_payload["out_features"]),
            int(projection_payload["in_features"]),
            int(projection_payload["fan_in"]),
            generator=constructor_generator,
            init_scale=float(config.weight_init_scale),
            max_weight_norm=float(config.max_weight_norm),
            device=device,
        )
        projection.load_payload(projection_payload)
        gate_projection_payload = payload.get("gate_projection")
        gate_projection = None
        if gate_projection_payload is not None:
            if not isinstance(gate_projection_payload, Mapping):
                raise ValueError("adaptive residual shadow gate payload is invalid")
            gate_projection = SparseSynapses(
                int(gate_projection_payload["out_features"]),
                int(gate_projection_payload["in_features"]),
                int(gate_projection_payload["fan_in"]),
                generator=constructor_generator,
                init_scale=float(config.weight_init_scale),
                max_weight_norm=float(config.max_weight_norm),
                device=device,
            )
            gate_projection.load_payload(gate_projection_payload)
        shadow = cls(
            config,
            candidate=candidate,
            region=region,
            output_projection=projection,
            parent_bridge_digest=str(payload["parent_bridge_digest"]),
            residual_gain=float(payload["residual_gain"]),
            gate_projection=gate_projection,
            birth_anchor_unit_id=payload.get("birth_anchor_unit_id"),
            birth_anchor_unit_ids=(
                None
                if payload.get("birth_anchor_unit_ids") is None
                else tuple(str(item) for item in payload["birth_anchor_unit_ids"])
            ),
            birth_anchor_weights=(
                None
                if payload.get("birth_anchor_weights") is None
                else tuple(float(value) for value in payload["birth_anchor_weights"])
            ),
            device=device,
        )
        shadow.set_gate(float(payload["gate"]))
        shadow._lesioned = bool(payload.get("lesioned", False))
        last_input = payload["last_input"].detach().to(shadow.device).clone()
        last_activity = payload["last_activity"].detach().to(shadow.device).clone()
        if last_input.shape != (shadow.output_dim,):
            raise ValueError("adaptive residual shadow last_input shape mismatch")
        if last_activity.shape != (shadow.unit_count,):
            raise ValueError("adaptive residual shadow last_activity shape mismatch")
        shadow._last_input = last_input
        shadow._last_activity = last_activity
        last_parent_context = payload.get(
            "last_parent_context",
            torch.zeros(shadow.output_dim, device=shadow.device),
        )
        last_parent_context = last_parent_context.detach().to(shadow.device).clone()
        if last_parent_context.shape != (shadow.output_dim,):
            raise ValueError("adaptive residual shadow last_parent_context shape mismatch")
        shadow._last_parent_context = last_parent_context
        parent_probabilities = payload.get(
            "counterfactual_parent_probabilities",
            torch.zeros(shadow.config.alphabet_size, device=shadow.device),
        )
        parent_probabilities = parent_probabilities.detach().to(shadow.device).clone()
        if parent_probabilities.shape != (shadow.config.alphabet_size,):
            raise ValueError(
                "adaptive residual shadow counterfactual probability shape mismatch"
            )
        if not bool(torch.isfinite(parent_probabilities).all()) or bool(
            (parent_probabilities < 0.0).any()
        ):
            raise ValueError(
                "adaptive residual shadow counterfactual probabilities must be finite and non-negative"
            )
        shadow._last_counterfactual_parent_probabilities = parent_probabilities
        shadow._counterfactual_parent_ready = bool(
            payload.get("counterfactual_parent_ready", False)
        )
        shadow._counterfactual_utility_ready = bool(
            payload.get("counterfactual_utility_ready", False)
        )
        if gate_projection is not None:
            gate_input = payload.get("last_gate_input")
            if gate_input is None:
                raise ValueError("adaptive residual shadow gate input state is missing")
            shadow._last_gate_input = gate_input.detach().to(shadow.device).clone()
            if shadow._last_gate_input.shape != (gate_projection.in_features,):
                raise ValueError("adaptive residual shadow gate input shape mismatch")
            gate_bias = payload.get("candidate_gate_bias", 0.0)
            shadow._candidate_gate_bias = torch.as_tensor(
                gate_bias,
                device=shadow.device,
                dtype=torch.float32,
            ).clone()
            if shadow._candidate_gate_bias.shape != ():
                raise ValueError("adaptive residual shadow gate bias shape mismatch")
            shadow._candidate_utility_baseline = float(
                payload.get("candidate_utility_baseline", 0.0)
            )
        diagnostics = payload.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            raise ValueError("adaptive residual shadow diagnostics payload is invalid")
        for name in (
            "candidate_activity",
            "candidate_eligibility_norm",
            "parent_residual_norm",
            "candidate_residual_norm",
            "candidate_credit_norm",
            "candidate_projection_update_norm",
            "candidate_utility",
            "candidate_counterfactual_utility",
            "candidate_gate",
        ):
            default = 1.0 if name == "candidate_gate" else 0.0
            value = float(diagnostics.get(name, default))
            if not math.isfinite(value) or (
                name
                not in {"candidate_utility", "candidate_counterfactual_utility"}
                and value < 0.0
            ):
                raise ValueError("adaptive residual shadow diagnostics must be finite")
            if name == "candidate_counterfactual_utility":
                shadow._last_counterfactual_utility = value
            else:
                setattr(shadow, f"_last_{name}", value)
        return shadow


__all__ = [
    "ADAPTIVE_RESIDUAL_SHADOW_FORMAT",
    "ADAPTIVE_RESIDUAL_SHADOW_VERSION",
    "AdaptiveResidualShadow",
]

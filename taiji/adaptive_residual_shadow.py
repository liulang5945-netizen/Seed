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
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.candidate = candidate
        self.region = region
        self.output_projection = output_projection
        self.parent_bridge_digest = str(parent_bridge_digest).strip()
        self.residual_gain = float(residual_gain)
        self._validate_gain(self.residual_gain)
        if not self.parent_bridge_digest:
            raise ValueError("adaptive residual shadow parent bridge digest must not be empty")
        if self.region.region_id != candidate.bridge_id:
            raise ValueError("adaptive residual shadow region does not match candidate")
        if candidate.unit_id not in self.region.unit_ids:
            raise ValueError("adaptive residual shadow is missing the candidate unit")
        if self.region.unit_count != candidate.proposed_unit_count:
            raise ValueError("adaptive residual shadow unit count does not match candidate")
        if output_projection.out_features != self.output_dim:
            raise ValueError("adaptive residual shadow projection output dimension mismatch")
        if output_projection.in_features != self.unit_count:
            raise ValueError("adaptive residual shadow projection input dimension mismatch")
        self._gate = 0.0
        self._lesioned = False
        self._last_input = torch.zeros(self.output_dim, device=self.device)
        self._last_activity = torch.zeros(self.unit_count, device=self.device)
        self._last_candidate_activity = 0.0
        self._last_candidate_eligibility_norm = 0.0
        self._last_parent_residual_norm = 0.0
        self._last_candidate_residual_norm = 0.0
        self._last_candidate_credit_norm = 0.0
        self._last_candidate_projection_update_norm = 0.0

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
        return self.region.edge_count + self.output_projection.edge_count

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
    def diagnostics(self) -> dict[str, float]:
        return {
            "candidate_activity": self.candidate_activity,
            "candidate_eligibility_norm": self.candidate_eligibility_norm,
            "parent_residual_norm": self.parent_residual_norm,
            "candidate_residual_norm": self.candidate_residual_norm,
            "candidate_credit_norm": self.candidate_credit_norm,
            "candidate_projection_update_norm": self.candidate_projection_update_norm,
        }

    @property
    def last_activity(self) -> torch.Tensor:
        return self._last_activity.detach().clone()

    @torch.no_grad()
    def set_gate(self, gate: float) -> None:
        value = float(gate)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("adaptive residual shadow gate must be finite and in [0, 1]")
        self._gate = value

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

    @torch.no_grad()
    def reset_dynamics(self) -> None:
        self.region.membrane.zero_()
        self.region.activity.zero_()
        self.region.trace.zero_()
        self._last_input.zero_()
        self._last_activity.zero_()
        self._last_candidate_activity = 0.0
        self._last_candidate_eligibility_norm = 0.0
        self._last_parent_residual_norm = 0.0
        self._last_candidate_residual_norm = 0.0
        self._last_candidate_credit_norm = 0.0
        self._last_candidate_projection_update_norm = 0.0

    @torch.no_grad()
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape != (self.output_dim,):
            raise ValueError("adaptive residual shadow context dimension mismatch")
        if self._gate == 0.0 or self._lesioned:
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
        self._last_candidate_activity = float(abs(activity[candidate_index]).item())
        self._last_candidate_eligibility_norm = float(abs(candidate_eligibility).item())
        self._last_candidate_residual_norm = float(candidate_residual.norm().item())
        self._last_parent_residual_norm = float((residual - candidate_residual).norm().item())
        return bound_norm(
            context.to(self.device) + float(self._gate) * float(self.residual_gain) * residual,
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
        return (*self.region.parameter_tensors(), self.output_projection.edge_weight)

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
            "diagnostics": dict(self.diagnostics),
        }
        return {**payload, "shadow_digest": content_digest(payload)}

    @classmethod
    def from_parent_bridge(
        cls,
        config: TaijiConfig,
        parent_bridge_payload: Mapping[str, Any],
        candidate: AdaptiveResidualGrowthCandidate,
        *,
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
        projection = cls._new_identity_projection(
            config,
            unit_count=region.unit_count,
            candidate_index=region.unit_index(candidate.unit_id),
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
        shadow = cls(
            config,
            candidate=candidate,
            region=region,
            output_projection=projection,
            parent_bridge_digest=str(payload["parent_bridge_digest"]),
            residual_gain=float(payload["residual_gain"]),
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
        ):
            value = float(diagnostics.get(name, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("adaptive residual shadow diagnostics must be finite and non-negative")
            setattr(shadow, f"_last_{name}", value)
        return shadow


__all__ = [
    "ADAPTIVE_RESIDUAL_SHADOW_FORMAT",
    "ADAPTIVE_RESIDUAL_SHADOW_VERSION",
    "AdaptiveResidualShadow",
]

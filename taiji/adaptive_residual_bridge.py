"""A zero-gated adaptive population connected to Taiji's predictive path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .config import TaijiConfig
from .neuron_region import AdaptiveNeuronRegion, NeuronRegionDynamics
from .sparse import bound_norm

ADAPTIVE_RESIDUAL_BRIDGE_FORMAT = "taiji-adaptive-residual-bridge-v1"
ADAPTIVE_RESIDUAL_BRIDGE_VERSION = 1


class AdaptiveResidualBridge:
    """One adaptive residual population on the F1 context→readout path.

    The bridge is born with ``gate=0``.  In that state it does not step its
    region, mutate runtime state, or write credit, so attaching it is a true
    functional no-op.  Opening the gate explicitly lets the population
    consume the predictive context and receive the same causal readout error
    that trains F1 context, which makes it a main-path substrate rather than a
    side-channel experiment.
    """

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        unit_count: int | None = None,
        fan_in: int | None = None,
        gate: float = 0.0,
        residual_gain: float = 0.20,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.output_dim = int(config.motor_context_dim)
        selected_units = self.output_dim if unit_count is None else int(unit_count)
        selected_fan_in = (
            int(config.predictive_context_fan_in) if fan_in is None else int(fan_in)
        )
        if selected_units != self.output_dim:
            raise ValueError("R3 bridge unit_count must equal motor_context_dim")
        if not 0 < selected_fan_in <= self.output_dim:
            raise ValueError("R3 bridge fan_in must fit motor_context_dim")
        recurrent_fan_in = min(selected_fan_in, max(1, selected_units - 1))
        self.residual_gain = float(residual_gain)
        self._validate_gain(self.residual_gain)
        self._gate = 0.0
        self._lesioned = False
        dynamics = NeuronRegionDynamics(
            learning_rate=float(config.predictive_context_learning_rate),
            weight_decay=float(config.synapse_decay),
            weight_init_scale=float(config.weight_init_scale),
            max_weight_norm=float(config.max_weight_norm),
            max_state_norm=float(config.motor_context_norm),
        )
        self.region = AdaptiveNeuronRegion(
            region_id="predictive_residual.bridge",
            input_dim=self.output_dim,
            unit_ids=tuple(
                f"predictive_residual.bridge.u{index}" for index in range(selected_units)
            ),
            fan_in=selected_fan_in,
            input_source_id="predictive_context",
            dynamics=dynamics,
            recurrent_fan_in=recurrent_fan_in,
            generator=generator,
            device=self.device,
        )
        self._last_input = torch.zeros(self.output_dim, device=self.device)
        self._last_activity = torch.zeros(selected_units, device=self.device)
        self.set_gate(gate)

    @staticmethod
    def _validate_gain(value: float) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("R3 bridge residual_gain must be finite and positive")

    @property
    def gate(self) -> float:
        return self._gate

    @property
    def lesioned(self) -> bool:
        return self._lesioned

    @property
    def unit_count(self) -> int:
        return self.region.unit_count

    @property
    def edge_count(self) -> int:
        return self.region.edge_count

    @property
    def activity_saturation(self) -> float:
        """Fraction of units at or above their target activity set point."""

        target = float(self.region.dynamics.target_activity)
        if target <= 0.0:
            return 1.0 if bool(self.region.activity.abs().any()) else 0.0
        return float((self.region.activity.abs() >= target).to(torch.float32).mean().item())

    @property
    def last_activity(self) -> torch.Tensor:
        return self._last_activity.detach().clone()

    @torch.no_grad()
    def set_gate(self, gate: float) -> None:
        value = float(gate)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("R3 bridge gate must be finite and in [0, 1]")
        self._gate = value

    @torch.no_grad()
    def lesion(self) -> None:
        self._lesioned = True
        self.reset_dynamics()

    @torch.no_grad()
    def unlesion(self) -> None:
        self._lesioned = False

    @torch.no_grad()
    def reset_dynamics(self) -> None:
        self.region.membrane.zero_()
        self.region.activity.zero_()
        self.region.trace.zero_()
        self._last_input.zero_()
        self._last_activity.zero_()

    @torch.no_grad()
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape != (self.output_dim,):
            raise ValueError("R3 bridge context dimension mismatch")
        if self._gate == 0.0 or self._lesioned:
            return context.clone()
        input_context = context.detach().to(self.device).clone()
        activity = self.region.step(input_context)
        self._last_input.copy_(input_context)
        self._last_activity.copy_(activity)
        residual = float(self.residual_gain) * activity
        return bound_norm(
            context.to(self.device) + float(self._gate) * residual,
            float(self.config.motor_context_norm),
        )

    @torch.no_grad()
    def learn(self, postsynaptic_error: torch.Tensor) -> None:
        if postsynaptic_error.shape != (self.unit_count,):
            raise ValueError("R3 bridge credit dimension mismatch")
        if self._gate == 0.0 or self._lesioned:
            return
        self.region.learn(
            self._last_input,
            postsynaptic_error.to(self.device) * float(self._gate) * float(self.residual_gain),
        )

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.region.parameter_tensors()

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": ADAPTIVE_RESIDUAL_BRIDGE_FORMAT,
            "version": ADAPTIVE_RESIDUAL_BRIDGE_VERSION,
            "gate": self._gate,
            "lesioned": self._lesioned,
            "residual_gain": self.residual_gain,
            "region": self.region.to_payload(),
            "last_input": self._last_input.detach().cpu().clone(),
            "last_activity": self._last_activity.detach().cpu().clone(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != ADAPTIVE_RESIDUAL_BRIDGE_FORMAT:
            raise ValueError("unsupported adaptive residual bridge format")
        if int(payload.get("version", -1)) != ADAPTIVE_RESIDUAL_BRIDGE_VERSION:
            raise ValueError("unsupported adaptive residual bridge version")
        residual_gain = float(payload["residual_gain"])
        self._validate_gain(residual_gain)
        if residual_gain != self.residual_gain:
            raise ValueError("adaptive residual bridge gain does not match architecture")
        self.region.load_payload(payload["region"])
        self.set_gate(float(payload["gate"]))
        self._lesioned = bool(payload.get("lesioned", False))
        last_input = payload["last_input"].detach().to(self.device, dtype=torch.float32).clone()
        last_activity = payload["last_activity"].detach().to(self.device, dtype=torch.float32).clone()
        if last_input.shape != (self.output_dim,):
            raise ValueError("adaptive residual bridge last_input shape does not match")
        if last_activity.shape != (self.unit_count,):
            raise ValueError("adaptive residual bridge last_activity shape does not match")
        self._last_input = last_input
        self._last_activity = last_activity

    @classmethod
    def from_payload(
        cls,
        config: TaijiConfig,
        payload: Mapping[str, Any],
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> AdaptiveResidualBridge:
        region_payload = payload.get("region")
        if not isinstance(region_payload, Mapping):
            raise ValueError("adaptive residual bridge region payload is invalid")
        bridge = cls(
            config,
            generator=generator,
            unit_count=len(region_payload["unit_ids"]),
            fan_in=int(region_payload["fan_in"]),
            gate=float(payload["gate"]),
            residual_gain=float(payload["residual_gain"]),
            device=device,
        )
        bridge.load_payload(payload)
        return bridge

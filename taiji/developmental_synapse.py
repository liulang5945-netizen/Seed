"""Checkpoint-compatible fast/slow synapse state for Taiji growth.

R1 is a migration boundary: it copies an existing F1 fixed-fan-in bank into
``slow_weight`` and creates a zero ``fast_delta`` plus explicit developmental
metadata.  R2 adds deliberately small local update primitives.  The model
still owns the learning schedule and Gate; this module only owns the sparse
state transition and its content-addressed representation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .internalization import content_digest

DEVELOPMENTAL_SYNAPSE_FORMAT = "taiji-developmental-synapse-v1"
DEVELOPMENTAL_SYNAPSE_VERSION = 1
DEVELOPMENTAL_SYNAPSE_BUNDLE_FORMAT = "taiji-developmental-synapse-bundle-v1"
DEVELOPMENTAL_SYNAPSE_BUNDLE_VERSION = 1
DEVELOPMENTAL_REPLAY_EVENT_FORMAT = "taiji-developmental-replay-event-v1"
DEVELOPMENTAL_REPLAY_EVENT_VERSION = 1
DEVELOPMENTAL_REPLAY_BUFFER_FORMAT = "taiji-developmental-replay-buffer-v1"
DEVELOPMENTAL_REPLAY_BUFFER_VERSION = 1
_SPARSE_STORAGE_FORMAT = "fixed-fan-in-v1"


class DevelopmentalSynapseContractError(ValueError):
    """Raised when a developmental synapse payload is malformed or unsafe."""


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DevelopmentalSynapseContractError(f"{name} must not be empty")


def _require_finite(value: float, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise DevelopmentalSynapseContractError(f"{name} must be finite")


def _verify_envelope(payload: Mapping[str, Any], expected_format: str, version: int) -> None:
    if payload.get("format") != expected_format:
        raise DevelopmentalSynapseContractError(
            f"unsupported developmental payload format: {payload.get('format')!r}"
        )
    if int(payload.get("version", -1)) != version:
        raise DevelopmentalSynapseContractError(
            f"unsupported developmental payload version: {payload.get('version')!r}"
        )


def _verify_digest(payload: Mapping[str, Any], field_name: str) -> None:
    actual = payload.get(field_name)
    if not isinstance(actual, str) or not actual:
        raise DevelopmentalSynapseContractError(f"{field_name} must be present")
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    if actual != content_digest(unsigned):
        raise DevelopmentalSynapseContractError(f"{field_name} does not match payload")


@dataclass(frozen=True)
class DevelopmentalSynapseBank:
    """A fixed topology with separate slow, fast and developmental state."""

    owner_id: str
    out_features: int
    in_features: int
    fan_in: int
    row_fan_in: int
    max_weight_norm: float
    pre_index: torch.Tensor
    slow_weight: torch.Tensor
    fast_delta: torch.Tensor
    eligibility: torch.Tensor
    importance: torch.Tensor
    usage: torch.Tensor
    age: torch.Tensor
    plasticity: torch.Tensor
    source_synapse_digest: str

    def __post_init__(self) -> None:
        _require_text(self.owner_id, "owner_id")
        _require_text(self.source_synapse_digest, "source_synapse_digest")
        if self.out_features <= 0 or self.in_features <= 0:
            raise DevelopmentalSynapseContractError("synapse dimensions must be positive")
        if self.fan_in <= 0 or self.row_fan_in <= 0 or self.row_fan_in > self.fan_in:
            raise DevelopmentalSynapseContractError("synapse fan-in is invalid")
        _require_finite(self.max_weight_norm, "max_weight_norm")
        if self.max_weight_norm <= 0:
            raise DevelopmentalSynapseContractError("max_weight_norm must be positive")
        if self.pre_index.shape != (self.out_features, self.row_fan_in):
            raise DevelopmentalSynapseContractError("pre_index shape does not match bank")
        expected = self.slow_weight.shape
        if expected != (self.out_features, self.row_fan_in):
            raise DevelopmentalSynapseContractError("slow_weight shape does not match bank")
        for name in (
            "fast_delta",
            "eligibility",
            "importance",
            "usage",
            "age",
            "plasticity",
        ):
            if getattr(self, name).shape != expected:
                raise DevelopmentalSynapseContractError(f"{name} shape does not match bank")
        rows = self.pre_index.detach().to(dtype=torch.long)
        if bool(((rows < 0) | (rows >= self.in_features)).any()):
            raise DevelopmentalSynapseContractError("pre_index is outside the input population")
        if bool((rows.sort(dim=1).values.diff(dim=1) == 0).any()):
            raise DevelopmentalSynapseContractError("pre_index repeats a contact")
        for name in (
            "slow_weight",
            "fast_delta",
            "eligibility",
            "importance",
            "usage",
            "age",
            "plasticity",
        ):
            tensor = getattr(self, name)
            if not bool(torch.isfinite(tensor).all()):
                raise DevelopmentalSynapseContractError(f"{name} contains non-finite values")
        for name in ("importance", "usage", "age"):
            if bool((getattr(self, name) < 0).any()):
                raise DevelopmentalSynapseContractError(f"{name} cannot be negative")
        if bool((self.plasticity < 0).any()) or bool((self.plasticity > 1).any()):
            raise DevelopmentalSynapseContractError("plasticity must be in [0, 1]")

    @classmethod
    def from_sparse_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> DevelopmentalSynapseBank:
        """Migrate one existing ``SparseSynapses`` payload without changing it."""

        if payload.get("storage") != _SPARSE_STORAGE_FORMAT:
            raise DevelopmentalSynapseContractError("source is not fixed-fan-in synapse storage")
        edge_weight = payload["edge_weight"].detach().cpu().to(dtype=torch.float32).clone()
        pre_index = payload["pre_index"].detach().cpu().to(dtype=torch.int32).clone()
        zeros = torch.zeros_like(edge_weight)
        ones = torch.ones_like(edge_weight)
        return cls(
            owner_id=owner_id,
            out_features=int(payload["out_features"]),
            in_features=int(payload["in_features"]),
            fan_in=int(payload["fan_in"]),
            row_fan_in=int(payload["row_fan_in"]),
            max_weight_norm=float(payload["max_weight_norm"]),
            pre_index=pre_index,
            slow_weight=edge_weight,
            fast_delta=zeros.clone(),
            eligibility=zeros.clone(),
            importance=ones.clone(),
            usage=zeros.clone(),
            age=zeros.clone(),
            plasticity=ones.clone(),
            source_synapse_digest=content_digest(dict(payload)),
        )

    @property
    def effective_weight(self) -> torch.Tensor:
        return self.slow_weight + self.fast_delta

    @property
    def effective_parameter_count(self) -> int:
        """Number of effective learned contacts, excluding developmental state."""

        return int(self.slow_weight.numel())

    @property
    def state_scalar_count(self) -> int:
        """Stored scalar count including fast state and explicit metadata."""

        return sum(
            int(getattr(self, name).numel())
            for name in (
                "slow_weight",
                "fast_delta",
                "eligibility",
                "importance",
                "usage",
                "age",
                "plasticity",
            )
        )

    @property
    def fast_is_zero(self) -> bool:
        return not bool(torch.count_nonzero(self.fast_delta).item())

    def forward(self, presynaptic: torch.Tensor) -> torch.Tensor:
        if presynaptic.shape != (self.in_features,):
            raise ValueError(
                f"presynaptic shape must be ({self.in_features},), got {tuple(presynaptic.shape)}"
            )
        device = presynaptic.device
        pre_index = self.pre_index.to(device=device, dtype=torch.long)
        weight = self.effective_weight.to(device=device, dtype=presynaptic.dtype)
        return (weight * presynaptic[pre_index]).sum(dim=1)

    def backproject(self, postsynaptic_error: torch.Tensor) -> torch.Tensor:
        """Project an output error through the current effective contacts."""

        if postsynaptic_error.shape != (self.out_features,):
            raise ValueError(
                f"postsynaptic error shape must be ({self.out_features},), "
                f"got {tuple(postsynaptic_error.shape)}"
            )
        device = postsynaptic_error.device
        pre_index = self.pre_index.to(device=device, dtype=torch.long)
        weight = self.effective_weight.to(device=device, dtype=postsynaptic_error.dtype)
        projected = torch.zeros(
            self.in_features,
            device=device,
            dtype=postsynaptic_error.dtype,
        )
        projected.scatter_add_(
            0,
            pre_index.reshape(-1),
            (postsynaptic_error[:, None] * weight).reshape(-1),
        )
        return projected

    def _validate_local_update(
        self,
        postsynaptic_error: torch.Tensor,
        presynaptic_trace: torch.Tensor,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        if postsynaptic_error.shape != (self.out_features,):
            raise ValueError("developmental postsynaptic error shape mismatch")
        if presynaptic_trace.shape != (self.in_features,):
            raise ValueError("developmental presynaptic trace shape mismatch")
        learning_rate = float(learning_rate)
        weight_decay = float(weight_decay)
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("developmental learning_rate must be finite and non-negative")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("developmental weight_decay must be finite and non-negative")
        if not bool(torch.isfinite(postsynaptic_error).all()):
            raise ValueError("developmental postsynaptic error must be finite")
        if not bool(torch.isfinite(presynaptic_trace).all()):
            raise ValueError("developmental presynaptic trace must be finite")
        return (
            postsynaptic_error.to(self.slow_weight.device, dtype=self.slow_weight.dtype),
            presynaptic_trace.to(self.slow_weight.device, dtype=self.slow_weight.dtype),
            learning_rate,
            weight_decay,
        )

    @torch.no_grad()
    def _apply_local_update(
        self,
        target: torch.Tensor,
        postsynaptic_error: torch.Tensor,
        presynaptic_trace: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        error, trace, learning_rate, weight_decay = self._validate_local_update(
            postsynaptic_error,
            presynaptic_trace,
            learning_rate,
            weight_decay,
        )
        pre_index = self.pre_index.to(device=target.device, dtype=torch.long)
        active = trace[pre_index]
        active_scale = max(1.0, float(active.abs().sum().item()))
        if weight_decay > 0.0:
            target.mul_(max(0.0, 1.0 - weight_decay))
        if learning_rate > 0.0:
            target.add_(learning_rate * error[:, None] * active / active_scale)
        activity = active.abs()
        self.eligibility.mul_(0.95).add_(activity).clamp_(max=1e6)
        self.usage.add_((activity > 0.0).to(self.usage.dtype))
        self.age.add_(1.0)

    @torch.no_grad()
    def _bound_rows(self, weight: torch.Tensor) -> None:
        norms = weight.norm(dim=1, keepdim=True)
        scale = torch.clamp(
            float(self.max_weight_norm) / torch.clamp(norms, min=1e-12),
            max=1.0,
        )
        weight.mul_(scale)

    @torch.no_grad()
    def _bound_effective(self) -> None:
        """Keep the effective bank bounded without rewriting consolidated state."""

        effective = self.effective_weight
        norms = effective.norm(dim=1, keepdim=True)
        scale = torch.clamp(
            float(self.max_weight_norm) / torch.clamp(norms, min=1e-12),
            max=1.0,
        )
        # The slow store is the stable parent.  If a combined row exceeds its
        # budget, adjust only the fast residual to the bounded effective
        # target; rollback and consolidation can therefore audit slow state.
        self.fast_delta.copy_(effective * scale - self.slow_weight)

    @torch.no_grad()
    def learn_fast(
        self,
        postsynaptic_error: torch.Tensor,
        presynaptic_trace: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        """Apply one wake-style update to the reversible fast residual."""

        self._apply_local_update(
            self.fast_delta,
            postsynaptic_error,
            presynaptic_trace,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self._bound_effective()

    @torch.no_grad()
    def learn_slow(
        self,
        postsynaptic_error: torch.Tensor,
        presynaptic_trace: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        """Apply the controlled slow-only comparison update."""

        self._apply_local_update(
            self.slow_weight,
            postsynaptic_error,
            presynaptic_trace,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self._bound_rows(self.slow_weight)

    @torch.no_grad()
    def consolidate_fast(self, *, rate: float = 1.0, clear_fast: bool = True) -> None:
        """Move a validated fraction of fast state into the slow store."""

        rate = float(rate)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("consolidation rate must be finite and in [0, 1]")
        self.slow_weight.add_(rate * self.fast_delta * self.plasticity)
        self.importance.add_(rate * self.fast_delta.abs()).clamp_(max=1e6)
        self.age.add_(1.0)
        if clear_fast:
            self.fast_delta.zero_()
        else:
            self.fast_delta.mul_(1.0 - rate)
        self._bound_rows(self.slow_weight)
        self._bound_effective()

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "format": DEVELOPMENTAL_SYNAPSE_FORMAT,
            "version": DEVELOPMENTAL_SYNAPSE_VERSION,
            "owner_id": self.owner_id,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "fan_in": self.fan_in,
            "row_fan_in": self.row_fan_in,
            "max_weight_norm": self.max_weight_norm,
            "pre_index": self.pre_index.detach().cpu().clone(),
            "slow_weight": self.slow_weight.detach().cpu().clone(),
            "fast_delta": self.fast_delta.detach().cpu().clone(),
            "eligibility": self.eligibility.detach().cpu().clone(),
            "importance": self.importance.detach().cpu().clone(),
            "usage": self.usage.detach().cpu().clone(),
            "age": self.age.detach().cpu().clone(),
            "plasticity": self.plasticity.detach().cpu().clone(),
            "source_synapse_digest": self.source_synapse_digest,
        }

    def to_payload(self) -> dict[str, Any]:
        unsigned = self._unsigned_payload()
        return {**unsigned, "bank_digest": content_digest(unsigned)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DevelopmentalSynapseBank:
        _verify_envelope(payload, DEVELOPMENTAL_SYNAPSE_FORMAT, DEVELOPMENTAL_SYNAPSE_VERSION)
        _verify_digest(payload, "bank_digest")
        return cls(
            owner_id=str(payload["owner_id"]),
            out_features=int(payload["out_features"]),
            in_features=int(payload["in_features"]),
            fan_in=int(payload["fan_in"]),
            row_fan_in=int(payload["row_fan_in"]),
            max_weight_norm=float(payload["max_weight_norm"]),
            pre_index=payload["pre_index"].detach().cpu().to(dtype=torch.int32).clone(),
            slow_weight=payload["slow_weight"].detach().cpu().to(dtype=torch.float32).clone(),
            fast_delta=payload["fast_delta"].detach().cpu().to(dtype=torch.float32).clone(),
            eligibility=payload["eligibility"].detach().cpu().to(dtype=torch.float32).clone(),
            importance=payload["importance"].detach().cpu().to(dtype=torch.float32).clone(),
            usage=payload["usage"].detach().cpu().to(dtype=torch.float32).clone(),
            age=payload["age"].detach().cpu().to(dtype=torch.float32).clone(),
            plasticity=payload["plasticity"].detach().cpu().to(dtype=torch.float32).clone(),
            source_synapse_digest=str(payload["source_synapse_digest"]),
        )


@dataclass(frozen=True)
class DevelopmentalReplayEvent:
    """A local credit trace captured from one real wake prediction."""

    event_id: str
    source: str
    tick: int
    observed_symbol: int
    predicted_probability: float
    readout_error: torch.Tensor
    readout_trace: torch.Tensor
    context_feedback: torch.Tensor
    context_trace: torch.Tensor

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.source, "source")
        if int(self.tick) < 0:
            raise DevelopmentalSynapseContractError("replay event tick cannot be negative")
        if int(self.observed_symbol) < 0:
            raise DevelopmentalSynapseContractError("replay event symbol cannot be negative")
        _require_finite(self.predicted_probability, "predicted_probability")
        if not 0.0 <= float(self.predicted_probability) <= 1.0:
            raise DevelopmentalSynapseContractError("predicted_probability must be in [0, 1]")
        for name in (
            "readout_error",
            "readout_trace",
            "context_feedback",
            "context_trace",
        ):
            tensor = getattr(self, name)
            if tensor.ndim != 1 or tensor.numel() == 0:
                raise DevelopmentalSynapseContractError(
                    f"replay event {name} must be a non-empty vector"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise DevelopmentalSynapseContractError(f"replay event {name} is not finite")
            object.__setattr__(self, name, tensor.detach().cpu().to(dtype=torch.float32).clone())

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "format": DEVELOPMENTAL_REPLAY_EVENT_FORMAT,
            "version": DEVELOPMENTAL_REPLAY_EVENT_VERSION,
            "event_id": self.event_id,
            "source": self.source,
            "tick": int(self.tick),
            "observed_symbol": int(self.observed_symbol),
            "predicted_probability": float(self.predicted_probability),
            "readout_error": self.readout_error,
            "readout_trace": self.readout_trace,
            "context_feedback": self.context_feedback,
            "context_trace": self.context_trace,
        }

    def to_payload(self) -> dict[str, Any]:
        unsigned = self._unsigned_payload()
        return {**unsigned, "event_digest": content_digest(unsigned)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DevelopmentalReplayEvent:
        _verify_envelope(
            payload, DEVELOPMENTAL_REPLAY_EVENT_FORMAT, DEVELOPMENTAL_REPLAY_EVENT_VERSION
        )
        _verify_digest(payload, "event_digest")
        return cls(
            event_id=str(payload["event_id"]),
            source=str(payload["source"]),
            tick=int(payload["tick"]),
            observed_symbol=int(payload["observed_symbol"]),
            predicted_probability=float(payload["predicted_probability"]),
            readout_error=payload["readout_error"],
            readout_trace=payload["readout_trace"],
            context_feedback=payload["context_feedback"],
            context_trace=payload["context_trace"],
        )


@dataclass(frozen=True)
class DevelopmentalReplayBuffer:
    """Bounded, checkpointable local experiences used by sleep replay."""

    events: tuple[DevelopmentalReplayEvent, ...]
    max_events: int = 2048

    def __post_init__(self) -> None:
        if int(self.max_events) <= 0:
            raise DevelopmentalSynapseContractError("replay buffer max_events must be positive")
        if len(self.events) > int(self.max_events):
            raise DevelopmentalSynapseContractError("replay buffer exceeds max_events")
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise DevelopmentalSynapseContractError("replay event ids must be unique")

    @property
    def event_count(self) -> int:
        return len(self.events)

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "format": DEVELOPMENTAL_REPLAY_BUFFER_FORMAT,
            "version": DEVELOPMENTAL_REPLAY_BUFFER_VERSION,
            "max_events": int(self.max_events),
            "events": [event.to_payload() for event in self.events],
        }

    def to_payload(self) -> dict[str, Any]:
        unsigned = self._unsigned_payload()
        return {**unsigned, "buffer_digest": content_digest(unsigned)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DevelopmentalReplayBuffer:
        _verify_envelope(
            payload, DEVELOPMENTAL_REPLAY_BUFFER_FORMAT, DEVELOPMENTAL_REPLAY_BUFFER_VERSION
        )
        _verify_digest(payload, "buffer_digest")
        events = payload.get("events")
        if not isinstance(events, list):
            raise DevelopmentalSynapseContractError("replay buffer events must be a list")
        return cls(
            events=tuple(DevelopmentalReplayEvent.from_payload(event) for event in events),
            max_events=int(payload["max_events"]),
        )


@dataclass(frozen=True)
class DevelopmentalSynapseBundle:
    """The two F1 banks migrated from one immutable Taiji checkpoint."""

    source_checkpoint_digest: str
    source_checkpoint_format: str
    config_digest: str
    owner_graph_digest: str
    banks: tuple[tuple[str, DevelopmentalSynapseBank], ...]

    def __post_init__(self) -> None:
        for name in (
            "source_checkpoint_digest",
            "source_checkpoint_format",
            "config_digest",
            "owner_graph_digest",
        ):
            _require_text(getattr(self, name), name)
        if not self.banks:
            raise DevelopmentalSynapseContractError("developmental bundle needs at least one bank")
        names = [name for name, _ in self.banks]
        if len(set(names)) != len(names):
            raise DevelopmentalSynapseContractError("developmental bank owners must be unique")
        for name, bank in self.banks:
            if name != bank.owner_id:
                raise DevelopmentalSynapseContractError("bank key does not match owner_id")

    @classmethod
    def from_taiji_checkpoint(cls, checkpoint: Mapping[str, Any]) -> DevelopmentalSynapseBundle:
        for key in ("format", "config", "predictive_context", "predictive_readout"):
            if key not in checkpoint:
                raise DevelopmentalSynapseContractError(
                    f"F1 checkpoint is missing {key} for developmental migration"
                )
        context = checkpoint["predictive_context"]
        readout = checkpoint["predictive_readout"]
        if not isinstance(context, Mapping) or not isinstance(readout, Mapping):
            raise DevelopmentalSynapseContractError("F1 checkpoint owners must be mappings")
        context_recurrent = context.get("recurrent")
        readout_synapses = readout.get("synapses")
        if not isinstance(context_recurrent, Mapping) or not isinstance(readout_synapses, Mapping):
            raise DevelopmentalSynapseContractError("F1 checkpoint has invalid synapse owners")
        owners = (
            (
                "predictive_context.recurrent",
                DevelopmentalSynapseBank.from_sparse_payload(
                    context_recurrent,
                    owner_id="predictive_context.recurrent",
                ),
            ),
            (
                "predictive_readout.synapses",
                DevelopmentalSynapseBank.from_sparse_payload(
                    readout_synapses,
                    owner_id="predictive_readout.synapses",
                ),
            ),
        )
        return cls(
            source_checkpoint_digest=content_digest(dict(checkpoint)),
            source_checkpoint_format=str(checkpoint["format"]),
            config_digest=content_digest(checkpoint["config"]),
            owner_graph_digest=content_digest(
                {"owners": [name for name, _ in owners], "mode": "f1-read-only-migration"}
            ),
            banks=owners,
        )

    def bank(self, owner_id: str) -> DevelopmentalSynapseBank:
        for name, bank in self.banks:
            if name == owner_id:
                return bank
        raise DevelopmentalSynapseContractError(f"unknown developmental bank: {owner_id}")

    @property
    def fast_is_zero(self) -> bool:
        return all(bank.fast_is_zero for _, bank in self.banks)

    @property
    def effective_parameter_count(self) -> int:
        return sum(bank.effective_parameter_count for _, bank in self.banks)

    @property
    def state_scalar_count(self) -> int:
        return sum(bank.state_scalar_count for _, bank in self.banks)

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "format": DEVELOPMENTAL_SYNAPSE_BUNDLE_FORMAT,
            "version": DEVELOPMENTAL_SYNAPSE_BUNDLE_VERSION,
            "source_checkpoint_digest": self.source_checkpoint_digest,
            "source_checkpoint_format": self.source_checkpoint_format,
            "config_digest": self.config_digest,
            "owner_graph_digest": self.owner_graph_digest,
            "banks": {name: bank.to_payload() for name, bank in self.banks},
        }

    def to_payload(self) -> dict[str, Any]:
        unsigned = self._unsigned_payload()
        return {**unsigned, "bundle_digest": content_digest(unsigned)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DevelopmentalSynapseBundle:
        _verify_envelope(
            payload,
            DEVELOPMENTAL_SYNAPSE_BUNDLE_FORMAT,
            DEVELOPMENTAL_SYNAPSE_BUNDLE_VERSION,
        )
        _verify_digest(payload, "bundle_digest")
        banks = payload.get("banks")
        if not isinstance(banks, Mapping):
            raise DevelopmentalSynapseContractError("developmental bundle banks must be a mapping")
        return cls(
            source_checkpoint_digest=str(payload["source_checkpoint_digest"]),
            source_checkpoint_format=str(payload["source_checkpoint_format"]),
            config_digest=str(payload["config_digest"]),
            owner_graph_digest=str(payload["owner_graph_digest"]),
            banks=tuple(
                (str(name), DevelopmentalSynapseBank.from_payload(bank_payload))
                for name, bank_payload in sorted(banks.items(), key=lambda item: str(item[0]))
            ),
        )

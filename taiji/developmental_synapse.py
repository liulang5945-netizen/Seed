"""Checkpoint-compatible fast/slow synapse state for Taiji growth.

R1 is a migration boundary, not a learning rule.  It copies an existing F1
fixed-fan-in bank into ``slow_weight`` and creates a zero ``fast_delta`` plus
explicit developmental metadata.  The effective connection is
``slow_weight + fast_delta``; with the migration defaults this is exactly the
old connection.  The bundle is read-only until a later milestone defines and
gates fast adaptation and consolidation.
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
_SPARSE_STORAGE_FORMAT = "fixed-fan-in-v1"


class DevelopmentalSynapseContractError(ValueError):
    """Raised when a developmental synapse payload is malformed or unsafe."""


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DevelopmentalSynapseContractError(f"{name} must not be empty")


def _require_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
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

"""Declarative MCP client-capability activation proposal.

An activation proposal is the last E6-2 boundary before a future client-body
activation phase.  It records which shadow-validated candidate would be
eligible, under which MCP and Workbench capability snapshots, but contains no
executor, source, connection, credential value, or activation operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MCP_CLIENT_ACTIVATION_PROPOSAL_FORMAT = "seed-mcp-client-capability-activation-proposal-v1"
MCP_CLIENT_ACTIVATION_PROPOSAL_VERSION = 1
MCP_CLIENT_ACTIVATION_PROPOSAL_STATES = ("proposed", "rolled_back", "withdrawn")
_FORBIDDEN_KEYS = frozenset(
    {
        "command",
        "credential_value",
        "entrypoint",
        "exec",
        "executor",
        "executor_id",
        "module",
        "password",
        "script",
        "secret",
        "shell",
        "source",
        "source_path",
        "token",
        "url",
    }
)


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MCP activation proposal digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported MCP activation proposal value: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _assert_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_KEYS:
                raise ValueError("MCP activation proposal contains executable or secret fields")
            _assert_safe(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_safe(item)


@dataclass(frozen=True)
class McpClientCapabilityActivationProposal:
    """An auditable eligibility proposal, never an activated client organ."""

    candidate_digest: str
    shadow_record_digest: str
    mcp_registry_snapshot_id: str
    client_capability_snapshot_id: str
    parent_checkpoint_id: str
    approval_required: bool = False
    state: str = "proposed"
    format: str = MCP_CLIENT_ACTIVATION_PROPOSAL_FORMAT
    version: int = MCP_CLIENT_ACTIVATION_PROPOSAL_VERSION

    def __post_init__(self) -> None:
        if self.format != MCP_CLIENT_ACTIVATION_PROPOSAL_FORMAT:
            raise ValueError("unsupported MCP activation proposal format")
        if self.version != MCP_CLIENT_ACTIVATION_PROPOSAL_VERSION:
            raise ValueError("unsupported MCP activation proposal version")
        for name, value in (
            ("candidate_digest", self.candidate_digest),
            ("shadow_record_digest", self.shadow_record_digest),
            ("mcp_registry_snapshot_id", self.mcp_registry_snapshot_id),
            ("client_capability_snapshot_id", self.client_capability_snapshot_id),
            ("parent_checkpoint_id", self.parent_checkpoint_id),
        ):
            object.__setattr__(self, name, _required_text(value, name))
        if self.state not in MCP_CLIENT_ACTIVATION_PROPOSAL_STATES:
            raise ValueError("unsupported MCP activation proposal state")
        if not isinstance(self.approval_required, bool):
            raise TypeError("approval_required must be boolean")

    @property
    def proposal_id(self) -> str:
        return _digest(self._stable_payload())

    @property
    def proposal_digest(self) -> str:
        return _digest(self._identity_payload())

    def _stable_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "candidate_digest": self.candidate_digest,
            "shadow_record_digest": self.shadow_record_digest,
            "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
            "client_capability_snapshot_id": self.client_capability_snapshot_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "approval_required": self.approval_required,
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {**self._stable_payload(), "state": self.state}

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "proposal_id": self.proposal_id,
            "proposal_digest": self.proposal_digest,
            "activation": "proposal_only",
        }

    def rolled_back(
        self,
        *,
        shadow_record_digest: str | None = None,
    ) -> McpClientCapabilityActivationProposal:
        return McpClientCapabilityActivationProposal(
            candidate_digest=self.candidate_digest,
            shadow_record_digest=shadow_record_digest or self.shadow_record_digest,
            mcp_registry_snapshot_id=self.mcp_registry_snapshot_id,
            client_capability_snapshot_id=self.client_capability_snapshot_id,
            parent_checkpoint_id=self.parent_checkpoint_id,
            approval_required=self.approval_required,
            state="rolled_back",
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> McpClientCapabilityActivationProposal:
        if not isinstance(payload, Mapping):
            raise TypeError("MCP activation proposal must be an object")
        _assert_safe(payload)
        proposal = cls(
            candidate_digest=str(payload.get("candidate_digest", "")),
            shadow_record_digest=str(payload.get("shadow_record_digest", "")),
            mcp_registry_snapshot_id=str(payload.get("mcp_registry_snapshot_id", "")),
            client_capability_snapshot_id=str(payload.get("client_capability_snapshot_id", "")),
            parent_checkpoint_id=str(payload.get("parent_checkpoint_id", "")),
            approval_required=bool(payload.get("approval_required", False)),
            state=str(payload.get("state", "")),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", 0)),
        )
        if str(payload.get("proposal_id", "")) != proposal.proposal_id:
            raise ValueError("MCP activation proposal id mismatch")
        if str(payload.get("proposal_digest", "")) != proposal.proposal_digest:
            raise ValueError("MCP activation proposal digest mismatch")
        return proposal


__all__ = [
    "MCP_CLIENT_ACTIVATION_PROPOSAL_FORMAT",
    "MCP_CLIENT_ACTIVATION_PROPOSAL_STATES",
    "MCP_CLIENT_ACTIVATION_PROPOSAL_VERSION",
    "McpClientCapabilityActivationProposal",
]

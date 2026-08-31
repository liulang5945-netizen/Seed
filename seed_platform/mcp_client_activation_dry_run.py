"""Local, non-committing projection of an MCP activation proposal.

E6-3 uses the existing Seed client extension host as a shape validator.  It
builds a declarative manifest from a shadow-validated MCP candidate and calls
``prepare`` only; it never calls ``commit`` and therefore cannot activate a
client organ or execute an MCP tool.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .client_extension_host import (
    ClientExtensionHost,
    ClientPluginManifest,
    PreparedExtensionSnapshot,
)
from .mcp_capability_inheritance import McpCapabilityInheritanceCandidate
from .mcp_client_capability_activation import McpClientCapabilityActivationProposal

MCP_CLIENT_DRY_RUN_FORMAT = "seed-mcp-client-capability-activation-dry-run-v1"
MCP_CLIENT_DRY_RUN_VERSION = 1


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
            raise ValueError("MCP client dry-run digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported MCP client dry-run value: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_client_activation_manifest(
    candidate: McpCapabilityInheritanceCandidate,
    proposal: McpClientCapabilityActivationProposal,
    *,
    slots: Sequence[str] = ("command",),
) -> ClientPluginManifest:
    """Build a declarative local manifest without adding an executor field."""

    if not isinstance(candidate, McpCapabilityInheritanceCandidate):
        raise TypeError("client dry-run requires an MCP capability candidate")
    if not isinstance(proposal, McpClientCapabilityActivationProposal):
        raise TypeError("client dry-run requires an activation proposal")
    if proposal.state != "proposed":
        raise PermissionError("client activation dry-run requires a proposed activation")
    if proposal.candidate_digest != candidate.candidate_digest:
        raise ValueError("activation proposal candidate digest mismatch")
    if proposal.mcp_registry_snapshot_id != candidate.registry_snapshot_id:
        raise ValueError("activation proposal MCP snapshot mismatch")
    tool_ids = tuple(item.tool_id for item in candidate.tool_contracts)
    return ClientPluginManifest(
        plugin_id=f"mcp.shadow.{candidate.candidate_digest[:16]}",
        version=candidate.server_version,
        scope="workspace",
        slots=tuple(slots),
        capability_ids=tool_ids,
        disposer_id=f"seed.shadow.dispose.{candidate.candidate_digest[:16]}",
        disposer_version="1.0.0",
        metadata={
            "origin": "mcp_client_activation_dry_run",
            "candidate_digest": candidate.candidate_digest,
            "proposal_id": proposal.proposal_id,
            "shadow_record_digest": proposal.shadow_record_digest,
            "mcp_registry_snapshot_id": proposal.mcp_registry_snapshot_id,
            "client_capability_snapshot_id": proposal.client_capability_snapshot_id,
            "activation": "dry_run_only",
        },
    )


@dataclass(frozen=True)
class McpClientActivationDryRun:
    proposal_id: str
    candidate_digest: str
    expected_client_snapshot_id: str
    target_client_snapshot_id: str
    plugin_digest: str
    prepared_digest: str
    committed: bool = False
    format: str = MCP_CLIENT_DRY_RUN_FORMAT
    version: int = MCP_CLIENT_DRY_RUN_VERSION

    def __post_init__(self) -> None:
        if self.format != MCP_CLIENT_DRY_RUN_FORMAT or self.version != MCP_CLIENT_DRY_RUN_VERSION:
            raise ValueError("unsupported MCP client activation dry-run format")
        for name, value in (
            ("proposal_id", self.proposal_id),
            ("candidate_digest", self.candidate_digest),
            ("expected_client_snapshot_id", self.expected_client_snapshot_id),
            ("target_client_snapshot_id", self.target_client_snapshot_id),
            ("plugin_digest", self.plugin_digest),
            ("prepared_digest", self.prepared_digest),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} cannot be empty")
        if not isinstance(self.committed, bool):
            raise TypeError("dry-run committed flag must be boolean")
        if self.committed:
            raise ValueError("client activation dry-run cannot be committed")

    @property
    def dry_run_digest(self) -> str:
        return _digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "candidate_digest": self.candidate_digest,
            "expected_client_snapshot_id": self.expected_client_snapshot_id,
            "target_client_snapshot_id": self.target_client_snapshot_id,
            "plugin_digest": self.plugin_digest,
            "prepared_digest": self.prepared_digest,
            "committed": self.committed,
        }
        if include_digest:
            payload["dry_run_digest"] = self.dry_run_digest
        return payload


def run_client_activation_dry_run(
    host: ClientExtensionHost,
    candidate: McpCapabilityInheritanceCandidate,
    proposal: McpClientCapabilityActivationProposal,
    *,
    client_capability_snapshot_id: str,
    available_capabilities: Sequence[str],
    slots: Sequence[str] = ("command",),
) -> McpClientActivationDryRun:
    """Prepare a local manifest and deliberately stop before host.commit."""

    if not isinstance(host, ClientExtensionHost):
        raise TypeError("client dry-run requires a Seed client extension host")
    if client_capability_snapshot_id != proposal.client_capability_snapshot_id:
        raise ValueError("client capability snapshot does not match activation proposal")
    manifest = build_client_activation_manifest(candidate, proposal, slots=slots)
    prepared: PreparedExtensionSnapshot = host.prepare(
        (manifest,),
        capability_snapshot_id=client_capability_snapshot_id,
        available_capabilities=tuple(available_capabilities),
        states={manifest.plugin_id: {"lifecycle": "shadow"}},
    )
    return McpClientActivationDryRun(
        proposal_id=proposal.proposal_id,
        candidate_digest=candidate.candidate_digest,
        expected_client_snapshot_id=prepared.expected_current_snapshot_id,
        target_client_snapshot_id=prepared.target_snapshot.snapshot_id,
        plugin_digest=manifest.plugin_digest,
        prepared_digest=prepared.prepared_digest,
    )


__all__ = [
    "MCP_CLIENT_DRY_RUN_FORMAT",
    "MCP_CLIENT_DRY_RUN_VERSION",
    "McpClientActivationDryRun",
    "build_client_activation_manifest",
    "run_client_activation_dry_run",
]

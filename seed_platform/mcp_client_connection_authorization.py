"""Fail-closed authorization boundary before a real MCP client connection.

This module records an explicit, time-bounded authorization for a shadow-
validated client-capability proposal.  It stores only references and declared
scopes: no endpoint URL, credential value, executor, source, or connection
operation is representable.  Issuing or revoking an authorization never opens
an MCP connection or activates a client extension.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    preflight_inheritance_candidate,
)
from .mcp_client_capability_activation import McpClientCapabilityActivationProposal

MCP_CONNECTION_AUTHORIZATION_FORMAT = "seed-mcp-client-connection-authorization-v1"
MCP_CONNECTION_AUTHORIZATION_VERSION = 1
MCP_CONNECTION_AUTHORIZATION_STATES = ("authorized", "revoked")
MCP_CONNECTION_AUTHORIZATION_STORE_FORMAT = "seed-mcp-client-connection-authorization-store-v1"
MCP_CONNECTION_AUTHORIZATION_STORE_VERSION = 1
_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_FORBIDDEN_KEYS = frozenset(
    {
        "command",
        "credential_value",
        "endpoint",
        "entrypoint",
        "exec",
        "executor",
        "executor_id",
        "import_path",
        "module",
        "password",
        "path",
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
            raise ValueError("MCP authorization digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported MCP authorization value: {type(value).__name__}")


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


def _reference_tuple(value: Sequence[str] | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_required_text(item, f"{name} item") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} items must be unique")
    if any(not _REFERENCE_PATTERN.fullmatch(item) for item in normalized):
        raise ValueError(f"{name} must contain references, not secret values")
    return tuple(sorted(normalized))


def _assert_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_KEYS:
                raise ValueError("MCP authorization contains executable, endpoint, or secret fields")
            _assert_safe(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_safe(item)


def _epoch(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numeric epoch")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or normalized != int(normalized):
        raise ValueError(f"{name} must be a non-negative integer epoch")
    return int(normalized)


@dataclass(frozen=True)
class McpClientConnectionAuthorization:
    """Time-bounded authorization metadata with no connection capability."""

    proposal_id: str
    candidate_digest: str
    server_id: str
    server_version: str
    source_artifact_digest: str
    mcp_registry_snapshot_id: str
    client_capability_snapshot_id: str
    network_scopes: tuple[str, ...]
    credential_refs: tuple[str, ...]
    allowed_permissions: tuple[str, ...]
    approval_id: str
    issuer: str
    issued_at_epoch: int
    expires_at_epoch: int
    state: str = "authorized"
    revocation_reason: str = ""
    format: str = MCP_CONNECTION_AUTHORIZATION_FORMAT
    version: int = MCP_CONNECTION_AUTHORIZATION_VERSION

    def __post_init__(self) -> None:
        if self.format != MCP_CONNECTION_AUTHORIZATION_FORMAT:
            raise ValueError("unsupported MCP connection authorization format")
        if self.version != MCP_CONNECTION_AUTHORIZATION_VERSION:
            raise ValueError("unsupported MCP connection authorization version")
        for name, value in (
            ("proposal_id", self.proposal_id),
            ("candidate_digest", self.candidate_digest),
            ("server_id", self.server_id),
            ("server_version", self.server_version),
            ("source_artifact_digest", self.source_artifact_digest),
            ("mcp_registry_snapshot_id", self.mcp_registry_snapshot_id),
            ("client_capability_snapshot_id", self.client_capability_snapshot_id),
            ("approval_id", self.approval_id),
            ("issuer", self.issuer),
        ):
            object.__setattr__(self, name, _required_text(value, name))
        if self.state not in MCP_CONNECTION_AUTHORIZATION_STATES:
            raise ValueError("unsupported MCP connection authorization state")
        object.__setattr__(self, "network_scopes", _reference_tuple(self.network_scopes, "network_scopes"))
        object.__setattr__(self, "credential_refs", _reference_tuple(self.credential_refs, "credential_refs"))
        object.__setattr__(self, "allowed_permissions", _reference_tuple(self.allowed_permissions, "allowed_permissions"))
        issued = _epoch(self.issued_at_epoch, "issued_at_epoch")
        expires = _epoch(self.expires_at_epoch, "expires_at_epoch")
        if expires <= issued:
            raise ValueError("expires_at_epoch must be after issued_at_epoch")
        if self.state == "revoked":
            object.__setattr__(self, "revocation_reason", _required_text(self.revocation_reason, "revocation_reason"))
        else:
            object.__setattr__(self, "revocation_reason", str(self.revocation_reason).strip())
        object.__setattr__(self, "issued_at_epoch", issued)
        object.__setattr__(self, "expires_at_epoch", expires)

    @property
    def authorization_id(self) -> str:
        return _digest(self._stable_payload())

    @property
    def authorization_digest(self) -> str:
        return _digest(self._identity_payload())

    def is_valid(self, at_epoch: int) -> bool:
        point = _epoch(at_epoch, "at_epoch")
        return self.state == "authorized" and self.issued_at_epoch <= point < self.expires_at_epoch

    def _stable_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "candidate_digest": self.candidate_digest,
            "server_id": self.server_id,
            "server_version": self.server_version,
            "source_artifact_digest": self.source_artifact_digest,
            "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
            "client_capability_snapshot_id": self.client_capability_snapshot_id,
            "network_scopes": list(self.network_scopes),
            "credential_refs": list(self.credential_refs),
            "allowed_permissions": list(self.allowed_permissions),
            "approval_id": self.approval_id,
            "issuer": self.issuer,
            "issued_at_epoch": self.issued_at_epoch,
            "expires_at_epoch": self.expires_at_epoch,
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {
            **self._stable_payload(),
            "state": self.state,
            "revocation_reason": self.revocation_reason,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "authorization_id": self.authorization_id,
            "authorization_digest": self.authorization_digest,
            "connection_attempted": False,
        }

    def revoked(self, reason: str) -> McpClientConnectionAuthorization:
        return McpClientConnectionAuthorization(
            **self._stable_payload(),
            state="revoked",
            revocation_reason=reason,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> McpClientConnectionAuthorization:
        if not isinstance(payload, Mapping):
            raise TypeError("MCP connection authorization must be an object")
        _assert_safe(payload)
        if payload.get("connection_attempted", False) is not False:
            raise ValueError("MCP connection authorization cannot record a connection attempt")
        authorization = cls(
            proposal_id=str(payload.get("proposal_id", "")),
            candidate_digest=str(payload.get("candidate_digest", "")),
            server_id=str(payload.get("server_id", "")),
            server_version=str(payload.get("server_version", "")),
            source_artifact_digest=str(payload.get("source_artifact_digest", "")),
            mcp_registry_snapshot_id=str(payload.get("mcp_registry_snapshot_id", "")),
            client_capability_snapshot_id=str(payload.get("client_capability_snapshot_id", "")),
            network_scopes=tuple(payload.get("network_scopes", ())),
            credential_refs=tuple(payload.get("credential_refs", ())),
            allowed_permissions=tuple(payload.get("allowed_permissions", ())),
            approval_id=str(payload.get("approval_id", "")),
            issuer=str(payload.get("issuer", "")),
            issued_at_epoch=payload.get("issued_at_epoch", -1),
            expires_at_epoch=payload.get("expires_at_epoch", -1),
            state=str(payload.get("state", "")),
            revocation_reason=str(payload.get("revocation_reason", "")),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", 0)),
        )
        if str(payload.get("authorization_id", "")) != authorization.authorization_id:
            raise ValueError("MCP connection authorization id mismatch")
        if str(payload.get("authorization_digest", "")) != authorization.authorization_digest:
            raise ValueError("MCP connection authorization digest mismatch")
        return authorization


@dataclass(frozen=True)
class McpConnectionAuthorizationDecision:
    passed: bool
    decision: str
    reason_code: str
    proposal_id: str
    candidate_digest: str
    authorization_id: str = ""
    connection_attempted: bool = False
    authorization: McpClientConnectionAuthorization | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "proposal_id": self.proposal_id,
            "candidate_digest": self.candidate_digest,
            "authorization_id": self.authorization_id,
            "connection_attempted": self.connection_attempted,
            "authorization": self.authorization.to_payload() if self.authorization else None,
        }


def authorize_mcp_client_connection(
    candidate: McpCapabilityInheritanceCandidate,
    proposal: McpClientCapabilityActivationProposal,
    policy: McpCapabilityInheritancePolicy,
    *,
    current_mcp_registry_snapshot_id: str,
    current_client_capability_snapshot_id: str,
    network_scopes: Sequence[str] = (),
    credential_refs: Sequence[str] = (),
    approval_id: str,
    issuer: str,
    issued_at_epoch: int,
    expires_at_epoch: int,
    max_lifetime_seconds: int = 3_600,
) -> McpConnectionAuthorizationDecision:
    """Issue only explicit authorization metadata; never connect or activate."""

    if not isinstance(candidate, McpCapabilityInheritanceCandidate):
        raise TypeError("MCP authorization requires a capability candidate")
    if not isinstance(proposal, McpClientCapabilityActivationProposal):
        raise TypeError("MCP authorization requires an activation proposal")
    if not isinstance(policy, McpCapabilityInheritancePolicy):
        raise TypeError("MCP authorization requires an explicit policy")
    proposal_id = proposal.proposal_id
    digest = candidate.candidate_digest

    def deny(reason: str) -> McpConnectionAuthorizationDecision:
        return McpConnectionAuthorizationDecision(
            False,
            "deny",
            reason,
            proposal_id,
            digest,
        )

    if proposal.state != "proposed":
        return deny("activation_proposal_not_proposed")
    if proposal.candidate_digest != digest:
        return deny("activation_proposal_candidate_mismatch")
    if proposal.mcp_registry_snapshot_id != str(current_mcp_registry_snapshot_id):
        return deny("stale_mcp_registry")
    if proposal.client_capability_snapshot_id != str(current_client_capability_snapshot_id):
        return deny("stale_client_capability_snapshot")
    preflight = preflight_inheritance_candidate(
        candidate,
        policy,
        current_registry_snapshot_id=current_mcp_registry_snapshot_id,
    )
    if not preflight.passed:
        return deny(preflight.reason_code)
    requested_network = _reference_tuple(network_scopes, "network_scopes")
    requested_credentials = _reference_tuple(credential_refs, "credential_refs")
    if not set(requested_network).issubset(candidate.network_scopes):
        return deny("network_scope_not_declared_by_candidate")
    if not set(requested_network).issubset(policy.allowed_network_scopes):
        return deny("network_scope_not_allowed")
    if not set(requested_credentials).issubset(candidate.credential_refs):
        return deny("credential_ref_not_declared_by_candidate")
    if requested_credentials and not policy.allow_credentials:
        return deny("credentials_not_allowed")
    approval = str(approval_id).strip()
    if not approval:
        return deny("explicit_approval_required")
    issuer_value = str(issuer).strip()
    if not issuer_value:
        return deny("authorization_issuer_required")
    issued = _epoch(issued_at_epoch, "issued_at_epoch")
    expires = _epoch(expires_at_epoch, "expires_at_epoch")
    if expires <= issued:
        return deny("authorization_expiry_invalid")
    if expires - issued > int(max_lifetime_seconds):
        return deny("authorization_lifetime_exceeded")
    authorization = McpClientConnectionAuthorization(
        proposal_id=proposal_id,
        candidate_digest=digest,
        server_id=candidate.server_id,
        server_version=candidate.server_version,
        source_artifact_digest=candidate.source_artifact_digest,
        mcp_registry_snapshot_id=current_mcp_registry_snapshot_id,
        client_capability_snapshot_id=current_client_capability_snapshot_id,
        network_scopes=requested_network,
        credential_refs=requested_credentials,
        allowed_permissions=tuple(
            permission
            for contract in candidate.tool_contracts
            for permission in contract.permissions
        ),
        approval_id=approval,
        issuer=issuer_value,
        issued_at_epoch=issued,
        expires_at_epoch=expires,
    )
    return McpConnectionAuthorizationDecision(
        True,
        "authorized_for_connection",
        "explicit_authorization_recorded",
        proposal_id,
        digest,
        authorization_id=authorization.authorization_id,
        authorization=authorization,
    )


class McpClientConnectionAuthorizationStore:
    """Checkpointable authorization records with explicit revocation."""

    def __init__(
        self,
        *,
        mcp_registry_snapshot_id: str,
        client_capability_snapshot_id: str,
    ) -> None:
        self._mcp_registry_snapshot_id = _required_text(
            mcp_registry_snapshot_id, "mcp_registry_snapshot_id"
        )
        self._client_capability_snapshot_id = _required_text(
            client_capability_snapshot_id,
            "client_capability_snapshot_id",
        )
        self.revision = 0
        self._records: dict[str, McpClientConnectionAuthorization] = {}
        self._events: list[Mapping[str, Any]] = []

    @property
    def mcp_registry_snapshot_id(self) -> str:
        return self._mcp_registry_snapshot_id

    @property
    def client_capability_snapshot_id(self) -> str:
        return self._client_capability_snapshot_id

    @property
    def records(self) -> tuple[McpClientConnectionAuthorization, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def snapshot_id(self) -> str:
        return _digest(
            {
                "format": MCP_CONNECTION_AUTHORIZATION_STORE_FORMAT,
                "version": MCP_CONNECTION_AUTHORIZATION_STORE_VERSION,
                "revision": self.revision,
                "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
                "client_capability_snapshot_id": self.client_capability_snapshot_id,
                "records": [record.authorization_digest for record in self.records],
                "events": [dict(event) for event in self._events],
            }
        )

    def get(self, authorization_id: str) -> McpClientConnectionAuthorization | None:
        return self._records.get(str(authorization_id).strip())

    def issue(self, authorization: McpClientConnectionAuthorization) -> McpClientConnectionAuthorization:
        if not isinstance(authorization, McpClientConnectionAuthorization):
            raise TypeError("authorization store accepts authorization records only")
        if authorization.state != "authorized":
            raise ValueError("only authorized records can be issued")
        if authorization.mcp_registry_snapshot_id != self.mcp_registry_snapshot_id:
            raise ValueError("authorization MCP registry snapshot is stale")
        if authorization.client_capability_snapshot_id != self.client_capability_snapshot_id:
            raise ValueError("authorization client capability snapshot is stale")
        existing = self._records.get(authorization.authorization_id)
        if existing is not None:
            if existing.authorization_digest != authorization.authorization_digest:
                raise ValueError("authorization identity conflicts with an existing record")
            if existing.state == "revoked":
                raise PermissionError("authorization is terminal after revocation")
            return existing
        next_revision = self.revision + 1
        self._records[authorization.authorization_id] = authorization
        self._events.append(
            {
                "event_id": f"issue:{authorization.authorization_id}:{next_revision}",
                "event_kind": "authorization_issued",
                "authorization_id": authorization.authorization_id,
                "authorization_digest": authorization.authorization_digest,
                "revision": next_revision,
            }
        )
        self.revision = next_revision
        return authorization

    def revoke(self, authorization_id: str, *, reason: str) -> McpClientConnectionAuthorization:
        normalized = _required_text(authorization_id, "authorization_id")
        current = self._records.get(normalized)
        if current is None:
            raise KeyError("unknown MCP connection authorization")
        if current.state == "revoked":
            return current
        revoked = current.revoked(reason)
        next_revision = self.revision + 1
        self._records[normalized] = revoked
        self._events.append(
            {
                "event_id": f"revoke:{normalized}:{next_revision}",
                "event_kind": "authorization_revoked",
                "authorization_id": normalized,
                "authorization_digest": revoked.authorization_digest,
                "revision": next_revision,
            }
        )
        self.revision = next_revision
        return revoked

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": MCP_CONNECTION_AUTHORIZATION_STORE_FORMAT,
            "version": MCP_CONNECTION_AUTHORIZATION_STORE_VERSION,
            "revision": self.revision,
            "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
            "client_capability_snapshot_id": self.client_capability_snapshot_id,
            "snapshot_id": self.snapshot_id,
            "records": [record.to_payload() for record in self.records],
            "events": [dict(event) for event in self._events],
        }
        payload["checkpoint_digest"] = _digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> McpClientConnectionAuthorizationStore:
        if payload.get("format") != MCP_CONNECTION_AUTHORIZATION_STORE_FORMAT:
            raise ValueError("unsupported MCP authorization store format")
        if int(payload.get("version", 0)) != MCP_CONNECTION_AUTHORIZATION_STORE_VERSION:
            raise ValueError("unsupported MCP authorization store version")
        expected = _digest({key: value for key, value in payload.items() if key != "checkpoint_digest"})
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("MCP authorization store checkpoint digest mismatch")
        store = cls(
            mcp_registry_snapshot_id=str(payload.get("mcp_registry_snapshot_id", "")),
            client_capability_snapshot_id=str(payload.get("client_capability_snapshot_id", "")),
        )
        raw_records = payload.get("records", ())
        if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Sequence):
            raise ValueError("MCP authorization store records must be a sequence")
        for raw_record in raw_records:
            record = McpClientConnectionAuthorization.from_payload(raw_record)
            if record.authorization_id in store._records:
                raise ValueError("duplicate MCP connection authorization")
            if record.mcp_registry_snapshot_id != store.mcp_registry_snapshot_id:
                raise ValueError("MCP authorization record snapshot mismatch")
            if record.client_capability_snapshot_id != store.client_capability_snapshot_id:
                raise ValueError("client authorization record snapshot mismatch")
            store._records[record.authorization_id] = record
        raw_events = payload.get("events", ())
        if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
            raise ValueError("MCP authorization store events must be a sequence")
        store._events = [dict(event) for event in raw_events]
        store.revision = int(payload.get("revision", -1))
        if store.revision != len(store._events):
            raise ValueError("MCP authorization store revision mismatch")
        if str(payload.get("snapshot_id", "")) != store.snapshot_id:
            raise ValueError("MCP authorization store snapshot mismatch")
        return store


__all__ = [
    "MCP_CONNECTION_AUTHORIZATION_FORMAT",
    "MCP_CONNECTION_AUTHORIZATION_STATES",
    "MCP_CONNECTION_AUTHORIZATION_STORE_FORMAT",
    "MCP_CONNECTION_AUTHORIZATION_STORE_VERSION",
    "MCP_CONNECTION_AUTHORIZATION_VERSION",
    "McpClientConnectionAuthorization",
    "McpClientConnectionAuthorizationStore",
    "McpConnectionAuthorizationDecision",
    "authorize_mcp_client_connection",
]

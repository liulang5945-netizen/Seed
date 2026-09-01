"""Declarative target binding before a real MCP client connection.

E6-5 binds an already-authorized capability to an explicit target identity and
transport class.  It intentionally contains no endpoint, command, executor,
credential value, source, socket, or connection result.  Binding and
revocation are metadata-only operations; a later phase must still perform a
separate, user-authorized connection canary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .mcp_client_connection_authorization import McpClientConnectionAuthorization

MCP_CONNECTION_TARGET_FORMAT = "seed-mcp-client-connection-target-v1"
MCP_CONNECTION_TARGET_VERSION = 1
MCP_CONNECTION_TARGET_STATES = ("bound", "revoked")
MCP_CONNECTION_TRANSPORTS = ("stdio", "sse", "streamable_http")
MCP_CONNECTION_TARGET_STORE_FORMAT = "seed-mcp-client-connection-target-store-v1"
MCP_CONNECTION_TARGET_STORE_VERSION = 1
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
            raise ValueError("MCP target digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported MCP target value: {type(value).__name__}")


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
                raise ValueError("MCP target contains executable, endpoint, or secret fields")
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
class McpClientConnectionTarget:
    """A target/transport declaration with no operation capability."""

    target_id: str
    target_version: str
    transport: str
    authorization_id: str
    authorization_issued_at_epoch: int
    authorization_expires_at_epoch: int
    source_artifact_digest: str
    mcp_registry_snapshot_id: str
    client_capability_snapshot_id: str
    network_scopes: tuple[str, ...]
    credential_refs: tuple[str, ...]
    allowed_permissions: tuple[str, ...]
    connection_owner_id: str
    credential_owner_id: str
    approver_id: str
    state: str = "bound"
    revocation_reason: str = ""
    format: str = MCP_CONNECTION_TARGET_FORMAT
    version: int = MCP_CONNECTION_TARGET_VERSION

    def __post_init__(self) -> None:
        if self.format != MCP_CONNECTION_TARGET_FORMAT:
            raise ValueError("unsupported MCP connection target format")
        if self.version != MCP_CONNECTION_TARGET_VERSION:
            raise ValueError("unsupported MCP connection target version")
        for name, value in (
            ("target_id", self.target_id),
            ("target_version", self.target_version),
            ("authorization_id", self.authorization_id),
            ("source_artifact_digest", self.source_artifact_digest),
            ("mcp_registry_snapshot_id", self.mcp_registry_snapshot_id),
            ("client_capability_snapshot_id", self.client_capability_snapshot_id),
            ("connection_owner_id", self.connection_owner_id),
            ("credential_owner_id", self.credential_owner_id),
            ("approver_id", self.approver_id),
        ):
            object.__setattr__(self, name, _required_text(value, name))
        if self.transport not in MCP_CONNECTION_TRANSPORTS:
            raise ValueError("unsupported MCP connection transport")
        if self.state not in MCP_CONNECTION_TARGET_STATES:
            raise ValueError("unsupported MCP connection target state")
        object.__setattr__(self, "network_scopes", _reference_tuple(self.network_scopes, "network_scopes"))
        object.__setattr__(self, "credential_refs", _reference_tuple(self.credential_refs, "credential_refs"))
        object.__setattr__(self, "allowed_permissions", _reference_tuple(self.allowed_permissions, "allowed_permissions"))
        issued = _epoch(self.authorization_issued_at_epoch, "authorization_issued_at_epoch")
        expires = _epoch(self.authorization_expires_at_epoch, "authorization_expires_at_epoch")
        if expires <= issued:
            raise ValueError("authorization_expires_at_epoch must be after authorization_issued_at_epoch")
        object.__setattr__(self, "authorization_issued_at_epoch", issued)
        object.__setattr__(self, "authorization_expires_at_epoch", expires)
        if self.state == "revoked":
            object.__setattr__(self, "revocation_reason", _required_text(self.revocation_reason, "revocation_reason"))
        else:
            object.__setattr__(self, "revocation_reason", str(self.revocation_reason).strip())

    @property
    def binding_id(self) -> str:
        return _digest(self._stable_payload())

    @property
    def binding_digest(self) -> str:
        return _digest(self._identity_payload())

    def is_valid(self, at_epoch: int) -> bool:
        point = _epoch(at_epoch, "at_epoch")
        return (
            self.state == "bound"
            and self.authorization_issued_at_epoch <= point < self.authorization_expires_at_epoch
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "binding_id": self.binding_id,
            "binding_digest": self.binding_digest,
            "connection_attempted": False,
        }

    def _stable_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "target_id": self.target_id,
            "target_version": self.target_version,
            "transport": self.transport,
            "authorization_id": self.authorization_id,
            "authorization_issued_at_epoch": self.authorization_issued_at_epoch,
            "authorization_expires_at_epoch": self.authorization_expires_at_epoch,
            "source_artifact_digest": self.source_artifact_digest,
            "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
            "client_capability_snapshot_id": self.client_capability_snapshot_id,
            "network_scopes": list(self.network_scopes),
            "credential_refs": list(self.credential_refs),
            "allowed_permissions": list(self.allowed_permissions),
            "connection_owner_id": self.connection_owner_id,
            "credential_owner_id": self.credential_owner_id,
            "approver_id": self.approver_id,
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {
            **self._stable_payload(),
            "state": self.state,
            "revocation_reason": self.revocation_reason,
        }

    def revoked(self, reason: str) -> McpClientConnectionTarget:
        return McpClientConnectionTarget(
            **self._stable_payload(),
            state="revoked",
            revocation_reason=reason,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> McpClientConnectionTarget:
        if not isinstance(payload, Mapping):
            raise TypeError("MCP connection target must be an object")
        _assert_safe(payload)
        if payload.get("connection_attempted", False) is not False:
            raise ValueError("MCP connection target cannot record a connection attempt")
        target = cls(
            target_id=str(payload.get("target_id", "")),
            target_version=str(payload.get("target_version", "")),
            transport=str(payload.get("transport", "")),
            authorization_id=str(payload.get("authorization_id", "")),
            authorization_issued_at_epoch=payload.get("authorization_issued_at_epoch", -1),
            authorization_expires_at_epoch=payload.get("authorization_expires_at_epoch", -1),
            source_artifact_digest=str(payload.get("source_artifact_digest", "")),
            mcp_registry_snapshot_id=str(payload.get("mcp_registry_snapshot_id", "")),
            client_capability_snapshot_id=str(payload.get("client_capability_snapshot_id", "")),
            network_scopes=tuple(payload.get("network_scopes", ())),
            credential_refs=tuple(payload.get("credential_refs", ())),
            allowed_permissions=tuple(payload.get("allowed_permissions", ())),
            connection_owner_id=str(payload.get("connection_owner_id", "")),
            credential_owner_id=str(payload.get("credential_owner_id", "")),
            approver_id=str(payload.get("approver_id", "")),
            state=str(payload.get("state", "")),
            revocation_reason=str(payload.get("revocation_reason", "")),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", 0)),
        )
        if str(payload.get("binding_id", "")) != target.binding_id:
            raise ValueError("MCP connection target id mismatch")
        if str(payload.get("binding_digest", "")) != target.binding_digest:
            raise ValueError("MCP connection target digest mismatch")
        return target


@dataclass(frozen=True)
class McpConnectionTargetBindingDecision:
    passed: bool
    decision: str
    reason_code: str
    authorization_id: str
    binding_id: str = ""
    connection_attempted: bool = False
    target: McpClientConnectionTarget | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "authorization_id": self.authorization_id,
            "binding_id": self.binding_id,
            "connection_attempted": self.connection_attempted,
            "target": self.target.to_payload() if self.target else None,
        }


def bind_mcp_client_connection_target(
    authorization: McpClientConnectionAuthorization,
    *,
    target_id: str,
    target_version: str,
    transport: str,
    mcp_registry_snapshot_id: str,
    client_capability_snapshot_id: str,
    network_scopes: Sequence[str] = (),
    credential_refs: Sequence[str] = (),
    allowed_permissions: Sequence[str] = (),
    connection_owner_id: str,
    credential_owner_id: str,
    approver_id: str,
    at_epoch: int,
) -> McpConnectionTargetBindingDecision:
    """Bind a declared target to an authorization without performing I/O."""

    if not isinstance(authorization, McpClientConnectionAuthorization):
        raise TypeError("MCP target binding requires a connection authorization")
    authorization_id = authorization.authorization_id

    def deny(reason: str) -> McpConnectionTargetBindingDecision:
        return McpConnectionTargetBindingDecision(
            False,
            "deny",
            reason,
            authorization_id,
        )

    if authorization.state != "authorized":
        return deny("authorization_not_active")
    if not authorization.is_valid(at_epoch):
        return deny("authorization_expired")
    normalized_target = _required_text(target_id, "target_id")
    normalized_version = _required_text(target_version, "target_version")
    if normalized_target != authorization.server_id:
        return deny("target_id_mismatch")
    if normalized_version != authorization.server_version:
        return deny("target_version_mismatch")
    if transport not in MCP_CONNECTION_TRANSPORTS:
        return deny("transport_not_allowed")
    if str(mcp_registry_snapshot_id) != authorization.mcp_registry_snapshot_id:
        return deny("stale_mcp_registry")
    if str(client_capability_snapshot_id) != authorization.client_capability_snapshot_id:
        return deny("stale_client_capability_snapshot")
    requested_network = _reference_tuple(network_scopes, "network_scopes")
    requested_credentials = _reference_tuple(credential_refs, "credential_refs")
    requested_permissions = _reference_tuple(allowed_permissions, "allowed_permissions")
    if not set(requested_network).issubset(authorization.network_scopes):
        return deny("network_scope_exceeds_authorization")
    if not set(requested_credentials).issubset(authorization.credential_refs):
        return deny("credential_ref_exceeds_authorization")
    if not set(requested_permissions).issubset(authorization.allowed_permissions):
        return deny("permission_exceeds_authorization")
    connection_owner = _required_text(connection_owner_id, "connection_owner_id")
    credential_owner = _required_text(credential_owner_id, "credential_owner_id")
    approver = _required_text(approver_id, "approver_id")
    target = McpClientConnectionTarget(
        target_id=normalized_target,
        target_version=normalized_version,
        transport=transport,
        authorization_id=authorization_id,
        authorization_issued_at_epoch=authorization.issued_at_epoch,
        authorization_expires_at_epoch=authorization.expires_at_epoch,
        source_artifact_digest=authorization.source_artifact_digest,
        mcp_registry_snapshot_id=authorization.mcp_registry_snapshot_id,
        client_capability_snapshot_id=authorization.client_capability_snapshot_id,
        network_scopes=requested_network,
        credential_refs=requested_credentials,
        allowed_permissions=requested_permissions,
        connection_owner_id=connection_owner,
        credential_owner_id=credential_owner,
        approver_id=approver,
    )
    return McpConnectionTargetBindingDecision(
        True,
        "target_bound_for_connection",
        "target_declaration_recorded",
        authorization_id,
        binding_id=target.binding_id,
        target=target,
    )


class McpClientConnectionTargetStore:
    """Checkpointable target declarations with authorization revocation fan-out."""

    def __init__(
        self,
        *,
        mcp_registry_snapshot_id: str,
        client_capability_snapshot_id: str,
    ) -> None:
        self._mcp_registry_snapshot_id = _required_text(mcp_registry_snapshot_id, "mcp_registry_snapshot_id")
        self._client_capability_snapshot_id = _required_text(
            client_capability_snapshot_id,
            "client_capability_snapshot_id",
        )
        self.revision = 0
        self._records: dict[str, McpClientConnectionTarget] = {}
        self._events: list[Mapping[str, Any]] = []

    @property
    def mcp_registry_snapshot_id(self) -> str:
        return self._mcp_registry_snapshot_id

    @property
    def client_capability_snapshot_id(self) -> str:
        return self._client_capability_snapshot_id

    @property
    def records(self) -> tuple[McpClientConnectionTarget, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def snapshot_id(self) -> str:
        return _digest(
            {
                "format": MCP_CONNECTION_TARGET_STORE_FORMAT,
                "version": MCP_CONNECTION_TARGET_STORE_VERSION,
                "revision": self.revision,
                "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
                "client_capability_snapshot_id": self.client_capability_snapshot_id,
                "records": [record.binding_digest for record in self.records],
                "events": [dict(event) for event in self._events],
            }
        )

    def get(self, binding_id: str) -> McpClientConnectionTarget | None:
        return self._records.get(str(binding_id).strip())

    def issue(self, target: McpClientConnectionTarget) -> McpClientConnectionTarget:
        if not isinstance(target, McpClientConnectionTarget):
            raise TypeError("target store accepts target declarations only")
        if target.state != "bound":
            raise ValueError("only bound target declarations can be issued")
        if target.mcp_registry_snapshot_id != self.mcp_registry_snapshot_id:
            raise ValueError("target MCP registry snapshot is stale")
        if target.client_capability_snapshot_id != self.client_capability_snapshot_id:
            raise ValueError("target client capability snapshot is stale")
        existing = self._records.get(target.binding_id)
        if existing is not None:
            if existing.binding_digest != target.binding_digest:
                raise ValueError("target identity conflicts with an existing record")
            if existing.state == "revoked":
                raise PermissionError("target declaration is terminal after revocation")
            return existing
        next_revision = self.revision + 1
        self._records[target.binding_id] = target
        self._events.append(
            {
                "event_id": f"bind:{target.binding_id}:{next_revision}",
                "event_kind": "target_bound",
                "binding_id": target.binding_id,
                "authorization_id": target.authorization_id,
                "binding_digest": target.binding_digest,
                "revision": next_revision,
            }
        )
        self.revision = next_revision
        return target

    def revoke(self, binding_id: str, *, reason: str) -> McpClientConnectionTarget:
        normalized = _required_text(binding_id, "binding_id")
        current = self._records.get(normalized)
        if current is None:
            raise KeyError("unknown MCP connection target")
        if current.state == "revoked":
            return current
        revoked = current.revoked(reason)
        next_revision = self.revision + 1
        self._records[normalized] = revoked
        self._events.append(
            {
                "event_id": f"revoke:{normalized}:{next_revision}",
                "event_kind": "target_revoked",
                "binding_id": normalized,
                "authorization_id": revoked.authorization_id,
                "binding_digest": revoked.binding_digest,
                "revision": next_revision,
            }
        )
        self.revision = next_revision
        return revoked

    def revoke_for_authorization(self, authorization_id: str, *, reason: str) -> tuple[str, ...]:
        normalized = _required_text(authorization_id, "authorization_id")
        revoked_ids: list[str] = []
        for target in self.records:
            if target.authorization_id != normalized or target.state == "revoked":
                continue
            self.revoke(target.binding_id, reason=reason)
            revoked_ids.append(target.binding_id)
        return tuple(revoked_ids)

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": MCP_CONNECTION_TARGET_STORE_FORMAT,
            "version": MCP_CONNECTION_TARGET_STORE_VERSION,
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
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> McpClientConnectionTargetStore:
        if payload.get("format") != MCP_CONNECTION_TARGET_STORE_FORMAT:
            raise ValueError("unsupported MCP target store format")
        if int(payload.get("version", 0)) != MCP_CONNECTION_TARGET_STORE_VERSION:
            raise ValueError("unsupported MCP target store version")
        expected = _digest({key: value for key, value in payload.items() if key != "checkpoint_digest"})
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("MCP target store checkpoint digest mismatch")
        store = cls(
            mcp_registry_snapshot_id=str(payload.get("mcp_registry_snapshot_id", "")),
            client_capability_snapshot_id=str(payload.get("client_capability_snapshot_id", "")),
        )
        raw_records = payload.get("records", ())
        if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Sequence):
            raise ValueError("MCP target store records must be a sequence")
        for raw_record in raw_records:
            target = McpClientConnectionTarget.from_payload(raw_record)
            if target.binding_id in store._records:
                raise ValueError("duplicate MCP connection target")
            if target.mcp_registry_snapshot_id != store.mcp_registry_snapshot_id:
                raise ValueError("MCP target record snapshot mismatch")
            if target.client_capability_snapshot_id != store.client_capability_snapshot_id:
                raise ValueError("client target record snapshot mismatch")
            store._records[target.binding_id] = target
        raw_events = payload.get("events", ())
        if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
            raise ValueError("MCP target store events must be a sequence")
        store._events = [dict(event) for event in raw_events]
        store.revision = int(payload.get("revision", -1))
        if store.revision != len(store._events):
            raise ValueError("MCP target store revision mismatch")
        if str(payload.get("snapshot_id", "")) != store.snapshot_id:
            raise ValueError("MCP target store snapshot mismatch")
        return store


__all__ = [
    "MCP_CONNECTION_TARGET_FORMAT",
    "MCP_CONNECTION_TARGET_STATES",
    "MCP_CONNECTION_TARGET_STORE_FORMAT",
    "MCP_CONNECTION_TARGET_STORE_VERSION",
    "MCP_CONNECTION_TARGET_VERSION",
    "MCP_CONNECTION_TRANSPORTS",
    "McpClientConnectionTarget",
    "McpClientConnectionTargetStore",
    "McpConnectionTargetBindingDecision",
    "bind_mcp_client_connection_target",
]

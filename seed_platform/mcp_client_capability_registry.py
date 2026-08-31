"""Seed-owned shadow registry for MCP client-capability candidates.

E6-1 projects the declarative E6-0 candidate into a durable lifecycle record.
The registry binds every record to the current Seed-owned MCP snapshot, accepts
only digest-only shadow observations, and keeps the candidate outside both the
Workbench executor registry and Taiji cognition.  There is intentionally no
active state in this phase: a later phase must add a separately gated client
activation contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritanceDecision,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
    evaluate_mcp_capability_shadow,
    preflight_inheritance_candidate,
)

MCP_CLIENT_SHADOW_REGISTRY_FORMAT = "seed-mcp-client-capability-shadow-registry-v1"
MCP_CLIENT_SHADOW_REGISTRY_VERSION = 1
MCP_CLIENT_SHADOW_STATES = (
    "proposed",
    "shadow_pending",
    "shadow_validated",
    "rejected",
    "rolled_back",
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
            raise ValueError("MCP shadow registry digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported MCP shadow registry value: {type(value).__name__}")


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


def _decision_from_payload(payload: Mapping[str, Any]) -> McpCapabilityInheritanceDecision:
    if not isinstance(payload, Mapping):
        raise TypeError("MCP client capability decision must be an object")
    return McpCapabilityInheritanceDecision(
        passed=bool(payload.get("passed", False)),
        decision=str(payload.get("decision", "")),
        reason_code=str(payload.get("reason_code", "")),
        candidate_digest=str(payload.get("candidate_digest", "")),
        policy_digest=str(payload.get("policy_digest", "")),
        approval_required=bool(payload.get("approval_required", False)),
        shadow_required=bool(payload.get("shadow_required", True)),
    )


@dataclass(frozen=True)
class McpClientCapabilityShadowRecord:
    """One candidate's declarative shadow lifecycle, never an active organ."""

    candidate: McpCapabilityInheritanceCandidate
    policy: McpCapabilityInheritancePolicy
    registry_snapshot_id: str
    state: str
    decision: McpCapabilityInheritanceDecision
    observation: McpCapabilityShadowObservation | None = None
    events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.state not in MCP_CLIENT_SHADOW_STATES:
            raise ValueError("unsupported MCP client capability shadow state")
        snapshot_id = _required_text(self.registry_snapshot_id, "registry_snapshot_id")
        if snapshot_id != self.candidate.registry_snapshot_id:
            raise ValueError("shadow record registry snapshot must match candidate")
        if self.decision.candidate_digest != self.candidate.candidate_digest:
            raise ValueError("shadow record decision candidate mismatch")
        if self.decision.policy_digest != self.policy.policy_digest:
            raise ValueError("shadow record decision policy mismatch")
        if self.observation is not None:
            if self.observation.candidate_digest != self.candidate.candidate_digest:
                raise ValueError("shadow observation candidate mismatch")
            if self.observation.registry_snapshot_id != snapshot_id:
                raise ValueError("shadow observation registry snapshot mismatch")
        object.__setattr__(self, "registry_snapshot_id", snapshot_id)
        object.__setattr__(self, "events", tuple(dict(event) for event in self.events))

    @property
    def candidate_digest(self) -> str:
        return self.candidate.candidate_digest

    @property
    def policy_digest(self) -> str:
        return self.policy.policy_digest

    @property
    def record_digest(self) -> str:
        return _digest(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": MCP_CLIENT_SHADOW_REGISTRY_FORMAT,
            "version": MCP_CLIENT_SHADOW_REGISTRY_VERSION,
            "candidate_digest": self.candidate_digest,
            "policy_digest": self.policy_digest,
            "registry_snapshot_id": self.registry_snapshot_id,
            "state": self.state,
            "decision": self.decision.to_payload(),
            "observation": self.observation.to_payload() if self.observation else None,
            "events": [dict(event) for event in self.events],
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "candidate": self.candidate.to_payload(),
            "policy": self.policy.to_payload(),
            "record_digest": self.record_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> McpClientCapabilityShadowRecord:
        if payload.get("format") != MCP_CLIENT_SHADOW_REGISTRY_FORMAT:
            raise ValueError("unsupported MCP client shadow record format")
        if int(payload.get("version", 0)) != MCP_CLIENT_SHADOW_REGISTRY_VERSION:
            raise ValueError("unsupported MCP client shadow record version")
        candidate = McpCapabilityInheritanceCandidate.from_payload(payload.get("candidate") or {})
        policy_payload = payload.get("policy") or {}
        if "policy_digest" in policy_payload:
            expected_policy_digest = str(policy_payload["policy_digest"])
        else:
            expected_policy_digest = ""
        policy = McpCapabilityInheritancePolicy(
            allowed_server_ids=tuple(policy_payload.get("allowed_server_ids", ())),
            allowed_risks=tuple(policy_payload.get("allowed_risks", ())),
            allowed_permissions=tuple(policy_payload.get("allowed_permissions", ())),
            allowed_network_scopes=tuple(policy_payload.get("allowed_network_scopes", ())),
            allow_credentials=bool(policy_payload.get("allow_credentials", False)),
            max_tools=int(policy_payload.get("max_tools", 16)),
            max_timeout_seconds=float(policy_payload.get("max_timeout_seconds", 30.0)),
            max_output_bytes=int(policy_payload.get("max_output_bytes", 1_000_000)),
            max_resource_budget=policy_payload.get("max_resource_budget") or {},
            require_shadow=bool(policy_payload.get("require_shadow", True)),
            require_approval_for_side_effects=bool(
                policy_payload.get("require_approval_for_side_effects", True)
            ),
            format=str(policy_payload.get("format", "")),
            version=int(policy_payload.get("version", 0)),
        )
        if expected_policy_digest and expected_policy_digest != policy.policy_digest:
            raise ValueError("shadow record policy digest mismatch")
        observation_payload = payload.get("observation")
        observation = (
            McpCapabilityShadowObservation.from_payload(observation_payload)
            if observation_payload is not None
            else None
        )
        record = cls(
            candidate=candidate,
            policy=policy,
            registry_snapshot_id=str(payload.get("registry_snapshot_id", "")),
            state=str(payload.get("state", "")),
            decision=_decision_from_payload(payload.get("decision") or {}),
            observation=observation,
            events=tuple(payload.get("events", ())),
        )
        if str(payload.get("record_digest", "")) != record.record_digest:
            raise ValueError("MCP client shadow record digest mismatch")
        return record


class McpClientCapabilityShadowRegistry:
    """Content-addressed shadow state bound to a current MCP registry snapshot."""

    def __init__(self, *, mcp_registry_snapshot_id: str, parent_checkpoint_id: str) -> None:
        self._current_mcp_registry_snapshot_id = _required_text(
            mcp_registry_snapshot_id, "mcp_registry_snapshot_id"
        )
        self.parent_checkpoint_id = _required_text(parent_checkpoint_id, "parent_checkpoint_id")
        self.revision = 0
        self._records: dict[str, McpClientCapabilityShadowRecord] = {}
        self._binding_events: list[Mapping[str, Any]] = []

    @classmethod
    def from_mcp_registry(cls, registry: Any, *, parent_checkpoint_id: str) -> McpClientCapabilityShadowRegistry:
        from .mcp_registry import McpToolRegistry

        if not isinstance(registry, McpToolRegistry):
            raise TypeError("MCP client shadow registry requires a Seed-owned MCP registry")
        return cls(
            mcp_registry_snapshot_id=registry.snapshot_id,
            parent_checkpoint_id=parent_checkpoint_id,
        )

    @property
    def current_mcp_registry_snapshot_id(self) -> str:
        return self._current_mcp_registry_snapshot_id

    @property
    def records(self) -> tuple[McpClientCapabilityShadowRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def shadow_validated(self) -> tuple[McpClientCapabilityShadowRecord, ...]:
        return tuple(item for item in self.records if item.state == "shadow_validated")

    @property
    def snapshot_id(self) -> str:
        return _digest(
            {
                "format": MCP_CLIENT_SHADOW_REGISTRY_FORMAT,
                "version": MCP_CLIENT_SHADOW_REGISTRY_VERSION,
                "parent_checkpoint_id": self.parent_checkpoint_id,
                "revision": self.revision,
                "mcp_registry_snapshot_id": self.current_mcp_registry_snapshot_id,
                "records": [item.record_digest for item in self.records],
                "binding_events": [dict(event) for event in self._binding_events],
            }
        )

    def get(self, candidate_digest: str) -> McpClientCapabilityShadowRecord | None:
        return self._records.get(str(candidate_digest).strip())

    def bind_mcp_snapshot(
        self,
        snapshot_id: str,
        *,
        expected_current_snapshot_id: str | None = None,
    ) -> str:
        self._require_current_snapshot(expected_current_snapshot_id)
        normalized = _required_text(snapshot_id, "mcp_registry_snapshot_id")
        if normalized == self.current_mcp_registry_snapshot_id:
            return self.snapshot_id
        next_revision = self.revision + 1
        self._binding_events.append(
            {
                "event_id": f"bind:{next_revision}:{normalized}",
                "event_kind": "mcp_snapshot_rebound",
                "previous_snapshot_id": self.current_mcp_registry_snapshot_id,
                "mcp_registry_snapshot_id": normalized,
                "revision": next_revision,
            }
        )
        self._current_mcp_registry_snapshot_id = normalized
        self.revision = next_revision
        return self.snapshot_id

    def propose(
        self,
        candidate: McpCapabilityInheritanceCandidate,
        policy: McpCapabilityInheritancePolicy,
        *,
        expected_current_snapshot_id: str | None = None,
    ) -> McpClientCapabilityShadowRecord:
        if not isinstance(candidate, McpCapabilityInheritanceCandidate):
            raise TypeError("MCP shadow registry requires a client capability candidate")
        if not isinstance(policy, McpCapabilityInheritancePolicy):
            raise TypeError("MCP shadow registry requires an explicit policy")
        self._require_current_snapshot(expected_current_snapshot_id)
        if candidate.registry_snapshot_id != self.current_mcp_registry_snapshot_id:
            raise ValueError("MCP client candidate is stale for the current registry snapshot")
        existing = self.get(candidate.candidate_digest)
        if existing is not None:
            if existing.policy_digest != policy.policy_digest:
                raise ValueError("candidate is already registered with another policy")
            return existing

        decision = preflight_inheritance_candidate(
            candidate,
            policy,
            current_registry_snapshot_id=self.current_mcp_registry_snapshot_id,
        )
        state = "rejected"
        if decision.passed:
            state = "shadow_pending" if decision.shadow_required else "proposed"
        next_revision = self.revision + 1
        event = self._event(
            candidate,
            policy,
            decision,
            state=state,
            event_kind="candidate_proposed",
            revision=next_revision,
        )
        record = McpClientCapabilityShadowRecord(
            candidate=candidate,
            policy=policy,
            registry_snapshot_id=candidate.registry_snapshot_id,
            state=state,
            decision=decision,
            events=(event,),
        )
        self._records[candidate.candidate_digest] = record
        self.revision = next_revision
        return record

    def record_shadow(
        self,
        candidate_digest: str,
        observation: McpCapabilityShadowObservation,
        *,
        expected_current_snapshot_id: str | None = None,
        approval_id: str = "",
    ) -> McpClientCapabilityShadowRecord:
        normalized = _required_text(candidate_digest, "candidate_digest")
        self._require_current_snapshot(expected_current_snapshot_id)
        record = self._records.get(normalized)
        if record is None:
            raise KeyError("unknown MCP client capability candidate")
        if not isinstance(observation, McpCapabilityShadowObservation):
            raise TypeError("MCP shadow registry requires a shadow observation")
        if observation.candidate_digest != normalized:
            raise ValueError("shadow observation candidate digest mismatch")
        if observation.registry_snapshot_id != self.current_mcp_registry_snapshot_id:
            raise ValueError("MCP shadow observation is stale for the current registry snapshot")
        if record.state in {"rejected", "rolled_back"}:
            raise PermissionError("MCP client capability candidate is terminal")
        if record.state == "shadow_validated":
            if record.observation and record.observation.observation_digest == observation.observation_digest:
                return record
            raise PermissionError("MCP client capability candidate is already shadow validated")

        decision = evaluate_mcp_capability_shadow(
            record.candidate,
            record.policy,
            observation,
            current_registry_snapshot_id=self.current_mcp_registry_snapshot_id,
            approval_id=approval_id,
        )
        if decision.passed:
            state = "shadow_validated"
        elif decision.reason_code == "approval_required":
            state = "shadow_pending"
        else:
            state = "rejected"
        next_revision = self.revision + 1
        event = self._event(
            record.candidate,
            record.policy,
            decision,
            state=state,
            event_kind="shadow_evaluated",
            revision=next_revision,
            observation_digest=observation.observation_digest,
        )
        updated = McpClientCapabilityShadowRecord(
            candidate=record.candidate,
            policy=record.policy,
            registry_snapshot_id=record.registry_snapshot_id,
            state=state,
            decision=decision,
            observation=observation,
            events=(*record.events, event),
        )
        self._records[normalized] = updated
        self.revision = next_revision
        return updated

    def rollback(
        self,
        candidate_digest: str,
        *,
        expected_current_snapshot_id: str | None = None,
    ) -> McpClientCapabilityShadowRecord:
        normalized = _required_text(candidate_digest, "candidate_digest")
        self._require_current_snapshot(expected_current_snapshot_id)
        record = self._records.get(normalized)
        if record is None:
            raise KeyError("unknown MCP client capability candidate")
        if record.state in {"rejected", "rolled_back"}:
            raise PermissionError("MCP client capability candidate is terminal")
        decision = record.decision
        next_revision = self.revision + 1
        event = self._event(
            record.candidate,
            record.policy,
            decision,
            state="rolled_back",
            event_kind="candidate_rolled_back",
            revision=next_revision,
        )
        updated = McpClientCapabilityShadowRecord(
            candidate=record.candidate,
            policy=record.policy,
            registry_snapshot_id=record.registry_snapshot_id,
            state="rolled_back",
            decision=decision,
            observation=record.observation,
            events=(*record.events, event),
        )
        self._records[normalized] = updated
        self.revision = next_revision
        return updated

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": MCP_CLIENT_SHADOW_REGISTRY_FORMAT,
            "version": MCP_CLIENT_SHADOW_REGISTRY_VERSION,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "revision": self.revision,
            "mcp_registry_snapshot_id": self.current_mcp_registry_snapshot_id,
            "snapshot_id": self.snapshot_id,
            "records": [item.to_payload() for item in self.records],
            "binding_events": [dict(event) for event in self._binding_events],
        }
        payload["checkpoint_digest"] = _digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> McpClientCapabilityShadowRegistry:
        if payload.get("format") != MCP_CLIENT_SHADOW_REGISTRY_FORMAT:
            raise ValueError("unsupported MCP client shadow registry format")
        if int(payload.get("version", 0)) != MCP_CLIENT_SHADOW_REGISTRY_VERSION:
            raise ValueError("unsupported MCP client shadow registry version")
        expected_checkpoint = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected_checkpoint:
            raise ValueError("MCP client shadow registry checkpoint digest mismatch")
        registry = cls(
            mcp_registry_snapshot_id=str(payload.get("mcp_registry_snapshot_id", "")),
            parent_checkpoint_id=str(payload.get("parent_checkpoint_id", "")),
        )
        raw_records = payload.get("records", ())
        if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Sequence):
            raise ValueError("MCP client shadow registry records must be a sequence")
        for raw_record in raw_records:
            record = McpClientCapabilityShadowRecord.from_payload(raw_record)
            if record.candidate_digest in registry._records:
                raise ValueError("duplicate MCP client shadow candidate")
            registry._records[record.candidate_digest] = record
        raw_events = payload.get("binding_events", ())
        if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
            raise ValueError("MCP client shadow registry binding events must be a sequence")
        registry._binding_events = [dict(event) for event in raw_events]
        registry.revision = int(payload.get("revision", -1))
        expected_revision = sum(len(record.events) for record in registry.records) + len(
            registry._binding_events
        )
        if registry.revision != expected_revision:
            raise ValueError("MCP client shadow registry checkpoint revision mismatch")
        if str(payload.get("snapshot_id", "")) != registry.snapshot_id:
            raise ValueError("MCP client shadow registry snapshot mismatch")
        return registry

    def _require_current_snapshot(self, expected_snapshot_id: str | None) -> None:
        if expected_snapshot_id not in (None, "") and str(expected_snapshot_id) != self.current_mcp_registry_snapshot_id:
            raise ValueError("MCP client shadow registry snapshot is stale")

    def _event(
        self,
        candidate: McpCapabilityInheritanceCandidate,
        policy: McpCapabilityInheritancePolicy,
        decision: McpCapabilityInheritanceDecision,
        *,
        state: str,
        event_kind: str,
        revision: int,
        observation_digest: str = "",
    ) -> dict[str, Any]:
        return {
            "event_id": f"{event_kind}:{candidate.candidate_digest}:{revision}",
            "event_kind": event_kind,
            "candidate_digest": candidate.candidate_digest,
            "policy_digest": policy.policy_digest,
            "mcp_registry_snapshot_id": self.current_mcp_registry_snapshot_id,
            "state": state,
            "decision": decision.decision,
            "reason_code": decision.reason_code,
            "observation_digest": observation_digest,
            "revision": revision,
        }


__all__ = [
    "MCP_CLIENT_SHADOW_REGISTRY_FORMAT",
    "MCP_CLIENT_SHADOW_REGISTRY_VERSION",
    "MCP_CLIENT_SHADOW_STATES",
    "McpClientCapabilityShadowRecord",
    "McpClientCapabilityShadowRegistry",
]

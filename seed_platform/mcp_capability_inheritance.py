"""Governed MCP-to-client capability inheritance candidates.

This module is the E6-0 boundary between a governed MCP description and a
future Seed client capability.  It deliberately stores only declarative tool
contracts and non-secret references.  It does not connect to a server, load
an executor, mutate the Workbench registry, or alter Taiji cognition.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .mcp_registry import McpToolDescriptor, McpToolRegistry

MCP_CLIENT_CANDIDATE_FORMAT = "seed-mcp-client-capability-candidate-v1"
MCP_CLIENT_CANDIDATE_VERSION = 1
MCP_CLIENT_POLICY_FORMAT = "seed-mcp-client-capability-policy-v1"
MCP_CLIENT_SHADOW_FORMAT = "seed-mcp-client-capability-shadow-v1"
MCP_CLIENT_RISKS = ("read_only", "reversible_ui", "file_write", "terminal", "mcp_dispatch")
MCP_CLIENT_SIDE_EFFECT_RISKS = frozenset({"file_write", "terminal", "mcp_dispatch"})
_FORBIDDEN_CANDIDATE_KEYS = frozenset(
    {
        "command",
        "credential_value",
        "entrypoint",
        "exec",
        "executor",
        "executor_path",
        "import_path",
        "module",
        "password",
        "path",
        "script",
        "secret",
        "shell",
        "source_path",
        "token",
        "url",
    }
)
_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


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
            raise ValueError("MCP client candidate digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported MCP client candidate value: {type(value).__name__}")


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


def _text_tuple(value: Sequence[str] | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_required_text(item, f"{name} item") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} items must be unique")
    return tuple(sorted(normalized))


def _reference_tuple(value: Sequence[str] | None, name: str) -> tuple[str, ...]:
    normalized = _text_tuple(value, name)
    for item in normalized:
        if not _REFERENCE_PATTERN.fullmatch(item):
            raise ValueError(f"{name} must contain identifier references, not secret values")
    return normalized


def _resource_budget(value: Mapping[str, Any] | None, name: str) -> dict[str, float | int]:
    if value is None or not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must contain bounded numeric values")
    normalized: dict[str, float | int] = {}
    for key, raw in value.items():
        resource = _required_text(key, f"{name} key")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{name} values must be numeric")
        numeric = float(raw)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{name} values must be finite and positive")
        normalized[resource] = raw
    return {key: normalized[key] for key in sorted(normalized)}


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_CANDIDATE_KEYS:
                raise ValueError("MCP client candidate contains forbidden executable or secret fields")
            # Schema property names describe user input; they are not candidate
            # control fields and may legitimately include names such as path.
            if str(key) not in {"input_schema", "schema"}:
                _assert_no_forbidden_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_no_forbidden_keys(item)


def _schema_digest(schema: Mapping[str, Any]) -> str:
    return _digest({"schema": dict(schema)})


@dataclass(frozen=True)
class McpClientToolContract:
    """One MCP tool's client-facing contract without an executor identity."""

    tool_id: str
    input_schema: Mapping[str, Any]
    risk: str = "read_only"
    permissions: tuple[str, ...] = ()
    timeout_seconds: float = 5.0
    output_limit: int = 65_536
    schema_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_id", _required_text(self.tool_id, "tool_id"))
        if self.risk not in MCP_CLIENT_RISKS:
            raise ValueError("unsupported MCP client risk")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("MCP client input_schema must be a mapping")
        object.__setattr__(self, "input_schema", _canonical(self.input_schema))
        object.__setattr__(self, "permissions", _reference_tuple(self.permissions, "permissions"))
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(self.output_limit, bool) or int(self.output_limit) < 1:
            raise ValueError("output_limit must be positive")
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "output_limit", int(self.output_limit))
        expected = _schema_digest(self.input_schema)
        if self.schema_digest and self.schema_digest != expected:
            raise ValueError("MCP tool schema digest mismatch")
        object.__setattr__(self, "schema_digest", expected)

    @classmethod
    def from_descriptor(cls, descriptor: McpToolDescriptor) -> McpClientToolContract:
        if not isinstance(descriptor, McpToolDescriptor):
            raise TypeError("MCP client tool contract requires a tool descriptor")
        return cls(
            tool_id=descriptor.tool_id,
            input_schema=descriptor.input_schema,
            risk=descriptor.risk,
            timeout_seconds=descriptor.timeout_seconds,
            output_limit=descriptor.output_limit,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": MCP_CLIENT_CANDIDATE_FORMAT,
            "version": MCP_CLIENT_CANDIDATE_VERSION,
            "tool_id": self.tool_id,
            "input_schema": dict(self.input_schema),
            "risk": self.risk,
            "permissions": list(self.permissions),
            "timeout_seconds": self.timeout_seconds,
            "output_limit": self.output_limit,
            "schema_digest": self.schema_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> McpClientToolContract:
        _assert_no_forbidden_keys(payload)
        if payload.get("format") != MCP_CLIENT_CANDIDATE_FORMAT:
            raise ValueError("unsupported MCP client tool contract format")
        return cls(
            tool_id=str(payload.get("tool_id", "")),
            input_schema=payload.get("input_schema") or {},
            risk=str(payload.get("risk", "read_only")),
            permissions=tuple(payload.get("permissions", ())),
            timeout_seconds=float(payload.get("timeout_seconds", 5.0)),
            output_limit=int(payload.get("output_limit", 65_536)),
            schema_digest=str(payload.get("schema_digest", "")),
        )


@dataclass(frozen=True)
class McpCapabilityInheritanceCandidate:
    """Unevaluated client-organ candidate derived from a governed MCP registry."""

    server_id: str
    server_version: str
    source_artifact_digest: str
    registry_snapshot_id: str
    tool_contracts: tuple[McpClientToolContract, ...]
    network_scopes: tuple[str, ...] = ()
    credential_refs: tuple[str, ...] = ()
    resource_budget: Mapping[str, Any] = field(default_factory=dict)
    evidence_digests: tuple[str, ...] = ()
    evaluation_gates: tuple[str, ...] = ()
    parent_checkpoint_id: str = ""
    rationale: str = ""
    format: str = MCP_CLIENT_CANDIDATE_FORMAT
    version: int = MCP_CLIENT_CANDIDATE_VERSION

    def __post_init__(self) -> None:
        if self.format != MCP_CLIENT_CANDIDATE_FORMAT or self.version != MCP_CLIENT_CANDIDATE_VERSION:
            raise ValueError("unsupported MCP client capability candidate format")
        for name, value in (
            ("server_id", self.server_id),
            ("server_version", self.server_version),
            ("source_artifact_digest", self.source_artifact_digest),
            ("registry_snapshot_id", self.registry_snapshot_id),
            ("parent_checkpoint_id", self.parent_checkpoint_id),
            ("rationale", self.rationale),
        ):
            object.__setattr__(self, name, _required_text(value, name))
        if isinstance(self.tool_contracts, (str, bytes)) or not isinstance(self.tool_contracts, Sequence):
            raise TypeError("tool_contracts must be a sequence")
        contracts = tuple(self.tool_contracts)
        if not contracts or not all(isinstance(item, McpClientToolContract) for item in contracts):
            raise ValueError("MCP client candidate requires tool contracts")
        if len({item.tool_id for item in contracts}) != len(contracts):
            raise ValueError("MCP client tool ids must be unique")
        object.__setattr__(self, "tool_contracts", tuple(sorted(contracts, key=lambda item: item.tool_id)))
        object.__setattr__(self, "network_scopes", _reference_tuple(self.network_scopes, "network_scopes"))
        object.__setattr__(self, "credential_refs", _reference_tuple(self.credential_refs, "credential_refs"))
        object.__setattr__(self, "resource_budget", _resource_budget(self.resource_budget, "resource_budget"))
        object.__setattr__(self, "evidence_digests", _text_tuple(self.evidence_digests, "evidence_digests"))
        object.__setattr__(self, "evaluation_gates", _text_tuple(self.evaluation_gates, "evaluation_gates"))
        if not self.evidence_digests or not self.evaluation_gates:
            raise ValueError("MCP client candidate requires evidence digests and evaluation gates")

    @classmethod
    def from_registry(
        cls,
        registry: McpToolRegistry,
        *,
        server_id: str,
        server_version: str,
        source_artifact_digest: str,
        tool_ids: Sequence[str],
        resource_budget: Mapping[str, Any],
        evidence_digests: Sequence[str],
        evaluation_gates: Sequence[str],
        parent_checkpoint_id: str,
        network_scopes: Sequence[str] = (),
        credential_refs: Sequence[str] = (),
        rationale: str,
    ) -> McpCapabilityInheritanceCandidate:
        if not isinstance(registry, McpToolRegistry):
            raise TypeError("MCP client candidate requires a Seed-owned MCP registry")
        requested = _text_tuple(tool_ids, "tool_ids")
        if not requested:
            raise ValueError("tool_ids cannot be empty")
        descriptors = []
        for tool_id in requested:
            descriptor = registry.get(tool_id)
            if descriptor is None:
                raise KeyError(f"unknown MCP tool: {tool_id}")
            descriptors.append(McpClientToolContract.from_descriptor(descriptor))
        return cls(
            server_id=server_id,
            server_version=server_version,
            source_artifact_digest=source_artifact_digest,
            registry_snapshot_id=registry.snapshot_id,
            tool_contracts=tuple(descriptors),
            network_scopes=tuple(network_scopes),
            credential_refs=tuple(credential_refs),
            resource_budget=resource_budget,
            evidence_digests=tuple(evidence_digests),
            evaluation_gates=tuple(evaluation_gates),
            parent_checkpoint_id=parent_checkpoint_id,
            rationale=rationale,
        )

    @property
    def candidate_digest(self) -> str:
        return _digest(self._identity_payload())

    @property
    def requires_approval(self) -> bool:
        return any(item.risk in MCP_CLIENT_SIDE_EFFECT_RISKS for item in self.tool_contracts)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "server_id": self.server_id,
            "server_version": self.server_version,
            "source_artifact_digest": self.source_artifact_digest,
            "registry_snapshot_id": self.registry_snapshot_id,
            "tool_contracts": [item.to_payload() for item in self.tool_contracts],
            "network_scopes": list(self.network_scopes),
            "credential_refs": list(self.credential_refs),
            "resource_budget": dict(self.resource_budget),
            "evidence_digests": list(self.evidence_digests),
            "evaluation_gates": list(self.evaluation_gates),
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "rationale": self.rationale,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "candidate_digest": self.candidate_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> McpCapabilityInheritanceCandidate:
        if not isinstance(payload, Mapping):
            raise TypeError("MCP client capability candidate must be an object")
        _assert_no_forbidden_keys(payload)
        candidate = cls(
            server_id=str(payload.get("server_id", "")),
            server_version=str(payload.get("server_version", "")),
            source_artifact_digest=str(payload.get("source_artifact_digest", "")),
            registry_snapshot_id=str(payload.get("registry_snapshot_id", "")),
            tool_contracts=tuple(
                McpClientToolContract.from_payload(item)
                for item in payload.get("tool_contracts", ())
            ),
            network_scopes=tuple(payload.get("network_scopes", ())),
            credential_refs=tuple(payload.get("credential_refs", ())),
            resource_budget=payload.get("resource_budget") or {},
            evidence_digests=tuple(payload.get("evidence_digests", ())),
            evaluation_gates=tuple(payload.get("evaluation_gates", ())),
            parent_checkpoint_id=str(payload.get("parent_checkpoint_id", "")),
            rationale=str(payload.get("rationale", "")),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", 0)),
        )
        if str(payload.get("candidate_digest", "")) != candidate.candidate_digest:
            raise ValueError("MCP client capability candidate digest mismatch")
        return candidate


@dataclass(frozen=True)
class McpCapabilityInheritancePolicy:
    """Explicit policy for candidate preflight; empty allowlists mean no scope."""

    allowed_server_ids: tuple[str, ...] = ()
    allowed_risks: tuple[str, ...] = ("read_only", "reversible_ui")
    allowed_permissions: tuple[str, ...] = ()
    allowed_network_scopes: tuple[str, ...] = ()
    allow_credentials: bool = False
    max_tools: int = 16
    max_timeout_seconds: float = 30.0
    max_output_bytes: int = 1_000_000
    max_resource_budget: Mapping[str, Any] = field(
        default_factory=lambda: {"max_cpu_ms": 100_000, "max_output_bytes": 1_000_000}
    )
    require_shadow: bool = True
    require_approval_for_side_effects: bool = True
    format: str = MCP_CLIENT_POLICY_FORMAT
    version: int = MCP_CLIENT_CANDIDATE_VERSION

    def __post_init__(self) -> None:
        if self.format != MCP_CLIENT_POLICY_FORMAT or self.version != MCP_CLIENT_CANDIDATE_VERSION:
            raise ValueError("unsupported MCP client capability policy format")
        object.__setattr__(self, "allowed_server_ids", _reference_tuple(self.allowed_server_ids, "allowed_server_ids"))
        if isinstance(self.allowed_risks, (str, bytes)) or not isinstance(self.allowed_risks, Sequence):
            raise TypeError("allowed_risks must be a sequence")
        risks = _text_tuple(self.allowed_risks, "allowed_risks")
        if any(risk not in MCP_CLIENT_RISKS for risk in risks):
            raise ValueError("policy contains unsupported MCP risk")
        object.__setattr__(self, "allowed_risks", risks)
        object.__setattr__(self, "allowed_permissions", _reference_tuple(self.allowed_permissions, "allowed_permissions"))
        object.__setattr__(self, "allowed_network_scopes", _reference_tuple(self.allowed_network_scopes, "allowed_network_scopes"))
        if not isinstance(self.allow_credentials, bool) or not isinstance(self.require_shadow, bool):
            raise TypeError("MCP client policy flags must be boolean")
        if not isinstance(self.require_approval_for_side_effects, bool):
            raise TypeError("require_approval_for_side_effects must be boolean")
        if isinstance(self.max_tools, bool) or int(self.max_tools) < 1:
            raise ValueError("max_tools must be positive")
        timeout = float(self.max_timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("max_timeout_seconds must be finite and positive")
        if isinstance(self.max_output_bytes, bool) or int(self.max_output_bytes) < 1:
            raise ValueError("max_output_bytes must be positive")
        object.__setattr__(self, "max_tools", int(self.max_tools))
        object.__setattr__(self, "max_timeout_seconds", timeout)
        object.__setattr__(self, "max_output_bytes", int(self.max_output_bytes))
        object.__setattr__(
            self,
            "max_resource_budget",
            _resource_budget(self.max_resource_budget, "max_resource_budget"),
        )

    @property
    def policy_digest(self) -> str:
        return _digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "allowed_server_ids": list(self.allowed_server_ids),
            "allowed_risks": list(self.allowed_risks),
            "allowed_permissions": list(self.allowed_permissions),
            "allowed_network_scopes": list(self.allowed_network_scopes),
            "allow_credentials": self.allow_credentials,
            "max_tools": self.max_tools,
            "max_timeout_seconds": self.max_timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "max_resource_budget": dict(self.max_resource_budget),
            "require_shadow": self.require_shadow,
            "require_approval_for_side_effects": self.require_approval_for_side_effects,
        }
        if include_digest:
            payload["policy_digest"] = self.policy_digest
        return payload


@dataclass(frozen=True)
class McpCapabilityInheritanceDecision:
    passed: bool
    decision: str
    reason_code: str
    candidate_digest: str
    policy_digest: str
    approval_required: bool = False
    shadow_required: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "candidate_digest": self.candidate_digest,
            "policy_digest": self.policy_digest,
            "approval_required": self.approval_required,
            "shadow_required": self.shadow_required,
        }


def preflight_inheritance_candidate(
    candidate: McpCapabilityInheritanceCandidate,
    policy: McpCapabilityInheritancePolicy,
    *,
    current_registry_snapshot_id: str,
) -> McpCapabilityInheritanceDecision:
    """Validate a candidate without registering, connecting, or activating it."""

    digest = candidate.candidate_digest

    def deny(reason: str) -> McpCapabilityInheritanceDecision:
        return McpCapabilityInheritanceDecision(
            False,
            "deny",
            reason,
            digest,
            policy.policy_digest,
            approval_required=candidate.requires_approval,
            shadow_required=policy.require_shadow,
        )

    if candidate.registry_snapshot_id != str(current_registry_snapshot_id):
        return deny("stale_mcp_registry")
    if policy.allowed_server_ids and candidate.server_id not in policy.allowed_server_ids:
        return deny("server_not_allowed")
    if len(candidate.tool_contracts) > policy.max_tools:
        return deny("tool_count_exceeded")
    if candidate.network_scopes and not set(candidate.network_scopes).issubset(policy.allowed_network_scopes):
        return deny("network_scope_not_allowed")
    if candidate.credential_refs and not policy.allow_credentials:
        return deny("credentials_not_allowed")
    for contract in candidate.tool_contracts:
        if contract.risk not in policy.allowed_risks:
            return deny("risk_not_allowed")
        if not set(contract.permissions).issubset(policy.allowed_permissions):
            return deny("permission_not_allowed")
        if contract.timeout_seconds > policy.max_timeout_seconds:
            return deny("tool_timeout_exceeded")
        if contract.output_limit > policy.max_output_bytes:
            return deny("tool_output_limit_exceeded")
    for resource, amount in candidate.resource_budget.items():
        maximum = policy.max_resource_budget.get(resource)
        if maximum is not None and float(amount) > float(maximum):
            return deny("resource_budget_exceeded")
    if candidate.requires_approval and policy.require_approval_for_side_effects:
        return McpCapabilityInheritanceDecision(
            True,
            "shadow_pending_approval",
            "approval_required_after_shadow",
            digest,
            policy.policy_digest,
            approval_required=True,
            shadow_required=policy.require_shadow,
        )
    if policy.require_shadow:
        return McpCapabilityInheritanceDecision(
            True,
            "shadow_pending",
            "shadow_required",
            digest,
            policy.policy_digest,
            approval_required=candidate.requires_approval,
            shadow_required=True,
        )
    return McpCapabilityInheritanceDecision(
        True,
        "candidate_admissible",
        "policy_preflight_passed",
        digest,
        policy.policy_digest,
        approval_required=candidate.requires_approval,
        shadow_required=False,
    )


@dataclass(frozen=True)
class McpCapabilityShadowObservation:
    """Digest-only no-side-effect comparison for an inheritance candidate."""

    candidate_digest: str
    registry_snapshot_id: str
    input_digest: str
    baseline_output_digest: str
    candidate_output_digest: str
    baseline_after_state_digest: str
    candidate_after_state_digest: str
    baseline_resources: Mapping[str, Any] = field(default_factory=dict)
    candidate_resources: Mapping[str, Any] = field(default_factory=dict)
    external_calls_performed: bool = False
    credential_accessed: bool = False
    format: str = MCP_CLIENT_SHADOW_FORMAT

    def __post_init__(self) -> None:
        if self.format != MCP_CLIENT_SHADOW_FORMAT:
            raise ValueError("unsupported MCP client capability shadow format")
        for name, value in (
            ("candidate_digest", self.candidate_digest),
            ("registry_snapshot_id", self.registry_snapshot_id),
            ("input_digest", self.input_digest),
            ("baseline_output_digest", self.baseline_output_digest),
            ("candidate_output_digest", self.candidate_output_digest),
            ("baseline_after_state_digest", self.baseline_after_state_digest),
            ("candidate_after_state_digest", self.candidate_after_state_digest),
        ):
            object.__setattr__(self, name, _required_text(value, name))
        object.__setattr__(self, "baseline_resources", _resource_budget(self.baseline_resources, "baseline_resources"))
        object.__setattr__(self, "candidate_resources", _resource_budget(self.candidate_resources, "candidate_resources"))
        if not isinstance(self.external_calls_performed, bool) or not isinstance(self.credential_accessed, bool):
            raise TypeError("MCP shadow side-effect flags must be boolean")

    @classmethod
    def from_execution(
        cls,
        *,
        candidate_digest: str,
        registry_snapshot_id: str,
        input_payload: Any,
        baseline_output: Any,
        candidate_output: Any,
        baseline_after_state: Any,
        candidate_after_state: Any,
        baseline_resources: Mapping[str, Any],
        candidate_resources: Mapping[str, Any],
        external_calls_performed: bool = False,
        credential_accessed: bool = False,
    ) -> McpCapabilityShadowObservation:
        return cls(
            candidate_digest=candidate_digest,
            registry_snapshot_id=registry_snapshot_id,
            input_digest=_digest(input_payload),
            baseline_output_digest=_digest(baseline_output),
            candidate_output_digest=_digest(candidate_output),
            baseline_after_state_digest=_digest(baseline_after_state),
            candidate_after_state_digest=_digest(candidate_after_state),
            baseline_resources=baseline_resources,
            candidate_resources=candidate_resources,
            external_calls_performed=external_calls_performed,
            credential_accessed=credential_accessed,
        )

    @property
    def observation_digest(self) -> str:
        return _digest(self._identity_payload())

    @property
    def output_equal(self) -> bool:
        return self.baseline_output_digest == self.candidate_output_digest

    @property
    def after_state_equal(self) -> bool:
        return self.baseline_after_state_digest == self.candidate_after_state_digest

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "candidate_digest": self.candidate_digest,
            "registry_snapshot_id": self.registry_snapshot_id,
            "input_digest": self.input_digest,
            "baseline_output_digest": self.baseline_output_digest,
            "candidate_output_digest": self.candidate_output_digest,
            "baseline_after_state_digest": self.baseline_after_state_digest,
            "candidate_after_state_digest": self.candidate_after_state_digest,
            "baseline_resources": dict(self.baseline_resources),
            "candidate_resources": dict(self.candidate_resources),
            "external_calls_performed": self.external_calls_performed,
            "credential_accessed": self.credential_accessed,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "observation_digest": self.observation_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> McpCapabilityShadowObservation:
        _assert_no_forbidden_keys(payload)
        observation = cls(
            candidate_digest=str(payload.get("candidate_digest", "")),
            registry_snapshot_id=str(payload.get("registry_snapshot_id", "")),
            input_digest=str(payload.get("input_digest", "")),
            baseline_output_digest=str(payload.get("baseline_output_digest", "")),
            candidate_output_digest=str(payload.get("candidate_output_digest", "")),
            baseline_after_state_digest=str(payload.get("baseline_after_state_digest", "")),
            candidate_after_state_digest=str(payload.get("candidate_after_state_digest", "")),
            baseline_resources=payload.get("baseline_resources") or {},
            candidate_resources=payload.get("candidate_resources") or {},
            external_calls_performed=bool(payload.get("external_calls_performed", False)),
            credential_accessed=bool(payload.get("credential_accessed", False)),
            format=str(payload.get("format", "")),
        )
        if str(payload.get("observation_digest", "")) != observation.observation_digest:
            raise ValueError("MCP shadow observation digest mismatch")
        return observation


def evaluate_mcp_capability_shadow(
    candidate: McpCapabilityInheritanceCandidate,
    policy: McpCapabilityInheritancePolicy,
    observation: McpCapabilityShadowObservation,
    *,
    current_registry_snapshot_id: str,
    approval_id: str = "",
) -> McpCapabilityInheritanceDecision:
    """Admit only an equivalent, no-side-effect shadow observation."""

    preflight = preflight_inheritance_candidate(
        candidate,
        policy,
        current_registry_snapshot_id=current_registry_snapshot_id,
    )
    if not preflight.passed:
        return preflight
    digest = candidate.candidate_digest

    def deny(reason: str) -> McpCapabilityInheritanceDecision:
        return McpCapabilityInheritanceDecision(
            False,
            "deny",
            reason,
            digest,
            policy.policy_digest,
            approval_required=candidate.requires_approval,
            shadow_required=policy.require_shadow,
        )

    if observation.candidate_digest != digest:
        return deny("candidate_digest_mismatch")
    if observation.registry_snapshot_id != str(current_registry_snapshot_id):
        return deny("stale_mcp_registry")
    if observation.external_calls_performed:
        return deny("shadow_external_call_detected")
    if observation.credential_accessed:
        return deny("shadow_credential_access_detected")
    if not observation.output_equal:
        return deny("shadow_output_mismatch")
    if not observation.after_state_equal:
        return deny("shadow_state_mutated")
    if candidate.requires_approval and not str(approval_id).strip():
        return deny("approval_required")
    return McpCapabilityInheritanceDecision(
        True,
        "shadow_equivalent",
        "shadow_equivalent",
        digest,
        policy.policy_digest,
        approval_required=candidate.requires_approval,
        shadow_required=policy.require_shadow,
    )


__all__ = [
    "McpCapabilityInheritanceCandidate",
    "McpCapabilityInheritanceDecision",
    "McpCapabilityInheritancePolicy",
    "McpCapabilityShadowObservation",
    "McpClientToolContract",
    "evaluate_mcp_capability_shadow",
    "preflight_inheritance_candidate",
]

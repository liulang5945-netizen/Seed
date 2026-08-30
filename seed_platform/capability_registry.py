"""Seed-owned capability lifecycle and content-addressed snapshot contract.

R5B-L0 moves effector identity out of ad-hoc dispatch strings. The registry
does not choose goals, produce affordances, or execute an executor. It only
validates a versioned bundle, records lifecycle, and resolves an admitted
capability against the current snapshot.

Only logical executor/disposer identities are stored. Executable source is
never imported or evaluated, and file presence never activates a bundle.
Side-effecting bundles must declare a disposer before registration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

CAPABILITY_REGISTRY_FORMAT = "seed-capability-registry-v1"
CAPABILITY_REGISTRY_VERSION = 1
CAPABILITY_REGISTRY_CHECKPOINT_FORMAT = "seed-capability-registry-checkpoint-v1"
CAPABILITY_CANDIDATE_FORMAT = "seed-capability-candidate-v1"
CAPABILITY_CANDIDATE_STATUSES = ("proposed", "validated", "rejected")
CAPABILITY_LIFECYCLE_STATUSES = (
    "proposed",
    "validated",
    "shadow",
    "active",
    "retired",
    "rolled_back",
    "tombstoned",
)
CAPABILITY_EFFECTS = (
    "read_only",
    "reversible_ui",
    "file_write",
    "terminal",
    "mcp_dispatch",
)
CAPABILITY_SIDE_EFFECTS = frozenset({"file_write", "terminal"})
CAPABILITY_RISKS = frozenset(CAPABILITY_EFFECTS)
_SIDE_EFFECTING = CAPABILITY_SIDE_EFFECTS
DEFAULT_RESOURCE_LIMITS = {
    "active_bundle_count": 64,
    "max_cpu_ms": 1_000_000,
    "max_output_bytes": 100_000_000,
    "max_file_changes": 1_024,
}
_FORBIDDEN_BUNDLE_KEYS = frozenset(
    {"path", "source_path", "executor_path", "module", "command", "delete", "auto_activate"}
)
_FORBIDDEN_CANDIDATE_KEYS = _FORBIDDEN_BUNDLE_KEYS | {
    "script",
    "entrypoint_path",
    "import_path",
    "shell",
    "url",
}


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
            raise ValueError("registry digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported registry digest value: {type(value).__name__}")


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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be positive")
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


def _event_tuple(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("events must be a sequence")
    normalized = tuple(_required_text(item, "event") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("events must be unique")
    return normalized


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): _canonical(item) for key, item in value.items()}


def _assert_no_forbidden_keys(value: Any, forbidden: frozenset[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            if name in forbidden:
                raise ValueError("capability artifact contains forbidden executable-source fields")
            if name != "schema":
                _assert_no_forbidden_keys(item, forbidden)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_no_forbidden_keys(item, forbidden)


def _resource_budget(
    value: Mapping[str, Any] | None, *, allow_empty: bool = False
) -> dict[str, float | int]:
    if value is None or not isinstance(value, Mapping):
        raise ValueError("resource_budget must contain bounded numeric values")
    if not value:
        if allow_empty:
            return {}
        raise ValueError("resource_budget must contain bounded numeric values")
    normalized: dict[str, float | int] = {}
    for key, item in value.items():
        name = _required_text(key, "resource budget key")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("resource budget values must be numeric")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError("resource budget values must be finite and positive")
        normalized[name] = item
    return {key: normalized[key] for key in sorted(normalized)}


@dataclass(frozen=True)
class CapabilityBundle:
    """Declarative identity of one capability and its executor boundary."""

    capability_id: str
    schema: Mapping[str, Any]
    effect: str = "read_only"
    risk: str = "read_only"
    reversible: bool = True
    permissions: tuple[str, ...] = ()
    executor_id: str = ""
    executor_version: str = ""
    disposer_id: str = ""
    disposer_version: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = CAPABILITY_REGISTRY_VERSION

    def __post_init__(self) -> None:
        capability_id = _required_text(self.capability_id, "capability_id")
        object.__setattr__(self, "capability_id", capability_id)
        if self.version != CAPABILITY_REGISTRY_VERSION:
            raise ValueError("unsupported capability bundle version")
        if self.effect not in CAPABILITY_EFFECTS:
            raise ValueError("unsupported capability effect")
        if self.risk not in CAPABILITY_RISKS:
            raise ValueError("unsupported capability risk")
        if not isinstance(self.schema, Mapping):
            raise TypeError("capability schema must be a mapping")
        object.__setattr__(self, "schema", _mapping(self.schema, "schema"))
        object.__setattr__(self, "permissions", _text_tuple(self.permissions, "permissions"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        executor_id = _required_text(self.executor_id, "executor_id")
        executor_version = _required_text(self.executor_version, "executor_version")
        object.__setattr__(self, "executor_id", executor_id)
        object.__setattr__(self, "executor_version", executor_version)
        disposer_id = str(self.disposer_id).strip()
        disposer_version = str(self.disposer_version).strip()
        if (disposer_id and not disposer_version) or (disposer_version and not disposer_id):
            raise ValueError("disposer_id and disposer_version must be provided together")
        if self.effect in _SIDE_EFFECTING or self.risk in _SIDE_EFFECTING:
            if not disposer_id or not disposer_version:
                raise ValueError("side-effecting capability requires a disposer")
        object.__setattr__(self, "disposer_id", disposer_id)
        object.__setattr__(self, "disposer_version", disposer_version)

    @property
    def bundle_digest(self) -> str:
        return _digest(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": CAPABILITY_REGISTRY_FORMAT,
            "version": self.version,
            "capability_id": self.capability_id,
            "schema": dict(self.schema),
            "effect": self.effect,
            "risk": self.risk,
            "reversible": self.reversible,
            "permissions": list(self.permissions),
            "executor_id": self.executor_id,
            "executor_version": self.executor_version,
            "disposer_id": self.disposer_id,
            "disposer_version": self.disposer_version,
            "metadata": dict(self.metadata),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "bundle_digest": self.bundle_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapabilityBundle:
        if not isinstance(payload, Mapping):
            raise TypeError("capability bundle must be an object")
        _assert_no_forbidden_keys(payload, _FORBIDDEN_BUNDLE_KEYS)
        bundle = cls(
            capability_id=str(payload.get("capability_id", "")),
            schema=payload.get("schema") or {},
            effect=str(payload.get("effect", "read_only")),
            risk=str(payload.get("risk", "read_only")),
            reversible=bool(payload.get("reversible", True)),
            permissions=tuple(payload.get("permissions", ())),
            executor_id=str(payload.get("executor_id", "")),
            executor_version=str(payload.get("executor_version", "")),
            disposer_id=str(payload.get("disposer_id", "")),
            disposer_version=str(payload.get("disposer_version", "")),
            metadata=payload.get("metadata") or {},
            version=int(payload.get("version", CAPABILITY_REGISTRY_VERSION)),
        )
        if str(payload.get("bundle_digest", "")) != bundle.bundle_digest:
            raise ValueError("capability bundle digest mismatch")
        return bundle


@dataclass(frozen=True)
class CapabilityCandidate:
    """Unevaluated capability proposal kept outside the executable registry."""

    bundle: CapabilityBundle
    rationale: str
    evidence_digests: tuple[str, ...]
    resource_budget: Mapping[str, Any]
    evaluation_gates: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = CAPABILITY_REGISTRY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, CapabilityBundle):
            raise TypeError("capability candidate requires a CapabilityBundle")
        if self.version != CAPABILITY_REGISTRY_VERSION:
            raise ValueError("unsupported capability candidate version")
        _required_text(self.rationale, "rationale")
        evidence_digests = _text_tuple(self.evidence_digests, "evidence_digests")
        evaluation_gates = _text_tuple(self.evaluation_gates, "evaluation_gates")
        if not evidence_digests:
            raise ValueError("capability candidate requires evidence digests")
        if not evaluation_gates:
            raise ValueError("capability candidate requires evaluation gates")
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        object.__setattr__(self, "evidence_digests", evidence_digests)
        resource_budget = _resource_budget(self.resource_budget)
        if "active_bundle_count" in resource_budget:
            raise ValueError("resource budget cannot reserve active_bundle_count")
        object.__setattr__(self, "resource_budget", resource_budget)
        object.__setattr__(self, "evaluation_gates", evaluation_gates)
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def candidate_digest(self) -> str:
        return _digest(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": CAPABILITY_CANDIDATE_FORMAT,
            "version": self.version,
            "bundle": self.bundle.to_payload(),
            "rationale": self.rationale,
            "evidence_digests": list(self.evidence_digests),
            "resource_budget": dict(self.resource_budget),
            "evaluation_gates": list(self.evaluation_gates),
            "metadata": dict(self.metadata),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "candidate_digest": self.candidate_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapabilityCandidate:
        if not isinstance(payload, Mapping):
            raise TypeError("capability candidate must be an object")
        _assert_no_forbidden_keys(payload, _FORBIDDEN_CANDIDATE_KEYS)
        bundle_payload = payload.get("bundle")
        if not isinstance(bundle_payload, Mapping):
            raise TypeError("capability candidate bundle must be an object")
        candidate = cls(
            bundle=CapabilityBundle.from_payload(bundle_payload),
            rationale=str(payload.get("rationale", "")),
            evidence_digests=tuple(payload.get("evidence_digests", ())),
            resource_budget=payload.get("resource_budget") or {},
            evaluation_gates=tuple(payload.get("evaluation_gates", ())),
            metadata=payload.get("metadata") or {},
            version=int(payload.get("version", CAPABILITY_REGISTRY_VERSION)),
        )
        if str(payload.get("candidate_digest", "")) != candidate.candidate_digest:
            raise ValueError("capability candidate digest mismatch")
        return candidate


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Content-addressed active capability set for one registry revision."""

    snapshot_digest: str
    revision: int
    policy_revision: int
    parent_checkpoint_id: str
    bundles: tuple[CapabilityBundle, ...]
    format: str = CAPABILITY_REGISTRY_FORMAT

    def __post_init__(self) -> None:
        if self.format != CAPABILITY_REGISTRY_FORMAT:
            raise ValueError("unsupported capability registry format")
        _positive_int(self.revision, "snapshot revision")
        _positive_int(self.policy_revision, "policy revision")
        _required_text(self.parent_checkpoint_id, "parent_checkpoint_id")
        ids = tuple(bundle.capability_id for bundle in self.bundles)
        if len(set(ids)) != len(ids):
            raise ValueError("active capability ids must be unique")
        if self.snapshot_digest != _digest(self._identity_payload()):
            raise ValueError("capability snapshot digest mismatch")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": CAPABILITY_REGISTRY_VERSION,
            "revision": self.revision,
            "policy_revision": self.policy_revision,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "bundles": [bundle.to_payload() for bundle in self.bundles],
        }

    @property
    def snapshot_id(self) -> str:
        return self.snapshot_digest

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "snapshot_digest": self.snapshot_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapabilitySnapshot:
        if payload.get("format") != CAPABILITY_REGISTRY_FORMAT:
            raise ValueError("unsupported capability registry format")
        bundles = tuple(CapabilityBundle.from_payload(item) for item in payload.get("bundles", ()))
        revision = int(payload.get("revision", 0))
        policy_revision = int(payload.get("policy_revision", 0))
        parent_checkpoint_id = str(payload.get("parent_checkpoint_id", ""))
        identity = {
            "format": str(payload["format"]),
            "version": CAPABILITY_REGISTRY_VERSION,
            "revision": revision,
            "policy_revision": policy_revision,
            "parent_checkpoint_id": parent_checkpoint_id,
            "bundles": [bundle.to_payload() for bundle in bundles],
        }
        snapshot_digest = str(payload.get("snapshot_digest", ""))
        if snapshot_digest != _digest(identity):
            raise ValueError("capability snapshot digest mismatch")
        return cls(snapshot_digest, revision, policy_revision, parent_checkpoint_id, bundles)


@dataclass(frozen=True)
class CapabilityLifecycleRecord:
    capability_id: str
    bundle_digest: str
    status: str
    snapshot_digest: str
    snapshot_revision: int
    policy_revision: int
    parent_checkpoint_id: str
    executor_version: str
    disposer_version: str
    events: tuple[str, ...] = ()
    audit_digest: str = ""

    def __post_init__(self) -> None:
        if self.status not in CAPABILITY_LIFECYCLE_STATUSES:
            raise ValueError("unsupported capability lifecycle status")
        for name, value in (
            ("capability_id", self.capability_id),
            ("bundle_digest", self.bundle_digest),
            ("snapshot_digest", self.snapshot_digest),
            ("parent_checkpoint_id", self.parent_checkpoint_id),
            ("executor_version", self.executor_version),
        ):
            _required_text(value, name)
        if self.disposer_version:
            _required_text(self.disposer_version, "disposer_version")
        _positive_int(self.snapshot_revision, "snapshot revision")
        _positive_int(self.policy_revision, "policy revision")
        events = _event_tuple(self.events)
        object.__setattr__(self, "events", events)
        expected = _digest(self._identity_payload())
        if self.audit_digest and self.audit_digest != expected:
            raise ValueError("capability lifecycle audit digest mismatch")
        object.__setattr__(self, "audit_digest", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": CAPABILITY_REGISTRY_FORMAT,
            "version": CAPABILITY_REGISTRY_VERSION,
            "capability_id": self.capability_id,
            "bundle_digest": self.bundle_digest,
            "status": self.status,
            "snapshot_digest": self.snapshot_digest,
            "snapshot_revision": self.snapshot_revision,
            "policy_revision": self.policy_revision,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "executor_version": self.executor_version,
            "disposer_version": self.disposer_version,
            "events": list(self.events),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "audit_digest": self.audit_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapabilityLifecycleRecord:
        return cls(
            capability_id=str(payload.get("capability_id", "")),
            bundle_digest=str(payload.get("bundle_digest", "")),
            status=str(payload.get("status", "")),
            snapshot_digest=str(payload.get("snapshot_digest", "")),
            snapshot_revision=int(payload.get("snapshot_revision", 0)),
            policy_revision=int(payload.get("policy_revision", 0)),
            parent_checkpoint_id=str(payload.get("parent_checkpoint_id", "")),
            executor_version=str(payload.get("executor_version", "")),
            disposer_version=str(payload.get("disposer_version", "")),
            events=tuple(payload.get("events", ())),
            audit_digest=str(payload.get("audit_digest", "")),
        )


@dataclass(frozen=True)
class CapabilityCandidateRecord:
    """Auditable state of a candidate before it enters bundle lifecycle."""

    candidate_digest: str
    bundle_digest: str
    status: str
    parent_checkpoint_id: str
    decision_ref: str = ""
    reason: str = ""
    events: tuple[str, ...] = ()
    audit_digest: str = ""

    def __post_init__(self) -> None:
        if self.status not in CAPABILITY_CANDIDATE_STATUSES:
            raise ValueError("unsupported capability candidate status")
        for name, value in (
            ("candidate_digest", self.candidate_digest),
            ("bundle_digest", self.bundle_digest),
            ("parent_checkpoint_id", self.parent_checkpoint_id),
        ):
            _required_text(value, name)
        if self.status != "proposed":
            _required_text(self.decision_ref, "decision_ref")
        if self.status == "rejected":
            _required_text(self.reason, "reason")
        events = _event_tuple(self.events)
        object.__setattr__(self, "events", events)
        expected = _digest(self._identity_payload())
        if self.audit_digest and self.audit_digest != expected:
            raise ValueError("capability candidate audit digest mismatch")
        object.__setattr__(self, "audit_digest", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": CAPABILITY_CANDIDATE_FORMAT,
            "version": CAPABILITY_REGISTRY_VERSION,
            "candidate_digest": self.candidate_digest,
            "bundle_digest": self.bundle_digest,
            "status": self.status,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "decision_ref": self.decision_ref,
            "reason": self.reason,
            "events": list(self.events),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "audit_digest": self.audit_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapabilityCandidateRecord:
        return cls(
            candidate_digest=str(payload.get("candidate_digest", "")),
            bundle_digest=str(payload.get("bundle_digest", "")),
            status=str(payload.get("status", "")),
            parent_checkpoint_id=str(payload.get("parent_checkpoint_id", "")),
            decision_ref=str(payload.get("decision_ref", "")),
            reason=str(payload.get("reason", "")),
            events=tuple(payload.get("events", ())),
            audit_digest=str(payload.get("audit_digest", "")),
        )


class CapabilityRegistry:
    """Registry owner for capability validation, snapshots, and lifecycle."""

    def __init__(
        self,
        *,
        policy_revision: int = 1,
        parent_checkpoint_id: str = "checkpoint:registry-root",
        resource_limits: Mapping[str, Any] | None = None,
    ) -> None:
        self.policy_revision = _positive_int(policy_revision, "policy revision")
        self.parent_checkpoint_id = _required_text(parent_checkpoint_id, "parent_checkpoint_id")
        self.resource_limits = _resource_budget(resource_limits or DEFAULT_RESOURCE_LIMITS)
        self._bundles: dict[str, CapabilityBundle] = {}
        self._records: dict[str, CapabilityLifecycleRecord] = {}
        self._candidates: dict[str, CapabilityCandidate] = {}
        self._candidate_records: dict[str, CapabilityCandidateRecord] = {}
        self._resource_budgets: dict[str, dict[str, float | int]] = {}
        self._resource_reservations: dict[str, dict[str, float | int]] = {}
        self._prior_resource_reservations: dict[str, dict[str, dict[str, float | int]]] = {}
        self._active: set[str] = set()
        self._prior_active: dict[str, tuple[str, ...]] = {}
        self._snapshot = self._make_snapshot(1, ())

    @classmethod
    def from_workbench_descriptors(
        cls,
        descriptors: Sequence[Any],
        *,
        policy_revision: int = 1,
        parent_checkpoint_id: str = "checkpoint:workbench-bootstrap",
    ) -> CapabilityRegistry:
        """Build an explicit active registry for trusted built-in descriptors.

        Bootstrap is a registry-owner operation. It still passes every bundle
        through validated and shadow states and uses an explicit bootstrap
        approval; it is not file-presence activation and is not callable by
        Taiji/provider/frontend code.
        """

        registry = cls(
            policy_revision=policy_revision,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        for descriptor in descriptors:
            if not bool(getattr(descriptor, "enabled", True)):
                continue
            risk = str(getattr(descriptor, "risk", "read_only"))
            effect = risk if risk in CAPABILITY_EFFECTS else "read_only"
            parameters = getattr(descriptor, "parameters", ())
            schema = {
                "type": "object",
                "properties": {
                    str(name): {"description": str(description)} for name, description in parameters
                },
                "additionalProperties": False,
            }
            disposer_id = ""
            disposer_version = ""
            if effect in _SIDE_EFFECTING:
                disposer_id = "seed.workbench.workspace.undo"
                disposer_version = "1.0.0"
            bundle = CapabilityBundle(
                capability_id=str(descriptor.capability_id),
                schema=schema,
                effect=effect,
                risk=effect,
                reversible=bool(getattr(descriptor, "reversible", True)),
                permissions=(str(descriptor.capability_id),),
                executor_id=str(descriptor.capability_id),
                executor_version=f"workbench-descriptor-v{int(getattr(descriptor, 'version', 1))}",
                disposer_id=disposer_id,
                disposer_version=disposer_version,
                metadata={"category": str(getattr(descriptor, "category", "workbench"))},
            )
            registry.register(bundle)
            registry.shadow(bundle.bundle_digest)
            registry.activate(
                bundle.bundle_digest,
                approval_id=f"approval:bootstrap:{bundle.capability_id}",
            )
        return registry

    @property
    def snapshot(self) -> CapabilitySnapshot:
        return self._snapshot

    @property
    def snapshot_id(self) -> str:
        return self._snapshot.snapshot_digest

    @property
    def records(self) -> tuple[CapabilityLifecycleRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def candidate_records(self) -> tuple[CapabilityCandidateRecord, ...]:
        return tuple(self._candidate_records[key] for key in sorted(self._candidate_records))

    def get_record(self, bundle_digest: str) -> CapabilityLifecycleRecord | None:
        return self._records.get(str(bundle_digest).strip())

    def get_bundle(self, bundle_digest: str) -> CapabilityBundle | None:
        return self._bundles.get(str(bundle_digest).strip())

    @property
    def resource_ledger(self) -> dict[str, Any]:
        usage: dict[str, float] = {}
        for reservation in self._resource_reservations.values():
            for key, value in reservation.items():
                usage[key] = usage.get(key, 0.0) + float(value)
        identity = {
            "format": CAPABILITY_REGISTRY_FORMAT,
            "policy_revision": self.policy_revision,
            "snapshot_id": self.snapshot_id,
            "limits": dict(self.resource_limits),
            "reservations": {
                digest: dict(values)
                for digest, values in sorted(self._resource_reservations.items())
            },
            "usage": {key: usage[key] for key in sorted(usage)},
        }
        return {**identity, "ledger_digest": _digest(identity)}

    def get_candidate_record(self, candidate_digest: str) -> CapabilityCandidateRecord | None:
        return self._candidate_records.get(str(candidate_digest).strip())

    def _make_snapshot(
        self, revision: int, bundles: Sequence[CapabilityBundle]
    ) -> CapabilitySnapshot:
        ordered = tuple(sorted(bundles, key=lambda bundle: bundle.capability_id))
        identity = {
            "format": CAPABILITY_REGISTRY_FORMAT,
            "version": CAPABILITY_REGISTRY_VERSION,
            "revision": revision,
            "policy_revision": self.policy_revision,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "bundles": [bundle.to_payload() for bundle in ordered],
        }
        return CapabilitySnapshot(
            snapshot_digest=_digest(identity),
            revision=revision,
            policy_revision=self.policy_revision,
            parent_checkpoint_id=self.parent_checkpoint_id,
            bundles=ordered,
        )

    def _set_record(
        self,
        bundle: CapabilityBundle,
        *,
        status: str,
        events: Sequence[str],
        snapshot: CapabilitySnapshot | None = None,
    ) -> CapabilityLifecycleRecord:
        current = self._records.get(bundle.bundle_digest)
        merged_events = (*current.events, *events) if current else tuple(events)
        record = CapabilityLifecycleRecord(
            capability_id=bundle.capability_id,
            bundle_digest=bundle.bundle_digest,
            status=status,
            snapshot_digest=(snapshot or self._snapshot).snapshot_digest,
            snapshot_revision=(snapshot or self._snapshot).revision,
            policy_revision=self.policy_revision,
            parent_checkpoint_id=self.parent_checkpoint_id,
            executor_version=bundle.executor_version,
            disposer_version=bundle.disposer_version,
            events=merged_events,
        )
        self._records[bundle.bundle_digest] = record
        return record

    def _sync_active_records(self) -> None:
        for bundle_digest in tuple(self._active):
            current = self._records[bundle_digest]
            self._records[bundle_digest] = replace(
                current,
                snapshot_digest=self._snapshot.snapshot_digest,
                snapshot_revision=self._snapshot.revision,
                policy_revision=self.policy_revision,
                audit_digest="",
            )

    def _reservation_for(self, bundle_digest: str) -> dict[str, float | int]:
        reservation: dict[str, float | int] = {"active_bundle_count": 1}
        reservation.update(self._resource_budgets.get(bundle_digest, {}))
        return reservation

    def _compute_resource_reservations(
        self, active_digests: set[str]
    ) -> dict[str, dict[str, float | int]]:
        reservations = {digest: self._reservation_for(digest) for digest in sorted(active_digests)}
        usage: dict[str, float] = {}
        for reservation in reservations.values():
            for key, value in reservation.items():
                usage[key] = usage.get(key, 0.0) + float(value)
        for key, amount in usage.items():
            limit = self.resource_limits.get(key)
            if limit is None or amount > float(limit):
                raise ValueError(f"capability resource budget exhausted: {key}")
        return reservations

    def _commit_active_set(self, active_digests: set[str]) -> None:
        reservations = self._compute_resource_reservations(active_digests)
        snapshot = self._make_snapshot(
            self._snapshot.revision + 1,
            tuple(self._bundles[digest] for digest in sorted(active_digests)),
        )
        self._active = set(active_digests)
        self._resource_reservations = reservations
        self._snapshot = snapshot

    def _set_candidate_record(
        self,
        candidate: CapabilityCandidate,
        *,
        status: str,
        events: Sequence[str],
        decision_ref: str = "",
        reason: str = "",
    ) -> CapabilityCandidateRecord:
        current = self._candidate_records.get(candidate.candidate_digest)
        merged_events = (*current.events, *events) if current else tuple(events)
        record = CapabilityCandidateRecord(
            candidate_digest=candidate.candidate_digest,
            bundle_digest=candidate.bundle.bundle_digest,
            status=status,
            parent_checkpoint_id=self.parent_checkpoint_id,
            decision_ref=decision_ref or (current.decision_ref if current else ""),
            reason=reason or (current.reason if current else ""),
            events=merged_events,
        )
        self._candidate_records[candidate.candidate_digest] = record
        return record

    def propose(self, candidate: CapabilityCandidate) -> CapabilityCandidateRecord:
        """Store a candidate proposal without registering or activating its bundle."""

        if not isinstance(candidate, CapabilityCandidate):
            raise TypeError("registry accepts CapabilityCandidate objects only")
        if candidate.candidate_digest in self._candidates:
            raise ValueError("capability candidate is already proposed")
        self._candidates[candidate.candidate_digest] = candidate
        return self._set_candidate_record(
            candidate,
            status="proposed",
            events=("candidate_proposed", "bundle_not_registered"),
        )

    def validate_candidate(
        self,
        candidate_digest: str,
        *,
        validation_ref: str,
    ) -> CapabilityLifecycleRecord:
        """Admit a proposed candidate to validated bundle state, never to active state."""

        candidate = self._require_candidate(candidate_digest)
        candidate_record = self._require_candidate_record(candidate.candidate_digest)
        validation_ref = _required_text(validation_ref, "validation_ref")
        if candidate_record.status != "proposed":
            raise ValueError("only proposed capability candidates can be validated")
        unknown_resources = set(candidate.resource_budget) - set(self.resource_limits)
        if unknown_resources:
            raise ValueError(
                "resource budget has no configured limit: " + ", ".join(sorted(unknown_resources))
            )
        bundle_record = self.register(candidate.bundle)
        self._resource_budgets[candidate.bundle.bundle_digest] = dict(candidate.resource_budget)
        self._set_candidate_record(
            candidate,
            status="validated",
            decision_ref=validation_ref,
            events=("candidate_validated",),
        )
        return bundle_record

    def reject_candidate(
        self,
        candidate_digest: str,
        *,
        decision_ref: str,
        reason: str,
    ) -> CapabilityCandidateRecord:
        candidate = self._require_candidate(candidate_digest)
        candidate_record = self._require_candidate_record(candidate.candidate_digest)
        if candidate_record.status != "proposed":
            raise ValueError("only proposed capability candidates can be rejected")
        return self._set_candidate_record(
            candidate,
            status="rejected",
            decision_ref=_required_text(decision_ref, "decision_ref"),
            reason=_required_text(reason, "reason"),
            events=("candidate_rejected",),
        )

    def register(self, bundle: CapabilityBundle) -> CapabilityLifecycleRecord:
        """Validate and register a bundle without making it executable."""

        if not isinstance(bundle, CapabilityBundle):
            raise TypeError("registry accepts CapabilityBundle objects only")
        if bundle.bundle_digest in self._bundles:
            raise ValueError("capability bundle is already registered")
        if any(
            current.capability_id == bundle.capability_id
            and self._records[digest].status in {"validated", "shadow"}
            for digest, current in self._bundles.items()
        ):
            raise ValueError("capability id already has a live bundle")
        self._bundles[bundle.bundle_digest] = bundle
        self._resource_budgets.setdefault(bundle.bundle_digest, {})
        return self._set_record(
            bundle,
            status="validated",
            events=("bundle_registered", "schema_validated", "executor_precompiled"),
        )

    def shadow(self, bundle_digest: str) -> CapabilityLifecycleRecord:
        bundle = self._require_bundle(bundle_digest)
        record = self._require_record(bundle.bundle_digest)
        if record.status != "validated":
            raise ValueError("only validated capability bundles can enter shadow")
        return self._set_record(bundle, status="shadow", events=("candidate_shadowed",))

    def activate(
        self,
        bundle_digest: str,
        *,
        approval_id: str,
        expected_snapshot_id: str | None = None,
    ) -> CapabilityLifecycleRecord:
        bundle = self._require_bundle(bundle_digest)
        record = self._require_record(bundle.bundle_digest)
        self._require_current_snapshot(expected_snapshot_id)
        if record.status != "shadow":
            raise PermissionError("capability activation requires a shadow candidate")
        if not str(approval_id).strip():
            raise PermissionError("capability activation requires approval")
        if any(
            self._bundles[digest].capability_id == bundle.capability_id for digest in self._active
        ):
            raise ValueError("capability id is already active; use replace")
        self._prior_active[bundle.bundle_digest] = tuple(sorted(self._active))
        self._prior_resource_reservations[bundle.bundle_digest] = {
            digest: dict(values) for digest, values in self._resource_reservations.items()
        }
        self._commit_active_set(self._active | {bundle.bundle_digest})
        activated = self._set_record(
            bundle,
            status="active",
            events=("approval_recorded", "snapshot_activated"),
            snapshot=self._snapshot,
        )
        self._sync_active_records()
        return activated

    def replace(
        self,
        current_bundle_digest: str,
        candidate_bundle_digest: str,
        *,
        approval_id: str,
        expected_snapshot_id: str | None = None,
    ) -> CapabilityLifecycleRecord:
        """Atomically replace one active bundle with a shadow candidate."""

        current_bundle = self._require_bundle(current_bundle_digest)
        candidate_bundle = self._require_bundle(candidate_bundle_digest)
        current_record = self._require_record(current_bundle.bundle_digest)
        candidate_record = self._require_record(candidate_bundle.bundle_digest)
        self._require_current_snapshot(expected_snapshot_id)
        if current_record.status != "active":
            raise ValueError("replacement source must be active")
        if candidate_record.status != "shadow":
            raise PermissionError("replacement candidate must be shadow")
        if current_bundle.capability_id != candidate_bundle.capability_id:
            raise ValueError("replacement capability ids must match")
        if not str(approval_id).strip():
            raise PermissionError("capability replacement requires approval")
        self._prior_active[candidate_bundle.bundle_digest] = tuple(sorted(self._active))
        self._prior_resource_reservations[candidate_bundle.bundle_digest] = {
            digest: dict(values) for digest, values in self._resource_reservations.items()
        }
        self._commit_active_set(
            (self._active - {current_bundle.bundle_digest}) | {candidate_bundle.bundle_digest}
        )
        self._set_record(
            current_bundle,
            status="retired",
            events=("bundle_retired", "replacement_committed"),
            snapshot=self._snapshot,
        )
        activated = self._set_record(
            candidate_bundle,
            status="active",
            events=("approval_recorded", "snapshot_activated", "replacement_committed"),
            snapshot=self._snapshot,
        )
        self._sync_active_records()
        return activated

    def retire(
        self, bundle_digest: str, *, expected_snapshot_id: str | None = None
    ) -> CapabilityLifecycleRecord:
        bundle = self._require_bundle(bundle_digest)
        record = self._require_record(bundle.bundle_digest)
        self._require_current_snapshot(expected_snapshot_id)
        if record.status != "active":
            raise ValueError("only active capability bundles can be retired")
        self._commit_active_set(self._active - {bundle.bundle_digest})
        retired = self._set_record(
            bundle,
            status="retired",
            events=("bundle_retired",),
            snapshot=self._snapshot,
        )
        self._sync_active_records()
        return retired

    def rollback(
        self, bundle_digest: str, *, expected_snapshot_id: str | None = None
    ) -> CapabilityLifecycleRecord:
        bundle = self._require_bundle(bundle_digest)
        record = self._require_record(bundle.bundle_digest)
        self._require_current_snapshot(expected_snapshot_id)
        if record.status in {"rolled_back", "tombstoned", "retired"}:
            raise ValueError("terminal capability bundle cannot be rolled back")
        if record.status == "active":
            previous_active = self._prior_active.pop(bundle.bundle_digest, None)
            previous_reservations = self._prior_resource_reservations.pop(
                bundle.bundle_digest, None
            )
            self._active = set(previous_active or ())
            self._resource_reservations = {
                digest: dict(values) for digest, values in (previous_reservations or {}).items()
            }
            if not previous_reservations:
                self._resource_reservations = self._compute_resource_reservations(self._active)
            self._snapshot = self._make_snapshot(
                self._snapshot.revision + 1, self._active_bundles()
            )
            if previous_active:
                for previous_digest in previous_active:
                    previous_bundle = self._bundles[previous_digest]
                    previous_record = self._records[previous_digest]
                    if previous_record.status == "retired":
                        self._set_record(
                            previous_bundle,
                            status="active",
                            events=("replacement_rollback",),
                            snapshot=self._snapshot,
                        )
            self._sync_active_records()
        rollback_events = ["rollback_completed"]
        if bundle.disposer_id:
            rollback_events.insert(0, "disposer_release_recorded")
        return self._set_record(
            bundle,
            status="rolled_back",
            events=tuple(rollback_events),
            snapshot=self._snapshot,
        )

    def tombstone(self, bundle_digest: str) -> CapabilityLifecycleRecord:
        bundle = self._require_bundle(bundle_digest)
        record = self._require_record(bundle.bundle_digest)
        if record.status not in {"retired", "rolled_back"}:
            raise ValueError("only retired or rolled-back bundles can be tombstoned")
        return self._set_record(bundle, status="tombstoned", events=("bundle_tombstoned",))

    def resolve(self, capability_id: str, *, snapshot_id: str | None = None) -> CapabilityBundle:
        self._require_current_snapshot(snapshot_id)
        normalized = _required_text(capability_id, "capability_id")
        for bundle in self._snapshot.bundles:
            if bundle.capability_id == normalized:
                return bundle
        raise PermissionError("capability is not active in the current snapshot")

    def _active_bundles(self) -> tuple[CapabilityBundle, ...]:
        return tuple(self._bundles[digest] for digest in sorted(self._active))

    def _require_bundle(self, bundle_digest: str) -> CapabilityBundle:
        normalized = _required_text(bundle_digest, "bundle_digest")
        bundle = self._bundles.get(normalized)
        if bundle is None:
            raise KeyError("unknown capability bundle")
        return bundle

    def _require_record(self, bundle_digest: str) -> CapabilityLifecycleRecord:
        record = self._records.get(bundle_digest)
        if record is None:
            raise RuntimeError("capability bundle has no lifecycle record")
        return record

    def _require_candidate(self, candidate_digest: str) -> CapabilityCandidate:
        normalized = _required_text(candidate_digest, "candidate_digest")
        candidate = self._candidates.get(normalized)
        if candidate is None:
            raise KeyError("unknown capability candidate")
        return candidate

    def _require_candidate_record(self, candidate_digest: str) -> CapabilityCandidateRecord:
        record = self._candidate_records.get(candidate_digest)
        if record is None:
            raise RuntimeError("capability candidate has no lifecycle record")
        return record

    def _require_current_snapshot(self, expected_snapshot_id: str | None) -> None:
        if expected_snapshot_id not in (None, "") and expected_snapshot_id != self.snapshot_id:
            raise ValueError("capability snapshot is stale")

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": CAPABILITY_REGISTRY_CHECKPOINT_FORMAT,
            "version": CAPABILITY_REGISTRY_VERSION,
            "policy_revision": self.policy_revision,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "snapshot": self._snapshot.to_payload(),
            "bundles": [self._bundles[key].to_payload() for key in sorted(self._bundles)],
            "lifecycles": [record.to_payload() for record in self.records],
            "candidates": [self._candidates[key].to_payload() for key in sorted(self._candidates)],
            "candidate_lifecycles": [record.to_payload() for record in self.candidate_records],
            "resource_limits": dict(self.resource_limits),
            "resource_budgets": {
                digest: dict(values) for digest, values in sorted(self._resource_budgets.items())
            },
            "resource_reservations": {
                digest: dict(values)
                for digest, values in sorted(self._resource_reservations.items())
            },
            "prior_resource_reservations": {
                digest: {
                    bundle_digest: dict(values)
                    for bundle_digest, values in sorted(reservations.items())
                }
                for digest, reservations in sorted(self._prior_resource_reservations.items())
            },
            "active_bundle_digests": sorted(self._active),
            "prior_active": {
                digest: list(active) for digest, active in sorted(self._prior_active.items())
            },
        }
        return {**payload, "checkpoint_digest": _digest(payload)}

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> CapabilityRegistry:
        if payload.get("format") != CAPABILITY_REGISTRY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported capability registry checkpoint format")
        identity = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        if str(payload.get("checkpoint_digest", "")) != _digest(identity):
            raise ValueError("capability registry checkpoint digest mismatch")
        registry = cls(
            policy_revision=int(payload.get("policy_revision", 0)),
            parent_checkpoint_id=str(payload.get("parent_checkpoint_id", "")),
            resource_limits=payload.get("resource_limits") or DEFAULT_RESOURCE_LIMITS,
        )
        registry._bundles = {
            bundle.bundle_digest: bundle
            for bundle in (
                CapabilityBundle.from_payload(item) for item in payload.get("bundles", ())
            )
        }
        registry._records = {
            record.bundle_digest: record
            for record in (
                CapabilityLifecycleRecord.from_payload(item)
                for item in payload.get("lifecycles", ())
            )
        }
        registry._candidates = {
            candidate.candidate_digest: candidate
            for candidate in (
                CapabilityCandidate.from_payload(item) for item in payload.get("candidates", ())
            )
        }
        registry._candidate_records = {
            record.candidate_digest: record
            for record in (
                CapabilityCandidateRecord.from_payload(item)
                for item in payload.get("candidate_lifecycles", ())
            )
        }
        if set(registry._candidates) != set(registry._candidate_records):
            raise ValueError("checkpoint candidate and lifecycle records diverge")
        if any(
            record.bundle_digest != registry._candidates[digest].bundle.bundle_digest
            for digest, record in registry._candidate_records.items()
        ):
            raise ValueError("checkpoint candidate bundle diverged")
        raw_resource_budgets = payload.get("resource_budgets", {})
        if not isinstance(raw_resource_budgets, Mapping):
            raise ValueError("checkpoint resource_budgets must be an object")
        registry._resource_budgets = {digest: {} for digest in registry._bundles}
        for digest, values in raw_resource_budgets.items():
            normalized_digest = str(digest)
            if normalized_digest not in registry._bundles:
                raise ValueError("checkpoint resource budget references missing bundle")
            registry._resource_budgets[normalized_digest] = _resource_budget(
                values, allow_empty=True
            )
        for candidate in registry._candidates.values():
            candidate_record = registry._candidate_records[candidate.candidate_digest]
            if candidate_record.status == "validated":
                registry._resource_budgets[candidate.bundle.bundle_digest] = dict(
                    candidate.resource_budget
                )
        registry._active = {str(item) for item in payload.get("active_bundle_digests", ())}
        raw_prior_active = payload.get("prior_active", {})
        if not isinstance(raw_prior_active, Mapping):
            raise ValueError("checkpoint prior_active must be an object")
        registry._prior_active = {
            str(digest): tuple(str(item) for item in active)
            for digest, active in raw_prior_active.items()
        }
        if not registry._active.issubset(registry._bundles):
            raise ValueError("checkpoint active bundle is missing")
        if any(
            digest not in registry._bundles or not set(active).issubset(registry._bundles)
            for digest, active in registry._prior_active.items()
        ):
            raise ValueError("checkpoint prior active bundle is missing")
        raw_resource_reservations = payload.get("resource_reservations", {})
        if not isinstance(raw_resource_reservations, Mapping):
            raise ValueError("checkpoint resource_reservations must be an object")
        registry._resource_reservations = {
            str(digest): _resource_budget(values, allow_empty=True)
            for digest, values in raw_resource_reservations.items()
        }
        if not registry._resource_reservations and registry._active:
            registry._resource_reservations = registry._compute_resource_reservations(
                registry._active
            )
        if set(registry._resource_reservations) != registry._active:
            raise ValueError("checkpoint resource reservations diverge from active bundles")
        raw_prior_resources = payload.get("prior_resource_reservations", {})
        if not isinstance(raw_prior_resources, Mapping):
            raise ValueError("checkpoint prior_resource_reservations must be an object")
        registry._prior_resource_reservations = {
            str(digest): {
                str(bundle_digest): _resource_budget(values, allow_empty=True)
                for bundle_digest, values in reservations.items()
            }
            for digest, reservations in raw_prior_resources.items()
        }
        if any(
            digest not in registry._bundles
            or any(bundle_digest not in registry._bundles for bundle_digest in reservations)
            for digest, reservations in registry._prior_resource_reservations.items()
        ):
            raise ValueError("checkpoint prior resource reservation references missing bundle")
        for bundle_digest in registry._active:
            record = registry._records.get(bundle_digest)
            if record is None or record.status != "active":
                raise ValueError("checkpoint active bundle has invalid lifecycle")
        registry._snapshot = CapabilitySnapshot.from_payload(payload["snapshot"])
        expected_active = tuple(
            bundle.bundle_digest
            for bundle in sorted(registry._active_bundles(), key=lambda item: item.capability_id)
        )
        actual_active = tuple(bundle.bundle_digest for bundle in registry._snapshot.bundles)
        if actual_active != expected_active:
            raise ValueError("checkpoint snapshot and active bundles diverge")
        if registry._snapshot.policy_revision != registry.policy_revision:
            raise ValueError("checkpoint policy revision drifted")
        return registry


__all__ = [
    "CAPABILITY_EFFECTS",
    "CAPABILITY_SIDE_EFFECTS",
    "CAPABILITY_CANDIDATE_FORMAT",
    "CAPABILITY_CANDIDATE_STATUSES",
    "CAPABILITY_LIFECYCLE_STATUSES",
    "CAPABILITY_REGISTRY_CHECKPOINT_FORMAT",
    "CAPABILITY_REGISTRY_FORMAT",
    "CAPABILITY_REGISTRY_VERSION",
    "CapabilityBundle",
    "CapabilityCandidate",
    "CapabilityCandidateRecord",
    "CapabilityLifecycleRecord",
    "CapabilityRegistry",
    "CapabilitySnapshot",
]

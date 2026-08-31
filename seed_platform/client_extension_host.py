"""Seed-owned client extension host and reversible lifecycle contract.

This module owns the *client body* side of evolution.  It stores declarative
plugin manifests, content-addressed client snapshots, dependency health,
state migration and lifecycle audit records.  It never imports plugin source,
loads an entrypoint, executes a command, or writes a Taiji cognition
checkpoint.  A later desktop/Vue integration can consume the committed
snapshot through stable slots while keeping the protected root shell outside
the extension host.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

CLIENT_EXTENSION_FORMAT = "seed-client-extension-v1"
CLIENT_EXTENSION_VERSION = 1
CLIENT_EXTENSION_CHECKPOINT_FORMAT = "seed-client-extension-checkpoint-v1"
CLIENT_EXTENSION_LIFECYCLE_STATES = (
    "prepared",
    "shadow",
    "active",
    "dependency_lost",
    "dependency_recovered",
    "draining",
    "retired",
    "failed",
    "quarantined",
    "rolled_back",
)
DEFAULT_EXTENSION_SLOTS = (
    "command",
    "ide.panel",
    "route",
    "settings",
    "sidebar",
    "visualization",
)
DEFAULT_PROTECTED_SLOTS = (
    "desktop.process_manager",
    "desktop.qwebchannel",
    "desktop.root_shell",
    "desktop.taskbar",
    "desktop.tray",
)
_FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "command",
        "entrypoint",
        "entrypoint_path",
        "exec",
        "executor",
        "executor_path",
        "import_path",
        "install_path",
        "module",
        "path",
        "script",
        "shell",
        "source_path",
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
            raise ValueError("client extension digest input must contain finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported client extension value: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, name: str) -> str:
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


def _text_tuple(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_text(item, f"{name} item") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} items must be unique")
    return tuple(sorted(normalized))


def _service_tuple(
    value: Mapping[str, str] | Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = value
    else:
        raise TypeError("service_dependencies must be a mapping or sequence")
    normalized: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise TypeError("service dependencies must contain name/version pairs")
        normalized.append((_text(item[0], "service name"), _text(item[1], "service version")))
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("service dependency names must be unique")
    return tuple(sorted(normalized))


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): _canonical(item) for key, item in value.items()}


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_MANIFEST_KEYS:
                raise ValueError("client extension manifest contains executable-source fields")
            _assert_no_forbidden_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_no_forbidden_keys(item)


def _state(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("client extension state must be a mapping")
    return copy.deepcopy(_mapping(value, "state"))


@dataclass(frozen=True)
class ClientPluginManifest:
    """Declarative plugin identity; no executable source is representable."""

    plugin_id: str
    version: str
    scope: str
    slots: tuple[str, ...]
    capability_ids: tuple[str, ...] = ()
    service_dependencies: tuple[tuple[str, str], ...] = ()
    state_schema_version: int = 1
    migration_id: str = ""
    migration_version: str = ""
    disposer_id: str = ""
    disposer_version: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = CLIENT_EXTENSION_FORMAT

    def __post_init__(self) -> None:
        if self.format != CLIENT_EXTENSION_FORMAT:
            raise ValueError("unsupported client extension format")
        object.__setattr__(self, "plugin_id", _text(self.plugin_id, "plugin_id"))
        object.__setattr__(self, "version", _text(self.version, "plugin version"))
        object.__setattr__(self, "scope", _text(self.scope, "plugin scope"))
        object.__setattr__(self, "slots", _text_tuple(self.slots, "slots"))
        object.__setattr__(self, "capability_ids", _text_tuple(self.capability_ids, "capability_ids"))
        object.__setattr__(self, "service_dependencies", _service_tuple(self.service_dependencies))
        object.__setattr__(
            self,
            "state_schema_version",
            _positive_int(self.state_schema_version, "state schema version"),
        )
        migration_id = str(self.migration_id).strip()
        migration_version = str(self.migration_version).strip()
        if (migration_id and not migration_version) or (migration_version and not migration_id):
            raise ValueError("migration_id and migration_version must be provided together")
        object.__setattr__(self, "migration_id", migration_id)
        object.__setattr__(self, "migration_version", migration_version)
        disposer_id = _text(self.disposer_id, "disposer_id")
        disposer_version = _text(self.disposer_version, "disposer_version")
        object.__setattr__(self, "disposer_id", disposer_id)
        object.__setattr__(self, "disposer_version", disposer_version)
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": CLIENT_EXTENSION_VERSION,
            "plugin_id": self.plugin_id,
            "plugin_version": self.version,
            "scope": self.scope,
            "slots": list(self.slots),
            "capability_ids": list(self.capability_ids),
            "service_dependencies": {
                name: version for name, version in self.service_dependencies
            },
            "state_schema_version": self.state_schema_version,
            "migration_id": self.migration_id,
            "migration_version": self.migration_version,
            "disposer_id": self.disposer_id,
            "disposer_version": self.disposer_version,
            "metadata": dict(self.metadata),
        }

    @property
    def plugin_digest(self) -> str:
        return _digest(self._identity_payload())

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "plugin_digest": self.plugin_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ClientPluginManifest:
        if not isinstance(payload, Mapping):
            raise TypeError("client plugin manifest must be a mapping")
        _assert_no_forbidden_keys(payload)
        manifest = cls(
            plugin_id=str(payload.get("plugin_id", "")),
            version=str(payload.get("plugin_version", payload.get("version", ""))),
            scope=str(payload.get("scope", "")),
            slots=tuple(payload.get("slots", ())),
            capability_ids=tuple(payload.get("capability_ids", ())),
            service_dependencies=payload.get("service_dependencies") or {},
            state_schema_version=int(payload.get("state_schema_version", 1)),
            migration_id=str(payload.get("migration_id", "")),
            migration_version=str(payload.get("migration_version", "")),
            disposer_id=str(payload.get("disposer_id", "")),
            disposer_version=str(payload.get("disposer_version", "")),
            metadata=payload.get("metadata") or {},
            format=str(payload.get("format", "")),
        )
        if str(payload.get("plugin_digest", "")) != manifest.plugin_digest:
            raise ValueError("client plugin manifest digest mismatch")
        return manifest


@dataclass(frozen=True)
class ExtensionHostPolicy:
    """Configurable client policy; protected shell slots are never mountable."""

    allowed_slots: tuple[str, ...] = DEFAULT_EXTENSION_SLOTS
    protected_slots: tuple[str, ...] = DEFAULT_PROTECTED_SLOTS
    max_active_extensions: int = 64
    revision: int = 1

    def __post_init__(self) -> None:
        allowed = _text_tuple(self.allowed_slots, "allowed_slots")
        protected = _text_tuple(self.protected_slots, "protected_slots")
        if set(allowed) & set(protected):
            raise ValueError("protected slots cannot also be allowed slots")
        object.__setattr__(self, "allowed_slots", allowed)
        object.__setattr__(self, "protected_slots", protected)
        object.__setattr__(self, "max_active_extensions", _positive_int(self.max_active_extensions, "max active extensions"))
        object.__setattr__(self, "revision", _positive_int(self.revision, "policy revision"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": CLIENT_EXTENSION_FORMAT,
            "version": CLIENT_EXTENSION_VERSION,
            "allowed_slots": list(self.allowed_slots),
            "protected_slots": list(self.protected_slots),
            "max_active_extensions": self.max_active_extensions,
            "revision": self.revision,
        }

    @property
    def policy_digest(self) -> str:
        return _digest(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExtensionHostPolicy:
        if payload.get("format") != CLIENT_EXTENSION_FORMAT:
            raise ValueError("unsupported extension host policy format")
        policy = cls(
            allowed_slots=tuple(payload.get("allowed_slots", ())),
            protected_slots=tuple(payload.get("protected_slots", ())),
            max_active_extensions=int(payload.get("max_active_extensions", 0)),
            revision=int(payload.get("revision", 0)),
        )
        if str(payload.get("policy_digest", policy.policy_digest)) != policy.policy_digest:
            raise ValueError("extension host policy digest mismatch")
        return policy


@dataclass(frozen=True)
class ClientExtensionSnapshot:
    snapshot_id: str
    revision: int
    parent_snapshot_id: str
    capability_snapshot_id: str
    policy_digest: str
    manifests: tuple[ClientPluginManifest, ...]
    state_digests: Mapping[str, str]
    format: str = CLIENT_EXTENSION_FORMAT

    def __post_init__(self) -> None:
        if self.format != CLIENT_EXTENSION_FORMAT:
            raise ValueError("unsupported client extension snapshot format")
        _positive_int(self.revision, "extension snapshot revision")
        _text(self.parent_snapshot_id, "parent snapshot id")
        _text(self.capability_snapshot_id, "capability snapshot id")
        _text(self.policy_digest, "policy digest")
        ids = tuple(item.plugin_id for item in self.manifests)
        if len(set(ids)) != len(ids):
            raise ValueError("client extension plugin ids must be unique")
        state_digests = {str(key): _text(value, "state digest") for key, value in self.state_digests.items()}
        if set(state_digests) != set(ids):
            raise ValueError("client extension state digests must match manifests")
        object.__setattr__(self, "state_digests", state_digests)
        if self.snapshot_id != _digest(self._identity_payload()):
            raise ValueError("client extension snapshot digest mismatch")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": CLIENT_EXTENSION_VERSION,
            "revision": self.revision,
            "parent_snapshot_id": self.parent_snapshot_id,
            "capability_snapshot_id": self.capability_snapshot_id,
            "policy_digest": self.policy_digest,
            "manifests": [item.to_payload() for item in self.manifests],
            "state_digests": dict(sorted(self.state_digests.items())),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "snapshot_id": self.snapshot_id}

    @classmethod
    def build(
        cls,
        *,
        revision: int,
        parent_snapshot_id: str,
        capability_snapshot_id: str,
        policy_digest: str,
        manifests: Sequence[ClientPluginManifest],
        states: Mapping[str, Mapping[str, Any]],
    ) -> ClientExtensionSnapshot:
        ordered = tuple(sorted(manifests, key=lambda item: item.plugin_id))
        state_digests = {
            item.plugin_id: _digest(_state(states.get(item.plugin_id))) for item in ordered
        }
        identity = {
            "format": CLIENT_EXTENSION_FORMAT,
            "version": CLIENT_EXTENSION_VERSION,
            "revision": revision,
            "parent_snapshot_id": parent_snapshot_id,
            "capability_snapshot_id": capability_snapshot_id,
            "policy_digest": policy_digest,
            "manifests": [item.to_payload() for item in ordered],
            "state_digests": dict(sorted(state_digests.items())),
        }
        return cls(
            snapshot_id=_digest(identity),
            revision=revision,
            parent_snapshot_id=parent_snapshot_id,
            capability_snapshot_id=capability_snapshot_id,
            policy_digest=policy_digest,
            manifests=ordered,
            state_digests=state_digests,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ClientExtensionSnapshot:
        manifests = tuple(ClientPluginManifest.from_payload(item) for item in payload.get("manifests", ()))
        return cls(
            snapshot_id=str(payload.get("snapshot_id", "")),
            revision=int(payload.get("revision", 0)),
            parent_snapshot_id=str(payload.get("parent_snapshot_id", "")),
            capability_snapshot_id=str(payload.get("capability_snapshot_id", "")),
            policy_digest=str(payload.get("policy_digest", "")),
            manifests=manifests,
            state_digests=dict(payload.get("state_digests", {})),
            format=str(payload.get("format", "")),
        )


@dataclass(frozen=True)
class ExtensionLifecycleRecord:
    plugin_id: str
    plugin_digest: str
    plugin_version: str
    state: str
    snapshot_id: str
    sequence: int
    reason: str = ""
    events: tuple[str, ...] = ()
    audit_digest: str = ""

    def __post_init__(self) -> None:
        if self.state not in CLIENT_EXTENSION_LIFECYCLE_STATES:
            raise ValueError("unsupported client extension lifecycle state")
        for name, value in (
            ("plugin_id", self.plugin_id),
            ("plugin_digest", self.plugin_digest),
            ("plugin_version", self.plugin_version),
            ("snapshot_id", self.snapshot_id),
        ):
            _text(value, name)
        _positive_int(self.sequence, "lifecycle sequence")
        events = _text_tuple(self.events, "lifecycle events") if self.events else ()
        object.__setattr__(self, "events", events)
        expected = _digest(self._identity_payload())
        if self.audit_digest and self.audit_digest != expected:
            raise ValueError("client extension lifecycle audit digest mismatch")
        object.__setattr__(self, "audit_digest", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": CLIENT_EXTENSION_FORMAT,
            "version": CLIENT_EXTENSION_VERSION,
            "plugin_id": self.plugin_id,
            "plugin_digest": self.plugin_digest,
            "plugin_version": self.plugin_version,
            "state": self.state,
            "snapshot_id": self.snapshot_id,
            "sequence": self.sequence,
            "reason": self.reason,
            "events": list(self.events),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "audit_digest": self.audit_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExtensionLifecycleRecord:
        return cls(
            plugin_id=str(payload.get("plugin_id", "")),
            plugin_digest=str(payload.get("plugin_digest", "")),
            plugin_version=str(payload.get("plugin_version", "")),
            state=str(payload.get("state", "")),
            snapshot_id=str(payload.get("snapshot_id", "")),
            sequence=int(payload.get("sequence", 0)),
            reason=str(payload.get("reason", "")),
            events=tuple(payload.get("events", ())),
            audit_digest=str(payload.get("audit_digest", "")),
        )


class ExtensionHostError(ValueError):
    """Base error for fail-closed client extension operations."""


class ExtensionDependencyError(ExtensionHostError):
    """A required service is missing or unhealthy."""


class ExtensionHostBusyError(ExtensionHostError):
    """An in-flight call prevents a snapshot change."""


class ExtensionDisposalError(ExtensionHostError):
    """A trusted host-owned disposer reported one or more cleanup failures."""


class DisposerNode:
    """Host-owned recursive disposer tree; callbacks never come from a manifest."""

    def __init__(self, name: str, callback: Callable[[], None] | None = None) -> None:
        self.name = _text(name, "disposer name")
        self._callback = callback
        self._children: list[DisposerNode] = []
        self._disposed = False

    @property
    def disposed(self) -> bool:
        return self._disposed

    def add(self, child: DisposerNode) -> DisposerNode:
        if self._disposed:
            raise ExtensionDisposalError("cannot add a child to a disposed disposer")
        if not isinstance(child, DisposerNode):
            raise TypeError("disposer child must be a DisposerNode")
        self._children.append(child)
        return child

    def dispose(self) -> tuple[str, ...]:
        if self._disposed:
            return ()
        disposed: list[str] = []
        errors: list[str] = []
        for child in reversed(self._children):
            try:
                disposed.extend(child.dispose())
            except ExtensionDisposalError as exc:
                errors.extend(str(exc).split("; "))
        if self._callback is not None:
            try:
                self._callback()
                disposed.append(self.name)
            except Exception as exc:  # pragma: no cover - message is asserted by host tests
                errors.append(f"{self.name}: {exc}")
        self._disposed = True
        if errors:
            raise ExtensionDisposalError("; ".join(errors))
        if self._callback is None:
            disposed.append(self.name)
        return tuple(disposed)


@dataclass(frozen=True)
class PreparedExtensionSnapshot:
    expected_current_snapshot_id: str
    target_snapshot: ClientExtensionSnapshot
    states: Mapping[str, Mapping[str, Any]]
    dependency_health: Mapping[str, bool]
    prepared_digest: str = ""

    def __post_init__(self) -> None:
        states = {str(key): _state(value) for key, value in self.states.items()}
        object.__setattr__(self, "states", states)
        health = {str(key): bool(value) for key, value in self.dependency_health.items()}
        object.__setattr__(self, "dependency_health", health)
        identity = self._identity_payload()
        expected = _digest(identity)
        if self.prepared_digest and self.prepared_digest != expected:
            raise ValueError("prepared extension snapshot digest mismatch")
        object.__setattr__(self, "prepared_digest", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": CLIENT_EXTENSION_FORMAT,
            "version": CLIENT_EXTENSION_VERSION,
            "expected_current_snapshot_id": self.expected_current_snapshot_id,
            "target_snapshot_id": self.target_snapshot.snapshot_id,
            "states": {key: _digest(value) for key, value in sorted(self.states.items())},
            "dependency_health": dict(sorted(self.dependency_health.items())),
        }


class ClientExtensionHost:
    """Atomic Seed-owned extension lifecycle; plugin source remains out of scope."""

    def __init__(
        self,
        *,
        policy: ExtensionHostPolicy | None = None,
        capability_snapshot_id: str = "capability:root",
        parent_checkpoint_id: str = "checkpoint:client-extension-root",
    ) -> None:
        self.policy = policy or ExtensionHostPolicy()
        capability_snapshot_id = _text(capability_snapshot_id, "capability snapshot id")
        parent_checkpoint_id = _text(parent_checkpoint_id, "parent checkpoint id")
        self._snapshot = ClientExtensionSnapshot.build(
            revision=1,
            parent_snapshot_id=parent_checkpoint_id,
            capability_snapshot_id=capability_snapshot_id,
            policy_digest=self.policy.policy_digest,
            manifests=(),
            states={},
        )
        self._states: dict[str, dict[str, Any]] = {}
        self._health: dict[str, bool] = {}
        self._records: list[ExtensionLifecycleRecord] = []
        self._history: dict[str, tuple[ClientExtensionSnapshot, dict[str, dict[str, Any]]]] = {}
        self._known_manifests: dict[str, ClientPluginManifest] = {}
        self._inflight: dict[str, int] = {}
        self._disposers: dict[str, DisposerNode] = {}
        self._migrators: dict[tuple[str, int, int, str], Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}

    @property
    def snapshot(self) -> ClientExtensionSnapshot:
        return self._snapshot

    @property
    def active_manifests(self) -> tuple[ClientPluginManifest, ...]:
        return self._snapshot.manifests

    @property
    def lifecycle_records(self) -> tuple[ExtensionLifecycleRecord, ...]:
        return tuple(self._records)

    @property
    def dependency_health(self) -> dict[str, bool]:
        return dict(self._health)

    def state(self, plugin_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._states.get(_text(plugin_id, "plugin id"), {}))

    def inflight(self, plugin_id: str) -> int:
        return self._inflight.get(_text(plugin_id, "plugin id"), 0)

    def register_state_migrator(
        self,
        plugin_id: str,
        from_schema_version: int,
        to_schema_version: int,
        migration_id: str,
        callback: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        if not callable(callback):
            raise TypeError("state migrator must be callable")
        key = (
            _text(plugin_id, "plugin id"),
            _positive_int(from_schema_version, "from schema version"),
            _positive_int(to_schema_version, "to schema version"),
            _text(migration_id, "migration id"),
        )
        if key in self._migrators:
            raise ValueError("state migrator is already registered")
        self._migrators[key] = callback

    def attach_disposer(self, plugin_id: str, disposer: DisposerNode) -> None:
        plugin_id = _text(plugin_id, "plugin id")
        if not isinstance(disposer, DisposerNode):
            raise TypeError("disposer must be a DisposerNode")
        if plugin_id not in {item.plugin_id for item in self.active_manifests}:
            raise ExtensionHostError("cannot attach disposer to an inactive extension")
        if plugin_id in self._disposers:
            raise ValueError("plugin already has a disposer")
        self._disposers[plugin_id] = disposer

    def prepare(
        self,
        manifests: Sequence[ClientPluginManifest],
        *,
        capability_snapshot_id: str,
        available_capabilities: Any,
        dependency_health: Mapping[str, bool] | None = None,
        states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> PreparedExtensionSnapshot:
        if isinstance(manifests, (str, bytes)) or not isinstance(manifests, Sequence):
            raise TypeError("client extension manifests must be a sequence")
        if len(manifests) > self.policy.max_active_extensions:
            raise ExtensionHostError("client extension active limit exceeded")
        normalized = tuple(manifests)
        if any(not isinstance(item, ClientPluginManifest) for item in normalized):
            raise TypeError("client extension host accepts ClientPluginManifest objects only")
        ids = tuple(item.plugin_id for item in normalized)
        if len(set(ids)) != len(ids):
            raise ExtensionHostError("client extension plugin ids must be unique")
        allowed = set(self.policy.allowed_slots)
        protected = set(self.policy.protected_slots)
        for manifest in normalized:
            unknown_slots = set(manifest.slots) - allowed
            if unknown_slots:
                raise ExtensionHostError(
                    f"plugin {manifest.plugin_id} requests unsupported slots: "
                    + ", ".join(sorted(unknown_slots))
                )
            if set(manifest.slots) & protected:
                raise ExtensionHostError("client extension cannot mount a protected shell slot")
        capability_ids = self._capability_ids(available_capabilities)
        for manifest in normalized:
            missing = set(manifest.capability_ids) - capability_ids
            if missing:
                raise ExtensionHostError(
                    f"plugin {manifest.plugin_id} requests unavailable capabilities: "
                    + ", ".join(sorted(missing))
                )
        health = dict(self._health)
        if dependency_health is not None:
            health.update({str(key): bool(value) for key, value in dependency_health.items()})
        for manifest in normalized:
            unhealthy = [name for name, _ in manifest.service_dependencies if health.get(name) is not True]
            if unhealthy:
                raise ExtensionDependencyError(
                    f"plugin {manifest.plugin_id} dependencies are unavailable: "
                    + ", ".join(sorted(unhealthy))
                )
        supplied_states = {key: _state(value) for key, value in self._states.items()}
        if states is not None:
            supplied_states.update({str(key): _state(value) for key, value in states.items()})
        current = {item.plugin_id: item for item in self.active_manifests}
        next_states: dict[str, dict[str, Any]] = {}
        for manifest in normalized:
            previous = current.get(manifest.plugin_id)
            previous_state = supplied_states.get(manifest.plugin_id, {})
            if previous is not None and previous.state_schema_version != manifest.state_schema_version:
                if not manifest.migration_id:
                    raise ExtensionHostError(
                        f"plugin {manifest.plugin_id} requires an explicit state migration"
                    )
                migrator = self._migrators.get(
                    (
                        manifest.plugin_id,
                        previous.state_schema_version,
                        manifest.state_schema_version,
                        manifest.migration_id,
                    )
                )
                if migrator is None:
                    raise ExtensionHostError("required client extension state migrator is not registered")
                migrated = migrator(copy.deepcopy(previous_state))
                next_states[manifest.plugin_id] = _state(migrated)
            else:
                next_states[manifest.plugin_id] = previous_state
        target = ClientExtensionSnapshot.build(
            revision=self._snapshot.revision + 1,
            parent_snapshot_id=self._snapshot.snapshot_id,
            capability_snapshot_id=_text(capability_snapshot_id, "capability snapshot id"),
            policy_digest=self.policy.policy_digest,
            manifests=normalized,
            states=next_states,
        )
        prepared = PreparedExtensionSnapshot(
            expected_current_snapshot_id=self._snapshot.snapshot_id,
            target_snapshot=target,
            states=next_states,
            dependency_health=health,
        )
        self._record_all(normalized, "prepared", target.snapshot_id, "two_phase_prepare")
        return prepared

    def commit(self, prepared: PreparedExtensionSnapshot) -> ClientExtensionSnapshot:
        if not isinstance(prepared, PreparedExtensionSnapshot):
            raise TypeError("commit accepts a PreparedExtensionSnapshot")
        if prepared.expected_current_snapshot_id != self._snapshot.snapshot_id:
            raise ExtensionHostError("prepared client extension snapshot is stale")
        self._verify_state_digests(prepared.target_snapshot, prepared.states)
        target_ids = {item.plugin_id for item in prepared.target_snapshot.manifests}
        current_by_id = {item.plugin_id: item for item in self.active_manifests}
        target_by_id = {item.plugin_id: item for item in prepared.target_snapshot.manifests}
        changed_ids = {
            plugin_id
            for plugin_id in set(current_by_id) | target_ids
            if current_by_id.get(plugin_id) != target_by_id.get(plugin_id)
        }
        busy = [plugin_id for plugin_id in changed_ids if self.inflight(plugin_id)]
        if busy:
            raise ExtensionHostBusyError(", ".join(sorted(busy)) + " has in-flight calls")
        for manifest in prepared.target_snapshot.manifests:
            unhealthy = [
                name
                for name, _ in manifest.service_dependencies
                if self._health.get(name, prepared.dependency_health.get(name)) is not True
            ]
            if unhealthy:
                raise ExtensionDependencyError("dependency health changed during prepare")
        old_snapshot = self._snapshot
        self._history[old_snapshot.snapshot_id] = (old_snapshot, copy.deepcopy(self._states))
        self._snapshot = prepared.target_snapshot
        self._states = {key: _state(value) for key, value in prepared.states.items()}
        self._health.update(prepared.dependency_health)
        for manifest in old_snapshot.manifests:
            if manifest.plugin_id not in target_by_id:
                self._record(manifest, "retired", self._snapshot.snapshot_id, "snapshot_replaced")
            elif target_by_id[manifest.plugin_id] != manifest:
                self._record(manifest, "retired", self._snapshot.snapshot_id, "version_replaced")
        self._known_manifests.update({item.plugin_id: item for item in old_snapshot.manifests})
        self._known_manifests.update({item.plugin_id: item for item in self.active_manifests})
        self._record_all(self.active_manifests, "active", self._snapshot.snapshot_id, "two_phase_commit")
        return self._snapshot

    def mount(
        self,
        manifest: ClientPluginManifest,
        *,
        capability_snapshot_id: str,
        available_capabilities: Any,
        dependency_health: Mapping[str, bool] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> ClientExtensionSnapshot:
        states = {manifest.plugin_id: _state(state)}
        prepared = self.prepare(
            (*self.active_manifests, manifest),
            capability_snapshot_id=capability_snapshot_id,
            available_capabilities=available_capabilities,
            dependency_health=dependency_health,
            states=states,
        )
        return self.commit(prepared)

    def retire(self, plugin_id: str, *, reason: str = "explicit_retire") -> ClientExtensionSnapshot:
        plugin_id = _text(plugin_id, "plugin id")
        manifest = next((item for item in self.active_manifests if item.plugin_id == plugin_id), None)
        if manifest is None:
            raise ExtensionHostError("client extension is not active")
        prepared = self.prepare(
            tuple(item for item in self.active_manifests if item.plugin_id != plugin_id),
            capability_snapshot_id=self.snapshot.capability_snapshot_id,
            available_capabilities=self._active_capability_ids(excluding=plugin_id),
            dependency_health=self._health,
        )
        snapshot = self.commit(prepared)
        self._record(manifest, "draining", snapshot.snapshot_id, reason)
        if self.inflight(plugin_id):
            raise ExtensionHostBusyError(f"{plugin_id} has in-flight calls")
        self.release(plugin_id)
        self._record(manifest, "retired", snapshot.snapshot_id, reason)
        return snapshot

    def quarantine(self, plugin_id: str, *, reason: str) -> ClientExtensionSnapshot:
        plugin_id = _text(plugin_id, "plugin id")
        manifest = next((item for item in self.active_manifests if item.plugin_id == plugin_id), None)
        if manifest is None:
            raise ExtensionHostError("client extension is not active")
        prepared = self.prepare(
            tuple(item for item in self.active_manifests if item.plugin_id != plugin_id),
            capability_snapshot_id=self.snapshot.capability_snapshot_id,
            available_capabilities=self._active_capability_ids(excluding=plugin_id),
            dependency_health=self._health,
        )
        snapshot = self.commit(prepared)
        self._record(manifest, "quarantined", snapshot.snapshot_id, reason)
        return snapshot

    def report_dependency(self, service: str, healthy: bool) -> tuple[str, ...]:
        service = _text(service, "service name")
        self._health[service] = bool(healthy)
        affected = tuple(
            item.plugin_id
            for item in self.active_manifests
            if any(name == service for name, _ in item.service_dependencies)
        )
        if healthy:
            for plugin_id in affected:
                self._record(
                    self._known_manifests.get(plugin_id)
                    or next(item for item in self.active_manifests if item.plugin_id == plugin_id),
                    "dependency_recovered",
                    self.snapshot.snapshot_id,
                    f"service:{service}",
                )
            return affected
        for plugin_id in affected:
            self.quarantine(plugin_id, reason=f"dependency_lost:{service}")
        return affected

    def begin_call(self, plugin_id: str) -> None:
        plugin_id = _text(plugin_id, "plugin id")
        if plugin_id not in {item.plugin_id for item in self.active_manifests}:
            raise ExtensionHostError("cannot begin a call for an inactive extension")
        self._inflight[plugin_id] = self.inflight(plugin_id) + 1

    def end_call(self, plugin_id: str) -> None:
        plugin_id = _text(plugin_id, "plugin id")
        count = self.inflight(plugin_id)
        if count < 1:
            raise ExtensionHostError("client extension has no in-flight call")
        if count == 1:
            self._inflight.pop(plugin_id, None)
        else:
            self._inflight[plugin_id] = count - 1

    def drain(self, plugin_id: str) -> bool:
        plugin_id = _text(plugin_id, "plugin id")
        if self.inflight(plugin_id):
            self._record(
                self._known_manifests.get(plugin_id)
                or next(item for item in self.active_manifests if item.plugin_id == plugin_id),
                "draining",
                self.snapshot.snapshot_id,
                "in_flight_calls",
            )
            return False
        return True

    def release(self, plugin_id: str) -> tuple[str, ...]:
        plugin_id = _text(plugin_id, "plugin id")
        if self.inflight(plugin_id):
            raise ExtensionHostBusyError(f"{plugin_id} has in-flight calls")
        disposer = self._disposers.pop(plugin_id, None)
        if disposer is None:
            return ()
        try:
            return disposer.dispose()
        except ExtensionDisposalError as exc:
            manifest = self._known_manifests.get(plugin_id)
            if manifest is not None:
                self._record(manifest, "failed", self.snapshot.snapshot_id, str(exc))
            raise

    def rollback(self, snapshot_id: str) -> ClientExtensionSnapshot:
        snapshot_id = _text(snapshot_id, "snapshot id")
        if snapshot_id == self._snapshot.snapshot_id:
            return self._snapshot
        entry = self._history.get(snapshot_id)
        if entry is None:
            raise ExtensionHostError("requested client extension snapshot is not in rollback history")
        target, states = entry
        current_ids = {item.plugin_id for item in self.active_manifests}
        target_ids = {item.plugin_id for item in target.manifests}
        busy = [plugin_id for plugin_id in current_ids | target_ids if self.inflight(plugin_id)]
        if busy:
            raise ExtensionHostBusyError(", ".join(sorted(busy)) + " has in-flight calls")
        old = self._snapshot
        self._history[old.snapshot_id] = (old, copy.deepcopy(self._states))
        self._snapshot = target
        self._states = copy.deepcopy(states)
        self._record_all(self.active_manifests, "rolled_back", target.snapshot_id, "parent_snapshot_restore")
        return target

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": CLIENT_EXTENSION_CHECKPOINT_FORMAT,
            "version": CLIENT_EXTENSION_VERSION,
            "policy": {**self.policy.to_payload(), "policy_digest": self.policy.policy_digest},
            "snapshot": self._snapshot.to_payload(),
            "states": copy.deepcopy(self._states),
            "dependency_health": dict(sorted(self._health.items())),
            "history": [
                {
                    "snapshot": snapshot.to_payload(),
                    "states": copy.deepcopy(states),
                }
                for snapshot, states in (self._history[key] for key in sorted(self._history))
            ],
            "lifecycle_records": [item.to_payload() for item in self._records],
        }
        payload["checkpoint_digest"] = _digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> ClientExtensionHost:
        if payload.get("format") != CLIENT_EXTENSION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported client extension checkpoint format")
        expected = _digest({key: value for key, value in payload.items() if key != "checkpoint_digest"})
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("client extension checkpoint digest mismatch")
        policy_payload = dict(payload.get("policy") or {})
        policy = ExtensionHostPolicy.from_payload(policy_payload)
        snapshot = ClientExtensionSnapshot.from_payload(payload.get("snapshot") or {})
        if snapshot.policy_digest != policy.policy_digest:
            raise ValueError("client extension checkpoint policy mismatch")
        host = cls(
            policy=policy,
            capability_snapshot_id=snapshot.capability_snapshot_id,
            parent_checkpoint_id=snapshot.parent_snapshot_id,
        )
        host._snapshot = snapshot
        host._states = {str(key): _state(value) for key, value in (payload.get("states") or {}).items()}
        if set(host._states) != {item.plugin_id for item in snapshot.manifests}:
            raise ValueError("client extension checkpoint state mismatch")
        host._verify_state_digests(snapshot, host._states)
        host._health = {str(key): bool(value) for key, value in (payload.get("dependency_health") or {}).items()}
        host._history = {}
        for item in payload.get("history", ()):
            history_snapshot = ClientExtensionSnapshot.from_payload(item.get("snapshot") or {})
            history_states = {
                str(key): _state(value) for key, value in (item.get("states") or {}).items()
            }
            if set(history_states) != {manifest.plugin_id for manifest in history_snapshot.manifests}:
                raise ValueError("client extension history state mismatch")
            host._verify_state_digests(history_snapshot, history_states)
            host._history[history_snapshot.snapshot_id] = (history_snapshot, history_states)
        host._records = [
            ExtensionLifecycleRecord.from_payload(item)
            for item in payload.get("lifecycle_records", ())
        ]
        expected_sequence = tuple(range(1, len(host._records) + 1))
        if tuple(item.sequence for item in host._records) != expected_sequence:
            raise ValueError("client extension lifecycle sequence mismatch")
        known_manifests = list(snapshot.manifests)
        for history_snapshot, _ in host._history.values():
            known_manifests.extend(history_snapshot.manifests)
        host._known_manifests = {item.plugin_id: item for item in known_manifests}
        return host

    @staticmethod
    def _capability_ids(value: Any) -> set[str]:
        if hasattr(value, "capabilities"):
            value = getattr(value, "capabilities")
        if isinstance(value, Mapping):
            return {str(key) for key in value}
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError("available_capabilities must be a capability snapshot or sequence")
        ids: set[str] = set()
        for item in value:
            ids.add(str(getattr(item, "capability_id", item)))
        return ids

    def _active_capability_ids(self, *, excluding: str = "") -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    capability_id
                    for manifest in self.active_manifests
                    if manifest.plugin_id != excluding
                    for capability_id in manifest.capability_ids
                }
            )
        )

    @staticmethod
    def _verify_state_digests(
        snapshot: ClientExtensionSnapshot,
        states: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for plugin_id, state_digest in snapshot.state_digests.items():
            if _digest(_state(states.get(plugin_id))) != state_digest:
                raise ExtensionHostError("client extension state digest mismatch")

    def _record_all(
        self,
        manifests: Sequence[ClientPluginManifest],
        state: str,
        snapshot_id: str,
        reason: str,
    ) -> None:
        for manifest in sorted(manifests, key=lambda item: item.plugin_id):
            self._record(manifest, state, snapshot_id, reason)

    def _record(
        self,
        manifest: ClientPluginManifest,
        state: str,
        snapshot_id: str,
        reason: str,
    ) -> ExtensionLifecycleRecord:
        self._known_manifests[manifest.plugin_id] = manifest
        record = ExtensionLifecycleRecord(
            plugin_id=manifest.plugin_id,
            plugin_digest=manifest.plugin_digest,
            plugin_version=manifest.version,
            state=state,
            snapshot_id=snapshot_id,
            sequence=len(self._records) + 1,
            reason=str(reason).strip(),
            events=(state, reason) if reason else (state,),
        )
        self._records.append(record)
        return record


__all__ = [
    "CLIENT_EXTENSION_CHECKPOINT_FORMAT",
    "CLIENT_EXTENSION_FORMAT",
    "CLIENT_EXTENSION_LIFECYCLE_STATES",
    "CLIENT_EXTENSION_VERSION",
    "ClientExtensionHost",
    "ClientExtensionSnapshot",
    "ClientPluginManifest",
    "DisposerNode",
    "ExtensionDependencyError",
    "ExtensionDisposalError",
    "ExtensionHostBusyError",
    "ExtensionHostError",
    "ExtensionHostPolicy",
    "ExtensionLifecycleRecord",
    "PreparedExtensionSnapshot",
]

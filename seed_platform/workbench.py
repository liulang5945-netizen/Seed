"""Seed-owned read-only workbench capability contract and environment.

The workbench is an execution boundary, not a second cognitive system. Taiji
produces a structured ``ActionIntent``/``ToolCall``; this module exposes the
currently available read-only capabilities, validates the capability snapshot,
executes safe filesystem reads, and records an auditable event stream.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taiji.environment import EnvironmentOutcome

from .paths import get_external_path
from .settings import get_setting

WORKBENCH_CONTRACT_FORMAT = "seed-workbench-contract-v1"
WORKBENCH_CONTRACT_VERSION = 1
WORKBENCH_MAX_READ_BYTES = 1_048_576
WORKBENCH_MAX_SEARCH_RESULTS = 100
# Workbench sensations cross into Taiji's native byte sensor. Keep the
# digest-derived marker inside the raw-byte domain; Taiji's boundary symbol
# is reserved for stream framing and is not emitted here.
WORKBENCH_SENSATION_SYMBOL_COUNT = 256


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_workspace_root() -> Path:
    """Resolve the current product workspace without importing Legacy."""

    configured = str(get_setting("workspace_path", "") or "").strip()
    if configured and Path(configured).is_dir():
        return Path(configured).resolve()
    return Path(get_external_path("agent_workspace")).resolve()


@dataclass(frozen=True)
class CapabilityDescriptor:
    capability_id: str
    description: str
    risk: str = "read_only"
    reversible: bool = True
    source: str = "seed.workbench"
    category: str = "workbench"
    parameters: tuple[tuple[str, str], ...] = ()
    enabled: bool = True
    version: int = WORKBENCH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version != WORKBENCH_CONTRACT_VERSION:
            raise ValueError("unsupported workbench capability version")
        if not self.capability_id.strip():
            raise ValueError("capability_id cannot be empty")
        if not self.description.strip():
            raise ValueError("capability description cannot be empty")
        if len({name for name, _ in self.parameters}) != len(self.parameters):
            raise ValueError("capability parameter names must be unique")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "capability_id": self.capability_id,
            "description": self.description,
            "risk": self.risk,
            "reversible": self.reversible,
            "source": self.source,
            "category": self.category,
            "parameters": {name: schema for name, schema in self.parameters},
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Content-addressed set of capabilities visible to one workbench."""

    snapshot_id: str
    revision: int
    capabilities: tuple[CapabilityDescriptor, ...]
    format: str = WORKBENCH_CONTRACT_FORMAT

    def __post_init__(self) -> None:
        if self.format != WORKBENCH_CONTRACT_FORMAT:
            raise ValueError("unsupported workbench contract format")
        if self.revision < 1:
            raise ValueError("workbench revision must be positive")
        ids = tuple(item.capability_id for item in self.capabilities)
        if len(set(ids)) != len(ids):
            raise ValueError("workbench capability ids must be unique")
        if not self.snapshot_id:
            raise ValueError("workbench snapshot_id cannot be empty")

    @classmethod
    def default(cls) -> CapabilitySnapshot:
        capabilities = (
            CapabilityDescriptor(
                "workspace.list",
                "List entries in the active workspace.",
                parameters=(("path", "relative directory path, default ."),),
            ),
            CapabilityDescriptor(
                "workspace.read",
                "Read one UTF-8 or binary-safe file snapshot.",
                parameters=(("path", "relative file path"),),
            ),
            CapabilityDescriptor(
                "workspace.stat",
                "Read metadata for one workspace entry.",
                parameters=(("path", "relative path"),),
            ),
            CapabilityDescriptor(
                "workspace.search",
                "Search text within the active workspace.",
                parameters=(
                    ("query", "non-empty text"),
                    ("path", "optional relative directory path"),
                ),
            ),
            CapabilityDescriptor(
                "editor.open",
                "Project an existing workspace file as the active editor tab.",
                risk="reversible_ui",
                category="editor",
                parameters=(("path", "relative file path"),),
            ),
            CapabilityDescriptor(
                "editor.diagnostics.read",
                "Read diagnostics from the connected editor.",
                risk="read_only",
                category="editor",
                enabled=False,
            ),
        )
        body = {
            "format": WORKBENCH_CONTRACT_FORMAT,
            "version": WORKBENCH_CONTRACT_VERSION,
            "revision": 1,
            "capabilities": [item.to_payload() for item in capabilities],
        }
        return cls(
            snapshot_id=_canonical_digest(body),
            revision=1,
            capabilities=capabilities,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": WORKBENCH_CONTRACT_VERSION,
            "snapshot_id": self.snapshot_id,
            "revision": self.revision,
            "capabilities": [item.to_payload() for item in self.capabilities],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CapabilitySnapshot:
        if payload.get("format") != WORKBENCH_CONTRACT_FORMAT:
            raise ValueError("unsupported workbench contract format")
        capabilities: list[CapabilityDescriptor] = []
        for raw in payload.get("capabilities", []):
            if not isinstance(raw, Mapping):
                raise ValueError("workbench capability must be an object")
            parameters = raw.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise ValueError("workbench capability parameters must be an object")
            capabilities.append(
                CapabilityDescriptor(
                    capability_id=str(raw["capability_id"]),
                    description=str(raw["description"]),
                    risk=str(raw.get("risk", "read_only")),
                    reversible=bool(raw.get("reversible", True)),
                    source=str(raw.get("source", "seed.workbench")),
                    category=str(raw.get("category", "workbench")),
                    parameters=tuple(
                        (str(name), str(schema)) for name, schema in parameters.items()
                    ),
                    enabled=bool(raw.get("enabled", True)),
                    version=int(raw.get("version", WORKBENCH_CONTRACT_VERSION)),
                )
            )
        revision = int(payload.get("revision", 0))
        body = {
            "format": WORKBENCH_CONTRACT_FORMAT,
            "version": WORKBENCH_CONTRACT_VERSION,
            "revision": revision,
            "capabilities": [item.to_payload() for item in capabilities],
        }
        snapshot_id = str(payload.get("snapshot_id", ""))
        if snapshot_id != _canonical_digest(body):
            raise ValueError("workbench capability snapshot digest mismatch")
        return cls(snapshot_id=snapshot_id, revision=revision, capabilities=tuple(capabilities))

    def get(self, capability_id: str) -> CapabilityDescriptor | None:
        return next(
            (item for item in self.capabilities if item.capability_id == capability_id),
            None,
        )


@dataclass(frozen=True)
class WorkbenchActionRequest:
    request_id: str
    intent_id: str
    capability_id: str
    parameters: Mapping[str, Any]
    snapshot_id: str
    source: str = "taiji"
    version: int = WORKBENCH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version != WORKBENCH_CONTRACT_VERSION:
            raise ValueError("unsupported workbench action version")
        for value, name in (
            (self.request_id, "request_id"),
            (self.intent_id, "intent_id"),
            (self.capability_id, "capability_id"),
            (self.snapshot_id, "snapshot_id"),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} cannot be empty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("workbench action parameters must be a mapping")

    @classmethod
    def from_action_intent(
        cls,
        intent: Any,
        *,
        snapshot_id: str,
        request_id: str | None = None,
    ) -> WorkbenchActionRequest:
        """Bind a Taiji-owned intent to the current Seed capability snapshot."""

        if not hasattr(intent, "intent_id") or not hasattr(intent, "kind"):
            raise TypeError("intent must provide intent_id and kind")
        return cls(
            request_id=request_id or f"workbench:{intent.intent_id}",
            intent_id=str(intent.intent_id),
            capability_id=str(intent.kind),
            parameters=dict(getattr(intent, "parameters", {}) or {}),
            snapshot_id=str(snapshot_id),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "intent_id": self.intent_id,
            "capability_id": self.capability_id,
            "parameters": dict(self.parameters),
            "snapshot_id": self.snapshot_id,
            "source": self.source,
        }


@dataclass(frozen=True)
class ExecutionPolicyDecision:
    request_id: str
    capability_id: str
    decision: str
    reason_code: str
    snapshot_id: str
    version: int = WORKBENCH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.decision not in {"allow", "deny", "ask_user"}:
            raise ValueError("unsupported workbench policy decision")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "capability_id": self.capability_id,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True)
class WorkbenchTransaction:
    operation: str
    path: str
    before_digest: str = ""
    after_digest: str = ""
    reversible: bool = True
    version: int = WORKBENCH_CONTRACT_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "operation": self.operation,
            "path": self.path,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "reversible": self.reversible,
        }


@dataclass(frozen=True)
class WorkbenchOutcome:
    request_id: str
    intent_id: str
    call_id: str
    capability_id: str
    snapshot_id: str
    status: str
    success: bool
    result: Mapping[str, Any] = field(default_factory=dict)
    error_code: str = ""
    error: str = ""
    transaction: WorkbenchTransaction | None = None
    tick: int = 0
    version: int = WORKBENCH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"success", "rejected", "error"}:
            raise ValueError("unsupported workbench outcome status")
        if self.tick < 0:
            raise ValueError("workbench outcome tick cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "intent_id": self.intent_id,
            "call_id": self.call_id,
            "capability_id": self.capability_id,
            "snapshot_id": self.snapshot_id,
            "status": self.status,
            "success": self.success,
            "result": dict(self.result),
            "error_code": self.error_code,
            "error": self.error,
            "transaction": (None if self.transaction is None else self.transaction.to_payload()),
            "tick": self.tick,
        }


@dataclass(frozen=True)
class WorkbenchEvent:
    sequence: int
    phase: str
    request_id: str
    tick: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    version: int = WORKBENCH_CONTRACT_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sequence": self.sequence,
            "phase": self.phase,
            "request_id": self.request_id,
            "tick": self.tick,
            "payload": dict(self.payload),
        }


class WorkbenchAuditLog:
    """Small bounded event stream shared by chat, IDE and API observers."""

    def __init__(self, *, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("workbench audit capacity must be positive")
        self.capacity = int(capacity)
        self._events: list[WorkbenchEvent] = []
        self._next_sequence = 1
        self._lock = threading.RLock()

    @property
    def events(self) -> tuple[WorkbenchEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def append(
        self,
        phase: str,
        request_id: str,
        *,
        tick: int,
        payload: Mapping[str, Any] | None = None,
    ) -> WorkbenchEvent:
        if phase not in {"planned", "policy", "executing", "outcome"}:
            raise ValueError("unsupported workbench event phase")
        with self._lock:
            event = WorkbenchEvent(
                sequence=self._next_sequence,
                phase=phase,
                request_id=request_id,
                tick=int(tick),
                payload=dict(payload or {}),
            )
            self._next_sequence += 1
            self._events.append(event)
            del self._events[: -self.capacity]
            return event

    def to_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "format": WORKBENCH_CONTRACT_FORMAT,
                "version": WORKBENCH_CONTRACT_VERSION,
                "capacity": self.capacity,
                "next_sequence": self._next_sequence,
                "events": [event.to_payload() for event in self._events],
            }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> WorkbenchAuditLog:
        data = dict(payload or {})
        if not data:
            return cls()
        if data.get("format") != WORKBENCH_CONTRACT_FORMAT:
            raise ValueError("unsupported workbench audit format")
        audit = cls(capacity=int(data.get("capacity", 256)))
        events: list[WorkbenchEvent] = []
        for raw in data.get("events", []):
            if not isinstance(raw, Mapping):
                continue
            events.append(
                WorkbenchEvent(
                    sequence=int(raw["sequence"]),
                    phase=str(raw["phase"]),
                    request_id=str(raw["request_id"]),
                    tick=int(raw.get("tick", 0)),
                    payload=dict(raw.get("payload") or {}),
                )
            )
        audit._events = events[-audit.capacity :]
        audit._next_sequence = max(
            int(data.get("next_sequence", 1)),
            max((event.sequence for event in audit._events), default=0) + 1,
        )
        return audit


class WorkbenchPathError(ValueError):
    """A requested path is not safe within the active workspace."""


class WorkbenchEnvironment:
    """Read-only Seed workbench implementing Taiji's tool-environment protocol."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        snapshot: CapabilitySnapshot | None = None,
    ) -> None:
        self.root = Path(root or default_workspace_root()).resolve()
        self.snapshot = snapshot or CapabilitySnapshot.default()
        self._last_result: dict[str, Any] = {}
        self._request_id = ""
        self._lock = threading.RLock()

    @property
    def capability_snapshot(self) -> CapabilitySnapshot:
        return self.snapshot

    @property
    def last_result(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_result)

    @contextlib.contextmanager
    def request_context(self, request_id: str) -> Iterator[None]:
        with self._lock:
            previous = self._request_id
            self._request_id = str(request_id)
        try:
            yield
        finally:
            with self._lock:
                self._request_id = previous

    def status(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "snapshot_id": self.snapshot.snapshot_id,
            "revision": self.snapshot.revision,
            "capabilities": [item.to_payload() for item in self.snapshot.capabilities],
        }

    def policy_for(self, request: WorkbenchActionRequest) -> ExecutionPolicyDecision:
        if request.snapshot_id != self.snapshot.snapshot_id:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "deny",
                "stale_capability_snapshot",
                self.snapshot.snapshot_id,
            )
        descriptor = self.snapshot.get(request.capability_id)
        if descriptor is None:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "deny",
                "unknown_capability",
                self.snapshot.snapshot_id,
            )
        if not descriptor.enabled:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "deny",
                "capability_not_connected",
                self.snapshot.snapshot_id,
            )
        if descriptor.risk not in {"read_only", "reversible_ui"}:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "ask_user",
                "capability_requires_approval",
                self.snapshot.snapshot_id,
            )
        return ExecutionPolicyDecision(
            request.request_id,
            request.capability_id,
            "allow",
            "read_only_or_reversible",
            self.snapshot.snapshot_id,
        )

    def execute_tool(
        self,
        tool_name: str,
        parameters: Mapping[str, Any],
    ) -> EnvironmentOutcome:
        """Execute one registered read-only capability."""

        with self._lock:
            self._last_result = {}
        try:
            if tool_name == "workspace.list":
                result = self._list_workspace(parameters)
            elif tool_name == "workspace.read":
                result = self._read_workspace(parameters)
            elif tool_name == "workspace.stat":
                result = self._stat_workspace(parameters)
            elif tool_name == "workspace.search":
                result = self._search_workspace(parameters)
            elif tool_name == "editor.open":
                result = self._open_editor(parameters)
            else:
                return self._failure("unknown_capability", "capability is not registered")
        except WorkbenchPathError as exc:
            return self._failure("unsafe_path", str(exc))
        except FileNotFoundError:
            return self._failure("not_found", "workspace entry does not exist")
        except IsADirectoryError:
            return self._failure("not_a_file", "workspace path is a directory")
        except OSError as exc:
            return self._failure("workspace_io_error", str(exc))
        except ValueError as exc:
            return self._failure("invalid_parameters", str(exc))

        with self._lock:
            self._last_result = dict(result)
        return EnvironmentOutcome(
            sensation=self._sensation(tool_name, result),
            reward=1.0,
            terminal=True,
            success=True,
        )

    def _failure(self, code: str, message: str) -> EnvironmentOutcome:
        with self._lock:
            self._last_result = {"error_code": code, "error": message}
        return EnvironmentOutcome(
            sensation=self._sensation(code, {"error": message}),
            reward=-1.0,
            terminal=True,
            success=False,
        )

    @staticmethod
    def _sensation(name: str, result: Mapping[str, Any]) -> int:
        digest = _canonical_digest({"name": name, "result": dict(result)})
        return int(digest[:2], 16) % WORKBENCH_SENSATION_SYMBOL_COUNT

    def _resolve_path(self, raw: Any, *, allow_root: bool = True) -> tuple[Path, str]:
        value = str(raw or ".").strip()
        candidate = Path(value)
        if candidate.is_absolute():
            raise WorkbenchPathError("absolute paths are not accepted")
        resolved = (self.root / candidate).resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkbenchPathError("path escapes the active workspace") from exc
        if not allow_root and resolved == self.root:
            raise WorkbenchPathError("workspace root is not a file")
        relative_name = "." if str(relative) == "." else relative.as_posix()
        return resolved, relative_name

    def _list_workspace(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        directory, relative_name = self._resolve_path(parameters.get("path", "."))
        if not directory.exists():
            raise FileNotFoundError(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        entries = []
        for item in sorted(directory.iterdir(), key=lambda path: path.name.lower()):
            stat = item.stat()
            entries.append(
                {
                    "name": item.name,
                    "path": item.relative_to(self.root).as_posix(),
                    "type": "directory" if item.is_dir() else "file",
                    "size": int(stat.st_size) if item.is_file() else 0,
                }
            )
        return {"path": relative_name, "entries": entries}

    def _read_workspace(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        path, relative_name = self._resolve_path(parameters.get("path"), allow_root=False)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise IsADirectoryError(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        truncated = len(raw) > WORKBENCH_MAX_READ_BYTES
        visible = raw[:WORKBENCH_MAX_READ_BYTES]
        try:
            content = visible.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = visible.decode("utf-8", errors="replace")
            encoding = "binary-utf8-replacement"
        return {
            "path": relative_name,
            "content": content,
            "encoding": encoding,
            "byte_length": len(raw),
            "digest": digest,
            "truncated": truncated,
        }

    def _stat_workspace(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        path, relative_name = self._resolve_path(parameters.get("path", "."))
        if not path.exists():
            return {"path": relative_name, "exists": False, "type": "missing"}
        stat = path.stat()
        return {
            "path": relative_name,
            "exists": True,
            "type": "directory" if path.is_dir() else "file",
            "size": int(stat.st_size) if path.is_file() else 0,
            "modified_ns": int(stat.st_mtime_ns),
        }

    def _search_workspace(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        query = str(parameters.get("query", ""))
        if not query:
            raise ValueError("search query cannot be empty")
        base, _ = self._resolve_path(parameters.get("path", "."))
        if not base.exists():
            raise FileNotFoundError(base)
        if not base.is_dir():
            raise NotADirectoryError(base)
        results: list[dict[str, Any]] = []
        ignored = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}
        for directory, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(name for name in dirnames if name not in ignored)
            for filename in sorted(filenames):
                path = Path(directory) / filename
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if query in line:
                        results.append(
                            {
                                "path": path.relative_to(self.root).as_posix(),
                                "line": line_number,
                                "preview": line[:500],
                            }
                        )
                        if len(results) >= WORKBENCH_MAX_SEARCH_RESULTS:
                            return {"query": query, "results": results, "truncated": True}
        return {"query": query, "results": results, "truncated": False}

    def _open_editor(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        path, relative_name = self._resolve_path(parameters.get("path"), allow_root=False)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise IsADirectoryError(path)
        return {
            "path": relative_name,
            "opened": True,
            "editor_state": "projected",
        }

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
import math
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taiji.environment import EnvironmentOutcome

from .mcp_registry import McpToolRegistry
from .paths import get_external_path
from .programming_languages import (
    LANGUAGE_CONFIDENCE_THRESHOLD,
    ProgrammingLanguageAssessment,
    ProgrammingLanguageRegistry,
)
from .settings import get_setting

WORKBENCH_CONTRACT_FORMAT = "seed-workbench-contract-v1"
WORKBENCH_CONTRACT_VERSION = 1
WORKBENCH_MAX_READ_BYTES = 1_048_576
WORKBENCH_MAX_SEARCH_RESULTS = 100
WORKBENCH_MAX_WRITE_BYTES = 1_048_576
WORKBENCH_MAX_TERMINAL_OUTPUT_BYTES = 1_048_576
WORKBENCH_MAX_TERMINAL_TIMEOUT_SECONDS = 120.0
WORKBENCH_APPROVAL_TTL_SECONDS = 300.0
WORKBENCH_MAX_UNDO_RECORDS = 64
WORKBENCH_LOOP_CONTRACT_FORMAT = "seed-workbench-loop-v1"
WORKBENCH_LOOP_CONTRACT_VERSION = 1
WORKBENCH_MAX_LOOP_STEPS = 8
WORKBENCH_MAX_LOOP_BUDGET_UNITS = 32.0
WORKBENCH_TAIJI_EVIDENCE_KIND = "workbench.evidence"
WORKBENCH_TAIJI_SUCCESSOR_GRAPH_FORMAT = "seed-taiji-successor-graph-v1"
WORKBENCH_TAIJI_SUCCESSOR_GRAPH_VERSION = 1
WORKBENCH_TAIJI_SUCCESSOR_STEP_BUDGET_UNITS = 1.0
WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_FORMAT = "seed-taiji-recovery-portfolio-v1"
WORKBENCH_TAIJI_RECOVERY_PORTFOLIO_VERSION = 1
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


def validate_workspace_root(raw_root: Any) -> Path:
    """Validate and normalize a user-selected workspace directory.

    Workspace selection is a platform concern, not a Legacy workspace route
    concern.  Keep the policy derived from the host environment so it works
    on Windows and POSIX without embedding one machine's absolute paths.
    """

    value = str(raw_root or "").strip()
    if not value:
        raise ValueError("workspace path cannot be empty")
    candidate = Path(os.path.expandvars(os.path.expanduser(value)))
    if not candidate.is_absolute():
        raise ValueError("workspace path must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("workspace path cannot be resolved") from exc
    if not resolved.is_dir():
        raise ValueError("workspace path must be an existing directory")

    filesystem_root = Path(resolved.anchor)
    if resolved == filesystem_root or resolved.parent == filesystem_root:
        raise ValueError("workspace path cannot be a filesystem root or direct child")

    protected_roots = {
        Path(value).resolve()
        for key in ("WINDIR", "ProgramFiles", "ProgramFiles(x86)", "ProgramData")
        if (value := os.environ.get(key))
    }
    protected_roots.update(
        Path(path).resolve()
        for path in ("/etc", "/bin", "/usr", "/var", "/root", "/home")
        if Path(path).exists()
    )
    if any(resolved == root or root in resolved.parents for root in protected_roots):
        raise ValueError("workspace path is a protected system directory")
    return resolved


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

    @property
    def parameter_names(self) -> frozenset[str]:
        """Return the parameter names declared by this capability contract."""

        return frozenset(name for name, _ in self.parameters)


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
                category="workspace",
                parameters=(("path", "relative directory path, default ."),),
            ),
            CapabilityDescriptor(
                "workspace.read",
                "Read one UTF-8 or binary-safe file snapshot.",
                category="workspace",
                parameters=(("path", "relative file path"),),
            ),
            CapabilityDescriptor(
                "workspace.stat",
                "Read metadata for one workspace entry.",
                category="workspace",
                parameters=(("path", "relative path"),),
            ),
            CapabilityDescriptor(
                "workspace.search",
                "Search text within the active workspace.",
                category="workspace",
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
            CapabilityDescriptor(
                "workspace.programming_language.resolve",
                "Resolve a file's programming language from content and workspace evidence.",
                category="language",
                parameters=(
                    ("path", "relative file path"),
                    (
                        "lsp_language_id",
                        "optional connected language-service selection",
                    ),
                ),
            ),
            CapabilityDescriptor(
                "editor.set_language",
                "Project a reversible programming-language selection into the editor.",
                risk="reversible_ui",
                category="editor",
                parameters=(
                    ("path", "relative file path"),
                    ("programming_language_id", "registry language id"),
                    ("user_override", "whether to persist an explicit user override"),
                ),
            ),
        )
        capabilities += (
            CapabilityDescriptor(
                "workspace.apply_patch",
                "Apply a structured UTF-8 text patch with digest and undo checks.",
                risk="file_write",
                category="workspace",
                parameters=(
                    ("path", "relative file path"),
                    ("before_digest", "required current file SHA-256"),
                    ("patch", "structured text replacement operations"),
                    ("expected_after_digest", "required expected resulting SHA-256"),
                ),
            ),
            CapabilityDescriptor(
                "workspace.create",
                "Create one new UTF-8 file with an undoable transaction.",
                risk="file_write",
                category="workspace",
                parameters=(
                    ("path", "relative new file path"),
                    ("content", "UTF-8 file content"),
                ),
            ),
            CapabilityDescriptor(
                "workspace.rename",
                "Rename one workspace file with a digest-checked transaction.",
                risk="file_write",
                category="workspace",
                parameters=(
                    ("path", "relative current file path"),
                    ("new_path", "relative destination file path"),
                    ("before_digest", "required current file SHA-256"),
                ),
            ),
            CapabilityDescriptor(
                "workspace.delete",
                "Delete one file with a digest check and undo token.",
                risk="file_write",
                category="workspace",
                parameters=(
                    ("path", "relative file path"),
                    ("before_digest", "required current file SHA-256"),
                ),
            ),
            CapabilityDescriptor(
                "workspace.undo",
                "Undo one previously committed file transaction if state is unchanged.",
                risk="file_write",
                category="workspace",
                parameters=(("undo_token", "single-use undo token"),),
            ),
            CapabilityDescriptor(
                "terminal.run",
                "Run one bounded argv command without a shell and return auditable output.",
                risk="terminal",
                category="terminal",
                parameters=(
                    ("argv", "non-empty argument vector; shell syntax is not accepted"),
                    ("cwd", "relative workspace directory"),
                    ("timeout_seconds", "bounded command timeout"),
                    ("env", "environment additions"),
                    ("env_allowlist", "allowed environment variable names"),
                    ("output_limit", "maximum captured bytes per stream"),
                    ("expected_artifacts", "relative paths expected after execution"),
                ),
            ),
            CapabilityDescriptor(
                "mcp.list",
                "List native MCP-shaped tools from the Seed-owned registry.",
                category="mcp",
            ),
            CapabilityDescriptor(
                "mcp.invoke",
                "Invoke one registry-validated MCP-shaped tool through Workbench policy.",
                risk="mcp_dispatch",
                category="mcp",
                parameters=(
                    ("tool_id", "registered MCP tool id"),
                    ("arguments", "validated tool arguments"),
                    ("registry_revision", "optional expected registry revision"),
                ),
            ),
        )
        body = {
            "format": WORKBENCH_CONTRACT_FORMAT,
            "version": WORKBENCH_CONTRACT_VERSION,
            "revision": 4,
            "capabilities": [item.to_payload() for item in capabilities],
        }
        return cls(
            snapshot_id=_canonical_digest(body),
            revision=4,
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

    def to_taiji_affordances(
        self,
        parameter_bindings: Mapping[str, Any],
        *,
        evidence_id: str = "",
        after_state_digest: str = "",
    ) -> tuple[Any, ...]:
        """Project explicit read-only capability bindings into Taiji affordances.

        The snapshot supplies availability and contract identity; callers must
        supply structured parameters from workspace/IDE evidence.  This method
        never extracts an action from prose and never fabricates parameters for
        a capability whose evidence is absent.
        """

        from taiji import WorldAffordance

        if not isinstance(parameter_bindings, Mapping):
            raise TypeError("Taiji capability bindings must be a mapping")
        if bool(evidence_id) != bool(after_state_digest):
            raise ValueError("Taiji evidence identity requires both event and after-state digest")
        affordances: list[Any] = []
        affordance_ids: set[str] = set()
        for capability_id in sorted(parameter_bindings, key=str):
            raw_binding_group = parameter_bindings[capability_id]
            if isinstance(raw_binding_group, Mapping):
                raw_bindings = (raw_binding_group,)
            elif isinstance(raw_binding_group, Sequence) and not isinstance(
                raw_binding_group, (str, bytes)
            ):
                raw_bindings = tuple(raw_binding_group)
            else:
                raise TypeError(
                    f"Taiji capability binding {capability_id!r} must be a mapping "
                    "or a sequence of mappings"
                )
            descriptor = self.get(str(capability_id))
            if descriptor is None:
                raise ValueError(
                    f"Taiji capability binding is not in the snapshot: {capability_id}"
                )
            if not descriptor.enabled:
                raise ValueError(f"Taiji capability binding is disabled: {capability_id}")
            if descriptor.risk != "read_only":
                raise ValueError(
                    f"Taiji capability projection only admits read-only capabilities: {capability_id}"
                )
            for raw_parameters in raw_bindings:
                if not isinstance(raw_parameters, Mapping):
                    raise TypeError(
                        f"Taiji capability binding {capability_id!r} must contain mappings"
                    )
                parameters = {str(name): value for name, value in raw_parameters.items()}
                unknown = sorted(set(parameters) - descriptor.parameter_names)
                if unknown:
                    raise ValueError(
                        f"Taiji capability binding contains undeclared parameters for "
                        f"{capability_id}: {unknown}"
                    )
                identity = _canonical_digest(
                    {
                        "capability_id": str(capability_id),
                        "parameters": parameters,
                        "evidence_id": str(evidence_id),
                        "after_state_digest": str(after_state_digest),
                    }
                )[:16]
                affordance_id = f"workbench:{capability_id}:{identity}"
                if affordance_id in affordance_ids:
                    raise ValueError(
                        f"Taiji capability bindings contain duplicate affordance content: "
                        f"{capability_id}"
                    )
                affordance_ids.add(affordance_id)
                lineage = [
                    f"workbench-snapshot:{self.snapshot_id}",
                    f"workbench-capability-revision:{self.revision}",
                    f"workbench-capability:{capability_id}",
                ]
                if evidence_id:
                    lineage.extend(
                        (
                            f"workbench-evidence:{evidence_id}",
                            f"workbench-after-state:{after_state_digest}",
                        )
                    )
                affordances.append(
                    WorldAffordance(
                        affordance_id=affordance_id,
                        action_kind=str(capability_id),
                        parameters=parameters,
                        confidence=1.0,
                        feature_provenance="workbench-capability-snapshot",
                        grounding_lineage=tuple(lineage),
                    )
                )
        return tuple(affordances)


@dataclass(frozen=True)
class WorkbenchActionRequest:
    request_id: str
    intent_id: str
    capability_id: str
    parameters: Mapping[str, Any]
    snapshot_id: str
    confidence: float = 0.0
    tick: int = 0
    source: str = "taiji"
    version: int = WORKBENCH_CONTRACT_VERSION
    approval_token: str = ""
    mcp_registry_snapshot_id: str = ""

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
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("workbench action confidence must be in [0, 1]")
        if int(self.tick) < 0:
            raise ValueError("workbench action tick cannot be negative")

    @classmethod
    def from_action_intent(
        cls,
        intent: Any,
        *,
        snapshot_id: str,
        request_id: str | None = None,
        approval_token: str = "",
        mcp_registry_snapshot_id: str = "",
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
            confidence=float(getattr(intent, "confidence", 0.0)),
            tick=int(getattr(intent, "tick", 0)),
            approval_token=str(approval_token or ""),
            mcp_registry_snapshot_id=str(mcp_registry_snapshot_id or ""),
        )

    def binding_payload(self) -> dict[str, str | int]:
        """Return the non-secret identity binding carried across the boundary."""

        return {
            "version": self.version,
            "capability_snapshot_id": self.snapshot_id,
            "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "intent_id": self.intent_id,
            "capability_id": self.capability_id,
            "parameters": dict(self.parameters),
            "snapshot_id": self.snapshot_id,
            "confidence": self.confidence,
            "tick": self.tick,
            "source": self.source,
            "approval_granted": bool(self.approval_token),
            "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
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
class TaijiTaskAdmission:
    """The read-only Gate binding one live Taiji candidate to Workbench."""

    accepted: bool
    candidate_id: str
    snapshot_id: str
    capability_revision: int
    reason_code: str
    reason: str
    candidate: Any | None = None
    request: WorkbenchActionRequest | None = None
    policy: ExecutionPolicyDecision | None = None

    def to_payload(self) -> dict[str, Any]:
        candidate_payload = None
        if self.candidate is not None:
            candidate_payload = self.candidate.to_payload()
        return {
            "accepted": self.accepted,
            "candidate_id": self.candidate_id,
            "snapshot_id": self.snapshot_id,
            "capability_revision": self.capability_revision,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "candidate": candidate_payload,
            "request": None if self.request is None else self.request.to_payload(),
            "policy": None if self.policy is None else self.policy.to_payload(),
        }


@dataclass(frozen=True)
class WorkbenchTransaction:
    operation: str
    path: str
    before_digest: str = ""
    after_digest: str = ""
    undo_token: str = ""
    reversible: bool = True
    version: int = WORKBENCH_CONTRACT_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "operation": self.operation,
            "path": self.path,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "undo_token": self.undo_token,
            "reversible": self.reversible,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WorkbenchTransaction:
        return cls(
            operation=str(payload.get("operation", "")),
            path=str(payload.get("path", "")),
            before_digest=str(payload.get("before_digest", "")),
            after_digest=str(payload.get("after_digest", "")),
            undo_token=str(payload.get("undo_token", "")),
            reversible=bool(payload.get("reversible", True)),
            version=int(payload.get("version", WORKBENCH_CONTRACT_VERSION)),
        )


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
    mcp_registry_snapshot_id: str = ""

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
            "mcp_registry_snapshot_id": self.mcp_registry_snapshot_id,
        }


@dataclass(frozen=True)
class WorkbenchTaijiEvidence:
    """Bounded after-state evidence that can be committed to Taiji's world."""

    request_id: str
    intent_id: str
    call_id: str
    capability_id: str
    snapshot_id: str
    tick: int
    status: str
    success: bool
    parameters: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] = field(default_factory=dict)
    version: int = WORKBENCH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version != WORKBENCH_CONTRACT_VERSION:
            raise ValueError("unsupported workbench Taiji evidence version")
        for value, name in (
            (self.request_id, "request_id"),
            (self.intent_id, "intent_id"),
            (self.call_id, "call_id"),
            (self.capability_id, "capability_id"),
            (self.snapshot_id, "snapshot_id"),
        ):
            if not str(value).strip():
                raise ValueError(f"workbench Taiji evidence {name} cannot be empty")
        if self.status not in {"success", "error"}:
            raise ValueError("unsupported workbench Taiji evidence status")
        if int(self.tick) < 0:
            raise ValueError("workbench Taiji evidence tick cannot be negative")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("workbench Taiji evidence parameters must be a mapping")
        if not isinstance(self.result, Mapping):
            raise TypeError("workbench Taiji evidence result must be a mapping")

    @property
    def after_state_digest(self) -> str:
        """Return the content identity of the observed Workbench result."""

        return _canonical_digest(
            {
                "capability_id": self.capability_id,
                "parameters": dict(self.parameters),
                "result": dict(self.result),
            }
        )

    @property
    def evidence_id(self) -> str:
        """Return a stable event id bound to one tool call and after-state."""

        return (
            "workbench-evidence:"
            + _canonical_digest(
                {
                    "request_id": self.request_id,
                    "intent_id": self.intent_id,
                    "call_id": self.call_id,
                    "capability_id": self.capability_id,
                    "snapshot_id": self.snapshot_id,
                    "tick": int(self.tick),
                    "after_state_digest": self.after_state_digest,
                }
            )[:32]
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "event_id": self.evidence_id,
            "kind": WORKBENCH_TAIJI_EVIDENCE_KIND,
            "request_id": self.request_id,
            "intent_id": self.intent_id,
            "call_id": self.call_id,
            "capability_id": self.capability_id,
            "snapshot_id": self.snapshot_id,
            "tick": int(self.tick),
            "status": self.status,
            "success": self.success,
            "parameters": dict(self.parameters),
            "result": dict(self.result),
            "after_state_digest": self.after_state_digest,
        }

    def to_taiji_event(self) -> Any:
        """Convert the evidence to the Taiji world-event contract lazily."""

        from taiji import WorldEvent

        return WorldEvent(
            event_id=self.evidence_id,
            kind=WORKBENCH_TAIJI_EVIDENCE_KIND,
            tick=int(self.tick),
            subject_id=self.capability_id,
            attributes=(
                ("request_id", self.request_id),
                ("intent_id", self.intent_id),
                ("call_id", self.call_id),
                ("capability_id", self.capability_id),
                ("snapshot_id", self.snapshot_id),
                ("status", self.status),
                ("success", bool(self.success)),
                ("parameters", dict(self.parameters)),
                ("result", dict(self.result)),
                ("after_state_digest", self.after_state_digest),
            ),
            provenance="workbench-observed",
        )

    @classmethod
    def from_taiji_event(cls, event: Any) -> WorkbenchTaijiEvidence:
        """Recover and verify Workbench evidence from a persisted world event."""

        from taiji import WorldEvent

        if not isinstance(event, WorldEvent) or event.kind != WORKBENCH_TAIJI_EVIDENCE_KIND:
            raise ValueError("event is not a Workbench Taiji evidence event")
        attributes = dict(event.attributes)
        required = (
            "request_id",
            "intent_id",
            "call_id",
            "capability_id",
            "snapshot_id",
            "status",
            "success",
            "parameters",
            "result",
            "after_state_digest",
        )
        missing = [name for name in required if name not in attributes]
        if missing:
            raise ValueError(f"WorkBench evidence event is missing attributes: {missing}")
        parameters = attributes["parameters"]
        result = attributes["result"]
        if not isinstance(parameters, Mapping) or not isinstance(result, Mapping):
            raise ValueError("WorkBench evidence event parameters and result must be mappings")
        evidence = cls(
            request_id=str(attributes["request_id"]),
            intent_id=str(attributes["intent_id"]),
            call_id=str(attributes["call_id"]),
            capability_id=str(attributes["capability_id"]),
            snapshot_id=str(attributes["snapshot_id"]),
            tick=int(event.tick),
            status=str(attributes["status"]),
            success=bool(attributes["success"]),
            parameters=dict(parameters),
            result=dict(result),
        )
        if evidence.evidence_id != event.event_id:
            raise ValueError("WorkBench evidence event identity does not match its attributes")
        if evidence.after_state_digest != str(attributes["after_state_digest"]):
            raise ValueError("WorkBench evidence after-state digest does not match its result")
        return evidence

    @property
    def projection_bindings(self) -> Mapping[str, Any]:
        """Derive deterministic next-step bindings from observed workspace state."""

        if not self.success:
            return {}
        bindings: dict[str, list[Mapping[str, Any]]] = {}
        seen: set[tuple[str, str]] = set()

        def add(capability_id: str, parameters: Mapping[str, Any]) -> None:
            identity = (capability_id, _canonical_digest(dict(parameters)))
            if identity in seen:
                return
            seen.add(identity)
            bindings.setdefault(capability_id, []).append(dict(parameters))

        if self.capability_id == "workspace.list":
            entries = self.result.get("entries", ())
            if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    path = str(entry.get("path", "")).strip()
                    if not path:
                        continue
                    add("workspace.stat", {"path": path})
                    if str(entry.get("type", "")) == "file":
                        add("workspace.read", {"path": path})
        elif self.capability_id == "workspace.search":
            results = self.result.get("results", ())
            if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
                for match in results:
                    if isinstance(match, Mapping):
                        path = str(match.get("path", "")).strip()
                        if path:
                            add("workspace.read", {"path": path})
        elif self.capability_id == "workspace.read":
            path = str(self.result.get("path", "")).strip()
            if path:
                add("workspace.stat", {"path": path})
        elif self.capability_id == "workspace.stat":
            if bool(self.result.get("exists")) and self.result.get("type") == "file":
                path = str(self.result.get("path", "")).strip()
                if path:
                    add("workspace.read", {"path": path})
        return {capability_id: tuple(items) for capability_id, items in bindings.items()}

    def to_taiji_affordances(self, snapshot: CapabilitySnapshot) -> tuple[Any, ...]:
        """Re-project this evidence through the current capability snapshot."""

        if not isinstance(snapshot, CapabilitySnapshot):
            raise TypeError("snapshot must be a CapabilitySnapshot")
        if self.snapshot_id != snapshot.snapshot_id:
            raise ValueError("WorkBench evidence capability snapshot is stale")
        if not self.success:
            return ()
        return snapshot.to_taiji_affordances(
            self.projection_bindings,
            evidence_id=self.evidence_id,
            after_state_digest=self.after_state_digest,
        )


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


class WorkbenchConflictError(ValueError):
    """A transaction observed a different state than its precondition."""


class WorkbenchResourceLimitError(ValueError):
    """A connected capability exceeded its declared resource budget."""


class WorkbenchEnvironment:
    """Read-only Seed workbench implementing Taiji's tool-environment protocol."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        snapshot: CapabilitySnapshot | None = None,
        programming_language_registry: ProgrammingLanguageRegistry | None = None,
        mcp_registry: McpToolRegistry | None = None,
    ) -> None:
        self.root = Path(root or default_workspace_root()).resolve()
        self.snapshot = snapshot or CapabilitySnapshot.default()
        self.programming_language_registry = (
            programming_language_registry or ProgrammingLanguageRegistry.default()
        )
        self.mcp_registry = mcp_registry or McpToolRegistry.default()
        self._language_selections: dict[str, ProgrammingLanguageAssessment] = {}
        self._undo_records: dict[str, dict[str, Any]] = {}
        self._approval_records: dict[str, dict[str, Any]] = {}
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
            "programming_languages": self.programming_language_registry.public_descriptors(),
            "programming_language_registry_revision": self.programming_language_registry.revision,
            "language_selections": [
                item.to_payload() for item in self._language_selections.values()
            ],
            "undoable_transactions": len(self._undo_records),
            "pending_approvals": len(self._approval_records),
            "mcp_registry": self.mcp_registry.to_payload(),
        }

    def preflight_loop(
        self,
        requests: Sequence[WorkbenchActionRequest],
        *,
        loop_id: str,
        max_steps: int = WORKBENCH_MAX_LOOP_STEPS,
        max_budget_units: float = WORKBENCH_MAX_LOOP_BUDGET_UNITS,
        on_failure: str = "stop",
        checkpoint_boundary: str = "after_each_step",
    ) -> dict[str, Any]:
        """Admit a bounded sequence without executing or mutating any step."""

        base = {
            "format": WORKBENCH_LOOP_CONTRACT_FORMAT,
            "version": WORKBENCH_LOOP_CONTRACT_VERSION,
            "loop_id": str(loop_id),
            "snapshot_id": self.snapshot.snapshot_id,
            "mcp_registry_snapshot_id": self.mcp_registry.snapshot_id,
        }

        def rejected(code: str, message: str, *, index: int | None = None) -> dict[str, Any]:
            payload = {
                **base,
                "accepted": False,
                "error_code": code,
                "error": message,
            }
            if index is not None:
                payload["step_index"] = index
            return payload

        if not str(loop_id).strip():
            return rejected("invalid_loop_id", "loop_id cannot be empty")
        try:
            step_limit = int(max_steps)
            budget_limit = float(max_budget_units)
        except (TypeError, ValueError):
            return rejected("invalid_loop_budget", "loop limits must be numeric")
        if not 1 <= step_limit <= WORKBENCH_MAX_LOOP_STEPS:
            return rejected("invalid_loop_steps", "max_steps must be between 1 and 8")
        if (
            not math.isfinite(budget_limit)
            or not 0.0 < budget_limit <= WORKBENCH_MAX_LOOP_BUDGET_UNITS
        ):
            return rejected("invalid_loop_budget", "max_budget_units must be in (0, 32]")
        if on_failure != "stop":
            return rejected("unsupported_failure_mode", "loop failure mode must be stop")
        if checkpoint_boundary != "after_each_step":
            return rejected(
                "unsupported_checkpoint_boundary",
                "loop checkpoint boundary must be after_each_step",
            )
        if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
            return rejected("invalid_loop_requests", "loop requests must be a sequence")
        if not requests:
            return rejected("empty_loop", "loop must contain at least one request")
        if len(requests) > step_limit:
            return rejected("loop_step_limit", "loop contains more than max_steps")

        steps: list[dict[str, Any]] = []
        seen: set[str] = set()
        seen_request_ids: set[str] = set()
        budget_units = 0.0
        for index, request in enumerate(requests):
            if not isinstance(request, WorkbenchActionRequest):
                return rejected(
                    "invalid_loop_request",
                    "loop item is not a workbench request",
                    index=index,
                )
            if request.snapshot_id != self.snapshot.snapshot_id:
                return rejected(
                    "stale_capability_snapshot",
                    "loop capability snapshot drifted",
                    index=index,
                )
            if request.request_id in seen_request_ids:
                return rejected(
                    "duplicate_request_id",
                    "loop contains a duplicate request_id",
                    index=index,
                )
            seen_request_ids.add(request.request_id)
            if request.capability_id in {"mcp.list", "mcp.invoke"} and (
                request.mcp_registry_snapshot_id != self.mcp_registry.snapshot_id
            ):
                return rejected(
                    "stale_mcp_registry",
                    "loop MCP registry snapshot drifted",
                    index=index,
                )
            request_key = _canonical_digest(
                {
                    "capability_id": request.capability_id,
                    "parameters": dict(request.parameters),
                }
            )
            if request_key in seen:
                return rejected(
                    "repeated_call",
                    "loop contains a repeated capability call",
                    index=index,
                )
            seen.add(request_key)
            policy = self.policy_for(request)
            if policy.decision != "allow":
                return rejected(
                    policy.reason_code,
                    "loop contains a request that is not admitted",
                    index=index,
                )
            step_budget = 1.0
            if request.capability_id == "mcp.invoke":
                try:
                    descriptor = self.mcp_registry.validate_call(
                        request.parameters.get("tool_id", ""),
                        request.parameters.get("arguments", {}),
                    )
                except (KeyError, PermissionError, TypeError, ValueError) as exc:
                    return rejected("invalid_parameters", str(exc), index=index)
                step_budget = max(
                    1.0,
                    float(descriptor.timeout_seconds) / 5.0
                    + float(descriptor.output_limit) / 65_536.0,
                )
            budget_units += step_budget
            if budget_units > budget_limit:
                return rejected("loop_budget_limit", "loop exceeds max_budget_units", index=index)
            steps.append(
                {
                    "index": index,
                    "request_id": request.request_id,
                    "intent_id": request.intent_id,
                    "capability_id": request.capability_id,
                    "policy": policy.to_payload(),
                    "budget_units": step_budget,
                    "mcp_registry_snapshot_id": request.mcp_registry_snapshot_id,
                }
            )

        payload = {
            **base,
            "accepted": True,
            "step_count": len(steps),
            "max_steps": step_limit,
            "budget_units": budget_units,
            "max_budget_units": budget_limit,
            "on_failure": "stop",
            "checkpoint": {
                "boundary": "after_each_step",
                "required": True,
                "resume_key": "request_id",
            },
            "steps": steps,
        }
        payload["preflight_id"] = _canonical_digest(payload)
        return payload

    def language_state_checkpoint(self) -> dict[str, Any]:
        """Persist language choices without persisting an executable action."""

        with self._lock:
            return {
                "format": "seed-workbench-language-state-v1",
                "version": 1,
                "registry_revision": self.programming_language_registry.revision,
                "selections": [item.to_payload() for item in self._language_selections.values()],
            }

    def restore_language_state(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        if payload.get("format") != "seed-workbench-language-state-v1":
            raise ValueError("unsupported workbench language state format")
        if str(payload.get("registry_revision", "")) != self.programming_language_registry.revision:
            raise ValueError("programming language registry drifted during restore")
        restored: dict[str, ProgrammingLanguageAssessment] = {}
        for raw in payload.get("selections", []):
            if not isinstance(raw, Mapping):
                continue
            assessment = ProgrammingLanguageAssessment.from_payload(raw)
            if assessment.registry_revision != self.programming_language_registry.revision:
                raise ValueError("programming language assessment registry drifted")
            restored[assessment.path] = assessment
        with self._lock:
            self._language_selections = restored

    def transaction_state_checkpoint(self) -> dict[str, Any]:
        """Persist undo data, but deliberately discard approval grants on restart."""

        with self._lock:
            records = [
                {"token": token, "record": dict(record)}
                for token, record in self._undo_records.items()
            ]
        return {
            "format": "seed-workbench-transaction-state-v1",
            "version": 1,
            "max_records": WORKBENCH_MAX_UNDO_RECORDS,
            "records": records[-WORKBENCH_MAX_UNDO_RECORDS:],
            "approvals_restored": False,
        }

    def restore_transaction_state(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        if payload.get("format") != "seed-workbench-transaction-state-v1":
            raise ValueError("unsupported workbench transaction state format")
        restored: dict[str, dict[str, Any]] = {}
        raw_records = payload.get("records", [])
        if not isinstance(raw_records, (list, tuple)):
            raise ValueError("workbench transaction records must be a list")
        allowed_operations = {
            "workspace.apply_patch",
            "workspace.create",
            "workspace.rename",
            "workspace.delete",
        }
        for raw_item in raw_records[-WORKBENCH_MAX_UNDO_RECORDS:]:
            if not isinstance(raw_item, Mapping):
                continue
            token = str(raw_item.get("token", "")).strip()
            record = raw_item.get("record")
            if not token or not isinstance(record, Mapping):
                continue
            normalized = dict(record)
            if normalized.get("operation") not in allowed_operations:
                continue
            if not isinstance(normalized.get("path"), str):
                continue
            before_bytes = normalized.get("before_bytes")
            if before_bytes is not None and not isinstance(before_bytes, bytes):
                raise ValueError("workbench undo bytes are invalid")
            restored[token] = normalized
        with self._lock:
            self._undo_records = restored
            # Approval is intentionally session-scoped and never resurrected.
            self._approval_records = {}

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
        if request.capability_id in {"mcp.list", "mcp.invoke"}:
            return self._mcp_policy_for(request)
        if descriptor.risk not in {"read_only", "reversible_ui"}:
            if request.approval_token:
                if self._approval_is_valid(request):
                    return ExecutionPolicyDecision(
                        request.request_id,
                        request.capability_id,
                        "allow",
                        "explicit_approval",
                        self.snapshot.snapshot_id,
                    )
                return ExecutionPolicyDecision(
                    request.request_id,
                    request.capability_id,
                    "ask_user",
                    "approval_invalid",
                    self.snapshot.snapshot_id,
                )
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "ask_user",
                "capability_requires_approval",
                self.snapshot.snapshot_id,
            )
        if request.capability_id == "editor.set_language":
            language_policy = self._language_policy_for(request)
            if language_policy is not None:
                return language_policy
        return ExecutionPolicyDecision(
            request.request_id,
            request.capability_id,
            "allow",
            "read_only_or_reversible",
            self.snapshot.snapshot_id,
        )

    def admit_taiji_candidate(
        self,
        candidate: Any,
        *,
        snapshot_id: str,
        current_tick: int | None = None,
        current_affordance_ids: Sequence[str] | None = None,
        current_affordances: Sequence[Any] | None = None,
    ) -> TaijiTaskAdmission:
        """Bind a live Taiji executive candidate to a read-only capability.

        This is deliberately narrower than :meth:`policy_for`: an ordinary
        Workbench request may be user-authored and may enter the approval flow,
        while this Gate only admits candidates that Taiji's executive already
        selected and that resolve to a ``read_only`` capability.  The language
        provider, frontend, and request prose never participate in selection.
        """

        from taiji import ExecutiveCandidate

        candidate_id = str(getattr(candidate, "candidate_id", ""))

        def reject(code: str, reason: str) -> TaijiTaskAdmission:
            return TaijiTaskAdmission(
                accepted=False,
                candidate_id=candidate_id,
                snapshot_id=self.snapshot.snapshot_id,
                capability_revision=self.snapshot.revision,
                reason_code=code,
                reason=reason,
                candidate=candidate if isinstance(candidate, ExecutiveCandidate) else None,
            )

        if str(snapshot_id) != self.snapshot.snapshot_id:
            return reject("stale_capability_snapshot", "Taiji task capability snapshot drifted")
        if not isinstance(candidate, ExecutiveCandidate):
            return reject("invalid_taiji_candidate", "Taiji task requires an ExecutiveCandidate")
        if not candidate.source_affordance_id:
            return reject(
                "taiji_candidate_not_grounded",
                "Taiji task candidate must retain its world affordance lineage",
            )
        if not candidate.provenance.startswith("affordance-derived/"):
            return reject(
                "taiji_candidate_not_grounded",
                "Taiji task candidate provenance is not Taiji affordance-derived",
            )
        if current_tick is not None and candidate.action_intent.tick != int(current_tick):
            return reject(
                "stale_taiji_candidate", "Taiji task candidate is not at the current tick"
            )
        if current_affordance_ids is not None and candidate.source_affordance_id not in {
            str(item) for item in current_affordance_ids
        }:
            return reject(
                "stale_taiji_affordance",
                "Taiji task candidate no longer refers to a current world affordance",
            )
        if current_affordances is not None:
            source_affordance = next(
                (
                    item
                    for item in current_affordances
                    if getattr(item, "affordance_id", "") == candidate.source_affordance_id
                ),
                None,
            )
            if source_affordance is None:
                return reject(
                    "stale_taiji_affordance",
                    "Taiji task candidate no longer refers to a live world affordance",
                )
            expected_lineage = f"workbench-snapshot:{self.snapshot.snapshot_id}"
            if expected_lineage not in tuple(
                str(item) for item in getattr(source_affordance, "grounding_lineage", ())
            ):
                return reject(
                    "stale_capability_snapshot",
                    "Taiji task candidate was projected from a different capability snapshot",
                )

        descriptor = self.snapshot.get(candidate.action_intent.kind)
        if descriptor is None:
            return reject("unknown_capability", "Taiji candidate capability is not in the snapshot")
        if not descriptor.enabled:
            return reject("capability_not_connected", "Taiji candidate capability is disabled")
        if descriptor.risk != "read_only":
            return reject(
                "taiji_read_only_gate_rejects_risk",
                "the first Taiji task Gate only admits read-only capabilities",
            )

        parameters = dict(candidate.action_intent.parameters)
        unknown = sorted(set(parameters) - descriptor.parameter_names)
        if unknown:
            return reject(
                "capability_parameter_drift",
                f"Taiji candidate contains undeclared capability parameters: {unknown}",
            )

        request = WorkbenchActionRequest.from_action_intent(
            candidate.action_intent,
            snapshot_id=self.snapshot.snapshot_id,
        )
        policy = self.policy_for(request)
        if policy.decision != "allow":
            return TaijiTaskAdmission(
                accepted=False,
                candidate_id=candidate.candidate_id,
                snapshot_id=self.snapshot.snapshot_id,
                capability_revision=self.snapshot.revision,
                reason_code=policy.reason_code,
                reason="Taiji candidate policy was not admitted",
                candidate=candidate,
                request=request,
                policy=policy,
            )
        return TaijiTaskAdmission(
            accepted=True,
            candidate_id=candidate.candidate_id,
            snapshot_id=self.snapshot.snapshot_id,
            capability_revision=self.snapshot.revision,
            reason_code="taiji_read_only_admitted",
            reason="Taiji candidate is bound to the current read-only capability snapshot",
            candidate=candidate,
            request=request,
            policy=policy,
        )

    def _mcp_policy_for(self, request: WorkbenchActionRequest) -> ExecutionPolicyDecision:
        registry_snapshot_id = str(request.mcp_registry_snapshot_id or "")
        if registry_snapshot_id and registry_snapshot_id != self.mcp_registry.snapshot_id:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "deny",
                "stale_mcp_registry",
                self.snapshot.snapshot_id,
            )
        if request.capability_id == "mcp.list":
            if request.parameters:
                return ExecutionPolicyDecision(
                    request.request_id,
                    request.capability_id,
                    "deny",
                    "invalid_parameters",
                    self.snapshot.snapshot_id,
                )
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "allow",
                "mcp_registry_read_only",
                self.snapshot.snapshot_id,
            )
        tool_id = str(request.parameters.get("tool_id", "")).strip()
        descriptor = self.mcp_registry.get(tool_id)
        if descriptor is None:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "deny",
                "unknown_mcp_tool",
                self.snapshot.snapshot_id,
            )
        if not descriptor.enabled:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "deny",
                "mcp_tool_not_connected",
                self.snapshot.snapshot_id,
            )
        if descriptor.risk not in {"read_only", "reversible_ui"}:
            if request.approval_token:
                if self._approval_is_valid(request):
                    return ExecutionPolicyDecision(
                        request.request_id,
                        request.capability_id,
                        "allow",
                        "explicit_approval",
                        self.snapshot.snapshot_id,
                    )
                return ExecutionPolicyDecision(
                    request.request_id,
                    request.capability_id,
                    "ask_user",
                    "approval_invalid",
                    self.snapshot.snapshot_id,
                )
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
            "mcp_tool_read_only",
            self.snapshot.snapshot_id,
        )

    @staticmethod
    def _approval_request_digest(request: WorkbenchActionRequest) -> str:
        return _canonical_digest(
            {
                "request_id": request.request_id,
                "intent_id": request.intent_id,
                "capability_id": request.capability_id,
                "parameters": dict(request.parameters),
                "snapshot_id": request.snapshot_id,
                "confidence": request.confidence,
                "tick": request.tick,
                "source": request.source,
                "mcp_registry_snapshot_id": request.mcp_registry_snapshot_id,
            }
        )

    def _approval_is_valid(self, request: WorkbenchActionRequest) -> bool:
        with self._lock:
            record = self._approval_records.get(request.approval_token)
        if record is None:
            return False
        if float(record.get("expires_at", 0.0)) < time.time():
            with self._lock:
                self._approval_records.pop(request.approval_token, None)
            return False
        return (
            record.get("request_digest") == self._approval_request_digest(request)
            and record.get("snapshot_id") == self.snapshot.snapshot_id
        )

    def issue_approval(self, request: WorkbenchActionRequest) -> dict[str, Any]:
        """Create a short-lived, single-use approval for one exact action."""

        policy = self.policy_for(request)
        if policy.reason_code != "capability_requires_approval":
            raise ValueError("only approval-required capabilities can be previewed")
        preview = self.preview_tool(request.capability_id, request.parameters)
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._approval_records[token] = {
                "request_digest": self._approval_request_digest(request),
                "snapshot_id": self.snapshot.snapshot_id,
                "expires_at": now + WORKBENCH_APPROVAL_TTL_SECONDS,
            }
        return {
            "format": "seed-workbench-approval-v1",
            "version": 1,
            "approval_token": token,
            "expires_in_seconds": WORKBENCH_APPROVAL_TTL_SECONDS,
            "preview": preview,
        }

    def consume_approval(self, request: WorkbenchActionRequest) -> None:
        """Consume an approval immediately before the side effect starts."""

        descriptor = self.snapshot.get(request.capability_id)
        if request.capability_id == "mcp.invoke":
            mcp_descriptor = self.mcp_registry.get(str(request.parameters.get("tool_id", "")))
            if mcp_descriptor is None or mcp_descriptor.risk in {
                "read_only",
                "reversible_ui",
            }:
                return
        if descriptor is None or descriptor.risk in {"read_only", "reversible_ui"}:
            return
        if not self._approval_is_valid(request):
            raise WorkbenchConflictError("approval expired, changed, or was already consumed")
        with self._lock:
            self._approval_records.pop(request.approval_token, None)

    def preview_tool(self, tool_name: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a side effect without mutating the workspace or running a process."""

        descriptor = self.snapshot.get(tool_name)
        if descriptor is None or not descriptor.enabled:
            raise ValueError("capability is not available for preview")
        if tool_name in {
            "workspace.apply_patch",
            "workspace.create",
            "workspace.rename",
            "workspace.delete",
            "workspace.undo",
        }:
            preview = self._preview_file_transaction(tool_name, parameters)
        elif tool_name == "terminal.run":
            preview = self._preview_terminal(parameters)
        elif tool_name == "mcp.invoke":
            preview = self._preview_mcp_invoke(parameters)
        else:
            raise ValueError("capability does not have a side-effect preview")
        return {
            "capability_id": tool_name,
            "risk": descriptor.risk,
            "reversible": descriptor.reversible,
            "parameters": self._preview_parameters(parameters),
            "validated": True,
            "mutation": preview,
        }

    def _language_policy_for(
        self, request: WorkbenchActionRequest
    ) -> ExecutionPolicyDecision | None:
        parameters = request.parameters
        if bool(parameters.get("user_override", False)):
            return None
        if bool(parameters.get("clear_override", False)):
            if request.confidence < LANGUAGE_CONFIDENCE_THRESHOLD:
                return ExecutionPolicyDecision(
                    request.request_id,
                    request.capability_id,
                    "ask_user",
                    "language_action_confidence_low",
                    self.snapshot.snapshot_id,
                )
            return None
        language_id = str(parameters.get("programming_language_id", "")).strip()
        if not language_id:
            editor_language_id = str(parameters.get("editor_language_id", "")).strip()
            definition = self.programming_language_registry.get_by_editor_language(
                editor_language_id
            )
            language_id = "" if definition is None else definition.language_id
        if self.programming_language_registry.get(language_id) is None:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "deny",
                "unknown_programming_language",
                self.snapshot.snapshot_id,
            )
        if request.confidence < LANGUAGE_CONFIDENCE_THRESHOLD:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "ask_user",
                "language_action_confidence_low",
                self.snapshot.snapshot_id,
            )
        try:
            assessment = self._resolve_programming_language(
                {"path": parameters.get("path")},
                remember=False,
                apply_selection=False,
            )
        except (
            WorkbenchPathError,
            FileNotFoundError,
            IsADirectoryError,
            OSError,
            ValueError,
        ):
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "deny",
                "language_assessment_unavailable",
                self.snapshot.snapshot_id,
            )
        if assessment["selection_state"] != "resolved":
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "ask_user",
                "language_evidence_ambiguous",
                self.snapshot.snapshot_id,
            )
        if assessment["programming_language_id"] != language_id:
            return ExecutionPolicyDecision(
                request.request_id,
                request.capability_id,
                "ask_user",
                "language_evidence_conflict",
                self.snapshot.snapshot_id,
            )
        return None

    def execute_tool(
        self,
        tool_name: str,
        parameters: Mapping[str, Any],
    ) -> EnvironmentOutcome:
        """Execute one registered capability after the policy boundary admits it."""

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
            elif tool_name == "workspace.programming_language.resolve":
                result = self._resolve_programming_language(parameters)
            elif tool_name == "editor.set_language":
                result = self._set_editor_language(parameters)
            elif tool_name == "workspace.apply_patch":
                result = self._apply_patch(parameters)
            elif tool_name == "workspace.create":
                result = self._create_file(parameters)
            elif tool_name == "workspace.rename":
                result = self._rename_file(parameters)
            elif tool_name == "workspace.delete":
                result = self._delete_file(parameters)
            elif tool_name == "workspace.undo":
                result = self._undo_transaction(parameters)
            elif tool_name == "terminal.run":
                result = self._run_terminal(parameters)
            elif tool_name == "mcp.list":
                result = self._mcp_list(parameters)
            elif tool_name == "mcp.invoke":
                result = self._mcp_invoke(parameters)
            else:
                return self._failure("unknown_capability", "capability is not registered")
        except WorkbenchPathError as exc:
            return self._failure("unsafe_path", str(exc))
        except WorkbenchConflictError as exc:
            return self._failure("transaction_conflict", str(exc))
        except KeyError as exc:
            return self._failure("unknown_mcp_tool", str(exc))
        except PermissionError as exc:
            return self._failure("mcp_tool_not_connected", str(exc))
        except WorkbenchResourceLimitError as exc:
            return self._failure("mcp_output_limit", str(exc))
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
        success = bool(result.get("success", True))
        return EnvironmentOutcome(
            sensation=self._sensation(tool_name, result),
            reward=1.0 if success else -1.0,
            terminal=True,
            success=success,
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
                            return {
                                "query": query,
                                "results": results,
                                "truncated": True,
                            }
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

    def _workspace_context_names(self, path: Path) -> tuple[set[str], set[str]]:
        """Collect names from the file's directory and workspace ancestors."""

        manifest_names: set[str] = set()
        neighbor_names: set[str] = set()
        current = path.parent
        while True:
            try:
                names = {item.name for item in current.iterdir()}
            except OSError:
                names = set()
            manifest_names.update(name.lower() for name in names)
            if current == path.parent:
                neighbor_names.update(name for name in names if name != path.name)
            if current == self.root or current.parent == current:
                break
            try:
                current = current.parent
                current.relative_to(self.root)
            except ValueError:
                break
        return manifest_names, neighbor_names

    def _resolve_programming_language(
        self,
        parameters: Mapping[str, Any],
        *,
        remember: bool = True,
        apply_selection: bool = True,
    ) -> dict[str, Any]:
        path, relative_name = self._resolve_path(parameters.get("path"), allow_root=False)
        raw = self._read_workspace({"path": relative_name})
        manifest_names, neighbor_names = self._workspace_context_names(path)
        assessment = self.programming_language_registry.resolve(
            path=relative_name,
            content=str(raw["content"]),
            file_digest=str(raw["digest"]),
            manifest_names=manifest_names,
            neighbor_names=neighbor_names,
            available_toolchains=self.programming_language_registry.available_toolchains(),
            lsp_language_id=(
                None
                if parameters.get("lsp_language_id") in (None, "")
                else str(parameters["lsp_language_id"]).strip()
            ),
            capability_revision=self.snapshot.revision,
        )
        if apply_selection:
            with self._lock:
                previous = self._language_selections.get(relative_name)
            if (
                previous is not None
                and previous.user_override
                and previous.file_digest == assessment.file_digest
            ):
                assessment = self.programming_language_registry.select(
                    assessment,
                    previous.user_override,
                    source="user_override",
                )
        if remember:
            with self._lock:
                self._language_selections[relative_name] = assessment
        return assessment.to_payload()

    def _set_editor_language(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        path, relative_name = self._resolve_path(parameters.get("path"), allow_root=False)
        if not path.is_file():
            raise FileNotFoundError(path)
        clear_override = bool(parameters.get("clear_override", False))
        if clear_override:
            with self._lock:
                self._language_selections.pop(relative_name, None)
        baseline = self._resolve_programming_language({"path": relative_name})
        assessment = ProgrammingLanguageAssessment.from_payload(baseline)
        if clear_override:
            with self._lock:
                self._language_selections[relative_name] = assessment
            return assessment.to_payload()
        language_id = str(parameters.get("programming_language_id", "")).strip()
        if not language_id:
            language_id = str(parameters.get("editor_language_id", "")).strip()
            definition = self.programming_language_registry.get_by_editor_language(language_id)
            language_id = definition.language_id if definition is not None else ""
        source = (
            "user_override" if bool(parameters.get("user_override", False)) else "taiji_selection"
        )
        selected = self.programming_language_registry.select(
            assessment,
            language_id,
            source=source,
        )
        with self._lock:
            self._language_selections[relative_name] = selected
        return selected.to_payload()

    @staticmethod
    def _preview_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
        preview = dict(parameters)
        raw_env = preview.get("env")
        if isinstance(raw_env, Mapping):
            preview["env"] = {str(key): "<redacted>" for key in raw_env}
        if isinstance(preview.get("content"), str):
            content = str(preview["content"]).encode("utf-8")
            preview["content"] = {
                "byte_length": len(content),
                "digest": hashlib.sha256(content).hexdigest(),
            }
        if "undo_token" in preview:
            preview["undo_token"] = "<redacted>"
        return preview

    def _preview_file_transaction(
        self, tool_name: str, parameters: Mapping[str, Any]
    ) -> dict[str, Any]:
        if tool_name == "workspace.create":
            path, relative_name = self._resolve_path(parameters.get("path"), allow_root=False)
            if path.exists() or path.is_symlink():
                raise WorkbenchConflictError("create target already exists")
            if not path.parent.is_dir() or path.parent.is_symlink():
                raise WorkbenchPathError("create parent directory is unavailable")
            content = parameters.get("content", "")
            if not isinstance(content, str):
                raise ValueError("create content must be text")
            raw = content.encode("utf-8")
            if len(raw) > WORKBENCH_MAX_WRITE_BYTES:
                raise ValueError("created file exceeds the writable size limit")
            return {
                "operation": tool_name,
                "path": relative_name,
                "before_digest": "",
                "after_digest": self._bytes_digest(raw),
                "byte_length": len(raw),
            }
        if tool_name == "workspace.undo":
            token = str(parameters.get("undo_token", "")).strip()
            if not token:
                raise ValueError("undo_token is required")
            with self._lock:
                record = self._undo_records.get(token)
            if record is None:
                raise ValueError("undo token is unknown or already consumed")
            path, relative_name = self._resolve_path(record["path"], allow_root=False)
            operation = str(record["operation"])
            if operation in {"workspace.apply_patch", "workspace.create"}:
                current = path.read_bytes() if path.is_file() else b""
                if self._bytes_digest(current) != record["after_digest"]:
                    raise WorkbenchConflictError("file changed before undo preview")
            elif operation == "workspace.delete":
                if path.exists() or path.is_symlink():
                    raise WorkbenchConflictError("delete target reappeared before undo preview")
            elif operation == "workspace.rename":
                new_path, _ = self._resolve_path(record["new_path"], allow_root=False)
                if (
                    path.exists()
                    or not new_path.is_file()
                    or self._bytes_digest(new_path.read_bytes()) != record["after_digest"]
                ):
                    raise WorkbenchConflictError("rename state changed before undo preview")
            else:
                raise ValueError("unsupported undo operation")
            return {
                "operation": "workspace.undo",
                "undo_of": operation,
                "path": relative_name,
                "before_digest": str(record.get("after_digest", "")),
                "after_digest": str(record.get("before_digest", "")),
            }
        path, relative_name = self._regular_file(parameters.get("path"))
        raw, before_digest = self._checked_file_bytes(path, parameters.get("before_digest"))
        if tool_name == "workspace.delete":
            return {
                "operation": tool_name,
                "path": relative_name,
                "before_digest": before_digest,
                "after_digest": "",
                "byte_length": len(raw),
            }
        if tool_name == "workspace.rename":
            new_path, new_relative_name = self._resolve_path(
                parameters.get("new_path"), allow_root=False
            )
            if new_path.exists() or new_path.is_symlink():
                raise WorkbenchConflictError("rename target already exists")
            if not new_path.parent.is_dir() or new_path.parent.is_symlink():
                raise WorkbenchPathError("rename parent directory is unavailable")
            return {
                "operation": tool_name,
                "path": relative_name,
                "new_path": new_relative_name,
                "before_digest": before_digest,
                "after_digest": before_digest,
                "byte_length": len(raw),
            }
        if tool_name != "workspace.apply_patch":
            raise ValueError("unsupported file transaction preview")
        patch = parameters.get("patch")
        if not isinstance(patch, Mapping) or patch.get("kind") != "text_replace":
            raise ValueError("patch.kind must be text_replace")
        raw_operations = patch.get("operations")
        if not isinstance(raw_operations, (list, tuple)) or not raw_operations:
            raise ValueError("patch.operations must be a non-empty list")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("workspace.apply_patch only accepts UTF-8 text") from exc
        operations: list[tuple[int, int, str]] = []
        for raw_operation in raw_operations:
            if not isinstance(raw_operation, Mapping):
                raise ValueError("patch operation must be an object")
            start = raw_operation.get("start")
            end = raw_operation.get("end")
            replacement = raw_operation.get("text", "")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not isinstance(replacement, str)
            ):
                raise ValueError("patch operation requires integer range and text")
            operations.append((start, end, replacement))
        operations.sort(key=lambda item: (item[0], item[1]))
        previous_end = 0
        for start, end, _replacement in operations:
            if start < 0 or end < start or end > len(content) or start < previous_end:
                raise ValueError("patch operations overlap or exceed the file")
            previous_end = end
        updated = content
        for start, end, replacement in reversed(operations):
            updated = updated[:start] + replacement + updated[end:]
        updated_raw = updated.encode("utf-8")
        if len(updated_raw) > WORKBENCH_MAX_WRITE_BYTES:
            raise ValueError("patched file exceeds the writable size limit")
        expected_after = (
            str(parameters.get("expected_after_digest", parameters.get("after_digest", "")))
            .strip()
            .lower()
        )
        if len(expected_after) != 64:
            raise ValueError("expected_after_digest must be a SHA-256 digest")
        after_digest = self._bytes_digest(updated_raw)
        if after_digest != expected_after:
            raise WorkbenchConflictError(
                f"patch result digest mismatch: expected {expected_after}, got {after_digest}"
            )
        return {
            "operation": tool_name,
            "path": relative_name,
            "before_digest": before_digest,
            "after_digest": after_digest,
            "before_byte_length": len(raw),
            "after_byte_length": len(updated_raw),
            "operations": len(operations),
        }

    def _preview_terminal(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        argv, cwd, relative_cwd, timeout_seconds, output_limit, allowlist, raw_env = (
            self._normalize_terminal_parameters(parameters)
        )
        return {
            "operation": "terminal.run",
            "argv": argv,
            "cwd": relative_cwd,
            "timeout_seconds": timeout_seconds,
            "output_limit": output_limit,
            "env_allowlist": sorted(allowlist),
            "env_keys": sorted(str(key) for key in raw_env),
            "expected_artifacts_before": self._artifact_state(
                parameters.get("expected_artifacts", [])
            ),
            "will_execute": True,
        }

    def _preview_mcp_invoke(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        tool_id = str(parameters.get("tool_id", "")).strip()
        arguments = parameters.get("arguments", {})
        descriptor = self.mcp_registry.validate_call(
            tool_id,
            arguments,
            registry_revision=parameters.get("registry_revision"),
        )
        return {
            "operation": "mcp.invoke",
            "tool_id": descriptor.tool_id,
            "source": descriptor.source,
            "executor_id": descriptor.executor_id,
            "risk": descriptor.risk,
            "registry_revision": self.mcp_registry.revision,
            "timeout_seconds": descriptor.timeout_seconds,
            "output_limit": descriptor.output_limit,
            "arguments": dict(arguments),
            "will_execute": True,
        }

    def _regular_file(self, raw_path: Any) -> tuple[Path, str]:
        path, relative_name = self._resolve_path(raw_path, allow_root=False)
        if path.is_symlink():
            raise WorkbenchPathError("symbolic links are not writable")
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise IsADirectoryError(path)
        return path, relative_name

    @staticmethod
    def _bytes_digest(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def _checked_file_bytes(self, path: Path, expected_digest: Any) -> tuple[bytes, str]:
        expected = str(expected_digest or "").strip().lower()
        if len(expected) != 64:
            raise ValueError("before_digest must be a SHA-256 digest")
        raw = path.read_bytes()
        actual = self._bytes_digest(raw)
        if actual != expected:
            raise WorkbenchConflictError(
                f"file digest changed before transaction: expected {expected}, got {actual}"
            )
        return raw, actual

    @staticmethod
    def _atomic_write(path: Path, raw: bytes) -> None:
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".seed-transaction-",
                suffix=".tmp",
                dir=str(path.parent),
                delete=False,
            ) as handle:
                temporary_path = handle.name
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    Path(temporary_path).unlink()

    def _new_undo_token(self, record: dict[str, Any]) -> str:
        token = secrets.token_urlsafe(24)
        with self._lock:
            while len(self._undo_records) >= WORKBENCH_MAX_UNDO_RECORDS:
                self._undo_records.pop(next(iter(self._undo_records)))
            self._undo_records[token] = record
        return token

    @staticmethod
    def _transaction_result(
        transaction: WorkbenchTransaction,
        *,
        digest: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": transaction.path,
            "digest": digest,
            "before_digest": transaction.before_digest,
            "after_digest": transaction.after_digest,
            "transaction": transaction.to_payload(),
        }
        result.update(dict(extra or {}))
        return result

    def _apply_patch(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        path, relative_name = self._regular_file(parameters.get("path"))
        raw, before_digest = self._checked_file_bytes(path, parameters.get("before_digest"))
        if len(raw) > WORKBENCH_MAX_WRITE_BYTES:
            raise ValueError("file exceeds the writable size limit")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("workspace.apply_patch only accepts UTF-8 text") from exc
        patch = parameters.get("patch")
        if not isinstance(patch, Mapping) or patch.get("kind") != "text_replace":
            raise ValueError("patch.kind must be text_replace")
        raw_operations = patch.get("operations")
        if not isinstance(raw_operations, (list, tuple)) or not raw_operations:
            raise ValueError("patch.operations must be a non-empty list")
        operations: list[tuple[int, int, str]] = []
        for raw_operation in raw_operations:
            if not isinstance(raw_operation, Mapping):
                raise ValueError("patch operation must be an object")
            start = raw_operation.get("start")
            end = raw_operation.get("end")
            replacement = raw_operation.get("text", "")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not isinstance(replacement, str)
            ):
                raise ValueError("patch operation requires integer range and text")
            operations.append((start, end, replacement))
        operations.sort(key=lambda item: (item[0], item[1]))
        previous_end = 0
        for start, end, _replacement in operations:
            if start < 0 or end < start or end > len(content) or start < previous_end:
                raise ValueError("patch operations overlap or exceed the file")
            previous_end = end
        updated = content
        for start, end, replacement in reversed(operations):
            updated = updated[:start] + replacement + updated[end:]
        updated_raw = updated.encode("utf-8")
        if len(updated_raw) > WORKBENCH_MAX_WRITE_BYTES:
            raise ValueError("patched file exceeds the writable size limit")
        expected_after = (
            str(parameters.get("expected_after_digest", parameters.get("after_digest", "")))
            .strip()
            .lower()
        )
        if len(expected_after) != 64:
            raise ValueError("expected_after_digest must be a SHA-256 digest")
        after_digest = self._bytes_digest(updated_raw)
        if after_digest != expected_after:
            raise WorkbenchConflictError(
                f"patch result digest mismatch: expected {expected_after}, got {after_digest}"
            )
        self._atomic_write(path, updated_raw)
        token = self._new_undo_token(
            {
                "operation": "workspace.apply_patch",
                "path": relative_name,
                "before_digest": before_digest,
                "after_digest": after_digest,
                "before_bytes": raw,
            }
        )
        transaction = WorkbenchTransaction(
            operation="workspace.apply_patch",
            path=relative_name,
            before_digest=before_digest,
            after_digest=after_digest,
            undo_token=token,
        )
        return self._transaction_result(transaction, digest=after_digest)

    def _create_file(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        path, relative_name = self._resolve_path(parameters.get("path"), allow_root=False)
        if path.exists() or path.is_symlink():
            raise WorkbenchConflictError("create target already exists")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise WorkbenchPathError("create parent directory is unavailable")
        content = parameters.get("content", "")
        if not isinstance(content, str):
            raise ValueError("create content must be text")
        raw = content.encode("utf-8")
        if len(raw) > WORKBENCH_MAX_WRITE_BYTES:
            raise ValueError("created file exceeds the writable size limit")
        try:
            descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise WorkbenchConflictError("create target appeared during transaction") from exc
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            raise
        after_digest = self._bytes_digest(raw)
        token = self._new_undo_token(
            {
                "operation": "workspace.create",
                "path": relative_name,
                "before_digest": "",
                "after_digest": after_digest,
            }
        )
        transaction = WorkbenchTransaction(
            operation="workspace.create",
            path=relative_name,
            after_digest=after_digest,
            undo_token=token,
        )
        return self._transaction_result(transaction, digest=after_digest)

    def _rename_file(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        path, relative_name = self._regular_file(parameters.get("path"))
        new_path, new_relative_name = self._resolve_path(
            parameters.get("new_path"), allow_root=False
        )
        if new_path.exists() or new_path.is_symlink():
            raise WorkbenchConflictError("rename target already exists")
        if not new_path.parent.is_dir() or new_path.parent.is_symlink():
            raise WorkbenchPathError("rename parent directory is unavailable")
        raw, before_digest = self._checked_file_bytes(path, parameters.get("before_digest"))
        path.rename(new_path)
        token = self._new_undo_token(
            {
                "operation": "workspace.rename",
                "path": relative_name,
                "new_path": new_relative_name,
                "before_digest": before_digest,
                "after_digest": before_digest,
            }
        )
        transaction = WorkbenchTransaction(
            operation="workspace.rename",
            path=relative_name,
            before_digest=before_digest,
            after_digest=before_digest,
            undo_token=token,
        )
        return self._transaction_result(
            transaction,
            digest=before_digest,
            extra={"new_path": new_relative_name, "byte_length": len(raw)},
        )

    def _delete_file(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        path, relative_name = self._regular_file(parameters.get("path"))
        raw, before_digest = self._checked_file_bytes(path, parameters.get("before_digest"))
        path.unlink()
        token = self._new_undo_token(
            {
                "operation": "workspace.delete",
                "path": relative_name,
                "before_digest": before_digest,
                "after_digest": "",
                "before_bytes": raw,
            }
        )
        transaction = WorkbenchTransaction(
            operation="workspace.delete",
            path=relative_name,
            before_digest=before_digest,
            after_digest="",
            undo_token=token,
        )
        return self._transaction_result(transaction, digest="", extra={"byte_length": len(raw)})

    def _undo_transaction(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        token = str(parameters.get("undo_token", "")).strip()
        if not token:
            raise ValueError("undo_token is required")
        with self._lock:
            record = self._undo_records.get(token)
        if record is None:
            raise ValueError("undo token is unknown or already consumed")
        path, relative_name = self._resolve_path(record["path"], allow_root=False)
        operation = str(record["operation"])
        if operation == "workspace.apply_patch":
            current = path.read_bytes() if path.is_file() else b""
            if self._bytes_digest(current) != record["after_digest"]:
                raise WorkbenchConflictError("file changed before undo")
            self._atomic_write(path, record["before_bytes"])
            after_digest = record["before_digest"]
        elif operation == "workspace.create":
            if (
                not path.is_file()
                or self._bytes_digest(path.read_bytes()) != record["after_digest"]
            ):
                raise WorkbenchConflictError("created file changed before undo")
            path.unlink()
            after_digest = ""
        elif operation == "workspace.delete":
            if path.exists() or path.is_symlink():
                raise WorkbenchConflictError("delete target reappeared before undo")
            raw = record["before_bytes"]
            descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            after_digest = record["before_digest"]
        elif operation == "workspace.rename":
            new_path, _ = self._resolve_path(record["new_path"], allow_root=False)
            if (
                path.exists()
                or not new_path.is_file()
                or self._bytes_digest(new_path.read_bytes()) != record["after_digest"]
            ):
                raise WorkbenchConflictError("rename state changed before undo")
            new_path.rename(path)
            after_digest = record["before_digest"]
        else:  # pragma: no cover - only internal records can reach this branch
            raise ValueError("unsupported undo operation")
        with self._lock:
            self._undo_records.pop(token, None)
        transaction = WorkbenchTransaction(
            operation="workspace.undo",
            path=relative_name,
            before_digest=str(record.get("after_digest", "")),
            after_digest=after_digest,
            reversible=False,
        )
        return self._transaction_result(transaction, digest=after_digest)

    @staticmethod
    def _bounded_text(raw: bytes | str | None, limit: int) -> tuple[str, bool]:
        data = raw.encode("utf-8", errors="replace") if isinstance(raw, str) else bytes(raw or b"")
        return data[:limit].decode("utf-8", errors="replace"), len(data) > limit

    def _artifact_state(self, raw_paths: Any) -> list[dict[str, Any]]:
        if raw_paths is None:
            return []
        if not isinstance(raw_paths, (list, tuple)):
            raise ValueError("expected_artifacts must be a list")
        artifacts: list[dict[str, Any]] = []
        for raw_path in raw_paths:
            path, relative_name = self._resolve_path(raw_path, allow_root=False)
            if path.is_symlink():
                raise WorkbenchPathError("symbolic-link artifacts are not accepted")
            item: dict[str, Any] = {
                "path": relative_name,
                "exists": path.exists(),
                "type": "missing",
            }
            if path.exists():
                item["type"] = "directory" if path.is_dir() else "file"
                item["size"] = int(path.stat().st_size) if path.is_file() else 0
                if path.is_file():
                    item["digest"] = self._bytes_digest(path.read_bytes())
            artifacts.append(item)
        return artifacts

    def _mcp_list(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        if parameters:
            raise ValueError("mcp.list does not accept parameters")
        return {
            "registry": self.mcp_registry.to_payload(),
            "tools": [item.to_payload() for item in self.mcp_registry.list_tools()],
        }

    def _mcp_invoke(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        tool_id = str(parameters.get("tool_id", "")).strip()
        arguments = parameters.get("arguments", {})
        descriptor = self.mcp_registry.validate_call(
            tool_id,
            arguments,
            registry_revision=parameters.get("registry_revision"),
        )
        if descriptor.executor_id == "workspace.list":
            result = self._list_workspace(arguments)
        else:  # pragma: no cover - registry prevents unbound executor ids
            raise ValueError("MCP executor is not connected")
        result_bytes = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(result_bytes) > descriptor.output_limit:
            raise WorkbenchResourceLimitError(f"MCP output exceeds {descriptor.output_limit} bytes")
        return {
            "tool_id": descriptor.tool_id,
            "source": descriptor.source,
            "registry_revision": self.mcp_registry.revision,
            "result": result,
            "provenance": {
                "kind": "mcp",
                "source": descriptor.source,
                "executor_id": descriptor.executor_id,
            },
            "success": True,
        }

    def _normalize_terminal_parameters(
        self, parameters: Mapping[str, Any]
    ) -> tuple[list[str], Path, str, float, int, set[str], Mapping[str, Any]]:
        raw_argv = parameters.get("argv")
        if not isinstance(raw_argv, (list, tuple)) or not raw_argv:
            raise ValueError("terminal.run requires a non-empty argv list")
        if any(not isinstance(item, str) or not item for item in raw_argv):
            raise ValueError("terminal argv entries must be non-empty strings")
        argv = list(raw_argv)
        cwd, relative_cwd = self._resolve_path(parameters.get("cwd", "."))
        if not cwd.is_dir() or cwd.is_symlink():
            raise WorkbenchPathError("terminal cwd must be a regular workspace directory")
        timeout_seconds = float(parameters.get("timeout_seconds", 30.0))
        if (
            not math.isfinite(timeout_seconds)
            or not 0.01 <= timeout_seconds <= WORKBENCH_MAX_TERMINAL_TIMEOUT_SECONDS
        ):
            raise ValueError("terminal timeout is outside the allowed range")
        output_limit = int(parameters.get("output_limit", 64 * 1024))
        if not 1 <= output_limit <= WORKBENCH_MAX_TERMINAL_OUTPUT_BYTES:
            raise ValueError("terminal output_limit is outside the allowed range")
        env_allowlist = parameters.get("env_allowlist", ())
        if not isinstance(env_allowlist, (list, tuple, set)):
            raise ValueError("env_allowlist must be a list")
        allowlist = {str(item) for item in env_allowlist}
        raw_env = parameters.get("env", {})
        if not isinstance(raw_env, Mapping):
            raise ValueError("terminal env must be an object")
        if any(str(key) not in allowlist for key in raw_env):
            raise ValueError("terminal env contains a variable outside env_allowlist")
        return (
            argv,
            cwd,
            relative_cwd,
            timeout_seconds,
            output_limit,
            allowlist,
            raw_env,
        )

    @staticmethod
    def _parse_diagnostics(*streams: tuple[str, str]) -> list[dict[str, Any]]:
        pattern = re.compile(
            r"^(?P<path>[^:\r\n]+):(?P<line>\d+)"
            r"(?::(?P<column>\d+))?:\s*"
            r"(?P<severity>error|warning|info)\b[:\s-]*(?P<message>.*)$",
            re.IGNORECASE,
        )
        diagnostics: list[dict[str, Any]] = []
        for source, text in streams:
            for raw_line in text.splitlines():
                match = pattern.match(raw_line.strip())
                if match is None:
                    continue
                diagnostics.append(
                    {
                        "source": source,
                        "path": match.group("path").strip(),
                        "line": int(match.group("line")),
                        "column": (
                            None if match.group("column") is None else int(match.group("column"))
                        ),
                        "severity": match.group("severity").lower(),
                        "message": match.group("message").strip(),
                    }
                )
                if len(diagnostics) >= 100:
                    return diagnostics
        return diagnostics

    def _run_terminal(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        (
            argv,
            cwd,
            relative_cwd,
            timeout_seconds,
            output_limit,
            _allowlist,
            raw_env,
        ) = self._normalize_terminal_parameters(parameters)
        process_env = os.environ.copy()
        process_env.update({str(key): str(value) for key, value in raw_env.items()})
        started = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                env=process_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                shell=False,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code: int | None = int(completed.returncode)
            stdout_raw: bytes | str | None = completed.stdout
            stderr_raw: bytes | str | None = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = None
            stdout_raw = exc.stdout
            stderr_raw = exc.stderr
        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
        stdout, stdout_truncated = self._bounded_text(stdout_raw, output_limit)
        stderr, stderr_truncated = self._bounded_text(stderr_raw, output_limit)
        diagnostics = self._parse_diagnostics(("stdout", stdout), ("stderr", stderr))
        expected_artifacts = self._artifact_state(parameters.get("expected_artifacts", []))
        artifact_failures = [item["path"] for item in expected_artifacts if not item["exists"]]
        execution_kind = str(parameters.get("execution_kind", "command"))
        if execution_kind not in {"command", "diagnostics", "test", "build"}:
            raise ValueError("execution_kind must be command, diagnostics, test, or build")
        return {
            "argv": argv,
            "cwd": relative_cwd,
            "shell": False,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "output_limit": output_limit,
            "execution_kind": execution_kind,
            "diagnostics": diagnostics,
            "expected_artifacts": expected_artifacts,
            "after_state": {
                "cwd": relative_cwd,
                "artifacts": expected_artifacts,
            },
            "artifact_failures": artifact_failures,
            "success": (
                not timed_out
                and exit_code == 0
                and not artifact_failures
                and not any(item["severity"] == "error" for item in diagnostics)
            ),
        }

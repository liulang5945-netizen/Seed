"""Content-addressed Workbench task boundaries for readout generation.

This module is a narrow contract between Seed's Workbench and Taiji's output
readout.  It carries task context and authorization evidence only.  It does
not carry prompts, targets, phase labels, answer tables, or executable tool
parameters, so a boundary cannot become a covert teacher signal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

WORKBENCH_TASK_BOUNDARY_FORMAT = "taiji-workbench-task-boundary-v1"
WORKBENCH_TASK_BOUNDARY_VERSION = 1
WORKBENCH_BOUNDARY_LIFECYCLES = ("active", "closed")
WORKBENCH_READOUT_GENERATIONS = ("protected", "active")
WORKBENCH_BOUNDARY_USAGES = ("execute", "read_only_replay")


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _text_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_required_text(value, name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class WorkbenchBoundaryAuthorization:
    """Live Seed evidence used to authorize one boundary token."""

    project_id: str
    task_id: str
    session_id: str
    capability_snapshot_id: str
    authorized_capability_ids: tuple[str, ...] = ()
    active_boundary_digest: str = ""
    current_tick: int = 0
    usage: str = "execute"
    version: int = WORKBENCH_TASK_BOUNDARY_VERSION

    def __post_init__(self) -> None:
        if self.version != WORKBENCH_TASK_BOUNDARY_VERSION:
            raise ValueError("unsupported Workbench boundary authorization version")
        for value, name in (
            (self.project_id, "project_id"),
            (self.task_id, "task_id"),
            (self.session_id, "session_id"),
            (self.capability_snapshot_id, "capability_snapshot_id"),
        ):
            _required_text(value, name)
        if self.active_boundary_digest:
            _required_text(self.active_boundary_digest, "active_boundary_digest")
        object.__setattr__(
            self,
            "authorized_capability_ids",
            _text_tuple(self.authorized_capability_ids, "authorized_capability_ids"),
        )
        if self.usage not in WORKBENCH_BOUNDARY_USAGES:
            raise ValueError("unsupported Workbench boundary usage")
        if int(self.current_tick) < 0:
            raise ValueError("current_tick cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "capability_snapshot_id": self.capability_snapshot_id,
            "authorized_capability_ids": list(self.authorized_capability_ids),
            "active_boundary_digest": self.active_boundary_digest,
            "current_tick": self.current_tick,
            "usage": self.usage,
        }


@dataclass(frozen=True)
class WorkbenchBoundaryDecision:
    """Read-only result of validating a task boundary."""

    accepted: bool
    reason_code: str
    boundary_digest: str
    generation_scope: str = ""
    read_only: bool = True
    version: int = WORKBENCH_TASK_BOUNDARY_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "boundary_digest": self.boundary_digest,
            "generation_scope": self.generation_scope,
            "read_only": self.read_only,
        }


@dataclass(frozen=True)
class WorkbenchTaskBoundary:
    """Immutable, content-addressed task context for readout selection."""

    token_digest: str
    project_id: str
    task_id: str
    session_id: str
    lifecycle: str
    language_id: str
    capability_snapshot_id: str
    capability_ids: tuple[str, ...]
    generation_scope: str
    issued_tick: int
    expires_tick: int
    parent_token_digest: str = ""
    version: int = WORKBENCH_TASK_BOUNDARY_VERSION
    format: str = WORKBENCH_TASK_BOUNDARY_FORMAT

    def __post_init__(self) -> None:
        if self.format != WORKBENCH_TASK_BOUNDARY_FORMAT:
            raise ValueError("unsupported Workbench task boundary format")
        if self.version != WORKBENCH_TASK_BOUNDARY_VERSION:
            raise ValueError("unsupported Workbench task boundary version")
        for value, name in (
            (self.token_digest, "token_digest"),
            (self.project_id, "project_id"),
            (self.task_id, "task_id"),
            (self.session_id, "session_id"),
            (self.language_id, "language_id"),
            (self.capability_snapshot_id, "capability_snapshot_id"),
            (self.generation_scope, "generation_scope"),
        ):
            _required_text(value, name)
        if self.lifecycle not in WORKBENCH_BOUNDARY_LIFECYCLES:
            raise ValueError("unsupported Workbench task boundary lifecycle")
        if self.generation_scope not in WORKBENCH_READOUT_GENERATIONS:
            raise ValueError("unsupported Workbench readout generation")
        object.__setattr__(self, "capability_ids", _text_tuple(self.capability_ids, "capability_ids"))
        if int(self.issued_tick) < 0:
            raise ValueError("issued_tick cannot be negative")
        if int(self.expires_tick) < int(self.issued_tick):
            raise ValueError("expires_tick cannot precede issued_tick")
        if self.parent_token_digest:
            _required_text(self.parent_token_digest, "parent_token_digest")
        if self.token_digest != _canonical_digest(self._identity_payload()):
            raise ValueError("Workbench task boundary digest mismatch")

    @classmethod
    def issue(
        cls,
        *,
        project_id: str,
        task_id: str,
        session_id: str,
        language_id: str,
        capability_snapshot_id: str,
        capability_ids: Sequence[str],
        generation_scope: str,
        issued_tick: int,
        ttl_ticks: int,
        parent_token_digest: str = "",
    ) -> WorkbenchTaskBoundary:
        issued = int(issued_tick)
        ttl = int(ttl_ticks)
        if issued < 0:
            raise ValueError("issued_tick cannot be negative")
        if ttl <= 0:
            raise ValueError("ttl_ticks must be positive")
        identity = {
            "format": WORKBENCH_TASK_BOUNDARY_FORMAT,
            "version": WORKBENCH_TASK_BOUNDARY_VERSION,
            "project_id": _required_text(project_id, "project_id"),
            "task_id": _required_text(task_id, "task_id"),
            "session_id": _required_text(session_id, "session_id"),
            "lifecycle": "active",
            "language_id": _required_text(language_id, "language_id"),
            "capability_snapshot_id": _required_text(
                capability_snapshot_id, "capability_snapshot_id"
            ),
            "capability_ids": list(_text_tuple(capability_ids, "capability_ids")),
            "generation_scope": _required_text(generation_scope, "generation_scope"),
            "issued_tick": issued,
            "expires_tick": issued + ttl,
            "parent_token_digest": str(parent_token_digest or "").strip(),
        }
        return cls(token_digest=_canonical_digest(identity), **identity)

    def successor(
        self,
        *,
        task_id: str,
        generation_scope: str,
        issued_tick: int,
        ttl_ticks: int,
    ) -> WorkbenchTaskBoundary:
        """Issue a new active task generation linked to this boundary."""

        return self.issue(
            project_id=self.project_id,
            task_id=task_id,
            session_id=self.session_id,
            language_id=self.language_id,
            capability_snapshot_id=self.capability_snapshot_id,
            capability_ids=self.capability_ids,
            generation_scope=generation_scope,
            issued_tick=issued_tick,
            ttl_ticks=ttl_ticks,
            parent_token_digest=self.token_digest,
        )

    def close(self, *, closed_tick: int) -> WorkbenchTaskBoundary:
        """Seal this task for read-only replay while preserving its lineage."""

        tick = int(closed_tick)
        if tick < self.issued_tick:
            raise ValueError("closed_tick cannot precede issued_tick")
        identity = {
            **self._identity_payload(),
            "lifecycle": "closed",
            "parent_token_digest": self.token_digest,
        }
        return type(self)(token_digest=_canonical_digest(identity), **identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "lifecycle": self.lifecycle,
            "language_id": self.language_id,
            "capability_snapshot_id": self.capability_snapshot_id,
            "capability_ids": list(self.capability_ids),
            "generation_scope": self.generation_scope,
            "issued_tick": self.issued_tick,
            "expires_tick": self.expires_tick,
            "parent_token_digest": self.parent_token_digest,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "token_digest": self.token_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WorkbenchTaskBoundary:
        if not isinstance(payload, Mapping):
            raise TypeError("Workbench task boundary must be a mapping")
        allowed = {
            "format",
            "version",
            "token_digest",
            "project_id",
            "task_id",
            "session_id",
            "lifecycle",
            "language_id",
            "capability_snapshot_id",
            "capability_ids",
            "generation_scope",
            "issued_tick",
            "expires_tick",
            "parent_token_digest",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Workbench task boundary contains unknown fields: {unknown}")
        return cls(
            token_digest=str(payload.get("token_digest", "")),
            project_id=str(payload.get("project_id", "")),
            task_id=str(payload.get("task_id", "")),
            session_id=str(payload.get("session_id", "")),
            lifecycle=str(payload.get("lifecycle", "")),
            language_id=str(payload.get("language_id", "")),
            capability_snapshot_id=str(payload.get("capability_snapshot_id", "")),
            capability_ids=tuple(payload.get("capability_ids", ())),
            generation_scope=str(payload.get("generation_scope", "")),
            issued_tick=int(payload.get("issued_tick", -1)),
            expires_tick=int(payload.get("expires_tick", -1)),
            parent_token_digest=str(payload.get("parent_token_digest", "")),
            version=int(payload.get("version", WORKBENCH_TASK_BOUNDARY_VERSION)),
            format=str(payload.get("format", WORKBENCH_TASK_BOUNDARY_FORMAT)),
        )

    def authorize(self, context: WorkbenchBoundaryAuthorization) -> WorkbenchBoundaryDecision:
        """Validate live task evidence and return the permitted readout scope."""

        def deny(reason_code: str) -> WorkbenchBoundaryDecision:
            return WorkbenchBoundaryDecision(
                accepted=False,
                reason_code=reason_code,
                boundary_digest=self.token_digest,
            )

        if context.project_id != self.project_id:
            return deny("cross_project_boundary")
        if context.task_id != self.task_id:
            return deny("task_boundary_mismatch")
        if context.session_id != self.session_id:
            return deny("session_boundary_mismatch")
        if context.capability_snapshot_id != self.capability_snapshot_id:
            return deny("stale_capability_snapshot")
        if context.current_tick > self.expires_tick:
            return deny("expired_boundary")
        if context.current_tick < self.issued_tick:
            return deny("future_boundary")
        if not set(self.capability_ids).issubset(context.authorized_capability_ids):
            return deny("unauthorized_capability")
        if context.usage == "execute":
            if self.lifecycle != "active":
                return deny("closed_boundary_not_executable")
            if context.active_boundary_digest != self.token_digest:
                return deny("stale_task_generation")
        elif self.lifecycle != "closed":
            return deny("replay_requires_closed_boundary")
        return WorkbenchBoundaryDecision(
            accepted=True,
            reason_code=(
                "active_generation_authorized"
                if context.usage == "execute"
                else "closed_generation_read_only"
            ),
            boundary_digest=self.token_digest,
            generation_scope=self.generation_scope,
            read_only=context.usage == "read_only_replay",
        )


def select_readout_generation(
    boundary: WorkbenchTaskBoundary | Mapping[str, Any],
    context: WorkbenchBoundaryAuthorization,
) -> str:
    """Return a validated generation scope for a Taiji readout caller."""

    resolved = (
        boundary
        if isinstance(boundary, WorkbenchTaskBoundary)
        else WorkbenchTaskBoundary.from_payload(boundary)
    )
    decision = resolved.authorize(context)
    if not decision.accepted:
        raise PermissionError(decision.reason_code)
    return decision.generation_scope

"""Native Taiji observation contract for verified Workbench evidence.

The Workbench remains the source of file, language, capability, and toolchain
facts.  This module only turns those already-admitted facts into a stable
numeric ``PerceptEvent`` for Taiji-owned semantic learners.  It deliberately
does not parse prose, select a tool, or create an executable intent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .contracts import PerceptEvent

WORKBENCH_OBSERVATION_FORMAT = "taiji-workbench-observation-v1"
WORKBENCH_OBSERVATION_VERSION = 1


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _unit(value: Any, name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


def _unique_texts(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(sorted({_text(item, name) for item in values}))
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


@dataclass(frozen=True)
class WorkbenchObservationSchema:
    """Versioned categorical vocabulary used by one observation course."""

    language_ids: tuple[str, ...]
    selection_states: tuple[str, ...]
    task_kinds: tuple[str, ...]
    extensions: tuple[str, ...]
    format: str = WORKBENCH_OBSERVATION_FORMAT
    version: int = WORKBENCH_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if self.format != WORKBENCH_OBSERVATION_FORMAT:
            raise ValueError("unsupported Workbench observation schema format")
        if int(self.version) != WORKBENCH_OBSERVATION_VERSION:
            raise ValueError("unsupported Workbench observation schema version")
        for value, name in (
            (self.language_ids, "language_ids"),
            (self.selection_states, "selection_states"),
            (self.task_kinds, "task_kinds"),
            (self.extensions, "extensions"),
        ):
            normalized = _unique_texts(value, name)
            object.__setattr__(self, name, normalized)
        if "unknown" not in self.language_ids:
            raise ValueError("observation language_ids must include unknown")
        if "unknown" not in self.selection_states:
            raise ValueError("observation selection_states must include unknown")
        if "<none>" not in self.extensions:
            raise ValueError("observation extensions must include <none>")

    @property
    def feature_names(self) -> tuple[str, ...]:
        categorical = tuple(
            [f"language:{item}" for item in self.language_ids]
            + [f"selection:{item}" for item in self.selection_states]
            + [f"task:{item}" for item in self.task_kinds]
            + [f"extension:{item}" for item in self.extensions]
        )
        scalar = (
            "read_success",
            "file_is_file",
            "byte_length_fraction",
            "language_confidence",
            "toolchain_available",
            "diagnostics_connected",
        )
        return categorical + scalar

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def schema_digest(self) -> str:
        return _digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "language_ids": list(self.language_ids),
            "selection_states": list(self.selection_states),
            "task_kinds": list(self.task_kinds),
            "extensions": list(self.extensions),
            "feature_names": list(self.feature_names),
            "feature_dim": self.feature_dim,
        }
        if include_digest:
            payload["schema_digest"] = self.schema_digest
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WorkbenchObservationSchema:
        schema = cls(
            language_ids=tuple(str(item) for item in payload.get("language_ids", ())),
            selection_states=tuple(str(item) for item in payload.get("selection_states", ())),
            task_kinds=tuple(str(item) for item in payload.get("task_kinds", ())),
            extensions=tuple(str(item) for item in payload.get("extensions", ())),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
        )
        if tuple(payload.get("feature_names", ())) != schema.feature_names:
            raise ValueError("Workbench observation feature names drifted")
        if int(payload.get("feature_dim", -1)) != schema.feature_dim:
            raise ValueError("Workbench observation feature dimension drifted")
        if str(payload.get("schema_digest", "")) != schema.schema_digest:
            raise ValueError("Workbench observation schema digest mismatch")
        return schema


@dataclass(frozen=True)
class WorkbenchObservation:
    """One content-addressed, non-executable Workbench sensation."""

    observation_id: str
    project_id: str
    task_id: str
    capability_snapshot_id: str
    capability_revision: int
    path: str
    file_digest: str
    read_success: bool
    file_is_file: bool
    byte_length: int
    language_id: str
    selection_state: str
    language_confidence: float
    toolchain_available: bool
    task_kind: str
    diagnostics_connected: bool
    schema: WorkbenchObservationSchema
    format: str = WORKBENCH_OBSERVATION_FORMAT
    version: int = WORKBENCH_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if self.format != WORKBENCH_OBSERVATION_FORMAT:
            raise ValueError("unsupported Workbench observation format")
        if int(self.version) != WORKBENCH_OBSERVATION_VERSION:
            raise ValueError("unsupported Workbench observation version")
        for value, name in (
            (self.observation_id, "observation_id"),
            (self.project_id, "project_id"),
            (self.task_id, "task_id"),
            (self.capability_snapshot_id, "capability_snapshot_id"),
            (self.path, "path"),
        ):
            _text(value, name)
        if int(self.capability_revision) < 1:
            raise ValueError("capability_revision must be positive")
        if int(self.byte_length) < 0:
            raise ValueError("byte_length cannot be negative")
        if self.language_id not in self.schema.language_ids:
            raise ValueError("observation language_id is outside its schema")
        if self.selection_state not in self.schema.selection_states:
            raise ValueError("observation selection_state is outside its schema")
        if self.task_kind not in self.schema.task_kinds:
            raise ValueError("observation task_kind is outside its schema")
        _unit(self.language_confidence, "language_confidence")
        if self.read_success and not self.file_is_file:
            raise ValueError("successful observation must describe a file")
        if self.read_success and len(self.file_digest) != 64:
            raise ValueError("successful observation file_digest is invalid")

    @property
    def file_extension(self) -> str:
        extension = Path(self.path).suffix.lower()
        return extension if extension in self.schema.extensions else "<none>"

    @property
    def observation_digest(self) -> str:
        return _digest(self.to_payload(include_digest=False))

    @property
    def feature_vector(self) -> torch.Tensor:
        categorical = [0.0] * (
            len(self.schema.language_ids)
            + len(self.schema.selection_states)
            + len(self.schema.task_kinds)
            + len(self.schema.extensions)
        )
        offset = 0
        categorical[offset + self.schema.language_ids.index(self.language_id)] = 1.0
        offset += len(self.schema.language_ids)
        categorical[offset + self.schema.selection_states.index(self.selection_state)] = 1.0
        offset += len(self.schema.selection_states)
        categorical[offset + self.schema.task_kinds.index(self.task_kind)] = 1.0
        offset += len(self.schema.task_kinds)
        categorical[offset + self.schema.extensions.index(self.file_extension)] = 1.0
        scalar = (
            float(self.read_success),
            float(self.file_is_file),
            min(float(self.byte_length), 4096.0) / 4096.0,
            float(self.language_confidence),
            float(self.toolchain_available),
            float(self.diagnostics_connected),
        )
        return torch.tensor(categorical + list(scalar), dtype=torch.float32)

    def to_percept_event(self, *, tick: int) -> PerceptEvent:
        confidence = self.language_confidence if self.read_success else 0.0
        return PerceptEvent(
            event_id=f"workbench-observation:{self.observation_digest[:32]}",
            observation_tick=int(tick),
            modality=WORKBENCH_OBSERVATION_FORMAT,
            features=self.feature_vector,
            assembly_id=f"workbench-assembly:{self.observation_digest[:32]}",
            duration=max(1, min(int(self.byte_length), 32)),
            boundary_score=1.0,
            prediction_error=0.0 if self.read_success else 1.0,
            boundary=True,
            confidence=confidence,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "observation_id": self.observation_id,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "capability_snapshot_id": self.capability_snapshot_id,
            "capability_revision": int(self.capability_revision),
            "path": self.path,
            "file_digest": self.file_digest,
            "read_success": bool(self.read_success),
            "file_is_file": bool(self.file_is_file),
            "byte_length": int(self.byte_length),
            "language_id": self.language_id,
            "selection_state": self.selection_state,
            "language_confidence": float(self.language_confidence),
            "toolchain_available": bool(self.toolchain_available),
            "task_kind": self.task_kind,
            "diagnostics_connected": bool(self.diagnostics_connected),
            "schema": self.schema.to_payload(),
        }
        if include_digest:
            payload["observation_digest"] = self.observation_digest
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WorkbenchObservation:
        observation = cls(
            observation_id=str(payload["observation_id"]),
            project_id=str(payload["project_id"]),
            task_id=str(payload["task_id"]),
            capability_snapshot_id=str(payload["capability_snapshot_id"]),
            capability_revision=int(payload["capability_revision"]),
            path=str(payload["path"]),
            file_digest=str(payload.get("file_digest", "")),
            read_success=bool(payload.get("read_success", False)),
            file_is_file=bool(payload.get("file_is_file", False)),
            byte_length=int(payload.get("byte_length", 0)),
            language_id=str(payload.get("language_id", "unknown")),
            selection_state=str(payload.get("selection_state", "unknown")),
            language_confidence=float(payload.get("language_confidence", 0.0)),
            toolchain_available=bool(payload.get("toolchain_available", False)),
            task_kind=str(payload["task_kind"]),
            diagnostics_connected=bool(payload.get("diagnostics_connected", False)),
            schema=WorkbenchObservationSchema.from_payload(payload["schema"]),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
        )
        if str(payload.get("observation_digest", "")) != observation.observation_digest:
            raise ValueError("Workbench observation digest mismatch")
        return observation

    @classmethod
    def from_workbench_evidence(
        cls,
        *,
        observation_id: str,
        project_id: str,
        task_id: str,
        path: str,
        capability_snapshot_id: str,
        capability_revision: int,
        read_result: Mapping[str, Any],
        language_result: Mapping[str, Any] | None,
        task_kind: str,
        schema: WorkbenchObservationSchema,
        diagnostics_connected: bool = False,
    ) -> WorkbenchObservation:
        read_success = bool(read_result.get("success", "error_code" not in read_result))
        # ``truncated`` describes the returned content window, not the target
        # kind.  A successful read of a large file must remain a file
        # observation; otherwise the learner would receive a false structural
        # fact merely because the Workbench applied its safety limit.
        file_is_file = read_success
        selection_state = "unknown"
        language_id = "unknown"
        language_confidence = 0.0
        toolchain_available = False
        file_digest = str(read_result.get("digest", "")) if read_success else ""
        if language_result is not None:
            selection_state = str(language_result.get("selection_state", "unknown"))
            if selection_state not in schema.selection_states:
                selection_state = "unknown"
            language_id = str(language_result.get("programming_language_id", "unknown"))
            if language_id not in schema.language_ids:
                language_id = "unknown"
            language_confidence = float(language_result.get("confidence", 0.0))
            execution = language_result.get("execution_snapshot", {})
            if isinstance(execution, Mapping):
                toolchain_available = bool(execution.get("available_for_language", ()))
            if read_success and file_digest != str(language_result.get("file_digest", "")):
                raise ValueError("Workbench read/language evidence digest mismatch")
        return cls(
            observation_id=observation_id,
            project_id=project_id,
            task_id=task_id,
            capability_snapshot_id=capability_snapshot_id,
            capability_revision=capability_revision,
            path=path,
            file_digest=file_digest,
            read_success=read_success,
            file_is_file=file_is_file,
            byte_length=int(read_result.get("byte_length", 0)),
            language_id=language_id,
            selection_state=selection_state,
            language_confidence=language_confidence,
            toolchain_available=toolchain_available,
            task_kind=task_kind,
            diagnostics_connected=diagnostics_connected,
            schema=schema,
        )


__all__ = [
    "WORKBENCH_OBSERVATION_FORMAT",
    "WORKBENCH_OBSERVATION_VERSION",
    "WorkbenchObservationSchema",
    "WorkbenchObservation",
]

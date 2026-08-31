"""Taiji-owned task interpretation and goal evidence.

This module is intentionally narrower than a planner. It records what the
architecture currently knows about an external task input and turns that
evidence into a Goal. It cannot select a tool, create an ActionIntent, or
execute a Workbench operation. A future learned semantic interpreter may
populate the same contract, but provenance and uncertainty remain visible.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import Goal
from .input_boundary import InputFrame
from .internalization import content_digest

TASK_INTERPRETATION_FORMAT = "taiji-task-interpretation-v1"
TASK_INTERPRETATION_VERSION = 1
TASK_INTERPRETATION_STATUSES = ("candidate", "resolved", "ambiguous", "rejected")
# This is a policy boundary, not a prompt-to-tool mapping. It remains
# explicit so a later task-planning config can calibrate it from evidence.
TASK_PLANNER_CONFIDENCE_FLOOR = 0.5
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def task_input_digest(payload: bytes) -> str:
    """Return the content address of the exact input bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("task input payload must be bytes")
    if not payload:
        raise ValueError("task input payload cannot be empty")
    return hashlib.sha256(payload).hexdigest()


def _required_text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _digest_text(value: str, name: str, *, optional: bool = False) -> str:
    normalized = str(value).strip().lower()
    if not normalized and optional:
        return ""
    if not _HEX_DIGEST.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _normalize_constraints(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("task constraints must be a sequence of strings")
    constraints = tuple(sorted({_required_text(item, "task constraint") for item in value}))
    return constraints


@dataclass(frozen=True)
class TaskInterpretation:
    """A content-addressed semantic candidate before planning.

    ``goal_description`` is a goal candidate, not a claim that Taiji has
    fully understood the task. The current implementation uses validated text
    at the input boundary as evidence and marks it unresolved with zero
    semantic confidence. A learned interpreter can later produce a better
    candidate without changing planner/effector ownership rules.
    """

    interpretation_id: str
    input_id: str
    input_digest: str
    modality: str
    goal_id: str
    goal_description: str
    constraints: tuple[str, ...] = ()
    context_digest: str = ""
    confidence: float = 0.0
    ambiguity: float = 1.0
    status: str = "candidate"
    provenance: str = "taiji.input"
    tick: int = 0
    version: int = TASK_INTERPRETATION_VERSION
    evidence_digest: str = ""

    def __post_init__(self) -> None:
        if int(self.version) != TASK_INTERPRETATION_VERSION:
            raise ValueError(f"unsupported task interpretation version: {self.version}")
        _required_text(self.interpretation_id, "interpretation_id")
        _required_text(self.input_id, "input_id")
        input_digest = _digest_text(self.input_digest, "input_digest")
        _required_text(self.modality, "modality")
        goal_id = _required_text(self.goal_id, "goal_id")
        goal_description = _required_text(self.goal_description, "goal_description")
        constraints = _normalize_constraints(self.constraints)
        context_digest = _digest_text(self.context_digest, "context_digest", optional=True)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("task interpretation confidence must be in [0, 1]")
        if not 0.0 <= float(self.ambiguity) <= 1.0:
            raise ValueError("task interpretation ambiguity must be in [0, 1]")
        status = _required_text(self.status, "task interpretation status")
        if status not in TASK_INTERPRETATION_STATUSES:
            raise ValueError(f"unsupported task interpretation status: {status}")
        provenance = _required_text(self.provenance, "task interpretation provenance")
        if int(self.tick) < 0:
            raise ValueError("task interpretation tick cannot be negative")

        identity = self._identity_payload(
            input_id=str(self.input_id),
            input_digest=input_digest,
            modality=str(self.modality),
            goal_description=goal_description,
            constraints=constraints,
            context_digest=context_digest,
            confidence=float(self.confidence),
            ambiguity=float(self.ambiguity),
            status=status,
            provenance=provenance,
            tick=int(self.tick),
        )
        identity_digest = content_digest(identity)
        expected_goal_id = f"goal:{identity_digest[:24]}"
        expected_interpretation_id = f"task-interpretation:{identity_digest[:24]}"
        if goal_id != expected_goal_id:
            raise ValueError("goal_id is not content-addressed to task interpretation evidence")
        if str(self.interpretation_id) != expected_interpretation_id:
            raise ValueError(
                "interpretation_id is not content-addressed to task interpretation evidence"
            )
        expected_evidence_digest = content_digest(
            {
                "format": TASK_INTERPRETATION_FORMAT,
                "version": TASK_INTERPRETATION_VERSION,
                **identity,
                "goal_id": goal_id,
                "interpretation_id": expected_interpretation_id,
            }
        )
        evidence_digest = _digest_text(self.evidence_digest, "evidence_digest")
        if evidence_digest != expected_evidence_digest:
            raise ValueError("task interpretation evidence digest does not match its payload")

        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "goal_description", goal_description)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "evidence_digest", evidence_digest)

    @staticmethod
    def _identity_payload(
        *,
        input_id: str,
        input_digest: str,
        modality: str,
        goal_description: str,
        constraints: tuple[str, ...],
        context_digest: str,
        confidence: float,
        ambiguity: float,
        status: str,
        provenance: str,
        tick: int,
    ) -> dict[str, Any]:
        return {
            "input_id": input_id,
            "input_digest": input_digest,
            "modality": modality,
            "goal_description": goal_description,
            "constraints": list(constraints),
            "context_digest": context_digest,
            "confidence": confidence,
            "ambiguity": ambiguity,
            "status": status,
            "provenance": provenance,
            "tick": tick,
        }

    def to_goal(self) -> Goal:
        """Project this evidence into the existing Taiji Goal contract."""

        return Goal(goal_id=self.goal_id, description=self.goal_description)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": TASK_INTERPRETATION_FORMAT,
            "version": self.version,
            "interpretation_id": self.interpretation_id,
            "input_id": self.input_id,
            "input_digest": self.input_digest,
            "modality": self.modality,
            "goal_id": self.goal_id,
            "goal_description": self.goal_description,
            "constraints": list(self.constraints),
            "context_digest": self.context_digest,
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "status": self.status,
            "provenance": self.provenance,
            "tick": self.tick,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_input(
        cls,
        frame: InputFrame,
        *,
        goal_description: str | None = None,
        constraints: Sequence[str] = (),
        context_digest: str = "",
        confidence: float = 0.0,
        ambiguity: float = 1.0,
        status: str = "candidate",
        tick: int = 0,
        provenance: str | None = None,
    ) -> TaskInterpretation:
        """Create evidence from an input frame without choosing an action.

        For text frames, omitting ``goal_description`` preserves exact UTF-8
        text as a candidate. Non-text modalities must be interpreted by an
        explicit semantic organ later; this boundary refuses to invent a
        description for them.
        """

        if not isinstance(frame, InputFrame):
            raise TypeError("frame must be a Taiji InputFrame")
        description = goal_description
        if description is None:
            if frame.modality not in {"text", "text-utf8"}:
                raise ValueError("non-text task interpretation requires semantic goal evidence")
            description = frame.payload.decode("utf-8")
        input_digest = task_input_digest(bytes(frame.payload))
        normalized_constraints = _normalize_constraints(constraints)
        normalized_context_digest = _digest_text(context_digest, "context_digest", optional=True)
        evidence_provenance = frame.provenance if provenance is None else _required_text(
            provenance, "provenance"
        )
        identity = cls._identity_payload(
            input_id=frame.input_id,
            input_digest=input_digest,
            modality=frame.modality,
            goal_description=_required_text(description, "goal_description"),
            constraints=normalized_constraints,
            context_digest=normalized_context_digest,
            confidence=float(confidence),
            ambiguity=float(ambiguity),
            status=status,
            provenance=evidence_provenance,
            tick=int(tick),
        )
        identity_digest = content_digest(identity)
        interpretation_id = f"task-interpretation:{identity_digest[:24]}"
        goal_id = f"goal:{identity_digest[:24]}"
        evidence_payload = {
            "format": TASK_INTERPRETATION_FORMAT,
            "version": TASK_INTERPRETATION_VERSION,
            **identity,
            "goal_id": goal_id,
            "interpretation_id": interpretation_id,
        }
        return cls(
            interpretation_id=interpretation_id,
            input_id=frame.input_id,
            input_digest=input_digest,
            modality=frame.modality,
            goal_id=goal_id,
            goal_description=identity["goal_description"],
            constraints=normalized_constraints,
            context_digest=normalized_context_digest,
            confidence=float(confidence),
            ambiguity=float(ambiguity),
            status=status,
            provenance=evidence_provenance,
            tick=int(tick),
            evidence_digest=content_digest(evidence_payload),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskInterpretation:
        if payload.get("format") != TASK_INTERPRETATION_FORMAT:
            raise ValueError("unsupported task interpretation format")
        return cls(
            version=int(payload["version"]),
            interpretation_id=str(payload["interpretation_id"]),
            input_id=str(payload["input_id"]),
            input_digest=str(payload["input_digest"]),
            modality=str(payload["modality"]),
            goal_id=str(payload["goal_id"]),
            goal_description=str(payload["goal_description"]),
            constraints=tuple(str(item) for item in payload.get("constraints", ())),
            context_digest=str(payload.get("context_digest", "")),
            confidence=float(payload.get("confidence", 0.0)),
            ambiguity=float(payload.get("ambiguity", 1.0)),
            status=str(payload.get("status", "candidate")),
            provenance=str(payload.get("provenance", "taiji.input")),
            tick=int(payload.get("tick", 0)),
            evidence_digest=str(payload["evidence_digest"]),
        )

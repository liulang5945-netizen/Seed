"""Taiji-owned semantic task decomposition evidence.

This contract is deliberately between language understanding and planning. It
can describe a bounded sequence of semantic steps, but it cannot carry a
capability id, tool name, executor, command, or ActionIntent. Workbench
grounding remains a later Taiji decision against live affordances.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .internalization import content_digest
from .task_interpretation import TASK_PLANNER_CONFIDENCE_FLOOR, TaskInterpretation

TASK_DECOMPOSITION_FORMAT = "taiji-task-decomposition-v1"
TASK_DECOMPOSITION_VERSION = 1
TASK_DECOMPOSITION_STATUSES = ("candidate", "resolved", "ambiguous", "rejected")
TASK_DECOMPOSITION_MAX_STEPS = 8
_FORBIDDEN_SEMANTIC_KEYS = frozenset(
    {
        "action",
        "action_kind",
        "argv",
        "capability",
        "capability_id",
        "command",
        "executor",
        "intent",
        "intent_id",
        "parameter_binding",
        "parameter_bindings",
        "parameters",
        "shell",
        "tool",
        "tool_id",
    }
)


def _required_text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _bounded(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _validate_no_execution_keys(value: Any, *, path: str = "semantic_slots") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_name = str(key).strip().lower()
            if key_name in _FORBIDDEN_SEMANTIC_KEYS:
                raise ValueError(
                    f"{path} cannot carry execution field {key_name!r}; "
                    "grounding belongs to Taiji planning"
                )
            _validate_no_execution_keys(nested, path=f"{path}.{key_name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _validate_no_execution_keys(nested, path=f"{path}[{index}]")


def _normalize_slots(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("task semantic_slots must be a mapping")
    _validate_no_execution_keys(value)
    normalized = {str(key).strip(): nested for key, nested in value.items()}
    if any(not key for key in normalized):
        raise ValueError("task semantic_slots keys cannot be empty")
    content_digest(normalized)
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class TaskStepEvidence:
    """One semantic step with no direct capability or executable binding."""

    step_id: str
    description: str
    semantic_slots: Mapping[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    confidence: float = 0.0
    ambiguity: float = 1.0
    provenance: str = "taiji.semantic"
    tick: int = 0
    version: int = TASK_DECOMPOSITION_VERSION
    evidence_digest: str = ""

    def __post_init__(self) -> None:
        if int(self.version) != TASK_DECOMPOSITION_VERSION:
            raise ValueError("unsupported task step evidence version")
        description = _required_text(self.description, "task step description")
        slots = _normalize_slots(self.semantic_slots)
        expected = str(self.expected_outcome).strip()
        confidence = _bounded(self.confidence, "task step confidence")
        ambiguity = _bounded(self.ambiguity, "task step ambiguity")
        provenance = _required_text(self.provenance, "task step provenance")
        tick = int(self.tick)
        if tick < 0:
            raise ValueError("task step tick cannot be negative")
        identity = self._identity_payload(
            description=description,
            semantic_slots=slots,
            expected_outcome=expected,
            confidence=confidence,
            ambiguity=ambiguity,
            provenance=provenance,
            tick=tick,
        )
        expected_step_id = f"task-step:{content_digest(identity)[:24]}"
        if str(self.step_id) != expected_step_id:
            raise ValueError("task step id is not content-addressed to semantic evidence")
        expected_evidence_digest = content_digest(
            {
                "format": TASK_DECOMPOSITION_FORMAT,
                "version": TASK_DECOMPOSITION_VERSION,
                "step_id": expected_step_id,
                **identity,
            }
        )
        if str(self.evidence_digest) != expected_evidence_digest:
            raise ValueError("task step evidence digest does not match its payload")
        object.__setattr__(self, "step_id", expected_step_id)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "semantic_slots", slots)
        object.__setattr__(self, "expected_outcome", expected)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "ambiguity", ambiguity)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "tick", tick)
        object.__setattr__(self, "evidence_digest", expected_evidence_digest)

    @staticmethod
    def _identity_payload(
        *,
        description: str,
        semantic_slots: Mapping[str, Any],
        expected_outcome: str,
        confidence: float,
        ambiguity: float,
        provenance: str,
        tick: int,
    ) -> dict[str, Any]:
        return {
            "description": description,
            "semantic_slots": dict(semantic_slots),
            "expected_outcome": expected_outcome,
            "confidence": confidence,
            "ambiguity": ambiguity,
            "provenance": provenance,
            "tick": tick,
        }

    @classmethod
    def from_semantic(
        cls,
        *,
        description: str,
        semantic_slots: Mapping[str, Any],
        expected_outcome: str = "",
        confidence: float = 0.0,
        ambiguity: float = 1.0,
        provenance: str = "taiji.semantic",
        tick: int = 0,
    ) -> TaskStepEvidence:
        normalized_description = _required_text(description, "task step description")
        normalized_slots = _normalize_slots(semantic_slots)
        identity = cls._identity_payload(
            description=normalized_description,
            semantic_slots=normalized_slots,
            expected_outcome=str(expected_outcome).strip(),
            confidence=_bounded(confidence, "task step confidence"),
            ambiguity=_bounded(ambiguity, "task step ambiguity"),
            provenance=_required_text(provenance, "task step provenance"),
            tick=int(tick),
        )
        step_id = f"task-step:{content_digest(identity)[:24]}"
        return cls(
            step_id=step_id,
            description=normalized_description,
            semantic_slots=normalized_slots,
            expected_outcome=identity["expected_outcome"],
            confidence=identity["confidence"],
            ambiguity=identity["ambiguity"],
            provenance=identity["provenance"],
            tick=identity["tick"],
            evidence_digest=content_digest(
                {
                    "format": TASK_DECOMPOSITION_FORMAT,
                    "version": TASK_DECOMPOSITION_VERSION,
                    "step_id": step_id,
                    **identity,
                }
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": TASK_DECOMPOSITION_FORMAT,
            "version": self.version,
            "step_id": self.step_id,
            "description": self.description,
            "semantic_slots": dict(self.semantic_slots),
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "provenance": self.provenance,
            "tick": self.tick,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaskStepEvidence:
        if payload.get("format") != TASK_DECOMPOSITION_FORMAT:
            raise ValueError("unsupported task step evidence format")
        return cls(
            version=int(payload["version"]),
            step_id=str(payload["step_id"]),
            description=str(payload["description"]),
            semantic_slots=dict(payload.get("semantic_slots", {})),
            expected_outcome=str(payload.get("expected_outcome", "")),
            confidence=float(payload.get("confidence", 0.0)),
            ambiguity=float(payload.get("ambiguity", 1.0)),
            provenance=str(payload.get("provenance", "taiji.semantic")),
            tick=int(payload.get("tick", 0)),
            evidence_digest=str(payload["evidence_digest"]),
        )


@dataclass(frozen=True)
class TaskDecomposition:
    """A bounded semantic sequence awaiting Taiji Workbench grounding."""

    decomposition_id: str
    interpretation_id: str
    goal_id: str
    steps: tuple[TaskStepEvidence, ...]
    confidence: float = 0.0
    ambiguity: float = 1.0
    status: str = "candidate"
    provenance: str = "taiji.semantic"
    tick: int = 0
    version: int = TASK_DECOMPOSITION_VERSION
    evidence_digest: str = ""

    def __post_init__(self) -> None:
        if int(self.version) != TASK_DECOMPOSITION_VERSION:
            raise ValueError("unsupported task decomposition version")
        interpretation_id = _required_text(self.interpretation_id, "interpretation_id")
        goal_id = _required_text(self.goal_id, "goal_id")
        steps = tuple(self.steps)
        if not 1 <= len(steps) <= TASK_DECOMPOSITION_MAX_STEPS:
            raise ValueError("task decomposition step count must be between 1 and 8")
        if any(not isinstance(step, TaskStepEvidence) for step in steps):
            raise TypeError("task decomposition steps must be TaskStepEvidence values")
        if len({step.step_id for step in steps}) != len(steps):
            raise ValueError("task decomposition step ids must be unique")
        tick = int(self.tick)
        if tick < 0 or any(step.tick != tick for step in steps):
            raise ValueError("task decomposition and steps must share a non-negative tick")
        confidence = _bounded(self.confidence, "task decomposition confidence")
        ambiguity = _bounded(self.ambiguity, "task decomposition ambiguity")
        status = _required_text(self.status, "task decomposition status")
        if status not in TASK_DECOMPOSITION_STATUSES:
            raise ValueError(f"unsupported task decomposition status: {status}")
        provenance = _required_text(self.provenance, "task decomposition provenance")
        identity = self._identity_payload(
            interpretation_id=interpretation_id,
            goal_id=goal_id,
            steps=steps,
            confidence=confidence,
            ambiguity=ambiguity,
            status=status,
            provenance=provenance,
            tick=tick,
        )
        expected_id = f"task-decomposition:{content_digest(identity)[:24]}"
        if str(self.decomposition_id) != expected_id:
            raise ValueError("task decomposition id is not content-addressed")
        expected_digest = content_digest(
            {
                "format": TASK_DECOMPOSITION_FORMAT,
                "version": TASK_DECOMPOSITION_VERSION,
                "decomposition_id": expected_id,
                **identity,
            }
        )
        if str(self.evidence_digest) != expected_digest:
            raise ValueError("task decomposition evidence digest does not match its payload")
        object.__setattr__(self, "interpretation_id", interpretation_id)
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "ambiguity", ambiguity)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "tick", tick)
        object.__setattr__(self, "evidence_digest", expected_digest)

    @staticmethod
    def _identity_payload(
        *,
        interpretation_id: str,
        goal_id: str,
        steps: tuple[TaskStepEvidence, ...],
        confidence: float,
        ambiguity: float,
        status: str,
        provenance: str,
        tick: int,
    ) -> dict[str, Any]:
        return {
            "interpretation_id": interpretation_id,
            "goal_id": goal_id,
            "steps": [
                {
                    "step_id": step.step_id,
                    **TaskStepEvidence._identity_payload(
                        description=step.description,
                        semantic_slots=step.semantic_slots,
                        expected_outcome=step.expected_outcome,
                        confidence=step.confidence,
                        ambiguity=step.ambiguity,
                        provenance=step.provenance,
                        tick=step.tick,
                    ),
                }
                for step in steps
            ],
            "confidence": confidence,
            "ambiguity": ambiguity,
            "status": status,
            "provenance": provenance,
            "tick": tick,
        }

    @classmethod
    def from_interpretation(
        cls,
        interpretation: TaskInterpretation,
        semantic_steps: Sequence[Mapping[str, Any]],
        *,
        confidence: float | None = None,
        ambiguity: float | None = None,
        status: str = "resolved",
        provenance: str = "taiji.semantic",
    ) -> TaskDecomposition:
        if not isinstance(interpretation, TaskInterpretation):
            raise TypeError("task decomposition requires TaskInterpretation evidence")
        if interpretation.status != "resolved":
            raise ValueError("task decomposition requires resolved task interpretation")
        if interpretation.confidence < TASK_PLANNER_CONFIDENCE_FLOOR:
            raise ValueError("task decomposition requires high-confidence interpretation")
        if isinstance(semantic_steps, (str, bytes, bytearray)):
            raise TypeError("semantic_steps must be a sequence of mappings")
        if not isinstance(semantic_steps, Sequence):
            raise TypeError("semantic_steps must be a sequence of mappings")
        if any(not isinstance(item, Mapping) for item in semantic_steps):
            raise TypeError("every semantic step must be a mapping")
        steps = tuple(
            TaskStepEvidence.from_semantic(
                description=str(item["description"]),
                semantic_slots=dict(item.get("semantic_slots", {})),
                expected_outcome=str(item.get("expected_outcome", "")),
                confidence=float(item.get("confidence", interpretation.confidence)),
                ambiguity=float(item.get("ambiguity", interpretation.ambiguity)),
                provenance=str(item.get("provenance", provenance)),
                tick=interpretation.tick,
            )
            for item in semantic_steps
        )
        if not steps:
            raise ValueError("task decomposition requires at least one semantic step")
        normalized_confidence = (
            interpretation.confidence if confidence is None else _bounded(confidence, "confidence")
        )
        normalized_ambiguity = (
            interpretation.ambiguity if ambiguity is None else _bounded(ambiguity, "ambiguity")
        )
        identity = cls._identity_payload(
            interpretation_id=interpretation.interpretation_id,
            goal_id=interpretation.goal_id,
            steps=steps,
            confidence=normalized_confidence,
            ambiguity=normalized_ambiguity,
            status=status,
            provenance=provenance,
            tick=interpretation.tick,
        )
        decomposition_id = f"task-decomposition:{content_digest(identity)[:24]}"
        return cls(
            decomposition_id=decomposition_id,
            interpretation_id=interpretation.interpretation_id,
            goal_id=interpretation.goal_id,
            steps=steps,
            confidence=normalized_confidence,
            ambiguity=normalized_ambiguity,
            status=status,
            provenance=provenance,
            tick=interpretation.tick,
            evidence_digest=content_digest(
                {
                    "format": TASK_DECOMPOSITION_FORMAT,
                    "version": TASK_DECOMPOSITION_VERSION,
                    "decomposition_id": decomposition_id,
                    **identity,
                }
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": TASK_DECOMPOSITION_FORMAT,
            "version": self.version,
            "decomposition_id": self.decomposition_id,
            "interpretation_id": self.interpretation_id,
            "goal_id": self.goal_id,
            "steps": [step.to_payload() for step in self.steps],
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "status": self.status,
            "provenance": self.provenance,
            "tick": self.tick,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaskDecomposition:
        if payload.get("format") != TASK_DECOMPOSITION_FORMAT:
            raise ValueError("unsupported task decomposition format")
        raw_steps = payload.get("steps", ())
        if isinstance(raw_steps, (str, bytes, bytearray)) or not isinstance(raw_steps, Sequence):
            raise TypeError("task decomposition steps must be a sequence")
        if any(not isinstance(item, Mapping) for item in raw_steps):
            raise TypeError("every task decomposition step must be a mapping")
        return cls(
            version=int(payload["version"]),
            decomposition_id=str(payload["decomposition_id"]),
            interpretation_id=str(payload["interpretation_id"]),
            goal_id=str(payload["goal_id"]),
            steps=tuple(
                TaskStepEvidence.from_payload(dict(item))
                for item in raw_steps
            ),
            confidence=float(payload.get("confidence", 0.0)),
            ambiguity=float(payload.get("ambiguity", 1.0)),
            status=str(payload.get("status", "candidate")),
            provenance=str(payload.get("provenance", "taiji.semantic")),
            tick=int(payload.get("tick", 0)),
            evidence_digest=str(payload["evidence_digest"]),
        )

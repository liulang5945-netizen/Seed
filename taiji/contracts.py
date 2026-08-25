"""Versioned contracts owned by the Taiji cognitive architecture.

The contracts in this module are deliberately small and serializable.  They
are the boundary between organs, cognition and effectors; they are not a
second model hidden in Seed and they do not prescribe the representation used
inside a future Taiji implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

CONTRACT_VERSION = 1
CONTRACT_FORMAT = "taiji-native-v1"


def _check_version(version: int) -> None:
    if int(version) != CONTRACT_VERSION:
        raise ValueError(f"unsupported Taiji contract version: {version}")


def _check_text(value: str, name: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _check_unit(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return value


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {str(key): _clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return value


def _encode_value(value: Any) -> Any:
    """Encode contract metadata without making torch tensors implicit."""

    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "value": value.detach().cpu().clone()}
    if isinstance(value, bytes):
        return {"kind": "bytes", "value": bytes(value)}
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "value": {str(key): _encode_value(item) for key, item in value.items()},
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "value": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "value": [_encode_value(item) for item in value]}
    return value


def _decode_value(value: Any, *, device: torch.device | str) -> Any:
    if not isinstance(value, Mapping) or "kind" not in value:
        return value
    kind = value["kind"]
    payload = value.get("value")
    if kind == "tensor":
        return payload.detach().to(device).clone()
    if kind == "bytes":
        return bytes(payload)
    if kind == "mapping":
        return {str(key): _decode_value(item, device=device) for key, item in payload.items()}
    if kind == "tuple":
        return tuple(_decode_value(item, device=device) for item in payload)
    if kind == "list":
        return [_decode_value(item, device=device) for item in payload]
    raise ValueError(f"unsupported contract value kind: {kind}")


def _normalize_pairs(value: Any, name: str) -> tuple[tuple[str, Any], ...]:
    """Freeze a user-facing mapping into deterministic contract metadata."""

    items = value.items() if isinstance(value, Mapping) else value
    normalized: dict[str, Any] = {}
    try:
        iterator = iter(items)
    except TypeError as exc:
        raise ValueError(f"{name} must be a mapping or pair sequence") from exc
    for pair in iterator:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{name} entries must be key/value pairs")
        key = _check_text(str(pair[0]), f"{name} key")
        if key in normalized:
            raise ValueError(f"{name} contains duplicate key: {key}")
        normalized[key] = _clone_value(pair[1])
    return tuple(sorted(normalized.items()))


def _normalize_tags(value: Any, name: str) -> tuple[str, ...]:
    tags = tuple(_check_text(str(item), name) for item in value)
    if len(set(tags)) != len(tags):
        raise ValueError(f"{name} cannot contain duplicate values")
    return tuple(sorted(tags))


@dataclass(frozen=True)
class Observation:
    """A versioned sensation entering Taiji through an organ boundary."""

    modality: str
    value: Any
    timestamp: int
    source: str
    provenance: str = "external"
    confidence: float = 1.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.modality, "modality")
        _check_text(self.source, "source")
        _check_text(self.provenance, "provenance")
        if int(self.timestamp) < 0:
            raise ValueError("timestamp cannot be negative")
        _check_unit(self.confidence, "confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "modality": self.modality,
            "value": _encode_value(self.value),
            "timestamp": self.timestamp,
            "source": self.source,
            "provenance": self.provenance,
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> Observation:
        return cls(
            version=int(payload["version"]),
            modality=str(payload["modality"]),
            value=_decode_value(payload["value"], device=device),
            timestamp=int(payload["timestamp"]),
            source=str(payload["source"]),
            provenance=str(payload.get("provenance", "external")),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class PerceptEvent:
    """A learned or provisional event emitted by a perceptual hierarchy."""

    event_id: str
    observation_tick: int
    modality: str
    features: torch.Tensor
    assembly_id: str = ""
    duration: int = 1
    boundary_score: float = 0.0
    prediction_error: float = 0.0
    boundary: bool = False
    confidence: float = 1.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.event_id, "event_id")
        _check_text(self.modality, "modality")
        if not self.assembly_id:
            object.__setattr__(self, "assembly_id", self.event_id)
        if int(self.observation_tick) < 0:
            raise ValueError("observation_tick cannot be negative")
        if self.features.ndim != 1:
            raise ValueError("percept features must be a vector")
        if int(self.duration) <= 0:
            raise ValueError("percept duration must be positive")
        _check_unit(self.boundary_score, "boundary_score")
        _check_unit(self.prediction_error, "prediction_error")
        _check_unit(self.confidence, "confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "observation_tick": self.observation_tick,
            "modality": self.modality,
            "features": self.features.detach().cpu().clone(),
            "assembly_id": self.assembly_id,
            "duration": self.duration,
            "boundary_score": self.boundary_score,
            "prediction_error": self.prediction_error,
            "boundary": self.boundary,
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> PerceptEvent:
        return cls(
            version=int(payload["version"]),
            event_id=str(payload["event_id"]),
            observation_tick=int(payload["observation_tick"]),
            modality=str(payload["modality"]),
            features=payload["features"].detach().to(device).clone(),
            assembly_id=str(payload.get("assembly_id", payload["event_id"])),
            duration=int(payload.get("duration", 1)),
            boundary_score=float(payload.get("boundary_score", 0.0)),
            prediction_error=float(payload.get("prediction_error", 0.0)),
            boundary=bool(payload.get("boundary", False)),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class WorkspaceState:
    """Capacity-limited focus and broadcast state."""

    tick: int
    focus: tuple[str, ...] = ()
    bindings: tuple[tuple[str, str], ...] = ()
    broadcast: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    capacity: int = 0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if int(self.tick) < 0:
            raise ValueError("workspace tick cannot be negative")
        if int(self.capacity) < 0:
            raise ValueError("workspace capacity cannot be negative")
        if len(self.focus) > int(self.capacity) and self.capacity != 0:
            raise ValueError("workspace focus exceeds capacity")
        if self.broadcast.ndim != 1:
            raise ValueError("workspace broadcast must be a vector")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "focus": list(self.focus),
            "bindings": [list(pair) for pair in self.bindings],
            "broadcast": self.broadcast.detach().cpu().clone(),
            "capacity": self.capacity,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorkspaceState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            focus=tuple(str(item) for item in payload.get("focus", ())),
            bindings=tuple((str(pair[0]), str(pair[1])) for pair in payload.get("bindings", ())),
            broadcast=payload["broadcast"].detach().to(device).clone(),
            capacity=int(payload.get("capacity", 0)),
        )


@dataclass(frozen=True)
class WorldObject:
    """A persistent object with learned-facing, schema-free attributes."""

    object_id: str
    attributes: tuple[tuple[str, Any], ...] = ()
    tags: tuple[str, ...] = ()
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.object_id, "object_id")
        object.__setattr__(self, "attributes", _normalize_pairs(self.attributes, "object attributes"))
        object.__setattr__(self, "tags", _normalize_tags(self.tags, "object tags"))

    def attribute(self, name: str, default: Any = None) -> Any:
        key = str(name)
        return dict(self.attributes).get(key, default)

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "object_id": self.object_id,
            "attributes": _encode_value(dict(self.attributes)),
            "tags": list(self.tags),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldObject:
        attributes = _decode_value(payload.get("attributes", {}), device=device)
        return cls(
            version=int(payload["version"]),
            object_id=str(payload["object_id"]),
            attributes=attributes,
            tags=tuple(str(item) for item in payload.get("tags", ())),
        )


@dataclass(frozen=True)
class WorldEvent:
    """A time-indexed event that can update or explain world state."""

    event_id: str
    kind: str
    tick: int
    subject_id: str = ""
    object_id: str = ""
    attributes: tuple[tuple[str, Any], ...] = ()
    provenance: str = "experienced"
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.event_id, "event_id")
        _check_text(self.kind, "event kind")
        _check_text(self.provenance, "event provenance")
        if int(self.tick) < 0:
            raise ValueError("event tick cannot be negative")
        if self.subject_id:
            _check_text(self.subject_id, "event subject_id")
        if self.object_id:
            _check_text(self.object_id, "event object_id")
        object.__setattr__(self, "attributes", _normalize_pairs(self.attributes, "event attributes"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "kind": self.kind,
            "tick": self.tick,
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "attributes": _encode_value(dict(self.attributes)),
            "provenance": self.provenance,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldEvent:
        attributes = _decode_value(payload.get("attributes", {}), device=device)
        return cls(
            version=int(payload["version"]),
            event_id=str(payload["event_id"]),
            kind=str(payload["kind"]),
            tick=int(payload["tick"]),
            subject_id=str(payload.get("subject_id", "")),
            object_id=str(payload.get("object_id", "")),
            attributes=attributes,
            provenance=str(payload.get("provenance", "experienced")),
        )


@dataclass(frozen=True)
class WorldAffordance:
    """An action possibility grounded in the current world state."""

    affordance_id: str
    action_kind: str
    actor_id: str = ""
    target_id: str = ""
    parameters: tuple[tuple[str, Any], ...] = ()
    confidence: float = 1.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.affordance_id, "affordance_id")
        _check_text(self.action_kind, "affordance action_kind")
        if self.actor_id:
            _check_text(self.actor_id, "affordance actor_id")
        if self.target_id:
            _check_text(self.target_id, "affordance target_id")
        _check_unit(self.confidence, "affordance confidence")
        object.__setattr__(self, "parameters", _normalize_pairs(self.parameters, "affordance parameters"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "affordance_id": self.affordance_id,
            "action_kind": self.action_kind,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "parameters": _encode_value(dict(self.parameters)),
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldAffordance:
        parameters = _decode_value(payload.get("parameters", {}), device=device)
        return cls(
            version=int(payload["version"]),
            affordance_id=str(payload["affordance_id"]),
            action_kind=str(payload["action_kind"]),
            actor_id=str(payload.get("actor_id", "")),
            target_id=str(payload.get("target_id", "")),
            parameters=parameters,
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class WorldState:
    """Persistent object/event state and explicit relation handles."""

    tick: int
    latent: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    entities: tuple[str, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    objects: tuple[WorldObject, ...] = ()
    events: tuple[WorldEvent, ...] = ()
    affordances: tuple[WorldAffordance, ...] = ()
    uncertainty: float = 1.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if int(self.tick) < 0:
            raise ValueError("world tick cannot be negative")
        if self.latent.ndim != 1:
            raise ValueError("world latent must be a vector")
        _check_unit(self.uncertainty, "uncertainty")
        object_ids = tuple(item.object_id for item in self.objects)
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("world objects must have unique object_id values")
        if self.entities and object_ids and set(self.entities) != set(object_ids):
            raise ValueError("world entities and objects must describe the same ids")
        if object_ids and not self.entities:
            object.__setattr__(self, "entities", object_ids)
        event_ids = tuple(item.event_id for item in self.events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("world events must have unique event_id values")
        affordance_ids = tuple(item.affordance_id for item in self.affordances)
        if len(set(affordance_ids)) != len(affordance_ids):
            raise ValueError("world affordances must have unique affordance_id values")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "latent": self.latent.detach().cpu().clone(),
            "entities": list(self.entities),
            "relations": [list(item) for item in self.relations],
            "objects": [item.to_payload() for item in self.objects],
            "events": [item.to_payload() for item in self.events],
            "affordances": [item.to_payload() for item in self.affordances],
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            latent=payload["latent"].detach().to(device).clone(),
            entities=tuple(str(item) for item in payload.get("entities", ())),
            relations=tuple(
                (str(item[0]), str(item[1]), str(item[2])) for item in payload.get("relations", ())
            ),
            objects=tuple(
                WorldObject.from_payload(item, device=device)
                for item in payload.get("objects", ())
            ),
            events=tuple(
                WorldEvent.from_payload(item, device=device)
                for item in payload.get("events", ())
            ),
            affordances=tuple(
                WorldAffordance.from_payload(item, device=device)
                for item in payload.get("affordances", ())
            ),
            uncertainty=float(payload.get("uncertainty", 1.0)),
        )


@dataclass(frozen=True)
class MemoryState:
    """Cross-system memory summary owned by Taiji, not a Seed cache."""

    tick: int
    episodic_confidence: float = 0.0
    semantic_context: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    procedural_context: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if int(self.tick) < 0:
            raise ValueError("memory tick cannot be negative")
        _check_unit(self.episodic_confidence, "episodic_confidence")
        if self.semantic_context.ndim != 1 or self.procedural_context.ndim != 1:
            raise ValueError("memory contexts must be vectors")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "episodic_confidence": self.episodic_confidence,
            "semantic_context": self.semantic_context.detach().cpu().clone(),
            "procedural_context": self.procedural_context.detach().cpu().clone(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> MemoryState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            episodic_confidence=float(payload.get("episodic_confidence", 0.0)),
            semantic_context=payload["semantic_context"].detach().to(device).clone(),
            procedural_context=payload["procedural_context"].detach().to(device).clone(),
        )


@dataclass(frozen=True)
class Goal:
    goal_id: str
    description: str
    priority: float = 0.0
    progress: float = 0.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.goal_id, "goal_id")
        _check_text(self.description, "description")
        _check_unit(self.priority, "priority")
        _check_unit(self.progress, "progress")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "progress": self.progress,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Goal:
        return cls(
            version=int(payload["version"]),
            goal_id=str(payload["goal_id"]),
            description=str(payload["description"]),
            priority=float(payload.get("priority", 0.0)),
            progress=float(payload.get("progress", 0.0)),
        )


@dataclass(frozen=True)
class GoalState:
    tick: int
    goals: tuple[Goal, ...] = ()
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if int(self.tick) < 0:
            raise ValueError("goal tick cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "goals": [goal.to_payload() for goal in self.goals],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GoalState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            goals=tuple(Goal.from_payload(item) for item in payload.get("goals", ())),
        )


@dataclass(frozen=True)
class PlanCandidate:
    plan_id: str
    action_kind: str
    expected_value: float = 0.0
    risk: float = 0.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.plan_id, "plan_id")
        _check_text(self.action_kind, "action_kind")
        if not math.isfinite(float(self.expected_value)):
            raise ValueError("expected_value must be finite")
        _check_unit(self.risk, "risk")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "plan_id": self.plan_id,
            "action_kind": self.action_kind,
            "expected_value": self.expected_value,
            "risk": self.risk,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlanCandidate:
        return cls(
            version=int(payload["version"]),
            plan_id=str(payload["plan_id"]),
            action_kind=str(payload["action_kind"]),
            expected_value=float(payload.get("expected_value", 0.0)),
            risk=float(payload.get("risk", 0.0)),
        )


@dataclass(frozen=True)
class PlanState:
    tick: int
    candidates: tuple[PlanCandidate, ...] = ()
    selected_plan_id: str | None = None
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if int(self.tick) < 0:
            raise ValueError("plan tick cannot be negative")
        ids = {candidate.plan_id for candidate in self.candidates}
        if self.selected_plan_id is not None and self.selected_plan_id not in ids:
            raise ValueError("selected plan must be one of the candidates")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "selected_plan_id": self.selected_plan_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlanState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            candidates=tuple(
                PlanCandidate.from_payload(item) for item in payload.get("candidates", ())
            ),
            selected_plan_id=payload.get("selected_plan_id"),
        )


@dataclass(frozen=True)
class WorldAction:
    """A concrete intervention against a world object or relation."""

    action_id: str
    kind: str
    tick: int
    actor_id: str = ""
    target_id: str = ""
    parameters: tuple[tuple[str, Any], ...] = ()
    provenance: str = "experienced"
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.action_id, "action_id")
        _check_text(self.kind, "action kind")
        _check_text(self.provenance, "action provenance")
        if int(self.tick) < 0:
            raise ValueError("action tick cannot be negative")
        if self.actor_id:
            _check_text(self.actor_id, "action actor_id")
        if self.target_id:
            _check_text(self.target_id, "action target_id")
        object.__setattr__(self, "parameters", _normalize_pairs(self.parameters, "action parameters"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "action_id": self.action_id,
            "kind": self.kind,
            "tick": self.tick,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "parameters": _encode_value(dict(self.parameters)),
            "provenance": self.provenance,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldAction:
        parameters = _decode_value(payload.get("parameters", {}), device=device)
        return cls(
            version=int(payload["version"]),
            action_id=str(payload["action_id"]),
            kind=str(payload["kind"]),
            tick=int(payload["tick"]),
            actor_id=str(payload.get("actor_id", "")),
            target_id=str(payload.get("target_id", "")),
            parameters=parameters,
            provenance=str(payload.get("provenance", "experienced")),
        )


@dataclass(frozen=True)
class ActionIntent:
    """A Taiji decision before an organ encodes it for an environment."""

    intent_id: str
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    source_goal_id: str | None = None
    expected_outcome: str = ""
    confidence: float = 0.0
    tick: int = 0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.intent_id, "intent_id")
        _check_text(self.kind, "kind")
        if self.source_goal_id is not None:
            _check_text(self.source_goal_id, "source_goal_id")
        if int(self.tick) < 0:
            raise ValueError("intent tick cannot be negative")
        _check_unit(self.confidence, "confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "intent_id": self.intent_id,
            "kind": self.kind,
            "parameters": _encode_value(self.parameters),
            "source_goal_id": self.source_goal_id,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "tick": self.tick,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> ActionIntent:
        parameters = _decode_value(payload.get("parameters", {}), device=device)
        if not isinstance(parameters, Mapping):
            raise ValueError("action parameters must be a mapping")
        return cls(
            version=int(payload["version"]),
            intent_id=str(payload["intent_id"]),
            kind=str(payload["kind"]),
            parameters=parameters,
            source_goal_id=payload.get("source_goal_id"),
            expected_outcome=str(payload.get("expected_outcome", "")),
            confidence=float(payload.get("confidence", 0.0)),
            tick=int(payload.get("tick", 0)),
        )


@dataclass(frozen=True)
class Outcome:
    """An environment consequence bound back to a Taiji action intent."""

    intent_id: str
    reward: float
    success: bool | None = None
    terminal: bool = False
    observation: Observation | None = None
    provenance: str = "experienced"
    tick: int = 0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.intent_id, "intent_id")
        if not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        _check_text(self.provenance, "provenance")
        if int(self.tick) < 0:
            raise ValueError("outcome tick cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "intent_id": self.intent_id,
            "reward": self.reward,
            "success": self.success,
            "terminal": self.terminal,
            "observation": None if self.observation is None else self.observation.to_payload(),
            "provenance": self.provenance,
            "tick": self.tick,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> Outcome:
        observation = payload.get("observation")
        return cls(
            version=int(payload["version"]),
            intent_id=str(payload["intent_id"]),
            reward=float(payload["reward"]),
            success=None if payload.get("success") is None else bool(payload["success"]),
            terminal=bool(payload.get("terminal", False)),
            observation=(
                None
                if observation is None
                else Observation.from_payload(observation, device=device)
            ),
            provenance=str(payload.get("provenance", "experienced")),
            tick=int(payload.get("tick", 0)),
        )


@dataclass(frozen=True)
class WorldTransition:
    """A single causal step joining an action, outcome and next world state."""

    before: WorldState
    action: WorldAction
    after: WorldState
    outcome: Outcome
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if self.action.tick != self.before.tick:
            raise ValueError("world action must act at the before-state tick")
        if self.after.tick <= self.before.tick:
            raise ValueError("world transition must advance the world tick")
        if self.outcome.intent_id != self.action.action_id:
            raise ValueError("world outcome must reference the world action")
        if self.outcome.tick != self.after.tick:
            raise ValueError("world outcome must be recorded at the after-state tick")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "before": self.before.to_payload(),
            "action": self.action.to_payload(),
            "after": self.after.to_payload(),
            "outcome": self.outcome.to_payload(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldTransition:
        return cls(
            version=int(payload["version"]),
            before=WorldState.from_payload(payload["before"], device=device),
            action=WorldAction.from_payload(payload["action"], device=device),
            after=WorldState.from_payload(payload["after"], device=device),
            outcome=Outcome.from_payload(payload["outcome"], device=device),
        )


@dataclass(frozen=True)
class WorldInterventionCase:
    """One registered causal intervention with hidden expected consequence."""

    case_id: str
    initial: WorldState
    action: WorldAction
    expected_state: WorldState
    expected_outcome: Outcome
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.case_id, "intervention case_id")
        if self.action.tick != self.initial.tick:
            raise ValueError("intervention action must start at the initial tick")
        if self.expected_state.tick <= self.initial.tick:
            raise ValueError("intervention expected state must advance the initial tick")
        if self.expected_outcome.intent_id != self.action.action_id:
            raise ValueError("intervention outcome must reference the action")
        if self.expected_outcome.tick != self.expected_state.tick:
            raise ValueError("intervention outcome must match the expected state tick")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "case_id": self.case_id,
            "initial": self.initial.to_payload(),
            "action": self.action.to_payload(),
            "expected_state": self.expected_state.to_payload(),
            "expected_outcome": self.expected_outcome.to_payload(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldInterventionCase:
        return cls(
            version=int(payload["version"]),
            case_id=str(payload["case_id"]),
            initial=WorldState.from_payload(payload["initial"], device=device),
            action=WorldAction.from_payload(payload["action"], device=device),
            expected_state=WorldState.from_payload(payload["expected_state"], device=device),
            expected_outcome=Outcome.from_payload(payload["expected_outcome"], device=device),
        )


@dataclass(frozen=True)
class WorldInterventionCorpus:
    """Train/holdout intervention cases with disjoint registered identities."""

    train: tuple[WorldInterventionCase, ...] = ()
    holdout: tuple[WorldInterventionCase, ...] = ()
    time_shuffled: tuple[WorldInterventionCase, ...] = ()
    format: str = "taiji-world-intervention-v1"
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if self.format != "taiji-world-intervention-v1":
            raise ValueError(f"unsupported intervention corpus format: {self.format}")
        train_ids = {case.case_id for case in self.train}
        holdout_ids = {case.case_id for case in self.holdout}
        shuffled_ids = {case.case_id for case in self.time_shuffled}
        if (
            len(train_ids) != len(self.train)
            or len(holdout_ids) != len(self.holdout)
            or len(shuffled_ids) != len(self.time_shuffled)
        ):
            raise ValueError("intervention case ids must be unique within each split")
        if (train_ids & holdout_ids) or (train_ids & shuffled_ids) or (holdout_ids & shuffled_ids):
            raise ValueError("intervention splits must be disjoint")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "train": [case.to_payload() for case in self.train],
            "holdout": [case.to_payload() for case in self.holdout],
            "time_shuffled": [case.to_payload() for case in self.time_shuffled],
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldInterventionCorpus:
        return cls(
            format=str(payload["format"]),
            version=int(payload["version"]),
            train=tuple(
                WorldInterventionCase.from_payload(item, device=device)
                for item in payload.get("train", ())
            ),
            holdout=tuple(
                WorldInterventionCase.from_payload(item, device=device)
                for item in payload.get("holdout", ())
            ),
            time_shuffled=tuple(
                WorldInterventionCase.from_payload(item, device=device)
                for item in payload.get("time_shuffled", ())
            ),
        )


@dataclass(frozen=True)
class SelfState:
    tick: int
    confidence: float = 0.0
    resource_fraction: float = 1.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_unit(self.confidence, "self confidence")
        _check_unit(self.resource_fraction, "resource_fraction")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "confidence": self.confidence,
            "resource_fraction": self.resource_fraction,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SelfState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            confidence=float(payload.get("confidence", 0.0)),
            resource_fraction=float(payload.get("resource_fraction", 1.0)),
        )


@dataclass(frozen=True)
class HomeostaticState:
    tick: int
    curiosity: float = 0.0
    fatigue: float = 0.0
    stress: float = 0.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_unit(self.curiosity, "curiosity")
        _check_unit(self.fatigue, "fatigue")
        _check_unit(self.stress, "stress")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "curiosity": self.curiosity,
            "fatigue": self.fatigue,
            "stress": self.stress,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HomeostaticState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            curiosity=float(payload.get("curiosity", 0.0)),
            fatigue=float(payload.get("fatigue", 0.0)),
            stress=float(payload.get("stress", 0.0)),
        )


@dataclass(frozen=True)
class DevelopmentState:
    tick: int
    stage: str = "bootstrap"
    structural_budget: int = 0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.stage, "development stage")
        if int(self.tick) < 0 or int(self.structural_budget) < 0:
            raise ValueError("development counters cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "stage": self.stage,
            "structural_budget": self.structural_budget,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DevelopmentState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            stage=str(payload.get("stage", "bootstrap")),
            structural_budget=int(payload.get("structural_budget", 0)),
        )


@dataclass(frozen=True)
class LearningState:
    tick: int
    local_updates: int = 0
    lifetime_updates: int = 0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if min(int(self.tick), int(self.local_updates), int(self.lifetime_updates)) < 0:
            raise ValueError("learning counters cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "local_updates": self.local_updates,
            "lifetime_updates": self.lifetime_updates,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LearningState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            local_updates=int(payload.get("local_updates", 0)),
            lifetime_updates=int(payload.get("lifetime_updates", 0)),
        )


@dataclass(frozen=True)
class CognitiveState:
    """Top-level Taiji state shared by all future cognitive subsystems."""

    episode_id: str
    tick: int
    observation: Observation | None
    percept: PerceptEvent | None
    workspace: WorkspaceState
    world: WorldState
    memory: MemoryState
    goals: GoalState
    plan: PlanState
    self_state: SelfState
    homeostasis: HomeostaticState
    development: DevelopmentState
    learning: LearningState
    action_intent: ActionIntent | None = None
    outcome: Outcome | None = None
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.episode_id, "episode_id")
        if int(self.tick) < 0:
            raise ValueError("cognitive tick cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "tick": self.tick,
            "observation": None if self.observation is None else self.observation.to_payload(),
            "percept": None if self.percept is None else self.percept.to_payload(),
            "workspace": self.workspace.to_payload(),
            "world": self.world.to_payload(),
            "memory": self.memory.to_payload(),
            "goals": self.goals.to_payload(),
            "plan": self.plan.to_payload(),
            "self_state": self.self_state.to_payload(),
            "homeostasis": self.homeostasis.to_payload(),
            "development": self.development.to_payload(),
            "learning": self.learning.to_payload(),
            "action_intent": (
                None if self.action_intent is None else self.action_intent.to_payload()
            ),
            "outcome": None if self.outcome is None else self.outcome.to_payload(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> CognitiveState:
        observation = payload.get("observation")
        percept = payload.get("percept")
        intent = payload.get("action_intent")
        outcome = payload.get("outcome")
        return cls(
            version=int(payload["version"]),
            episode_id=str(payload["episode_id"]),
            tick=int(payload["tick"]),
            observation=(
                None
                if observation is None
                else Observation.from_payload(observation, device=device)
            ),
            percept=(
                None if percept is None else PerceptEvent.from_payload(percept, device=device)
            ),
            workspace=WorkspaceState.from_payload(payload["workspace"], device=device),
            world=WorldState.from_payload(payload["world"], device=device),
            memory=MemoryState.from_payload(payload["memory"], device=device),
            goals=GoalState.from_payload(payload["goals"]),
            plan=PlanState.from_payload(payload["plan"]),
            self_state=SelfState.from_payload(payload["self_state"]),
            homeostasis=HomeostaticState.from_payload(payload["homeostasis"]),
            development=DevelopmentState.from_payload(payload["development"]),
            learning=LearningState.from_payload(payload["learning"]),
            action_intent=(
                None if intent is None else ActionIntent.from_payload(intent, device=device)
            ),
            outcome=None if outcome is None else Outcome.from_payload(outcome, device=device),
        )


@dataclass(frozen=True)
class NativeCheckpoint:
    """Atomic Taiji envelope around a kernel checkpoint."""

    kernel: Mapping[str, Any]
    cognitive_state: CognitiveState
    components: Mapping[str, Any] = field(default_factory=dict)
    adapter: str = "tsk-v8"
    format: str = CONTRACT_FORMAT
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if self.format != CONTRACT_FORMAT:
            raise ValueError(f"unsupported Taiji checkpoint format: {self.format}")
        _check_text(self.adapter, "adapter")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "adapter": self.adapter,
            "kernel": dict(self.kernel),
            "cognitive_state": self.cognitive_state.to_payload(),
            "components": dict(self.components),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> NativeCheckpoint:
        return cls(
            format=str(payload["format"]),
            version=int(payload["version"]),
            adapter=str(payload["adapter"]),
            kernel=payload["kernel"],
            cognitive_state=CognitiveState.from_payload(payload["cognitive_state"], device=device),
            components=payload.get("components", {}),
        )

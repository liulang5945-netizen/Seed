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
    boundary: bool = False
    confidence: float = 1.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.event_id, "event_id")
        _check_text(self.modality, "modality")
        if int(self.observation_tick) < 0:
            raise ValueError("observation_tick cannot be negative")
        if self.features.ndim != 1:
            raise ValueError("percept features must be a vector")
        _check_unit(self.confidence, "confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "observation_tick": self.observation_tick,
            "modality": self.modality,
            "features": self.features.detach().cpu().clone(),
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
class WorldState:
    """Persistent latent world state and explicit relation handles."""

    tick: int
    latent: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    entities: tuple[str, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    uncertainty: float = 1.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if int(self.tick) < 0:
            raise ValueError("world tick cannot be negative")
        if self.latent.ndim != 1:
            raise ValueError("world latent must be a vector")
        _check_unit(self.uncertainty, "uncertainty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "latent": self.latent.detach().cpu().clone(),
            "entities": list(self.entities),
            "relations": [list(item) for item in self.relations],
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
        )

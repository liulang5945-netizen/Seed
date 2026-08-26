"""Versioned contracts owned by the Taiji cognitive architecture.

The contracts in this module are deliberately small and serializable.  They
are the boundary between organs, cognition and effectors; they are not a
second model hidden in Seed and they do not prescribe the representation used
inside a future Taiji implementation.
"""

from __future__ import annotations

import hashlib
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


def _normalize_ids(value: Any, name: str) -> tuple[str, ...]:
    ids = tuple(_check_text(str(item), name) for item in value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} cannot contain duplicate values")
    return ids


def _normalize_sequences(value: Any, name: str) -> tuple[tuple[str, ...], ...]:
    sequences: list[tuple[str, ...]] = []
    for sequence in value:
        normalized = tuple(_check_text(str(item), f"{name} item") for item in sequence)
        if not normalized:
            raise ValueError(f"{name} cannot contain empty sequences")
        if normalized in sequences:
            raise ValueError(f"{name} cannot contain duplicate sequences")
        sequences.append(normalized)
    return tuple(sequences)


def _normalize_text_sequence(value: Any, name: str) -> tuple[str, ...]:
    sequence = tuple(_check_text(str(item), f"{name} item") for item in value)
    if not sequence:
        raise ValueError(f"{name} cannot be empty")
    return sequence


def _normalize_unit_pairs(value: Any, name: str) -> tuple[tuple[str, float], ...]:
    pairs = _normalize_pairs(value, name)
    normalized: list[tuple[str, float]] = []
    for key, item in pairs:
        normalized.append((key, _check_unit(float(item), f"{name}[{key}]")))
    return tuple(normalized)


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
class Assembly:
    """A time-bounded, causally testable coalition of neural activity."""

    assembly_id: str
    start_tick: int
    end_tick: int
    activity: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    member_indices: tuple[int, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    coherence: float = 0.0
    prediction_error: float = 0.0
    route_score: float = 0.0
    provenance: str = "learned"
    confidence: float = 0.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.assembly_id, "assembly_id")
        _check_text(self.provenance, "assembly provenance")
        if int(self.start_tick) < 0 or int(self.end_tick) < int(self.start_tick):
            raise ValueError("assembly ticks must be non-negative and ordered")
        if self.activity.ndim != 1:
            raise ValueError("assembly activity must be a vector")
        members = tuple(int(index) for index in self.member_indices)
        if any(index < 0 for index in members):
            raise ValueError("assembly member indices cannot be negative")
        if len(set(members)) != len(members):
            raise ValueError("assembly member indices cannot contain duplicates")
        object.__setattr__(self, "member_indices", members)
        object.__setattr__(
            self, "source_event_ids", _normalize_ids(self.source_event_ids, "assembly source_event_ids")
        )
        _check_unit(self.coherence, "assembly coherence")
        _check_unit(self.prediction_error, "assembly prediction_error")
        _check_unit(self.route_score, "assembly route_score")
        _check_unit(self.confidence, "assembly confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "assembly_id": self.assembly_id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "activity": self.activity.detach().cpu().clone(),
            "member_indices": list(self.member_indices),
            "source_event_ids": list(self.source_event_ids),
            "coherence": self.coherence,
            "prediction_error": self.prediction_error,
            "route_score": self.route_score,
            "provenance": self.provenance,
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> Assembly:
        return cls(
            version=int(payload["version"]),
            assembly_id=str(payload["assembly_id"]),
            start_tick=int(payload["start_tick"]),
            end_tick=int(payload["end_tick"]),
            activity=payload["activity"].detach().to(device).clone(),
            member_indices=tuple(int(index) for index in payload.get("member_indices", ())),
            source_event_ids=tuple(str(item) for item in payload.get("source_event_ids", ())),
            coherence=float(payload.get("coherence", 0.0)),
            prediction_error=float(payload.get("prediction_error", 0.0)),
            route_score=float(payload.get("route_score", 0.0)),
            provenance=str(payload.get("provenance", "learned")),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class Event:
    """A learned temporal structure composed from perceptual assemblies."""

    event_id: str
    start_tick: int
    end_tick: int
    latent: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    assembly_ids: tuple[str, ...] = ()
    parent_event_ids: tuple[str, ...] = ()
    object_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    prediction_error: float = 0.0
    confidence: float = 0.0
    provenance: str = "learned"
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.event_id, "event_id")
        _check_text(self.provenance, "event provenance")
        if int(self.start_tick) < 0 or int(self.end_tick) < int(self.start_tick):
            raise ValueError("event ticks must be non-negative and ordered")
        if self.latent.ndim != 1:
            raise ValueError("event latent must be a vector")
        for field_name in ("assembly_ids", "parent_event_ids", "object_ids", "relation_ids"):
            object.__setattr__(
                self,
                field_name,
                _normalize_ids(getattr(self, field_name), f"event {field_name}"),
            )
        _check_unit(self.prediction_error, "event prediction_error")
        _check_unit(self.confidence, "event confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "latent": self.latent.detach().cpu().clone(),
            "assembly_ids": list(self.assembly_ids),
            "parent_event_ids": list(self.parent_event_ids),
            "object_ids": list(self.object_ids),
            "relation_ids": list(self.relation_ids),
            "prediction_error": self.prediction_error,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> Event:
        return cls(
            version=int(payload["version"]),
            event_id=str(payload["event_id"]),
            start_tick=int(payload["start_tick"]),
            end_tick=int(payload["end_tick"]),
            latent=payload["latent"].detach().to(device).clone(),
            assembly_ids=tuple(str(item) for item in payload.get("assembly_ids", ())),
            parent_event_ids=tuple(str(item) for item in payload.get("parent_event_ids", ())),
            object_ids=tuple(str(item) for item in payload.get("object_ids", ())),
            relation_ids=tuple(str(item) for item in payload.get("relation_ids", ())),
            prediction_error=float(payload.get("prediction_error", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            provenance=str(payload.get("provenance", "learned")),
        )


@dataclass(frozen=True)
class ConceptSequenceTrace:
    """A learned sequence trace with state-conditioned credit."""

    action_kinds: tuple[str, ...]
    before_prototype: torch.Tensor
    after_prototypes: tuple[torch.Tensor, ...]
    step_credit: tuple[float, ...]
    prediction_errors: tuple[float, ...]
    trace_id: str = ""
    outcome_mean: float = 0.0
    visits: int = 1
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        object.__setattr__(
            self,
            "action_kinds",
            _normalize_text_sequence(self.action_kinds, "sequence trace action_kinds"),
        )
        if self.before_prototype.ndim != 1:
            raise ValueError("sequence trace before_prototype must be a vector")
        after_prototypes = tuple(item.detach().clone() for item in self.after_prototypes)
        if len(after_prototypes) != len(self.action_kinds):
            raise ValueError("sequence trace after_prototypes must match action_kinds")
        if any(item.ndim != 1 for item in after_prototypes):
            raise ValueError("sequence trace after_prototypes must be vectors")
        if any(item.numel() != self.before_prototype.numel() for item in after_prototypes):
            raise ValueError("sequence trace state prototypes must share one dimension")
        object.__setattr__(self, "before_prototype", self.before_prototype.detach().clone())
        object.__setattr__(self, "after_prototypes", after_prototypes)
        trace_id = str(self.trace_id)
        if not trace_id:
            signature = repr(
                (
                    self.action_kinds,
                    tuple(round(float(value), 6) for value in self.before_prototype.tolist()),
                    tuple(
                        tuple(round(float(value), 6) for value in prototype.tolist())
                        for prototype in after_prototypes
                    ),
                )
            )
            trace_id = f"trace:{hashlib.sha256(signature.encode('utf-8')).hexdigest()[:16]}"
        object.__setattr__(self, "trace_id", _check_text(trace_id, "sequence trace trace_id"))
        if len(self.step_credit) != len(self.action_kinds):
            raise ValueError("sequence trace step_credit must match action_kinds")
        if len(self.prediction_errors) != len(self.action_kinds):
            raise ValueError("sequence trace prediction_errors must match action_kinds")
        object.__setattr__(
            self,
            "step_credit",
            tuple(_check_unit(value, "sequence trace step_credit") for value in self.step_credit),
        )
        object.__setattr__(
            self,
            "prediction_errors",
            tuple(
                _check_unit(value, "sequence trace prediction_error")
                for value in self.prediction_errors
            ),
        )
        _check_unit(self.outcome_mean, "sequence trace outcome_mean")
        if int(self.visits) <= 0:
            raise ValueError("sequence trace visits must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "trace_id": self.trace_id,
            "action_kinds": list(self.action_kinds),
            "before_prototype": self.before_prototype.detach().cpu().clone(),
            "after_prototypes": [item.detach().cpu().clone() for item in self.after_prototypes],
            "step_credit": list(self.step_credit),
            "prediction_errors": list(self.prediction_errors),
            "outcome_mean": self.outcome_mean,
            "visits": self.visits,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> ConceptSequenceTrace:
        return cls(
            version=int(payload["version"]),
            trace_id=str(payload.get("trace_id", "")),
            action_kinds=tuple(str(item) for item in payload["action_kinds"]),
            before_prototype=payload["before_prototype"].detach().to(device).clone(),
            after_prototypes=tuple(
                item.detach().to(device).clone() for item in payload["after_prototypes"]
            ),
            step_credit=tuple(float(item) for item in payload["step_credit"]),
            prediction_errors=tuple(float(item) for item in payload["prediction_errors"]),
            outcome_mean=float(payload.get("outcome_mean", 0.0)),
            visits=int(payload.get("visits", 1)),
        )


@dataclass(frozen=True)
class Concept:
    """A cross-experience invariant, not a label or an episode copy."""

    concept_id: str
    prototype: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    support_event_ids: tuple[str, ...] = ()
    support_assembly_ids: tuple[str, ...] = ()
    object_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    action_kinds: tuple[str, ...] = ()
    action_sequences: tuple[tuple[str, ...], ...] = ()
    sequence_traces: tuple[ConceptSequenceTrace, ...] = ()
    sequence_traces_lesioned: bool = False
    maturity: float = 0.0
    stability: float = 0.0
    confidence: float = 0.0
    outcome_mean: float = 0.0
    outcome_consistency: float = 0.0
    update_count: int = 0
    last_updated_tick: int = 0
    provenance: str = "consolidated"
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.concept_id, "concept_id")
        _check_text(self.provenance, "concept provenance")
        if self.prototype.ndim != 1:
            raise ValueError("concept prototype must be a vector")
        for field_name in (
            "support_event_ids",
            "support_assembly_ids",
            "object_ids",
            "relation_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_ids(getattr(self, field_name), f"concept {field_name}"),
            )
        object.__setattr__(
            self, "action_kinds", _normalize_ids(self.action_kinds, "concept action_kinds")
        )
        object.__setattr__(
            self,
            "action_sequences",
            _normalize_sequences(self.action_sequences, "concept action_sequences"),
        )
        if any(not isinstance(item, ConceptSequenceTrace) for item in self.sequence_traces):
            raise ValueError("concept sequence_traces must contain ConceptSequenceTrace values")
        object.__setattr__(self, "sequence_traces", tuple(self.sequence_traces))
        _check_unit(self.maturity, "concept maturity")
        _check_unit(self.stability, "concept stability")
        _check_unit(self.confidence, "concept confidence")
        _check_unit(self.outcome_mean, "concept outcome_mean")
        _check_unit(self.outcome_consistency, "concept outcome_consistency")
        if int(self.update_count) < 0 or int(self.last_updated_tick) < 0:
            raise ValueError("concept update counters cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "concept_id": self.concept_id,
            "prototype": self.prototype.detach().cpu().clone(),
            "support_event_ids": list(self.support_event_ids),
            "support_assembly_ids": list(self.support_assembly_ids),
            "object_ids": list(self.object_ids),
            "relation_ids": list(self.relation_ids),
            "action_kinds": list(self.action_kinds),
            "action_sequences": [list(sequence) for sequence in self.action_sequences],
            "sequence_traces": [item.to_payload() for item in self.sequence_traces],
            "sequence_traces_lesioned": self.sequence_traces_lesioned,
            "maturity": self.maturity,
            "stability": self.stability,
            "confidence": self.confidence,
            "outcome_mean": self.outcome_mean,
            "outcome_consistency": self.outcome_consistency,
            "update_count": self.update_count,
            "last_updated_tick": self.last_updated_tick,
            "provenance": self.provenance,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> Concept:
        return cls(
            version=int(payload["version"]),
            concept_id=str(payload["concept_id"]),
            prototype=payload["prototype"].detach().to(device).clone(),
            support_event_ids=tuple(str(item) for item in payload.get("support_event_ids", ())),
            support_assembly_ids=tuple(
                str(item) for item in payload.get("support_assembly_ids", ())
            ),
            object_ids=tuple(str(item) for item in payload.get("object_ids", ())),
            relation_ids=tuple(str(item) for item in payload.get("relation_ids", ())),
            action_kinds=tuple(str(item) for item in payload.get("action_kinds", ())),
            action_sequences=tuple(
                tuple(str(action) for action in sequence)
                for sequence in payload.get("action_sequences", ())
            ),
            sequence_traces=tuple(
                ConceptSequenceTrace.from_payload(item, device=device)
                for item in payload.get("sequence_traces", ())
            ),
            sequence_traces_lesioned=bool(payload.get("sequence_traces_lesioned", False)),
            maturity=float(payload.get("maturity", 0.0)),
            stability=float(payload.get("stability", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            outcome_mean=float(payload.get("outcome_mean", 0.0)),
            outcome_consistency=float(payload.get("outcome_consistency", 0.0)),
            update_count=int(payload.get("update_count", 0)),
            last_updated_tick=int(payload.get("last_updated_tick", 0)),
            provenance=str(payload.get("provenance", "consolidated")),
        )


@dataclass(frozen=True)
class WorkspaceCandidate:
    """A candidate that may enter the shared workspace."""

    candidate_id: str
    features: torch.Tensor
    salience: float = 0.0
    source: str = "external"
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.candidate_id, "workspace candidate_id")
        _check_text(self.source, "workspace candidate source")
        _check_unit(self.salience, "workspace candidate salience")
        if self.features.ndim != 1:
            raise ValueError("workspace candidate features must be a vector")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "candidate_id": self.candidate_id,
            "features": self.features.detach().cpu().clone(),
            "salience": self.salience,
            "source": self.source,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorkspaceCandidate:
        return cls(
            version=int(payload["version"]),
            candidate_id=str(payload["candidate_id"]),
            features=payload["features"].detach().to(device).clone(),
            salience=float(payload.get("salience", 0.0)),
            source=str(payload.get("source", "external")),
        )


@dataclass(frozen=True)
class WorkspaceSelection:
    """The auditable result of routing candidates through the workspace."""

    tick: int
    mode: str
    candidate_ids: tuple[str, ...] = ()
    selected_ids: tuple[str, ...] = ()
    scores: tuple[float, ...] = ()
    broadcast: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    capacity: int = 0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if int(self.tick) < 0:
            raise ValueError("workspace selection tick cannot be negative")
        if self.mode not in {"learned", "none", "random"}:
            raise ValueError(f"unsupported workspace selection mode: {self.mode}")
        if int(self.capacity) < 0:
            raise ValueError("workspace selection capacity cannot be negative")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("workspace candidate ids must be unique")
        if len(self.scores) != len(self.candidate_ids):
            raise ValueError("workspace scores must align with candidate ids")
        if not set(self.selected_ids).issubset(self.candidate_ids):
            raise ValueError("workspace selection contains an unknown candidate")
        if self.capacity != 0 and len(self.selected_ids) > self.capacity:
            raise ValueError("workspace selection exceeds capacity")
        if any(not math.isfinite(float(score)) for score in self.scores):
            raise ValueError("workspace scores must be finite")
        if self.broadcast.ndim != 1:
            raise ValueError("workspace selection broadcast must be a vector")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "mode": self.mode,
            "candidate_ids": list(self.candidate_ids),
            "selected_ids": list(self.selected_ids),
            "scores": list(self.scores),
            "broadcast": self.broadcast.detach().cpu().clone(),
            "capacity": self.capacity,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorkspaceSelection:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            mode=str(payload.get("mode", "learned")),
            candidate_ids=tuple(str(item) for item in payload.get("candidate_ids", ())),
            selected_ids=tuple(str(item) for item in payload.get("selected_ids", ())),
            scores=tuple(float(item) for item in payload.get("scores", ())),
            broadcast=payload["broadcast"].detach().to(device).clone(),
            capacity=int(payload.get("capacity", 0)),
        )


@dataclass(frozen=True)
class WorkspaceState:
    """Capacity-limited focus and broadcast state."""

    tick: int
    focus: tuple[str, ...] = ()
    bindings: tuple[tuple[str, str], ...] = ()
    broadcast: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    capacity: int = 0
    candidates: tuple[WorkspaceCandidate, ...] = ()
    selection: WorkspaceSelection | None = None
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
        if self.selection is not None:
            if self.selection.tick != self.tick:
                raise ValueError("workspace selection tick must match workspace tick")
            candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
            if candidate_ids != self.selection.candidate_ids:
                raise ValueError("workspace selection must align with workspace candidates")
            if tuple(self.focus) != self.selection.selected_ids:
                raise ValueError("workspace focus must match workspace selection")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "focus": list(self.focus),
            "bindings": [list(pair) for pair in self.bindings],
            "broadcast": self.broadcast.detach().cpu().clone(),
            "capacity": self.capacity,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "selection": None if self.selection is None else self.selection.to_payload(),
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
            candidates=tuple(
                WorkspaceCandidate.from_payload(item, device=device)
                for item in payload.get("candidates", ())
            ),
            selection=(
                None
                if payload.get("selection") is None
                else WorkspaceSelection.from_payload(payload["selection"], device=device)
            ),
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
    features: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    feature_provenance: str = "world-organ"
    grounding_lineage: tuple[str, ...] = ()
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
        if self.features.ndim != 1:
            raise ValueError("affordance features must be a vector")
        if self.features.numel() and not bool(torch.isfinite(self.features).all()):
            raise ValueError("affordance features must be finite")
        _check_text(self.feature_provenance, "affordance feature_provenance")
        object.__setattr__(self, "parameters", _normalize_pairs(self.parameters, "affordance parameters"))
        object.__setattr__(
            self,
            "grounding_lineage",
            _normalize_tags(self.grounding_lineage, "affordance grounding_lineage"),
        )
        object.__setattr__(self, "features", self.features.detach().clone())

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "affordance_id": self.affordance_id,
            "action_kind": self.action_kind,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "parameters": _encode_value(dict(self.parameters)),
            "confidence": self.confidence,
            "features": self.features.detach().cpu().clone(),
            "feature_provenance": self.feature_provenance,
            "grounding_lineage": list(self.grounding_lineage),
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
            features=payload.get("features", torch.empty(0)).detach().to(device).clone(),
            feature_provenance=str(payload.get("feature_provenance", "world-organ")),
            grounding_lineage=tuple(
                str(item) for item in payload.get("grounding_lineage", ())
            ),
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
                WorldObject.from_payload(item, device=device) for item in payload.get("objects", ())
            ),
            events=tuple(
                WorldEvent.from_payload(item, device=device) for item in payload.get("events", ())
            ),
            affordances=tuple(
                WorldAffordance.from_payload(item, device=device)
                for item in payload.get("affordances", ())
            ),
            uncertainty=float(payload.get("uncertainty", 1.0)),
        )


@dataclass(frozen=True)
class WorkingMemoryItem:
    """One capacity-governed item held in the current working context."""

    item_id: str
    value: Any
    salience: float = 0.0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.item_id, "working memory item_id")
        _check_unit(self.salience, "working memory salience")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "item_id": self.item_id,
            "value": _encode_value(self.value),
            "salience": self.salience,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorkingMemoryItem:
        return cls(
            version=int(payload["version"]),
            item_id=str(payload["item_id"]),
            value=_decode_value(payload.get("value"), device=device),
            salience=float(payload.get("salience", 0.0)),
        )


@dataclass(frozen=True)
class MemoryState:
    """Cross-system memory summary owned by Taiji, not a Seed cache."""

    tick: int
    episodic_confidence: float = 0.0
    semantic_context: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    procedural_context: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    working_ids: tuple[str, ...] = ()
    episodic_ids: tuple[str, ...] = ()
    concept_ids: tuple[str, ...] = ()
    concept_confidence: float = 0.0
    working_items: tuple[WorkingMemoryItem, ...] = ()
    working_capacity: int = 4
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if int(self.tick) < 0:
            raise ValueError("memory tick cannot be negative")
        _check_unit(self.episodic_confidence, "episodic_confidence")
        if self.semantic_context.ndim != 1 or self.procedural_context.ndim != 1:
            raise ValueError("memory contexts must be vectors")
        if any(not str(item) for item in (*self.working_ids, *self.episodic_ids)):
            raise ValueError("memory ids cannot be empty")
        if any(not str(item) for item in self.concept_ids):
            raise ValueError("memory concept ids cannot be empty")
        _check_unit(self.concept_confidence, "concept confidence")
        if int(self.working_capacity) <= 0:
            raise ValueError("working memory capacity must be positive")
        if len(self.working_items) > int(self.working_capacity):
            raise ValueError("working memory items exceed capacity")
        if len({item.item_id for item in self.working_items}) != len(self.working_items):
            raise ValueError("working memory item ids must be unique")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "episodic_confidence": self.episodic_confidence,
            "semantic_context": self.semantic_context.detach().cpu().clone(),
            "procedural_context": self.procedural_context.detach().cpu().clone(),
            "working_ids": list(self.working_ids),
            "episodic_ids": list(self.episodic_ids),
            "concept_ids": list(self.concept_ids),
            "concept_confidence": self.concept_confidence,
            "working_items": [item.to_payload() for item in self.working_items],
            "working_capacity": self.working_capacity,
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
            working_ids=tuple(str(item) for item in payload.get("working_ids", ())),
            episodic_ids=tuple(str(item) for item in payload.get("episodic_ids", ())),
            concept_ids=tuple(str(item) for item in payload.get("concept_ids", ())),
            concept_confidence=float(payload.get("concept_confidence", 0.0)),
            working_items=tuple(
                WorkingMemoryItem.from_payload(item, device=device)
                for item in payload.get("working_items", ())
            ),
            working_capacity=int(payload.get("working_capacity", 4)),
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
        object.__setattr__(
            self, "parameters", _normalize_pairs(self.parameters, "action parameters")
        )

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
class WorldPredictionRecord:
    """A runtime prediction that is later scored against a real transition."""

    action: WorldAction
    predicted_state: WorldState
    predicted_reward: float
    predicted_success_probability: float
    state_error: float | None = None
    raw_state_error: float | None = None
    reward_error: float | None = None
    online_update_count: int = 0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if not math.isfinite(float(self.predicted_reward)):
            raise ValueError("predicted_reward must be finite")
        _check_unit(self.predicted_success_probability, "predicted_success_probability")
        if self.state_error is not None and (
            not math.isfinite(float(self.state_error)) or float(self.state_error) < 0.0
        ):
            raise ValueError("state_error must be a finite non-negative value")
        if self.raw_state_error is not None and (
            not math.isfinite(float(self.raw_state_error)) or float(self.raw_state_error) < 0.0
        ):
            raise ValueError("raw_state_error must be a finite non-negative value")
        if self.reward_error is not None and not math.isfinite(float(self.reward_error)):
            raise ValueError("reward_error must be finite")
        if int(self.online_update_count) < 0:
            raise ValueError("online_update_count cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "action": self.action.to_payload(),
            "predicted_state": self.predicted_state.to_payload(),
            "predicted_reward": self.predicted_reward,
            "predicted_success_probability": self.predicted_success_probability,
            "state_error": self.state_error,
            "raw_state_error": self.raw_state_error,
            "reward_error": self.reward_error,
            "online_update_count": self.online_update_count,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldPredictionRecord:
        return cls(
            version=int(payload["version"]),
            action=WorldAction.from_payload(payload["action"], device=device),
            predicted_state=WorldState.from_payload(payload["predicted_state"], device=device),
            predicted_reward=float(payload["predicted_reward"]),
            predicted_success_probability=float(payload["predicted_success_probability"]),
            state_error=(
                None if payload.get("state_error") is None else float(payload["state_error"])
            ),
            raw_state_error=(
                None
                if payload.get("raw_state_error") is None
                else float(payload["raw_state_error"])
            ),
            reward_error=(
                None if payload.get("reward_error") is None else float(payload["reward_error"])
            ),
            online_update_count=int(payload.get("online_update_count", 0)),
        )


@dataclass(frozen=True)
class WorldCalibrationTrace:
    """One recoverable prediction/error/update record for a real transition."""

    transition: WorldTransition
    prediction: WorldPredictionRecord
    calibration_applied: bool
    online_update_count_before: int
    online_update_count_after: int
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if self.transition.action.action_id != self.prediction.action.action_id:
            raise ValueError("calibration trace prediction must reference its transition action")
        if self.prediction.state_error is None or self.prediction.reward_error is None:
            raise ValueError("calibration trace requires scored prediction errors")
        if int(self.online_update_count_before) < 0:
            raise ValueError("online_update_count_before cannot be negative")
        if int(self.online_update_count_after) < int(self.online_update_count_before):
            raise ValueError("online update count cannot move backwards")
        if int(self.prediction.online_update_count) != int(self.online_update_count_after):
            raise ValueError("calibration trace count must match the prediction record")
        if self.calibration_applied and int(self.online_update_count_after) <= int(
            self.online_update_count_before
        ):
            raise ValueError("applied calibration must advance the online update count")
        if not self.calibration_applied and int(self.online_update_count_after) != int(
            self.online_update_count_before
        ):
            raise ValueError("disabled calibration cannot advance the online update count")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "transition": self.transition.to_payload(),
            "prediction": self.prediction.to_payload(),
            "calibration_applied": self.calibration_applied,
            "online_update_count_before": self.online_update_count_before,
            "online_update_count_after": self.online_update_count_after,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldCalibrationTrace:
        return cls(
            version=int(payload["version"]),
            transition=WorldTransition.from_payload(payload["transition"], device=device),
            prediction=WorldPredictionRecord.from_payload(payload["prediction"], device=device),
            calibration_applied=bool(payload["calibration_applied"]),
            online_update_count_before=int(payload["online_update_count_before"]),
            online_update_count_after=int(payload["online_update_count_after"]),
        )


@dataclass(frozen=True)
class PlanningRecoveryState:
    """Explicit runtime state entered after an imagined-world error."""

    mode: str
    trigger: str
    prediction_error: float
    threshold: float
    source_rollout_id: str | None = None
    remaining_rollout_steps: int = 0
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.mode, "planning recovery mode")
        _check_text(self.trigger, "planning recovery trigger")
        if not math.isfinite(float(self.prediction_error)) or float(self.prediction_error) < 0.0:
            raise ValueError("planning recovery prediction_error must be finite and non-negative")
        if not math.isfinite(float(self.threshold)) or float(self.threshold) < 0.0:
            raise ValueError("planning recovery threshold must be finite and non-negative")
        if self.source_rollout_id is not None and not str(self.source_rollout_id):
            raise ValueError("planning recovery source_rollout_id cannot be empty")
        if int(self.remaining_rollout_steps) < 0:
            raise ValueError("planning recovery remaining_rollout_steps cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "trigger": self.trigger,
            "prediction_error": self.prediction_error,
            "threshold": self.threshold,
            "source_rollout_id": self.source_rollout_id,
            "remaining_rollout_steps": self.remaining_rollout_steps,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlanningRecoveryState:
        return cls(
            version=int(payload["version"]),
            mode=str(payload["mode"]),
            trigger=str(payload["trigger"]),
            prediction_error=float(payload["prediction_error"]),
            threshold=float(payload["threshold"]),
            source_rollout_id=(
                None
                if payload.get("source_rollout_id") is None
                else str(payload["source_rollout_id"])
            ),
            remaining_rollout_steps=int(payload.get("remaining_rollout_steps", 0)),
        )


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    """One real experience owned by Taiji's episodic memory system."""

    memory_id: str
    episode_id: str
    tick: int
    cue: torch.Tensor
    action_intent: ActionIntent | None = None
    outcome: Outcome | None = None
    world_transition: WorldTransition | None = None
    prediction_error: float = 0.0
    provenance: str = "experienced"
    event_ids: tuple[str, ...] = ()
    assembly_ids: tuple[str, ...] = ()
    object_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.memory_id, "episodic memory_id")
        _check_text(self.episode_id, "episodic episode_id")
        _check_text(self.provenance, "episodic provenance")
        object.__setattr__(self, "event_ids", _normalize_ids(self.event_ids, "episodic event_ids"))
        object.__setattr__(
            self, "assembly_ids", _normalize_ids(self.assembly_ids, "episodic assembly_ids")
        )
        object.__setattr__(self, "object_ids", _normalize_ids(self.object_ids, "episodic object_ids"))
        object.__setattr__(
            self, "relation_ids", _normalize_ids(self.relation_ids, "episodic relation_ids")
        )
        if int(self.tick) < 0:
            raise ValueError("episodic memory tick cannot be negative")
        if self.cue.ndim != 1:
            raise ValueError("episodic memory cue must be a vector")
        if self.action_intent is not None and self.action_intent.tick > self.tick:
            raise ValueError("episodic action intent cannot occur after the record tick")
        if self.outcome is not None and self.outcome.tick != self.tick:
            raise ValueError("episodic outcome tick must match the record tick")
        if self.world_transition is not None:
            if self.world_transition.after.tick != self.tick:
                raise ValueError("episodic world transition tick must match the record tick")
            if self.world_transition.outcome.tick != self.tick:
                raise ValueError("episodic transition outcome tick must match the record tick")
        if not math.isfinite(float(self.prediction_error)) or float(self.prediction_error) < 0.0:
            raise ValueError("episodic prediction_error must be finite and non-negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "memory_id": self.memory_id,
            "episode_id": self.episode_id,
            "tick": self.tick,
            "cue": self.cue.detach().cpu().clone(),
            "action_intent": (
                None if self.action_intent is None else self.action_intent.to_payload()
            ),
            "outcome": None if self.outcome is None else self.outcome.to_payload(),
            "world_transition": (
                None if self.world_transition is None else self.world_transition.to_payload()
            ),
            "prediction_error": self.prediction_error,
            "provenance": self.provenance,
            "event_ids": list(self.event_ids),
            "assembly_ids": list(self.assembly_ids),
            "object_ids": list(self.object_ids),
            "relation_ids": list(self.relation_ids),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> EpisodicMemoryRecord:
        action_intent = payload.get("action_intent")
        outcome = payload.get("outcome")
        world_transition = payload.get("world_transition")
        return cls(
            version=int(payload["version"]),
            memory_id=str(payload["memory_id"]),
            episode_id=str(payload["episode_id"]),
            tick=int(payload["tick"]),
            cue=payload["cue"].detach().to(device).clone(),
            action_intent=(
                None
                if action_intent is None
                else ActionIntent.from_payload(action_intent, device=device)
            ),
            outcome=None if outcome is None else Outcome.from_payload(outcome, device=device),
            world_transition=(
                None
                if world_transition is None
                else WorldTransition.from_payload(world_transition, device=device)
            ),
            prediction_error=float(payload.get("prediction_error", 0.0)),
            provenance=str(payload.get("provenance", "experienced")),
            event_ids=tuple(str(item) for item in payload.get("event_ids", ())),
            assembly_ids=tuple(str(item) for item in payload.get("assembly_ids", ())),
            object_ids=tuple(str(item) for item in payload.get("object_ids", ())),
            relation_ids=tuple(str(item) for item in payload.get("relation_ids", ())),
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
class WorldEpisode:
    """A contiguous multi-step world experience owned by Taiji."""

    episode_id: str
    initial: WorldState
    transitions: tuple[WorldTransition, ...] = ()
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.episode_id, "world episode_id")
        if not self.transitions:
            raise ValueError("world episode must contain at least one transition")
        action_ids = tuple(item.action.action_id for item in self.transitions)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("world episode actions must have unique action_id values")
        if self.transitions[0].before.tick != self.initial.tick:
            raise ValueError("world episode must start at the initial-state tick")
        for previous, current in zip(self.transitions, self.transitions[1:], strict=False):
            if previous.after.tick != current.before.tick:
                raise ValueError("world episode transitions must be contiguous")

    @property
    def final_state(self) -> WorldState:
        return self.transitions[-1].after

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "initial": self.initial.to_payload(),
            "transitions": [item.to_payload() for item in self.transitions],
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldEpisode:
        return cls(
            version=int(payload["version"]),
            episode_id=str(payload["episode_id"]),
            initial=WorldState.from_payload(payload["initial"], device=device),
            transitions=tuple(
                WorldTransition.from_payload(item, device=device)
                for item in payload.get("transitions", ())
            ),
        )


@dataclass(frozen=True)
class WorldEpisodeCorpus:
    """Train/holdout episodes for multi-step and cross-episode evaluation."""

    train: tuple[WorldEpisode, ...] = ()
    holdout: tuple[WorldEpisode, ...] = ()
    format: str = "taiji-world-episode-v1"
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        if self.format != "taiji-world-episode-v1":
            raise ValueError(f"unsupported world episode corpus format: {self.format}")
        train_ids = {episode.episode_id for episode in self.train}
        holdout_ids = {episode.episode_id for episode in self.holdout}
        if len(train_ids) != len(self.train) or len(holdout_ids) != len(self.holdout):
            raise ValueError("world episode ids must be unique within each split")
        if train_ids & holdout_ids:
            raise ValueError("world episode train and holdout splits must be disjoint")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "train": [episode.to_payload() for episode in self.train],
            "holdout": [episode.to_payload() for episode in self.holdout],
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> WorldEpisodeCorpus:
        return cls(
            format=str(payload["format"]),
            version=int(payload["version"]),
            train=tuple(
                WorldEpisode.from_payload(item, device=device)
                for item in payload.get("train", ())
            ),
            holdout=tuple(
                WorldEpisode.from_payload(item, device=device)
                for item in payload.get("holdout", ())
            ),
        )


@dataclass(frozen=True)
class SelfState:
    tick: int
    confidence: float = 0.0
    resource_fraction: float = 1.0
    capability_confidence: tuple[tuple[str, float], ...] = ()
    available_tool_ids: tuple[str, ...] = ()
    autobiographical_ids: tuple[str, ...] = ()
    commitment_ids: tuple[str, ...] = ()
    last_outcome_id: str | None = None
    last_update_source: str = "bootstrap"
    last_prediction_error: float = 0.0
    update_count: int = 0
    lineage: tuple[str, ...] = ()
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_unit(self.confidence, "self confidence")
        _check_unit(self.resource_fraction, "resource_fraction")
        _check_text(self.last_update_source, "self update source")
        object.__setattr__(
            self,
            "capability_confidence",
            _normalize_unit_pairs(self.capability_confidence, "capability_confidence"),
        )
        for field_name in (
            "available_tool_ids",
            "autobiographical_ids",
            "commitment_ids",
            "lineage",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_ids(getattr(self, field_name), f"self {field_name}"),
            )
        if self.last_outcome_id is not None:
            _check_text(self.last_outcome_id, "self last_outcome_id")
        if not math.isfinite(float(self.last_prediction_error)) or float(self.last_prediction_error) < 0.0:
            raise ValueError("self last_prediction_error must be finite and non-negative")
        if int(self.tick) < 0 or int(self.update_count) < 0:
            raise ValueError("self counters cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "confidence": self.confidence,
            "resource_fraction": self.resource_fraction,
            "capability_confidence": list(self.capability_confidence),
            "available_tool_ids": list(self.available_tool_ids),
            "autobiographical_ids": list(self.autobiographical_ids),
            "commitment_ids": list(self.commitment_ids),
            "last_outcome_id": self.last_outcome_id,
            "last_update_source": self.last_update_source,
            "last_prediction_error": self.last_prediction_error,
            "update_count": self.update_count,
            "lineage": list(self.lineage),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SelfState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            confidence=float(payload.get("confidence", 0.0)),
            resource_fraction=float(payload.get("resource_fraction", 1.0)),
            capability_confidence=tuple(
                (str(key), float(value))
                for key, value in payload.get("capability_confidence", ())
            ),
            available_tool_ids=tuple(str(item) for item in payload.get("available_tool_ids", ())),
            autobiographical_ids=tuple(
                str(item) for item in payload.get("autobiographical_ids", ())
            ),
            commitment_ids=tuple(str(item) for item in payload.get("commitment_ids", ())),
            last_outcome_id=(
                None
                if payload.get("last_outcome_id") is None
                else str(payload["last_outcome_id"])
            ),
            last_update_source=str(payload.get("last_update_source", "bootstrap")),
            last_prediction_error=float(payload.get("last_prediction_error", 0.0)),
            update_count=int(payload.get("update_count", 0)),
            lineage=tuple(str(item) for item in payload.get("lineage", ())),
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
    resource_utilization: float = 0.0
    capability_gaps: tuple[str, ...] = ()
    proposal_ids: tuple[str, ...] = ()
    parent_checkpoint_id: str | None = None
    last_update_source: str = "bootstrap"
    last_validation_status: str = "none"
    validation_evidence_ids: tuple[str, ...] = ()
    growth_count: int = 0
    prune_count: int = 0
    split_merge_count: int = 0
    lineage: tuple[str, ...] = ()
    version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _check_version(self.version)
        _check_text(self.stage, "development stage")
        _check_text(self.last_update_source, "development update source")
        if self.last_validation_status not in {"none", "pending", "accepted", "rejected", "rolled_back"}:
            raise ValueError("unsupported development validation status")
        _check_unit(self.resource_utilization, "development resource_utilization")
        object.__setattr__(
            self,
            "capability_gaps",
            _normalize_ids(self.capability_gaps, "development capability_gaps"),
        )
        for field_name in ("proposal_ids", "validation_evidence_ids", "lineage"):
            object.__setattr__(
                self,
                field_name,
                _normalize_ids(getattr(self, field_name), f"development {field_name}"),
            )
        if self.parent_checkpoint_id is not None:
            _check_text(self.parent_checkpoint_id, "development parent_checkpoint_id")
        if (
            int(self.tick) < 0
            or int(self.structural_budget) < 0
            or int(self.growth_count) < 0
            or int(self.prune_count) < 0
            or int(self.split_merge_count) < 0
        ):
            raise ValueError("development counters cannot be negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tick": self.tick,
            "stage": self.stage,
            "structural_budget": self.structural_budget,
            "resource_utilization": self.resource_utilization,
            "capability_gaps": list(self.capability_gaps),
            "proposal_ids": list(self.proposal_ids),
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "last_update_source": self.last_update_source,
            "last_validation_status": self.last_validation_status,
            "validation_evidence_ids": list(self.validation_evidence_ids),
            "growth_count": self.growth_count,
            "prune_count": self.prune_count,
            "split_merge_count": self.split_merge_count,
            "lineage": list(self.lineage),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DevelopmentState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            stage=str(payload.get("stage", "bootstrap")),
            structural_budget=int(payload.get("structural_budget", 0)),
            resource_utilization=float(payload.get("resource_utilization", 0.0)),
            capability_gaps=tuple(str(item) for item in payload.get("capability_gaps", ())),
            proposal_ids=tuple(str(item) for item in payload.get("proposal_ids", ())),
            parent_checkpoint_id=(
                None
                if payload.get("parent_checkpoint_id") is None
                else str(payload["parent_checkpoint_id"])
            ),
            last_update_source=str(payload.get("last_update_source", "bootstrap")),
            last_validation_status=str(payload.get("last_validation_status", "none")),
            validation_evidence_ids=tuple(
                str(item) for item in payload.get("validation_evidence_ids", ())
            ),
            growth_count=int(payload.get("growth_count", 0)),
            prune_count=int(payload.get("prune_count", 0)),
            split_merge_count=int(payload.get("split_merge_count", 0)),
            lineage=tuple(str(item) for item in payload.get("lineage", ())),
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
    world_transition: WorldTransition | None = None
    world_prediction: WorldPredictionRecord | None = None
    world_calibration_trace: tuple[WorldCalibrationTrace, ...] = ()
    planning_recovery: PlanningRecoveryState | None = None
    assemblies: tuple[Assembly, ...] = ()
    events: tuple[Event, ...] = ()
    concepts: tuple[Concept, ...] = ()
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
            "world_transition": (
                None if self.world_transition is None else self.world_transition.to_payload()
            ),
            "world_prediction": (
                None if self.world_prediction is None else self.world_prediction.to_payload()
            ),
            "world_calibration_trace": [
                item.to_payload() for item in self.world_calibration_trace
            ],
            "planning_recovery": (
                None if self.planning_recovery is None else self.planning_recovery.to_payload()
            ),
            "assemblies": [item.to_payload() for item in self.assemblies],
            "events": [item.to_payload() for item in self.events],
            "concepts": [item.to_payload() for item in self.concepts],
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> CognitiveState:
        observation = payload.get("observation")
        percept = payload.get("percept")
        intent = payload.get("action_intent")
        outcome = payload.get("outcome")
        world_transition = payload.get("world_transition")
        world_prediction = payload.get("world_prediction")
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
            world_transition=(
                None
                if world_transition is None
                else WorldTransition.from_payload(world_transition, device=device)
            ),
            world_prediction=(
                None
                if world_prediction is None
                else WorldPredictionRecord.from_payload(world_prediction, device=device)
            ),
            world_calibration_trace=tuple(
                WorldCalibrationTrace.from_payload(item, device=device)
                for item in payload.get("world_calibration_trace", ())
            ),
            planning_recovery=(
                None
                if payload.get("planning_recovery") is None
                else PlanningRecoveryState.from_payload(payload["planning_recovery"])
            ),
            assemblies=tuple(
                Assembly.from_payload(item, device=device)
                for item in payload.get("assemblies", ())
            ),
            events=tuple(
                Event.from_payload(item, device=device) for item in payload.get("events", ())
            ),
            concepts=tuple(
                Concept.from_payload(item, device=device)
                for item in payload.get("concepts", ())
            ),
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

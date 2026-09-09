"""Taiji-owned observed-outcome projection for dependent tasks.

This module is deliberately a shadow boundary for M5.K3.  It does not execute
an action, select a capability, or update a learner.  A typed ``WorldEvent``
and an explicit dependency specification are converted into a content-addressed
projection that can enrich an owned ``WorldState``.  The projection is kept
separate from predictive transition output so an observed result cannot be
silently counted as a prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import WorldEvent, WorldState
from .internalization import content_digest

OUTCOME_DEPENDENCY_PROJECTION_FORMAT = "taiji-outcome-dependency-projection-v1"
OUTCOME_DEPENDENCY_PROJECTION_VERSION = 1


def _text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _digest(value: Any, name: str) -> str:
    normalized = _text(value, name)
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _facts(values: Sequence[Sequence[Any]]) -> tuple[tuple[str, str, str], ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("dependency facts must be a sequence of triples")
    normalized: list[tuple[str, str, str]] = []
    for index, value in enumerate(values):
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise TypeError(f"dependency fact {index} must be a triple")
        if len(value) != 3:
            raise ValueError(f"dependency fact {index} must contain three values")
        item = tuple(_text(part, f"dependency fact {index}") for part in value)
        normalized.append((item[0], item[1], item[2]))
    if len(set(normalized)) != len(normalized):
        raise ValueError("dependency facts must be unique")
    return tuple(normalized)


def _lineage(values: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("dependency lineage must be a sequence")
    normalized = tuple(_text(value, "dependency lineage item") for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("dependency lineage must be unique")
    return normalized


@dataclass(frozen=True)
class OutcomeDependencySpec:
    """A fixture- or task-owner-declared next-step dependency contract."""

    dependency_id: str
    next_task_id: str
    capability_id: str
    required_outcome: str
    target_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependency_id", _text(self.dependency_id, "dependency_id"))
        object.__setattr__(self, "next_task_id", _text(self.next_task_id, "next_task_id"))
        object.__setattr__(self, "capability_id", _text(self.capability_id, "capability_id"))
        required = _text(self.required_outcome, "required_outcome")
        if required not in {"success", "failure"}:
            raise ValueError("required_outcome must be 'success' or 'failure'")
        object.__setattr__(self, "required_outcome", required)
        if self.target_digest:
            object.__setattr__(self, "target_digest", _digest(self.target_digest, "target_digest"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "dependency_id": self.dependency_id,
            "next_task_id": self.next_task_id,
            "capability_id": self.capability_id,
            "required_outcome": self.required_outcome,
            "target_digest": self.target_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> OutcomeDependencySpec:
        return cls(
            dependency_id=str(payload["dependency_id"]),
            next_task_id=str(payload["next_task_id"]),
            capability_id=str(payload["capability_id"]),
            required_outcome=str(payload["required_outcome"]),
            target_digest=str(payload.get("target_digest", "")),
        )


@dataclass(frozen=True)
class OutcomeDependencyProjection:
    """Auditable observed outcome and its typed next-step dependency."""

    scope_id: str
    event: WorldEvent
    spec: OutcomeDependencySpec
    outcome_signature: str
    dependency_digest: str
    outcome_class: str
    parent_tick: int
    next_tick: int
    accepted: bool
    reason_code: str
    facts: tuple[tuple[str, str, str], ...] = ()
    lineage: tuple[str, ...] = ()
    format: str = OUTCOME_DEPENDENCY_PROJECTION_FORMAT
    version: int = OUTCOME_DEPENDENCY_PROJECTION_VERSION

    def __post_init__(self) -> None:
        if self.format != OUTCOME_DEPENDENCY_PROJECTION_FORMAT:
            raise ValueError("unsupported outcome dependency projection format")
        if int(self.version) != OUTCOME_DEPENDENCY_PROJECTION_VERSION:
            raise ValueError("unsupported outcome dependency projection version")
        object.__setattr__(self, "scope_id", _text(self.scope_id, "scope_id"))
        if not isinstance(self.event, WorldEvent):
            raise TypeError("projection event must be a WorldEvent")
        if not isinstance(self.spec, OutcomeDependencySpec):
            raise TypeError("projection spec must be an OutcomeDependencySpec")
        object.__setattr__(
            self, "outcome_signature", _digest(self.outcome_signature, "outcome_signature")
        )
        object.__setattr__(
            self, "dependency_digest", _digest(self.dependency_digest, "dependency_digest")
        )
        object.__setattr__(self, "outcome_class", str(self.outcome_class).strip())
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code"))
        if int(self.parent_tick) < 0 or int(self.next_tick) != int(self.parent_tick) + 1:
            raise ValueError("projection ticks must advance by one")
        normalized_facts = _facts(self.facts)
        normalized_lineage = _lineage(self.lineage)
        object.__setattr__(self, "facts", normalized_facts)
        object.__setattr__(self, "lineage", normalized_lineage)
        if self.accepted:
            if self.event.tick != int(self.parent_tick):
                raise ValueError("accepted projection event must match parent_tick")
            if self.outcome_class not in {"success", "failure"}:
                raise ValueError("accepted projection outcome_class is invalid")
            if self.reason_code != "accepted":
                raise ValueError("accepted projection must use the accepted reason code")
            if not self.facts or len(self.lineage) != 4:
                raise ValueError("accepted projection must carry facts and complete lineage")
        elif self.facts or self.lineage:
            raise ValueError("rejected projection cannot carry executable dependency facts")

    @property
    def projection_digest(self) -> str:
        return content_digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "version": self.version,
            "scope_id": self.scope_id,
            "event": self.event.to_payload(),
            "spec": self.spec.to_payload(),
            "outcome_signature": self.outcome_signature,
            "dependency_digest": self.dependency_digest,
            "outcome_class": self.outcome_class,
            "parent_tick": int(self.parent_tick),
            "next_tick": int(self.next_tick),
            "accepted": bool(self.accepted),
            "reason_code": self.reason_code,
            "facts": [list(item) for item in self.facts],
            "lineage": list(self.lineage),
        }
        if include_digest:
            payload["projection_digest"] = self.projection_digest
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> OutcomeDependencyProjection:
        projection = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            scope_id=str(payload["scope_id"]),
            event=WorldEvent.from_payload(payload["event"]),
            spec=OutcomeDependencySpec.from_payload(payload["spec"]),
            outcome_signature=str(payload["outcome_signature"]),
            dependency_digest=str(payload["dependency_digest"]),
            outcome_class=str(payload.get("outcome_class", "")),
            parent_tick=int(payload["parent_tick"]),
            next_tick=int(payload["next_tick"]),
            accepted=bool(payload["accepted"]),
            reason_code=str(payload["reason_code"]),
            facts=_facts(payload.get("facts", ())),
            lineage=tuple(str(item) for item in payload.get("lineage", ())),
        )
        if str(payload.get("projection_digest", "")) != projection.projection_digest:
            raise ValueError("outcome dependency projection digest mismatch")
        return projection


class OutcomeDependencyProjector:
    """Create and apply typed outcome feedback inside one episode scope."""

    CHECKPOINT_FORMAT = OUTCOME_DEPENDENCY_PROJECTION_FORMAT
    CHECKPOINT_VERSION = OUTCOME_DEPENDENCY_PROJECTION_VERSION

    def __init__(self, scope_id: str, *, lesioned: bool = False) -> None:
        self.scope_id = _text(scope_id, "scope_id")
        self.lesioned = bool(lesioned)

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "scope_id": self.scope_id,
            "lesioned": self.lesioned,
        }
        return {**payload, "checkpoint_digest": content_digest(payload)}

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> OutcomeDependencyProjector:
        if payload.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported outcome dependency projector format")
        if int(payload.get("version", -1)) != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported outcome dependency projector version")
        unsigned = {
            "format": payload["format"],
            "version": int(payload["version"]),
            "scope_id": payload["scope_id"],
            "lesioned": bool(payload.get("lesioned", False)),
        }
        if str(payload.get("checkpoint_digest", "")) != content_digest(unsigned):
            raise ValueError("outcome dependency projector checkpoint digest mismatch")
        return cls(str(unsigned["scope_id"]), lesioned=bool(unsigned["lesioned"]))

    @staticmethod
    def _outcome_class(event: WorldEvent) -> tuple[str | None, str | None]:
        attributes = dict(event.attributes)
        success = attributes.get("success")
        if not isinstance(success, bool):
            return None, "invalid_outcome_success"
        capability_id = str(attributes.get("capability_id", event.subject_id)).strip()
        if not capability_id:
            return None, "missing_outcome_capability"
        return ("success" if success else "failure"), None

    def _rejected(
        self,
        *,
        event: WorldEvent,
        spec: OutcomeDependencySpec,
        reason_code: str,
        parent_tick: int,
        outcome_signature: str,
        outcome_class: str = "",
    ) -> OutcomeDependencyProjection:
        return OutcomeDependencyProjection(
            scope_id=self.scope_id,
            event=event,
            spec=spec,
            outcome_signature=(
                outcome_signature
                if outcome_signature
                else content_digest(event.to_payload())
            ),
            dependency_digest=content_digest(
                {
                    "scope_id": self.scope_id,
                    "event_id": event.event_id,
                    "spec": spec.to_payload(),
                    "reason_code": reason_code,
                }
            ),
            outcome_class=outcome_class,
            parent_tick=max(0, int(parent_tick)),
            next_tick=max(0, int(parent_tick)) + 1,
            accepted=False,
            reason_code=reason_code,
        )

    def project(
        self,
        world: WorldState,
        event: WorldEvent,
        spec: OutcomeDependencySpec,
    ) -> OutcomeDependencyProjection:
        if not isinstance(world, WorldState):
            raise TypeError("outcome dependency projection requires a WorldState")
        if not isinstance(event, WorldEvent):
            raise TypeError("outcome dependency projection requires a WorldEvent")
        if not isinstance(spec, OutcomeDependencySpec):
            raise TypeError("outcome dependency projection requires an OutcomeDependencySpec")
        outcome_signature = content_digest(event.to_payload())
        outcome_class, outcome_error = self._outcome_class(event)
        if self.lesioned:
            return self._rejected(
                event=event,
                spec=spec,
                reason_code="outcome_feedback_lesioned",
                parent_tick=world.tick,
                outcome_signature=outcome_signature,
                outcome_class=outcome_class or "",
            )
        if event.tick != world.tick:
            return self._rejected(
                event=event,
                spec=spec,
                reason_code="stale_event_tick",
                parent_tick=world.tick,
                outcome_signature=outcome_signature,
                outcome_class=outcome_class or "",
            )
        existing = {item.event_id: item for item in world.events}
        if event.event_id in existing and existing[event.event_id].to_payload() != event.to_payload():
            return self._rejected(
                event=event,
                spec=spec,
                reason_code="event_identity_conflict",
                parent_tick=world.tick,
                outcome_signature=outcome_signature,
                outcome_class=outcome_class or "",
            )
        if outcome_error is not None:
            return self._rejected(
                event=event,
                spec=spec,
                reason_code=outcome_error,
                parent_tick=world.tick,
                outcome_signature=outcome_signature,
            )
        if outcome_class != spec.required_outcome:
            return self._rejected(
                event=event,
                spec=spec,
                reason_code="dependency_outcome_mismatch",
                parent_tick=world.tick,
                outcome_signature=outcome_signature,
                outcome_class=outcome_class or "",
            )
        dependency_digest = content_digest(
            {
                "scope_id": self.scope_id,
                "event_id": event.event_id,
                "outcome_signature": outcome_signature,
                "spec": spec.to_payload(),
                "parent_tick": int(world.tick),
            }
        )
        facts = (
            ("outcome", "class", outcome_class or ""),
            ("outcome", "capability", spec.capability_id),
            ("dependency", "id", spec.dependency_id),
            ("dependency", "task", spec.next_task_id),
            ("dependency", "digest", dependency_digest),
        )
        return OutcomeDependencyProjection(
            scope_id=self.scope_id,
            event=event,
            spec=spec,
            outcome_signature=outcome_signature,
            dependency_digest=dependency_digest,
            outcome_class=outcome_class or "",
            parent_tick=int(world.tick),
            next_tick=int(world.tick) + 1,
            accepted=True,
            reason_code="accepted",
            facts=facts,
            lineage=(self.scope_id, event.event_id, outcome_signature, dependency_digest),
        )

    def apply(self, world: WorldState, projection: OutcomeDependencyProjection) -> WorldState:
        if not isinstance(world, WorldState):
            raise TypeError("outcome dependency application requires a WorldState")
        if not isinstance(projection, OutcomeDependencyProjection):
            raise TypeError("outcome dependency application requires a projection")
        if projection.scope_id != self.scope_id:
            raise ValueError("outcome dependency projection scope mismatch")
        if not projection.accepted:
            raise ValueError(f"cannot apply rejected outcome dependency: {projection.reason_code}")
        if world.tick != projection.parent_tick:
            raise ValueError("outcome dependency projection is stale for the current world")
        existing = {item.event_id: item for item in world.events}
        if projection.event.event_id in existing:
            if existing[projection.event.event_id].to_payload() != projection.event.to_payload():
                raise ValueError("outcome dependency event identity conflict")
        if ("dependency", "digest", projection.dependency_digest) in world.relations:
            raise ValueError("outcome dependency projection was already applied")
        events = world.events
        if projection.event.event_id not in existing:
            events = (*events, projection.event)
        return replace(world, events=events, relations=(*world.relations, *projection.facts))


__all__ = [
    "OUTCOME_DEPENDENCY_PROJECTION_FORMAT",
    "OUTCOME_DEPENDENCY_PROJECTION_VERSION",
    "OutcomeDependencyProjector",
    "OutcomeDependencyProjection",
    "OutcomeDependencySpec",
]

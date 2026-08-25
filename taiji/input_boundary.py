"""Versioned input boundary between product clients and Taiji perception.

The frame is a transport contract, not a semantic interpreter.  Text may be
validated as UTF-8 and then remains raw bytes until Taiji perception emits
``Observation`` and ``PerceptEvent`` values.  No token IDs, prompt templates,
or fixed intent mappings are introduced here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import ActionIntent, Observation, PerceptEvent

INPUT_BOUNDARY_FORMAT = "taiji-input-boundary-v1"


@dataclass(frozen=True)
class InputFrame:
    """One external input payload before Taiji's learned perception."""

    input_id: str
    modality: str
    payload: bytes
    source: str
    timestamp: int = 0
    provenance: str = "external"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not str(self.input_id):
            raise ValueError("input_id cannot be empty")
        if not str(self.modality):
            raise ValueError("input modality cannot be empty")
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise TypeError("input payload must be bytes-like")
        payload = bytes(self.payload)
        if not payload:
            raise ValueError("input payload cannot be empty")
        if not str(self.source):
            raise ValueError("input source cannot be empty")
        if not str(self.provenance):
            raise ValueError("input provenance cannot be empty")
        if int(self.timestamp) < 0:
            raise ValueError("input timestamp cannot be negative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("input confidence must be in [0, 1]")
        if self.modality in {"text", "text-utf8"}:
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("text input payload must be valid UTF-8") from exc
        object.__setattr__(self, "payload", payload)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INPUT_BOUNDARY_FORMAT,
            "input_id": self.input_id,
            "modality": self.modality,
            "payload": bytes(self.payload),
            "source": self.source,
            "timestamp": self.timestamp,
            "provenance": self.provenance,
            "confidence": self.confidence,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InputFrame:
        if payload.get("format") != INPUT_BOUNDARY_FORMAT:
            raise ValueError("unsupported Taiji input boundary format")
        return cls(
            input_id=str(payload["input_id"]),
            modality=str(payload["modality"]),
            payload=payload["payload"],
            source=str(payload["source"]),
            timestamp=int(payload.get("timestamp", 0)),
            provenance=str(payload.get("provenance", "external")),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class InputTrace:
    """Inspectable result of one frame entering Taiji perception."""

    input_id: str
    modality: str
    observations: tuple[Observation, ...]
    percepts: tuple[PerceptEvent, ...]
    action_intent: ActionIntent | None = None

    def __post_init__(self) -> None:
        if not str(self.input_id):
            raise ValueError("input trace input_id cannot be empty")
        if not str(self.modality):
            raise ValueError("input trace modality cannot be empty")
        if len(self.observations) != len(self.percepts):
            raise ValueError("input observations and percepts must have equal lengths")
        if any(not isinstance(item, Observation) for item in self.observations):
            raise TypeError("input trace observations must be Observation values")
        if any(not isinstance(item, PerceptEvent) for item in self.percepts):
            raise TypeError("input trace percepts must be PerceptEvent values")
        if self.action_intent is not None and not isinstance(self.action_intent, ActionIntent):
            raise TypeError("input trace action_intent must be an ActionIntent or None")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INPUT_BOUNDARY_FORMAT,
            "input_id": self.input_id,
            "modality": self.modality,
            "observations": [item.to_payload() for item in self.observations],
            "percepts": [item.to_payload() for item in self.percepts],
            "action_intent": (
                None if self.action_intent is None else self.action_intent.to_payload()
            ),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: str = "cpu",
    ) -> InputTrace:
        if payload.get("format") != INPUT_BOUNDARY_FORMAT:
            raise ValueError("unsupported Taiji input trace format")
        observations = payload.get("observations", ())
        percepts = payload.get("percepts", ())
        if not isinstance(observations, (list, tuple)) or not isinstance(percepts, (list, tuple)):
            raise ValueError("input trace observations and percepts must be sequences")
        intent = payload.get("action_intent")
        return cls(
            input_id=str(payload["input_id"]),
            modality=str(payload["modality"]),
            observations=tuple(
                Observation.from_payload(item, device=device) for item in observations
            ),
            percepts=tuple(PerceptEvent.from_payload(item, device=device) for item in percepts),
            action_intent=(
                None if intent is None else ActionIntent.from_payload(intent, device=device)
            ),
        )

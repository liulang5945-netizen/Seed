"""Provider-to-Taiji semantic evidence contract.

The provider is an optional semantic sensor.  It may propose a goal, context,
and bounded semantic steps, but it cannot claim a resolved Goal, name a
capability, create an ActionIntent, select a tool, or execute anything.
Taiji validates the proposal against the live input frame and decides whether
the evidence is strong enough to become a task interpretation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .input_boundary import InputFrame
from .internalization import content_digest
from .task_interpretation import task_input_digest

SEMANTIC_PROVIDER_EVIDENCE_FORMAT = "taiji-semantic-provider-evidence-v1"
SEMANTIC_PROVIDER_EVIDENCE_VERSION = 1
SEMANTIC_PROVIDER_REQUEST_FORMAT = "taiji-semantic-provider-request-v1"
SEMANTIC_PROVIDER_INTERFACE_FORMAT = "taiji-semantic-provider-interface-v1"
SEMANTIC_PROVIDER_MAX_STEPS = 8
SEMANTIC_PROVIDER_CONFIDENCE_FLOOR = 0.5
SEMANTIC_PROVIDER_AMBIGUITY_CEILING = 0.5
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_STEP_FIELDS = frozenset(
    {"description", "semantic_slots", "expected_outcome", "confidence", "ambiguity", "provenance"}
)
_FORBIDDEN_FIELDS = frozenset(
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


def _digest_text(value: Any, name: str, *, optional: bool = False) -> str:
    normalized = str(value).strip().lower()
    if not normalized and optional:
        return ""
    if not _HEX_DIGEST.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _bounded(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _normalize_constraints(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("provider constraints must be a sequence of strings")
    return tuple(sorted({_required_text(item, "provider constraint") for item in value}))


def _reject_execution_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_name = str(key).strip().lower()
            if key_name in _FORBIDDEN_FIELDS:
                raise ValueError(
                    f"{path} cannot carry execution field {key_name!r}; "
                    "provider evidence stops before Taiji grounding"
                )
            _reject_execution_fields(nested, path=f"{path}.{key_name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_execution_fields(nested, path=f"{path}[{index}]")


def _normalize_steps(value: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("provider semantic_steps must be a sequence of mappings")
    if len(value) > SEMANTIC_PROVIDER_MAX_STEPS:
        raise ValueError("provider semantic_steps cannot exceed 8 entries")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"provider semantic step {index} must be a mapping")
        _reject_execution_fields(item, path=f"semantic_steps[{index}]")
        unknown = set(item) - _STEP_FIELDS
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"provider semantic step {index} has unsupported fields: {names}")
        description = _required_text(item.get("description", ""), "provider step description")
        slots = item.get("semantic_slots", {})
        if not isinstance(slots, Mapping):
            raise TypeError(f"provider semantic step {index} semantic_slots must be a mapping")
        normalized_slots = dict(slots)
        _reject_execution_fields(normalized_slots, path=f"semantic_steps[{index}].semantic_slots")
        normalized_item: dict[str, Any] = {
            "description": description,
            "semantic_slots": normalized_slots,
            "expected_outcome": str(item.get("expected_outcome", "")).strip(),
        }
        if "confidence" in item:
            normalized_item["confidence"] = _bounded(
                item["confidence"], f"provider step {index} confidence"
            )
        if "ambiguity" in item:
            normalized_item["ambiguity"] = _bounded(
                item["ambiguity"], f"provider step {index} ambiguity"
            )
        if "provenance" in item:
            normalized_item["provenance"] = _required_text(
                item["provenance"], f"provider step {index} provenance"
            )
        normalized.append(normalized_item)
    return tuple(normalized)


@dataclass(frozen=True)
class SemanticProviderRequest:
    """Content-addressed request presented to an independent semantic organ.

    The request exposes the exact input frame and a digest of conversational
    context, but it carries no capability, tool, parameter, intent, or
    execution authority.  A provider may read the frame and propose semantic
    evidence; Taiji remains the only owner that can admit, ground, or execute
    the proposal.
    """

    frame: InputFrame
    context_digest: str = ""
    constraints: tuple[str, ...] = ()
    request_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.frame, InputFrame):
            raise TypeError("semantic provider request frame must be a Taiji InputFrame")
        context_digest = _digest_text(self.context_digest, "context_digest", optional=True)
        constraints = _normalize_constraints(self.constraints)
        identity = self._identity_payload(
            input_id=self.frame.input_id,
            input_digest=task_input_digest(bytes(self.frame.payload)),
            modality=self.frame.modality,
            context_digest=context_digest,
            constraints=constraints,
            tick=int(self.frame.timestamp),
        )
        expected_id = f"semantic-request:{content_digest(identity)[:24]}"
        if self.request_id and str(self.request_id) != expected_id:
            raise ValueError("semantic provider request_id is not content-addressed")
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "request_id", expected_id)

    @staticmethod
    def _identity_payload(
        *,
        input_id: str,
        input_digest: str,
        modality: str,
        context_digest: str,
        constraints: tuple[str, ...],
        tick: int,
    ) -> dict[str, Any]:
        return {
            "input_id": input_id,
            "input_digest": input_digest,
            "modality": modality,
            "context_digest": context_digest,
            "constraints": list(constraints),
            "tick": tick,
        }

    @classmethod
    def from_frame(
        cls,
        frame: InputFrame,
        *,
        context_digest: str = "",
        constraints: Sequence[str] = (),
    ) -> SemanticProviderRequest:
        return cls(
            frame=frame,
            context_digest=context_digest,
            constraints=tuple(constraints),
        )

    @property
    def input_digest(self) -> str:
        return task_input_digest(bytes(self.frame.payload))

    def to_payload(self) -> dict[str, Any]:
        """Return auditable metadata without duplicating the raw input bytes."""

        return {
            "format": SEMANTIC_PROVIDER_REQUEST_FORMAT,
            "request_id": self.request_id,
            "input_id": self.frame.input_id,
            "input_digest": self.input_digest,
            "modality": self.frame.modality,
            "context_digest": self.context_digest,
            "constraints": list(self.constraints),
            "tick": int(self.frame.timestamp),
        }


@dataclass(frozen=True)
class SemanticEvidenceProposal:
    """Content-addressed provider evidence before Taiji interpretation."""

    proposal_id: str
    provider_id: str
    input_id: str
    input_digest: str
    modality: str
    goal_description: str
    constraints: tuple[str, ...] = ()
    context_digest: str = ""
    semantic_steps: tuple[Mapping[str, Any], ...] = ()
    confidence: float = 0.0
    ambiguity: float = 1.0
    provenance: str = "semantic-provider"
    tick: int = 0
    version: int = SEMANTIC_PROVIDER_EVIDENCE_VERSION
    evidence_digest: str = ""

    def __post_init__(self) -> None:
        if int(self.version) != SEMANTIC_PROVIDER_EVIDENCE_VERSION:
            raise ValueError("unsupported semantic provider evidence version")
        provider_id = _required_text(self.provider_id, "provider_id")
        input_id = _required_text(self.input_id, "input_id")
        input_digest = _digest_text(self.input_digest, "input_digest")
        modality = _required_text(self.modality, "modality")
        goal_description = _required_text(self.goal_description, "goal_description")
        constraints = _normalize_constraints(self.constraints)
        context_digest = _digest_text(self.context_digest, "context_digest", optional=True)
        steps = _normalize_steps(self.semantic_steps)
        confidence = _bounded(self.confidence, "provider confidence")
        ambiguity = _bounded(self.ambiguity, "provider ambiguity")
        provenance = _required_text(self.provenance, "provider provenance")
        tick = int(self.tick)
        if tick < 0:
            raise ValueError("provider evidence tick cannot be negative")
        identity = self._identity_payload(
            provider_id=provider_id,
            input_id=input_id,
            input_digest=input_digest,
            modality=modality,
            goal_description=goal_description,
            constraints=constraints,
            context_digest=context_digest,
            semantic_steps=steps,
            confidence=confidence,
            ambiguity=ambiguity,
            provenance=provenance,
            tick=tick,
        )
        expected_id = f"semantic-evidence:{content_digest(identity)[:24]}"
        if str(self.proposal_id) != expected_id:
            raise ValueError("semantic provider proposal_id is not content-addressed")
        expected_digest = content_digest(
            {
                "format": SEMANTIC_PROVIDER_EVIDENCE_FORMAT,
                "version": SEMANTIC_PROVIDER_EVIDENCE_VERSION,
                "proposal_id": expected_id,
                **identity,
            }
        )
        if str(self.evidence_digest) != expected_digest:
            raise ValueError("semantic provider evidence digest does not match its payload")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "input_id", input_id)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "goal_description", goal_description)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(self, "semantic_steps", steps)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "ambiguity", ambiguity)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "tick", tick)
        object.__setattr__(self, "evidence_digest", expected_digest)

    @staticmethod
    def _identity_payload(
        *,
        provider_id: str,
        input_id: str,
        input_digest: str,
        modality: str,
        goal_description: str,
        constraints: tuple[str, ...],
        context_digest: str,
        semantic_steps: tuple[Mapping[str, Any], ...],
        confidence: float,
        ambiguity: float,
        provenance: str,
        tick: int,
    ) -> dict[str, Any]:
        return {
            "provider_id": provider_id,
            "input_id": input_id,
            "input_digest": input_digest,
            "modality": modality,
            "goal_description": goal_description,
            "constraints": list(constraints),
            "context_digest": context_digest,
            "semantic_steps": [dict(item) for item in semantic_steps],
            "confidence": confidence,
            "ambiguity": ambiguity,
            "provenance": provenance,
            "tick": tick,
        }

    @classmethod
    def from_frame(
        cls,
        frame: InputFrame,
        *,
        provider_id: str,
        goal_description: str,
        semantic_steps: Sequence[Mapping[str, Any]] = (),
        constraints: Sequence[str] = (),
        context_digest: str = "",
        confidence: float = 0.0,
        ambiguity: float = 1.0,
        provenance: str = "semantic-provider",
        tick: int | None = None,
    ) -> SemanticEvidenceProposal:
        if not isinstance(frame, InputFrame):
            raise TypeError("frame must be a Taiji InputFrame")
        normalized_tick = frame.timestamp if tick is None else int(tick)
        normalized_steps = _normalize_steps(semantic_steps)
        identity = cls._identity_payload(
            provider_id=_required_text(provider_id, "provider_id"),
            input_id=frame.input_id,
            input_digest=task_input_digest(bytes(frame.payload)),
            modality=frame.modality,
            goal_description=_required_text(goal_description, "goal_description"),
            constraints=_normalize_constraints(constraints),
            context_digest=_digest_text(context_digest, "context_digest", optional=True),
            semantic_steps=normalized_steps,
            confidence=_bounded(confidence, "provider confidence"),
            ambiguity=_bounded(ambiguity, "provider ambiguity"),
            provenance=_required_text(provenance, "provider provenance"),
            tick=normalized_tick,
        )
        proposal_id = f"semantic-evidence:{content_digest(identity)[:24]}"
        evidence_digest = content_digest(
            {
                "format": SEMANTIC_PROVIDER_EVIDENCE_FORMAT,
                "version": SEMANTIC_PROVIDER_EVIDENCE_VERSION,
                "proposal_id": proposal_id,
                **identity,
            }
        )
        return cls(
            proposal_id=proposal_id,
            provider_id=identity["provider_id"],
            input_id=identity["input_id"],
            input_digest=identity["input_digest"],
            modality=identity["modality"],
            goal_description=identity["goal_description"],
            constraints=tuple(identity["constraints"]),
            context_digest=identity["context_digest"],
            semantic_steps=tuple(identity["semantic_steps"]),
            confidence=identity["confidence"],
            ambiguity=identity["ambiguity"],
            provenance=identity["provenance"],
            tick=identity["tick"],
            evidence_digest=evidence_digest,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": SEMANTIC_PROVIDER_EVIDENCE_FORMAT,
            "version": self.version,
            "proposal_id": self.proposal_id,
            "provider_id": self.provider_id,
            "input_id": self.input_id,
            "input_digest": self.input_digest,
            "modality": self.modality,
            "goal_description": self.goal_description,
            "constraints": list(self.constraints),
            "context_digest": self.context_digest,
            "semantic_steps": [dict(item) for item in self.semantic_steps],
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "provenance": self.provenance,
            "tick": self.tick,
            "evidence_digest": self.evidence_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SemanticEvidenceProposal:
        if not isinstance(payload, Mapping):
            raise TypeError("semantic provider evidence payload must be a mapping")
        if payload.get("format") != SEMANTIC_PROVIDER_EVIDENCE_FORMAT:
            raise ValueError("unsupported semantic provider evidence format")
        return cls(
            version=int(payload["version"]),
            proposal_id=str(payload["proposal_id"]),
            provider_id=str(payload["provider_id"]),
            input_id=str(payload["input_id"]),
            input_digest=str(payload["input_digest"]),
            modality=str(payload["modality"]),
            goal_description=str(payload["goal_description"]),
            constraints=tuple(payload.get("constraints", ())),
            context_digest=str(payload.get("context_digest", "")),
            semantic_steps=tuple(payload.get("semantic_steps", ())),
            confidence=float(payload.get("confidence", 0.0)),
            ambiguity=float(payload.get("ambiguity", 1.0)),
            provenance=str(payload.get("provenance", "semantic-provider")),
            tick=int(payload.get("tick", 0)),
            evidence_digest=str(payload["evidence_digest"]),
        )


@runtime_checkable
class SemanticEvidenceProvider(Protocol):
    """Independent semantic-organ interface accepted by the Seed edge.

    Implementations may be a local model, a remote connector, or a learned
    Taiji-side organ.  The interface deliberately returns only
    ``SemanticEvidenceProposal``; it cannot return a tool call, ActionIntent,
    parameter binding, patch, or execution result.
    """

    @property
    def provider_id(self) -> str:
        """Stable provider identifier recorded in evidence provenance."""

    def propose(self, request: SemanticProviderRequest) -> SemanticEvidenceProposal:
        """Propose content-addressed semantic evidence for one request."""

    def checkpoint(self) -> Mapping[str, Any]:
        """Return a serializable descriptor for explicit provider rebinding."""

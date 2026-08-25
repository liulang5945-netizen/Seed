"""Terminal language-organ boundary for Taiji expression plans.

The language organ is an effector.  It receives an already selected
``ExpressionPlan`` and returns text bytes; it does not own goals, memory,
planning, content selection, or ``ActionIntent`` creation.  A mature decoder
can implement the same protocol later without entering Taiji's cognitive
core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .generation import ExpressionPlan, TextExpressionCodec

LANGUAGE_ORGAN_CHECKPOINT_FORMAT = "taiji-language-organ-v1"


@dataclass(frozen=True)
class LanguageEmission:
    """Inspectable output of one terminal language-organ emission."""

    expression: ExpressionPlan
    text_bytes: bytes
    backend: str
    provenance: str = "language-organ"

    def __post_init__(self) -> None:
        if not isinstance(self.expression, ExpressionPlan):
            raise TypeError("language emission expression must be an ExpressionPlan")
        if not isinstance(self.text_bytes, bytes):
            raise TypeError("language emission text_bytes must be bytes")
        if not self.text_bytes:
            raise ValueError("language emission text_bytes cannot be empty")
        if not str(self.backend):
            raise ValueError("language emission backend cannot be empty")
        if not str(self.provenance):
            raise ValueError("language emission provenance cannot be empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_ORGAN_CHECKPOINT_FORMAT,
            "expression": self.expression.to_payload(),
            "text_bytes": self.text_bytes,
            "backend": self.backend,
            "provenance": self.provenance,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LanguageEmission:
        if payload.get("format") != LANGUAGE_ORGAN_CHECKPOINT_FORMAT:
            raise ValueError("unsupported language emission format")
        text_bytes = payload.get("text_bytes", b"")
        if not isinstance(text_bytes, bytes):
            text_bytes = bytes(text_bytes)
        expression = payload.get("expression")
        if not isinstance(expression, Mapping):
            raise ValueError("language emission payload is missing expression")
        return cls(
            expression=ExpressionPlan.from_payload(expression),
            text_bytes=text_bytes,
            backend=str(payload["backend"]),
            provenance=str(payload.get("provenance", "language-organ")),
        )


@runtime_checkable
class LanguageOrgan(Protocol):
    """Replaceable terminal language-organ contract."""

    backend_id: str

    def emit(self, expression: ExpressionPlan) -> LanguageEmission:
        """Render a Taiji-owned text expression without changing cognition."""

    def checkpoint(self) -> dict[str, Any]:
        """Return a backend descriptor suitable for a Taiji checkpoint."""


class StructuredTextLanguageOrgan:
    """Deterministic baseline organ used to prove the boundary.

    This organ deliberately emits the lossless structured text codec instead of
    pretending that serialization is natural-language realization.  It is a
    replaceable stub and is not part of Taiji's cognitive state or parameters.
    """

    BACKEND_ID = "structured-stub"

    def __init__(self, *, max_bytes: int = 1_000_000) -> None:
        if int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)

    @property
    def backend_id(self) -> str:
        return self.BACKEND_ID

    def emit(self, expression: ExpressionPlan) -> LanguageEmission:
        if not isinstance(expression, ExpressionPlan):
            raise TypeError("language organ requires an ExpressionPlan")
        if expression.modality != "text":
            raise ValueError("language organ only accepts text ExpressionPlan values")
        encoded = TextExpressionCodec.encode(expression)
        if len(encoded) > self.max_bytes:
            raise ValueError("language expression exceeds language organ max_bytes")
        return LanguageEmission(
            expression=expression,
            text_bytes=encoded,
            backend=self.backend_id,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_ORGAN_CHECKPOINT_FORMAT,
            "backend": self.backend_id,
            "max_bytes": self.max_bytes,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> StructuredTextLanguageOrgan:
        if payload.get("format") != LANGUAGE_ORGAN_CHECKPOINT_FORMAT:
            raise ValueError("unsupported language organ checkpoint format")
        if payload.get("backend") != cls.BACKEND_ID:
            raise ValueError("unsupported structured language organ backend")
        return cls(max_bytes=int(payload.get("max_bytes", 1_000_000)))

"""Terminal language-organ boundary for Taiji expression plans.

The language organ is an effector.  It receives an already selected
``ExpressionPlan`` and returns text bytes; it does not own goals, memory,
planning, content selection, or ``ActionIntent`` creation.  A mature decoder
can implement the same protocol later without entering Taiji's cognitive
core.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from .generation import ExpressionPlan, TextExpressionCodec

LANGUAGE_ORGAN_CHECKPOINT_FORMAT = "taiji-language-organ-v1"
LANGUAGE_TRAINING_EXAMPLE_FORMAT = "taiji-language-training-example-v1"
LANGUAGE_TRAINING_CORPUS_FORMAT = "taiji-language-training-corpus-v1"
LANGUAGE_BACKEND_SPEC_FORMAT = "taiji-language-backend-spec-v1"
LANGUAGE_BACKEND_REGISTRY_FORMAT = "taiji-language-backend-registry-v1"
LANGUAGE_VALIDATION_FORMAT = "taiji-language-validation-v1"
LANGUAGE_PROVIDER_ARTIFACT_FORMAT = "taiji-language-provider-artifact-v1"
LANGUAGE_REALIZATION_GATE_FORMAT = "taiji-language-realization-gate-v1"


@dataclass(frozen=True)
class LanguageValidation:
    """Result of checking whether emitted text preserves required semantics."""

    accepted: bool
    required_terms: tuple[str, ...]
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    coverage: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.coverage) <= 1.0:
            raise ValueError("language validation coverage must be in [0, 1]")
        if not str(self.reason):
            raise ValueError("language validation reason cannot be empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_VALIDATION_FORMAT,
            "accepted": self.accepted,
            "required_terms": list(self.required_terms),
            "matched_terms": list(self.matched_terms),
            "missing_terms": list(self.missing_terms),
            "coverage": self.coverage,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LanguageValidation:
        if payload.get("format") != LANGUAGE_VALIDATION_FORMAT:
            raise ValueError("unsupported language validation format")
        return cls(
            accepted=bool(payload.get("accepted", False)),
            required_terms=tuple(str(term) for term in payload.get("required_terms", ())),
            matched_terms=tuple(str(term) for term in payload.get("matched_terms", ())),
            missing_terms=tuple(str(term) for term in payload.get("missing_terms", ())),
            coverage=float(payload.get("coverage", 0.0)),
            reason=str(payload.get("reason", "unknown")),
        )


@dataclass(frozen=True)
class LanguageEmission:
    """Inspectable output of one terminal language-organ emission."""

    expression: ExpressionPlan
    text_bytes: bytes
    backend: str
    provenance: str = "language-organ"
    validation: LanguageValidation | None = None
    fallback_used: bool = False

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
        if self.validation is not None and not isinstance(self.validation, LanguageValidation):
            raise TypeError("language emission validation must be a LanguageValidation")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_ORGAN_CHECKPOINT_FORMAT,
            "expression": self.expression.to_payload(),
            "text_bytes": self.text_bytes,
            "backend": self.backend,
            "provenance": self.provenance,
            "validation": None if self.validation is None else self.validation.to_payload(),
            "fallback_used": self.fallback_used,
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
        validation = payload.get("validation")
        return cls(
            expression=ExpressionPlan.from_payload(expression),
            text_bytes=text_bytes,
            backend=str(payload["backend"]),
            provenance=str(payload.get("provenance", "language-organ")),
            validation=(
                None if validation is None else LanguageValidation.from_payload(dict(validation))
            ),
            fallback_used=bool(payload.get("fallback_used", False)),
        )


@dataclass(frozen=True)
class LanguageTrainingExample:
    """Supervision contract for a text organ, without cognitive labels."""

    example_id: str
    expression: ExpressionPlan
    target_text: str
    split: str = "train"
    weight: float = 1.0
    provenance: str = "supervised"

    def __post_init__(self) -> None:
        if not str(self.example_id):
            raise ValueError("language training example_id cannot be empty")
        if not isinstance(self.expression, ExpressionPlan):
            raise TypeError("language training expression must be an ExpressionPlan")
        if self.expression.modality != "text":
            raise ValueError("language training expression must use text modality")
        if not isinstance(self.target_text, str) or not self.target_text.strip():
            raise ValueError("language training target_text cannot be empty")
        if not str(self.split):
            raise ValueError("language training split cannot be empty")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("language training weight must be finite and positive")
        if not str(self.provenance):
            raise ValueError("language training provenance cannot be empty")

    @property
    def target_bytes(self) -> bytes:
        return self.target_text.encode("utf-8")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_TRAINING_EXAMPLE_FORMAT,
            "example_id": self.example_id,
            "expression": self.expression.to_payload(),
            "target_text": self.target_text,
            "split": self.split,
            "weight": self.weight,
            "provenance": self.provenance,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LanguageTrainingExample:
        if payload.get("format") != LANGUAGE_TRAINING_EXAMPLE_FORMAT:
            raise ValueError("unsupported language training example format")
        expression = payload.get("expression")
        if not isinstance(expression, Mapping):
            raise ValueError("language training example is missing expression")
        return cls(
            example_id=str(payload["example_id"]),
            expression=ExpressionPlan.from_payload(expression),
            target_text=str(payload["target_text"]),
            split=str(payload.get("split", "train")),
            weight=float(payload.get("weight", 1.0)),
            provenance=str(payload.get("provenance", "supervised")),
        )


@dataclass(frozen=True)
class LanguageTrainingCorpus:
    """Disjoint train/holdout supervision owned by the provider boundary."""

    train: tuple[LanguageTrainingExample, ...]
    holdout: tuple[LanguageTrainingExample, ...]

    def __post_init__(self) -> None:
        train = tuple(self.train)
        holdout = tuple(self.holdout)
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "holdout", holdout)
        if not train or not holdout:
            raise ValueError("language training corpus requires non-empty train and holdout splits")
        examples = train + holdout
        if any(not isinstance(example, LanguageTrainingExample) for example in examples):
            raise TypeError("language training corpus accepts LanguageTrainingExample values")
        if any(example.split != "train" for example in train):
            raise ValueError("language training corpus train split contains a non-train example")
        if any(example.split != "holdout" for example in holdout):
            raise ValueError(
                "language training corpus holdout split contains a non-holdout example"
            )
        example_ids = tuple(example.example_id for example in examples)
        if len(set(example_ids)) != len(example_ids):
            raise ValueError("language training corpus example IDs must be unique")
        expression_ids = tuple(example.expression.expression_id for example in examples)
        if len(set(expression_ids)) != len(expression_ids):
            raise ValueError("language training corpus expression IDs must be unique")

    @property
    def size(self) -> int:
        return len(self.train) + len(self.holdout)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_TRAINING_CORPUS_FORMAT,
            "train": [example.to_payload() for example in self.train],
            "holdout": [example.to_payload() for example in self.holdout],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LanguageTrainingCorpus:
        if payload.get("format") != LANGUAGE_TRAINING_CORPUS_FORMAT:
            raise ValueError("unsupported language training corpus format")
        train = payload.get("train", ())
        holdout = payload.get("holdout", ())
        if not isinstance(train, (list, tuple)) or not isinstance(holdout, (list, tuple)):
            raise ValueError("language training corpus splits must be sequences")
        return cls(
            train=tuple(LanguageTrainingExample.from_payload(dict(example)) for example in train),
            holdout=tuple(
                LanguageTrainingExample.from_payload(dict(example)) for example in holdout
            ),
        )


@dataclass(frozen=True)
class LanguageProviderArtifact:
    """Auditable external provider selection and rollback metadata."""

    artifact_id: str
    backend_id: str
    mode: str
    base_model: str
    adapter_path: str | None = None
    training_corpus: str | None = None
    training_report: str | None = None
    safety_report: str | None = None
    rollback_strategy: str = "disable-adapter"
    default_enabled: bool = False
    provenance: str = "external-provider"

    def __post_init__(self) -> None:
        for value, name in (
            (self.artifact_id, "artifact_id"),
            (self.backend_id, "backend_id"),
            (self.base_model, "base_model"),
            (self.rollback_strategy, "rollback_strategy"),
            (self.provenance, "provenance"),
        ):
            if not str(value):
                raise ValueError(f"provider artifact {name} cannot be empty")
        if self.mode not in {"raw", "lora", "guarded"}:
            raise ValueError("provider artifact mode must be raw, lora, or guarded")
        if self.mode in {"lora", "guarded"} and not str(self.adapter_path or ""):
            raise ValueError(f"provider artifact mode {self.mode} requires adapter_path")
        if self.mode == "raw" and self.adapter_path is not None:
            raise ValueError("raw provider artifact cannot carry adapter_path")
        if self.mode == "guarded" and self.default_enabled:
            raise ValueError("guarded provider artifacts must remain opt-in")
        for optional_value, name in (
            (self.adapter_path, "adapter_path"),
            (self.training_corpus, "training_corpus"),
            (self.training_report, "training_report"),
            (self.safety_report, "safety_report"),
        ):
            if optional_value is not None and not str(optional_value):
                raise ValueError(f"provider artifact {name} cannot be empty when provided")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_PROVIDER_ARTIFACT_FORMAT,
            "artifact_id": self.artifact_id,
            "backend_id": self.backend_id,
            "mode": self.mode,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "training_corpus": self.training_corpus,
            "training_report": self.training_report,
            "safety_report": self.safety_report,
            "rollback_strategy": self.rollback_strategy,
            "default_enabled": self.default_enabled,
            "provenance": self.provenance,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LanguageProviderArtifact:
        if payload.get("format") != LANGUAGE_PROVIDER_ARTIFACT_FORMAT:
            raise ValueError("unsupported language provider artifact format")
        return cls(
            artifact_id=str(payload["artifact_id"]),
            backend_id=str(payload["backend_id"]),
            mode=str(payload["mode"]),
            base_model=str(payload["base_model"]),
            adapter_path=(
                None if payload.get("adapter_path") is None else str(payload["adapter_path"])
            ),
            training_corpus=(
                None if payload.get("training_corpus") is None else str(payload["training_corpus"])
            ),
            training_report=(
                None if payload.get("training_report") is None else str(payload["training_report"])
            ),
            safety_report=(
                None if payload.get("safety_report") is None else str(payload["safety_report"])
            ),
            rollback_strategy=str(payload.get("rollback_strategy", "disable-adapter")),
            default_enabled=bool(payload.get("default_enabled", False)),
            provenance=str(payload.get("provenance", "external-provider")),
        )


@dataclass(frozen=True)
class LanguageBackendSpec:
    """Declarative backend metadata enforced at the effector boundary."""

    backend_id: str
    family: str
    training_contract: str
    supports_training: bool = True
    modalities: tuple[str, ...] = ("text",)
    owns_cognition: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.backend_id, "backend_id"),
            (self.family, "family"),
            (self.training_contract, "training_contract"),
        ):
            if not str(value):
                raise ValueError(f"language backend {name} cannot be empty")
        if not self.modalities or "text" not in self.modalities:
            raise ValueError("language backend must support text modality")
        if self.owns_cognition:
            raise ValueError("language backend cannot own Taiji cognition")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_BACKEND_SPEC_FORMAT,
            "backend_id": self.backend_id,
            "family": self.family,
            "training_contract": self.training_contract,
            "supports_training": self.supports_training,
            "modalities": list(self.modalities),
            "owns_cognition": self.owns_cognition,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LanguageBackendSpec:
        if payload.get("format") != LANGUAGE_BACKEND_SPEC_FORMAT:
            raise ValueError("unsupported language backend spec format")
        modalities = payload.get("modalities", ("text",))
        if not isinstance(modalities, (list, tuple)):
            raise ValueError("language backend modalities must be a sequence")
        return cls(
            backend_id=str(payload["backend_id"]),
            family=str(payload["family"]),
            training_contract=str(payload["training_contract"]),
            supports_training=bool(payload.get("supports_training", True)),
            modalities=tuple(str(modality) for modality in modalities),
            owns_cognition=bool(payload.get("owns_cognition", False)),
        )


class LanguageBackendRegistry:
    """Registry of replaceable text-organ descriptors, not cognitive modules."""

    def __init__(self, specs: tuple[LanguageBackendSpec, ...] = ()) -> None:
        self._specs: dict[str, LanguageBackendSpec] = {}
        for spec in specs:
            self.register(spec)

    @classmethod
    def default(cls) -> LanguageBackendRegistry:
        registry = cls()
        registry.register(
            LanguageBackendSpec(
                backend_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
                family="native-readable-surface",
                training_contract="none",
                supports_training=False,
            )
        )
        registry.register(
            LanguageBackendSpec(
                backend_id=StructuredTextLanguageOrgan.BACKEND_ID,
                family="structured-codec",
                training_contract="none",
                supports_training=False,
            )
        )
        return registry

    @property
    def specs(self) -> tuple[LanguageBackendSpec, ...]:
        return tuple(self._specs.values())

    def register(self, spec: LanguageBackendSpec) -> None:
        if not isinstance(spec, LanguageBackendSpec):
            raise TypeError("language backend registry accepts LanguageBackendSpec values")
        if spec.backend_id in self._specs:
            raise ValueError(f"language backend is already registered: {spec.backend_id}")
        self._specs[spec.backend_id] = spec

    def get(self, backend_id: str) -> LanguageBackendSpec:
        try:
            return self._specs[str(backend_id)]
        except KeyError as exc:
            raise KeyError(f"language backend is not registered: {backend_id}") from exc

    def validate(self, organ: LanguageOrgan) -> LanguageBackendSpec:
        if not isinstance(organ, LanguageOrgan):
            raise TypeError("organ must implement the LanguageOrgan protocol")
        spec = self.get(organ.backend_id)
        if "text" not in spec.modalities:
            raise ValueError("registered language backend does not support text")
        return spec

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_BACKEND_REGISTRY_FORMAT,
            "specs": [spec.to_payload() for spec in self.specs],
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> LanguageBackendRegistry:
        if payload.get("format") != LANGUAGE_BACKEND_REGISTRY_FORMAT:
            raise ValueError("unsupported language backend registry format")
        specs = payload.get("specs", ())
        if not isinstance(specs, (list, tuple)):
            raise ValueError("language backend registry specs must be a sequence")
        return cls(tuple(LanguageBackendSpec.from_payload(spec) for spec in specs))


@runtime_checkable
class LanguageOrgan(Protocol):
    """Replaceable terminal language-organ contract."""

    @property
    def backend_id(self) -> str:
        """Stable registry identifier for this organ."""

    def emit(self, expression: ExpressionPlan) -> LanguageEmission:
        """Render a Taiji-owned text expression without changing cognition."""

    def checkpoint(self) -> dict[str, Any]:
        """Return a backend descriptor suitable for a Taiji checkpoint."""


class LanguageRealizationValidator:
    """Taiji-owned semantic safety check for terminal text emissions."""

    def __init__(
        self,
        *,
        minimum_coverage: float = 1.0,
        reject_structured_leakage: bool = True,
    ) -> None:
        if not 0.0 <= float(minimum_coverage) <= 1.0:
            raise ValueError("minimum_coverage must be in [0, 1]")
        self.minimum_coverage = float(minimum_coverage)
        self.reject_structured_leakage = bool(reject_structured_leakage)

    def validate(
        self,
        expression: ExpressionPlan,
        text_bytes: bytes,
        *,
        required_terms: tuple[str, ...] = (),
    ) -> LanguageValidation:
        if not isinstance(expression, ExpressionPlan):
            raise TypeError("language validation requires an ExpressionPlan")
        if expression.modality != "text":
            raise ValueError("language validation requires a text ExpressionPlan")
        if not isinstance(text_bytes, bytes):
            raise TypeError("language validation text_bytes must be bytes")
        try:
            text = text_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return LanguageValidation(
                accepted=False,
                required_terms=tuple(required_terms),
                matched_terms=(),
                missing_terms=tuple(required_terms),
                coverage=0.0,
                reason="invalid-utf8",
            )
        terms = tuple(dict.fromkeys(str(term) for term in required_terms if str(term)))
        matched = tuple(term for term in terms if term in text)
        missing = tuple(term for term in terms if term not in text)
        coverage = len(matched) / max(1, len(terms))
        leaked = self.reject_structured_leakage and any(
            marker in text
            for marker in ("semantic_slots", "intent_kind", "expected_outcome", "{", "}")
        )
        accepted = coverage >= self.minimum_coverage and not leaked and bool(text.strip())
        if leaked:
            reason = "structured-leakage"
        elif coverage < self.minimum_coverage:
            reason = "missing-required-terms"
        elif not text.strip():
            reason = "empty-text"
        else:
            reason = "accepted"
        return LanguageValidation(
            accepted=accepted,
            required_terms=terms,
            matched_terms=matched,
            missing_terms=missing,
            coverage=coverage,
            reason=reason,
        )


def _readable_surface(value: Any) -> str | None:
    """Return a safe surface candidate, or ``None`` when it is not text."""

    if not isinstance(value, str):
        return None
    text = value.replace("\x00", "").strip()
    if not text or "\ufffd" in text:
        return None
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        return None
    if not any(char.isalnum() for char in text):
        return None
    return text


@runtime_checkable
class TextDecoder(Protocol):
    """Minimal external decoder surface accepted by Taiji."""

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        """Generate text from a terminal-organ prompt."""


class NativeReadableTextLanguageOrgan:
    """Native, dependency-free surface realizer for text expressions.

    The TSK byte predictor is allowed to remain a low-level compatibility
    path, but arbitrary predicted bytes must not be presented as language.
    This organ accepts a candidate surface string when one is available and
    otherwise returns a truthful, readable status message.  It does not
    invent goals or answer semantics; a mature decoder can replace it at the
    same terminal boundary.
    """

    BACKEND_ID = "native-readable"

    def __init__(self, *, max_bytes: int = 1_000_000) -> None:
        if int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = int(max_bytes)

    @property
    def backend_id(self) -> str:
        return self.BACKEND_ID

    @staticmethod
    def _readable(value: Any) -> str | None:
        return _readable_surface(value)

    @classmethod
    def _fallback_text(cls, expression: ExpressionPlan) -> str:
        fields = expression.fields
        slots = fields.get("semantic_slots", {})
        if not isinstance(slots, Mapping):
            slots = {}
        required_terms = tuple(str(term) for term in expression.required_terms if str(term))
        if required_terms:
            return "当前表达包含以下关键信息：" + "、".join(required_terms) + "。"
        prompt = cls._readable(slots.get("prompt"))
        if prompt:
            return f"我已收到你的问题：“{prompt}”。当前原生语言表层正在形成稳定表达。"
        return "Taiji 已完成内部处理，但当前还没有稳定的可读语言输出。"

    def emit(self, expression: ExpressionPlan) -> LanguageEmission:
        if not isinstance(expression, ExpressionPlan):
            raise TypeError("native readable organ requires an ExpressionPlan")
        if expression.modality != "text":
            raise ValueError("native readable organ only accepts text ExpressionPlan values")
        fields = expression.fields
        candidate = None
        if isinstance(fields, Mapping):
            for key in ("surface_text", "answer", "native_prediction"):
                candidate = self._readable(fields.get(key))
                if candidate is not None:
                    break
        text = candidate or self._fallback_text(expression)
        encoded = text.encode("utf-8")
        if len(encoded) > self.max_bytes:
            encoded = encoded[: self.max_bytes]
            text = encoded.decode("utf-8", errors="ignore").rstrip()
            if not text:
                raise ValueError("native readable expression exceeds max_bytes")
            encoded = text.encode("utf-8")
        return LanguageEmission(
            expression=expression,
            text_bytes=encoded,
            backend=self.backend_id,
            provenance="native-readable-surface",
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_ORGAN_CHECKPOINT_FORMAT,
            "backend": self.backend_id,
            "max_bytes": self.max_bytes,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeReadableTextLanguageOrgan:
        if payload.get("format") != LANGUAGE_ORGAN_CHECKPOINT_FORMAT:
            raise ValueError("unsupported language organ checkpoint format")
        if payload.get("backend") != cls.BACKEND_ID:
            raise ValueError("unsupported native readable language organ backend")
        return cls(max_bytes=int(payload.get("max_bytes", 1_000_000)))


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


class ValidatedLanguageOrgan:
    """Guard an external organ and fall back when semantic safety fails."""

    def __init__(
        self,
        primary: LanguageOrgan,
        *,
        fallback: LanguageOrgan | None = None,
        validator: LanguageRealizationValidator | None = None,
        required_terms_builder: Callable[[ExpressionPlan], tuple[str, ...]] | None = None,
    ) -> None:
        if not isinstance(primary, LanguageOrgan):
            raise TypeError("primary must implement the LanguageOrgan protocol")
        selected_fallback = fallback or NativeReadableTextLanguageOrgan()
        if not isinstance(selected_fallback, LanguageOrgan):
            raise TypeError("fallback must implement the LanguageOrgan protocol")
        if validator is not None and not isinstance(validator, LanguageRealizationValidator):
            raise TypeError("validator must be a LanguageRealizationValidator or None")
        if required_terms_builder is not None and not callable(required_terms_builder):
            raise TypeError("required_terms_builder must be callable or None")
        self.primary = primary
        self.fallback = selected_fallback
        self.validator = validator or LanguageRealizationValidator()
        self.required_terms_builder = required_terms_builder

    @property
    def backend_id(self) -> str:
        return self.primary.backend_id

    def emit(self, expression: ExpressionPlan) -> LanguageEmission:
        candidate = self.primary.emit(expression)
        required_terms = (
            expression.required_terms
            if self.required_terms_builder is None
            else tuple(self.required_terms_builder(expression))
        )
        validation = self.validator.validate(
            expression,
            candidate.text_bytes,
            required_terms=required_terms,
        )
        if validation.accepted:
            return replace(candidate, validation=validation, fallback_used=False)
        fallback = self.fallback.emit(expression)
        return replace(
            fallback,
            provenance="validated-fallback",
            validation=validation,
            fallback_used=True,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_ORGAN_CHECKPOINT_FORMAT,
            "backend": self.backend_id,
            "wrapper": "validated-language-organ",
            "primary": self.primary.checkpoint(),
            "fallback": self.fallback.checkpoint(),
            "minimum_coverage": self.validator.minimum_coverage,
            "reject_structured_leakage": self.validator.reject_structured_leakage,
        }


class ExternalTextDecoderLanguageOrgan:
    """Adapt a mature external decoder without importing it into Taiji.

    The caller supplies both the decoder and the expression-to-prompt codec.
    This keeps tokenization, model loading, device placement, and decoder
    training outside Taiji while preserving a strict one-way effector input.
    """

    def __init__(
        self,
        decoder: TextDecoder,
        *,
        prompt_builder: Callable[[ExpressionPlan], str],
        backend_id: str = "mature-decoder-v1",
        max_tokens: int = 128,
        temperature: float = 0.2,
        prompt_contract: str = "expression-to-text-v1",
    ) -> None:
        if not isinstance(decoder, TextDecoder):
            raise TypeError("decoder must implement the TextDecoder protocol")
        if not callable(prompt_builder):
            raise TypeError("prompt_builder must be callable")
        if not str(backend_id):
            raise ValueError("external decoder backend_id cannot be empty")
        if int(max_tokens) <= 0:
            raise ValueError("external decoder max_tokens must be positive")
        if float(temperature) < 0.0:
            raise ValueError("external decoder temperature cannot be negative")
        if not str(prompt_contract):
            raise ValueError("external decoder prompt_contract cannot be empty")
        self.decoder = decoder
        self.prompt_builder = prompt_builder
        self._backend_id = str(backend_id)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.prompt_contract = str(prompt_contract)

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def emit(self, expression: ExpressionPlan) -> LanguageEmission:
        if not isinstance(expression, ExpressionPlan):
            raise TypeError("external decoder requires an ExpressionPlan")
        if expression.modality != "text":
            raise ValueError("external decoder only accepts text ExpressionPlan values")
        prompt = self.prompt_builder(expression)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt_builder must return non-empty text")
        text = self.decoder.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if not isinstance(text, str) or not text:
            raise ValueError("external decoder must return non-empty text")
        return LanguageEmission(
            expression=expression,
            text_bytes=text.encode("utf-8"),
            backend=self.backend_id,
            provenance="external-decoder",
        )

    def checkpoint(self) -> dict[str, Any]:
        """Persist only the boundary config; model state stays external."""

        return {
            "format": LANGUAGE_ORGAN_CHECKPOINT_FORMAT,
            "backend": self.backend_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "prompt_contract": self.prompt_contract,
            "model_state": "external",
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        decoder: TextDecoder,
        *,
        prompt_builder: Callable[[ExpressionPlan], str],
    ) -> ExternalTextDecoderLanguageOrgan:
        if payload.get("format") != LANGUAGE_ORGAN_CHECKPOINT_FORMAT:
            raise ValueError("unsupported external language organ checkpoint format")
        if payload.get("model_state") != "external":
            raise ValueError(
                "external language organ checkpoint must reference external model state"
            )
        return cls(
            decoder,
            prompt_builder=prompt_builder,
            backend_id=str(payload["backend"]),
            max_tokens=int(payload.get("max_tokens", 128)),
            temperature=float(payload.get("temperature", 0.2)),
            prompt_contract=str(payload.get("prompt_contract", "expression-to-text-v1")),
        )


class LanguageRealizationGate:
    """Admission gate for an ``ExpressionPlan`` to real language output.

    The gate is deliberately an evaluator, not a trainer.  It measures a
    candidate language organ on Taiji-owned train/holdout examples and only
    passes when readable output, required semantic terms, exact rollback, and
    checkpoint continuation are all demonstrated.  External model state is
    supplied through ``checkpoint_loader`` and therefore never enters Taiji
    cognition or its native checkpoint.
    """

    def __init__(
        self,
        *,
        minimum_required_term_coverage: float = 1.0,
        minimum_readable_rate: float = 1.0,
        maximum_fallback_rate: float = 0.0,
    ) -> None:
        for value, name in (
            (minimum_required_term_coverage, "minimum_required_term_coverage"),
            (minimum_readable_rate, "minimum_readable_rate"),
            (maximum_fallback_rate, "maximum_fallback_rate"),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.minimum_required_term_coverage = float(minimum_required_term_coverage)
        self.minimum_readable_rate = float(minimum_readable_rate)
        self.maximum_fallback_rate = float(maximum_fallback_rate)
        self._validator = LanguageRealizationValidator(
            minimum_coverage=self.minimum_required_term_coverage
        )

    @staticmethod
    def _structured_leakage(text: str) -> bool:
        return any(
            marker in text
            for marker in ("semantic_slots", "intent_kind", "expected_outcome", "{", "}")
        )

    def _measure_split(
        self,
        organ: LanguageOrgan,
        examples: tuple[LanguageTrainingExample, ...],
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for example in examples:
            row: dict[str, Any] = {
                "example_id": example.example_id,
                "output_text": "",
                "output_nonempty": False,
                "readable": False,
                "required_terms": list(example.expression.required_terms),
                "matched_terms": [],
                "required_term_coverage": 0.0,
                "structured_leakage": False,
                "fallback_used": False,
                "accepted": False,
                "error": None,
            }
            try:
                emission = organ.emit(example.expression)
                text = emission.text_bytes.decode("utf-8", errors="strict")
                readable = _readable_surface(text) is not None
                terms = tuple(example.expression.required_terms)
                matched = tuple(term for term in terms if term in text)
                validation = self._validator.validate(
                    example.expression,
                    emission.text_bytes,
                    required_terms=terms,
                )
                leakage = self._structured_leakage(text)
                row.update(
                    {
                        "output_text": text,
                        "output_nonempty": bool(text.strip()),
                        "readable": readable,
                        "matched_terms": list(matched),
                        "required_term_coverage": len(matched) / max(1, len(terms)),
                        "structured_leakage": leakage,
                        "fallback_used": bool(emission.fallback_used),
                        "accepted": bool(
                            readable
                            and validation.accepted
                            and not leakage
                            and not emission.fallback_used
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - evaluator records failed cases
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

        count = max(1, len(rows))
        output_nonempty_rate = sum(bool(row["output_nonempty"]) for row in rows) / count
        readable_rate = sum(bool(row["readable"]) for row in rows) / count
        term_coverage = sum(float(row["required_term_coverage"]) for row in rows) / count
        leakage_free_rate = sum(not bool(row["structured_leakage"]) for row in rows) / count
        fallback_rate = sum(bool(row["fallback_used"]) for row in rows) / count
        accepted_rate = sum(bool(row["accepted"]) for row in rows) / count
        passed = bool(
            output_nonempty_rate == 1.0
            and readable_rate >= self.minimum_readable_rate
            and term_coverage >= self.minimum_required_term_coverage
            and leakage_free_rate == 1.0
            and fallback_rate <= self.maximum_fallback_rate
            and accepted_rate == 1.0
            and all(row["error"] is None for row in rows)
        )
        return {
            "examples": rows,
            "output_nonempty_rate": output_nonempty_rate,
            "readable_rate": readable_rate,
            "required_term_coverage": term_coverage,
            "structured_leakage_free_rate": leakage_free_rate,
            "fallback_rate": fallback_rate,
            "accepted_rate": accepted_rate,
            "passed": passed,
        }

    @staticmethod
    def _outputs(
        organ: LanguageOrgan,
        examples: tuple[LanguageTrainingExample, ...],
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        outputs: dict[str, str] = {}
        errors: list[str] = []
        for example in examples:
            try:
                outputs[example.example_id] = organ.emit(example.expression).text_bytes.decode(
                    "utf-8", errors="strict"
                )
            except Exception as exc:  # noqa: BLE001 - evaluator records failed cases
                errors.append(f"{example.example_id}: {type(exc).__name__}: {exc}")
        return outputs, tuple(errors)

    def evaluate(
        self,
        organ: LanguageOrgan,
        corpus: LanguageTrainingCorpus,
        *,
        rollback_organ: LanguageOrgan | None = None,
        rollback_reference_organ: LanguageOrgan | None = None,
        checkpoint_loader: Callable[[Mapping[str, Any]], LanguageOrgan] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(organ, LanguageOrgan):
            raise TypeError("language realization gate organ must implement LanguageOrgan")
        if not isinstance(corpus, LanguageTrainingCorpus):
            raise TypeError("language realization gate corpus must be LanguageTrainingCorpus")

        restored_corpus = LanguageTrainingCorpus.from_payload(corpus.to_payload())
        corpus_round_trip = restored_corpus == corpus
        train_ids = {example.example_id for example in corpus.train}
        holdout_ids = {example.example_id for example in corpus.holdout}
        train_expression_ids = {example.expression.expression_id for example in corpus.train}
        holdout_expression_ids = {example.expression.expression_id for example in corpus.holdout}
        split_disjoint = train_ids.isdisjoint(holdout_ids) and train_expression_ids.isdisjoint(
            holdout_expression_ids
        )

        train = self._measure_split(organ, corpus.train)
        holdout = self._measure_split(organ, corpus.holdout)

        rollback = {
            "checked": rollback_organ is not None and rollback_reference_organ is not None,
            "outputs_match_reference": False,
            "errors": [],
        }
        if rollback["checked"]:
            actual, actual_errors = self._outputs(rollback_organ, corpus.holdout)  # type: ignore[arg-type]
            reference, reference_errors = self._outputs(
                rollback_reference_organ, corpus.holdout  # type: ignore[arg-type]
            )
            rollback["outputs_match_reference"] = bool(
                not actual_errors and not reference_errors and actual == reference
            )
            rollback["errors"] = list(actual_errors + reference_errors)

        checkpoint = {
            "checked": checkpoint_loader is not None,
            "outputs_match": False,
            "errors": [],
        }
        if checkpoint_loader is not None:
            try:
                payload = organ.checkpoint()
                restored = checkpoint_loader(payload)
                if not isinstance(restored, LanguageOrgan):
                    raise TypeError("checkpoint_loader did not return a LanguageOrgan")
                original, original_errors = self._outputs(organ, corpus.holdout)
                resumed, resumed_errors = self._outputs(restored, corpus.holdout)
                checkpoint["outputs_match"] = bool(
                    not original_errors and not resumed_errors and original == resumed
                )
                checkpoint["errors"] = list(original_errors + resumed_errors)
            except Exception as exc:  # noqa: BLE001 - evaluator records failed gate
                checkpoint["errors"] = [f"{type(exc).__name__}: {exc}"]

        passed = bool(
            corpus_round_trip
            and split_disjoint
            and bool(train["passed"])
            and bool(holdout["passed"])
            and bool(rollback["checked"])
            and bool(rollback["outputs_match_reference"])
            and bool(checkpoint["checked"])
            and bool(checkpoint["outputs_match"])
        )
        return {
            "format": LANGUAGE_REALIZATION_GATE_FORMAT,
            "corpus": {
                "train_examples": len(corpus.train),
                "holdout_examples": len(corpus.holdout),
                "round_trip": corpus_round_trip,
                "split_disjoint": split_disjoint,
            },
            "train": train,
            "holdout": holdout,
            "rollback": rollback,
            "checkpoint": checkpoint,
            "thresholds": {
                "minimum_required_term_coverage": self.minimum_required_term_coverage,
                "minimum_readable_rate": self.minimum_readable_rate,
                "maximum_fallback_rate": self.maximum_fallback_rate,
            },
            "gate": {
                "passed": passed,
                "criterion": "train and holdout ExpressionPlan values produce readable, semantically complete, non-structured text without fallback, while rollback and checkpoint continuation reproduce their references",
            },
        }

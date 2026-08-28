"""Terminal language-organ boundary for Taiji expression plans.

The language organ is an effector.  It receives an already selected
``ExpressionPlan`` and returns text bytes; it does not own goals, memory,
planning, content selection, or ``ActionIntent`` creation.  A mature decoder
can implement the same protocol later without entering Taiji's cognitive
core.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .generation import ExpressionPlan, TextExpressionCodec

LANGUAGE_ORGAN_CHECKPOINT_FORMAT = "taiji-language-organ-v1"
LANGUAGE_TRAINING_EXAMPLE_FORMAT = "taiji-language-training-example-v1"
LANGUAGE_TRAINING_CORPUS_FORMAT = "taiji-language-training-corpus-v1"
LANGUAGE_BACKEND_SPEC_FORMAT = "taiji-language-backend-spec-v1"
LANGUAGE_BACKEND_REGISTRY_FORMAT = "taiji-language-backend-registry-v1"
LANGUAGE_VALIDATION_FORMAT = "taiji-language-validation-v1"
LANGUAGE_PROVIDER_ARTIFACT_FORMAT = "taiji-language-provider-artifact-v1"
LANGUAGE_PROVIDER_CONTENT_ADDRESS_FORMAT = "taiji-language-provider-content-address-v1"
LANGUAGE_PROVIDER_CANARY_FORMAT = "taiji-language-provider-canary-v1"
LANGUAGE_PROVIDER_REGISTRY_FORMAT = "taiji-language-provider-registry-v1"
LANGUAGE_PROVIDER_HEALTH_FORMAT = "taiji-language-provider-health-v1"
LANGUAGE_PROVIDER_HEALTH_GATE_FORMAT = "taiji-language-provider-health-gate-v1"
LANGUAGE_REALIZATION_GATE_FORMAT = "taiji-language-realization-gate-v1"

_LANGUAGE_PROVIDER_CONTENT_ROLES = frozenset(
    {"base_model", "adapter", "training_corpus", "training_report", "safety_report"}
)


def _hash_part(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big", signed=False))
    hasher.update(value)


def _hash_file_content(hasher: Any, path: Path) -> None:
    hasher.update(path.stat().st_size.to_bytes(8, "big", signed=False))
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)


def language_provider_content_digest(path: str | Path) -> str:
    """Return a path-independent SHA-256 digest for a provider asset.

    Files are addressed by their bytes.  Directories are addressed by the
    sorted relative POSIX names, sizes, and bytes of every regular file.  No
    absolute path, mtime, or filesystem ordering enters the digest, so an
    artifact can be relocated without becoming a different artifact.
    """

    asset = Path(path)
    if asset.is_symlink():
        raise ValueError(f"provider content address does not accept symlinks: {asset}")
    if not asset.exists():
        raise FileNotFoundError(f"provider content address target not found: {asset}")

    hasher = hashlib.sha256()
    _hash_part(hasher, LANGUAGE_PROVIDER_CONTENT_ADDRESS_FORMAT.encode("utf-8"))
    if asset.is_file():
        _hash_part(hasher, b"file")
        _hash_file_content(hasher, asset)
    elif asset.is_dir():
        _hash_part(hasher, b"directory")
        entries = sorted(asset.rglob("*"), key=lambda item: item.relative_to(asset).as_posix())
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"provider content address does not accept symlinks: {entry}")
            if not entry.is_file():
                continue
            _hash_part(hasher, entry.relative_to(asset).as_posix().encode("utf-8"))
            _hash_file_content(hasher, entry)
    else:
        raise ValueError(f"provider content address target must be a file or directory: {asset}")
    return hasher.hexdigest()


def _provider_artifact_digest(
    *,
    artifact_id: str,
    backend_id: str,
    mode: str,
    content_address_format: str,
    canary_id: str,
    rollback_strategy: str,
    default_enabled: bool,
    provenance: str,
    content_digests: tuple[tuple[str, str], ...],
    expires_at: float | None,
) -> str:
    payload = {
        "format": content_address_format,
        "artifact_id": artifact_id,
        "backend_id": backend_id,
        "mode": mode,
        "canary_id": canary_id,
        "rollback_strategy": rollback_strategy,
        "default_enabled": bool(default_enabled),
        "provenance": provenance,
        "content_digests": {role: digest for role, digest in content_digests},
        "expires_at": None if expires_at is None else float(expires_at),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    content_digests: tuple[tuple[str, str], ...] = ()
    artifact_digest: str = ""
    expires_at: float | None = None
    canary_id: str = LANGUAGE_PROVIDER_CANARY_FORMAT
    content_address_format: str = LANGUAGE_PROVIDER_CONTENT_ADDRESS_FORMAT

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
        if self.content_address_format != LANGUAGE_PROVIDER_CONTENT_ADDRESS_FORMAT:
            raise ValueError("unsupported provider artifact content address format")
        if not str(self.canary_id):
            raise ValueError("provider artifact canary_id cannot be empty")
        if self.expires_at is not None:
            if isinstance(self.expires_at, bool) or not math.isfinite(float(self.expires_at)):
                raise ValueError("provider artifact expires_at must be finite")
            object.__setattr__(self, "expires_at", float(self.expires_at))

        if isinstance(self.content_digests, (str, bytes, bytearray)):
            raise TypeError("provider artifact content_digests must be role/digest pairs")
        normalized_digests: list[tuple[str, str]] = []
        for item in self.content_digests:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("provider artifact content_digests must contain role/digest pairs")
            role, digest = str(item[0]), str(item[1]).lower()
            if role not in _LANGUAGE_PROVIDER_CONTENT_ROLES:
                raise ValueError(f"unsupported provider artifact content role: {role}")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"provider artifact digest for {role} must be SHA-256 hex")
            normalized_digests.append((role, digest))
        if len({role for role, _ in normalized_digests}) != len(normalized_digests):
            raise ValueError("provider artifact content roles must be unique")
        normalized_digests.sort()
        normalized = tuple(normalized_digests)
        object.__setattr__(self, "content_digests", normalized)

        if self.artifact_digest:
            artifact_digest = str(self.artifact_digest).lower()
            if len(artifact_digest) != 64 or any(
                char not in "0123456789abcdef" for char in artifact_digest
            ):
                raise ValueError("provider artifact artifact_digest must be SHA-256 hex")
            if not normalized:
                raise ValueError("provider artifact artifact_digest requires content_digests")
            expected = _provider_artifact_digest(
                artifact_id=self.artifact_id,
                backend_id=self.backend_id,
                mode=self.mode,
                content_address_format=self.content_address_format,
                canary_id=self.canary_id,
                rollback_strategy=self.rollback_strategy,
                default_enabled=self.default_enabled,
                provenance=self.provenance,
                content_digests=normalized,
                expires_at=self.expires_at,
            )
            if artifact_digest != expected:
                raise ValueError("provider artifact artifact_digest does not match its manifest")
            object.__setattr__(self, "artifact_digest", artifact_digest)
        elif normalized:
            object.__setattr__(
                self,
                "artifact_digest",
                _provider_artifact_digest(
                    artifact_id=self.artifact_id,
                    backend_id=self.backend_id,
                    mode=self.mode,
                    content_address_format=self.content_address_format,
                    canary_id=self.canary_id,
                    rollback_strategy=self.rollback_strategy,
                    default_enabled=self.default_enabled,
                    provenance=self.provenance,
                    content_digests=normalized,
                    expires_at=self.expires_at,
                ),
            )

        if normalized:
            required_roles = {"base_model"}
            if self.mode in {"lora", "guarded"}:
                required_roles.add("adapter")
            if self.mode in {"lora", "guarded"}:
                required_roles.update({"training_corpus", "training_report"})
            if self.mode == "guarded":
                required_roles.add("safety_report")
            missing_roles = required_roles.difference(dict(normalized))
            if missing_roles:
                raise ValueError(
                    "provider artifact content_digests missing roles: "
                    + ", ".join(sorted(missing_roles))
                )

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
            "content_address_format": self.content_address_format,
            "content_digests": {role: digest for role, digest in self.content_digests},
            "artifact_digest": self.artifact_digest,
            "expires_at": self.expires_at,
            "canary_id": self.canary_id,
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
            content_digests=_provider_digest_pairs(payload.get("content_digests", {})),
            artifact_digest=str(payload.get("artifact_digest", "")),
            expires_at=(
                None if payload.get("expires_at") is None else float(payload["expires_at"])
            ),
            canary_id=str(payload.get("canary_id", LANGUAGE_PROVIDER_CANARY_FORMAT)),
            content_address_format=str(
                payload.get("content_address_format", LANGUAGE_PROVIDER_CONTENT_ADDRESS_FORMAT)
            ),
        )


def _provider_digest_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        return tuple((str(role), str(digest)) for role, digest in value.items())
    if isinstance(value, (list, tuple)):
        pairs: list[tuple[str, str]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("provider artifact content_digests must contain role/digest pairs")
            pairs.append((str(item[0]), str(item[1])))
        return tuple(pairs)
    raise ValueError("provider artifact content_digests must be a mapping or sequence")


def language_provider_artifact_digest(artifact: LanguageProviderArtifact) -> str:
    """Recompute the path-independent digest of an artifact manifest."""

    if not isinstance(artifact, LanguageProviderArtifact):
        raise TypeError("language provider artifact digest requires a LanguageProviderArtifact")
    return _provider_artifact_digest(
        artifact_id=artifact.artifact_id,
        backend_id=artifact.backend_id,
        mode=artifact.mode,
        content_address_format=artifact.content_address_format,
        canary_id=artifact.canary_id,
        rollback_strategy=artifact.rollback_strategy,
        default_enabled=artifact.default_enabled,
        provenance=artifact.provenance,
        content_digests=artifact.content_digests,
        expires_at=artifact.expires_at,
    )


@dataclass(frozen=True)
class LanguageProviderArtifactRegistry:
    """Versioned provider manifests with an explicit activation allowlist.

    The registry is metadata only: it does not load a decoder or participate in
    Taiji cognition.  Every state transition returns a new snapshot so a
    runtime can validate and stage a candidate before publishing the snapshot
    that marks it active.
    """

    artifacts: tuple[LanguageProviderArtifact, ...] = ()
    allowed_artifact_ids: tuple[str, ...] = ()
    active_artifact_id: str | None = None
    previous_artifact_id: str | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.artifacts, (str, bytes, bytearray)):
            raise TypeError("provider artifact registry artifacts must be a sequence")
        normalized_artifacts = tuple(sorted(self.artifacts, key=lambda item: item.artifact_id))
        if any(not isinstance(item, LanguageProviderArtifact) for item in normalized_artifacts):
            raise TypeError("provider artifact registry accepts LanguageProviderArtifact values")
        artifact_ids = tuple(item.artifact_id for item in normalized_artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("provider artifact registry artifact_ids must be unique")
        allowed = tuple(sorted(dict.fromkeys(str(item) for item in self.allowed_artifact_ids)))
        unknown_allowed = set(allowed).difference(artifact_ids)
        if unknown_allowed:
            raise ValueError(
                "provider artifact registry allowlist contains unknown artifacts: "
                + ", ".join(sorted(unknown_allowed))
            )
        for selected, label in (
            (self.active_artifact_id, "active_artifact_id"),
            (self.previous_artifact_id, "previous_artifact_id"),
        ):
            if selected is not None and selected not in artifact_ids:
                raise ValueError(f"provider artifact registry {label} is unknown: {selected}")
            if selected is not None and selected not in allowed:
                raise ValueError(
                    f"provider artifact registry {label} is not allowlisted: {selected}"
                )
        if isinstance(self.revision, bool) or int(self.revision) < 0:
            raise ValueError("provider artifact registry revision must be non-negative")
        object.__setattr__(self, "artifacts", normalized_artifacts)
        object.__setattr__(self, "allowed_artifact_ids", allowed)
        object.__setattr__(self, "revision", int(self.revision))

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        """Return deterministic version identifiers in this registry."""

        return tuple(artifact.artifact_id for artifact in self.artifacts)

    @property
    def active_artifact(self) -> LanguageProviderArtifact | None:
        """Return the active manifest, if the registry has selected one."""

        return self.get(self.active_artifact_id) if self.active_artifact_id else None

    @property
    def previous_artifact(self) -> LanguageProviderArtifact | None:
        """Return the previous manifest retained for an explicit rollback."""

        return self.get(self.previous_artifact_id) if self.previous_artifact_id else None

    def get(self, artifact_id: str | None) -> LanguageProviderArtifact | None:
        if artifact_id is None:
            return None
        for artifact in self.artifacts:
            if artifact.artifact_id == str(artifact_id):
                return artifact
        return None

    def require_allowed(self, artifact: LanguageProviderArtifact) -> None:
        """Require the same path-independent manifest identity before staging."""

        if not isinstance(artifact, LanguageProviderArtifact):
            raise TypeError("provider artifact registry requires a LanguageProviderArtifact")
        registered = self.get(artifact.artifact_id)
        if registered is None:
            raise KeyError(f"provider artifact is not registered: {artifact.artifact_id}")
        if language_provider_artifact_digest(registered) != language_provider_artifact_digest(
            artifact
        ):
            raise ValueError(
                f"provider artifact manifest does not match registered version: {artifact.artifact_id}"
            )
        if artifact.artifact_id not in self.allowed_artifact_ids:
            raise PermissionError(f"provider artifact is not allowlisted: {artifact.artifact_id}")

    def with_artifact(
        self,
        artifact: LanguageProviderArtifact,
        *,
        allow: bool = False,
    ) -> LanguageProviderArtifactRegistry:
        """Return a new registry snapshot containing one immutable manifest."""

        if not isinstance(artifact, LanguageProviderArtifact):
            raise TypeError("provider artifact registry accepts LanguageProviderArtifact values")
        existing = self.get(artifact.artifact_id)
        if existing is not None and language_provider_artifact_digest(
            existing
        ) != language_provider_artifact_digest(artifact):
            raise ValueError(
                f"provider artifact version already has a different manifest: {artifact.artifact_id}"
            )
        artifacts = (
            tuple(
                artifact if item.artifact_id == artifact.artifact_id else item
                for item in self.artifacts
            )
            if existing is not None
            else (*self.artifacts, artifact)
        )
        allowed = self.allowed_artifact_ids
        if allow and artifact.artifact_id not in allowed:
            allowed = (*allowed, artifact.artifact_id)
        return replace(
            self,
            artifacts=artifacts,
            allowed_artifact_ids=allowed,
            revision=self.revision + 1,
        )

    def allow(self, artifact_id: str) -> LanguageProviderArtifactRegistry:
        """Return a snapshot that explicitly permits one registered version."""

        artifact = self.get(artifact_id)
        if artifact is None:
            raise KeyError(f"provider artifact is not registered: {artifact_id}")
        if artifact.artifact_id in self.allowed_artifact_ids:
            return self
        return replace(
            self,
            allowed_artifact_ids=(*self.allowed_artifact_ids, artifact.artifact_id),
            revision=self.revision + 1,
        )

    def activate(self, artifact_id: str) -> LanguageProviderArtifactRegistry:
        """Select an allowlisted version after an external runtime Gate passes."""

        artifact = self.get(artifact_id)
        if artifact is None:
            raise KeyError(f"provider artifact is not registered: {artifact_id}")
        if artifact.artifact_id not in self.allowed_artifact_ids:
            raise PermissionError(f"provider artifact is not allowlisted: {artifact.artifact_id}")
        if artifact.artifact_id == self.active_artifact_id:
            return self
        return replace(
            self,
            active_artifact_id=artifact.artifact_id,
            previous_artifact_id=self.active_artifact_id,
            revision=self.revision + 1,
        )

    def rollback(self) -> LanguageProviderArtifactRegistry:
        """Swap active and previous versions without deleting either manifest."""

        if self.previous_artifact_id is None:
            raise ValueError("provider artifact registry has no previous version to roll back to")
        return replace(
            self,
            active_artifact_id=self.previous_artifact_id,
            previous_artifact_id=self.active_artifact_id,
            revision=self.revision + 1,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_PROVIDER_REGISTRY_FORMAT,
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
            "allowed_artifact_ids": list(self.allowed_artifact_ids),
            "active_artifact_id": self.active_artifact_id,
            "previous_artifact_id": self.previous_artifact_id,
            "revision": self.revision,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> LanguageProviderArtifactRegistry:
        if payload.get("format") != LANGUAGE_PROVIDER_REGISTRY_FORMAT:
            raise ValueError("unsupported language provider artifact registry format")
        artifacts = payload.get("artifacts", ())
        allowed = payload.get("allowed_artifact_ids", ())
        if not isinstance(artifacts, (list, tuple)):
            raise ValueError("provider artifact registry artifacts must be a sequence")
        if not isinstance(allowed, (list, tuple)):
            raise ValueError("provider artifact registry allowlist must be a sequence")
        return cls(
            artifacts=tuple(
                LanguageProviderArtifact.from_payload(dict(artifact)) for artifact in artifacts
            ),
            allowed_artifact_ids=tuple(str(item) for item in allowed),
            active_artifact_id=(
                None
                if payload.get("active_artifact_id") is None
                else str(payload["active_artifact_id"])
            ),
            previous_artifact_id=(
                None
                if payload.get("previous_artifact_id") is None
                else str(payload["previous_artifact_id"])
            ),
            revision=int(payload.get("revision", 0)),
        )


@dataclass(frozen=True)
class LanguageProviderHealthPolicy:
    """Deterministic thresholds for the provider runtime health watchdog.

    The policy is data only.  It never loads a decoder and never decides which
    artifact is allowed; it only bounds how many consecutive rejected probes an
    active provider may produce before the runtime must fall back, and how long
    the runtime must stay quiet afterwards so a rollback cannot flap.
    """

    failure_threshold: int = 3
    cooldown_seconds: float = 300.0
    minimum_accepted_rate: float = 0.5
    minimum_rate_probes: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.failure_threshold, bool) or int(self.failure_threshold) < 1:
            raise ValueError("provider health failure_threshold must be a positive integer")
        cooldown = float(self.cooldown_seconds)
        if not math.isfinite(cooldown) or cooldown < 0.0:
            raise ValueError("provider health cooldown_seconds must be finite and non-negative")
        rate = float(self.minimum_accepted_rate)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("provider health minimum_accepted_rate must fall within [0, 1]")
        if isinstance(self.minimum_rate_probes, bool) or int(self.minimum_rate_probes) < 1:
            raise ValueError("provider health minimum_rate_probes must be a positive integer")
        object.__setattr__(self, "failure_threshold", int(self.failure_threshold))
        object.__setattr__(self, "cooldown_seconds", cooldown)
        object.__setattr__(self, "minimum_accepted_rate", rate)
        object.__setattr__(self, "minimum_rate_probes", int(self.minimum_rate_probes))

    def to_payload(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "minimum_accepted_rate": self.minimum_accepted_rate,
            "minimum_rate_probes": self.minimum_rate_probes,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LanguageProviderHealthPolicy:
        return cls(
            failure_threshold=int(payload.get("failure_threshold", 3)),
            cooldown_seconds=float(payload.get("cooldown_seconds", 300.0)),
            minimum_accepted_rate=float(payload.get("minimum_accepted_rate", 0.5)),
            minimum_rate_probes=int(payload.get("minimum_rate_probes", 8)),
        )


@dataclass(frozen=True)
class LanguageProviderHealthState:
    """Immutable runtime health record for one active provider version.

    The record is anchored on ``artifact_id``: observing a different version
    starts a fresh record, so counters can never be inherited across a
    rotation or a rollback.  Every transition returns a new snapshot, and the
    whole record is checkpointable so a degraded provider stays degraded across
    a restart instead of silently becoming healthy again.
    """

    artifact_id: str | None = None
    probe_count: int = 0
    accepted_count: int = 0
    consecutive_failures: int = 0
    rollback_count: int = 0
    degraded: bool = False
    rollback_pending: bool = False
    cooldown_until: float = 0.0
    last_reason_code: str = ""
    last_probe_at: float | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.probe_count, "probe_count"),
            (self.accepted_count, "accepted_count"),
            (self.consecutive_failures, "consecutive_failures"),
            (self.rollback_count, "rollback_count"),
        ):
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"provider health {label} must be non-negative")
        if int(self.accepted_count) > int(self.probe_count):
            raise ValueError("provider health accepted_count cannot exceed probe_count")
        cooldown_until = float(self.cooldown_until)
        if not math.isfinite(cooldown_until) or cooldown_until < 0.0:
            raise ValueError("provider health cooldown_until must be finite and non-negative")
        if self.last_probe_at is not None and not math.isfinite(float(self.last_probe_at)):
            raise ValueError("provider health last_probe_at must be finite")
        if self.rollback_pending and not self.degraded:
            raise ValueError("provider health rollback_pending requires a degraded record")
        object.__setattr__(
            self, "artifact_id", None if self.artifact_id is None else str(self.artifact_id)
        )
        object.__setattr__(self, "probe_count", int(self.probe_count))
        object.__setattr__(self, "accepted_count", int(self.accepted_count))
        object.__setattr__(self, "consecutive_failures", int(self.consecutive_failures))
        object.__setattr__(self, "rollback_count", int(self.rollback_count))
        object.__setattr__(self, "degraded", bool(self.degraded))
        object.__setattr__(self, "rollback_pending", bool(self.rollback_pending))
        object.__setattr__(self, "cooldown_until", cooldown_until)
        object.__setattr__(self, "last_reason_code", str(self.last_reason_code))
        object.__setattr__(
            self, "last_probe_at", None if self.last_probe_at is None else float(self.last_probe_at)
        )

    @property
    def accepted_rate(self) -> float:
        """Return the accepted probe rate for this version, 1.0 when unprobed."""

        if self.probe_count == 0:
            return 1.0
        return self.accepted_count / self.probe_count

    def in_cooldown(self, now: float) -> bool:
        """Return whether the watchdog is still inside its quiet window."""

        return float(now) < self.cooldown_until

    def for_artifact(self, artifact_id: str | None) -> LanguageProviderHealthState:
        """Return this record when the version matches, else a fresh record.

        The cooldown window and rollback tally survive the rebase so a
        post-rollback quiet window cannot be erased by pointing the watchdog at
        another version.
        """

        target = None if artifact_id is None else str(artifact_id)
        if target == self.artifact_id:
            return self
        return LanguageProviderHealthState(
            artifact_id=target,
            rollback_count=self.rollback_count,
            cooldown_until=self.cooldown_until,
        )

    def observe(
        self,
        *,
        artifact_id: str | None,
        accepted: bool,
        reason_code: str,
        now: float,
        policy: LanguageProviderHealthPolicy,
    ) -> LanguageProviderHealthState:
        """Fold one probe outcome into a new record."""

        if not isinstance(policy, LanguageProviderHealthPolicy):
            raise TypeError("provider health observation requires a LanguageProviderHealthPolicy")
        base = self.for_artifact(artifact_id)
        probe_count = base.probe_count + 1
        accepted_count = base.accepted_count + (1 if accepted else 0)
        consecutive_failures = 0 if accepted else base.consecutive_failures + 1
        threshold_breached = consecutive_failures >= policy.failure_threshold
        rate_breached = (
            probe_count >= policy.minimum_rate_probes
            and (accepted_count / probe_count) < policy.minimum_accepted_rate
        )
        degraded = bool(base.degraded or threshold_breached or rate_breached)
        return replace(
            base,
            probe_count=probe_count,
            accepted_count=accepted_count,
            consecutive_failures=consecutive_failures,
            degraded=degraded,
            rollback_pending=degraded,
            last_reason_code=str(reason_code),
            last_probe_at=float(now),
        )

    def after_rollback(
        self,
        *,
        artifact_id: str | None,
        now: float,
        policy: LanguageProviderHealthPolicy,
        reason_code: str,
    ) -> LanguageProviderHealthState:
        """Return a fresh record for the fallback target inside a cooldown window."""

        if not isinstance(policy, LanguageProviderHealthPolicy):
            raise TypeError("provider health rollback requires a LanguageProviderHealthPolicy")
        return LanguageProviderHealthState(
            artifact_id=None if artifact_id is None else str(artifact_id),
            rollback_count=self.rollback_count + 1,
            cooldown_until=float(now) + policy.cooldown_seconds,
            last_reason_code=str(reason_code),
            last_probe_at=float(now),
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": LANGUAGE_PROVIDER_HEALTH_FORMAT,
            "artifact_id": self.artifact_id,
            "probe_count": self.probe_count,
            "accepted_count": self.accepted_count,
            "consecutive_failures": self.consecutive_failures,
            "rollback_count": self.rollback_count,
            "degraded": self.degraded,
            "rollback_pending": self.rollback_pending,
            "cooldown_until": self.cooldown_until,
            "last_reason_code": self.last_reason_code,
            "last_probe_at": self.last_probe_at,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> LanguageProviderHealthState:
        if payload.get("format") != LANGUAGE_PROVIDER_HEALTH_FORMAT:
            raise ValueError("unsupported language provider health state format")
        return cls(
            artifact_id=(
                None if payload.get("artifact_id") is None else str(payload["artifact_id"])
            ),
            probe_count=int(payload.get("probe_count", 0)),
            accepted_count=int(payload.get("accepted_count", 0)),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
            rollback_count=int(payload.get("rollback_count", 0)),
            degraded=bool(payload.get("degraded", False)),
            rollback_pending=bool(payload.get("rollback_pending", False)),
            cooldown_until=float(payload.get("cooldown_until", 0.0)),
            last_reason_code=str(payload.get("last_reason_code", "")),
            last_probe_at=(
                None if payload.get("last_probe_at") is None else float(payload["last_probe_at"])
            ),
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


class LanguageProviderCanaryGate:
    """Run deterministic first-chat expressions against a loaded provider.

    This is intentionally smaller than the train/holdout realization gate:
    it checks the exact post-load organ that will serve the first product chat
    request.  The cases reuse the reviewed semantic vocabulary used by the
    provider's holdout set, so a provider can pass only by producing readable
    text that preserves Taiji-owned meaning without validated fallback.
    """

    CANARY_ID = LANGUAGE_PROVIDER_CANARY_FORMAT

    @classmethod
    def cases(cls) -> tuple[ExpressionPlan, ...]:
        return (
            ExpressionPlan(
                expression_id=f"{cls.CANARY_ID}:database-status",
                content_id=f"{cls.CANARY_ID}:database-status:content",
                modality="text",
                channel="message",
                fields={
                    "intent_kind": "render_database_notice",
                    "semantic_slots": {"系统": "数据库", "状态": "正常", "受众": "操作员"},
                    "expected_outcome": "operator receives a concise message",
                },
                required_terms=("数据库", "正常"),
                confidence=1.0,
                provenance="language-provider-canary",
                tick=0,
            ),
            ExpressionPlan(
                expression_id=f"{cls.CANARY_ID}:interface-recovery",
                content_id=f"{cls.CANARY_ID}:interface-recovery:content",
                modality="text",
                channel="message",
                fields={
                    "intent_kind": "render_recovery_notice",
                    "semantic_slots": {"状态": "恢复", "服务": "接口", "受众": "操作员"},
                    "expected_outcome": "operator receives a concise message",
                },
                required_terms=("接口", "恢复"),
                confidence=1.0,
                provenance="language-provider-canary",
                tick=0,
            ),
        )

    def __init__(self) -> None:
        self._validator = LanguageRealizationValidator(minimum_coverage=1.0)

    def evaluate(self, organ: LanguageOrgan) -> dict[str, Any]:
        if not isinstance(organ, LanguageOrgan):
            raise TypeError("language provider canary organ must implement LanguageOrgan")
        rows: list[dict[str, Any]] = []
        for expression in self.cases():
            row: dict[str, Any] = {
                "case_id": expression.expression_id,
                "output_text": "",
                "output_nonempty": False,
                "readable": False,
                "required_terms": list(expression.required_terms),
                "matched_terms": [],
                "required_term_coverage": 0.0,
                "structured_leakage": False,
                "fallback_used": False,
                "accepted": False,
                "error": None,
            }
            try:
                emission = organ.emit(expression)
                text = emission.text_bytes.decode("utf-8", errors="strict")
                matched = tuple(term for term in expression.required_terms if term in text)
                validation = self._validator.validate(
                    expression,
                    emission.text_bytes,
                    required_terms=expression.required_terms,
                )
                leakage = any(
                    marker in text
                    for marker in ("semantic_slots", "intent_kind", "expected_outcome", "{", "}")
                )
                row.update(
                    {
                        "output_text": text,
                        "output_nonempty": bool(text.strip()),
                        "readable": _readable_surface(text) is not None,
                        "matched_terms": list(matched),
                        "required_term_coverage": len(matched)
                        / max(1, len(expression.required_terms)),
                        "structured_leakage": leakage,
                        "fallback_used": bool(emission.fallback_used),
                        "accepted": bool(
                            _readable_surface(text) is not None
                            and validation.accepted
                            and not leakage
                            and not emission.fallback_used
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - canary records failed cases
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

        count = max(1, len(rows))
        metrics = {
            "case_count": len(rows),
            "output_nonempty_rate": sum(bool(row["output_nonempty"]) for row in rows) / count,
            "readable_rate": sum(bool(row["readable"]) for row in rows) / count,
            "required_term_coverage": sum(float(row["required_term_coverage"]) for row in rows)
            / count,
            "structured_leakage_free_rate": sum(not bool(row["structured_leakage"]) for row in rows)
            / count,
            "fallback_rate": sum(bool(row["fallback_used"]) for row in rows) / count,
            "accepted_rate": sum(bool(row["accepted"]) for row in rows) / count,
        }
        passed = bool(
            metrics["case_count"] == len(self.cases())
            and metrics["output_nonempty_rate"] == 1.0
            and metrics["readable_rate"] == 1.0
            and metrics["required_term_coverage"] == 1.0
            and metrics["structured_leakage_free_rate"] == 1.0
            and metrics["fallback_rate"] == 0.0
            and metrics["accepted_rate"] == 1.0
            and all(row["error"] is None for row in rows)
        )
        return {
            "format": LANGUAGE_PROVIDER_CANARY_FORMAT,
            "canary_id": self.CANARY_ID,
            "cases": rows,
            "metrics": metrics,
            "gate": {
                "passed": passed,
                "criterion": "the loaded provider emits readable, semantically complete canary messages without structured leakage or validated fallback",
            },
        }


class LanguageProviderHealthProbe:
    """Judge one live emission without requiring Taiji-owned required terms.

    Product chat expressions carry no ``required_terms``, so this probe cannot
    reuse the canary's coverage metric.  It judges exactly what remains
    observable per request: the surface must be readable text, it must not leak
    structured plan keys, and the organ must not have taken its validated
    fallback.  A raised exception is a rejected probe, never a healthy one.
    """

    LEAKAGE_MARKERS = ("semantic_slots", "intent_kind", "expected_outcome", "{", "}")

    def evaluate(self, expression: ExpressionPlan, emission: Any) -> dict[str, Any]:
        """Return a probe row for one emission produced from ``expression``."""

        if not isinstance(expression, ExpressionPlan):
            raise TypeError("provider health probe requires an ExpressionPlan")
        row: dict[str, Any] = {
            "expression_id": expression.expression_id,
            "readable": False,
            "structured_leakage": False,
            "fallback_used": False,
            "accepted": False,
            "reason_code": "probe_emission_invalid",
            "error": None,
        }
        if not isinstance(emission, LanguageEmission):
            return row
        try:
            text = emission.text_bytes.decode("utf-8", errors="strict")
        except Exception as exc:  # noqa: BLE001 - an undecodable emission is a failed probe
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["reason_code"] = "probe_undecodable"
            row["fallback_used"] = bool(emission.fallback_used)
            return row
        readable = _readable_surface(text) is not None
        leakage = any(marker in text for marker in self.LEAKAGE_MARKERS)
        fallback_used = bool(emission.fallback_used)
        if fallback_used:
            reason_code = "probe_fallback_used"
        elif not readable:
            reason_code = "probe_unreadable"
        elif leakage:
            reason_code = "probe_structured_leakage"
        else:
            reason_code = "probe_accepted"
        row.update(
            {
                "readable": readable,
                "structured_leakage": leakage,
                "fallback_used": fallback_used,
                "accepted": bool(readable and not leakage and not fallback_used),
                "reason_code": reason_code,
            }
        )
        return row

    def probe(self, organ: LanguageOrgan, expression: ExpressionPlan) -> dict[str, Any]:
        """Emit through ``organ`` and judge the result, recording raises."""

        if not isinstance(organ, LanguageOrgan):
            raise TypeError("provider health probe organ must implement LanguageOrgan")
        try:
            emission = organ.emit(expression)
        except Exception as exc:  # noqa: BLE001 - a raising organ is a failed probe
            return {
                "expression_id": expression.expression_id,
                "readable": False,
                "structured_leakage": False,
                "fallback_used": False,
                "accepted": False,
                "reason_code": "probe_emission_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return self.evaluate(expression, emission)


class LanguageProviderHealthGate:
    """Prove the watchdog degrades, rolls back once, and survives a restart.

    The gate does not merely report whether one organ is currently healthy.  It
    exercises the full decision path the runtime depends on: a healthy provider
    stays active and undegraded, a failing provider trips the configured
    consecutive-failure threshold exactly once, the post-rollback cooldown
    window suppresses a second rollback, and a checkpoint round-trip reproduces
    the degraded record so a restart cannot silently re-declare health.
    """

    GATE_ID = LANGUAGE_PROVIDER_HEALTH_GATE_FORMAT

    def __init__(self, policy: LanguageProviderHealthPolicy | None = None) -> None:
        self._policy = policy if policy is not None else LanguageProviderHealthPolicy()
        if not isinstance(self._policy, LanguageProviderHealthPolicy):
            raise TypeError("provider health gate requires a LanguageProviderHealthPolicy")
        self._probe = LanguageProviderHealthProbe()

    @property
    def policy(self) -> LanguageProviderHealthPolicy:
        return self._policy

    @classmethod
    def cases(cls) -> tuple[ExpressionPlan, ...]:
        """Reuse the reviewed canary expressions as live health probes."""

        return LanguageProviderCanaryGate.cases()

    def evaluate(
        self,
        organ: LanguageOrgan,
        *,
        degraded_organ: LanguageOrgan | None = None,
        artifact_id: str = "health-gate-active",
        rollback_artifact_id: str = NativeReadableTextLanguageOrgan.BACKEND_ID,
        now: float = 0.0,
    ) -> dict[str, Any]:
        policy = self._policy
        healthy_rows: list[dict[str, Any]] = []
        healthy_state = LanguageProviderHealthState()
        clock = float(now)
        for expression in self.cases():
            clock += 1.0
            row = self._probe.probe(organ, expression)
            healthy_rows.append(row)
            healthy_state = healthy_state.observe(
                artifact_id=artifact_id,
                accepted=bool(row["accepted"]),
                reason_code=str(row["reason_code"]),
                now=clock,
                policy=policy,
            )

        degraded_rows: list[dict[str, Any]] = []
        degraded_state = healthy_state
        current_artifact_id = artifact_id
        rollbacks = 0
        rollback_suppressed = 0
        probes_before_rollback = 0
        if degraded_organ is not None:
            probe_budget = policy.failure_threshold * 2
            for index in range(probe_budget):
                clock += 1.0
                expression = self.cases()[index % len(self.cases())]
                row = self._probe.probe(degraded_organ, expression)
                degraded_rows.append(row)
                degraded_state = degraded_state.observe(
                    artifact_id=current_artifact_id,
                    accepted=bool(row["accepted"]),
                    reason_code=str(row["reason_code"]),
                    now=clock,
                    policy=policy,
                )
                if not degraded_state.rollback_pending:
                    continue
                if degraded_state.in_cooldown(clock):
                    rollback_suppressed += 1
                    continue
                rollbacks += 1
                if probes_before_rollback == 0:
                    probes_before_rollback = index + 1
                degraded_state = degraded_state.after_rollback(
                    artifact_id=rollback_artifact_id,
                    now=clock,
                    policy=policy,
                    reason_code="provider_health_rolled_back",
                )
                current_artifact_id = rollback_artifact_id

        restored = LanguageProviderHealthState.from_checkpoint(degraded_state.checkpoint())
        metrics = {
            "healthy_probe_count": len(healthy_rows),
            "healthy_accepted_rate": sum(bool(row["accepted"]) for row in healthy_rows)
            / max(1, len(healthy_rows)),
            "healthy_degraded": bool(healthy_state.degraded),
            "degraded_probe_count": len(degraded_rows),
            "degraded_accepted_rate": sum(bool(row["accepted"]) for row in degraded_rows)
            / max(1, len(degraded_rows)),
            "probes_before_rollback": probes_before_rollback,
            "rollback_count": rollbacks,
            "rollback_suppressed_count": rollback_suppressed,
            "checkpoint_roundtrip_matches": bool(restored == degraded_state),
            "cooldown_active_after_rollback": bool(degraded_state.in_cooldown(clock)),
            "rollback_target": degraded_state.artifact_id,
        }
        healthy_passed = bool(
            metrics["healthy_probe_count"] == len(self.cases())
            and metrics["healthy_accepted_rate"] == 1.0
            and not metrics["healthy_degraded"]
        )
        if degraded_organ is None:
            degraded_passed = True
        else:
            degraded_passed = bool(
                metrics["rollback_count"] == 1
                and metrics["probes_before_rollback"] == policy.failure_threshold
                and metrics["rollback_suppressed_count"] >= 1
                and metrics["cooldown_active_after_rollback"]
                and metrics["rollback_target"] == rollback_artifact_id
            )
        passed = bool(
            healthy_passed and degraded_passed and metrics["checkpoint_roundtrip_matches"]
        )
        return {
            "format": LANGUAGE_PROVIDER_HEALTH_GATE_FORMAT,
            "gate_id": self.GATE_ID,
            "policy": policy.to_payload(),
            "healthy_probes": healthy_rows,
            "degraded_probes": degraded_rows,
            "health_state": degraded_state.checkpoint(),
            "metrics": metrics,
            "gate": {
                "passed": passed,
                "criterion": "a healthy provider stays active while a degrading provider trips the consecutive-failure threshold exactly once, falls back to the declared target, is held by the cooldown window, and reproduces its degraded record from a checkpoint",
            },
        }

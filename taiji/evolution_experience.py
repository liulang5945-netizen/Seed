"""Taiji-owned corpus and runtime experience contracts.

This module is intentionally free of Seed, Workbench and provider imports.
It defines the versioned data that Taiji may consume after an external
source has been normalized, redacted and admitted by a platform ledger.
Corpus material is knowledge input; an experience is an observed episode.
Neither object contains executable source or a direct mutation operation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .internalization import content_digest

EVOLUTION_CORPUS_FORMAT = "taiji-evolution-corpus-v1"
EVOLUTION_EXPERIENCE_FORMAT = "taiji-evolution-experience-v1"
EVOLUTION_CONTRACT_VERSION = 1

EVOLUTION_CORPUS_UNIT_KINDS = (
    "knowledge",
    "procedure",
    "affordance",
    "constraint",
    "example",
    "counterexample",
)
EVOLUTION_CORPUS_STATUSES = ("candidate", "admitted", "rejected", "quarantined")
EVOLUTION_PARTITIONS = ("train", "holdout", "retention", "security")
EVOLUTION_EXPERIENCE_SOURCE_KINDS = (
    "workbench",
    "skill",
    "mcp",
    "client_plugin",
    "user_correction",
    "provider",
)
EVOLUTION_EXPERIENCE_STATUSES = ("success", "rejected", "error", "cancelled")
EVOLUTION_REDACTION_REVISION = "taiji-evolution-redaction-v1"
REDACTION_PLACEHOLDER = "<redacted>"

_SENSITIVE_KEYS = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "password",
        "private_key",
        "secret",
        "token",
    }
)
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "command",
        "entrypoint_path",
        "executable_source",
        "import_path",
        "prompt",
        "raw_transcript",
        "shell",
        "source_code",
        "transcript",
    }
)


def _required_text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _optional_text(value: Any) -> str:
    return str(value).strip()


def _digest_text(value: Any, name: str, *, optional: bool = False) -> str:
    normalized = str(value).strip()
    if optional and not normalized:
        return ""
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _safe_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _required_text(key, f"{name} key")
            if normalized_key in normalized:
                raise ValueError(f"{name} contains duplicate key: {normalized_key}")
            normalized[normalized_key] = _safe_value(item, f"{name}.{normalized_key}")
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, f"{name} item") for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite floats")
        return value
    raise TypeError(f"{name} contains unsupported value: {type(value).__name__}")


def _safe_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = _safe_value(value, name)
    assert isinstance(normalized, dict)
    return normalized


def _text_tuple(value: Sequence[Any], name: str, *, sort: bool = True) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(_required_text(item, f"{name} item") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(normalized)) if sort else normalized


def _digest_tuple(value: Sequence[Any], name: str) -> tuple[str, ...]:
    return _text_tuple(
        tuple(_digest_text(item, f"{name} item") for item in value),
        name,
    )


def _numeric_mapping(
    value: Mapping[str, Any],
    name: str,
    *,
    non_negative: bool,
) -> tuple[tuple[str, float], ...]:
    normalized = _safe_mapping(value, name)
    pairs: list[tuple[str, float]] = []
    for key, item in normalized.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name}.{key} must be numeric")
        number = float(item)
        if not math.isfinite(number) or (non_negative and number < 0.0):
            constraint = "finite and non-negative" if non_negative else "finite"
            raise ValueError(f"{name}.{key} must be {constraint}")
        pairs.append((key, number))
    return tuple(sorted(pairs))


def _validate_redaction(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if key_text in _SENSITIVE_KEYS and not isinstance(item, (Mapping, list, tuple)):
                if item != REDACTION_PLACEHOLDER:
                    raise ValueError(f"{path}.{key} contains unredacted sensitive data")
            _validate_redaction(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_redaction(item, f"{path}[{index}]")


def _validate_metadata_keys(value: Mapping[str, Any], name: str) -> None:
    for key in value:
        if str(key).strip().lower() in _FORBIDDEN_METADATA_KEYS:
            raise ValueError(f"{name} contains forbidden field: {key}")
        child = value[key]
        if isinstance(child, Mapping):
            _validate_metadata_keys(child, f"{name}.{key}")
        elif isinstance(child, (list, tuple)):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    _validate_metadata_keys(item, f"{name}.{key}[{index}]")


def _partition(value: str) -> str:
    normalized = _required_text(value, "partition")
    if normalized not in EVOLUTION_PARTITIONS:
        raise ValueError(f"unsupported evolution partition: {normalized}")
    return normalized


@dataclass(frozen=True)
class EvolutionCorpusArtifact:
    """A content-addressed, potentially admissible knowledge unit."""

    corpus_id: str
    source_kind: str
    source_id: str
    source_version: str
    source_digest: str
    unit_kind: str
    content: Mapping[str, Any]
    relation_digests: tuple[str, ...] = ()
    publisher: str = ""
    scope_id: str = ""
    capability_semantics: Mapping[str, Any] = field(default_factory=dict)
    input_schema_digest: str = ""
    output_schema_digest: str = ""
    constraint_digests: tuple[str, ...] = ()
    language: str = "und"
    confidence: float = 1.0
    license_use_policy: str = "unspecified"
    taint_flags: tuple[str, ...] = ()
    redaction_revision: str = EVOLUTION_REDACTION_REVISION
    partition: str = "train"
    retention_policy: str = "default"
    dependency_digests: tuple[str, ...] = ()
    supersedes_digest: str = ""
    status: str = "candidate"
    admission_revision: str = ""
    content_digest: str = ""
    chunk_digest: str = ""
    artifact_digest: str = ""
    version: int = EVOLUTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if int(self.version) != EVOLUTION_CONTRACT_VERSION:
            raise ValueError("unsupported evolution corpus version")
        for value, name in (
            (self.corpus_id, "corpus_id"),
            (self.source_kind, "source_kind"),
            (self.source_id, "source_id"),
            (self.source_version, "source_version"),
            (self.unit_kind, "unit_kind"),
            (self.language, "language"),
            (self.license_use_policy, "license_use_policy"),
            (self.redaction_revision, "redaction_revision"),
            (self.retention_policy, "retention_policy"),
        ):
            object.__setattr__(self, name, _required_text(value, name))
        object.__setattr__(self, "source_digest", _digest_text(self.source_digest, "source_digest"))
        if self.source_kind not in {"skill_artifact", "mcp_artifact", "client_plugin_artifact", "verified_domain_material"}:
            raise ValueError("unsupported evolution corpus source_kind")
        if self.unit_kind not in EVOLUTION_CORPUS_UNIT_KINDS:
            raise ValueError("unsupported evolution corpus unit_kind")
        object.__setattr__(self, "partition", _partition(self.partition))
        if self.status not in EVOLUTION_CORPUS_STATUSES:
            raise ValueError("unsupported evolution corpus status")
        if self.status == "admitted" and not str(self.admission_revision).strip():
            raise ValueError("admitted corpus requires admission_revision")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("corpus confidence must be in [0, 1]")
        normalized_content = _safe_mapping(self.content, "content")
        _validate_redaction(normalized_content, "content")
        object.__setattr__(self, "content", normalized_content)
        object.__setattr__(self, "publisher", _optional_text(self.publisher))
        object.__setattr__(self, "scope_id", _optional_text(self.scope_id))
        normalized_capability_semantics = _safe_mapping(
            self.capability_semantics,
            "capability_semantics",
        )
        _validate_redaction(normalized_capability_semantics, "capability_semantics")
        _validate_metadata_keys(normalized_capability_semantics, "capability_semantics")
        object.__setattr__(self, "capability_semantics", normalized_capability_semantics)
        expected_content_digest = content_digest(normalized_content)
        supplied_content_digest = str(self.content_digest).strip()
        if supplied_content_digest and supplied_content_digest != expected_content_digest:
            raise ValueError("evolution corpus content_digest mismatch")
        object.__setattr__(self, "content_digest", expected_content_digest)
        relation_digests = _digest_tuple(self.relation_digests, "relation_digests")
        dependency_digests = _digest_tuple(self.dependency_digests, "dependency_digests")
        object.__setattr__(self, "relation_digests", relation_digests)
        object.__setattr__(self, "dependency_digests", dependency_digests)
        object.__setattr__(
            self,
            "input_schema_digest",
            _digest_text(self.input_schema_digest, "input_schema_digest", optional=True),
        )
        object.__setattr__(
            self,
            "output_schema_digest",
            _digest_text(self.output_schema_digest, "output_schema_digest", optional=True),
        )
        object.__setattr__(self, "constraint_digests", _digest_tuple(self.constraint_digests, "constraint_digests"))
        supersedes_digest = _digest_text(self.supersedes_digest, "supersedes_digest", optional=True)
        object.__setattr__(self, "supersedes_digest", supersedes_digest)
        object.__setattr__(self, "taint_flags", _text_tuple(self.taint_flags, "taint_flags"))
        identity_digest = content_digest(self._identity_payload())
        supplied_artifact_digest = str(self.artifact_digest).strip()
        if supplied_artifact_digest and supplied_artifact_digest != identity_digest:
            raise ValueError("evolution corpus artifact_digest mismatch")
        object.__setattr__(self, "artifact_digest", identity_digest)
        supplied_chunk_digest = str(self.chunk_digest).strip()
        if supplied_chunk_digest and supplied_chunk_digest != self.content_digest:
            raise ValueError("evolution corpus chunk_digest mismatch")
        object.__setattr__(self, "chunk_digest", self.content_digest)
        object.__setattr__(self, "admission_revision", str(self.admission_revision).strip())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "format": EVOLUTION_CORPUS_FORMAT,
            "version": self.version,
            "corpus_id": self.corpus_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_digest": self.source_digest,
            "unit_kind": self.unit_kind,
            "content_digest": self.content_digest,
            "relation_digests": list(self.relation_digests),
            "publisher": self.publisher,
            "scope_id": self.scope_id,
            "capability_semantics": dict(self.capability_semantics),
            "input_schema_digest": self.input_schema_digest,
            "output_schema_digest": self.output_schema_digest,
            "constraint_digests": list(self.constraint_digests),
            "language": self.language,
            "confidence": float(self.confidence),
            "license_use_policy": self.license_use_policy,
            "taint_flags": list(self.taint_flags),
            "redaction_revision": self.redaction_revision,
            "partition": self.partition,
            "retention_policy": self.retention_policy,
            "dependency_digests": list(self.dependency_digests),
            "supersedes_digest": self.supersedes_digest,
        }

    def with_status(self, status: str, *, admission_revision: str = "") -> EvolutionCorpusArtifact:
        return replace(
            self,
            status=str(status),
            admission_revision=str(admission_revision),
            artifact_digest=self.artifact_digest,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "format": EVOLUTION_CORPUS_FORMAT,
            "content": dict(self.content),
            "status": self.status,
            "admission_revision": self.admission_revision,
            "chunk_digest": self.chunk_digest,
            "artifact_digest": self.artifact_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvolutionCorpusArtifact:
        if payload.get("format") != EVOLUTION_CORPUS_FORMAT:
            raise ValueError("unsupported evolution corpus format")
        artifact = cls(
            corpus_id=str(payload["corpus_id"]),
            source_kind=str(payload["source_kind"]),
            source_id=str(payload["source_id"]),
            source_version=str(payload["source_version"]),
            source_digest=str(payload["source_digest"]),
            unit_kind=str(payload["unit_kind"]),
            content=payload.get("content") or {},
            relation_digests=tuple(payload.get("relation_digests", ())),
            publisher=str(payload.get("publisher", "")),
            scope_id=str(payload.get("scope_id", "")),
            capability_semantics=payload.get("capability_semantics") or {},
            input_schema_digest=str(payload.get("input_schema_digest", "")),
            output_schema_digest=str(payload.get("output_schema_digest", "")),
            constraint_digests=tuple(payload.get("constraint_digests", ())),
            language=str(payload.get("language", "und")),
            confidence=float(payload.get("confidence", 1.0)),
            license_use_policy=str(payload.get("license_use_policy", "unspecified")),
            taint_flags=tuple(payload.get("taint_flags", ())),
            redaction_revision=str(payload.get("redaction_revision", EVOLUTION_REDACTION_REVISION)),
            partition=str(payload.get("partition", "train")),
            retention_policy=str(payload.get("retention_policy", "default")),
            dependency_digests=tuple(payload.get("dependency_digests", ())),
            supersedes_digest=str(payload.get("supersedes_digest", "")),
            status=str(payload.get("status", "candidate")),
            admission_revision=str(payload.get("admission_revision", "")),
            content_digest=str(payload.get("content_digest", "")),
            chunk_digest=str(payload.get("chunk_digest", "")),
            artifact_digest=str(payload.get("artifact_digest", "")),
            version=int(payload.get("version", EVOLUTION_CONTRACT_VERSION)),
        )
        return artifact


@dataclass(frozen=True)
class EvolutionExperience:
    """One redacted runtime episode that can later feed a learner view."""

    experience_id: str
    source_kind: str
    source_id: str
    source_version: str
    source_digest: str
    parent_checkpoint_digest: str
    partition: str
    status: str
    success: bool
    request_id: str = ""
    intent_id: str = ""
    call_id: str = ""
    outcome_id: str = ""
    episode_id: str = ""
    tick: int = 0
    input_digest: str = ""
    percept_digest: str = ""
    goal_digest: str = ""
    world_state_digest: str = ""
    plan_digest: str = ""
    uncertainty: float = 0.0
    capability_id: str = ""
    capability_snapshot_id: str = ""
    arguments_digest: str = ""
    approval_id: str = ""
    result_digest: str = ""
    error_code: str = ""
    reward_components: tuple[tuple[str, float], ...] = ()
    resource_usage: tuple[tuple[str, float], ...] = ()
    user_correction_digest: str = ""
    client_snapshot_id: str = ""
    skill_digest: str = ""
    mcp_server_digest: str = ""
    mcp_schema_digest: str = ""
    plugin_digest: str = ""
    taint_flags: tuple[str, ...] = ()
    redaction_revision: str = EVOLUTION_REDACTION_REVISION
    retention_policy: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    event_sequence: int = 0
    previous_event_digest: str = ""
    experience_digest: str = ""
    event_digest: str = ""
    version: int = EVOLUTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if int(self.version) != EVOLUTION_CONTRACT_VERSION:
            raise ValueError("unsupported evolution experience version")
        for value, name in (
            (self.experience_id, "experience_id"),
            (self.source_kind, "source_kind"),
            (self.source_id, "source_id"),
            (self.source_version, "source_version"),
            (self.redaction_revision, "redaction_revision"),
            (self.retention_policy, "retention_policy"),
        ):
            object.__setattr__(self, name, _required_text(value, name))
        if self.source_kind not in EVOLUTION_EXPERIENCE_SOURCE_KINDS:
            raise ValueError("unsupported evolution experience source_kind")
        object.__setattr__(self, "source_digest", _digest_text(self.source_digest, "source_digest"))
        object.__setattr__(
            self,
            "parent_checkpoint_digest",
            _digest_text(self.parent_checkpoint_digest, "parent_checkpoint_digest"),
        )
        object.__setattr__(self, "partition", _partition(self.partition))
        if self.status not in EVOLUTION_EXPERIENCE_STATUSES:
            raise ValueError("unsupported evolution experience status")
        if int(self.tick) < 0 or int(self.event_sequence) < 0:
            raise ValueError("experience tick and event_sequence cannot be negative")
        if not math.isfinite(float(self.uncertainty)) or not 0.0 <= float(self.uncertainty) <= 1.0:
            raise ValueError("experience uncertainty must be in [0, 1]")
        object.__setattr__(self, "tick", int(self.tick))
        object.__setattr__(self, "event_sequence", int(self.event_sequence))
        for name in (
            "request_id",
            "intent_id",
            "call_id",
            "outcome_id",
            "episode_id",
            "capability_id",
            "capability_snapshot_id",
            "approval_id",
            "error_code",
            "client_snapshot_id",
            "retention_policy",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        for name in (
            "input_digest",
            "percept_digest",
            "goal_digest",
            "world_state_digest",
            "plan_digest",
            "arguments_digest",
            "result_digest",
            "user_correction_digest",
            "skill_digest",
            "mcp_server_digest",
            "mcp_schema_digest",
            "plugin_digest",
            "previous_event_digest",
        ):
            object.__setattr__(
                self,
                name,
                _digest_text(getattr(self, name), name, optional=True),
            )
        object.__setattr__(
            self,
            "reward_components",
            _numeric_mapping(dict(self.reward_components), "reward_components", non_negative=False),
        )
        object.__setattr__(
            self,
            "resource_usage",
            _numeric_mapping(dict(self.resource_usage), "resource_usage", non_negative=True),
        )
        normalized_metadata = _safe_mapping(self.metadata, "metadata")
        _validate_metadata_keys(normalized_metadata, "metadata")
        _validate_redaction(normalized_metadata, "metadata")
        object.__setattr__(self, "metadata", normalized_metadata)
        object.__setattr__(self, "taint_flags", _text_tuple(self.taint_flags, "taint_flags"))
        expected_experience_digest = content_digest(self._content_payload())
        supplied_experience_digest = str(self.experience_digest).strip()
        if supplied_experience_digest and supplied_experience_digest != expected_experience_digest:
            raise ValueError("evolution experience experience_digest mismatch")
        object.__setattr__(self, "experience_digest", expected_experience_digest)
        if self.event_sequence == 0:
            if self.previous_event_digest:
                raise ValueError("unbound experience cannot have previous_event_digest")
            expected_event_digest = content_digest(self._event_payload())
        else:
            if not self.previous_event_digest:
                raise ValueError("bound experience requires previous_event_digest")
            expected_event_digest = content_digest(self._event_payload())
        supplied_event_digest = str(self.event_digest).strip()
        if supplied_event_digest and supplied_event_digest != expected_event_digest:
            raise ValueError("evolution experience event_digest mismatch")
        object.__setattr__(self, "event_digest", expected_event_digest)

    def _content_payload(self) -> dict[str, Any]:
        return {
            "format": EVOLUTION_EXPERIENCE_FORMAT,
            "version": self.version,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_digest": self.source_digest,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "partition": self.partition,
            "status": self.status,
            "success": bool(self.success),
            "request_id": self.request_id,
            "intent_id": self.intent_id,
            "call_id": self.call_id,
            "outcome_id": self.outcome_id,
            "episode_id": self.episode_id,
            "tick": self.tick,
            "input_digest": self.input_digest,
            "percept_digest": self.percept_digest,
            "goal_digest": self.goal_digest,
            "world_state_digest": self.world_state_digest,
            "plan_digest": self.plan_digest,
            "uncertainty": float(self.uncertainty),
            "capability_id": self.capability_id,
            "capability_snapshot_id": self.capability_snapshot_id,
            "arguments_digest": self.arguments_digest,
            "approval_id": self.approval_id,
            "result_digest": self.result_digest,
            "error_code": self.error_code,
            "reward_components": list(self.reward_components),
            "resource_usage": list(self.resource_usage),
            "user_correction_digest": self.user_correction_digest,
            "client_snapshot_id": self.client_snapshot_id,
            "skill_digest": self.skill_digest,
            "mcp_server_digest": self.mcp_server_digest,
            "mcp_schema_digest": self.mcp_schema_digest,
            "plugin_digest": self.plugin_digest,
            "taint_flags": list(self.taint_flags),
            "redaction_revision": self.redaction_revision,
            "retention_policy": self.retention_policy,
            "metadata": dict(self.metadata),
        }

    def _event_payload(self) -> dict[str, Any]:
        return {
            "format": EVOLUTION_EXPERIENCE_FORMAT,
            "version": self.version,
            "experience_digest": self.experience_digest,
            "event_sequence": self.event_sequence,
            "previous_event_digest": self.previous_event_digest,
        }

    def bind_to_chain(self, sequence: int, previous_event_digest: str) -> EvolutionExperience:
        if int(sequence) <= 0:
            raise ValueError("bound experience sequence must be positive")
        return replace(
            self,
            event_sequence=int(sequence),
            previous_event_digest=_digest_text(previous_event_digest, "previous_event_digest"),
            event_digest="",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._content_payload(),
            "experience_id": self.experience_id,
            "event_sequence": self.event_sequence,
            "previous_event_digest": self.previous_event_digest,
            "experience_digest": self.experience_digest,
            "event_digest": self.event_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvolutionExperience:
        if payload.get("format") != EVOLUTION_EXPERIENCE_FORMAT:
            raise ValueError("unsupported evolution experience format")
        return cls(
            experience_id=str(payload["experience_id"]),
            source_kind=str(payload["source_kind"]),
            source_id=str(payload["source_id"]),
            source_version=str(payload["source_version"]),
            source_digest=str(payload["source_digest"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            partition=str(payload["partition"]),
            status=str(payload["status"]),
            success=bool(payload["success"]),
            request_id=str(payload.get("request_id", "")),
            intent_id=str(payload.get("intent_id", "")),
            call_id=str(payload.get("call_id", "")),
            outcome_id=str(payload.get("outcome_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            tick=int(payload.get("tick", 0)),
            input_digest=str(payload.get("input_digest", "")),
            percept_digest=str(payload.get("percept_digest", "")),
            goal_digest=str(payload.get("goal_digest", "")),
            world_state_digest=str(payload.get("world_state_digest", "")),
            plan_digest=str(payload.get("plan_digest", "")),
            uncertainty=float(payload.get("uncertainty", 0.0)),
            capability_id=str(payload.get("capability_id", "")),
            capability_snapshot_id=str(payload.get("capability_snapshot_id", "")),
            arguments_digest=str(payload.get("arguments_digest", "")),
            approval_id=str(payload.get("approval_id", "")),
            result_digest=str(payload.get("result_digest", "")),
            error_code=str(payload.get("error_code", "")),
            reward_components=tuple(tuple(item) for item in payload.get("reward_components", ())),
            resource_usage=tuple(tuple(item) for item in payload.get("resource_usage", ())),
            user_correction_digest=str(payload.get("user_correction_digest", "")),
            client_snapshot_id=str(payload.get("client_snapshot_id", "")),
            skill_digest=str(payload.get("skill_digest", "")),
            mcp_server_digest=str(payload.get("mcp_server_digest", "")),
            mcp_schema_digest=str(payload.get("mcp_schema_digest", "")),
            plugin_digest=str(payload.get("plugin_digest", "")),
            taint_flags=tuple(payload.get("taint_flags", ())),
            redaction_revision=str(payload.get("redaction_revision", EVOLUTION_REDACTION_REVISION)),
            retention_policy=str(payload.get("retention_policy", "default")),
            metadata=payload.get("metadata") or {},
            event_sequence=int(payload.get("event_sequence", 0)),
            previous_event_digest=str(payload.get("previous_event_digest", "")),
            experience_digest=str(payload.get("experience_digest", "")),
            event_digest=str(payload.get("event_digest", "")),
            version=int(payload.get("version", EVOLUTION_CONTRACT_VERSION)),
        )


__all__ = [
    "EVOLUTION_CONTRACT_VERSION",
    "EVOLUTION_CORPUS_FORMAT",
    "EVOLUTION_CORPUS_STATUSES",
    "EVOLUTION_CORPUS_UNIT_KINDS",
    "EVOLUTION_EXPERIENCE_FORMAT",
    "EVOLUTION_EXPERIENCE_SOURCE_KINDS",
    "EVOLUTION_EXPERIENCE_STATUSES",
    "EVOLUTION_PARTITIONS",
    "EVOLUTION_REDACTION_REVISION",
    "EvolutionCorpusArtifact",
    "EvolutionExperience",
    "REDACTION_PLACEHOLDER",
]

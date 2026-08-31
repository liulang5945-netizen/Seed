"""Deterministic adapters from external artifacts and lifecycle events.

The adapters are a boundary, not a learner.  They select documented,
non-executable fields, redact secrets, and project source-specific events into
the E1 Taiji contracts.  Nothing in this module installs, imports, connects
to, or executes a Skill, MCP server, or client plugin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from seed_platform.evolution_ledger import redact_sensitive_payload
from taiji.evolution_experience import (
    EVOLUTION_REDACTION_REVISION,
    EvolutionCorpusArtifact,
    EvolutionExperience,
)
from taiji.internalization import content_digest

_UNSAFE_SOURCE_KEYS = frozenset(
    {
        "entrypoint_path",
        "executable_source",
        "import_path",
        "install_script",
        "prompt",
        "raw_transcript",
        "shell",
        "source_code",
        "transcript",
    }
)
_EVENT_SOURCE_KINDS = frozenset({"skill", "mcp", "client_plugin"})


def _safe_source(value: Any) -> tuple[Any, tuple[str, ...]]:
    """Return redacted JSON-like data without executable source fields."""

    redacted, flags = redact_sensitive_payload(value)

    def strip(item: Any, path: str = "") -> Any:
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for raw_key, child in item.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                if key.strip().lower() in _UNSAFE_SOURCE_KEYS:
                    continue
                result[key] = strip(child, child_path)
            return {key: result[key] for key in sorted(result)}
        if isinstance(item, (list, tuple)):
            return [strip(child, f"{path}[{index}]") for index, child in enumerate(item)]
        return item

    return strip(redacted), flags


def _required(source: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = str(source.get(name, "")).strip()
        if value:
            return value
    raise ValueError(f"artifact is missing required field: {'/'.join(names)}")


def _optional(source: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = str(source.get(name, "")).strip()
        if value:
            return value
    return ""


def _entries(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        result: list[Any] = []
        for key, child in value.items():
            if isinstance(child, Mapping):
                entry = dict(child)
                entry.setdefault("name", str(key))
                result.append(entry)
            else:
                result.append({"name": str(key), "value": child})
        return tuple(result)
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _content(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _digest_for_schema(value: Any) -> str:
    if isinstance(value, str) and len(value.strip()) == 64:
        candidate = value.strip()
        if all(char in "0123456789abcdef" for char in candidate):
            return candidate
    if value is None or value == "":
        return ""
    safe, _ = _safe_source(value)
    return content_digest(safe)


def _make_unit(
    *,
    source_kind: str,
    source_id: str,
    source_version: str,
    source_digest: str,
    publisher: str,
    scope_id: str,
    unit_kind: str,
    content: Mapping[str, Any],
    partition: str,
    capability_semantics: Mapping[str, Any] | None = None,
    input_schema: Any = None,
    output_schema: Any = None,
    redaction_flags: Sequence[str] = (),
) -> EvolutionCorpusArtifact:
    safe_content, content_flags = _safe_source(content)
    if not isinstance(safe_content, Mapping):  # pragma: no cover - _content ensures this
        raise TypeError("corpus unit content must be a mapping")
    combined_flags = tuple(sorted(set(redaction_flags) | set(content_flags)))
    taint_flags = ("redacted_external_source",) if combined_flags else ()
    capability = {} if capability_semantics is None else capability_semantics
    safe_capability, capability_flags = _safe_source(capability)
    if not isinstance(safe_capability, Mapping):
        raise TypeError("capability semantics must be a mapping")
    combined_flags = tuple(sorted(set(combined_flags) | set(capability_flags)))
    if combined_flags:
        taint_flags = (*taint_flags, "redacted_external_source")
    unit_digest = content_digest(safe_content)
    return EvolutionCorpusArtifact(
        corpus_id=f"{source_kind}:{source_id}:{unit_kind}:{unit_digest[:16]}",
        source_kind=f"{source_kind}_artifact",
        source_id=source_id,
        source_version=source_version,
        source_digest=source_digest,
        publisher=publisher,
        scope_id=scope_id,
        unit_kind=unit_kind,
        content=safe_content,
        relation_digests=(source_digest,),
        capability_semantics=safe_capability,
        input_schema_digest=_digest_for_schema(input_schema),
        output_schema_digest=_digest_for_schema(output_schema),
        language="und",
        partition=partition,
        taint_flags=tuple(sorted(set(taint_flags))),
        redaction_revision=EVOLUTION_REDACTION_REVISION,
    )


@dataclass(frozen=True)
class ArtifactCorpusProjection:
    source_kind: str
    source_id: str
    source_version: str
    source_digest: str
    scope_id: str
    publisher: str
    corpus: tuple[EvolutionCorpusArtifact, ...]
    redaction_flags: tuple[str, ...] = ()

    def project_event(
        self,
        event: Mapping[str, Any],
        *,
        parent_checkpoint_digest: str,
        partition: str = "train",
    ) -> EvolutionExperience:
        return runtime_event_to_experience(
            event,
            source_kind=self.source_kind,
            source_id=self.source_id,
            source_version=self.source_version,
            source_digest=self.source_digest,
            scope_id=self.scope_id,
            parent_checkpoint_digest=parent_checkpoint_digest,
            partition=partition,
        )


class SkillArtifactAdapter:
    """Project a declarative external Skill manifest into corpus units."""

    source_kind = "skill"

    def project(self, artifact: Mapping[str, Any], *, partition: str = "train") -> ArtifactCorpusProjection:
        if not isinstance(artifact, Mapping):
            raise TypeError("Skill artifact must be a mapping")
        safe, flags = _safe_source(artifact)
        if not isinstance(safe, Mapping):
            raise TypeError("Skill artifact must be mapping-like")
        source_id = _required(safe, "skill_id", "id", "name")
        source_version = _required(safe, "version")
        scope_id = _optional(safe, "scope_id", "scope")
        publisher = _optional(safe, "publisher", "author")
        source_digest = content_digest(safe)
        units: list[EvolutionCorpusArtifact] = []
        knowledge = {
            "name": str(safe.get("name", source_id)),
            "description": str(safe.get("description", "")),
            "scope_id": scope_id,
            "references": safe.get("references", ()),
            "license_use_policy": _optional(safe, "license_use_policy", "use_policy"),
        }
        units.append(
            _make_unit(
                source_kind=self.source_kind,
                source_id=source_id,
                source_version=source_version,
                source_digest=source_digest,
                publisher=publisher,
                scope_id=scope_id,
                unit_kind="knowledge",
                content=knowledge,
                partition=partition,
                redaction_flags=flags,
            )
        )
        procedures = safe.get("steps", safe.get("instructions", safe.get("procedure")))
        if procedures is not None:
            units.append(
                _make_unit(
                    source_kind=self.source_kind,
                    source_id=source_id,
                    source_version=source_version,
                    source_digest=source_digest,
                    publisher=publisher,
                    scope_id=scope_id,
                    unit_kind="procedure",
                    content={"steps": _entries(procedures)},
                    partition=partition,
                    redaction_flags=flags,
                )
            )
        for field_name, unit_kind in (
            ("capabilities", "affordance"),
            ("constraints", "constraint"),
            ("examples", "example"),
            ("counterexamples", "counterexample"),
        ):
            for item in _entries(safe.get(field_name)):
                units.append(
                    _make_unit(
                        source_kind=self.source_kind,
                        source_id=source_id,
                        source_version=source_version,
                        source_digest=source_digest,
                        publisher=publisher,
                        scope_id=scope_id,
                        unit_kind=unit_kind,
                        content=_content(item),
                        partition=partition,
                        redaction_flags=flags,
                    )
                )
        return ArtifactCorpusProjection(
            self.source_kind,
            source_id,
            source_version,
            source_digest,
            scope_id,
            publisher,
            tuple(units),
            tuple(sorted(flags)),
        )


class McpArtifactAdapter:
    """Project MCP server docs, schemas and constraints without execution."""

    source_kind = "mcp"

    def project(self, artifact: Mapping[str, Any], *, partition: str = "train") -> ArtifactCorpusProjection:
        if not isinstance(artifact, Mapping):
            raise TypeError("MCP artifact must be a mapping")
        safe, flags = _safe_source(artifact)
        if not isinstance(safe, Mapping):
            raise TypeError("MCP artifact must be mapping-like")
        source_id = _required(safe, "server_id", "id", "name")
        source_version = _required(safe, "version")
        scope_id = _optional(safe, "scope_id", "scope")
        publisher = _optional(safe, "publisher", "maintainer")
        source_digest = content_digest(safe)
        units = [
            _make_unit(
                source_kind=self.source_kind,
                source_id=source_id,
                source_version=source_version,
                source_digest=source_digest,
                publisher=publisher,
                scope_id=scope_id,
                unit_kind="knowledge",
                content={
                    "name": str(safe.get("name", source_id)),
                    "description": str(safe.get("description", "")),
                    "documentation": safe.get("documentation", ()),
                },
                partition=partition,
                redaction_flags=flags,
            )
        ]
        for tool in _entries(safe.get("tools")):
            tool_content = _content(tool)
            units.append(
                _make_unit(
                    source_kind=self.source_kind,
                    source_id=source_id,
                    source_version=source_version,
                    source_digest=source_digest,
                    publisher=publisher,
                    scope_id=scope_id,
                    unit_kind="affordance",
                    content=tool_content,
                    partition=partition,
                    capability_semantics={"kind": "mcp_tool", "server_id": source_id},
                    input_schema=tool.get("input_schema", tool.get("schema"))
                    if isinstance(tool, Mapping)
                    else None,
                    output_schema=tool.get("output_schema") if isinstance(tool, Mapping) else None,
                    redaction_flags=flags,
                )
            )
            if isinstance(tool, Mapping):
                for constraint in _entries(tool.get("constraints")):
                    units.append(
                        _make_unit(
                            source_kind=self.source_kind,
                            source_id=source_id,
                            source_version=source_version,
                            source_digest=source_digest,
                            publisher=publisher,
                            scope_id=scope_id,
                            unit_kind="constraint",
                            content={"tool": tool.get("name", ""), "constraint": constraint},
                            partition=partition,
                            redaction_flags=flags,
                        )
                    )
        for field_name, unit_kind in (
            ("permissions", "constraint"),
            ("resources", "constraint"),
            ("errors", "constraint"),
            ("examples", "example"),
        ):
            for item in _entries(safe.get(field_name)):
                units.append(
                    _make_unit(
                        source_kind=self.source_kind,
                        source_id=source_id,
                        source_version=source_version,
                        source_digest=source_digest,
                        publisher=publisher,
                        scope_id=scope_id,
                        unit_kind=unit_kind,
                        content=_content(item),
                        partition=partition,
                        redaction_flags=flags,
                    )
                )
        call_flow = safe.get("call_flow")
        if call_flow is not None:
            units.append(
                _make_unit(
                    source_kind=self.source_kind,
                    source_id=source_id,
                    source_version=source_version,
                    source_digest=source_digest,
                    publisher=publisher,
                    scope_id=scope_id,
                    unit_kind="procedure",
                    content={"steps": _entries(call_flow)},
                    partition=partition,
                    redaction_flags=flags,
                )
            )
        return ArtifactCorpusProjection(
            self.source_kind,
            source_id,
            source_version,
            source_digest,
            scope_id,
            publisher,
            tuple(units),
            tuple(sorted(flags)),
        )


class ClientPluginArtifactAdapter:
    """Project client extension affordances, not plugin executable code."""

    source_kind = "client_plugin"

    def project(self, artifact: Mapping[str, Any], *, partition: str = "train") -> ArtifactCorpusProjection:
        if not isinstance(artifact, Mapping):
            raise TypeError("client plugin artifact must be a mapping")
        safe, flags = _safe_source(artifact)
        if not isinstance(safe, Mapping):
            raise TypeError("client plugin artifact must be mapping-like")
        source_id = _required(safe, "plugin_id", "id", "name")
        source_version = _required(safe, "version")
        scope_id = _optional(safe, "scope_id", "scope")
        publisher = _optional(safe, "publisher", "author")
        source_digest = content_digest(safe)
        units = [
            _make_unit(
                source_kind=self.source_kind,
                source_id=source_id,
                source_version=source_version,
                source_digest=source_digest,
                publisher=publisher,
                scope_id=scope_id,
                unit_kind="knowledge",
                content={
                    "name": str(safe.get("name", source_id)),
                    "description": str(safe.get("description", "")),
                    "lifecycle": safe.get("lifecycle", ()),
                },
                partition=partition,
                redaction_flags=flags,
            )
        ]
        ui = safe.get("ui", {})
        units.append(
            _make_unit(
                source_kind=self.source_kind,
                source_id=source_id,
                source_version=source_version,
                source_digest=source_digest,
                publisher=publisher,
                scope_id=scope_id,
                unit_kind="affordance",
                content={"ui": ui, "capabilities": safe.get("capabilities", ())},
                partition=partition,
                capability_semantics={"kind": "client_extension", "plugin_id": source_id},
                redaction_flags=flags,
            )
        )
        for field_name in ("dependencies", "permissions", "resources"):
            for item in _entries(safe.get(field_name)):
                units.append(
                    _make_unit(
                        source_kind=self.source_kind,
                        source_id=source_id,
                        source_version=source_version,
                        source_digest=source_digest,
                        publisher=publisher,
                        scope_id=scope_id,
                        unit_kind="constraint",
                        content={field_name: item},
                        partition=partition,
                        redaction_flags=flags,
                    )
                )
        for item in _entries(safe.get("examples", safe.get("lifecycle_examples"))):
            units.append(
                _make_unit(
                    source_kind=self.source_kind,
                    source_id=source_id,
                    source_version=source_version,
                    source_digest=source_digest,
                    publisher=publisher,
                    scope_id=scope_id,
                    unit_kind="example",
                    content=_content(item),
                    partition=partition,
                    redaction_flags=flags,
                )
            )
        return ArtifactCorpusProjection(
            self.source_kind,
            source_id,
            source_version,
            source_digest,
            scope_id,
            publisher,
            tuple(units),
            tuple(sorted(flags)),
        )


def runtime_event_to_experience(
    event: Mapping[str, Any],
    *,
    source_kind: str,
    source_id: str,
    source_version: str,
    source_digest: str,
    parent_checkpoint_digest: str,
    partition: str = "train",
    scope_id: str = "",
) -> EvolutionExperience:
    """Project one lifecycle event while retaining only digest-bound payloads."""

    if source_kind not in _EVENT_SOURCE_KINDS:
        raise ValueError(f"unsupported lifecycle source_kind: {source_kind}")
    if not isinstance(event, Mapping):
        raise TypeError("lifecycle event must be a mapping")
    safe, flags = _safe_source(event)
    if not isinstance(safe, Mapping):
        raise TypeError("lifecycle event must be mapping-like")
    event_kind = _required(safe, "event_kind", "kind", "operation")
    event_id = _optional(safe, "event_id", "id")
    if not event_id:
        event_id = content_digest(safe)[:24]
    status = _optional(safe, "status") or ("success" if bool(safe.get("success")) else "error")
    status = {"failed": "error", "failure": "error", "timeout": "cancelled"}.get(status, status)
    success = bool(safe.get("success", status == "success"))
    result_value = safe.get("result", safe.get("output", safe.get("response")))
    arguments_value = safe.get("arguments", safe.get("input"))
    correction_value = safe.get("user_correction")
    snapshot_value = safe.get("capability_snapshot", safe.get("snapshot"))
    schema_value = safe.get("schema", safe.get("input_schema"))
    resource_usage = safe.get("resource_usage") or {
        key: safe[key]
        for key in ("latency_ms", "cpu_ms", "memory_bytes", "output_bytes", "side_effect_count")
        if key in safe
    }
    reward_components = safe.get("reward_components") or safe.get("reward") or {}
    raw_taint_flags = safe.get("taint_flags", ())
    if isinstance(raw_taint_flags, Mapping):
        raise TypeError("lifecycle taint_flags must be a sequence of strings")
    taint_flags = [str(item) for item in _entries(raw_taint_flags)]
    if flags:
        taint_flags.append("redacted_external_source")
    metadata: dict[str, Any] = {
        "event_kind": event_kind,
        "scope_id": scope_id,
        "redaction_paths": list(flags),
    }
    if isinstance(safe.get("metadata"), Mapping):
        metadata["event_metadata_digest"] = content_digest(safe["metadata"])
    return EvolutionExperience(
        experience_id=f"{source_kind}:{source_id}:{event_id}",
        source_kind=source_kind,
        source_id=source_id,
        source_version=source_version,
        source_digest=source_digest,
        parent_checkpoint_digest=parent_checkpoint_digest,
        partition=partition,
        status=status,
        success=success,
        request_id=_optional(safe, "request_id"),
        intent_id=_optional(safe, "intent_id"),
        call_id=_optional(safe, "call_id", "call"),
        outcome_id=event_id,
        episode_id=_optional(safe, "episode_id"),
        tick=int(safe.get("tick", 0)),
        input_digest=content_digest(arguments_value) if arguments_value is not None else "",
        capability_id=_optional(safe, "capability_id", "tool_id", "operation"),
        capability_snapshot_id=_digest_for_schema(snapshot_value),
        arguments_digest=content_digest(arguments_value) if arguments_value is not None else "",
        approval_id=_optional(safe, "approval_id"),
        result_digest=content_digest(result_value) if result_value is not None else "",
        error_code=_optional(safe, "error_code"),
        reward_components=reward_components,
        resource_usage=resource_usage,
        user_correction_digest=content_digest(correction_value) if correction_value is not None else "",
        client_snapshot_id=_optional(safe, "client_snapshot_id"),
        skill_digest=source_digest if source_kind == "skill" else "",
        mcp_server_digest=source_digest if source_kind == "mcp" else "",
        mcp_schema_digest=_digest_for_schema(schema_value),
        plugin_digest=source_digest if source_kind == "client_plugin" else "",
        taint_flags=tuple(taint_flags),
        redaction_revision=EVOLUTION_REDACTION_REVISION,
        metadata=metadata,
    )


__all__ = [
    "ArtifactCorpusProjection",
    "ClientPluginArtifactAdapter",
    "McpArtifactAdapter",
    "SkillArtifactAdapter",
    "runtime_event_to_experience",
]

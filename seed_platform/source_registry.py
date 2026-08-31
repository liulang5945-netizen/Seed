"""Seed-owned lifecycle registry for declarative evolution sources.

This registry is deliberately narrower than a plugin host.  It records a
versioned source projection, lifecycle state and digest-only events.  Loading,
networking and execution remain outside this module; the resulting events can
be projected into the E1 evolution ledger after policy review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from seed_platform.evolution_adapters import ArtifactCorpusProjection
from taiji.evolution_experience import EvolutionCorpusArtifact
from taiji.internalization import content_digest

SOURCE_REGISTRY_CHECKPOINT_FORMAT = "seed-evolution-source-registry-v1"
SOURCE_REGISTRY_VERSION = 1
SOURCE_LIFECYCLE_STATES = (
    "discovered",
    "staged",
    "shadow",
    "active",
    "failed",
    "quarantined",
    "retired",
)
_ALLOWED_TRANSITIONS = {
    "discovered": frozenset({"staged", "failed", "quarantined"}),
    "staged": frozenset({"shadow", "active", "failed", "quarantined"}),
    "shadow": frozenset({"active", "failed", "quarantined"}),
    "active": frozenset({"failed", "quarantined", "retired"}),
    "failed": frozenset({"staged", "quarantined", "retired"}),
    "quarantined": frozenset({"staged", "retired"}),
    "retired": frozenset(),
}


@dataclass(frozen=True)
class SourceRegistryEntry:
    projection: ArtifactCorpusProjection
    partition: str
    state: str
    snapshot_digest: str
    events: tuple[Mapping[str, Any], ...]

    @property
    def source_id(self) -> str:
        return self.projection.source_id

    @property
    def source_version(self) -> str:
        return self.projection.source_version

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_kind": self.projection.source_kind,
            "source_id": self.projection.source_id,
            "source_version": self.projection.source_version,
            "source_digest": self.projection.source_digest,
            "scope_id": self.projection.scope_id,
            "publisher": self.projection.publisher,
            "partition": self.partition,
            "state": self.state,
            "snapshot_digest": self.snapshot_digest,
            "redaction_flags": list(self.projection.redaction_flags),
            "corpus": [item.to_payload() for item in self.projection.corpus],
            "events": [dict(event) for event in self.events],
        }


class DeclarativeSourceRegistry:
    """Register and recover one source family without executing it."""

    def __init__(self, adapter: Any) -> None:
        if not isinstance(getattr(adapter, "source_kind", None), str):
            raise TypeError("source registry adapter must declare source_kind")
        self.adapter = adapter
        self.source_kind = str(adapter.source_kind)
        self.revision = 0
        self._entries: dict[tuple[str, str], SourceRegistryEntry] = {}

    @property
    def entries(self) -> tuple[SourceRegistryEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    @property
    def snapshot_id(self) -> str:
        return content_digest(
            {
                "format": SOURCE_REGISTRY_CHECKPOINT_FORMAT,
                "version": SOURCE_REGISTRY_VERSION,
                "source_kind": self.source_kind,
                "revision": self.revision,
                "entries": [
                    {
                        "source_id": entry.source_id,
                        "source_version": entry.source_version,
                        "source_digest": entry.projection.source_digest,
                        "state": entry.state,
                        "snapshot_digest": entry.snapshot_digest,
                    }
                    for entry in self.entries
                ],
            }
        )

    def get(self, source_id: str, source_version: str) -> SourceRegistryEntry | None:
        return self._entries.get((str(source_id).strip(), str(source_version).strip()))

    def register(
        self,
        artifact: Mapping[str, Any],
        *,
        partition: str = "train",
    ) -> ArtifactCorpusProjection:
        projection = self.adapter.project(artifact, partition=partition)
        key = (projection.source_id, projection.source_version)
        existing = self._entries.get(key)
        if existing is not None:
            if existing.projection.source_digest != projection.source_digest:
                raise ValueError("source identity is already registered with another digest")
            return existing.projection
        snapshot_digest = self._entry_snapshot_digest(projection, "discovered")
        event = self._lifecycle_event(
            projection,
            state="discovered",
            snapshot_digest=snapshot_digest,
            event_id=f"discover:{projection.source_id}:{projection.source_version}",
        )
        self.revision += 1
        self._entries[key] = SourceRegistryEntry(
            projection=projection,
            partition=partition,
            state="discovered",
            snapshot_digest=snapshot_digest,
            events=(event,),
        )
        return projection

    def transition(
        self,
        source_id: str,
        source_version: str,
        state: str,
        *,
        error_code: str = "",
    ) -> SourceRegistryEntry:
        key = (str(source_id).strip(), str(source_version).strip())
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"unknown {self.source_kind} source: {source_id}:{source_version}")
        target = str(state).strip()
        if target not in SOURCE_LIFECYCLE_STATES:
            raise ValueError(f"unsupported source lifecycle state: {target}")
        if target == entry.state:
            return entry
        if target not in _ALLOWED_TRANSITIONS[entry.state]:
            raise ValueError(f"invalid source lifecycle transition: {entry.state}->{target}")
        next_revision = self.revision + 1
        snapshot_digest = self._entry_snapshot_digest(entry.projection, target)
        event = self._lifecycle_event(
            entry.projection,
            state=target,
            snapshot_digest=snapshot_digest,
            event_id=f"{target}:{entry.source_id}:{entry.source_version}:{next_revision}",
            error_code=error_code,
        )
        self.revision = next_revision
        updated = SourceRegistryEntry(
            projection=entry.projection,
            partition=entry.partition,
            state=target,
            snapshot_digest=snapshot_digest,
            events=(*entry.events, event),
        )
        self._entries[key] = updated
        return updated

    def project_to_ledger(self, ledger: Any, *, parent_checkpoint_digest: str) -> tuple[Any, ...]:
        """Project source corpus and lifecycle events into an E1 ledger."""

        results: list[Any] = []
        for entry in self.entries:
            for artifact in entry.projection.corpus:
                ledger.add_corpus(artifact)
            for event in entry.events:
                experience = entry.projection.project_event(
                    event,
                    parent_checkpoint_digest=parent_checkpoint_digest,
                    partition=entry.partition,
                )
                results.append(ledger.append(experience))
        return tuple(results)

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": SOURCE_REGISTRY_CHECKPOINT_FORMAT,
            "version": SOURCE_REGISTRY_VERSION,
            "source_kind": self.source_kind,
            "revision": self.revision,
            "snapshot_id": self.snapshot_id,
            "entries": [entry.to_payload() for entry in self.entries],
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any], *, adapter: Any) -> DeclarativeSourceRegistry:
        if payload.get("format") != SOURCE_REGISTRY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported source registry checkpoint format")
        if int(payload.get("version", -1)) != SOURCE_REGISTRY_VERSION:
            raise ValueError("unsupported source registry checkpoint version")
        expected_digest = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected_digest:
            raise ValueError("source registry checkpoint digest mismatch")
        registry = cls(adapter)
        if str(payload.get("source_kind", "")) != registry.source_kind:
            raise ValueError("source registry adapter kind mismatch")
        raw_entries = payload.get("entries", ())
        if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence):
            raise ValueError("source registry entries must be a sequence")
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise TypeError("source registry entry must be a mapping")
            raw_corpus = raw_entry.get("corpus", ())
            if isinstance(raw_corpus, (str, bytes)) or not isinstance(raw_corpus, Sequence):
                raise ValueError("source registry corpus must be a sequence")
            corpus = tuple(EvolutionCorpusArtifact.from_payload(item) for item in raw_corpus)
            source_kind = str(raw_entry.get("source_kind", ""))
            if source_kind != registry.source_kind:
                raise ValueError("source registry corpus kind mismatch")
            if any(item.source_kind != f"{registry.source_kind}_artifact" for item in corpus):
                raise ValueError("source registry artifact kind mismatch")
            projection = ArtifactCorpusProjection(
                source_kind=registry.source_kind,
                source_id=str(raw_entry["source_id"]),
                source_version=str(raw_entry["source_version"]),
                source_digest=str(raw_entry["source_digest"]),
                scope_id=str(raw_entry.get("scope_id", "")),
                publisher=str(raw_entry.get("publisher", "")),
                corpus=corpus,
                redaction_flags=tuple(str(item) for item in raw_entry.get("redaction_flags", ())),
            )
            state = str(raw_entry["state"])
            if state not in SOURCE_LIFECYCLE_STATES:
                raise ValueError("source registry entry has unsupported state")
            snapshot_digest = str(raw_entry["snapshot_digest"])
            if snapshot_digest != registry._entry_snapshot_digest(projection, state):
                raise ValueError("source registry entry snapshot mismatch")
            raw_events = raw_entry.get("events", ())
            if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
                raise ValueError("source registry events must be a sequence")
            events = tuple(dict(event) for event in raw_events)
            if not events:
                raise ValueError("source registry entry must retain discovery event")
            key = (projection.source_id, projection.source_version)
            if key in registry._entries:
                raise ValueError("duplicate source registry identity")
            registry._entries[key] = SourceRegistryEntry(
                projection=projection,
                partition=str(raw_entry["partition"]),
                state=state,
                snapshot_digest=snapshot_digest,
                events=events,
            )
        registry.revision = int(payload.get("revision", -1))
        expected_revision = sum(len(entry.events) for entry in registry.entries)
        if registry.revision != expected_revision:
            raise ValueError("source registry checkpoint revision mismatch")
        if str(payload.get("snapshot_id", "")) != registry.snapshot_id:
            raise ValueError("source registry checkpoint snapshot mismatch")
        return registry

    @staticmethod
    def _entry_snapshot_digest(projection: ArtifactCorpusProjection, state: str) -> str:
        return content_digest(
            {
                "source_kind": projection.source_kind,
                "source_id": projection.source_id,
                "source_version": projection.source_version,
                "source_digest": projection.source_digest,
                "scope_id": projection.scope_id,
                "publisher": projection.publisher,
                "state": state,
                "corpus_digests": [item.artifact_digest for item in projection.corpus],
            }
        )

    def _lifecycle_event(
        self,
        projection: ArtifactCorpusProjection,
        *,
        state: str,
        snapshot_digest: str,
        event_id: str,
        error_code: str = "",
    ) -> dict[str, Any]:
        failed = state == "failed"
        return {
            "event_id": event_id,
            "event_kind": f"lifecycle.{state}",
            "status": "error" if failed else "success",
            "success": not failed,
            "capability_id": projection.source_id,
            "capability_snapshot": {
                "source_kind": projection.source_kind,
                "source_id": projection.source_id,
                "source_version": projection.source_version,
                "state": state,
                "snapshot_digest": snapshot_digest,
            },
            "result": {"state": state},
            "error_code": error_code if failed else "",
            "tick": self.revision + 1,
            "metadata": {"registry": self.source_kind, "lifecycle_state": state},
        }


__all__ = [
    "SOURCE_LIFECYCLE_STATES",
    "SOURCE_REGISTRY_CHECKPOINT_FORMAT",
    "SOURCE_REGISTRY_VERSION",
    "DeclarativeSourceRegistry",
    "SourceRegistryEntry",
]

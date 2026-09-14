"""Content-addressed checkpoint state for resumable Taiji continuation.

This contract is deliberately narrower than a full S/G/K runtime.  It records
the state needed to resume one deterministic continuation stream: worker
checkpoint digests and references, phase cursor, ordered experience stream,
RNG state, budgets, and explicit parent lineage.  It does not decide whether
growth or promotion is allowed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .internalization import content_digest

TAIJI_CONTINUATION_CHECKPOINT_FORMAT = "taiji-continuation-checkpoint-v1"
TAIJI_CONTINUATION_CHECKPOINT_VERSION = 1
CONTINUATION_PHASES = ("wake", "replay", "consolidate")


def _text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _digest(value: Any, name: str) -> str:
    normalized = _text(value, name)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _pairs(value: Any, name: str, *, digest_values: bool = False) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = tuple(value)
    else:
        raise TypeError(f"{name} must be a mapping or pair sequence")
    normalized: list[tuple[str, Any]] = []
    for item in items:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"{name} entries must be pairs")
        key = _text(item[0], f"{name} key")
        value_item = item[1]
        if digest_values:
            value_item = _digest(value_item, f"{name}[{key}]")
        normalized.append((key, value_item))
    if not normalized or len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} must contain unique non-empty keys")
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _digest_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(_digest(item, f"{name} item") for item in value)


@dataclass(frozen=True)
class ContinuationPhaseCursor:
    """Position inside one phase of a deterministic continuation stream."""

    phase: str
    index: int
    total: int

    def __post_init__(self) -> None:
        phase = _text(self.phase, "phase")
        if phase not in CONTINUATION_PHASES:
            raise ValueError(f"unsupported continuation phase: {phase}")
        index = int(self.index)
        total = int(self.total)
        if total < 0 or index < 0 or index > total:
            raise ValueError("continuation phase cursor is outside its phase")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "total", total)

    def to_payload(self) -> dict[str, Any]:
        return {"phase": self.phase, "index": self.index, "total": self.total}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContinuationPhaseCursor:
        if not isinstance(payload, Mapping):
            raise TypeError("continuation phase cursor must be a mapping")
        return cls(
            phase=str(payload["phase"]),
            index=int(payload["index"]),
            total=int(payload["total"]),
        )


@dataclass(frozen=True)
class TaijiContinuationCheckpoint:
    """Signed-by-content state for a resumable continuation boundary."""

    parent_checkpoint_digest: str
    worker_checkpoint_digests: tuple[tuple[str, str], ...]
    worker_checkpoint_refs: tuple[tuple[str, str], ...]
    phase_cursor: ContinuationPhaseCursor
    experience_digests: tuple[str, ...]
    stream_digest: str
    rng_state: Any
    rng_state_digest: str
    budget_counts: tuple[tuple[str, int], ...]
    origin_parent_digest: str
    attached_parent_digest: str
    lineage_chain: tuple[str, ...]
    checkpoint_digest: str
    format: str = TAIJI_CONTINUATION_CHECKPOINT_FORMAT
    version: int = TAIJI_CONTINUATION_CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_CONTINUATION_CHECKPOINT_FORMAT:
            raise ValueError("unsupported continuation checkpoint format")
        if int(self.version) != TAIJI_CONTINUATION_CHECKPOINT_VERSION:
            raise ValueError("unsupported continuation checkpoint version")
        parent = _digest(self.parent_checkpoint_digest, "parent_checkpoint_digest")
        worker_digests = _pairs(
            self.worker_checkpoint_digests,
            "worker_checkpoint_digests",
            digest_values=True,
        )
        worker_refs = _pairs(self.worker_checkpoint_refs, "worker_checkpoint_refs")
        if tuple(key for key, _ in worker_digests) != tuple(key for key, _ in worker_refs):
            raise ValueError("worker checkpoint digests and refs must have the same owners")
        if len(worker_digests) < 2:
            raise ValueError("continuation checkpoint needs at least two worker payloads")
        experiences = _digest_sequence(self.experience_digests, "experience_digests")
        stream = _digest(self.stream_digest, "stream_digest")
        rng_digest = _digest(self.rng_state_digest, "rng_state_digest")
        if content_digest(self.rng_state) != rng_digest:
            raise ValueError("continuation RNG state digest mismatch")
        budgets = _pairs(self.budget_counts, "budget_counts")
        normalized_budgets: list[tuple[str, int]] = []
        for key, value in budgets:
            count = int(value)
            if count < 0:
                raise ValueError("continuation budget counts cannot be negative")
            normalized_budgets.append((key, count))
        origin = _digest(self.origin_parent_digest, "origin_parent_digest")
        attached = _digest(self.attached_parent_digest, "attached_parent_digest")
        lineage = _digest_sequence(self.lineage_chain, "lineage_chain")
        if not lineage or lineage[0] != origin or lineage[-1] != attached:
            raise ValueError("continuation lineage must start at origin and end at attached parent")
        checkpoint = _digest(self.checkpoint_digest, "checkpoint_digest")
        object.__setattr__(self, "parent_checkpoint_digest", parent)
        object.__setattr__(self, "worker_checkpoint_digests", worker_digests)
        object.__setattr__(self, "worker_checkpoint_refs", worker_refs)
        object.__setattr__(self, "experience_digests", experiences)
        object.__setattr__(self, "stream_digest", stream)
        object.__setattr__(self, "rng_state_digest", rng_digest)
        object.__setattr__(self, "budget_counts", tuple(normalized_budgets))
        object.__setattr__(self, "origin_parent_digest", origin)
        object.__setattr__(self, "attached_parent_digest", attached)
        object.__setattr__(self, "lineage_chain", lineage)
        object.__setattr__(self, "checkpoint_digest", checkpoint)

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "worker_checkpoint_digests": dict(self.worker_checkpoint_digests),
            "worker_checkpoint_refs": dict(self.worker_checkpoint_refs),
            "phase_cursor": self.phase_cursor.to_payload(),
            "experience_digests": list(self.experience_digests),
            "stream_digest": self.stream_digest,
            "rng_state": self.rng_state,
            "rng_state_digest": self.rng_state_digest,
            "budget_counts": dict(self.budget_counts),
            "origin_parent_digest": self.origin_parent_digest,
            "attached_parent_digest": self.attached_parent_digest,
            "lineage_chain": list(self.lineage_chain),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._unsigned_payload(), "checkpoint_digest": self.checkpoint_digest}

    def assert_parent(self, expected_parent_digest: str) -> None:
        expected = _digest(expected_parent_digest, "expected_parent_digest")
        if expected != self.parent_checkpoint_digest:
            raise ValueError("continuation checkpoint crosses the expected parent")

    @classmethod
    def create(
        cls,
        *,
        parent_checkpoint_digest: str,
        worker_checkpoint_digests: Mapping[str, str] | Sequence[tuple[str, str]],
        worker_checkpoint_refs: Mapping[str, str] | Sequence[tuple[str, str]],
        phase_cursor: ContinuationPhaseCursor,
        experience_digests: Sequence[str],
        stream_digest: str,
        rng_state: Any,
        budget_counts: Mapping[str, int] | Sequence[tuple[str, int]],
        origin_parent_digest: str,
        attached_parent_digest: str,
        lineage_chain: Sequence[str],
    ) -> TaijiContinuationCheckpoint:
        provisional = {
            "format": TAIJI_CONTINUATION_CHECKPOINT_FORMAT,
            "version": TAIJI_CONTINUATION_CHECKPOINT_VERSION,
            "parent_checkpoint_digest": _digest(
                parent_checkpoint_digest, "parent_checkpoint_digest"
            ),
            "worker_checkpoint_digests": dict(
                _pairs(worker_checkpoint_digests, "worker_checkpoint_digests", digest_values=True)
            ),
            "worker_checkpoint_refs": dict(
                _pairs(worker_checkpoint_refs, "worker_checkpoint_refs")
            ),
            "phase_cursor": phase_cursor.to_payload(),
            "experience_digests": list(_digest_sequence(experience_digests, "experience_digests")),
            "stream_digest": _digest(stream_digest, "stream_digest"),
            "rng_state": rng_state,
            "rng_state_digest": content_digest(rng_state),
            "budget_counts": dict(_pairs(budget_counts, "budget_counts")),
            "origin_parent_digest": _digest(origin_parent_digest, "origin_parent_digest"),
            "attached_parent_digest": _digest(attached_parent_digest, "attached_parent_digest"),
            "lineage_chain": list(_digest_sequence(lineage_chain, "lineage_chain")),
        }
        checkpoint_digest = content_digest(provisional)
        provisional.pop("phase_cursor")
        return cls(
            **provisional,
            phase_cursor=phase_cursor,
            checkpoint_digest=checkpoint_digest,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaijiContinuationCheckpoint:
        if not isinstance(payload, Mapping):
            raise TypeError("continuation checkpoint payload must be a mapping")
        without_digest = {
            key: value for key, value in payload.items() if key != "checkpoint_digest"
        }
        expected = content_digest(without_digest)
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("continuation checkpoint digest mismatch")
        checkpoint = cls(
            format=str(payload["format"]),
            version=int(payload["version"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            worker_checkpoint_digests=tuple(payload["worker_checkpoint_digests"].items()),
            worker_checkpoint_refs=tuple(payload["worker_checkpoint_refs"].items()),
            phase_cursor=ContinuationPhaseCursor.from_payload(payload["phase_cursor"]),
            experience_digests=tuple(payload["experience_digests"]),
            stream_digest=str(payload["stream_digest"]),
            rng_state=payload["rng_state"],
            rng_state_digest=str(payload["rng_state_digest"]),
            budget_counts=tuple(payload["budget_counts"].items()),
            origin_parent_digest=str(payload["origin_parent_digest"]),
            attached_parent_digest=str(payload["attached_parent_digest"]),
            lineage_chain=tuple(payload["lineage_chain"]),
            checkpoint_digest=str(payload["checkpoint_digest"]),
        )
        if content_digest(checkpoint._unsigned_payload()) != checkpoint.checkpoint_digest:
            raise ValueError("continuation checkpoint normalized digest mismatch")
        return checkpoint


__all__ = [
    "CONTINUATION_PHASES",
    "TAIJI_CONTINUATION_CHECKPOINT_FORMAT",
    "TAIJI_CONTINUATION_CHECKPOINT_VERSION",
    "ContinuationPhaseCursor",
    "TaijiContinuationCheckpoint",
]

"""Content-addressed collections of replay-bound structural validation artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

STRUCTURAL_VALIDATION_ARTIFACT_BATCH_FORMAT = (
    "taiji-workbench-structural-validation-artifact-batch-v1"
)


def _batch_digest(
    *,
    batch_id: str,
    expected_candidate_ids: tuple[str, ...],
    artifact_digests_by_candidate: Mapping[str, str],
) -> str:
    payload = {
        "format": STRUCTURAL_VALIDATION_ARTIFACT_BATCH_FORMAT,
        "batch_id": str(batch_id),
        "expected_candidate_ids": list(expected_candidate_ids),
        "artifact_digests_by_candidate": {
            str(candidate_id): str(artifact_digest)
            for candidate_id, artifact_digest in sorted(
                artifact_digests_by_candidate.items()
            )
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StructuralValidationArtifactBatch:
    """A checkpointable, incrementally consumable artifact collection."""

    batch_id: str
    expected_candidate_ids: tuple[str, ...]
    artifact_digests_by_candidate: tuple[tuple[str, str], ...]
    batch_digest: str

    def __post_init__(self) -> None:
        batch_id = str(self.batch_id)
        expected = tuple(str(item) for item in self.expected_candidate_ids)
        entries = tuple(
            (str(candidate_id), str(artifact_digest))
            for candidate_id, artifact_digest in self.artifact_digests_by_candidate
        )
        if not batch_id:
            raise ValueError("structural validation artifact batch_id must not be empty")
        if not expected or any(not item for item in expected):
            raise ValueError(
                "structural validation artifact expected candidates must not be empty"
            )
        if len(set(expected)) != len(expected):
            raise ValueError(
                "structural validation artifact expected candidates must be unique"
            )
        if len({candidate_id for candidate_id, _ in entries}) != len(entries):
            raise ValueError(
                "structural validation artifact batch candidates must be unique"
            )
        if any(candidate_id not in expected for candidate_id, _ in entries):
            raise ValueError(
                "structural validation artifact batch contains an unexpected candidate"
            )
        if any(not candidate_id or not artifact_digest for candidate_id, artifact_digest in entries):
            raise ValueError(
                "structural validation artifact batch entries must not be empty"
            )
        if not str(self.batch_digest):
            raise ValueError("structural validation artifact batch digest must not be empty")
        entries = tuple(sorted(entries))
        expected = tuple(sorted(expected))
        calculated = _batch_digest(
            batch_id=batch_id,
            expected_candidate_ids=expected,
            artifact_digests_by_candidate=dict(entries),
        )
        if calculated != str(self.batch_digest):
            raise ValueError("structural validation artifact batch digest mismatch")
        object.__setattr__(self, "batch_id", batch_id)
        object.__setattr__(self, "expected_candidate_ids", expected)
        object.__setattr__(self, "artifact_digests_by_candidate", entries)
        object.__setattr__(self, "batch_digest", str(self.batch_digest))

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate_id for candidate_id, _ in self.artifact_digests_by_candidate)

    @property
    def complete(self) -> bool:
        return set(self.candidate_ids) == set(self.expected_candidate_ids)

    @classmethod
    def from_digest_map(
        cls,
        *,
        batch_id: str,
        expected_candidate_ids: tuple[str, ...],
        artifact_digests_by_candidate: Mapping[str, str],
    ) -> StructuralValidationArtifactBatch:
        expected = tuple(sorted(str(item) for item in expected_candidate_ids))
        entries = {
            str(candidate_id): str(artifact_digest)
            for candidate_id, artifact_digest in artifact_digests_by_candidate.items()
        }
        return cls(
            batch_id=str(batch_id),
            expected_candidate_ids=expected,
            artifact_digests_by_candidate=tuple(sorted(entries.items())),
            batch_digest=_batch_digest(
                batch_id=str(batch_id),
                expected_candidate_ids=expected,
                artifact_digests_by_candidate=entries,
            ),
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        batch_id: str,
        expected_candidate_ids: tuple[str, ...],
        artifacts: Mapping[str, Any],
    ) -> StructuralValidationArtifactBatch:
        digest_map = {
            str(candidate_id): str(getattr(artifact, "artifact_digest"))
            for candidate_id, artifact in artifacts.items()
        }
        return cls.from_digest_map(
            batch_id=batch_id,
            expected_candidate_ids=expected_candidate_ids,
            artifact_digests_by_candidate=digest_map,
        )

    def merge(
        self,
        other: StructuralValidationArtifactBatch,
    ) -> StructuralValidationArtifactBatch:
        if self.batch_id != other.batch_id:
            raise ValueError("cannot merge validation artifact batches with different ids")
        if self.expected_candidate_ids != other.expected_candidate_ids:
            raise ValueError(
                "cannot merge validation artifact batches with different candidate sets"
            )
        merged = dict(self.artifact_digests_by_candidate)
        for candidate_id, artifact_digest in other.artifact_digests_by_candidate:
            previous = merged.get(candidate_id)
            if previous is not None and previous != artifact_digest:
                raise ValueError(
                    "validation artifact candidate is bound to conflicting artifact digests"
                )
            merged[candidate_id] = artifact_digest
        return self.from_digest_map(
            batch_id=self.batch_id,
            expected_candidate_ids=self.expected_candidate_ids,
            artifact_digests_by_candidate=merged,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_VALIDATION_ARTIFACT_BATCH_FORMAT,
            "batch_id": self.batch_id,
            "expected_candidate_ids": list(self.expected_candidate_ids),
            "artifact_digests_by_candidate": {
                candidate_id: artifact_digest
                for candidate_id, artifact_digest in self.artifact_digests_by_candidate
            },
            "complete": self.complete,
            "batch_digest": self.batch_digest,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> StructuralValidationArtifactBatch:
        if payload.get("format") != STRUCTURAL_VALIDATION_ARTIFACT_BATCH_FORMAT:
            raise ValueError("unsupported structural validation artifact batch format")
        digest_map = payload.get("artifact_digests_by_candidate", {})
        if not isinstance(digest_map, Mapping):
            raise TypeError("validation artifact batch digest map must be a mapping")
        return cls(
            batch_id=str(payload["batch_id"]),
            expected_candidate_ids=tuple(
                str(item) for item in payload.get("expected_candidate_ids", ())
            ),
            artifact_digests_by_candidate=tuple(
                (str(candidate_id), str(artifact_digest))
                for candidate_id, artifact_digest in digest_map.items()
            ),
            batch_digest=str(payload["batch_digest"]),
        )

"""Content-addressed long-horizon evidence windows for Taiji structural growth.

This module is deliberately an evidence ledger, not a growth controller.  It
preserves bounded, replayable runtime observations so a later stage can make a
growth decision from a window instead of from one tick or a scale target.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .structural_growth import STRUCTURAL_EVIDENCE_PARTITIONS, StructuralRuntimeObservation

STRUCTURAL_EVIDENCE_WINDOW_CHECKPOINT_FORMAT = "taiji-structural-evidence-window-v1"
STRUCTURAL_EVIDENCE_LEDGER_CHECKPOINT_FORMAT = "taiji-structural-evidence-ledger-v1"
STRUCTURAL_EVIDENCE_COMPACTED_WINDOW_FORMAT = "taiji-structural-evidence-compacted-window-v1"
STRUCTURAL_EVIDENCE_CONSUMPTION_AUDIT_FORMAT = "taiji-structural-evidence-consumption-audit-v1"
STRUCTURAL_EVIDENCE_COMPACTION_RESULT_FORMAT = "taiji-structural-evidence-compaction-result-v1"
STRUCTURAL_EVIDENCE_PRESSURE_SNAPSHOT_FORMAT = "taiji-structural-evidence-pressure-snapshot-v1"
DEFAULT_MAX_COMPACTED_WINDOWS = 256


def _content_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


@dataclass(frozen=True)
class StructuralEvidenceWindowSummary:
    """Immutable aggregate of one bounded observation window."""

    window_id: str
    network_id: str
    region_id: str
    task_slice_ids: tuple[str, ...]
    partition_counts: tuple[tuple[str, int], ...]
    first_tick: int
    last_tick: int
    observation_count: int
    prediction_observation_count: int
    mean_usage: float
    mean_resource_pressure: float
    mean_prediction_error: float | None
    mean_learning_gain: float
    mean_holdout_transfer: float
    evidence_ids: tuple[str, ...]
    window_digest: str

    def __post_init__(self) -> None:
        for name in ("window_id", "network_id", "region_id", "window_digest"):
            if not str(getattr(self, name)):
                raise ValueError(f"structural evidence {name} must not be empty")
        if int(self.first_tick) <= 0 or int(self.last_tick) < int(self.first_tick):
            raise ValueError("structural evidence window ticks are invalid")
        if int(self.observation_count) <= 0:
            raise ValueError("structural evidence observation_count must be positive")
        if not 0 <= int(self.prediction_observation_count) <= int(self.observation_count):
            raise ValueError("structural evidence prediction count is invalid")
        for name in (
            "mean_usage",
            "mean_resource_pressure",
            "mean_learning_gain",
            "mean_holdout_transfer",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"structural evidence {name} must be in [0, 1]")
        if (
            self.mean_prediction_error is not None
            and not 0.0 <= float(self.mean_prediction_error) <= 1.0
        ):
            raise ValueError("structural evidence mean_prediction_error must be in [0, 1]")
        ids = tuple(str(item) for item in self.evidence_ids)
        if len(ids) != int(self.observation_count) or len(set(ids)) != len(ids):
            raise ValueError("structural evidence ids must match unique observation count")
        if any(not item for item in ids):
            raise ValueError("structural evidence ids must not be empty")
        task_slice_ids = tuple(str(item) for item in self.task_slice_ids)
        if len(set(task_slice_ids)) != len(task_slice_ids) or any(
            not item for item in task_slice_ids
        ):
            raise ValueError("structural evidence task_slice_ids must be unique and non-empty")
        partition_counts = tuple((str(partition), int(count)) for partition, count in self.partition_counts)
        if any(partition not in STRUCTURAL_EVIDENCE_PARTITIONS for partition, _ in partition_counts):
            raise ValueError("structural evidence partition is not supported")
        if any(count <= 0 for _, count in partition_counts):
            raise ValueError("structural evidence partition counts must be positive")
        if sum(count for _, count in partition_counts) != int(self.observation_count):
            raise ValueError("structural evidence partition counts must match observations")
        if len({partition for partition, _ in partition_counts}) != len(partition_counts):
            raise ValueError("structural evidence partitions must be unique")
        object.__setattr__(self, "window_id", str(self.window_id))
        object.__setattr__(self, "network_id", str(self.network_id))
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "task_slice_ids", task_slice_ids)
        object.__setattr__(self, "partition_counts", partition_counts)
        object.__setattr__(self, "first_tick", int(self.first_tick))
        object.__setattr__(self, "last_tick", int(self.last_tick))
        object.__setattr__(self, "observation_count", int(self.observation_count))
        object.__setattr__(
            self,
            "prediction_observation_count",
            int(self.prediction_observation_count),
        )
        for name in (
            "mean_usage",
            "mean_resource_pressure",
            "mean_learning_gain",
            "mean_holdout_transfer",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(
            self,
            "mean_prediction_error",
            None if self.mean_prediction_error is None else float(self.mean_prediction_error),
        )
        object.__setattr__(self, "evidence_ids", ids)
        object.__setattr__(self, "window_digest", str(self.window_digest))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_EVIDENCE_WINDOW_CHECKPOINT_FORMAT,
            "window_id": self.window_id,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "task_slice_ids": list(self.task_slice_ids),
            "partition_counts": {
                partition: count for partition, count in self.partition_counts
            },
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "observation_count": self.observation_count,
            "prediction_observation_count": self.prediction_observation_count,
            "mean_usage": self.mean_usage,
            "mean_resource_pressure": self.mean_resource_pressure,
            "mean_prediction_error": self.mean_prediction_error,
            "mean_learning_gain": self.mean_learning_gain,
            "mean_holdout_transfer": self.mean_holdout_transfer,
            "evidence_ids": list(self.evidence_ids),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "window_digest": self.window_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralEvidenceWindowSummary:
        if payload.get("format") != STRUCTURAL_EVIDENCE_WINDOW_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural evidence window format")
        expected_digest = _content_digest(
            {key: value for key, value in payload.items() if key != "window_digest"}
        )
        if str(payload.get("window_digest")) != expected_digest:
            raise ValueError("structural evidence window digest mismatch")
        return cls(
            window_id=str(payload["window_id"]),
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            task_slice_ids=tuple(str(item) for item in payload.get("task_slice_ids", ())),
            partition_counts=tuple(
                (str(partition), int(count))
                for partition, count in dict(payload.get("partition_counts", {})).items()
            ),
            first_tick=int(payload["first_tick"]),
            last_tick=int(payload["last_tick"]),
            observation_count=int(payload["observation_count"]),
            prediction_observation_count=int(payload.get("prediction_observation_count", 0)),
            mean_usage=float(payload["mean_usage"]),
            mean_resource_pressure=float(payload["mean_resource_pressure"]),
            mean_prediction_error=(
                None
                if payload.get("mean_prediction_error") is None
                else float(payload["mean_prediction_error"])
            ),
            mean_learning_gain=float(payload["mean_learning_gain"]),
            mean_holdout_transfer=float(payload["mean_holdout_transfer"]),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
            window_digest=str(payload["window_digest"]),
        )


@dataclass(frozen=True)
class StructuralEvidenceCompactedWindow:
    """Bounded provenance retained after a consumed window is compacted.

    Raw observations and aggregate metrics are intentionally discarded from
    the active ledger, while source identity, task/partition attribution,
    evidence identity, and the scheduler revision that consumed the window
    remain replay-auditable.
    """

    window_id: str
    network_id: str
    region_id: str
    task_slice_ids: tuple[str, ...]
    partition_counts: tuple[tuple[str, int], ...]
    first_tick: int
    last_tick: int
    observation_count: int
    prediction_observation_count: int
    evidence_ids: tuple[str, ...]
    evidence_digests: tuple[tuple[str, str], ...]
    window_digest: str
    consumed_scheduler_revision: int
    provenance_digest: str

    def __post_init__(self) -> None:
        for name in (
            "window_id",
            "network_id",
            "region_id",
            "window_digest",
            "provenance_digest",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"compacted structural evidence {name} must not be empty")
        if int(self.first_tick) <= 0 or int(self.last_tick) < int(self.first_tick):
            raise ValueError("compacted structural evidence ticks are invalid")
        if int(self.observation_count) <= 0:
            raise ValueError("compacted structural evidence observation_count must be positive")
        if not 0 <= int(self.prediction_observation_count) <= int(self.observation_count):
            raise ValueError("compacted structural evidence prediction count is invalid")
        if int(self.consumed_scheduler_revision) < 0:
            raise ValueError("compacted structural evidence scheduler revision cannot be negative")
        task_slice_ids = tuple(str(item) for item in self.task_slice_ids)
        if len(set(task_slice_ids)) != len(task_slice_ids) or any(
            not item for item in task_slice_ids
        ):
            raise ValueError("compacted structural evidence task slices must be unique and non-empty")
        partition_counts = tuple((str(partition), int(count)) for partition, count in self.partition_counts)
        if any(partition not in STRUCTURAL_EVIDENCE_PARTITIONS for partition, _ in partition_counts):
            raise ValueError("compacted structural evidence partition is not supported")
        if any(count <= 0 for _, count in partition_counts):
            raise ValueError("compacted structural evidence partition counts must be positive")
        if sum(count for _, count in partition_counts) != int(self.observation_count):
            raise ValueError("compacted structural evidence partition counts must match observations")
        if len({partition for partition, _ in partition_counts}) != len(partition_counts):
            raise ValueError("compacted structural evidence partitions must be unique")
        evidence_ids = tuple(str(item) for item in self.evidence_ids)
        if len(evidence_ids) != int(self.observation_count) or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("compacted structural evidence ids must match unique observation count")
        if any(not item for item in evidence_ids):
            raise ValueError("compacted structural evidence ids must not be empty")
        evidence_digests = tuple((str(evidence_id), str(digest)) for evidence_id, digest in self.evidence_digests)
        if len(evidence_digests) != len(evidence_ids) or tuple(item[0] for item in evidence_digests) != evidence_ids:
            raise ValueError("compacted structural evidence digests must align with evidence ids")
        if any(not evidence_id or not digest for evidence_id, digest in evidence_digests):
            raise ValueError("compacted structural evidence digests must not be empty")
        object.__setattr__(self, "window_id", str(self.window_id))
        object.__setattr__(self, "network_id", str(self.network_id))
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "task_slice_ids", task_slice_ids)
        object.__setattr__(self, "partition_counts", partition_counts)
        object.__setattr__(self, "first_tick", int(self.first_tick))
        object.__setattr__(self, "last_tick", int(self.last_tick))
        object.__setattr__(self, "observation_count", int(self.observation_count))
        object.__setattr__(self, "prediction_observation_count", int(self.prediction_observation_count))
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "evidence_digests", evidence_digests)
        object.__setattr__(self, "window_digest", str(self.window_digest))
        object.__setattr__(self, "consumed_scheduler_revision", int(self.consumed_scheduler_revision))
        object.__setattr__(self, "provenance_digest", str(self.provenance_digest))
        if self.provenance_digest != _content_digest(self._payload_without_digest()):
            raise ValueError("compacted structural evidence provenance digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_EVIDENCE_COMPACTED_WINDOW_FORMAT,
            "window_id": self.window_id,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "task_slice_ids": list(self.task_slice_ids),
            "partition_counts": {partition: count for partition, count in self.partition_counts},
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "observation_count": self.observation_count,
            "prediction_observation_count": self.prediction_observation_count,
            "evidence_ids": list(self.evidence_ids),
            "evidence_digests": {evidence_id: digest for evidence_id, digest in self.evidence_digests},
            "window_digest": self.window_digest,
            "consumed_scheduler_revision": self.consumed_scheduler_revision,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "provenance_digest": self.provenance_digest}

    @classmethod
    def from_summary(
        cls,
        summary: StructuralEvidenceWindowSummary,
        *,
        evidence_digests: Sequence[tuple[str, str]],
        consumed_scheduler_revision: int,
    ) -> StructuralEvidenceCompactedWindow:
        payload = {
            "format": STRUCTURAL_EVIDENCE_COMPACTED_WINDOW_FORMAT,
            "window_id": summary.window_id,
            "network_id": summary.network_id,
            "region_id": summary.region_id,
            "task_slice_ids": list(summary.task_slice_ids),
            "partition_counts": {partition: count for partition, count in summary.partition_counts},
            "first_tick": summary.first_tick,
            "last_tick": summary.last_tick,
            "observation_count": summary.observation_count,
            "prediction_observation_count": summary.prediction_observation_count,
            "evidence_ids": list(summary.evidence_ids),
            "evidence_digests": {evidence_id: digest for evidence_id, digest in evidence_digests},
            "window_digest": summary.window_digest,
            "consumed_scheduler_revision": int(consumed_scheduler_revision),
        }
        return cls(
            window_id=summary.window_id,
            network_id=summary.network_id,
            region_id=summary.region_id,
            task_slice_ids=summary.task_slice_ids,
            partition_counts=summary.partition_counts,
            first_tick=summary.first_tick,
            last_tick=summary.last_tick,
            observation_count=summary.observation_count,
            prediction_observation_count=summary.prediction_observation_count,
            evidence_ids=summary.evidence_ids,
            evidence_digests=tuple((str(key), str(value)) for key, value in evidence_digests),
            window_digest=summary.window_digest,
            consumed_scheduler_revision=int(consumed_scheduler_revision),
            provenance_digest=_content_digest(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralEvidenceCompactedWindow:
        if payload.get("format") != STRUCTURAL_EVIDENCE_COMPACTED_WINDOW_FORMAT:
            raise ValueError("unsupported compacted structural evidence format")
        expected_digest = _content_digest(
            {key: value for key, value in payload.items() if key != "provenance_digest"}
        )
        if str(payload.get("provenance_digest")) != expected_digest:
            raise ValueError("compacted structural evidence provenance digest mismatch")
        raw_evidence_digests = payload.get("evidence_digests", {})
        if not isinstance(raw_evidence_digests, Mapping):
            raise ValueError("compacted structural evidence digests must be a mapping")
        return cls(
            window_id=str(payload["window_id"]),
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            task_slice_ids=tuple(str(item) for item in payload.get("task_slice_ids", ())),
            partition_counts=tuple(
                (str(partition), int(count))
                for partition, count in dict(payload.get("partition_counts", {})).items()
            ),
            first_tick=int(payload["first_tick"]),
            last_tick=int(payload["last_tick"]),
            observation_count=int(payload["observation_count"]),
            prediction_observation_count=int(payload.get("prediction_observation_count", 0)),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
            evidence_digests=tuple(
                (str(evidence_id), str(raw_evidence_digests[evidence_id]))
                for evidence_id in payload.get("evidence_ids", ())
            ),
            window_digest=str(payload["window_digest"]),
            consumed_scheduler_revision=int(payload.get("consumed_scheduler_revision", 0)),
            provenance_digest=str(payload["provenance_digest"]),
        )


@dataclass(frozen=True)
class StructuralEvidencePressureSnapshot:
    """Aggregate pressure facts retained for already-consumed windows."""

    network_id: str
    region_id: str
    window_digests: tuple[str, ...]
    first_tick: int
    last_tick: int
    train_task_slice_ids: tuple[str, ...]
    holdout_task_slice_ids: tuple[str, ...]
    retention_task_slice_ids: tuple[str, ...]
    train_window_count: int
    holdout_window_count: int
    retention_window_count: int
    prediction_observation_count: int
    train_prediction_error_sum: float
    train_resource_state_sum: float
    holdout_transfer_sum: float
    evidence_ids: tuple[str, ...]
    snapshot_digest: str

    def __post_init__(self) -> None:
        for name in ("network_id", "region_id", "snapshot_digest"):
            if not str(getattr(self, name)):
                raise ValueError(f"structural evidence pressure snapshot {name} must not be empty")
        window_digests = tuple(str(item) for item in self.window_digests)
        if not window_digests or len(set(window_digests)) != len(window_digests):
            raise ValueError("structural evidence pressure snapshot windows must be unique")
        if int(self.first_tick) <= 0 or int(self.last_tick) < int(self.first_tick):
            raise ValueError("structural evidence pressure snapshot ticks are invalid")
        for name in (
            "train_window_count",
            "holdout_window_count",
            "retention_window_count",
            "prediction_observation_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError("structural evidence pressure snapshot counters cannot be negative")
        if int(self.train_window_count + self.holdout_window_count + self.retention_window_count) != len(
            window_digests
        ):
            raise ValueError("structural evidence pressure snapshot window counts do not match digests")
        for name, count in (
            ("train_prediction_error_sum", self.train_window_count),
            ("train_resource_state_sum", self.train_window_count),
            ("holdout_transfer_sum", self.holdout_window_count),
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= float(count):
                raise ValueError(f"structural evidence pressure snapshot {name} is out of range")
        task_fields = (
            "train_task_slice_ids",
            "holdout_task_slice_ids",
            "retention_task_slice_ids",
        )
        normalized_tasks: dict[str, tuple[str, ...]] = {}
        for name in task_fields:
            values = tuple(str(item) for item in getattr(self, name))
            if len(set(values)) != len(values) or any(not item for item in values):
                raise ValueError(f"structural evidence pressure snapshot {name} is invalid")
            normalized_tasks[name] = values
        evidence_ids = tuple(str(item) for item in self.evidence_ids)
        if not evidence_ids or len(set(evidence_ids)) != len(evidence_ids) or any(
            not item for item in evidence_ids
        ):
            raise ValueError("structural evidence pressure snapshot evidence ids are invalid")
        object.__setattr__(self, "network_id", str(self.network_id))
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "window_digests", window_digests)
        object.__setattr__(self, "first_tick", int(self.first_tick))
        object.__setattr__(self, "last_tick", int(self.last_tick))
        for name in (
            "train_window_count",
            "holdout_window_count",
            "retention_window_count",
            "prediction_observation_count",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in (
            "train_prediction_error_sum",
            "train_resource_state_sum",
            "holdout_transfer_sum",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name, values in normalized_tasks.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "snapshot_digest", str(self.snapshot_digest))
        if self.snapshot_digest != _content_digest(self._payload_without_digest()):
            raise ValueError("structural evidence pressure snapshot digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_EVIDENCE_PRESSURE_SNAPSHOT_FORMAT,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "window_digests": list(self.window_digests),
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "train_task_slice_ids": list(self.train_task_slice_ids),
            "holdout_task_slice_ids": list(self.holdout_task_slice_ids),
            "retention_task_slice_ids": list(self.retention_task_slice_ids),
            "train_window_count": self.train_window_count,
            "holdout_window_count": self.holdout_window_count,
            "retention_window_count": self.retention_window_count,
            "prediction_observation_count": self.prediction_observation_count,
            "train_prediction_error_sum": self.train_prediction_error_sum,
            "train_resource_state_sum": self.train_resource_state_sum,
            "holdout_transfer_sum": self.holdout_transfer_sum,
            "evidence_ids": list(self.evidence_ids),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "snapshot_digest": self.snapshot_digest}

    @staticmethod
    def _partition(summary: StructuralEvidenceWindowSummary) -> str:
        partitions = tuple(name for name, count in summary.partition_counts if int(count) > 0)
        if len(partitions) != 1:
            raise ValueError("structural evidence pressure snapshot requires one partition per window")
        return partitions[0]

    @classmethod
    def from_summaries(
        cls,
        summaries: Sequence[StructuralEvidenceWindowSummary],
    ) -> StructuralEvidencePressureSnapshot:
        items = tuple(summaries)
        if not items:
            raise ValueError("structural evidence pressure snapshot requires summaries")
        if any(not isinstance(item, StructuralEvidenceWindowSummary) for item in items):
            raise TypeError("structural evidence pressure snapshot accepts window summaries")
        substrates = {(item.network_id, item.region_id) for item in items}
        if len(substrates) != 1:
            raise ValueError("structural evidence pressure snapshot windows must share one substrate")
        train = tuple(item for item in items if cls._partition(item) == "train")
        holdout = tuple(item for item in items if cls._partition(item) == "holdout")
        retention = tuple(item for item in items if cls._partition(item) == "retention")
        if any(item.mean_prediction_error is None for item in train):
            raise ValueError("structural evidence pressure snapshot train windows require prediction error")
        evidence_ids = tuple(dict.fromkeys(item_id for item in items for item_id in item.evidence_ids))
        payload = {
            "format": STRUCTURAL_EVIDENCE_PRESSURE_SNAPSHOT_FORMAT,
            "network_id": items[0].network_id,
            "region_id": items[0].region_id,
            "window_digests": [item.window_digest for item in items],
            "first_tick": min(item.first_tick for item in items),
            "last_tick": max(item.last_tick for item in items),
            "train_task_slice_ids": list(
                dict.fromkeys(task for item in train for task in item.task_slice_ids if task)
            ),
            "holdout_task_slice_ids": list(
                dict.fromkeys(task for item in holdout for task in item.task_slice_ids if task)
            ),
            "retention_task_slice_ids": list(
                dict.fromkeys(task for item in retention for task in item.task_slice_ids if task)
            ),
            "train_window_count": len(train),
            "holdout_window_count": len(holdout),
            "retention_window_count": len(retention),
            "prediction_observation_count": sum(item.prediction_observation_count for item in train),
            "train_prediction_error_sum": float(
                sum(float(item.mean_prediction_error) for item in train)
            ),
            "train_resource_state_sum": float(
                sum(1.0 - float(item.mean_resource_pressure) for item in train)
            ),
            "holdout_transfer_sum": float(
                sum(float(item.mean_holdout_transfer) for item in holdout)
            ),
            "evidence_ids": list(evidence_ids),
        }
        return cls(
            network_id=items[0].network_id,
            region_id=items[0].region_id,
            window_digests=tuple(payload["window_digests"]),
            first_tick=int(payload["first_tick"]),
            last_tick=int(payload["last_tick"]),
            train_task_slice_ids=tuple(payload["train_task_slice_ids"]),
            holdout_task_slice_ids=tuple(payload["holdout_task_slice_ids"]),
            retention_task_slice_ids=tuple(payload["retention_task_slice_ids"]),
            train_window_count=len(train),
            holdout_window_count=len(holdout),
            retention_window_count=len(retention),
            prediction_observation_count=int(payload["prediction_observation_count"]),
            train_prediction_error_sum=float(payload["train_prediction_error_sum"]),
            train_resource_state_sum=float(payload["train_resource_state_sum"]),
            holdout_transfer_sum=float(payload["holdout_transfer_sum"]),
            evidence_ids=evidence_ids,
            snapshot_digest=_content_digest(payload),
        )

    def merge(self, other: StructuralEvidencePressureSnapshot) -> StructuralEvidencePressureSnapshot:
        if (self.network_id, self.region_id) != (other.network_id, other.region_id):
            raise ValueError("structural evidence pressure snapshots must share one substrate")
        if set(self.window_digests) & set(other.window_digests):
            raise ValueError("structural evidence pressure snapshots cannot reuse windows")
        def _merge_ids(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys((*first, *second)))
        payload = {
            "format": STRUCTURAL_EVIDENCE_PRESSURE_SNAPSHOT_FORMAT,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "window_digests": list(_merge_ids(self.window_digests, other.window_digests)),
            "first_tick": min(self.first_tick, other.first_tick),
            "last_tick": max(self.last_tick, other.last_tick),
            "train_task_slice_ids": list(_merge_ids(self.train_task_slice_ids, other.train_task_slice_ids)),
            "holdout_task_slice_ids": list(_merge_ids(self.holdout_task_slice_ids, other.holdout_task_slice_ids)),
            "retention_task_slice_ids": list(_merge_ids(self.retention_task_slice_ids, other.retention_task_slice_ids)),
            "train_window_count": self.train_window_count + other.train_window_count,
            "holdout_window_count": self.holdout_window_count + other.holdout_window_count,
            "retention_window_count": self.retention_window_count + other.retention_window_count,
            "prediction_observation_count": self.prediction_observation_count + other.prediction_observation_count,
            "train_prediction_error_sum": self.train_prediction_error_sum + other.train_prediction_error_sum,
            "train_resource_state_sum": self.train_resource_state_sum + other.train_resource_state_sum,
            "holdout_transfer_sum": self.holdout_transfer_sum + other.holdout_transfer_sum,
            "evidence_ids": list(_merge_ids(self.evidence_ids, other.evidence_ids)),
        }
        return StructuralEvidencePressureSnapshot(
            network_id=self.network_id,
            region_id=self.region_id,
            window_digests=tuple(payload["window_digests"]),
            first_tick=int(payload["first_tick"]),
            last_tick=int(payload["last_tick"]),
            train_task_slice_ids=tuple(payload["train_task_slice_ids"]),
            holdout_task_slice_ids=tuple(payload["holdout_task_slice_ids"]),
            retention_task_slice_ids=tuple(payload["retention_task_slice_ids"]),
            train_window_count=int(payload["train_window_count"]),
            holdout_window_count=int(payload["holdout_window_count"]),
            retention_window_count=int(payload["retention_window_count"]),
            prediction_observation_count=int(payload["prediction_observation_count"]),
            train_prediction_error_sum=float(payload["train_prediction_error_sum"]),
            train_resource_state_sum=float(payload["train_resource_state_sum"]),
            holdout_transfer_sum=float(payload["holdout_transfer_sum"]),
            evidence_ids=tuple(payload["evidence_ids"]),
            snapshot_digest=_content_digest(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralEvidencePressureSnapshot:
        if payload.get("format") != STRUCTURAL_EVIDENCE_PRESSURE_SNAPSHOT_FORMAT:
            raise ValueError("unsupported structural evidence pressure snapshot format")
        expected_digest = _content_digest(
            {key: value for key, value in payload.items() if key != "snapshot_digest"}
        )
        if str(payload.get("snapshot_digest")) != expected_digest:
            raise ValueError("structural evidence pressure snapshot digest mismatch")
        return cls(
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            window_digests=tuple(str(item) for item in payload.get("window_digests", ())),
            first_tick=int(payload["first_tick"]),
            last_tick=int(payload["last_tick"]),
            train_task_slice_ids=tuple(str(item) for item in payload.get("train_task_slice_ids", ())),
            holdout_task_slice_ids=tuple(str(item) for item in payload.get("holdout_task_slice_ids", ())),
            retention_task_slice_ids=tuple(str(item) for item in payload.get("retention_task_slice_ids", ())),
            train_window_count=int(payload.get("train_window_count", 0)),
            holdout_window_count=int(payload.get("holdout_window_count", 0)),
            retention_window_count=int(payload.get("retention_window_count", 0)),
            prediction_observation_count=int(payload.get("prediction_observation_count", 0)),
            train_prediction_error_sum=float(payload.get("train_prediction_error_sum", 0.0)),
            train_resource_state_sum=float(payload.get("train_resource_state_sum", 0.0)),
            holdout_transfer_sum=float(payload.get("holdout_transfer_sum", 0.0)),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
            snapshot_digest=str(payload["snapshot_digest"]),
        )


@dataclass(frozen=True)
class StructuralEvidenceConsumptionAudit:
    """Content-addressed view of cross-round evidence consumption."""

    ledger_digest: str
    scheduler_revision: int
    evaluated_window_digests: tuple[str, ...]
    consumed_window_digests: tuple[str, ...]
    unconsumed_window_digests: tuple[str, ...]
    orphaned_evaluated_window_digests: tuple[str, ...]
    retained_window_digests: tuple[str, ...]
    compacted_window_digests: tuple[str, ...]
    stream_status: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    audit_digest: str

    def __post_init__(self) -> None:
        if not str(self.ledger_digest) or not str(self.audit_digest):
            raise ValueError("structural evidence audit digests must not be empty")
        if int(self.scheduler_revision) < 0:
            raise ValueError("structural evidence audit scheduler revision cannot be negative")
        fields = (
            "evaluated_window_digests",
            "consumed_window_digests",
            "unconsumed_window_digests",
            "orphaned_evaluated_window_digests",
            "retained_window_digests",
            "compacted_window_digests",
        )
        normalized: dict[str, tuple[str, ...]] = {}
        for name in fields:
            values = tuple(str(item) for item in getattr(self, name))
            if any(not item for item in values) or len(set(values)) != len(values):
                raise ValueError(f"structural evidence audit {name} must be unique and non-empty")
            normalized[name] = values
        stream_status = tuple(
            (
                str(stream),
                tuple(str(item) for item in consumed),
                tuple(str(item) for item in unconsumed),
            )
            for stream, consumed, unconsumed in self.stream_status
        )
        if any(not stream for stream, _, _ in stream_status) or len(
            {stream for stream, _, _ in stream_status}
        ) != len(stream_status):
            raise ValueError("structural evidence audit streams must be unique and non-empty")
        for stream, consumed, unconsumed in stream_status:
            if any(not item for item in (*consumed, *unconsumed)):
                raise ValueError(f"structural evidence audit stream {stream} has an empty digest")
            if set(consumed) & set(unconsumed):
                raise ValueError(f"structural evidence audit stream {stream} overlaps consumption state")
        object.__setattr__(self, "ledger_digest", str(self.ledger_digest))
        object.__setattr__(self, "scheduler_revision", int(self.scheduler_revision))
        for name, values in normalized.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "stream_status", tuple(sorted(stream_status)))
        object.__setattr__(self, "audit_digest", str(self.audit_digest))
        if self.audit_digest != _content_digest(self._payload_without_digest()):
            raise ValueError("structural evidence consumption audit digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_EVIDENCE_CONSUMPTION_AUDIT_FORMAT,
            "ledger_digest": self.ledger_digest,
            "scheduler_revision": self.scheduler_revision,
            "evaluated_window_digests": list(self.evaluated_window_digests),
            "consumed_window_digests": list(self.consumed_window_digests),
            "unconsumed_window_digests": list(self.unconsumed_window_digests),
            "orphaned_evaluated_window_digests": list(self.orphaned_evaluated_window_digests),
            "retained_window_digests": list(self.retained_window_digests),
            "compacted_window_digests": list(self.compacted_window_digests),
            "stream_status": [
                {
                    "stream": stream,
                    "consumed": list(consumed),
                    "unconsumed": list(unconsumed),
                }
                for stream, consumed, unconsumed in self.stream_status
            ],
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "audit_digest": self.audit_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralEvidenceConsumptionAudit:
        if payload.get("format") != STRUCTURAL_EVIDENCE_CONSUMPTION_AUDIT_FORMAT:
            raise ValueError("unsupported structural evidence consumption audit format")
        expected_digest = _content_digest(
            {key: value for key, value in payload.items() if key != "audit_digest"}
        )
        if str(payload.get("audit_digest")) != expected_digest:
            raise ValueError("structural evidence consumption audit digest mismatch")
        raw_stream_status = payload.get("stream_status", ())
        if not isinstance(raw_stream_status, (tuple, list)):
            raise ValueError("structural evidence audit stream status must be a sequence")
        stream_status = []
        for item in raw_stream_status:
            if not isinstance(item, Mapping):
                raise ValueError("structural evidence audit stream entry must be a mapping")
            stream_status.append(
                (
                    str(item["stream"]),
                    tuple(str(value) for value in item.get("consumed", ())),
                    tuple(str(value) for value in item.get("unconsumed", ())),
                )
            )
        return cls(
            ledger_digest=str(payload["ledger_digest"]),
            scheduler_revision=int(payload.get("scheduler_revision", 0)),
            evaluated_window_digests=tuple(str(item) for item in payload.get("evaluated_window_digests", ())),
            consumed_window_digests=tuple(str(item) for item in payload.get("consumed_window_digests", ())),
            unconsumed_window_digests=tuple(str(item) for item in payload.get("unconsumed_window_digests", ())),
            orphaned_evaluated_window_digests=tuple(
                str(item) for item in payload.get("orphaned_evaluated_window_digests", ())
            ),
            retained_window_digests=tuple(str(item) for item in payload.get("retained_window_digests", ())),
            compacted_window_digests=tuple(str(item) for item in payload.get("compacted_window_digests", ())),
            stream_status=tuple(stream_status),
            audit_digest=str(payload["audit_digest"]),
        )


@dataclass(frozen=True)
class StructuralEvidenceCompactionResult:
    """Audit record for one atomic consumed-window compaction."""

    status: str
    source_ledger_digest: str
    target_ledger_digest: str
    scheduler_revision: int
    compacted_window_digests: tuple[str, ...]
    retained_window_digests: tuple[str, ...]
    compacted_evidence_count: int
    result_digest: str

    def __post_init__(self) -> None:
        if self.status not in {"compacted", "nothing_to_compact"}:
            raise ValueError("unsupported structural evidence compaction status")
        if not str(self.source_ledger_digest) or not str(self.target_ledger_digest):
            raise ValueError("structural evidence compaction ledger digests must not be empty")
        if int(self.scheduler_revision) < 0 or int(self.compacted_evidence_count) < 0:
            raise ValueError("structural evidence compaction counters are invalid")
        compacted = tuple(str(item) for item in self.compacted_window_digests)
        retained = tuple(str(item) for item in self.retained_window_digests)
        if any(not item for item in (*compacted, *retained)):
            raise ValueError("structural evidence compaction window digests must not be empty")
        if len(set(compacted)) != len(compacted) or len(set(retained)) != len(retained):
            raise ValueError("structural evidence compaction window digests must be unique")
        object.__setattr__(self, "source_ledger_digest", str(self.source_ledger_digest))
        object.__setattr__(self, "target_ledger_digest", str(self.target_ledger_digest))
        object.__setattr__(self, "scheduler_revision", int(self.scheduler_revision))
        object.__setattr__(self, "compacted_window_digests", compacted)
        object.__setattr__(self, "retained_window_digests", retained)
        object.__setattr__(self, "compacted_evidence_count", int(self.compacted_evidence_count))
        object.__setattr__(self, "result_digest", str(self.result_digest))
        if not self.result_digest:
            raise ValueError("structural evidence compaction result digest must not be empty")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_EVIDENCE_COMPACTION_RESULT_FORMAT,
            "status": self.status,
            "source_ledger_digest": self.source_ledger_digest,
            "target_ledger_digest": self.target_ledger_digest,
            "scheduler_revision": self.scheduler_revision,
            "compacted_window_digests": list(self.compacted_window_digests),
            "retained_window_digests": list(self.retained_window_digests),
            "compacted_evidence_count": self.compacted_evidence_count,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "result_digest": self.result_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralEvidenceCompactionResult:
        if payload.get("format") != STRUCTURAL_EVIDENCE_COMPACTION_RESULT_FORMAT:
            raise ValueError("unsupported structural evidence compaction result format")
        expected_digest = _content_digest(
            {key: value for key, value in payload.items() if key != "result_digest"}
        )
        if str(payload.get("result_digest")) != expected_digest:
            raise ValueError("structural evidence compaction result digest mismatch")
        return cls(
            status=str(payload["status"]),
            source_ledger_digest=str(payload["source_ledger_digest"]),
            target_ledger_digest=str(payload["target_ledger_digest"]),
            scheduler_revision=int(payload.get("scheduler_revision", 0)),
            compacted_window_digests=tuple(str(item) for item in payload.get("compacted_window_digests", ())),
            retained_window_digests=tuple(str(item) for item in payload.get("retained_window_digests", ())),
            compacted_evidence_count=int(payload.get("compacted_evidence_count", 0)),
            result_digest=str(payload["result_digest"]),
        )


@dataclass
class StructuralEvidenceWindow:
    """A bounded, monotonic, replayable window for one substrate region."""

    window_id: str
    network_id: str
    region_id: str
    capacity: int
    task_slice_id: str = ""
    partition: str = "runtime"
    sealed: bool = False
    _observations: list[StructuralRuntimeObservation] = field(
        default_factory=list,
        repr=False,
    )

    def __post_init__(self) -> None:
        for name in ("window_id", "network_id", "region_id"):
            if not str(getattr(self, name)):
                raise ValueError(f"structural evidence {name} must not be empty")
        if int(self.capacity) <= 0:
            raise ValueError("structural evidence window capacity must be positive")
        self.window_id = str(self.window_id)
        self.network_id = str(self.network_id)
        self.region_id = str(self.region_id)
        self.capacity = int(self.capacity)
        self.task_slice_id = str(self.task_slice_id)
        self.partition = str(self.partition)
        if self.partition not in STRUCTURAL_EVIDENCE_PARTITIONS:
            raise ValueError("unsupported structural evidence window partition")
        self.sealed = bool(self.sealed)
        self._validate_observations()
        if self.sealed and not self._observations:
            raise ValueError("sealed structural evidence window cannot be empty")

    @property
    def observations(self) -> tuple[StructuralRuntimeObservation, ...]:
        return tuple(self._observations)

    @property
    def full(self) -> bool:
        return len(self._observations) >= self.capacity

    def _validate_observations(self) -> None:
        previous_tick = 0
        seen_ids: set[str] = set()
        for observation in self._observations:
            if not isinstance(observation, StructuralRuntimeObservation):
                raise TypeError("structural evidence window accepts runtime observations")
            if observation.network_id != self.network_id or observation.region_id != self.region_id:
                raise ValueError("structural evidence observation substrate mismatch")
            if (
                observation.task_slice_id != self.task_slice_id
                or observation.partition != self.partition
            ):
                raise ValueError("structural evidence observation context mismatch")
            if observation.tick <= previous_tick:
                raise ValueError("structural evidence observation ticks must increase")
            if observation.evidence_id in seen_ids:
                raise ValueError("structural evidence observation ids must be unique")
            previous_tick = observation.tick
            seen_ids.add(observation.evidence_id)
        if len(self._observations) > self.capacity:
            raise ValueError("structural evidence window exceeds capacity")

    def append(self, observation: StructuralRuntimeObservation) -> bool:
        """Append one observation; exact replay is idempotent."""

        if not isinstance(observation, StructuralRuntimeObservation):
            raise TypeError("structural evidence window accepts runtime observations")
        if observation.network_id != self.network_id or observation.region_id != self.region_id:
            raise ValueError("structural evidence observation substrate mismatch")
        if (
            observation.task_slice_id != self.task_slice_id
            or observation.partition != self.partition
        ):
            raise ValueError("structural evidence observation context mismatch")
        for existing in self._observations:
            if existing.evidence_id == observation.evidence_id:
                if existing.to_payload() == observation.to_payload():
                    return False
                raise ValueError("structural evidence id was reused with different content")
        if self.sealed:
            raise ValueError("structural evidence window is sealed")
        if self.full:
            raise OverflowError("structural evidence window capacity exhausted")
        if self._observations and observation.tick <= self._observations[-1].tick:
            raise ValueError("structural evidence observation ticks must increase")
        self._observations.append(observation)
        return True

    def seal(self) -> StructuralEvidenceWindowSummary:
        if not self._observations:
            raise ValueError("cannot seal an empty structural evidence window")
        if self.sealed:
            return self.summary
        self.sealed = True
        return self.summary

    @property
    def summary(self) -> StructuralEvidenceWindowSummary:
        if not self._observations:
            raise ValueError("empty structural evidence window has no summary")
        observations = self._observations
        prediction_errors = [
            float(item.prediction_error)
            for item in observations
            if item.prediction_error is not None
        ]
        task_slice_ids = tuple(
            dict.fromkeys(item.task_slice_id for item in observations if item.task_slice_id)
        )
        partition_counts = tuple(
            (partition, sum(item.partition == partition for item in observations))
            for partition in sorted({item.partition for item in observations})
        )
        payload = {
            "format": STRUCTURAL_EVIDENCE_WINDOW_CHECKPOINT_FORMAT,
            "window_id": self.window_id,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "task_slice_ids": list(task_slice_ids),
            "partition_counts": {
                partition: count for partition, count in partition_counts
            },
            "first_tick": observations[0].tick,
            "last_tick": observations[-1].tick,
            "observation_count": len(observations),
            "prediction_observation_count": len(prediction_errors),
            "mean_usage": _mean([item.usage for item in observations]),
            "mean_resource_pressure": _mean([item.resource_pressure for item in observations]),
            "mean_prediction_error": _mean(prediction_errors),
            "mean_learning_gain": _mean([item.learning_gain for item in observations]),
            "mean_holdout_transfer": _mean([item.holdout_transfer for item in observations]),
            "evidence_ids": [item.evidence_id for item in observations],
        }
        digest = _content_digest(payload)
        return StructuralEvidenceWindowSummary(
            window_id=self.window_id,
            network_id=self.network_id,
            region_id=self.region_id,
            task_slice_ids=task_slice_ids,
            partition_counts=partition_counts,
            first_tick=observations[0].tick,
            last_tick=observations[-1].tick,
            observation_count=len(observations),
            prediction_observation_count=len(prediction_errors),
            mean_usage=float(payload["mean_usage"]),
            mean_resource_pressure=float(payload["mean_resource_pressure"]),
            mean_prediction_error=(
                None
                if payload["mean_prediction_error"] is None
                else float(payload["mean_prediction_error"])
            ),
            mean_learning_gain=float(payload["mean_learning_gain"]),
            mean_holdout_transfer=float(payload["mean_holdout_transfer"]),
            evidence_ids=tuple(item.evidence_id for item in observations),
            window_digest=digest,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_EVIDENCE_WINDOW_CHECKPOINT_FORMAT,
            "window_id": self.window_id,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "task_slice_id": self.task_slice_id,
            "partition": self.partition,
            "capacity": self.capacity,
            "sealed": self.sealed,
            "observations": [item.to_payload() for item in self._observations],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralEvidenceWindow:
        if payload.get("format") != STRUCTURAL_EVIDENCE_WINDOW_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural evidence window format")
        observations_payload = payload.get("observations", ())
        if not isinstance(observations_payload, (tuple, list)):
            raise ValueError("structural evidence observations must be a sequence")
        observations = [
            StructuralRuntimeObservation.from_payload(item)
            for item in observations_payload
            if isinstance(item, Mapping)
        ]
        if len(observations) != len(observations_payload):
            raise ValueError("structural evidence observation entry must be a mapping")
        return cls(
            window_id=str(payload["window_id"]),
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            capacity=int(payload["capacity"]),
            task_slice_id=str(payload.get("task_slice_id", "")),
            partition=str(payload.get("partition", "runtime")),
            sealed=bool(payload.get("sealed", False)),
            _observations=observations,
        )


@dataclass(frozen=True)
class StructuralEvidenceAppendResult:
    evidence_id: str
    status: str
    window_id: str
    sealed_window_digest: str | None = None

    def __post_init__(self) -> None:
        if not str(self.evidence_id) or not str(self.window_id):
            raise ValueError("structural evidence append identifiers must not be empty")
        if self.status not in {"accepted", "duplicate", "window_sealed"}:
            raise ValueError("unsupported structural evidence append status")
        if self.status == "window_sealed" and not self.sealed_window_digest:
            raise ValueError("sealed evidence result must include a digest")


@dataclass
class StructuralEvidenceLedger:
    """Bounded ledger that turns runtime ticks into long-horizon summaries."""

    window_capacity: int = 8
    max_sealed_windows: int = 128
    max_evidence_index: int = 4096
    max_compacted_windows: int = DEFAULT_MAX_COMPACTED_WINDOWS
    _open_windows: dict[str, StructuralEvidenceWindow] = field(
        default_factory=dict,
        repr=False,
    )
    _sealed_windows: list[StructuralEvidenceWindowSummary] = field(
        default_factory=list,
        repr=False,
    )
    _evidence_index: dict[str, str] = field(default_factory=dict, repr=False)
    _compacted_windows: list[StructuralEvidenceCompactedWindow] = field(
        default_factory=list,
        repr=False,
    )
    _compacted_evidence_index: dict[str, str] = field(default_factory=dict, repr=False)
    _pressure_snapshots: list[StructuralEvidencePressureSnapshot] = field(
        default_factory=list,
        repr=False,
    )
    _next_ordinals: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if int(self.window_capacity) <= 0:
            raise ValueError("structural evidence window_capacity must be positive")
        if int(self.max_sealed_windows) <= 0:
            raise ValueError("structural evidence max_sealed_windows must be positive")
        if int(self.max_evidence_index) <= 0:
            raise ValueError("structural evidence max_evidence_index must be positive")
        if int(self.max_compacted_windows) <= 0:
            raise ValueError("structural evidence max_compacted_windows must be positive")
        self.window_capacity = int(self.window_capacity)
        self.max_sealed_windows = int(self.max_sealed_windows)
        self.max_evidence_index = int(self.max_evidence_index)
        self.max_compacted_windows = int(self.max_compacted_windows)
        self._compacted_windows = list(self._compacted_windows)
        self._compacted_evidence_index = {
            str(key): str(value) for key, value in self._compacted_evidence_index.items()
        }
        self._pressure_snapshots = list(self._pressure_snapshots)
        self._validate_state()

    @staticmethod
    def _substrate_key(
        network_id: str,
        region_id: str,
        task_slice_id: str = "",
        partition: str = "runtime",
    ) -> str:
        return f"{network_id}\x00{region_id}\x00{task_slice_id}\x00{partition}"

    def _window_for(self, observation: StructuralRuntimeObservation) -> StructuralEvidenceWindow:
        key = self._substrate_key(
            observation.network_id,
            observation.region_id,
            observation.task_slice_id,
            observation.partition,
        )
        window = self._open_windows.get(key)
        if window is not None:
            return window
        ordinal = int(self._next_ordinals.get(key, 0)) + 1
        self._next_ordinals[key] = ordinal
        window = StructuralEvidenceWindow(
            window_id=(f"window:{observation.network_id}:{observation.region_id}:{ordinal}"),
            network_id=observation.network_id,
            region_id=observation.region_id,
            capacity=self.window_capacity,
            task_slice_id=observation.task_slice_id,
            partition=observation.partition,
        )
        self._open_windows[key] = window
        return window

    def append(self, observation: StructuralRuntimeObservation) -> StructuralEvidenceAppendResult:
        if not isinstance(observation, StructuralRuntimeObservation):
            raise TypeError("structural evidence ledger accepts runtime observations")
        observation_digest = _content_digest(observation.to_payload())
        existing_digest = self._evidence_index.get(observation.evidence_id)
        compacted_window_id = None
        if existing_digest is None:
            existing_digest = self._compacted_evidence_index.get(observation.evidence_id)
            if existing_digest is not None:
                compacted_window_id = next(
                    (
                        item.window_id
                        for item in self._compacted_windows
                        if observation.evidence_id in item.evidence_ids
                    ),
                    "compacted-history",
                )
        if existing_digest is not None:
            if existing_digest == observation_digest:
                key = self._substrate_key(
                    observation.network_id,
                    observation.region_id,
                    observation.task_slice_id,
                    observation.partition,
                )
                window = self._open_windows.get(key)
                window_id = window.window_id if window is not None else "sealed-history"
                if compacted_window_id is not None:
                    window_id = f"compacted:{compacted_window_id}"
                return StructuralEvidenceAppendResult(
                    evidence_id=observation.evidence_id,
                    status="duplicate",
                    window_id=window_id,
                )
            raise ValueError("structural evidence id was reused with different content")
        if len(self._evidence_index) >= self.max_evidence_index:
            raise OverflowError("structural evidence index capacity exhausted")
        window = self._window_for(observation)
        if window.full and len(self._sealed_windows) >= self.max_sealed_windows:
            raise OverflowError("structural evidence sealed-window capacity exhausted")
        window.append(observation)
        self._evidence_index[observation.evidence_id] = observation_digest
        if not window.full:
            return StructuralEvidenceAppendResult(
                evidence_id=observation.evidence_id,
                status="accepted",
                window_id=window.window_id,
            )
        summary = window.seal()
        key = self._substrate_key(
            observation.network_id,
            observation.region_id,
            observation.task_slice_id,
            observation.partition,
        )
        self._sealed_windows.append(summary)
        del self._open_windows[key]
        return StructuralEvidenceAppendResult(
            evidence_id=observation.evidence_id,
            status="window_sealed",
            window_id=summary.window_id,
            sealed_window_digest=summary.window_digest,
        )

    def seal(
        self,
        network_id: str,
        region_id: str,
        *,
        task_slice_id: str = "",
        partition: str = "runtime",
    ) -> StructuralEvidenceWindowSummary | None:
        key = self._substrate_key(
            str(network_id),
            str(region_id),
            str(task_slice_id),
            str(partition),
        )
        window = self._open_windows.get(key)
        if window is None:
            return None
        if len(self._sealed_windows) >= self.max_sealed_windows:
            raise OverflowError("structural evidence sealed-window capacity exhausted")
        summary = window.seal()
        self._sealed_windows.append(summary)
        del self._open_windows[key]
        return summary

    def seal_all(self) -> tuple[StructuralEvidenceWindowSummary, ...]:
        summaries = []
        for key in tuple(self._open_windows):
            window = self._open_windows[key]
            if len(self._sealed_windows) + len(summaries) >= self.max_sealed_windows:
                raise OverflowError("structural evidence sealed-window capacity exhausted")
            summary = window.seal()
            summaries.append(summary)
            del self._open_windows[key]
        self._sealed_windows.extend(summaries)
        return tuple(summaries)

    @property
    def open_windows(self) -> tuple[StructuralEvidenceWindow, ...]:
        return tuple(self._open_windows.values())

    @property
    def sealed_summaries(self) -> tuple[StructuralEvidenceWindowSummary, ...]:
        return tuple(self._sealed_windows)

    @property
    def observed_count(self) -> int:
        return len(self._evidence_index) + len(self._compacted_evidence_index)

    @property
    def active_observed_count(self) -> int:
        """Return observations still available to active-window projection."""

        return len(self._evidence_index)

    @property
    def compacted_windows(self) -> tuple[StructuralEvidenceCompactedWindow, ...]:
        """Return bounded provenance records excluded from active projection."""

        return tuple(self._compacted_windows)

    @property
    def pressure_snapshots(self) -> tuple[StructuralEvidencePressureSnapshot, ...]:
        """Return consumed-history aggregates available for explicit projection."""

        return tuple(self._pressure_snapshots)

    @property
    def digest(self) -> str:
        payload = self._payload_without_digest()
        return _content_digest(payload)

    def _payload_without_digest(self) -> dict[str, Any]:
        payload = {
            "format": STRUCTURAL_EVIDENCE_LEDGER_CHECKPOINT_FORMAT,
            "window_capacity": self.window_capacity,
            "max_sealed_windows": self.max_sealed_windows,
            "max_evidence_index": self.max_evidence_index,
            "open_windows": [item.to_payload() for item in self._open_windows.values()],
            "sealed_windows": [item.to_payload() for item in self._sealed_windows],
            "evidence_index": dict(sorted(self._evidence_index.items())),
            "next_ordinals": dict(sorted(self._next_ordinals.items())),
        }
        if self.max_compacted_windows != DEFAULT_MAX_COMPACTED_WINDOWS:
            payload["max_compacted_windows"] = self.max_compacted_windows
        if self._compacted_windows:
            payload["compacted_windows"] = [item.to_payload() for item in self._compacted_windows]
            payload["compacted_evidence_index"] = dict(
                sorted(self._compacted_evidence_index.items())
            )
            payload["pressure_snapshots"] = [item.to_payload() for item in self._pressure_snapshots]
        return payload

    def to_payload(self) -> dict[str, Any]:
        payload = self._payload_without_digest()
        return {**payload, "ledger_digest": _content_digest(payload)}

    def _validate_state(self) -> None:
        if len(self._sealed_windows) > self.max_sealed_windows:
            raise ValueError("structural evidence sealed windows exceed capacity")
        if len(self._evidence_index) > self.max_evidence_index:
            raise ValueError("structural evidence index exceeds capacity")
        if len(self._compacted_windows) > self.max_compacted_windows:
            raise ValueError("structural evidence compacted windows exceed capacity")
        observed_ids: set[str] = set()
        for window in self._open_windows.values():
            if window.sealed:
                raise ValueError("open structural evidence window cannot be sealed")
            for observation in window.observations:
                if observation.evidence_id in observed_ids:
                    raise ValueError("structural evidence id appears more than once")
                observed_ids.add(observation.evidence_id)
        for summary in self._sealed_windows:
            for evidence_id in summary.evidence_ids:
                if evidence_id in observed_ids:
                    raise ValueError("structural evidence id appears in open and sealed history")
                observed_ids.add(evidence_id)
        if observed_ids != set(self._evidence_index):
            raise ValueError("structural evidence index does not match stored observations")
        window_digests = {summary.window_digest for summary in self._sealed_windows}
        compacted_ids: set[str] = set()
        compacted_index: dict[str, str] = {}
        for record in self._compacted_windows:
            if record.window_digest in window_digests:
                raise ValueError("structural evidence window is both active and compacted")
            window_digests.add(record.window_digest)
            for evidence_id, evidence_digest in record.evidence_digests:
                if evidence_id in observed_ids or evidence_id in compacted_ids:
                    raise ValueError("structural evidence id appears more than once")
                compacted_ids.add(evidence_id)
                compacted_index[evidence_id] = evidence_digest
        if compacted_index != self._compacted_evidence_index:
            raise ValueError("compacted structural evidence index does not match provenance")
        snapshot_digests: set[str] = set()
        snapshot_windows: set[str] = set()
        for snapshot in self._pressure_snapshots:
            if snapshot.snapshot_digest in snapshot_digests:
                raise ValueError("structural evidence pressure snapshot digest appears more than once")
            snapshot_digests.add(snapshot.snapshot_digest)
            if snapshot.network_id == "" or snapshot.region_id == "":
                raise ValueError("structural evidence pressure snapshot substrate is empty")
            if snapshot_windows & set(snapshot.window_digests):
                raise ValueError("structural evidence pressure snapshot reuses a window")
            snapshot_windows.update(snapshot.window_digests)
        if snapshot_windows != {item.window_digest for item in self._compacted_windows}:
            raise ValueError("structural evidence pressure snapshots do not match compacted windows")

    def audit_consumption(
        self,
        *,
        evaluated_window_digests: Sequence[str],
        scheduler_revision: int = 0,
    ) -> StructuralEvidenceConsumptionAudit:
        """Audit current and compacted windows against scheduler consumption."""

        evaluated = tuple(dict.fromkeys(str(item) for item in evaluated_window_digests))
        if any(not item for item in evaluated):
            raise ValueError("structural evidence evaluated digests must not be empty")
        if int(scheduler_revision) < 0:
            raise ValueError("structural evidence audit scheduler revision cannot be negative")
        active = tuple(item.window_digest for item in self._sealed_windows)
        compacted = tuple(item.window_digest for item in self._compacted_windows)
        known = (*active, *compacted)
        known_set = set(known)
        evaluated_set = set(evaluated)
        consumed = tuple(item for item in known if item in evaluated_set)
        unconsumed = tuple(item for item in active if item not in evaluated_set)
        orphaned = tuple(item for item in evaluated if item not in known_set)
        stream_windows: dict[str, list[tuple[str, bool]]] = {}
        for item in self._sealed_windows:
            stream = f"{item.network_id}:{item.region_id}"
            stream_windows.setdefault(stream, []).append(
                (item.window_digest, item.window_digest in evaluated_set)
            )
        for item in self._compacted_windows:
            stream = f"{item.network_id}:{item.region_id}"
            stream_windows.setdefault(stream, []).append(
                (item.window_digest, item.window_digest in evaluated_set)
            )
        stream_status = tuple(
            (
                stream,
                tuple(digest for digest, is_consumed in entries if is_consumed),
                tuple(digest for digest, is_consumed in entries if not is_consumed),
            )
            for stream, entries in sorted(stream_windows.items())
        )
        payload = {
            "format": STRUCTURAL_EVIDENCE_CONSUMPTION_AUDIT_FORMAT,
            "ledger_digest": self.digest,
            "scheduler_revision": int(scheduler_revision),
            "evaluated_window_digests": list(evaluated),
            "consumed_window_digests": list(consumed),
            "unconsumed_window_digests": list(unconsumed),
            "orphaned_evaluated_window_digests": list(orphaned),
            "retained_window_digests": list(active),
            "compacted_window_digests": list(compacted),
            "stream_status": [
                {
                    "stream": stream,
                    "consumed": list(consumed_digests),
                    "unconsumed": list(unconsumed_digests),
                }
                for stream, consumed_digests, unconsumed_digests in stream_status
            ],
        }
        ledger_digest = self.digest
        return StructuralEvidenceConsumptionAudit(
            ledger_digest=ledger_digest,
            scheduler_revision=int(scheduler_revision),
            evaluated_window_digests=evaluated,
            consumed_window_digests=consumed,
            unconsumed_window_digests=unconsumed,
            orphaned_evaluated_window_digests=orphaned,
            retained_window_digests=active,
            compacted_window_digests=compacted,
            stream_status=stream_status,
            audit_digest=_content_digest({**payload, "ledger_digest": ledger_digest}),
        )

    def compact_consumed_windows(
        self,
        *,
        evaluated_window_digests: Sequence[str],
        scheduler_revision: int = 0,
        keep_latest_per_stream: int = 1,
        protected_window_digests: Sequence[str] = (),
    ) -> StructuralEvidenceCompactionResult:
        """Compact only consumed, older windows into bounded provenance records."""

        evaluated = {str(item) for item in evaluated_window_digests}
        protected = {str(item) for item in protected_window_digests}
        if any(not item for item in (*evaluated, *protected)):
            raise ValueError("structural evidence compaction digests must not be empty")
        if int(scheduler_revision) < 0:
            raise ValueError("structural evidence compaction scheduler revision cannot be negative")
        if int(keep_latest_per_stream) < 0:
            raise ValueError("structural evidence compaction keep_latest_per_stream cannot be negative")

        by_stream: dict[str, list[StructuralEvidenceWindowSummary]] = {}
        for summary in self._sealed_windows:
            if summary.window_digest in evaluated and summary.window_digest not in protected:
                stream = f"{summary.network_id}:{summary.region_id}"
                by_stream.setdefault(stream, []).append(summary)
        selected: list[StructuralEvidenceWindowSummary] = []
        for summaries in by_stream.values():
            if keep_latest_per_stream == 0:
                selected.extend(summaries)
            elif len(summaries) > keep_latest_per_stream:
                selected.extend(summaries[:-keep_latest_per_stream])
        selected_digests = {item.window_digest for item in selected}
        source_digest = self.digest
        if not selected:
            retained = tuple(item.window_digest for item in self._sealed_windows)
            payload = {
                "format": STRUCTURAL_EVIDENCE_COMPACTION_RESULT_FORMAT,
                "status": "nothing_to_compact",
                "source_ledger_digest": source_digest,
                "target_ledger_digest": source_digest,
                "scheduler_revision": int(scheduler_revision),
                "compacted_window_digests": [],
                "retained_window_digests": list(retained),
                "compacted_evidence_count": 0,
            }
            return StructuralEvidenceCompactionResult(
                status="nothing_to_compact",
                source_ledger_digest=source_digest,
                target_ledger_digest=source_digest,
                scheduler_revision=int(scheduler_revision),
                compacted_window_digests=(),
                retained_window_digests=retained,
                compacted_evidence_count=0,
                result_digest=_content_digest(payload),
            )
        if len(self._compacted_windows) + len(selected) > self.max_compacted_windows:
            raise OverflowError("structural evidence compacted-window capacity exhausted")
        evidence_digests = dict(self._evidence_index)
        selected_evidence_ids = {
            evidence_id for item in selected for evidence_id in item.evidence_ids
        }
        records = [
            StructuralEvidenceCompactedWindow.from_summary(
                summary,
                evidence_digests=tuple(
                    (evidence_id, evidence_digests[evidence_id])
                    for evidence_id in summary.evidence_ids
                ),
                consumed_scheduler_revision=int(scheduler_revision),
            )
            for summary in selected
        ]
        snapshots_by_stream: dict[str, StructuralEvidencePressureSnapshot] = {
            f"{snapshot.network_id}:{snapshot.region_id}": snapshot
            for snapshot in self._pressure_snapshots
        }
        for stream, summaries in by_stream.items():
            selected_for_stream = tuple(
                item for item in selected if f"{item.network_id}:{item.region_id}" == stream
            )
            if not selected_for_stream:
                continue
            incoming = StructuralEvidencePressureSnapshot.from_summaries(selected_for_stream)
            prior = snapshots_by_stream.get(stream)
            snapshots_by_stream[stream] = incoming if prior is None else prior.merge(incoming)
        original_sealed = self._sealed_windows
        original_index = self._evidence_index
        original_compacted = self._compacted_windows
        original_compacted_index = self._compacted_evidence_index
        original_snapshots = self._pressure_snapshots
        try:
            self._sealed_windows = [
                item for item in self._sealed_windows if item.window_digest not in selected_digests
            ]
            self._compacted_windows = [*self._compacted_windows, *records]
            self._evidence_index = {
                evidence_id: digest
                for evidence_id, digest in self._evidence_index.items()
                if evidence_id not in selected_evidence_ids
            }
            self._compacted_evidence_index = {
                **self._compacted_evidence_index,
                **{
                    evidence_id: digest
                    for record in records
                    for evidence_id, digest in record.evidence_digests
                },
            }
            self._pressure_snapshots = [
                snapshots_by_stream[key] for key in sorted(snapshots_by_stream)
            ]
            self._validate_state()
        except Exception:
            self._sealed_windows = original_sealed
            self._evidence_index = original_index
            self._compacted_windows = original_compacted
            self._compacted_evidence_index = original_compacted_index
            self._pressure_snapshots = original_snapshots
            raise
        target_digest = self.digest
        retained = tuple(item.window_digest for item in self._sealed_windows)
        compacted = tuple(item.window_digest for item in records)
        compacted_evidence_count = sum(item.observation_count for item in records)
        payload = {
            "format": STRUCTURAL_EVIDENCE_COMPACTION_RESULT_FORMAT,
            "status": "compacted",
            "source_ledger_digest": source_digest,
            "target_ledger_digest": target_digest,
            "scheduler_revision": int(scheduler_revision),
            "compacted_window_digests": list(compacted),
            "retained_window_digests": list(retained),
            "compacted_evidence_count": compacted_evidence_count,
        }
        return StructuralEvidenceCompactionResult(
            status="compacted",
            source_ledger_digest=source_digest,
            target_ledger_digest=target_digest,
            scheduler_revision=int(scheduler_revision),
            compacted_window_digests=compacted,
            retained_window_digests=retained,
            compacted_evidence_count=compacted_evidence_count,
            result_digest=_content_digest(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralEvidenceLedger:
        if payload.get("format") != STRUCTURAL_EVIDENCE_LEDGER_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural evidence ledger format")
        expected_digest = _content_digest(
            {key: value for key, value in payload.items() if key != "ledger_digest"}
        )
        if str(payload.get("ledger_digest")) != expected_digest:
            raise ValueError("structural evidence ledger digest mismatch")
        open_payload = payload.get("open_windows", ())
        sealed_payload = payload.get("sealed_windows", ())
        compacted_payload = payload.get("compacted_windows", ())
        evidence_index = payload.get("evidence_index", {})
        compacted_evidence_index = payload.get("compacted_evidence_index", {})
        next_ordinals = payload.get("next_ordinals", {})
        if not isinstance(open_payload, (tuple, list)):
            raise ValueError("structural evidence open_windows must be a sequence")
        if not isinstance(sealed_payload, (tuple, list)):
            raise ValueError("structural evidence sealed_windows must be a sequence")
        if not isinstance(compacted_payload, (tuple, list)):
            raise ValueError("structural evidence compacted_windows must be a sequence")
        if (
            not isinstance(evidence_index, Mapping)
            or not isinstance(compacted_evidence_index, Mapping)
            or not isinstance(next_ordinals, Mapping)
        ):
            raise ValueError("structural evidence ledger indexes must be mappings")
        windows = [
            StructuralEvidenceWindow.from_payload(item)
            for item in open_payload
            if isinstance(item, Mapping)
        ]
        if len(windows) != len(open_payload):
            raise ValueError("structural evidence open window entry must be a mapping")
        open_windows: dict[str, StructuralEvidenceWindow] = {}
        for window in windows:
            key = cls._substrate_key(
                window.network_id,
                window.region_id,
                window.task_slice_id,
                window.partition,
            )
            if key in open_windows:
                raise ValueError("multiple open structural evidence windows share a substrate")
            open_windows[key] = window
        summaries = [
            StructuralEvidenceWindowSummary.from_payload(item)
            for item in sealed_payload
            if isinstance(item, Mapping)
        ]
        if len(summaries) != len(sealed_payload):
            raise ValueError("structural evidence sealed window entry must be a mapping")
        compacted_windows = [
            StructuralEvidenceCompactedWindow.from_payload(item)
            for item in compacted_payload
            if isinstance(item, Mapping)
        ]
        if len(compacted_windows) != len(compacted_payload):
            raise ValueError("structural evidence compacted window entry must be a mapping")
        pressure_snapshot_payload = payload.get("pressure_snapshots", ())
        if not isinstance(pressure_snapshot_payload, (tuple, list)):
            raise ValueError("structural evidence pressure_snapshots must be a sequence")
        pressure_snapshots = [
            StructuralEvidencePressureSnapshot.from_payload(item)
            for item in pressure_snapshot_payload
            if isinstance(item, Mapping)
        ]
        if len(pressure_snapshots) != len(pressure_snapshot_payload):
            raise ValueError("structural evidence pressure snapshot entry must be a mapping")
        ledger = cls(
            window_capacity=int(payload["window_capacity"]),
            max_sealed_windows=int(payload["max_sealed_windows"]),
            max_evidence_index=int(payload["max_evidence_index"]),
            max_compacted_windows=int(
                payload.get("max_compacted_windows", DEFAULT_MAX_COMPACTED_WINDOWS)
            ),
            _open_windows=open_windows,
            _sealed_windows=summaries,
            _evidence_index={str(key): str(value) for key, value in evidence_index.items()},
            _compacted_windows=compacted_windows,
            _compacted_evidence_index={
                str(key): str(value) for key, value in compacted_evidence_index.items()
            },
            _pressure_snapshots=pressure_snapshots,
            _next_ordinals={str(key): int(value) for key, value in next_ordinals.items()},
        )
        for window in ledger._open_windows.values():
            for observation in window.observations:
                expected = _content_digest(observation.to_payload())
                if ledger._evidence_index.get(observation.evidence_id) != expected:
                    raise ValueError("structural evidence observation digest mismatch")
        return ledger

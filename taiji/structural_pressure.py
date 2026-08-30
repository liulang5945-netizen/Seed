"""Evidence-only projection for structural growth pressure.

The projector is intentionally not a proposal writer.  It accepts sealed
evidence windows, enforces task/partition separation, and returns metrics that
the existing growth controller may evaluate in a later stage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .structural_evidence import StructuralEvidenceWindowSummary

STRUCTURAL_PRESSURE_PROJECTION_FORMAT = "taiji-structural-pressure-projection-v1"


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _partition_count(summary: StructuralEvidenceWindowSummary, partition: str) -> int:
    return dict(summary.partition_counts).get(partition, 0)


@dataclass(frozen=True)
class StructuralGrowthEvidenceProjection:
    """Cross-task metrics derived only from sealed evidence windows."""

    network_id: str
    region_id: str
    first_tick: int
    last_tick: int
    window_digests: tuple[str, ...]
    train_task_slice_ids: tuple[str, ...]
    holdout_task_slice_ids: tuple[str, ...]
    retention_task_slice_ids: tuple[str, ...]
    train_window_count: int
    holdout_window_count: int
    retention_window_count: int
    prediction_observation_count: int
    mean_prediction_error: float
    mean_resource_state: float
    mean_holdout_transfer: float | None
    evidence_ids: tuple[str, ...]
    projection_digest: str

    def __post_init__(self) -> None:
        if not self.network_id or not self.region_id or not self.projection_digest:
            raise ValueError("structural pressure projection identifiers must not be empty")
        if int(self.first_tick) <= 0 or int(self.last_tick) < int(self.first_tick):
            raise ValueError("structural pressure projection ticks are invalid")
        if not self.window_digests or len(set(self.window_digests)) != len(self.window_digests):
            raise ValueError("structural pressure window digests must be unique")
        for name in (
            "train_window_count",
            "holdout_window_count",
            "retention_window_count",
            "prediction_observation_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError("structural pressure counters cannot be negative")
        if int(self.train_window_count) <= 0:
            raise ValueError("structural pressure requires train windows")
        if not 0.0 <= float(self.mean_prediction_error) <= 1.0:
            raise ValueError("structural pressure prediction error must be in [0, 1]")
        if not 0.0 <= float(self.mean_resource_state) <= 1.0:
            raise ValueError("structural pressure resource state must be in [0, 1]")
        if self.mean_holdout_transfer is not None and not 0.0 <= float(
            self.mean_holdout_transfer
        ) <= 1.0:
            raise ValueError("structural pressure holdout transfer must be in [0, 1]")
        evidence_ids = tuple(str(item) for item in self.evidence_ids)
        if not evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("structural pressure evidence ids must be unique and non-empty")
        object.__setattr__(self, "network_id", str(self.network_id))
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "first_tick", int(self.first_tick))
        object.__setattr__(self, "last_tick", int(self.last_tick))
        object.__setattr__(self, "window_digests", tuple(str(item) for item in self.window_digests))
        object.__setattr__(
            self,
            "train_task_slice_ids",
            tuple(str(item) for item in self.train_task_slice_ids),
        )
        object.__setattr__(
            self,
            "holdout_task_slice_ids",
            tuple(str(item) for item in self.holdout_task_slice_ids),
        )
        object.__setattr__(
            self,
            "retention_task_slice_ids",
            tuple(str(item) for item in self.retention_task_slice_ids),
        )
        for name in (
            "train_window_count",
            "holdout_window_count",
            "retention_window_count",
            "prediction_observation_count",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "mean_prediction_error", float(self.mean_prediction_error))
        object.__setattr__(self, "mean_resource_state", float(self.mean_resource_state))
        object.__setattr__(
            self,
            "mean_holdout_transfer",
            None if self.mean_holdout_transfer is None else float(self.mean_holdout_transfer),
        )
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "projection_digest", str(self.projection_digest))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_PRESSURE_PROJECTION_FORMAT,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "window_digests": list(self.window_digests),
            "train_task_slice_ids": list(self.train_task_slice_ids),
            "holdout_task_slice_ids": list(self.holdout_task_slice_ids),
            "retention_task_slice_ids": list(self.retention_task_slice_ids),
            "train_window_count": self.train_window_count,
            "holdout_window_count": self.holdout_window_count,
            "retention_window_count": self.retention_window_count,
            "prediction_observation_count": self.prediction_observation_count,
            "mean_prediction_error": self.mean_prediction_error,
            "mean_resource_state": self.mean_resource_state,
            "mean_holdout_transfer": self.mean_holdout_transfer,
            "evidence_ids": list(self.evidence_ids),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "projection_digest": self.projection_digest}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StructuralGrowthEvidenceProjection:
        if payload.get("format") != STRUCTURAL_PRESSURE_PROJECTION_FORMAT:
            raise ValueError("unsupported structural pressure projection format")
        expected = _digest(
            {key: value for key, value in payload.items() if key != "projection_digest"}
        )
        if payload.get("projection_digest") != expected:
            raise ValueError("structural pressure projection digest mismatch")
        return cls(
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            first_tick=int(payload["first_tick"]),
            last_tick=int(payload["last_tick"]),
            window_digests=tuple(str(item) for item in payload["window_digests"]),
            train_task_slice_ids=tuple(str(item) for item in payload["train_task_slice_ids"]),
            holdout_task_slice_ids=tuple(str(item) for item in payload["holdout_task_slice_ids"]),
            retention_task_slice_ids=tuple(
                str(item) for item in payload["retention_task_slice_ids"]
            ),
            train_window_count=int(payload["train_window_count"]),
            holdout_window_count=int(payload["holdout_window_count"]),
            retention_window_count=int(payload["retention_window_count"]),
            prediction_observation_count=int(payload["prediction_observation_count"]),
            mean_prediction_error=float(payload["mean_prediction_error"]),
            mean_resource_state=float(payload["mean_resource_state"]),
            mean_holdout_transfer=(
                None
                if payload.get("mean_holdout_transfer") is None
                else float(payload["mean_holdout_transfer"])
            ),
            evidence_ids=tuple(str(item) for item in payload["evidence_ids"]),
            projection_digest=str(payload["projection_digest"]),
        )


def project_structural_growth_pressure(
    summaries: Sequence[StructuralEvidenceWindowSummary],
    *,
    minimum_train_task_slices: int = 2,
    minimum_train_windows: int = 2,
    require_holdout: bool = True,
    require_retention: bool = False,
) -> StructuralGrowthEvidenceProjection:
    """Build a non-mutating pressure projection from sealed windows.

    ``train`` windows provide pressure metrics.  ``holdout`` and ``retention``
    windows are validation evidence only and never contribute to the train
    error or resource state.
    """

    if minimum_train_task_slices <= 0 or minimum_train_windows <= 0:
        raise ValueError("structural pressure minimums must be positive")
    items = tuple(summaries)
    if not items:
        raise ValueError("structural pressure requires sealed evidence windows")
    if any(not isinstance(item, StructuralEvidenceWindowSummary) for item in items):
        raise TypeError("structural pressure accepts sealed evidence window summaries")
    if len({item.window_digest for item in items}) != len(items):
        raise ValueError("structural pressure cannot reuse a window digest")
    substrates = {(item.network_id, item.region_id) for item in items}
    if len(substrates) != 1:
        raise ValueError("structural pressure windows must share one substrate")

    def partition(summary: StructuralEvidenceWindowSummary) -> str:
        partitions = tuple(
            name for name, count in summary.partition_counts if int(count) > 0
        )
        if len(partitions) != 1:
            raise ValueError("structural pressure requires one partition per window")
        return partitions[0]

    train = tuple(item for item in items if partition(item) == "train")
    holdout = tuple(item for item in items if partition(item) == "holdout")
    retention = tuple(item for item in items if partition(item) == "retention")
    if len(train) < minimum_train_windows:
        raise ValueError("structural pressure lacks enough train windows")
    train_task_slices = tuple(
        dict.fromkeys(
            task_slice
            for item in train
            for task_slice in item.task_slice_ids
            if task_slice
        )
    )
    if len(train_task_slices) < minimum_train_task_slices:
        raise ValueError("structural pressure lacks independent train task slices")
    if require_holdout and not holdout:
        raise ValueError("structural pressure requires a separate holdout window")
    if require_retention and not retention:
        raise ValueError("structural pressure requires a separate retention window")
    if any(item.mean_prediction_error is None for item in train):
        raise ValueError("train windows require prediction error evidence")
    train_errors = [float(item.mean_prediction_error) for item in train]
    train_resource_states = [1.0 - float(item.mean_resource_pressure) for item in train]
    holdout_transfer = (
        None
        if not holdout
        else sum(float(item.mean_holdout_transfer) for item in holdout) / len(holdout)
    )
    evidence_ids = tuple(
        dict.fromkeys(evidence_id for item in items for evidence_id in item.evidence_ids)
    )
    payload = {
        "format": STRUCTURAL_PRESSURE_PROJECTION_FORMAT,
        "network_id": items[0].network_id,
        "region_id": items[0].region_id,
        "first_tick": min(item.first_tick for item in items),
        "last_tick": max(item.last_tick for item in items),
        "window_digests": [item.window_digest for item in items],
        "train_task_slice_ids": list(train_task_slices),
        "holdout_task_slice_ids": list(
            dict.fromkeys(
                task_slice for item in holdout for task_slice in item.task_slice_ids if task_slice
            )
        ),
        "retention_task_slice_ids": list(
            dict.fromkeys(
                task_slice for item in retention for task_slice in item.task_slice_ids if task_slice
            )
        ),
        "train_window_count": len(train),
        "holdout_window_count": len(holdout),
        "retention_window_count": len(retention),
        "prediction_observation_count": sum(
            item.prediction_observation_count for item in train
        ),
        "mean_prediction_error": sum(train_errors) / len(train_errors),
        "mean_resource_state": sum(train_resource_states) / len(train_resource_states),
        "mean_holdout_transfer": holdout_transfer,
        "evidence_ids": list(evidence_ids),
    }
    return StructuralGrowthEvidenceProjection(
        network_id=items[0].network_id,
        region_id=items[0].region_id,
        first_tick=int(payload["first_tick"]),
        last_tick=int(payload["last_tick"]),
        window_digests=tuple(payload["window_digests"]),
        train_task_slice_ids=tuple(payload["train_task_slice_ids"]),
        holdout_task_slice_ids=tuple(payload["holdout_task_slice_ids"]),
        retention_task_slice_ids=tuple(payload["retention_task_slice_ids"]),
        train_window_count=len(train),
        holdout_window_count=len(holdout),
        retention_window_count=len(retention),
        prediction_observation_count=int(payload["prediction_observation_count"]),
        mean_prediction_error=float(payload["mean_prediction_error"]),
        mean_resource_state=float(payload["mean_resource_state"]),
        mean_holdout_transfer=holdout_transfer,
        evidence_ids=evidence_ids,
        projection_digest=_digest(payload),
    )

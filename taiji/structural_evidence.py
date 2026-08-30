"""Content-addressed long-horizon evidence windows for Taiji structural growth.

This module is deliberately an evidence ledger, not a growth controller.  It
preserves bounded, replayable runtime observations so a later stage can make a
growth decision from a window instead of from one tick or a scale target.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .structural_growth import STRUCTURAL_EVIDENCE_PARTITIONS, StructuralRuntimeObservation

STRUCTURAL_EVIDENCE_WINDOW_CHECKPOINT_FORMAT = "taiji-structural-evidence-window-v1"
STRUCTURAL_EVIDENCE_LEDGER_CHECKPOINT_FORMAT = "taiji-structural-evidence-ledger-v1"


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
    _open_windows: dict[str, StructuralEvidenceWindow] = field(
        default_factory=dict,
        repr=False,
    )
    _sealed_windows: list[StructuralEvidenceWindowSummary] = field(
        default_factory=list,
        repr=False,
    )
    _evidence_index: dict[str, str] = field(default_factory=dict, repr=False)
    _next_ordinals: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if int(self.window_capacity) <= 0:
            raise ValueError("structural evidence window_capacity must be positive")
        if int(self.max_sealed_windows) <= 0:
            raise ValueError("structural evidence max_sealed_windows must be positive")
        if int(self.max_evidence_index) <= 0:
            raise ValueError("structural evidence max_evidence_index must be positive")
        self.window_capacity = int(self.window_capacity)
        self.max_sealed_windows = int(self.max_sealed_windows)
        self.max_evidence_index = int(self.max_evidence_index)
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
        return len(self._evidence_index)

    @property
    def digest(self) -> str:
        payload = self._payload_without_digest()
        return _content_digest(payload)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_EVIDENCE_LEDGER_CHECKPOINT_FORMAT,
            "window_capacity": self.window_capacity,
            "max_sealed_windows": self.max_sealed_windows,
            "max_evidence_index": self.max_evidence_index,
            "open_windows": [item.to_payload() for item in self._open_windows.values()],
            "sealed_windows": [item.to_payload() for item in self._sealed_windows],
            "evidence_index": dict(sorted(self._evidence_index.items())),
            "next_ordinals": dict(sorted(self._next_ordinals.items())),
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self._payload_without_digest()
        return {**payload, "ledger_digest": _content_digest(payload)}

    def _validate_state(self) -> None:
        if len(self._sealed_windows) > self.max_sealed_windows:
            raise ValueError("structural evidence sealed windows exceed capacity")
        if len(self._evidence_index) > self.max_evidence_index:
            raise ValueError("structural evidence index exceeds capacity")
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
        evidence_index = payload.get("evidence_index", {})
        next_ordinals = payload.get("next_ordinals", {})
        if not isinstance(open_payload, (tuple, list)):
            raise ValueError("structural evidence open_windows must be a sequence")
        if not isinstance(sealed_payload, (tuple, list)):
            raise ValueError("structural evidence sealed_windows must be a sequence")
        if not isinstance(evidence_index, Mapping) or not isinstance(next_ordinals, Mapping):
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
        ledger = cls(
            window_capacity=int(payload["window_capacity"]),
            max_sealed_windows=int(payload["max_sealed_windows"]),
            max_evidence_index=int(payload["max_evidence_index"]),
            _open_windows=open_windows,
            _sealed_windows=summaries,
            _evidence_index={str(key): str(value) for key, value in evidence_index.items()},
            _next_ordinals={str(key): int(value) for key, value in next_ordinals.items()},
        )
        for window in ledger._open_windows.values():
            for observation in window.observations:
                expected = _content_digest(observation.to_payload())
                if ledger._evidence_index.get(observation.evidence_id) != expected:
                    raise ValueError("structural evidence observation digest mismatch")
        return ledger

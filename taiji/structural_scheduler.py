"""Checkpointable scheduling state for bounded Taiji structural growth."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

STRUCTURAL_GROWTH_SCHEDULER_CHECKPOINT_FORMAT = "taiji-structural-growth-scheduler-v1"
STRUCTURAL_GROWTH_SCHEDULE_RESULT_FORMAT = "taiji-structural-growth-schedule-result-v1"
STRUCTURAL_WORKBENCH_BATCH_SCHEDULE_FORMAT = "taiji-structural-workbench-batch-schedule-v1"


@dataclass(frozen=True)
class StructuralGrowthScheduleState:
    """Persisted cursors preventing one stream from starving another.

    ``last_evaluated_tick`` remains as the legacy aggregate cursor for old
    checkpoints and observability.  Cooldown decisions use the stream-scoped
    cursors so independent network/region evidence streams cannot block one
    another merely because batch request ordering differs.
    """

    window_interval_ticks: int = 1
    last_evaluated_tick: int = 0
    evaluated_window_digests: tuple[str, ...] = ()
    revision: int = 0
    stream_last_evaluated_ticks: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if int(self.window_interval_ticks) <= 0:
            raise ValueError("structural scheduler window interval must be positive")
        if int(self.last_evaluated_tick) < 0:
            raise ValueError("structural scheduler last tick cannot be negative")
        if int(self.revision) < 0:
            raise ValueError("structural scheduler revision cannot be negative")
        digests = tuple(str(item) for item in self.evaluated_window_digests)
        if any(not item for item in digests) or len(set(digests)) != len(digests):
            raise ValueError("structural scheduler window digests must be unique and non-empty")
        stream_ticks = tuple((str(key), int(value)) for key, value in self.stream_last_evaluated_ticks)
        if any(not key for key, _ in stream_ticks) or len({key for key, _ in stream_ticks}) != len(
            stream_ticks
        ):
            raise ValueError("structural scheduler stream cursors must have unique non-empty keys")
        if any(value < 0 for _, value in stream_ticks):
            raise ValueError("structural scheduler stream cursor cannot be negative")
        object.__setattr__(self, "window_interval_ticks", int(self.window_interval_ticks))
        object.__setattr__(self, "last_evaluated_tick", int(self.last_evaluated_tick))
        object.__setattr__(self, "evaluated_window_digests", digests)
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "stream_last_evaluated_ticks", tuple(sorted(stream_ticks)))

    def last_evaluated_tick_for(self, stream_key: str | None = None) -> int:
        """Return the cooldown cursor for one evidence stream.

        A legacy checkpoint has no stream map.  In that case its aggregate
        cursor is used as a conservative fallback.  A new state starts at
        zero, and once stream cursors exist an unseen stream also starts at
        zero instead of inheriting another region's clock.
        """

        if stream_key is None:
            return self.last_evaluated_tick
        key = str(stream_key)
        if not key:
            raise ValueError("structural scheduler stream key must not be empty")
        stream_ticks = dict(self.stream_last_evaluated_ticks)
        if key in stream_ticks:
            return stream_ticks[key]
        if not stream_ticks and self.last_evaluated_tick > 0:
            return self.last_evaluated_tick
        return 0

    def advance(
        self,
        *,
        last_evaluated_tick: int,
        window_digests: tuple[str, ...],
        stream_key: str | None = None,
    ) -> StructuralGrowthScheduleState:
        merged = tuple(dict.fromkeys((*self.evaluated_window_digests, *window_digests)))
        stream_ticks = dict(self.stream_last_evaluated_ticks)
        if stream_key is None:
            key = "global"
        else:
            key = str(stream_key)
            if not key:
                raise ValueError("structural scheduler stream key must not be empty")
        prior_stream_tick = self.last_evaluated_tick_for(key)
        stream_ticks[key] = max(prior_stream_tick, int(last_evaluated_tick))
        return StructuralGrowthScheduleState(
            window_interval_ticks=self.window_interval_ticks,
            last_evaluated_tick=max(self.last_evaluated_tick, int(last_evaluated_tick)),
            evaluated_window_digests=merged[-256:],
            revision=self.revision + 1,
            stream_last_evaluated_ticks=tuple(stream_ticks.items()),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_GROWTH_SCHEDULER_CHECKPOINT_FORMAT,
            "window_interval_ticks": self.window_interval_ticks,
            "last_evaluated_tick": self.last_evaluated_tick,
            "evaluated_window_digests": list(self.evaluated_window_digests),
            "revision": self.revision,
            "stream_last_evaluated_ticks": {
                key: value for key, value in self.stream_last_evaluated_ticks
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralGrowthScheduleState:
        if payload.get("format") != STRUCTURAL_GROWTH_SCHEDULER_CHECKPOINT_FORMAT:
            raise ValueError("unsupported structural scheduler state format")
        raw_digests = payload.get("evaluated_window_digests", ())
        if isinstance(raw_digests, (str, bytes)) or not isinstance(raw_digests, (tuple, list)):
            raise ValueError("structural scheduler window digests must be a sequence")
        raw_stream_ticks = payload.get("stream_last_evaluated_ticks", {})
        if isinstance(raw_stream_ticks, Mapping):
            stream_ticks = tuple(
                (str(key), int(value)) for key, value in raw_stream_ticks.items()
            )
        elif isinstance(raw_stream_ticks, (tuple, list)):
            stream_ticks = tuple(
                (str(item[0]), int(item[1]))
                for item in raw_stream_ticks
                if isinstance(item, (tuple, list)) and len(item) == 2
            )
            if len(stream_ticks) != len(raw_stream_ticks):
                raise ValueError("structural scheduler stream cursors must be key/value pairs")
        else:
            raise ValueError("structural scheduler stream cursors must be a mapping")
        return cls(
            window_interval_ticks=int(payload.get("window_interval_ticks", 1)),
            last_evaluated_tick=int(payload.get("last_evaluated_tick", 0)),
            evaluated_window_digests=tuple(str(item) for item in raw_digests),
            revision=int(payload.get("revision", 0)),
            stream_last_evaluated_ticks=stream_ticks,
        )


@dataclass(frozen=True)
class StructuralGrowthScheduleResult:
    """One non-admitting scheduler decision bound to new sealed windows."""

    status: str
    trigger_tick: int
    new_window_digests: tuple[str, ...] = ()
    projection_digest: str | None = None
    candidate_id: str | None = None
    reason: str = ""
    scheduler_revision: int = 0

    def __post_init__(self) -> None:
        if self.status not in {"waiting", "candidate_created", "no_growth", "failed_closed"}:
            raise ValueError("unsupported structural scheduler result status")
        if int(self.trigger_tick) < 0:
            raise ValueError("structural scheduler trigger tick cannot be negative")
        digests = tuple(str(item) for item in self.new_window_digests)
        if any(not item for item in digests) or len(set(digests)) != len(digests):
            raise ValueError("structural scheduler result window digests must be unique")
        if self.projection_digest is not None and not str(self.projection_digest):
            raise ValueError("structural scheduler projection digest must not be empty")
        if self.candidate_id is not None and not str(self.candidate_id):
            raise ValueError("structural scheduler candidate id must not be empty")
        if int(self.scheduler_revision) < 0:
            raise ValueError("structural scheduler result revision cannot be negative")
        object.__setattr__(self, "trigger_tick", int(self.trigger_tick))
        object.__setattr__(self, "new_window_digests", digests)
        object.__setattr__(
            self,
            "projection_digest",
            None if self.projection_digest is None else str(self.projection_digest),
        )
        object.__setattr__(
            self,
            "candidate_id",
            None if self.candidate_id is None else str(self.candidate_id),
        )
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "scheduler_revision", int(self.scheduler_revision))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_GROWTH_SCHEDULE_RESULT_FORMAT,
            "status": self.status,
            "trigger_tick": self.trigger_tick,
            "new_window_digests": list(self.new_window_digests),
            "projection_digest": self.projection_digest,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "scheduler_revision": self.scheduler_revision,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralGrowthScheduleResult:
        if payload.get("format") != STRUCTURAL_GROWTH_SCHEDULE_RESULT_FORMAT:
            raise ValueError("unsupported structural scheduler result format")
        raw_digests = payload.get("new_window_digests", ())
        if isinstance(raw_digests, (str, bytes)) or not isinstance(raw_digests, (tuple, list)):
            raise ValueError("structural scheduler result digests must be a sequence")
        return cls(
            status=str(payload["status"]),
            trigger_tick=int(payload.get("trigger_tick", 0)),
            new_window_digests=tuple(str(item) for item in raw_digests),
            projection_digest=(
                None
                if payload.get("projection_digest") is None
                else str(payload["projection_digest"])
            ),
            candidate_id=(None if payload.get("candidate_id") is None else str(payload["candidate_id"])),
            reason=str(payload.get("reason", "")),
            scheduler_revision=int(payload.get("scheduler_revision", 0)),
        )


@dataclass(frozen=True)
class StructuralWorkbenchBatchScheduleResult:
    """One multi-region Workbench scheduling decision before admission."""

    status: str
    trigger_tick: int
    request_digest: str
    region_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    source_window_digests: tuple[str, ...] = ()
    batch_id: str | None = None
    reason: str = ""
    scheduler_revision: int = 0

    def __post_init__(self) -> None:
        if self.status not in {"waiting", "batch_created", "no_candidates", "failed_closed"}:
            raise ValueError("unsupported Workbench batch schedule result status")
        if int(self.trigger_tick) < 0:
            raise ValueError("Workbench batch schedule trigger tick cannot be negative")
        if not str(self.request_digest):
            raise ValueError("Workbench batch schedule request digest must not be empty")
        for name, values in (
            ("region_ids", self.region_ids),
            ("candidate_ids", self.candidate_ids),
            ("source_window_digests", self.source_window_digests),
        ):
            normalized = tuple(str(item) for item in values)
            if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
                raise ValueError(f"Workbench batch schedule {name} must be unique and non-empty")
            object.__setattr__(self, name, normalized)
        if self.batch_id is not None and not str(self.batch_id):
            raise ValueError("Workbench batch schedule batch id must not be empty")
        if int(self.scheduler_revision) < 0:
            raise ValueError("Workbench batch schedule revision cannot be negative")
        object.__setattr__(self, "trigger_tick", int(self.trigger_tick))
        object.__setattr__(self, "request_digest", str(self.request_digest))
        object.__setattr__(self, "batch_id", None if self.batch_id is None else str(self.batch_id))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "scheduler_revision", int(self.scheduler_revision))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_WORKBENCH_BATCH_SCHEDULE_FORMAT,
            "status": self.status,
            "trigger_tick": self.trigger_tick,
            "request_digest": self.request_digest,
            "region_ids": list(self.region_ids),
            "candidate_ids": list(self.candidate_ids),
            "source_window_digests": list(self.source_window_digests),
            "batch_id": self.batch_id,
            "reason": self.reason,
            "scheduler_revision": self.scheduler_revision,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralWorkbenchBatchScheduleResult:
        if payload.get("format") != STRUCTURAL_WORKBENCH_BATCH_SCHEDULE_FORMAT:
            raise ValueError("unsupported Workbench batch schedule result format")

        def _sequence(name: str) -> tuple[str, ...]:
            raw = payload.get(name, ())
            if isinstance(raw, (str, bytes)) or not isinstance(raw, (tuple, list)):
                raise ValueError(f"Workbench batch schedule {name} must be a sequence")
            return tuple(str(item) for item in raw)

        return cls(
            status=str(payload["status"]),
            trigger_tick=int(payload.get("trigger_tick", 0)),
            request_digest=str(payload["request_digest"]),
            region_ids=_sequence("region_ids"),
            candidate_ids=_sequence("candidate_ids"),
            source_window_digests=_sequence("source_window_digests"),
            batch_id=None if payload.get("batch_id") is None else str(payload["batch_id"]),
            reason=str(payload.get("reason", "")),
            scheduler_revision=int(payload.get("scheduler_revision", 0)),
        )

"""Train-only utility learning for selecting trace-grounded interaction groups."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .interaction_groups import InteractionGroupRecord

INTERACTION_GROUP_LEARNING_CHECKPOINT_FORMAT = "taiji-interaction-group-learning-v1"


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(value: str, name: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class InteractionGroupSelection:
    """One learned group choice with its train-evidence lineage."""

    group_id: str
    member_ids: tuple[str, ...]
    source_trace_digest: str
    checkpoint_revision: int
    utility: float
    resource_cost: float
    observations: int
    version: int = 1

    def __post_init__(self) -> None:
        _text(self.group_id, "interaction selection group_id")
        if len(self.member_ids) < 2 or tuple(sorted(set(self.member_ids))) != self.member_ids:
            raise ValueError("interaction selection member_ids must be sorted and unique")
        if len(self.source_trace_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_trace_digest
        ):
            raise ValueError("interaction selection source_trace_digest must be lowercase SHA-256")
        if int(self.checkpoint_revision) < 0:
            raise ValueError("interaction selection checkpoint_revision cannot be negative")
        _finite(self.utility, "interaction selection utility")
        if _finite(self.resource_cost, "interaction selection resource_cost") < 0.0:
            raise ValueError("interaction selection resource_cost cannot be negative")
        if int(self.observations) <= 0:
            raise ValueError("interaction selection observations must be positive")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction selection version: {self.version}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_GROUP_LEARNING_CHECKPOINT_FORMAT,
            "version": self.version,
            "group_id": self.group_id,
            "member_ids": list(self.member_ids),
            "source_trace_digest": self.source_trace_digest,
            "checkpoint_revision": self.checkpoint_revision,
            "utility": self.utility,
            "resource_cost": self.resource_cost,
            "observations": self.observations,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionGroupSelection:
        if payload.get("format", INTERACTION_GROUP_LEARNING_CHECKPOINT_FORMAT) != (
            INTERACTION_GROUP_LEARNING_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported interaction selection format")
        return cls(
            version=int(payload.get("version", 1)),
            group_id=str(payload["group_id"]),
            member_ids=tuple(str(item) for item in payload.get("member_ids", ())),
            source_trace_digest=str(payload["source_trace_digest"]),
            checkpoint_revision=int(payload["checkpoint_revision"]),
            utility=float(payload["utility"]),
            resource_cost=float(payload["resource_cost"]),
            observations=int(payload["observations"]),
        )


class InteractionGroupUtilityLearner:
    """Learn group utility from train-only interaction evidence.

    The learner consumes opaque group records and their measured interaction
    values.  It never reads holdout-derived fields, semantic role labels, or
    provider/tool identifiers.  Selection is a reversible checkpointable
    projection and does not mutate Workbench policy.
    """

    def __init__(self, *, learning_rate: float = 1.0, minimum_utility: float = 0.0) -> None:
        if not 0.0 < float(learning_rate) <= 1.0:
            raise ValueError("interaction group learning_rate must be in (0, 1]")
        if not math.isfinite(float(minimum_utility)):
            raise ValueError("interaction group minimum_utility must be finite")
        self.learning_rate = float(learning_rate)
        self.minimum_utility = float(minimum_utility)
        self._groups: dict[str, InteractionGroupSelection] = {}
        self.total_observations = 0
        self._source_trace_digest: str | None = None
        self._checkpoint_revision: int | None = None

    @property
    def groups(self) -> tuple[InteractionGroupSelection, ...]:
        return tuple(self._groups[key] for key in sorted(self._groups))

    @property
    def source_trace_digest(self) -> str | None:
        return self._source_trace_digest

    @property
    def checkpoint_revision(self) -> int | None:
        return self._checkpoint_revision

    def observe(self, records: Iterable[InteractionGroupRecord]) -> tuple[str, ...]:
        """Update utilities from records that contain no holdout evidence."""

        observed_ids: list[str] = []
        for record in records:
            if not isinstance(record, InteractionGroupRecord):
                raise TypeError("interaction learner records must be InteractionGroupRecord values")
            if record.status not in {"candidate", "admitted"}:
                raise ValueError("interaction learner cannot consume terminal group records")
            if record.holdout_interaction is not None or record.holdout_recovery_effect is not None:
                raise ValueError("interaction learner cannot consume holdout-derived evidence")
            if self._source_trace_digest is None:
                self._source_trace_digest = record.source_trace_digest
                self._checkpoint_revision = int(record.checkpoint_revision)
            elif (
                record.source_trace_digest != self._source_trace_digest
                or int(record.checkpoint_revision) != self._checkpoint_revision
            ):
                raise ValueError("interaction learner evidence crosses trace or checkpoint lineage")
            previous = self._groups.get(record.group_id)
            if previous is None:
                utility = float(record.interaction)
                observations = 1
            else:
                utility = (1.0 - self.learning_rate) * previous.utility + self.learning_rate * float(
                    record.interaction
                )
                observations = previous.observations + 1
            self._groups[record.group_id] = InteractionGroupSelection(
                group_id=record.group_id,
                member_ids=record.member_ids,
                source_trace_digest=record.source_trace_digest,
                checkpoint_revision=int(record.checkpoint_revision),
                utility=utility,
                resource_cost=float(record.resource_cost),
                observations=observations,
            )
            observed_ids.append(record.group_id)
            self.total_observations += 1
        return tuple(observed_ids)

    def select(self, *, resource_budget: float | None = None) -> InteractionGroupSelection | None:
        """Select the highest train-learned utility group within the budget."""

        if resource_budget is not None and float(resource_budget) < 0.0:
            raise ValueError("interaction group resource_budget cannot be negative")
        candidates = [
            item
            for item in self.groups
            if item.utility >= self.minimum_utility
            and (resource_budget is None or item.resource_cost <= float(resource_budget))
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: (-item.utility, item.resource_cost, item.group_id))

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": INTERACTION_GROUP_LEARNING_CHECKPOINT_FORMAT,
            "version": 1,
            "learning_rate": self.learning_rate,
            "minimum_utility": self.minimum_utility,
            "groups": [item.to_payload() for item in self.groups],
            "total_observations": self.total_observations,
            "source_trace_digest": self._source_trace_digest,
            "checkpoint_revision": self._checkpoint_revision,
        }
        payload["checkpoint_digest"] = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> InteractionGroupUtilityLearner:
        if payload.get("format") != INTERACTION_GROUP_LEARNING_CHECKPOINT_FORMAT:
            raise ValueError("unsupported interaction group learner checkpoint format")
        expected_digest = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest")) != expected_digest:
            raise ValueError("interaction group learner checkpoint digest mismatch")
        learner = cls(
            learning_rate=float(payload["learning_rate"]),
            minimum_utility=float(payload["minimum_utility"]),
        )
        learner._groups = {
            item.group_id: item
            for item in (
                InteractionGroupSelection.from_payload(entry)
                for entry in payload.get("groups", ())
            )
        }
        if len(learner._groups) != len(payload.get("groups", ())):
            raise ValueError("interaction group learner checkpoint contains duplicate groups")
        learner.total_observations = int(payload.get("total_observations", 0))
        if learner.total_observations < 0:
            raise ValueError("interaction group learner total_observations cannot be negative")
        learner._source_trace_digest = (
            None if payload.get("source_trace_digest") is None else str(payload["source_trace_digest"])
        )
        learner._checkpoint_revision = (
            None
            if payload.get("checkpoint_revision") is None
            else int(payload["checkpoint_revision"])
        )
        if learner._groups:
            lineages = {
                (item.source_trace_digest, int(item.checkpoint_revision))
                for item in learner._groups.values()
            }
            if len(lineages) != 1:
                raise ValueError("interaction group learner checkpoint crosses evidence lineage")
            if (learner._source_trace_digest, learner._checkpoint_revision) != next(iter(lineages)):
                raise ValueError("interaction group learner checkpoint lineage is stale")
        return learner


__all__ = [
    "INTERACTION_GROUP_LEARNING_CHECKPOINT_FORMAT",
    "InteractionGroupSelection",
    "InteractionGroupUtilityLearner",
]

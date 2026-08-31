"""Train-only transfer for interaction groups with opaque member evidence.

The identity-based interaction learner is intentionally conservative: it can
remember a measured group, but it cannot score a group whose member set has
never been observed.  This module adds the next boundary without introducing
semantic role labels.  It derives a compact continuous profile for each
opaque member from singleton trace evidence and fits a regularized symmetric
relation over observed group interactions.  The result is a candidate
prediction, not an admission: policy, execution, lesion, and rollback remain
outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .interaction_group_learning import InteractionGroupSelection
from .interaction_groups import InteractionGroupRecord, InteractionTraceEpisode

INTERACTION_GROUP_TRANSFER_CHECKPOINT_FORMAT = "taiji-interaction-group-transfer-v1"
INTERACTION_GROUP_TRANSFER_MODEL_REVISION = 1


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


def _nonnegative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _digest_text(value: str, name: str) -> str:
    value = _text(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class InteractionGroupMemberEvidence:
    """A role-free, train-only profile for one opaque interaction member."""

    member_id: str
    source_trace_digest: str
    checkpoint_revision: int
    contribution: float
    recovery_effect: float
    resource_cost: float
    observations: int
    context_count: int
    version: int = 1

    def __post_init__(self) -> None:
        _text(self.member_id, "interaction member_id")
        _digest_text(self.source_trace_digest, "interaction member source_trace_digest")
        if int(self.checkpoint_revision) < 0:
            raise ValueError("interaction member checkpoint_revision cannot be negative")
        _finite(self.contribution, "interaction member contribution")
        _finite(self.recovery_effect, "interaction member recovery_effect")
        _nonnegative(self.resource_cost, "interaction member resource_cost")
        if int(self.observations) <= 0:
            raise ValueError("interaction member observations must be positive")
        if int(self.context_count) <= 0:
            raise ValueError("interaction member context_count must be positive")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction member evidence version: {self.version}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_GROUP_TRANSFER_CHECKPOINT_FORMAT,
            "version": self.version,
            "member_id": self.member_id,
            "source_trace_digest": self.source_trace_digest,
            "checkpoint_revision": self.checkpoint_revision,
            "contribution": self.contribution,
            "recovery_effect": self.recovery_effect,
            "resource_cost": self.resource_cost,
            "observations": self.observations,
            "context_count": self.context_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionGroupMemberEvidence:
        if payload.get("format", INTERACTION_GROUP_TRANSFER_CHECKPOINT_FORMAT) != (
            INTERACTION_GROUP_TRANSFER_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported interaction member evidence format")
        return cls(
            version=int(payload.get("version", 1)),
            member_id=str(payload["member_id"]),
            source_trace_digest=str(payload["source_trace_digest"]),
            checkpoint_revision=int(payload["checkpoint_revision"]),
            contribution=float(payload["contribution"]),
            recovery_effect=float(payload.get("recovery_effect", 0.0)),
            resource_cost=float(payload.get("resource_cost", 0.0)),
            observations=int(payload["observations"]),
            context_count=int(payload["context_count"]),
        )


@dataclass(frozen=True)
class InteractionGroupTransferCandidate:
    """A predicted group that has not been admitted or executed."""

    group_id: str
    member_ids: tuple[str, ...]
    source_trace_digest: str
    checkpoint_revision: int
    predicted_interaction: float
    uncertainty: float
    resource_cost: float
    support: int
    method: str = "train-only-member-profile-relation"
    version: int = 1

    def __post_init__(self) -> None:
        _text(self.group_id, "interaction transfer group_id")
        if len(self.member_ids) < 2 or tuple(sorted(set(self.member_ids))) != self.member_ids:
            raise ValueError("interaction transfer member_ids must be sorted and unique")
        _digest_text(self.source_trace_digest, "interaction transfer source_trace_digest")
        if int(self.checkpoint_revision) < 0:
            raise ValueError("interaction transfer checkpoint_revision cannot be negative")
        _finite(self.predicted_interaction, "interaction transfer predicted_interaction")
        _nonnegative(self.uncertainty, "interaction transfer uncertainty")
        _nonnegative(self.resource_cost, "interaction transfer resource_cost")
        if int(self.support) < 0:
            raise ValueError("interaction transfer support cannot be negative")
        _text(self.method, "interaction transfer method")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction transfer version: {self.version}")

    @property
    def utility(self) -> float:
        return float(self.predicted_interaction)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_GROUP_TRANSFER_CHECKPOINT_FORMAT,
            "version": self.version,
            "group_id": self.group_id,
            "member_ids": list(self.member_ids),
            "source_trace_digest": self.source_trace_digest,
            "checkpoint_revision": self.checkpoint_revision,
            "predicted_interaction": self.predicted_interaction,
            "uncertainty": self.uncertainty,
            "resource_cost": self.resource_cost,
            "support": self.support,
            "method": self.method,
        }


def build_member_evidence(
    episodes: Sequence[InteractionTraceEpisode],
    *,
    source_trace_digest: str,
    checkpoint_revision: int,
) -> tuple[InteractionGroupMemberEvidence, ...]:
    """Derive member profiles from singleton-vs-baseline train evidence.

    A member is admitted to the profile set only when at least one context
    contains both an inactive baseline and a singleton episode.  This makes a
    genuinely unknown member fail closed instead of receiving an arbitrary
    default vector.  No owner name is interpreted as a semantic role.
    """

    _digest_text(source_trace_digest, "interaction member source_trace_digest")
    if int(checkpoint_revision) < 0:
        raise ValueError("interaction member checkpoint_revision cannot be negative")
    contexts: dict[str, dict[frozenset[str], list[InteractionTraceEpisode]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode in episodes:
        if int(episode.checkpoint_revision) != int(checkpoint_revision):
            raise ValueError("member evidence cannot cross checkpoint revisions")
        contexts[episode.context_id][frozenset(episode.member_ids)].append(episode)

    observations: dict[str, list[tuple[float, float, float, str]]] = defaultdict(list)
    all_members = sorted(
        {
            member
            for episode in episodes
            for member in episode.member_ids
        }
    )
    for context_id, cells in contexts.items():
        baseline = cells.get(frozenset())
        if not baseline:
            continue
        baseline_outcome = sum(item.outcome for item in baseline) / len(baseline)
        baseline_recovery = sum(item.recovery_effect for item in baseline) / len(baseline)
        for member in all_members:
            singleton = cells.get(frozenset((member,)))
            if not singleton:
                continue
            for episode in singleton:
                event_cost = sum(
                    event.resource_cost for event in episode.events if event.owner_id == member
                )
                observations[member].append(
                    (
                        float(episode.outcome - baseline_outcome),
                        float(episode.recovery_effect - baseline_recovery),
                        float(event_cost),
                        context_id,
                    )
                )

    profiles: list[InteractionGroupMemberEvidence] = []
    for member in all_members:
        values = observations.get(member, [])
        if not values:
            continue
        profiles.append(
            InteractionGroupMemberEvidence(
                member_id=member,
                source_trace_digest=source_trace_digest,
                checkpoint_revision=int(checkpoint_revision),
                contribution=sum(item[0] for item in values) / len(values),
                recovery_effect=sum(item[1] for item in values) / len(values),
                resource_cost=sum(item[2] for item in values) / len(values),
                observations=len(values),
                context_count=len({item[3] for item in values}),
            )
        )
    return tuple(profiles)


class InteractionGroupTransferLearner:
    """Fit and select unseen member combinations from train-only evidence."""

    def __init__(
        self,
        *,
        ridge: float = 0.1,
        minimum_utility: float = 0.0,
        maximum_uncertainty: float = 1.0,
    ) -> None:
        if not 0.0 < float(ridge):
            raise ValueError("interaction transfer ridge must be positive")
        if not math.isfinite(float(minimum_utility)):
            raise ValueError("interaction transfer minimum_utility must be finite")
        if not 0.0 <= float(maximum_uncertainty):
            raise ValueError("interaction transfer maximum_uncertainty cannot be negative")
        self.ridge = float(ridge)
        self.minimum_utility = float(minimum_utility)
        self.maximum_uncertainty = float(maximum_uncertainty)
        self._profiles: dict[str, InteractionGroupMemberEvidence] = {}
        self._records: list[InteractionGroupRecord] = []
        self._coefficients: tuple[float, ...] = ()
        self._residual_rmse = 0.0
        self._source_trace_digest: str | None = None
        self._checkpoint_revision: int | None = None

    @property
    def profiles(self) -> tuple[InteractionGroupMemberEvidence, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))

    @property
    def observed_records(self) -> tuple[InteractionGroupRecord, ...]:
        return tuple(self._records)

    @property
    def source_trace_digest(self) -> str | None:
        return self._source_trace_digest

    @property
    def checkpoint_revision(self) -> int | None:
        return self._checkpoint_revision

    @property
    def model_digest(self) -> str:
        return _digest(
            {
                "revision": INTERACTION_GROUP_TRANSFER_MODEL_REVISION,
                "coefficients": list(self._coefficients),
                "residual_rmse": self._residual_rmse,
                "profiles": [item.to_payload() for item in self.profiles],
            }
        )

    def observe_members(self, profiles: Iterable[InteractionGroupMemberEvidence]) -> tuple[str, ...]:
        """Register role-free member evidence from one train lineage."""

        observed: list[str] = []
        for profile in profiles:
            if not isinstance(profile, InteractionGroupMemberEvidence):
                raise TypeError("interaction transfer profiles must be member evidence values")
            self._bind_lineage(profile.source_trace_digest, profile.checkpoint_revision)
            if profile.member_id in self._profiles:
                raise ValueError(f"duplicate interaction member profile: {profile.member_id}")
            self._profiles[profile.member_id] = profile
            observed.append(profile.member_id)
        self._fit()
        return tuple(observed)

    def observe_records(self, records: Iterable[InteractionGroupRecord]) -> tuple[str, ...]:
        """Register only candidate/admitted records with no holdout fields."""

        observed: list[str] = []
        for record in records:
            if not isinstance(record, InteractionGroupRecord):
                raise TypeError("interaction transfer records must be InteractionGroupRecord values")
            if record.status not in {"candidate", "admitted"}:
                raise ValueError("interaction transfer cannot consume terminal group records")
            if record.holdout_interaction is not None or record.holdout_recovery_effect is not None:
                raise ValueError("interaction transfer cannot consume holdout-derived evidence")
            self._bind_lineage(record.source_trace_digest, record.checkpoint_revision)
            if any(member not in self._profiles for member in record.member_ids):
                raise ValueError("interaction transfer record references an unknown member profile")
            self._records.append(record)
            observed.append(record.group_id)
        self._fit()
        return tuple(observed)

    def candidate(
        self,
        member_ids: Sequence[str],
        *,
        allow_observed: bool = False,
    ) -> InteractionGroupTransferCandidate | None:
        """Predict one pair; unknown members and duplicate pairs fail closed."""

        members = tuple(sorted(set(str(item) for item in member_ids)))
        if len(members) != len(tuple(member_ids)) or len(members) < 2:
            raise ValueError("interaction transfer candidates need sorted unique member_ids")
        if self._source_trace_digest is None or self._checkpoint_revision is None:
            return None
        if any(member not in self._profiles for member in members):
            return None
        pair = frozenset(members)
        if not allow_observed and any(frozenset(record.member_ids) == pair for record in self._records):
            return None
        if not self._coefficients:
            return None
        features = self._pair_features(members)
        prediction = sum(left * right for left, right in zip(self._coefficients, features))
        support = sum(
            1
            for record in self._records
            if set(record.member_ids) & set(members)
        )
        profile_observations = min(self._profiles[member].observations for member in members)
        uncertainty = self._residual_rmse + 1.0 / math.sqrt(float(max(1, profile_observations)))
        resource_cost = sum(self._profiles[member].resource_cost for member in members)
        group_id = "transfer-group:" + _digest(
            {
                "members": list(members),
                "source_trace_digest": self._source_trace_digest,
                "checkpoint_revision": self._checkpoint_revision,
                "model_digest": self.model_digest,
            }
        )[:24]
        return InteractionGroupTransferCandidate(
            group_id=group_id,
            member_ids=members,
            source_trace_digest=self._source_trace_digest,
            checkpoint_revision=self._checkpoint_revision,
            predicted_interaction=float(prediction),
            uncertainty=float(uncertainty),
            resource_cost=float(resource_cost),
            support=support,
        )

    def select(
        self,
        candidate_member_sets: Iterable[Sequence[str]],
        *,
        resource_budget: float | None = None,
        unseen_only: bool = True,
    ) -> tuple[InteractionGroupSelection, InteractionGroupTransferCandidate] | None:
        """Select a predicted candidate within utility/resource/confidence bounds."""

        if resource_budget is not None and float(resource_budget) < 0.0:
            raise ValueError("interaction transfer resource_budget cannot be negative")
        candidates = [
            candidate
            for member_ids in candidate_member_sets
            if (candidate := self.candidate(member_ids, allow_observed=not unseen_only)) is not None
            and candidate.predicted_interaction >= self.minimum_utility
            and candidate.uncertainty <= self.maximum_uncertainty
            and (
                resource_budget is None
                or candidate.resource_cost <= float(resource_budget)
            )
        ]
        if not candidates:
            return None
        selected = min(
            candidates,
            key=lambda item: (
                -item.predicted_interaction,
                item.uncertainty,
                item.resource_cost,
                item.group_id,
            ),
        )
        return (
            InteractionGroupSelection(
                group_id=selected.group_id,
                member_ids=selected.member_ids,
                source_trace_digest=selected.source_trace_digest,
                checkpoint_revision=selected.checkpoint_revision,
                utility=selected.predicted_interaction,
                resource_cost=selected.resource_cost,
                observations=max(1, selected.support),
            ),
            selected,
        )

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": INTERACTION_GROUP_TRANSFER_CHECKPOINT_FORMAT,
            "version": 1,
            "model_revision": INTERACTION_GROUP_TRANSFER_MODEL_REVISION,
            "ridge": self.ridge,
            "minimum_utility": self.minimum_utility,
            "maximum_uncertainty": self.maximum_uncertainty,
            "profiles": [item.to_payload() for item in self.profiles],
            "records": [item.to_payload() for item in self._records],
            "coefficients": list(self._coefficients),
            "residual_rmse": self._residual_rmse,
            "source_trace_digest": self._source_trace_digest,
            "checkpoint_revision": self._checkpoint_revision,
        }
        payload["checkpoint_digest"] = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> InteractionGroupTransferLearner:
        if payload.get("format") != INTERACTION_GROUP_TRANSFER_CHECKPOINT_FORMAT:
            raise ValueError("unsupported interaction transfer checkpoint format")
        if int(payload.get("model_revision", -1)) != INTERACTION_GROUP_TRANSFER_MODEL_REVISION:
            raise ValueError("unsupported interaction transfer model revision")
        expected_digest = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest")) != expected_digest:
            raise ValueError("interaction transfer checkpoint digest mismatch")
        learner = cls(
            ridge=float(payload["ridge"]),
            minimum_utility=float(payload["minimum_utility"]),
            maximum_uncertainty=float(payload["maximum_uncertainty"]),
        )
        profiles = tuple(
            InteractionGroupMemberEvidence.from_payload(item)
            for item in payload.get("profiles", ())
        )
        learner.observe_members(profiles)
        records = tuple(
            InteractionGroupRecord.from_payload(item) for item in payload.get("records", ())
        )
        learner.observe_records(records)
        if learner._source_trace_digest != payload.get("source_trace_digest"):
            raise ValueError("interaction transfer checkpoint lineage is stale")
        if learner._checkpoint_revision != payload.get("checkpoint_revision"):
            raise ValueError("interaction transfer checkpoint revision is stale")
        coefficients = tuple(float(item) for item in payload.get("coefficients", ()))
        residual_rmse = float(payload.get("residual_rmse", 0.0))
        if coefficients != learner._coefficients or not math.isclose(
            residual_rmse, learner._residual_rmse, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("interaction transfer checkpoint model state is stale")
        return learner

    def _bind_lineage(self, source_trace_digest: str, checkpoint_revision: int) -> None:
        _digest_text(source_trace_digest, "interaction transfer source_trace_digest")
        if int(checkpoint_revision) < 0:
            raise ValueError("interaction transfer checkpoint_revision cannot be negative")
        if self._source_trace_digest is None:
            self._source_trace_digest = source_trace_digest
            self._checkpoint_revision = int(checkpoint_revision)
        elif (
            source_trace_digest != self._source_trace_digest
            or int(checkpoint_revision) != self._checkpoint_revision
        ):
            raise ValueError("interaction transfer evidence crosses trace or checkpoint lineage")

    def _pair_features(self, members: Sequence[str]) -> tuple[float, ...]:
        if len(members) != 2:
            raise ValueError("interaction transfer relation currently supports pairs only")
        first = self._profiles[members[0]]
        second = self._profiles[members[1]]
        first_value = float(first.contribution)
        second_value = float(second.contribution)
        return (
            1.0,
            (first_value + second_value) / 2.0,
            first_value * second_value,
        )

    def _fit(self) -> None:
        if not self._records:
            self._coefficients = ()
            self._residual_rmse = 0.0
            return
        rows = [self._pair_features(record.member_ids) for record in self._records]
        targets = [float(record.interaction) for record in self._records]
        self._coefficients = tuple(self._ridge_fit(rows, targets, self.ridge))
        errors = [
            float(target - sum(weight * feature for weight, feature in zip(self._coefficients, row)))
            for row, target in zip(rows, targets)
        ]
        self._residual_rmse = math.sqrt(sum(error * error for error in errors) / len(errors))

    @staticmethod
    def _ridge_fit(
        rows: Sequence[Sequence[float]], targets: Sequence[float], ridge: float
    ) -> list[float]:
        width = len(rows[0])
        normal = [[0.0 for _ in range(width)] for _ in range(width)]
        right = [0.0 for _ in range(width)]
        for row, target in zip(rows, targets):
            for left in range(width):
                right[left] += row[left] * target
                for column in range(width):
                    normal[left][column] += row[left] * row[column]
        for index in range(1, width):
            normal[index][index] += float(ridge)
        for pivot_index in range(width):
            pivot = max(
                range(pivot_index, width), key=lambda index: abs(normal[index][pivot_index])
            )
            if abs(normal[pivot][pivot_index]) < 1e-12:
                raise ValueError("interaction transfer relation matrix is singular")
            if pivot != pivot_index:
                normal[pivot_index], normal[pivot] = normal[pivot], normal[pivot_index]
                right[pivot_index], right[pivot] = right[pivot], right[pivot_index]
            divisor = normal[pivot_index][pivot_index]
            normal[pivot_index] = [value / divisor for value in normal[pivot_index]]
            right[pivot_index] /= divisor
            for row_index in range(width):
                if row_index == pivot_index:
                    continue
                factor = normal[row_index][pivot_index]
                if factor == 0.0:
                    continue
                normal[row_index] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(normal[row_index], normal[pivot_index])
                ]
                right[row_index] -= factor * right[pivot_index]
        return right


__all__ = [
    "INTERACTION_GROUP_TRANSFER_CHECKPOINT_FORMAT",
    "INTERACTION_GROUP_TRANSFER_MODEL_REVISION",
    "InteractionGroupMemberEvidence",
    "InteractionGroupTransferCandidate",
    "InteractionGroupTransferLearner",
    "build_member_evidence",
]

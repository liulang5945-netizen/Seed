"""Trace-grounded interaction groups for the Taiji native substrate.

This module owns an evidence evaluator, not a new executive.  It accepts
opaque owner identifiers from a versioned native trace and estimates pairwise
contribution, interaction, recovery effect, resource cost, and lesion impact.
No biological role names are accepted or inferred here.  The resulting state
is checkpointable and can be handed to an owner policy only after a later
admission step; this S0 evaluator never mutates a runtime policy.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

INTERACTION_TRACE_FORMAT = "taiji-interaction-trace-v1"
INTERACTION_GROUP_CHECKPOINT_FORMAT = "taiji-interaction-group-v1"
INTERACTION_GROUP_ESTIMATOR_REVISION = 1


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


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _digest_text(value: str, name: str) -> str:
    value = _text(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class InteractionTraceEvent:
    """One observed native interaction surface in an episode.

    ``owner_id`` is intentionally opaque.  It identifies the state owner that
    emitted the event, but it is not a semantic role or a neuron class.
    """

    event_id: str
    owner_id: str
    episode_id: str
    checkpoint_revision: int
    outcome_id: str
    resource_cost: float = 0.0
    version: int = 1

    def __post_init__(self) -> None:
        _text(self.event_id, "interaction event_id")
        _text(self.owner_id, "interaction owner_id")
        _text(self.episode_id, "interaction episode_id")
        _text(self.outcome_id, "interaction outcome_id")
        if int(self.checkpoint_revision) < 0:
            raise ValueError("interaction checkpoint_revision cannot be negative")
        _nonnegative(self.resource_cost, "interaction event resource_cost")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction trace event version: {self.version}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_TRACE_FORMAT,
            "version": self.version,
            "event_id": self.event_id,
            "owner_id": self.owner_id,
            "episode_id": self.episode_id,
            "checkpoint_revision": self.checkpoint_revision,
            "outcome_id": self.outcome_id,
            "resource_cost": self.resource_cost,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionTraceEvent:
        if payload.get("format", INTERACTION_TRACE_FORMAT) != INTERACTION_TRACE_FORMAT:
            raise ValueError("unsupported interaction trace event format")
        return cls(
            version=int(payload.get("version", 1)),
            event_id=str(payload["event_id"]),
            owner_id=str(payload["owner_id"]),
            episode_id=str(payload["episode_id"]),
            checkpoint_revision=int(payload["checkpoint_revision"]),
            outcome_id=str(payload["outcome_id"]),
            resource_cost=float(payload.get("resource_cost", 0.0)),
        )


def project_native_adapter_episode(
    adapter: Any,
    *,
    context_id: str,
    owner_id_by_event_id: Mapping[str, str | None],
    outcome: Any | None = None,
    recovery_effect: float = 0.0,
    resource_cost_by_event_id: Mapping[str, float] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> InteractionTraceEpisode:
    """Project one settled native adapter state into the trace contract.

    The adapter remains the source of truth for event and outcome identity.
    ``owner_id_by_event_id`` is a projection boundary: callers may omit
    non-candidate native events by mapping them to ``None``, but every emitted
    trace event keeps its actual native ``event_id``.  Owner IDs are therefore
    opaque evidence handles, not a semantic neuron taxonomy.

    ``checkpoint`` is optional for callers that already captured the atomic
    native checkpoint.  The envelope version is used as the trace revision so
    stale or mixed checkpoint formats cannot silently enter an evaluation.
    """

    if not hasattr(adapter, "cognitive_snapshot") or not hasattr(adapter, "native_checkpoint"):
        raise TypeError("adapter must expose cognitive_snapshot() and native_checkpoint()")
    state = adapter.cognitive_snapshot()
    native_checkpoint = checkpoint if checkpoint is not None else adapter.native_checkpoint()
    if not isinstance(native_checkpoint, Mapping):
        raise TypeError("native checkpoint must be a mapping")
    if native_checkpoint.get("format") != "taiji-native-v1":
        raise ValueError("native adapter trace requires taiji-native-v1 checkpoint format")
    checkpoint_revision = int(native_checkpoint.get("version", -1))
    if checkpoint_revision < 0:
        raise ValueError("native checkpoint version must be non-negative")
    projected_outcome = outcome if outcome is not None else state.outcome
    if projected_outcome is None:
        raise RuntimeError("native adapter trace projection requires a settled Outcome")

    episode_id = _text(str(state.episode_id), "native projected episode_id")
    outcome_id = (
        f"{episode_id}:outcome:{int(projected_outcome.tick)}:{str(projected_outcome.intent_id)}"
    )
    native_events = {str(event.event_id): event for event in state.events}
    unknown_event_ids = set(owner_id_by_event_id) - set(native_events)
    if unknown_event_ids:
        raise ValueError(
            "native trace owner mapping references unknown event IDs: "
            + ", ".join(sorted(unknown_event_ids))
        )
    costs = resource_cost_by_event_id or {}
    unknown_cost_ids = set(costs) - set(native_events)
    if unknown_cost_ids:
        raise ValueError(
            "native trace resource mapping references unknown event IDs: "
            + ", ".join(sorted(unknown_cost_ids))
        )

    events: list[InteractionTraceEvent] = []
    for event_id in sorted(owner_id_by_event_id):
        owner_id = owner_id_by_event_id[event_id]
        if owner_id is None:
            continue
        event = native_events[event_id]
        resource_cost = float(
            costs.get(event_id, max(1, int(event.end_tick - event.start_tick + 1)))
        )
        events.append(
            InteractionTraceEvent(
                event_id=event.event_id,
                owner_id=_text(str(owner_id), "native projected owner_id"),
                episode_id=episode_id,
                checkpoint_revision=checkpoint_revision,
                outcome_id=outcome_id,
                resource_cost=resource_cost,
            )
        )
    return InteractionTraceEpisode(
        episode_id=episode_id,
        checkpoint_revision=checkpoint_revision,
        outcome_id=outcome_id,
        events=tuple(events),
        outcome=float(projected_outcome.reward),
        recovery_effect=float(recovery_effect),
        context_id=_text(str(context_id), "native projected context_id"),
    )


@dataclass(frozen=True)
class InteractionTraceEpisode:
    """A factorially observed native episode and its experienced outcome."""

    episode_id: str
    checkpoint_revision: int
    outcome_id: str
    events: tuple[InteractionTraceEvent, ...]
    outcome: float
    recovery_effect: float = 0.0
    context_id: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        _text(self.episode_id, "interaction episode_id")
        _text(self.outcome_id, "interaction outcome_id")
        if int(self.checkpoint_revision) < 0:
            raise ValueError("interaction episode checkpoint_revision cannot be negative")
        event_ids = tuple(event.event_id for event in self.events)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("interaction episode event_ids must be unique")
        for event in self.events:
            if not isinstance(event, InteractionTraceEvent):
                raise TypeError("interaction episode events must be InteractionTraceEvent values")
            if (
                event.episode_id != self.episode_id
                or int(event.checkpoint_revision) != int(self.checkpoint_revision)
                or event.outcome_id != self.outcome_id
            ):
                raise ValueError("interaction event binding does not match its episode")
        _finite(self.outcome, "interaction outcome")
        _finite(self.recovery_effect, "interaction recovery_effect")
        _text(self.context_id, "interaction context_id")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction episode version: {self.version}")

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(sorted({event.owner_id for event in self.events}))

    @property
    def resource_cost(self) -> float:
        return float(sum(event.resource_cost for event in self.events))

    def to_payload(self, *, include_outcome: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": INTERACTION_TRACE_FORMAT,
            "version": self.version,
            "episode_id": self.episode_id,
            "checkpoint_revision": self.checkpoint_revision,
            "outcome_id": self.outcome_id,
            "events": [event.to_payload() for event in self.events],
            "context_id": self.context_id,
        }
        if include_outcome:
            payload["outcome"] = self.outcome
            payload["recovery_effect"] = self.recovery_effect
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionTraceEpisode:
        if payload.get("format", INTERACTION_TRACE_FORMAT) != INTERACTION_TRACE_FORMAT:
            raise ValueError("unsupported interaction trace episode format")
        return cls(
            version=int(payload.get("version", 1)),
            episode_id=str(payload["episode_id"]),
            checkpoint_revision=int(payload["checkpoint_revision"]),
            outcome_id=str(payload["outcome_id"]),
            events=tuple(
                InteractionTraceEvent.from_payload(item) for item in payload.get("events", ())
            ),
            outcome=float(payload.get("outcome", 0.0)),
            recovery_effect=float(payload.get("recovery_effect", 0.0)),
            context_id=str(payload.get("context_id", "")),
        )


@dataclass(frozen=True)
class InteractionTraceCorpus:
    """Train and independent holdout trace splits.

    The train digest deliberately excludes outcome values from the holdout
    split.  This makes it impossible for a holdout result to silently become
    part of the learned group identity.
    """

    train: tuple[InteractionTraceEpisode, ...]
    holdout: tuple[InteractionTraceEpisode, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if not self.train or not self.holdout:
            raise ValueError("interaction trace train and holdout cannot be empty")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction trace corpus version: {self.version}")
        train_ids = {item.episode_id for item in self.train}
        holdout_ids = {item.episode_id for item in self.holdout}
        if len(train_ids) != len(self.train) or len(holdout_ids) != len(self.holdout):
            raise ValueError("interaction episode ids must be unique within a split")
        if train_ids & holdout_ids:
            raise ValueError("interaction train and holdout episodes must be disjoint")
        for episode in (*self.train, *self.holdout):
            if not isinstance(episode, InteractionTraceEpisode):
                raise TypeError("interaction corpus entries must be InteractionTraceEpisode values")

    @property
    def train_checkpoint_revisions(self) -> frozenset[int]:
        return frozenset(int(item.checkpoint_revision) for item in self.train)

    @property
    def train_trace_digest(self) -> str:
        payload = [
            item.to_payload(include_outcome=False)
            for item in sorted(self.train, key=lambda value: value.episode_id)
        ]
        return _digest({"format": INTERACTION_TRACE_FORMAT, "split": "train", "episodes": payload})

    @property
    def holdout_trace_digest(self) -> str:
        payload = [
            item.to_payload(include_outcome=False)
            for item in sorted(self.holdout, key=lambda value: value.episode_id)
        ]
        return _digest(
            {"format": INTERACTION_TRACE_FORMAT, "split": "holdout", "episodes": payload}
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_TRACE_FORMAT,
            "version": self.version,
            "train": [item.to_payload() for item in self.train],
            "holdout": [item.to_payload() for item in self.holdout],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionTraceCorpus:
        if payload.get("format") != INTERACTION_TRACE_FORMAT:
            raise ValueError("unsupported interaction trace corpus format")
        return cls(
            version=int(payload.get("version", 1)),
            train=tuple(
                InteractionTraceEpisode.from_payload(item) for item in payload.get("train", ())
            ),
            holdout=tuple(
                InteractionTraceEpisode.from_payload(item) for item in payload.get("holdout", ())
            ),
        )


@dataclass(frozen=True)
class InteractionGroupEvaluatorConfig:
    """Resource and confidence budget for deterministic pairwise S0 evidence."""

    estimator_revision: int = INTERACTION_GROUP_ESTIMATOR_REVISION
    minimum_interaction: float = 0.1
    maximum_uncertainty: float = 0.12
    maximum_group_cardinality: int = 2
    maximum_pairwise_candidates: int = 32
    maximum_resource_cost: float = 10.0

    def __post_init__(self) -> None:
        if int(self.estimator_revision) <= 0:
            raise ValueError("interaction estimator_revision must be positive")
        if _finite(self.minimum_interaction, "interaction minimum_interaction") < 0.0:
            raise ValueError("interaction minimum_interaction cannot be negative")
        if _nonnegative(self.maximum_uncertainty, "interaction maximum_uncertainty") < 0.0:
            raise ValueError("interaction maximum_uncertainty cannot be negative")
        if int(self.maximum_group_cardinality) != 2:
            raise ValueError("S0 interaction evaluator only supports pair groups")
        if int(self.maximum_pairwise_candidates) <= 0:
            raise ValueError("interaction maximum_pairwise_candidates must be positive")
        _nonnegative(self.maximum_resource_cost, "interaction maximum_resource_cost")


@dataclass(frozen=True)
class InteractionGroupRecord:
    """A trace-bound, signed attribution record for one candidate group."""

    group_id: str
    member_ids: tuple[str, ...]
    source_trace_digest: str
    checkpoint_revision: int
    contribution: float
    interaction: float
    uncertainty: float
    resource_cost: float
    owner_policy: str
    recovery_effect: float = 0.0
    holdout_interaction: float | None = None
    holdout_recovery_effect: float | None = None
    status: str = "admitted"
    method: str = "factorial-counterfactual+lesion"
    event_ids: tuple[str, ...] = ()
    outcome_ids: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        _text(self.group_id, "interaction group_id")
        if len(self.member_ids) < 2:
            raise ValueError("interaction group needs at least two members")
        if tuple(sorted(set(self.member_ids))) != tuple(self.member_ids):
            raise ValueError("interaction group member_ids must be unique and sorted")
        _digest_text(self.source_trace_digest, "interaction source_trace_digest")
        if int(self.checkpoint_revision) < 0:
            raise ValueError("interaction group checkpoint_revision cannot be negative")
        _finite(self.contribution, "interaction group contribution")
        _finite(self.interaction, "interaction group interaction")
        _nonnegative(self.uncertainty, "interaction group uncertainty")
        _nonnegative(self.resource_cost, "interaction group resource_cost")
        _text(self.owner_policy, "interaction group owner_policy")
        _finite(self.recovery_effect, "interaction group recovery_effect")
        if self.holdout_interaction is not None:
            _finite(self.holdout_interaction, "interaction group holdout_interaction")
        if self.holdout_recovery_effect is not None:
            _finite(self.holdout_recovery_effect, "interaction group holdout_recovery_effect")
        if self.status not in {"candidate", "admitted", "rejected", "rolled_back"}:
            raise ValueError("unsupported interaction group status")
        _text(self.method, "interaction group method")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("interaction group event_ids must be unique")
        if len(set(self.outcome_ids)) != len(self.outcome_ids):
            raise ValueError("interaction group outcome_ids must be unique")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction group version: {self.version}")

    @property
    def interaction_kind(self) -> str:
        if self.interaction > 0.0:
            return "complementary"
        if self.interaction < 0.0:
            return "conflicting"
        return "neutral"

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_GROUP_CHECKPOINT_FORMAT,
            "version": self.version,
            "group_id": self.group_id,
            "member_ids": list(self.member_ids),
            "source_trace_digest": self.source_trace_digest,
            "checkpoint_revision": self.checkpoint_revision,
            "contribution": self.contribution,
            "interaction": self.interaction,
            "uncertainty": self.uncertainty,
            "resource_cost": self.resource_cost,
            "owner_policy": self.owner_policy,
            "recovery_effect": self.recovery_effect,
            "holdout_interaction": self.holdout_interaction,
            "holdout_recovery_effect": self.holdout_recovery_effect,
            "status": self.status,
            "method": self.method,
            "event_ids": list(self.event_ids),
            "outcome_ids": list(self.outcome_ids),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionGroupRecord:
        if payload.get("format", INTERACTION_GROUP_CHECKPOINT_FORMAT) != (
            INTERACTION_GROUP_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported interaction group checkpoint format")
        return cls(
            version=int(payload.get("version", 1)),
            group_id=str(payload["group_id"]),
            member_ids=tuple(str(item) for item in payload.get("member_ids", ())),
            source_trace_digest=str(payload["source_trace_digest"]),
            checkpoint_revision=int(payload["checkpoint_revision"]),
            contribution=float(payload.get("contribution", 0.0)),
            interaction=float(payload.get("interaction", 0.0)),
            uncertainty=float(payload.get("uncertainty", 0.0)),
            resource_cost=float(payload.get("resource_cost", 0.0)),
            owner_policy=str(payload["owner_policy"]),
            recovery_effect=float(payload.get("recovery_effect", 0.0)),
            holdout_interaction=(
                None
                if payload.get("holdout_interaction") is None
                else float(payload["holdout_interaction"])
            ),
            holdout_recovery_effect=(
                None
                if payload.get("holdout_recovery_effect") is None
                else float(payload["holdout_recovery_effect"])
            ),
            status=str(payload.get("status", "admitted")),
            method=str(payload.get("method", "factorial-counterfactual+lesion")),
            event_ids=tuple(str(item) for item in payload.get("event_ids", ())),
            outcome_ids=tuple(str(item) for item in payload.get("outcome_ids", ())),
        )


@dataclass(frozen=True)
class InteractionGroupTombstone:
    """A rejected or rolled-back candidate retained for provenance."""

    candidate_id: str
    member_ids: tuple[str, ...]
    source_trace_digest: str
    checkpoint_revision: int
    reason: str
    version: int = 1

    def __post_init__(self) -> None:
        _text(self.candidate_id, "interaction tombstone candidate_id")
        if len(self.member_ids) < 2 or tuple(sorted(set(self.member_ids))) != self.member_ids:
            raise ValueError("interaction tombstone member_ids must be sorted and unique")
        _digest_text(self.source_trace_digest, "interaction tombstone source_trace_digest")
        if int(self.checkpoint_revision) < 0:
            raise ValueError("interaction tombstone checkpoint_revision cannot be negative")
        _text(self.reason, "interaction tombstone reason")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction tombstone version: {self.version}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_GROUP_CHECKPOINT_FORMAT,
            "version": self.version,
            "candidate_id": self.candidate_id,
            "member_ids": list(self.member_ids),
            "source_trace_digest": self.source_trace_digest,
            "checkpoint_revision": self.checkpoint_revision,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionGroupTombstone:
        if payload.get("format", INTERACTION_GROUP_CHECKPOINT_FORMAT) != (
            INTERACTION_GROUP_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported interaction tombstone checkpoint format")
        return cls(
            version=int(payload.get("version", 1)),
            candidate_id=str(payload["candidate_id"]),
            member_ids=tuple(str(item) for item in payload.get("member_ids", ())),
            source_trace_digest=str(payload["source_trace_digest"]),
            checkpoint_revision=int(payload["checkpoint_revision"]),
            reason=str(payload["reason"]),
        )


@dataclass(frozen=True)
class InteractionGroupState:
    """Checkpoint state for admitted groups and rejected candidates."""

    checkpoint_revision: int
    source_trace_digest: str
    estimator_revision: int
    groups: tuple[InteractionGroupRecord, ...] = ()
    rejected_candidates: tuple[InteractionGroupTombstone, ...] = ()
    owner_policy_lineage: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if int(self.checkpoint_revision) < 0:
            raise ValueError("interaction state checkpoint_revision cannot be negative")
        _digest_text(self.source_trace_digest, "interaction state source_trace_digest")
        if int(self.estimator_revision) <= 0:
            raise ValueError("interaction state estimator_revision must be positive")
        group_ids = tuple(item.group_id for item in self.groups)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("interaction state group_ids must be unique")
        for group in self.groups:
            if not isinstance(group, InteractionGroupRecord):
                raise TypeError("interaction state groups must be InteractionGroupRecord values")
            if group.source_trace_digest != self.source_trace_digest or int(
                group.checkpoint_revision
            ) != int(self.checkpoint_revision):
                raise ValueError("interaction group is bound to a different trace revision")
        tombstone_ids = tuple(item.candidate_id for item in self.rejected_candidates)
        if len(set(tombstone_ids)) != len(tombstone_ids):
            raise ValueError("interaction state tombstone ids must be unique")
        for tombstone in self.rejected_candidates:
            if not isinstance(tombstone, InteractionGroupTombstone):
                raise TypeError(
                    "interaction state rejected candidates must be InteractionGroupTombstone values"
                )
            if tombstone.source_trace_digest != self.source_trace_digest or int(
                tombstone.checkpoint_revision
            ) != int(self.checkpoint_revision):
                raise ValueError("interaction tombstone is bound to a different trace revision")
        if any(not str(item) for item in self.owner_policy_lineage):
            raise ValueError("interaction state owner_policy_lineage cannot contain empty ids")
        if tuple(item.owner_policy for item in self.groups) != self.owner_policy_lineage:
            raise ValueError("interaction state owner_policy_lineage is stale")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction state version: {self.version}")

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_GROUP_CHECKPOINT_FORMAT,
            "version": self.version,
            "checkpoint_revision": self.checkpoint_revision,
            "source_trace_digest": self.source_trace_digest,
            "estimator_revision": self.estimator_revision,
            "groups": [item.to_payload() for item in self.groups],
            "rejected_candidates": [item.to_payload() for item in self.rejected_candidates],
            "owner_policy_lineage": list(self.owner_policy_lineage),
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> InteractionGroupState:
        if payload.get("format") != INTERACTION_GROUP_CHECKPOINT_FORMAT:
            raise ValueError("unsupported interaction group state format")
        return cls(
            version=int(payload.get("version", 1)),
            checkpoint_revision=int(payload["checkpoint_revision"]),
            source_trace_digest=str(payload["source_trace_digest"]),
            estimator_revision=int(payload["estimator_revision"]),
            groups=tuple(
                InteractionGroupRecord.from_payload(item) for item in payload.get("groups", ())
            ),
            rejected_candidates=tuple(
                InteractionGroupTombstone.from_payload(item)
                for item in payload.get("rejected_candidates", ())
            ),
            owner_policy_lineage=tuple(
                str(item) for item in payload.get("owner_policy_lineage", ())
            ),
        )


@dataclass(frozen=True)
class InteractionGroupEvaluation:
    """Deterministic S0 output plus an auditable event trace."""

    state: InteractionGroupState
    holdout_trace_digest: str
    events: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]

    @property
    def source_trace_digest(self) -> str:
        return self.state.source_trace_digest

    @property
    def passed(self) -> bool:
        return bool(self.metrics.get("gate_passed", False))

    def to_report(self) -> dict[str, Any]:
        return {
            "format": "taiji-interaction-group-evaluation-v1",
            "estimator_revision": self.state.estimator_revision,
            "source_trace_digest": self.state.source_trace_digest,
            "holdout_trace_digest": self.holdout_trace_digest,
            "checkpoint": self.state.checkpoint(),
            "events": [dict(item) for item in self.events],
            "metrics": dict(self.metrics),
            "gate": {
                "passed": self.passed,
                "criterion": (
                    "trace-bound factorial counterfactuals must recover a complementary pair "
                    "and a conflicting pair on holdout, show member/group lesion effects, "
                    "preserve ownership and evidence through checkpoint, and reject role-label "
                    "or holdout-leakage shortcuts"
                ),
            },
        }


class InteractionGroupEvaluator:
    """Estimate bounded pair groups from observed factorial trace cells."""

    def __init__(self, config: InteractionGroupEvaluatorConfig | None = None) -> None:
        self.config = config or InteractionGroupEvaluatorConfig()

    def evaluate(self, corpus: InteractionTraceCorpus) -> InteractionGroupEvaluation:
        if len(corpus.train_checkpoint_revisions) != 1:
            return self._failed_evaluation(corpus, "mixed_checkpoint_revision")
        revision = next(iter(corpus.train_checkpoint_revisions))
        member_ids = tuple(
            sorted({member for episode in corpus.train for member in episode.member_ids})
        )
        pairs = tuple(itertools.combinations(member_ids, 2))
        if len(pairs) > int(self.config.maximum_pairwise_candidates):
            return self._failed_evaluation(corpus, "pairwise_budget_exceeded")

        groups: list[InteractionGroupRecord] = []
        rejected: list[InteractionGroupTombstone] = []
        events: list[Mapping[str, Any]] = []
        for members in pairs:
            group_id = self._group_id(
                members,
                corpus.train_trace_digest,
                revision,
                int(self.config.estimator_revision),
            )
            events.append(
                {
                    "event": "group_candidate_created",
                    "group_id": group_id,
                    "member_ids": list(members),
                    "source_trace_digest": corpus.train_trace_digest,
                    "checkpoint_revision": revision,
                }
            )
            estimate = self._estimate_pair(corpus.train, members)
            if estimate is None:
                rejected.append(
                    self._tombstone(
                        group_id, members, corpus.train_trace_digest, revision, "insufficient_trace"
                    )
                )
                events.append(
                    {
                        "event": "group_rejected",
                        "group_id": group_id,
                        "reason": "insufficient_trace",
                    }
                )
                continue
            events.append(
                {
                    "event": "counterfactual_evaluated",
                    "group_id": group_id,
                    "cells": estimate["cells"],
                    "source_trace_digest": corpus.train_trace_digest,
                }
            )
            holdout = self._estimate_pair(corpus.holdout, members)
            if holdout is None:
                rejected.append(
                    self._tombstone(
                        group_id,
                        members,
                        corpus.train_trace_digest,
                        revision,
                        "holdout_insufficient_trace",
                    )
                )
                events.append(
                    {
                        "event": "group_rejected",
                        "group_id": group_id,
                        "reason": "holdout_insufficient_trace",
                    }
                )
                continue
            events.append(
                {
                    "event": "lesion_evaluated",
                    "group_id": group_id,
                    "group_effect": estimate["pair_contribution"],
                    "remove_first_member_effect": estimate["pair_contribution"]
                    - estimate["first_contribution"],
                    "remove_second_member_effect": estimate["pair_contribution"]
                    - estimate["second_contribution"],
                }
            )
            reason = self._rejection_reason(estimate, holdout)
            if reason is not None:
                rejected.append(
                    self._tombstone(group_id, members, corpus.train_trace_digest, revision, reason)
                )
                events.append({"event": "group_rejected", "group_id": group_id, "reason": reason})
                continue
            record = InteractionGroupRecord(
                group_id=group_id,
                member_ids=members,
                source_trace_digest=corpus.train_trace_digest,
                checkpoint_revision=revision,
                contribution=estimate["pair_contribution"],
                interaction=estimate["interaction"],
                uncertainty=estimate["uncertainty"],
                resource_cost=estimate["pair_resource_cost"],
                owner_policy=f"policy:{group_id}",
                recovery_effect=estimate["recovery_interaction"],
                holdout_interaction=holdout["interaction"],
                holdout_recovery_effect=holdout["recovery_interaction"],
                event_ids=estimate["event_ids"],
                outcome_ids=estimate["outcome_ids"],
            )
            groups.append(record)
            events.append(
                {
                    "event": "group_admitted",
                    "group_id": group_id,
                    "interaction": record.interaction,
                    "holdout_interaction": record.holdout_interaction,
                    "owner_policy": record.owner_policy,
                }
            )

        state = InteractionGroupState(
            checkpoint_revision=revision,
            source_trace_digest=corpus.train_trace_digest,
            estimator_revision=int(self.config.estimator_revision),
            groups=tuple(groups),
            rejected_candidates=tuple(rejected),
            owner_policy_lineage=tuple(item.owner_policy for item in groups),
        )
        restored = InteractionGroupState.from_checkpoint(state.checkpoint())
        checkpoint_roundtrip = restored == state
        positive = next((item for item in groups if item.interaction > 0.0), None)
        negative = next((item for item in groups if item.interaction < 0.0), None)
        holdout_direction = bool(
            positive is not None
            and negative is not None
            and positive.holdout_interaction is not None
            and negative.holdout_interaction is not None
            and positive.holdout_interaction > 0.0
            and negative.holdout_interaction < 0.0
        )
        metrics = {
            "train_group_count": len(groups),
            "rejected_candidate_count": len(rejected),
            "complementary_group": None if positive is None else positive.group_id,
            "conflicting_group": None if negative is None else negative.group_id,
            "holdout_direction_preserved": holdout_direction,
            "checkpoint_roundtrip": checkpoint_roundtrip,
            "checkpoint_owner_lineage_preserved": restored.owner_policy_lineage
            == state.owner_policy_lineage,
            "lesion_effects_observed": all(abs(float(item.contribution)) > 0.0 for item in groups),
            "source_digest_excludes_holdout_outcome": (
                corpus.train_trace_digest != corpus.holdout_trace_digest
            ),
            "role_label_input_count": 0,
            "gate_passed": bool(
                positive is not None
                and negative is not None
                and holdout_direction
                and checkpoint_roundtrip
                and restored.owner_policy_lineage == state.owner_policy_lineage
                and all(abs(float(item.contribution)) > 0.0 for item in groups)
                and corpus.train_trace_digest != corpus.holdout_trace_digest
            ),
        }
        return InteractionGroupEvaluation(
            state=state,
            holdout_trace_digest=corpus.holdout_trace_digest,
            events=tuple(events),
            metrics=metrics,
        )

    def train_only_candidates(
        self, corpus: InteractionTraceCorpus
    ) -> tuple[InteractionGroupRecord, ...]:
        """Return pair candidates estimated from train traces only.

        This is intentionally separate from :meth:`evaluate`: the latter is a
        train-plus-holdout attribution Gate, while a learner must select from
        evidence that existed before holdout evaluation.  Candidates returned
        here are not admitted and carry no holdout-derived field.
        """

        if len(corpus.train_checkpoint_revisions) != 1:
            raise ValueError("train-only interaction candidates require one checkpoint revision")
        revision = next(iter(corpus.train_checkpoint_revisions))
        member_ids = tuple(
            sorted({member for episode in corpus.train for member in episode.member_ids})
        )
        pairs = tuple(itertools.combinations(member_ids, 2))
        if len(pairs) > int(self.config.maximum_pairwise_candidates):
            raise ValueError("train-only interaction candidate budget exceeded")
        candidates: list[InteractionGroupRecord] = []
        for members in pairs:
            estimate = self._estimate_pair(corpus.train, members)
            if estimate is None or float(estimate["pair_resource_cost"]) > float(
                self.config.maximum_resource_cost
            ):
                continue
            group_id = self._group_id(
                members,
                corpus.train_trace_digest,
                revision,
                int(self.config.estimator_revision),
            )
            candidates.append(
                InteractionGroupRecord(
                    group_id=group_id,
                    member_ids=members,
                    source_trace_digest=corpus.train_trace_digest,
                    checkpoint_revision=revision,
                    contribution=float(estimate["pair_contribution"]),
                    interaction=float(estimate["interaction"]),
                    uncertainty=float(estimate["uncertainty"]),
                    resource_cost=float(estimate["pair_resource_cost"]),
                    owner_policy=f"policy:{group_id}",
                    recovery_effect=float(estimate["recovery_interaction"]),
                    event_ids=estimate["event_ids"],
                    outcome_ids=estimate["outcome_ids"],
                    status="candidate",
                    method="train-only-factorial-counterfactual",
                )
            )
        return tuple(candidates)

    def _failed_evaluation(
        self, corpus: InteractionTraceCorpus, reason: str
    ) -> InteractionGroupEvaluation:
        revision = max(corpus.train_checkpoint_revisions, default=0)
        state = InteractionGroupState(
            checkpoint_revision=revision,
            source_trace_digest=corpus.train_trace_digest,
            estimator_revision=int(self.config.estimator_revision),
            rejected_candidates=(),
        )
        return InteractionGroupEvaluation(
            state=state,
            holdout_trace_digest=corpus.holdout_trace_digest,
            events=({"event": "evaluation_failed", "reason": reason},),
            metrics={
                "train_group_count": 0,
                "rejected_candidate_count": 0,
                "failure_reason": reason,
                "checkpoint_roundtrip": InteractionGroupState.from_checkpoint(state.checkpoint())
                == state,
                "role_label_input_count": 0,
                "gate_passed": False,
            },
        )

    def _estimate_pair(
        self, episodes: Sequence[InteractionTraceEpisode], members: tuple[str, str]
    ) -> dict[str, Any] | None:
        contexts: dict[str, dict[tuple[bool, bool], list[InteractionTraceEpisode]]] = defaultdict(
            lambda: {
                (False, False): [],
                (True, False): [],
                (False, True): [],
                (True, True): [],
            }
        )
        for episode in episodes:
            active = set(episode.member_ids)
            contexts[episode.context_id][(members[0] in active, members[1] in active)].append(
                episode
            )
        usable_contexts = [
            cells
            for cells in contexts.values()
            if all(
                cells[key] for key in ((False, False), (True, False), (False, True), (True, True))
            )
        ]
        if not usable_contexts:
            return None

        estimates: list[dict[str, Any]] = []
        for cells in usable_contexts:
            means = {
                key: float(sum(item.outcome for item in values) / len(values))
                for key, values in cells.items()
            }
            recovery_means = {
                key: float(sum(item.recovery_effect for item in values) / len(values))
                for key, values in cells.items()
            }
            baseline = means[(False, False)]
            first = means[(True, False)] - baseline
            second = means[(False, True)] - baseline
            pair = means[(True, True)] - baseline
            interaction = pair - first - second
            recovery_interaction = (
                recovery_means[(True, True)]
                - recovery_means[(False, False)]
                - (recovery_means[(True, False)] - recovery_means[(False, False)])
                - (recovery_means[(False, True)] - recovery_means[(False, False)])
            )
            pair_episodes = cells[(True, True)]
            estimates.append(
                {
                    "first_contribution": first,
                    "second_contribution": second,
                    "pair_contribution": pair,
                    "interaction": interaction,
                    "recovery_interaction": recovery_interaction,
                    "pair_resource_cost": float(
                        sum(item.resource_cost for item in pair_episodes) / len(pair_episodes)
                    ),
                    "episodes": tuple(item for values in cells.values() for item in values),
                }
            )
        interaction_values = [float(item["interaction"]) for item in estimates]
        uncertainty = self._sample_uncertainty(interaction_values)
        used_episodes = tuple(episode for estimate in estimates for episode in estimate["episodes"])
        cell_counts = {
            str(key): sum(len(context[key]) for context in usable_contexts)
            for key in ((False, False), (True, False), (False, True), (True, True))
        }
        event_ids = tuple(
            sorted(event.event_id for episode in used_episodes for event in episode.events)
        )
        outcome_ids = tuple(sorted(episode.outcome_id for episode in used_episodes))
        return {
            "cells": cell_counts,
            "pair_contribution": self._mean(item["pair_contribution"] for item in estimates),
            "first_contribution": self._mean(item["first_contribution"] for item in estimates),
            "second_contribution": self._mean(item["second_contribution"] for item in estimates),
            "interaction": self._mean(interaction_values),
            "recovery_interaction": self._mean(item["recovery_interaction"] for item in estimates),
            "uncertainty": uncertainty,
            "pair_resource_cost": self._mean(item["pair_resource_cost"] for item in estimates),
            "event_ids": event_ids,
            "outcome_ids": outcome_ids,
        }

    def _rejection_reason(self, train: dict[str, Any], holdout: dict[str, Any]) -> str | None:
        if train["uncertainty"] > float(self.config.maximum_uncertainty):
            return "low_confidence"
        if train["pair_resource_cost"] > float(self.config.maximum_resource_cost):
            return "resource_pressure"
        if abs(float(train["interaction"])) < float(self.config.minimum_interaction):
            return "interaction_below_threshold"
        if float(train["interaction"]) * float(holdout["interaction"]) <= 0.0:
            return "holdout_direction_changed"
        return None

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values = tuple(float(value) for value in values)
        if not values:
            raise ValueError("interaction estimator cannot average an empty sequence")
        return float(sum(values) / len(values))

    @classmethod
    def _sample_uncertainty(cls, values: Sequence[float]) -> float:
        values = tuple(float(value) for value in values)
        if len(values) <= 1:
            return 0.0
        mean = cls._mean(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return float(math.sqrt(variance / len(values)))

    @staticmethod
    def _group_id(
        members: tuple[str, ...], trace_digest: str, revision: int, estimator_revision: int
    ) -> str:
        return (
            "group:"
            + _digest(
                {
                    "members": list(members),
                    "trace_digest": trace_digest,
                    "checkpoint_revision": int(revision),
                    "estimator_revision": int(estimator_revision),
                }
            )[:24]
        )

    @staticmethod
    def _tombstone(
        group_id: str,
        members: tuple[str, ...],
        trace_digest: str,
        revision: int,
        reason: str,
    ) -> InteractionGroupTombstone:
        return InteractionGroupTombstone(
            candidate_id=group_id,
            member_ids=members,
            source_trace_digest=trace_digest,
            checkpoint_revision=revision,
            reason=reason,
        )


__all__ = [
    "INTERACTION_GROUP_CHECKPOINT_FORMAT",
    "INTERACTION_GROUP_ESTIMATOR_REVISION",
    "INTERACTION_TRACE_FORMAT",
    "InteractionGroupEvaluation",
    "InteractionGroupEvaluator",
    "InteractionGroupEvaluatorConfig",
    "InteractionGroupRecord",
    "InteractionGroupState",
    "InteractionGroupTombstone",
    "InteractionTraceCorpus",
    "InteractionTraceEpisode",
    "InteractionTraceEvent",
    "project_native_adapter_episode",
]

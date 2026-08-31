"""Online Outcome writeback for trace-bound interaction-group transfer.

The transfer learner predicts a candidate; this module owns the boundary where
an actually experienced native ``Outcome`` may update that learner.  It keeps
the mutation explicit, content-addressed, checkpointable, and reversible.  A
failed, non-terminal, low-confidence, resource-invalid, stale, or holdout
feedback value is retained as an audit admission but never enters the train
lineage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import Outcome
from .interaction_group_transfer import (
    InteractionGroupTransferCandidate,
    InteractionGroupTransferLearner,
)
from .interaction_groups import InteractionGroupRecord, InteractionTraceEpisode

INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT = "taiji-interaction-group-online-v1"
INTERACTION_GROUP_ONLINE_MODEL_REVISION = 1


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(value: str, name: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _digest_text(value: str, name: str) -> str:
    value = _text(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
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


def _ids(values: Sequence[str], name: str, *, sorted_unique: bool = False) -> tuple[str, ...]:
    result = tuple(_text(str(value), name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    if sorted_unique and tuple(sorted(result)) != result:
        raise ValueError(f"{name} must be sorted")
    return result


@dataclass(frozen=True)
class InteractionGroupOutcomeFeedback:
    """One actual online Outcome eligible for learner admission.

    ``source_split`` is deliberately fixed to ``online``.  There is no
    holdout field in this contract, so an evaluation result cannot be silently
    replayed as training evidence.  The parent checkpoint binds the feedback
    to the candidate that was selected before the environment was run.
    """

    feedback_id: str
    candidate_id: str
    member_ids: tuple[str, ...]
    source_trace_digest: str
    checkpoint_revision: int
    parent_checkpoint_digest: str
    outcome: Outcome
    outcome_id: str
    event_ids: tuple[str, ...]
    realized_interaction: float
    contribution: float
    resource_cost: float
    recovery_effect: float = 0.0
    uncertainty: float = 0.0
    source_split: str = "online"
    version: int = 1

    def __post_init__(self) -> None:
        _digest_text(self.feedback_id, "online feedback_id")
        _text(self.candidate_id, "online candidate_id")
        members = _ids(self.member_ids, "online member_ids", sorted_unique=True)
        if len(members) < 2:
            raise ValueError("online feedback needs at least two members")
        _digest_text(self.source_trace_digest, "online source_trace_digest")
        if int(self.checkpoint_revision) < 0:
            raise ValueError("online checkpoint_revision cannot be negative")
        _digest_text(self.parent_checkpoint_digest, "online parent_checkpoint_digest")
        if not isinstance(self.outcome, Outcome):
            raise TypeError("online feedback outcome must be an Outcome")
        _text(self.outcome_id, "online outcome_id")
        _ids(self.event_ids, "online event_ids")
        _finite(self.realized_interaction, "online realized_interaction")
        _finite(self.contribution, "online contribution")
        _nonnegative(self.resource_cost, "online resource_cost")
        _finite(self.recovery_effect, "online recovery_effect")
        _nonnegative(self.uncertainty, "online uncertainty")
        if self.source_split != "online":
            raise ValueError("online feedback source_split must be exactly online")
        if int(self.version) != 1:
            raise ValueError(f"unsupported online feedback version: {self.version}")

    @classmethod
    def from_episode(
        cls,
        *,
        candidate: InteractionGroupTransferCandidate,
        parent_checkpoint_digest: str,
        episode: InteractionTraceEpisode,
        outcome: Outcome,
        realized_interaction: float,
        contribution: float,
        uncertainty: float = 0.0,
    ) -> InteractionGroupOutcomeFeedback:
        """Build feedback from a settled native trace and its actual Outcome."""

        if not isinstance(candidate, InteractionGroupTransferCandidate):
            raise TypeError("online feedback candidate must be a transfer candidate")
        if not isinstance(episode, InteractionTraceEpisode):
            raise TypeError("online feedback episode must be an InteractionTraceEpisode")
        if not isinstance(outcome, Outcome):
            raise TypeError("online feedback outcome must be an Outcome")
        if not math.isclose(float(episode.outcome), float(outcome.reward), abs_tol=1e-12):
            raise ValueError("online feedback Outcome reward does not match native trace")
        if episode.member_ids != candidate.member_ids:
            raise ValueError("online feedback native members do not match candidate members")
        if int(episode.checkpoint_revision) != int(candidate.checkpoint_revision):
            raise ValueError("online feedback native episode crosses candidate revision")
        if not episode.events:
            raise ValueError("online feedback requires native event evidence")
        payload = {
            "candidate_id": candidate.group_id,
            "member_ids": list(candidate.member_ids),
            "source_trace_digest": candidate.source_trace_digest,
            "checkpoint_revision": candidate.checkpoint_revision,
            "parent_checkpoint_digest": parent_checkpoint_digest,
            "outcome": outcome.to_payload(),
            "outcome_id": episode.outcome_id,
            "event_ids": [event.event_id for event in episode.events],
            "realized_interaction": float(realized_interaction),
            "contribution": float(contribution),
            "uncertainty": float(uncertainty),
        }
        return cls(
            feedback_id=_digest(payload),
            candidate_id=candidate.group_id,
            member_ids=candidate.member_ids,
            source_trace_digest=candidate.source_trace_digest,
            checkpoint_revision=candidate.checkpoint_revision,
            parent_checkpoint_digest=parent_checkpoint_digest,
            outcome=outcome,
            outcome_id=episode.outcome_id,
            event_ids=tuple(event.event_id for event in episode.events),
            realized_interaction=float(realized_interaction),
            contribution=float(contribution),
            resource_cost=float(episode.resource_cost),
            recovery_effect=float(episode.recovery_effect),
            uncertainty=float(uncertainty),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT,
            "version": self.version,
            "feedback_id": self.feedback_id,
            "candidate_id": self.candidate_id,
            "member_ids": list(self.member_ids),
            "source_trace_digest": self.source_trace_digest,
            "checkpoint_revision": self.checkpoint_revision,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "outcome": self.outcome.to_payload(),
            "outcome_id": self.outcome_id,
            "event_ids": list(self.event_ids),
            "realized_interaction": self.realized_interaction,
            "contribution": self.contribution,
            "resource_cost": self.resource_cost,
            "recovery_effect": self.recovery_effect,
            "uncertainty": self.uncertainty,
            "source_split": self.source_split,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionGroupOutcomeFeedback:
        if payload.get("format", INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT) != (
            INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported online feedback format")
        return cls(
            version=int(payload.get("version", 1)),
            feedback_id=str(payload["feedback_id"]),
            candidate_id=str(payload["candidate_id"]),
            member_ids=tuple(str(item) for item in payload.get("member_ids", ())),
            source_trace_digest=str(payload["source_trace_digest"]),
            checkpoint_revision=int(payload["checkpoint_revision"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            outcome=Outcome.from_payload(payload["outcome"]),
            outcome_id=str(payload["outcome_id"]),
            event_ids=tuple(str(item) for item in payload.get("event_ids", ())),
            realized_interaction=float(payload["realized_interaction"]),
            contribution=float(payload["contribution"]),
            resource_cost=float(payload["resource_cost"]),
            recovery_effect=float(payload.get("recovery_effect", 0.0)),
            uncertainty=float(payload.get("uncertainty", 0.0)),
            source_split=str(payload.get("source_split", "online")),
        )


@dataclass(frozen=True)
class InteractionGroupOnlineAdmission:
    """Audit result for an online feedback admission attempt."""

    feedback_id: str
    candidate_id: str
    member_ids: tuple[str, ...]
    outcome_id: str
    event_ids: tuple[str, ...]
    parent_checkpoint_digest: str
    post_learner_checkpoint_digest: str
    status: str
    reason: str
    version: int = 1

    def __post_init__(self) -> None:
        _digest_text(self.feedback_id, "online admission feedback_id")
        _text(self.candidate_id, "online admission candidate_id")
        if len(_ids(self.member_ids, "online admission member_ids", sorted_unique=True)) < 2:
            raise ValueError("online admission needs at least two members")
        _text(self.outcome_id, "online admission outcome_id")
        _ids(self.event_ids, "online admission event_ids")
        _digest_text(self.parent_checkpoint_digest, "online admission parent_checkpoint_digest")
        _digest_text(
            self.post_learner_checkpoint_digest,
            "online admission post_learner_checkpoint_digest",
        )
        if self.status not in {"applied", "rejected", "rolled_back"}:
            raise ValueError("unsupported online admission status")
        _text(self.reason, "online admission reason")
        if int(self.version) != 1:
            raise ValueError(f"unsupported online admission version: {self.version}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT,
            "version": self.version,
            "feedback_id": self.feedback_id,
            "candidate_id": self.candidate_id,
            "member_ids": list(self.member_ids),
            "outcome_id": self.outcome_id,
            "event_ids": list(self.event_ids),
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "post_learner_checkpoint_digest": self.post_learner_checkpoint_digest,
            "status": self.status,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InteractionGroupOnlineAdmission:
        if payload.get("format", INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT) != (
            INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT
        ):
            raise ValueError("unsupported online admission format")
        return cls(
            version=int(payload.get("version", 1)),
            feedback_id=str(payload["feedback_id"]),
            candidate_id=str(payload["candidate_id"]),
            member_ids=tuple(str(item) for item in payload.get("member_ids", ())),
            outcome_id=str(payload["outcome_id"]),
            event_ids=tuple(str(item) for item in payload.get("event_ids", ())),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            post_learner_checkpoint_digest=str(payload["post_learner_checkpoint_digest"]),
            status=str(payload["status"]),
            reason=str(payload["reason"]),
        )


class InteractionGroupOnlineLearner:
    """Gate actual Outcome writeback into a transfer learner.

    ``apply_feedback`` is the only mutating entry point.  It performs
    candidate, lineage, terminal-Outcome, resource, uncertainty, and duplicate
    checks before calling ``observe_records``.  ``rollback_to`` restores an
    earlier outer checkpoint and leaves a tombstone-like audit admission so a
    rolled-back trial cannot be silently re-applied.
    """

    def __init__(
        self,
        learner: InteractionGroupTransferLearner,
        *,
        minimum_interaction: float = 0.0,
        maximum_feedback_uncertainty: float = 1.0,
        maximum_resource_cost: float = 10.0,
    ) -> None:
        if not isinstance(learner, InteractionGroupTransferLearner):
            raise TypeError("online learner requires an InteractionGroupTransferLearner")
        if not math.isfinite(float(minimum_interaction)):
            raise ValueError("online minimum_interaction must be finite")
        if not 0.0 <= float(maximum_feedback_uncertainty):
            raise ValueError("online maximum_feedback_uncertainty cannot be negative")
        if not 0.0 <= float(maximum_resource_cost):
            raise ValueError("online maximum_resource_cost cannot be negative")
        self._learner = learner
        self.minimum_interaction = float(minimum_interaction)
        self.maximum_feedback_uncertainty = float(maximum_feedback_uncertainty)
        self.maximum_resource_cost = float(maximum_resource_cost)
        self._admissions: list[InteractionGroupOnlineAdmission] = []

    @property
    def learner(self) -> InteractionGroupTransferLearner:
        return self._learner

    @property
    def admissions(self) -> tuple[InteractionGroupOnlineAdmission, ...]:
        return tuple(self._admissions)

    @property
    def applied_feedback_ids(self) -> tuple[str, ...]:
        return tuple(
            item.feedback_id for item in self._admissions if item.status == "applied"
        )

    @property
    def blocked_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.candidate_id
                    for item in self._admissions
                    if item.status in {"rejected", "rolled_back"}
                }
            )
        )

    @property
    def blocked_member_sets(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            sorted(
                {
                    item.member_ids
                    for item in self._admissions
                    if item.status in {"rejected", "rolled_back"}
                }
            )
        )

    def select(
        self,
        candidate_member_sets: Sequence[Sequence[str]],
        *,
        resource_budget: float | None = None,
    ) -> tuple[Any, InteractionGroupTransferCandidate] | None:
        """Select a candidate not blocked by a prior rejected/rolled-back trial."""

        selected = self._learner.select(
            candidate_member_sets,
            resource_budget=resource_budget,
            unseen_only=True,
        )
        if selected is None or not self._candidate_is_blocked(selected[1]):
            return selected
        remaining = [
            members
            for members in candidate_member_sets
            if (
                (candidate := self._learner.candidate(members, allow_observed=False)) is not None
                and not self._candidate_is_blocked(candidate)
            )
        ]
        if not remaining:
            return None
        return self._learner.select(
            remaining,
            resource_budget=resource_budget,
            unseen_only=True,
        )

    def apply_feedback(
        self, feedback: InteractionGroupOutcomeFeedback
    ) -> InteractionGroupOnlineAdmission:
        """Admit one actual Outcome or retain a non-mutating rejection audit."""

        if not isinstance(feedback, InteractionGroupOutcomeFeedback):
            raise TypeError("online feedback must be InteractionGroupOutcomeFeedback")
        parent = self.checkpoint()
        parent_digest = str(parent["checkpoint_digest"])
        if feedback.parent_checkpoint_digest != parent_digest:
            raise ValueError("online feedback parent checkpoint is stale")
        if feedback.feedback_id in {item.feedback_id for item in self._admissions}:
            raise ValueError("online feedback is a duplicate")
        self._validate_lineage(feedback)
        candidate = self._learner.candidate(feedback.member_ids, allow_observed=False)
        rejection = self._rejection_reason(feedback, candidate)
        if rejection is not None:
            admission = self._audit(feedback, parent_digest, "rejected", rejection)
            self._admissions.append(admission)
            return admission
        assert candidate is not None
        if candidate.group_id != feedback.candidate_id:
            admission = self._audit(
                feedback,
                parent_digest,
                "rejected",
                "candidate_id_does_not_match_current_prediction",
            )
            self._admissions.append(admission)
            return admission
        record = InteractionGroupRecord(
            group_id=feedback.candidate_id,
            member_ids=feedback.member_ids,
            source_trace_digest=feedback.source_trace_digest,
            checkpoint_revision=feedback.checkpoint_revision,
            contribution=feedback.contribution,
            interaction=feedback.realized_interaction,
            uncertainty=feedback.uncertainty,
            resource_cost=feedback.resource_cost,
            owner_policy="online-outcome-admission",
            recovery_effect=feedback.recovery_effect,
            status="admitted",
            method="native-outcome-online-writeback",
            event_ids=feedback.event_ids,
            outcome_ids=(feedback.outcome_id,),
        )
        self._learner.observe_records((record,))
        post_digest = str(self._learner.checkpoint()["checkpoint_digest"])
        admission = self._audit(
            feedback,
            parent_digest,
            "applied",
            "native_terminal_outcome_admitted",
            post_learner_checkpoint_digest=post_digest,
        )
        self._admissions.append(admission)
        return admission

    def rollback_to(
        self,
        parent_checkpoint: Mapping[str, Any],
        *,
        feedback_id: str,
    ) -> InteractionGroupOnlineAdmission:
        """Rollback the latest applied feedback to its pre-trial checkpoint."""

        _digest_text(feedback_id, "online rollback feedback_id")
        expected_digest = _digest(
            {key: value for key, value in parent_checkpoint.items() if key != "checkpoint_digest"}
        )
        if str(parent_checkpoint.get("checkpoint_digest")) != expected_digest:
            raise ValueError("online rollback parent checkpoint digest mismatch")
        applied = [item for item in self._admissions if item.status == "applied"]
        if not applied or applied[-1].feedback_id != feedback_id:
            raise ValueError("online rollback must target the latest applied feedback")
        target = applied[-1]
        if target.parent_checkpoint_digest != expected_digest:
            raise ValueError("online rollback checkpoint does not own the applied feedback")
        restored = self.from_checkpoint(parent_checkpoint)
        self._learner = restored._learner
        self._admissions = list(restored._admissions)
        rolled_back = replace(
            target,
            status="rolled_back",
            reason="explicit_trial_rollback",
            post_learner_checkpoint_digest=str(self._learner.checkpoint()["checkpoint_digest"]),
        )
        self._admissions.append(rolled_back)
        return rolled_back

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT,
            "version": 1,
            "model_revision": INTERACTION_GROUP_ONLINE_MODEL_REVISION,
            "minimum_interaction": self.minimum_interaction,
            "maximum_feedback_uncertainty": self.maximum_feedback_uncertainty,
            "maximum_resource_cost": self.maximum_resource_cost,
            "learner": self._learner.checkpoint(),
            "admissions": [item.to_payload() for item in self._admissions],
        }
        payload["checkpoint_digest"] = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        return payload

    @classmethod
    def from_checkpoint(
        cls, payload: Mapping[str, Any]
    ) -> InteractionGroupOnlineLearner:
        if payload.get("format") != INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported online learner checkpoint format")
        if int(payload.get("model_revision", -1)) != INTERACTION_GROUP_ONLINE_MODEL_REVISION:
            raise ValueError("unsupported online learner model revision")
        expected_digest = _digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest")) != expected_digest:
            raise ValueError("online learner checkpoint digest mismatch")
        learner = cls(
            InteractionGroupTransferLearner.from_checkpoint(payload["learner"]),
            minimum_interaction=float(payload["minimum_interaction"]),
            maximum_feedback_uncertainty=float(payload["maximum_feedback_uncertainty"]),
            maximum_resource_cost=float(payload["maximum_resource_cost"]),
        )
        learner._admissions = [
            InteractionGroupOnlineAdmission.from_payload(item)
            for item in payload.get("admissions", ())
        ]
        feedback_ids = [item.feedback_id for item in learner._admissions]
        if len(set(feedback_ids)) != len(feedback_ids):
            raise ValueError("online learner checkpoint contains duplicate feedback IDs")
        return learner

    def _validate_lineage(self, feedback: InteractionGroupOutcomeFeedback) -> None:
        if feedback.source_trace_digest != self._learner.source_trace_digest:
            raise ValueError("online feedback crosses source trace lineage")
        if feedback.checkpoint_revision != self._learner.checkpoint_revision:
            raise ValueError("online feedback crosses checkpoint revision lineage")

    def _rejection_reason(
        self,
        feedback: InteractionGroupOutcomeFeedback,
        candidate: InteractionGroupTransferCandidate | None,
    ) -> str | None:
        if candidate is None:
            return "candidate_unknown_or_already_observed"
        if feedback.candidate_id in self.blocked_candidate_ids or tuple(
            feedback.member_ids
        ) in self.blocked_member_sets:
            return "candidate_blocked_after_prior_rejection_or_rollback"
        if not feedback.outcome.terminal:
            return "outcome_not_terminal"
        if feedback.outcome.success is not True:
            return "outcome_unsuccessful"
        if feedback.realized_interaction < self.minimum_interaction:
            return "interaction_below_admission_threshold"
        if feedback.uncertainty > self.maximum_feedback_uncertainty:
            return "feedback_uncertainty_exceeds_admission_threshold"
        if feedback.resource_cost > self.maximum_resource_cost:
            return "feedback_resource_cost_exceeds_admission_budget"
        if not feedback.event_ids:
            return "native_event_evidence_missing"
        return None

    def _candidate_is_blocked(self, candidate: InteractionGroupTransferCandidate) -> bool:
        return candidate.group_id in self.blocked_candidate_ids or candidate.member_ids in self.blocked_member_sets

    @staticmethod
    def _audit(
        feedback: InteractionGroupOutcomeFeedback,
        parent_digest: str,
        status: str,
        reason: str,
        *,
        post_learner_checkpoint_digest: str | None = None,
    ) -> InteractionGroupOnlineAdmission:
        return InteractionGroupOnlineAdmission(
            feedback_id=feedback.feedback_id,
            candidate_id=feedback.candidate_id,
            member_ids=feedback.member_ids,
            outcome_id=feedback.outcome_id,
            event_ids=feedback.event_ids,
            parent_checkpoint_digest=parent_digest,
            post_learner_checkpoint_digest=(
                parent_digest
                if post_learner_checkpoint_digest is None
                else post_learner_checkpoint_digest
            ),
            status=status,
            reason=reason,
        )


__all__ = [
    "INTERACTION_GROUP_ONLINE_CHECKPOINT_FORMAT",
    "INTERACTION_GROUP_ONLINE_MODEL_REVISION",
    "InteractionGroupOnlineAdmission",
    "InteractionGroupOnlineLearner",
    "InteractionGroupOutcomeFeedback",
]

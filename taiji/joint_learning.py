"""Read-only outcome credit routing for the N177 joint task candidate.

Routes identify eligible parameter owners; they never apply an update. Environment
authenticity remains the responsibility of a future executor-signed receipt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .internalization import content_digest
from .joint_task_state import JointTaskStateV1


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _feedback_id_for(
    joint_state: JointTaskStateV1,
    input_id: str,
    kind: str,
    target_sha256: str,
) -> str:
    identity = "\0".join(
        (
            joint_state.joint_bundle_id,
            joint_state.owner_id,
            joint_state.scope_id,
            joint_state.session_id,
            joint_state.task_id,
            input_id,
            kind,
            target_sha256,
        )
    )
    return "joint-feedback:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JointLanguageFeedbackV1:
    """An explicit text target bound to one input in the current joint task."""

    feedback_id: str
    input_id: str
    kind: str
    target_bytes: bytes
    target_sha256: str

    FORMAT = "taiji-joint-language-feedback-v1"
    KINDS = frozenset({"correction", "supervised_target"})

    def __post_init__(self) -> None:
        _required_text(self.feedback_id, "feedback_id")
        _required_text(self.input_id, "input_id")
        if self.kind not in self.KINDS:
            raise ValueError("joint language feedback kind is unsupported")
        if not isinstance(self.target_bytes, bytes) or not self.target_bytes:
            raise ValueError("joint language feedback requires nonempty target bytes")
        try:
            self.target_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("joint language feedback target must be valid UTF-8") from error
        digest = hashlib.sha256(self.target_bytes).hexdigest()
        if self.target_sha256 != digest:
            raise ValueError("joint language feedback target digest differs")

    @classmethod
    def bind(
        cls,
        joint_state: JointTaskStateV1,
        *,
        input_id: str,
        kind: str,
        target_text: str | bytes,
    ) -> JointLanguageFeedbackV1:
        joint_state._validate_live_owner()
        if not any(item.input_id == input_id for item in joint_state.input_bindings):
            raise ValueError("language feedback input is outside the current joint task")
        target_bytes = target_text.encode("utf-8") if isinstance(target_text, str) else target_text
        if not isinstance(target_bytes, bytes) or not target_bytes:
            raise ValueError("language feedback target must be nonempty text")
        target_sha256 = hashlib.sha256(target_bytes).hexdigest()
        feedback_id = _feedback_id_for(joint_state, input_id, kind, target_sha256)
        return cls(feedback_id, input_id, kind, target_bytes, target_sha256)

    def validate_for(self, joint_state: JointTaskStateV1) -> None:
        joint_state._validate_live_owner()
        if not any(item.input_id == self.input_id for item in joint_state.input_bindings):
            raise ValueError("language feedback input is outside the current joint task")
        if hashlib.sha256(self.target_bytes).hexdigest() != self.target_sha256:
            raise ValueError("language feedback target changed after binding")
        if self.feedback_id != _feedback_id_for(
            joint_state, self.input_id, self.kind, self.target_sha256
        ):
            raise ValueError("language feedback belongs to a different joint task identity")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "feedback_id": self.feedback_id,
            "input_id": self.input_id,
            "kind": self.kind,
            "target_hex": self.target_bytes.hex(),
            "target_sha256": self.target_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointLanguageFeedbackV1:
        required = {
            "format",
            "version",
            "feedback_id",
            "input_id",
            "kind",
            "target_hex",
            "target_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("joint language feedback schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported joint language feedback version")
        try:
            target_bytes = bytes.fromhex(str(payload["target_hex"]))
        except ValueError as error:
            raise ValueError("joint language feedback target encoding is invalid") from error
        return cls(
            feedback_id=str(payload["feedback_id"]),
            input_id=str(payload["input_id"]),
            kind=str(payload["kind"]),
            target_bytes=target_bytes,
            target_sha256=str(payload["target_sha256"]),
        )


@dataclass(frozen=True)
class JointCreditRouteV1:
    """A frozen state/parameter owner map with all parameter updates disabled."""

    joint_bundle_id: str
    owner_id: str
    scope_id: str
    session_id: str
    task_id: str
    joint_revision: int
    intent_id: str
    execution_id: str
    outcome_sha256: str
    outcome_trust: str
    state_owner: str
    eligible_receivers: tuple[str, ...]
    world_model_sha256: str | None = None
    world_prediction_sha256: str | None = None
    executive_sha256: str | None = None
    decision_sha256: str | None = None
    language_feedback_id: str | None = None
    language_feedback_sha256: str | None = None
    language_parameter_revision: int | None = None
    apply_updates: bool = False

    FORMAT = "taiji-joint-credit-route-v1"
    RECEIVERS = frozenset({"world_model", "executive", "language_candidate"})

    def __post_init__(self) -> None:
        for name in (
            "joint_bundle_id",
            "owner_id",
            "scope_id",
            "session_id",
            "task_id",
            "intent_id",
            "execution_id",
            "outcome_sha256",
            "outcome_trust",
            "state_owner",
        ):
            _required_text(getattr(self, name), name)
        if self.joint_revision < 0:
            raise ValueError("credit route joint revision cannot be negative")
        receivers = tuple(self.eligible_receivers)
        if len(set(receivers)) != len(receivers) or not set(receivers) <= self.RECEIVERS:
            raise ValueError("joint credit route has duplicate or unsupported receivers")
        object.__setattr__(self, "eligible_receivers", receivers)
        if self.apply_updates:
            raise ValueError("N177.3 credit routing is read-only")
        if self.outcome_trust != "caller_asserted_experienced_not_authenticated":
            raise ValueError("joint credit route cannot claim authenticated environment evidence")
        if self.state_owner != "CognitiveState":
            raise ValueError(
                "joint state feedback must return to the canonical CognitiveState owner"
            )
        if "world_model" in receivers and (
            self.world_model_sha256 is None or self.world_prediction_sha256 is None
        ):
            raise ValueError(
                "world-model credit requires its model and selected prediction digests"
            )
        if "executive" in receivers and (
            self.executive_sha256 is None or self.decision_sha256 is None
        ):
            raise ValueError("executive credit requires its model and decision snapshot digests")
        if "language_candidate" in receivers and (
            self.language_feedback_id is None or self.language_feedback_sha256 is None
        ):
            raise ValueError("language credit requires explicit text feedback")

    @classmethod
    def from_settled_outcome(
        cls,
        joint_state: JointTaskStateV1,
        *,
        language_feedback: JointLanguageFeedbackV1 | None = None,
    ) -> JointCreditRouteV1:
        joint_state._validate_live_owner()
        receipt = joint_state.action_receipt
        outcome = joint_state.cognitive_state.outcome
        intent = joint_state.cognitive_state.action_intent
        if (
            receipt is None
            or receipt.status != "settled"
            or intent is None
            or outcome is None
            or receipt.intent_id != intent.intent_id
            or outcome.intent_id != intent.intent_id
            or outcome.success is None
            or outcome.provenance != "experienced"
        ):
            raise ValueError(
                "credit routing requires a settled caller-supplied experienced outcome"
            )
        transition = joint_state.cognitive_state.world_transition
        result_digest = content_digest(
            {
                "outcome": outcome.to_payload(),
                "world_transition": None if transition is None else transition.to_payload(),
            }
        )
        if receipt.outcome_sha256 != result_digest:
            raise ValueError("settled outcome differs from its persisted execution receipt")
        if transition is not None:
            prediction = joint_state._validate_world_transition(
                joint_state.cognitive_state,
                receipt,
                intent,
                validate_before_state=False,
            )
            transition_outcome = transition.outcome.to_payload()
            cognitive_outcome = outcome.to_payload()
            transition_outcome.pop("intent_id")
            cognitive_outcome.pop("intent_id")
            if content_digest(transition_outcome) != content_digest(cognitive_outcome):
                raise ValueError("world transition and CognitiveState outcome disagree")

        trace = joint_state.decision_trace
        if trace is not None and trace.intent_id != intent.intent_id:
            raise ValueError("decision trace belongs to a different ActionIntent")
        receivers: list[str] = []
        world_digest = None
        executive_digest = None
        decision_digest = None
        prediction_digest = None
        if trace is not None:
            receivers.append("executive")
            executive_digest = trace.executive_sha256
            decision_digest = trace.decision_sha256
            if transition is not None and transition.action.provenance == "experienced":
                receivers.append("world_model")
                world_digest = trace.world_model_sha256
                if prediction is None:
                    raise ValueError("world-model credit lacks its selected prediction")
                prediction_digest = content_digest(prediction.to_payload())

        feedback_id = None
        feedback_digest = None
        if language_feedback is not None:
            language_feedback.validate_for(joint_state)
            receivers.append("language_candidate")
            feedback_id = language_feedback.feedback_id
            feedback_digest = language_feedback.target_sha256

        return cls(
            joint_bundle_id=joint_state.joint_bundle_id,
            owner_id=joint_state.owner_id,
            scope_id=joint_state.scope_id,
            session_id=joint_state.session_id,
            task_id=joint_state.task_id,
            joint_revision=joint_state.joint_revision,
            intent_id=intent.intent_id,
            execution_id=receipt.execution_id or "",
            outcome_sha256=receipt.outcome_sha256 or "",
            outcome_trust="caller_asserted_experienced_not_authenticated",
            state_owner="CognitiveState",
            eligible_receivers=tuple(receivers),
            world_model_sha256=world_digest,
            world_prediction_sha256=prediction_digest,
            executive_sha256=executive_digest,
            decision_sha256=decision_digest,
            language_feedback_id=feedback_id,
            language_feedback_sha256=feedback_digest,
            language_parameter_revision=(
                joint_state.language_state.parameter_revision
                if "language_candidate" in receivers
                else None
            ),
            apply_updates=False,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "format": self.FORMAT,
            "version": 1,
            "joint_bundle_id": self.joint_bundle_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "joint_revision": self.joint_revision,
            "intent_id": self.intent_id,
            "execution_id": self.execution_id,
            "outcome_sha256": self.outcome_sha256,
            "outcome_trust": self.outcome_trust,
            "state_owner": self.state_owner,
            "eligible_receivers": list(self.eligible_receivers),
            "world_model_sha256": self.world_model_sha256,
            "world_prediction_sha256": self.world_prediction_sha256,
            "executive_sha256": self.executive_sha256,
            "decision_sha256": self.decision_sha256,
            "language_feedback_id": self.language_feedback_id,
            "language_feedback_sha256": self.language_feedback_sha256,
            "language_parameter_revision": self.language_parameter_revision,
            "apply_updates": self.apply_updates,
        }
        payload["route_sha256"] = content_digest(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointCreditRouteV1:
        required = {
            "format",
            "version",
            "joint_bundle_id",
            "owner_id",
            "scope_id",
            "session_id",
            "task_id",
            "joint_revision",
            "intent_id",
            "execution_id",
            "outcome_sha256",
            "outcome_trust",
            "state_owner",
            "eligible_receivers",
            "world_model_sha256",
            "world_prediction_sha256",
            "executive_sha256",
            "decision_sha256",
            "language_feedback_id",
            "language_feedback_sha256",
            "language_parameter_revision",
            "apply_updates",
            "route_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("joint credit route schema differs")
        unsigned = {key: value for key, value in payload.items() if key != "route_sha256"}
        if payload["route_sha256"] != content_digest(unsigned):
            raise ValueError("joint credit route digest differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported joint credit route version")
        return cls(
            joint_bundle_id=str(payload["joint_bundle_id"]),
            owner_id=str(payload["owner_id"]),
            scope_id=str(payload["scope_id"]),
            session_id=str(payload["session_id"]),
            task_id=str(payload["task_id"]),
            joint_revision=int(payload["joint_revision"]),
            intent_id=str(payload["intent_id"]),
            execution_id=str(payload["execution_id"]),
            outcome_sha256=str(payload["outcome_sha256"]),
            outcome_trust=str(payload["outcome_trust"]),
            state_owner=str(payload["state_owner"]),
            eligible_receivers=tuple(str(item) for item in payload["eligible_receivers"]),
            world_model_sha256=(
                None
                if payload["world_model_sha256"] is None
                else str(payload["world_model_sha256"])
            ),
            world_prediction_sha256=(
                None
                if payload["world_prediction_sha256"] is None
                else str(payload["world_prediction_sha256"])
            ),
            executive_sha256=(
                None if payload["executive_sha256"] is None else str(payload["executive_sha256"])
            ),
            decision_sha256=(
                None if payload["decision_sha256"] is None else str(payload["decision_sha256"])
            ),
            language_feedback_id=(
                None
                if payload["language_feedback_id"] is None
                else str(payload["language_feedback_id"])
            ),
            language_feedback_sha256=(
                None
                if payload["language_feedback_sha256"] is None
                else str(payload["language_feedback_sha256"])
            ),
            language_parameter_revision=(
                None
                if payload["language_parameter_revision"] is None
                else int(payload["language_parameter_revision"])
            ),
            apply_updates=bool(payload["apply_updates"]),
        )

"""Versioned bridge from Taiji's canonical cognitive state to a joint task candidate.

The CognitiveState remains the sole owner of world, memory, goals and actions.
TaskStateV3/V4 remains a neural language/attention state. This component binds
them to one task/scope and records the exact observation-to-byte provenance;
it does not copy world facts into a second mutable store or claim that an
unlinked workspace candidate is trustworthy.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .conditional_language_v3 import TaskStateV3
from .contracts import CognitiveState, NativeCheckpoint
from .episodic_memory import EpisodicMemoryStore
from .input_boundary import InputTrace
from .internalization import content_digest

JOINT_TASK_STATE_FORMAT_V1 = "taiji-joint-task-state-v1"
JOINT_TASK_STATE_FORMAT_V2 = "taiji-joint-task-state-v2"
JOINT_TASK_STATE_FORMAT = "taiji-joint-task-state-v3"
JOINT_NATIVE_COMPONENT_V1 = "joint_task_state_v1"
JOINT_NATIVE_COMPONENT_V2 = "joint_task_state_v2"
JOINT_NATIVE_COMPONENT = "joint_task_state_v3"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _unique_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(_required_text(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def cognitive_state_digest(state: CognitiveState) -> str:
    if not isinstance(state, CognitiveState):
        raise TypeError("joint task state requires Taiji's canonical CognitiveState")
    return content_digest(state.to_payload())


def _neural_observation_index(state: TaskStateV3) -> dict[str, bytes]:
    sources = [source for source, _ in state.observations]
    if len(set(sources)) != len(sources):
        raise ValueError("neural task state repeats an observation source")
    return dict(state.observations)


def _world_action_matches_intent(action: Any, intent: Any, world_action_id: str) -> bool:
    expected_parameters = intent.parameters.get("parameters", {})
    if not isinstance(expected_parameters, Mapping):
        return False
    return (
        action.action_id == world_action_id
        and action.kind == intent.kind
        and action.tick == intent.tick
        and action.actor_id == str(intent.parameters.get("actor_id", ""))
        and action.target_id == str(intent.parameters.get("target_id", ""))
        and content_digest(dict(action.parameters)) == content_digest(dict(expected_parameters))
    )


@dataclass(frozen=True)
class JointInputBindingV1:
    """One real input frame mapped to its neural source and perception events."""

    input_id: str
    neural_source_id: str
    modality: str
    source: str
    provenance: str
    timestamp_start: int
    timestamp_end: int
    byte_count: int
    payload_sha256: str
    percept_event_ids: tuple[str, ...]
    assembly_ids: tuple[str, ...]
    percept_tick_start: int | None = None
    percept_tick_end: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_id",
            "neural_source_id",
            "modality",
            "source",
            "provenance",
            "payload_sha256",
        ):
            _required_text(getattr(self, field_name), field_name)
        if self.timestamp_start < 0 or self.timestamp_end < self.timestamp_start:
            raise ValueError("input binding timestamps must be nonnegative and ordered")
        if self.byte_count <= 0:
            raise ValueError("input binding must contain at least one source byte")
        if len(self.percept_event_ids) != self.byte_count:
            raise ValueError("each input byte needs a recorded percept event reference")
        # The adapter may reuse one percept event for several bytes in a learned
        # assembly. Keep one reference per byte, but do not invent uniqueness.
        object.__setattr__(
            self,
            "percept_event_ids",
            tuple(_required_text(x, "percept event id") for x in self.percept_event_ids),
        )
        # Variable-duration bytes legitimately reuse an assembly identity.
        assemblies = tuple(_required_text(x, "assembly id") for x in self.assembly_ids)
        object.__setattr__(self, "assembly_ids", tuple(dict.fromkeys(assemblies)))
        if (self.percept_tick_start is None) != (self.percept_tick_end is None):
            raise ValueError("percept tick span requires both endpoints")
        tick_start = self.percept_tick_start
        tick_end = self.percept_tick_end
        if (
            tick_start is not None
            and tick_end is not None
            and (tick_start < 0 or tick_end < tick_start)
        ):
            raise ValueError("percept tick span must be nonnegative and ordered")

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "neural_source_id": self.neural_source_id,
            "modality": self.modality,
            "source": self.source,
            "provenance": self.provenance,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "byte_count": self.byte_count,
            "payload_sha256": self.payload_sha256,
            "percept_event_ids": list(self.percept_event_ids),
            "assembly_ids": list(self.assembly_ids),
            "percept_tick_start": self.percept_tick_start,
            "percept_tick_end": self.percept_tick_end,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointInputBindingV1:
        required = {
            "input_id",
            "neural_source_id",
            "modality",
            "source",
            "provenance",
            "timestamp_start",
            "timestamp_end",
            "byte_count",
            "payload_sha256",
            "percept_event_ids",
            "assembly_ids",
        }
        if not isinstance(payload, Mapping) or frozenset(payload) not in {
            frozenset(required),
            frozenset(required | {"percept_tick_start", "percept_tick_end"}),
        }:
            raise ValueError("joint input binding has an unsupported schema")
        return cls(
            input_id=str(payload["input_id"]),
            neural_source_id=str(payload["neural_source_id"]),
            modality=str(payload["modality"]),
            source=str(payload["source"]),
            provenance=str(payload["provenance"]),
            timestamp_start=int(payload["timestamp_start"]),
            timestamp_end=int(payload["timestamp_end"]),
            byte_count=int(payload["byte_count"]),
            payload_sha256=str(payload["payload_sha256"]),
            percept_event_ids=tuple(str(x) for x in payload["percept_event_ids"]),
            assembly_ids=tuple(str(x) for x in payload["assembly_ids"]),
            percept_tick_start=(
                None
                if payload.get("percept_tick_start") is None
                else int(payload["percept_tick_start"])
            ),
            percept_tick_end=(
                None
                if payload.get("percept_tick_end") is None
                else int(payload["percept_tick_end"])
            ),
        )


@dataclass(frozen=True)
class JointActionExecutionReceiptV1:
    """Checkpointed two-phase execution receipt; it never invokes an environment."""

    intent_id: str
    world_action_id: str
    idempotency_key: str
    status: str
    proposed_at_revision: int
    execution_id: str | None = None
    outcome_sha256: str | None = None

    STATUSES = frozenset(
        {
            "proposed",
            "dispatch_started",
            "awaiting_outcome",
            "outcome_unknown",
            "settled",
            "superseded",
            "cancelled",
        }
    )

    def __post_init__(self) -> None:
        for field_name in ("intent_id", "world_action_id", "idempotency_key"):
            _required_text(getattr(self, field_name), field_name)
        if self.status not in self.STATUSES:
            raise ValueError("unsupported joint action execution status")
        if self.proposed_at_revision < 0:
            raise ValueError("action proposal revision cannot be negative")
        if self.execution_id is not None:
            _required_text(self.execution_id, "execution_id")
        if self.outcome_sha256 is not None:
            _required_text(self.outcome_sha256, "outcome_sha256")
        if self.status == "proposed" and (
            self.execution_id is not None or self.outcome_sha256 is not None
        ):
            raise ValueError("proposed action cannot carry an execution or outcome receipt")
        if self.status == "dispatch_started" and self.execution_id != self.idempotency_key:
            raise ValueError("dispatch-started receipt must expose its idempotency key")
        if (
            self.status in {"awaiting_outcome", "outcome_unknown", "settled"}
            and self.execution_id is None
        ):
            raise ValueError("executed action status requires an execution receipt")
        if self.status in {"outcome_unknown", "settled"} and self.outcome_sha256 is None:
            raise ValueError("recorded action outcome requires an outcome digest")
        if self.status in {"superseded", "cancelled"} and (
            self.execution_id is not None or self.outcome_sha256 is not None
        ):
            raise ValueError("undispatched terminal proposal cannot carry an execution receipt")

    @classmethod
    def proposed(
        cls,
        *,
        joint_bundle_id: str,
        owner_id: str,
        scope_id: str,
        session_id: str,
        task_id: str,
        joint_revision: int,
        intent_id: str,
        world_action_id: str,
    ) -> JointActionExecutionReceiptV1:
        identity = "\0".join((joint_bundle_id, owner_id, scope_id, session_id, task_id, intent_id))
        key = "taiji-action:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return cls(
            intent_id=intent_id,
            world_action_id=world_action_id,
            idempotency_key=key,
            status="proposed",
            proposed_at_revision=joint_revision,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": "taiji-joint-action-execution-receipt-v1",
            "version": 1,
            "intent_id": self.intent_id,
            "world_action_id": self.world_action_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "proposed_at_revision": self.proposed_at_revision,
            "execution_id": self.execution_id,
            "outcome_sha256": self.outcome_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointActionExecutionReceiptV1:
        required = {
            "format",
            "version",
            "intent_id",
            "world_action_id",
            "idempotency_key",
            "status",
            "proposed_at_revision",
            "execution_id",
            "outcome_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("joint action execution receipt schema differs")
        if (payload["format"], payload["version"]) != (
            "taiji-joint-action-execution-receipt-v1",
            1,
        ):
            raise ValueError("unsupported joint action execution receipt version")
        return cls(
            intent_id=str(payload["intent_id"]),
            world_action_id=str(payload["world_action_id"]),
            idempotency_key=str(payload["idempotency_key"]),
            status=str(payload["status"]),
            proposed_at_revision=int(payload["proposed_at_revision"]),
            execution_id=(
                None if payload["execution_id"] is None else str(payload["execution_id"])
            ),
            outcome_sha256=(
                None if payload["outcome_sha256"] is None else str(payload["outcome_sha256"])
            ),
        )


@dataclass(frozen=True)
class JointDecisionTraceV1:
    """The exact read-only decision models and executive choice bound to one intent."""

    intent_id: str
    world_model_snapshot: Mapping[str, Any]
    world_prediction: Mapping[str, Any]
    executive_checkpoint: Mapping[str, Any]
    executive_decision: Mapping[str, Any]
    world_model_sha256: str
    world_prediction_sha256: str
    executive_sha256: str
    decision_sha256: str

    FORMAT = "taiji-joint-decision-trace-v1"

    def __post_init__(self) -> None:
        _required_text(self.intent_id, "intent_id")
        for field_name in (
            "world_model_snapshot",
            "world_prediction",
            "executive_checkpoint",
            "executive_decision",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_name} must be a checkpoint mapping")
            object.__setattr__(self, field_name, copy.deepcopy(dict(value)))
        expected = (
            ("world_model_sha256", self.world_model_snapshot),
            ("world_prediction_sha256", self.world_prediction),
            ("executive_sha256", self.executive_checkpoint),
            ("decision_sha256", self.executive_decision),
        )
        for name, payload in expected:
            digest = getattr(self, name)
            if digest != content_digest(payload):
                raise ValueError(f"joint decision trace {name} differs from its payload")
        selected = self.executive_decision.get("selected")
        selected_intent = selected.get("action_intent") if isinstance(selected, Mapping) else None
        if (
            not isinstance(selected_intent, Mapping)
            or selected_intent.get("intent_id") != self.intent_id
        ):
            raise ValueError("joint decision trace executive choice differs from its ActionIntent")

    @classmethod
    def create(
        cls,
        *,
        intent_id: str,
        world_model_snapshot: Mapping[str, Any],
        world_prediction: Mapping[str, Any],
        executive_checkpoint: Mapping[str, Any],
        executive_decision: Mapping[str, Any],
    ) -> JointDecisionTraceV1:
        return cls(
            intent_id=intent_id,
            world_model_snapshot=world_model_snapshot,
            world_prediction=world_prediction,
            world_prediction_sha256=content_digest(world_prediction),
            executive_checkpoint=executive_checkpoint,
            executive_decision=executive_decision,
            world_model_sha256=content_digest(world_model_snapshot),
            executive_sha256=content_digest(executive_checkpoint),
            decision_sha256=content_digest(executive_decision),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "intent_id": self.intent_id,
            "world_model_snapshot": copy.deepcopy(dict(self.world_model_snapshot)),
            "world_prediction": copy.deepcopy(dict(self.world_prediction)),
            "executive_checkpoint": copy.deepcopy(dict(self.executive_checkpoint)),
            "executive_decision": copy.deepcopy(dict(self.executive_decision)),
            "world_model_sha256": self.world_model_sha256,
            "world_prediction_sha256": self.world_prediction_sha256,
            "executive_sha256": self.executive_sha256,
            "decision_sha256": self.decision_sha256,
        }

    def restore_models(self) -> tuple[Any, Any, Any]:
        """Rebuild both models and deserialize the saved choice; this does not replay candidates."""
        import torch

        from .executive import ExecutiveController, ExecutiveDecision
        from .world_learning import WorldDynamicsLearner, WorldSchema

        schema_payload = self.world_model_snapshot.get("schema")
        if not isinstance(schema_payload, Mapping):
            raise ValueError("joint decision trace lacks the world model schema")
        rng_state = torch.random.get_rng_state()
        try:
            world_model = WorldDynamicsLearner(
                WorldSchema.from_payload(dict(schema_payload)),
                hidden_dim=int(self.world_model_snapshot["hidden_dim"]),
                seed=0,
            )
        finally:
            torch.random.set_rng_state(rng_state)
        world_model.restore(self.world_model_snapshot)
        executive = ExecutiveController.from_checkpoint(self.executive_checkpoint)
        decision = ExecutiveDecision.from_payload(self.executive_decision)
        if content_digest(world_model.snapshot()) != self.world_model_sha256:
            raise ValueError("restored world model differs from its decision trace")
        if content_digest(executive.checkpoint()) != self.executive_sha256:
            raise ValueError("restored executive differs from its decision trace")
        if content_digest(decision.to_payload()) != self.decision_sha256:
            raise ValueError("restored executive decision differs from its decision trace")
        if decision.action_intent.intent_id != self.intent_id:
            raise ValueError("restored executive decision names another ActionIntent")
        return world_model, executive, decision

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointDecisionTraceV1:
        required = {
            "format",
            "version",
            "intent_id",
            "world_model_snapshot",
            "world_prediction",
            "executive_checkpoint",
            "executive_decision",
            "world_model_sha256",
            "world_prediction_sha256",
            "executive_sha256",
            "decision_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("joint decision trace schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported joint decision trace version")
        return cls(
            intent_id=str(payload["intent_id"]),
            world_model_snapshot=payload["world_model_snapshot"],
            world_prediction=payload["world_prediction"],
            executive_checkpoint=payload["executive_checkpoint"],
            executive_decision=payload["executive_decision"],
            world_model_sha256=str(payload["world_model_sha256"]),
            world_prediction_sha256=str(payload["world_prediction_sha256"]),
            executive_sha256=str(payload["executive_sha256"]),
            decision_sha256=str(payload["decision_sha256"]),
        )


@dataclass(frozen=True)
class JointTaskStateV3:
    """Canonical Taiji state plus a source-bound neural task state."""

    joint_bundle_id: str
    owner_id: str
    scope_id: str
    session_id: str
    task_id: str
    joint_revision: int
    cognitive_state: CognitiveState
    cognitive_state_sha256: str
    language_state: TaskStateV3
    input_bindings: tuple[JointInputBindingV1, ...] = ()
    lifecycle: str = "active"
    action_receipts: tuple[JointActionExecutionReceiptV1, ...] = ()
    decision_trace: JointDecisionTraceV1 | None = None
    FORMAT = JOINT_TASK_STATE_FORMAT
    PENDING_ACTION_STATUSES = frozenset(
        {"proposed", "dispatch_started", "awaiting_outcome", "outcome_unknown"}
    )

    def __post_init__(self) -> None:
        for field_name in ("joint_bundle_id", "owner_id", "scope_id", "session_id", "task_id"):
            _required_text(getattr(self, field_name), field_name)
        if self.joint_revision < 0:
            raise ValueError("joint revision cannot be negative")
        if self.lifecycle not in {
            "active",
            "response_complete",
            "awaiting_outcome",
            "cancelled",
            "complete",
        }:
            raise ValueError("unsupported joint task lifecycle")
        if not isinstance(self.cognitive_state, CognitiveState):
            raise TypeError("joint state must reference a canonical CognitiveState")
        if not isinstance(self.language_state, TaskStateV3):
            raise TypeError("joint state requires the versioned V3/V4 neural task state")
        if self.task_id != self.cognitive_state.episode_id:
            raise ValueError("joint task identity differs from its cognitive owner")
        if (self.language_state.scope_id, self.language_state.session_id) != (
            self.scope_id,
            self.session_id,
        ):
            raise ValueError("joint and neural task scope/session differ")
        if self.cognitive_state_sha256 != cognitive_state_digest(self.cognitive_state):
            raise ValueError("joint cognitive owner digest differs")
        object.__setattr__(self, "input_bindings", tuple(self.input_bindings))
        input_ids = _unique_ids(tuple(x.input_id for x in self.input_bindings), "input id")
        source_ids = _unique_ids(
            tuple(x.neural_source_id for x in self.input_bindings), "neural input source id"
        )
        if len(input_ids) != len(source_ids):
            raise ValueError("joint input/source identities are not one-to-one")
        observations = _neural_observation_index(self.language_state)
        if set(observations) != set(source_ids):
            raise ValueError("joint input bindings do not exactly cover neural observations")
        for binding in self.input_bindings:
            raw = observations[binding.neural_source_id]
            if (
                len(raw) != binding.byte_count
                or hashlib.sha256(raw).hexdigest() != binding.payload_sha256
            ):
                raise ValueError("joint input binding does not match its exact neural byte source")
        receipts = tuple(self.action_receipts)
        if any(not isinstance(item, JointActionExecutionReceiptV1) for item in receipts):
            raise TypeError("joint action history must contain execution receipts")
        if len({item.intent_id for item in receipts}) != len(receipts):
            raise ValueError("joint action receipt history repeats an ActionIntent ID")
        latest_receipt = None if not receipts else receipts[-1]
        if (
            latest_receipt is not None
            and latest_receipt.status in {"awaiting_outcome", "outcome_unknown"}
            and self.lifecycle != "awaiting_outcome"
        ):
            raise ValueError("joint lifecycle must reflect an accepted pending action")
        if self.lifecycle == "awaiting_outcome" and (
            latest_receipt is None
            or latest_receipt.status not in {"awaiting_outcome", "outcome_unknown"}
        ):
            raise ValueError("awaiting_outcome lifecycle requires its matching action receipt")
        if receipts and receipts[-1].status in {
            "proposed",
            "dispatch_started",
            "awaiting_outcome",
            "outcome_unknown",
        }:
            current_intent = self.cognitive_state.action_intent
            if current_intent is None or current_intent.intent_id != receipts[-1].intent_id:
                raise ValueError(
                    "pending execution receipt differs from the canonical ActionIntent"
                )
        object.__setattr__(self, "action_receipts", receipts)
        if self.decision_trace is not None:
            if not isinstance(self.decision_trace, JointDecisionTraceV1):
                raise TypeError("joint decision trace must use its versioned contract")
            intent = self.cognitive_state.action_intent
            if intent is None or intent.intent_id != self.decision_trace.intent_id:
                raise ValueError("joint decision trace differs from the canonical ActionIntent")
            if (
                intent.parameters.get("world_model_sha256")
                != self.decision_trace.world_model_sha256
                or intent.parameters.get("executive_sha256") != self.decision_trace.executive_sha256
            ):
                raise ValueError("joint decision trace model digests differ from ActionIntent")
            selected = self.decision_trace.executive_decision.get("selected")
            selected_intent = (
                selected.get("action_intent") if isinstance(selected, Mapping) else None
            )
            if not isinstance(selected_intent, Mapping) or content_digest(
                selected_intent
            ) != content_digest(intent.to_payload()):
                raise ValueError("joint decision trace selected a different ActionIntent payload")
            receipt = None if not receipts else receipts[-1]
            prediction = self.cognitive_state.world_prediction
            if (
                receipt is None
                or receipt.intent_id != intent.intent_id
                or prediction is None
                or not _world_action_matches_intent(
                    prediction.action, intent, receipt.world_action_id
                )
                or content_digest(prediction.to_payload())
                != self.decision_trace.world_prediction_sha256
                or content_digest(self.decision_trace.world_prediction)
                != self.decision_trace.world_prediction_sha256
            ):
                raise ValueError("joint decision trace lacks its matching selected prediction")
        if self.language_state.bundle_id == "":
            raise ValueError("joint neural bundle identity cannot be empty")

    @classmethod
    def begin(
        cls,
        candidate: Any,
        cognitive_state: CognitiveState,
        *,
        joint_bundle_id: str,
        owner_id: str,
        scope_id: str,
        session_id: str,
    ) -> JointTaskStateV1:
        if getattr(candidate, "bundle_id", None) is None:
            raise TypeError("candidate must expose an explicit bundle id")
        neural = candidate.new_state(
            scope_id=scope_id,
            session_id=session_id,
            native_state_ref=f"joint:{joint_bundle_id}:{cognitive_state.episode_id}",
        )
        if neural.observations:
            raise ValueError("new joint task must start at an explicit zero observation cursor")
        return cls(
            joint_bundle_id=joint_bundle_id,
            owner_id=owner_id,
            scope_id=scope_id,
            session_id=session_id,
            task_id=cognitive_state.episode_id,
            joint_revision=0,
            cognitive_state=cognitive_state,
            cognitive_state_sha256=cognitive_state_digest(cognitive_state),
            language_state=neural,
        )

    @staticmethod
    def _frame_bytes(trace: InputTrace) -> bytes:
        if not isinstance(trace, InputTrace) or not trace.observations:
            raise TypeError("joint task input must be a nonempty, typed InputTrace")
        if len(trace.observations) != len(trace.percepts):
            raise ValueError("joint task input lost perception lineage")
        source = trace.observations[0].source
        provenance = trace.observations[0].provenance
        modality = trace.observations[0].modality
        values = []
        previous_time = -1
        previous_percept_tick = -1
        for observation, percept in zip(trace.observations, trace.percepts, strict=True):
            if observation.modality != "text-byte" or type(observation.value) is not int:
                raise ValueError("V4 joint byte input only accepts Taiji text-byte observations")
            if not 0 <= observation.value <= 255:
                raise ValueError("text-byte observation is outside the byte alphabet")
            if (observation.source, observation.provenance, observation.modality) != (
                source,
                provenance,
                modality,
            ):
                raise ValueError("one joint input frame cannot mix source/provenance/modality")
            if observation.timestamp < previous_time:
                raise ValueError("input observation timestamps must be ordered")
            if percept.observation_tick < previous_percept_tick:
                raise ValueError("input percept ticks must be ordered")
            if percept.modality != "text-byte":
                raise ValueError("percept modality differs from the text-byte source")
            values.append(observation.value)
            previous_time = observation.timestamp
            previous_percept_tick = percept.observation_tick
        return bytes(values)

    def append_input(
        self,
        candidate: Any,
        trace: InputTrace,
        cognitive_state: CognitiveState,
        *,
        resource_snapshot: Mapping[str, int],
    ) -> JointTaskStateV1:
        self._validate_live_owner()
        if self.lifecycle in {"awaiting_outcome", "cancelled", "complete"}:
            raise ValueError(f"joint task status {self.lifecycle!r} cannot accept input")
        receipts = self.action_receipts
        latest_receipt = None if not receipts else receipts[-1]
        if latest_receipt is not None and latest_receipt.status in {
            "dispatch_started",
            "awaiting_outcome",
            "outcome_unknown",
        }:
            raise ValueError(
                "an external action may have executed; reconcile its receipt before new input"
            )
        if latest_receipt is not None and latest_receipt.status == "proposed":
            if (
                cognitive_state.action_intent is not None
                and cognitive_state.action_intent.intent_id == latest_receipt.intent_id
            ):
                raise ValueError("new input still carries the unexecuted proposed ActionIntent")
            receipts = (*receipts[:-1], replace(latest_receipt, status="superseded"))
        if getattr(candidate, "bundle_id", None) != self.language_state.bundle_id:
            raise ValueError("joint candidate bundle differs")
        if (self.scope_id, self.session_id) != (
            self.language_state.scope_id,
            self.language_state.session_id,
        ):
            raise ValueError("joint scope changed without an explicit reset")
        raw = self._frame_bytes(trace)
        if any(binding.input_id == trace.input_id for binding in self.input_bindings):
            raise ValueError("joint input frame was already consumed")
        if (
            cognitive_state.episode_id != self.task_id
            or cognitive_state.tick < self.cognitive_state.tick
        ):
            raise ValueError(
                "updated cognitive owner crossed task identity or moved time backwards"
            )
        if cognitive_state.observation is None or cognitive_state.percept is None:
            raise ValueError("updated cognitive owner lacks the traced observation/percept")
        if content_digest(cognitive_state.observation.to_payload()) != content_digest(
            trace.observations[-1].to_payload()
        ) or content_digest(cognitive_state.percept.to_payload()) != content_digest(
            trace.percepts[-1].to_payload()
        ):
            raise ValueError("InputTrace is not the source of the updated CognitiveState")
        if trace.action_intent is not None and (
            cognitive_state.action_intent is None
            or content_digest(trace.action_intent.to_payload())
            != content_digest(cognitive_state.action_intent.to_payload())
        ):
            raise ValueError("input-trace intent differs from its cognitive owner")
        source_id = "joint-input:" + hashlib.sha256(trace.input_id.encode("utf-8")).hexdigest()[:32]
        if (
            len(raw) + sum(len(item) for _, item in self.language_state.observations)
            > candidate.config.max_observation_bytes
        ):
            raise ValueError(
                "joint session exceeds the V4 observation budget; open an explicit new scope"
            )
        neural = candidate.step_task(
            raw,
            self.language_state,
            scope_id=self.scope_id,
            session_id=self.session_id,
            source_id=source_id,
            resource_snapshot=dict(resource_snapshot),
        )
        percept_ids = tuple(percept.event_id for percept in trace.percepts)
        assembly_ids = tuple(percept.assembly_id for percept in trace.percepts)
        binding = JointInputBindingV1(
            input_id=trace.input_id,
            neural_source_id=source_id,
            modality=trace.modality,
            source=trace.observations[0].source,
            provenance=trace.observations[0].provenance,
            timestamp_start=trace.observations[0].timestamp,
            timestamp_end=trace.observations[-1].timestamp,
            byte_count=len(raw),
            payload_sha256=hashlib.sha256(raw).hexdigest(),
            percept_event_ids=percept_ids,
            assembly_ids=assembly_ids,
            percept_tick_start=min(percept.observation_tick for percept in trace.percepts),
            percept_tick_end=max(percept.observation_tick for percept in trace.percepts),
        )
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            cognitive_state=cognitive_state,
            cognitive_state_sha256=cognitive_state_digest(cognitive_state),
            language_state=neural,
            input_bindings=self.input_bindings + (binding,),
            lifecycle="active" if self.lifecycle == "response_complete" else self.lifecycle,
            action_receipts=receipts,
            decision_trace=(
                self.decision_trace
                if cognitive_state.action_intent is not None
                and self.decision_trace is not None
                and cognitive_state.action_intent.intent_id == self.decision_trace.intent_id
                else None
            ),
        )

    def advance_cognitive_state(self, cognitive_state: CognitiveState) -> JointTaskStateV1:
        """Commit a Taiji owner-state revision without copying its content into shadow fields."""
        self._validate_live_owner()
        if cognitive_state.episode_id != self.task_id:
            raise ValueError("task identity change requires an explicit new task")
        if cognitive_state.tick < self.cognitive_state.tick:
            raise ValueError("canonical cognitive-state time cannot move backwards")
        receipts = self.action_receipts
        latest = None if not receipts else receipts[-1]
        pending = {"proposed", "dispatch_started", "awaiting_outcome", "outcome_unknown"}
        if latest is not None and latest.status in pending:
            current_intent = cognitive_state.action_intent
            if current_intent is None or current_intent.intent_id != latest.intent_id:
                raise ValueError(
                    "pending action must be cancelled or reconciled by its execution receipt"
                )
            if cognitive_state_digest(cognitive_state) != self.cognitive_state_sha256:
                raise ValueError(
                    "pending action state must be changed through its execution receipt"
                )
        previous_intent = self.cognitive_state.action_intent
        current_intent = cognitive_state.action_intent
        if (
            previous_intent is not None
            and current_intent is not None
            and previous_intent.intent_id == current_intent.intent_id
            and content_digest(previous_intent.to_payload())
            != content_digest(current_intent.to_payload())
        ):
            raise ValueError("an ActionIntent identity cannot be rebound to different parameters")
        if current_intent is not None and (
            previous_intent is None or current_intent.intent_id != previous_intent.intent_id
        ):
            if latest is not None and latest.status in pending:
                raise ValueError("a second ActionIntent cannot replace a pending execution")
            world_action_id = str(
                current_intent.parameters.get("source_world_action_id", current_intent.intent_id)
            )
            receipts = (
                *receipts,
                JointActionExecutionReceiptV1.proposed(
                    joint_bundle_id=self.joint_bundle_id,
                    owner_id=self.owner_id,
                    scope_id=self.scope_id,
                    session_id=self.session_id,
                    task_id=self.task_id,
                    joint_revision=self.joint_revision + 1,
                    intent_id=current_intent.intent_id,
                    world_action_id=world_action_id,
                ),
            )
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            cognitive_state=cognitive_state,
            cognitive_state_sha256=cognitive_state_digest(cognitive_state),
            action_receipts=receipts,
            decision_trace=(
                self.decision_trace
                if current_intent is not None
                and self.decision_trace is not None
                and current_intent.intent_id == self.decision_trace.intent_id
                else None
            ),
        )

    def with_decision_trace(
        self,
        intent_id: str,
        *,
        world_model_snapshot: Mapping[str, Any],
        executive_checkpoint: Mapping[str, Any],
        executive_decision: Mapping[str, Any],
    ) -> JointTaskStateV1:
        self._validate_live_owner()
        intent = self.cognitive_state.action_intent
        if intent is None or intent.intent_id != intent_id:
            raise ValueError("decision trace requires the current canonical ActionIntent")
        trace = JointDecisionTraceV1.create(
            intent_id=intent_id,
            world_model_snapshot=world_model_snapshot,
            world_prediction=(
                self.cognitive_state.world_prediction.to_payload()
                if self.cognitive_state.world_prediction is not None
                else {}
            ),
            executive_checkpoint=executive_checkpoint,
            executive_decision=executive_decision,
        )
        if (
            intent.parameters.get("world_model_sha256") != trace.world_model_sha256
            or intent.parameters.get("executive_sha256") != trace.executive_sha256
        ):
            raise ValueError("decision model snapshots do not match the ActionIntent hashes")
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            decision_trace=trace,
        )

    @property
    def action_receipt(self) -> JointActionExecutionReceiptV1 | None:
        return None if not self.action_receipts else self.action_receipts[-1]

    def begin_action_dispatch(self, intent_id: str) -> tuple[JointTaskStateV1, str]:
        """Checkpoint this intent's idempotency key before the caller crosses an executor boundary."""
        self._validate_live_owner()
        receipt = self.action_receipt
        if self.lifecycle not in {"active", "response_complete"}:
            raise ValueError("joint task lifecycle does not permit action dispatch")
        if receipt is None or receipt.status != "proposed" or receipt.intent_id != intent_id:
            raise ValueError("only the current proposed ActionIntent can begin dispatch")
        started = replace(
            receipt,
            status="dispatch_started",
            execution_id=receipt.idempotency_key,
        )
        state = replace(
            self,
            joint_revision=self.joint_revision + 1,
            action_receipts=(*self.action_receipts[:-1], started),
        )
        return state, started.idempotency_key

    def mark_action_execution_accepted(self, intent_id: str, execution_id: str) -> JointTaskStateV1:
        """Record a durable executor acknowledgement; this method does not dispatch anything."""
        self._validate_live_owner()
        _required_text(execution_id, "execution_id")
        receipt = self.action_receipt
        if receipt is None or receipt.intent_id != intent_id:
            raise ValueError("execution acknowledgement does not match the pending ActionIntent")
        if receipt.status == "awaiting_outcome" and receipt.execution_id == execution_id:
            return self
        if receipt.status != "dispatch_started":
            raise ValueError("execution acknowledgement requires a checkpointed dispatch start")
        accepted = replace(receipt, status="awaiting_outcome", execution_id=execution_id)
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            lifecycle="awaiting_outcome",
            action_receipts=(*self.action_receipts[:-1], accepted),
        )

    def cancel_action_proposal(self, intent_id: str) -> JointTaskStateV1:
        """Cancel a proposal only before any executor call may have started."""
        self._validate_live_owner()
        receipt = self.action_receipt
        if receipt is None or receipt.status != "proposed" or receipt.intent_id != intent_id:
            raise ValueError("only an undispatched proposed action can be cancelled")
        root = self.cognitive_state
        if root.action_intent is None or root.action_intent.intent_id != intent_id:
            raise ValueError("proposed action differs from canonical CognitiveState")
        cancelled = replace(receipt, status="cancelled")
        owner = replace(
            root,
            action_intent=None,
            outcome=None,
            world_transition=None,
            world_prediction=None,
            plan=replace(root.plan, selected_plan_id=None),
        )
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            cognitive_state=owner,
            cognitive_state_sha256=cognitive_state_digest(owner),
            action_receipts=(*self.action_receipts[:-1], cancelled),
            decision_trace=None,
        )

    def reset_turn(self) -> JointTaskStateV1:
        self._validate_live_owner()
        if self.lifecycle not in {"active", "response_complete"}:
            raise ValueError("turn reset cannot discard an unresolved environment outcome")
        if (
            self.action_receipt is not None
            and self.action_receipt.status in self.PENDING_ACTION_STATUSES
        ):
            raise ValueError("resolve or cancel the pending ActionIntent before turn reset")
        neural = self.language_state.reset("turn")
        if not isinstance(neural, TaskStateV3):
            raise TypeError("turn reset lost the candidate task-state version")
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            language_state=neural,
            lifecycle="active",
        )

    def with_language_state(self, language_state: TaskStateV3) -> JointTaskStateV1:
        """Bind a later neural revision only if it preserves this task's observed-source ledger."""
        self._validate_live_owner()
        if not isinstance(language_state, TaskStateV3):
            raise TypeError("language state must retain its explicit V3/V4 schema")
        if (language_state.bundle_id, language_state.scope_id, language_state.session_id) != (
            self.language_state.bundle_id,
            self.scope_id,
            self.session_id,
        ):
            raise ValueError("updated neural state changed bundle or scope ownership")
        known = _neural_observation_index(self.language_state)
        supplied = _neural_observation_index(language_state)
        if known != supplied:
            raise ValueError(
                "parameter update cannot insert, rewrite or drop joint source observations"
            )
        if language_state.parameter_revision < self.language_state.parameter_revision:
            raise ValueError("candidate parameter revision moved backwards")
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            language_state=language_state,
        )

    def attach_to_native_checkpoint(
        self,
        native_payload: Mapping[str, Any],
        language_candidate_checkpoint: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Put root cognition, bound task state and its V4 model/optimizer in one native envelope."""
        self._validate_live_owner()
        envelope = NativeCheckpoint.from_payload(native_payload)
        if cognitive_state_digest(envelope.cognitive_state) != self.cognitive_state_sha256:
            raise ValueError("native checkpoint and joint state have different cognitive owners")
        from .conditional_language_v4 import ConditionalLanguageTrainerV4

        trainer, neural, _ = ConditionalLanguageTrainerV4.from_checkpoint(
            dict(language_candidate_checkpoint)
        )
        if trainer.model.bundle_id != self.language_state.bundle_id:
            raise ValueError("joint language checkpoint belongs to another candidate bundle")
        if content_digest(neural.to_payload()) != content_digest(self.language_state.to_payload()):
            raise ValueError("joint task state differs from language checkpoint state")
        if {
            JOINT_NATIVE_COMPONENT,
            JOINT_NATIVE_COMPONENT_V2,
            JOINT_NATIVE_COMPONENT_V1,
        }.intersection(envelope.components):
            raise ValueError("native checkpoint already contains a joint task component")
        frame = {
            "format": JOINT_TASK_STATE_FORMAT,
            "version": 3,
            "joint_bundle_id": self.joint_bundle_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "joint_revision": self.joint_revision,
            "cognitive_state_sha256": self.cognitive_state_sha256,
            "input_bindings": [item.to_payload() for item in self.input_bindings],
            "lifecycle": self.lifecycle,
            "action_receipts": [item.to_payload() for item in self.action_receipts],
            "decision_trace": (
                None if self.decision_trace is None else self.decision_trace.to_payload()
            ),
        }
        component = {
            "format": "taiji-joint-cognitive-component-v3",
            "frame": frame,
            "language_candidate_checkpoint": dict(language_candidate_checkpoint),
        }
        component["component_digest"] = content_digest(component)
        return NativeCheckpoint(
            kernel=envelope.kernel,
            cognitive_state=envelope.cognitive_state,
            components={**envelope.components, JOINT_NATIVE_COMPONENT: component},
            adapter=envelope.adapter,
        ).to_payload()

    @classmethod
    def restore_native_checkpoint(
        cls,
        native_payload: Mapping[str, Any],
        *,
        joint_bundle_id: str,
        owner_id: str,
        scope_id: str,
        session_id: str,
    ) -> tuple[JointTaskStateV1, Any]:
        """Restore both cognitive owner and candidate from the same native checkpoint."""
        envelope = NativeCheckpoint.from_payload(native_payload)
        component_specs = (
            (
                JOINT_NATIVE_COMPONENT,
                "taiji-joint-cognitive-component-v3",
                JOINT_TASK_STATE_FORMAT,
                3,
            ),
            (
                JOINT_NATIVE_COMPONENT_V2,
                "taiji-joint-cognitive-component-v2",
                JOINT_TASK_STATE_FORMAT_V2,
                2,
            ),
            (
                JOINT_NATIVE_COMPONENT_V1,
                "taiji-joint-cognitive-component-v1",
                JOINT_TASK_STATE_FORMAT_V1,
                1,
            ),
        )
        present_specs = [item for item in component_specs if item[0] in envelope.components]
        if len(present_specs) != 1:
            raise ValueError("native checkpoint must contain exactly one joint cognition component")
        component_key, expected_component_format, expected_frame_format, expected_frame_version = (
            present_specs[0]
        )
        component = envelope.components[component_key]
        if (
            not isinstance(component, Mapping)
            or component.get("format") != expected_component_format
            or set(component)
            != {"format", "frame", "language_candidate_checkpoint", "component_digest"}
        ):
            raise ValueError("native checkpoint joint cognition component schema differs")
        expected_component = content_digest(
            {key: value for key, value in component.items() if key != "component_digest"}
        )
        if component.get("component_digest") != expected_component:
            raise ValueError("joint cognition component digest differs")
        frame = component.get("frame")
        if not isinstance(frame, Mapping):
            raise ValueError("joint cognition component frame is missing")
        base_frame_fields = {
            "format",
            "version",
            "joint_bundle_id",
            "owner_id",
            "scope_id",
            "session_id",
            "task_id",
            "joint_revision",
            "cognitive_state_sha256",
            "input_bindings",
            "lifecycle",
        }
        expected_frame_fields = (
            base_frame_fields | {"action_receipts", "decision_trace"}
            if expected_frame_version == 3
            else base_frame_fields
        )
        if (frame.get("format"), frame.get("version")) != (
            expected_frame_format,
            expected_frame_version,
        ) or set(frame) != expected_frame_fields:
            raise ValueError("joint cognition component and frame versions differ")
        from .conditional_language_v4 import ConditionalLanguageTrainerV4

        trainer, neural, _ = ConditionalLanguageTrainerV4.from_checkpoint(
            component["language_candidate_checkpoint"]
        )
        if frame.get("cognitive_state_sha256") != cognitive_state_digest(envelope.cognitive_state):
            raise ValueError("joint frame cognitive owner differs from native checkpoint")
        joint_payload = {
            "format": str(frame.get("format", JOINT_TASK_STATE_FORMAT_V1)),
            "version": int(frame.get("version", 1)),
            "joint_bundle_id": str(frame["joint_bundle_id"]),
            "owner_id": str(frame["owner_id"]),
            "scope_id": str(frame["scope_id"]),
            "session_id": str(frame["session_id"]),
            "task_id": str(frame["task_id"]),
            "joint_revision": int(frame["joint_revision"]),
            "cognitive_state_sha256": str(frame["cognitive_state_sha256"]),
            "cognitive_state": envelope.cognitive_state.to_payload(),
            "language_state": neural.to_payload(),
            "input_bindings": list(frame["input_bindings"]),
            "lifecycle": str(frame["lifecycle"]),
        }
        if expected_frame_version == 3:
            joint_payload["action_receipts"] = list(frame["action_receipts"])
            joint_payload["decision_trace"] = frame["decision_trace"]
        joint = cls.from_payload(
            joint_payload,
            joint_bundle_id=joint_bundle_id,
            owner_id=owner_id,
            scope_id=scope_id,
            session_id=session_id,
            candidate_bundle_id=trainer.model.bundle_id,
        )
        for name, expected in (
            ("joint_bundle_id", joint_bundle_id),
            ("owner_id", owner_id),
            ("scope_id", scope_id),
            ("session_id", session_id),
        ):
            if getattr(joint, name) != expected:
                raise ValueError(f"restored joint {name} differs from its authenticated scope")
        return joint, trainer

    def cancel_task(self) -> JointTaskStateV1:
        self._validate_live_owner()
        if self.lifecycle not in {"active", "response_complete"}:
            raise ValueError("cancellation cannot discard an unresolved environment outcome")
        receipt = self.action_receipt
        if receipt is not None and receipt.status in {
            "dispatch_started",
            "awaiting_outcome",
            "outcome_unknown",
        }:
            raise ValueError("external action may have executed; reconcile it before cancelling")
        receipts = self.action_receipts
        if receipt is not None and receipt.status == "proposed":
            receipts = (*receipts[:-1], replace(receipt, status="cancelled"))
        neural = self.language_state.reset("turn")
        if not isinstance(neural, TaskStateV3):
            raise TypeError("task cancellation lost the candidate task-state version")
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            language_state=neural,
            lifecycle="cancelled",
            action_receipts=receipts,
        )

    def complete_response(self) -> JointTaskStateV1:
        """Close this response turn while allowing a later user turn in the same task."""
        self._validate_live_owner()
        if self.lifecycle != "active":
            raise ValueError("only an active joint task can be completed")
        if not self.language_state.generated.endswith(b"<|end|>"):
            raise ValueError("response completion requires the candidate end marker")
        return replace(self, joint_revision=self.joint_revision + 1, lifecycle="response_complete")

    def _validate_world_transition(
        self,
        cognitive_state: CognitiveState,
        receipt: JointActionExecutionReceiptV1,
        intent: Any,
        *,
        validate_before_state: bool = True,
    ) -> Any | None:
        transition = cognitive_state.world_transition
        if transition is None:
            return None
        if not _world_action_matches_intent(transition.action, intent, receipt.world_action_id):
            raise ValueError("world transition action differs from the selected ActionIntent")
        if validate_before_state and content_digest(
            transition.before.to_payload()
        ) != content_digest(self.cognitive_state.world.to_payload()):
            raise ValueError("world transition before-state differs from the dispatched state")
        if content_digest(transition.after.to_payload()) != content_digest(
            cognitive_state.world.to_payload()
        ):
            raise ValueError("world transition after-state differs from the current CognitiveState")
        prediction = cognitive_state.world_prediction
        if (
            prediction is None
            or not _world_action_matches_intent(prediction.action, intent, receipt.world_action_id)
            or prediction.action.tick != transition.action.tick
        ):
            raise ValueError("world transition lacks its matching selected prediction")
        if (
            self.decision_trace is not None
            and self.decision_trace.intent_id == intent.intent_id
            and content_digest(prediction.to_payload())
            != self.decision_trace.world_prediction_sha256
        ):
            raise ValueError("world prediction differs from its selection-time decision trace")
        return prediction

    def record_action_outcome(
        self,
        cognitive_state: CognitiveState,
        intent_id: str,
        *,
        execution_id: str,
    ) -> JointTaskStateV1:
        """Bind a caller-supplied outcome; this does not authenticate an executor or environment."""
        self._validate_live_owner()
        if self.lifecycle not in {"active", "response_complete", "awaiting_outcome"}:
            raise ValueError("a closed task cannot accept an action outcome")
        _required_text(execution_id, "execution_id")
        receipt = self.action_receipt
        if receipt is None or receipt.intent_id != intent_id:
            raise ValueError("action outcome has no matching persisted execution receipt")
        outcome = cognitive_state.outcome
        if (
            outcome is None
            or outcome.intent_id != intent_id
            or outcome.provenance
            in {
                "generated",
                "imagined",
            }
        ):
            raise ValueError("action outcome requires a linked non-generated environment receipt")
        intent = self.cognitive_state.action_intent
        if intent is None or intent.intent_id != intent_id:
            raise ValueError("action outcome does not match the canonical ActionIntent")
        world_transition = cognitive_state.world_transition
        if world_transition is not None:
            self._validate_world_transition(
                cognitive_state,
                receipt,
                intent,
                validate_before_state=receipt.status not in {"settled", "outcome_unknown"},
            )
            transition_outcome = world_transition.outcome.to_payload()
            cognitive_outcome = outcome.to_payload()
            transition_outcome.pop("intent_id")
            cognitive_outcome.pop("intent_id")
            if content_digest(transition_outcome) != content_digest(cognitive_outcome):
                raise ValueError("world transition and CognitiveState outcome disagree")
        result_digest = content_digest(
            {
                "outcome": outcome.to_payload(),
                "world_transition": (
                    None if world_transition is None else world_transition.to_payload()
                ),
            }
        )
        if receipt.status == "settled":
            if receipt.execution_id == execution_id and receipt.outcome_sha256 == result_digest:
                return self
            raise ValueError("conflicting duplicate outcome for a settled action")
        if receipt.status not in {"dispatch_started", "awaiting_outcome", "outcome_unknown"}:
            raise ValueError("action outcome requires a started or acknowledged execution")
        if receipt.execution_id != execution_id:
            raise ValueError("action outcome execution receipt ID differs")
        if (
            receipt.status == "outcome_unknown"
            and outcome.success is None
            and receipt.outcome_sha256 != result_digest
        ):
            raise ValueError("conflicting duplicate unknown outcome for one execution receipt")
        if (
            receipt.status == "outcome_unknown"
            and outcome.success is None
            and receipt.outcome_sha256 == result_digest
        ):
            return self
        if (
            cognitive_state.episode_id != self.task_id
            or cognitive_state.tick < self.cognitive_state.tick
        ):
            raise ValueError("action outcome crossed task identity or moved time backwards")
        current_intent = cognitive_state.action_intent
        if (
            intent is None
            or (
                current_intent is not None
                and content_digest(current_intent.to_payload())
                != content_digest(intent.to_payload())
            )
            or outcome is None
            or intent.intent_id != intent_id
            or outcome.intent_id != intent_id
        ):
            raise ValueError("action outcome requires the same intent and a non-generated receipt")
        if outcome.success is None:
            updated_receipt = replace(
                receipt, status="outcome_unknown", outcome_sha256=result_digest
            )
            lifecycle = "awaiting_outcome"
        else:
            updated_receipt = replace(receipt, status="settled", outcome_sha256=result_digest)
            lifecycle = "complete" if outcome.terminal else "active"
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            cognitive_state=cognitive_state,
            cognitive_state_sha256=cognitive_state_digest(cognitive_state),
            lifecycle=lifecycle,
            action_receipts=(*self.action_receipts[:-1], updated_receipt),
        )

    def new_session(
        self,
        candidate: Any,
        cognitive_state: CognitiveState,
        *,
        session_id: str,
        authorized_carry_refs: Sequence[str],
    ) -> JointTaskStateV1:
        self._validate_live_owner()
        if (
            self.action_receipt is not None
            and self.action_receipt.status in self.PENDING_ACTION_STATUSES
        ):
            raise ValueError("resolve or cancel the pending action before opening a new session")
        _required_text(session_id, "new session_id")
        if session_id == self.session_id:
            raise ValueError("new_session requires a new session identity")
        if cognitive_state.episode_id == self.task_id:
            raise ValueError("new session must be built on an explicitly supplied task owner state")
        if getattr(candidate, "bundle_id", None) != self.language_state.bundle_id:
            raise ValueError("new session candidate bundle differs")
        old_refs = self.cognitive_reference_ids(self.cognitive_state)
        new_refs = self.cognitive_reference_ids(cognitive_state)
        carried = old_refs & new_refs
        authorized = set(_unique_ids(tuple(authorized_carry_refs), "authorized carry reference"))
        if carried != authorized:
            raise ValueError(
                "cross-session carry must explicitly match all repeated cognitive references"
            )
        neural = candidate.new_state(
            scope_id=self.scope_id,
            session_id=session_id,
            native_state_ref=f"joint:{self.joint_bundle_id}:{cognitive_state.episode_id}",
        )
        return JointTaskStateV1(
            joint_bundle_id=self.joint_bundle_id,
            owner_id=self.owner_id,
            scope_id=self.scope_id,
            session_id=session_id,
            task_id=cognitive_state.episode_id,
            joint_revision=self.joint_revision + 1,
            cognitive_state=cognitive_state,
            cognitive_state_sha256=cognitive_state_digest(cognitive_state),
            language_state=neural,
        )

    @staticmethod
    def cognitive_reference_ids(state: CognitiveState) -> set[str]:
        refs = {x.object_id for x in state.world.objects}
        refs.update(x.event_id for x in state.world.events)
        refs.update(state.memory.working_ids)
        refs.update(state.memory.episodic_ids)
        refs.update(state.memory.concept_ids)
        refs.update(x.item_id for x in state.memory.working_items)
        refs.update(x.event_id for x in state.events)
        refs.update(x.assembly_id for x in state.assemblies)
        for assembly in state.assemblies:
            refs.update(assembly.source_event_ids)
        refs.update(x.concept_id for x in state.concepts)
        refs.update(x.goal_id for x in state.goals.goals)
        refs.update(x.plan_id for x in state.plan.candidates)
        refs.update(x.candidate_id for x in state.workspace.candidates)
        refs.update(x.source for x in state.workspace.candidates)
        refs.update(x for x, _ in state.workspace.bindings)
        refs.update(y for _, y in state.workspace.bindings)
        refs.update(state.workspace.percept_event_id.split())
        refs.update(state.workspace.percept_assembly_id.split())
        refs.update(state.world.percept_event_id.split())
        refs.update(state.world.percept_assembly_id.split())
        if state.observation is not None:
            refs.update((state.observation.source, state.observation.provenance))
        if state.percept is not None:
            refs.update((state.percept.event_id, state.percept.assembly_id))
        for world_event in state.world.events:
            refs.add(world_event.provenance)
        for cognitive_event in state.events:
            refs.update(cognitive_event.provenance.split())
        for field_name in ("autobiographical_ids", "commitment_ids", "available_tool_ids"):
            refs.update(getattr(state.self_state, field_name))
        refs.update(state.development.capability_gaps)
        refs.update(state.development.proposal_ids)
        refs.update(state.development.validation_evidence_ids)
        refs.update(state.development.lineage)
        refs.update(state.self_state.lineage)
        if state.action_intent is not None:
            refs.add(state.action_intent.intent_id)
        if state.outcome is not None:
            refs.add(state.outcome.intent_id)
        return refs

    def content_view(self) -> JointContentReadViewV1:
        self._validate_live_owner()
        return JointContentReadViewV1.from_joint_state(self)

    def _validate_live_owner(self) -> None:
        if cognitive_state_digest(self.cognitive_state) != self.cognitive_state_sha256:
            raise ValueError("canonical CognitiveState changed outside the joint revision")
        observations = _neural_observation_index(self.language_state)
        if set(observations) != {x.neural_source_id for x in self.input_bindings}:
            raise ValueError("joint raw observations and provenance map differ")
        for binding in self.input_bindings:
            raw = observations[binding.neural_source_id]
            if (
                len(raw) != binding.byte_count
                or hashlib.sha256(raw).hexdigest() != binding.payload_sha256
            ):
                raise ValueError("joint raw evidence source digest differs")

    def to_payload(self) -> dict[str, Any]:
        self._validate_live_owner()
        return {
            "format": self.FORMAT,
            "version": 3,
            "joint_bundle_id": self.joint_bundle_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "joint_revision": self.joint_revision,
            "cognitive_state_sha256": self.cognitive_state_sha256,
            "cognitive_state": self.cognitive_state.to_payload(),
            "language_state": self.language_state.to_payload(),
            "input_bindings": [x.to_payload() for x in self.input_bindings],
            "lifecycle": self.lifecycle,
            "action_receipts": [x.to_payload() for x in self.action_receipts],
            "decision_trace": (
                None if self.decision_trace is None else self.decision_trace.to_payload()
            ),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        joint_bundle_id: str,
        owner_id: str,
        scope_id: str,
        session_id: str,
        candidate_bundle_id: str,
    ) -> JointTaskStateV1:
        legacy_fields = {
            "format",
            "version",
            "joint_bundle_id",
            "owner_id",
            "scope_id",
            "session_id",
            "task_id",
            "joint_revision",
            "cognitive_state_sha256",
            "cognitive_state",
            "language_state",
            "input_bindings",
            "lifecycle",
        }
        current_fields = legacy_fields | {"action_receipts", "decision_trace"}
        if not isinstance(payload, Mapping) or frozenset(payload) not in {
            frozenset(legacy_fields),
            frozenset(current_fields),
        }:
            raise ValueError("joint task state schema differs")
        payload_version = (payload["format"], payload["version"])
        if payload_version not in {
            (JOINT_TASK_STATE_FORMAT_V1, 1),
            (JOINT_TASK_STATE_FORMAT_V2, 2),
            (cls.FORMAT, 3),
        }:
            raise ValueError("unsupported joint task state version")
        if payload_version == (cls.FORMAT, 3) and set(payload) != current_fields:
            raise ValueError("joint task state v3 requires action and decision trace fields")
        if payload_version != (cls.FORMAT, 3) and set(payload) != legacy_fields:
            raise ValueError("legacy joint task state cannot carry v3 action receipts")
        for name, expected in (
            ("joint_bundle_id", joint_bundle_id),
            ("owner_id", owner_id),
            ("scope_id", scope_id),
            ("session_id", session_id),
        ):
            if payload[name] != expected:
                raise ValueError(f"restored joint {name} differs from its authorized owner")
        cognitive = CognitiveState.from_payload(payload["cognitive_state"])
        neural = TaskStateV3.from_payload(payload["language_state"])
        if neural.bundle_id != candidate_bundle_id:
            raise ValueError("restored neural candidate bundle differs")
        return cls(
            joint_bundle_id=str(payload["joint_bundle_id"]),
            owner_id=str(payload["owner_id"]),
            scope_id=str(payload["scope_id"]),
            session_id=str(payload["session_id"]),
            task_id=str(payload["task_id"]),
            joint_revision=int(payload["joint_revision"]),
            cognitive_state=cognitive,
            cognitive_state_sha256=str(payload["cognitive_state_sha256"]),
            language_state=neural,
            input_bindings=tuple(
                JointInputBindingV1.from_payload(x) for x in payload["input_bindings"]
            ),
            lifecycle=str(payload["lifecycle"]),
            action_receipts=tuple(
                JointActionExecutionReceiptV1.from_payload(item)
                for item in payload.get("action_receipts", ())
            ),
            decision_trace=(
                None
                if payload.get("decision_trace") is None
                else JointDecisionTraceV1.from_payload(payload["decision_trace"])
            ),
        )


@dataclass(frozen=True)
class JointEpisodeReadV1:
    """One store-selected episode bound to the current query and source frames."""

    FORMAT = "taiji-joint-episode-read-v1"

    joint_bundle_id: str
    owner_id: str
    scope_id: str
    session_id: str
    joint_revision: int
    cognitive_state_sha256: str
    query_input_id: str
    query_neural_source_id: str
    memory_id: str
    record_sha256: str
    cue_sha256: str
    retrieval_score: float
    retrieval_limit: int
    memory_neural_source_ids: tuple[str, ...]
    selected_neural_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "joint_bundle_id",
            "owner_id",
            "scope_id",
            "session_id",
            "cognitive_state_sha256",
            "query_input_id",
            "query_neural_source_id",
            "memory_id",
            "record_sha256",
            "cue_sha256",
        ):
            _required_text(getattr(self, field_name), field_name)
        if (
            self.joint_revision < 0
            or not math.isfinite(float(self.retrieval_score))
            or self.retrieval_limit <= 0
        ):
            raise ValueError("joint episode read revision or score is invalid")
        memory_sources = _unique_ids(self.memory_neural_source_ids, "episode memory source id")
        selected_sources = _unique_ids(
            self.selected_neural_source_ids, "selected episode source id"
        )
        if not memory_sources or self.query_neural_source_id in memory_sources:
            raise ValueError("episode read must link a distinct historical input source")
        if self.query_neural_source_id not in selected_sources or not set(memory_sources).issubset(
            selected_sources
        ):
            raise ValueError("episode read does not include both query and memory sources")
        object.__setattr__(self, "memory_neural_source_ids", memory_sources)
        object.__setattr__(self, "selected_neural_source_ids", selected_sources)

    @classmethod
    def retrieve(
        cls,
        state: JointTaskStateV1,
        store: EpisodicMemoryStore,
        *,
        limit: int = 3,
    ) -> JointEpisodeReadV1:
        """Retrieve only a record already named by Taiji's current memory state."""
        if not isinstance(store, EpisodicMemoryStore):
            raise TypeError("episode retrieval requires Taiji's EpisodicMemoryStore")
        state._validate_live_owner()
        if state.lifecycle != "active":
            raise ValueError("joint task lifecycle does not permit episode retrieval")
        root = state.cognitive_state
        if not state.input_bindings or root.percept is None:
            raise ValueError("episode retrieval requires a current input-bound percept")
        if not root.memory.episodic_ids:
            raise ValueError("canonical CognitiveState has no authorized episode references")
        query_binding = state.input_bindings[-1]
        if (
            root.observation is None
            or root.observation.source != query_binding.source
            or root.observation.provenance != query_binding.provenance
            or root.percept.event_id not in query_binding.percept_event_ids
        ):
            raise ValueError("current percept does not belong to the latest joint input frame")
        query_cue = root.percept.features
        hits = store.retrieve(query_cue, limit=limit)
        allowed_memory_ids = set(root.memory.episodic_ids)
        hit = next(
            (
                item
                for item in hits
                if item.record.memory_id in allowed_memory_ids
                and item.record.provenance not in {"generated", "imagined"}
            ),
            None,
        )
        if hit is None:
            raise ValueError("episodic retrieval found no authorized real-experience hit")

        record = hit.record
        event_ids = set(record.event_ids)
        assembly_ids = set(record.assembly_ids)
        percept_ids = set(record.event_ids)
        cognitive_events = {item.event_id: item for item in root.events}
        assemblies = {item.assembly_id: item for item in root.assemblies}
        event_windows: list[tuple[int, int]] = []
        for event_id in record.event_ids:
            event = cognitive_events.get(event_id)
            if event is None:
                continue
            event_windows.append((event.start_tick, event.end_tick))
            assembly_ids.update(event.assembly_ids)
            for assembly_id in event.assembly_ids:
                assembly = assemblies.get(assembly_id)
                if assembly is not None:
                    percept_ids.update(assembly.source_event_ids)

        candidates = [
            binding
            for binding in state.input_bindings
            if assembly_ids.intersection(binding.assembly_ids)
            or event_ids.intersection(binding.percept_event_ids)
            or percept_ids.intersection(binding.percept_event_ids)
        ]
        tick_candidates = [
            binding
            for binding in candidates
            if binding.percept_tick_start is not None
            and binding.percept_tick_end is not None
            and (
                any(
                    max(binding.percept_tick_start, event_start)
                    <= min(binding.percept_tick_end, event_end)
                    for event_start, event_end in event_windows
                )
                if event_windows
                else binding.percept_tick_start <= record.tick <= binding.percept_tick_end
            )
        ]
        if len(tick_candidates) > 1:
            raise ValueError(
                "episode source identity is ambiguous across reused event/assembly IDs"
            )
        if not tick_candidates:
            raise ValueError("episode record has no source frame with a matching percept time span")
        memory_sources = {tick_candidates[0].neural_source_id}
        memory_sources.discard(query_binding.neural_source_id)
        if not memory_sources:
            raise ValueError("retrieved episode has no source frame in the joint task ledger")
        selected_sources = tuple(
            dict.fromkeys((query_binding.neural_source_id, *sorted(memory_sources)))
        )
        return cls(
            joint_bundle_id=state.joint_bundle_id,
            owner_id=state.owner_id,
            scope_id=state.scope_id,
            session_id=state.session_id,
            joint_revision=state.joint_revision,
            cognitive_state_sha256=state.cognitive_state_sha256,
            query_input_id=query_binding.input_id,
            query_neural_source_id=query_binding.neural_source_id,
            memory_id=record.memory_id,
            record_sha256=content_digest(record.to_payload()),
            cue_sha256=content_digest(query_cue),
            retrieval_score=float(hit.score),
            retrieval_limit=int(limit),
            memory_neural_source_ids=tuple(sorted(memory_sources)),
            selected_neural_source_ids=selected_sources,
        )

    def validate_for(self, state: JointTaskStateV1, store: EpisodicMemoryStore) -> None:
        state._validate_live_owner()
        if (
            self.joint_bundle_id,
            self.owner_id,
            self.scope_id,
            self.session_id,
            self.joint_revision,
            self.cognitive_state_sha256,
        ) != (
            state.joint_bundle_id,
            state.owner_id,
            state.scope_id,
            state.session_id,
            state.joint_revision,
            state.cognitive_state_sha256,
        ):
            raise ValueError("episode read belongs to a stale or different joint revision")
        if not state.input_bindings or (
            state.input_bindings[-1].input_id,
            state.input_bindings[-1].neural_source_id,
        ) != (self.query_input_id, self.query_neural_source_id):
            raise ValueError("episode read query source differs from the current joint input")
        if (
            state.cognitive_state.percept is None
            or content_digest(state.cognitive_state.percept.features) != self.cue_sha256
        ):
            raise ValueError("episode retrieval cue differs from the current cognitive percept")
        if self.memory_id not in state.cognitive_state.memory.episodic_ids:
            raise ValueError("episode read is no longer authorized by canonical memory state")
        records = {record.memory_id: record for record in store.records}
        record = records.get(self.memory_id)
        if record is None or content_digest(record.to_payload()) != self.record_sha256:
            raise ValueError("episode read record is missing or changed in the memory store")
        known_sources = {item.neural_source_id for item in state.input_bindings}
        if not set(self.selected_neural_source_ids).issubset(known_sources):
            raise ValueError("episode read references an input source outside the joint ledger")
        expected = type(self).retrieve(state, store, limit=self.retrieval_limit)
        if expected != self:
            raise ValueError("episode read receipt differs from the authorized store selection")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "joint_bundle_id": self.joint_bundle_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "joint_revision": self.joint_revision,
            "cognitive_state_sha256": self.cognitive_state_sha256,
            "query_input_id": self.query_input_id,
            "query_neural_source_id": self.query_neural_source_id,
            "memory_id": self.memory_id,
            "record_sha256": self.record_sha256,
            "cue_sha256": self.cue_sha256,
            "retrieval_score": self.retrieval_score,
            "retrieval_limit": self.retrieval_limit,
            "memory_neural_source_ids": list(self.memory_neural_source_ids),
            "selected_neural_source_ids": list(self.selected_neural_source_ids),
        }


@dataclass(frozen=True)
class JointCandidateReferenceV1:
    candidate_id: str
    target_id: str | None
    target_kind: str
    source_status: str


@dataclass(frozen=True)
class JointContentReadViewV1:
    """Read-only references into the canonical Taiji state; values are never re-authored here."""

    bundle_id: str
    state_revision: int
    source_ids: tuple[str, ...]
    selected_neural_source_ids: tuple[str, ...]
    selected_candidates: tuple[JointCandidateReferenceV1, ...]
    unresolved_focus_ids: tuple[str, ...]
    world_objects: tuple[Any, ...]
    world_events: tuple[Any, ...]
    available_episode_refs: tuple[str, ...]
    goal_ids: tuple[str, ...]
    plan_ids: tuple[str, ...]
    action_intent_id: str | None
    outcome_intent_id: str | None

    @classmethod
    def from_joint_state(cls, state: JointTaskStateV1) -> JointContentReadViewV1:
        state._validate_live_owner()
        root = state.cognitive_state
        known_sources = {state.task_id}
        for item in state.input_bindings:
            known_sources.update({item.input_id, item.neural_source_id})
            known_sources.update(item.percept_event_ids)
            known_sources.update(item.assembly_ids)
        candidate_map = {item.candidate_id: item for item in root.workspace.candidates}
        requested_ids = (
            tuple(root.workspace.selection.selected_ids)
            if root.workspace.selection is not None
            else tuple(root.workspace.focus)
        )
        unresolved_focus = tuple(item for item in requested_ids if item not in candidate_map)
        selected_ids = tuple(item for item in requested_ids if item in candidate_map)
        bindings = dict(root.workspace.bindings)
        if len(bindings) != len(root.workspace.bindings):
            raise ValueError("canonical workspace has duplicate candidate-content bindings")
        objects = {item.object_id: item for item in root.world.objects}
        world_events = {item.event_id: item for item in root.world.events}
        root_events = {item.event_id: item for item in root.events}
        assemblies = {item.assembly_id: item for item in root.assemblies}
        references = []
        selected_objects: dict[str, Any] = {}
        selected_events: dict[str, Any] = {}
        selected_neural_source_ids: set[str] = set()

        def linked_neural_sources(event_id: str, provenance: str) -> set[str]:
            """Map an evidence event to exact consumed input identities, never generic origin labels."""
            owner_event = root_events.get(event_id)
            owner_assembly_ids = set(() if owner_event is None else owner_event.assembly_ids)
            owner_percept_ids: set[str] = set()
            for assembly_id in owner_assembly_ids:
                assembly = assemblies.get(assembly_id)
                if assembly is not None:
                    owner_percept_ids.update(assembly.source_event_ids)
            exact: set[str] = set()
            for binding in state.input_bindings:
                if (
                    provenance in {binding.input_id, binding.neural_source_id}
                    or event_id in binding.percept_event_ids
                ):
                    exact.add(binding.neural_source_id)
            if exact:
                return exact
            return {
                binding.neural_source_id
                for binding in state.input_bindings
                if owner_assembly_ids.intersection(binding.assembly_ids)
                or owner_percept_ids.intersection(binding.percept_event_ids)
            }

        for candidate_id in selected_ids:
            target = bindings.get(candidate_id)
            target_kind = "unresolved"
            source_status = "unverified"
            if target in objects:
                target_kind = "world_object"
                selected_objects[target] = objects[target]
                linked_fields: list[bool] = []
                candidate_sources: set[str] = set()
                for attribute, value in objects[target].attributes:
                    matching = [
                        event
                        for event in root.world.events
                        if event.object_id == target
                        and event.tick <= root.world.tick
                        and (attribute, value) in event.attributes
                    ]
                    if not matching:
                        linked_fields.append(False)
                        continue
                    latest_tick = max(event.tick for event in matching)
                    latest = [event for event in matching if event.tick == latest_tick]
                    field_linked = False
                    for event in latest:
                        event_sources = linked_neural_sources(event.event_id, event.provenance)
                        if event_sources:
                            selected_events[event.event_id] = event
                            candidate_sources.update(event_sources)
                            field_linked = True
                    linked_fields.append(field_linked)
                if linked_fields and all(linked_fields):
                    source_status = "source_linked"
                    selected_neural_source_ids.update(candidate_sources)
                elif any(linked_fields):
                    source_status = "partially_source_linked"
                else:
                    source_status = "world_fact_source_unlinked"
            elif target in world_events:
                target_kind = "world_event"
                event_sources = linked_neural_sources(target, world_events[target].provenance)
                if event_sources:
                    selected_events[target] = world_events[target]
                    selected_neural_source_ids.update(event_sources)
                    source_status = "source_linked"
                else:
                    source_status = "world_event_source_unlinked"
            elif target in root_events:
                target_kind = "cognitive_event"
                event_sources = linked_neural_sources(target, root_events[target].provenance)
                if event_sources:
                    selected_neural_source_ids.update(event_sources)
                    source_status = "source_linked"
                else:
                    source_status = "cognitive_event_source_unlinked"
            elif target in set(root.memory.episodic_ids):
                target_kind, source_status = "episodic_memory_ref", "ref_only_owner_lookup_required"
            elif target in set(root.memory.working_ids):
                target_kind, source_status = "working_memory_ref", "ref_only"
            elif target in {g.goal_id for g in root.goals.goals}:
                target_kind, source_status = "goal", "canonical_state_ref"
            elif target in {p.plan_id for p in root.plan.candidates}:
                target_kind, source_status = "plan_candidate", "canonical_state_ref"
            elif target in set(root.memory.concept_ids) | {c.concept_id for c in root.concepts}:
                target_kind, source_status = "concept_ref", "ref_only_owner_lookup_required"
            elif target in known_sources:
                target_kind, source_status = "input_source", "source_linked"
                selected_neural_source_ids.update(
                    binding.neural_source_id
                    for binding in state.input_bindings
                    if target
                    in {
                        binding.input_id,
                        binding.neural_source_id,
                        *binding.percept_event_ids,
                        *binding.assembly_ids,
                    }
                )
            elif target is None:
                source_status = "no_content_binding"
            else:
                raise ValueError("workspace binding target is absent from the canonical owner")
            references.append(
                JointCandidateReferenceV1(candidate_id, target, target_kind, source_status)
            )
        return cls(
            bundle_id=state.joint_bundle_id,
            state_revision=state.joint_revision,
            source_ids=tuple(item.input_id for item in state.input_bindings),
            selected_neural_source_ids=tuple(sorted(selected_neural_source_ids)),
            selected_candidates=tuple(references),
            unresolved_focus_ids=unresolved_focus,
            world_objects=tuple(selected_objects.values()),
            world_events=tuple(selected_events.values()),
            available_episode_refs=tuple(root.memory.episodic_ids),
            goal_ids=tuple(goal.goal_id for goal in root.goals.goals),
            plan_ids=tuple(plan.plan_id for plan in root.plan.candidates),
            action_intent_id=None if root.action_intent is None else root.action_intent.intent_id,
            outcome_intent_id=None if root.outcome is None else root.outcome.intent_id,
        )


# Compatibility names for existing callers; serialized data migrates forward
# through the exact v1/v2/v3 formats in from_payload.
JointTaskStateV2 = JointTaskStateV3
JointTaskStateV1 = JointTaskStateV3

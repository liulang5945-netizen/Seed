"""Versioned bridge from Taiji's canonical cognitive state to a joint task candidate.

The CognitiveState remains the sole owner of world, memory, goals and actions.
TaskStateV3/V4 remains a neural language/attention state. This component binds
them to one task/scope and records the exact observation-to-byte provenance;
it does not copy world facts into a second mutable store or claim that an
unlinked workspace candidate is trustworthy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .conditional_language_v3 import TaskStateV3
from .contracts import CognitiveState, NativeCheckpoint
from .input_boundary import InputTrace
from .internalization import content_digest

JOINT_TASK_STATE_FORMAT = "taiji-joint-task-state-v1"
JOINT_NATIVE_COMPONENT = "joint_task_state_v1"


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
        object.__setattr__(
            self, "percept_event_ids", _unique_ids(self.percept_event_ids, "percept event id")
        )
        # Variable-duration bytes legitimately reuse an assembly identity.
        assemblies = tuple(_required_text(x, "assembly id") for x in self.assembly_ids)
        object.__setattr__(self, "assembly_ids", tuple(dict.fromkeys(assemblies)))

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
        if not isinstance(payload, Mapping) or set(payload) != required:
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
        )


@dataclass(frozen=True)
class JointTaskStateV1:
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
    FORMAT = JOINT_TASK_STATE_FORMAT

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
            if percept.modality != "text-byte":
                raise ValueError("percept modality differs from the text-byte source")
            values.append(observation.value)
            previous_time = observation.timestamp
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
        )
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            cognitive_state=cognitive_state,
            cognitive_state_sha256=cognitive_state_digest(cognitive_state),
            language_state=neural,
            input_bindings=self.input_bindings + (binding,),
            lifecycle="active" if self.lifecycle == "response_complete" else self.lifecycle,
        )

    def advance_cognitive_state(self, cognitive_state: CognitiveState) -> JointTaskStateV1:
        """Commit a Taiji owner-state revision without copying its content into shadow fields."""
        self._validate_live_owner()
        if cognitive_state.episode_id != self.task_id:
            raise ValueError("task identity change requires an explicit new task")
        if cognitive_state.tick < self.cognitive_state.tick:
            raise ValueError("canonical cognitive-state time cannot move backwards")
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            cognitive_state=cognitive_state,
            cognitive_state_sha256=cognitive_state_digest(cognitive_state),
        )

    def reset_turn(self) -> JointTaskStateV1:
        self._validate_live_owner()
        if self.lifecycle not in {"active", "response_complete"}:
            raise ValueError("turn reset cannot discard an unresolved environment outcome")
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
        if JOINT_NATIVE_COMPONENT in envelope.components:
            raise ValueError("native checkpoint already contains a joint task component")
        frame = {
            "format": JOINT_TASK_STATE_FORMAT,
            "version": 1,
            "joint_bundle_id": self.joint_bundle_id,
            "owner_id": self.owner_id,
            "scope_id": self.scope_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "joint_revision": self.joint_revision,
            "cognitive_state_sha256": self.cognitive_state_sha256,
            "input_bindings": [item.to_payload() for item in self.input_bindings],
            "lifecycle": self.lifecycle,
        }
        component = {
            "format": "taiji-joint-cognitive-component-v1",
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
        component = envelope.components.get(JOINT_NATIVE_COMPONENT)
        if (
            not isinstance(component, Mapping)
            or component.get("format") != "taiji-joint-cognitive-component-v1"
        ):
            raise ValueError("native checkpoint lacks the joint cognition component")
        expected_component = content_digest(
            {key: value for key, value in component.items() if key != "component_digest"}
        )
        if component.get("component_digest") != expected_component:
            raise ValueError("joint cognition component digest differs")
        from .conditional_language_v4 import ConditionalLanguageTrainerV4

        trainer, neural, _ = ConditionalLanguageTrainerV4.from_checkpoint(
            component["language_candidate_checkpoint"]
        )
        frame = component.get("frame")
        if not isinstance(frame, Mapping):
            raise ValueError("joint cognition component frame is missing")
        if frame.get("cognitive_state_sha256") != cognitive_state_digest(envelope.cognitive_state):
            raise ValueError("joint frame cognitive owner differs from native checkpoint")
        joint = cls(
            joint_bundle_id=str(frame["joint_bundle_id"]),
            owner_id=str(frame["owner_id"]),
            scope_id=str(frame["scope_id"]),
            session_id=str(frame["session_id"]),
            task_id=str(frame["task_id"]),
            joint_revision=int(frame["joint_revision"]),
            cognitive_state=envelope.cognitive_state,
            cognitive_state_sha256=str(frame["cognitive_state_sha256"]),
            language_state=neural,
            input_bindings=tuple(
                JointInputBindingV1.from_payload(item) for item in frame["input_bindings"]
            ),
            lifecycle=str(frame["lifecycle"]),
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
        neural = self.language_state.reset("turn")
        if not isinstance(neural, TaskStateV3):
            raise TypeError("task cancellation lost the candidate task-state version")
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            language_state=neural,
            lifecycle="cancelled",
        )

    def complete_response(self) -> JointTaskStateV1:
        """Close this response turn while allowing a later user turn in the same task."""
        self._validate_live_owner()
        if self.lifecycle != "active":
            raise ValueError("only an active joint task can be completed")
        if not self.language_state.generated.endswith(b"<|end|>"):
            raise ValueError("response completion requires the candidate end marker")
        return replace(self, joint_revision=self.joint_revision + 1, lifecycle="response_complete")

    def record_action_outcome(
        self, cognitive_state: CognitiveState, intent_id: str
    ) -> JointTaskStateV1:
        """Record a linked environment outcome; failure may leave the parent task active."""
        self._validate_live_owner()
        if self.lifecycle not in {"active", "awaiting_outcome"}:
            raise ValueError("a closed task cannot accept an action outcome")
        if (
            cognitive_state.episode_id != self.task_id
            or cognitive_state.tick < self.cognitive_state.tick
        ):
            raise ValueError("action outcome crossed task identity or moved time backwards")
        intent = self.cognitive_state.action_intent
        current_intent, outcome = cognitive_state.action_intent, cognitive_state.outcome
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
            or outcome.provenance == "generated"
        ):
            raise ValueError("action outcome requires the same intent and a non-generated receipt")
        lifecycle = (
            "awaiting_outcome"
            if outcome.success is None
            else "complete" if outcome.terminal else "active"
        )
        return replace(
            self,
            joint_revision=self.joint_revision + 1,
            cognitive_state=cognitive_state,
            cognitive_state_sha256=cognitive_state_digest(cognitive_state),
            lifecycle=lifecycle,
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
            "version": 1,
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
        fields = {
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
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("joint task state schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported joint task state version")
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
        )


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
            known_sources.update({item.input_id, item.neural_source_id, item.source})
            known_sources.update(item.percept_event_ids)
            known_sources.update(item.assembly_ids)
        known_sources.add(root.observation.source if root.observation is not None else "")
        known_sources.discard("")
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
        for candidate_id in selected_ids:
            target = bindings.get(candidate_id)
            target_kind = "unresolved"
            source_status = "unverified"
            if target in objects:
                target_kind = "world_object"
                selected_objects[target] = objects[target]
                linked_fields: list[bool] = []
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
                        owner_event = root_events.get(event.event_id)
                        direct_source = event.provenance in known_sources
                        traced_owner = owner_event is not None and any(
                            assembly_id in assemblies
                            and bool(set(assemblies[assembly_id].source_event_ids) & known_sources)
                            for assembly_id in owner_event.assembly_ids
                        )
                        if direct_source or traced_owner:
                            selected_events[event.event_id] = event
                            field_linked = True
                    linked_fields.append(field_linked)
                if linked_fields and all(linked_fields):
                    source_status = "source_linked"
                elif any(linked_fields):
                    source_status = "partially_source_linked"
                else:
                    source_status = "world_fact_source_unlinked"
            elif target in world_events:
                target_kind = "world_event"
                selected_events[target] = world_events[target]
                source_status = (
                    "source_linked"
                    if world_events[target].provenance in known_sources or target in root_events
                    else "world_event_source_unlinked"
                )
            elif target in root_events:
                target_kind = "cognitive_event"
                source_status = (
                    "source_linked"
                    if any(
                        assembly_id in assemblies
                        and set(assemblies[assembly_id].source_event_ids).intersection(
                            known_sources
                        )
                        for assembly_id in root_events[target].assembly_ids
                    )
                    else "cognitive_event_source_unlinked"
                )
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

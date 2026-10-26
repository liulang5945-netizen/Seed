"""Opt-in one-step world prediction and executive selection on a joint task frame."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .contracts import (
    ActionIntent,
    PlanCandidate,
    PlanState,
    WorldAction,
    WorldPredictionRecord,
    WorldState,
)
from .episodic_memory import EpisodicMemoryStore
from .executive import ExecutiveCandidate, ExecutiveContext, ExecutiveController, ExecutiveDecision
from .generation import ContentPlan
from .internalization import content_digest
from .joint_task_state import JointEpisodeReadV1, JointTaskStateV1
from .world_learning import WorldDynamicsLearner, WorldPrediction


def _json_projection(value: Any, *, path: str = "projection") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, torch.Tensor):
        return [
            _json_projection(item, path=path) for item in value.detach().cpu().flatten().tolist()
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _json_projection(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_projection(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


@dataclass(frozen=True)
class JointWorldCandidatePredictionV1:
    candidate: ExecutiveCandidate
    action: WorldAction
    prediction: WorldPrediction
    goal_alignment: float
    state_change: float


@dataclass(frozen=True)
class JointGoalWorldTargetV1:
    """Structured target state tied to a canonical Goal and its input evidence."""

    goal_id: str
    target_state: WorldState
    source_input_ids: tuple[str, ...]
    provenance: str
    state_slots: tuple[tuple[str, str], ...]
    relation_targets: tuple[tuple[tuple[str, str, str], bool], ...] = ()

    def __post_init__(self) -> None:
        if not self.goal_id.strip() or not self.provenance.strip():
            raise ValueError("joint goal target requires goal and provenance identities")
        if self.provenance in {"generated", "imagined"}:
            raise ValueError("generated or imagined goal targets cannot ground action selection")
        if not self.source_input_ids or len(set(self.source_input_ids)) != len(
            self.source_input_ids
        ):
            raise ValueError("joint goal target requires unique, source-bound input IDs")
        if any(not object_id or not attribute for object_id, attribute in self.state_slots):
            raise ValueError("joint goal state constraints require object and attribute identities")
        if not self.state_slots and not self.relation_targets:
            raise ValueError("joint goal target must explicitly name at least one state constraint")
        if len(set(self.state_slots)) != len(self.state_slots):
            raise ValueError("joint goal state slots must be unique")
        relation_keys = tuple(key for key, _ in self.relation_targets)
        if len(set(relation_keys)) != len(relation_keys):
            raise ValueError("joint goal relation targets must be unique")
        if any(
            len(relation) != 3 or any(not part for part in relation) or type(present) is not bool
            for relation, present in self.relation_targets
        ):
            raise ValueError("joint goal relation targets require a triple and a boolean value")

    def validate_for(self, state: JointTaskStateV1, active_goal_id: str, schema: Any) -> None:
        if self.goal_id != active_goal_id:
            raise ValueError("structured world target belongs to another active goal")
        known_inputs = {item.input_id for item in state.input_bindings}
        if not set(self.source_input_ids).issubset(known_inputs):
            raise ValueError("structured world target cites an input outside the joint ledger")
        if not set(self.state_slots).issubset(set(schema.state_slots)):
            raise ValueError(
                "structured world target contains a state slot outside the model schema"
            )
        if not set(key for key, _ in self.relation_targets).issubset(set(schema.relation_slots)):
            raise ValueError("structured world target contains a relation outside the model schema")
        target_objects = {item.object_id: item for item in self.target_state.objects}
        for object_id, attribute in self.state_slots:
            if object_id not in target_objects or attribute not in dict(
                target_objects[object_id].attributes
            ):
                raise ValueError("structured world target omits an explicitly required state value")
            raw_value = target_objects[object_id].attribute(attribute)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError("structured goal state constraints must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("structured world target values must be finite numbers")

    def distance(self, predicted_state: WorldState, schema: Any) -> float:
        predicted_objects = {item.object_id: item for item in predicted_state.objects}
        target_objects = {item.object_id: item for item in self.target_state.objects}
        errors: list[float] = []
        for slot in self.state_slots:
            object_id, attribute = slot
            if object_id not in predicted_objects or attribute not in dict(
                predicted_objects[object_id].attributes
            ):
                raise ValueError("world prediction omitted an explicitly targeted state value")
            index = schema.state_slots.index(slot)
            scale = schema.state_scales[index]
            predicted_value = predicted_objects[object_id].attribute(attribute)
            target_value = target_objects[object_id].attribute(attribute)
            if (
                isinstance(predicted_value, bool)
                or not isinstance(predicted_value, (int, float))
                or isinstance(target_value, bool)
                or not isinstance(target_value, (int, float))
            ):
                raise ValueError("goal-aligned state values must remain numeric")
            delta = (float(predicted_value) - float(target_value)) / float(scale)
            errors.append(delta * delta)
        predicted_relations = set(predicted_state.relations)
        target_relation_indices = {
            relation: index for index, relation in enumerate(schema.relation_slots)
        }
        for relation, expected_present in self.relation_targets:
            index = schema.object_state_dim + target_relation_indices[relation]
            scale = float(schema.state_scales[index])
            delta = (float(relation in predicted_relations) - float(expected_present)) / scale
            errors.append(delta * delta)
        return math.sqrt(sum(errors) / len(errors))

    def to_payload(self) -> dict[str, Any]:
        target_objects = {item.object_id: item for item in self.target_state.objects}
        return {
            "format": "taiji-joint-goal-world-target-v1",
            "goal_id": self.goal_id,
            "state_constraints": [
                {
                    "object_id": object_id,
                    "attribute": attribute,
                    "value": float(target_objects[object_id].attribute(attribute)),
                }
                for object_id, attribute in self.state_slots
            ],
            "relation_constraints": [
                {"relation": list(relation), "present": present}
                for relation, present in self.relation_targets
            ],
            "source_input_ids": list(self.source_input_ids),
            "provenance": self.provenance,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True)
class JointWorldDecisionV1:
    """A read-only candidate comparison whose selected intent is committed to the joint owner."""

    source_joint_revision: int
    joint_state: JointTaskStateV1
    goal_id: str
    goal_target: JointGoalWorldTargetV1
    candidates: tuple[JointWorldCandidatePredictionV1, ...]
    decision: ExecutiveDecision
    world_schema_revision: int
    world_model_sha256: str
    executive_sha256: str

    @classmethod
    def select(
        cls,
        joint_state: JointTaskStateV1,
        world_model: WorldDynamicsLearner,
        executive: ExecutiveController,
        goal_target: JointGoalWorldTargetV1,
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> JointWorldDecisionV1:
        joint_state._validate_live_owner()
        if joint_state.lifecycle != "active":
            raise ValueError("joint lifecycle does not permit a new world decision")
        root = joint_state.cognitive_state
        if root.action_intent is not None:
            raise ValueError("settle or clear the existing ActionIntent before another decision")
        if len(root.world.affordances) < 2:
            raise ValueError("joint world selection requires at least two Taiji affordances")
        if executive.candidate_feature_dim != 6:
            raise ValueError("joint world bridge requires the six-feature executive contract")
        context = ExecutiveContext.from_state(
            root,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        if context.goal_id is None:
            raise ValueError("joint world selection requires an active canonical goal")
        if not isinstance(goal_target, JointGoalWorldTargetV1):
            raise TypeError("joint world selection requires a structured goal target")
        goal_target.validate_for(joint_state, context.goal_id, world_model.schema)

        world_snapshot = world_model.snapshot()
        world_digest = content_digest(world_snapshot)
        executive_checkpoint = executive.checkpoint()
        executive_digest = content_digest(executive_checkpoint)
        was_training = world_model.training
        candidates: list[JointWorldCandidatePredictionV1] = []
        try:
            for affordance in root.world.affordances:
                candidate_id = (
                    f"{joint_state.joint_bundle_id}:{joint_state.joint_revision}:"
                    f"{affordance.affordance_id}"
                )
                intent_id = f"{candidate_id}:intent"
                action = WorldAction(
                    action_id=intent_id,
                    kind=affordance.action_kind,
                    tick=root.world.tick,
                    actor_id=affordance.actor_id,
                    target_id=affordance.target_id,
                    parameters=affordance.parameters,
                    provenance="planned",
                )
                prediction = world_model.predict(
                    root.world,
                    action,
                    register_parameters=False,
                )
                state_change = world_model.schema.normalized_state_error(
                    prediction.state, root.world
                )
                goal_distance = goal_target.distance(prediction.state, world_model.schema)
                goal_alignment = 1.0 / (1.0 + goal_distance)
                information_gain = min(
                    1.0, max(0.0, root.world.uncertainty - prediction.uncertainty)
                )
                resource_cost = float(dict(affordance.parameters).get("resource_cost", 0.0))
                if not math.isfinite(resource_cost) or not 0.0 <= resource_cost <= 1.0:
                    raise ValueError("affordance resource_cost must be finite and in [0, 1]")

                action_parameters = {
                    "candidate_id": candidate_id,
                    "source_world_action_id": action.action_id,
                    "goal_target_sha256": goal_target.digest,
                    "goal_target": goal_target.to_payload(),
                    "world_model_sha256": world_digest,
                    "executive_sha256": executive_digest,
                    "world_schema_revision": world_model.schema_registry.active_version,
                    "affordance_id": affordance.affordance_id,
                    "actor_id": affordance.actor_id,
                    "target_id": affordance.target_id,
                    "parameters": dict(affordance.parameters),
                    "predicted_reward": prediction.reward,
                    "predicted_success_probability": prediction.success_probability,
                    "predicted_uncertainty": prediction.uncertainty,
                    "uncertainty_mode": prediction.uncertainty_mode,
                }
                expected_outcome = (
                    f"model-predicted reward={prediction.reward:.6g}; "
                    f"success_probability={prediction.success_probability:.6g}; "
                    f"uncertainty={prediction.uncertainty:.6g}"
                )
                intent = ActionIntent(
                    intent_id=intent_id,
                    kind=affordance.action_kind,
                    parameters=action_parameters,
                    source_goal_id=context.goal_id,
                    expected_outcome=expected_outcome,
                    confidence=prediction.success_probability,
                    tick=root.world.tick,
                )
                content = ContentPlan(
                    content_id=f"{intent_id}:content",
                    intent_id=intent_id,
                    intent_kind=affordance.action_kind,
                    semantic_slots={
                        "action_id": action.action_id,
                        "action_kind": action.kind,
                        "actor_id": action.actor_id,
                        "target_id": action.target_id,
                        "predicted_reward": prediction.reward,
                        "predicted_success_probability": prediction.success_probability,
                        "predicted_state_sha256": content_digest(prediction.state.to_payload()),
                        "goal_target_sha256": goal_target.digest,
                        "goal_target": goal_target.to_payload(),
                        "world_model_sha256": world_digest,
                        "executive_sha256": executive_digest,
                        "world_schema_revision": world_model.schema_registry.active_version,
                    },
                    source_goal_id=context.goal_id,
                    expected_outcome=expected_outcome,
                    confidence=prediction.success_probability,
                    provenance="joint-world-prediction",
                    tick=root.world.tick,
                )
                features = (
                    goal_alignment,
                    min(1.0, max(0.0, state_change)),
                    information_gain,
                    prediction.success_probability,
                    prediction.uncertainty,
                    resource_cost,
                )
                candidate = ExecutiveCandidate(
                    candidate_id=candidate_id,
                    action_intent=intent,
                    content_plan=content,
                    features=features,
                    provenance="joint-world-prediction",
                    source_percept_id=(None if root.percept is None else root.percept.event_id),
                    source_affordance_id=affordance.affordance_id,
                )
                candidates.append(
                    JointWorldCandidatePredictionV1(
                        candidate=candidate,
                        action=action,
                        prediction=prediction,
                        goal_alignment=goal_alignment,
                        state_change=state_change,
                    )
                )
        finally:
            if world_model.training != was_training:
                world_model.train(was_training)

        if content_digest(world_model.snapshot()) != world_digest:
            world_model.restore(world_snapshot)
            if world_model.training != was_training:
                world_model.train(was_training)
            raise RuntimeError("read-only joint prediction changed world model state")
        if content_digest(executive.checkpoint()) != executive_digest:
            raise RuntimeError("read-only executive selection changed controller state")

        candidate_ids = [item.candidate.candidate_id for item in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("joint world affordances must produce unique candidate identities")
        decision = executive.select(tuple(item.candidate for item in candidates), context)
        if content_digest(executive.checkpoint()) != executive_digest:
            raise RuntimeError("read-only executive selection changed controller state")
        best_score = max(decision.scores.values())
        tied = tuple(
            candidate_id
            for candidate_id, score in decision.scores.items()
            if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-10)
        )
        if len(tied) != 1:
            raise ValueError("executive controller cannot distinguish predicted action candidates")
        selected = next(
            item
            for item in candidates
            if item.candidate.candidate_id == decision.selected.candidate_id
        )
        plan_candidates = tuple(
            PlanCandidate(
                plan_id=item.candidate.candidate_id,
                action_kind=item.action.kind,
                expected_value=float(decision.scores[item.candidate.candidate_id]),
                risk=1.0 - item.prediction.success_probability,
            )
            for item in candidates
        )
        selected_prediction = WorldPredictionRecord(
            action=selected.action,
            predicted_state=selected.prediction.state,
            predicted_reward=selected.prediction.reward,
            predicted_success_probability=selected.prediction.success_probability,
            online_update_count=world_model.online_updates,
            uncertainty=selected.prediction.uncertainty,
            uncertainty_mode=selected.prediction.uncertainty_mode,
        )
        updated_owner = replace(
            root,
            plan=PlanState(
                tick=root.tick,
                candidates=plan_candidates,
                selected_plan_id=selected.candidate.candidate_id,
            ),
            action_intent=decision.action_intent,
            world_prediction=selected_prediction,
        )
        updated_joint = joint_state.advance_cognitive_state(updated_owner)
        return cls(
            source_joint_revision=joint_state.joint_revision,
            joint_state=updated_joint,
            goal_id=context.goal_id,
            goal_target=goal_target,
            candidates=tuple(candidates),
            decision=decision,
            world_schema_revision=world_model.schema_registry.active_version,
            world_model_sha256=world_digest,
            executive_sha256=executive_digest,
        )

    @property
    def selected_prediction(self) -> JointWorldCandidatePredictionV1:
        return next(
            item
            for item in self.candidates
            if item.candidate.candidate_id == self.decision.selected.candidate_id
        )


@dataclass(frozen=True)
class JointActionReadV1:
    """A same-revision action intent/prediction projection for the V4 decoder."""

    joint_bundle_id: str
    owner_id: str
    scope_id: str
    session_id: str
    joint_revision: int
    cognitive_state_sha256: str
    action_intent_id: str
    selected_plan_id: str
    episode_memory_id: str | None
    source_neural_ids: tuple[str, ...]
    projection_bytes: bytes
    projection_sha256: str
    action_state_sha256: str

    @classmethod
    def from_joint_state(
        cls,
        state: JointTaskStateV1,
        *,
        episode_read: JointEpisodeReadV1 | None = None,
        memory_store: EpisodicMemoryStore | None = None,
    ) -> JointActionReadV1:
        state._validate_live_owner()
        if state.lifecycle != "active":
            raise ValueError("joint action read requires an active task")
        root = state.cognitive_state
        intent, prediction = root.action_intent, root.world_prediction
        if intent is None or prediction is None or root.plan.selected_plan_id is None:
            raise ValueError("joint action read requires the selected ActionIntent and prediction")
        candidate_id = intent.parameters.get("candidate_id")
        action_id = intent.parameters.get("source_world_action_id")
        goal_target_payload = intent.parameters.get("goal_target")
        goal_target_digest = intent.parameters.get("goal_target_sha256")
        input_ids = {item.input_id for item in state.input_bindings}
        if (
            candidate_id != root.plan.selected_plan_id
            or action_id != prediction.action.action_id
            or intent.source_goal_id is None
            or intent.source_goal_id not in {goal.goal_id for goal in root.goals.goals}
            or not isinstance(goal_target_payload, Mapping)
            or goal_target_payload.get("goal_id") != intent.source_goal_id
            or content_digest(goal_target_payload) != goal_target_digest
            or not set(goal_target_payload.get("source_input_ids", ())).issubset(input_ids)
        ):
            raise ValueError("selected plan, intent, prediction and goal target are not linked")
        if not state.input_bindings:
            raise ValueError("joint action read requires a current query source")

        joint_view = state.content_view()
        unresolved = tuple(
            f"{item.candidate_id}:{item.source_status}"
            for item in joint_view.selected_candidates
            if item.source_status != "source_linked"
        )
        if unresolved:
            raise ValueError(
                "joint action context has unlinked selected content: " + ", ".join(unresolved)
            )
        sources = [
            state.input_bindings[-1].neural_source_id,
            *joint_view.selected_neural_source_ids,
        ]
        if episode_read is not None:
            if memory_store is None:
                raise ValueError(
                    "joint action read with episodic context requires its memory store"
                )
            episode_read.validate_for(state, memory_store)
            sources.extend(episode_read.selected_neural_source_ids)
        elif memory_store is not None:
            raise ValueError("episodic memory store requires a joint episode read")
        source_ids = tuple(dict.fromkeys(sources))

        predicted_objects = [
            {
                "object_id": item.object_id,
                "attributes": _json_projection(dict(item.attributes)),
                "tags": list(item.tags),
            }
            for item in prediction.predicted_state.objects
        ]
        projection = {
            "format": "taiji-joint-action-condition-v1",
            "goal_id": intent.source_goal_id,
            "selected_plan_id": root.plan.selected_plan_id,
            "action_intent": {
                "intent_id": intent.intent_id,
                "kind": intent.kind,
                "parameters": _json_projection(intent.parameters),
                "expected_outcome": intent.expected_outcome,
                "confidence": intent.confidence,
            },
            "world_prediction": {
                "action_id": prediction.action.action_id,
                "predicted_reward": prediction.predicted_reward,
                "predicted_success_probability": prediction.predicted_success_probability,
                "uncertainty": prediction.uncertainty,
                "uncertainty_mode": prediction.uncertainty_mode,
                "objects": predicted_objects,
                "relations": _json_projection(prediction.predicted_state.relations),
            },
        }
        raw = json.dumps(
            projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        action_state_sha256 = content_digest(
            {
                "intent": intent.to_payload(),
                "prediction": prediction.to_payload(),
                "plan": root.plan.to_payload(),
            }
        )
        return cls(
            joint_bundle_id=state.joint_bundle_id,
            owner_id=state.owner_id,
            scope_id=state.scope_id,
            session_id=state.session_id,
            joint_revision=state.joint_revision,
            cognitive_state_sha256=state.cognitive_state_sha256,
            action_intent_id=intent.intent_id,
            selected_plan_id=root.plan.selected_plan_id,
            episode_memory_id=None if episode_read is None else episode_read.memory_id,
            source_neural_ids=source_ids,
            projection_bytes=raw,
            projection_sha256=hashlib.sha256(raw).hexdigest(),
            action_state_sha256=action_state_sha256,
        )

    def validate_for(
        self,
        state: JointTaskStateV1,
        *,
        episode_read: JointEpisodeReadV1 | None = None,
        memory_store: EpisodicMemoryStore | None = None,
    ) -> None:
        if (
            type(self).from_joint_state(state, episode_read=episode_read, memory_store=memory_store)
            != self
        ):
            raise ValueError("joint action read differs from the current canonical selection")

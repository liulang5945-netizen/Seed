"""Joint task frames bind one canonical cognitive owner to versioned neural evidence."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from taiji.adapter import TSKV8Adapter
from taiji.conditional_language_v3 import TaskCoreConfigV3
from taiji.conditional_language_v4 import (
    ConditionalLanguageCandidateV4,
    ConditionalLanguageTrainerV4,
)
from taiji.config import TaijiConfig
from taiji.contracts import (
    ActionIntent,
    Assembly,
    CognitiveState,
    DevelopmentState,
    EpisodicMemoryRecord,
    Event,
    Goal,
    GoalState,
    HomeostaticState,
    LearningState,
    MemoryState,
    NativeCheckpoint,
    Observation,
    Outcome,
    PerceptEvent,
    PlanCandidate,
    PlanState,
    SelfState,
    WorkspaceCandidate,
    WorkspaceSelection,
    WorkspaceState,
    WorldAction,
    WorldAffordance,
    WorldEvent,
    WorldObject,
    WorldPredictionRecord,
    WorldState,
    WorldTransition,
)
from taiji.episodic_memory import EpisodicMemoryStore
from taiji.executive import ExecutiveController
from taiji.input_boundary import InputFrame, InputTrace
from taiji.internalization import content_digest
from taiji.joint_action import (
    JointActionReadV1,
    JointGoalWorldTargetV1,
    JointWorldDecisionV1,
)
from taiji.joint_learning import JointCreditRouteV1, JointLanguageFeedbackV1
from taiji.joint_task_state import JointDecisionTraceV1, JointEpisodeReadV1, JointTaskStateV1
from taiji.world_learning import WorldDynamicsLearner, WorldSchema


def make_trace(text: str, *, input_id: str, source: str = "client", timestamp: int = 4):
    raw = text.encode("utf-8")
    observations = tuple(
        Observation("text-byte", value, timestamp, source, "external") for value in raw
    )
    assembly_id = f"assembly:{input_id}"
    percepts = tuple(
        PerceptEvent(
            event_id=f"percept:{input_id}:{i}",
            assembly_id=assembly_id,
            observation_tick=timestamp,
            modality="text-byte",
            features=torch.tensor([float(value), float(i)], dtype=torch.float32),
        )
        for i, value in enumerate(raw)
    )
    return raw, InputTrace(input_id, "text", observations, percepts)


def make_root(
    trace: InputTrace,
    text: str,
    *,
    task_id: str = "task-1",
    color: str = "蓝色",
    goal_id: str = "goal-1",
    object_id: str = "object-1",
):
    raw, _ = make_trace(
        text,
        input_id=trace.input_id,
        source=trace.observations[0].source,
        timestamp=trace.observations[0].timestamp,
    )
    candidate_id = "candidate-object-1"
    assembly_id = trace.percepts[-1].assembly_id
    world_event_id = f"fact:{trace.input_id}:{color}"
    observation = trace.observations[-1]
    percept = trace.percepts[-1]
    selection = WorkspaceSelection(
        tick=observation.timestamp,
        mode="learned",
        candidate_ids=(candidate_id,),
        selected_ids=(candidate_id,),
        scores=(0.8,),
        broadcast=torch.tensor([0.5, 0.5], dtype=torch.float32),
        capacity=1,
    )
    return CognitiveState(
        episode_id=task_id,
        tick=observation.timestamp,
        observation=observation,
        percept=percept,
        workspace=WorkspaceState(
            tick=observation.timestamp,
            focus=(candidate_id,),
            bindings=((candidate_id, object_id),),
            broadcast=torch.tensor([0.5, 0.5], dtype=torch.float32),
            capacity=1,
            candidates=(
                WorkspaceCandidate(
                    candidate_id,
                    torch.tensor([0.5, 0.5], dtype=torch.float32),
                    source=trace.input_id,
                ),
            ),
            selection=selection,
            percept_event_id=percept.event_id,
            percept_assembly_id=assembly_id,
        ),
        world=WorldState(
            tick=observation.timestamp,
            objects=(WorldObject(object_id, attributes=(("color", color),)),),
            events=(
                WorldEvent(
                    event_id=world_event_id,
                    kind="observed_attribute",
                    tick=observation.timestamp,
                    subject_id=object_id,
                    object_id=object_id,
                    attributes=(("color", color),),
                    provenance=trace.input_id,
                ),
            ),
        ),
        memory=MemoryState(tick=observation.timestamp, episodic_ids=("episode-1",)),
        goals=GoalState(tick=observation.timestamp, goals=(Goal(goal_id, "answer user"),)),
        plan=PlanState(
            tick=observation.timestamp, candidates=(PlanCandidate(f"plan-{goal_id}", "answer"),)
        ),
        self_state=SelfState(tick=observation.timestamp),
        homeostasis=HomeostaticState(tick=observation.timestamp),
        development=DevelopmentState(tick=observation.timestamp),
        learning=LearningState(tick=observation.timestamp),
        events=(
            Event(
                event_id=world_event_id,
                start_tick=observation.timestamp,
                end_tick=observation.timestamp,
                assembly_ids=(assembly_id,),
                object_ids=(object_id,),
            ),
        ),
        assemblies=(
            Assembly(
                assembly_id=assembly_id,
                start_tick=observation.timestamp,
                end_tick=observation.timestamp,
                activity=torch.tensor([0.5, 0.5], dtype=torch.float32),
                source_event_ids=tuple(p.event_id for p in trace.percepts),
                provenance="learned",
            ),
        ),
    )


def setup_joint(
    text="杯子是蓝色的.",
    *,
    owner="user-1",
    scope="scope-1",
    session="session-1",
    goal_id="goal-1",
    max_observation_bytes=256,
):
    _, trace = make_trace(text, input_id=f"input-{session}-{text}")
    root = make_root(trace, text, goal_id=goal_id)
    model = ConditionalLanguageCandidateV4(
        TaskCoreConfigV3(
            width=8,
            embedding_width=4,
            key_width=4,
            slots=4,
            chunk_bytes=4,
            max_observation_bytes=max_observation_bytes,
            max_input_evidence_bytes=max_observation_bytes,
            max_output_position=128,
            max_generation_bytes=128,
        )
    )
    joint = JointTaskStateV1.begin(
        model,
        root,
        joint_bundle_id="joint-v1:test",
        owner_id=owner,
        scope_id=scope,
        session_id=session,
    )
    return model, root, trace, joint


def setup_two_source_joint(*, scope="scope-1", goal_id="goal-1", session="source-pair"):
    model, blue_root, blue_trace, initial = setup_joint(
        text="杯子是蓝色的。", scope=scope, session=session, goal_id=goal_id
    )
    budget = {"remaining_steps": 1, "max_observation_bytes": 256}
    one_source = initial.append_input(model, blue_trace, blue_root, resource_snapshot=budget)
    red_trace = make_trace("杯子是红色的。", input_id="input-source-red", timestamp=5)[1]
    red_root = make_root(
        red_trace,
        "杯子是红色的。",
        task_id=one_source.task_id,
        color="红色",
        goal_id=goal_id,
        object_id="object-red",
    )
    candidate_blue = "candidate-blue"
    candidate_red = "candidate-red"
    broadcast = torch.tensor([0.5, 0.5], dtype=torch.float32)
    workspace = WorkspaceState(
        tick=red_root.tick,
        focus=(candidate_blue,),
        bindings=((candidate_blue, "object-1"), (candidate_red, "object-red")),
        broadcast=broadcast,
        capacity=1,
        candidates=(
            WorkspaceCandidate(
                candidate_blue,
                torch.tensor([1.0, 0.0], dtype=torch.float32),
                source=blue_trace.input_id,
            ),
            WorkspaceCandidate(
                candidate_red,
                torch.tensor([0.0, 1.0], dtype=torch.float32),
                source=red_trace.input_id,
            ),
        ),
        selection=WorkspaceSelection(
            tick=red_root.tick,
            mode="learned",
            candidate_ids=(candidate_blue, candidate_red),
            selected_ids=(candidate_blue,),
            scores=(1.0, 0.0),
            broadcast=broadcast,
            capacity=1,
        ),
        percept_event_id=red_trace.percepts[-1].event_id,
        percept_assembly_id=red_trace.percepts[-1].assembly_id,
    )
    red_root = replace(
        red_root,
        workspace=workspace,
        world=replace(
            red_root.world,
            entities=("object-1", "object-red"),
            objects=blue_root.world.objects + red_root.world.objects,
            events=blue_root.world.events + red_root.world.events,
        ),
        events=blue_root.events + red_root.events,
        assemblies=blue_root.assemblies + red_root.assemblies,
    )
    two_sources = one_source.append_input(model, red_trace, red_root, resource_snapshot=budget)
    return model, red_root, two_sources


def select_workspace_candidate(root: CognitiveState, selected_id: str) -> CognitiveState:
    workspace = root.workspace
    assert workspace.selection is not None
    score_by_id = dict(
        zip(workspace.selection.candidate_ids, workspace.selection.scores, strict=True)
    )
    ordered_scores = tuple(
        score_by_id[candidate_id] for candidate_id in workspace.selection.candidate_ids
    )
    selection = replace(
        workspace.selection,
        selected_ids=(selected_id,),
        scores=ordered_scores,
    )
    return replace(
        root,
        workspace=replace(workspace, focus=(selected_id,), selection=selection),
    )


def test_same_cognitive_owner_workspace_and_provenance_reach_the_joint_view():
    model, root, trace, joint = setup_joint()
    _, trace2 = make_trace("杯子是蓝色的。", input_id=trace.input_id)
    root = make_root(trace2, "杯子是蓝色的。")
    joined = joint.append_input(
        model, trace2, root, resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256}
    )
    view = joined.content_view()
    assert joined.cognitive_state is root
    assert joined.cognitive_state_sha256 != joint.cognitive_state_sha256
    assert joined.joint_revision == joint.joint_revision + 1
    assert len(joined.input_bindings) == 1
    assert joined.input_bindings[0].byte_count == len("杯子是蓝色的。".encode())
    assert view.selected_candidates[0].candidate_id == "candidate-object-1"
    assert view.world_objects[0] is root.world.objects[0]
    assert view.world_events[0].event_id == root.world.events[0].event_id
    assert joined.input_bindings[0].assembly_ids == (trace.percepts[0].assembly_id,)
    assert view.available_episode_refs == root.memory.episodic_ids
    assert view.action_intent_id is None and view.outcome_intent_id is None


def test_raw_input_cognitive_snapshot_and_neural_state_must_share_one_trace():
    model, root, trace, joint = setup_joint()
    _, wrong_trace = make_trace("桌上有杯子。", input_id="input-other")
    root_wrong = make_root(wrong_trace, "桌上有杯子。")
    with pytest.raises(ValueError, match="InputTrace"):
        joint.append_input(
            model,
            trace,
            root_wrong,
            resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
        )
    with pytest.raises(ValueError, match="already consumed"):
        joined = joint.append_input(
            model,
            trace,
            root,
            resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
        )
        joined.append_input(
            model,
            trace,
            root,
            resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
        )


def test_unknown_workspace_candidate_is_reported_not_falsely_read_as_content():
    model, root, trace, joint = setup_joint()
    workspace = replace(root.workspace, focus=("predictive-context",), selection=None, bindings=())
    root = replace(root, workspace=workspace)
    unbound = JointTaskStateV1.begin(
        model,
        root,
        joint_bundle_id="joint-v1:test",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="session-1",
    )
    view = unbound.content_view()
    assert view.selected_candidates == ()
    assert view.unresolved_focus_ids == ("predictive-context",)


def test_root_mutation_invalidates_joint_projection():
    model, root, trace, joint = setup_joint()
    changed = replace(root, world=replace(root.world, uncertainty=0.25))
    advanced = joint.advance_cognitive_state(changed)
    assert advanced.content_view().state_revision == 1
    root.workspace.candidates[0].features.add_(1.0)
    with pytest.raises(ValueError, match="changed outside the joint revision"):
        advanced.content_view()


def test_turn_reset_preserves_cognitive_owner_and_provenance_but_clears_decoder_cursor():
    model, root, trace, joint = setup_joint()
    joined = joint.append_input(
        model, trace, root, resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256}
    )
    neural = replace(
        joined.language_state,
        generated=b"A",
        previous_byte=ord("A"),
        decoder_hidden=torch.ones_like(joined.language_state.hidden),
    )
    joined = replace(joined, language_state=neural)
    reset = joined.reset_turn()
    assert reset.cognitive_state is root
    assert reset.input_bindings == joined.input_bindings
    assert reset.language_state.generated == b"" and reset.language_state.decoder_hidden is None
    assert reset.language_state.observations == joined.language_state.observations


def test_session_reset_requires_explicit_exact_memory_carry_and_resets_neural_evidence():
    model, root, trace, joint = setup_joint()
    joined = joint.append_input(
        model, trace, root, resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256}
    )
    new_trace = make_trace("新的任务。", input_id="input-session-2")[1]
    new_root = make_root(new_trace, "新的任务。", task_id="task-2")
    # Deliberately repeat one episodic reference: carry must be authorized.
    new_root = replace(new_root, memory=replace(new_root.memory, episodic_ids=("episode-1",)))
    old_refs = JointTaskStateV1.cognitive_reference_ids(joined.cognitive_state)
    new_refs = JointTaskStateV1.cognitive_reference_ids(new_root)
    carries = tuple(sorted(old_refs & new_refs))
    with pytest.raises(ValueError, match="carry"):
        joined.new_session(model, new_root, session_id="session-2", authorized_carry_refs=())
    carried = joined.new_session(
        model, new_root, session_id="session-2", authorized_carry_refs=carries
    )
    assert carried.session_id == "session-2" and carried.task_id == "task-2"
    assert carried.input_bindings == () and carried.language_state.observations == ()


def test_action_failure_replans_unknown_blocks_replay_and_only_terminal_outcome_closes_task():
    model, root, trace, joint = setup_joint()
    intent = ActionIntent("intent-1", "move", tick=root.tick)
    pending = replace(root, action_intent=intent, outcome=None)
    proposed = joint.advance_cognitive_state(pending)
    assert proposed.action_receipt is not None
    assert proposed.action_receipt.status == "proposed"
    frame, idempotency_key = proposed.begin_action_dispatch("intent-1")
    assert frame.action_receipt.status == "dispatch_started"
    assert frame.action_receipt.execution_id == idempotency_key
    with pytest.raises(ValueError, match="proposed ActionIntent"):
        frame.begin_action_dispatch("intent-1")
    with pytest.raises(ValueError, match="non-generated"):
        frame.record_action_outcome(
            replace(pending, outcome=Outcome("intent-1", 0.0, provenance="generated")),
            "intent-1",
            execution_id=idempotency_key,
        )
    accepted = frame.mark_action_execution_accepted("intent-1", "executor-1")
    assert accepted.action_receipt.status == "awaiting_outcome"
    assert accepted.lifecycle == "awaiting_outcome"
    with pytest.raises(ValueError, match="dispatch start"):
        accepted.mark_action_execution_accepted("intent-1", "executor-2")
    failure = Outcome(
        "intent-1",
        reward=-1.0,
        success=False,
        terminal=False,
        tick=root.tick,
        provenance="experienced",
    )
    retry = accepted.record_action_outcome(
        replace(pending, outcome=failure), "intent-1", execution_id="executor-1"
    )
    assert retry.lifecycle == "active" and retry.cognitive_state.outcome.success is False
    assert retry.action_receipt.status == "settled"
    assert (
        retry.record_action_outcome(
            replace(pending, outcome=failure), "intent-1", execution_id="executor-1"
        )
        is retry
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        retry.record_action_outcome(
            replace(pending, outcome=replace(failure, reward=-2.0)),
            "intent-1",
            execution_id="executor-1",
        )
    unknown = Outcome(
        "intent-1",
        reward=0.0,
        success=None,
        terminal=False,
        tick=root.tick,
        provenance="experienced",
    )
    unknown_start = proposed.begin_action_dispatch("intent-1")[0].mark_action_execution_accepted(
        "intent-1", "executor-unknown"
    )
    awaiting = unknown_start.record_action_outcome(
        replace(unknown_start.cognitive_state, outcome=unknown),
        "intent-1",
        execution_id="executor-unknown",
    )
    assert awaiting.lifecycle == "awaiting_outcome"
    assert awaiting.action_receipt.status == "outcome_unknown"
    assert (
        awaiting.record_action_outcome(
            replace(unknown_start.cognitive_state, outcome=unknown),
            "intent-1",
            execution_id="executor-unknown",
        )
        is awaiting
    )
    with pytest.raises(ValueError, match="conflicting duplicate unknown"):
        awaiting.record_action_outcome(
            replace(unknown_start.cognitive_state, outcome=replace(unknown, reward=0.25)),
            "intent-1",
            execution_id="executor-unknown",
        )
    with pytest.raises(ValueError, match="awaiting_outcome"):
        awaiting.append_input(
            model,
            trace,
            awaiting.cognitive_state,
            resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
        )
    resolved = Outcome(
        "intent-1",
        reward=1.0,
        success=True,
        terminal=True,
        tick=root.tick,
        provenance="experienced",
    )
    settled = awaiting.record_action_outcome(
        replace(awaiting.cognitive_state, outcome=resolved),
        "intent-1",
        execution_id="executor-unknown",
    )
    assert settled.lifecycle == "complete"
    generated = replace(failure, provenance="generated")
    with pytest.raises(ValueError, match="non-generated"):
        frame.record_action_outcome(
            replace(pending, outcome=generated),
            "intent-1",
            execution_id=idempotency_key,
        )


def test_action_outcome_can_settle_after_response_completion():
    _, root, _, joint = setup_joint()
    intent = ActionIntent("intent-after-response", "move", tick=root.tick)
    proposed = joint.advance_cognitive_state(replace(root, action_intent=intent, outcome=None))
    neural = replace(
        proposed.language_state,
        generated=b"<|end|>",
        previous_byte=ord(">"),
        decoder_hidden=torch.zeros_like(proposed.language_state.hidden),
    )
    response = proposed.with_language_state(neural).complete_response()
    dispatch, _ = response.begin_action_dispatch(intent.intent_id)
    accepted = dispatch.mark_action_execution_accepted(intent.intent_id, "executor-after-response")
    result = Outcome(intent.intent_id, -0.5, success=False, terminal=False, tick=root.tick)
    settled = accepted.record_action_outcome(
        replace(accepted.cognitive_state, outcome=result),
        intent.intent_id,
        execution_id="executor-after-response",
    )
    assert settled.lifecycle == "active"
    assert settled.action_receipt.status == "settled"


def test_world_transition_and_cognitive_outcome_must_agree_before_settlement():
    _, root, _, joint = setup_joint()
    intent = ActionIntent(
        "intent-transition",
        "move",
        parameters={"source_world_action_id": "world-action-transition"},
        tick=root.tick,
    )
    proposed = joint.advance_cognitive_state(replace(root, action_intent=intent, outcome=None))
    dispatch, _ = proposed.begin_action_dispatch(intent.intent_id)
    accepted = dispatch.mark_action_execution_accepted(intent.intent_id, "executor-transition")
    after = replace(root.world, tick=root.world.tick + 1)
    action = WorldAction(
        "world-action-transition", "move", tick=root.world.tick, provenance="experienced"
    )
    transition_outcome = Outcome(
        action.action_id,
        1.0,
        success=True,
        terminal=False,
        tick=after.tick,
        provenance="experienced",
    )
    transition = WorldTransition(root.world, action, after, transition_outcome)
    prediction = WorldPredictionRecord(
        action=replace(action, provenance="planned"),
        predicted_state=after,
        predicted_reward=0.8,
        predicted_success_probability=0.9,
    )
    contradictory = Outcome(
        intent.intent_id,
        -1.0,
        success=False,
        terminal=False,
        tick=after.tick,
        provenance="experienced",
    )
    invalid_owner = replace(
        accepted.cognitive_state,
        tick=after.tick,
        world=after,
        world_prediction=prediction,
        world_transition=transition,
        outcome=contradictory,
    )
    same_id_wrong_action = replace(action, kind="open")
    wrong_action_transition = replace(transition, action=same_id_wrong_action)
    with pytest.raises(ValueError, match="selected ActionIntent"):
        accepted.record_action_outcome(
            replace(invalid_owner, world_transition=wrong_action_transition),
            intent.intent_id,
            execution_id="executor-transition",
        )
    with pytest.raises(ValueError, match="matching selected prediction"):
        accepted.record_action_outcome(
            replace(invalid_owner, world_prediction=None),
            intent.intent_id,
            execution_id="executor-transition",
        )
    inconsistent_after = replace(after, tick=after.tick + 1)
    with pytest.raises(ValueError, match="after-state"):
        accepted.record_action_outcome(
            replace(invalid_owner, world=inconsistent_after),
            intent.intent_id,
            execution_id="executor-transition",
        )
    with pytest.raises(ValueError, match="disagree"):
        accepted.record_action_outcome(
            invalid_owner,
            intent.intent_id,
            execution_id="executor-transition",
        )
    matching = replace(contradictory, reward=1.0, success=True)
    settled = accepted.record_action_outcome(
        replace(invalid_owner, outcome=matching),
        intent.intent_id,
        execution_id="executor-transition",
    )
    assert settled.action_receipt.status == "settled"


def test_response_end_marker_closes_a_turn_and_a_follow_up_continues_the_same_task():
    model, root, trace, frame = setup_joint()
    neural = replace(
        frame.language_state,
        generated=b"<|end|>",
        previous_byte=ord(">"),
        decoder_hidden=torch.zeros_like(frame.language_state.hidden),
    )
    response = frame.with_language_state(neural).complete_response()
    assert response.lifecycle == "response_complete"
    with pytest.raises(ValueError, match="lifecycle"):
        model.next_logits_from_joint(response)
    next_trace = make_trace("还有问题。", input_id="second-turn")[1]
    next_root = make_root(next_trace, "还有问题。", task_id=response.task_id)
    resumed = response.append_input(
        model,
        next_trace,
        next_root,
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
    )
    assert resumed.lifecycle == "active" and resumed.joint_revision == response.joint_revision + 1


def test_versioned_joint_payload_roundtrips_and_rejects_owner_neural_or_cognitive_drift():
    model, root, trace, joint = setup_joint()
    joined = joint.append_input(
        model, trace, root, resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256}
    )
    payload = joined.to_payload()
    restored = JointTaskStateV1.from_payload(
        payload,
        joint_bundle_id="joint-v1:test",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="session-1",
        candidate_bundle_id=model.bundle_id,
    )
    assert restored.to_payload()["cognitive_state_sha256"] == joined.cognitive_state_sha256
    legacy_v2_payload = copy.deepcopy(payload)
    legacy_v2_payload["format"] = "taiji-joint-task-state-v2"
    legacy_v2_payload["version"] = 2
    legacy_v2_payload.pop("action_receipts")
    legacy_v2_payload.pop("decision_trace")
    migrated_v2 = JointTaskStateV1.from_payload(
        legacy_v2_payload,
        joint_bundle_id="joint-v1:test",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="session-1",
        candidate_bundle_id=model.bundle_id,
    )
    assert migrated_v2.to_payload()["format"] == "taiji-joint-task-state-v3"
    legacy_payload = copy.deepcopy(payload)
    legacy_payload["format"] = "taiji-joint-task-state-v1"
    legacy_payload["version"] = 1
    legacy_payload.pop("action_receipts")
    legacy_payload.pop("decision_trace")
    for binding in legacy_payload["input_bindings"]:
        binding.pop("percept_tick_start", None)
        binding.pop("percept_tick_end", None)
    migrated = JointTaskStateV1.from_payload(
        legacy_payload,
        joint_bundle_id="joint-v1:test",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="session-1",
        candidate_bundle_id=model.bundle_id,
    )
    assert migrated.to_payload()["format"] == "taiji-joint-task-state-v3"
    assert migrated.input_bindings[0].percept_tick_start is None
    with pytest.raises(ValueError, match="owner"):
        JointTaskStateV1.from_payload(
            payload,
            joint_bundle_id="joint-v1:test",
            owner_id="user-2",
            scope_id="scope-1",
            session_id="session-1",
            candidate_bundle_id=model.bundle_id,
        )
    damaged = copy.deepcopy(payload)
    damaged["cognitive_state"]["world"]["tick"] += 1
    with pytest.raises(ValueError, match="digest"):
        JointTaskStateV1.from_payload(
            damaged,
            joint_bundle_id="joint-v1:test",
            owner_id="user-1",
            scope_id="scope-1",
            session_id="session-1",
            candidate_bundle_id=model.bundle_id,
        )


def test_one_native_checkpoint_contains_one_cognitive_owner_and_its_candidate_optimizer():
    model, root, trace, joint = setup_joint()
    joined = joint.append_input(
        model, trace, root, resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256}
    )
    trainer = ConditionalLanguageTrainerV4(model)
    language_checkpoint = trainer.checkpoint(joined.language_state, cursor={"step": 0})
    native = NativeCheckpoint(kernel={"format": "synthetic-task-kernel-v1"}, cognitive_state=root)
    envelope = joined.attach_to_native_checkpoint(native.to_payload(), language_checkpoint)
    restored, candidate = JointTaskStateV1.restore_native_checkpoint(
        envelope,
        joint_bundle_id=joined.joint_bundle_id,
        owner_id=joined.owner_id,
        scope_id=joined.scope_id,
        session_id=joined.session_id,
    )
    assert restored.cognitive_state_sha256 == joined.cognitive_state_sha256
    assert (
        candidate.steps == 0
        and restored.language_state.observations == joined.language_state.observations
    )
    legacy_envelope = copy.deepcopy(envelope)
    for version, component_key, component_format in (
        (2, "joint_task_state_v2", "taiji-joint-cognitive-component-v2"),
        (1, "joint_task_state_v1", "taiji-joint-cognitive-component-v1"),
    ):
        legacy_envelope = copy.deepcopy(envelope)
        legacy_component = legacy_envelope["components"].pop("joint_task_state_v3")
        legacy_component["format"] = component_format
        legacy_component["frame"]["format"] = f"taiji-joint-task-state-v{version}"
        legacy_component["frame"]["version"] = version
        legacy_component["frame"].pop("action_receipts")
        legacy_component["frame"].pop("decision_trace")
        if version == 1:
            for binding in legacy_component["frame"]["input_bindings"]:
                binding.pop("percept_tick_start", None)
                binding.pop("percept_tick_end", None)
        legacy_component.pop("component_digest")
        legacy_component["component_digest"] = content_digest(legacy_component)
        legacy_envelope["components"][component_key] = legacy_component
        migrated, _ = JointTaskStateV1.restore_native_checkpoint(
            legacy_envelope,
            joint_bundle_id=joined.joint_bundle_id,
            owner_id=joined.owner_id,
            scope_id=joined.scope_id,
            session_id=joined.session_id,
        )
        assert migrated.to_payload()["format"] == "taiji-joint-task-state-v3"
        assert migrated.action_receipts == () and migrated.decision_trace is None
        if version == 1:
            assert migrated.input_bindings[0].percept_tick_start is None
    damaged = copy.deepcopy(envelope)
    damaged["components"]["joint_task_state_v3"]["frame"]["owner_id"] = "user-other"
    with pytest.raises(ValueError, match="component digest"):
        JointTaskStateV1.restore_native_checkpoint(
            damaged,
            joint_bundle_id=joined.joint_bundle_id,
            owner_id=joined.owner_id,
            scope_id=joined.scope_id,
            session_id=joined.session_id,
        )
    mismatched_frame = copy.deepcopy(envelope)
    mismatched_component = mismatched_frame["components"]["joint_task_state_v3"]
    mismatched_component["frame"]["format"] = "taiji-joint-task-state-v2"
    mismatched_component["frame"]["version"] = 2
    mismatched_component["frame"].pop("action_receipts")
    mismatched_component["frame"].pop("decision_trace")
    mismatched_component.pop("component_digest")
    mismatched_component["component_digest"] = content_digest(mismatched_component)
    with pytest.raises(ValueError, match="versions differ"):
        JointTaskStateV1.restore_native_checkpoint(
            mismatched_frame,
            joint_bundle_id=joined.joint_bundle_id,
            owner_id=joined.owner_id,
            scope_id=joined.scope_id,
            session_id=joined.session_id,
        )
    duplicate_component = copy.deepcopy(envelope)
    duplicate_component["components"]["joint_task_state_v2"] = copy.deepcopy(
        duplicate_component["components"]["joint_task_state_v3"]
    )
    with pytest.raises(ValueError, match="exactly one"):
        JointTaskStateV1.restore_native_checkpoint(
            duplicate_component,
            joint_bundle_id=joined.joint_bundle_id,
            owner_id=joined.owner_id,
            scope_id=joined.scope_id,
            session_id=joined.session_id,
        )


@pytest.mark.parametrize(
    "stage,expected_status",
    [
        ("proposed", "proposed"),
        ("dispatch_started", "dispatch_started"),
        ("awaiting_outcome", "awaiting_outcome"),
        ("outcome_unknown", "outcome_unknown"),
        ("settled", "settled"),
        ("cancelled", "cancelled"),
    ],
)
@pytest.mark.parametrize("scope", ["scope-a", "scope-b"])
def test_action_execution_receipt_survives_native_checkpoint(stage, expected_status, scope):
    model, root, _, joint = setup_joint(scope=scope, session=f"session-{scope}")
    intent = ActionIntent("intent-checkpoint", "move", tick=root.tick)
    frame = joint.advance_cognitive_state(replace(root, action_intent=intent, outcome=None))
    idempotency_key = None
    execution_id = None
    if stage not in {"proposed", "cancelled"}:
        frame, idempotency_key = frame.begin_action_dispatch(intent.intent_id)
    if stage in {"awaiting_outcome", "outcome_unknown", "settled"}:
        execution_id = "executor-checkpoint"
        frame = frame.mark_action_execution_accepted(intent.intent_id, execution_id)
    if stage == "outcome_unknown":
        unknown = Outcome(intent.intent_id, 0.0, success=None, provenance="experienced")
        frame = frame.record_action_outcome(
            replace(frame.cognitive_state, outcome=unknown),
            intent.intent_id,
            execution_id=execution_id,
        )
    elif stage == "settled":
        result = Outcome(intent.intent_id, 1.0, success=True, terminal=False)
        frame = frame.record_action_outcome(
            replace(frame.cognitive_state, outcome=result),
            intent.intent_id,
            execution_id=execution_id,
        )
    elif stage == "cancelled":
        frame = frame.cancel_action_proposal(intent.intent_id)
    trainer = ConditionalLanguageTrainerV4(model)
    language_checkpoint = trainer.checkpoint(frame.language_state, cursor={"step": 0})
    native = NativeCheckpoint(
        kernel={"format": "synthetic-task-kernel-v1"},
        cognitive_state=frame.cognitive_state,
    )
    envelope = frame.attach_to_native_checkpoint(native.to_payload(), language_checkpoint)
    restored, _ = JointTaskStateV1.restore_native_checkpoint(
        envelope,
        joint_bundle_id=frame.joint_bundle_id,
        owner_id=frame.owner_id,
        scope_id=frame.scope_id,
        session_id=frame.session_id,
    )
    assert restored.action_receipt is not None
    assert restored.action_receipt.status == expected_status
    if stage == "proposed":
        _, restored_key = restored.begin_action_dispatch(intent.intent_id)
        assert restored_key == restored.action_receipt.idempotency_key
    elif stage == "dispatch_started":
        assert restored.action_receipt.execution_id == idempotency_key
    elif stage in {"awaiting_outcome", "outcome_unknown", "settled"}:
        assert restored.action_receipt.execution_id == execution_id


def test_unknown_action_checkpoint_restores_in_a_fresh_process(tmp_path):
    model, root, _, joint = setup_joint(scope="scope-fresh", session="session-fresh")
    intent = ActionIntent("intent-fresh-restore", "move", tick=root.tick)
    proposed = joint.advance_cognitive_state(replace(root, action_intent=intent, outcome=None))
    dispatch, idempotency_key = proposed.begin_action_dispatch(intent.intent_id)
    accepted = dispatch.mark_action_execution_accepted(intent.intent_id, "executor-fresh-restore")
    unknown = Outcome(intent.intent_id, 0.0, success=None, provenance="experienced")
    pending = accepted.record_action_outcome(
        replace(accepted.cognitive_state, outcome=unknown),
        intent.intent_id,
        execution_id="executor-fresh-restore",
    )
    trainer = ConditionalLanguageTrainerV4(model)
    language_checkpoint = trainer.checkpoint(pending.language_state, cursor={"step": 0})
    native = NativeCheckpoint(
        kernel={"format": "synthetic-task-kernel-v1"},
        cognitive_state=pending.cognitive_state,
    )
    envelope = pending.attach_to_native_checkpoint(native.to_payload(), language_checkpoint)
    checkpoint_path = tmp_path / "joint-outcome-unknown.pt"
    torch.save(envelope, checkpoint_path)
    child = """
import json, sys, torch
from taiji.joint_task_state import JointTaskStateV1
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state, trainer = JointTaskStateV1.restore_native_checkpoint(
    payload,
    joint_bundle_id='joint-v1:test',
    owner_id='user-1',
    scope_id='scope-fresh',
    session_id='session-fresh',
)
print(json.dumps({
    'status': state.action_receipt.status,
    'execution_id': state.action_receipt.execution_id,
    'idempotency_key': state.action_receipt.idempotency_key,
    'candidate_steps': trainer.steps,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(checkpoint_path)],
        cwd=str(tmp_path.parents[2]),
        check=True,
        capture_output=True,
        text=True,
    )
    restored = json.loads(result.stdout.strip())
    assert restored == {
        "status": "outcome_unknown",
        "execution_id": "executor-fresh-restore",
        "idempotency_key": idempotency_key,
        "candidate_steps": 0,
    }


@pytest.mark.parametrize("scope", ["scope-a", "scope-b"])
@pytest.mark.parametrize("goal", ["goal-x", "goal-y"])
@pytest.mark.parametrize(
    "event",
    ["turn_reset", "new_session", "cancel", "new_fact", "restore"],
)
def test_twenty_scope_goal_lifecycle_contract_cases(scope, goal, event):
    model, root, trace, initial = setup_joint(scope=scope, session=f"session-{scope}", goal_id=goal)
    frame = initial.append_input(
        model, trace, root, resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256}
    )
    if event == "turn_reset":
        result = frame.reset_turn()
        assert result.cognitive_state is root and result.input_bindings == frame.input_bindings
        assert result.language_state.workspace is frame.language_state.workspace
        return
    if event == "new_session":
        new_trace = make_trace("任务继续。", input_id=f"new-{scope}-{goal}")[1]
        new_root = make_root(new_trace, "任务继续。", task_id=f"task-{scope}-{goal}", goal_id=goal)
        new_root = replace(
            new_root, memory=replace(new_root.memory, episodic_ids=root.memory.episodic_ids)
        )
        allowed = tuple(
            sorted(
                JointTaskStateV1.cognitive_reference_ids(root)
                & JointTaskStateV1.cognitive_reference_ids(new_root)
            )
        )
        result = frame.new_session(
            model,
            new_root,
            session_id=f"next-{scope}-{goal}",
            authorized_carry_refs=allowed,
        )
        assert result.input_bindings == () and result.language_state.observations == ()
        assert result.task_id == new_root.episode_id and result.owner_id == frame.owner_id
        return
    if event == "cancel":
        result = frame.cancel_task()
        assert result.lifecycle == "cancelled" and result.cognitive_state is root
        with pytest.raises(ValueError, match="cancelled"):
            result.append_input(
                model,
                trace,
                root,
                resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
            )
        return
    if event == "new_fact":
        updated_trace = make_trace("杯子现在是红色。", input_id=f"update-{scope}-{goal}")[1]
        updated_root = make_root(
            updated_trace,
            "杯子现在是红色。",
            task_id=root.episode_id,
            color="红色",
            goal_id=goal,
        )
        result = frame.append_input(
            model,
            updated_trace,
            updated_root,
            resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
        )
        assert result.content_view().world_objects[0].attribute("color") == "红色"
        assert result.joint_revision == frame.joint_revision + 1
        return
    result = JointTaskStateV1.from_payload(
        frame.to_payload(),
        joint_bundle_id=frame.joint_bundle_id,
        owner_id=frame.owner_id,
        scope_id=frame.scope_id,
        session_id=frame.session_id,
        candidate_bundle_id=model.bundle_id,
    )
    assert content_digest(result.to_payload()) == content_digest(frame.to_payload())


def test_joint_content_read_view_maps_selected_fact_to_exact_neural_source():
    _, _, joint = setup_two_source_joint()

    view = joint.content_view()

    assert view.selected_candidates[0].source_status == "source_linked"
    assert view.selected_neural_source_ids == (joint.input_bindings[0].neural_source_id,)


@pytest.mark.parametrize("scope", ["scope-a", "scope-b"])
@pytest.mark.parametrize("goal_id", ["goal-x", "goal-y"])
@pytest.mark.parametrize("intervention", ["related_flip", "unrelated_hold", "cut", "rescue"])
def test_v4_joint_logits_pass_sixteen_scope_goal_intervention_cases(scope, goal_id, intervention):
    session = f"{scope}-{goal_id}"
    model, root, joint = setup_two_source_joint(scope=scope, goal_id=goal_id, session=session)
    selected_sources = joint.content_view().selected_neural_source_ids
    logits_selected, _ = model.next_logits_from_joint(joint)

    if intervention == "related_flip":
        flipped_root = select_workspace_candidate(root, "candidate-red")
        flipped = joint.advance_cognitive_state(flipped_root)
        assert flipped.content_view().selected_neural_source_ids != selected_sources
        logits_flipped, _ = model.next_logits_from_joint(flipped)
        assert not torch.allclose(logits_selected, logits_flipped)
    elif intervention == "unrelated_hold":
        unrelated_object = WorldObject("object-unrelated", attributes=(("shape", "round"),))
        unrelated_root = replace(
            root,
            world=replace(
                root.world,
                entities=tuple(item.object_id for item in root.world.objects)
                + ("object-unrelated",),
                objects=root.world.objects + (unrelated_object,),
            ),
        )
        unrelated = joint.advance_cognitive_state(unrelated_root)
        assert unrelated.content_view().selected_neural_source_ids == selected_sources
        logits_unrelated, _ = model.next_logits_from_joint(unrelated)
        assert torch.equal(logits_selected, logits_unrelated)
    elif intervention == "cut":
        logits_cut, _ = model.next_logits(joint.language_state)
        assert not torch.allclose(logits_selected, logits_cut)
    else:
        flipped_root = select_workspace_candidate(root, "candidate-red")
        flipped = joint.advance_cognitive_state(flipped_root)
        restored_selection = flipped.advance_cognitive_state(root)
        logits_rescued, _ = model.next_logits_from_joint(restored_selection)
        assert torch.equal(logits_selected, logits_rescued)


def test_joint_language_generation_uses_selected_source_cache_without_parameter_updates():
    model, _, joint = setup_two_source_joint()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    generated, updated_joint_state = model.generate_from_joint(joint, max_new_bytes=8)

    assert isinstance(generated, bytes) and 0 < len(generated) <= 8
    assert updated_joint_state.language_state.generated.endswith(generated)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert joint.language_state.generated == b""


def test_joint_language_consumer_rejects_generic_origin_without_exact_source_lineage():
    model, root, joint = setup_two_source_joint()
    selected_event = replace(root.world.events[0], provenance="client")
    unlinked_root = replace(
        root,
        world=replace(root.world, events=(selected_event,)),
        events=(),
        assemblies=(),
    )
    unlinked = joint.advance_cognitive_state(unlinked_root)

    assert unlinked.content_view().selected_candidates[0].source_status == (
        "world_fact_source_unlinked"
    )
    with pytest.raises(ValueError, match="exact source-linked"):
        model.next_logits_from_joint(unlinked)


def test_actual_adapter_trace_can_feed_the_joint_v4_consumer_interface():
    adapter = TSKV8Adapter(
        TaijiConfig(region_sizes=(24, 16), synapse_fan_in=6, motor_fan_in=8, seed=41)
    )
    model = ConditionalLanguageCandidateV4(
        TaskCoreConfigV3(
            width=8,
            embedding_width=4,
            key_width=4,
            slots=4,
            chunk_bytes=4,
            max_observation_bytes=256,
            max_input_evidence_bytes=256,
            max_output_position=16,
            max_generation_bytes=16,
            seed=17,
        )
    )
    initial_root = adapter.cognitive_snapshot()
    initial_joint = JointTaskStateV1.begin(
        model,
        initial_root,
        joint_bundle_id="joint-v1:adapter-test",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="adapter-session",
    )
    frame = InputFrame(
        input_id="adapter-input-1",
        modality="text",
        payload="杯子是蓝色的。".encode(),
        source="test.client",
        timestamp=3,
    )
    trace = adapter.ingest_input(frame, learn=False)
    root = adapter.cognitive_snapshot()
    candidate_id, object_id = "adapter-fact", "adapter-cup"
    world_event = WorldEvent(
        event_id="adapter-world-fact",
        kind="observed_attribute",
        tick=root.world.tick,
        subject_id=object_id,
        object_id=object_id,
        attributes=(("color", "蓝色"),),
        provenance=trace.input_id,
    )
    broadcast = torch.zeros(2, dtype=torch.float32)
    workspace = WorkspaceState(
        tick=root.workspace.tick,
        focus=(candidate_id,),
        bindings=((candidate_id, object_id),),
        broadcast=broadcast,
        capacity=1,
        candidates=(
            WorkspaceCandidate(
                candidate_id,
                torch.tensor([1.0, 0.0], dtype=torch.float32),
                source=trace.input_id,
            ),
        ),
        selection=WorkspaceSelection(
            tick=root.workspace.tick,
            mode="learned",
            candidate_ids=(candidate_id,),
            selected_ids=(candidate_id,),
            scores=(1.0,),
            broadcast=broadcast,
            capacity=1,
        ),
    )
    root = replace(
        root,
        workspace=workspace,
        world=replace(
            root.world,
            entities=(object_id,),
            objects=(WorldObject(object_id, attributes=(("color", "蓝色"),)),),
            events=(world_event,),
        ),
    )
    joint = initial_joint.append_input(
        model,
        trace,
        root,
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
    )

    logits, _ = model.next_logits_from_joint(joint)

    assert logits.shape == (257,)
    assert joint.content_view().selected_neural_source_ids == (
        joint.input_bindings[0].neural_source_id,
    )


def test_actual_episodic_store_hit_is_source_bound_and_consumed_by_v4():
    adapter = TSKV8Adapter(
        TaijiConfig(region_sizes=(24, 16), synapse_fan_in=6, motor_fan_in=8, seed=51)
    )
    store = EpisodicMemoryStore(capacity=8)
    adapter.attach_episodic_memory(store)
    model = ConditionalLanguageCandidateV4(
        TaskCoreConfigV3(
            width=8,
            embedding_width=4,
            key_width=4,
            slots=4,
            chunk_bytes=4,
            max_observation_bytes=256,
            max_input_evidence_bytes=256,
            max_output_position=16,
            max_generation_bytes=16,
            seed=23,
        )
    )
    initial_root = adapter.cognitive_snapshot()
    joint = JointTaskStateV1.begin(
        model,
        initial_root,
        joint_bundle_id="joint-v1:episode-read-test",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="episode-read-session",
    )
    budget = {"remaining_steps": 1, "max_observation_bytes": 256}

    memory_frame = InputFrame(
        "episode-input-1", "text", "之前杯子是蓝色。".encode(), "test.client", timestamp=1
    )
    memory_trace = adapter.ingest_input(memory_frame, learn=False)
    memory_root = adapter.cognitive_snapshot()
    joint = joint.append_input(model, memory_trace, memory_root, resource_snapshot=budget)
    memory_record = EpisodicMemoryRecord(
        memory_id="episode-memory-1",
        episode_id=joint.task_id,
        tick=memory_root.tick,
        cue=memory_trace.percepts[-1].features.detach().clone(),
        provenance="external",
        event_ids=(memory_trace.percepts[-1].event_id,),
        assembly_ids=(memory_trace.percepts[-1].assembly_id,),
    )
    store.write(memory_record)

    unrelated_frame = InputFrame(
        "episode-input-2", "text", "今天天气晴朗。".encode(), "test.client", timestamp=2
    )
    unrelated_trace = adapter.ingest_input(unrelated_frame, learn=False)
    unrelated_root = adapter.cognitive_snapshot()
    joint = joint.append_input(model, unrelated_trace, unrelated_root, resource_snapshot=budget)

    query_frame = InputFrame(
        "episode-query-1", "text", "杯子之前是什么颜色？".encode(), "test.client", timestamp=3
    )
    query_trace = adapter.ingest_input(query_frame, learn=False)
    query_root = adapter.cognitive_snapshot()
    assert memory_record.memory_id in query_root.memory.episodic_ids
    joint = joint.append_input(model, query_trace, query_root, resource_snapshot=budget)
    read = JointEpisodeReadV1.retrieve(joint, store)

    memory_source = joint.input_bindings[0].neural_source_id
    unrelated_source = joint.input_bindings[1].neural_source_id
    query_source = joint.input_bindings[2].neural_source_id
    assert read.memory_id == memory_record.memory_id
    assert read.memory_neural_source_ids == (memory_source,)
    assert read.query_neural_source_id == query_source
    assert read.selected_neural_source_ids == (query_source, memory_source)
    assert unrelated_source not in read.selected_neural_source_ids
    forged_read = replace(
        read,
        selected_neural_source_ids=(query_source, memory_source, unrelated_source),
    )
    with pytest.raises(ValueError, match="receipt differs"):
        model.next_logits_from_episode(joint, forged_read, store)

    legacy_payload = copy.deepcopy(joint.to_payload())
    legacy_payload["format"] = "taiji-joint-task-state-v1"
    legacy_payload["version"] = 1
    legacy_payload.pop("action_receipts")
    legacy_payload.pop("decision_trace")
    for binding in legacy_payload["input_bindings"]:
        binding.pop("percept_tick_start", None)
        binding.pop("percept_tick_end", None)
    legacy_joint = JointTaskStateV1.from_payload(
        legacy_payload,
        joint_bundle_id=joint.joint_bundle_id,
        owner_id=joint.owner_id,
        scope_id=joint.scope_id,
        session_id=joint.session_id,
        candidate_bundle_id=model.bundle_id,
    )
    with pytest.raises(ValueError, match="source frame"):
        JointEpisodeReadV1.retrieve(legacy_joint, store)

    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    memory_logits, _ = model.next_logits_from_episode(joint, read, store)
    query_only_logits, _ = model.next_logits_from_episode_query_only(joint, read, store)
    unconditioned_logits, _ = model.next_logits(joint.language_state)
    assert not torch.allclose(memory_logits, query_only_logits)
    assert not torch.allclose(memory_logits, unconditioned_logits)
    generated, resumed_joint = model.generate_from_episode(joint, read, store, max_new_bytes=8)
    assert isinstance(generated, bytes) and generated
    assert resumed_joint.language_state.generated.endswith(generated)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_joint_episode_read_rejects_unapproved_or_unlinked_store_records():
    adapter = TSKV8Adapter(
        TaijiConfig(region_sizes=(24, 16), synapse_fan_in=6, motor_fan_in=8, seed=61)
    )
    store = EpisodicMemoryStore(capacity=4)
    adapter.attach_episodic_memory(store)
    model = ConditionalLanguageCandidateV4(
        TaskCoreConfigV3(
            width=8,
            embedding_width=4,
            key_width=4,
            slots=4,
            chunk_bytes=4,
            max_observation_bytes=256,
            max_input_evidence_bytes=256,
            max_output_position=16,
            max_generation_bytes=16,
            seed=29,
        )
    )
    initial_root = adapter.cognitive_snapshot()
    joint = JointTaskStateV1.begin(
        model,
        initial_root,
        joint_bundle_id="joint-v1:episode-unlinked-test",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="episode-unlinked-session",
    )
    frame = InputFrame(
        "episode-unlinked-input", "text", "再问一次。".encode(), "test.client", timestamp=1
    )
    trace = adapter.ingest_input(frame, learn=False)
    root = adapter.cognitive_snapshot()
    orphan = EpisodicMemoryRecord(
        memory_id="orphan-memory",
        episode_id=root.episode_id,
        tick=root.tick,
        cue=trace.percepts[-1].features.detach().clone(),
        provenance="external",
        assembly_ids=("assembly-not-in-joint-ledger",),
    )
    store.write(orphan)
    root = replace(root, memory=replace(root.memory, episodic_ids=(orphan.memory_id,)))
    joint = joint.append_input(
        model,
        trace,
        root,
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
    )

    with pytest.raises(ValueError, match="no source frame"):
        JointEpisodeReadV1.retrieve(joint, store)


def _world_predictor_for_action(preferred_kind: str) -> WorldDynamicsLearner:
    schema = WorldSchema(
        object_ids=("object-1",),
        state_slots=(("object-1", "on_table"),),
        relation_slots=(),
        action_kinds=("leave", "place"),
        actor_ids=("agent",),
        target_ids=("object-1",),
        parameter_names=(),
    )
    model = WorldDynamicsLearner(schema, hidden_dim=4, seed=73)
    preferred_index = schema.input_feature_keys.index(("action-kind", preferred_kind))
    rejected_kind = "leave" if preferred_kind == "place" else "place"
    rejected_index = schema.input_feature_keys.index(("action-kind", rejected_kind))
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].bias.zero_()
        model.network[0].weight[0, preferred_index] = 1.0
        model.network[0].weight[0, rejected_index] = -1.0
        model.network[2].weight.zero_()
        model.network[2].bias.zero_()
        model.network[2].weight[0, 0] = 1.0
        model.network[2].weight[schema.state_dim, 0] = 1.0
        model.network[2].weight[schema.state_dim + 1, 0] = 1.0
    return model


def _executive_preferring_goal_alignment() -> ExecutiveController:
    controller = ExecutiveController(candidate_feature_dim=6)
    with torch.no_grad():
        controller._model.weight.zero_()
        controller._model.bias.zero_()
        controller._model.weight[0, 0] = 1.0
    return controller


def _setup_joint_world_decision():
    model, root, trace, initial = setup_joint(
        text="杯子在桌上。", session="joint-world", max_observation_bytes=2048
    )
    action_affordances = (
        WorldAffordance(
            "aff-place",
            "place",
            actor_id="agent",
            target_id="object-1",
            confidence=0.9,
        ),
        WorldAffordance(
            "aff-leave",
            "leave",
            actor_id="agent",
            target_id="object-1",
            confidence=0.9,
        ),
    )
    world_event = replace(
        root.world.events[0],
        attributes=(("on_table", 0.0),),
    )
    structured_root = replace(
        root,
        goals=GoalState(tick=root.tick, goals=(Goal("goal-place", "place cup", priority=1.0),)),
        world=replace(
            root.world,
            entities=("object-1",),
            objects=(WorldObject("object-1", attributes=(("on_table", 0.0),)),),
            events=(world_event,),
            affordances=action_affordances,
        ),
    )
    joint = initial.append_input(
        model,
        trace,
        structured_root,
        resource_snapshot={
            "remaining_steps": 1,
            "max_observation_bytes": model.config.max_observation_bytes,
        },
    )
    return model, structured_root, joint


def test_world_prediction_executive_selection_and_v4_action_condition_share_joint_state():
    language_model, root, initial_joint = _setup_joint_world_decision()
    executive = _executive_preferring_goal_alignment()
    place_predictor = _world_predictor_for_action("place")
    leave_predictor = _world_predictor_for_action("leave")
    source_input_ids = (initial_joint.input_bindings[-1].input_id,)
    place_goal_state = replace(
        root.world,
        objects=(WorldObject("object-1", attributes=(("on_table", 1.0),)),),
    )
    leave_goal_state = replace(
        root.world,
        objects=(WorldObject("object-1", attributes=(("on_table", -1.0),)),),
    )
    place_target = JointGoalWorldTargetV1(
        "goal-place",
        place_goal_state,
        source_input_ids,
        "structured-interface-test",
        state_slots=(("object-1", "on_table"),),
    )
    leave_root = replace(
        root,
        goals=GoalState(tick=root.tick, goals=(Goal("goal-leave", "leave cup", priority=1.0),)),
    )
    leave_joint = initial_joint.advance_cognitive_state(leave_root)
    leave_target = JointGoalWorldTargetV1(
        "goal-leave",
        leave_goal_state,
        source_input_ids,
        "structured-interface-test",
        state_slots=(("object-1", "on_table"),),
    )
    incomplete_target = JointGoalWorldTargetV1(
        "goal-place",
        replace(
            place_goal_state,
            objects=(WorldObject("object-1", attributes=()),),
        ),
        source_input_ids,
        "structured-interface-test",
        state_slots=(("object-1", "on_table"),),
    )
    with pytest.raises(ValueError, match="omits an explicitly required"):
        JointWorldDecisionV1.select(
            initial_joint,
            place_predictor,
            executive,
            goal_target=incomplete_target,
        )
    with pytest.raises(ValueError, match="cannot distinguish"):
        JointWorldDecisionV1.select(
            initial_joint,
            place_predictor,
            ExecutiveController(candidate_feature_dim=6),
            goal_target=place_target,
        )
    language_parameters = {
        name: value.detach().clone() for name, value in language_model.state_dict().items()
    }
    executive_before = executive.checkpoint()
    place_before = place_predictor.snapshot()
    leave_before = leave_predictor.snapshot()

    place_decision = JointWorldDecisionV1.select(
        initial_joint, place_predictor, executive, goal_target=place_target
    )
    leave_decision = JointWorldDecisionV1.select(
        leave_joint, place_predictor, executive, goal_target=leave_target
    )
    world_flip_decision = JointWorldDecisionV1.select(
        initial_joint, leave_predictor, executive, goal_target=place_target
    )
    reversed_root = replace(
        root,
        world=replace(root.world, affordances=tuple(reversed(root.world.affordances))),
    )
    reversed_joint = initial_joint.advance_cognitive_state(reversed_root)
    reversed_decision = JointWorldDecisionV1.select(
        reversed_joint, place_predictor, executive, goal_target=place_target
    )

    assert place_decision.decision.action_intent.kind == "place"
    assert leave_decision.decision.action_intent.kind == "leave"
    assert world_flip_decision.decision.action_intent.kind == "leave"
    assert reversed_decision.decision.selected.source_affordance_id == (
        place_decision.decision.selected.source_affordance_id
    )
    assert place_decision.joint_state.cognitive_state.action_intent == (
        place_decision.decision.action_intent
    )
    assert place_decision.joint_state.cognitive_state.world_prediction.action.action_id == (
        place_decision.decision.action_intent.parameters["source_world_action_id"]
    )
    assert place_decision.joint_state.cognitive_state.plan.selected_plan_id == (
        place_decision.decision.action_intent.parameters["candidate_id"]
    )
    assert (
        place_decision.joint_state.cognitive_state.action_intent.parameters["world_model_sha256"]
        == place_decision.world_model_sha256
    )
    assert (
        place_decision.joint_state.cognitive_state.action_intent.parameters["executive_sha256"]
        == place_decision.executive_sha256
    )
    decision_trace = place_decision.joint_state.decision_trace
    assert decision_trace is not None
    assert decision_trace.intent_id == place_decision.decision.action_intent.intent_id
    assert decision_trace.world_model_sha256 == place_decision.world_model_sha256
    assert decision_trace.executive_sha256 == place_decision.executive_sha256
    altered_decision = copy.deepcopy(dict(decision_trace.executive_decision))
    altered_decision["selected"]["action_intent"]["expected_outcome"] += ";altered"
    altered_trace = JointDecisionTraceV1.create(
        intent_id=decision_trace.intent_id,
        world_model_snapshot=decision_trace.world_model_snapshot,
        world_prediction=decision_trace.world_prediction,
        executive_checkpoint=decision_trace.executive_checkpoint,
        executive_decision=altered_decision,
    )
    with pytest.raises(ValueError, match="different ActionIntent payload"):
        replace(place_decision.joint_state, decision_trace=altered_trace)
    decision_trainer = ConditionalLanguageTrainerV4(language_model)
    decision_checkpoint = decision_trainer.checkpoint(
        place_decision.joint_state.language_state, cursor={"step": 0}
    )
    decision_native = NativeCheckpoint(
        kernel={"format": "synthetic-task-kernel-v1"},
        cognitive_state=place_decision.joint_state.cognitive_state,
    )
    decision_envelope = place_decision.joint_state.attach_to_native_checkpoint(
        decision_native.to_payload(), decision_checkpoint
    )
    decision_restored, _ = JointTaskStateV1.restore_native_checkpoint(
        decision_envelope,
        joint_bundle_id=place_decision.joint_state.joint_bundle_id,
        owner_id=place_decision.joint_state.owner_id,
        scope_id=place_decision.joint_state.scope_id,
        session_id=place_decision.joint_state.session_id,
    )
    assert content_digest(decision_restored.decision_trace.to_payload()) == content_digest(
        decision_trace.to_payload()
    )
    restored_world, restored_executive, restored_decision = (
        decision_restored.decision_trace.restore_models()
    )
    assert content_digest(restored_world.snapshot()) == decision_trace.world_model_sha256
    assert content_digest(restored_executive.checkpoint()) == decision_trace.executive_sha256
    assert content_digest(restored_decision.to_payload()) == decision_trace.decision_sha256
    place_read = JointActionReadV1.from_joint_state(place_decision.joint_state)
    leave_read = JointActionReadV1.from_joint_state(leave_decision.joint_state)
    world_flip_read = JointActionReadV1.from_joint_state(world_flip_decision.joint_state)
    place_logits, _ = language_model.next_logits_from_action(place_decision.joint_state, place_read)
    leave_logits, _ = language_model.next_logits_from_action(leave_decision.joint_state, leave_read)
    world_flip_logits, _ = language_model.next_logits_from_action(
        world_flip_decision.joint_state, world_flip_read
    )
    assert not torch.allclose(place_logits, leave_logits)
    assert not torch.allclose(place_logits, world_flip_logits)
    generated, updated_joint = language_model.generate_from_action(
        place_decision.joint_state, place_read, max_new_bytes=8
    )
    assert generated and updated_joint.language_state.generated.endswith(generated)
    assert content_digest(executive.checkpoint()) == content_digest(executive_before)
    assert content_digest(place_predictor.snapshot()) == content_digest(place_before)
    assert content_digest(leave_predictor.snapshot()) == content_digest(leave_before)
    assert all(
        torch.equal(language_parameters[name], value)
        for name, value in language_model.state_dict().items()
    )
    decision_state = place_decision.joint_state
    intent = decision_state.cognitive_state.action_intent
    assert intent is not None
    dispatch, _ = decision_state.begin_action_dispatch(intent.intent_id)
    execution_id = "trusted-fixture-executor-1"
    accepted = dispatch.mark_action_execution_accepted(intent.intent_id, execution_id)
    selected_action = replace(place_decision.selected_prediction.action, provenance="experienced")
    after = replace(root.world, tick=root.world.tick + 1)
    transition_result = Outcome(
        selected_action.action_id,
        reward=0.75,
        success=True,
        terminal=False,
        tick=after.tick,
        provenance="experienced",
    )
    transition = WorldTransition(root.world, selected_action, after, transition_result)
    intent_result = replace(transition_result, intent_id=intent.intent_id)
    outcome_owner = replace(
        accepted.cognitive_state,
        tick=after.tick,
        world=after,
        outcome=intent_result,
        world_transition=transition,
    )
    altered_prediction = replace(
        outcome_owner.world_prediction,
        predicted_reward=outcome_owner.world_prediction.predicted_reward + 0.01,
    )
    with pytest.raises(ValueError, match="selection-time decision trace"):
        accepted.record_action_outcome(
            replace(outcome_owner, world_prediction=altered_prediction),
            intent.intent_id,
            execution_id=execution_id,
        )
    settled = accepted.record_action_outcome(
        outcome_owner,
        intent.intent_id,
        execution_id=execution_id,
    )
    feedback = JointLanguageFeedbackV1.bind(
        settled,
        input_id=source_input_ids[0],
        kind="correction",
        target_text="杯子已放在桌上。",
    )
    _, reused_id_trace = make_trace("请继续核对。", input_id=source_input_ids[0])
    reused_id_root = make_root(reused_id_trace, "请继续核对。", task_id="other-task")
    other_joint = JointTaskStateV1.begin(
        language_model,
        reused_id_root,
        joint_bundle_id="joint-v1:other-scope",
        owner_id="user-1",
        scope_id="scope-other",
        session_id="session-other",
    ).append_input(
        language_model,
        reused_id_trace,
        reused_id_root,
        resource_snapshot={
            "remaining_steps": 1,
            "max_observation_bytes": language_model.config.max_observation_bytes,
        },
    )
    with pytest.raises(ValueError, match="different joint task"):
        feedback.validate_for(other_joint)
    route = JointCreditRouteV1.from_settled_outcome(settled, language_feedback=feedback)
    assert set(route.eligible_receivers) == {"world_model", "executive", "language_candidate"}
    assert route.world_prediction_sha256 is not None
    assert route.state_owner == "CognitiveState"
    assert route.apply_updates is False
    assert route.outcome_trust == "caller_asserted_experienced_not_authenticated"
    restored_route = JointCreditRouteV1.from_payload(route.to_payload())
    assert content_digest(restored_route.to_payload()) == content_digest(route.to_payload())
    reward_only_route = JointCreditRouteV1.from_settled_outcome(settled)
    assert set(reward_only_route.eligible_receivers) == {"world_model", "executive"}
    assert content_digest(executive.checkpoint()) == content_digest(executive_before)
    assert content_digest(place_predictor.snapshot()) == content_digest(place_before)

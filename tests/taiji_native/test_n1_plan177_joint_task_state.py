"""Joint task frames bind one canonical cognitive owner to versioned neural evidence."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from taiji.conditional_language_v3 import TaskCoreConfigV3
from taiji.conditional_language_v4 import (
    ConditionalLanguageCandidateV4,
    ConditionalLanguageTrainerV4,
)
from taiji.contracts import (
    ActionIntent,
    Assembly,
    CognitiveState,
    DevelopmentState,
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
    WorldEvent,
    WorldObject,
    WorldState,
)
from taiji.input_boundary import InputTrace
from taiji.internalization import content_digest
from taiji.joint_task_state import JointTaskStateV1


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
            max_observation_bytes=256,
            max_input_evidence_bytes=256,
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
    frame = joint.advance_cognitive_state(pending)
    with pytest.raises(ValueError, match="non-generated"):
        frame.record_action_outcome(pending, "intent-1")
    failure = Outcome(
        "intent-1",
        reward=-1.0,
        success=False,
        terminal=False,
        tick=root.tick,
        provenance="experienced",
    )
    retry = frame.record_action_outcome(replace(pending, outcome=failure), "intent-1")
    assert retry.lifecycle == "active" and retry.cognitive_state.outcome.success is False
    unknown = Outcome(
        "intent-1",
        reward=0.0,
        success=None,
        terminal=False,
        tick=root.tick,
        provenance="experienced",
    )
    awaiting = retry.record_action_outcome(
        replace(retry.cognitive_state, outcome=unknown), "intent-1"
    )
    assert awaiting.lifecycle == "awaiting_outcome"
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
        replace(awaiting.cognitive_state, outcome=resolved), "intent-1"
    )
    assert settled.lifecycle == "complete"
    generated = replace(failure, provenance="generated")
    with pytest.raises(ValueError, match="non-generated"):
        frame.record_action_outcome(replace(pending, outcome=generated), "intent-1")


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
    damaged = copy.deepcopy(envelope)
    damaged["components"]["joint_task_state_v1"]["frame"]["owner_id"] = "user-other"
    with pytest.raises(ValueError, match="component digest"):
        JointTaskStateV1.restore_native_checkpoint(
            damaged,
            joint_bundle_id=joined.joint_bundle_id,
            owner_id=joined.owner_id,
            scope_id=joined.scope_id,
            session_id=joined.session_id,
        )


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

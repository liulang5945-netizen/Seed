from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scripts.training.eval_taiji_a2_world import build_corpus
from seed import Seed, SeedConfig
from taiji import (
    CONTRACT_FORMAT,
    ActionIntent,
    Assembly,
    CognitiveState,
    Concept,
    DevelopmentState,
    EnvironmentOutcome,
    Event,
    Goal,
    GoalPlanner,
    NativeMemoryState,
    Observation,
    Outcome,
    PlanningCandidate,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
    WorldDynamicsLearner,
    WorldObject,
    WorldSchema,
    WorldState,
)


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(12, 8),
        synapse_fan_in=4,
        motor_fan_in=6,
        memory_units=16,
        memory_fan_in=4,
        memory_readout_fan_in=6,
        memory_meta_dim=6,
        memory_iterations=2,
        memory_time_dim=4,
        memory_episode_dim=4,
        lateral_fan_in=4,
        seed=71,
    )


def test_core_object_contracts_round_trip_and_cognitive_state_lineage() -> None:
    assembly = Assembly(
        assembly_id="assembly-1",
        start_tick=2,
        end_tick=3,
        activity=torch.tensor([0.1, 0.9]),
        member_indices=(1, 4),
        source_event_ids=("percept-1",),
        coherence=0.8,
        prediction_error=0.2,
        route_score=0.7,
        provenance="learned",
        confidence=0.9,
    )
    event = Event(
        event_id="event-1",
        start_tick=2,
        end_tick=3,
        latent=torch.tensor([0.2, 0.8]),
        assembly_ids=(assembly.assembly_id,),
        object_ids=("object-1",),
        relation_ids=("relation-1",),
        confidence=0.75,
        provenance="experienced",
    )
    concept = Concept(
        concept_id="concept-1",
        prototype=torch.tensor([0.3, 0.7]),
        support_event_ids=(event.event_id,),
        support_assembly_ids=(assembly.assembly_id,),
        relation_ids=("relation-1",),
        maturity=0.4,
        stability=0.6,
        confidence=0.8,
        update_count=2,
        last_updated_tick=3,
    )

    restored_assembly = Assembly.from_payload(assembly.to_payload())
    restored_event = Event.from_payload(event.to_payload())
    restored_concept = Concept.from_payload(concept.to_payload())

    assert torch.equal(restored_assembly.activity, assembly.activity)
    assert restored_event.assembly_ids == (assembly.assembly_id,)
    assert torch.equal(restored_concept.prototype, concept.prototype)

    model = TSKV8Adapter(_config(), episode_id="core-object-lineage")
    snapshot = model.cognitive_snapshot()
    enriched = replace(
        snapshot,
        assemblies=(assembly,),
        events=(event,),
        concepts=(concept,),
        self_state=replace(
            snapshot.self_state,
            capability_confidence=(("planning", 0.75),),
            available_tool_ids=("tool-1",),
            autobiographical_ids=("memory-1",),
            commitment_ids=("goal-1",),
            last_outcome_id="outcome-1",
            last_update_source="outcome-1",
            last_prediction_error=0.25,
            update_count=2,
            lineage=("checkpoint-0",),
        ),
        development=replace(
            snapshot.development,
            stage="specializing",
            structural_budget=3,
            resource_utilization=0.5,
            capability_gaps=("long-horizon",),
            proposal_ids=("proposal-1",),
            parent_checkpoint_id="checkpoint-0",
            last_update_source="evaluation-1",
            last_validation_status="pending",
            validation_evidence_ids=("report-1",),
            growth_count=1,
            lineage=("checkpoint-0",),
        ),
    )
    restored_state = CognitiveState.from_payload(enriched.to_payload())

    assert restored_state.assemblies[0].assembly_id == assembly.assembly_id
    assert restored_state.events[0].event_id == event.event_id
    assert restored_state.concepts[0].concept_id == concept.concept_id
    assert restored_state.self_state.capability_confidence == (("planning", 0.75),)
    assert restored_state.development.parent_checkpoint_id == "checkpoint-0"


def test_core_object_contracts_reject_invalid_lineage_and_growth_state() -> None:
    with pytest.raises(ValueError, match="assembly ticks"):
        Assembly(assembly_id="bad", start_tick=2, end_tick=1)

    with pytest.raises(ValueError, match="duplicate"):
        Event(event_id="bad", start_tick=0, end_tick=0, assembly_ids=("a", "a"))

    with pytest.raises(ValueError, match="validation status"):
        DevelopmentState(tick=0, last_validation_status="unknown")


def test_v1_contracts_round_trip_tensor_and_action_data() -> None:
    observation = Observation(
        modality="text-byte",
        value=b"a",
        timestamp=3,
        source="unit-test",
        provenance="experienced",
        confidence=0.75,
    )
    intent = ActionIntent(
        intent_id="intent-1",
        kind="tool",
        parameters={"vector": torch.tensor([1.0, 2.0]), "args": ("x", 2)},
        source_goal_id="goal-1",
        expected_outcome="done",
        confidence=0.5,
        tick=3,
    )
    outcome = Outcome(
        intent_id=intent.intent_id,
        reward=1.25,
        success=True,
        observation=observation,
        tick=4,
    )

    restored_intent = ActionIntent.from_payload(intent.to_payload())
    restored = Outcome.from_payload(outcome.to_payload())

    assert restored == outcome
    assert torch.equal(restored_intent.parameters["vector"], torch.tensor([1.0, 2.0]))
    assert restored_intent.parameters["args"] == ("x", 2)
    assert restored.observation is not None
    assert restored.observation.value == b"a"


def test_tsk_v8_adapter_exposes_observation_action_outcome_contracts() -> None:
    model = TSKV8Adapter(_config(), episode_id="contract-loop")
    observation = Observation(
        modality="text-byte",
        value=97,
        timestamp=0,
        source="test-environment",
    )

    model.observe_event(observation, learn=False)
    decision = model.act((97, 98), sample=False)
    state = model.cognitive_snapshot()

    assert state.observation == observation
    assert state.percept is not None
    assert state.action_intent is not None
    assert state.action_intent.parameters["action_symbol"] == decision.action_symbol
    assert state.plan.selected_plan_id is not None

    model.settle_action(1.0, learn=False, provenance="experienced")
    state = model.cognitive_snapshot()
    assert state.outcome is not None
    assert state.outcome.intent_id == state.action_intent.intent_id
    assert state.outcome.reward == 1.0


def test_tsk_v8_adapter_commits_structured_world_transition_lineage() -> None:
    model = TSKV8Adapter(_config(), episode_id="world-lineage")
    model.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="test-environment",
        ),
        learn=False,
    )
    model.act((97, 98), sample=False)
    before = model.cognitive_snapshot().world
    after = WorldState(
        tick=before.tick + 1,
        latent=torch.zeros(2),
        objects=(WorldObject("token", attributes={"position": 1.0}),),
    )

    model.settle_action(1.0, learn=False, learn_world=True, world_state=after, success=True)
    state = model.cognitive_snapshot()

    assert state.world.objects[0].attribute("position") == 1.0
    assert state.world_transition is not None
    assert state.world_transition.before.tick == before.tick
    assert state.world_transition.after.tick == after.tick
    assert state.world_transition.outcome.intent_id == state.action_intent.intent_id

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    restored_state = restored.cognitive_snapshot()
    assert restored_state.world_transition is not None
    assert (
        restored_state.world_transition.action.action_id == state.world_transition.action.action_id
    )
    assert restored_state.world.objects[0].attribute("position") == 1.0

    restored.observe(100, learn=False)
    assert restored.cognitive_snapshot().world.objects[0].attribute("position") == 1.0


def test_tsk_v8_adapter_scores_runtime_world_prediction_and_restores_learner() -> None:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=11)
    learner.fit(corpus.train, epochs=200, learning_rate=0.01)

    model = TSKV8Adapter(_config(), episode_id="world-prediction")
    model.attach_world_dynamics(learner)
    initial = corpus.train[0].initial
    initial = WorldState(
        tick=1,
        latent=torch.zeros(2),
        objects=initial.objects,
        relations=initial.relations,
        events=initial.events,
        affordances=initial.affordances,
    )
    model.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="test-environment",
        ),
        learn=False,
        world_state=initial,
    )
    model.act(
        (97, 98),
        sample=False,
        world_action=WorldAction(
            action_id="pending",
            kind="move",
            tick=model.tick,
            actor_id="agent",
            target_id="red",
            parameters={"step": 1.0},
        ),
    )
    after = WorldState(
        tick=2,
        latent=torch.zeros(2),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": 1.0}),
            WorldObject("blue", attributes={"position": 0.0}),
        ),
    )

    model.settle_action(1.0, learn=False, learn_world=True, world_state=after, success=True)
    state = model.cognitive_snapshot()
    assert state.world_prediction is not None
    assert state.world_prediction.state_error is not None
    assert state.world_prediction.raw_state_error is not None
    assert state.world_prediction.reward_error is not None
    assert state.world_prediction.state_error < 1.0
    assert state.world_prediction.online_update_count == 1
    assert len(state.world_calibration_trace) == 1
    trace = state.world_calibration_trace[0]
    assert trace.transition.action.action_id == state.action_intent.intent_id
    assert trace.calibration_applied is True
    assert trace.online_update_count_before == 0
    assert trace.online_update_count_after == 1

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    restored_state = restored.cognitive_snapshot()
    assert restored_state.world_prediction is not None
    assert restored_state.world_prediction.state_error == state.world_prediction.state_error
    assert len(restored_state.world_calibration_trace) == 1
    restored_trace = restored_state.world_calibration_trace[0]
    assert restored_trace.transition.action.action_id == trace.transition.action.action_id
    assert restored_trace.prediction.state_error == trace.prediction.state_error
    assert restored_trace.prediction.raw_state_error == trace.prediction.raw_state_error
    assert restored_trace.online_update_count_after == trace.online_update_count_after
    assert restored._world_dynamics is not None
    assert restored._world_dynamics.online_updates == 1


def test_world_schema_scale_separates_raw_and_normalized_error() -> None:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    case = corpus.train[0]

    raw_error = float(
        torch.mean(
            (schema.state_values(case.initial) - schema.state_values(case.expected_state)) ** 2
        )
    )
    normalized_error = schema.normalized_state_error(case.initial, case.expected_state)
    scaled_schema = replace(
        schema,
        state_scales=tuple(scale * 10.0 for scale in schema.state_scales),
    )

    assert raw_error > 0.0
    assert 0.0 < normalized_error <= raw_error
    assert (
        scaled_schema.normalized_state_error(case.initial, case.expected_state) < normalized_error
    )
    assert WorldSchema.from_payload(schema.payload()).state_scales == schema.state_scales


def test_world_schema_scale_transfer_preserves_normalized_policy() -> None:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    case = corpus.train[0]
    factor = 10.0

    def scale_state(state):
        return WorldState(
            tick=state.tick,
            latent=state.latent,
            entities=state.entities,
            relations=state.relations,
            objects=tuple(
                WorldObject(
                    item.object_id,
                    attributes={name: float(value) * factor for name, value in item.attributes},
                    tags=item.tags,
                )
                for item in state.objects
            ),
            events=state.events,
            affordances=state.affordances,
            uncertainty=state.uncertainty,
        )

    scaled_schema = replace(
        schema,
        state_scales=tuple(
            scale * (factor if index < schema.object_state_dim else 1.0)
            for index, scale in enumerate(schema.state_scales)
        ),
    )
    base_normalized = schema.normalized_state_error(case.initial, case.expected_state)
    scaled_initial = scale_state(case.initial)
    scaled_expected = scale_state(case.expected_state)
    scaled_raw = float(
        torch.mean(
            (
                scaled_schema.state_values(scaled_initial)
                - scaled_schema.state_values(scaled_expected)
            )
            ** 2
        )
    )
    base_raw = float(
        torch.mean(
            (schema.state_values(case.initial) - schema.state_values(case.expected_state)) ** 2
        )
    )

    assert scaled_raw > base_raw
    assert scaled_schema.normalized_state_error(scaled_initial, scaled_expected) == pytest.approx(
        base_normalized
    )
    planner = GoalPlanner()
    base_threshold = planner.calibrate_world_prediction_errors((base_normalized,))
    scaled_threshold = planner.calibrate_world_prediction_errors(
        (scaled_schema.normalized_state_error(scaled_initial, scaled_expected),)
    )
    assert scaled_threshold == pytest.approx(base_threshold)


def test_world_prediction_projects_into_planner_and_triggers_replan_lesion() -> None:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=11)
    learner.fit(corpus.train, epochs=200, learning_rate=0.01)

    model = TSKV8Adapter(_config(), episode_id="world-planner")
    model.attach_world_dynamics(learner)
    model.attach_goal_planner(GoalPlanner())
    model.set_goals((Goal("reach-world", "reach the world target", priority=1.0),))
    corpus_initial = corpus.train[0].initial
    initial = WorldState(
        tick=1,
        latent=torch.zeros(2),
        objects=corpus_initial.objects,
        relations=corpus_initial.relations,
        events=corpus_initial.events,
        affordances=corpus_initial.affordances,
    )
    model.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="test-environment",
        ),
        learn=False,
        world_state=initial,
    )
    action = WorldAction(
        action_id="world-candidate",
        kind="move",
        tick=model.cognitive_snapshot().world.tick,
        actor_id="agent",
        target_id="red",
        parameters={"step": 1.0},
    )
    candidate = PlanningCandidate(
        candidate_id="model-route",
        action=action,
        predicted_reward=123.0,
        success_probability=0.0,
        expected_progress=0.8,
    )
    projected = model.predict_world_candidates((candidate,))
    assert projected[0].predicted_reward != candidate.predicted_reward
    assert projected[0].success_probability != candidate.success_probability
    decision = model.plan_world_actions((candidate,))
    assert decision.selected.candidate_id == "model-route"
    second_action = WorldAction(
        action_id="world-candidate-next",
        kind="move",
        tick=action.tick + 1,
        actor_id="agent",
        target_id="blue",
        parameters={"step": 1.0},
    )
    rollout = model.imagine_world_rollout(
        "world-rollout",
        "reach-world",
        (
            candidate,
            PlanningCandidate(
                candidate_id="model-route-next",
                action=second_action,
                predicted_reward=0.0,
                success_probability=0.0,
                expected_progress=0.9,
            ),
        ),
    )
    assert len(rollout.steps) == 2
    assert all(step.prediction_provenance == "world-dynamics" for step in rollout.steps)
    assert all(step.action.provenance == "world-dynamics" for step in rollout.steps)
    rollout_decision = model.plan_rollouts((rollout,))
    assert rollout_decision.selected.rollout_id == "world-rollout"

    model.act((97, 98), sample=False, world_action=action)
    bad_objects = []
    for item in initial.objects:
        attributes = dict(item.attributes)
        if item.object_id == "red":
            attributes["position"] = 9.0
        bad_objects.append(WorldObject(item.object_id, attributes=attributes, tags=item.tags))
    bad_after = WorldState(
        tick=initial.tick + 1,
        latent=initial.latent,
        objects=tuple(bad_objects),
    )
    model.settle_action(
        1.0,
        learn=False,
        learn_world=False,
        world_state=bad_after,
        success=True,
    )
    state = model.cognitive_snapshot()
    assert state.world_prediction is not None
    assert state.world_prediction.state_error is not None
    assert state.world_prediction.state_error > model._goal_planner.config.replan_error_threshold
    assert model.replan_required is True

    lesion = TSKV8Adapter(_config(), episode_id="world-planner-lesion")
    with pytest.raises(RuntimeError, match="world dynamics is not attached"):
        lesion.predict_world_candidates((candidate,))


def test_imagined_rollout_executes_real_steps_and_consumes_remaining_plan() -> None:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=11)
    learner.fit(corpus.train, epochs=250, learning_rate=0.01)

    class RolloutEnvironment:
        def __init__(self, states: tuple[WorldState, ...]) -> None:
            self.states = states
            self.index = 0
            self.actions: list[int] = []

        def reset(self) -> tuple[int, tuple[int, ...]]:
            self.index = 0
            self.actions.clear()
            return 97, (10, 11)

        def step(self, action_symbol: int) -> EnvironmentOutcome:
            self.actions.append(action_symbol)
            state = self.states[self.index]
            self.index += 1
            return EnvironmentOutcome(
                sensation=97 + self.index,
                reward=1.0,
                success=True,
                terminal=self.index == len(self.states),
                world_state=state,
            )

    base = corpus.train[0].initial
    initial = WorldState(
        tick=1,
        latent=base.latent,
        objects=base.objects,
        relations=base.relations,
        events=base.events,
        affordances=base.affordances,
    )
    after_one = WorldState(
        tick=2,
        latent=base.latent,
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": 1.0}),
            WorldObject("blue", attributes={"position": 0.0}),
        ),
    )
    after_two = WorldState(
        tick=3,
        latent=base.latent,
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": 1.0}),
            WorldObject("blue", attributes={"position": 1.0}),
        ),
    )
    model = TSKV8Adapter(_config(), episode_id="imagined-execution")
    model.attach_world_dynamics(learner)
    model.attach_goal_planner(GoalPlanner())
    model.set_goals((Goal("reach-world", "reach the world target", priority=1.0),))
    model.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="test-environment",
        ),
        learn=False,
        world_state=initial,
    )
    start_tick = model.cognitive_snapshot().world.tick
    rollout = model.imagine_world_rollout(
        "imagined-execution-rollout",
        "reach-world",
        (
            PlanningCandidate(
                candidate_id="red-step",
                action=WorldAction(
                    action_id="red-step",
                    kind="move",
                    tick=start_tick,
                    actor_id="agent",
                    target_id="red",
                    parameters={"step": 1.0, "action_symbol": 10},
                ),
                predicted_reward=0.0,
                success_probability=0.0,
                expected_progress=0.5,
            ),
            PlanningCandidate(
                candidate_id="blue-step",
                action=WorldAction(
                    action_id="blue-step",
                    kind="move",
                    tick=start_tick + 1,
                    actor_id="agent",
                    target_id="blue",
                    parameters={"step": 1.0, "action_symbol": 11},
                ),
                predicted_reward=0.0,
                success_probability=0.0,
                expected_progress=1.0,
            ),
        ),
    )
    model.plan_rollouts((rollout,))
    environment = RolloutEnvironment((after_one, after_two))

    first = model.execute_imagined_rollout_step(
        environment,
        available_actions=(10, 11),
        action_kinds=("move", "idle"),
        learn=False,
        learn_world=True,
    )
    assert first.success is True
    assert model._planned_rollout is not None
    assert len(model._planned_rollout.steps) == 1

    second = model.execute_imagined_rollout_step(
        environment,
        available_actions=(10, 11),
        action_kinds=("idle", "move"),
        learn=False,
        learn_world=True,
    )
    assert second.success is True
    assert environment.actions == [10, 11]
    assert model._planned_rollout is None
    snapshot = model.cognitive_snapshot()
    assert snapshot.world.tick == 3
    assert len(snapshot.world_calibration_trace) == 2
    assert all(item.calibration_applied for item in snapshot.world_calibration_trace)


def test_native_checkpoint_is_atomic_and_deterministic() -> None:
    model = TSKV8Adapter(_config(), episode_id="native-checkpoint")
    for symbol in (256, 97, 98, 99):
        model.observe(symbol, learn=True)
    model.act((97, 98), sample=False)
    model.settle_action(0.25, learn=True)
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)

    assert checkpoint["format"] == CONTRACT_FORMAT
    assert checkpoint["adapter"] == "tsk-v8"
    original_state = model.cognitive_snapshot()
    restored_state = restored.cognitive_snapshot()
    assert restored_state.episode_id == original_state.episode_id
    assert restored_state.tick == original_state.tick
    assert restored_state.action_intent is not None
    assert original_state.action_intent is not None
    assert restored_state.action_intent.intent_id == original_state.action_intent.intent_id
    assert restored_state.action_intent.kind == original_state.action_intent.kind
    assert (
        restored_state.action_intent.parameters["action_symbol"]
        == original_state.action_intent.parameters["action_symbol"]
    )
    assert restored_state.outcome == original_state.outcome
    assert torch.equal(restored_state.workspace.broadcast, original_state.workspace.broadcast)
    assert torch.equal(restored_state.world.latent, original_state.world.latent)

    left = model.observe(100, learn=True)
    right = restored.observe(100, learn=True)
    assert left.predicted_symbol == right.predicted_symbol
    assert torch.equal(left.probabilities, right.probabilities)


def test_seed_delegates_cognitive_checkpoint_to_taiji() -> None:
    model = Seed(SeedConfig(taiji=_config()), episode_id="seed-contract")
    model.observe(97, learn=False)
    checkpoint = model.checkpoint()

    assert isinstance(model.architecture, TSKV8Adapter)
    assert checkpoint["taiji"]["format"] == CONTRACT_FORMAT
    assert checkpoint["taiji"]["cognitive_state"]["tick"] == model.tick
    assert checkpoint["substrate"]["format"] == "taiji-native-v8"


def test_seed_reads_pre_p1_checkpoint_with_substrate_only() -> None:
    model = Seed(SeedConfig(taiji=_config()), episode_id="legacy-seed")
    model.observe(97, learn=True)
    checkpoint = model.checkpoint()
    legacy = dict(checkpoint)
    legacy.pop("taiji")

    restored = Seed.from_checkpoint(legacy)

    assert restored.tick == model.tick
    assert restored.substrate.checkpoint()["format"] == "taiji-native-v8"


def test_native_memory_state_is_distinct_from_kernel_memory_state() -> None:
    state = NativeMemoryState(
        tick=1,
        episodic_confidence=0.5,
        semantic_context=torch.zeros(2),
        procedural_context=torch.ones(2),
    )
    restored = NativeMemoryState.from_payload(state.to_payload())

    assert restored.tick == state.tick
    assert restored.episodic_confidence == state.episodic_confidence
    assert torch.equal(restored.semantic_context, state.semantic_context)
    assert torch.equal(restored.procedural_context, torch.ones(2))

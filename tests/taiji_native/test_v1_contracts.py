from __future__ import annotations

import torch

from scripts.training.eval_taiji_a2_world import build_corpus
from seed import Seed, SeedConfig
from taiji import (
    CONTRACT_FORMAT,
    ActionIntent,
    NativeMemoryState,
    Observation,
    Outcome,
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
    assert restored_state.world_transition.action.action_id == state.world_transition.action.action_id
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
    assert state.world_prediction.reward_error is not None
    assert state.world_prediction.state_error < 1.0
    assert state.world_prediction.online_update_count == 1

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    restored_state = restored.cognitive_snapshot()
    assert restored_state.world_prediction is not None
    assert restored_state.world_prediction.state_error == state.world_prediction.state_error
    assert restored._world_dynamics is not None
    assert restored._world_dynamics.online_updates == 1


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

from __future__ import annotations

import torch

from taiji import (
    ActionIntent,
    EpisodicMemoryRecord,
    EpisodicMemoryStore,
    Goal,
    GoalPlanner,
    Observation,
    Outcome,
    PerceptionConfig,
    PlanningCandidate,
    PlanningConfig,
    SemanticMemoryLearner,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
    WorldState,
)


def test_percept_lineage_reaches_workspace_and_world_and_survives_checkpoint() -> None:
    config = TaijiConfig(
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
        perception=PerceptionConfig(maximum_assembly_duration=2),
    )
    model = TSKV8Adapter(config, episode_id="percept-lineage")

    closed_state = None
    for tick, symbol in enumerate((97, 98, 99)):
        world_state = None if tick != 2 else WorldState(tick=model.tick + 1, latent=torch.zeros(2))
        model.observe_event(
            Observation(
                modality="text-byte",
                value=symbol,
                timestamp=tick,
                source="lineage-test",
            ),
            learn=False,
            world_state=world_state,
        )
        if tick == 1:
            closed_state = model.cognitive_snapshot()

    state = model.cognitive_snapshot()
    assert closed_state is not None
    assert closed_state.percept is not None
    assert closed_state.percept.boundary is True
    assert closed_state.workspace.percept_boundary_closed is True
    assert closed_state.world.percept_boundary_closed is True
    assert state.percept is not None
    assert state.events
    event = state.events[-1]
    assert state.workspace.percept_event_id == event.event_id
    assert state.workspace.percept_assembly_id == state.percept.assembly_id
    assert state.workspace.percept_boundary_closed == state.percept.boundary
    assert state.world.percept_event_id == event.event_id
    assert state.world.percept_assembly_id == state.percept.assembly_id
    assert state.world.percept_boundary_closed == state.percept.boundary

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    restored_state = restored.cognitive_snapshot()
    assert restored_state.workspace.percept_event_id == event.event_id
    assert restored_state.world.percept_assembly_id == state.percept.assembly_id
    assert restored_state.world.percept_boundary_closed == state.percept.boundary


def test_concept_registry_is_consumed_by_planning_and_lesion_removes_prior() -> None:
    model = TSKV8Adapter(
        TaijiConfig(
            region_sizes=(32, 24),
            synapse_fan_in=8,
            motor_fan_in=8,
            memory_units=32,
            memory_fan_in=8,
            memory_readout_fan_in=8,
            memory_meta_dim=8,
            lateral_fan_in=4,
            memory_time_dim=4,
            memory_episode_dim=4,
        ),
        episode_id="concept-runtime",
    )
    episodic = EpisodicMemoryStore(capacity=8, cue_dim=model.perception.feature_dim)
    model.attach_episodic_memory(episodic)
    model.attach_semantic_memory(SemanticMemoryLearner(model.perception.feature_dim))
    model.observe_event(
        Observation(modality="text-byte", value=97, timestamp=0, source="test"),
        learn=False,
    )
    percept = model.cognitive_snapshot().percept
    assert percept is not None
    cue = percept.features.detach().clone()
    for index in range(2):
        intent_id = f"concept-intent-{index}"
        episodic.write(
            EpisodicMemoryRecord(
                memory_id=f"concept-memory-{index}",
                episode_id=f"concept-episode-{index}",
                tick=1,
                cue=cue,
                action_intent=ActionIntent(intent_id, "preferred", tick=0),
                outcome=Outcome(intent_id, reward=1.0, success=True, tick=1),
                event_ids=(f"concept-event-{index}",),
                assembly_ids=(f"concept-assembly-{index}",),
            )
        )

    model.consolidate_semantic_memory(epochs=1, learning_rate=0.01)
    concepts = model.cognitive_snapshot().concepts
    assert len(concepts) == 1
    assert concepts[0].action_kinds == ("preferred",)
    assert model.concept_formation.retrieve(cue)[0].concept.concept_id == concepts[0].concept_id

    model.attach_goal_planner(GoalPlanner(PlanningConfig(concept_weight=0.40)))
    model.set_goals((Goal("goal", "prefer the experienced route", priority=1.0),))
    tick = model.cognitive_snapshot().world.tick
    preferred = PlanningCandidate(
        candidate_id="preferred-candidate",
        action=WorldAction("preferred-action", "preferred", tick=tick),
        predicted_reward=0.0,
        success_probability=0.5,
        expected_progress=0.5,
    )
    fallback = PlanningCandidate(
        candidate_id="fallback-candidate",
        action=WorldAction("fallback-action", "fallback", tick=tick),
        predicted_reward=0.05,
        success_probability=0.5,
        expected_progress=0.5,
    )
    decision = model.plan_actions((preferred, fallback))
    assert decision.selected.action.kind == "preferred"
    assert decision.selected.concept_affinity > 0.5

    concept_id = concepts[0].concept_id
    assert model.concept_formation.lesion((concept_id,)) == (concept_id,)
    lesioned_decision = model.plan_actions((preferred, fallback))
    assert lesioned_decision.selected.action.kind == "fallback"
    assert lesioned_decision.selected.concept_affinity == 0.0

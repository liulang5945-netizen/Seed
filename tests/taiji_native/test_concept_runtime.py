from __future__ import annotations

from taiji import (
    ActionIntent,
    EpisodicMemoryRecord,
    EpisodicMemoryStore,
    Goal,
    GoalPlanner,
    Observation,
    Outcome,
    PlanningCandidate,
    PlanningConfig,
    SemanticMemoryLearner,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
)


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

    model.attach_goal_planner(
        GoalPlanner(PlanningConfig(concept_weight=0.40))
    )
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

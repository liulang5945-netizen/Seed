from __future__ import annotations

import pytest
import torch

from taiji import (
    Observation,
    Outcome,
    TaijiConfig,
    TSKV8Adapter,
    WorldAction,
    WorldDynamicsLearner,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldObject,
    WorldSchema,
    WorldSchemaBudgetError,
    WorldSchemaConflictError,
    WorldSchemaRegistry,
    WorldState,
    WorldTransition,
)


def _state(tick: int, *, target_id: str = "red", include_blue: bool = False) -> WorldState:
    objects = [
        WorldObject("agent", attributes={"energy": 1.0}),
        WorldObject(target_id, attributes={"position": float(tick)}),
    ]
    if include_blue and target_id != "blue":
        objects.append(WorldObject("blue", attributes={"charge": 0.5, "position": 2.0}))
    return WorldState(
        tick=tick,
        latent=torch.zeros(2),
        objects=tuple(objects),
        relations=(("agent", "near", target_id),),
    )


def _learner() -> WorldDynamicsLearner:
    initial = _state(0)
    expected = _state(1)
    action = WorldAction(
        "assemble-0",
        "assemble",
        0,
        actor_id="agent",
        target_id="red",
        parameters={"step": 1.0},
    )
    corpus = WorldInterventionCorpus(
        train=(
            WorldInterventionCase(
                case_id="assemble-0",
                initial=initial,
                action=action,
                expected_state=expected,
                expected_outcome=Outcome(
                    intent_id=action.action_id,
                    reward=1.0,
                    success=True,
                    tick=1,
                ),
            ),
        )
    )
    return WorldDynamicsLearner(WorldSchema.from_corpus(corpus), hidden_dim=12, seed=11)


def test_registry_alias_revision_rollback_and_conflict_are_checkpointable() -> None:
    learner = _learner()
    registry = learner.schema_registry
    assert registry.register_alias("ruby", "red") is True
    normalized = registry.normalize_state(_state(0, target_id="ruby"))
    assert normalized.objects[1].object_id == "red"
    assert (
        registry.normalize_action(
            WorldAction("alias-action", "assemble", 0, actor_id="agent", target_id="ruby")
        ).target_id
        == "red"
    )

    proposal = registry.propose_open_set(
        states=(_state(1, target_id="blue", include_blue=True),),
        actions=(
            WorldAction(
                "secure-1",
                "secure",
                1,
                actor_id="agent",
                target_id="blue",
                parameters={"strength": 0.75},
            ),
        ),
        evidence_ids=("episode-b",),
    )
    assert proposal is not None
    version = registry.commit(proposal)
    assert version == 1
    assert registry.active_version == version
    assert registry.lineage[-1]["evidence_ids"] == ["episode-b"]

    relation_key = ("relation", "agent", "tracks", "blue")
    assert registry.record_feedback(relation_key, 1.0) is True
    assert registry.record_feedback(relation_key, 0.0) is False
    assert registry.contradiction_count == 1
    assert registry.feature_confidence[relation_key] > 0.0

    checkpoint = registry.checkpoint()
    restored = WorldSchemaRegistry.from_checkpoint(checkpoint)
    assert restored.schema.payload() == registry.schema.payload()
    assert restored.aliases == registry.aliases
    assert restored.revision_versions == registry.revision_versions
    assert restored.contradiction_count == 1
    assert restored.lineage == registry.lineage

    assert learner.rollback_schema(0) is True
    assert learner.schema_registry.active_version == 0
    assert learner.schema == learner.schema_registry.schema


def test_registry_prune_tombstone_and_budget_fail_closed() -> None:
    learner = _learner()
    registry = learner.schema_registry
    feature = ("parameter", "step")
    budget = WorldSchemaRegistry(
        registry.schema, max_feature_count=registry.schema.input_dim + registry.schema.state_dim
    )
    with pytest.raises(WorldSchemaBudgetError):
        budget.propose_open_set(
            states=(_state(1, target_id="blue", include_blue=True),),
            actions=(
                WorldAction(
                    "secure-1",
                    "secure",
                    1,
                    actor_id="agent",
                    target_id="blue",
                    parameters={"strength": 0.75},
                ),
            ),
        )
    proposal = registry.propose_prune((feature,), evidence_ids=("resource-pressure",))
    registry.commit(proposal)
    assert feature in registry.tombstones
    with pytest.raises(WorldSchemaConflictError):
        registry.propose_open_set(
            states=(_state(0),),
            actions=(
                WorldAction(
                    "assemble-1",
                    "assemble",
                    0,
                    actor_id="agent",
                    target_id="red",
                    parameters={"step": 1.0},
                ),
            ),
        )


def _transition(
    action_id: str, *, after_position: float, reward: float, success: bool
) -> WorldTransition:
    before = _state(0)
    action = WorldAction(
        action_id,
        "assemble",
        0,
        actor_id="agent",
        target_id="red",
        parameters={"step": 1.0},
    )
    after = WorldState(
        tick=1,
        latent=torch.zeros(2),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": after_position}),
        ),
        relations=(("agent", "near", "red"),),
    )
    return WorldTransition(
        before=before,
        action=action,
        after=after,
        outcome=Outcome(
            intent_id=action_id,
            reward=reward,
            success=success,
            tick=1,
        ),
    )


def test_transition_adjudication_is_cross_episode_and_fail_closed() -> None:
    learner = _learner()
    registry = learner.schema_registry
    first = _transition("assemble-1", after_position=1.0, reward=1.0, success=True)
    repeat = _transition("assemble-2", after_position=1.0, reward=1.0, success=True)
    contradiction = _transition("assemble-3", after_position=2.0, reward=-1.0, success=False)

    assert registry.record_transition_outcome(first) is True
    key = registry.transition_evidence_key(first)
    assert registry.transition_outcome_count == 1
    assert registry.transition_confidence[key] == pytest.approx(1.0)
    assert registry.record_transition_outcome(repeat) is True
    assert registry.transition_confidence[key] == pytest.approx(1.0)
    assert registry.transition_outcome_estimate(key) == pytest.approx((1.0, 1.0))
    assert registry.transition_uncertainty(key) == (0.0, "deterministic")
    assert registry.record_transition_outcome(contradiction) is False
    assert registry.transition_outcome_count == 1
    assert registry.transition_confidence[key] == pytest.approx(2.0 / 3.0)
    assert registry.transition_outcome_estimate(key) == pytest.approx((1.0 / 3.0, 2.0 / 3.0))
    assert registry.transition_uncertainty(key) == (1.0, "conflicted")
    assert registry.transition_outcome_mode(key) == "conflicted"
    assert sorted(item["evidence_count"] for item in registry.transition_hypotheses[key]) == [1, 2]
    assert registry.contradiction_count == 1

    learner = _learner()
    assert learner.online_update(first, learning_rate=0.01, repeats=1)
    assert learner.online_update(repeat, learning_rate=0.01, repeats=1)
    updates_before = learner.online_updates
    assert learner.online_update(contradiction, learning_rate=0.01, repeats=1) == []
    assert learner.online_updates == updates_before == 2
    assert learner.transition_acceptances == 2
    assert learner.transition_rejections == 1

    restored = WorldSchemaRegistry.from_checkpoint(registry.checkpoint())
    assert restored.transition_outcome_count == 1
    assert restored.transition_confidence == registry.transition_confidence
    assert restored.record_transition_outcome(contradiction) is False


def test_transition_ledger_identifies_repeatable_stochastic_outcomes() -> None:
    registry = _learner().schema_registry
    first = _transition("stochastic-a1", after_position=1.0, reward=1.0, success=True)
    alternate = _transition("stochastic-b1", after_position=2.0, reward=-1.0, success=False)
    key = registry.transition_evidence_key(first)

    assert registry.record_transition_outcome(first) is True
    assert registry.transition_outcome_mode(key) == "deterministic"
    assert registry.record_transition_outcome(alternate) is False
    assert registry.transition_outcome_mode(key) == "conflicted"
    assert registry.record_transition_outcome(alternate) is True
    assert registry.record_transition_outcome(first) is False
    assert registry.transition_outcome_mode(key) == "stochastic"
    assert registry.record_transition_outcome(alternate) is True
    hypotheses = registry.transition_hypotheses[key]
    assert sorted(item["evidence_count"] for item in hypotheses) == [2, 3]
    assert registry.transition_confidence[key] == pytest.approx(0.6)
    assert registry.transition_outcome_mode(key) == "stochastic"
    uncertainty, uncertainty_mode = registry.transition_uncertainty(key)
    assert uncertainty == pytest.approx(0.4)
    assert uncertainty_mode == "stochastic"
    assert registry.transition_outcome_estimate(key) == pytest.approx((-0.2, 0.4))


def test_world_prediction_exposes_ledger_uncertainty_and_observed_outcome() -> None:
    learner = _learner()
    state = _state(0)
    action = WorldAction(
        "prediction",
        "assemble",
        0,
        actor_id="agent",
        target_id="red",
        parameters={"step": 1.0},
    )
    unseen = learner.predict(state, action)
    assert unseen.uncertainty == pytest.approx(1.0)
    assert unseen.uncertainty_mode == "unseen"

    learner.online_update(_transition("observed", after_position=1.0, reward=1.0, success=True))
    observed = learner.predict(state, action)
    assert observed.reward == pytest.approx(1.0)
    assert observed.success_probability == pytest.approx(1.0)
    assert observed.uncertainty == pytest.approx(0.0)
    assert observed.uncertainty_mode == "deterministic"


def _runtime_state(tick: int, *, position: float) -> WorldState:
    return WorldState(
        tick=tick,
        latent=torch.zeros(2),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": position}),
        ),
        relations=(("agent", "near", "red"),),
    )


def _run_runtime_transition(
    learner: WorldDynamicsLearner,
    *,
    episode_id: str,
    after_position: float,
    reward: float,
    success: bool,
) -> TSKV8Adapter:
    model = TSKV8Adapter(TaijiConfig(seed=11), episode_id=episode_id)
    model.attach_world_dynamics(learner)
    model.observe_event(
        Observation(
            modality="text-byte",
            value=97,
            timestamp=0,
            source="transition-adjudication-test",
        ),
        learn=False,
        world_state=_runtime_state(1, position=0.0),
    )
    before = model.cognitive_snapshot().world
    action = WorldAction(
        f"{episode_id}:assemble",
        "assemble",
        before.tick,
        actor_id="agent",
        target_id="red",
        parameters={"step": 1.0},
    )
    model.act((97, 98), sample=False, world_action=action)
    model.settle_action(
        reward,
        learn=False,
        learn_world=True,
        world_state=_runtime_state(before.tick + 1, position=after_position),
        success=success,
    )
    return model


def test_adapter_outcome_loop_calibrates_and_rejects_contradictory_episode() -> None:
    learner = _learner()
    first = _run_runtime_transition(
        learner,
        episode_id="episode-a",
        after_position=1.0,
        reward=1.0,
        success=True,
    )
    second = _run_runtime_transition(
        learner,
        episode_id="episode-b",
        after_position=1.0,
        reward=1.0,
        success=True,
    )
    rejected = _run_runtime_transition(
        learner,
        episode_id="episode-c",
        after_position=2.0,
        reward=-1.0,
        success=False,
    )

    assert learner.online_updates == 2
    assert learner.transition_acceptances == 2
    assert learner.transition_rejections == 1
    assert learner.schema_registry.transition_outcome_count == 1
    assert learner.schema_registry.contradiction_count == 1
    assert first.cognitive_snapshot().world_calibration_trace[-1].calibration_applied is True
    assert second.cognitive_snapshot().world_calibration_trace[-1].calibration_applied is True
    assert (
        second.cognitive_snapshot().world_calibration_trace[-1].prediction.uncertainty_mode
        == "deterministic"
    )
    assert second.cognitive_snapshot().world_calibration_trace[-1].prediction.uncertainty == 0.0
    rejected_trace = rejected.cognitive_snapshot().world_calibration_trace[-1]
    assert rejected_trace.calibration_applied is False
    assert rejected_trace.prediction.uncertainty_mode == "deterministic"
    assert rejected_trace.prediction.uncertainty == 0.0
    assert rejected_trace.online_update_count_before == 2
    assert rejected_trace.online_update_count_after == 2
    assert rejected.replan_required is True
    assert rejected_trace.adjudication == "rejected"
    assert rejected_trace.ledger_uncertainty_mode == "conflicted"
    assert rejected_trace.ledger_uncertainty == pytest.approx(1.0)
    assert rejected_trace.ledger_evidence_count == 3


def test_adapter_checkpoint_restores_registry_and_network_schema_snapshots() -> None:
    learner = _learner()
    new_state = _state(1, target_id="blue", include_blue=True)
    new_action = WorldAction(
        "secure-1",
        "secure",
        1,
        actor_id="agent",
        target_id="blue",
        parameters={"strength": 0.75},
    )
    learner.register_open_set(new_state, action=new_action)
    model = TSKV8Adapter(TaijiConfig(seed=11), episode_id="registry-checkpoint")
    model.attach_world_dynamics(learner)
    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())

    assert restored._world_dynamics is not None
    assert restored._world_dynamics.schema_registry.revision_versions == (0, 1)
    assert restored._world_dynamics.schema_registry.active_version == 1
    assert restored._world_dynamics.rollback_schema(0) is True
    assert restored._world_dynamics.schema_registry.active_version == 0

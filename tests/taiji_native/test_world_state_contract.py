from __future__ import annotations

import pytest
import torch

from taiji import (
    Observation,
    Outcome,
    TaijiWorldState,
    WorldAction,
    WorldAffordance,
    WorldEvent,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldObject,
    WorldState,
    WorldTransition,
)


def _world(tick: int, position: int) -> WorldState:
    return WorldState(
        tick=tick,
        latent=torch.tensor([float(position), float(tick)]),
        relations=(("agent", "near", "token"),),
        objects=(
            WorldObject(
                "agent",
                attributes={"position": position, "energy": 1.0},
                tags=("actor",),
            ),
            WorldObject("token", attributes={"position": position}, tags=("target",)),
        ),
        events=(
            WorldEvent(
                event_id=f"observation-{tick}",
                kind="observed",
                tick=tick,
                subject_id="agent",
                object_id="token",
            ),
        ),
        affordances=(
            WorldAffordance(
                affordance_id=f"move-{tick}",
                action_kind="move",
                actor_id="agent",
                target_id="token",
                parameters={"step": 1},
            ),
        ),
        uncertainty=0.2,
    )


def _transition(before: WorldState, after: WorldState, action_id: str) -> WorldTransition:
    action = WorldAction(
        action_id=action_id,
        kind="move",
        tick=before.tick,
        actor_id="agent",
        target_id="token",
        parameters={"step": 1},
    )
    outcome = Outcome(
        intent_id=action_id,
        reward=1.0,
        success=True,
        observation=Observation(
            modality="world-event",
            value={"kind": "moved", "position": after.objects[0].attribute("position")},
            timestamp=after.tick,
            source="test-world",
        ),
        tick=after.tick,
    )
    return WorldTransition(before=before, action=action, after=after, outcome=outcome)


def test_world_objects_events_and_interventions_round_trip() -> None:
    initial = _world(0, 0)
    expected = _world(1, 1)
    transition = _transition(initial, expected, "move-0")
    case = WorldInterventionCase(
        case_id="case-0",
        initial=initial,
        action=transition.action,
        expected_state=expected,
        expected_outcome=transition.outcome,
    )
    corpus = WorldInterventionCorpus(train=(case,), holdout=())

    restored = WorldInterventionCorpus.from_payload(corpus.to_payload())
    restored_case = restored.train[0]

    assert restored_case.case_id == case.case_id
    assert restored_case.initial.objects[0].attribute("position") == 0
    assert restored_case.initial.entities == ("agent", "token")
    assert restored_case.action.parameters == (("step", 1),)
    assert restored_case.expected_outcome.success is True
    assert torch.equal(restored_case.expected_state.latent, expected.latent)


def test_intervention_corpus_round_trips_time_shuffled_split() -> None:
    initial = _world(0, 0)
    expected = _world(1, 1)
    transition = _transition(initial, expected, "move-0")
    shuffled = WorldInterventionCase(
        case_id="shuffled-0",
        initial=initial,
        action=transition.action,
        expected_state=expected,
        expected_outcome=transition.outcome,
    )
    corpus = WorldInterventionCorpus(time_shuffled=(shuffled,))

    restored = WorldInterventionCorpus.from_payload(corpus.to_payload())

    assert len(restored.time_shuffled) == 1
    assert restored.time_shuffled[0].case_id == "shuffled-0"


def test_taiji_world_state_owns_transition_and_checkpoint() -> None:
    initial = _world(0, 0)
    expected = _world(1, 1)
    transition = _transition(initial, expected, "move-0")
    world = TaijiWorldState(initial)

    committed = world.apply(transition)
    assert committed.objects[0].attribute("position") == 1
    assert len(world.history) == 1

    restored = TaijiWorldState.from_checkpoint(world.checkpoint())
    assert restored.state.objects == world.state.objects
    assert torch.equal(restored.state.latent, world.state.latent)
    assert len(restored.history) == 1
    assert restored.history[0].action.action_id == "move-0"

    with pytest.raises(ValueError, match="owned current state"):
        restored.apply(_transition(_world(0, 0), _world(1, 1), "replayed-0"))


def test_taiji_world_state_synchronizes_same_tick_observation_without_transition() -> None:
    initial = _world(0, 0)
    expected = _world(1, 1)
    transition = _transition(initial, expected, "move-0")
    world = TaijiWorldState(initial)
    world.apply(transition)
    observed = WorldState(
        tick=expected.tick,
        latent=torch.tensor([9.0, 1.0]),
        objects=expected.objects,
        relations=expected.relations,
        events=expected.events + (WorldEvent("percept-1", "percept", expected.tick),),
        percept_event_id="percept-1",
        percept_assembly_id="assembly-1",
        percept_boundary_closed=True,
    )

    synchronized = world.synchronize_observation(observed)
    assert synchronized.percept_event_id == "percept-1"
    assert len(world.history) == 1
    assert world.history[0].after.percept_assembly_id == "assembly-1"
    restored = TaijiWorldState.from_checkpoint(world.checkpoint())
    assert restored.state.percept_boundary_closed is True

    with pytest.raises(ValueError, match="owned current tick"):
        world.synchronize_observation(_world(2, 2))


def test_taiji_world_state_advances_passive_observation_and_preserves_history() -> None:
    initial = _world(0, 0)
    expected = _world(1, 1)
    transition = _transition(initial, expected, "move-0")
    world = TaijiWorldState(initial)
    world.apply(transition)
    observed = _world(3, 2)

    advanced = world.advance_observation(observed)
    assert advanced.tick == 3
    assert len(world.history) == 1
    assert world.history[0].after.tick == 1
    restored = TaijiWorldState.from_checkpoint(world.checkpoint())
    assert restored.state.tick == 3
    assert len(restored.history) == 1


def test_world_intervention_splits_must_not_share_case_ids() -> None:
    initial = _world(0, 0)
    expected = _world(1, 1)
    transition = _transition(initial, expected, "move-0")
    case = WorldInterventionCase(
        case_id="duplicate",
        initial=initial,
        action=transition.action,
        expected_state=expected,
        expected_outcome=transition.outcome,
    )

    with pytest.raises(ValueError, match="disjoint"):
        WorldInterventionCorpus(train=(case,), holdout=(case,))

from __future__ import annotations

import torch

from taiji import (
    Outcome,
    WorldAction,
    WorldEpisode,
    WorldEpisodeCorpus,
    WorldEpisodeEvaluationConfig,
    WorldEpisodeEvaluator,
    WorldObject,
    WorldState,
    WorldTransition,
)


def _initial() -> WorldState:
    return WorldState(
        tick=0,
        latent=torch.zeros(1),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": 0.0}),
            WorldObject("blue", attributes={"position": 0.0}),
        ),
    )


def _step(state: WorldState, target: str, amount: float, index: int) -> WorldTransition:
    updated = {obj.object_id: dict(obj.attributes) for obj in state.objects}
    updated[target]["position"] += amount
    after = WorldState(
        tick=state.tick + 1,
        latent=state.latent,
        objects=tuple(
            WorldObject(obj.object_id, attributes=updated[obj.object_id], tags=obj.tags)
            for obj in state.objects
        ),
    )
    action = WorldAction(
        action_id=f"move-{index}",
        kind="move",
        tick=state.tick,
        actor_id="agent",
        target_id=target,
        parameters={"step": amount},
    )
    return WorldTransition(
        before=state,
        action=action,
        after=after,
        outcome=Outcome(
            intent_id=action.action_id,
            reward=amount,
            success=amount > 0.0,
            tick=after.tick,
        ),
    )


def _episode(episode_id: str, steps: tuple[tuple[str, float], ...]) -> WorldEpisode:
    state = _initial()
    transitions = []
    for index, (target, amount) in enumerate(steps):
        transition = _step(state, target, amount, index)
        transitions.append(transition)
        state = transition.after
    return WorldEpisode(episode_id=episode_id, initial=transitions[0].before, transitions=tuple(transitions))


def test_world_episode_contract_round_trip_and_contiguity() -> None:
    episode = _episode("episode-0", (("red", 1.0), ("blue", -1.0)))
    corpus = WorldEpisodeCorpus(train=(episode,), holdout=())

    restored = WorldEpisodeCorpus.from_payload(corpus.to_payload())

    assert restored.train[0].final_state.tick == 2
    assert restored.train[0].transitions[1].before.tick == 1


def test_world_episode_evaluator_rolls_out_and_recovers_checkpoint() -> None:
    corpus = WorldEpisodeCorpus(
        train=(
            _episode("train-0", (("red", 1.0), ("blue", -1.0))),
            _episode("train-1", (("red", -1.0), ("blue", 1.0))),
        ),
        holdout=(_episode("unseen-episode", (("red", 1.0), ("blue", 1.0))),),
    )

    report = WorldEpisodeEvaluator(
        WorldEpisodeEvaluationConfig(seeds=(11, 29), hidden_dim=32, epochs=350)
    ).evaluate(corpus)

    assert report["gate"]["checkpoint_recovery"] is True
    assert report["gate"]["passed"] is True
    assert all(item["schema_uses_episode_id"] is False for item in report["seeds"])

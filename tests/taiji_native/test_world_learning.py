from __future__ import annotations

from copy import deepcopy

import torch

from scripts.training.eval_taiji_a2_world import build_corpus
from taiji import (
    Outcome,
    WorldAction,
    WorldDynamicsLearner,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldInterventionEvaluationConfig,
    WorldInterventionEvaluator,
    WorldObject,
    WorldSchema,
    WorldState,
    WorldTransition,
)


def _state() -> WorldState:
    return WorldState(
        tick=0,
        latent=torch.zeros(2),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": 0.0}),
            WorldObject("blue", attributes={"position": 0.0}),
        ),
    )


def _case(target: str, step: float, index: int) -> WorldInterventionCase:
    initial = _state()
    updated = {
        object_id: dict(obj.attributes) for object_id, obj in ((obj.object_id, obj) for obj in initial.objects)
    }
    updated[target]["position"] += step
    expected = WorldState(
        tick=1,
        latent=initial.latent,
        objects=tuple(
            WorldObject(obj.object_id, attributes=updated[obj.object_id], tags=obj.tags)
            for obj in initial.objects
        ),
    )
    action_id = f"move-{index}"
    action = WorldAction(
        action_id=action_id,
        kind="move",
        tick=0,
        actor_id="agent",
        target_id=target,
        parameters={"step": step},
    )
    return WorldInterventionCase(
        case_id=f"case-{index}",
        initial=initial,
        action=action,
        expected_state=expected,
        expected_outcome=Outcome(
            intent_id=action_id,
            reward=step,
            success=step > 0.0,
            tick=1,
        ),
    )


def _corpus() -> WorldInterventionCorpus:
    cases = []
    index = 0
    for target in ("red", "blue"):
        for step in (-2.0, -1.0, 1.0, 2.0):
            if (target, step) in (("red", 2.0), ("blue", -2.0)):
                continue
            cases.append(_case(target, step, index))
            index += 1
    holdout = (_case("red", 2.0, 100), _case("blue", -2.0, 101))
    return WorldInterventionCorpus(train=tuple(cases), holdout=holdout)


def test_world_dynamics_learns_target_bound_state_change() -> None:
    corpus = _corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=11)
    learner.fit(corpus.train, epochs=350, learning_rate=0.01)

    full = tuple(learner.predict(case.initial, case.action) for case in corpus.holdout)
    lesion = tuple(
        learner.predict(case.initial, case.action, bind_target=False) for case in corpus.holdout
    )
    full_error = sum(
        float(torch.mean((schema.state_values(prediction.state) - schema.state_values(case.expected_state)) ** 2))
        for prediction, case in zip(full, corpus.holdout, strict=True)
    ) / len(full)
    lesion_error = sum(
        float(torch.mean((schema.state_values(prediction.state) - schema.state_values(case.expected_state)) ** 2))
        for prediction, case in zip(lesion, corpus.holdout, strict=True)
    ) / len(lesion)

    assert full_error < 0.2
    assert lesion_error - full_error > 0.1


def test_world_intervention_evaluator_reports_a2_gate() -> None:
    report = WorldInterventionEvaluator(
        WorldInterventionEvaluationConfig(
            seeds=(11, 29),
            hidden_dim=32,
            epochs=350,
            learning_rate=0.01,
        )
    ).evaluate(_corpus())

    assert report["format"] == "taiji-a2-world-intervention-v1"
    assert report["schema"]["target_ids"] == ["blue", "red"]
    assert report["gate"]["passed"] is True
    assert report["gate"]["binding_drop_min"] > 0.1


def test_expanded_world_benchmark_contains_relation_and_time_controls() -> None:
    corpus = build_corpus()
    schema = WorldSchema.from_corpus(corpus)
    report = WorldInterventionEvaluator(
        WorldInterventionEvaluationConfig(seeds=(11,), hidden_dim=32, epochs=350)
    ).evaluate(corpus)

    assert schema.relation_slots == (("blue", "near", "red"), ("red", "near", "blue"))
    assert report["time_shuffled_cases"] == 1
    assert report["seeds"][0]["time_shuffled"]["success_accuracy"] == 1.0


def test_online_world_error_update_beats_no_update_lesion() -> None:
    corpus = _corpus()
    schema = WorldSchema.from_corpus(corpus)
    learner = WorldDynamicsLearner(schema, hidden_dim=32, seed=11)
    learner.fit(corpus.train, epochs=350, learning_rate=0.01)
    lesion = deepcopy(learner)
    case = corpus.holdout[0]
    transition = WorldTransition(
        before=case.initial,
        action=case.action,
        after=case.expected_state,
        outcome=case.expected_outcome,
    )

    def error(model: WorldDynamicsLearner) -> float:
        prediction = model.predict(case.initial, case.action)
        state_error = torch.mean(
            (schema.state_values(prediction.state) - schema.state_values(case.expected_state)) ** 2
        )
        return float(state_error + (prediction.reward - case.expected_outcome.reward) ** 2)

    before = error(learner)
    no_update = error(lesion)
    learner.online_update(transition, learning_rate=0.01, repeats=50)
    after = error(learner)

    assert after < before
    assert abs(no_update - before) < 1e-8
    assert learner.online_updates == 1

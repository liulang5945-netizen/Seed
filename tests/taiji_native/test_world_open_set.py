from __future__ import annotations

import torch

from taiji import (
    Outcome,
    WorldAction,
    WorldDynamicsLearner,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldObject,
    WorldSchema,
    WorldState,
)


def _state(tick: int, *, target_id: str = "red", include_blue: bool = False) -> WorldState:
    objects = [
        WorldObject("agent", attributes={"energy": 1.0}),
        WorldObject(target_id, attributes={"position": float(tick)}),
    ]
    if include_blue:
        objects.append(WorldObject("blue", attributes={"position": 2.0}))
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
    learner = WorldDynamicsLearner(WorldSchema.from_corpus(corpus), hidden_dim=12, seed=11)
    learner.fit(corpus.train, epochs=3, learning_rate=0.01)
    return learner


def test_open_set_registration_grows_schema_and_preserves_old_weights() -> None:
    learner = _learner()
    old_schema = learner.schema
    old_input = learner._input_layer.weight.detach().clone()
    old_output = learner._output_layer.weight.detach().clone()
    old_input_keys = old_schema.input_feature_keys
    old_output_keys = old_schema.state_feature_keys

    new_state = WorldState(
        tick=1,
        latent=torch.zeros(2),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": 1.0}),
            WorldObject("blue", attributes={"position": 2.0, "charge": 0.5}),
        ),
        relations=(
            ("agent", "near", "red"),
            ("agent", "tracks", "blue"),
        ),
    )
    new_action = WorldAction(
        "secure-1",
        "secure",
        1,
        actor_id="agent",
        target_id="blue",
        parameters={"strength": 0.75},
    )

    assert learner.register_open_set(new_state, action=new_action) is True
    assert learner.schema.open_set is True
    assert "blue" in learner.schema.object_ids
    assert ("blue", "charge") in learner.schema.state_slots
    assert ("agent", "tracks", "blue") in learner.schema.relation_slots
    assert "secure" in learner.schema.action_kinds
    assert "blue" in learner.schema.target_ids
    assert "strength" in learner.schema.parameter_names

    new_input_keys = learner.schema.input_feature_keys
    new_output_keys = learner.schema.state_feature_keys
    for key, old_index in zip(old_input_keys, range(len(old_input_keys)), strict=True):
        assert torch.equal(
            learner._input_layer.weight[:, new_input_keys.index(key)], old_input[:, old_index]
        )
    for key, old_index in zip(old_output_keys, range(len(old_output_keys)), strict=True):
        assert torch.equal(
            learner._output_layer.weight[new_output_keys.index(key)], old_output[old_index]
        )

    missing_new_object = _state(1)
    assert learner.schema.state_values(missing_new_object).shape == (learner.schema.state_dim,)
    prediction = learner.predict(missing_new_object, new_action)
    assert prediction.state.tick == missing_new_object.tick + 1
    assert learner.schema == WorldSchema.from_payload(learner.schema.payload())


def test_open_set_runtime_can_ignore_action_metadata_parameters() -> None:
    learner = _learner()
    action = WorldAction(
        "assemble-1",
        "assemble",
        0,
        actor_id="agent",
        target_id="red",
        parameters={"runtime_metadata": 1.0},
    )

    assert learner.register_open_set(_state(0), action=action, register_parameters=False) is False
    assert "runtime_metadata" not in learner.schema.parameter_names
    learner.predict(_state(0), action, register_parameters=False)

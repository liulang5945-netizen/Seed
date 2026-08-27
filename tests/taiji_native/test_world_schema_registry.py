from __future__ import annotations

import pytest
import torch

from taiji import (
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

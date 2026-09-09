from __future__ import annotations

import copy

import pytest

from taiji import (
    KAdapterExchange,
    KAdapterInput,
    KAdapterOutput,
    KContinualAdapter,
    OutcomeDependencyProjection,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    WorldEvent,
    WorldState,
    content_digest,
)


def _projection(scope_id: str = "r6-scope"):
    world = WorldState(tick=0, entities=("workbench",), uncertainty=0.0)
    event = WorldEvent(
        event_id="outcome-0",
        kind="workbench.evidence",
        tick=0,
        subject_id="workspace.read",
        attributes=(
            ("capability_id", "workspace.read"),
            ("success", True),
        ),
        provenance="workbench-observed",
    )
    spec = OutcomeDependencySpec(
        dependency_id="dep-0",
        next_task_id="task-follow-up",
        capability_id="workspace.read",
        required_outcome="success",
    )
    return OutcomeDependencyProjector(scope_id).project(world, event, spec)


def _adapter() -> tuple[KContinualAdapter, dict[str, object]]:
    parent = {
        "format": "r6-parent-v1",
        "owner": "taiji",
        "revision": 0,
    }
    adapter = KContinualAdapter(
        parent_checkpoint_digest=content_digest(parent),
        owner_graph_digest="1" * 64,
        source_manifest_digest="2" * 64,
        resource_manifest_digest="3" * 64,
        dependency_scope_id="r6-scope",
    )
    return adapter, parent


def _exchange(
    adapter: KContinualAdapter,
    projection: OutcomeDependencyProjection,
    parent: dict[str, object],
) -> KAdapterExchange:
    input_item = KAdapterInput(
        episode_id="r6-smoke-episode",
        parent_checkpoint_digest=content_digest(parent),
        observation_digest=content_digest({"path": "missing_00.txt"}),
        world_digest=content_digest({"world": "r6"}),
        goal_digest=content_digest({"goal": "read"}),
        content_plan_digest=content_digest({"content": "inspect"}),
        source_manifest_digest=adapter.source_manifest_digest,
        tick=1,
    )
    output_item = KAdapterOutput(
        parent_checkpoint_digest=content_digest(parent),
        input_digest=input_item.input_digest,
        action_digest=content_digest({"kind": "workspace.read"}),
        outcome_signature=projection.outcome_signature,
        dependency_digest=projection.dependency_digest,
        dependency_projection_digest=projection.projection_digest,
        success=True,
        lineage=(
            adapter.dependency_scope_id,
            input_item.input_digest,
            projection.projection_digest,
            projection.dependency_digest,
        ),
    )
    return KAdapterExchange.create(
        scope_id=adapter.dependency_scope_id,
        input=input_item,
        output=output_item,
    )


def test_same_parent_adapter_roundtrip_and_rollback() -> None:
    adapter, parent = _adapter()
    projection = _projection()

    assert adapter.parent_checkpoint_matches(parent)
    adapter.bind_dependency_projection(projection)
    preflight_checkpoint = adapter.checkpoint()
    restored = KContinualAdapter.from_checkpoint(preflight_checkpoint)
    assert restored.checkpoint() == preflight_checkpoint

    rollback_token = adapter.stage_candidate(
        candidate_checkpoint_digest="4" * 64,
        candidate_owner_graph_digest="5" * 64,
        candidate_source_manifest_digest="6" * 64,
        candidate_parent_checkpoint_digest=content_digest(parent),
    )
    staged_checkpoint = adapter.checkpoint()
    staged_restored = KContinualAdapter.from_checkpoint(staged_checkpoint)
    assert staged_restored.checkpoint() == staged_checkpoint
    assert staged_restored.active_namespace == staged_restored.candidate_namespace

    record = adapter.rollback(rollback_token)
    assert record.status == "rolled_back"
    assert record.reason == "explicit_parent_restore"
    assert record.trial_id == staged_restored._staged_trial_id

    rollback_checkpoint = adapter.checkpoint()
    rollback_restored = KContinualAdapter.from_checkpoint(rollback_checkpoint)
    assert rollback_restored.checkpoint() == rollback_checkpoint
    assert rollback_restored.active_namespace == rollback_restored.parent_namespace
    assert rollback_restored.dependency_projection == projection
    assert rollback_restored.training_steps == 0


def test_same_parent_adapter_records_typed_exchange_and_preserves_it_on_rollback() -> None:
    adapter, parent = _adapter()
    projection = _projection()
    adapter.bind_dependency_projection(projection)
    exchange = _exchange(adapter, projection, parent)

    assert adapter.record_exchange(exchange) == exchange.exchange_digest
    checkpoint = adapter.checkpoint()
    restored = KContinualAdapter.from_checkpoint(checkpoint)
    assert restored.last_exchange == exchange

    token = adapter.stage_candidate(
        candidate_checkpoint_digest="4" * 64,
        candidate_owner_graph_digest="5" * 64,
        candidate_source_manifest_digest="6" * 64,
        candidate_parent_checkpoint_digest=content_digest(parent),
    )
    adapter.rollback(token)
    assert adapter.last_exchange == exchange
    assert KContinualAdapter.from_checkpoint(adapter.checkpoint()).last_exchange == exchange


def test_same_parent_adapter_rejects_tampered_exchange() -> None:
    adapter, parent = _adapter()
    projection = _projection()
    adapter.bind_dependency_projection(projection)
    payload = copy.deepcopy(_exchange(adapter, projection, parent).to_payload())
    payload["output"]["action_digest"] = "8" * 64

    with pytest.raises(ValueError, match="output digest mismatch"):
        KAdapterExchange.from_payload(payload)


def test_same_parent_adapter_rejects_scope_crossing_and_parent_crossing() -> None:
    adapter, parent = _adapter()
    with pytest.raises(ValueError, match="dependency scope mismatch"):
        adapter.bind_dependency_projection(_projection("other-scope"))

    adapter.bind_dependency_projection(_projection())
    with pytest.raises(ValueError, match="crosses the parent checkpoint"):
        adapter.stage_candidate(
            candidate_checkpoint_digest="4" * 64,
            candidate_owner_graph_digest="5" * 64,
            candidate_source_manifest_digest="6" * 64,
            candidate_parent_checkpoint_digest="7" * 64,
        )
    changed_parent = dict(parent)
    changed_parent["revision"] = 1
    assert not adapter.parent_checkpoint_matches(changed_parent)


def test_same_parent_adapter_checkpoint_is_tamper_evident() -> None:
    adapter, _ = _adapter()
    adapter.bind_dependency_projection(_projection())
    payload = copy.deepcopy(adapter.checkpoint())
    payload["owner_graph_digest"] = "8" * 64

    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        KContinualAdapter.from_checkpoint(payload)

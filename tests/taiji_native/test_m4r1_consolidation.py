from __future__ import annotations

from copy import deepcopy

from taiji import (
    Taiji,
    TaijiConfig,
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
)
from taiji.internalization import content_digest


def _boundary(*, task_id: str, generation_scope: str) -> WorkbenchTaskBoundary:
    return WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id=task_id,
        session_id="session:m4r1",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope=generation_scope,
        issued_tick=10,
        ttl_ticks=20,
    )


def _authorization(boundary: WorkbenchTaskBoundary) -> WorkbenchBoundaryAuthorization:
    return WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=("workspace.read",),
        active_boundary_digest=boundary.token_digest,
        current_tick=12,
        usage="execute",
    )


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(32,),
        synapse_fan_in=8,
        motor_fan_in=16,
        memory_units=32,
        memory_fan_in=8,
        memory_readout_fan_in=16,
        memory_meta_dim=16,
        seed=71,
    )


def test_m4r1_zero_strength_is_backward_compatible() -> None:
    source = Taiji(_config(), episode_id="m4r1-zero-source")
    source.learn_bytes(b"old-old-old", epochs=2, learn_fabric=False, learn_predictive_context=False)
    source_checkpoint = source.checkpoint()
    active = _boundary(task_id="task:m4r1-zero", generation_scope="active")
    authorization = _authorization(active)

    baseline = Taiji.from_checkpoint(deepcopy(source_checkpoint))
    candidate = Taiji.from_checkpoint(deepcopy(source_checkpoint))
    for model in (baseline, candidate):
        model.clone_protected_predictive_readout_as_active(boundary_digest=active.token_digest)
    baseline.learn_bytes(
        b"new-new-new",
        learn_fabric=False,
        learn_predictive_context=False,
        boundary=active,
        authorization=authorization,
    )
    candidate.learn_bytes(
        b"new-new-new",
        learn_fabric=False,
        learn_predictive_context=False,
        consolidation_strength=0.0,
        boundary=active,
        authorization=authorization,
    )

    assert content_digest(baseline.checkpoint()) == content_digest(candidate.checkpoint())


def test_m4r1_consolidation_isolated_and_checkpointable() -> None:
    model = Taiji(_config(), episode_id="m4r1-positive")
    model.learn_bytes(b"old-old-old", epochs=2, learn_fabric=False, learn_predictive_context=False)
    active = _boundary(task_id="task:m4r1-positive", generation_scope="active")
    authorization = _authorization(active)
    model.clone_protected_predictive_readout_as_active(boundary_digest=active.token_digest)
    protected_before = model.readout_registry_status()["protected"]["readout_digest"]
    active_before = model.active_predictive_readout_metadata
    assert active_before is not None
    fabric_before = content_digest(model.fabric.to_payload())
    context_before = content_digest(model.predictive_context.to_payload())

    model.learn_bytes(
        b"new-new-new",
        epochs=2,
        learn_fabric=False,
        learn_predictive_context=False,
        consolidation_strength=0.5,
        boundary=active,
        authorization=authorization,
    )

    active_after = model.active_predictive_readout_metadata
    assert active_after is not None
    assert active_after["readout_digest"] != active_before["readout_digest"]
    assert model.readout_registry_status()["protected"]["readout_digest"] == protected_before
    assert content_digest(model.fabric.to_payload()) == fabric_before
    assert content_digest(model.predictive_context.to_payload()) == context_before

    checkpoint = model.checkpoint()
    restored = Taiji.from_checkpoint(deepcopy(checkpoint))
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)
    assert restored.active_predictive_readout_metadata == active_after


def test_m4r1_consolidation_requires_active_execution_boundary() -> None:
    model = Taiji(_config(), episode_id="m4r1-gate")
    try:
        model.learn_bytes(b"new", consolidation_strength=0.5)
    except RuntimeError as exc:
        assert "active readout boundary" in str(exc)
    else:
        raise AssertionError("consolidation must require an active boundary")

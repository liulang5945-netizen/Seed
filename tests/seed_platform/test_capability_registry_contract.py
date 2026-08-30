from __future__ import annotations

import pytest

from seed_platform.capability_registry import (
    CapabilityBundle,
    CapabilityCandidate,
    CapabilityRegistry,
)
from seed_platform.workbench import WorkbenchActionRequest, WorkbenchEnvironment


def _read_bundle() -> CapabilityBundle:
    return CapabilityBundle(
        capability_id="workspace.read",
        schema={"type": "object", "properties": {"path": {"type": "string"}}},
        permissions=("workspace.read",),
        executor_id="seed.workbench.workspace.read",
        executor_version="1.0.0",
    )


def _write_bundle() -> CapabilityBundle:
    return CapabilityBundle(
        capability_id="workspace.apply_patch",
        schema={"type": "object", "properties": {"patch": {"type": "string"}}},
        effect="file_write",
        risk="file_write",
        permissions=("workspace.write",),
        executor_id="seed.workbench.workspace.apply_patch",
        executor_version="1.0.0",
        disposer_id="seed.workbench.workspace.undo",
        disposer_version="1.0.0",
    )


def _read_bundle_v2() -> CapabilityBundle:
    return CapabilityBundle(
        capability_id="workspace.read",
        schema={"type": "object", "properties": {"path": {"type": "string"}}},
        permissions=("workspace.read",),
        executor_id="seed.workbench.workspace.read.v2",
        executor_version="2.0.0",
    )


def _read_candidate() -> CapabilityCandidate:
    return CapabilityCandidate(
        bundle=_read_bundle(),
        rationale="replace ad hoc read dispatch with a verified registry identity",
        evidence_digests=("evidence:workbench-read", "evidence:rollback-canary"),
        resource_budget={"max_cpu_ms": 25, "max_output_bytes": 2048},
        evaluation_gates=("shadow_equivalence", "rollback"),
        metadata={"origin": "r5b-l1-canary"},
    )


def test_registration_is_content_addressed_and_does_not_auto_activate() -> None:
    registry = CapabilityRegistry(parent_checkpoint_id="checkpoint:parent")
    bundle = _read_bundle()

    record = registry.register(bundle)

    assert record.status == "validated"
    assert registry.snapshot.bundles == ()
    with pytest.raises(PermissionError, match="not active"):
        registry.resolve(bundle.capability_id, snapshot_id=registry.snapshot_id)

    restored = CapabilityRegistry.from_checkpoint(registry.checkpoint())
    assert restored.snapshot.to_payload() == registry.snapshot.to_payload()
    assert restored.records[0].audit_digest == record.audit_digest


def test_candidate_proposal_is_content_addressed_and_not_executable() -> None:
    registry = CapabilityRegistry(parent_checkpoint_id="checkpoint:candidate")
    candidate = _read_candidate()

    proposed = registry.propose(candidate)

    assert proposed.status == "proposed"
    assert registry.snapshot.bundles == ()
    assert registry.get_record(candidate.bundle.bundle_digest) is None
    assert registry.get_candidate_record(candidate.candidate_digest) == proposed
    with pytest.raises(PermissionError, match="not active"):
        registry.resolve(candidate.bundle.capability_id, snapshot_id=registry.snapshot_id)

    restored = CapabilityRegistry.from_checkpoint(registry.checkpoint())
    assert restored.get_candidate_record(candidate.candidate_digest) == proposed


def test_candidate_validation_stays_non_active_until_shadow_and_approval() -> None:
    registry = CapabilityRegistry()
    candidate = _read_candidate()
    registry.propose(candidate)

    validated = registry.validate_candidate(
        candidate.candidate_digest,
        validation_ref="validation:r5b-l1",
    )

    assert validated.status == "validated"
    assert registry.snapshot.bundles == ()
    assert registry.get_candidate_record(candidate.candidate_digest).status == "validated"
    with pytest.raises(PermissionError, match="activation requires"):
        registry.activate(candidate.bundle.bundle_digest, approval_id="")

    registry.shadow(candidate.bundle.bundle_digest)
    active = registry.activate(
        candidate.bundle.bundle_digest,
        approval_id="approval:r5b-l1",
        expected_snapshot_id=registry.snapshot_id,
    )
    assert active.status == "active"


def test_candidate_rejection_and_nested_executable_source_fail_closed() -> None:
    registry = CapabilityRegistry()
    candidate = _read_candidate()
    registry.propose(candidate)
    rejected = registry.reject_candidate(
        candidate.candidate_digest,
        decision_ref="review:r5b-l1",
        reason="oracle coverage is insufficient",
    )

    assert rejected.status == "rejected"
    assert registry.get_record(candidate.bundle.bundle_digest) is None

    payload = _read_candidate().to_payload()
    payload["bundle"]["metadata"]["source_path"] = "executor.py"
    with pytest.raises(ValueError, match="forbidden executable-source"):
        CapabilityCandidate.from_payload(payload)


def test_shadow_requires_approval_before_activation_and_binds_snapshot() -> None:
    registry = CapabilityRegistry()
    bundle = _read_bundle()
    registry.register(bundle)
    shadow = registry.shadow(bundle.bundle_digest)
    assert shadow.status == "shadow"

    with pytest.raises(PermissionError, match="activation requires"):
        registry.activate(
            bundle.bundle_digest,
            approval_id="",
            expected_snapshot_id=registry.snapshot_id,
        )

    active = registry.activate(
        bundle.bundle_digest,
        approval_id="approval:read",
        expected_snapshot_id=registry.snapshot_id,
    )
    assert active.status == "active"
    assert registry.resolve(bundle.capability_id, snapshot_id=registry.snapshot_id) == bundle
    assert {
        "bundle_registered",
        "schema_validated",
        "executor_precompiled",
        "candidate_shadowed",
        "approval_recorded",
        "snapshot_activated",
    }.issubset(active.events)


def test_side_effecting_bundle_requires_disposer_and_source_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires a disposer"):
        CapabilityBundle(
            capability_id="workspace.apply_patch",
            schema={},
            effect="file_write",
            risk="file_write",
            executor_id="executor",
            executor_version="1.0.0",
        )

    payload = _read_bundle().to_payload()
    payload["source_path"] = "executor.py"
    with pytest.raises(ValueError, match="forbidden executable-source"):
        CapabilityBundle.from_payload(payload)


def test_stale_snapshot_and_tombstoned_bundle_fail_closed() -> None:
    registry = CapabilityRegistry()
    bundle = _read_bundle()
    registry.register(bundle)
    registry.shadow(bundle.bundle_digest)
    parent_snapshot = registry.snapshot_id
    registry.activate(bundle.bundle_digest, approval_id="approval:read")

    with pytest.raises(ValueError, match="snapshot is stale"):
        registry.resolve(bundle.capability_id, snapshot_id=parent_snapshot)

    registry.retire(bundle.bundle_digest, expected_snapshot_id=registry.snapshot_id)
    registry.tombstone(bundle.bundle_digest)
    with pytest.raises(ValueError, match="terminal"):
        registry.rollback(bundle.bundle_digest)
    with pytest.raises(PermissionError, match="not active"):
        registry.resolve(bundle.capability_id, snapshot_id=registry.snapshot_id)


def test_checkpoint_restores_active_registry_without_resurrecting_terminal_bundle() -> None:
    registry = CapabilityRegistry(policy_revision=3, parent_checkpoint_id="checkpoint:trial")
    read_bundle = _read_bundle()
    write_bundle = _write_bundle()
    for bundle in (read_bundle, write_bundle):
        registry.register(bundle)
        registry.shadow(bundle.bundle_digest)
    registry.activate(read_bundle.bundle_digest, approval_id="approval:read")
    registry.activate(write_bundle.bundle_digest, approval_id="approval:write")
    registry.retire(write_bundle.bundle_digest, expected_snapshot_id=registry.snapshot_id)
    registry.tombstone(write_bundle.bundle_digest)

    restored = CapabilityRegistry.from_checkpoint(registry.checkpoint())

    assert restored.snapshot_id == registry.snapshot_id
    assert (
        restored.resolve(read_bundle.capability_id, snapshot_id=restored.snapshot_id) == read_bundle
    )
    assert restored.get_record(write_bundle.bundle_digest).status == "tombstoned"
    with pytest.raises(PermissionError, match="not active"):
        restored.resolve(write_bundle.capability_id, snapshot_id=restored.snapshot_id)


def test_replacement_checkpoint_and_rollback_restore_the_parent_active_bundle() -> None:
    registry = CapabilityRegistry(parent_checkpoint_id="checkpoint:replacement")
    old_bundle = _read_bundle()
    new_bundle = _read_bundle_v2()
    registry.register(old_bundle)
    registry.shadow(old_bundle.bundle_digest)
    registry.activate(old_bundle.bundle_digest, approval_id="approval:old")
    registry.register(new_bundle)
    registry.shadow(new_bundle.bundle_digest)

    replaced = registry.replace(
        old_bundle.bundle_digest,
        new_bundle.bundle_digest,
        approval_id="approval:replacement",
        expected_snapshot_id=registry.snapshot_id,
    )
    restored = CapabilityRegistry.from_checkpoint(registry.checkpoint())
    rolled_back = restored.rollback(
        new_bundle.bundle_digest,
        expected_snapshot_id=restored.snapshot_id,
    )

    assert replaced.status == "active"
    assert rolled_back.status == "rolled_back"
    assert restored.resolve("workspace.read", snapshot_id=restored.snapshot_id) == old_bundle
    assert restored.get_record(old_bundle.bundle_digest).status == "active"


def test_workbench_dispatch_resolves_executor_from_active_registry(tmp_path) -> None:
    with (tmp_path / "README.md").open("w", encoding="utf-8", newline="") as handle:
        handle.write("registry dispatch\n")
    environment = WorkbenchEnvironment(tmp_path)

    registered = environment.capability_registry.resolve("workspace.read")
    result = environment.execute_tool("workspace.read", {"path": "README.md"})

    assert registered.executor_id == "workspace.read"
    assert result.success is True
    assert environment.last_result["content"] == "registry dispatch\n"

    environment._capability_executors["workspace.read"] = lambda _parameters: {
        "success": True,
        "operation": "registry-canary",
    }
    canary = environment.execute_tool("workspace.read", {"path": "README.md"})
    assert canary.success is True
    assert environment.last_result["operation"] == "registry-canary"


def test_workbench_policy_fails_closed_when_registry_lacks_capability(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path, capability_registry=CapabilityRegistry())
    request = WorkbenchActionRequest(
        request_id="registry-policy-request",
        intent_id="registry-policy-intent",
        capability_id="workspace.read",
        parameters={"path": "README.md"},
        snapshot_id=environment.capability_snapshot.snapshot_id,
    )

    decision = environment.policy_for(request)

    assert decision.decision == "deny"
    assert decision.reason_code == "capability_not_connected"


def test_workbench_request_binds_registry_snapshot_and_rejects_stale_dispatch(tmp_path) -> None:
    environment = WorkbenchEnvironment(tmp_path)
    request = WorkbenchActionRequest(
        request_id="stale-registry-request",
        intent_id="stale-registry-intent",
        capability_id="workspace.read",
        parameters={"path": "README.md"},
        snapshot_id=environment.capability_snapshot.snapshot_id,
        capability_registry_snapshot_id="stale-registry-snapshot",
    )

    decision = environment.policy_for(request)
    result = environment.execute_tool(
        "workspace.read",
        request.parameters,
        capability_registry_snapshot_id=request.capability_registry_snapshot_id,
    )

    assert decision.reason_code == "stale_capability_registry"
    assert result.success is False
    assert environment.last_result["error_code"] == "stale_capability_registry"


def test_workbench_registry_covers_every_enabled_capability_without_legacy_chain(tmp_path) -> None:
    from inspect import getsource

    environment = WorkbenchEnvironment(tmp_path)
    enabled_ids = {
        item.capability_id for item in environment.capability_snapshot.capabilities if item.enabled
    }
    registry_ids = {item.capability_id for item in environment.capability_registry.snapshot.bundles}

    assert registry_ids == enabled_ids
    assert registry_ids == set(environment._capability_executors)
    assert "elif tool_name" not in getsource(WorkbenchEnvironment.execute_tool)


def test_workbench_replacement_is_stale_bound_and_rollback_restores_old_executor(tmp_path) -> None:
    with (tmp_path / "README.md").open("w", encoding="utf-8", newline="") as handle:
        handle.write("replacement dispatch\n")
    environment = WorkbenchEnvironment(tmp_path)
    registry = environment.capability_registry
    old_bundle = registry.resolve("workspace.read")
    old_registry_snapshot = registry.snapshot_id
    new_bundle = _read_bundle_v2()
    registry.register(new_bundle)
    registry.shadow(new_bundle.bundle_digest)
    registry.replace(
        old_bundle.bundle_digest,
        new_bundle.bundle_digest,
        approval_id="approval:replacement",
        expected_snapshot_id=old_registry_snapshot,
    )
    environment._capability_executors[new_bundle.executor_id] = environment._read_workspace

    stale = environment.execute_tool(
        "workspace.read",
        {"path": "README.md"},
        capability_registry_snapshot_id=old_registry_snapshot,
    )
    stale_error = dict(environment.last_result)
    current = environment.execute_tool(
        "workspace.read",
        {"path": "README.md"},
        capability_registry_snapshot_id=registry.snapshot_id,
    )
    restored_registry = CapabilityRegistry.from_checkpoint(registry.checkpoint())
    restored_registry.rollback(
        new_bundle.bundle_digest,
        expected_snapshot_id=restored_registry.snapshot_id,
    )

    assert stale.success is False
    assert stale_error["error_code"] == "stale_capability_registry"
    assert current.success is True
    assert restored_registry.resolve("workspace.read") == old_bundle

from __future__ import annotations

import pytest

from seed_platform.mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
)
from seed_platform.mcp_client_capability_registry import McpClientCapabilityShadowRegistry
from seed_platform.mcp_client_connection_authorization import authorize_mcp_client_connection
from seed_platform.mcp_client_connection_target import (
    McpClientConnectionTarget,
    McpClientConnectionTargetStore,
    bind_mcp_client_connection_target,
)
from seed_platform.mcp_registry import McpToolRegistry


def _authorized():
    registry = McpToolRegistry.default()
    candidate = McpCapabilityInheritanceCandidate.from_registry(
        registry,
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:target:test",
        tool_ids=(registry.list_tools()[0].tool_id,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:target:test",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:target:test",
        rationale="test an explicit target declaration",
    )
    policy = McpCapabilityInheritancePolicy(allowed_server_ids=(candidate.server_id,))
    shadow_registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        registry,
        parent_checkpoint_id="checkpoint:mcp:target:test",
    )
    observation = McpCapabilityShadowObservation.from_execution(
        candidate_digest=candidate.candidate_digest,
        registry_snapshot_id=registry.snapshot_id,
        input_payload={"action": "list"},
        baseline_output={"entries": ["README.md"]},
        candidate_output={"entries": ["README.md"]},
        baseline_after_state={"files": ["README.md"]},
        candidate_after_state={"files": ["README.md"]},
        baseline_resources={"cpu_ms": 1, "output_bytes": 20},
        candidate_resources={"cpu_ms": 2, "output_bytes": 21},
    )
    shadow_registry.propose(candidate, policy)
    shadow_registry.record_shadow(candidate.candidate_digest, observation)
    proposal = shadow_registry.propose_activation(
        candidate.candidate_digest,
        client_capability_snapshot_id="capability:target",
    )
    authorization = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:target",
        approval_id="approval:target",
        issuer="user:target",
        issued_at_epoch=100,
        expires_at_epoch=200,
    ).authorization
    assert authorization is not None
    return registry, candidate, authorization


def _bind(authorization):
    return bind_mcp_client_connection_target(
        authorization,
        target_id=authorization.server_id,
        target_version=authorization.server_version,
        transport="streamable_http",
        mcp_registry_snapshot_id=authorization.mcp_registry_snapshot_id,
        client_capability_snapshot_id=authorization.client_capability_snapshot_id,
        connection_owner_id="owner:target",
        credential_owner_id="owner:credential",
        approver_id="approver:target",
        at_epoch=100,
    )


def test_target_binding_is_declarative_checkpointable_and_revocable():
    registry, _, authorization = _authorized()
    decision = _bind(authorization)
    assert decision.passed is True
    assert decision.target is not None
    target = decision.target
    payload = target.to_payload()
    assert target.is_valid(100) is True
    assert target.is_valid(200) is False
    assert payload["connection_attempted"] is False
    assert "endpoint" not in payload
    assert "url" not in payload
    assert "credential_value" not in payload
    assert McpClientConnectionTarget.from_payload(payload) == target

    store = McpClientConnectionTargetStore(
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:target",
    )
    store.issue(target)
    restored = McpClientConnectionTargetStore.from_checkpoint(store.checkpoint())
    assert restored.get(target.binding_id) == target
    revoked = restored.revoke(target.binding_id, reason="target test stop")
    assert revoked.state == "revoked"
    assert revoked.to_payload()["connection_attempted"] is False
    assert restored.revoke(target.binding_id, reason="ignored") == revoked


def test_target_binding_fails_closed_for_identity_scope_transport_snapshot_and_expiry():
    registry, _, authorization = _authorized()
    mismatch = _bind(authorization)
    assert mismatch.passed is True

    wrong_target = bind_mcp_client_connection_target(
        authorization,
        target_id="other.mcp.server",
        target_version=authorization.server_version,
        transport="stdio",
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:target",
        connection_owner_id="owner:target",
        credential_owner_id="owner:credential",
        approver_id="approver:target",
        at_epoch=100,
    )
    assert wrong_target.passed is False
    assert wrong_target.reason_code == "target_id_mismatch"

    wrong_transport = bind_mcp_client_connection_target(
        authorization,
        target_id=authorization.server_id,
        target_version=authorization.server_version,
        transport="unknown",
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:target",
        connection_owner_id="owner:target",
        credential_owner_id="owner:credential",
        approver_id="approver:target",
        at_epoch=100,
    )
    assert wrong_transport.passed is False
    assert wrong_transport.reason_code == "transport_not_allowed"

    stale = bind_mcp_client_connection_target(
        authorization,
        target_id=authorization.server_id,
        target_version=authorization.server_version,
        transport="stdio",
        mcp_registry_snapshot_id="mcp:stale",
        client_capability_snapshot_id="capability:target",
        connection_owner_id="owner:target",
        credential_owner_id="owner:credential",
        approver_id="approver:target",
        at_epoch=100,
    )
    assert stale.passed is False
    assert stale.reason_code == "stale_mcp_registry"

    expired = bind_mcp_client_connection_target(
        authorization,
        target_id=authorization.server_id,
        target_version=authorization.server_version,
        transport="stdio",
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:target",
        connection_owner_id="owner:target",
        credential_owner_id="owner:credential",
        approver_id="approver:target",
        at_epoch=200,
    )
    assert expired.passed is False
    assert expired.reason_code == "authorization_expired"

    tampered = mismatch.target.to_payload()
    tampered["connection_attempted"] = True
    with pytest.raises(ValueError):
        McpClientConnectionTarget.from_payload(tampered)

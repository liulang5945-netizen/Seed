from __future__ import annotations

import pytest

from seed_platform.mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
)
from seed_platform.mcp_client_capability_registry import (
    McpClientCapabilityShadowRegistry,
)
from seed_platform.mcp_client_connection_authorization import (
    McpClientConnectionAuthorization,
    McpClientConnectionAuthorizationStore,
    authorize_mcp_client_connection,
)
from seed_platform.mcp_registry import McpToolRegistry


def _candidate(registry: McpToolRegistry) -> McpCapabilityInheritanceCandidate:
    return McpCapabilityInheritanceCandidate.from_registry(
        registry,
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:authorization:test",
        tool_ids=(registry.list_tools()[0].tool_id,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:authorization:test",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:authorization:test",
        rationale="test an explicit connection authorization boundary",
    )


def _observation(candidate: McpCapabilityInheritanceCandidate) -> McpCapabilityShadowObservation:
    return McpCapabilityShadowObservation.from_execution(
        candidate_digest=candidate.candidate_digest,
        registry_snapshot_id=candidate.registry_snapshot_id,
        input_payload={"action": "list"},
        baseline_output={"entries": ["README.md"]},
        candidate_output={"entries": ["README.md"]},
        baseline_after_state={"files": ["README.md"]},
        candidate_after_state={"files": ["README.md"]},
        baseline_resources={"cpu_ms": 1, "output_bytes": 20},
        candidate_resources={"cpu_ms": 2, "output_bytes": 21},
    )


def _proposed():
    registry = McpToolRegistry.default()
    candidate = _candidate(registry)
    policy = McpCapabilityInheritancePolicy(allowed_server_ids=(candidate.server_id,))
    shadow_registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        registry,
        parent_checkpoint_id="checkpoint:mcp:authorization:test",
    )
    shadow_registry.propose(candidate, policy)
    shadow_registry.record_shadow(candidate.candidate_digest, _observation(candidate))
    proposal = shadow_registry.propose_activation(
        candidate.candidate_digest,
        client_capability_snapshot_id="capability:test",
    )
    return registry, candidate, policy, proposal


def test_authorization_is_explicit_bounded_and_checkpointable():
    registry, candidate, policy, proposal = _proposed()
    decision = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:test",
        approval_id="approval:test",
        issuer="user:test",
        issued_at_epoch=100,
        expires_at_epoch=200,
    )

    assert decision.passed is True
    assert decision.decision == "authorized_for_connection"
    assert decision.authorization is not None
    authorization = decision.authorization
    assert authorization.is_valid(100) is True
    assert authorization.is_valid(200) is False
    assert authorization.to_payload()["connection_attempted"] is False
    assert "endpoint" not in authorization.to_payload()
    assert "credential_value" not in authorization.to_payload()
    assert McpClientConnectionAuthorization.from_payload(authorization.to_payload()) == authorization

    store = McpClientConnectionAuthorizationStore(
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:test",
    )
    store.issue(authorization)
    restored = McpClientConnectionAuthorizationStore.from_checkpoint(store.checkpoint())
    assert restored.get(authorization.authorization_id) == authorization
    revoked = restored.revoke(authorization.authorization_id, reason="test stop")
    assert revoked.state == "revoked"
    assert revoked.is_valid(100) is False
    assert restored.revoke(authorization.authorization_id, reason="ignored") == revoked


def test_authorization_fails_closed_for_missing_approval_stale_snapshot_and_scope():
    registry, candidate, policy, proposal = _proposed()
    missing_approval = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:test",
        approval_id="",
        issuer="user:test",
        issued_at_epoch=100,
        expires_at_epoch=200,
    )
    assert missing_approval.passed is False
    assert missing_approval.reason_code == "explicit_approval_required"

    stale = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:stale",
        approval_id="approval:test",
        issuer="user:test",
        issued_at_epoch=100,
        expires_at_epoch=200,
    )
    assert stale.passed is False
    assert stale.reason_code == "stale_client_capability_snapshot"

    out_of_scope = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:test",
        network_scopes=("network:undeclared",),
        approval_id="approval:test",
        issuer="user:test",
        issued_at_epoch=100,
        expires_at_epoch=200,
    )
    assert out_of_scope.passed is False
    assert out_of_scope.reason_code == "network_scope_not_declared_by_candidate"


def test_authorization_rejects_unbounded_lifetime_and_tampered_record():
    registry, candidate, policy, proposal = _proposed()
    decision = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:test",
        approval_id="approval:test",
        issuer="user:test",
        issued_at_epoch=100,
        expires_at_epoch=3_701,
        max_lifetime_seconds=3_600,
    )
    assert decision.passed is False
    assert decision.reason_code == "authorization_lifetime_exceeded"

    valid = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:test",
        approval_id="approval:test",
        issuer="user:test",
        issued_at_epoch=100,
        expires_at_epoch=200,
    ).authorization
    assert valid is not None
    payload = valid.to_payload()
    payload["connection_attempted"] = True
    with pytest.raises(ValueError):
        McpClientConnectionAuthorization.from_payload(payload)

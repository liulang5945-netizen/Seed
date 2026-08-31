from dataclasses import replace

import pytest

from seed_platform.mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
    McpClientToolContract,
)
from seed_platform.mcp_client_capability_registry import (
    McpClientCapabilityShadowRegistry,
)
from seed_platform.mcp_registry import McpToolRegistry


def _candidate(
    registry: McpToolRegistry,
    *,
    risk: str = "read_only",
) -> McpCapabilityInheritanceCandidate:
    tool = McpClientToolContract.from_descriptor(registry.list_tools()[0])
    tool = replace(tool, risk=risk)
    return McpCapabilityInheritanceCandidate(
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:local:v1",
        registry_snapshot_id=registry.snapshot_id,
        tool_contracts=(tool,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:local",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:local",
        rationale="project a governed MCP contract into a client shadow registry",
    )


def _policy(**kwargs) -> McpCapabilityInheritancePolicy:
    return McpCapabilityInheritancePolicy(
        allowed_server_ids=("seed.mcp.local",),
        **kwargs,
    )


def _observation(candidate: McpCapabilityInheritanceCandidate, **kwargs):
    values = {
        "candidate_digest": candidate.candidate_digest,
        "registry_snapshot_id": candidate.registry_snapshot_id,
        "input_payload": {"path": "."},
        "baseline_output": {"entries": ["README.md"]},
        "candidate_output": {"entries": ["README.md"]},
        "baseline_after_state": {"files": ["README.md"]},
        "candidate_after_state": {"files": ["README.md"]},
        "baseline_resources": {"cpu_ms": 1, "output_bytes": 20},
        "candidate_resources": {"cpu_ms": 2, "output_bytes": 21},
    }
    values.update(kwargs)
    return McpCapabilityShadowObservation.from_execution(**values)


def test_registry_projects_candidate_to_shadow_and_restores_checkpoint() -> None:
    mcp_registry = McpToolRegistry.default()
    registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        mcp_registry,
        parent_checkpoint_id="checkpoint:shadow-registry",
    )
    candidate = _candidate(mcp_registry)
    proposed = registry.propose(
        candidate,
        _policy(),
        expected_current_snapshot_id=mcp_registry.snapshot_id,
    )

    assert proposed.state == "shadow_pending"
    assert registry.shadow_validated == ()
    assert "executor_id" not in proposed.to_payload()["candidate"]["tool_contracts"][0]

    validated = registry.record_shadow(
        candidate.candidate_digest,
        _observation(candidate),
        expected_current_snapshot_id=mcp_registry.snapshot_id,
    )
    assert validated.state == "shadow_validated"
    assert len(registry.shadow_validated) == 1

    restored = McpClientCapabilityShadowRegistry.from_checkpoint(registry.checkpoint())
    assert restored.snapshot_id == registry.snapshot_id
    assert restored.get(candidate.candidate_digest).state == "shadow_validated"
    assert restored.get(candidate.candidate_digest).candidate == candidate


def test_registry_rejects_bad_shadow_and_keeps_client_activation_unavailable() -> None:
    mcp_registry = McpToolRegistry.default()
    registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        mcp_registry,
        parent_checkpoint_id="checkpoint:shadow-reject",
    )
    candidate = _candidate(mcp_registry)
    registry.propose(candidate, _policy())
    rejected = registry.record_shadow(
        candidate.candidate_digest,
        _observation(candidate, external_calls_performed=True),
    )

    assert rejected.state == "rejected"
    assert registry.shadow_validated == ()
    with pytest.raises(PermissionError, match="terminal"):
        registry.record_shadow(candidate.candidate_digest, _observation(candidate))


def test_registry_binding_drift_is_fail_closed_without_mutating_record() -> None:
    mcp_registry = McpToolRegistry.default()
    registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        mcp_registry,
        parent_checkpoint_id="checkpoint:shadow-drift",
    )
    candidate = _candidate(mcp_registry)
    pending = registry.propose(candidate, _policy())
    before = registry.snapshot_id

    registry.bind_mcp_snapshot("mcp-snapshot:new", expected_current_snapshot_id=mcp_registry.snapshot_id)
    with pytest.raises(ValueError, match="stale"):
        registry.record_shadow(candidate.candidate_digest, _observation(candidate))

    assert registry.get(candidate.candidate_digest) == pending
    assert registry.snapshot_id != before


def test_registry_rollback_is_terminal_and_does_not_activate_candidate() -> None:
    mcp_registry = McpToolRegistry.default()
    registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        mcp_registry,
        parent_checkpoint_id="checkpoint:shadow-rollback",
    )
    candidate = _candidate(mcp_registry)
    registry.propose(candidate, _policy())
    registry.record_shadow(candidate.candidate_digest, _observation(candidate))

    rolled_back = registry.rollback(candidate.candidate_digest)

    assert rolled_back.state == "rolled_back"
    assert registry.shadow_validated == ()
    assert all(record.state != "active" for record in registry.records)
    with pytest.raises(PermissionError, match="terminal"):
        registry.rollback(candidate.candidate_digest)

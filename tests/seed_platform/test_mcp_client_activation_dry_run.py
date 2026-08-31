import pytest

from seed_platform.client_extension_host import ClientExtensionHost
from seed_platform.mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
    McpClientToolContract,
)
from seed_platform.mcp_client_activation_dry_run import (
    build_client_activation_manifest,
    run_client_activation_dry_run,
)
from seed_platform.mcp_client_capability_registry import (
    McpClientCapabilityShadowRegistry,
)
from seed_platform.mcp_registry import McpToolRegistry


def _candidate(registry: McpToolRegistry) -> McpCapabilityInheritanceCandidate:
    tool = McpClientToolContract.from_descriptor(registry.list_tools()[0])
    return McpCapabilityInheritanceCandidate(
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:dry-run:v1",
        registry_snapshot_id=registry.snapshot_id,
        tool_contracts=(tool,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:dry-run",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:dry-run",
        rationale="validate a local client-organ shape without committing it",
    )


def _observation(candidate: McpCapabilityInheritanceCandidate) -> McpCapabilityShadowObservation:
    return McpCapabilityShadowObservation.from_execution(
        candidate_digest=candidate.candidate_digest,
        registry_snapshot_id=candidate.registry_snapshot_id,
        input_payload={"path": "."},
        baseline_output={"entries": ["README.md"]},
        candidate_output={"entries": ["README.md"]},
        baseline_after_state={"files": ["README.md"]},
        candidate_after_state={"files": ["README.md"]},
        baseline_resources={"cpu_ms": 1, "output_bytes": 20},
        candidate_resources={"cpu_ms": 2, "output_bytes": 21},
    )


def _proposal(registry: McpToolRegistry, client_snapshot_id: str):
    candidate = _candidate(registry)
    shadow_registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        registry,
        parent_checkpoint_id="checkpoint:mcp:dry-run",
    )
    shadow_registry.propose(
        candidate,
        McpCapabilityInheritancePolicy(allowed_server_ids=("seed.mcp.local",)),
    )
    shadow_registry.record_shadow(candidate.candidate_digest, _observation(candidate))
    proposal = shadow_registry.propose_activation(
        candidate.candidate_digest,
        client_capability_snapshot_id=client_snapshot_id,
    )
    return candidate, proposal


def test_dry_run_prepares_manifest_without_committing_client_snapshot() -> None:
    registry = McpToolRegistry.default()
    candidate, proposal = _proposal(registry, "capability-snapshot:dry-run")
    host = ClientExtensionHost(
        capability_snapshot_id="capability-snapshot:dry-run",
        parent_checkpoint_id="checkpoint:client-dry-run",
    )
    before_snapshot = host.snapshot.snapshot_id

    result = run_client_activation_dry_run(
        host,
        candidate,
        proposal,
        client_capability_snapshot_id="capability-snapshot:dry-run",
        available_capabilities=tuple(item.tool_id for item in candidate.tool_contracts),
    )

    manifest = build_client_activation_manifest(candidate, proposal)
    assert result.committed is False
    assert result.proposal_id == proposal.proposal_id
    assert host.snapshot.snapshot_id == before_snapshot
    assert host.active_manifests == ()
    assert manifest.plugin_digest == result.plugin_digest
    assert "executor_id" not in manifest.to_payload()
    assert '"executor"' not in manifest.to_payload()["metadata"]


def test_dry_run_rejects_snapshot_mismatch_and_rolled_back_proposal() -> None:
    registry = McpToolRegistry.default()
    candidate, proposal = _proposal(registry, "capability-snapshot:dry-run")
    host = ClientExtensionHost(
        capability_snapshot_id="capability-snapshot:dry-run",
        parent_checkpoint_id="checkpoint:client-dry-run",
    )
    with pytest.raises(ValueError, match="does not match"):
        run_client_activation_dry_run(
            host,
            candidate,
            proposal,
            client_capability_snapshot_id="stale-client-snapshot",
            available_capabilities=tuple(item.tool_id for item in candidate.tool_contracts),
        )

    shadow_registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        registry,
        parent_checkpoint_id="checkpoint:mcp:dry-run-rollback",
    )
    shadow_registry.propose(
        candidate,
        McpCapabilityInheritancePolicy(allowed_server_ids=("seed.mcp.local",)),
    )
    shadow_registry.record_shadow(candidate.candidate_digest, _observation(candidate))
    shadow_registry.propose_activation(
        candidate.candidate_digest,
        client_capability_snapshot_id="capability-snapshot:dry-run",
    )
    shadow_registry.rollback(candidate.candidate_digest)
    rolled_back = shadow_registry.activation_proposals[0]
    with pytest.raises(PermissionError, match="proposed activation"):
        run_client_activation_dry_run(
            host,
            candidate,
            rolled_back,
            client_capability_snapshot_id="capability-snapshot:dry-run",
            available_capabilities=tuple(item.tool_id for item in candidate.tool_contracts),
        )

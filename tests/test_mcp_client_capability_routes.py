from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import routes_mcp_client_capabilities
from api.app import create_app
from seed_platform.mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityShadowObservation,
)
from seed_platform.mcp_registry import McpToolRegistry


@pytest.fixture
def client():
    routes_mcp_client_capabilities._registry = None
    app = create_app(startup_tasks=False)
    with TestClient(app) as test_client:
        yield test_client
    routes_mcp_client_capabilities._registry = None


def _candidate() -> McpCapabilityInheritanceCandidate:
    registry = McpToolRegistry.default()
    return McpCapabilityInheritanceCandidate.from_registry(
        registry,
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:route:v1",
        tool_ids=(registry.list_tools()[0].tool_id,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:route",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:route",
        rationale="exercise the client capability shadow API",
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


def test_mcp_client_capability_api_projects_shadow_lifecycle_and_rollback(client):
    candidate = _candidate()
    proposed = client.post(
        "/api/mcp-client-capabilities/proposals",
        json={"candidate": candidate.to_payload(), "policy": {}},
    )
    assert proposed.status_code == 200
    assert proposed.json()["status"] == "shadow_pending"
    record = proposed.json()["record"]
    assert "executor_id" not in proposed.text
    assert '"executor"' not in proposed.text

    shadow = client.post(
        f"/api/mcp-client-capabilities/{candidate.candidate_digest}/shadow",
        json={"observation": _observation(candidate).to_payload()},
    )
    assert shadow.status_code == 200
    assert shadow.json()["status"] == "shadow_validated"
    assert shadow.json()["record"]["candidate_digest"] == record["candidate_digest"]

    status = client.get("/api/mcp-client-capabilities")
    assert status.status_code == 200
    assert len(status.json()["shadow_validated"]) == 1
    assert status.json()["client_activation"] == "proposal_only_in_e6_2"
    client_capability_snapshot_id = status.json()["client_capability_snapshot_id"]

    activation_proposal = client.post(
        f"/api/mcp-client-capabilities/{candidate.candidate_digest}/activation-proposals",
        json={"client_capability_snapshot_id": client_capability_snapshot_id},
    )
    assert activation_proposal.status_code == 200
    assert activation_proposal.json()["status"] == "proposed"
    assert activation_proposal.json()["activation"] == "proposal_only"
    assert activation_proposal.json()["proposal"]["proposal_id"]

    dry_run = client.post(
        f"/api/mcp-client-capabilities/{candidate.candidate_digest}/activation-dry-run",
        json={"client_capability_snapshot_id": client_capability_snapshot_id},
    )
    assert dry_run.status_code == 200
    assert dry_run.json()["status"] == "dry_run"
    assert dry_run.json()["activation"] == "not_committed"
    assert dry_run.json()["dry_run"]["committed"] is False

    rollback = client.post(
        f"/api/mcp-client-capabilities/{candidate.candidate_digest}/rollback",
    )
    assert rollback.status_code == 200
    assert rollback.json()["status"] == "rolled_back"
    status_after_rollback = client.get("/api/mcp-client-capabilities").json()
    assert status_after_rollback["shadow_validated"] == []
    assert status_after_rollback["activation_proposals"][0]["state"] == "rolled_back"

    stale_activation = client.post(
        f"/api/mcp-client-capabilities/{candidate.candidate_digest}/activation-proposals",
        json={"client_capability_snapshot_id": "stale-client-snapshot"},
    )
    assert stale_activation.status_code == 409


def test_mcp_client_capability_api_rejects_stale_or_executable_candidate(client):
    candidate = _candidate()
    stale_payload = candidate.to_payload()
    stale_payload["registry_snapshot_id"] = "stale-mcp-registry"
    with pytest.raises(ValueError):
        McpCapabilityInheritanceCandidate.from_payload(stale_payload)

    forbidden = candidate.to_payload()
    forbidden["executor"] = "seed.executor"
    response = client.post(
        "/api/mcp-client-capabilities/proposals",
        json={"candidate": forbidden, "policy": {}},
    )
    assert response.status_code == 400
    assert "forbidden" in response.json()["detail"]

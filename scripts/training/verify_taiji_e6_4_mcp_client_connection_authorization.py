"""Verify the E6-4 explicit authorization boundary before real MCP connection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api import routes_mcp_client_capabilities  # noqa: E402
from api.app import create_app  # noqa: E402
from seed_platform.mcp_capability_inheritance import (  # noqa: E402
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
)
from seed_platform.mcp_client_capability_registry import (  # noqa: E402
    McpClientCapabilityShadowRegistry,
)
from seed_platform.mcp_client_connection_authorization import (  # noqa: E402
    McpClientConnectionAuthorizationStore,
    authorize_mcp_client_connection,
)
from seed_platform.mcp_registry import McpToolRegistry  # noqa: E402


def _candidate(registry: McpToolRegistry) -> McpCapabilityInheritanceCandidate:
    return McpCapabilityInheritanceCandidate.from_registry(
        registry,
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:e6-4:v1",
        tool_ids=(registry.list_tools()[0].tool_id,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:e6-4",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:e6-4",
        rationale="validate explicit authorization before any real connection",
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


def _prepared():
    registry = McpToolRegistry.default()
    candidate = _candidate(registry)
    policy = McpCapabilityInheritancePolicy(allowed_server_ids=(candidate.server_id,))
    shadow_registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        registry,
        parent_checkpoint_id="checkpoint:mcp:e6-4",
    )
    shadow_registry.propose(candidate, policy)
    shadow_registry.record_shadow(candidate.candidate_digest, _observation(candidate))
    proposal = shadow_registry.propose_activation(
        candidate.candidate_digest,
        client_capability_snapshot_id="capability:e6-4",
    )
    return registry, candidate, policy, proposal, shadow_registry


def run_gate() -> dict[str, object]:
    registry, candidate, policy, proposal, shadow_registry = _prepared()
    mcp_snapshot_before = registry.snapshot_id
    decision = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:e6-4",
        approval_id="approval:e6-4",
        issuer="user:e6-4",
        issued_at_epoch=100,
        expires_at_epoch=200,
    )
    authorization = decision.authorization
    assert authorization is not None

    store = McpClientConnectionAuthorizationStore(
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:e6-4",
    )
    store.issue(authorization)
    restored_store = McpClientConnectionAuthorizationStore.from_checkpoint(store.checkpoint())
    restored_authorization = restored_store.get(authorization.authorization_id)
    revoked = restored_store.revoke(authorization.authorization_id, reason="gate stop")

    missing_approval = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:e6-4",
        approval_id="",
        issuer="user:e6-4",
        issued_at_epoch=100,
        expires_at_epoch=200,
    )
    stale_client = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:stale",
        approval_id="approval:e6-4",
        issuer="user:e6-4",
        issued_at_epoch=100,
        expires_at_epoch=200,
    )
    out_of_scope = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:e6-4",
        network_scopes=("network:undeclared",),
        approval_id="approval:e6-4",
        issuer="user:e6-4",
        issued_at_epoch=100,
        expires_at_epoch=200,
    )
    too_long = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:e6-4",
        approval_id="approval:e6-4",
        issuer="user:e6-4",
        issued_at_epoch=100,
        expires_at_epoch=3_701,
    )

    routes_mcp_client_capabilities._registry = None
    routes_mcp_client_capabilities._authorization_store = None
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        api_candidate = _candidate(McpToolRegistry.default())
        proposed = client.post(
            "/api/mcp-client-capabilities/proposals",
            json={"candidate": api_candidate.to_payload(), "policy": {}},
        )
        shadow = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/shadow",
            json={"observation": _observation(api_candidate).to_payload()},
        )
        status = client.get("/api/mcp-client-capabilities").json()
        activation = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/activation-proposals",
            json={"client_capability_snapshot_id": status["client_capability_snapshot_id"]},
        ).json()["proposal"]
        dry_run = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/activation-dry-run",
            json={"client_capability_snapshot_id": status["client_capability_snapshot_id"]},
        ).json()["dry_run"]
        api_authorization = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/connection-authorization",
            json={
                "proposal_id": activation["proposal_id"],
                "dry_run_digest": dry_run["dry_run_digest"],
                "client_capability_snapshot_id": status["client_capability_snapshot_id"],
                "approval_id": "approval:api:e6-4",
                "issuer": "user:api:e6-4",
                "issued_at_epoch": 100,
                "expires_at_epoch": 200,
            },
        )
        api_auth_payload = api_authorization.json()
        api_revoke = client.post(
            "/api/mcp-client-capabilities/connection-authorizations/"
            f"{api_auth_payload['authorization']['authorization_id']}/revoke",
            json={"reason": "api gate stop"},
        )
        api_status = client.get("/api/mcp-client-capabilities")
    routes_mcp_client_capabilities._registry = None
    routes_mcp_client_capabilities._authorization_store = None

    auth_source = (
        PROJECT_ROOT / "seed_platform" / "mcp_client_connection_authorization.py"
    ).read_text(encoding="utf-8")
    route_source = (
        PROJECT_ROOT / "api" / "routes_mcp_client_capabilities.py"
    ).read_text(encoding="utf-8")
    forbidden_markers = (
        "import requests",
        "import socket",
        "import httpx",
        "import subprocess",
        "from taiji",
        "ClientExtensionHost.commit",
    )
    payload = authorization.to_payload()
    checks = {
        "requires_shadow_validated_activation": decision.passed
        and proposal.state == "proposed"
        and shadow_registry.get(candidate.candidate_digest).state == "shadow_validated",
        "requires_explicit_approval_issuer_and_bounded_lifetime": decision.passed
        and bool(authorization.approval_id)
        and bool(authorization.issuer)
        and authorization.expires_at_epoch - authorization.issued_at_epoch <= 3_600,
        "binds_current_mcp_and_client_snapshots": authorization.mcp_registry_snapshot_id
        == registry.snapshot_id
        and authorization.client_capability_snapshot_id == "capability:e6-4",
        "stores_references_without_secrets_or_endpoint": set(payload).isdisjoint(
            {"endpoint", "url", "token", "secret", "credential_value", "executor", "source"}
        )
        and isinstance(payload["credential_refs"], list),
        "connection_is_never_attempted": payload["connection_attempted"] is False
        and decision.connection_attempted is False
        and revoked.to_payload()["connection_attempted"] is False,
        "store_issue_revoke_checkpoint_roundtrip": restored_authorization == authorization
        and restored_store.get(authorization.authorization_id).state == "revoked"
        and revoked.state == "revoked"
        and revoked.is_valid(100) is False,
        "fail_closed_on_approval_snapshot_scope_and_lifetime": missing_approval.reason_code
        == "explicit_approval_required"
        and stale_client.reason_code == "stale_client_capability_snapshot"
        and out_of_scope.reason_code == "network_scope_not_declared_by_candidate"
        and too_long.reason_code == "authorization_lifetime_exceeded",
        "api_requires_dry_run_and_supports_revoke": proposed.status_code == 200
        and shadow.status_code == 200
        and api_authorization.status_code == 200
        and api_auth_payload["connection"] == "not_attempted"
        and api_revoke.status_code == 200
        and api_revoke.json()["status"] == "revoked"
        and api_status.status_code == 200
        and api_status.json()["connection"] == "not_attempted",
        "mcp_registry_and_client_activation_remain_unchanged": registry.snapshot_id
        == mcp_snapshot_before
        and all(item.state != "active" for item in shadow_registry.activation_proposals),
        "no_external_or_execution_boundary": all(
            marker not in auth_source + route_source for marker in forbidden_markers
        ),
    }
    return {
        "gate": "taiji-e6-4-mcp-client-connection-authorization",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scope": {
            "explicit_authorization_record_only": True,
            "real_third_party_mcp_connected": False,
            "client_organ_activated": False,
            "connection_attempted": False,
            "taiji_cognition_mutated": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_e6_4_mcp_client_connection_authorization_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

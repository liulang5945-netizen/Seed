"""Verify E6-5 target binding before any real MCP connection."""

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
    authorize_mcp_client_connection,
)
from seed_platform.mcp_client_connection_target import (  # noqa: E402
    McpClientConnectionTargetStore,
    bind_mcp_client_connection_target,
)
from seed_platform.mcp_registry import McpToolRegistry  # noqa: E402


def _prepared():
    registry = McpToolRegistry.default()
    candidate = McpCapabilityInheritanceCandidate.from_registry(
        registry,
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:e6-5:v1",
        tool_ids=(registry.list_tools()[0].tool_id,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:e6-5",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:e6-5",
        rationale="validate target identity and ownership before a connection",
    )
    policy = McpCapabilityInheritancePolicy(allowed_server_ids=(candidate.server_id,))
    shadow_registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        registry,
        parent_checkpoint_id="checkpoint:mcp:e6-5",
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
        client_capability_snapshot_id="capability:e6-5",
    )
    authorization = authorize_mcp_client_connection(
        candidate,
        proposal,
        policy,
        current_mcp_registry_snapshot_id=registry.snapshot_id,
        current_client_capability_snapshot_id="capability:e6-5",
        approval_id="approval:e6-5",
        issuer="user:e6-5",
        issued_at_epoch=100,
        expires_at_epoch=200,
    ).authorization
    assert authorization is not None
    return registry, candidate, authorization, shadow_registry


def run_gate() -> dict[str, object]:
    registry, candidate, authorization, shadow_registry = _prepared()
    mcp_snapshot_before = registry.snapshot_id
    decision = bind_mcp_client_connection_target(
        authorization,
        target_id=authorization.server_id,
        target_version=authorization.server_version,
        transport="streamable_http",
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:e6-5",
        connection_owner_id="owner:e6-5",
        credential_owner_id="credential-owner:e6-5",
        approver_id="approver:e6-5",
        at_epoch=100,
    )
    target = decision.target
    assert target is not None
    target_store = McpClientConnectionTargetStore(
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:e6-5",
    )
    target_store.issue(target)
    restored = McpClientConnectionTargetStore.from_checkpoint(target_store.checkpoint())
    restored_target = restored.get(target.binding_id)
    restored.revoke_for_authorization(authorization.authorization_id, reason="gate stop")

    wrong_identity = bind_mcp_client_connection_target(
        authorization,
        target_id="other.mcp.server",
        target_version=authorization.server_version,
        transport="stdio",
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:e6-5",
        connection_owner_id="owner:e6-5",
        credential_owner_id="credential-owner:e6-5",
        approver_id="approver:e6-5",
        at_epoch=100,
    )
    stale = bind_mcp_client_connection_target(
        authorization,
        target_id=authorization.server_id,
        target_version=authorization.server_version,
        transport="stdio",
        mcp_registry_snapshot_id="mcp:stale",
        client_capability_snapshot_id="capability:e6-5",
        connection_owner_id="owner:e6-5",
        credential_owner_id="credential-owner:e6-5",
        approver_id="approver:e6-5",
        at_epoch=100,
    )
    expired = bind_mcp_client_connection_target(
        authorization,
        target_id=authorization.server_id,
        target_version=authorization.server_version,
        transport="stdio",
        mcp_registry_snapshot_id=registry.snapshot_id,
        client_capability_snapshot_id="capability:e6-5",
        connection_owner_id="owner:e6-5",
        credential_owner_id="credential-owner:e6-5",
        approver_id="approver:e6-5",
        at_epoch=200,
    )

    routes_mcp_client_capabilities._registry = None
    routes_mcp_client_capabilities._authorization_store = None
    routes_mcp_client_capabilities._target_store = None
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        api_candidate = McpCapabilityInheritanceCandidate.from_registry(
            McpToolRegistry.default(),
            server_id="seed.mcp.local",
            server_version="1.0.0",
            source_artifact_digest="artifact:mcp:e6-5:api",
            tool_ids=(McpToolRegistry.default().list_tools()[0].tool_id,),
            resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
            evidence_digests=("evidence:mcp:e6-5:api",),
            evaluation_gates=("policy", "shadow", "rollback"),
            parent_checkpoint_id="checkpoint:mcp:e6-5:api",
            rationale="exercise target binding API",
        )
        proposed = client.post(
            "/api/mcp-client-capabilities/proposals",
            json={"candidate": api_candidate.to_payload(), "policy": {}},
        )
        shadow = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/shadow",
            json={"observation": McpCapabilityShadowObservation.from_execution(
                candidate_digest=api_candidate.candidate_digest,
                registry_snapshot_id=api_candidate.registry_snapshot_id,
                input_payload={"action": "list"},
                baseline_output={"entries": ["README.md"]},
                candidate_output={"entries": ["README.md"]},
                baseline_after_state={"files": ["README.md"]},
                candidate_after_state={"files": ["README.md"]},
                baseline_resources={"cpu_ms": 1, "output_bytes": 20},
                candidate_resources={"cpu_ms": 2, "output_bytes": 21},
            ).to_payload()},
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
        api_auth = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/connection-authorization",
            json={
                "proposal_id": activation["proposal_id"],
                "dry_run_digest": dry_run["dry_run_digest"],
                "client_capability_snapshot_id": status["client_capability_snapshot_id"],
                "approval_id": "approval:e6-5:api",
                "issuer": "user:e6-5:api",
                "issued_at_epoch": 100,
                "expires_at_epoch": 200,
            },
        )
        api_auth_payload = api_auth.json()
        api_target = client.post(
            "/api/mcp-client-capabilities/connection-authorizations/"
            f"{api_auth_payload['authorization']['authorization_id']}/target-binding",
            json={
                "target_id": api_candidate.server_id,
                "target_version": api_candidate.server_version,
                "transport": "stdio",
                "mcp_registry_snapshot_id": status["mcp_registry_snapshot_id"],
                "client_capability_snapshot_id": status["client_capability_snapshot_id"],
                "connection_owner_id": "owner:e6-5:api",
                "credential_owner_id": "credential-owner:e6-5:api",
                "approver_id": "approver:e6-5:api",
                "at_epoch": 100,
            },
        )
        api_target_payload = api_target.json()
        api_revoke = client.post(
            "/api/mcp-client-capabilities/connection-authorizations/"
            f"{api_auth_payload['authorization']['authorization_id']}/revoke",
            json={"reason": "api gate stop"},
        )
    routes_mcp_client_capabilities._registry = None
    routes_mcp_client_capabilities._authorization_store = None
    routes_mcp_client_capabilities._target_store = None

    target_source = (
        PROJECT_ROOT / "seed_platform" / "mcp_client_connection_target.py"
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
    payload = target.to_payload()
    checks = {
        "binds_only_authorized_target": decision.passed
        and target.authorization_id == authorization.authorization_id
        and target.target_id == authorization.server_id,
        "requires_explicit_transport_identity_and_owners": target.transport == "streamable_http"
        and target.target_version == authorization.server_version
        and target.connection_owner_id == "owner:e6-5"
        and target.credential_owner_id == "credential-owner:e6-5"
        and target.approver_id == "approver:e6-5",
        "inherits_authorization_time_window": target.authorization_issued_at_epoch
        == authorization.issued_at_epoch
        and target.authorization_expires_at_epoch == authorization.expires_at_epoch
        and target.is_valid(100) is True
        and target.is_valid(200) is False,
        "binds_current_snapshots_and_authorized_scope": target.mcp_registry_snapshot_id
        == registry.snapshot_id
        and target.client_capability_snapshot_id == "capability:e6-5"
        and set(target.network_scopes).issubset(authorization.network_scopes)
        and set(target.credential_refs).issubset(authorization.credential_refs)
        and set(target.allowed_permissions).issubset(authorization.allowed_permissions),
        "checkpoint_roundtrip_and_revocation_hold": restored_target == target
        and restored.get(target.binding_id).state == "revoked"
        and restored.get(target.binding_id).to_payload()["connection_attempted"] is False,
        "target_payload_has_no_endpoint_or_secret": set(payload).isdisjoint(
            {"endpoint", "url", "token", "secret", "credential_value", "command", "executor", "source"}
        )
        and payload["connection_attempted"] is False,
        "fail_closed_on_identity_snapshot_and_expiry": wrong_identity.reason_code
        == "target_id_mismatch"
        and stale.reason_code == "stale_mcp_registry"
        and expired.reason_code == "authorization_expired",
        "authorization_revoke_cascades_to_target": proposed.status_code == 200
        and shadow.status_code == 200
        and api_auth.status_code == 200
        and api_target.status_code == 200
        and api_target_payload["connection"] == "not_attempted"
        and api_revoke.status_code == 200
        and api_target_payload["target"]["binding_id"]
        in api_revoke.json()["revoked_target_binding_ids"],
        "no_client_activation_or_external_connection": all(
            item.state != "active" for item in shadow_registry.activation_proposals
        )
        and registry.snapshot_id == mcp_snapshot_before,
        "no_external_or_execution_boundary": all(
            marker not in target_source + route_source for marker in forbidden_markers
        ),
        "connection_is_never_attempted": decision.connection_attempted is False
        and target.to_payload()["connection_attempted"] is False
        and api_revoke.json()["connection"] == "not_attempted",
    }
    return {
        "gate": "taiji-e6-5-mcp-client-target-binding",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scope": {
            "declarative_target_binding_only": True,
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
        / "taiji_w7_e6_5_mcp_client_target_binding_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

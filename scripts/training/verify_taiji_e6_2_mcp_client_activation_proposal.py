"""Verify the E6-2 MCP client activation-proposal boundary."""

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
    McpClientToolContract,
)
from seed_platform.mcp_client_capability_registry import (  # noqa: E402
    McpClientCapabilityShadowRegistry,
)
from seed_platform.mcp_registry import McpToolRegistry  # noqa: E402


def _candidate(registry: McpToolRegistry) -> McpCapabilityInheritanceCandidate:
    tool = McpClientToolContract.from_descriptor(registry.list_tools()[0])
    return McpCapabilityInheritanceCandidate(
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:e6-2:v1",
        registry_snapshot_id=registry.snapshot_id,
        tool_contracts=(tool,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:e6-2",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:e6-2",
        rationale="propose a client capability only after a governed shadow canary",
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


def run_gate() -> dict[str, object]:
    mcp_registry = McpToolRegistry.default()
    mcp_snapshot_before = mcp_registry.snapshot_id
    candidate = _candidate(mcp_registry)
    policy = McpCapabilityInheritancePolicy(allowed_server_ids=("seed.mcp.local",))
    registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        mcp_registry,
        parent_checkpoint_id="checkpoint:mcp:e6-2",
    )
    pending = registry.propose(candidate, policy)
    blocked_before_shadow = False
    try:
        registry.propose_activation(
            candidate.candidate_digest,
            client_capability_snapshot_id="capability-snapshot:e6-2",
        )
    except PermissionError as exc:
        blocked_before_shadow = "shadow validation" in str(exc)
    validated = registry.record_shadow(candidate.candidate_digest, _observation(candidate))
    proposal = registry.propose_activation(
        candidate.candidate_digest,
        client_capability_snapshot_id="capability-snapshot:e6-2",
    )
    restored = McpClientCapabilityShadowRegistry.from_checkpoint(registry.checkpoint())
    restored_proposal = restored.activation_proposals[0]
    rolled_back = registry.rollback(candidate.candidate_digest)
    rollback_checkpoint = McpClientCapabilityShadowRegistry.from_checkpoint(registry.checkpoint())

    routes_mcp_client_capabilities._registry = None
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        api_status = client.get("/api/mcp-client-capabilities")
        api_candidate = _candidate(McpToolRegistry.default())
        api_proposal = client.post(
            "/api/mcp-client-capabilities/proposals",
            json={"candidate": api_candidate.to_payload(), "policy": {}},
        )
        client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/shadow",
            json={
                "observation": _observation(api_candidate).to_payload(),
            },
        )
        api_status_after_shadow = client.get("/api/mcp-client-capabilities").json()
        api_activation = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/activation-proposals",
            json={
                "client_capability_snapshot_id": api_status_after_shadow[
                    "client_capability_snapshot_id"
                ]
            },
        )
    routes_mcp_client_capabilities._registry = None

    source = (
        PROJECT_ROOT / "seed_platform" / "mcp_client_capability_activation.py"
    ).read_text(encoding="utf-8")
    registry_source = (
        PROJECT_ROOT / "seed_platform" / "mcp_client_capability_registry.py"
    ).read_text(encoding="utf-8")
    api_source = (
        PROJECT_ROOT / "api" / "routes_mcp_client_capabilities.py"
    ).read_text(encoding="utf-8")
    checks = {
        "activation_is_blocked_before_shadow": pending.state == "shadow_pending"
        and blocked_before_shadow,
        "shadow_canary_validates_candidate": validated.state == "shadow_validated",
        "proposal_is_explicit_and_not_active": proposal.state == "proposed"
        and proposal.to_payload()["activation"] == "proposal_only",
        "proposal_checkpoint_roundtrip_holds": restored_proposal.proposal_id == proposal.proposal_id
        and restored_proposal.state == "proposed",
        "candidate_rollback_rolls_back_proposal": rolled_back.state == "rolled_back"
        and registry.activation_proposals[0].state == "rolled_back"
        and rollback_checkpoint.activation_proposals[0].state == "rolled_back",
        "api_exposes_proposal_only_boundary": api_status.status_code == 200
        and api_status.json()["client_activation"] == "proposal_only_in_e6_2"
        and api_proposal.status_code == 200
        and api_activation.status_code == 200
        and api_activation.json()["activation"] == "proposal_only",
        "executor_and_source_are_not_exported": "executor_id" not in proposal.to_payload()
        and '"executor"' not in api_activation.text,
        "mcp_registry_is_not_mutated": mcp_registry.snapshot_id == mcp_snapshot_before,
        "no_active_client_state_exists": all(item.state != "active" for item in registry.records)
        and all(item.state != "active" for item in registry.activation_proposals),
        "no_external_or_cognitive_execution_boundary": all(
            marker not in source + registry_source + api_source
            for marker in ("import requests", "import socket", "subprocess", "httpx", "from taiji")
        ),
    }
    return {
        "gate": "taiji-e6-2-mcp-client-activation-proposal",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scope": {
            "shadow_canary_required": True,
            "activation_proposal_is_content_addressed": True,
            "activation_is_not_implemented": True,
            "rollback_is_supported": True,
            "real_third_party_mcp_connected": False,
            "client_organ_activated": False,
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
        / "taiji_w7_e6_2_mcp_client_activation_proposal_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

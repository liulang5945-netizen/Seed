"""Verify the E6-1 MCP client-capability shadow projection boundary."""

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
        source_artifact_digest="artifact:mcp:e6-1:v1",
        registry_snapshot_id=registry.snapshot_id,
        tool_contracts=(tool,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:e6-1",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:e6-1",
        rationale="project an MCP contract into a Seed-owned shadow registry",
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
    shadow_registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        mcp_registry,
        parent_checkpoint_id="checkpoint:mcp:e6-1",
    )
    proposed = shadow_registry.propose(
        candidate,
        policy,
        expected_current_snapshot_id=mcp_registry.snapshot_id,
    )
    validated = shadow_registry.record_shadow(
        candidate.candidate_digest,
        _observation(candidate),
        expected_current_snapshot_id=mcp_registry.snapshot_id,
    )
    checkpoint = shadow_registry.checkpoint()
    restored = McpClientCapabilityShadowRegistry.from_checkpoint(checkpoint)
    restored_shadow_snapshot_id = restored.snapshot_id
    restored_shadow_state = restored.get(candidate.candidate_digest).state
    rollback = restored.rollback(candidate.candidate_digest)
    before_drift_record = shadow_registry.get(candidate.candidate_digest)
    shadow_registry.bind_mcp_snapshot(
        "mcp-snapshot:e6-1:new",
        expected_current_snapshot_id=mcp_registry.snapshot_id,
    )
    stale_rejected = False
    try:
        shadow_registry.propose(
            candidate,
            policy,
            expected_current_snapshot_id=shadow_registry.current_mcp_registry_snapshot_id,
        )
    except ValueError as exc:
        stale_rejected = "stale" in str(exc)
    after_drift_record = shadow_registry.get(candidate.candidate_digest)

    routes_mcp_client_capabilities._registry = None
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        api_status = client.get("/api/mcp-client-capabilities")
        api_candidate = _candidate(McpToolRegistry.default())
        api_proposal = client.post(
            "/api/mcp-client-capabilities/proposals",
            json={"candidate": api_candidate.to_payload(), "policy": {}},
        )
    routes_mcp_client_capabilities._registry = None

    registry_source = (
        PROJECT_ROOT / "seed_platform" / "mcp_client_capability_registry.py"
    ).read_text(encoding="utf-8")
    api_source = (
        PROJECT_ROOT / "api" / "routes_mcp_client_capabilities.py"
    ).read_text(encoding="utf-8")
    checks = {
        "candidate_projects_to_shadow_pending": proposed.state == "shadow_pending",
        "equivalent_observation_projects_to_shadow_validated": validated.state
        == "shadow_validated",
        "api_exposes_seed_owned_shadow_snapshot": api_status.status_code == 200
        and api_status.json()["client_activation"] == "not_available_in_e6_1",
        "api_proposal_is_shadow_only": api_proposal.status_code == 200
        and api_proposal.json()["status"] == "shadow_pending",
        "executor_and_source_are_not_exported": "executor_id" not in candidate.to_payload()[
            "tool_contracts"
        ][0]
        and '"executor"' not in api_proposal.text,
        "mcp_registry_is_not_mutated": mcp_registry.snapshot_id == mcp_snapshot_before,
        "checkpoint_roundtrip_preserves_shadow": restored_shadow_snapshot_id
        == checkpoint["snapshot_id"]
        and restored_shadow_state == "shadow_validated",
        "rollback_is_terminal_without_active_state": rollback.state == "rolled_back"
        and all(item.state != "active" for item in restored.records),
        "snapshot_drift_fails_closed_without_record_mutation": stale_rejected
        and before_drift_record == after_drift_record,
        "no_external_or_cognitive_execution_boundary": all(
            marker not in registry_source + api_source
            for marker in ("import requests", "import socket", "subprocess", "httpx", "from taiji")
        ),
    }
    return {
        "gate": "taiji-e6-1-mcp-client-capability-shadow-projection",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scope": {
            "candidate_is_seed_owned": True,
            "mcp_snapshot_is_bound": True,
            "shadow_registry_is_checkpointable": True,
            "rollback_is_supported": True,
            "client_activation": False,
            "real_third_party_mcp_connected": False,
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
        / "taiji_w7_e6_1_mcp_client_capability_shadow_projection_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

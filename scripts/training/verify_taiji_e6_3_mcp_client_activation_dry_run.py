"""Verify the E6-3 local MCP client-organ activation dry-run boundary."""

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
from seed_platform.client_extension_host import ClientExtensionHost  # noqa: E402
from seed_platform.mcp_capability_inheritance import (  # noqa: E402
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
    McpClientToolContract,
)
from seed_platform.mcp_client_activation_dry_run import (  # noqa: E402
    build_client_activation_manifest,
    run_client_activation_dry_run,
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
        source_artifact_digest="artifact:mcp:e6-3:v1",
        registry_snapshot_id=registry.snapshot_id,
        tool_contracts=(tool,),
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:e6-3",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:e6-3",
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


def run_gate() -> dict[str, object]:
    mcp_registry = McpToolRegistry.default()
    mcp_snapshot_before = mcp_registry.snapshot_id
    candidate = _candidate(mcp_registry)
    policy = McpCapabilityInheritancePolicy(allowed_server_ids=("seed.mcp.local",))
    shadow_registry = McpClientCapabilityShadowRegistry.from_mcp_registry(
        mcp_registry,
        parent_checkpoint_id="checkpoint:mcp:e6-3",
    )
    shadow_registry.propose(candidate, policy)
    shadow_registry.record_shadow(candidate.candidate_digest, _observation(candidate))
    proposal = shadow_registry.propose_activation(
        candidate.candidate_digest,
        client_capability_snapshot_id="capability-snapshot:e6-3",
    )
    host = ClientExtensionHost(
        capability_snapshot_id="capability-snapshot:e6-3",
        parent_checkpoint_id="checkpoint:client:e6-3",
    )
    before_host_snapshot = host.snapshot.snapshot_id
    manifest = build_client_activation_manifest(candidate, proposal)
    dry_run = run_client_activation_dry_run(
        host,
        candidate,
        proposal,
        client_capability_snapshot_id="capability-snapshot:e6-3",
        available_capabilities=tuple(item.tool_id for item in candidate.tool_contracts),
    )
    dry_run_target = dry_run.target_client_snapshot_id

    routes_mcp_client_capabilities._registry = None
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        status = client.get("/api/mcp-client-capabilities")
        api_candidate = _candidate(McpToolRegistry.default())
        client.post(
            "/api/mcp-client-capabilities/proposals",
            json={"candidate": api_candidate.to_payload(), "policy": {}},
        )
        client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/shadow",
            json={"observation": _observation(api_candidate).to_payload()},
        )
        after_shadow = client.get("/api/mcp-client-capabilities").json()
        client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/activation-proposals",
            json={
                "client_capability_snapshot_id": after_shadow[
                    "client_capability_snapshot_id"
                ]
            },
        )
        api_dry_run = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/activation-dry-run",
            json={
                "client_capability_snapshot_id": after_shadow[
                    "client_capability_snapshot_id"
                ]
            },
        )
        stale_dry_run = client.post(
            f"/api/mcp-client-capabilities/{api_candidate.candidate_digest}/activation-dry-run",
            json={"client_capability_snapshot_id": "stale-client-snapshot"},
        )
    routes_mcp_client_capabilities._registry = None

    dry_run_source = (
        PROJECT_ROOT / "seed_platform" / "mcp_client_activation_dry_run.py"
    ).read_text(encoding="utf-8")
    api_source = (
        PROJECT_ROOT / "api" / "routes_mcp_client_capabilities.py"
    ).read_text(encoding="utf-8")
    checks = {
        "manifest_is_declarative": "executor_id" not in manifest.to_payload()
        and '"executor"' not in manifest.to_payload()["metadata"],
        "prepare_dry_run_is_not_committed": dry_run.committed is False,
        "host_active_snapshot_is_unchanged": host.snapshot.snapshot_id == before_host_snapshot
        and host.active_manifests == (),
        "prepared_target_is_content_addressed": bool(dry_run_target)
        and dry_run.plugin_digest == manifest.plugin_digest,
        "api_exposes_dry_run_only": status.status_code == 200
        and api_dry_run.status_code == 200
        and api_dry_run.json()["status"] == "dry_run"
        and api_dry_run.json()["activation"] == "not_committed"
        and api_dry_run.json()["dry_run"]["committed"] is False,
        "api_rejects_stale_client_snapshot": stale_dry_run.status_code == 409,
        "proposal_binding_is_preserved": proposal.client_capability_snapshot_id
        == "capability-snapshot:e6-3"
        and dry_run.candidate_digest == candidate.candidate_digest,
        "mcp_registry_is_not_mutated": mcp_registry.snapshot_id == mcp_snapshot_before,
        "no_active_client_organ_exists": all(
            item.state != "active" for item in shadow_registry.activation_proposals
        ),
        "no_external_or_cognitive_execution_boundary": all(
            marker not in dry_run_source + api_source
            for marker in ("import requests", "import socket", "subprocess", "httpx", "from taiji")
        ),
    }
    return {
        "gate": "taiji-e6-3-mcp-client-activation-dry-run",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scope": {
            "client_extension_host_prepare_only": True,
            "client_extension_host_commit_called": False,
            "synthetic_local_manifest_only": True,
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
        / "taiji_w7_e6_3_mcp_client_activation_dry_run_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

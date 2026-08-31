"""Verify the E6-0 MCP client-capability inheritance boundary."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.mcp_capability_inheritance import (  # noqa: E402
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
    McpClientToolContract,
    evaluate_mcp_capability_shadow,
    preflight_inheritance_candidate,
)
from seed_platform.mcp_registry import McpToolRegistry  # noqa: E402


def _candidate(registry: McpToolRegistry) -> McpCapabilityInheritanceCandidate:
    tool = McpClientToolContract.from_descriptor(registry.list_tools()[0])
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
        rationale="inherit a governed local read-only MCP contract",
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
    registry = McpToolRegistry.default()
    snapshot_before = registry.snapshot_id
    candidate = _candidate(registry)
    policy = McpCapabilityInheritancePolicy(allowed_server_ids=("seed.mcp.local",))
    pending = preflight_inheritance_candidate(
        candidate,
        policy,
        current_registry_snapshot_id=registry.snapshot_id,
    )
    stale = preflight_inheritance_candidate(
        candidate,
        policy,
        current_registry_snapshot_id="stale-registry",
    )
    observation = _observation(candidate)
    shadow = evaluate_mcp_capability_shadow(
        candidate,
        policy,
        observation,
        current_registry_snapshot_id=registry.snapshot_id,
    )
    rejected_shadow = evaluate_mcp_capability_shadow(
        candidate,
        policy,
        replace(observation, external_calls_performed=True),
        current_registry_snapshot_id=registry.snapshot_id,
    )
    module_source = (PROJECT_ROOT / "seed_platform" / "mcp_capability_inheritance.py").read_text(
        encoding="utf-8"
    )
    payload = candidate.to_payload()
    restored = McpCapabilityInheritanceCandidate.from_payload(payload)
    checks = {
        "candidate_roundtrip_is_content_addressed": restored == candidate,
        "schema_digest_is_preserved": payload["tool_contracts"][0]["schema_digest"]
        == restored.tool_contracts[0].schema_digest,
        "executor_source_is_not_exported": "executor_id" not in payload["tool_contracts"][0]
        and "source" not in payload["tool_contracts"][0],
        "preflight_stops_at_shadow": pending.passed and pending.decision == "shadow_pending",
        "registry_drift_fails_closed": not stale.passed and stale.reason_code == "stale_mcp_registry",
        "equivalent_shadow_is_admissible": shadow.passed and shadow.decision == "shadow_equivalent",
        "shadow_external_call_is_rejected": not rejected_shadow.passed
        and rejected_shadow.reason_code == "shadow_external_call_detected",
        "candidate_does_not_mutate_registry": registry.snapshot_id == snapshot_before,
        "candidate_has_no_cognitive_owner": "from taiji" not in module_source
        and "CognitiveInternalizationArtifact" not in module_source,
        "candidate_has_no_external_executor": all(
            marker not in module_source for marker in ("import requests", "import socket", "subprocess", "httpx")
        ),
    }
    return {
        "gate": "taiji-e6-0-mcp-client-capability-boundary",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scope": {
            "cognitive_artifact_and_client_candidate_are_separate": True,
            "network_and_credentials_are_policy_fields_only": True,
            "shadow_performs_no_external_call": True,
            "client_capability_is_not_active": True,
            "taiji_cognition_checkpoint_mutated": False,
            "real_third_party_mcp_connected": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e6_0_mcp_client_capability_boundary_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

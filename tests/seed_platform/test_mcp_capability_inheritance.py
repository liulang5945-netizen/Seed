from dataclasses import replace

import pytest

from seed_platform.mcp_capability_inheritance import (
    McpCapabilityInheritanceCandidate,
    McpCapabilityInheritancePolicy,
    McpCapabilityShadowObservation,
    McpClientToolContract,
    evaluate_mcp_capability_shadow,
    preflight_inheritance_candidate,
)
from seed_platform.mcp_registry import McpToolRegistry


def _candidate(
    registry: McpToolRegistry | None = None,
    *,
    network_scopes: tuple[str, ...] = (),
    credential_refs: tuple[str, ...] = (),
    permissions: tuple[str, ...] = (),
    risk: str = "read_only",
) -> McpCapabilityInheritanceCandidate:
    registry = registry or McpToolRegistry.default()
    tool = McpClientToolContract.from_descriptor(registry.list_tools()[0])
    tool = replace(tool, permissions=permissions, risk=risk)
    return McpCapabilityInheritanceCandidate(
        server_id="seed.mcp.local",
        server_version="1.0.0",
        source_artifact_digest="artifact:mcp:local:v1",
        registry_snapshot_id=registry.snapshot_id,
        tool_contracts=(tool,),
        network_scopes=network_scopes,
        credential_refs=credential_refs,
        resource_budget={"max_cpu_ms": 100, "max_output_bytes": 1024},
        evidence_digests=("evidence:mcp:local",),
        evaluation_gates=("policy", "shadow", "rollback"),
        parent_checkpoint_id="checkpoint:mcp:local",
        rationale="inherit a governed read-only MCP contract as a client candidate",
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


def test_candidate_is_content_addressed_and_does_not_export_executor_source() -> None:
    registry = McpToolRegistry.default()
    candidate = _candidate(registry)

    payload = candidate.to_payload()
    restored = McpCapabilityInheritanceCandidate.from_payload(payload)

    assert restored == candidate
    assert payload["tool_contracts"][0]["input_schema"]["properties"]["path"]["type"] == "string"
    assert "executor_id" not in payload["tool_contracts"][0]
    assert "source" not in payload["tool_contracts"][0]


def test_preflight_requires_current_registry_and_shadow_before_activation() -> None:
    registry = McpToolRegistry.default()
    candidate = _candidate(registry)
    policy = _policy()

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

    assert pending.passed is True
    assert pending.decision == "shadow_pending"
    assert pending.reason_code == "shadow_required"
    assert stale.passed is False
    assert stale.reason_code == "stale_mcp_registry"


def test_network_credentials_and_permissions_fail_closed() -> None:
    registry = McpToolRegistry.default()
    policy = _policy()

    network = preflight_inheritance_candidate(
        _candidate(registry, network_scopes=("network:remote",)),
        policy,
        current_registry_snapshot_id=registry.snapshot_id,
    )
    credentials = preflight_inheritance_candidate(
        _candidate(registry, credential_refs=("credential-ref:github",)),
        policy,
        current_registry_snapshot_id=registry.snapshot_id,
    )
    permissions = preflight_inheritance_candidate(
        _candidate(registry, permissions=("workspace.read",)),
        policy,
        current_registry_snapshot_id=registry.snapshot_id,
    )

    assert network.reason_code == "network_scope_not_allowed"
    assert credentials.reason_code == "credentials_not_allowed"
    assert permissions.reason_code == "permission_not_allowed"


def test_equivalent_shadow_is_admissible_and_roundtrips() -> None:
    registry = McpToolRegistry.default()
    candidate = _candidate(registry)
    policy = _policy()
    observation = _observation(candidate)

    result = evaluate_mcp_capability_shadow(
        candidate,
        policy,
        observation,
        current_registry_snapshot_id=registry.snapshot_id,
    )

    assert result.passed is True
    assert result.decision == "shadow_equivalent"
    assert McpCapabilityShadowObservation.from_payload(observation.to_payload()) == observation


def test_shadow_external_call_credential_access_and_mismatch_are_rejected() -> None:
    registry = McpToolRegistry.default()
    candidate = _candidate(registry)
    policy = _policy()

    external = evaluate_mcp_capability_shadow(
        candidate,
        policy,
        _observation(candidate, external_calls_performed=True),
        current_registry_snapshot_id=registry.snapshot_id,
    )
    credential = evaluate_mcp_capability_shadow(
        candidate,
        policy,
        _observation(candidate, credential_accessed=True),
        current_registry_snapshot_id=registry.snapshot_id,
    )
    mismatch = evaluate_mcp_capability_shadow(
        candidate,
        policy,
        _observation(candidate, candidate_output={"entries": ["different"]}),
        current_registry_snapshot_id=registry.snapshot_id,
    )

    assert external.reason_code == "shadow_external_call_detected"
    assert credential.reason_code == "shadow_credential_access_detected"
    assert mismatch.reason_code == "shadow_output_mismatch"


def test_side_effect_candidate_needs_explicit_approval_after_shadow() -> None:
    registry = McpToolRegistry.default()
    candidate = _candidate(registry, risk="mcp_dispatch")
    policy = _policy(allowed_risks=("mcp_dispatch",))
    observation = _observation(candidate)

    without_approval = evaluate_mcp_capability_shadow(
        candidate,
        policy,
        observation,
        current_registry_snapshot_id=registry.snapshot_id,
    )
    with_approval = evaluate_mcp_capability_shadow(
        candidate,
        policy,
        observation,
        current_registry_snapshot_id=registry.snapshot_id,
        approval_id="approval:mcp:local",
    )

    assert without_approval.reason_code == "approval_required"
    assert with_approval.passed is True
    assert with_approval.approval_required is True


def test_forbidden_top_level_source_and_secret_fields_are_rejected() -> None:
    registry = McpToolRegistry.default()
    payload = _candidate(registry).to_payload()

    with pytest.raises(ValueError, match="forbidden"):
        McpCapabilityInheritanceCandidate.from_payload({**payload, "source_path": "plugin.py"})
    with pytest.raises(ValueError, match="forbidden"):
        McpCapabilityInheritanceCandidate.from_payload({**payload, "secret": "plaintext"})

from __future__ import annotations

from seed_platform.capability_registry import (
    CapabilityBundle,
    CapabilityCandidate,
    CapabilityRegistry,
)
from seed_platform.capability_shadow import CapabilityShadowObservation, evaluate_shadow


def _read_candidate() -> CapabilityCandidate:
    return CapabilityCandidate(
        bundle=CapabilityBundle(
            capability_id="workspace.read",
            schema={"type": "object", "properties": {"path": {"type": "string"}}},
            permissions=("workspace.read",),
            executor_id="workspace.read.shadow",
            executor_version="1.0.0",
        ),
        rationale="compare a candidate read executor without invoking source",
        evidence_digests=("evidence:shadow-read",),
        resource_budget={"max_cpu_ms": 10, "max_output_bytes": 1024},
        evaluation_gates=("shadow_equivalence",),
    )


def _write_candidate() -> CapabilityCandidate:
    return CapabilityCandidate(
        bundle=CapabilityBundle(
            capability_id="workspace.apply_patch",
            schema={"type": "object", "properties": {"patch": {"type": "string"}}},
            effect="file_write",
            risk="file_write",
            permissions=("workspace.write",),
            executor_id="workspace.apply_patch.shadow",
            executor_version="1.0.0",
            disposer_id="workspace.undo",
            disposer_version="1.0.0",
        ),
        rationale="compare a write candidate only through a no-side-effect simulation",
        evidence_digests=("evidence:shadow-write",),
        resource_budget={"max_cpu_ms": 20, "max_file_changes": 1},
        evaluation_gates=("policy", "approval", "shadow_equivalence"),
    )


def _shadow_candidate(registry: CapabilityRegistry, candidate: CapabilityCandidate) -> None:
    registry.propose(candidate)
    registry.validate_candidate(candidate.candidate_digest, validation_ref="validation:shadow")
    registry.shadow(candidate.bundle.bundle_digest)


def _observation(registry: CapabilityRegistry, candidate: CapabilityCandidate, **kwargs):
    parameters = {
        "capability_id": candidate.bundle.capability_id,
        "candidate_bundle_digest": candidate.bundle.bundle_digest,
        "registry_snapshot_id": registry.snapshot_id,
        "input_payload": {"path": "README.md"},
        "baseline_output": {"content": "same"},
        "candidate_output": {"content": "same"},
        "baseline_after_state": {"files": ["README.md"]},
        "candidate_after_state": {"files": ["README.md"]},
        "baseline_resources": {"cpu_ms": 1, "output_bytes": 10},
        "candidate_resources": {"cpu_ms": 2, "output_bytes": 12},
    }
    parameters.update(kwargs)
    return CapabilityShadowObservation.from_execution(
        **parameters,
    )


def test_read_only_shadow_is_content_addressed_and_equivalent() -> None:
    registry = CapabilityRegistry()
    candidate = _read_candidate()
    _shadow_candidate(registry, candidate)

    observation = _observation(registry, candidate)
    result = evaluate_shadow(registry, observation)

    assert result.passed is True
    assert result.reason_code == "shadow_equivalent"
    assert result.approval_required is False
    assert result.resource_delta == {"cpu_ms": 1.0, "output_bytes": 2.0}
    assert CapabilityShadowObservation.from_payload(observation.to_payload()) == observation


def test_side_effecting_shadow_requires_approval_but_never_executes_side_effects() -> None:
    registry = CapabilityRegistry()
    candidate = _write_candidate()
    _shadow_candidate(registry, candidate)

    without_approval = evaluate_shadow(registry, _observation(registry, candidate))
    with_approval = evaluate_shadow(
        registry,
        _observation(registry, candidate, approval_id="approval:shadow-write"),
    )

    assert without_approval.passed is False
    assert without_approval.reason_code == "approval_required"
    assert with_approval.passed is True
    assert with_approval.approval_required is True


def test_shadow_policy_stale_and_side_effect_red_proofs_fail_closed() -> None:
    registry = CapabilityRegistry()
    candidate = _read_candidate()
    _shadow_candidate(registry, candidate)

    policy_denied = evaluate_shadow(
        registry,
        _observation(registry, candidate, policy_allowed=False),
    )
    stale = evaluate_shadow(
        registry,
        _observation(registry, candidate, registry_snapshot_id="stale"),
    )
    side_effect = evaluate_shadow(
        registry,
        _observation(registry, candidate, side_effects_performed=True),
    )
    mismatch = evaluate_shadow(
        registry,
        _observation(
            registry,
            candidate,
            candidate_output={"content": "different"},
            require_output_equivalence=True,
        ),
    )

    assert policy_denied.reason_code == "policy_denied"
    assert stale.reason_code == "stale_capability_registry"
    assert side_effect.reason_code == "shadow_side_effect_detected"
    assert mismatch.reason_code == "output_mismatch"

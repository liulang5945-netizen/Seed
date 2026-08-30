"""Run the deterministic R5B-L2 no-side-effect shadow canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.capability_registry import (  # noqa: E402
    CapabilityBundle,
    CapabilityCandidate,
    CapabilityRegistry,
)
from seed_platform.capability_shadow import (  # noqa: E402
    CapabilityShadowObservation,
    evaluate_shadow,
)

REPORT_FORMAT = "taiji-w7-r5b-l2-capability-shadow-v1"


def _candidate(*, capability_id: str, effect: str = "read_only") -> CapabilityCandidate:
    side_effecting = effect in {"file_write", "terminal"}
    return CapabilityCandidate(
        bundle=CapabilityBundle(
            capability_id=capability_id,
            schema={"type": "object", "properties": {"path": {"type": "string"}}},
            effect=effect,
            risk=effect,
            permissions=(capability_id,),
            executor_id=f"{capability_id}.shadow",
            executor_version="1.0.0",
            disposer_id="workspace.undo" if side_effecting else "",
            disposer_version="1.0.0" if side_effecting else "",
        ),
        rationale="compare a candidate through digest-only no-side-effect shadow execution",
        evidence_digests=(f"evidence:{capability_id}:shadow",),
        resource_budget={"max_cpu_ms": 20, "max_output_bytes": 1024},
        evaluation_gates=("policy", "approval", "shadow_equivalence"),
    )


def _observation(
    registry: CapabilityRegistry,
    candidate: CapabilityCandidate,
    **overrides: object,
) -> CapabilityShadowObservation:
    parameters: dict[str, object] = {
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
    parameters.update(overrides)
    return CapabilityShadowObservation.from_execution(**parameters)


def _prepare(registry: CapabilityRegistry, candidate: CapabilityCandidate) -> None:
    registry.propose(candidate)
    registry.validate_candidate(candidate.candidate_digest, validation_ref="validation:r5b-l2")
    registry.shadow(candidate.bundle.bundle_digest)


def evaluate_shadow_canary() -> dict[str, object]:
    registry = CapabilityRegistry(parent_checkpoint_id="checkpoint:r5b-l2-canary")
    read_candidate = _candidate(capability_id="workspace.read")
    _prepare(registry, read_candidate)

    read_observation = _observation(registry, read_candidate)
    read_result = evaluate_shadow(registry, read_observation)
    observation_roundtrip = (
        CapabilityShadowObservation.from_payload(read_observation.to_payload()) == read_observation
    )
    policy_denied = evaluate_shadow(
        registry,
        _observation(registry, read_candidate, policy_allowed=False),
    )
    stale = evaluate_shadow(
        registry,
        _observation(registry, read_candidate, registry_snapshot_id="stale"),
    )
    side_effect_detected = evaluate_shadow(
        registry,
        _observation(registry, read_candidate, side_effects_performed=True),
    )

    write_candidate = _candidate(capability_id="workspace.apply_patch", effect="file_write")
    _prepare(registry, write_candidate)
    write_without_approval = evaluate_shadow(registry, _observation(registry, write_candidate))
    write_with_approval = evaluate_shadow(
        registry,
        _observation(registry, write_candidate, approval_id="approval:r5b-l2"),
    )

    passed = (
        read_result.passed
        and observation_roundtrip
        and policy_denied.reason_code == "policy_denied"
        and stale.reason_code == "stale_capability_registry"
        and side_effect_detected.reason_code == "shadow_side_effect_detected"
        and write_without_approval.reason_code == "approval_required"
        and write_with_approval.passed
    )
    return {
        "format": REPORT_FORMAT,
        "read_only_shadow": read_result.to_payload(),
        "side_effect_shadow": {
            "without_approval": write_without_approval.to_payload(),
            "with_approval": write_with_approval.to_payload(),
        },
        "red_proofs": {
            "policy_denied": policy_denied.to_payload(),
            "stale_registry": stale.to_payload(),
            "side_effect_detected": side_effect_detected.to_payload(),
        },
        "observation_roundtrip": observation_roundtrip,
        "gate": {
            "passed": passed,
            "criterion": "shadow observations are digest-bound, after-state safe, policy-aware, and approval-aware without executing side effects",
        },
        "boundary": "R5B-L2 shadow comparison only; no executor source evaluation, cognition ownership, training, CUDA, or physical deletion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5b_l2_capability_shadow_20260830.json",
    )
    args = parser.parse_args()
    report = evaluate_shadow_canary()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit("R5B-L2 capability shadow canary failed")


if __name__ == "__main__":
    main()

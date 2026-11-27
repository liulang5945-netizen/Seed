"""Run the deterministic R5B-L1 capability-candidate canary."""

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

REPORT_FORMAT = "taiji-w7-r5b-l1-capability-candidate-v1"


def _read_candidate(*, variant: str = "base") -> CapabilityCandidate:
    return CapabilityCandidate(
        bundle=CapabilityBundle(
            capability_id="workspace.read",
            schema={"type": "object", "properties": {"path": {"type": "string"}}},
            permissions=("workspace.read",),
            executor_id="workspace.read.candidate",
            executor_version="1.0.0",
        ),
        rationale="propose a read capability through a reviewable artifact boundary",
        evidence_digests=("evidence:workbench-read", "evidence:rollback-canary"),
        resource_budget={"max_cpu_ms": 25, "max_output_bytes": 2048},
        evaluation_gates=("shadow_equivalence", "rollback"),
        metadata={"origin": "r5b-l1-canary", "variant": variant},
    )


def evaluate_candidate() -> dict[str, object]:
    registry = CapabilityRegistry(parent_checkpoint_id="checkpoint:r5b-l1-canary")
    candidate = _read_candidate()
    initial_snapshot = registry.snapshot_id

    proposed = registry.propose(candidate)
    proposed_checkpoint = registry.checkpoint()
    restored = CapabilityRegistry.from_checkpoint(proposed_checkpoint)
    checkpoint_roundtrip = (
        restored.snapshot.to_payload() == registry.snapshot.to_payload()
        and restored.get_candidate_record(candidate.candidate_digest) == proposed
    )
    bundle_registered_before_validation = (
        registry.get_record(candidate.bundle.bundle_digest) is not None
    )
    snapshot_unchanged_before_validation = registry.snapshot_id == initial_snapshot

    validated = registry.validate_candidate(
        candidate.candidate_digest,
        validation_ref="validation:r5b-l1-canary",
    )
    remains_non_active_after_validation = registry.snapshot.bundles == ()
    shadowed = registry.shadow(candidate.bundle.bundle_digest)
    activated = registry.activate(
        candidate.bundle.bundle_digest,
        approval_id="approval:r5b-l1-canary",
        expected_snapshot_id=registry.snapshot_id,
    )

    rejected_registry = CapabilityRegistry(parent_checkpoint_id="checkpoint:r5b-l1-rejected")
    rejected_candidate = _read_candidate(variant="rejected")
    rejected_registry.propose(rejected_candidate)
    rejected = rejected_registry.reject_candidate(
        rejected_candidate.candidate_digest,
        decision_ref="review:r5b-l1-canary",
        reason="oracle coverage is insufficient",
    )

    tampered_payload = _read_candidate().to_payload()
    tampered_payload["bundle"]["metadata"]["source_path"] = "executor.py"
    try:
        CapabilityCandidate.from_payload(tampered_payload)
    except ValueError as error:
        tamper_error = str(error)
    else:
        tamper_error = ""

    passed = (
        proposed.status == "proposed"
        and bundle_registered_before_validation is False
        and snapshot_unchanged_before_validation
        and checkpoint_roundtrip
        and validated.status == "validated"
        and remains_non_active_after_validation
        and shadowed.status == "shadow"
        and activated.status == "active"
        and rejected.status == "rejected"
        and rejected_registry.get_record(rejected_candidate.bundle.bundle_digest) is None
        and bool(tamper_error)
    )
    return {
        "format": REPORT_FORMAT,
        "candidate": {
            "candidate_digest": candidate.candidate_digest,
            "bundle_digest": candidate.bundle.bundle_digest,
            "resource_budget": dict(candidate.resource_budget),
            "evidence_digests": list(candidate.evidence_digests),
            "evaluation_gates": list(candidate.evaluation_gates),
        },
        "lifecycle": {
            "proposed": proposed.status,
            "validated": validated.status,
            "shadowed": shadowed.status,
            "activated": activated.status,
            "rejected": rejected.status,
        },
        "safety": {
            "bundle_registered_before_validation": bundle_registered_before_validation,
            "snapshot_unchanged_before_validation": snapshot_unchanged_before_validation,
            "remains_non_active_after_validation": remains_non_active_after_validation,
            "checkpoint_roundtrip": checkpoint_roundtrip,
            "tamper_error": tamper_error,
        },
        "gate": {
            "passed": passed,
            "criterion": "candidate packages are content-addressed, evidence/resource bounded, non-executable while proposed, and require validation, shadow, and approval before activation",
        },
        "boundary": "R5B-L1 candidate package only; no cognition ownership, provider selection, training, CUDA, or physical deletion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5b_l1_capability_candidate_20260830.json",
    )
    args = parser.parse_args()
    report = evaluate_candidate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit("R5B-L1 capability candidate canary failed")


if __name__ == "__main__":
    main()

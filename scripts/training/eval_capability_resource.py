"""Run the deterministic R5B-L3 resource and rollback canary."""

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

REPORT_FORMAT = "taiji-w7-r5b-l3-capability-resource-v1"


def _candidate(
    capability_id: str,
    *,
    cpu_ms: int,
    effect: str = "read_only",
) -> CapabilityCandidate:
    side_effecting = effect in {"file_write", "terminal"}
    return CapabilityCandidate(
        bundle=CapabilityBundle(
            capability_id=capability_id,
            schema={"type": "object"},
            effect=effect,
            risk=effect,
            permissions=(capability_id,),
            executor_id=f"{capability_id}.resource.{cpu_ms}",
            executor_version=f"1.0.{cpu_ms}",
            disposer_id="workspace.undo" if side_effecting else "",
            disposer_version="1.0.0" if side_effecting else "",
        ),
        rationale="reserve a bounded resource budget before activation",
        evidence_digests=(f"evidence:{capability_id}:resource:{cpu_ms}",),
        resource_budget={"max_cpu_ms": cpu_ms, "max_output_bytes": 100},
        evaluation_gates=("resource", "rollback"),
    )


def _prepare(registry: CapabilityRegistry, candidate: CapabilityCandidate) -> None:
    registry.propose(candidate)
    registry.validate_candidate(candidate.candidate_digest, validation_ref="validation:r5b-l3")
    registry.shadow(candidate.bundle.bundle_digest)


def evaluate_resource_canary() -> dict[str, object]:
    registry = CapabilityRegistry(
        parent_checkpoint_id="checkpoint:r5b-l3-canary",
        resource_limits={
            "active_bundle_count": 2,
            "max_cpu_ms": 5,
            "max_output_bytes": 500,
        },
    )
    first = _candidate("workspace.read", cpu_ms=3)
    second = _candidate("workspace.stat", cpu_ms=3)
    _prepare(registry, first)
    _prepare(registry, second)
    registry.activate(
        first.bundle.bundle_digest,
        approval_id="approval:r5b-l3:first",
        expected_snapshot_id=registry.snapshot_id,
    )
    before_snapshot = registry.snapshot_id
    before_ledger = registry.resource_ledger
    try:
        registry.activate(
            second.bundle.bundle_digest,
            approval_id="approval:r5b-l3:second",
            expected_snapshot_id=registry.snapshot_id,
        )
    except ValueError as error:
        exhaustion_error = str(error)
    else:
        exhaustion_error = ""
    exhaustion_atomic = (
        bool(exhaustion_error)
        and registry.snapshot_id == before_snapshot
        and registry.resource_ledger == before_ledger
        and registry.get_record(second.bundle.bundle_digest).status == "shadow"
    )

    replacement_registry = CapabilityRegistry(
        parent_checkpoint_id="checkpoint:r5b-l3-replacement",
        resource_limits={
            "active_bundle_count": 1,
            "max_cpu_ms": 10,
            "max_output_bytes": 500,
        },
    )
    old = _candidate("workspace.apply_patch", cpu_ms=2, effect="file_write")
    new = _candidate("workspace.apply_patch", cpu_ms=7, effect="file_write")
    _prepare(replacement_registry, old)
    replacement_registry.activate(
        old.bundle.bundle_digest,
        approval_id="approval:r5b-l3:old",
        expected_snapshot_id=replacement_registry.snapshot_id,
    )
    _prepare(replacement_registry, new)
    replacement_registry.replace(
        old.bundle.bundle_digest,
        new.bundle.bundle_digest,
        approval_id="approval:r5b-l3:replace",
        expected_snapshot_id=replacement_registry.snapshot_id,
    )
    replacement_usage = replacement_registry.resource_ledger["usage"]
    restored = CapabilityRegistry.from_checkpoint(replacement_registry.checkpoint())
    rolled_back = restored.rollback(
        new.bundle.bundle_digest, expected_snapshot_id=restored.snapshot_id
    )
    rollback_restored = restored.resource_ledger["usage"]

    passed = (
        exhaustion_atomic
        and replacement_usage["max_cpu_ms"] == 7.0
        and rolled_back.status == "rolled_back"
        and "disposer_release_recorded" in rolled_back.events
        and restored.resolve("workspace.apply_patch") == old.bundle
        and rollback_restored["max_cpu_ms"] == 2.0
    )
    return {
        "format": REPORT_FORMAT,
        "resource_limits": dict(registry.resource_limits),
        "exhaustion": {
            "error": exhaustion_error,
            "atomic": exhaustion_atomic,
            "unchanged_snapshot": registry.snapshot_id == before_snapshot,
        },
        "replacement": {
            "candidate_bundle_digest": new.bundle.bundle_digest,
            "usage_after_replace": dict(replacement_usage),
            "status_after_rollback": rolled_back.status,
            "disposer_release_recorded": "disposer_release_recorded" in rolled_back.events,
            "usage_after_rollback": dict(rollback_restored),
        },
        "checkpoint": {
            "resource_ledger_roundtrip": restored.resource_ledger["limits"]
            == replacement_registry.resource_ledger["limits"],
            "checkpoint_digest": replacement_registry.checkpoint()["checkpoint_digest"],
        },
        "gate": {
            "passed": passed,
            "criterion": "activation and replacement reserve resources atomically, persist reservations through checkpoint, and restore the parent ledger with disposer rollback evidence",
        },
        "boundary": "R5B-L3 resource and lifecycle rollback only; no executor source evaluation, cognition ownership, training, CUDA, or physical deletion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5b_l3_capability_resource_20260830.json",
    )
    args = parser.parse_args()
    report = evaluate_resource_canary()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit("R5B-L3 capability resource canary failed")


if __name__ == "__main__":
    main()

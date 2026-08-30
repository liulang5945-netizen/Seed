from __future__ import annotations

import pytest

from seed_platform.capability_registry import (
    CapabilityBundle,
    CapabilityCandidate,
    CapabilityRegistry,
)


def _candidate(
    capability_id: str,
    *,
    cpu_ms: int,
    effect: str = "read_only",
    disposer: bool = False,
) -> CapabilityCandidate:
    return CapabilityCandidate(
        bundle=CapabilityBundle(
            capability_id=capability_id,
            schema={"type": "object"},
            effect=effect,
            risk=effect,
            permissions=(capability_id,),
            executor_id=f"{capability_id}.resource.{cpu_ms}",
            executor_version=f"1.0.{cpu_ms}",
            disposer_id="workspace.undo" if disposer else "",
            disposer_version="1.0.0" if disposer else "",
        ),
        rationale="reserve bounded resources before activation",
        evidence_digests=(f"evidence:{capability_id}:resource",),
        resource_budget={"max_cpu_ms": cpu_ms, "max_output_bytes": 100},
        evaluation_gates=("resource", "rollback"),
    )


def _prepare(registry: CapabilityRegistry, candidate: CapabilityCandidate) -> None:
    registry.propose(candidate)
    registry.validate_candidate(candidate.candidate_digest, validation_ref="validation:resource")
    registry.shadow(candidate.bundle.bundle_digest)


def test_activation_reserves_resources_and_checkpoint_restores_the_ledger() -> None:
    registry = CapabilityRegistry(
        resource_limits={
            "active_bundle_count": 1,
            "max_cpu_ms": 5,
            "max_output_bytes": 500,
        }
    )
    candidate = _candidate("workspace.read", cpu_ms=3)
    _prepare(registry, candidate)

    registry.activate(
        candidate.bundle.bundle_digest,
        approval_id="approval:resource",
        expected_snapshot_id=registry.snapshot_id,
    )
    ledger = registry.resource_ledger
    restored = CapabilityRegistry.from_checkpoint(registry.checkpoint())

    assert ledger["usage"] == {
        "active_bundle_count": 1.0,
        "max_cpu_ms": 3.0,
        "max_output_bytes": 100.0,
    }
    assert restored.resource_ledger == ledger
    restored.rollback(candidate.bundle.bundle_digest, expected_snapshot_id=restored.snapshot_id)
    assert restored.resource_ledger["usage"] == {}


def test_resource_exhaustion_is_atomic_and_keeps_candidate_shadowed() -> None:
    registry = CapabilityRegistry(
        resource_limits={
            "active_bundle_count": 2,
            "max_cpu_ms": 5,
            "max_output_bytes": 500,
        }
    )
    first = _candidate("workspace.read", cpu_ms=3)
    second = _candidate("workspace.stat", cpu_ms=3)
    _prepare(registry, first)
    _prepare(registry, second)
    registry.activate(
        first.bundle.bundle_digest,
        approval_id="approval:first",
        expected_snapshot_id=registry.snapshot_id,
    )
    before_snapshot = registry.snapshot_id
    before_ledger = registry.resource_ledger

    with pytest.raises(ValueError, match="resource budget exhausted"):
        registry.activate(
            second.bundle.bundle_digest,
            approval_id="approval:second",
            expected_snapshot_id=registry.snapshot_id,
        )

    assert registry.snapshot_id == before_snapshot
    assert registry.resource_ledger == before_ledger
    assert registry.get_record(second.bundle.bundle_digest).status == "shadow"


def test_replacement_rollback_restores_parent_resources_and_records_disposer_release() -> None:
    registry = CapabilityRegistry(
        resource_limits={
            "active_bundle_count": 1,
            "max_cpu_ms": 10,
            "max_output_bytes": 500,
        }
    )
    old = _candidate("workspace.apply_patch", cpu_ms=2, effect="file_write", disposer=True)
    new = _candidate("workspace.apply_patch", cpu_ms=7, effect="file_write", disposer=True)
    _prepare(registry, old)
    registry.activate(
        old.bundle.bundle_digest,
        approval_id="approval:old",
        expected_snapshot_id=registry.snapshot_id,
    )
    _prepare(registry, new)
    registry.replace(
        old.bundle.bundle_digest,
        new.bundle.bundle_digest,
        approval_id="approval:replace",
        expected_snapshot_id=registry.snapshot_id,
    )
    assert registry.resource_ledger["usage"]["max_cpu_ms"] == 7.0

    rolled_back = registry.rollback(
        new.bundle.bundle_digest, expected_snapshot_id=registry.snapshot_id
    )

    assert rolled_back.status == "rolled_back"
    assert "disposer_release_recorded" in rolled_back.events
    assert registry.resolve("workspace.apply_patch") == old.bundle
    assert registry.resource_ledger["usage"]["max_cpu_ms"] == 2.0

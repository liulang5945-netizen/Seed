from __future__ import annotations

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_structural_lineage_compaction import _record_terminal_subgraph
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from seed import Seed
from taiji import StructuralMaintenanceAudit


def _runtime_with_terminal_lineage() -> tuple[SeedRuntime, str]:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    model = runtime.model.architecture
    active_batch_id = str(schedule["batch_id"])
    active = next(item for item in model.structural_candidate_batches if item.batch_id == active_batch_id)
    _record_terminal_subgraph(model, active)
    return runtime, active_batch_id


def _empty_maintenance(runtime: SeedRuntime, *, max_batches: int | None = None) -> dict:
    return runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_max_batches=max_batches,
    )


def test_seed_runtime_projects_a_stable_default_audit_without_retention() -> None:
    runtime, _ = _runtime_with_terminal_lineage()

    payload = _empty_maintenance(runtime)
    audit = StructuralMaintenanceAudit.from_payload(payload)

    assert audit.maintenance_results == ()
    assert audit.lineage_retention is None
    assert payload["audit_digest"] == audit.audit_digest


def test_seed_runtime_projects_explicit_retention_and_rejects_tampering() -> None:
    runtime, _ = _runtime_with_terminal_lineage()

    payload = _empty_maintenance(runtime, max_batches=1)
    audit = StructuralMaintenanceAudit.from_payload(payload)
    assert audit.lineage_retention is not None
    assert audit.lineage_retention.status == "compacted"
    assert runtime.model.architecture.structural_lineage_retention_result == audit.lineage_retention

    tampered = dict(payload)
    tampered["audit_digest"] = "0" * 64
    try:
        StructuralMaintenanceAudit.from_payload(tampered)
    except ValueError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered runtime maintenance audit must fail closed")


def test_seed_runtime_checkpoint_restore_keeps_audit_but_default_call_does_not_replay_it() -> None:
    runtime, _ = _runtime_with_terminal_lineage()
    first = StructuralMaintenanceAudit.from_payload(_empty_maintenance(runtime, max_batches=1))
    checkpoint = runtime.model.checkpoint()

    restored_runtime = SeedRuntime(Seed.from_checkpoint(checkpoint))
    restored_model = restored_runtime.model.architecture
    assert restored_model.structural_lineage_retention_result == first.lineage_retention

    default_after_restore = StructuralMaintenanceAudit.from_payload(
        _empty_maintenance(restored_runtime)
    )
    assert default_after_restore.lineage_retention is None
    assert restored_model.structural_lineage_retention_result == first.lineage_retention

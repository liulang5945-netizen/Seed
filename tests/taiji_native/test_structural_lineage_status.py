from __future__ import annotations

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_structural_lineage_compaction import _record_terminal_subgraph
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from seed import Seed
from taiji import STRUCTURAL_MAINTENANCE_STATUS_FORMAT
from taiji.adapter import _checkpoint_digest


def _runtime_with_terminal_lineage() -> SeedRuntime:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    model = runtime.model.architecture
    active = next(
        item
        for item in model.structural_candidate_batches
        if item.batch_id == str(schedule["batch_id"])
    )
    _record_terminal_subgraph(model, active)
    return runtime


def _maintain(runtime: SeedRuntime, *, max_batches: int | None = None) -> dict[str, object]:
    return runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_max_batches=max_batches,
    )


def test_status_has_explicit_empty_state_without_audit_or_side_effect() -> None:
    runtime = _runtime_with_terminal_lineage()
    before = _checkpoint_digest(runtime.model.architecture.native_checkpoint())

    status = runtime.status()["structural_maintenance"]

    assert status == {
        "format": STRUCTURAL_MAINTENANCE_STATUS_FORMAT,
        "status": "no_audit",
        "structural_runtime_tick": runtime.model.architecture.structural_runtime_tick,
        "has_retention_audit": False,
        "last_retention_audit": None,
        "last_retention_policy": None,
        "last_retention_policy_migration": None,
        "retention_pressure": False,
    }
    assert _checkpoint_digest(runtime.model.architecture.native_checkpoint()) == before


def test_status_projects_latest_retention_audit_without_becoming_a_decision_input() -> None:
    runtime = _runtime_with_terminal_lineage()
    audit = _maintain(runtime, max_batches=1)["lineage_retention"]
    status = runtime.structural_maintenance_status()

    assert status["status"] == "audit_available"
    assert status["has_retention_audit"] is True
    assert status["last_retention_audit"] == audit
    assert status["retention_pressure"] is False


def test_status_projection_survives_seed_checkpoint_restore() -> None:
    runtime = _runtime_with_terminal_lineage()
    _maintain(runtime, max_batches=1)
    expected = runtime.structural_maintenance_status()

    restored = SeedRuntime(Seed.from_checkpoint(runtime.model.checkpoint()))

    assert restored.structural_maintenance_status() == expected
    assert restored.status()["structural_maintenance"] == expected

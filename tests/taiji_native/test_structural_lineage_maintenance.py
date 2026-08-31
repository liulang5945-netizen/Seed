from __future__ import annotations

from dataclasses import replace

from scripts.training.eval_taiji_structural_lineage_compaction import _record_terminal_subgraph
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji import TSKV8Adapter
from taiji.adapter import _checkpoint_digest


def _model_with_active_and_terminal_batch() -> tuple[TSKV8Adapter, str, str]:
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
    return model, active_batch_id, "batch:terminal-lineage"


def _empty_maintenance(model: TSKV8Adapter, *, max_batches: int | None = None) -> tuple:
    return model.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_max_batches=max_batches,
    )


def test_lineage_compaction_is_only_triggered_by_explicit_maintenance() -> None:
    model, active_batch_id, terminal_batch_id = _model_with_active_and_terminal_batch()
    before = _checkpoint_digest(model.native_checkpoint())

    assert _empty_maintenance(model) == ()

    assert _checkpoint_digest(model.native_checkpoint()) == before
    assert {item.batch_id for item in model.structural_candidate_batches} == {
        active_batch_id,
        terminal_batch_id,
    }
    assert model.structural_lineage_retention_result is None

    assert _empty_maintenance(model, max_batches=1) == ()

    audit = model.structural_lineage_retention_result
    assert audit is not None
    assert audit.status == "compacted"
    assert audit.removed_batch_ids == (terminal_batch_id,)
    assert {item.batch_id for item in model.structural_candidate_batches} == {active_batch_id}


def test_maintenance_retention_audit_is_checkpointed_and_idempotent() -> None:
    model, active_batch_id, _ = _model_with_active_and_terminal_batch()
    _empty_maintenance(model, max_batches=1)
    active = next(item for item in model.structural_candidate_batches if item.batch_id == active_batch_id)
    model._record_structural_candidate_batch(
        replace(active, batch_id="batch:protected-copy", revision=active.revision + 1)
    )
    _empty_maintenance(model, max_batches=1)
    pressure_audit = model.structural_lineage_retention_result
    assert pressure_audit is not None
    assert pressure_audit.retention_pressure is True
    assert pressure_audit.status == "nothing_to_compact"

    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    assert restored.structural_lineage_retention_result == pressure_audit
    assert _checkpoint_digest(restored.native_checkpoint()) == _checkpoint_digest(checkpoint)

    _empty_maintenance(restored, max_batches=1)

    assert restored.structural_lineage_retention_result == pressure_audit
    assert {item.batch_id for item in restored.structural_candidate_batches} == {
        active_batch_id,
        "batch:protected-copy",
    }

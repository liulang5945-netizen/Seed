from __future__ import annotations

from dataclasses import replace

from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji import StructuralCandidateBatch


def _terminal_batch(batch: StructuralCandidateBatch, batch_id: str) -> StructuralCandidateBatch:
    return replace(
        batch,
        batch_id=batch_id,
        selected_candidate_ids=(),
        deferred_candidate_ids=(),
        rejected_candidate_ids=batch.candidate_ids,
        candidate_states=tuple((candidate_id, "rejected") for candidate_id in batch.candidate_ids),
        reserved_resource_cost=0,
        reservation_remaining=0,
        arbitration_digest=f"terminal:{batch_id}",
        revision=batch.revision + 1,
        status="completed",
    )


def test_lineage_retention_preserves_active_batch_and_pending_candidates() -> None:
    runtime = _build_runtime()
    model = runtime.model.architecture
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    batch_id = str(schedule["batch_id"])
    active = next(item for item in model.structural_candidate_batches if item.batch_id == batch_id)
    pending_ids = set(active.candidate_ids)
    model.config = replace(model.config, cognitive_lineage_history_limit=1)

    model._record_structural_candidate_batch(_terminal_batch(active, "batch:terminal"))
    assert batch_id in {item.batch_id for item in model.structural_candidate_batches}
    assert "batch:terminal" not in {item.batch_id for item in model.structural_candidate_batches}

    candidate = next(
        item for item in model.structural_proposal_candidates if item.candidate_id == active.candidate_ids[0]
    )
    extra = replace(
        candidate,
        candidate_id="candidate:extra-retention",
        specification=(
            ("region_id", dict(candidate.specification)["region_id"]),
            ("unit_id", "retention-extra"),
        ),
    )
    model._queue_structural_proposal_candidate(extra)
    remaining_ids = {item.candidate_id for item in model.structural_proposal_candidates}
    assert pending_ids.issubset(remaining_ids)
    assert "candidate:extra-retention" not in remaining_ids


def test_lineage_retention_preserves_admitted_batch_until_rollback_audit_exists() -> None:
    runtime = _build_runtime()
    model = runtime.model.architecture
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    batch_id = str(schedule["batch_id"])
    active = next(item for item in model.structural_candidate_batches if item.batch_id == batch_id)
    admitted = replace(
        active,
        candidate_states=tuple((candidate_id, "admitted") for candidate_id in active.candidate_ids),
        selected_candidate_ids=active.candidate_ids,
        reserved_resource_cost=0,
        reservation_remaining=0,
        status="completed",
    )
    model.config = replace(model.config, cognitive_lineage_history_limit=1)
    model._record_structural_candidate_batch(admitted)
    model._record_structural_candidate_batch(_terminal_batch(active, "batch:terminal-admitted"))

    batches = {item.batch_id: item for item in model.structural_candidate_batches}
    assert batches[batch_id].state_by_candidate == admitted.state_by_candidate

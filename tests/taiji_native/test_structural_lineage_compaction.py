from __future__ import annotations

from dataclasses import replace

from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji import (
    StructuralAdmissionResult,
    StructuralCandidateBatch,
    StructuralCandidateRollback,
    StructuralCandidateValidation,
    StructuralRuntimeObservation,
    StructuralValidationArtifactBatch,
    TSKV8Adapter,
    WorkbenchStructuralValidationArtifact,
    evaluate_structural_candidate_validation,
)
from taiji.adapter import _checkpoint_digest


def _terminal_batch(
    source: StructuralCandidateBatch,
    *,
    batch_id: str,
    candidate_id: str,
) -> StructuralCandidateBatch:
    return replace(
        source,
        batch_id=batch_id,
        candidate_ids=(candidate_id,),
        selected_candidate_ids=(),
        deferred_candidate_ids=(),
        rejected_candidate_ids=(candidate_id,),
        candidate_states=((candidate_id, "rejected"),),
        reasons=((candidate_id, "terminal"),),
        reserved_resource_cost=0,
        reservation_remaining=0,
        arbitration_digest=f"terminal:{batch_id}",
        revision=source.revision + 1,
        status="completed",
    )


def _terminal_records(
    model: TSKV8Adapter,
    *,
    batch_id: str,
    candidate_id: str,
) -> None:
    artifact = WorkbenchStructuralValidationArtifact.from_measurements(
        candidate_id=candidate_id,
        network_id="workbench",
        region_id="workbench.code",
        task_slice_id="terminal-lineage",
        outcome_digests=("terminal:outcome",),
        parent_checkpoint_digest="terminal:parent",
        trial_checkpoint_digest="terminal:trial",
        holdout_inputs=(),
        holdout_outputs=(),
        retention_baseline=(),
        retention_candidate=(),
        lesion_baseline=(),
        lesion_candidate=(),
        resource_measurement=(),
        holdout_gain=0.0,
        retention_regression=0.0,
        lesion_effect=0.0,
        resource_state=0.0,
        resource_cost=1,
        evidence_ids=("terminal:evidence",),
    )
    model._record_structural_validation_artifact(artifact)
    model._record_structural_validation_artifact_batch(
        StructuralValidationArtifactBatch.from_digest_map(
            batch_id=batch_id,
            expected_candidate_ids=(candidate_id,),
            artifact_digests_by_candidate={candidate_id: artifact.artifact_digest},
        )
    )
    model._record_structural_candidate_validation(
        StructuralCandidateValidation(
            candidate_id=candidate_id,
            proposal_id=None,
            status="rejected",
            validation_score=0.0,
            parent_checkpoint_digest="terminal:parent",
            validation_checkpoint_digest="terminal:validation",
            topology_before_digest="terminal:topology-before",
            topology_after_digest="terminal:topology-after",
            structural_budget_before=1,
            structural_budget_after=1,
            evidence_ids=("terminal:evidence",),
            error="terminal validation",
        )
    )
    decision = evaluate_structural_candidate_validation(
        candidate_id,
        holdout_gain=0.0,
        retention_regression=0.0,
        lesion_effect=0.0,
        resource_state=0.0,
        resource_cost=1,
        structural_budget=1,
        evidence_ids=("terminal:evidence",),
    )
    assert not decision.passed
    model._record_structural_validation_gate_decision(decision)
    model._record_structural_admission_result(
        StructuralAdmissionResult(
            candidate_id=candidate_id,
            proposal_id="terminal:proposal",
            status="rejected",
            decision_digest=decision.decision_digest,
            parent_checkpoint_digest="terminal:parent",
            child_checkpoint_digest="terminal:child",
            topology_before_digest="terminal:topology-before",
            topology_after_digest="terminal:topology-after",
            structural_budget_before=1,
            structural_budget_after=1,
            error="terminal admission",
        )
    )
    model._record_structural_candidate_rollback(
        StructuralCandidateRollback(
            batch_id=batch_id,
            candidate_id=candidate_id,
            proposal_id="terminal:proposal",
            status="not_rollbackable",
            admission_parent_checkpoint_digest="terminal:parent",
            admission_child_checkpoint_digest="terminal:child",
            rollback_checkpoint_digest="terminal:rollback",
            topology_before_digest="terminal:topology-before",
            topology_after_digest="terminal:topology-after",
            structural_budget_before=1,
            structural_budget_after=1,
            resource_cost=1,
            reason="terminal test record",
        )
    )
    model._record_structural_workbench_batch_schedule_result(
        replace(
            model.structural_workbench_batch_schedule_results[-1],
            batch_id=batch_id,
            candidate_ids=(candidate_id,),
            status="batch_created",
        )
    )


def _prepare_model() -> tuple[TSKV8Adapter, str, str, object]:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    model = runtime.model.architecture
    active_batch_id = str(schedule["batch_id"])
    active = next(item for item in model.structural_candidate_batches if item.batch_id == active_batch_id)
    terminal_batch_id = "batch:terminal-lineage"
    terminal_candidate_id = "candidate:terminal-lineage"
    model._record_structural_candidate_batch(
        _terminal_batch(active, batch_id=terminal_batch_id, candidate_id=terminal_candidate_id)
    )
    _terminal_records(
        model,
        batch_id=terminal_batch_id,
        candidate_id=terminal_candidate_id,
    )
    return model, active_batch_id, terminal_batch_id, schedule


def test_coordinated_compaction_removes_only_terminal_subgraph() -> None:
    model, active_batch_id, terminal_batch_id, schedule = _prepare_model()
    result = model.compact_structural_lineage_history(max_batches=1)

    assert result.status == "compacted"
    assert result.removed_batch_ids == (terminal_batch_id,)
    assert result.retention_pressure is False
    assert active_batch_id in result.retained_batch_ids
    assert terminal_batch_id not in {item.batch_id for item in model.structural_candidate_batches}
    assert terminal_batch_id not in {
        item.batch_id for item in model.structural_validation_artifact_batches
    }
    assert not any(
        item.candidate_id == "candidate:terminal-lineage"
        for item in model.structural_candidate_validations
    )
    assert not any(
        item.candidate_id == "candidate:terminal-lineage"
        for item in model.structural_validation_artifacts
    )
    assert not any(
        item.candidate_id == "candidate:terminal-lineage"
        for item in model.structural_validation_gate_decisions
    )
    assert not any(
        item.candidate_id == "candidate:terminal-lineage"
        for item in model.structural_admission_results
    )
    assert not any(
        item.batch_id == terminal_batch_id for item in model.structural_candidate_rollbacks
    )
    assert schedule["batch_id"] == active_batch_id


def test_compaction_is_checkpointable_and_old_lineage_fails_closed() -> None:
    model, active_batch_id, terminal_batch_id, _ = _prepare_model()
    old_artifact = model.structural_validation_artifacts[-1]

    result = model.compact_structural_lineage_history(max_batches=1)
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)

    assert _checkpoint_digest(restored.native_checkpoint()) == _checkpoint_digest(checkpoint)
    assert active_batch_id in {item.batch_id for item in restored.structural_candidate_batches}
    assert terminal_batch_id not in {item.batch_id for item in restored.structural_candidate_batches}
    assert restored.materialize_structural_candidate(old_artifact.candidate_id) is None
    stale = restored.continue_structural_candidate_from_validation_artifact(
        old_artifact,
        holdout_inputs=(),
        expected_activities=(),
    )
    assert stale["status"] == "failed_closed"
    try:
        restored.rollback_structural_candidate_batch(
            terminal_batch_id,
            old_artifact.candidate_id,
        )
    except ValueError as exc:
        assert "unknown structural candidate batch" in str(exc)
    else:
        raise AssertionError("removed terminal lineage unexpectedly became rollbackable")
    repeated = restored.compact_structural_lineage_history(max_batches=1)
    assert repeated.status == "nothing_to_compact"
    assert repeated.removed_batch_ids == ()
    assert result.result_digest != repeated.result_digest


def test_protected_lineage_reports_pressure_without_deletion() -> None:
    model, active_batch_id, _, _ = _prepare_model()
    active = next(item for item in model.structural_candidate_batches if item.batch_id == active_batch_id)
    model._record_structural_candidate_batch(
        replace(active, batch_id="batch:protected-copy", revision=active.revision + 1)
    )

    result = model.compact_structural_lineage_history(max_batches=1)

    assert result.status == "compacted"
    assert result.retention_pressure is True
    assert result.removed_batch_ids == ("batch:terminal-lineage",)
    assert {item.batch_id for item in model.structural_candidate_batches} == {
        active_batch_id,
        "batch:protected-copy",
    }


def test_checkpoint_restore_continues_new_evidence_deterministically() -> None:
    model, _, _, _ = _prepare_model()
    first = model.compact_structural_lineage_history(max_batches=1)
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    observation = StructuralRuntimeObservation(
        network_id="workbench",
        region_id="workbench.code",
        tick=model.structural_runtime_tick + 1,
        usage=0.8,
        resource_pressure=0.2,
        prediction_error=0.2,
        learning_gain=0.1,
        holdout_transfer=0.0,
        evidence_id="s23:fresh",
        task_slice_id="s23-fresh",
        partition="train",
    )
    for current in (model, restored):
        current.record_structural_runtime_observation(observation)
        assert (
            current.seal_structural_evidence_window(
                "workbench",
                "workbench.code",
                task_slice_id="s23-fresh",
                partition="train",
            )
            is not None
        )

    left = model.schedule_structural_candidate_batch_from_workbench_evidence(_schedule_requests())
    right = restored.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )

    assert first.status == "compacted"
    assert left == right
    assert left.status == "batch_created"
    assert left.trigger_tick == observation.tick

from __future__ import annotations

import os
from pathlib import Path

import torch

from scripts.training.eval_taiji_structural_lineage_restart_continuation import (
    _build_migrated_runtime,
    _continuation_requests,
    _record_continuation_evidence,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from taiji import TSKV8Adapter
from taiji.adapter import _checkpoint_digest


def _save_native_checkpoint(model: TSKV8Adapter, path: Path) -> None:
    torch.save(model.native_checkpoint(), path)


def _load_native_checkpoint(path: Path) -> TSKV8Adapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return TSKV8Adapter.from_native_checkpoint(checkpoint)


def test_artifact_provenance_survives_rollback_and_terminal_compaction() -> None:
    runtime = _build_migrated_runtime()
    continuation_evidence = _record_continuation_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _continuation_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_candidate, second_candidate = batch.selected_candidate_ids

    checkpoint_root = Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    before_path = checkpoint_root / f"s33-before-{suffix}.pt"
    after_first_path = checkpoint_root / f"s33-after-first-{suffix}.pt"
    after_rollback_path = checkpoint_root / f"s33-after-rollback-{suffix}.pt"
    after_compaction_path = checkpoint_root / f"s33-after-compaction-{suffix}.pt"
    try:
        _save_native_checkpoint(runtime.model.architecture, before_path)
        restored = _load_native_checkpoint(before_path)
        first_artifact, first_replay, _ = _build_artifact(
            restored,
            first_candidate,
            continuation_evidence,
        )
        _save_native_checkpoint(restored, before_path)
        restored = _load_native_checkpoint(before_path)
        first_result = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={first_candidate: first_artifact},
            replays_by_candidate={first_candidate: first_replay},
        )
        assert first_result["results"][first_candidate]["status"] == "admitted"
        _save_native_checkpoint(restored, after_first_path)
        first_resumed = _load_native_checkpoint(after_first_path)

        second_artifact, second_replay, _ = _build_artifact(
            first_resumed,
            second_candidate,
            continuation_evidence,
        )
        second_result = first_resumed.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact},
            replays_by_candidate={second_candidate: second_replay},
        )
        assert second_result["results"][second_candidate]["status"] == "admitted"
        second_rollback = first_resumed.rollback_structural_candidate_batch(
            batch.batch_id,
            second_candidate,
        )
        assert second_rollback["status"] == "rolled_back"
        second_replay_after_rollback = (
            first_resumed.continue_structural_candidate_from_validation_artifact(
                second_artifact,
                holdout_inputs=second_replay["holdout_inputs"],
                expected_activities=second_replay["expected_activities"],
            )
        )
        assert second_replay_after_rollback["status"] == "rolled_back"
        _save_native_checkpoint(first_resumed, after_rollback_path)
        rollback_resumed = _load_native_checkpoint(after_rollback_path)
        persisted_second_artifact = next(
            item
            for item in rollback_resumed.structural_validation_artifacts
            if item.artifact_digest == second_artifact.artifact_digest
        )
        assert persisted_second_artifact.artifact_digest == second_artifact.artifact_digest

        before_first_rollback = rollback_resumed.cognitive_snapshot().development.structural_budget
        first_rollback = rollback_resumed.rollback_structural_candidate_batch(
            batch.batch_id,
            first_candidate,
        )
        assert first_rollback["status"] == "rolled_back"
        assert (
            rollback_resumed.cognitive_snapshot().development.structural_budget
            == before_first_rollback + first_rollback["resource_cost"]
        )
        maintenance = rollback_resumed.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=(
                rollback_resumed.structural_lineage_retention_policy.to_payload()
            ),
        )
        assert maintenance == ()
        retention = rollback_resumed.structural_lineage_retention_result
        assert retention is not None
        assert retention.status == "compacted"
        assert batch.batch_id in retention.removed_batch_ids
        assert all(
            item.artifact_digest not in {
                first_artifact.artifact_digest,
                second_artifact.artifact_digest,
            }
            for item in rollback_resumed.structural_validation_artifacts
        )

        before_replay_after_compaction = _checkpoint_digest(
            rollback_resumed.native_checkpoint()
        )
        try:
            rollback_resumed.continue_structural_candidate_batch_from_validation_artifacts(
                batch.batch_id,
                artifacts_by_candidate={
                    first_candidate: first_artifact,
                    second_candidate: second_artifact,
                },
                replays_by_candidate={
                    first_candidate: first_replay,
                    second_candidate: second_replay,
                },
            )
        except ValueError as exc:
            assert "unknown structural candidate batch" in str(exc)
        else:
            raise AssertionError("compacted artifact batch unexpectedly replayed")
        assert _checkpoint_digest(rollback_resumed.native_checkpoint()) == (
            before_replay_after_compaction
        )
        _save_native_checkpoint(rollback_resumed, after_compaction_path)
        final = _load_native_checkpoint(after_compaction_path)
        assert batch.batch_id not in {item.batch_id for item in final.structural_candidate_batches}
        assert batch.batch_id not in {item.batch_id for item in final.structural_candidate_rollbacks}
    finally:
        before_path.unlink(missing_ok=True)
        after_first_path.unlink(missing_ok=True)
        after_rollback_path.unlink(missing_ok=True)
        after_compaction_path.unlink(missing_ok=True)

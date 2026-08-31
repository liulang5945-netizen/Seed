from __future__ import annotations

import os
from pathlib import Path

import torch

from scripts.training.eval_taiji_structural_lineage_restart_continuation import (
    _build_migrated_runtime,
    _continuation_requests,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from scripts.training.eval_taiji_workbench_multi_region_batch import _execute_observation
from taiji import TSKV8Adapter
from taiji.adapter import _checkpoint_digest


def _save_native_checkpoint(model: TSKV8Adapter, path: Path) -> None:
    torch.save(model.native_checkpoint(), path)


def _load_native_checkpoint(path: Path) -> TSKV8Adapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return TSKV8Adapter.from_native_checkpoint(checkpoint)


def _record_second_round(runtime) -> tuple[dict[str, object], ...]:
    rows = (
        (13, "workbench.code", "code-isolation-read", "train", "README.md"),
        (14, "workbench.code", "code-isolation-config", "train", "pyproject.toml"),
        (15, "workbench.code", "code-isolation-holdout", "holdout", "plans/README.md"),
        (16, "workbench.docs", "docs-isolation-roadmap", "train", "plans/README.md"),
        (17, "workbench.docs", "docs-isolation-frontend", "train", "frontend/package.json"),
        (18, "workbench.docs", "docs-isolation-holdout", "holdout", "README.md"),
    )
    return tuple(
        _execute_observation(
            runtime,
            ordinal=ordinal,
            region_id=region_id,
            task_slice_id=task_slice_id,
            partition=partition,
            path=path,
            prediction_error=0.1 if partition == "holdout" else 0.8,
            holdout_transfer=0.9 if partition == "holdout" else 0.0,
        )
        for ordinal, region_id, task_slice_id, partition, path in rows
    )


def test_multi_batch_artifact_retention_preserves_active_lineage() -> None:
    runtime = _build_migrated_runtime()
    active_batch = runtime.model.architecture.structural_candidate_batches[-1]
    active_batch_id = active_batch.batch_id
    second_round = _record_second_round(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _continuation_requests()
    )
    assert schedule.get("status") == "batch_created"
    terminal_batch = runtime.model.architecture.structural_candidate_batches[-1]
    assert terminal_batch.batch_id != active_batch_id
    first_candidate, second_candidate = terminal_batch.selected_candidate_ids

    checkpoint_root = Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    before_maintenance_path = checkpoint_root / f"s34-before-maintenance-{suffix}.pt"
    after_maintenance_path = checkpoint_root / f"s34-after-maintenance-{suffix}.pt"
    try:
        first_artifact, first_replay, _ = _build_artifact(
            runtime.model.architecture,
            first_candidate,
            second_round,
        )
        first_result = runtime.model.architecture.continue_structural_candidate_batch_from_validation_artifacts(
            terminal_batch.batch_id,
            artifacts_by_candidate={first_candidate: first_artifact},
            replays_by_candidate={first_candidate: first_replay},
        )
        assert first_result["results"][first_candidate]["status"] == "admitted"
        second_artifact, second_replay, _ = _build_artifact(
            runtime.model.architecture,
            second_candidate,
            second_round,
        )
        second_result = runtime.model.architecture.continue_structural_candidate_batch_from_validation_artifacts(
            terminal_batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact},
            replays_by_candidate={second_candidate: second_replay},
        )
        assert second_result["results"][second_candidate]["status"] == "admitted"
        assert (
            runtime.model.architecture.rollback_structural_candidate_batch(
                terminal_batch.batch_id,
                second_candidate,
            )["status"]
            == "rolled_back"
        )
        assert (
            runtime.model.architecture.rollback_structural_candidate_batch(
                terminal_batch.batch_id,
                first_candidate,
            )["status"]
            == "rolled_back"
        )
        before_maintenance = runtime.model.architecture.native_checkpoint()
        active_payload = next(
            item.to_payload()
            for item in runtime.model.architecture.structural_candidate_batches
            if item.batch_id == active_batch_id
        )
        budget_before = runtime.model.architecture.cognitive_snapshot().development.structural_budget
        _save_native_checkpoint(runtime.model.architecture, before_maintenance_path)
        restored = _load_native_checkpoint(before_maintenance_path)
        maintenance = restored.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=restored.structural_lineage_retention_policy.to_payload(),
        )
        retention = restored.structural_lineage_retention_result
        assert maintenance == ()
        assert retention is not None
        assert retention.status == "compacted"
        assert terminal_batch.batch_id in retention.removed_batch_ids
        assert active_batch_id in retention.protected_batch_ids
        assert next(
            item.to_payload()
            for item in restored.structural_candidate_batches
            if item.batch_id == active_batch_id
        ) == active_payload
        assert restored.cognitive_snapshot().development.structural_budget == budget_before
        assert not (
            {first_artifact.artifact_digest, second_artifact.artifact_digest}
            & {item.artifact_digest for item in restored.structural_validation_artifacts}
        )

        before_replay = _checkpoint_digest(restored.native_checkpoint())
        try:
            restored.continue_structural_candidate_batch_from_validation_artifacts(
                terminal_batch.batch_id,
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
            raise AssertionError("terminal batch artifact unexpectedly crossed retention boundary")
        assert _checkpoint_digest(restored.native_checkpoint()) == before_replay
        _save_native_checkpoint(restored, after_maintenance_path)
        final = _load_native_checkpoint(after_maintenance_path)
        assert active_batch_id in {item.batch_id for item in final.structural_candidate_batches}
        assert terminal_batch.batch_id not in {item.batch_id for item in final.structural_candidate_batches}
        assert _checkpoint_digest(before_maintenance) != _checkpoint_digest(restored.native_checkpoint())
    finally:
        before_maintenance_path.unlink(missing_ok=True)
        after_maintenance_path.unlink(missing_ok=True)

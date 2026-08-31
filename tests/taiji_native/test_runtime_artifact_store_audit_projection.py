from __future__ import annotations

import os
from pathlib import Path

import pytest

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_runtime_structural_artifact_multi_round import (
    _batch,
    _record_round,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from taiji import (
    STRUCTURAL_ARTIFACT_STORE_PROJECTION_FORMAT,
    ArtifactConsumptionPolicy,
    StructuralLineageRetentionPolicy,
    StructuralValidationArtifactStore,
)
from taiji.adapter import _checkpoint_digest


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def test_runtime_projects_external_store_audit_without_mutation() -> None:
    runtime = _build_runtime()
    _record_round(runtime, first_ordinal=1, round_id="active-round")
    active_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert active_schedule.get("status") == "batch_created"
    active_batch_id = str(active_schedule["batch_id"])

    terminal_evidence = _record_round(runtime, first_ordinal=7, round_id="terminal-round")
    terminal_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert terminal_schedule.get("status") == "batch_created"
    terminal_batch_id = str(terminal_schedule["batch_id"])
    first_id, second_id = _batch(runtime, terminal_batch_id).selected_candidate_ids

    store_root = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "manual-r5-canary"
        / f"s47-store-{os.getpid()}"
    )
    before_retention_path = store_root.parent / f"s47-before-retention-{os.getpid()}.pt"
    after_retention_path = store_root.parent / f"s47-after-retention-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
    legacy_policy = ArtifactConsumptionPolicy.legacy_compatible(
        reason="historical-s47-audit-test"
    )
    try:
        first_artifact, first_replay, _ = _build_artifact(
            runtime.model.architecture,
            first_id,
            terminal_evidence,
        )
        store.put(first_artifact)
        first_result = runtime.continue_structural_candidate_batch_from_artifact_store(
            terminal_batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={first_id: first_artifact.artifact_digest},
            replays_by_candidate={first_id: first_replay},
            artifact_consumption_policy=legacy_policy,
        )
        second_artifact, second_replay, _ = _build_artifact(
            runtime.model.architecture,
            second_id,
            terminal_evidence,
        )
        store.put(second_artifact)
        second_result = runtime.continue_structural_candidate_batch_from_artifact_store(
            terminal_batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={second_id: second_artifact.artifact_digest},
            replays_by_candidate={second_id: second_replay},
            artifact_consumption_policy=legacy_policy,
        )
        assert first_result["results"][first_id]["status"] == "admitted"
        assert second_result["results"][second_id]["status"] == "admitted"
        assert runtime.rollback_structural_candidate_batch(terminal_batch_id, second_id)[
            "status"
        ] == "rolled_back"
        assert runtime.rollback_structural_candidate_batch(terminal_batch_id, first_id)[
            "status"
        ] == "rolled_back"

        before_projection_checkpoint = _checkpoint_digest(
            runtime.model.architecture.native_checkpoint()
        )
        before_artifact_count = len(runtime.model.architecture.structural_validation_artifacts)
        before_batch_count = len(runtime.model.architecture.structural_validation_artifact_batches)
        before_store_bytes = {
            artifact.artifact_digest: store.path_for(artifact.artifact_digest).read_bytes()
            for artifact in (first_artifact, second_artifact)
        }
        projection = runtime.project_structural_artifact_store_audit(artifact_store=store)
        repeated_projection = runtime.project_structural_artifact_store_audit(
            artifact_store=store
        )
        assert projection == repeated_projection
        assert projection["format"] == STRUCTURAL_ARTIFACT_STORE_PROJECTION_FORMAT
        assert projection["audit_digest"]
        assert {
            item["runtime_visibility"] for item in projection["entries"]
        } == {"runtime_recorded"}
        assert all(item["runtime_batch_ids"] for item in projection["entries"])
        assert _checkpoint_digest(
            runtime.model.architecture.native_checkpoint()
        ) == before_projection_checkpoint
        assert len(runtime.model.architecture.structural_validation_artifacts) == before_artifact_count
        assert len(runtime.model.architecture.structural_validation_artifact_batches) == before_batch_count
        assert {
            artifact.artifact_digest: store.path_for(artifact.artifact_digest).read_bytes()
            for artifact in (first_artifact, second_artifact)
        } == before_store_bytes

        runtime.save(before_retention_path)
        restored = SeedRuntime.load(before_retention_path)
        restored_projection = restored.project_structural_artifact_store_audit(
            artifact_store=StructuralValidationArtifactStore(store_root)
        )
        assert restored_projection == projection

        policy = StructuralLineageRetentionPolicy.create(1, revision=2)
        maintenance = restored.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=policy.to_payload(),
        )
        restored.save(after_retention_path)
        after_retention = SeedRuntime.load(after_retention_path)
        before_tampered_query = _checkpoint_digest(
            after_retention.model.architecture.native_checkpoint()
        )
        after_projection = after_retention.project_structural_artifact_store_audit(
            artifact_store=store
        )
        assert {
            item["runtime_visibility"] for item in after_projection["entries"]
        } == {"external_orphan"}
        assert after_projection["audit_digest"] != projection["audit_digest"]
        assert terminal_batch_id in after_retention.model.architecture.structural_lineage_retention_result.removed_batch_ids
        assert active_batch_id in {
            item.batch_id
            for item in after_retention.model.architecture.structural_candidate_batches
        }
        assert maintenance["maintenance_results"] == []

        first_path = store.path_for(first_artifact.artifact_digest)
        original_bytes = first_path.read_bytes()
        first_path.write_bytes(b"{}")
        with pytest.raises(ValueError):
            after_retention.project_structural_artifact_store_audit(artifact_store=store)
        first_path.write_bytes(original_bytes)
        assert _checkpoint_digest(
            after_retention.model.architecture.native_checkpoint()
        ) == before_tampered_query
        assert {
            artifact.artifact_digest: store.path_for(artifact.artifact_digest).read_bytes()
            for artifact in (first_artifact, second_artifact)
        } == before_store_bytes
    finally:
        before_retention_path.unlink(missing_ok=True)
        after_retention_path.unlink(missing_ok=True)
        _remove_directory(store_root)

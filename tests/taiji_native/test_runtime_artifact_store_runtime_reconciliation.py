from __future__ import annotations

import os
from pathlib import Path

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
    ArtifactConsumptionPolicy,
    StructuralLineageRetentionPolicy,
    StructuralValidationArtifactStore,
)


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def test_runtime_store_reconciliation_distinguishes_missing_and_orphan() -> None:
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
        / f"s48-store-{os.getpid()}"
    )
    before_retention_path = store_root.parent / f"s48-before-retention-{os.getpid()}.pt"
    after_retention_path = store_root.parent / f"s48-after-retention-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
    legacy_policy = ArtifactConsumptionPolicy.legacy_compatible(
        reason="historical-s48-reconciliation-test"
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
        second_result = runtime.continue_structural_candidate_batch_from_validation_artifacts(
            terminal_batch_id,
            artifacts_by_candidate={second_id: second_artifact.to_payload()},
            replays_by_candidate={second_id: second_replay},
        )
        assert first_result["results"][first_id]["status"] == "admitted"
        assert second_result["results"][second_id]["status"] == "admitted"
        assert runtime.rollback_structural_candidate_batch(terminal_batch_id, second_id)[
            "status"
        ] == "rolled_back"
        assert runtime.rollback_structural_candidate_batch(terminal_batch_id, first_id)[
            "status"
        ] == "rolled_back"

        projection = runtime.project_structural_artifact_store_audit(artifact_store=store)
        repeated = runtime.project_structural_artifact_store_audit(artifact_store=store)
        assert projection == repeated
        assert projection["audit_digest"]
        assert projection["runtime_artifact_digests"] == sorted(
            (first_artifact.artifact_digest, second_artifact.artifact_digest)
        )
        assert projection["runtime_batch_artifact_digests"] == sorted(
            (first_artifact.artifact_digest, second_artifact.artifact_digest)
        )
        assert projection["missing_runtime_artifact_digests"] == [
            second_artifact.artifact_digest
        ]
        assert projection["missing_runtime_batch_artifact_digests"] == [
            second_artifact.artifact_digest
        ]
        assert {
            item["runtime_visibility"] for item in projection["entries"]
        } == {"runtime_recorded"}

        runtime.save(before_retention_path)
        restored = SeedRuntime.load(before_retention_path)
        assert restored.project_structural_artifact_store_audit(
            artifact_store=StructuralValidationArtifactStore(store_root)
        ) == projection

        policy = StructuralLineageRetentionPolicy.create(1, revision=2)
        restored.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=policy.to_payload(),
        )
        restored.save(after_retention_path)
        after_retention = SeedRuntime.load(after_retention_path)
        orphan_projection = after_retention.project_structural_artifact_store_audit(
            artifact_store=store
        )
        assert orphan_projection == after_retention.project_structural_artifact_store_audit(
            artifact_store=store
        )
        assert orphan_projection["runtime_artifact_digests"] == []
        assert orphan_projection["runtime_batch_artifact_digests"] == []
        assert orphan_projection["missing_runtime_artifact_digests"] == []
        assert orphan_projection["missing_runtime_batch_artifact_digests"] == []
        assert {
            item["runtime_visibility"] for item in orphan_projection["entries"]
        } == {"external_orphan"}
        assert second_artifact.artifact_digest not in {
            item["artifact_digest"] for item in orphan_projection["entries"]
        }
        assert first_artifact.artifact_digest in {
            item["artifact_digest"] for item in orphan_projection["entries"]
        }
        assert terminal_batch_id in after_retention.model.architecture.structural_lineage_retention_result.removed_batch_ids
        assert active_batch_id in {
            item.batch_id
            for item in after_retention.model.architecture.structural_candidate_batches
        }
    finally:
        before_retention_path.unlink(missing_ok=True)
        after_retention_path.unlink(missing_ok=True)
        _remove_directory(store_root)

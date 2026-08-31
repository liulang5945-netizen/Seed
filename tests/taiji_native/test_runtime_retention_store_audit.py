from __future__ import annotations

import json
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
from taiji import StructuralLineageRetentionPolicy, StructuralValidationArtifactStore
from taiji.adapter import _checkpoint_digest


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def test_store_audit_is_read_only_and_reports_runtime_orphans() -> None:
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
    terminal_batch = _batch(runtime, terminal_batch_id)
    first_id, second_id = terminal_batch.selected_candidate_ids

    store_root = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "manual-r5-canary"
        / f"s46-store-{os.getpid()}"
    )
    checkpoint_path = store_root.parent / f"s46-runtime-{os.getpid()}.pt"
    before_retention_path = store_root.parent / f"s46-before-retention-{os.getpid()}.pt"
    after_retention_path = store_root.parent / f"s46-after-retention-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
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
        )
        assert first_result["results"][first_id]["status"] == "admitted"
        assert second_result["results"][second_id]["status"] == "admitted"
        assert runtime.rollback_structural_candidate_batch(terminal_batch_id, second_id)[
            "status"
        ] == "rolled_back"
        assert runtime.rollback_structural_candidate_batch(terminal_batch_id, first_id)[
            "status"
        ] == "rolled_back"

        runtime.save(before_retention_path)
        restored = SeedRuntime.load(before_retention_path)
        retention = StructuralLineageRetentionPolicy.create(1, revision=2)
        maintenance = restored.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=retention.to_payload(),
        )
        assert maintenance["maintenance_results"] == []
        restored.save(after_retention_path)
        after_retention = SeedRuntime.load(after_retention_path)
        before_audit_digest = _checkpoint_digest(
            after_retention.model.architecture.native_checkpoint()
        )

        healthy_inventory = store.inventory()
        assert store.audit() == healthy_inventory
        assert tuple(item["artifact_digest"] for item in healthy_inventory) == tuple(
            sorted((first_artifact.artifact_digest, second_artifact.artifact_digest))
        )
        records_by_digest = {
            item["artifact_digest"]: item for item in healthy_inventory
        }
        for artifact in (first_artifact, second_artifact):
            record = records_by_digest[artifact.artifact_digest]
            assert record["measurement_digest"] == artifact.measurement_digest
            assert record["candidate_id"] == artifact.candidate_id
            assert record["resource_cost"] == artifact.resource_cost
            assert store.load(artifact.artifact_digest) == artifact

        assert terminal_batch_id in after_retention.model.architecture.structural_lineage_retention_result.removed_batch_ids
        assert active_batch_id in {
            item.batch_id
            for item in after_retention.model.architecture.structural_candidate_batches
        }
        try:
            after_retention.continue_structural_candidate_batch_from_artifact_store(
                terminal_batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={
                    second_id: second_artifact.artifact_digest
                },
                replays_by_candidate={second_id: second_replay},
            )
        except ValueError as exc:
            assert "unknown structural candidate batch" in str(exc)
        else:
            raise AssertionError("runtime orphan unexpectedly resurrected a deleted batch")
        assert _checkpoint_digest(
            after_retention.model.architecture.native_checkpoint()
        ) == before_audit_digest

        first_path = store.path_for(first_artifact.artifact_digest)
        original_bytes = first_path.read_bytes()
        tampered_payload = json.loads(original_bytes.decode("utf-8"))
        tampered_payload["measurement_digest"] = "0" * 64
        first_path.write_text(
            json.dumps(
                tampered_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            store.audit()
        assert first_path.exists()
        first_path.write_bytes(original_bytes)

        invalid_path = store_root / "invalid-name.json"
        invalid_path.write_bytes(original_bytes)
        with pytest.raises(ValueError, match="unexpected file|SHA-256"):
            store.inventory()
        assert invalid_path.exists()
        invalid_path.unlink()

        assert store.inventory() == healthy_inventory
        assert _checkpoint_digest(
            after_retention.model.architecture.native_checkpoint()
        ) == before_audit_digest
    finally:
        checkpoint_path.unlink(missing_ok=True)
        before_retention_path.unlink(missing_ok=True)
        after_retention_path.unlink(missing_ok=True)
        _remove_directory(store_root)

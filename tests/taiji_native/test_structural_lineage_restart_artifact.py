from __future__ import annotations

import copy
import json
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


def test_restart_replay_bound_artifact_continues_and_rejects_tampering() -> None:
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
    before_artifact_path = checkpoint_root / f"s32-before-artifact-{suffix}.pt"
    first_artifact_path = checkpoint_root / f"s32-first-artifact-{suffix}.json"
    after_first_path = checkpoint_root / f"s32-after-first-{suffix}.pt"
    second_artifact_path = checkpoint_root / f"s32-second-artifact-{suffix}.json"
    final_path = checkpoint_root / f"s32-final-{suffix}.pt"
    try:
        _save_native_checkpoint(runtime.model.architecture, before_artifact_path)
        restored = _load_native_checkpoint(before_artifact_path)
        assert _checkpoint_digest(restored.native_checkpoint()) == _checkpoint_digest(
            runtime.model.architecture.native_checkpoint()
        )
        first_artifact, first_replay, first_measurements = _build_artifact(
            restored,
            first_candidate,
            continuation_evidence,
        )
        assert _checkpoint_digest(restored.native_checkpoint()) == (
            first_artifact.parent_checkpoint_digest
        )
        _save_native_checkpoint(restored, before_artifact_path)
        first_artifact_path.write_text(
            json.dumps(first_artifact.to_payload(), sort_keys=True),
            encoding="utf-8",
        )
        restored_artifact = json.loads(first_artifact_path.read_text(encoding="utf-8"))
        assert restored_artifact["measurement_digest"] == first_measurements.measurement_digest

        tampered_payload = copy.deepcopy(first_artifact.to_payload())
        tampered_payload["measurement_digest"] = "0" * 64
        tamper_branch = _load_native_checkpoint(before_artifact_path)
        before_tamper = _checkpoint_digest(tamper_branch.native_checkpoint())
        tampered = tamper_branch.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={first_candidate: tampered_payload},
            replays_by_candidate={first_candidate: first_replay},
        )
        after_tamper = _checkpoint_digest(tamper_branch.native_checkpoint())
        assert tampered["results"][first_candidate]["status"] == "failed_closed"
        assert before_tamper != after_tamper
        assert (
            tamper_branch.cognitive_snapshot().development.structural_budget
            == restored.cognitive_snapshot().development.structural_budget
        )
        assert tuple(
            region.unit_ids for region in tamper_branch.neuron_regions
        ) == tuple(region.unit_ids for region in restored.neuron_regions)

        artifact_restored = _load_native_checkpoint(before_artifact_path)
        assert _checkpoint_digest(artifact_restored.native_checkpoint()) == (
            first_artifact.parent_checkpoint_digest
        )
        first_result = artifact_restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={first_candidate: restored_artifact},
            replays_by_candidate={first_candidate: first_replay},
        )
        assert first_result["results"][first_candidate]["status"] == "admitted"
        _save_native_checkpoint(artifact_restored, after_first_path)
        first_resumed = _load_native_checkpoint(after_first_path)
        persisted_first_artifact = next(
            item
            for item in first_resumed.structural_validation_artifacts
            if item.artifact_digest == first_artifact.artifact_digest
        )
        assert persisted_first_artifact.measurement_digest == first_measurements.measurement_digest

        second_artifact, second_replay, second_measurements = _build_artifact(
            first_resumed,
            second_candidate,
            continuation_evidence,
        )
        second_artifact_path.write_text(
            json.dumps(second_artifact.to_payload(), sort_keys=True),
            encoding="utf-8",
        )
        second_artifact_payload = json.loads(
            second_artifact_path.read_text(encoding="utf-8")
        )
        second_result = first_resumed.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact_payload},
            replays_by_candidate={second_candidate: second_replay},
        )
        assert second_result["results"][second_candidate]["status"] == "admitted"
        _save_native_checkpoint(first_resumed, final_path)
        final = _load_native_checkpoint(final_path)
        repeated = final.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={
                first_candidate: restored_artifact,
                second_candidate: second_artifact,
            },
            replays_by_candidate={
                first_candidate: first_replay,
                second_candidate: second_replay,
            },
        )
        assert repeated["results"][first_candidate]["status"] == "already_applied"
        assert repeated["results"][second_candidate]["status"] == "already_applied"
        assert repeated["artifact_batch"]["complete"] is True
        assert second_measurements.measurement_digest == second_artifact.measurement_digest
    finally:
        before_artifact_path.unlink(missing_ok=True)
        first_artifact_path.unlink(missing_ok=True)
        after_first_path.unlink(missing_ok=True)
        second_artifact_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)

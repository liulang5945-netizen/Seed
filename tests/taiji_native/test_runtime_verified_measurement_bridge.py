from __future__ import annotations

import os
from pathlib import Path

import pytest

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_runtime_structural_artifact_multi_round import _record_round
from scripts.training.eval_taiji_workbench_measured_artifact_batch import _build_artifact
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from taiji import ArtifactConsumptionPolicy, StructuralValidationArtifactStore
from taiji.adapter import _checkpoint_digest


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def _prepare_runtime() -> tuple[SeedRuntime, str, tuple[dict[str, object], ...]]:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="verified-bridge")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"verified bridge batch was not created: {schedule}")
    return runtime, str(schedule["batch_id"]), evidence


def test_verified_measurement_bridge_is_opt_in_and_all_or_nothing() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "manual-r5-canary"
        / f"s51-store-{os.getpid()}"
    )
    verified_checkpoint = root.parent / f"s51-verified-{os.getpid()}.pt"
    legacy_checkpoint = root.parent / f"s51-legacy-{os.getpid()}.pt"
    try:
        verified_runtime, verified_batch_id, verified_evidence = _prepare_runtime()
        verified_candidate_id = verified_runtime.model.architecture.structural_candidate_batches[-1].selected_candidate_ids[0]
        verified_artifact, verified_replay, verified_measurements = _build_artifact(
            verified_runtime.model.architecture,
            verified_candidate_id,
            verified_evidence,
        )
        verified_store = StructuralValidationArtifactStore(root / "verified")
        verified_store.put_measured_artifact(verified_artifact, verified_measurements)
        verified_runtime.save(verified_checkpoint)
        verified_restored = SeedRuntime.load(verified_checkpoint)
        verified_result = verified_restored.continue_structural_candidate_batch_from_artifact_store(
            verified_batch_id,
            artifact_store=verified_store,
            artifact_digests_by_candidate={
                verified_candidate_id: verified_artifact.artifact_digest
            },
            replays_by_candidate={verified_candidate_id: verified_replay},
            require_verified_measurements=True,
        )
        assert verified_result["results"][verified_candidate_id]["status"] == "admitted"

        legacy_runtime, legacy_batch_id, legacy_evidence = _prepare_runtime()
        legacy_candidate_id = legacy_runtime.model.architecture.structural_candidate_batches[-1].selected_candidate_ids[0]
        legacy_artifact, legacy_replay, _ = _build_artifact(
            legacy_runtime.model.architecture,
            legacy_candidate_id,
            legacy_evidence,
        )
        legacy_store = StructuralValidationArtifactStore(root / "legacy")
        legacy_store.put(legacy_artifact)
        legacy_runtime.save(legacy_checkpoint)
        strict_legacy = SeedRuntime.load(legacy_checkpoint)
        before_strict_legacy = _checkpoint_digest(
            strict_legacy.model.architecture.native_checkpoint()
        )
        with pytest.raises(FileNotFoundError):
            strict_legacy.continue_structural_candidate_batch_from_artifact_store(
                legacy_batch_id,
                artifact_store=legacy_store,
                artifact_digests_by_candidate={
                    legacy_candidate_id: legacy_artifact.artifact_digest
                },
                replays_by_candidate={legacy_candidate_id: legacy_replay},
                require_verified_measurements=True,
            )
        assert _checkpoint_digest(
            strict_legacy.model.architecture.native_checkpoint()
        ) == before_strict_legacy
        default_legacy = SeedRuntime.load(legacy_checkpoint)
        default_result = default_legacy.continue_structural_candidate_batch_from_artifact_store(
            legacy_batch_id,
            artifact_store=legacy_store,
            artifact_digests_by_candidate={legacy_candidate_id: legacy_artifact.artifact_digest},
            replays_by_candidate={legacy_candidate_id: legacy_replay},
            artifact_consumption_policy=ArtifactConsumptionPolicy.legacy_compatible(
                reason="historical-replay-test"
            ),
        )
        assert default_result["results"][legacy_candidate_id]["status"] == "admitted"

        partial_runtime, partial_batch_id, partial_evidence = _prepare_runtime()
        first_id, second_id = partial_runtime.model.architecture.structural_candidate_batches[-1].selected_candidate_ids
        first_artifact, first_replay, first_measurements = _build_artifact(
            partial_runtime.model.architecture,
            first_id,
            partial_evidence,
        )
        second_artifact, second_replay, _ = _build_artifact(
            partial_runtime.model.architecture,
            second_id,
            partial_evidence,
        )
        partial_store = StructuralValidationArtifactStore(root / "partial")
        partial_store.put_measured_artifact(first_artifact, first_measurements)
        partial_store.put(second_artifact)
        before_partial = _checkpoint_digest(
            partial_runtime.model.architecture.native_checkpoint()
        )
        before_partial_budget = partial_runtime.model.architecture.cognitive_snapshot().development.structural_budget
        with pytest.raises(FileNotFoundError):
            partial_runtime.continue_structural_candidate_batch_from_artifact_store(
                partial_batch_id,
                artifact_store=partial_store,
                artifact_digests_by_candidate={
                    first_id: first_artifact.artifact_digest,
                    second_id: second_artifact.artifact_digest,
                },
                replays_by_candidate={
                    first_id: first_replay,
                    second_id: second_replay,
                },
                require_verified_measurements=True,
            )
        assert _checkpoint_digest(
            partial_runtime.model.architecture.native_checkpoint()
        ) == before_partial
        assert partial_runtime.model.architecture.cognitive_snapshot().development.structural_budget == before_partial_budget
    finally:
        verified_checkpoint.unlink(missing_ok=True)
        legacy_checkpoint.unlink(missing_ok=True)
        _remove_directory(root / "verified")
        _remove_directory(root / "legacy")
        _remove_directory(root / "partial")
        _remove_directory(root)

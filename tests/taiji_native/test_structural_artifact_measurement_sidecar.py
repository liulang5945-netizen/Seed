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
from taiji import StructuralValidationArtifactStore
from taiji.adapter import _checkpoint_digest


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def test_measured_artifact_sidecar_is_verified_and_legacy_is_explicit() -> None:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="sidecar-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_id = batch.selected_candidate_ids[0]
    first_artifact, first_replay, first_measurements = _build_artifact(
        runtime.model.architecture,
        first_id,
        evidence,
    )
    store_root = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "manual-r5-canary"
        / f"s49-store-{os.getpid()}"
    )
    legacy_root = store_root.parent / f"s49-legacy-{os.getpid()}"
    checkpoint_path = store_root.parent / f"s49-runtime-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
    legacy_store = StructuralValidationArtifactStore(legacy_root)
    try:
        stored = store.put_measured_artifact(first_artifact, first_measurements)
        assert stored == first_artifact
        assert store.put_measured_artifact(
            first_artifact.to_payload(), first_measurements.to_payload()
        ) == first_artifact
        assert store.contains(first_artifact.artifact_digest)
        assert store.contains_measurement(first_measurements.measurement_digest)
        assert store.load(first_artifact.artifact_digest) == first_artifact
        assert store.load_measurements(first_measurements.measurement_digest) == first_measurements
        inventory = store.inventory()
        assert len(inventory) == 1
        assert inventory[0]["measurement_status"] == "verified"
        assert inventory[0]["measurement_digest"] == first_measurements.measurement_digest

        with pytest.raises(ValueError):
            store.put_measured_artifact(
                first_artifact,
                {**first_measurements.to_payload(), "measurement_digest": "0" * 64},
            )

        measurement_path = store.measurement_path_for(first_measurements.measurement_digest)
        original_measurement_bytes = measurement_path.read_bytes()
        measurement_path.write_bytes(b"{}")
        with pytest.raises(ValueError):
            store.load_measurements(first_measurements.measurement_digest)
        with pytest.raises(ValueError):
            store.inventory()
        assert measurement_path.exists()
        measurement_path.write_bytes(original_measurement_bytes)
        assert store.inventory() == inventory

        legacy_store.put(first_artifact)
        legacy_inventory = legacy_store.inventory()
        assert legacy_inventory[0]["measurement_status"] == "legacy_unverified"
        assert legacy_store.load(first_artifact.artifact_digest) == first_artifact

        runtime.save(checkpoint_path)
        restored = SeedRuntime.load(checkpoint_path)
        before_checkpoint = _checkpoint_digest(
            restored.model.architecture.native_checkpoint()
        )
        result = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={first_id: first_artifact.artifact_digest},
            replays_by_candidate={first_id: first_replay},
        )
        assert result["results"][first_id]["status"] == "admitted"
        assert _checkpoint_digest(
            restored.model.architecture.native_checkpoint()
        ) != before_checkpoint
    finally:
        checkpoint_path.unlink(missing_ok=True)
        _remove_directory(store_root)
        _remove_directory(legacy_root)

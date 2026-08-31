from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

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


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_partial_measurement_bundle_fails_closed_then_recovers_explicitly() -> None:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="bundle-recovery")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, _, measurements = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        evidence,
    )
    before_runtime = _checkpoint_digest(runtime.model.architecture.native_checkpoint())

    root = (
        Path(__file__).resolve().parents[2]
        / "output"
        / "manual-r5-canary"
        / f"s50-store-{os.getpid()}"
    )
    legacy_root = root.parent / f"s50-legacy-{os.getpid()}"
    try:
        partial_store = StructuralValidationArtifactStore(root)
        partial_store.root.mkdir(parents=True, exist_ok=True)
        sidecar_path = partial_store.measurement_path_for(measurements.measurement_digest)
        sidecar_bytes = _canonical_bytes(measurements.to_payload())
        sidecar_path.write_bytes(sidecar_bytes)
        with pytest.raises(ValueError, match="unreferenced measurement sidecar"):
            partial_store.inventory()
        assert sidecar_path.exists()

        partial_store.put_measured_artifact(artifact, measurements)
        recovered_inventory = partial_store.inventory()
        assert recovered_inventory[0]["measurement_status"] == "verified"
        assert sidecar_path.read_bytes() == sidecar_bytes
        assert partial_store.put_measured_artifact(artifact, measurements) == artifact
        assert partial_store.inventory() == recovered_inventory

        legacy_store = StructuralValidationArtifactStore(legacy_root)
        legacy_store.put(artifact)
        legacy_artifact_bytes = legacy_store.path_for(artifact.artifact_digest).read_bytes()
        assert legacy_store.inventory()[0]["measurement_status"] == "legacy_unverified"
        legacy_store.put_measured_artifact(artifact, measurements)
        assert legacy_store.inventory()[0]["measurement_status"] == "verified"
        assert legacy_store.path_for(artifact.artifact_digest).read_bytes() == legacy_artifact_bytes

        recovered_sidecar_bytes = sidecar_path.read_bytes()
        sidecar_path.write_bytes(b"{}")
        with pytest.raises(ValueError):
            partial_store.put_measured_artifact(artifact, measurements)
        assert sidecar_path.read_bytes() == b"{}"
        sidecar_path.write_bytes(recovered_sidecar_bytes)
        assert partial_store.inventory() == recovered_inventory
        assert _checkpoint_digest(runtime.model.architecture.native_checkpoint()) == before_runtime
    finally:
        _remove_directory(root)
        _remove_directory(legacy_root)

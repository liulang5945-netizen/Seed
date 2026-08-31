"""Run the R5C-S50 measurement-bundle partial-write recovery canary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_runtime_structural_artifact_multi_round import (  # noqa: E402
    _record_round,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from taiji import StructuralValidationArtifactStore  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s50-structural-artifact-measurement-bundle-recovery-v1"


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


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="bundle-recovery")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"bundle recovery batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, _, measurements = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        evidence,
    )
    before_runtime = _checkpoint_digest(runtime.model.architecture.native_checkpoint())

    root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s50-store-{os.getpid()}"
    legacy_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s50-legacy-{os.getpid()}"
    try:
        partial_store = StructuralValidationArtifactStore(root)
        partial_store.root.mkdir(parents=True, exist_ok=True)
        sidecar_path = partial_store.measurement_path_for(measurements.measurement_digest)
        sidecar_bytes = _canonical_bytes(measurements.to_payload())
        sidecar_path.write_bytes(sidecar_bytes)
        try:
            partial_store.inventory()
        except ValueError as exc:
            sidecar_only_rejected = "unreferenced measurement sidecar" in str(exc)
        else:
            sidecar_only_rejected = False
        sidecar_preserved = sidecar_path.exists()

        partial_store.put_measured_artifact(artifact, measurements)
        recovered_inventory = partial_store.inventory()
        explicit_recovery = (
            recovered_inventory[0]["measurement_status"] == "verified"
            and sidecar_path.read_bytes() == sidecar_bytes
            and partial_store.put_measured_artifact(artifact, measurements) == artifact
            and partial_store.inventory() == recovered_inventory
        )

        legacy_store = StructuralValidationArtifactStore(legacy_root)
        legacy_store.put(artifact)
        legacy_before = legacy_store.inventory()[0]["measurement_status"]
        artifact_bytes = legacy_store.path_for(artifact.artifact_digest).read_bytes()
        legacy_store.put_measured_artifact(artifact, measurements)
        artifact_only_recovered = (
            legacy_before == "legacy_unverified"
            and legacy_store.inventory()[0]["measurement_status"] == "verified"
            and legacy_store.path_for(artifact.artifact_digest).read_bytes() == artifact_bytes
        )

        recovered_sidecar_bytes = sidecar_path.read_bytes()
        sidecar_path.write_bytes(b"{}")
        try:
            partial_store.put_measured_artifact(artifact, measurements)
        except ValueError:
            conflict_rejected = True
        else:
            conflict_rejected = False
        conflict_preserved = sidecar_path.read_bytes() == b"{}"
        sidecar_path.write_bytes(recovered_sidecar_bytes)

        runtime_unchanged = (
            _checkpoint_digest(runtime.model.architecture.native_checkpoint())
            == before_runtime
        )
        metrics = {
            "sidecar_only_fails_closed_without_deletion": (
                sidecar_only_rejected and sidecar_preserved
            ),
            "sidecar_only_recovers_by_explicit_retry": explicit_recovery,
            "artifact_only_legacy_upgrades_explicitly": artifact_only_recovered,
            "conflicting_retry_does_not_overwrite": (
                conflict_rejected and conflict_preserved
            ),
            "recovery_is_runtime_read_only": runtime_unchanged,
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "artifact_digest": artifact.artifact_digest,
            "measurement_digest": measurements.measurement_digest,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "partial measured bundles must fail closed without cleanup and recover only "
                    "through an explicit idempotent complete-bundle retry"
                ),
            },
            "boundary": (
                "This canary covers native CPU partial measurement-bundle recovery. It does not "
                "claim automatic repair, deletion, unlimited storage, open-domain quality, CUDA, "
                "frontend behavior, Windows shell, CI completion, or general intelligence."
            ),
        }
    finally:
        _remove_directory(root)
        _remove_directory(legacy_root)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s50_structural_artifact_measurement_bundle_recovery_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

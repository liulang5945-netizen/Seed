"""Run the R5C-S49 measured-artifact measurement-sidecar canary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
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

REPORT_FORMAT = "taiji-w7-r5c-s49-structural-artifact-measurement-sidecar-v1"


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="sidecar-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"sidecar batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_id = batch.selected_candidate_ids[0]
    first_artifact, first_replay, first_measurements = _build_artifact(
        runtime.model.architecture,
        first_id,
        evidence,
    )

    store_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s49-store-{os.getpid()}"
    legacy_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s49-legacy-{os.getpid()}"
    checkpoint_path = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s49-runtime-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
    legacy_store = StructuralValidationArtifactStore(legacy_root)
    try:
        store.put_measured_artifact(first_artifact, first_measurements)
        inventory = store.inventory()
        verified = (
            len(inventory) == 1
            and inventory[0]["measurement_status"] == "verified"
            and store.load_measurements(first_measurements.measurement_digest)
            == first_measurements
        )

        legacy_store.put(first_artifact)
        legacy_explicit = legacy_store.inventory()[0]["measurement_status"] == "legacy_unverified"
        before_binding_files = tuple(sorted(item.name for item in store_root.iterdir()))
        try:
            store.put_measured_artifact(first_artifact, {**first_measurements.to_payload(), "measurement_digest": "0" * 64})
        except ValueError:
            binding_rejected = True
        else:
            binding_rejected = False
        binding_is_non_mutating = tuple(sorted(item.name for item in store_root.iterdir())) == before_binding_files

        measurement_path = store.measurement_path_for(first_measurements.measurement_digest)
        original_measurement_bytes = measurement_path.read_bytes()
        measurement_path.write_bytes(b"{}")
        try:
            store.inventory()
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        tamper_is_non_destructive = measurement_path.exists()
        measurement_path.write_bytes(original_measurement_bytes)

        runtime.save(checkpoint_path)
        restored = SeedRuntime.load(checkpoint_path)
        before_runtime = _checkpoint_digest(
            restored.model.architecture.native_checkpoint()
        )
        result = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={first_id: first_artifact.artifact_digest},
            replays_by_candidate={first_id: first_replay},
        )
        runtime_consumption_is_explicit = (
            result["results"][first_id]["status"] == "admitted"
            and _checkpoint_digest(restored.model.architecture.native_checkpoint())
            != before_runtime
        )

        metrics = {
            "new_bundle_is_independently_verified": verified,
            "legacy_artifact_is_explicitly_unverified": legacy_explicit,
            "artifact_measurement_binding_fails_closed": (
                binding_rejected and binding_is_non_mutating
            ),
            "tampered_measurement_sidecar_fails_closed_without_deletion": (
                tamper_rejected and tamper_is_non_destructive
            ),
            "runtime_consumption_remains_explicit": runtime_consumption_is_explicit,
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "artifact_digest": first_artifact.artifact_digest,
            "measurement_digest": first_measurements.measurement_digest,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "new measured bundles must carry independently verifiable canonical measurement "
                    "facts while legacy artifact-only files remain explicit and runtime consumption "
                    "continues through the existing contract"
                ),
            },
            "boundary": (
                "This canary covers native CPU measurement-fact sidecars. It does not claim automatic "
                "repair, deletion, unlimited storage, open-domain quality, unlimited growth, CUDA, "
                "frontend behavior, Windows shell, CI completion, or general intelligence."
            ),
        }
    finally:
        checkpoint_path.unlink(missing_ok=True)
        _remove_directory(store_root)
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
        / "taiji_w7_r5c_s49_structural_artifact_measurement_sidecar_20260831.json",
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

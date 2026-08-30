"""Run the R5C-S17 validation artifact and measurement integrity canary."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import (  # noqa: E402
    StructuralValidationMeasurements,
    TSKV8Adapter,
)
from taiji.structural_validation_artifact import _digest_value  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s17-workbench-integrity-v1"


def _expect_rejection(factory: object) -> bool:
    try:
        factory()  # type: ignore[operator]
    except (KeyError, TypeError, ValueError):
        return True
    return False


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"integrity canary batch was not created: {schedule}")
    batch_id = str(schedule["batch_id"])
    batch = next(
        item
        for item in runtime.model.architecture.structural_candidate_batches
        if item.batch_id == batch_id
    )
    candidate_id = batch.selected_candidate_ids[0]
    model = TSKV8Adapter.from_native_checkpoint(runtime.model.architecture.native_checkpoint())
    artifact, replay, measurements = _build_artifact(
        model,
        candidate_id,
        executions,
    )

    measurement_payload = measurements.to_payload()
    tampered_measurement_metric = copy.deepcopy(measurement_payload)
    tampered_measurement_metric["holdout_gain"] = min(
        1.0,
        float(tampered_measurement_metric["holdout_gain"]) + 0.01,
    )
    tampered_measurement_digest = copy.deepcopy(measurement_payload)
    tampered_measurement_digest["measurement_digest"] = "0" * 64

    artifact_payload = artifact.to_payload()
    tampered_artifact_measurement = copy.deepcopy(artifact_payload)
    tampered_artifact_measurement["measurement_digest"] = "1" * 64
    legacy_artifact = artifact_payload.copy()
    legacy_artifact.pop("measurement_digest", None)
    legacy_artifact["artifact_digest"] = _digest_value(
        {key: value for key, value in legacy_artifact.items() if key != "artifact_digest"}
    )

    model.record_structural_validation_artifact(artifact)
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_artifact = next(
        item
        for item in restored.structural_validation_artifacts
        if item.artifact_digest == artifact.artifact_digest
    )
    metrics = {
        "measurement_roundtrip_recomputes_digest": (
            StructuralValidationMeasurements.from_payload(measurement_payload)
            == measurements
        ),
        "tampered_measurement_metric_fails_closed": _expect_rejection(
            lambda: StructuralValidationMeasurements.from_payload(tampered_measurement_metric)
        ),
        "tampered_measurement_digest_fails_closed": _expect_rejection(
            lambda: StructuralValidationMeasurements.from_payload(tampered_measurement_digest)
        ),
        "artifact_binds_measurement_digest": (
            artifact.measurement_digest == measurements.measurement_digest
        ),
        "tampered_artifact_measurement_binding_fails_closed": _expect_rejection(
            lambda: type(artifact).from_payload(tampered_artifact_measurement)
        ),
        "legacy_artifact_without_measurement_digest_roundtrips": (
            type(artifact).from_payload(legacy_artifact).measurement_digest == ""
        ),
        "artifact_ledger_checkpoint_roundtrip_preserves_binding": (
            restored_artifact.measurement_digest == measurements.measurement_digest
            and restored_artifact.artifact_digest == artifact.artifact_digest
        ),
        "measurement_inputs_remain_raw_digest_bound": all(
            bool(value)
            for value in (
                measurements.holdout_baseline_digest,
                measurements.holdout_candidate_digest,
                measurements.retention_baseline_digest,
                measurements.retention_candidate_digest,
                measurements.lesion_full_digest,
                measurements.lesion_lesioned_digest,
                measurements.resource_measurement_digest,
            )
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "candidate_id": candidate_id,
        "schedule": schedule,
        "measurement": measurements.to_payload(),
        "artifact": artifact.to_payload(),
        "checkpoint": {
            "artifact_count": len(restored.structural_validation_artifacts),
            "checkpointed_artifact_digest": restored_artifact.artifact_digest,
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "measurement and artifact payloads must reject tampering, preserve raw input "
                "digests, explicitly bind measurement provenance, and remain legacy-checkpoint compatible"
            ),
        },
        "boundary": (
            "This canary closes measurement/artifact integrity and provenance binding; it does "
            "not claim open-domain quality, unlimited growth, CUDA, or CI completion."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s17_workbench_integrity_20260830.json",
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

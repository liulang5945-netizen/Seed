"""Run the R5C-S36 SeedRuntime artifact-batch projection canary."""

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
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s36-runtime-structural-artifact-batch-v1"


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S36 batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, replay, measurements = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        executions,
    )

    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    checkpoint_path = checkpoint_root / f"s36-runtime-artifact-{suffix}.pt"
    artifact_path = checkpoint_root / f"s36-runtime-artifact-{suffix}.json"
    try:
        artifact_path.write_text(
            json.dumps(artifact.to_payload(), sort_keys=True),
            encoding="utf-8",
        )
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        runtime.save(checkpoint_path)
        restored = SeedRuntime.load(checkpoint_path)
        parent_matches = (
            _checkpoint_digest(restored.model.architecture.native_checkpoint())
            == artifact.parent_checkpoint_digest
        )
        before_unknown = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        try:
            restored.continue_structural_candidate_batch_from_validation_artifacts(
                batch.batch_id,
                artifacts_by_candidate={"candidate:foreign:runtime": artifact_payload},
                replays_by_candidate={},
            )
        except ValueError as exc:
            unknown_key_rejected = "outside the selected batch" in str(exc)
        else:
            unknown_key_rejected = False
        unknown_key_atomic = (
            _checkpoint_digest(restored.model.architecture.native_checkpoint()) == before_unknown
        )
        first = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: artifact_payload},
            replays_by_candidate={candidate_id: replay},
        )
        restored.save(checkpoint_path)
        resumed = SeedRuntime.load(checkpoint_path)
        repeated = resumed.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: artifact_payload},
            replays_by_candidate={candidate_id: replay},
        )
        persisted = next(
            item
            for item in resumed.model.architecture.structural_validation_artifacts
            if item.artifact_digest == artifact.artifact_digest
        )
        metrics = {
            "runtime_parent_matches_measured_checkpoint": parent_matches,
            "artifact_and_measurement_payload_roundtrip": (
                artifact_payload["artifact_digest"] == artifact.artifact_digest
                and artifact_payload["measurement_digest"] == measurements.measurement_digest
            ),
            "unknown_key_rejected_atomically": unknown_key_rejected and unknown_key_atomic,
            "runtime_wrapper_admits_measured_artifact": (
                first["results"][candidate_id]["status"] == "admitted"
            ),
            "artifact_provenance_survives_runtime_restart": (
                persisted.artifact_digest == artifact.artifact_digest
                and persisted.measurement_digest == measurements.measurement_digest
            ),
            "repeated_runtime_consumption_is_idempotent": (
                repeated["results"][candidate_id]["status"] == "already_applied"
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_id": candidate_id,
            "artifact_digest": artifact.artifact_digest,
            "measurement_digest": measurements.measurement_digest,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "SeedRuntime must project the native artifact-batch contract without "
                    "recomputing facts, survive checkpoint restart, reject unknown inputs "
                    "atomically, and keep repeated consumption idempotent"
                ),
            },
            "boundary": (
                "This canary covers SeedRuntime/native CPU artifact-batch projection and restart. "
                "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
            ),
        }
    finally:
        checkpoint_path.unlink(missing_ok=True)
        artifact_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s36_runtime_structural_artifact_batch_20260831.json",
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

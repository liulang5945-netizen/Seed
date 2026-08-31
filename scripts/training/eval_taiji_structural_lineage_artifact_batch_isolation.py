"""Run the R5C-S35 artifact-batch input isolation canary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_structural_lineage_restart_continuation import (  # noqa: E402
    _build_migrated_runtime,
    _continuation_requests,
    _record_continuation_evidence,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from taiji import TSKV8Adapter  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s35-structural-lineage-artifact-batch-isolation-v1"


def _save_native_checkpoint(model: TSKV8Adapter, path: Path) -> None:
    torch.save(model.native_checkpoint(), path)


def _load_native_checkpoint(path: Path) -> TSKV8Adapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return TSKV8Adapter.from_native_checkpoint(checkpoint)


def evaluate() -> dict[str, object]:
    runtime = _build_migrated_runtime()
    continuation_evidence = _record_continuation_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _continuation_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S35 continuation batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_candidate, second_candidate = batch.selected_candidate_ids
    unknown_candidate = "candidate:foreign:unselected"
    before_unknown = _checkpoint_digest(runtime.model.architecture.native_checkpoint())

    unknown_key_rejected = True
    for artifacts, replays in (
        ({unknown_candidate: {}}, {}),
        ({}, {unknown_candidate: {}}),
    ):
        try:
            runtime.model.architecture.continue_structural_candidate_batch_from_validation_artifacts(
                batch.batch_id,
                artifacts_by_candidate=artifacts,
                replays_by_candidate=replays,
            )
        except ValueError as exc:
            unknown_key_rejected = unknown_key_rejected and (
                "outside the selected batch" in str(exc)
            )
        else:
            unknown_key_rejected = False
    unknown_key_atomic = (
        _checkpoint_digest(runtime.model.architecture.native_checkpoint()) == before_unknown
    )

    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    before_failure_path = checkpoint_root / f"s35-before-failure-{suffix}.pt"
    after_failure_path = checkpoint_root / f"s35-after-failure-{suffix}.pt"
    after_success_path = checkpoint_root / f"s35-after-success-{suffix}.pt"
    try:
        first_artifact, first_replay, _ = _build_artifact(
            runtime.model.architecture,
            first_candidate,
            continuation_evidence,
        )
        _save_native_checkpoint(runtime.model.architecture, before_failure_path)
        restored = _load_native_checkpoint(before_failure_path)
        malformed_payload = dict(first_artifact.to_payload())
        malformed_payload["artifact_digest"] = "0" * 64
        failed = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={first_candidate: malformed_payload},
            replays_by_candidate={first_candidate: first_replay},
        )
        failed_batch = next(
            item for item in restored.structural_candidate_batches if item.batch_id == batch.batch_id
        )
        first_failed_second_reserved = (
            failed["results"][first_candidate]["status"] == "failed_closed"
            and failed_batch.state_by_candidate[first_candidate] == "failed_closed"
            and failed_batch.state_by_candidate[second_candidate] == "reserved"
        )
        _save_native_checkpoint(restored, after_failure_path)

        resumed = _load_native_checkpoint(after_failure_path)
        second_artifact, second_replay, second_measurements = _build_artifact(
            resumed,
            second_candidate,
            continuation_evidence,
        )
        succeeded = resumed.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact},
            replays_by_candidate={second_candidate: second_replay},
        )
        _save_native_checkpoint(resumed, after_success_path)
        final = _load_native_checkpoint(after_success_path)
        final_batch = next(
            item for item in final.structural_candidate_batches if item.batch_id == batch.batch_id
        )
        before_repeat_topology = tuple(
            (region.region_id, region.unit_ids) for region in final.neuron_regions
        )
        before_repeat_budget = final.cognitive_snapshot().development.structural_budget
        repeated = final.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact},
            replays_by_candidate={second_candidate: second_replay},
        )
        after_repeat_topology = tuple(
            (region.region_id, region.unit_ids) for region in final.neuron_regions
        )
        metrics = {
            "unknown_artifact_and_replay_keys_rejected": unknown_key_rejected,
            "unknown_key_rejection_is_atomic": unknown_key_atomic,
            "malformed_candidate_failure_isolated": first_failed_second_reserved,
            "valid_sibling_artifact_continues_after_restart": (
                succeeded["results"][second_candidate]["status"] == "admitted"
                and final_batch.state_by_candidate[first_candidate] == "failed_closed"
                and final_batch.state_by_candidate[second_candidate] == "admitted"
                and second_measurements.measurement_digest == second_artifact.measurement_digest
            ),
            "repeated_valid_artifact_is_idempotent": (
                repeated["results"][second_candidate]["status"] == "already_applied"
                and after_repeat_topology == before_repeat_topology
                and final.cognitive_snapshot().development.structural_budget == before_repeat_budget
            ),
            "checkpoint_preserves_partial_batch_state": (
                first_candidate in final_batch.state_by_candidate
                and second_candidate in final_batch.state_by_candidate
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_ids": [first_candidate, second_candidate],
            "first_artifact_digest": first_artifact.artifact_digest,
            "second_artifact_digest": second_artifact.artifact_digest,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "artifact batch inputs must reject unknown candidates atomically, isolate "
                    "malformed artifacts, preserve a valid sibling continuation across checkpoint, "
                    "and keep repeated valid consumption topology/budget idempotent"
                ),
            },
            "boundary": (
                "This canary covers native CPU artifact-batch input isolation and partial failure. "
                "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
            ),
        }
    finally:
        before_failure_path.unlink(missing_ok=True)
        after_failure_path.unlink(missing_ok=True)
        after_success_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s35_structural_lineage_artifact_batch_isolation_20260831.json",
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

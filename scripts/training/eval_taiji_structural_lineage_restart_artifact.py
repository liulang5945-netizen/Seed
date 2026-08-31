"""Run the R5C-S32 restart replay-bound validation artifact canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s32-structural-lineage-restart-artifact-v1"


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
        raise AssertionError(f"S32 continuation batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_candidate, second_candidate = batch.selected_candidate_ids

    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    before_path = checkpoint_root / f"s32-before-artifact-{suffix}.pt"
    first_artifact_path = checkpoint_root / f"s32-first-artifact-{suffix}.json"
    after_first_path = checkpoint_root / f"s32-after-first-{suffix}.pt"
    second_artifact_path = checkpoint_root / f"s32-second-artifact-{suffix}.json"
    final_path = checkpoint_root / f"s32-final-{suffix}.pt"
    try:
        _save_native_checkpoint(runtime.model.architecture, before_path)
        restored = _load_native_checkpoint(before_path)
        first_artifact, first_replay, first_measurements = _build_artifact(
            restored,
            first_candidate,
            continuation_evidence,
        )
        measured_parent_matches = (
            _checkpoint_digest(restored.native_checkpoint())
            == first_artifact.parent_checkpoint_digest
        )
        # Capacity measurement is part of the measured parent state. Persist
        # that state before asking a later process to consume the artifact.
        _save_native_checkpoint(restored, before_path)
        first_artifact_path.write_text(
            json.dumps(first_artifact.to_payload(), sort_keys=True),
            encoding="utf-8",
        )
        first_artifact_payload = json.loads(
            first_artifact_path.read_text(encoding="utf-8")
        )

        tamper_branch = _load_native_checkpoint(before_path)
        before_tamper = _checkpoint_digest(tamper_branch.native_checkpoint())
        before_budget = tamper_branch.cognitive_snapshot().development.structural_budget
        before_topology = tuple(
            region.unit_ids for region in tamper_branch.neuron_regions
        )
        tampered_payload = dict(first_artifact_payload)
        tampered_payload["measurement_digest"] = "0" * 64
        tampered = tamper_branch.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={first_candidate: tampered_payload},
            replays_by_candidate={first_candidate: first_replay},
        )
        tamper_after = _checkpoint_digest(tamper_branch.native_checkpoint())
        tamper_topology = tuple(
            region.unit_ids for region in tamper_branch.neuron_regions
        )
        tamper_failed_closed = (
            tampered["results"][first_candidate]["status"] == "failed_closed"
        )

        artifact_restored = _load_native_checkpoint(before_path)
        first_result = artifact_restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={first_candidate: first_artifact_payload},
            replays_by_candidate={first_candidate: first_replay},
        )
        _save_native_checkpoint(artifact_restored, after_first_path)
        first_resumed = _load_native_checkpoint(after_first_path)
        persisted_first_artifact = next(
            item
            for item in first_resumed.structural_validation_artifacts
            if item.artifact_digest == first_artifact.artifact_digest
        )

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
        _save_native_checkpoint(first_resumed, final_path)
        final = _load_native_checkpoint(final_path)
        repeated = final.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={
                first_candidate: first_artifact_payload,
                second_candidate: second_artifact_payload,
            },
            replays_by_candidate={
                first_candidate: first_replay,
                second_candidate: second_replay,
            },
        )

        metrics = {
            "measured_parent_matches_checkpoint": measured_parent_matches,
            "artifact_payload_roundtrips_by_content": (
                first_artifact_payload["artifact_digest"] == first_artifact.artifact_digest
                and first_artifact_payload["measurement_digest"]
                == first_measurements.measurement_digest
            ),
            "tampered_artifact_fails_closed": tamper_failed_closed,
            "tamper_preserves_budget_and_topology": (
                tamper_branch.cognitive_snapshot().development.structural_budget
                == before_budget
                and tamper_topology == before_topology
                and before_tamper != tamper_after
            ),
            "first_artifact_admits_after_restart": (
                first_result["results"][first_candidate]["status"] == "admitted"
                and persisted_first_artifact.measurement_digest
                == first_measurements.measurement_digest
            ),
            "second_measured_artifact_admits": (
                second_result["results"][second_candidate]["status"] == "admitted"
                and second_artifact.measurement_digest == second_measurements.measurement_digest
            ),
            "repeated_artifact_consumption_is_idempotent": (
                repeated["results"][first_candidate]["status"] == "already_applied"
                and repeated["results"][second_candidate]["status"] == "already_applied"
                and repeated["artifact_batch"]["complete"] is True
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_ids": [first_candidate, second_candidate],
            "first_artifact_digest": first_artifact.artifact_digest,
            "second_artifact_digest": second_artifact.artifact_digest,
            "first_measurement_digest": first_measurements.measurement_digest,
            "second_measurement_digest": second_measurements.measurement_digest,
            "tampered_result": tampered,
            "first_result": first_result,
            "second_result": second_result,
            "repeated_result": repeated,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "a measured, replay-bound validation artifact must survive a native "
                    "checkpoint restart, reject tampering fail-closed, admit each candidate, "
                    "and remain idempotent on repeated consumption"
                ),
            },
            "boundary": (
                "This canary covers native CPU checkpoint and artifact lifecycle semantics. "
                "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
            ),
        }
    finally:
        before_path.unlink(missing_ok=True)
        first_artifact_path.unlink(missing_ok=True)
        after_first_path.unlink(missing_ok=True)
        second_artifact_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s32_structural_lineage_restart_artifact_20260831.json",
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

"""Run the R5C-S33 artifact provenance and rollback/recovery canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s33-structural-lineage-artifact-rollback-v1"


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
        raise AssertionError(f"S33 continuation batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_candidate, second_candidate = batch.selected_candidate_ids

    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    before_path = checkpoint_root / f"s33-before-{suffix}.pt"
    after_first_path = checkpoint_root / f"s33-after-first-{suffix}.pt"
    after_rollback_path = checkpoint_root / f"s33-after-rollback-{suffix}.pt"
    after_compaction_path = checkpoint_root / f"s33-after-compaction-{suffix}.pt"
    try:
        _save_native_checkpoint(runtime.model.architecture, before_path)
        restored = _load_native_checkpoint(before_path)
        first_artifact, first_replay, _ = _build_artifact(
            restored,
            first_candidate,
            continuation_evidence,
        )
        _save_native_checkpoint(restored, before_path)
        restored = _load_native_checkpoint(before_path)
        first_result = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={first_candidate: first_artifact},
            replays_by_candidate={first_candidate: first_replay},
        )
        _save_native_checkpoint(restored, after_first_path)
        first_resumed = _load_native_checkpoint(after_first_path)

        second_artifact, second_replay, _ = _build_artifact(
            first_resumed,
            second_candidate,
            continuation_evidence,
        )
        second_result = first_resumed.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact},
            replays_by_candidate={second_candidate: second_replay},
        )
        second_rollback = first_resumed.rollback_structural_candidate_batch(
            batch.batch_id,
            second_candidate,
        )
        second_replay_after_rollback = (
            first_resumed.continue_structural_candidate_from_validation_artifact(
                second_artifact,
                holdout_inputs=second_replay["holdout_inputs"],
                expected_activities=second_replay["expected_activities"],
            )
        )
        _save_native_checkpoint(first_resumed, after_rollback_path)
        rollback_resumed = _load_native_checkpoint(after_rollback_path)
        persisted_artifacts = {
            item.artifact_digest for item in rollback_resumed.structural_validation_artifacts
        }
        before_first_rollback = (
            rollback_resumed.cognitive_snapshot().development.structural_budget
        )
        first_rollback = rollback_resumed.rollback_structural_candidate_batch(
            batch.batch_id,
            first_candidate,
        )
        maintenance = rollback_resumed.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=(
                rollback_resumed.structural_lineage_retention_policy.to_payload()
            ),
        )
        retention = rollback_resumed.structural_lineage_retention_result
        if retention is None:
            raise AssertionError("S33 retention audit was not recorded")
        before_replay_after_compaction = _checkpoint_digest(
            rollback_resumed.native_checkpoint()
        )
        try:
            rollback_resumed.continue_structural_candidate_batch_from_validation_artifacts(
                batch.batch_id,
                artifacts_by_candidate={
                    first_candidate: first_artifact,
                    second_candidate: second_artifact,
                },
                replays_by_candidate={
                    first_candidate: first_replay,
                    second_candidate: second_replay,
                },
            )
        except ValueError as exc:
            replay_after_compaction_failed_closed = (
                "unknown structural candidate batch" in str(exc)
            )
        else:
            replay_after_compaction_failed_closed = False
        after_replay_after_compaction = _checkpoint_digest(
            rollback_resumed.native_checkpoint()
        )
        _save_native_checkpoint(rollback_resumed, after_compaction_path)
        final = _load_native_checkpoint(after_compaction_path)
        metrics = {
            "both_candidates_admitted": (
                first_result["results"][first_candidate]["status"] == "admitted"
                and second_result["results"][second_candidate]["status"] == "admitted"
            ),
            "artifact_survives_rollback_checkpoint": (
                first_artifact.artifact_digest in persisted_artifacts
                and second_artifact.artifact_digest in persisted_artifacts
                and second_rollback["status"] == "rolled_back"
            ),
            "rolled_back_artifact_cannot_reactivate": (
                second_replay_after_rollback["status"] == "rolled_back"
            ),
            "first_rollback_restores_budget": (
                first_rollback["status"] == "rolled_back"
                and rollback_resumed.cognitive_snapshot().development.structural_budget
                == before_first_rollback + first_rollback["resource_cost"]
            ),
            "terminal_artifact_lineage_compacts_atomically": (
                maintenance == ()
                and retention.status == "compacted"
                and batch.batch_id in retention.removed_batch_ids
                and not (
                    {first_artifact.artifact_digest, second_artifact.artifact_digest}
                    & {
                        item.artifact_digest
                        for item in rollback_resumed.structural_validation_artifacts
                    }
                )
            ),
            "post_compaction_replay_fails_closed_without_mutation": (
                replay_after_compaction_failed_closed
                and before_replay_after_compaction == after_replay_after_compaction
            ),
            "final_restart_does_not_resurrect_rollback_lineage": (
                batch.batch_id not in {item.batch_id for item in final.structural_candidate_batches}
                and batch.batch_id
                not in {item.batch_id for item in final.structural_candidate_rollbacks}
                and not final.structural_validation_artifacts
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_ids": [first_candidate, second_candidate],
            "artifact_digests": [first_artifact.artifact_digest, second_artifact.artifact_digest],
            "first_rollback_status": first_rollback["status"],
            "second_rollback_status": second_rollback["status"],
            "retention_status": retention.status,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "rollback must be an explicit terminal boundary for replay-bound artifacts; "
                    "artifact provenance survives the rollback checkpoint, terminal lineage is "
                    "compacted atomically, and old replay cannot reactivate or resurrect it"
                ),
            },
            "boundary": (
                "This canary covers native CPU artifact rollback, retention, and checkpoint recovery. "
                "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
            ),
        }
    finally:
        before_path.unlink(missing_ok=True)
        after_first_path.unlink(missing_ok=True)
        after_rollback_path.unlink(missing_ok=True)
        after_compaction_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s33_structural_lineage_artifact_rollback_20260831.json",
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

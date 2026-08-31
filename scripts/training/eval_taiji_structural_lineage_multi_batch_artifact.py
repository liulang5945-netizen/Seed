"""Run the R5C-S34 multi-batch artifact retention isolation canary."""

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
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _execute_observation,
)
from taiji import TSKV8Adapter  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s34-structural-lineage-multi-batch-artifact-v1"


def _save_native_checkpoint(model: TSKV8Adapter, path: Path) -> None:
    torch.save(model.native_checkpoint(), path)


def _load_native_checkpoint(path: Path) -> TSKV8Adapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return TSKV8Adapter.from_native_checkpoint(checkpoint)


def _record_second_round(runtime) -> tuple[dict[str, object], ...]:
    rows = (
        (13, "workbench.code", "code-isolation-read", "train", "README.md"),
        (14, "workbench.code", "code-isolation-config", "train", "pyproject.toml"),
        (15, "workbench.code", "code-isolation-holdout", "holdout", "plans/README.md"),
        (16, "workbench.docs", "docs-isolation-roadmap", "train", "plans/README.md"),
        (17, "workbench.docs", "docs-isolation-frontend", "train", "frontend/package.json"),
        (18, "workbench.docs", "docs-isolation-holdout", "holdout", "README.md"),
    )
    return tuple(
        _execute_observation(
            runtime,
            ordinal=ordinal,
            region_id=region_id,
            task_slice_id=task_slice_id,
            partition=partition,
            path=path,
            prediction_error=0.1 if partition == "holdout" else 0.8,
            holdout_transfer=0.9 if partition == "holdout" else 0.0,
        )
        for ordinal, region_id, task_slice_id, partition, path in rows
    )


def evaluate() -> dict[str, object]:
    runtime = _build_migrated_runtime()
    active_batch = runtime.model.architecture.structural_candidate_batches[-1]
    active_batch_id = active_batch.batch_id
    second_round = _record_second_round(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _continuation_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S34 second batch was not created: {schedule}")
    terminal_batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_candidate, second_candidate = terminal_batch.selected_candidate_ids
    try:
        first_artifact, first_replay, _ = _build_artifact(
            runtime.model.architecture,
            first_candidate,
            second_round,
        )
        first_result = runtime.model.architecture.continue_structural_candidate_batch_from_validation_artifacts(
            terminal_batch.batch_id,
            artifacts_by_candidate={first_candidate: first_artifact},
            replays_by_candidate={first_candidate: first_replay},
        )
        second_artifact, second_replay, _ = _build_artifact(
            runtime.model.architecture,
            second_candidate,
            second_round,
        )
        second_result = runtime.model.architecture.continue_structural_candidate_batch_from_validation_artifacts(
            terminal_batch.batch_id,
            artifacts_by_candidate={second_candidate: second_artifact},
            replays_by_candidate={second_candidate: second_replay},
        )
        second_rollback = runtime.model.architecture.rollback_structural_candidate_batch(
            terminal_batch.batch_id,
            second_candidate,
        )
        first_rollback = runtime.model.architecture.rollback_structural_candidate_batch(
            terminal_batch.batch_id,
            first_candidate,
        )
        active_before = next(
            item.to_payload()
            for item in runtime.model.architecture.structural_candidate_batches
            if item.batch_id == active_batch_id
        )
        budget_before = runtime.model.architecture.cognitive_snapshot().development.structural_budget
        checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        suffix = os.getpid()
        before_path = checkpoint_root / f"s34-before-maintenance-{suffix}.pt"
        after_path = checkpoint_root / f"s34-after-maintenance-{suffix}.pt"
        try:
            _save_native_checkpoint(runtime.model.architecture, before_path)
            restored = _load_native_checkpoint(before_path)
            maintenance = restored.run_structural_maintenance_cycle(
                candidate_ids=(),
                holdout_inputs_by_candidate={},
                expected_activities_by_candidate={},
                lineage_retention_policy=restored.structural_lineage_retention_policy.to_payload(),
            )
            retention = restored.structural_lineage_retention_result
            if retention is None:
                raise AssertionError("S34 retention audit was not recorded")
            active_after = next(
                item.to_payload()
                for item in restored.structural_candidate_batches
                if item.batch_id == active_batch_id
            )
            terminal_artifact_digests = {
                first_artifact.artifact_digest,
                second_artifact.artifact_digest,
            }
            before_replay = _checkpoint_digest(restored.native_checkpoint())
            try:
                restored.continue_structural_candidate_batch_from_validation_artifacts(
                    terminal_batch.batch_id,
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
                replay_failed_closed = "unknown structural candidate batch" in str(exc)
            else:
                replay_failed_closed = False
            after_replay = _checkpoint_digest(restored.native_checkpoint())
            _save_native_checkpoint(restored, after_path)
            final = _load_native_checkpoint(after_path)
            metrics = {
                "independent_batches_exist": active_batch_id != terminal_batch.batch_id,
                "terminal_batch_artifacts_admitted": (
                    first_result["results"][first_candidate]["status"] == "admitted"
                    and second_result["results"][second_candidate]["status"] == "admitted"
                ),
                "terminal_rollback_is_explicit": (
                    first_rollback["status"] == "rolled_back"
                    and second_rollback["status"] == "rolled_back"
                ),
                "active_batch_is_protected": (
                    active_batch_id in retention.protected_batch_ids
                    and active_after == active_before
                    and restored.cognitive_snapshot().development.structural_budget
                    == budget_before
                ),
                "terminal_artifacts_compact_as_one_subgraph": (
                    maintenance == ()
                    and terminal_batch.batch_id in retention.removed_batch_ids
                    and not (
                        terminal_artifact_digests
                        & {
                            item.artifact_digest
                            for item in restored.structural_validation_artifacts
                        }
                    )
                ),
                "terminal_replay_fails_closed_without_mutation": (
                    replay_failed_closed and before_replay == after_replay
                ),
                "restart_preserves_isolation": (
                    active_batch_id in {item.batch_id for item in final.structural_candidate_batches}
                    and terminal_batch.batch_id
                    not in {item.batch_id for item in final.structural_candidate_batches}
                    and not (
                        terminal_artifact_digests
                        & {
                            item.artifact_digest
                            for item in final.structural_validation_artifacts
                        }
                    )
                ),
            }
            return {
                "format": REPORT_FORMAT,
                "active_batch_id": active_batch_id,
                "terminal_batch_id": terminal_batch.batch_id,
                "terminal_artifact_digests": sorted(terminal_artifact_digests),
                "retention_status": retention.status,
                "protected_batch_ids": list(retention.protected_batch_ids),
                "removed_batch_ids": list(retention.removed_batch_ids),
                "metrics": metrics,
                "gate": {
                    "passed": all(metrics.values()),
                    "criterion": (
                        "under a small retention limit, active multi-batch lineage must remain "
                        "isolated and checkpointable while a terminal artifact batch is compacted "
                        "as a complete subgraph and cannot be replayed"
                    ),
                },
                "boundary": (
                    "This canary covers native CPU multi-batch artifact retention and isolation. "
                    "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
                ),
            }
        finally:
            before_path.unlink(missing_ok=True)
            after_path.unlink(missing_ok=True)
    finally:
        pass


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s34_structural_lineage_multi_batch_artifact_20260831.json",
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

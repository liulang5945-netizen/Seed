"""Run the R5C-S45 runtime-retention/store-lifecycle separation canary."""

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
    _batch,
    _record_round,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from taiji import (  # noqa: E402
    StructuralLineageRetentionPolicy,
    StructuralValidationArtifactStore,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s45-runtime-retention-store-separation-v1"


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    _record_round(runtime, first_ordinal=1, round_id="active-round")
    active_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if active_schedule.get("status") != "batch_created":
        raise AssertionError(f"active batch was not created: {active_schedule}")
    active_batch_id = str(active_schedule["batch_id"])
    runtime = _checkpoint(runtime, "active")

    terminal_evidence = _record_round(runtime, first_ordinal=7, round_id="terminal-round")
    terminal_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if terminal_schedule.get("status") != "batch_created":
        raise AssertionError(f"terminal batch was not created: {terminal_schedule}")
    terminal_batch_id = str(terminal_schedule["batch_id"])
    terminal_batch = _batch(runtime, terminal_batch_id)
    first_id, second_id = terminal_batch.selected_candidate_ids
    runtime = _checkpoint(runtime, "terminal-scheduled")

    store_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s45-store-{os.getpid()}"
    store = StructuralValidationArtifactStore(store_root)
    paths = [
        PROJECT_ROOT / "output" / "manual-r5-canary" / f"s45-first-{os.getpid()}.pt",
        PROJECT_ROOT / "output" / "manual-r5-canary" / f"s45-second-{os.getpid()}.pt",
        PROJECT_ROOT / "output" / "manual-r5-canary" / f"s45-before-retention-{os.getpid()}.pt",
        PROJECT_ROOT / "output" / "manual-r5-canary" / f"s45-after-retention-{os.getpid()}.pt",
    ]
    try:
        first_artifact, first_replay, _ = _build_artifact(
            runtime.model.architecture,
            first_id,
            terminal_evidence,
        )
        store.put(first_artifact)
        first_bytes = store.path_for(first_artifact.artifact_digest).read_bytes()
        runtime = _checkpoint(runtime, "first-measured", paths[0])
        first_result = runtime.continue_structural_candidate_batch_from_artifact_store(
            terminal_batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={first_id: first_artifact.artifact_digest},
            replays_by_candidate={first_id: first_replay},
        )

        second_artifact, second_replay, _ = _build_artifact(
            runtime.model.architecture,
            second_id,
            terminal_evidence,
        )
        store.put(second_artifact)
        second_bytes = store.path_for(second_artifact.artifact_digest).read_bytes()
        runtime = _checkpoint(runtime, "second-measured", paths[1])
        second_result = runtime.continue_structural_candidate_batch_from_artifact_store(
            terminal_batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={second_id: second_artifact.artifact_digest},
            replays_by_candidate={second_id: second_replay},
        )
        second_rollback = runtime.rollback_structural_candidate_batch(terminal_batch_id, second_id)
        first_rollback = runtime.rollback_structural_candidate_batch(terminal_batch_id, first_id)
        runtime = _checkpoint(runtime, "before-retention", paths[2])
        before_retention = SeedRuntime.load(paths[2])
        policy = StructuralLineageRetentionPolicy.create(1, revision=2)
        maintenance = before_retention.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=policy.to_payload(),
        )
        retention = before_retention.model.architecture.structural_lineage_retention_result
        if retention is None:
            raise AssertionError("S45 retention audit was not recorded")
        before_terminal_replay = _checkpoint_digest(before_retention.model.architecture.native_checkpoint())
        before_retention.save(paths[3])
        after_retention = SeedRuntime.load(paths[3])
        first_external = StructuralValidationArtifactStore(store_root).load(
            first_artifact.artifact_digest
        )
        second_external = store.load(second_artifact.artifact_digest)
        try:
            after_retention.continue_structural_candidate_batch_from_artifact_store(
                terminal_batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={second_id: second_external.artifact_digest},
                replays_by_candidate={second_id: second_replay},
            )
        except ValueError as exc:
            terminal_replay_rejected = "unknown structural candidate batch" in str(exc)
        else:
            terminal_replay_rejected = False
        after_terminal_replay = _checkpoint_digest(after_retention.model.architecture.native_checkpoint())
        final_batch_ids = {
            item.batch_id for item in after_retention.model.architecture.structural_candidate_batches
        }
        metrics = {
            "terminal_artifacts_are_still_physically_present": (
                store.path_for(first_artifact.artifact_digest).read_bytes() == first_bytes
                and store.path_for(second_artifact.artifact_digest).read_bytes() == second_bytes
                and first_external == first_artifact
                and second_external == second_artifact
            ),
            "retention_removes_runtime_lineage_only": (
                terminal_batch_id in retention.removed_batch_ids
                and active_batch_id in retention.protected_batch_ids
                and terminal_batch_id not in final_batch_ids
                and active_batch_id in final_batch_ids
                and maintenance["maintenance_results"] == []
            ),
            "stored_old_artifact_cannot_resurrect_deleted_batch": (
                terminal_replay_rejected and before_terminal_replay == after_terminal_replay
            ),
            "terminal_admission_and_rollback_were_explicit": (
                first_result["results"][first_id]["status"] == "admitted"
                and second_result["results"][second_id]["status"] == "admitted"
                and first_rollback["status"] == "rolled_back"
                and second_rollback["status"] == "rolled_back"
            ),
            "checkpoint_restore_preserves_store_and_audit_ownership": (
                after_retention.model.architecture.structural_lineage_retention_policy
                == policy
                and after_retention.model.architecture.structural_lineage_retention_result
                == retention
                and store.load(first_artifact.artifact_digest) == first_artifact
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "active_batch_id": active_batch_id,
            "terminal_batch_id": terminal_batch_id,
            "artifact_digests": [first_artifact.artifact_digest, second_artifact.artifact_digest],
            "retention": retention.to_payload(),
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "runtime retention may remove terminal lineage references but must not delete "
                    "immutable external artifacts or let them resurrect a deleted batch"
                ),
            },
            "boundary": (
                "This canary covers native CPU runtime/store lifecycle separation. It does not claim "
                "automatic garbage collection, unlimited storage, open-domain quality, unlimited "
                "growth, CUDA, frontend behavior, Windows shell, CI completion, or general intelligence."
            ),
        }
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
        _remove_directory(store_root)


def _checkpoint(runtime: SeedRuntime, name: str, path: Path | None = None) -> SeedRuntime:
    target = path or (
        PROJECT_ROOT / "output" / "manual-r5-canary" / f"s45-{name}-{os.getpid()}.pt"
    )
    runtime.save(target)
    return SeedRuntime.load(target)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s45_runtime_retention_store_separation_20260831.json",
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

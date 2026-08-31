"""Run the R5C-S48 runtime/store read-only reconciliation canary."""

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
    ArtifactConsumptionPolicy,
    StructuralLineageRetentionPolicy,
    StructuralValidationArtifactStore,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s48-artifact-store-runtime-reconciliation-v1"


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

    terminal_evidence = _record_round(runtime, first_ordinal=7, round_id="terminal-round")
    terminal_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if terminal_schedule.get("status") != "batch_created":
        raise AssertionError(f"terminal batch was not created: {terminal_schedule}")
    terminal_batch_id = str(terminal_schedule["batch_id"])
    first_id, second_id = _batch(runtime, terminal_batch_id).selected_candidate_ids

    store_root = (
        PROJECT_ROOT
        / "output"
        / "manual-r5-canary"
        / f"s48-store-{os.getpid()}"
    )
    before_retention_path = store_root.parent / f"s48-before-retention-{os.getpid()}.pt"
    after_retention_path = store_root.parent / f"s48-after-retention-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
    legacy_policy = ArtifactConsumptionPolicy.legacy_compatible(
        reason="historical-s48-reconciliation-canary"
    )
    try:
        first_artifact, first_replay, _ = _build_artifact(
            runtime.model.architecture,
            first_id,
            terminal_evidence,
        )
        store.put(first_artifact)
        first_result = runtime.continue_structural_candidate_batch_from_artifact_store(
            terminal_batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={first_id: first_artifact.artifact_digest},
            replays_by_candidate={first_id: first_replay},
            artifact_consumption_policy=legacy_policy,
        )
        second_artifact, second_replay, _ = _build_artifact(
            runtime.model.architecture,
            second_id,
            terminal_evidence,
        )
        second_result = runtime.continue_structural_candidate_batch_from_validation_artifacts(
            terminal_batch_id,
            artifacts_by_candidate={second_id: second_artifact.to_payload()},
            replays_by_candidate={second_id: second_replay},
        )
        second_rollback = runtime.rollback_structural_candidate_batch(
            terminal_batch_id, second_id
        )
        first_rollback = runtime.rollback_structural_candidate_batch(terminal_batch_id, first_id)

        before_projection_checkpoint = _checkpoint_digest(
            runtime.model.architecture.native_checkpoint()
        )
        before_store_bytes = store.path_for(first_artifact.artifact_digest).read_bytes()
        projection = runtime.project_structural_artifact_store_audit(artifact_store=store)
        repeated = runtime.project_structural_artifact_store_audit(artifact_store=store)
        healthy_reconciliation = (
            projection == repeated
            and projection["runtime_artifact_digests"] == sorted(
                (first_artifact.artifact_digest, second_artifact.artifact_digest)
            )
            and projection["runtime_batch_artifact_digests"] == sorted(
                (first_artifact.artifact_digest, second_artifact.artifact_digest)
            )
            and projection["missing_runtime_artifact_digests"] == [
                second_artifact.artifact_digest
            ]
            and projection["missing_runtime_batch_artifact_digests"] == [
                second_artifact.artifact_digest
            ]
        )
        missing_is_not_orphan = (
            {
                item["runtime_visibility"] for item in projection["entries"]
            }
            == {"runtime_recorded"}
            and second_artifact.artifact_digest
            not in {item["artifact_digest"] for item in projection["entries"]}
        )
        projection_is_read_only = (
            _checkpoint_digest(runtime.model.architecture.native_checkpoint())
            == before_projection_checkpoint
            and store.path_for(first_artifact.artifact_digest).read_bytes()
            == before_store_bytes
        )

        runtime.save(before_retention_path)
        restored = SeedRuntime.load(before_retention_path)
        restored_projection = restored.project_structural_artifact_store_audit(
            artifact_store=StructuralValidationArtifactStore(store_root)
        )
        checkpoint_restore_stable = restored_projection == projection

        policy = StructuralLineageRetentionPolicy.create(1, revision=2)
        restored.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=policy.to_payload(),
        )
        restored.save(after_retention_path)
        after_retention = SeedRuntime.load(after_retention_path)
        after_projection = after_retention.project_structural_artifact_store_audit(
            artifact_store=store
        )
        orphan_is_distinct = (
            after_projection["runtime_artifact_digests"] == []
            and after_projection["runtime_batch_artifact_digests"] == []
            and after_projection["missing_runtime_artifact_digests"] == []
            and after_projection["missing_runtime_batch_artifact_digests"] == []
            and {
                item["runtime_visibility"] for item in after_projection["entries"]
            }
            == {"external_orphan"}
        )

        first_path = store.path_for(first_artifact.artifact_digest)
        original_bytes = first_path.read_bytes()
        before_tampered_query = _checkpoint_digest(
            after_retention.model.architecture.native_checkpoint()
        )
        first_path.write_bytes(b"{}")
        try:
            after_retention.project_structural_artifact_store_audit(artifact_store=store)
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        first_path.write_bytes(original_bytes)
        tamper_is_read_only = (
            tamper_rejected
            and _checkpoint_digest(after_retention.model.architecture.native_checkpoint())
            == before_tampered_query
            and first_path.read_bytes() == original_bytes
        )

        metrics = {
            "healthy_runtime_store_reconciliation": healthy_reconciliation,
            "runtime_missing_is_not_external_orphan": missing_is_not_orphan,
            "checkpoint_restore_preserves_reconciliation": checkpoint_restore_stable,
            "retention_orphan_is_distinct_from_runtime_missing": orphan_is_distinct,
            "projection_and_tamper_paths_are_read_only": (
                projection_is_read_only and tamper_is_read_only
            ),
            "lifecycle_actions_remain_explicit": (
                first_result["results"][first_id]["status"] == "admitted"
                and second_result["results"][second_id]["status"] == "admitted"
                and first_rollback["status"] == "rolled_back"
                and second_rollback["status"] == "rolled_back"
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "active_batch_id": active_batch_id,
            "terminal_batch_id": terminal_batch_id,
            "missing_runtime_artifact_digest": second_artifact.artifact_digest,
            "external_orphan_artifact_digest": first_artifact.artifact_digest,
            "projection_audit_digest": projection["audit_digest"],
            "orphan_projection_audit_digest": after_projection["audit_digest"],
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "runtime/store reconciliation must expose both directions of the reference "
                    "relationship without repairing, registering, deleting, or consuming artifacts"
                ),
            },
            "boundary": (
                "This canary covers native CPU read-only artifact-store/runtime reconciliation. "
                "It does not claim automatic repair, registration, deletion, unlimited storage, "
                "open-domain quality, unlimited growth, CUDA, frontend behavior, Windows shell, "
                "CI completion, or general intelligence."
            ),
        }
    finally:
        before_retention_path.unlink(missing_ok=True)
        after_retention_path.unlink(missing_ok=True)
        _remove_directory(store_root)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s48_artifact_store_runtime_reconciliation_20260831.json",
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

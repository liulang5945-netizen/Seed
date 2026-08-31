"""Run the R5C-S46 read-only external artifact-store audit canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s46-runtime-retention-store-audit-v1"


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
    terminal_batch = _batch(runtime, terminal_batch_id)
    first_id, second_id = terminal_batch.selected_candidate_ids

    store_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s46-store-{os.getpid()}"
    before_retention_path = (
        store_root.parent / f"s46-before-retention-{os.getpid()}.pt"
    )
    after_retention_path = store_root.parent / f"s46-after-retention-{os.getpid()}.pt"
    store = StructuralValidationArtifactStore(store_root)
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
        )
        second_artifact, second_replay, _ = _build_artifact(
            runtime.model.architecture,
            second_id,
            terminal_evidence,
        )
        store.put(second_artifact)
        second_result = runtime.continue_structural_candidate_batch_from_artifact_store(
            terminal_batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={second_id: second_artifact.artifact_digest},
            replays_by_candidate={second_id: second_replay},
        )
        second_rollback = runtime.rollback_structural_candidate_batch(
            terminal_batch_id, second_id
        )
        first_rollback = runtime.rollback_structural_candidate_batch(terminal_batch_id, first_id)
        runtime.save(before_retention_path)
        restored = SeedRuntime.load(before_retention_path)
        policy = StructuralLineageRetentionPolicy.create(1, revision=2)
        maintenance = restored.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=policy.to_payload(),
        )
        restored.save(after_retention_path)
        after_retention = SeedRuntime.load(after_retention_path)
        before_audit_digest = _checkpoint_digest(
            after_retention.model.architecture.native_checkpoint()
        )

        healthy_inventory = store.inventory()
        audit_inventory = store.audit()
        records_by_digest = {
            item["artifact_digest"]: item for item in healthy_inventory
        }
        healthy_facts_match = all(
            records_by_digest[artifact.artifact_digest]["measurement_digest"]
            == artifact.measurement_digest
            and records_by_digest[artifact.artifact_digest]["resource_cost"]
            == artifact.resource_cost
            for artifact in (first_artifact, second_artifact)
        )
        try:
            after_retention.continue_structural_candidate_batch_from_artifact_store(
                terminal_batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={
                    second_id: second_artifact.artifact_digest
                },
                replays_by_candidate={second_id: second_replay},
            )
        except ValueError as exc:
            orphan_replay_rejected = "unknown structural candidate batch" in str(exc)
        else:
            orphan_replay_rejected = False

        first_path = store.path_for(first_artifact.artifact_digest)
        original_bytes = first_path.read_bytes()
        tampered_payload = json.loads(original_bytes.decode("utf-8"))
        tampered_payload["measurement_digest"] = "0" * 64
        first_path.write_text(
            json.dumps(
                tampered_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        try:
            store.audit()
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        tampered_bytes_remain = first_path.read_bytes() != original_bytes
        first_path.write_bytes(original_bytes)

        invalid_path = store_root / "invalid-name.json"
        invalid_path.write_bytes(original_bytes)
        try:
            store.inventory()
        except ValueError:
            invalid_name_rejected = True
        else:
            invalid_name_rejected = False
        invalid_file_remains = invalid_path.exists()
        invalid_path.unlink(missing_ok=True)

        after_audit_digest = _checkpoint_digest(
            after_retention.model.architecture.native_checkpoint()
        )
        metrics = {
            "healthy_inventory_is_stable_and_audit_alias_matches": (
                len(healthy_inventory) == 2
                and audit_inventory == healthy_inventory
                and store.inventory() == healthy_inventory
                and healthy_facts_match
            ),
            "runtime_orphans_remain_auditable_but_not_consumable": (
                terminal_batch_id
                in after_retention.model.architecture.structural_lineage_retention_result.removed_batch_ids
                and active_batch_id
                in {
                    item.batch_id
                    for item in after_retention.model.architecture.structural_candidate_batches
                }
                and orphan_replay_rejected
            ),
            "tampered_measurement_facts_fail_closed_without_repair": (
                tamper_rejected and tampered_bytes_remain and first_path.exists()
            ),
            "invalid_filename_fails_closed_without_deletion": (
                invalid_name_rejected and invalid_file_remains
            ),
            "audit_is_runtime_read_only": (
                before_audit_digest == after_audit_digest
                and maintenance["maintenance_results"] == []
                and first_result["results"][first_id]["status"] == "admitted"
                and second_result["results"][second_id]["status"] == "admitted"
                and first_rollback["status"] == "rolled_back"
                and second_rollback["status"] == "rolled_back"
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "active_batch_id": active_batch_id,
            "terminal_batch_id": terminal_batch_id,
            "artifact_digests": [
                first_artifact.artifact_digest,
                second_artifact.artifact_digest,
            ],
            "inventory": list(healthy_inventory),
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "external artifact inventory must be deterministic and integrity-checked, "
                    "while remaining read-only and independent from runtime retention or replay"
                ),
            },
            "boundary": (
                "This canary covers native CPU read-only external artifact-store audit. It does "
                "not claim automatic garbage collection, deletion, unlimited storage, open-domain "
                "quality, unlimited growth, CUDA, frontend behavior, Windows shell, CI completion, "
                "or general intelligence."
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
        / "taiji_w7_r5c_s46_runtime_retention_store_audit_20260831.json",
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

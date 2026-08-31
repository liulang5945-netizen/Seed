"""Run the R5C-S29 SeedRuntime disk-checkpoint continuation canary."""

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

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_structural_lineage_compaction import (  # noqa: E402
    _record_terminal_subgraph,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import StructuralLineageRetentionPolicy  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s29-structural-lineage-disk-checkpoint-v1"


def _runtime_with_terminal_lineage() -> SeedRuntime:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S29 active batch was not created: {schedule}")
    model = runtime.model.architecture
    active = next(
        item for item in model.structural_candidate_batches if item.batch_id == schedule["batch_id"]
    )
    _record_terminal_subgraph(model, active)
    return runtime


def evaluate() -> dict[str, object]:
    runtime = _runtime_with_terminal_lineage()
    source = StructuralLineageRetentionPolicy.create(1)
    maintenance = runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_policy=source,
    )
    target = source.migrate_to_latest()
    migration = runtime.migrate_structural_lineage_retention_policy(target)
    expected_status = runtime.structural_maintenance_status()
    expected_result = runtime.model.architecture.structural_lineage_retention_result
    terminal_batch_id = "batch:terminal-lineage"
    terminal_removed_before_save = terminal_batch_id not in {
        item.batch_id for item in runtime.model.architecture.structural_candidate_batches
    }
    topology_before = tuple(
        (region.region_id, region.unit_ids) for region in runtime.model.architecture.neuron_regions
    )
    budget_before = runtime.model.architecture.cognitive_snapshot().development.structural_budget

    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    migrated_path = checkpoint_root / f"s29-migrated-{suffix}.pt"
    rolled_back_path = checkpoint_root / f"s29-rolled-back-{suffix}.pt"
    tampered_path = checkpoint_root / f"s29-tampered-{suffix}.pt"
    missing_field_path = checkpoint_root / f"s29-missing-field-{suffix}.pt"
    try:
        runtime.save(migrated_path)
        original_bytes = migrated_path.read_bytes()
        restored = SeedRuntime.load(migrated_path)
        restored_status = restored.structural_maintenance_status()
        restored_result = restored.model.architecture.structural_lineage_retention_result

        tampered_payload = torch.load(migrated_path, map_location="cpu", weights_only=False)
        tampered_taiji = dict(tampered_payload["taiji"])
        tampered_components = dict(tampered_taiji["components"])
        tampered_runtime = dict(tampered_components["structural_runtime"])
        tampered_migration = dict(
            tampered_runtime["lineage_retention_policy_migration"]
        )
        tampered_migration["migration_digest"] = "0" * 64
        tampered_runtime["lineage_retention_policy_migration"] = tampered_migration
        tampered_components["structural_runtime"] = tampered_runtime
        tampered_taiji["components"] = tampered_components
        tampered_payload["taiji"] = tampered_taiji
        torch.save(tampered_payload, tampered_path)
        try:
            SeedRuntime.load(tampered_path)
        except ValueError as exc:
            tampered_rejected = "migration digest mismatch" in str(exc)
        else:
            tampered_rejected = False

        incomplete_payload = dict(tampered_payload)
        incomplete_payload["config"] = dict(incomplete_payload["config"])
        incomplete_payload["config"].pop("taiji", None)
        torch.save(incomplete_payload, missing_field_path)
        try:
            SeedRuntime.load(missing_field_path)
        except (KeyError, TypeError, ValueError):
            incomplete_rejected = True
        else:
            incomplete_rejected = False

        rollback = restored.rollback_structural_lineage_retention_policy_migration(migration)
        restored.save(rolled_back_path)
        resumed = SeedRuntime.load(rolled_back_path)
        resumed_status = resumed.structural_maintenance_status()
        topology_after = tuple(
            (region.region_id, region.unit_ids) for region in resumed.model.architecture.neuron_regions
        )
        budget_after = resumed.model.architecture.cognitive_snapshot().development.structural_budget

        metrics = {
            "maintenance_audit_is_preserved": (
                maintenance["retention_policy"] == source.to_payload()
                and expected_result is not None
                and restored_result == expected_result
            ),
            "deleted_lineage_does_not_reappear": (
                terminal_removed_before_save
                and terminal_batch_id
                not in {item.batch_id for item in restored.model.architecture.structural_candidate_batches}
                and terminal_batch_id
                not in {item.batch_id for item in resumed.model.architecture.structural_candidate_batches}
            ),
            "migration_and_status_survive_disk_restore": (
                restored_status == expected_status
                and restored_status["last_retention_policy"]["revision"] == 2
                and restored_status["last_retention_policy_migration"]["status"] == "committed"
            ),
            "rollback_continues_after_restore": (
                rollback["status"] == "rolled_back"
                and resumed_status["last_retention_policy"]["revision"] == 1
                and resumed_status["last_retention_policy_migration"]["status"] == "rolled_back"
                and resumed.model.architecture.structural_lineage_retention_result == expected_result
            ),
            "lineage_and_budget_are_unchanged": (
                topology_after == topology_before and budget_after == budget_before
            ),
            "both_disk_artifacts_exist": (
                migrated_path.is_file()
                and rolled_back_path.is_file()
                and migrated_path.stat().st_size > 0
                and rolled_back_path.stat().st_size > 0
            ),
            "tampered_or_incomplete_checkpoint_fails_closed": (
                tampered_rejected
                and incomplete_rejected
                and migrated_path.read_bytes() == original_bytes
                and runtime.structural_maintenance_status() == expected_status
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "source_policy": source.to_payload(),
            "target_policy": target.to_payload(),
            "migration": migration,
            "rollback": rollback,
            "artifact_bytes": {
                "migrated": migrated_path.stat().st_size,
                "rolled_back": rolled_back_path.stat().st_size,
            },
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "a real SeedRuntime disk save/load must preserve retention audit, policy migration, "
                    "status and lineage, then permit explicit rollback and a second save/load without "
                    "changing topology or structural budget"
                ),
            },
            "boundary": (
                "This canary covers native CPU disk checkpoint continuation and rollback. "
                "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
            ),
        }
    finally:
        migrated_path.unlink(missing_ok=True)
        rolled_back_path.unlink(missing_ok=True)
        tampered_path.unlink(missing_ok=True)
        missing_field_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s29_structural_lineage_disk_checkpoint_20260831.json",
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

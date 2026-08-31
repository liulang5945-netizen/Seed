"""Run the R5C-S28 retention-policy migration and rollback canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from seed import Seed  # noqa: E402
from taiji import (  # noqa: E402
    StructuralLineageRetentionPolicy,
    StructuralLineageRetentionPolicyMigration,
    TSKV8Adapter,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s28-structural-lineage-policy-migration-v1"


def _runtime_with_terminal_lineage() -> SeedRuntime:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S28 active batch was not created: {schedule}")
    model = runtime.model.architecture
    active = next(item for item in model.structural_candidate_batches if item.batch_id == schedule["batch_id"])
    _record_terminal_subgraph(model, active)
    return runtime


def evaluate() -> dict[str, object]:
    runtime = _runtime_with_terminal_lineage()
    source = StructuralLineageRetentionPolicy.create(1)
    first_audit = runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_policy=source,
    )
    model = runtime.model.architecture
    first_result = model.structural_lineage_retention_result
    topology = tuple((item.region_id, item.unit_ids) for item in model.neuron_regions)
    budget = model.cognitive_snapshot().development.structural_budget
    target = source.migrate_to_latest()

    committed_payload = runtime.migrate_structural_lineage_retention_policy(target)
    committed = StructuralLineageRetentionPolicyMigration.from_payload(committed_payload)
    checkpoint = runtime.model.checkpoint()
    restored_runtime = SeedRuntime(Seed.from_checkpoint(checkpoint))
    restored_model = restored_runtime.model.architecture
    restored_status = restored_runtime.structural_maintenance_status()
    expected_status = runtime.structural_maintenance_status()

    rolled_back_payload = runtime.rollback_structural_lineage_retention_policy_migration(
        committed_payload
    )
    rolled_back = StructuralLineageRetentionPolicyMigration.from_payload(rolled_back_payload)

    invalid_before = _checkpoint_digest(model.native_checkpoint())
    try:
        runtime.migrate_structural_lineage_retention_policy(
            StructuralLineageRetentionPolicy.create(2, revision=2)
        )
    except ValueError as exc:
        invalid_failed_closed = "safety semantics" in str(exc)
    else:
        invalid_failed_closed = False
    invalid_after = _checkpoint_digest(model.native_checkpoint())

    tampered = dict(committed_payload)
    tampered["migration_digest"] = "0" * 64
    try:
        StructuralLineageRetentionPolicyMigration.from_payload(tampered)
    except ValueError as exc:
        tamper_failed_closed = "digest mismatch" in str(exc)
    else:
        tamper_failed_closed = False

    inconsistent = restored_model.native_checkpoint()
    runtime_component = dict(inconsistent["components"]["structural_runtime"])
    runtime_component["lineage_retention_policy_migration"] = tampered
    inconsistent["components"] = {
        **inconsistent["components"],
        "structural_runtime": runtime_component,
    }
    try:
        TSKV8Adapter.from_native_checkpoint(inconsistent)
    except ValueError as exc:
        inconsistent_failed_closed = "digest mismatch" in str(exc)
    else:
        inconsistent_failed_closed = False

    metrics = {
        "v1_to_latest_migration_is_explicit_and_safe": (
            committed.status == "committed"
            and committed.source_policy == source
            and committed.target_policy == target
            and target.max_batches == source.max_batches
            and target.protection_rules == source.protection_rules
        ),
        "migration_checkpoint_restore_is_consistent": (
            restored_model.structural_lineage_retention_policy == target
            and restored_model.structural_lineage_retention_policy_migration == committed
            and restored_model.structural_lineage_retention_result == first_result
            and restored_status == expected_status
        ),
        "rollback_preserves_old_audit_and_structure": (
            rolled_back.status == "rolled_back"
            and model.structural_lineage_retention_policy == source
            and model.structural_lineage_retention_result == first_result
            and tuple((item.region_id, item.unit_ids) for item in model.neuron_regions) == topology
            and model.cognitive_snapshot().development.structural_budget == budget
        ),
        "no_request_does_not_implicitly_migrate": (
            first_audit["retention_policy"] == source.to_payload()
            and model.structural_lineage_retention_policy_migration == rolled_back
        ),
        "invalid_migration_is_atomic": invalid_failed_closed and invalid_before == invalid_after,
        "tampered_migration_fails_closed": tamper_failed_closed,
        "inconsistent_checkpoint_migration_fails_closed": inconsistent_failed_closed,
    }
    return {
        "format": REPORT_FORMAT,
        "source_policy": source.to_payload(),
        "target_policy": target.to_payload(),
        "migration": committed_payload,
        "rolled_back": rolled_back_payload,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "retention policy revision migration is explicit, adjacent, safe-semantic preserving, "
                "checkpointable and reversible; malformed or inconsistent migrations fail closed"
            ),
        },
        "boundary": (
            "This canary covers native CPU policy lifecycle. It does not claim implicit migration, "
            "background cleanup, policy-driven growth, open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s28_structural_lineage_policy_migration_20260831.json",
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

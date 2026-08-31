"""Run the R5C-S25 SeedRuntime structural-maintenance audit canary."""

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
from taiji import StructuralMaintenanceAudit  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s25-structural-lineage-runtime-v1"
RETENTION_LIMIT = 1


def _runtime_with_terminal_lineage() -> tuple[SeedRuntime, str]:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S25 active batch was not created: {schedule}")
    model = runtime.model.architecture
    active_batch_id = str(schedule["batch_id"])
    active = next(item for item in model.structural_candidate_batches if item.batch_id == active_batch_id)
    _record_terminal_subgraph(model, active)
    return runtime, active_batch_id


def _maintenance(runtime: SeedRuntime, *, max_batches: int | None = None) -> dict[str, object]:
    return runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_max_batches=max_batches,
    )


def evaluate() -> dict[str, object]:
    runtime, active_batch_id = _runtime_with_terminal_lineage()
    model = runtime.model.architecture
    topology_before = tuple((item.region_id, item.unit_ids) for item in model.neuron_regions)
    budget_before = model.cognitive_snapshot().development.structural_budget

    before_default = _checkpoint_digest(model.native_checkpoint())
    default_audit = StructuralMaintenanceAudit.from_payload(_maintenance(runtime))
    after_default = _checkpoint_digest(model.native_checkpoint())

    explicit_audit = StructuralMaintenanceAudit.from_payload(
        _maintenance(runtime, max_batches=RETENTION_LIMIT)
    )
    retention = explicit_audit.lineage_retention
    checkpoint = runtime.model.checkpoint()
    restored_runtime = SeedRuntime(Seed.from_checkpoint(checkpoint))
    restored_model = restored_runtime.model.architecture
    restored_default = StructuralMaintenanceAudit.from_payload(_maintenance(restored_runtime))

    invalid_before = _checkpoint_digest(restored_model.native_checkpoint())
    try:
        _maintenance(restored_runtime, max_batches=0)
    except ValueError as exc:
        invalid_failed_closed = "max_batches must be positive" in str(exc)
    else:
        invalid_failed_closed = False
    invalid_after = _checkpoint_digest(restored_model.native_checkpoint())

    tampered = dict(explicit_audit.to_payload())
    tampered["audit_digest"] = "0" * 64
    try:
        StructuralMaintenanceAudit.from_payload(tampered)
    except ValueError as exc:
        tamper_failed_closed = "digest mismatch" in str(exc)
    else:
        tamper_failed_closed = False

    metrics = {
        "default_runtime_audit_is_valid_without_retention": (
            default_audit.maintenance_results == ()
            and default_audit.lineage_retention is None
            and before_default == after_default
        ),
        "explicit_runtime_audit_projects_retention": (
            retention is not None
            and retention.status == "compacted"
            and retention.removed_batch_ids
            and active_batch_id in retention.protected_batch_ids
        ),
        "checkpoint_restore_keeps_taiji_audit_state": (
            retention is not None
            and restored_model.structural_lineage_retention_result == retention
            and active_batch_id in {item.batch_id for item in restored_model.structural_candidate_batches}
        ),
        "default_runtime_call_does_not_replay_old_audit": (
            restored_default.lineage_retention is None
            and restored_model.structural_lineage_retention_result == retention
        ),
        "runtime_maintenance_preserves_topology_and_budget": (
            tuple((item.region_id, item.unit_ids) for item in model.neuron_regions)
            == topology_before
            and model.cognitive_snapshot().development.structural_budget == budget_before
        ),
        "invalid_limit_is_atomic_through_runtime": (
            invalid_failed_closed and invalid_before == invalid_after
        ),
        "tampered_runtime_audit_fails_closed": tamper_failed_closed,
    }
    return {
        "format": REPORT_FORMAT,
        "retention_limit": RETENTION_LIMIT,
        "default_audit": default_audit.to_payload(),
        "explicit_audit": explicit_audit.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "SeedRuntime exposes a stable content-addressed maintenance audit; "
                "default calls do not trigger or replay retention, explicit calls preserve "
                "Taiji ownership and checkpoint state, and invalid or tampered payloads fail closed"
            ),
        },
        "boundary": (
            "This canary covers the explicit native CPU SeedRuntime projection. "
            "It does not claim background cleanup, open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s25_structural_lineage_runtime_20260831.json",
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

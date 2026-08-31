"""Run the R5C-S24 explicit structural-maintenance retention canary."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from taiji import TSKV8Adapter  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s24-structural-lineage-maintenance-v1"
RETENTION_LIMIT = 1


def _empty_maintenance(
    model: TSKV8Adapter,
    *,
    max_batches: int | None = None,
) -> tuple:
    return model.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_max_batches=max_batches,
    )


def _topology(model: TSKV8Adapter) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((region.region_id, region.unit_ids) for region in model.neuron_regions)


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S24 active batch was not created: {schedule}")
    model = runtime.model.architecture
    active_batch_id = str(schedule["batch_id"])
    active = next(item for item in model.structural_candidate_batches if item.batch_id == active_batch_id)
    _record_terminal_subgraph(model, active)
    terminal_batch_id = "batch:terminal-lineage"
    topology_before = _topology(model)
    budget_before = model.cognitive_snapshot().development.structural_budget

    before_default_maintenance = _checkpoint_digest(model.native_checkpoint())
    default_results = _empty_maintenance(model)
    after_default_maintenance = _checkpoint_digest(model.native_checkpoint())
    explicit_results = _empty_maintenance(model, max_batches=RETENTION_LIMIT)
    first_audit = model.structural_lineage_retention_result
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_audit = restored.structural_lineage_retention_result

    invalid_before = _checkpoint_digest(restored.native_checkpoint())
    try:
        _empty_maintenance(restored, max_batches=0)
    except ValueError as exc:
        invalid_failed_closed = "max_batches must be positive" in str(exc)
    else:
        invalid_failed_closed = False
    invalid_after = _checkpoint_digest(restored.native_checkpoint())

    pressure_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    retained_active = next(
        item for item in pressure_model.structural_candidate_batches if item.batch_id == active_batch_id
    )
    pressure_model._record_structural_candidate_batch(
        replace(
            retained_active,
            batch_id="batch:protected-copy",
            revision=retained_active.revision + 1,
        )
    )
    pressure_audit = pressure_model
    _empty_maintenance(pressure_audit, max_batches=RETENTION_LIMIT)
    pressure_result = pressure_audit.structural_lineage_retention_result
    pressure_checkpoint = pressure_audit.native_checkpoint()
    pressure_restored = TSKV8Adapter.from_native_checkpoint(pressure_checkpoint)
    _empty_maintenance(pressure_restored, max_batches=RETENTION_LIMIT)
    repeated_pressure = pressure_restored.structural_lineage_retention_result

    metrics = {
        "default_maintenance_does_not_trigger_compaction": (
            default_results == ()
            and before_default_maintenance == after_default_maintenance
            and model.structural_lineage_retention_result is not None
        ),
        "explicit_maintenance_compacts_terminal_lineage": (
            explicit_results == ()
            and first_audit is not None
            and first_audit.status == "compacted"
            and first_audit.removed_batch_ids == (terminal_batch_id,)
            and {item.batch_id for item in model.structural_candidate_batches}
            == {active_batch_id}
        ),
        "maintenance_does_not_change_topology_or_budget": (
            _topology(model) == topology_before
            and model.cognitive_snapshot().development.structural_budget == budget_before
        ),
        "retention_audit_persists_through_checkpoint": (
            first_audit is not None
            and restored_audit == first_audit
            and _checkpoint_digest(restored.native_checkpoint()) == _checkpoint_digest(checkpoint)
            and active_batch_id in {item.batch_id for item in restored.structural_candidate_batches}
        ),
        "invalid_maintenance_limit_is_atomic": (
            invalid_failed_closed and invalid_before == invalid_after
        ),
        "protected_pressure_is_reported_and_idempotent": (
            pressure_result is not None
            and repeated_pressure == pressure_result
            and pressure_result.status == "nothing_to_compact"
            and pressure_result.retention_pressure
            and pressure_result.removed_batch_ids == ()
            and {item.batch_id for item in pressure_restored.structural_candidate_batches}
            == {active_batch_id, "batch:protected-copy"}
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "retention_limit": RETENTION_LIMIT,
        "schedule": schedule,
        "first_audit": None if first_audit is None else first_audit.to_payload(),
        "pressure_audit": None if pressure_result is None else pressure_result.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "lineage retention is invoked only at an explicit structural maintenance boundary; "
                "the audit is checkpointed and idempotent, terminal subgraphs compact atomically, "
                "protected pressure is observable, and invalid limits cannot partially mutate state"
            ),
        },
        "boundary": (
            "This canary covers native CPU maintenance integration using real Workbench evidence. "
            "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s24_structural_lineage_maintenance_20260831.json",
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

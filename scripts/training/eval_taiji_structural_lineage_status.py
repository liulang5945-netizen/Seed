"""Run the R5C-S26 read-only structural-maintenance status canary."""

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
from taiji import STRUCTURAL_MAINTENANCE_STATUS_FORMAT  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s26-structural-lineage-status-v1"
RETENTION_LIMIT = 1


def _runtime_with_terminal_lineage() -> SeedRuntime:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S26 active batch was not created: {schedule}")
    model = runtime.model.architecture
    active = next(item for item in model.structural_candidate_batches if item.batch_id == schedule["batch_id"])
    _record_terminal_subgraph(model, active)
    return runtime


def _maintain(runtime: SeedRuntime) -> dict[str, object]:
    return runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_max_batches=RETENTION_LIMIT,
    )


def evaluate() -> dict[str, object]:
    empty_runtime = _runtime_with_terminal_lineage()
    empty_before = _checkpoint_digest(empty_runtime.model.architecture.native_checkpoint())
    empty_status = empty_runtime.structural_maintenance_status()
    empty_after = _checkpoint_digest(empty_runtime.model.architecture.native_checkpoint())

    runtime = _runtime_with_terminal_lineage()
    audit = _maintain(runtime)
    model = runtime.model.architecture
    status_before_query = _checkpoint_digest(model.native_checkpoint())
    status = runtime.status()["structural_maintenance"]
    status_after_query = _checkpoint_digest(model.native_checkpoint())

    restored = SeedRuntime(Seed.from_checkpoint(runtime.model.checkpoint()))
    restored_status = restored.structural_maintenance_status()

    topology = tuple((item.region_id, item.unit_ids) for item in model.neuron_regions)
    budget = model.cognitive_snapshot().development.structural_budget
    tamper_is_not_consumed = (
        status["last_retention_audit"] == audit["lineage_retention"]
        and topology == tuple((item.region_id, item.unit_ids) for item in model.neuron_regions)
        and budget == model.cognitive_snapshot().development.structural_budget
    )
    metrics = {
        "empty_state_is_explicit": (
            empty_status["format"] == STRUCTURAL_MAINTENANCE_STATUS_FORMAT
            and empty_status["status"] == "no_audit"
            and empty_status["has_retention_audit"] is False
            and empty_status["last_retention_audit"] is None
            and empty_before == empty_after
        ),
        "status_projects_latest_audit": (
            status["status"] == "audit_available"
            and status["has_retention_audit"] is True
            and status["last_retention_audit"] == audit["lineage_retention"]
            and status["retention_pressure"] is False
        ),
        "status_query_has_no_structural_side_effect": (
            status_before_query == status_after_query
            and tamper_is_not_consumed
        ),
        "status_survives_seed_checkpoint_restore": restored_status == status,
        "status_does_not_change_topology_or_budget": (
            topology == tuple((item.region_id, item.unit_ids) for item in model.neuron_regions)
            and budget == model.cognitive_snapshot().development.structural_budget
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "retention_limit": RETENTION_LIMIT,
        "status": status,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "runtime status is a pure read-only projection with an explicit empty state; "
                "it exposes the latest Taiji retention audit without triggering maintenance or changing topology/budget"
            ),
        },
        "boundary": (
            "This canary covers native CPU status projection. It does not claim that status is a structural decision input, "
            "background cleanup, open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s26_structural_lineage_status_20260831.json",
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

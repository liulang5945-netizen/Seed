"""Run the R5C-S30 restart continuation and no-replay canary."""

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
from scripts.training.eval_taiji_structural_lineage_compaction import (  # noqa: E402
    _record_terminal_subgraph,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _execute_observation,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import StructuralLineageRetentionPolicy  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s30-structural-lineage-restart-continuation-v1"


def _continuation_requests() -> tuple[dict[str, object], ...]:
    return (
        {
            "network_id": "workbench",
            "region_id": "workbench.code",
            "controller_region_id": "adaptive.cortex",
            "target_kind": "neuron",
            "operation": "add",
            "substrate_ids": ("adaptive.cortex",),
            "specification": {"region_id": "adaptive.cortex", "unit_id": "u3"},
        },
        {
            "network_id": "workbench",
            "region_id": "workbench.docs",
            "controller_region_id": "adaptive.memory",
            "target_kind": "neuron",
            "operation": "add",
            "substrate_ids": ("adaptive.memory",),
            "specification": {"region_id": "adaptive.memory", "unit_id": "m3"},
        },
    )


def _record_continuation_evidence(runtime: SeedRuntime) -> tuple[dict[str, object], ...]:
    rows = (
        (7, "workbench.code", "code-continuation-read", "train", "README.md", 0.8, 0.0),
        (8, "workbench.code", "code-continuation-config", "train", "pyproject.toml", 0.8, 0.0),
        (9, "workbench.code", "code-continuation-holdout", "holdout", "plans/README.md", 0.1, 0.9),
        (10, "workbench.docs", "docs-continuation-roadmap", "train", "plans/README.md", 0.8, 0.0),
        (11, "workbench.docs", "docs-continuation-frontend", "train", "frontend/package.json", 0.8, 0.0),
        (12, "workbench.docs", "docs-continuation-holdout", "holdout", "README.md", 0.1, 0.9),
    )
    return tuple(
        _execute_observation(
            runtime,
            ordinal=ordinal,
            region_id=region_id,
            task_slice_id=task_slice_id,
            partition=partition,
            path=path,
            prediction_error=prediction_error,
            holdout_transfer=holdout_transfer,
        )
        for ordinal, region_id, task_slice_id, partition, path, prediction_error, holdout_transfer in rows
    )


def _build_migrated_runtime() -> SeedRuntime:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S30 initial batch was not created: {schedule}")
    active = next(
        item
        for item in runtime.model.architecture.structural_candidate_batches
        if item.batch_id == schedule["batch_id"]
    )
    _record_terminal_subgraph(runtime.model.architecture, active)
    source = StructuralLineageRetentionPolicy.create(1)
    runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_policy=source,
    )
    runtime.migrate_structural_lineage_retention_policy(source.migrate_to_latest())
    return runtime


def evaluate() -> dict[str, object]:
    runtime = _build_migrated_runtime()
    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    migrated_path = checkpoint_root / f"s30-migrated-{suffix}.pt"
    continued_path = checkpoint_root / f"s30-continued-{suffix}.pt"
    terminal_batch_id = "batch:terminal-lineage"
    try:
        runtime.save(migrated_path)
        restored = SeedRuntime.load(migrated_path)
        old_tick = restored.model.architecture.structural_runtime_tick
        old_scheduler_revision = restored.model.architecture.structural_growth_scheduler_state.revision
        old_status = restored.structural_maintenance_status()
        evidence = _record_continuation_evidence(restored)
        schedule = restored.schedule_structural_candidate_batch_from_workbench_evidence(
            _continuation_requests()
        )
        continuation = restored.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=restored.model.architecture.structural_lineage_retention_policy.to_payload(),
        )
        new_status = restored.structural_maintenance_status()
        restored.save(continued_path)
        resumed = SeedRuntime.load(continued_path)
        default_replay = resumed.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
        )

        metrics = {
            "new_workbench_evidence_advances_cursor": (
                all(item["outcome"]["status"] == "success" for item in evidence)
                and restored.model.architecture.structural_runtime_tick > old_tick
                and restored.model.architecture.structural_growth_scheduler_state.revision
                > old_scheduler_revision
            ),
            "only_new_windows_create_the_next_batch": (
                schedule["status"] == "batch_created"
                and schedule["source_window_digests"]
                and len(restored.model.architecture.structural_workbench_batch_schedule_results) >= 2
            ),
            "continuation_creates_a_new_explicit_audit": (
                continuation["lineage_retention"] is not None
                and continuation["structural_runtime_tick"]
                == restored.model.architecture.structural_runtime_tick
                and continuation["lineage_retention"]["result_digest"]
                != old_status["last_retention_audit"]["result_digest"]
            ),
            "old_audit_and_deleted_lineage_are_not_replayed": (
                terminal_batch_id
                not in {item.batch_id for item in restored.model.architecture.structural_candidate_batches}
                and terminal_batch_id
                not in {item.batch_id for item in resumed.model.architecture.structural_candidate_batches}
                and default_replay["maintenance_results"] == []
                and default_replay["lineage_retention"] is None
            ),
            "second_restart_preserves_continuation_state": (
                resumed.structural_maintenance_status() == new_status
                and resumed.model.architecture.structural_growth_scheduler_state.revision
                == restored.model.architecture.structural_growth_scheduler_state.revision
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "schedule": schedule,
            "continuation_audit": continuation,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "after restart, only new Workbench evidence may advance the structural cursor and create "
                    "a new explicit maintenance audit; old actions and deleted lineage must not replay, and "
                    "the continuation state must survive a second checkpoint roundtrip"
                ),
            },
            "boundary": (
                "This canary covers native CPU restart continuation and no-replay semantics. "
                "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
            ),
        }
    finally:
        migrated_path.unlink(missing_ok=True)
        continued_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s30_structural_lineage_restart_continuation_20260831.json",
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

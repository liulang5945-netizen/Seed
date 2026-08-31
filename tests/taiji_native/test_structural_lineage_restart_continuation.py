from __future__ import annotations

import os
from pathlib import Path

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_structural_lineage_compaction import _record_terminal_subgraph
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _execute_observation,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji import StructuralLineageRetentionPolicy


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
    return (
        _execute_observation(
            runtime,
            ordinal=7,
            region_id="workbench.code",
            task_slice_id="code-continuation-read",
            partition="train",
            path="README.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=8,
            region_id="workbench.code",
            task_slice_id="code-continuation-config",
            partition="train",
            path="pyproject.toml",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=9,
            region_id="workbench.code",
            task_slice_id="code-continuation-holdout",
            partition="holdout",
            path="plans/README.md",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
        _execute_observation(
            runtime,
            ordinal=10,
            region_id="workbench.docs",
            task_slice_id="docs-continuation-roadmap",
            partition="train",
            path="plans/README.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=11,
            region_id="workbench.docs",
            task_slice_id="docs-continuation-frontend",
            partition="train",
            path="frontend/package.json",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=12,
            region_id="workbench.docs",
            task_slice_id="docs-continuation-holdout",
            partition="holdout",
            path="README.md",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
    )


def _build_migrated_runtime() -> tuple[SeedRuntime, dict[str, object]]:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
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
    migration = runtime.migrate_structural_lineage_retention_policy(source.migrate_to_latest())
    return runtime, migration


def test_restart_continuation_consumes_only_new_evidence() -> None:
    runtime, migration = _build_migrated_runtime()
    checkpoint_root = Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    migrated_path = checkpoint_root / f"s30-migrated-{suffix}.pt"
    continued_path = checkpoint_root / f"s30-continued-{suffix}.pt"
    terminal_batch_id = "batch:terminal-lineage"
    try:
        runtime.save(migrated_path)
        restored = SeedRuntime.load(migrated_path)
        old_tick = restored.model.architecture.structural_runtime_tick
        old_revision = restored.model.architecture.structural_growth_scheduler_state.revision
        old_audit = restored.model.architecture.structural_lineage_retention_result

        evidence = _record_continuation_evidence(restored)
        schedule = restored.schedule_structural_candidate_batch_from_workbench_evidence(
            _continuation_requests()
        )
        assert all(item["outcome"]["status"] == "success" for item in evidence)
        assert schedule["status"] == "batch_created"
        assert restored.model.architecture.structural_runtime_tick > old_tick
        assert restored.model.architecture.structural_growth_scheduler_state.revision > old_revision
        assert terminal_batch_id not in {
            item.batch_id for item in restored.model.architecture.structural_candidate_batches
        }

        continuation_audit = restored.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=restored.model.architecture.structural_lineage_retention_policy.to_payload(),
        )
        assert continuation_audit["lineage_retention"] is not None
        assert continuation_audit["structural_runtime_tick"] == restored.model.architecture.structural_runtime_tick
        assert old_audit is not None
        assert continuation_audit["lineage_retention"]["result_digest"] != old_audit.result_digest

        restored.save(continued_path)
        resumed = SeedRuntime.load(continued_path)
        resumed_default = resumed.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
        )
        assert resumed_default["maintenance_results"] == []
        assert resumed_default["lineage_retention"] is None
        assert resumed.structural_maintenance_status()["last_retention_audit"] == (
            restored.structural_maintenance_status()["last_retention_audit"]
        )
        assert terminal_batch_id not in {
            item.batch_id for item in resumed.model.architecture.structural_candidate_batches
        }
        assert resumed.model.architecture.structural_growth_scheduler_state.revision == (
            restored.model.architecture.structural_growth_scheduler_state.revision
        )
    finally:
        migrated_path.unlink(missing_ok=True)
        continued_path.unlink(missing_ok=True)

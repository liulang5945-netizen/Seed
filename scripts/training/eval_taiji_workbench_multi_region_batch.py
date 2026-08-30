"""Run the R5C-S10 real-Workbench multi-region batch canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from seed_platform import workbench as workbench_module  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    ActionIntent,
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    StructuralGrowthDynamics,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-w7-r5c-s10-workbench-multi-region-batch-v1"


def _build_runtime() -> SeedRuntime:
    workbench_module.get_setting = (
        lambda key, default=None: str(PROJECT_ROOT) if key == "workspace_path" else default
    )
    runtime = SeedRuntime(Seed(episode_id="r5c-s10-workbench-multi-region"))
    runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
    for region_id, unit_ids, seed in (
        ("adaptive.cortex", ("u0", "u1"), 211),
        ("adaptive.memory", ("m0", "m1"), 212),
    ):
        runtime.model.architecture.attach_adaptive_neuron_region(
            AdaptiveNeuronRegion(
                region_id=region_id,
                input_dim=5,
                unit_ids=unit_ids,
                fan_in=2,
                generator=torch.Generator().manual_seed(seed),
            )
        )
    runtime.model.architecture.attach_structural_growth_controller(
        AdaptiveStructuralGrowthController(
            dynamics=StructuralGrowthDynamics(
                ema_rate=1.0,
                error_threshold=0.0,
                holdout_transfer_threshold=0.0,
                minimum_resource_state=0.0,
                required_error_steps=1,
            )
        )
    )
    return runtime


def _execute_observation(
    runtime: SeedRuntime,
    *,
    ordinal: int,
    region_id: str,
    task_slice_id: str,
    partition: str,
    path: str,
    prediction_error: float,
    holdout_transfer: float,
) -> dict[str, object]:
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
    result = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id=f"s10-workbench-read-{ordinal}",
            kind="workspace.read",
            parameters={"path": path},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
        structural_evidence={
            "network_id": "workbench",
            "region_id": region_id,
            "task_slice_id": task_slice_id,
            "partition": partition,
            "usage": 0.8,
            "resource_pressure": 0.2,
            "prediction_error": prediction_error,
            "learning_gain": 0.1,
            "holdout_transfer": holdout_transfer,
        },
    )
    outcome = result["outcome"]
    evidence = result.get("structural_evidence")
    if not isinstance(outcome, dict) or outcome.get("status") != "success":
        raise AssertionError(f"real Workbench outcome was not successful: {outcome}")
    if not isinstance(evidence, dict):
        raise AssertionError("successful Workbench outcome did not return structural evidence")
    sealed = runtime.model.architecture.seal_structural_evidence_window(
        "workbench",
        region_id,
        task_slice_id=task_slice_id,
        partition=partition,
    )
    if sealed is None:
        raise AssertionError(f"Workbench evidence window was not sealed for {region_id}/{task_slice_id}")
    return {
        "outcome": outcome,
        "evidence": evidence,
        "sealed_window": sealed.to_payload(),
    }


def _schedule_requests() -> tuple[dict[str, object], ...]:
    return (
        {
            "network_id": "workbench",
            "region_id": "workbench.code",
            "controller_region_id": "adaptive.cortex",
            "target_kind": "neuron",
            "operation": "add",
            "substrate_ids": ("adaptive.cortex",),
            "specification": {"region_id": "adaptive.cortex", "unit_id": "u2"},
        },
        {
            "network_id": "workbench",
            "region_id": "workbench.docs",
            "controller_region_id": "adaptive.memory",
            "target_kind": "neuron",
            "operation": "add",
            "substrate_ids": ("adaptive.memory",),
            "specification": {"region_id": "adaptive.memory", "unit_id": "m2"},
        },
    )


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = (
        _execute_observation(
            runtime,
            ordinal=1,
            region_id="workbench.code",
            task_slice_id="code-readme",
            partition="train",
            path="README.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=2,
            region_id="workbench.code",
            task_slice_id="code-config",
            partition="train",
            path="pyproject.toml",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=3,
            region_id="workbench.code",
            task_slice_id="code-holdout",
            partition="holdout",
            path="plans/README.md",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
        _execute_observation(
            runtime,
            ordinal=4,
            region_id="workbench.docs",
            task_slice_id="docs-roadmap",
            partition="train",
            path="plans/README.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=5,
            region_id="workbench.docs",
            task_slice_id="docs-frontend",
            partition="train",
            path="frontend/package.json",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=6,
            region_id="workbench.docs",
            task_slice_id="docs-holdout",
            partition="holdout",
            path="README.md",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
    )
    topology_before = tuple(
        (region.region_id, region.unit_ids) for region in runtime.model.architecture.neuron_regions
    )
    budget_before = runtime.model.architecture.cognitive_snapshot().development.structural_budget
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    checkpoint = runtime.model.architecture.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    repeated = restored.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    ).to_payload()
    batches = restored.structural_candidate_batches
    if not batches:
        raise AssertionError("real Workbench batch scheduler did not create a batch")
    batch = batches[-1]
    topology_after = tuple(
        (region.region_id, region.unit_ids) for region in restored.neuron_regions
    )
    budget_after = restored.cognitive_snapshot().development.structural_budget
    evidence_by_region = {
        region_id: [
            item
            for item in executions
            if item["evidence"]["observation"]["region_id"] == region_id
        ]
        for region_id in ("workbench.code", "workbench.docs")
    }
    candidate_ids = tuple(batch.candidate_ids)
    metrics = {
        "six_real_workbench_outcomes_succeeded": all(
            item["outcome"]["status"] == "success" for item in executions
        ),
        "outcome_digest_bound_to_each_evidence": all(
            bool(item["evidence"]["evidence"]["outcome_digest"])
            for item in executions
        ),
        "each_region_has_two_train_and_one_holdout_window": all(
            len(items) == 3
            and sum(item["evidence"]["observation"]["partition"] == "train" for item in items) == 2
            and sum(item["evidence"]["observation"]["partition"] == "holdout" for item in items) == 1
            for items in evidence_by_region.values()
        ),
        "two_distinct_candidates_created_from_real_regions": (
            schedule["status"] == "batch_created"
            and len(schedule["candidate_ids"]) == 2
            and len(set(schedule["candidate_ids"])) == 2
            and len(candidate_ids) == 2
        ),
        "batch_contains_both_real_region_candidates": (
            set(schedule["candidate_ids"]) == set(candidate_ids)
            and set(batch.selected_candidate_ids) == set(candidate_ids)
        ),
        "arbitration_keeps_topology_and_budget_unchanged": (
            topology_before == topology_after and budget_before == budget_after
        ),
        "source_windows_are_checkpointed": (
            len(schedule["source_window_digests"]) == 6
            and set(schedule["source_window_digests"])
            == set(restored.structural_workbench_batch_schedule_results[-1].source_window_digests)
        ),
        "batch_and_scheduler_restore": (
            restored.structural_workbench_batch_schedule_results[-1].batch_id == batch.batch_id
            and restored.structural_workbench_batch_schedule_results[-1].status == "batch_created"
        ),
        "repeated_multi_region_schedule_is_idempotent": repeated == schedule,
    }
    return {
        "format": REPORT_FORMAT,
        "executions": list(executions),
        "schedule": schedule,
        "repeated_schedule": repeated,
        "batch": batch.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "multiple real Workbench outcome streams must form distinct, checkpointable "
                "regional candidates in one deterministic batch without topology or budget mutation"
            ),
        },
        "boundary": (
            "This canary does not admit topology, invent metrics from provider output, expose "
            "structural controls to the frontend, expand budget, or make CUDA/CI claims."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s10_workbench_multi_region_batch_20260830.json",
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

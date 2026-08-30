"""Run the R5C-S6 real-Workbench structural-growth scheduler canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s6-workbench-structural-scheduler-v1"


def _build_runtime() -> SeedRuntime:
    workbench_module.get_setting = (
        lambda key, default=None: str(PROJECT_ROOT) if key == "workspace_path" else default
    )
    runtime = SeedRuntime(Seed(episode_id="r5c-s6-workbench"))
    runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
    region = AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(113),
    )
    runtime.model.architecture.attach_adaptive_neuron_region(region)
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
    task_slice_id: str,
    partition: str,
    prediction_error: float,
    holdout_transfer: float,
) -> dict[str, object]:
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
    result = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id=f"s6-workbench-read-{ordinal}",
            kind="workspace.read",
            parameters={"path": "README.md"},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
        structural_evidence={
            "network_id": "workbench",
            "region_id": "executor",
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
    if not isinstance(outcome, dict) or outcome.get("status") != "success":
        raise AssertionError(f"real Workbench outcome was not successful: {outcome}")
    evidence = result.get("structural_evidence")
    if not isinstance(evidence, dict):
        raise AssertionError("successful Workbench outcome did not return structural evidence")
    sealed = runtime.model.architecture.seal_structural_evidence_window(
        "workbench",
        "executor",
        task_slice_id=task_slice_id,
        partition=partition,
    )
    return {
        "outcome": outcome,
        "evidence": evidence,
        "sealed_window": sealed.to_payload(),
    }


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = (
        _execute_observation(
            runtime,
            ordinal=1,
            task_slice_id="task-a",
            partition="train",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=2,
            task_slice_id="task-b",
            partition="train",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=3,
            task_slice_id="task-a",
            partition="holdout",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
    )
    schedule = runtime.schedule_structural_growth_from_workbench_evidence(
        network_id="workbench",
        region_id="executor",
        controller_region_id="adaptive.cortex",
        target_kind="neuron",
        operation="add",
        substrate_ids=("adaptive.cortex",),
        specification={"region_id": "adaptive.cortex", "unit_id": "u2"},
    )
    if schedule.get("status") != "candidate_created":
        raise AssertionError(f"scheduler did not create a candidate: {schedule}")
    candidate_id = schedule.get("candidate_id")
    checkpoint = runtime.model.architecture.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    repeated = restored.schedule_structural_growth_from_evidence(
        network_id="workbench",
        region_id="executor",
        controller_region_id="adaptive.cortex",
        target_kind="neuron",
        operation="add",
        substrate_ids=("adaptive.cortex",),
        specification={"region_id": "adaptive.cortex", "unit_id": "u2"},
    ).to_payload()
    topology = restored.neuron_regions[0].unit_ids
    metrics = {
        "three_real_workbench_outcomes_succeeded": all(
            item["outcome"]["status"] == "success" for item in executions
        ),
        "structural_evidence_returned_on_each_outcome": all(
            bool(item["evidence"]) for item in executions
        ),
        "three_sealed_windows_recorded": len(restored.structural_evidence_summaries) == 3,
        "scheduler_created_candidate": schedule["status"] == "candidate_created",
        "candidate_is_bound_to_projection": bool(schedule.get("projection_digest")),
        "topology_remains_candidate_only": topology == ("u0", "u1"),
        "scheduler_state_checkpointed": (
            restored.structural_growth_scheduler_state.revision == 1
            and len(restored.structural_growth_scheduler_state.evaluated_window_digests) == 3
        ),
        "candidate_checkpointed": (
            len(restored.structural_proposal_candidates) == 1
            and restored.structural_proposal_candidates[0].candidate_id == candidate_id
        ),
        "repeated_schedule_is_idempotent": (
            repeated["status"] == "waiting"
            and repeated["reason"] == "no_new_sealed_window"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "executions": list(executions),
        "schedule": schedule,
        "repeated_schedule": repeated,
        "scheduler_state": restored.structural_growth_scheduler_state.to_payload(),
        "candidate": restored.structural_proposal_candidates[0].to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "real successful Workbench outcomes may supply explicit evaluator metrics to a "
                "content-addressed evidence ledger; the scheduler may queue one candidate after "
                "independent train and holdout windows, but it must not mutate topology"
            ),
        },
        "boundary": (
            "This canary does not admit the candidate, invent metrics from tool success, expand "
            "the structural budget, or make CUDA/CI claims."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s6_workbench_structural_scheduler_20260830.json",
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

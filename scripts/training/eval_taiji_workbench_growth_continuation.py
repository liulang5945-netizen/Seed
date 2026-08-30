"""Run the R5C-S7 Workbench candidate validation and admission canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_workbench_structural_scheduler import (  # noqa: E402
    _build_runtime,
    _execute_observation,
)
from taiji import AdaptiveNeuronRegion, TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s7-workbench-growth-continuation-v1"


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
    candidate_id = str(schedule["candidate_id"])
    model = runtime.model.architecture
    candidate = model.structural_proposal_candidates[0]
    proposal = model.materialize_structural_candidate(candidate.candidate_id)
    if proposal is None:
        raise AssertionError("scheduled candidate was not materialized")
    region = model.neuron_regions[0]
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(5)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    expected_activity = trial.step(holdout_input)
    workbench_evidence_id = executions[-1]["evidence"]["evidence"]["evidence_id"]
    budget_before = model.cognitive_snapshot().development.structural_budget
    continuation = runtime.continue_structural_candidate(
        candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected_activity,),
        retention_regression=0.0,
        lesion_effect=1.0,
        resource_state=0.8,
        evidence_ids=(str(workbench_evidence_id), "s7:retention", "s7:lesion"),
    )
    if continuation["status"] != "admitted":
        raise AssertionError(f"scheduled candidate did not complete lifecycle: {continuation}")
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    repeated = restored.continue_structural_candidate(
        candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected_activity,),
        retention_regression=0.0,
        lesion_effect=1.0,
        resource_state=0.8,
        evidence_ids=(str(workbench_evidence_id), "s7:retention", "s7:lesion"),
    )
    metrics = {
        "scheduler_candidate_was_created": schedule["status"] == "candidate_created",
        "shadow_validation_passed": continuation["validation"]["status"] == "validated",
        "policy_gate_passed": bool(continuation["decision"]["passed"]),
        "admission_completed": continuation["admission"]["status"] == "admitted",
        "topology_changed_only_after_admission": (
            model.neuron_regions[0].unit_ids == ("u0", "u1", "u2")
        ),
        "budget_debited_once": model.cognitive_snapshot().development.structural_budget == budget_before - candidate.resource_cost,
        "admission_checkpointed": (
            len(restored.structural_admission_results) == 1
            and restored.structural_admission_results[0].status == "admitted"
        ),
        "repeat_is_idempotent": (
            repeated["status"] == "admitted"
            and len(restored.structural_admission_results) == 1
            and restored.neuron_regions[0].unit_ids == ("u0", "u1", "u2")
        ),
    }
    metrics["restored_budget_matches"] = (
        restored.cognitive_snapshot().development.structural_budget
        == model.cognitive_snapshot().development.structural_budget
    )
    return {
        "format": REPORT_FORMAT,
        "schedule": schedule,
        "continuation": continuation,
        "repeated_continuation": repeated,
        "restored_topology": list(restored.neuron_regions[0].unit_ids),
        "restored_budget": restored.cognitive_snapshot().development.structural_budget,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "a candidate created from real Workbench evidence must pass the existing shadow, "
                "policy, and atomic admission lifecycle; restore and repeat must not duplicate "
                "the admission"
            ),
        },
        "boundary": (
            "The canary uses explicit holdout/retention/lesion metrics and inputs. It does not "
            "derive quality from tool success, expand budget, or bypass validation."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s7_workbench_growth_continuation_20260830.json",
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

"""Run the R5C-S11 real multi-region Workbench batch lifecycle canary."""

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
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _execute_observation,
    _schedule_requests,
)
from taiji import AdaptiveNeuronRegion, TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s11-workbench-multi-region-lifecycle-v1"


def _record_real_evidence(runtime: SeedRuntime) -> tuple[dict[str, object], ...]:
    return (
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


def _holdout_payload(model: TSKV8Adapter, candidate_id: str) -> dict[str, object]:
    candidate = next(
        item
        for item in model.structural_proposal_candidates
        if item.candidate_id == candidate_id
    )
    proposal = model.materialize_structural_candidate(candidate_id)
    if proposal is None:
        raise AssertionError(f"candidate {candidate_id} was not materialized")
    region_id = str(dict(candidate.specification)["region_id"])
    region = next(item for item in model.neuron_regions if item.region_id == region_id)
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(region.input_dim)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    return {
        "holdout_inputs": (holdout_input,),
        "expected_activities": (trial.step(holdout_input),),
        "retention_regression": 0.0,
        "lesion_effect": 1.0,
        "resource_state": 0.8,
        "evidence_ids": (f"s11:retention:{candidate_id}", f"s11:lesion:{candidate_id}"),
    }


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    executions = _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"real Workbench batch was not created: {schedule}")
    batch_id = str(schedule["batch_id"])
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    if len(batch.selected_candidate_ids) != 2:
        raise AssertionError(f"expected two selected candidates: {batch.to_payload()}")
    first_candidate_id, second_candidate_id = batch.selected_candidate_ids
    candidate_regions = {
        candidate.candidate_id: str(dict(candidate.specification)["region_id"])
        for candidate in runtime.model.architecture.structural_proposal_candidates
        if candidate.candidate_id in batch.selected_candidate_ids
    }
    second_region = candidate_regions[second_candidate_id]
    budget_before_admission = runtime.model.architecture.cognitive_snapshot().development.structural_budget
    topology_before_admission = tuple(
        (region.region_id, region.unit_ids) for region in runtime.model.architecture.neuron_regions
    )
    first_payload = _holdout_payload(runtime.model.architecture, first_candidate_id)
    first_continuation = runtime.continue_structural_candidate_batch(
        batch_id,
        continuations_by_candidate={first_candidate_id: first_payload},
    )
    first_batch = runtime.model.architecture.structural_candidate_batches[-1]
    checkpoint_after_first = runtime.model.architecture.native_checkpoint()
    after_first_topology = tuple(
        (region.region_id, region.unit_ids) for region in runtime.model.architecture.neuron_regions
    )
    restored_failure = TSKV8Adapter.from_native_checkpoint(checkpoint_after_first)
    failure_continuation = restored_failure.continue_structural_candidate_batch(
        batch_id,
        continuations_by_candidate={
            second_candidate_id: {
                "holdout_inputs": (),
                "expected_activities": (),
                "retention_regression": 0.0,
                "lesion_effect": 1.0,
                "resource_state": 0.8,
            }
        },
    )
    failure_topology = tuple(
        (region.region_id, region.unit_ids) for region in restored_failure.neuron_regions
    )

    restored_success = TSKV8Adapter.from_native_checkpoint(checkpoint_after_first)
    second_payload = _holdout_payload(restored_success, second_candidate_id)
    second_continuation = restored_success.continue_structural_candidate_batch(
        batch_id,
        continuations_by_candidate={second_candidate_id: second_payload},
    )
    admitted_topology = tuple(
        (region.region_id, region.unit_ids) for region in restored_success.neuron_regions
    )
    budget_after_admission = restored_success.cognitive_snapshot().development.structural_budget
    rollback = restored_success.rollback_structural_candidate_batch(
        batch_id,
        second_candidate_id,
    )
    rollback_topology = tuple(
        (region.region_id, region.unit_ids) for region in restored_success.neuron_regions
    )
    budget_after_rollback = restored_success.cognitive_snapshot().development.structural_budget
    rollback_checkpoint = restored_success.native_checkpoint()
    restored_rollback = TSKV8Adapter.from_native_checkpoint(rollback_checkpoint)
    repeated_rollback = restored_rollback.rollback_structural_candidate_batch(
        batch_id,
        second_candidate_id,
    )
    metrics = {
        "six_real_workbench_outcomes_succeeded": all(
            item["outcome"]["status"] == "success" for item in executions
        ),
        "real_batch_has_two_regions_and_two_candidates": (
            set(schedule["region_ids"]) == {"workbench.code", "workbench.docs"}
            and len(schedule["candidate_ids"]) == 2
        ),
        "first_candidate_admitted_before_checkpoint": (
            first_continuation["results"][first_candidate_id]["status"] == "admitted"
            and first_batch.state_by_candidate[first_candidate_id] == "admitted"
            and first_batch.state_by_candidate[second_candidate_id] == "reserved"
            and first_batch.reservation_remaining == 1
        ),
        "second_candidate_failure_isolated": (
            failure_continuation["results"][second_candidate_id]["status"] == "failed_closed"
            and dict(failure_topology) == dict(after_first_topology)
            and restored_failure.structural_candidate_batches[-1].state_by_candidate[
                first_candidate_id
            ]
            == "admitted"
        ),
        "second_candidate_admitted_after_independent_restore": (
            second_continuation["results"][second_candidate_id]["status"] == "admitted"
            and restored_success.structural_admission_results[-1].candidate_id
            == second_candidate_id
            and all(len(unit_ids) == 3 for _, unit_ids in admitted_topology)
        ),
        "admission_consumes_two_budget_units": (
            budget_after_admission == budget_before_admission - 2
        ),
        "rollback_restores_second_region_and_reopens_budget": (
            rollback["status"] == "rolled_back"
            and dict(rollback_topology)[second_region]
            == dict(topology_before_admission)[second_region]
            and budget_after_rollback == budget_before_admission - 1
        ),
        "rollback_is_checkpointed_and_idempotent": repeated_rollback == rollback,
        "failed_branch_does_not_change_first_region": (
            dict(failure_topology)["adaptive.cortex"]
            == dict(after_first_topology)["adaptive.cortex"]
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "executions": list(executions),
        "schedule": schedule,
        "first_continuation": first_continuation,
        "failure_continuation": failure_continuation,
        "second_continuation": second_continuation,
        "rollback": rollback,
        "repeated_rollback": repeated_rollback,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "a real multi-region Workbench batch must continue through independent "
                "shadow/policy/admission paths, isolate one candidate failure, and support "
                "checkpointed reversible rollback"
            ),
        },
        "boundary": (
            "This canary does not parallelize topology commits, expand structural budget, "
            "expose structural controls to frontend/provider, or make CUDA/CI claims."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s11_workbench_multi_region_lifecycle_20260830.json",
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

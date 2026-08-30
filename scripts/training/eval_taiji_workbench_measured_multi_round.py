"""Run the R5C-S16 multi-round measured evidence canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _execute_observation,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s16-workbench-measured-multi-round-v1"


def _record_round_two_evidence(runtime: SeedRuntime) -> tuple[dict[str, object], ...]:
    """Record a fresh set of Workbench observations with new task-slice identities."""

    return (
        _execute_observation(
            runtime,
            ordinal=7,
            region_id="workbench.code",
            task_slice_id="code-adapter-round-2",
            partition="train",
            path="taiji/adapter.py",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=8,
            region_id="workbench.code",
            task_slice_id="code-artifact-round-2",
            partition="train",
            path="taiji/structural_validation_artifact.py",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=9,
            region_id="workbench.code",
            task_slice_id="code-holdout-round-2",
            partition="holdout",
            path="plans/active/roadmap/03_CURRENT_EXECUTION.md",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
        _execute_observation(
            runtime,
            ordinal=10,
            region_id="workbench.docs",
            task_slice_id="docs-plan-round-2",
            partition="train",
            path="plans/active/roadmap/04_EXECUTION_PLAN.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=11,
            region_id="workbench.docs",
            task_slice_id="docs-status-round-2",
            partition="train",
            path="plans/reference/IMPLEMENTATION_STATUS_2026_08.md",
            prediction_error=0.8,
            holdout_transfer=0.0,
        ),
        _execute_observation(
            runtime,
            ordinal=12,
            region_id="workbench.docs",
            task_slice_id="docs-holdout-round-2",
            partition="holdout",
            path="README.md",
            prediction_error=0.1,
            holdout_transfer=0.9,
        ),
    )


def _round_two_schedule_requests() -> tuple[dict[str, object], ...]:
    """Use fresh structural target identities so round two is real growth."""

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


def _topology(model: TSKV8Adapter) -> dict[str, tuple[str, ...]]:
    return {region.region_id: region.unit_ids for region in model.neuron_regions}


def _budget(model: TSKV8Adapter) -> int:
    return int(model.cognitive_snapshot().development.structural_budget)


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()

    # Round one establishes a real parent lineage with two admitted regional
    # candidates.  The second candidate is deliberately re-measured after the
    # first admission, matching the sequential parent-binding contract.
    round_one_evidence = _record_real_evidence(runtime)
    round_one_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if round_one_schedule.get("status") != "batch_created":
        raise AssertionError(f"round one batch was not created: {round_one_schedule}")
    round_one_batch_id = str(round_one_schedule["batch_id"])
    round_one_batch = next(
        batch
        for batch in runtime.model.architecture.structural_candidate_batches
        if batch.batch_id == round_one_batch_id
    )
    if len(round_one_batch.selected_candidate_ids) != 2:
        raise AssertionError("round one must have two selected candidates")
    round_one_first_id, round_one_second_id = round_one_batch.selected_candidate_ids

    first_artifact, first_replay, first_measurements = _build_artifact(
        runtime.model.architecture,
        round_one_first_id,
        round_one_evidence,
    )
    first_result = runtime.continue_structural_candidate_batch_from_validation_artifacts(
        round_one_batch_id,
        artifacts_by_candidate={round_one_first_id: first_artifact},
        replays_by_candidate={round_one_first_id: first_replay},
    )
    round_one_second_artifact, round_one_second_replay, round_one_second_measurements = (
        _build_artifact(
            runtime.model.architecture,
            round_one_second_id,
            round_one_evidence,
        )
    )
    second_result = runtime.continue_structural_candidate_batch_from_validation_artifacts(
        round_one_batch_id,
        artifacts_by_candidate={round_one_second_id: round_one_second_artifact},
        replays_by_candidate={round_one_second_id: round_one_second_replay},
    )
    round_one_checkpoint = runtime.model.architecture.native_checkpoint()
    round_one_model = TSKV8Adapter.from_native_checkpoint(round_one_checkpoint)

    if second_result["batch"]["status"] != "completed":
        raise AssertionError(f"round one did not complete: {second_result}")

    # Round two must be driven by new Workbench windows and new target
    # identities, then start from round one's admitted checkpoint.
    runtime.model.substrate = round_one_model
    round_two_evidence = _record_round_two_evidence(runtime)
    round_two_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _round_two_schedule_requests()
    )
    if round_two_schedule.get("status") != "batch_created":
        raise AssertionError(f"round two batch was not created: {round_two_schedule}")
    round_two_batch_id = str(round_two_schedule["batch_id"])
    round_two_batch = next(
        batch
        for batch in runtime.model.architecture.structural_candidate_batches
        if batch.batch_id == round_two_batch_id
    )
    if len(round_two_batch.selected_candidate_ids) != 2:
        raise AssertionError(f"round two must have two selected candidates: {round_two_schedule}")
    round_two_first_id, round_two_second_id = round_two_batch.selected_candidate_ids

    round_two_pre_admission_checkpoint = runtime.model.architecture.native_checkpoint()
    round_two_first_model = TSKV8Adapter.from_native_checkpoint(
        round_two_pre_admission_checkpoint
    )
    round_two_first_artifact, round_two_first_replay, round_two_first_measurements = (
        _build_artifact(
            round_two_first_model,
            round_two_first_id,
            round_two_evidence,
            capacity_limit=8,
        )
    )
    # Keep a deliberately stale artifact from before the first round-two
    # admission.  It must fail after the parent checkpoint changes.
    stale_measurement_model = TSKV8Adapter.from_native_checkpoint(
        round_two_pre_admission_checkpoint
    )
    stale_round_two_second_artifact, stale_round_two_second_replay, _ = _build_artifact(
        stale_measurement_model,
        round_two_second_id,
        round_two_evidence,
        capacity_limit=8,
    )
    round_two_first_result = round_two_first_model.continue_structural_candidate_batch_from_validation_artifacts(
        round_two_batch_id,
        artifacts_by_candidate={round_two_first_id: round_two_first_artifact},
        replays_by_candidate={round_two_first_id: round_two_first_replay},
    )
    round_two_after_first_checkpoint = round_two_first_model.native_checkpoint()
    stale_branch = TSKV8Adapter.from_native_checkpoint(round_two_after_first_checkpoint)
    stale_topology_before = _topology(stale_branch)
    stale_budget_before = _budget(stale_branch)
    stale_result = stale_branch.continue_structural_candidate_batch_from_validation_artifacts(
        round_two_batch_id,
        artifacts_by_candidate={round_two_second_id: stale_round_two_second_artifact},
        replays_by_candidate={round_two_second_id: stale_round_two_second_replay},
    )
    stale_topology_after = _topology(stale_branch)
    stale_budget_after = _budget(stale_branch)

    # Rebuild the second artifact against the new parent and complete round two
    # on the success branch.
    round_two_success_model = TSKV8Adapter.from_native_checkpoint(
        round_two_after_first_checkpoint
    )
    round_two_second_artifact, round_two_second_replay, round_two_second_measurements = (
        _build_artifact(
            round_two_success_model,
            round_two_second_id,
            round_two_evidence,
            capacity_limit=8,
        )
    )
    round_two_second_result = round_two_success_model.continue_structural_candidate_batch_from_validation_artifacts(
        round_two_batch_id,
        artifacts_by_candidate={round_two_second_id: round_two_second_artifact},
        replays_by_candidate={round_two_second_id: round_two_second_replay},
    )
    round_two_success_checkpoint = round_two_success_model.native_checkpoint()
    round_two_success_topology = _topology(round_two_success_model)
    round_two_success_budget = _budget(round_two_success_model)
    round_two_after_first_topology = _topology(
        TSKV8Adapter.from_native_checkpoint(round_two_after_first_checkpoint)
    )

    rollback = round_two_success_model.rollback_structural_candidate_batch(
        round_two_batch_id,
        round_two_second_id,
    )
    rollback_topology = _topology(round_two_success_model)
    rollback_budget = _budget(round_two_success_model)
    rollback_checkpoint = round_two_success_model.native_checkpoint()
    restored_after_rollback = TSKV8Adapter.from_native_checkpoint(rollback_checkpoint)
    repeated_rollback = restored_after_rollback.rollback_structural_candidate_batch(
        round_two_batch_id,
        round_two_second_id,
    )

    round_one_topology = _topology(round_one_model)
    round_one_budget = _budget(round_one_model)
    round_two_initial_topology = _topology(
        TSKV8Adapter.from_native_checkpoint(round_one_checkpoint)
    )
    round_two_initial_budget = _budget(TSKV8Adapter.from_native_checkpoint(round_one_checkpoint))
    metrics = {
        "round_one_real_evidence_and_admission": (
            all(item["outcome"]["status"] == "success" for item in round_one_evidence)
            and first_result["results"][round_one_first_id]["status"] == "admitted"
            and second_result["results"][round_one_second_id]["status"] == "admitted"
            and second_result["batch"]["status"] == "completed"
        ),
        "round_two_uses_fresh_windows_and_candidates": (
            all(item["outcome"]["status"] == "success" for item in round_two_evidence)
            and set(round_two_schedule["source_window_digests"]).isdisjoint(
                set(round_one_schedule["source_window_digests"])
            )
            and set(round_two_schedule["candidate_ids"]).isdisjoint(
                set(round_one_schedule["candidate_ids"])
            )
            and round_two_schedule["batch_id"] != round_one_schedule["batch_id"]
        ),
        "round_two_parent_starts_from_round_one_checkpoint": (
            round_two_initial_topology == round_one_topology
            and round_two_initial_budget == round_one_budget
        ),
        "stale_artifact_fails_closed_after_parent_changes": (
            stale_result["results"][round_two_second_id]["status"] == "failed_closed"
            and stale_result["batch"]["candidate_states"][round_two_first_id] == "admitted"
            and stale_result["batch"]["candidate_states"][round_two_second_id]
            == "failed_closed"
            and stale_topology_after == stale_topology_before
            and stale_budget_after == stale_budget_before
        ),
        "round_two_remeasured_artifact_completes": (
            round_two_first_result["results"][round_two_first_id]["status"] == "admitted"
            and round_two_second_result["results"][round_two_second_id]["status"]
            == "admitted"
            and round_two_second_result["batch"]["status"] == "completed"
            and round_two_second_result["artifact_batch"]["complete"]
            and round_two_first_artifact.artifact_digest
            != round_two_second_artifact.artifact_digest
        ),
        "measurements_remain_owner_derived": (
            round_two_first_artifact.holdout_gain == round_two_first_measurements.holdout_gain
            and round_two_second_artifact.holdout_gain
            == round_two_second_measurements.holdout_gain
            and round_two_first_artifact.resource_measurement_digest
            == round_two_first_measurements.resource_measurement_digest
            and round_two_second_artifact.resource_measurement_digest
            == round_two_second_measurements.resource_measurement_digest
        ),
        "round_two_rollback_restores_only_latest_region_and_budget": (
            rollback["status"] == "rolled_back"
            and rollback_budget == round_two_success_budget + round_two_second_artifact.resource_cost
            and rollback_topology == round_two_after_first_topology
        ),
        "rollback_checkpoint_is_idempotent": repeated_rollback == rollback,
        "round_one_lineage_remains_visible_after_round_two": (
            len(round_two_success_model.structural_validation_artifact_batches) >= 2
            and len(round_two_success_model.structural_admission_results) >= 4
            and bool(round_two_success_checkpoint)
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "round_one": {
            "evidence": list(round_one_evidence),
            "schedule": round_one_schedule,
            "first_result": first_result,
            "second_result": second_result,
            "first_measurements": first_measurements.to_payload(),
            "second_measurements": round_one_second_measurements.to_payload(),
            "first_artifact": first_artifact.to_payload(),
            "second_artifact": round_one_second_artifact.to_payload(),
        },
        "round_two": {
            "evidence": list(round_two_evidence),
            "schedule": round_two_schedule,
            "first_result": round_two_first_result,
            "stale_result": stale_result,
            "second_result": round_two_second_result,
            "first_measurements": round_two_first_measurements.to_payload(),
            "second_measurements": round_two_second_measurements.to_payload(),
            "first_artifact": round_two_first_artifact.to_payload(),
            "second_artifact": round_two_second_artifact.to_payload(),
            "rollback": rollback,
            "repeated_rollback": repeated_rollback,
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "at least two real Workbench evidence rounds must create fresh measured "
                "artifacts, reject stale parent lineage, complete admission, and support "
                "checkpointed reversible rollback"
            ),
        },
        "boundary": (
            "This canary proves bounded multi-round evidence/artifact lifecycle and rollback; "
            "it does not claim open-domain quality, unlimited growth, CUDA, or CI completion."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s16_workbench_measured_multi_round_20260830.json",
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

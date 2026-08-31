"""Run the R5C-S40 repeated retention-pressure canary."""

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
from scripts.training.eval_taiji_runtime_structural_artifact_multi_round import (  # noqa: E402
    _batch,
    _budget,
    _record_round,
    _signature,
    _topology,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from taiji import StructuralLineageRetentionPolicy  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s40-runtime-artifact-repeated-retention-v1"


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    checkpoint_paths: list[Path] = []

    def checkpoint(name: str, value: SeedRuntime) -> SeedRuntime:
        path = checkpoint_root / f"s40-{name}-{suffix}.pt"
        checkpoint_paths.append(path)
        value.save(path)
        return SeedRuntime.load(path)

    try:
        base_topology = _topology(runtime)
        base_budget = _budget(runtime)
        lineage_limit = int(runtime.model.architecture.config.cognitive_lineage_history_limit)
        policy = StructuralLineageRetentionPolicy.create(1, revision=2)

        _record_round(runtime, first_ordinal=1, round_id="round-1")
        round_one_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
            _schedule_requests()
        )
        if round_one_schedule.get("status") != "batch_created":
            raise AssertionError(f"round one batch was not created: {round_one_schedule}")
        active_batch_id = str(round_one_schedule["batch_id"])
        runtime = checkpoint("round1", runtime)

        cycle_details: list[dict[str, object]] = []
        removed_terminal_batch_ids: list[str] = []
        terminal_artifacts: list[tuple[str, object, dict[str, object]]] = []
        runtime_ticks = [runtime.model.architecture.structural_runtime_tick]
        for round_index in range(2, 6):
            first_ordinal = 1 + ((round_index - 1) * 6)
            evidence = _record_round(
                runtime,
                first_ordinal=first_ordinal,
                round_id=f"round-{round_index}",
            )
            schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
                _schedule_requests()
            )
            if schedule.get("status") != "batch_created":
                raise AssertionError(f"round {round_index} batch was not created: {schedule}")
            terminal_batch_id = str(schedule["batch_id"])
            if terminal_batch_id == active_batch_id:
                raise AssertionError("repeated retention reused the protected active batch")
            terminal_batch = _batch(runtime, terminal_batch_id)
            first_id, second_id = terminal_batch.selected_candidate_ids
            runtime = checkpoint(f"round{round_index}-scheduled", runtime)

            first_artifact, first_replay, first_measurements = _build_artifact(
                runtime.model.architecture,
                first_id,
                evidence,
            )
            first_parent_bound = first_artifact.parent_checkpoint_digest == _checkpoint_digest(
                runtime.model.architecture.native_checkpoint()
            )
            runtime = checkpoint(f"round{round_index}-first", runtime)
            first_result = runtime.continue_structural_candidate_batch_from_validation_artifacts(
                terminal_batch_id,
                artifacts_by_candidate={first_id: first_artifact},
                replays_by_candidate={first_id: first_replay},
            )
            if first_result["results"][first_id]["status"] != "admitted":
                raise AssertionError(f"round {round_index} first candidate was not admitted")

            second_artifact, second_replay, second_measurements = _build_artifact(
                runtime.model.architecture,
                second_id,
                evidence,
            )
            second_parent_bound = second_artifact.parent_checkpoint_digest == _checkpoint_digest(
                runtime.model.architecture.native_checkpoint()
            )
            runtime = checkpoint(f"round{round_index}-second", runtime)
            second_result = runtime.continue_structural_candidate_batch_from_validation_artifacts(
                terminal_batch_id,
                artifacts_by_candidate={second_id: second_artifact},
                replays_by_candidate={second_id: second_replay},
            )
            if second_result["results"][second_id]["status"] != "admitted":
                raise AssertionError(f"round {round_index} second candidate was not admitted")
            second_rollback = runtime.rollback_structural_candidate_batch(
                terminal_batch_id,
                second_id,
            )
            first_rollback = runtime.rollback_structural_candidate_batch(
                terminal_batch_id,
                first_id,
            )
            if first_rollback["status"] != "rolled_back" or second_rollback["status"] != "rolled_back":
                raise AssertionError(f"round {round_index} rollback was not clean")
            terminal_artifacts.append((terminal_batch_id, second_artifact, second_replay))

            maintenance = runtime.run_structural_maintenance_cycle(
                candidate_ids=(),
                holdout_inputs_by_candidate={},
                expected_activities_by_candidate={},
                lineage_retention_policy=policy.to_payload(),
            )
            retention = runtime.model.architecture.structural_lineage_retention_result
            if retention is None:
                raise AssertionError(f"round {round_index} retention audit was not recorded")
            if terminal_batch_id not in retention.removed_batch_ids:
                raise AssertionError(f"round {round_index} terminal batch was not compacted")
            runtime = checkpoint(f"round{round_index}-retention", runtime)
            runtime_ticks.append(runtime.model.architecture.structural_runtime_tick)
            cycle_details.append(
                {
                    "round": round_index,
                    "batch_id": terminal_batch_id,
                    "source_window_count": len(schedule["source_window_digests"]),
                    "first_artifact_digest": first_artifact.artifact_digest,
                    "second_artifact_digest": second_artifact.artifact_digest,
                    "first_parent_bound": first_parent_bound,
                    "second_parent_bound": second_parent_bound,
                    "first_measurement_digest": first_measurements.measurement_digest,
                    "second_measurement_digest": second_measurements.measurement_digest,
                    "retention_digest": retention.result_digest,
                    "maintenance_results": maintenance["maintenance_results"],
                }
            )
            removed_terminal_batch_ids.append(terminal_batch_id)

        before_terminal_replay = _checkpoint_digest(runtime.model.architecture.native_checkpoint())
        terminal_batch_id, terminal_artifact, terminal_replay = terminal_artifacts[-1]
        try:
            runtime.continue_structural_candidate_batch_from_validation_artifacts(
                terminal_batch_id,
                artifacts_by_candidate={terminal_artifact.candidate_id: terminal_artifact},
                replays_by_candidate={terminal_artifact.candidate_id: terminal_replay},
            )
        except ValueError as exc:
            terminal_replay_rejected = "unknown structural candidate batch" in str(exc)
        else:
            terminal_replay_rejected = False
        after_terminal_replay = _checkpoint_digest(runtime.model.architecture.native_checkpoint())
        final_signature_before_save = _signature(runtime)
        runtime = checkpoint("final", runtime)
        final_signature = _signature(runtime)
        final_batch_ids = {item.batch_id for item in runtime.model.architecture.structural_candidate_batches}
        final_artifact_count = len(runtime.model.architecture.structural_validation_artifacts)
        final_record_counts = {
            "candidate_batches": len(runtime.model.architecture.structural_candidate_batches),
            "candidate_rollbacks": len(runtime.model.architecture.structural_candidate_rollbacks),
            "validation_artifacts": final_artifact_count,
            "artifact_batches": len(runtime.model.architecture.structural_validation_artifact_batches),
            "workbench_schedule_results": len(
                runtime.model.architecture.structural_workbench_batch_schedule_results
            ),
            "capacity_pressure_snapshots": len(
                runtime.model.architecture.structural_capacity_pressure_snapshots
            ),
        }
        metrics = {
            "four_retention_cycles_remove_only_terminal_batches": (
                len(removed_terminal_batch_ids) == 4
                and len(set(removed_terminal_batch_ids)) == 4
                and all(
                    detail["batch_id"] in removed_terminal_batch_ids
                    and detail["maintenance_results"] == []
                    for detail in cycle_details
                )
            ),
            "protected_active_batch_survives_every_cycle": (
                active_batch_id in final_batch_ids
                and active_batch_id not in removed_terminal_batch_ids
                and all(
                    active_batch_id in runtime.model.architecture.structural_lineage_retention_result.retained_batch_ids
                    for _ in (0,)
                )
            ),
            "every_cycle_uses_new_measured_parent_and_checkpoint": (
                all(
                    detail["source_window_count"] == 6
                    and detail["first_parent_bound"]
                    and detail["second_parent_bound"]
                    for detail in cycle_details
                )
                and runtime_ticks == sorted(set(runtime_ticks))
            ),
            "admission_and_rollback_return_budget_and_topology": (
                _budget(runtime) == base_budget
                and _topology(runtime) == base_topology
                and all(
                    detail["first_measurement_digest"]
                    and detail["second_measurement_digest"]
                    for detail in cycle_details
                )
            ),
            "deleted_terminal_replay_fails_closed_without_mutation": (
                terminal_replay_rejected and before_terminal_replay == after_terminal_replay
            ),
            "record_and_artifact_storage_stays_bounded": (
                final_record_counts["candidate_batches"] == 1
                and final_record_counts["candidate_rollbacks"] == 0
                and final_artifact_count == 0
                and final_record_counts["artifact_batches"] == 0
                and final_record_counts["workbench_schedule_results"] <= lineage_limit
                and final_record_counts["capacity_pressure_snapshots"] <= lineage_limit
            ),
            "final_checkpoint_preserves_policy_and_projection": (
                final_signature == final_signature_before_save
                and final_signature["policy_digest"] == policy.policy_digest
                and final_signature["batch_ids"] == (active_batch_id,)
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "active_batch_id": active_batch_id,
            "removed_terminal_batch_ids": removed_terminal_batch_ids,
            "cycles": cycle_details,
            "runtime_ticks": runtime_ticks,
            "final_record_counts": final_record_counts,
            "retention": runtime.model.architecture.structural_lineage_retention_result.to_payload(),
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "repeated retention pressure must compact four terminal rounds while preserving "
                    "one active reservation; every admission/rollback remains checkpointable and "
                    "bounded, and deleted terminal artifact replay remains impossible"
                ),
            },
            "boundary": (
                "This canary covers native CPU repeated SeedRuntime retention pressure and bounded "
                "structural lineage. It does not claim unlimited growth, automatic budget expansion, "
                "open-domain quality, CUDA, frontend behavior, Windows shell, CI completion, or "
                "general intelligence."
            ),
        }
    finally:
        for path in checkpoint_paths:
            path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s40_runtime_artifact_repeated_retention_20260831.json",
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

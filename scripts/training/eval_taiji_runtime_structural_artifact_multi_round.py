"""Run the R5C-S38 multi-round SeedRuntime artifact/retention canary."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

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
from taiji import StructuralLineageRetentionPolicy  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s38-runtime-artifact-multi-round-v1"


def _record_round(
    runtime: SeedRuntime,
    *,
    first_ordinal: int,
    round_id: str,
) -> tuple[dict[str, object], ...]:
    rows = (
        (
            first_ordinal,
            "workbench.code",
            f"{round_id}-code-read",
            "train",
            "README.md",
            0.8,
            0.0,
        ),
        (
            first_ordinal + 1,
            "workbench.code",
            f"{round_id}-code-config",
            "train",
            "pyproject.toml",
            0.8,
            0.0,
        ),
        (
            first_ordinal + 2,
            "workbench.code",
            f"{round_id}-code-holdout",
            "holdout",
            "plans/README.md",
            0.1,
            0.9,
        ),
        (
            first_ordinal + 3,
            "workbench.docs",
            f"{round_id}-docs-roadmap",
            "train",
            "plans/README.md",
            0.8,
            0.0,
        ),
        (
            first_ordinal + 4,
            "workbench.docs",
            f"{round_id}-docs-frontend",
            "train",
            "frontend/package.json",
            0.8,
            0.0,
        ),
        (
            first_ordinal + 5,
            "workbench.docs",
            f"{round_id}-docs-holdout",
            "holdout",
            "README.md",
            0.1,
            0.9,
        ),
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
        for (
            ordinal,
            region_id,
            task_slice_id,
            partition,
            path,
            prediction_error,
            holdout_transfer,
        ) in rows
    )


def _batch(runtime: SeedRuntime, batch_id: str) -> Any:
    return next(
        item
        for item in runtime.model.architecture.structural_candidate_batches
        if item.batch_id == batch_id
    )


def _topology(runtime: SeedRuntime) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (region.region_id, region.unit_ids)
        for region in runtime.model.architecture.neuron_regions
    )


def _budget(runtime: SeedRuntime) -> int:
    return int(runtime.model.architecture.cognitive_snapshot().development.structural_budget)


def _signature(runtime: SeedRuntime) -> dict[str, object]:
    architecture = runtime.model.architecture
    return {
        "runtime_tick": architecture.structural_runtime_tick,
        "scheduler_revision": architecture.structural_growth_scheduler_state.revision,
        "topology": _topology(runtime),
        "budget": _budget(runtime),
        "batch_ids": tuple(item.batch_id for item in architecture.structural_candidate_batches),
        "artifact_digests": tuple(
            item.artifact_digest for item in architecture.structural_validation_artifacts
        ),
        "policy_digest": (
            None
            if architecture.structural_lineage_retention_policy is None
            else architecture.structural_lineage_retention_policy.policy_digest
        ),
    }


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    paths = {
        name: checkpoint_root / f"s38-{name}-{suffix}.pt"
        for name in (
            "round1",
            "round2-pre",
            "round2-first",
            "round2-second",
            "round2-done",
            "round3-pre",
            "round3-measured",
            "round3-final",
        )
    }
    try:
        # Round one creates a protected live batch and proves its first restart.
        round_one_evidence = _record_round(runtime, first_ordinal=1, round_id="round-1")
        round_one_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
            _schedule_requests()
        )
        if round_one_schedule.get("status") != "batch_created":
            raise AssertionError(f"round one batch was not created: {round_one_schedule}")
        round_one_batch_id = str(round_one_schedule["batch_id"])
        round_one_before_restart = _signature(runtime)
        runtime.save(paths["round1"])
        round_one_restored = SeedRuntime.load(paths["round1"])
        round_one_restart_stable = _signature(round_one_restored) == round_one_before_restart

        # Round two starts only from new windows.  Both candidates are admitted,
        # then rolled back, leaving a terminal batch eligible for retention.
        round_two_evidence = _record_round(
            round_one_restored,
            first_ordinal=7,
            round_id="round-2",
        )
        round_two_schedule = round_one_restored.schedule_structural_candidate_batch_from_workbench_evidence(
            _schedule_requests()
        )
        if round_two_schedule.get("status") != "batch_created":
            raise AssertionError(f"round two batch was not created: {round_two_schedule}")
        round_two_batch_id = str(round_two_schedule["batch_id"])
        round_two_batch = _batch(round_one_restored, round_two_batch_id)
        round_two_first_id, round_two_second_id = round_two_batch.selected_candidate_ids
        round_one_restored.save(paths["round2-pre"])
        round_two_parent = SeedRuntime.load(paths["round2-pre"])

        # Produce a stale second artifact on an isolated branch before the first
        # admission.  Its failure must not poison the success branch.
        stale_branch = SeedRuntime.load(paths["round2-pre"])
        stale_artifact, stale_replay, _ = _build_artifact(
            stale_branch.model.architecture,
            round_two_second_id,
            round_two_evidence,
        )

        first_artifact, first_replay, first_measurements = _build_artifact(
            round_two_parent.model.architecture,
            round_two_first_id,
            round_two_evidence,
        )
        round_two_parent.save(paths["round2-first"])
        round_two_after_first_measurement = SeedRuntime.load(paths["round2-first"])
        first_result = round_two_after_first_measurement.continue_structural_candidate_batch_from_validation_artifacts(
            round_two_batch_id,
            artifacts_by_candidate={round_two_first_id: first_artifact},
            replays_by_candidate={round_two_first_id: first_replay},
        )

        stale_isolated = SeedRuntime.load(paths["round2-first"])
        stale_topology = _topology(stale_isolated)
        stale_budget = _budget(stale_isolated)
        stale_result = stale_isolated.continue_structural_candidate_batch_from_validation_artifacts(
            round_two_batch_id,
            artifacts_by_candidate={round_two_second_id: stale_artifact},
            replays_by_candidate={round_two_second_id: stale_replay},
        )
        stale_failed_without_mutation = (
            stale_result["results"][round_two_second_id]["status"] == "failed_closed"
            and "parent checkpoint" in stale_result["results"][round_two_second_id]["reason"]
            and _topology(stale_isolated) == stale_topology
            and _budget(stale_isolated) == stale_budget
        )

        second_artifact, second_replay, second_measurements = _build_artifact(
            round_two_after_first_measurement.model.architecture,
            round_two_second_id,
            round_two_evidence,
        )
        round_two_after_first_measurement.save(paths["round2-second"])
        round_two_before_second_admission = SeedRuntime.load(paths["round2-second"])
        second_result = round_two_before_second_admission.continue_structural_candidate_batch_from_validation_artifacts(
            round_two_batch_id,
            artifacts_by_candidate={round_two_second_id: second_artifact},
            replays_by_candidate={round_two_second_id: second_replay},
        )
        second_rollback = round_two_before_second_admission.rollback_structural_candidate_batch(
            round_two_batch_id,
            round_two_second_id,
        )
        first_rollback = round_two_before_second_admission.rollback_structural_candidate_batch(
            round_two_batch_id,
            round_two_first_id,
        )
        repeated_rollback = round_two_before_second_admission.continue_structural_candidate_batch_from_validation_artifacts(
            round_two_batch_id,
            artifacts_by_candidate={round_two_second_id: second_artifact},
            replays_by_candidate={round_two_second_id: second_replay},
        )
        round_two_terminal = _batch(round_two_before_second_admission, round_two_batch_id)
        round_two_before_restart = _signature(round_two_before_second_admission)
        round_two_before_second_admission.save(paths["round2-done"])
        round_two_restored = SeedRuntime.load(paths["round2-done"])
        round_two_restart_signature = _signature(round_two_restored)

        # Round three remains live after a candidate-level tamper failure.  This
        # gives retention two protected rounds and one removable terminal round.
        round_three_evidence = _record_round(
            round_two_restored,
            first_ordinal=13,
            round_id="round-3",
        )
        round_three_schedule = round_two_restored.schedule_structural_candidate_batch_from_workbench_evidence(
            _schedule_requests()
        )
        if round_three_schedule.get("status") != "batch_created":
            raise AssertionError(f"round three batch was not created: {round_three_schedule}")
        round_three_batch_id = str(round_three_schedule["batch_id"])
        round_three_batch = _batch(round_two_restored, round_three_batch_id)
        round_three_first_id = round_three_batch.selected_candidate_ids[0]
        round_three_second_id = round_three_batch.selected_candidate_ids[1]
        round_two_restored.save(paths["round3-pre"])
        round_three_parent = SeedRuntime.load(paths["round3-pre"])
        round_three_artifact, round_three_replay, round_three_measurements = _build_artifact(
            round_three_parent.model.architecture,
            round_three_first_id,
            round_three_evidence,
        )
        round_three_parent.save(paths["round3-measured"])
        round_three_measured = SeedRuntime.load(paths["round3-measured"])
        malformed = copy.deepcopy(round_three_artifact.to_payload())
        malformed["measurement_digest"] = "0" * 64
        malformed_result = round_three_measured.continue_structural_candidate_batch_from_validation_artifacts(
            round_three_batch_id,
            artifacts_by_candidate={round_three_first_id: malformed},
            replays_by_candidate={round_three_first_id: round_three_replay},
        )
        round_three_after_failure = _batch(round_three_measured, round_three_batch_id)
        round_three_live_after_failure = (
            malformed_result["results"][round_three_first_id]["status"] == "failed_closed"
            and round_three_after_failure.state_by_candidate[round_three_first_id]
            == "failed_closed"
            and round_three_after_failure.state_by_candidate[round_three_second_id] == "reserved"
        )

        policy = StructuralLineageRetentionPolicy.create(2, revision=2)
        retention = round_three_measured.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=policy.to_payload(),
        )
        retention_result = round_three_measured.model.architecture.structural_lineage_retention_result
        if retention_result is None:
            raise AssertionError("S38 retention audit was not recorded")
        before_replay = _checkpoint_digest(round_three_measured.model.architecture.native_checkpoint())
        try:
            round_three_measured.continue_structural_candidate_batch_from_validation_artifacts(
                round_two_batch_id,
                artifacts_by_candidate={round_two_second_id: second_artifact},
                replays_by_candidate={round_two_second_id: second_replay},
            )
        except ValueError as exc:
            terminal_replay_rejected = "unknown structural candidate batch" in str(exc)
        else:
            terminal_replay_rejected = False
        after_replay = _checkpoint_digest(round_three_measured.model.architecture.native_checkpoint())
        round_three_measured.save(paths["round3-final"])
        final = SeedRuntime.load(paths["round3-final"])
        final_batch_ids = {item.batch_id for item in final.model.architecture.structural_candidate_batches}
        final_signature = _signature(final)
        metrics = {
            "three_rounds_use_fresh_evidence_and_batches": (
                all(item["outcome"]["status"] == "success" for item in round_one_evidence)
                and all(item["outcome"]["status"] == "success" for item in round_two_evidence)
                and all(item["outcome"]["status"] == "success" for item in round_three_evidence)
                and len(
                    {
                        round_one_batch_id,
                        round_two_batch_id,
                        round_three_batch_id,
                    }
                )
                == 3
                and set(round_one_schedule["source_window_digests"]).isdisjoint(
                    round_two_schedule["source_window_digests"]
                )
                and set(round_two_schedule["source_window_digests"]).isdisjoint(
                    round_three_schedule["source_window_digests"]
                )
            ),
            "round_one_save_load_preserves_live_lineage": round_one_restart_stable,
            "round_two_save_load_and_measured_artifacts_are_owner_derived": (
                round_two_before_restart == round_two_restart_signature
                and first_artifact.holdout_gain == first_measurements.holdout_gain
                and second_artifact.holdout_gain == second_measurements.holdout_gain
                and first_artifact.resource_measurement_digest
                == first_measurements.resource_measurement_digest
                and second_artifact.resource_measurement_digest
                == second_measurements.resource_measurement_digest
            ),
            "stale_artifact_isolated_without_mutating_success_branch": stale_failed_without_mutation,
            "round_two_rollback_and_repeat_are_idempotent": (
                round_two_terminal.status == "completed"
                and first_result["results"][round_two_first_id]["status"] == "admitted"
                and second_result["results"][round_two_second_id]["status"] == "admitted"
                and first_rollback["status"] == "rolled_back"
                and second_rollback["status"] == "rolled_back"
                and repeated_rollback["results"][round_two_second_id]["status"] == "rolled_back"
            ),
            "round_three_malformed_candidate_failure_preserves_sibling_reservation": (
                round_three_live_after_failure
            ),
            "retention_removes_only_terminal_round_two_subgraph": (
                retention["maintenance_results"] == []
                and retention_result.status == "compacted"
                and retention_result.retention_pressure is False
                and round_two_batch_id in retention_result.removed_batch_ids
                and round_one_batch_id in retention_result.protected_batch_ids
                and round_three_batch_id in retention_result.protected_batch_ids
                and round_one_batch_id in final_batch_ids
                and round_three_batch_id in final_batch_ids
                and round_two_batch_id not in final_batch_ids
                and not (
                    {first_artifact.artifact_digest, second_artifact.artifact_digest}
                    & {
                        item.artifact_digest
                        for item in final.model.architecture.structural_validation_artifacts
                    }
                )
            ),
            "old_terminal_artifact_replay_fails_closed_without_mutation": (
                terminal_replay_rejected and before_replay == after_replay
            ),
            "round_three_checkpoint_preserves_policy_cursor_budget_and_topology": (
                final_signature["policy_digest"] == policy.policy_digest
                and final_signature["runtime_tick"]
                == round_three_measured.model.architecture.structural_runtime_tick
                and final_signature["topology"] == _topology(round_three_measured)
                and final_signature["budget"] == _budget(round_three_measured)
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "rounds": {
                "round_one": {
                    "batch_id": round_one_batch_id,
                    "evidence_count": len(round_one_evidence),
                    "source_window_digests": list(round_one_schedule["source_window_digests"]),
                },
                "round_two": {
                    "batch_id": round_two_batch_id,
                    "candidate_ids": list(round_two_batch.selected_candidate_ids),
                    "artifact_digests": [
                        first_artifact.artifact_digest,
                        second_artifact.artifact_digest,
                    ],
                },
                "round_three": {
                    "batch_id": round_three_batch_id,
                    "candidate_ids": list(round_three_batch.selected_candidate_ids),
                    "malformed_candidate_id": round_three_first_id,
                    "live_sibling_candidate_id": round_three_second_id,
                },
            },
            "retention": retention_result.to_payload(),
            "diagnostics": {
                "round_two_signature_before_restart": round_two_before_restart,
                "round_two_signature_after_restart": round_two_restart_signature,
                "round_two_owner_checks": {
                    "first_holdout_gain": first_artifact.holdout_gain
                    == first_measurements.holdout_gain,
                    "second_holdout_gain": second_artifact.holdout_gain
                    == second_measurements.holdout_gain,
                    "first_resource_digest": first_artifact.resource_measurement_digest
                    == first_measurements.resource_measurement_digest,
                    "second_resource_digest": second_artifact.resource_measurement_digest
                    == second_measurements.resource_measurement_digest,
                },
            },
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "three fresh Workbench evidence rounds must remain independently checkpointable; "
                    "measured artifacts, stale/tampered failure, rollback/repeat, and terminal-only "
                    "retention must preserve lineage, budget, topology, cursor, and policy boundaries"
                ),
            },
            "boundary": (
                "This canary covers native CPU multi-round SeedRuntime artifact lifecycle and bounded "
                "retention. It does not claim unlimited growth, automatic budget expansion, open-domain "
                "quality, CUDA, frontend behavior, Windows shell, CI completion, or general intelligence."
            ),
        }
    finally:
        for path in paths.values():
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
        / "taiji_w7_r5c_s38_runtime_artifact_multi_round_20260831.json",
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

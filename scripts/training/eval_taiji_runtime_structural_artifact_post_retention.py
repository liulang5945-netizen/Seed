"""Run the R5C-S39 post-retention active-lineage continuation canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s39-runtime-artifact-post-retention-v1"


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    paths = {
        name: checkpoint_root / f"s39-{name}-{suffix}.pt"
        for name in (
            "round1",
            "round2-pre",
            "round2-first",
            "round2-second",
            "round2-done",
            "round3-pre",
            "post-retention",
            "post-first",
            "post-second",
            "final",
        )
    }
    try:
        round_one_evidence = _record_round(runtime, first_ordinal=1, round_id="round-1")
        round_one_schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
            _schedule_requests()
        )
        if round_one_schedule.get("status") != "batch_created":
            raise AssertionError(f"round one batch was not created: {round_one_schedule}")
        round_one_batch_id = str(round_one_schedule["batch_id"])
        runtime.save(paths["round1"])
        round_one = SeedRuntime.load(paths["round1"])

        round_two_evidence = _record_round(
            round_one,
            first_ordinal=7,
            round_id="round-2",
        )
        round_two_schedule = round_one.schedule_structural_candidate_batch_from_workbench_evidence(
            _schedule_requests()
        )
        if round_two_schedule.get("status") != "batch_created":
            raise AssertionError(f"round two batch was not created: {round_two_schedule}")
        round_two_batch_id = str(round_two_schedule["batch_id"])
        round_two_batch = _batch(round_one, round_two_batch_id)
        round_two_first_id, round_two_second_id = round_two_batch.selected_candidate_ids
        round_one.save(paths["round2-pre"])
        round_two_parent = SeedRuntime.load(paths["round2-pre"])

        first_artifact, first_replay, first_measurements = _build_artifact(
            round_two_parent.model.architecture,
            round_two_first_id,
            round_two_evidence,
        )
        if first_artifact.parent_checkpoint_digest != _checkpoint_digest(
            round_two_parent.model.architecture.native_checkpoint()
        ):
            raise AssertionError("round two first artifact was not bound to its measured parent")
        round_two_parent.save(paths["round2-first"])
        round_two_after_first = SeedRuntime.load(paths["round2-first"])
        first_result = round_two_after_first.continue_structural_candidate_batch_from_validation_artifacts(
            round_two_batch_id,
            artifacts_by_candidate={round_two_first_id: first_artifact},
            replays_by_candidate={round_two_first_id: first_replay},
        )
        if first_result["results"][round_two_first_id]["status"] != "admitted":
            raise AssertionError(f"round two first candidate was not admitted: {first_result}")

        second_artifact, second_replay, second_measurements = _build_artifact(
            round_two_after_first.model.architecture,
            round_two_second_id,
            round_two_evidence,
        )
        if second_artifact.parent_checkpoint_digest != _checkpoint_digest(
            round_two_after_first.model.architecture.native_checkpoint()
        ):
            raise AssertionError("round two second artifact was not bound to its measured parent")
        round_two_after_first.save(paths["round2-second"])
        round_two_before_rollback = SeedRuntime.load(paths["round2-second"])
        second_result = round_two_before_rollback.continue_structural_candidate_batch_from_validation_artifacts(
            round_two_batch_id,
            artifacts_by_candidate={round_two_second_id: second_artifact},
            replays_by_candidate={round_two_second_id: second_replay},
        )
        if second_result["results"][round_two_second_id]["status"] != "admitted":
            raise AssertionError(f"round two second candidate was not admitted: {second_result}")
        second_rollback = round_two_before_rollback.rollback_structural_candidate_batch(
            round_two_batch_id,
            round_two_second_id,
        )
        first_rollback = round_two_before_rollback.rollback_structural_candidate_batch(
            round_two_batch_id,
            round_two_first_id,
        )
        if first_rollback["status"] != "rolled_back" or second_rollback["status"] != "rolled_back":
            raise AssertionError("round two terminal batch did not roll back cleanly")
        round_two_before_rollback.save(paths["round2-done"])
        round_two_done = SeedRuntime.load(paths["round2-done"])

        _record_round(
            round_two_done,
            first_ordinal=13,
            round_id="round-3",
        )
        round_three_schedule = round_two_done.schedule_structural_candidate_batch_from_workbench_evidence(
            _schedule_requests()
        )
        if round_three_schedule.get("status") != "batch_created":
            raise AssertionError(f"round three batch was not created: {round_three_schedule}")
        round_three_batch_id = str(round_three_schedule["batch_id"])
        round_two_done.save(paths["round3-pre"])
        round_three = SeedRuntime.load(paths["round3-pre"])
        policy = StructuralLineageRetentionPolicy.create(2, revision=2)
        round_three.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=policy.to_payload(),
        )
        retention_result = round_three.model.architecture.structural_lineage_retention_result
        if retention_result is None:
            raise AssertionError("S39 retention audit was not recorded")
        post_retention_signature = _signature(round_three)
        post_retention_topology = _topology(round_three)
        post_retention_budget = _budget(round_three)
        round_three.save(paths["post-retention"])
        post_retention = SeedRuntime.load(paths["post-retention"])
        post_retention_restart_stable = _signature(post_retention) == post_retention_signature

        before_terminal_replay = _checkpoint_digest(
            post_retention.model.architecture.native_checkpoint()
        )
        try:
            post_retention.continue_structural_candidate_batch_from_validation_artifacts(
                round_two_batch_id,
                artifacts_by_candidate={round_two_second_id: second_artifact},
                replays_by_candidate={round_two_second_id: second_replay},
            )
        except ValueError as exc:
            terminal_replay_rejected = "unknown structural candidate batch" in str(exc)
        else:
            terminal_replay_rejected = False
        after_terminal_replay = _checkpoint_digest(
            post_retention.model.architecture.native_checkpoint()
        )

        round_one_batch = _batch(post_retention, round_one_batch_id)
        retained_first_id, retained_second_id = round_one_batch.selected_candidate_ids
        retained_first_artifact, retained_first_replay, retained_first_measurements = _build_artifact(
            post_retention.model.architecture,
            retained_first_id,
            round_one_evidence,
        )
        retained_first_parent_bound = retained_first_artifact.parent_checkpoint_digest == _checkpoint_digest(
            post_retention.model.architecture.native_checkpoint()
        )
        post_retention.save(paths["post-first"])
        post_first = SeedRuntime.load(paths["post-first"])
        retained_first_checkpoint_bound = retained_first_artifact.parent_checkpoint_digest == _checkpoint_digest(
            post_first.model.architecture.native_checkpoint()
        )
        retained_first_budget_before = _budget(post_first)
        retained_first_result = post_first.continue_structural_candidate_batch_from_validation_artifacts(
            round_one_batch_id,
            artifacts_by_candidate={retained_first_id: retained_first_artifact},
            replays_by_candidate={retained_first_id: retained_first_replay},
        )
        retained_first_budget_after = _budget(post_first)

        retained_second_artifact, retained_second_replay, retained_second_measurements = _build_artifact(
            post_first.model.architecture,
            retained_second_id,
            round_one_evidence,
        )
        retained_second_parent_bound = retained_second_artifact.parent_checkpoint_digest == _checkpoint_digest(
            post_first.model.architecture.native_checkpoint()
        )
        post_first.save(paths["post-second"])
        post_second = SeedRuntime.load(paths["post-second"])
        retained_second_checkpoint_bound = retained_second_artifact.parent_checkpoint_digest == _checkpoint_digest(
            post_second.model.architecture.native_checkpoint()
        )
        retained_second_budget_before = _budget(post_second)
        retained_second_result = post_second.continue_structural_candidate_batch_from_validation_artifacts(
            round_one_batch_id,
            artifacts_by_candidate={retained_second_id: retained_second_artifact},
            replays_by_candidate={retained_second_id: retained_second_replay},
        )
        retained_second_budget_after = _budget(post_second)
        retained_second_rollback = post_second.rollback_structural_candidate_batch(
            round_one_batch_id,
            retained_second_id,
        )
        retained_first_rollback = post_second.rollback_structural_candidate_batch(
            round_one_batch_id,
            retained_first_id,
        )
        repeated_retained_second_rollback = post_second.rollback_structural_candidate_batch(
            round_one_batch_id,
            retained_second_id,
        )
        post_rollback_topology = _topology(post_second)
        post_rollback_budget = _budget(post_second)
        post_second.run_structural_maintenance_cycle(
            candidate_ids=(),
            holdout_inputs_by_candidate={},
            expected_activities_by_candidate={},
            lineage_retention_policy=policy.to_payload(),
        )
        post_second.save(paths["final"])
        final = SeedRuntime.load(paths["final"])
        final_batch_ids = {item.batch_id for item in final.model.architecture.structural_candidate_batches}
        final_signature = _signature(final)
        round_two_artifact_digests = {
            first_artifact.artifact_digest,
            second_artifact.artifact_digest,
        }
        metrics = {
            "retention_protects_live_batches": (
                round_one_batch_id in retention_result.protected_batch_ids
                and round_three_batch_id in retention_result.protected_batch_ids
                and round_two_batch_id in retention_result.removed_batch_ids
            ),
            "post_retention_restart_preserves_projection": post_retention_restart_stable,
            "deleted_terminal_artifact_stays_fail_closed": (
                terminal_replay_rejected and before_terminal_replay == after_terminal_replay
            ),
            "new_artifact_binds_current_checkpoint_after_restart": (
                retained_first_parent_bound
                and retained_first_checkpoint_bound
                and retained_second_parent_bound
                and retained_second_checkpoint_bound
            ),
            "retained_lineage_admits_after_retention": (
                retained_first_result["results"][retained_first_id]["status"] == "admitted"
                and retained_second_result["results"][retained_second_id]["status"] == "admitted"
                and retained_first_budget_after
                == retained_first_budget_before - retained_first_artifact.resource_cost
                and retained_second_budget_after
                == retained_second_budget_before - retained_second_artifact.resource_cost
            ),
            "post_retention_rollback_is_reversible_and_idempotent": (
                retained_first_rollback["status"] == "rolled_back"
                and retained_second_rollback["status"] == "rolled_back"
                and repeated_retained_second_rollback == retained_second_rollback
                and post_rollback_topology == post_retention_topology
                and post_rollback_budget == post_retention_budget
            ),
            "measured_metrics_remain_owner_derived": (
                retained_first_artifact.holdout_gain == retained_first_measurements.holdout_gain
                and retained_second_artifact.holdout_gain == retained_second_measurements.holdout_gain
                and retained_first_artifact.resource_measurement_digest
                == retained_first_measurements.resource_measurement_digest
                and retained_second_artifact.resource_measurement_digest
                == retained_second_measurements.resource_measurement_digest
            ),
            "final_checkpoint_preserves_active_round_and_removes_terminal_round": (
                round_one_batch_id in final_batch_ids
                and round_three_batch_id in final_batch_ids
                and round_two_batch_id not in final_batch_ids
                and not (
                    round_two_artifact_digests
                    & {
                        item.artifact_digest
                        for item in final.model.architecture.structural_validation_artifacts
                    }
                )
                and final_signature["topology"] == post_rollback_topology
                and final_signature["budget"] == post_rollback_budget
                and final_signature["policy_digest"] == policy.policy_digest
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "rounds": {
                "round_one_batch_id": round_one_batch_id,
                "round_two_batch_id": round_two_batch_id,
                "round_three_batch_id": round_three_batch_id,
                "retained_candidate_ids": [retained_first_id, retained_second_id],
            },
            "retention": retention_result.to_payload(),
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "after terminal-only retention and disk restart, protected active lineage must "
                    "consume newly measured artifacts against the current checkpoint, admit and "
                    "rollback exactly within budget, and keep deleted terminal artifacts unreplayable"
                ),
            },
            "boundary": (
                "This canary covers native CPU post-retention SeedRuntime continuation. It does not "
                "claim unlimited growth, automatic budget expansion, open-domain quality, CUDA, "
                "frontend behavior, Windows shell, CI completion, or general intelligence."
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
        / "taiji_w7_r5c_s39_runtime_artifact_post_retention_20260831.json",
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

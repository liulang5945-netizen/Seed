"""Run the R5C-S44 ordered multi-artifact external batch canary."""

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
    _record_round,
)
from scripts.training.eval_taiji_workbench_measured_artifact_batch import (  # noqa: E402
    _build_artifact,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from taiji import (  # noqa: E402
    ArtifactConsumptionPolicy,
    StructuralValidationArtifactStore,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s44-runtime-artifact-store-batch-v1"


def _budget(runtime: SeedRuntime) -> int:
    return int(runtime.model.architecture.cognitive_snapshot().development.structural_budget)


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="ordered-batch-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"ordered batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_id, second_id = batch.selected_candidate_ids
    store_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s44-batch-store-{os.getpid()}"
    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    paths = [
        checkpoint_root / f"s44-first-parent-{os.getpid()}.pt",
        checkpoint_root / f"s44-second-parent-{os.getpid()}.pt",
        checkpoint_root / f"s44-final-{os.getpid()}.pt",
    ]
    store = StructuralValidationArtifactStore(store_root)
    legacy_policy = ArtifactConsumptionPolicy.legacy_compatible(
        reason="historical-s44-batch-canary"
    )
    try:
        initial_budget = _budget(runtime)
        first_artifact, first_replay, first_measurements = _build_artifact(
            runtime.model.architecture,
            first_id,
            evidence,
        )
        first_parent_digest = _checkpoint_digest(runtime.model.architecture.native_checkpoint())
        store.put(first_artifact)
        runtime.save(paths[0])
        first_parent = SeedRuntime.load(paths[0])
        first_result = first_parent.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=StructuralValidationArtifactStore(store_root),
            artifact_digests_by_candidate={first_id: first_artifact.artifact_digest},
            replays_by_candidate={first_id: first_replay},
            artifact_consumption_policy=legacy_policy,
        )
        first_budget = _budget(first_parent)

        second_artifact, second_replay, second_measurements = _build_artifact(
            first_parent.model.architecture,
            second_id,
            evidence,
        )
        second_parent_digest = _checkpoint_digest(first_parent.model.architecture.native_checkpoint())
        store.put(second_artifact)
        first_parent.save(paths[1])
        second_parent = SeedRuntime.load(paths[1])
        second_result = second_parent.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={second_id: second_artifact.artifact_digest},
            replays_by_candidate={second_id: second_replay},
            artifact_consumption_policy=legacy_policy,
        )
        second_budget = _budget(second_parent)
        second_parent.save(paths[2])
        final = SeedRuntime.load(paths[2])
        repeated = final.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=StructuralValidationArtifactStore(store_root),
            artifact_digests_by_candidate={
                first_id: first_artifact.artifact_digest,
                second_id: second_artifact.artifact_digest,
            },
            replays_by_candidate={first_id: first_replay, second_id: second_replay},
            artifact_consumption_policy=legacy_policy,
        )
        final_batch = next(
            item
            for item in final.model.architecture.structural_candidate_batches
            if item.batch_id == batch.batch_id
        )
        provenance = {
            item.artifact_digest: item.measurement_digest
            for item in final.model.architecture.structural_validation_artifacts
        }
        metrics = {
            "first_artifact_uses_initial_parent": (
                first_artifact.parent_checkpoint_digest == first_parent_digest
                and first_result["results"][first_id]["status"] == "admitted"
            ),
            "second_artifact_uses_first_child_parent": (
                second_artifact.parent_checkpoint_digest == second_parent_digest
                and second_result["results"][second_id]["status"] == "admitted"
            ),
            "ordered_budget_charge_is_exact": (
                first_budget == initial_budget - first_artifact.resource_cost
                and second_budget == first_budget - second_artifact.resource_cost
            ),
            "batch_completes_and_full_repeat_is_idempotent": (
                final_batch.status == "completed"
                and repeated["results"][first_id]["status"] == "already_applied"
                and repeated["results"][second_id]["status"] == "already_applied"
            ),
            "provenance_survives_checkpoint_chain": (
                provenance.get(first_artifact.artifact_digest) == first_measurements.measurement_digest
                and provenance.get(second_artifact.artifact_digest)
                == second_measurements.measurement_digest
            ),
            "final_checkpoint_preserves_ordered_state": (
                _budget(final) == second_budget
                and final.model.architecture.structural_runtime_tick
                == second_parent.model.architecture.structural_runtime_tick
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_ids": [first_id, second_id],
            "artifact_digests": [first_artifact.artifact_digest, second_artifact.artifact_digest],
            "parent_digests": [first_parent_digest, second_parent_digest],
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "two external measured artifacts must be handed off in parent order across "
                    "checkpoints, complete one batch with exact budget charges, preserve provenance, "
                    "and return idempotent statuses on full repeat"
                ),
            },
            "boundary": (
                "This canary covers native CPU ordered multi-artifact external batch handoff. It does "
                "not claim unordered parallel admission, unlimited growth, open-domain quality, CUDA, "
                "frontend behavior, Windows shell, CI completion, or general intelligence."
            ),
        }
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
        _remove_directory(store_root)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s44_runtime_artifact_store_batch_20260831.json",
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

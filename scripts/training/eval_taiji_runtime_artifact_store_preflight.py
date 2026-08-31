"""Run the R5C-S43 multi-candidate artifact-store preflight canary."""

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
from taiji import StructuralValidationArtifactStore  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s43-runtime-artifact-store-preflight-v1"


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="preflight-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"preflight batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    first_id, second_id = batch.selected_candidate_ids
    artifact, replay, _ = _build_artifact(runtime.model.architecture, first_id, evidence)
    suffix = os.getpid()
    store_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s43-preflight-{suffix}"
    checkpoint_path = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s43-preflight-runtime-{suffix}.pt"
    store = StructuralValidationArtifactStore(store_root)
    try:
        store.put(artifact)
        runtime.save(checkpoint_path)
        restored = SeedRuntime.load(checkpoint_path)
        before = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        try:
            restored.continue_structural_candidate_batch_from_artifact_store(
                batch.batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={first_id: artifact.artifact_digest, second_id: "0" * 64},
                replays_by_candidate={first_id: replay},
            )
        except FileNotFoundError:
            missing_second_rejected = True
        else:
            missing_second_rejected = False
        after = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        unchanged_batch = next(
            item
            for item in restored.model.architecture.structural_candidate_batches
            if item.batch_id == batch.batch_id
        )
        preflight_atomic = (
            missing_second_rejected
            and before == after
            and all(state == "reserved" for _, state in unchanged_batch.candidate_states)
        )
        result = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=StructuralValidationArtifactStore(store_root),
            artifact_digests_by_candidate={first_id: artifact.artifact_digest},
            replays_by_candidate={first_id: replay},
        )
        repeat = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={first_id: artifact.artifact_digest},
            replays_by_candidate={first_id: replay},
        )
        metrics = {
            "all_candidate_references_preflight_before_mutation": preflight_atomic,
            "valid_candidate_survives_failed_preflight": (
                result["results"][first_id]["status"] == "admitted"
                and repeat["results"][first_id]["status"] == "already_applied"
            ),
            "single_budget_charge_after_preflight_failure": (
                len(restored.model.architecture.structural_admission_results) == 1
                and len(restored.model.architecture.structural_validation_artifacts) == 1
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_ids": [first_id, second_id],
            "artifact_digest": artifact.artifact_digest,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "multi-candidate artifact references must be fully resolved before native mutation; "
                    "a missing second artifact cannot partially consume the first, while a later valid "
                    "submission remains admissible and idempotent"
                ),
            },
            "boundary": (
                "This canary covers native CPU multi-candidate SeedRuntime artifact preflight. It does "
                "not claim automatic retries, unlimited growth, open-domain quality, CUDA, frontend "
                "behavior, Windows shell, CI completion, or general intelligence."
            ),
        }
    finally:
        checkpoint_path.unlink(missing_ok=True)
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
        / "taiji_w7_r5c_s43_runtime_artifact_store_preflight_20260831.json",
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

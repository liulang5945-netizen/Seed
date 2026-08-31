"""Run the R5C-S42 SeedRuntime artifact-store bridge canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s42-runtime-artifact-store-bridge-v1"


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="bridge-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"bridge batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, replay, _ = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        evidence,
    )
    suffix = os.getpid()
    store_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s42-bridge-{suffix}"
    checkpoint_path = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s42-bridge-runtime-{suffix}.pt"
    store = StructuralValidationArtifactStore(store_root)
    try:
        store.put(artifact)
        runtime.save(checkpoint_path)

        unknown = SeedRuntime.load(checkpoint_path)
        before_unknown = _checkpoint_digest(unknown.model.architecture.native_checkpoint())
        try:
            unknown.continue_structural_candidate_batch_from_artifact_store(
                batch.batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={"candidate:foreign": artifact.artifact_digest},
                replays_by_candidate={},
            )
        except ValueError as exc:
            unknown_rejected = "outside the selected batch" in str(exc)
        else:
            unknown_rejected = False
        unknown_atomic = _checkpoint_digest(unknown.model.architecture.native_checkpoint()) == before_unknown

        missing = SeedRuntime.load(checkpoint_path)
        before_missing = _checkpoint_digest(missing.model.architecture.native_checkpoint())
        try:
            missing.continue_structural_candidate_batch_from_artifact_store(
                batch.batch_id,
                artifact_store=store,
                artifact_digests_by_candidate={candidate_id: "0" * 64},
                replays_by_candidate={candidate_id: replay},
            )
        except FileNotFoundError:
            missing_rejected = True
        else:
            missing_rejected = False
        missing_atomic = _checkpoint_digest(missing.model.architecture.native_checkpoint()) == before_missing

        restored = SeedRuntime.load(checkpoint_path)
        handoff = StructuralValidationArtifactStore(store_root)
        result = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=handoff,
            artifact_digests_by_candidate={candidate_id: artifact.artifact_digest},
            replays_by_candidate={candidate_id: replay},
        )
        repeated = restored.continue_structural_candidate_batch_from_artifact_store(
            batch.batch_id,
            artifact_store=store,
            artifact_digests_by_candidate={candidate_id: artifact.artifact_digest},
            replays_by_candidate={candidate_id: replay},
        )
        metrics = {
            "unknown_key_rejected_before_store_resolution": unknown_rejected and unknown_atomic,
            "missing_store_artifact_rejected_atomically": missing_rejected and missing_atomic,
            "external_digest_admission_uses_native_contract": (
                result["results"][candidate_id]["status"] == "admitted"
                and repeated["results"][candidate_id]["status"] == "already_applied"
            ),
            "bridge_preserves_single_budget_charge": (
                len(restored.model.architecture.structural_admission_results) == 1
                and len(restored.model.architecture.structural_validation_artifacts) == 1
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_id": candidate_id,
            "artifact_digest": artifact.artifact_digest,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "SeedRuntime must validate candidate-to-artifact references before mutation, "
                    "then route valid external artifacts through the existing native batch contract "
                    "with one budget charge and idempotent repeat"
                ),
            },
            "boundary": (
                "This canary covers native CPU SeedRuntime artifact-store integration. It does not "
                "claim automatic deletion, unlimited growth, open-domain quality, CUDA, frontend "
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
        / "taiji_w7_r5c_s42_runtime_artifact_store_bridge_20260831.json",
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

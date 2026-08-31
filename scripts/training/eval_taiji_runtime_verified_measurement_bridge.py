"""Run the R5C-S51 verified-measurement runtime bridge canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s51-runtime-verified-measurement-bridge-v1"


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def _prepare_runtime(round_id: str) -> tuple[SeedRuntime, str, tuple[dict[str, object], ...]]:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id=round_id)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"verified bridge batch was not created: {schedule}")
    return runtime, str(schedule["batch_id"]), evidence


def evaluate() -> dict[str, object]:
    suffix = os.getpid()
    root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s51-store-{suffix}"
    verified_checkpoint = root.parent / f"s51-verified-{suffix}.pt"
    legacy_checkpoint = root.parent / f"s51-legacy-{suffix}.pt"
    try:
        verified_runtime, verified_batch_id, verified_evidence = _prepare_runtime(
            "verified-bridge"
        )
        verified_candidate_id = verified_runtime.model.architecture.structural_candidate_batches[-1].selected_candidate_ids[0]
        verified_artifact, verified_replay, verified_measurements = _build_artifact(
            verified_runtime.model.architecture,
            verified_candidate_id,
            verified_evidence,
        )
        verified_store = StructuralValidationArtifactStore(root / "verified")
        verified_store.put_measured_artifact(verified_artifact, verified_measurements)
        verified_runtime.save(verified_checkpoint)
        verified_restored = SeedRuntime.load(verified_checkpoint)
        verified_result = verified_restored.continue_structural_candidate_batch_from_artifact_store(
            verified_batch_id,
            artifact_store=verified_store,
            artifact_digests_by_candidate={
                verified_candidate_id: verified_artifact.artifact_digest
            },
            replays_by_candidate={verified_candidate_id: verified_replay},
            require_verified_measurements=True,
        )
        verified_accepts = (
            verified_result["results"][verified_candidate_id]["status"] == "admitted"
        )

        legacy_runtime, legacy_batch_id, legacy_evidence = _prepare_runtime("legacy-bridge")
        legacy_candidate_id = legacy_runtime.model.architecture.structural_candidate_batches[-1].selected_candidate_ids[0]
        legacy_artifact, legacy_replay, _ = _build_artifact(
            legacy_runtime.model.architecture,
            legacy_candidate_id,
            legacy_evidence,
        )
        legacy_store = StructuralValidationArtifactStore(root / "legacy")
        legacy_store.put(legacy_artifact)
        legacy_runtime.save(legacy_checkpoint)
        strict_legacy = SeedRuntime.load(legacy_checkpoint)
        before_strict_legacy = _checkpoint_digest(
            strict_legacy.model.architecture.native_checkpoint()
        )
        try:
            strict_legacy.continue_structural_candidate_batch_from_artifact_store(
                legacy_batch_id,
                artifact_store=legacy_store,
                artifact_digests_by_candidate={
                    legacy_candidate_id: legacy_artifact.artifact_digest
                },
                replays_by_candidate={legacy_candidate_id: legacy_replay},
                require_verified_measurements=True,
            )
        except ValueError:
            strict_legacy_rejected = True
        else:
            strict_legacy_rejected = False
        strict_legacy_read_only = (
            _checkpoint_digest(strict_legacy.model.architecture.native_checkpoint())
            == before_strict_legacy
        )
        default_legacy = SeedRuntime.load(legacy_checkpoint)
        default_result = default_legacy.continue_structural_candidate_batch_from_artifact_store(
            legacy_batch_id,
            artifact_store=legacy_store,
            artifact_digests_by_candidate={legacy_candidate_id: legacy_artifact.artifact_digest},
            replays_by_candidate={legacy_candidate_id: legacy_replay},
        )
        legacy_default_accepts = (
            default_result["results"][legacy_candidate_id]["status"] == "admitted"
        )

        partial_runtime, partial_batch_id, partial_evidence = _prepare_runtime("partial-bridge")
        first_id, second_id = partial_runtime.model.architecture.structural_candidate_batches[-1].selected_candidate_ids
        first_artifact, first_replay, first_measurements = _build_artifact(
            partial_runtime.model.architecture,
            first_id,
            partial_evidence,
        )
        second_artifact, second_replay, _ = _build_artifact(
            partial_runtime.model.architecture,
            second_id,
            partial_evidence,
        )
        partial_store = StructuralValidationArtifactStore(root / "partial")
        partial_store.put_measured_artifact(first_artifact, first_measurements)
        partial_store.put(second_artifact)
        before_partial = _checkpoint_digest(
            partial_runtime.model.architecture.native_checkpoint()
        )
        before_partial_budget = partial_runtime.model.architecture.cognitive_snapshot().development.structural_budget
        try:
            partial_runtime.continue_structural_candidate_batch_from_artifact_store(
                partial_batch_id,
                artifact_store=partial_store,
                artifact_digests_by_candidate={
                    first_id: first_artifact.artifact_digest,
                    second_id: second_artifact.artifact_digest,
                },
                replays_by_candidate={
                    first_id: first_replay,
                    second_id: second_replay,
                },
                require_verified_measurements=True,
            )
        except ValueError:
            partial_rejected = True
        else:
            partial_rejected = False
        partial_is_atomic = (
            partial_rejected
            and _checkpoint_digest(partial_runtime.model.architecture.native_checkpoint())
            == before_partial
            and partial_runtime.model.architecture.cognitive_snapshot().development.structural_budget
            == before_partial_budget
        )

        metrics = {
            "verified_bundle_strictly_consumes": verified_accepts,
            "legacy_default_compatibility_is_preserved": legacy_default_accepts,
            "legacy_strict_mode_fails_closed_read_only": (
                strict_legacy_rejected and strict_legacy_read_only
            ),
            "multi_candidate_strict_resolution_is_atomic": partial_is_atomic,
        }
        return {
            "format": REPORT_FORMAT,
            "verified_batch_id": verified_batch_id,
            "legacy_batch_id": legacy_batch_id,
            "partial_batch_id": partial_batch_id,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "strict verified-measurement bridge mode must validate every external artifact "
                    "before native consumption, while default mode remains legacy-compatible"
                ),
            },
            "boundary": (
                "This canary covers native CPU opt-in verified artifact consumption. It does not "
                "claim default forced migration, automatic registration, deletion, unlimited storage, "
                "open-domain quality, CUDA, frontend behavior, Windows shell, CI completion, or "
                "general intelligence."
            ),
        }
    finally:
        verified_checkpoint.unlink(missing_ok=True)
        legacy_checkpoint.unlink(missing_ok=True)
        _remove_directory(root / "verified")
        _remove_directory(root / "legacy")
        _remove_directory(root / "partial")
        _remove_directory(root)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s51_runtime_verified_measurement_bridge_20260831.json",
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

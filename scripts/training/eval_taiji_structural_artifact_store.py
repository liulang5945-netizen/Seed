"""Run the R5C-S41 external measured-artifact store canary."""

from __future__ import annotations

import argparse
import concurrent.futures
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

REPORT_FORMAT = "taiji-w7-r5c-s41-structural-artifact-store-v1"


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    evidence = _record_round(runtime, first_ordinal=1, round_id="store-round")
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"artifact store batch was not created: {schedule}")
    batch = runtime.model.architecture.structural_candidate_batches[-1]
    candidate_id = batch.selected_candidate_ids[0]
    artifact, replay, measurements = _build_artifact(
        runtime.model.architecture,
        candidate_id,
        evidence,
    )
    suffix = os.getpid()
    store_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s41-store-{suffix}"
    concurrent_root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s41-concurrent-{suffix}"
    checkpoint_path = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s41-runtime-{suffix}.pt"
    store = StructuralValidationArtifactStore(store_root)
    try:
        store.put(artifact)
        artifact_path = store.path_for(artifact.artifact_digest)
        original_bytes = artifact_path.read_bytes()
        handed_off = StructuralValidationArtifactStore(store_root).load(artifact.artifact_digest)
        concurrent_store = StructuralValidationArtifactStore(concurrent_root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_digests = tuple(
                item.artifact_digest
                for item in executor.map(lambda _: concurrent_store.put(artifact), range(4))
            )
        concurrent_roundtrip = concurrent_store.load(artifact.artifact_digest)

        runtime.save(checkpoint_path)
        restored = SeedRuntime.load(checkpoint_path)
        before_budget = restored.model.architecture.cognitive_snapshot().development.structural_budget
        admission = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: handed_off.to_payload()},
            replays_by_candidate={candidate_id: replay},
        )
        repeated = restored.continue_structural_candidate_batch_from_validation_artifacts(
            batch.batch_id,
            artifacts_by_candidate={candidate_id: concurrent_roundtrip.to_payload()},
            replays_by_candidate={candidate_id: replay},
        )

        before_tamper = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        artifact_path.write_bytes(b"{}")
        try:
            store.load(artifact.artifact_digest)
        except (KeyError, ValueError):
            tamper_rejected = True
        else:
            tamper_rejected = False
        try:
            store.put(artifact)
        except ValueError as exc:
            collision_rejected = "content collision" in str(exc)
        else:
            collision_rejected = False
        after_tamper = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        artifact_path.write_bytes(original_bytes)

        metrics = {
            "digest_named_canonical_roundtrip": (
                artifact_path.name == f"{artifact.artifact_digest}.json"
                and handed_off == artifact
                and artifact_path.read_bytes() == original_bytes
            ),
            "concurrent_initial_write_is_idempotent": (
                concurrent_digests == (artifact.artifact_digest,) * 4
                and concurrent_roundtrip == artifact
            ),
            "external_payload_consumes_through_runtime_contract": (
                measurements.measurement_digest == handed_off.measurement_digest
                and admission["results"][candidate_id]["status"] == "admitted"
                and repeated["results"][candidate_id]["status"] == "already_applied"
                and _budget_after(restored) == before_budget - artifact.resource_cost
            ),
            "tamper_and_content_collision_fail_closed": (
                tamper_rejected and collision_rejected and before_tamper == after_tamper
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "artifact_digest": artifact.artifact_digest,
            "measurement_digest": artifact.measurement_digest,
            "batch_id": batch.batch_id,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "measured artifacts must survive immutable digest-addressed external handoff, "
                    "be consumable only through the existing runtime batch contract, and reject "
                    "concurrent corruption, tamper, and byte collisions"
                ),
            },
            "boundary": (
                "This canary covers native CPU measured-artifact persistence and handoff. It does "
                "not claim automatic deletion, unbounded storage, open-domain quality, unlimited "
                "growth, CUDA, frontend behavior, Windows shell, CI completion, or general intelligence."
            ),
        }
    finally:
        checkpoint_path.unlink(missing_ok=True)
        _remove_directory(store_root)
        _remove_directory(concurrent_root)


def _budget_after(runtime: SeedRuntime) -> int:
    return int(runtime.model.architecture.cognitive_snapshot().development.structural_budget)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s41_structural_artifact_store_20260831.json",
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

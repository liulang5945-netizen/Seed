"""Run the R5C-S52 artifact-consumption policy canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s52-artifact-consumption-policy-v1"


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
        raise AssertionError(f"S52 batch was not created: {schedule}")
    return runtime, str(schedule["batch_id"]), evidence


def evaluate() -> dict[str, object]:
    suffix = os.getpid()
    root = PROJECT_ROOT / "output" / "manual-r5-canary" / f"s52-store-{suffix}"
    strict_checkpoint = root.parent / f"s52-strict-{suffix}.pt"
    legacy_checkpoint = root.parent / f"s52-legacy-{suffix}.pt"
    checkpointed_policy_checkpoint = root.parent / f"s52-policy-{suffix}.pt"
    try:
        strict_runtime, strict_batch_id, strict_evidence = _prepare_runtime("s52-strict")
        strict_candidate_id = (
            strict_runtime.model.architecture.structural_candidate_batches[-1]
            .selected_candidate_ids[0]
        )
        strict_artifact, strict_replay, strict_measurements = _build_artifact(
            strict_runtime.model.architecture,
            strict_candidate_id,
            strict_evidence,
        )
        strict_store = StructuralValidationArtifactStore(root / "strict")
        strict_store.put_measured_artifact(strict_artifact, strict_measurements)
        strict_runtime.save(strict_checkpoint)
        strict_restored = SeedRuntime.load(strict_checkpoint)
        strict_result = strict_restored.continue_structural_candidate_batch_from_artifact_store(
            strict_batch_id,
            artifact_store=strict_store,
            artifact_digests_by_candidate={
                strict_candidate_id: strict_artifact.artifact_digest
            },
            replays_by_candidate={strict_candidate_id: strict_replay},
        )
        strict_audit = strict_result["artifact_consumption"]
        strict_verified = (
            strict_result["results"][strict_candidate_id]["status"] == "admitted"
            and strict_audit["policy"]["mode"] == "verified-only"
            and strict_audit["artifact_statuses"][strict_candidate_id] == "verified"
        )

        legacy_runtime, legacy_batch_id, legacy_evidence = _prepare_runtime("s52-legacy")
        legacy_candidate_id = (
            legacy_runtime.model.architecture.structural_candidate_batches[-1]
            .selected_candidate_ids[0]
        )
        legacy_artifact, legacy_replay, _ = _build_artifact(
            legacy_runtime.model.architecture,
            legacy_candidate_id,
            legacy_evidence,
        )
        legacy_store = StructuralValidationArtifactStore(root / "legacy")
        legacy_store.put(legacy_artifact)
        legacy_runtime.save(legacy_checkpoint)
        legacy_restored = SeedRuntime.load(legacy_checkpoint)
        before_legacy_rejection = _checkpoint_digest(
            legacy_restored.model.architecture.native_checkpoint()
        )
        try:
            legacy_restored.continue_structural_candidate_batch_from_artifact_store(
                legacy_batch_id,
                artifact_store=legacy_store,
                artifact_digests_by_candidate={
                    legacy_candidate_id: legacy_artifact.artifact_digest
                },
                replays_by_candidate={legacy_candidate_id: legacy_replay},
            )
        except (FileNotFoundError, ValueError):
            strict_legacy_rejected = True
        else:
            strict_legacy_rejected = False
        strict_legacy_audit = legacy_restored.status()["artifact_consumption"]["last_audit"]
        strict_legacy_read_only = (
            _checkpoint_digest(legacy_restored.model.architecture.native_checkpoint())
            == before_legacy_rejection
        )
        explicit_legacy_policy = ArtifactConsumptionPolicy.legacy_compatible(
            reason="historical-s52-canary"
        )
        legacy_result = legacy_restored.continue_structural_candidate_batch_from_artifact_store(
            legacy_batch_id,
            artifact_store=legacy_store,
            artifact_digests_by_candidate={
                legacy_candidate_id: legacy_artifact.artifact_digest
            },
            replays_by_candidate={legacy_candidate_id: legacy_replay},
            artifact_consumption_policy=explicit_legacy_policy,
        )
        legacy_audit = legacy_result["artifact_consumption"]
        explicit_legacy = (
            legacy_result["results"][legacy_candidate_id]["status"] == "admitted"
            and legacy_audit["policy"] == explicit_legacy_policy.to_payload()
            and legacy_audit["artifact_statuses"][legacy_candidate_id]
            == "legacy_unverified"
        )

        checkpointed_runtime, checkpointed_batch_id, checkpointed_evidence = _prepare_runtime(
            "s52-checkpointed-policy"
        )
        checkpointed_candidate_id = (
            checkpointed_runtime.model.architecture.structural_candidate_batches[-1]
            .selected_candidate_ids[0]
        )
        checkpointed_artifact, checkpointed_replay, _ = _build_artifact(
            checkpointed_runtime.model.architecture,
            checkpointed_candidate_id,
            checkpointed_evidence,
        )
        checkpointed_store = StructuralValidationArtifactStore(root / "checkpointed")
        checkpointed_store.put(checkpointed_artifact)
        checkpointed_policy = checkpointed_runtime.model.architecture.set_artifact_consumption_policy(
            explicit_legacy_policy
        )
        checkpointed_runtime.save(checkpointed_policy_checkpoint)
        checkpointed_restored = SeedRuntime.load(checkpointed_policy_checkpoint)
        checkpointed_result = checkpointed_restored.continue_structural_candidate_batch_from_artifact_store(
            checkpointed_batch_id,
            artifact_store=checkpointed_store,
            artifact_digests_by_candidate={
                checkpointed_candidate_id: checkpointed_artifact.artifact_digest
            },
            replays_by_candidate={checkpointed_candidate_id: checkpointed_replay},
        )
        checkpointed_audit = checkpointed_result["artifact_consumption"]
        checkpointed_policy_roundtrip = (
            checkpointed_restored.model.architecture.artifact_consumption_policy
            == checkpointed_policy
            and checkpointed_audit["policy"] == checkpointed_policy.to_payload()
        )

        tampered_runtime, tampered_batch_id, tampered_evidence = _prepare_runtime("s52-tampered")
        tampered_candidate_id = (
            tampered_runtime.model.architecture.structural_candidate_batches[-1]
            .selected_candidate_ids[0]
        )
        tampered_artifact, tampered_replay, tampered_measurements = _build_artifact(
            tampered_runtime.model.architecture,
            tampered_candidate_id,
            tampered_evidence,
        )
        tampered_store = StructuralValidationArtifactStore(root / "tampered")
        tampered_store.put_measured_artifact(tampered_artifact, tampered_measurements)
        measurement_path = tampered_store.measurement_path_for(
            tampered_measurements.measurement_digest
        )
        original_measurement_bytes = measurement_path.read_bytes()
        tampered_payload = json.loads(original_measurement_bytes.decode("utf-8"))
        tampered_payload["resource_cost"] = int(tampered_payload["resource_cost"]) + 1
        measurement_path.write_text(
            json.dumps(
                tampered_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        before_tampered_rejection = _checkpoint_digest(
            tampered_runtime.model.architecture.native_checkpoint()
        )
        try:
            tampered_runtime.continue_structural_candidate_batch_from_artifact_store(
                tampered_batch_id,
                artifact_store=tampered_store,
                artifact_digests_by_candidate={
                    tampered_candidate_id: tampered_artifact.artifact_digest
                },
                replays_by_candidate={tampered_candidate_id: tampered_replay},
            )
        except ValueError:
            tampered_rejected = True
        else:
            tampered_rejected = False
        tampered_audit = tampered_runtime.status()["artifact_consumption"]["last_audit"]
        tampered_fails_closed = (
            tampered_rejected
            and tampered_audit["artifact_statuses"][tampered_candidate_id] == "tampered"
            and _checkpoint_digest(tampered_runtime.model.architecture.native_checkpoint())
            == before_tampered_rejection
        )
        measurement_path.write_bytes(original_measurement_bytes)

        multi_runtime, multi_batch_id, multi_evidence = _prepare_runtime("s52-multi")
        first_id, second_id = (
            multi_runtime.model.architecture.structural_candidate_batches[-1]
            .selected_candidate_ids
        )
        first_artifact, first_replay, first_measurements = _build_artifact(
            multi_runtime.model.architecture,
            first_id,
            multi_evidence,
        )
        second_artifact, second_replay, _ = _build_artifact(
            multi_runtime.model.architecture,
            second_id,
            multi_evidence,
        )
        multi_store = StructuralValidationArtifactStore(root / "multi")
        multi_store.put_measured_artifact(first_artifact, first_measurements)
        multi_store.put(second_artifact)
        before_multi_rejection = _checkpoint_digest(
            multi_runtime.model.architecture.native_checkpoint()
        )
        try:
            multi_runtime.continue_structural_candidate_batch_from_artifact_store(
                multi_batch_id,
                artifact_store=multi_store,
                artifact_digests_by_candidate={
                    first_id: first_artifact.artifact_digest,
                    second_id: second_artifact.artifact_digest,
                },
                replays_by_candidate={
                    first_id: first_replay,
                    second_id: second_replay,
                },
            )
        except (FileNotFoundError, ValueError):
            multi_rejected = True
        else:
            multi_rejected = False
        multi_audit = multi_runtime.status()["artifact_consumption"]["last_audit"]
        multi_batch = multi_runtime.model.architecture.structural_candidate_batches[-1]
        multi_is_atomic = (
            multi_rejected
            and multi_audit["artifact_statuses"] == {
                first_id: "verified",
                second_id: "legacy_unverified",
            }
            and _checkpoint_digest(multi_runtime.model.architecture.native_checkpoint())
            == before_multi_rejection
            and all(state == "reserved" for _, state in multi_batch.candidate_states)
        )

        metrics = {
            "fresh_runtime_defaults_verified_only": strict_verified,
            "strict_rejects_legacy_read_only": (
                strict_legacy_rejected
                and strict_legacy_read_only
                and strict_legacy_audit["artifact_statuses"][legacy_candidate_id]
                == "legacy_unverified"
            ),
            "explicit_legacy_replay_is_audited": explicit_legacy,
            "policy_checkpoint_roundtrip_is_stable": checkpointed_policy_roundtrip,
            "tampered_sidecar_fails_closed": tampered_fails_closed,
            "multi_candidate_resolution_is_atomic": multi_is_atomic,
        }
        return {
            "format": REPORT_FORMAT,
            "batch_ids": {
                "strict": strict_batch_id,
                "legacy": legacy_batch_id,
                "checkpointed": checkpointed_batch_id,
                "tampered": tampered_batch_id,
                "multi": multi_batch_id,
            },
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "new runtimes consume only independently verified artifacts by default; "
                    "historical replay requires an explicit audited compatibility policy; "
                    "all policy failures remain atomic before native mutation"
                ),
            },
            "boundary": (
                "This CPU canary proves policy resolution and artifact preflight boundaries. "
                "It does not claim open-domain quality, autonomous Workbench planning, CUDA, "
                "frontend behavior, CI completion, or general intelligence."
            ),
        }
    finally:
        for checkpoint in (
            strict_checkpoint,
            legacy_checkpoint,
            checkpointed_policy_checkpoint,
        ):
            checkpoint.unlink(missing_ok=True)
        for directory in (
            root / "strict",
            root / "legacy",
            root / "checkpointed",
            root / "tampered",
            root / "multi",
            root,
        ):
            _remove_directory(directory)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_r5c_s52_artifact_consumption_policy_20260831.json",
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

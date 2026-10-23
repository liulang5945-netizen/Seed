"""Verify actual resume-launch provenance without importing a model or reading prompts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scripts.training.taiji_r2_resume_ledger_anchor import BASELINE, verified_snapshot
from scripts.training.validate_taiji_r2_continuation_endpoint import EndpointRefuse

LAUNCH_SOURCES = (
    "scripts/training/run_taiji_r2_target_update_resume.py",
    "scripts/training/taiji_r2_target_update_projection.py",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py",
    "tests/taiji_native/test_r2_target_update_projection_contract.py",
)
SOURCE_REVISION_SHA256 = "2f6938afb910a7bca2c9d5bd5c3d4f79e5a67aa2cd19da08c0a9577589640bf7"
LAUNCH_SOURCE_SHA256 = {
    "scripts/training/run_taiji_r2_target_update_resume.py": "6895f2d79b66f5def4ef27758e28c206826082d0c85c96cf7541037cbf5a230d",
    "scripts/training/taiji_r2_target_update_projection.py": "c7a2d0821b59a35fc5beca94edab557e2c201b04a028311a0008091209d1aeba",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py": "fce64448dc8d9d49104509c1a1cd5a4f4beeb80f5ee7ea74c440a2c118ae34b5",
    "tests/taiji_native/test_r2_target_update_projection_contract.py": "acf6a0a4b41f9efd2a3f8475a562d387537e2b14f21793dfdf23f2d6850b2000",
}
FAST_LAUNCH_SOURCES = (
    "scripts/training/run_taiji_r2_fast_ledger_resume.py",
    "scripts/training/taiji_r2_resume_ledger_anchor.py",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py",
    "scripts/training/taiji_r2_target_update_projection.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_contract.py",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py",
    "scripts/training/run_taiji_r2_fast_ledger_resume_v2.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_v2_contract.py",
)
FAST_LAUNCH_SOURCE_SHA256 = {
    "scripts/training/run_taiji_r2_fast_ledger_resume.py": "d8ece312b17e6b84cdd8c45a332534ef2b2755b0e159b208151c65479e9a9cbf",
    "scripts/training/taiji_r2_resume_ledger_anchor.py": "c729243584d89bd8aded0bacfdd7d0e99d760ad72dbe6020aff5bd03b5dfd37b",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py": "fce64448dc8d9d49104509c1a1cd5a4f4beeb80f5ee7ea74c440a2c118ae34b5",
    "scripts/training/taiji_r2_target_update_projection.py": "c7a2d0821b59a35fc5beca94edab557e2c201b04a028311a0008091209d1aeba",
    "tests/taiji_native/test_r2_fast_ledger_resume_contract.py": "839e4f8bb85a7ce816cae33fafe185761f298717cd58a08807b767620a12bcc5",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py": "e9985a64ffc7a58c1d6b4a30d11b419d7a5bbcb0c7e2fe16e696bfa886a7ae4f",
}
FAST_LAUNCH_V2_SOURCES = (
    "scripts/training/run_taiji_r2_fast_ledger_resume_v2.py",
    *FAST_LAUNCH_SOURCES,
    "tests/taiji_native/test_r2_fast_ledger_resume_v2_contract.py",
)
FAST_LAUNCH_V2_SOURCE_SHA256 = {
    **FAST_LAUNCH_SOURCE_SHA256,
    "scripts/training/run_taiji_r2_fast_ledger_resume_v2.py": "1530e68150fe8264d1db509b52fbb82159c8d325f2fa24d44ca2b88a7869f9e6",
    "tests/taiji_native/test_r2_fast_ledger_resume_v2_contract.py": "3cd27ced8d3daacf52ddecfe4aa589da96240fea7666b9daaf186939f6358dbc",
}
CORE_SOURCES = {
    "budget_contract_test": "tests/taiji_native/test_r2_dialogue_continuation_extension_contract.py",
    "budget_helper": "scripts/training/taiji_r2_dialogue_continuation_extension.py",
    "runner": "scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py",
    "runner_contract_test": "tests/taiji_native/test_r2_dialogue_continuation_extension_runner_contract.py",
    "source_revision_contract_test": "tests/taiji_native/test_r2_dialogue_continuation_source_revision_contract.py",
    "source_validator": "scripts/training/taiji_r2_dialogue_continuation_source_validation.py",
    "source_validator_contract_test": "tests/taiji_native/test_r2_dialogue_continuation_source_validation_contract.py",
}
BOOT_SOURCES = (
    "scripts/training/taiji_r2_dialogue_boot_accounting.py",
    "tests/taiji_native/test_r2_dialogue_boot_accounting_contract.py",
)
VERIFICATION_SOURCES = (
    "scripts/training/validate_taiji_r2_continuation_launches.py",
    "tests/taiji_native/test_r2_continuation_launch_provenance_contract.py",
    "scripts/training/validate_taiji_r2_continuation_endpoint.py",
    "tests/taiji_native/test_r2_continuation_endpoint_contract.py",
    "scripts/training/prepare_taiji_r2_continuation_endpoints.py",
    "tests/taiji_native/test_r2_continuation_endpoint_preflight_contract.py",
    "scripts/training/run_taiji_r2_fast_ledger_resume.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_contract.py",
    "scripts/training/taiji_r2_resume_ledger_anchor.py",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py",
)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_path(path, root):
    path = Path(path)
    if (
        not path.is_absolute()
        or not path.resolve().is_relative_to(root.resolve())
        or any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
    ):
        raise EndpointRefuse("launch provenance file is missing, linked or outside repository")
    return path


def _read(path, root):
    _checked_path(path, root)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EndpointRefuse(f"cannot read launch provenance: {path.name}") from exc


def _bound(path, expected, root):
    path = _checked_path(path, root)
    if not isinstance(expected, str) or len(expected) != 64 or _sha(path) != expected:
        raise EndpointRefuse("launch provenance file SHA differs")
    return path.resolve().relative_to(root.resolve()).as_posix()


def _source_map(rows, expected_paths, root, *, expected_sha256=None):
    if not isinstance(rows, dict) or len(rows) != len(expected_paths):
        raise EndpointRefuse("launch source set is incomplete")
    result = {_bound(path, digest, root): digest for path, digest in rows.items()}
    if set(result) != set(expected_paths):
        raise EndpointRefuse("launch source set differs")
    if expected_sha256 is not None and result != expected_sha256:
        raise EndpointRefuse("launch sources differ from the approved adapter revision")
    return result


def _validate_forecast(forecast, arm, arms, *, selected_arm=False):
    if not isinstance(forecast, dict):
        raise EndpointRefuse("launch forecast is missing")
    safety = forecast.get("safety_factor")
    aggregate = forecast.get("projected_cpu_aggregate")
    aggregate_headroom = forecast.get("aggregate_headroom_cpu_seconds")
    per_arm = forecast.get("projected_cpu_by_arm")
    headroom = forecast.get("arm_headroom_by_arm_cpu_seconds")

    def valid_number(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )

    if (
        forecast.get("arm") != arm
        or forecast.get("projection_unit") != "target_byte_updates"
        or not valid_number(safety)
        or safety < 1.15
        or not isinstance(per_arm, dict)
        or set(per_arm) != set(arms)
        or not all(valid_number(value) for value in per_arm.values())
        or not isinstance(headroom, dict)
        or set(headroom) != set(arms)
        or not all(valid_number(value) for value in headroom.values())
        or not valid_number(aggregate)
        or not math.isclose(aggregate, sum(per_arm.values()), rel_tol=1e-9, abs_tol=0.01)
        or not valid_number(aggregate_headroom)
        or forecast.get("arm_headroom_cpu_seconds") != headroom[arm]
    ):
        raise EndpointRefuse("launch forecast is missing or malformed")
    flags = forecast.get("per_arm_feasible")
    if (
        not isinstance(flags, dict)
        or set(flags) != set(arms)
        or any(not isinstance(flags[name], bool) for name in arms)
    ):
        raise EndpointRefuse("launch forecast feasibility map is malformed")
    global_feasible = (
        all(flags.values())
        and aggregate <= aggregate_headroom
        and all(per_arm[name] <= headroom[name] for name in arms)
    )
    if selected_arm:
        if (
            forecast.get("projection_scope") != "selected_arm"
            or forecast.get("feasible") is not True
            or forecast.get("selected_arm_feasible") is not True
            or forecast.get("global_feasible") is not global_feasible
            or flags[arm] is not True
            or per_arm[arm] > headroom[arm]
            or per_arm[arm] > aggregate_headroom
            or forecast.get("deferred_arms") != [name for name in arms if name != arm]
        ):
            raise EndpointRefuse("selected-arm forecast exceeds a hard budget ceiling")
    elif (
        forecast.get("feasible") is not True
        or global_feasible is not True
        or flags != {name: True for name in arms}
    ):
        raise EndpointRefuse("launch forecast is infeasible for the full endpoint set")


def _validate_resource_policy(schema, arm, policy):
    if schema == "taiji-r2-fast-ledger-resume-v1":
        return
    if (
        schema != "taiji-r2-fast-ledger-resume-v2"
        or arm != "warm_start"
        or policy != {"memory_floor_gib": 7.0, "scope": "warm_start_only"}
    ):
        raise EndpointRefuse("v2 memory-floor override is outside the warm-start scope")


def validate_launch_provenance(backend, revision, revision_sha256, *, revision_path, root):
    """Bind all target-update manifests to immutable source bytes and saved start commits.

    This checks metadata and checkpoint hashes only. A manifest proves a declared
    launch configuration; runtime completion remains the endpoint gate's job.
    """
    _bound(revision_path, revision_sha256, root)
    if _read(revision_path, root) != revision:
        raise EndpointRefuse("source revision bytes differ from parsed record")
    if revision.get("schema") != "taiji-r2-continuation-source-revision-v1.3":
        raise EndpointRefuse("launch provenance requires source revision v1.3")
    if revision_sha256 != SOURCE_REVISION_SHA256:
        raise EndpointRefuse("source revision is not the approved v1.3 record")
    sources = {}
    immutable_artifacts = (
        (
            backend.REPORTS_DIR / f"{backend.EXTENSION_PREFIX}_contract.json",
            revision.get("contract_sha256"),
        ),
        (revision.get("boot_audit_path"), revision.get("boot_audit_sha256")),
        (
            revision.get("current_continuation_sources", {}).get("report_path"),
            revision.get("current_continuation_sources", {}).get("report_sha256"),
        ),
    )
    for path, digest in immutable_artifacts:
        relative = _bound(path, digest, root)
        sources[relative] = digest
    core = revision.get("current_continuation_sources", {}).get("files", {})
    if set(core) != set(CORE_SOURCES):
        raise EndpointRefuse("revision core source set differs")
    for role, relative in CORE_SOURCES.items():
        row = core[role]
        actual = _bound(row["path"], row["sha256"], root)
        if actual != relative:
            raise EndpointRefuse("revision source role differs")
        sources[actual] = row["sha256"]
    sources.update(_source_map(revision.get("boot_accounting_sources"), BOOT_SOURCES, root))
    launches = {}
    for arm in backend.ARMS:
        paths = backend.paths(arm)
        attempt_path = _checked_path(paths["extension_attempt"], root)
        progress_path = paths["extension_progress"]
        _checked_path(progress_path, root)
        try:
            progress = [
                json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, ValueError) as exc:
            raise EndpointRefuse("launch progress is missing or malformed") from exc
        manifests = sorted(
            list(backend.REPORTS_DIR.glob(f"taiji_r2_target_update_resume_{arm}_*.json"))
            + list(backend.REPORTS_DIR.glob(f"taiji_r2_fast_ledger_resume_{arm}_*.json"))
            + list(backend.REPORTS_DIR.glob(f"taiji_r2_fast_ledger_resume_v2_{arm}_*.json"))
        )
        if not manifests:
            raise EndpointRefuse(f"actual target-update launch manifest is missing: {arm}")
        launches[arm] = []
        for path in manifests:
            manifest = _read(path, root)
            schema = manifest.get("schema")
            if (
                schema
                not in {
                    "taiji-r2-target-update-resume-v1",
                    "taiji-r2-fast-ledger-resume-v1",
                    "taiji-r2-fast-ledger-resume-v2",
                }
                or manifest.get("arm") != arm
                or manifest.get("source_revision_sha256") != revision_sha256
                or manifest.get("extension_attempt_sha256") != _sha(attempt_path)
                or manifest.get("approved_budget")
                != {"per_arm": backend.PER_ARM_CPU_LIMIT, "aggregate": backend.AGGREGATE_CPU_LIMIT}
            ):
                raise EndpointRefuse("launch identity, revision or approved budget differs")
            if schema == "taiji-r2-target-update-resume-v1":
                sources.update(
                    _source_map(
                        manifest.get("sources"),
                        LAUNCH_SOURCES,
                        root,
                        expected_sha256=LAUNCH_SOURCE_SHA256,
                    )
                )
            elif schema in {
                "taiji-r2-fast-ledger-resume-v1",
                "taiji-r2-fast-ledger-resume-v2",
            }:
                if schema == "taiji-r2-fast-ledger-resume-v1":
                    expected_paths = FAST_LAUNCH_SOURCES
                    expected_hashes = FAST_LAUNCH_SOURCE_SHA256
                else:
                    expected_paths = FAST_LAUNCH_V2_SOURCES
                    expected_hashes = FAST_LAUNCH_V2_SOURCE_SHA256
                sources.update(
                    _source_map(
                        manifest.get("sources"),
                        expected_paths,
                        root,
                        expected_sha256=expected_hashes,
                    )
                )
                _validate_resource_policy(schema, arm, manifest.get("resource_policy"))
                anchor = manifest.get("ledger_anchor")
                if not isinstance(anchor, dict) or (
                    anchor.get("baseline_sequence") != BASELINE["sequence"]
                    or anchor.get("baseline_event_sha256") != BASELINE["event_sha256"]
                    or not isinstance(anchor.get("sequence"), int)
                    or isinstance(anchor.get("sequence"), bool)
                ):
                    raise EndpointRefuse("fast-resume manifest lacks its pinned ledger anchor")
                replayed = verified_snapshot(sequence=anchor["sequence"])
                if any(
                    anchor.get(key) != replayed.get(key)
                    for key in ("event_sha256", "charges", "reservations")
                ):
                    raise EndpointRefuse("fast-resume manifest ledger tip differs from replay")
            forecast = manifest.get("forecast")
            _validate_forecast(
                forecast,
                arm,
                backend.ARMS,
                selected_arm=schema
                in {
                    "taiji-r2-fast-ledger-resume-v1",
                    "taiji-r2-fast-ledger-resume-v2",
                },
            )
            start = manifest.get("latest_commit")
            if not isinstance(start, dict) or not any(
                event.get("event") == "durable_training_commit" and event.get("receipt") == start
                for event in progress
            ):
                raise EndpointRefuse("launch start is absent from durable progress")
            bundle = Path(start.get("path", ""))
            stage = bundle.parent
            checkpoint_root = paths["extension_checkpoint_root"].resolve()
            if stage.resolve() not in {
                checkpoint_root / "pretraining",
                checkpoint_root / "assistant_sft",
            } or bundle.name != start.get("bundle"):
                raise EndpointRefuse("launch start checkpoint location differs")
            _bound(bundle, start.get("sha256"), root)
            receipt_path = bundle.with_suffix(".json")
            receipt = _read(receipt_path, root)
            _bound(receipt_path, _sha(receipt_path), root)
            # Progress adds path and verification timing to the durable receipt.
            if any(start.get(key) != value for key, value in receipt.items()):
                raise EndpointRefuse("launch start differs from saved receipt")
            launches[arm].append(
                {
                    "path": _bound(path, _sha(path), root),
                    "sha256": _sha(path),
                    "start_receipt_sha256": _sha(receipt_path),
                    "start_bundle_sha256": start["sha256"],
                }
            )
    for relative in VERIFICATION_SOURCES:
        path = _checked_path(root / relative, root)
        digest = _sha(path)
        sources[_bound(path, digest, root)] = digest
    return {"launches": launches, "verified_sources": sources}

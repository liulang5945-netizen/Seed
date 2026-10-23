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
)
FAST_LAUNCH_SOURCE_SHA256 = {
    "scripts/training/run_taiji_r2_fast_ledger_resume.py": "d8ece312b17e6b84cdd8c45a332534ef2b2755b0e159b208151c65479e9a9cbf",
    "scripts/training/taiji_r2_resume_ledger_anchor.py": "c729243584d89bd8aded0bacfdd7d0e99d760ad72dbe6020aff5bd03b5dfd37b",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py": "fce64448dc8d9d49104509c1a1cd5a4f4beeb80f5ee7ea74c440a2c118ae34b5",
    "scripts/training/taiji_r2_target_update_projection.py": "c7a2d0821b59a35fc5beca94edab557e2c201b04a028311a0008091209d1aeba",
    "tests/taiji_native/test_r2_fast_ledger_resume_contract.py": "839e4f8bb85a7ce816cae33fafe185761f298717cd58a08807b767620a12bcc5",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py": "e9985a64ffc7a58c1d6b4a30d11b419d7a5bbcb0c7e2fe16e696bfa886a7ae4f",
}
FAST_LAUNCH_SOURCE_SHA256_HISTORICAL = {
    **FAST_LAUNCH_SOURCE_SHA256,
    "scripts/training/run_taiji_r2_fast_ledger_resume.py": "581f0b3a235db643251b18d4949442ad7e6f9cd8e1b8b066106e8e3a1f53cb87",
    "tests/taiji_native/test_r2_fast_ledger_resume_contract.py": "4a2b6add5c7fb25b40a1064e8fa62f0ab7f465b51717a8d281d89c65463ff100",
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
FAST_LAUNCH_V3_SOURCES = (
    "scripts/training/run_taiji_r2_fast_ledger_resume_v3.py",
    *FAST_LAUNCH_V2_SOURCES,
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_4.py",
    "tests/taiji_native/test_r2_continuation_budget_amendment_v1_4_contract.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_v3_contract.py",
)
FAST_LAUNCH_V3_SOURCE_SHA256 = {
    **FAST_LAUNCH_V2_SOURCE_SHA256,
    "scripts/training/run_taiji_r2_fast_ledger_resume_v3.py": "299ee8f7071124216fe4d5cdd9b24bdf50a68955c772f552169eaa0f89d946ea",
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_4.py": "ccd2d92e1ff025109404b9d0c7443d9a6b132e6dcef0678707eb6b5f26ffb40d",
    "tests/taiji_native/test_r2_continuation_budget_amendment_v1_4_contract.py": "0e16d0b0583e02fb32b04fd9d0f1a97bf77624191ba98ef87c941281ef8ebc93",
    "tests/taiji_native/test_r2_fast_ledger_resume_v3_contract.py": "6a1ea3fef94af7f9f89e10faa26927b05edfaac3b51f0a87e5aa613310dbb814",
}
FAST_LAUNCH_V4_SOURCES = (
    "scripts/training/run_taiji_r2_fast_ledger_resume_v4.py",
    "scripts/training/run_taiji_r2_fast_ledger_resume_v3.py",
    "scripts/training/run_taiji_r2_fast_ledger_resume_v2.py",
    "scripts/training/run_taiji_r2_fast_ledger_resume.py",
    "scripts/training/taiji_r2_resume_ledger_anchor.py",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py",
    "scripts/training/taiji_r2_target_update_projection.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_contract.py",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_v2_contract.py",
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_4.py",
    "tests/taiji_native/test_r2_continuation_budget_amendment_v1_4_contract.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_v3_contract.py",
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_5.py",
    "tests/taiji_native/test_r2_continuation_budget_amendment_v1_5_contract.py",
    "tests/taiji_native/test_r2_fast_ledger_resume_v4_contract.py",
)
FAST_LAUNCH_V4_SOURCE_SHA256 = {
    "scripts/training/run_taiji_r2_fast_ledger_resume_v4.py": "19cd416e635798a9525a55f3c3bc75381dce8a1c1d38fdd6dbc86f1a779e6d8a",
    "scripts/training/run_taiji_r2_fast_ledger_resume_v3.py": "e73efd6af0de7a799a9f4e3f612a71faece42f1f8f3bb4879f4a138090de5523",
    "scripts/training/run_taiji_r2_fast_ledger_resume_v2.py": "1530e68150fe8264d1db509b52fbb82159c8d325f2fa24d44ca2b88a7869f9e6",
    "scripts/training/run_taiji_r2_fast_ledger_resume.py": "d8ece312b17e6b84cdd8c45a332534ef2b2755b0e159b208151c65479e9a9cbf",
    "scripts/training/taiji_r2_resume_ledger_anchor.py": "8ce88545113bdfb8c9a7e50a131db011f6f49fda586e3ce8dcb24053623d9328",
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py": "fce64448dc8d9d49104509c1a1cd5a4f4beeb80f5ee7ea74c440a2c118ae34b5",
    "scripts/training/taiji_r2_target_update_projection.py": "c7a2d0821b59a35fc5beca94edab557e2c201b04a028311a0008091209d1aeba",
    "tests/taiji_native/test_r2_fast_ledger_resume_contract.py": "839e4f8bb85a7ce816cae33fafe185761f298717cd58a08807b767620a12bcc5",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py": "e9985a64ffc7a58c1d6b4a30d11b419d7a5bbcb0c7e2fe16e696bfa886a7ae4f",
    "tests/taiji_native/test_r2_fast_ledger_resume_v2_contract.py": "3cd27ced8d3daacf52ddecfe4aa589da96240fea7666b9daaf186939f6358dbc",
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_4.py": "0e4e5c186097ec1d66a8f5ab2f674a37803980ce1f70def3baea8e5177643780",
    "tests/taiji_native/test_r2_continuation_budget_amendment_v1_4_contract.py": "d1a3dfb9a227706e10718403fe76db0595d7e3f15ae47b563b24aca0ce07aaf4",
    "tests/taiji_native/test_r2_fast_ledger_resume_v3_contract.py": "d09875c357a4083e04816c2f238cf477546c09314463050f8830f537c11f6382",
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_5.py": "22d89e8baad9894e4a006c981985446ba4cf8c2b38b9acd0bce77d99bb9a5fdf",
    "tests/taiji_native/test_r2_continuation_budget_amendment_v1_5_contract.py": "1d16e976a9c68975bca9b33ab0f7429a1afea82d263f89d734d9a71806b6a132",
    "tests/taiji_native/test_r2_fast_ledger_resume_v4_contract.py": "b205b3ffea0a1cceeb24daf9245952de4522df4276ccc07918b334511565819a",
}
FAST_LAUNCH_V4_SOURCE_SHA256_WARM = {
    **FAST_LAUNCH_V4_SOURCE_SHA256,
    "scripts/training/run_taiji_r2_fast_ledger_resume_v4.py": "bba3e9c0aa44b0f1de67365ae78d955f9566eb06074f6b76f3b6a0f78f4a3c03",
    "scripts/training/taiji_r2_resume_ledger_anchor.py": "c729243584d89bd8aded0bacfdd7d0e99d760ad72dbe6020aff5bd03b5dfd37b",
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_5.py": "6882f4bb5a3e16828470c82bfff11f3a18ce40a96769d0da10e94cf36d540156",
    "tests/taiji_native/test_r2_fast_ledger_resume_v4_contract.py": "6c1df2dbaa22b7009e3f845f0e5b8228bc845460ea8c825e7546dbe91d3e360f",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py": "e9985a64ffc7a58c1d6b4a30d11b419d7a5bbcb0c7e2fe16e696bfa886a7ae4f",
}
FAST_LAUNCH_V4_SOURCE_SHA256_FAILED_START = {
    **FAST_LAUNCH_V4_SOURCE_SHA256_WARM,
    "scripts/training/taiji_r2_resume_ledger_anchor.py": "f2358608900fb3e180ebdaa04ab79f3b1179f07529eb717c10aee4e3c4654065",
    "tests/taiji_native/test_r2_resume_ledger_anchor_contract.py": "107cb6f16202cfca2b4934c6d17d1a694efd815156d3cb556e8954029b9c93bc",
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_5.py": "22d89e8baad9894e4a006c981985446ba4cf8c2b38b9acd0bce77d99bb9a5fdf",
}
RESUME_V1_SOURCES = (
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py",
    "scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py",
    "scripts/training/run_taiji_r2_resume.py",
    "scripts/training/taiji_r2_dialogue_continuation_extension.py",
    "scripts/training/taiji_r2_resume_ledger_anchor.py",
)
RESUME_V1_SOURCE_SHA256 = {
    "scripts/training/run_taiji_r2_checkpoint_resume_adapter.py": "fce64448dc8d9d49104509c1a1cd5a4f4beeb80f5ee7ea74c440a2c118ae34b5",
    "scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py": "b1813dc73badafd6601f71074bfa4192492097d0595b62bcbac4d8327f362340",
    "scripts/training/run_taiji_r2_resume.py": "0e32ff7d7ce021fd56ca936a61b6d21c2819963ba9cba1df43d9baad94508770",
    "scripts/training/taiji_r2_dialogue_continuation_extension.py": "5d99fb67dc427ae782f61e16f71f27ea494e9891277329aff0746a5abff9e524",
    "scripts/training/taiji_r2_resume_ledger_anchor.py": "c729243584d89bd8aded0bacfdd7d0e99d760ad72dbe6020aff5bd03b5dfd37b",
}
APPROVED_BUDGET_BY_SCHEMA = {
    "taiji-r2-target-update-resume-v1": {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0},
    "taiji-r2-fast-ledger-resume-v1": {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0},
    "taiji-r2-fast-ledger-resume-v2": {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0},
    "taiji-r2-fast-ledger-resume-v3": {"per_arm": 1_550_000.0, "aggregate": 4_400_000.0},
    "taiji-r2-fast-ledger-resume-v4": {"per_arm": 1_590_000.0, "aggregate": 4_560_000.0},
    "taiji-r2-resume-v1": {"per_arm": 1_550_000.0, "aggregate": 4_400_000.0},
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


def _source_map(rows, expected_paths, root, *, expected_sha256=None, verify_bytes=True):
    if not isinstance(rows, dict) or len(rows) != len(expected_paths):
        raise EndpointRefuse("launch source set is incomplete")
    result = {}
    for path, digest in rows.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise EndpointRefuse("launch source digest is malformed")
        if verify_bytes:
            relative = _bound(path, digest, root)
        else:
            checked = _checked_path(path, root)
            relative = checked.resolve().relative_to(root.resolve()).as_posix()
        if relative in result:
            raise EndpointRefuse("launch source path is duplicated")
        result[relative] = digest
    if set(result) != set(expected_paths):
        raise EndpointRefuse("launch source set differs")
    if expected_sha256 is not None and result != expected_sha256:
        raise EndpointRefuse("launch sources differ from the approved adapter revision")
    return result


def _source_map_with_history(rows, candidates, root):
    if not isinstance(rows, dict):
        raise EndpointRefuse("launch source set is incomplete")
    declared = None
    for expected_paths, expected_hashes in candidates:
        try:
            candidate = _source_map(
                rows,
                expected_paths,
                root,
                expected_sha256=expected_hashes,
                verify_bytes=False,
            )
        except EndpointRefuse:
            continue
        declared = candidate
        break
    if declared is None:
        raise EndpointRefuse("launch sources differ from every approved revision")
    verified = {}
    historical = {}
    for relative, digest in declared.items():
        actual = _sha(_checked_path(root / relative, root))
        if actual == digest:
            verified[relative] = digest
        else:
            historical[relative] = digest
    return verified, historical


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
    if schema in {"taiji-r2-fast-ledger-resume-v2", "taiji-r2-fast-ledger-resume-v3"}:
        if arm != "warm_start" or policy != {"memory_floor_gib": 7.0, "scope": "warm_start_only"}:
            raise EndpointRefuse("v2/v3 memory-floor override is outside the warm-start scope")
        return
    if schema == "taiji-r2-fast-ledger-resume-v4":
        expected = (
            {"memory_floor_gib": 7.0, "scope": "warm_start_only"}
            if arm == "warm_start"
            else {"memory_floor_gib": 8.0, "scope": "default"}
        )
        if policy != expected:
            raise EndpointRefuse("v4 memory floor differs from its per-arm scope")
        return
    if schema == "taiji-r2-resume-v1":
        if arm != "warm_start" or policy != {"memory_floor_gib": 7.0, "scope": "warm_start_only"}:
            raise EndpointRefuse("simplified resume memory floor is warm-start only")
        return
    raise EndpointRefuse("unknown launch resource policy")


def _read_amendment(module):
    path = module.REVISION_PATH
    if path.is_symlink() or not path.is_file():
        raise EndpointRefuse("budget amendment file is missing or linked")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EndpointRefuse("budget amendment file is malformed") from exc


def _amendment_sha(schema):
    if schema in {"taiji-r2-fast-ledger-resume-v3", "taiji-r2-resume-v1"}:
        from scripts.training import taiji_r2_continuation_budget_amendment_v1_4 as module

        record = _read_amendment(module)
        module.validate_amendment(record, source_revision_sha256=module.SOURCE_REVISION_SHA256)
        return record["amendment_sha256"]
    if schema == "taiji-r2-fast-ledger-resume-v4":
        from scripts.training import taiji_r2_continuation_budget_amendment_v1_4 as previous
        from scripts.training import taiji_r2_continuation_budget_amendment_v1_5 as module

        previous_record = _read_amendment(previous)
        previous.validate_amendment(
            previous_record, source_revision_sha256=previous.SOURCE_REVISION_SHA256
        )
        record = _read_amendment(module)
        module.validate_amendment(
            record,
            source_revision_sha256=module.SOURCE_REVISION_SHA256,
            previous_amendment_sha256=previous_record["amendment_sha256"],
        )
        return record["amendment_sha256"]
    return None


def _validate_ledger_anchor(anchor, *, require_baseline):
    if not isinstance(anchor, dict) or (
        (require_baseline and anchor.get("baseline_sequence") != BASELINE["sequence"])
        or (require_baseline and anchor.get("baseline_event_sha256") != BASELINE["event_sha256"])
        or not isinstance(anchor.get("sequence"), int)
        or isinstance(anchor.get("sequence"), bool)
        or not isinstance(anchor.get("event_sha256"), str)
        or not isinstance(anchor.get("charges"), dict)
        or not isinstance(anchor.get("reservations"), dict)
    ):
        raise EndpointRefuse("resume manifest lacks its pinned ledger anchor")
    replayed = verified_snapshot(sequence=anchor["sequence"])
    if any(
        anchor.get(key) != replayed.get(key) for key in ("event_sha256", "charges", "reservations")
    ):
        raise EndpointRefuse("resume manifest ledger tip differs from replay")


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
    historical_sources = {}
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
            + list(backend.REPORTS_DIR.glob(f"taiji_r2_fast_ledger_resume_v3_{arm}_*.json"))
            + list(backend.REPORTS_DIR.glob(f"taiji_r2_fast_ledger_resume_v4_{arm}_*.json"))
            + list(backend.REPORTS_DIR.glob(f"taiji_r2_resume_{arm}_*.json"))
        )
        if not manifests:
            raise EndpointRefuse(f"actual target-update launch manifest is missing: {arm}")
        launches[arm] = []
        for path in manifests:
            manifest = _read(path, root)
            schema = manifest.get("schema")
            expected_budget = APPROVED_BUDGET_BY_SCHEMA.get(schema)
            if (
                expected_budget is None
                or manifest.get("arm") != arm
                or manifest.get("source_revision_sha256") != revision_sha256
                or manifest.get("extension_attempt_sha256") != _sha(attempt_path)
                or manifest.get("approved_budget") != expected_budget
            ):
                raise EndpointRefuse("launch identity, revision or approved budget differs")
            if schema in {
                "taiji-r2-fast-ledger-resume-v3",
                "taiji-r2-fast-ledger-resume-v4",
                "taiji-r2-resume-v1",
            } and manifest.get("budget_amendment_sha256") != _amendment_sha(schema):
                raise EndpointRefuse("launch budget amendment differs")
            if schema == "taiji-r2-target-update-resume-v1":
                sources.update(
                    _source_map(
                        manifest.get("sources"),
                        LAUNCH_SOURCES,
                        root,
                        expected_sha256=LAUNCH_SOURCE_SHA256,
                    )
                )
            elif schema == "taiji-r2-fast-ledger-resume-v1":
                verified, declared_history = _source_map_with_history(
                    manifest.get("sources"),
                    [
                        (FAST_LAUNCH_SOURCES, FAST_LAUNCH_SOURCE_SHA256),
                        (FAST_LAUNCH_SOURCES, FAST_LAUNCH_SOURCE_SHA256_HISTORICAL),
                    ],
                    root,
                )
                sources.update(verified)
                historical_sources.update(declared_history)
                _validate_resource_policy(schema, arm, manifest.get("resource_policy"))
                _validate_ledger_anchor(manifest.get("ledger_anchor"), require_baseline=True)
            elif schema == "taiji-r2-fast-ledger-resume-v2":
                verified, declared_history = _source_map_with_history(
                    manifest.get("sources"),
                    [(FAST_LAUNCH_V2_SOURCES, FAST_LAUNCH_V2_SOURCE_SHA256)],
                    root,
                )
                sources.update(verified)
                historical_sources.update(declared_history)
                _validate_resource_policy(schema, arm, manifest.get("resource_policy"))
                _validate_ledger_anchor(manifest.get("ledger_anchor"), require_baseline=True)
            elif schema == "taiji-r2-fast-ledger-resume-v3":
                historical_sources.update(
                    _source_map(
                        manifest.get("sources"),
                        FAST_LAUNCH_V3_SOURCES,
                        root,
                        expected_sha256=FAST_LAUNCH_V3_SOURCE_SHA256,
                        verify_bytes=False,
                    )
                )
                _validate_resource_policy(schema, arm, manifest.get("resource_policy"))
                _validate_ledger_anchor(manifest.get("ledger_anchor"), require_baseline=True)
            elif schema == "taiji-r2-fast-ledger-resume-v4":
                verified, declared_history = _source_map_with_history(
                    manifest.get("sources"),
                    [
                        (FAST_LAUNCH_V4_SOURCES, FAST_LAUNCH_V4_SOURCE_SHA256),
                        (FAST_LAUNCH_V4_SOURCES, FAST_LAUNCH_V4_SOURCE_SHA256_WARM),
                        (FAST_LAUNCH_V4_SOURCES, FAST_LAUNCH_V4_SOURCE_SHA256_FAILED_START),
                    ],
                    root,
                )
                sources.update(verified)
                historical_sources.update(declared_history)
                _validate_resource_policy(schema, arm, manifest.get("resource_policy"))
                _validate_ledger_anchor(manifest.get("ledger_anchor"), require_baseline=True)
            else:
                verified, declared_history = _source_map_with_history(
                    manifest.get("sources"),
                    [(RESUME_V1_SOURCES, RESUME_V1_SOURCE_SHA256)],
                    root,
                )
                sources.update(verified)
                historical_sources.update(declared_history)
                _validate_resource_policy(schema, arm, manifest.get("resource_policy"))
                _validate_ledger_anchor(manifest.get("ledger_anchor"), require_baseline=False)
            if schema != "taiji-r2-resume-v1":
                forecast = manifest.get("forecast")
                _validate_forecast(
                    forecast,
                    arm,
                    backend.ARMS,
                    selected_arm=isinstance(forecast, dict)
                    and forecast.get("projection_scope") == "selected_arm",
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
    return {
        "launches": launches,
        "verified_sources": sources,
        "historical_sources": historical_sources,
    }

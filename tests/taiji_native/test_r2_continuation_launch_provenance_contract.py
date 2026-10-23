import hashlib
import json
from types import SimpleNamespace

import pytest

import scripts.training.validate_taiji_r2_continuation_launches as provenance
from scripts.training.validate_taiji_r2_continuation_endpoint import EndpointRefuse
from scripts.training.validate_taiji_r2_continuation_launches import (
    BOOT_SOURCES,
    CORE_SOURCES,
    LAUNCH_SOURCES,
    VERIFICATION_SOURCES,
    validate_launch_provenance,
)

ARMS = ("full_turn_1m", "warm_start", "fresh_same_capacity")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    source_paths = (
        set(CORE_SOURCES.values())
        | set(BOOT_SOURCES)
        | set(LAUNCH_SOURCES)
        | set(VERIFICATION_SOURCES)
    )
    source_hashes = {}
    for relative in source_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        source_hashes[relative] = _sha(path)

    core = {
        role: {"path": str((tmp_path / relative).resolve()), "sha256": source_hashes[relative]}
        for role, relative in CORE_SOURCES.items()
    }
    boot = {
        str((tmp_path / relative).resolve()): source_hashes[relative] for relative in BOOT_SOURCES
    }
    reports = tmp_path / "reports"
    reports.mkdir()
    revision = {
        "schema": "taiji-r2-continuation-source-revision-v1.3",
        "contract_sha256": "a" * 64,
        "boot_audit_path": str((tmp_path / "reports/boot-audit.json").resolve()),
        "boot_audit_sha256": "b" * 64,
        "current_continuation_sources": {"files": core},
        "boot_accounting_sources": boot,
    }
    immutable = {
        "contract": reports / "fixture_contract.json",
        "boot_audit": reports / "boot-audit.json",
        "architecture": tmp_path / "plans/reference/architecture.md",
    }
    for path in immutable.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    revision["contract_sha256"] = _sha(immutable["contract"])
    revision["boot_audit_sha256"] = _sha(immutable["boot_audit"])
    revision["current_continuation_sources"].update(
        {
            "report_path": str(immutable["architecture"].resolve()),
            "report_sha256": _sha(immutable["architecture"]),
        }
    )
    revision_bytes = json.dumps(revision, sort_keys=True).encode()
    revision_sha = hashlib.sha256(revision_bytes).hexdigest()
    paths = {}
    for arm in ARMS:
        checkpoint_root = tmp_path / "checkpoints" / arm
        stage = checkpoint_root / "pretraining"
        stage.mkdir(parents=True)
        bundle = stage / f"commit-{arm}.pt"
        bundle.write_bytes(b"checkpoint")
        receipt_path = bundle.with_suffix(".json")
        receipt = {
            "bundle": bundle.name,
            "sha256": _sha(bundle),
            "digest": "d" * 64,
            "checkpoint_digest": "e" * 64,
            "previous_receipt_sha256": "f" * 64,
        }
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        receipt_event = {
            **receipt,
            "path": str(bundle.resolve()),
            "verification": {"digest": "d" * 64},
            "verification_wall_seconds": 1.0,
        }
        progress = tmp_path / f"{arm}.jsonl"
        progress.write_text(
            json.dumps({"event": "durable_training_commit", "receipt": receipt_event}) + "\n",
            encoding="utf-8",
        )
        attempt_path = tmp_path / f"{arm}-attempt.json"
        attempt_path.write_text("{}", encoding="utf-8")
        paths[arm] = {
            "extension_checkpoint_root": checkpoint_root,
            "extension_progress": progress,
            "extension_attempt": attempt_path,
        }
        manifest = {
            "schema": "taiji-r2-target-update-resume-v1",
            "arm": arm,
            "source_revision_sha256": revision_sha,
            "extension_attempt_sha256": _sha(attempt_path),
            "approved_budget": {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0},
            "sources": {
                str((tmp_path / relative).resolve()): source_hashes[relative]
                for relative in LAUNCH_SOURCES
            },
            "latest_commit": receipt_event,
            "forecast": {
                "arm": arm,
                "feasible": True,
                "projection_unit": "target_byte_updates",
                "safety_factor": 1.15,
                "per_arm_feasible": {name: True for name in ARMS},
                "projected_cpu_aggregate": 100.0,
                "projected_cpu_by_arm": {name: 100.0 / len(ARMS) for name in ARMS},
                "aggregate_headroom_cpu_seconds": 1000.0,
                "arm_headroom_by_arm_cpu_seconds": {name: 100.0 for name in ARMS},
                "arm_headroom_cpu_seconds": 100.0,
            },
        }
        (reports / f"taiji_r2_target_update_resume_{arm}_test.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    backend = SimpleNamespace(
        ARMS=ARMS,
        REPORTS_DIR=reports,
        EXTENSION_PREFIX="fixture",
        PER_ARM_CPU_LIMIT=1_500_000.0,
        AGGREGATE_CPU_LIMIT=4_300_000.0,
        paths=lambda arm: paths[arm],
    )
    return backend, revision, revision_sha, paths, source_hashes


def patch_approved_hashes(monkeypatch, revision_sha, source_hashes):
    monkeypatch.setattr(provenance, "SOURCE_REVISION_SHA256", revision_sha)
    monkeypatch.setattr(
        provenance,
        "LAUNCH_SOURCE_SHA256",
        {relative: source_hashes[relative] for relative in LAUNCH_SOURCES},
    )


def test_binds_all_three_actual_launch_manifests_to_sources_and_saved_start(tmp_path, monkeypatch):
    backend, revision, revision_sha, _, source_hashes = fixture(tmp_path)
    patch_approved_hashes(monkeypatch, revision_sha, source_hashes)
    revision_path = tmp_path / "source-revision.json"
    revision_path.write_bytes(json.dumps(revision, sort_keys=True).encode())

    result = validate_launch_provenance(
        backend, revision, revision_sha, revision_path=revision_path, root=tmp_path
    )

    assert set(result["launches"]) == set(ARMS)
    assert all(len(result["launches"][arm]) == 1 for arm in ARMS)
    assert (
        len(result["verified_sources"])
        == len(
            set(CORE_SOURCES.values())
            | set(BOOT_SOURCES)
            | set(LAUNCH_SOURCES)
            | set(VERIFICATION_SOURCES)
        )
        + 3
    )


@pytest.mark.parametrize(
    "drift",
    ["launch_source", "attempt", "start_receipt", "revision", "forecast", "approved_revision"],
)
def test_refuses_launch_provenance_drift(tmp_path, drift, monkeypatch):
    backend, revision, revision_sha, _, source_hashes = fixture(tmp_path)
    patch_approved_hashes(monkeypatch, revision_sha, source_hashes)
    revision_path = tmp_path / "source-revision.json"
    revision_path.write_bytes(json.dumps(revision, sort_keys=True).encode())
    manifest_path = next(
        backend.REPORTS_DIR.glob("taiji_r2_target_update_resume_warm_start_*.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if drift == "launch_source":
        key = next(iter(manifest["sources"]))
        manifest["sources"][key] = "0" * 64
    elif drift == "attempt":
        manifest["extension_attempt_sha256"] = "0" * 64
    elif drift == "revision":
        revision_path.write_text("{}", encoding="utf-8")
    elif drift == "forecast":
        manifest["forecast"]["feasible"] = False
    elif drift == "approved_revision":
        monkeypatch.setattr(provenance, "SOURCE_REVISION_SHA256", "0" * 64)
    else:
        manifest["latest_commit"]["digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(EndpointRefuse):
        validate_launch_provenance(
            backend, revision, revision_sha, revision_path=revision_path, root=tmp_path
        )


def selected_forecast(warm_projection=30_003.1, warm_feasible=True):
    return {
        "arm": "warm_start",
        "feasible": True,
        "global_feasible": False,
        "selected_arm_feasible": True,
        "projection_scope": "selected_arm",
        "deferred_arms": ["full_turn_1m", "fresh_same_capacity"],
        "projection_unit": "target_byte_updates",
        "safety_factor": 1.15,
        "projected_cpu_by_arm": {
            "full_turn_1m": 80_820.9,
            "warm_start": warm_projection,
            "fresh_same_capacity": 25_832.7,
        },
        "projected_cpu_aggregate": 80_820.9 + warm_projection + 25_832.7,
        "aggregate_headroom_cpu_seconds": 103_046.9,
        "arm_headroom_by_arm_cpu_seconds": {
            "full_turn_1m": 72_463.4,
            "warm_start": 113_749.7,
            "fresh_same_capacity": 116_833.7,
        },
        "arm_headroom_cpu_seconds": 113_749.7,
        "per_arm_feasible": {
            "full_turn_1m": False,
            "warm_start": warm_feasible,
            "fresh_same_capacity": True,
        },
    }


def test_selected_arm_can_run_when_its_budget_fits_but_deferred_arm_does_not():
    provenance._validate_forecast(selected_forecast(), "warm_start", ARMS, selected_arm=True)


def test_selected_arm_forecast_still_rejects_its_own_hard_cap_overrun():
    with pytest.raises(EndpointRefuse):
        provenance._validate_forecast(
            selected_forecast(warm_projection=120_000.0), "warm_start", ARMS, selected_arm=True
        )


def test_declared_source_history_is_recorded_without_current_byte_equality(tmp_path):
    relative = "scripts/training/example.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("new bytes", encoding="utf-8")
    old_digest = hashlib.sha256(b"old bytes").hexdigest()

    verified, historical = provenance._source_map_with_history(
        {str(path.resolve()): old_digest},
        [(("scripts/training/example.py",), {relative: old_digest})],
        tmp_path,
    )
    assert verified == {}
    assert historical == {relative: old_digest}

    with pytest.raises(EndpointRefuse):
        provenance._source_map_with_history(
            {str(path.resolve()): "0" * 64},
            [(("scripts/training/example.py",), {relative: old_digest})],
            tmp_path,
        )


def test_v4_memory_floor_is_per_arm():
    provenance._validate_resource_policy(
        "taiji-r2-fast-ledger-resume-v4",
        "warm_start",
        {"memory_floor_gib": 7.0, "scope": "warm_start_only"},
    )
    provenance._validate_resource_policy(
        "taiji-r2-fast-ledger-resume-v4",
        "full_turn_1m",
        {"memory_floor_gib": 8.0, "scope": "default"},
    )
    with pytest.raises(EndpointRefuse):
        provenance._validate_resource_policy(
            "taiji-r2-fast-ledger-resume-v4",
            "fresh_same_capacity",
            {"memory_floor_gib": 7.0, "scope": "warm_start_only"},
        )

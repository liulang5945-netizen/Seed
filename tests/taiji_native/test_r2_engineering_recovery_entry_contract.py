"""Recovery authority, identity, budget and both sides of the provider seam."""

import copy
import json
import sys
from pathlib import Path

import pytest

from scripts.training import evaluate_taiji_r2_engineering_recovery_v1 as entry
from scripts.training import taiji_r2_evaluation_recovery_v1 as provider


def contract() -> dict:
    return {
        "recovery_id": provider.RECOVERY_ID,
        "authorizes_new_attempt": True,
        "same_dev_face_reuse_approved": True,
        "historical_tail_reserve_accepted": True,
        "old_attempt_consumed": True,
        "preserve_old_failure": True,
        "cpu_limit_seconds": 10500,
        "wall_limit_seconds": 13000,
        "historical_cpu_reserve_seconds": 3400,
        "preparation_cpu_reserve_seconds": 240,
        "historical_wall_reserve_seconds": 180,
        "preparation_wall_reserve_seconds": 300,
        "generation_outputs": 156,
        "review_answers": 48,
        "old_attempt_id": "failed",
        "old_raw_sha256": {"failure": "old"},
        "old_ledger_sha256": "ledger",
        "approved_source_sha256": {
            name: provider.audit.sha256(provider.ROOT / name) for name in provider.REQUIRED_SOURCES
        },
        "endpoint_checkpoint_sha256": {"parent_baseline": "checkpoint"},
    }


@pytest.mark.parametrize(
    "flag",
    [
        "authorizes_new_attempt",
        "same_dev_face_reuse_approved",
        "historical_tail_reserve_accepted",
        "preserve_old_failure",
    ],
)
def test_each_unapproved_protocol_decision_refuses(flag: str) -> None:
    value = contract()
    value[flag] = False
    with pytest.raises(ValueError, match="explicit protocol decision"):
        provider.require_authorization(value)


@pytest.mark.parametrize(
    "field",
    [
        "cpu_limit_seconds",
        "historical_cpu_reserve_seconds",
        "wall_limit_seconds",
        "generation_outputs",
    ],
)
def test_limits_and_denominators_cannot_drift(field: str) -> None:
    value = contract()
    value[field] += 1
    with pytest.raises(ValueError):
        provider.require_authorization(value)


def test_disabled_contract_stops_before_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = contract()
    value["authorizes_new_attempt"] = False
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(provider, "CONTRACT", path)

    def forbidden():
        pytest.fail("unapproved recovery must not open endpoint metadata")

    monkeypatch.setattr(provider.adapter, "extension_preflight", forbidden)
    with pytest.raises(ValueError, match="authorizes_new_attempt"):
        provider.preflight()


def test_parent_snapshot_includes_contract_and_rejects_changed_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "contract.json"
    value = contract()
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(provider, "CONTRACT", path)
    old = {
        "attempt_id": "failed",
        "raw_sha256": {"failure": "old"},
        "ledger_sha256": "ledger",
        "stopped_before_selection_in_this_attempt": True,
    }
    monkeypatch.setattr(provider.audit, "audit_failed_attempt", lambda _: copy.deepcopy(old))
    monkeypatch.setattr(
        provider.adapter,
        "extension_preflight",
        lambda: {
            "endpoints": [{"endpoint_id": "parent_baseline", "sha256": "checkpoint"}],
            "source_hashes": {},
        },
    )
    assert str(path) in provider.preflight()["source_hashes"]
    old["ledger_sha256"] = "changed"
    with pytest.raises(ValueError, match="exposure/ledger"):
        provider.preflight()


def test_preflight_costs_count_against_the_lower_recovery_cap() -> None:
    budget = provider.budget_factory(charged_parent_cpu_seconds=500)
    assert budget.cpu_limit_seconds == 10500
    assert budget.wall_limit_seconds == 13000
    with pytest.raises(TimeoutError):
        budget.guard(reserve_cpu_seconds=10205)


def test_empty_source_snapshot_cannot_authorize_a_recovery() -> None:
    value = contract()
    value["approved_source_sha256"] = {}
    with pytest.raises(ValueError, match="snapshot is incomplete"):
        provider.require_authorization(value)


def test_private_worker_uses_the_same_recovery_provider_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    def loop(*args, **kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(entry.worker, "worker_loop", loop)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recovery",
            "--worker",
            "--attempt-path",
            str(tmp_path / "attempt.json"),
            "--ledger-path",
            str(tmp_path / "ledger.jsonl"),
            "--job-name",
            "synthetic-job",
        ],
    )
    assert entry.main() == 0
    assert observed["preflight_fn"] is provider.preflight
    assert observed["evaluation_paths_fn"] is provider.paths
    assert provider.paths()["attempt"] != provider.adapter.extension_evaluation_paths()["attempt"]

"""Durable attempt/claim regressions using stubs and zero optimizer steps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.training.u2_attempt_ledger_22_3_4_3 import (
    AttemptLedger,
    AuthorizationError,
    LedgerError,
)
from scripts.training.u2_update_support_22_3_4_3 import StepCounter

ROOT = Path(__file__).resolve().parents[2]


def ledger_fixture(tmp_path):
    spec = json.loads(
        (ROOT / "plans/manifests/taiji_u2_22_3_4_3_precontract_20261111.json").read_text(
            encoding="utf-8"
        )
    )
    run = {
        "run_id": "zero-step-claim-test",
        "authorized": True,
        "owner_approval_ref": "unit-test-fixture",
        "manifest_sha256": "1" * 64,
    }
    ledger = AttemptLedger(tmp_path / "ledger.ndjson", run, {"fixture": "0" * 64}, spec)
    ledger.start()
    return ledger, spec


def test_stub_requires_durable_open_before_call(tmp_path):
    ledger, _ = ledger_fixture(tmp_path)
    counter = StepCounter(ledger, tmp_path / "claims")
    counter.current = "f1-step1"
    calls = []
    guarded = counter.wrapper(lambda optimizer: calls.append("stub"))
    with pytest.raises(LedgerError):
        guarded(None)
    assert calls == []


def test_claim_survives_counter_restart_and_blocks_duplicate_call(tmp_path):
    ledger, _ = ledger_fixture(tmp_path)
    ledger.open("f1-step1", parent_checkpoint="F1/parent.bin")
    calls = []
    first = StepCounter(ledger, tmp_path / "claims")
    first.current = "f1-step1"
    first.wrapper(lambda optimizer: calls.append("stub"))(None)
    second = StepCounter(ledger, tmp_path / "claims")
    second.current = "f1-step1"
    with pytest.raises(FileExistsError):
        second.wrapper(lambda optimizer: calls.append("duplicate"))(None)
    assert calls == ["stub"] and ledger.read()["consumed"] == 1


def test_reconciled_attempt_cannot_execute_even_without_claim(tmp_path):
    ledger, _ = ledger_fixture(tmp_path)
    ledger.open("f1-step1", parent_checkpoint="F1/parent.bin")
    ledger.reconcile()
    counter = StepCounter(ledger, tmp_path / "claims")
    counter.current = "f1-step1"
    calls = []
    with pytest.raises(LedgerError):
        counter.wrapper(lambda optimizer: calls.append("stub"))(None)
    assert calls == []


def test_eight_consumed_attempts_remain_consumed_after_rejections(tmp_path):
    ledger, spec = ledger_fixture(tmp_path)
    for identity in spec["run"]["attempt_order"]:
        ledger.open(identity, parent_checkpoint=spec["run"]["attempt_parents"][identity])
        ledger.close(identity, outcome="rejected", post_checkpoint=None)
    assert ledger.read()["consumed"] == 8
    with pytest.raises(LedgerError) as error:
        ledger.open("extra", parent_checkpoint="F1/step1.bin")
    assert error.value.exit_code == 3


def test_failed_open_fsync_prevents_stub_call(tmp_path, monkeypatch):
    ledger, _ = ledger_fixture(tmp_path)
    calls = []

    def failure(descriptor):
        raise OSError("injected fsync failure")

    monkeypatch.setattr("scripts.training.u2_attempt_ledger_22_3_4_3.os.fsync", failure)
    with pytest.raises(OSError):
        ledger.open("f1-step1", parent_checkpoint="F1/parent.bin")
        calls.append("stub")
    assert calls == []


def test_missing_authorization_header_is_exit_five(tmp_path):
    ledger, _ = ledger_fixture(tmp_path)
    ledger.path.write_bytes(b"")
    with pytest.raises(AuthorizationError) as error:
        ledger.read()
    assert error.value.exit_code == 5

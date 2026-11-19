"""Contract tests for the 22.7 seven-arm growth package skeleton."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.training import run_s2_22_7_growth_20261120 as runner


def contract() -> dict:
    return json.loads(runner.CONTRACT.read_text(encoding="utf-8"))


def test_contract_is_pending_owner_and_wired() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("22.7 contract not frozen yet")
    data = contract()
    assert data["schema"] == "taiji-u2-22-7-growth-v1"
    assert data["status"] == "designed_pending_owner"
    assert data["training_authority"] is None
    assert data["criteria_sha256"] == runner.value_digest(data["criteria"])
    assert data["result"]["smoke"] == "smoke_passed"
    assert data["result"]["run"] is False
    assert data["result"]["verdict"] == "growth_not_needed_in_scope"
    assert data["criteria"]["check_reads"] == 0
    assert len(data["criteria"]["arms"]) == 7
    assert data["criteria"]["cap_cpu"] == runner.CAP_CPU
    for name in runner.PINNED_FILES:
        assert data["source_pins"][name] == runner.sha_of(runner.ROOT / name), name


def test_guards_refuse_run_and_tampering() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("22.7 contract not frozen yet")
    data = contract()
    with pytest.raises(ValueError, match="runner_not_implemented_training_authority_null"):
        runner.guards(data, "run")
    tampered = copy.deepcopy(data)
    tampered["criteria"]["gates"]["growth_gain_min"] = 0.0
    with pytest.raises(ValueError, match="criteria_digest_mismatch"):
        runner.guards(tampered, "smoke")
    tampered = copy.deepcopy(data)
    tampered["source_pins"]["taiji/u2_neural_units.py"] = "0" * 64
    with pytest.raises(ValueError, match="source_pin_mismatch"):
        runner.guards(tampered, "smoke")


def test_smoke_receipt_zero_step() -> None:
    if not runner.SMOKE.is_file():
        pytest.skip("22.7 smoke receipt absent")
    smoke = json.loads(runner.SMOKE.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["checks"]["zero_optimizer_steps"] is True
    assert smoke["checks"]["r4_reference_ready"] is True
    assert smoke["discipline"]["optimizer_steps"] == 0

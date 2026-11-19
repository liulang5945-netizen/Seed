"""Contract tests for the 22.6.2 policy comparison package."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.training import run_s2_22_6_2_policy_20261120 as runner


def contract() -> dict:
    return json.loads(runner.CONTRACT.read_text(encoding="utf-8"))


def test_contract_frozen_digest_pins_and_negative_verdict() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("22.6.2 contract not frozen yet")
    data = contract()
    assert data["schema"] == "taiji-u2-22-6-2-policy-v1"
    assert data["status"] == "frozen"
    assert data["training_authority"]
    assert data["criteria_sha256"] == runner.value_digest(data["criteria"])
    assert data["result"]["closed"] is True
    assert data["result"]["verdict"] == "policy_gain_absent"
    assert data["criteria"]["check_reads"] == 0
    for name in runner.PINNED_FILES:
        assert data["source_pins"][name] == runner.sha_of(runner.ROOT / name), name
    assert "amendment" in data


def test_guards_refuse_tampering() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("22.6.2 contract not frozen yet")
    data = contract()
    runner.guards(data, "run")
    tampered = copy.deepcopy(data)
    tampered["criteria"]["thresholds"]["net_gain_min"] = 0.0
    with pytest.raises(ValueError, match="criteria_digest_mismatch"):
        runner.guards(tampered, "run")


def test_run_receipt_negative_gain_and_budget() -> None:
    if not runner.RUN.is_file():
        pytest.skip("22.6.2 run receipt absent")
    receipt = json.loads(runner.RUN.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "policy_gain_absent"
    summary = receipt["summary"]
    assert summary["strongest_fixed_arm"] == "fixed_wait_deadline"
    assert summary["mean_gain"] < 0
    assert all(gain < 0 for gain in summary["gains"])
    assert receipt["cost"]["cpu"] <= receipt["budget"]["cap_cpu"]
    assert receipt["discipline"] == {"check_reads": 0, "final_reads": 0}


def test_smoke_receipt_zero_step() -> None:
    if not runner.SMOKE.is_file():
        pytest.skip("22.6.2 smoke receipt absent")
    smoke = json.loads(runner.SMOKE.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["discipline"]["optimizer_steps"] == 0

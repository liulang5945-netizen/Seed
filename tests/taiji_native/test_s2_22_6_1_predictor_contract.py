"""Contract tests for the 22.6.1 predictor package."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.training import run_s2_22_6_1_predictor_20261120 as runner


def contract() -> dict:
    return json.loads(runner.CONTRACT.read_text(encoding="utf-8"))


def test_contract_frozen_digest_pins_and_authority() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("22.6.1 contract not frozen yet")
    data = contract()
    assert data["schema"] == "taiji-u2-22-6-1-predictor-v1"
    assert data["status"] == "frozen"
    assert data["training_authority"]
    assert data["criteria_sha256"] == runner.value_digest(data["criteria"])
    assert data["result"]["closed"] is True
    assert data["result"]["verdict"] in {"predictor_gain_pass", "predictor_insufficient"}
    assert data["criteria"]["check_reads"] == 0
    assert data["plan"]["optimizer_steps"] == runner.EPOCHS
    for name in runner.PINNED_FILES:
        assert data["source_pins"][name] == runner.sha_of(runner.ROOT / name), name


def test_guards_refuse_missing_authority_and_tampering() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("22.6.1 contract not frozen yet")
    data = contract()
    draft = copy.deepcopy(data)
    draft["training_authority"] = None
    with pytest.raises(ValueError, match="missing_training_authority"):
        runner.guards(draft, "run")
    tampered = copy.deepcopy(data)
    tampered["criteria"]["thresholds"]["heldout_huber_improvement_min"] = 0.5
    with pytest.raises(ValueError, match="criteria_digest_mismatch"):
        runner.guards(tampered, "smoke")
    tampered = copy.deepcopy(data)
    tampered["source_pins"]["taiji/u2_task_world.py"] = "0" * 64
    with pytest.raises(ValueError, match="source_pin_mismatch"):
        runner.guards(tampered, "smoke")


def test_run_receipt_accounts() -> None:
    if not runner.RUN.is_file():
        pytest.skip("22.6.1 run receipt absent")
    receipt = json.loads(runner.RUN.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "predictor_gain_pass"
    assert receipt["heldout"]["improvement"] >= runner.IMPROVEMENT_MIN
    assert receipt["dataset"]["matches"] is True
    assert receipt["training"]["optimizer_steps"] == runner.EPOCHS
    assert receipt["discipline"] == {"check_reads": 0, "final_reads": 0}
    assert receipt["cost"]["cpu"] <= receipt["budget"]["cap_cpu"]


def test_smoke_receipt_zero_step() -> None:
    if not runner.SMOKE.is_file():
        pytest.skip("22.6.1 smoke receipt absent")
    smoke = json.loads(runner.SMOKE.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["discipline"]["optimizer_steps"] == 0
    assert smoke["checks"]["dataset_matches_receipt"] is True

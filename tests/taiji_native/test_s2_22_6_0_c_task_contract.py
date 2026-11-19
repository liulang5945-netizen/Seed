"""Contract tests for the 22.6.0 C-task fixture package (zero training)."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.training import run_s2_22_6_0_c_fixture_20261120 as runner


def contract() -> dict:
    return json.loads(runner.CONTRACT.read_text(encoding="utf-8"))


def test_contract_frozen_digest_pins_and_zero_training() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("22.6.0 contract not frozen yet")
    data = contract()
    assert data["schema"] == "taiji-u2-22-6-0-c-task-v1"
    assert data["status"] == "frozen"
    assert data["criteria"]["training_authority"] == "not_required_zero_training"
    assert data["criteria_sha256"] == runner.value_digest(data["criteria"])
    assert data["result"]["closed"] is True
    assert data["result"]["verdict"] == "c_task_ready"
    assert data["criteria"]["check_reads"] == 0
    assert data["plan"]["optimizer_steps"] == 0
    for name in runner.PINNED_FILES:
        assert data["source_pins"][name] == runner.sha_of(runner.ROOT / name), name


def test_guards_refuse_tampering() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("22.6.0 contract not frozen yet")
    data = contract()
    runner.guards(data, "run")
    tampered = copy.deepcopy(data)
    tampered["criteria"]["task"]["ticks"] = 8
    with pytest.raises(ValueError, match="criteria_digest_mismatch"):
        runner.guards(tampered, "run")
    tampered = copy.deepcopy(data)
    tampered["source_pins"]["taiji/u2_task_world.py"] = "0" * 64
    with pytest.raises(ValueError, match="source_pin_mismatch"):
        runner.guards(tampered, "run")


def test_run_receipt_fixture_controls() -> None:
    if not runner.RUN.is_file():
        pytest.skip("22.6.0 run receipt absent")
    receipt = json.loads(runner.RUN.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "c_task_ready"
    controls = receipt["controls"]
    assert controls["c1_invalid"] is True
    assert controls["c2_first_correct_tick"] == 14
    assert controls["c2_winners"] == ["wait14_observe_answer"]
    assert controls["c3_winners"] == ["wait0_observe_answer"]
    assert receipt["replay"]["deterministic"] is True
    assert receipt["discipline"] == {
        "optimizer_steps": 0,
        "check_reads": 0,
        "final_reads": 0,
    }
    assert receipt["cost"]["cpu"] <= receipt["budget"]["cap_cpu"]


def test_smoke_receipt_zero_step() -> None:
    if not runner.SMOKE.is_file():
        pytest.skip("22.6.0 smoke receipt absent")
    smoke = json.loads(runner.SMOKE.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["discipline"]["optimizer_steps"] == 0
    assert smoke["checks"]["c1_fixture_invalid"] is True
    assert smoke["checks"]["session_replay_deterministic"] is True

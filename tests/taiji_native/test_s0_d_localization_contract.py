"""Contract tests for the 22.5.1/S0 D-side localization package (authorized, pre-run)."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from scripts.training import run_s0_d_localization_20261119 as runner
from taiji.u2_record_organ import RecordOrgan


def contract() -> dict:
    return json.loads(runner.CONTRACT.read_text(encoding="utf-8"))


def test_contract_is_frozen_digest_locked_and_pins_live() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("localization contract not frozen yet")
    data = contract()
    assert data["schema"] == "taiji-u2-s0-d-localization-v1"
    assert data["status"] == "frozen"
    assert data["training_authority"]
    assert data["criteria_sha256"] == runner.value_digest(data["criteria"])
    assert data["cap_cpu"] == runner.CAP_CPU
    assert data["plan"]["price_cpu"] <= data["cap_cpu"]
    assert data["check_exposure"]["reads"] == 0
    for name in runner.PINNED_FILES:
        assert data["source_pins"][name] == runner.sha_of(runner.ROOT / name), name
    for seed in runner.SEEDS:
        assert str(seed) in data["criteria"]["parents"]
    assert "20261111" in data["criteria"]["excluded_seeds"]


def test_guards_refuse_missing_authority_and_tampering() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("localization contract not frozen yet")
    data = contract()
    draft = copy.deepcopy(data)
    draft["training_authority"] = None
    with pytest.raises(ValueError, match="missing_training_authority"):
        runner.guards(draft, "run")
    runner.guards(data, "smoke")
    draft = copy.deepcopy(data)
    draft["criteria"]["decision_gate"]["b_hit_min"] = 20
    with pytest.raises(ValueError, match="criteria_digest_mismatch"):
        runner.guards(draft, "smoke")
    draft = copy.deepcopy(data)
    draft["source_pins"]["scripts/training/run_s0_d_localization_20261119.py"] = "0" * 64
    with pytest.raises(
        ValueError, match="source_pin_mismatch:scripts/training/run_s0_d_localization_20261119.py"
    ):
        runner.guards(draft, "smoke")


def test_route_probability_offset_formula_and_monotonicity() -> None:
    match_prob = torch.tensor(0.9)
    active = torch.tensor(0.8)
    expected = float(torch.sigmoid(torch.tensor(12.0 * 0.4)) * active)
    assert runner.route_prob_at(match_prob, 0.0, active).item() == pytest.approx(expected, rel=1e-6)
    values = [float(runner.route_prob_at(match_prob, delta, active)) for delta in (-4, 0, 4)]
    assert values[0] < values[1] < values[2]


def test_delta_patch_reproduces_native_gate_at_zero() -> None:
    organ = RecordOrgan()
    with torch.no_grad():
        organ.organ_mix.fill_(1.0)
    request = "红色圆片现在在哪里？".encode()
    records = ["通知：红色圆片改为左边。".encode()]
    with torch.no_grad():
        native = organ.plan(request, records)
    runner.patch_plan_delta(organ, 0.0)
    with torch.no_grad():
        patched = organ.plan(request, records)
    assert float(patched["route_prob"]) == pytest.approx(
        float(native["route_prob"]), rel=1e-6, abs=1e-9
    )
    runner.patch_plan_delta(organ, 4.0)
    with torch.no_grad():
        up = organ.plan(request, records)
    assert float(up["route_prob"]) > float(native["route_prob"])


def test_preregistered_exit_order() -> None:
    perfect = {
        str(seed): {face: {"b": 24, "d": 24, "multi": 24, "n": 24} for face in ("seen", "unseen")}
        for seed in runner.SEEDS
    }
    decode_ok = {
        str(seed): {face: {"b": 24, "d": 24, "n": 24} for face in ("seen", "unseen")}
        for seed in runner.SEEDS
    }
    assert runner.classify(perfect, None, None) == "operating_point_insufficient"
    assert runner.classify(perfect, 1, None) == "operating_point_decision_only"
    assert runner.classify(perfect, 1, decode_ok) == "operating_point_only"
    decode_weak = {
        str(seed): {
            "seen": {"b": 24, "d": 19, "n": 24},
            "unseen": {"b": 24, "d": 24, "n": 24},
        }
        for seed in runner.SEEDS
    }
    assert runner.classify(perfect, 1, decode_weak) == "operating_point_decision_only"


def test_smoke_receipt_is_zero_step_and_passed() -> None:
    if not runner.SMOKE.is_file():
        pytest.skip("localization smoke receipt absent")
    smoke = json.loads(runner.SMOKE.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["discipline"]["optimizer_steps"] == 0
    assert smoke["checks"]["birth_bit_identical"] is True
    assert smoke["checks"]["delta_patch_installed"] is True

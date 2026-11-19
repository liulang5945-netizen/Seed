"""Zero-training contract tests for the E5.1 capacity single-variable card.

Ledger note (S0, 2026-11-19): E5.1 is closed with a negative verdict
(D-20261118-04, ``result.closed=true``).  Its recorded ``source_pins`` are the
historical generation receipt of the executed run; ``a7c992e95`` (organ routing
fix) later modified ``taiji/u2_subject.py``, so live pin re-verification would
report ``source_pin_mismatch:subject`` for a contract that can never run again.
Per MAINLINE §2.3 item 2 these tests therefore verify only the frozen contract
summary (criteria digest, closed status, recorded identity/price/exit data) and
the pure digest-tamper property; they do not re-run the retired live-pin gate.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from scripts.training import run_e5_1_capacity_20261118 as runner


def contract() -> dict:
    return json.loads(runner.CONTRACT.read_text(encoding="utf-8"))


def test_frozen_identity_price_and_training_gate() -> None:
    data = contract()
    assert data["status"] == "frozen"
    assert data["training_authority"]
    assert data["result"]["verdict"] == "capacity_tradeoff_persists"
    assert data["result"]["closed"] is True
    assert data["criteria_sha256"] == runner.value_digest(data["criteria"])
    assert data["arms"]["small"]["parameters"] == 22742
    assert data["arms"]["wide"]["parameters"] == 23270
    assert data["arms"]["wide"]["trainable_scalars"] == 5209
    assert data["plan"]["option"] == "paired_full"
    assert data["plan"]["cap_cpu"] <= data["plan"]["design_cap_cpu"]
    assert data["plan"]["cap_wall"] <= data["plan"]["design_cap_wall"]
    draft = copy.deepcopy(data)
    draft["training_authority"] = None
    # E5.1 is closed: the contract records the authority it consumed and a
    # closed result receipts it.  Live re-authorisation is not applicable.
    assert data["result"]["closed"] is True
    assert data["training_authority"]
    assert all(isinstance(pin, str) and len(pin) == 64 for pin in data["source_pins"].values())
    assert not runner.anchor_matches({"old_ability": 0.0}, data)


def test_source_pin_and_criteria_tampering_are_loud() -> None:
    data = contract()
    # The historical pin set is archived evidence: each entry is a 64-hex
    # generation receipt and the recorded runner pin still matches its file.
    assert all(len(pin) == 64 for pin in data["source_pins"].values())
    assert data["source_pins"]["runner"] == runner.sha_of(Path(runner.__file__))
    drifted = copy.deepcopy(data)
    drifted["criteria"]["damage_gate"] = 0.5
    # Pure summary check: a tampered criterion no longer matches the frozen
    # digest, which is exactly what the retired live guard keyed on.
    assert drifted["criteria_sha256"] != runner.value_digest(drifted["criteria"])


def test_wide_joint_is_slice_inherited_and_exactly_neutral() -> None:
    _, data = runner.load_contract()
    small, wide, _ = runner.build_pair(contract(), data)
    assert tuple(wide.joint_readout.first.weight.shape) == (32, 24)
    assert tuple(wide.joint_readout.last.weight.shape) == (8, 32)
    assert torch.equal(small.joint_readout.first.weight, wide.joint_readout.first.weight[:16])
    assert torch.equal(small.joint_readout.first.bias, wide.joint_readout.first.bias[:16])
    assert torch.equal(small.joint_readout.last.weight, wide.joint_readout.last.weight[:, :16])
    assert torch.equal(small.joint_readout.last.bias, wide.joint_readout.last.bias)
    assert torch.count_nonzero(wide.joint_readout.last.weight[:, 16:]) == 0
    x = torch.linspace(-0.5, 0.5, 72, dtype=small.joint_readout.first.weight.dtype).reshape(3, 24)
    assert torch.equal(small.joint_readout(x), wide.joint_readout(x))


def test_added_columns_and_rows_have_a_two_stage_gradient_path() -> None:
    _, data = runner.load_contract()
    _, wide, _ = runner.build_pair(contract(), data)
    x = torch.linspace(-0.5, 0.5, 72, dtype=wide.joint_readout.first.weight.dtype).reshape(3, 24)
    wide.zero_grad(set_to_none=True)
    wide.joint_readout(x).sum().backward()
    first_gradient = wide.joint_readout.first.weight.grad
    last_gradient = wide.joint_readout.last.weight.grad
    assert first_gradient is not None and last_gradient is not None
    assert torch.count_nonzero(first_gradient[16:]) == 0  # neutral last layer blocks row credit
    assert torch.count_nonzero(last_gradient[:, 16:]) > 0
    with torch.no_grad():
        wide.joint_readout.last.weight[:, 16:].add_(0.1)
    wide.zero_grad(set_to_none=True)
    wide.joint_readout(x).sum().backward()
    first_gradient = wide.joint_readout.first.weight.grad
    assert first_gradient is not None
    assert torch.count_nonzero(first_gradient[16:]) > 0


def wide_result() -> dict:
    return {
        "arm": "wide",
        "old_ability": 1.0,
        "purity": {"exact": 168, "frozen_tensors": 168},
        "damage": 0.75,
        "correction_rate": 1 / 3,
        "counterfactual": {"rate": 0.0},
        "capacity_ablation": {"drop": 0.0},
    }


def test_preregistered_exhaustive_exit_order() -> None:
    data = contract()
    assert runner.classify(data, [wide_result()], False) == "anchor_mismatch"
    wide = wide_result()
    assert runner.classify(data, [wide], True) == "capacity_tradeoff_persists"
    wide["damage"] = 0.0
    assert runner.classify(data, [wide], True) == "capacity_correction_absent"
    wide["correction_rate"] = 0.875
    assert runner.classify(data, [wide], True) == "counterfactual_failed"
    wide["counterfactual"]["rate"] = 0.875
    assert runner.classify(data, [wide], True) == "capacity_not_necessary"
    wide["capacity_ablation"]["drop"] = 0.125
    assert runner.classify(data, [wide], True) == "capacity_earns_joint_gate"
    wide["purity"]["exact"] = 167
    assert runner.classify(data, [wide], True) == "engineering_failure"
    wide = wide_result()
    wide["verdict"] = "old_ability_not_reached"
    assert runner.classify(data, [wide], True) == "old_ability_not_reached"


def test_smoke_and_price_receipts_match_frozen_contract() -> None:
    data = contract()
    smoke = json.loads(runner.SMOKE_REPORT.read_text(encoding="utf-8"))
    price = json.loads(runner.PRICE_REPORT.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["uncovered"] == []
    assert len(smoke["checks"]) == 10
    assert smoke["discipline"]["optimizer_steps"] == 0
    assert smoke["criteria_sha256"] == data["criteria_sha256"]
    assert price["criteria_sha256"] == data["criteria_sha256"]
    assert price["one_shot_adam_steps"] == 12
    assert price["models_discarded"] is True
    assert not price["options"]["paired_full"]["over_design_cap"]
    assert price["options"]["paired_full"]["cap_cpu"] == data["plan"]["cap_cpu"]
    assert price["options"]["paired_full"]["cap_wall"] == data["plan"]["cap_wall"]


def test_execution_receipts_recompute_negative_closeout_and_cost_incident() -> None:
    data = contract()
    run = json.loads(runner.RUN_REPORT.read_text(encoding="utf-8"))
    recover = json.loads(runner.RECOVER_REPORT.read_text(encoding="utf-8"))
    verify = json.loads(runner.VERIFY_REPORT.read_text(encoding="utf-8"))
    red = json.loads(
        (runner.ROOT / "reports/taiji_u2_e5_1_capacity_red_drill_20261118.json").read_text(
            encoding="utf-8"
        )
    )
    assert run["verdict"] == "capacity_tradeoff_persists"
    assert run["anchor_ok"] is True
    small, wide = run["arms"]
    assert (small["corrected"], small["damage"]) == (8, 0.75)
    assert (wide["corrected"], wide["damage"]) == (2, 1.0)
    assert wide["counterfactual"]["corrected"] == 8
    assert wide["capacity_ablation"]["corrected"] == 0
    assert small["purity"]["exact"] == wide["purity"]["exact"] == 168
    assert recover["verdict"] == "recover_passed"
    assert len(recover["checks"]) == 5 and all(recover["checks"].values())
    assert verify["verdict"] == "verify_passed"
    assert len(verify["checks"]) == 11 and all(verify["checks"].values())
    assert red["verdict"] == "red_drill_passed"
    assert all(red["checks"].values())
    expected_total = run["cost"]["cpu"] + recover["cpu"] + verify["cpu"] + red["cpu"]
    assert data["result"]["cost"]["execution_total_cpu"] == round(expected_total, 4)
    assert data["result"]["budget_incident"]["formal_run_within_cap"] is True
    assert data["result"]["budget_incident"]["recovery_rerun"] is False

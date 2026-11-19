"""Contract tests for the 22.5.1 joint gate (fit face) S2 package."""

from __future__ import annotations

import copy
import json

import pytest

from scripts.training import run_s2_joint_gate_20261120 as runner


def contract() -> dict:
    return json.loads(runner.CONTRACT.read_text(encoding="utf-8"))


def row(
    *,
    b: int = 24,
    a: int = 24,
    d: int = 24,
    frozen_b: int = 0,
    parent_a: int = 32,
    parent_d: int = 24,
) -> dict:
    main_rows = [
        {
            "id": f"item-{index}",
            "a": index < a,
            "b": index < b,
            "d": index < d,
            "multi": index < b,
            "supersede": index < b,
            "parent_d": index < parent_d,
            "parent_a": index < min(parent_a, 24),
        }
        for index in range(24)
    ]
    frozen_rows = [
        {
            "id": f"item-{index}",
            "a": index < a,
            "b": index < frozen_b,
            "d": index < d,
            "multi": index < frozen_b,
            "supersede": index < frozen_b,
            "parent_d": index < parent_d,
            "parent_a": index < min(parent_a, 24),
        }
        for index in range(24)
    ]

    def totals(values: list[dict]) -> dict:
        return {
            "n": len(values),
            **{
                f"{key}_correct": sum(1 for value in values if value[key])
                for key in ("a", "b", "d", "multi", "supersede", "parent_d", "parent_a")
            },
        }

    return {
        "seed": 20261110,
        "parent_bit_identical": True,
        "overwrite_probe": {"a_correct": 0, "n": 24},
        "arms": {
            "main": {"rows": main_rows, **totals(main_rows)},
            "frozen_update": {"rows": frozen_rows, **totals(frozen_rows)},
            "permuted_feedback": {"rows": frozen_rows, **totals(frozen_rows)},
            "cuts": {
                "record_organ": {"b_correct": 0},
                "fast": {"b_correct": 24},
                "record_injection": {"b_correct": 24},
            },
        },
    }


def test_contract_frozen_digest_pins_and_authority() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("joint-gate contract not frozen yet")
    data = contract()
    assert data["schema"] == "taiji-u2-22-5-1-joint-gate-v1"
    assert data["status"] == "frozen"
    assert data["training_authority"]
    assert data["criteria_sha256"] == runner.value_digest(data["criteria"])
    assert data["check_exposure"]["reads"] == 0
    assert data["criteria"]["check_reads"] == 0
    assert data["criteria"]["architecture_gate"].startswith("not_run")
    assert data["cap_cpu"] == runner.CAP_CPU
    assert data["plan"]["price_cpu"] <= data["cap_cpu"]
    assert data["criteria"]["parent"]["preflight_mismatch"] is False
    for name in runner.PINNED_FILES:
        assert data["source_pins"][name] == runner.sha_of(runner.ROOT / name), name


def test_guards_refuse_missing_authority_and_tampering() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("joint-gate contract not frozen yet")
    data = contract()
    draft = copy.deepcopy(data)
    draft["training_authority"] = None
    with pytest.raises(ValueError, match="missing_training_authority"):
        runner.guards(draft, "run")
    runner.guards(data, "smoke")
    draft = copy.deepcopy(data)
    draft["criteria"]["thresholds"]["b_success_min"] = 0.5
    with pytest.raises(ValueError, match="criteria_digest_mismatch"):
        runner.guards(draft, "smoke")
    draft = copy.deepcopy(data)
    draft["source_pins"]["scripts/training/run_s2_joint_gate_20261120.py"] = "0" * 64
    with pytest.raises(ValueError, match="source_pin_mismatch"):
        runner.guards(draft, "smoke")


def test_preregistered_exit_order() -> None:
    criteria = contract()["criteria"] if runner.CONTRACT.is_file() else runner.criteria()
    good = [row()]
    verdict, summary = runner.classify(good, criteria)
    assert verdict == "joint_gate_fit_face_pass", summary
    breach = [row(a=12)]
    assert runner.classify(breach, criteria)[0] == "retention_breach"
    d_breach = [row(d=10, parent_d=24)]
    assert runner.classify(d_breach, criteria)[0] == "d_backoff_breach"
    not_failable = [row()]
    not_failable[0]["overwrite_probe"] = {"a_correct": 24, "n": 24}
    assert runner.classify(not_failable, criteria)[0] == "retention_gate_not_failable"
    causal = [row()]
    causal[0]["arms"]["cuts"]["record_organ"]["b_correct"] = 5
    assert runner.classify(causal, criteria)[0] == "causal_control_failure"
    no_correction = [row(b=10, frozen_b=0)]
    assert runner.classify(no_correction, criteria)[0] == "correction_absent"
    no_gain = [row(b=24, frozen_b=24)]
    assert runner.classify(no_gain, criteria)[0] == "learning_gain_absent"


def test_bootstrap_bounds_are_sane() -> None:
    assert runner.bootstrap_lower([1.0] * 24) == pytest.approx(1.0)
    assert runner.bootstrap_upper([0.0] * 24) == pytest.approx(0.0)
    mixed = [1.0] * 12 + [0.0] * 12
    assert 0.0 <= runner.bootstrap_lower(mixed) < runner.bootstrap_upper(mixed) <= 1.0


def test_smoke_receipt_is_zero_step() -> None:
    if not runner.SMOKE.is_file():
        pytest.skip("joint-gate smoke receipt absent")
    smoke = json.loads(runner.SMOKE.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["discipline"]["optimizer_steps"] == 0
    assert smoke["checks"]["parent_fit_recorded"] is True
    assert smoke["checks"]["birth_decode_matches_parent"] is True

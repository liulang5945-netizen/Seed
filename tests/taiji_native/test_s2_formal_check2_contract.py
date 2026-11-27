"""Contract tests for the 22.5.1 formal check② gate (Proposal B, D = parent_a_truth)."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from scripts.training import run_s2_formal_check2_20261120 as runner
from scripts.training.first_a_support_22_5_v2 import generate_examples as generate_v2
from taiji.u2_22_5_pipeline import DEFAULT_TEMPLATES, build_b_items


def contract() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(runner.CONTRACT.read_text(encoding="utf-8")))


def make_seed_row(
    *,
    a_child: int = 192,
    a_parent: int = 192,
    b_correct: int = 144,
    frozen_b_correct: int = 0,
    d_child_truth: int = 144,
    d_parent_a_truth: int = 144,
    cut_b: int = 0,
    probe_drop: float = 0.75,
    bit_id: bool = True,
) -> dict:
    n_a = 192
    n_b = 144
    return {
        "seed": 20261110,
        "a_n": n_a,
        "a_parent_correct": a_parent,
        "a_child_correct": a_child,
        "parent_a_192_hits": [i < a_parent for i in range(n_a)],
        "child_a_192_hits": [i < a_child for i in range(n_a)],
        "b_n": n_b,
        "b_correct": b_correct,
        "frozen_b_correct": frozen_b_correct,
        "b_hits": [i < b_correct for i in range(n_b)],
        "frozen_b_hits": [i < frozen_b_correct for i in range(n_b)],
        "parent_a_b144_hits": [i < d_parent_a_truth for i in range(n_b)],
        "d_parent_a_truth_correct": d_parent_a_truth,
        "d_child_truth_correct": d_child_truth,
        "child_d_truth_hits": [i < d_child_truth for i in range(n_b)],
        "d_parent_same_correct": 60,
        "d_child_same_correct": d_child_truth,
        "parent_d_same_hits": [i < 60 for i in range(n_b)],
        "child_d_same_hits": [i < d_child_truth for i in range(n_b)],
        "d_child_raw_correct": d_child_truth,
        "child_d_raw_hits": [i < d_child_truth for i in range(n_b)],
        "raw_mod_collisions": 12,
        "cut_record_organ_b_correct": cut_b,
        "overwrite_probe": {
            "n": 24,
            "a_before": 24,
            "a_after": 24 - int(round(probe_drop * 24)),
            "drop": probe_drop,
        },
        "parent_bit_identical": bit_id,
        "optional_reports": {"multi_record_b": "not_run", "superseding_b": "not_run"},
    }


def test_distinct_entity_pairing_never_collides_on_fit() -> None:
    fit = generate_v2("fit", 32)
    items = build_b_items(fit, templates=DEFAULT_TEMPLATES[:2])
    for idx, item in enumerate(items):
        _, rec_b, rec_d, rec_d2, _, _, _ = runner._records_distinct(
            items, idx, DEFAULT_TEMPLATES[:2]
        )
        assert rec_d.raw != item["records"][idx % 2]
        assert rec_d2.raw != rec_b.raw
        assert rec_d2.raw != rec_d.raw


def test_formal_check2_exit_order_and_strict_d() -> None:
    crit = contract()["criteria"] if runner.CONTRACT.is_file() else runner.criteria()
    assert crit["d_reference"] == "parent_a_truth"
    assert runner.classify([make_seed_row()], crit)[0] == "joint_gate_check2_pass"
    assert runner.classify([make_seed_row(bit_id=False)], crit)[0] == "engineering_failure"
    assert (
        runner.classify([make_seed_row(probe_drop=0.0)], crit)[0] == "retention_gate_not_failable"
    )
    assert runner.classify([make_seed_row(cut_b=5)], crit)[0] == "causal_control_failure"
    assert (
        runner.classify([make_seed_row(a_child=150, a_parent=192)], crit)[0] == "retention_breach"
    )
    assert (
        runner.classify([make_seed_row(d_child_truth=120, d_parent_a_truth=144)], crit)[0]
        == "d_backoff_breach"
    )
    assert (
        runner.classify([make_seed_row(b_correct=144, frozen_b_correct=144)], crit)[0]
        == "learning_gain_absent"
    )
    assert (
        runner.classify([make_seed_row(b_correct=100, frozen_b_correct=0)], crit)[0]
        == "correction_absent"
    )


def test_contract_frozen_pins_and_exposure_guards() -> None:
    if not runner.CONTRACT.is_file():
        pytest.skip("formal check2 contract not frozen yet")
    data = contract()
    assert data["schema"] == "taiji-u2-22-5-1-formal-check2-v1"
    assert data["status"] == "frozen"
    assert data["training_authority"]
    assert data["check_exposure"]["account"] == "formal_2251"
    assert data["check_exposure"]["allocated"] == 1
    assert data["criteria_sha256"] == runner.value_digest(data["criteria"])
    for name in runner.PINNED_FILES:
        assert data["source_pins"][name] == runner.sha_of(runner.ROOT / name), name
    for rel, expected in data["artifact_pins"].items():
        assert runner.bytes_sha_of(runner.ROOT / rel) == expected, rel

    no_auth = copy.deepcopy(data)
    no_auth["training_authority"] = None
    no_auth["check_exposure"]["used"] = 0
    no_auth["result"] = {"run": False, "closed": False, "verdict": None}
    with pytest.raises(ValueError, match="missing_training_authority"):
        runner.guards(no_auth, "run")

    bad_crit = copy.deepcopy(data)
    bad_crit["criteria"]["thresholds"]["b_success_min"] = 0.5
    with pytest.raises(ValueError, match="criteria_digest_mismatch"):
        runner.guards(bad_crit, "smoke")

    bad_pin = copy.deepcopy(data)
    bad_pin["source_pins"]["scripts/training/run_s2_formal_check2_20261120.py"] = "0" * 64
    with pytest.raises(ValueError, match="source_pin_mismatch"):
        runner.guards(bad_pin, "smoke")

    exhausted = copy.deepcopy(data)
    exhausted["check_exposure"]["used"] = 1
    with pytest.raises(ValueError, match="check_exposure_exhausted"):
        runner.guards(exhausted, "run")


def test_smoke_receipt_is_zero_step_and_zero_check() -> None:
    if not runner.SMOKE.is_file():
        pytest.skip("formal check2 smoke receipt absent")
    smoke = json.loads(runner.SMOKE.read_text(encoding="utf-8"))
    assert smoke["verdict"] == "smoke_passed"
    assert smoke["discipline"]["optimizer_steps"] == 0
    assert smoke["discipline"]["check_reads"] == 0
    assert smoke["discipline"]["final_reads"] == 0
    assert smoke["checks"]["saved_children_strict_and_bit_identical"] is True


def test_verify_amendment_receipt_and_non_shift_invariant_probe() -> None:
    if not runner.VERIFY.is_file():
        pytest.skip("formal check2 verify receipt absent")
    verify = json.loads(runner.VERIFY.read_text(encoding="utf-8"))
    if "amended_verdict" not in verify:
        pytest.skip("formal check2 verify amendment not executed yet")
    assert verify["original_verdict"] == "retention_gate_not_failable"
    assert verify["amended_verdict"] == "joint_gate_check2_pass"
    assert verify["checks"]["no_retraining"] is True
    assert verify["checks"]["no_additional_check_read"] is True
    for entry in verify["recomputed"]["per_seed"]:
        assert entry["retention_probe_drop"] >= 0.02
        assert entry["cut_excess_over_frozen"] == 0

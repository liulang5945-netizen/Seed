from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.training import run_taiji_n1_plan151_train_score_ledger_recovery as recovery


def _gate_summary() -> dict:
    return {
        "items": 192,
        "free_exact": 190,
        "free_by_kind": {"lookup": 48, "revision": 47, "transition": 48, "unknown": 47},
        "pair_exact": 96,
        "rename_free_exact": 187,
        "rename_free_by_kind": {"lookup": 48, "revision": 48, "transition": 48, "unknown": 43},
        "rename_pair_exact": 96,
        "teacher_exact": 190,
        "teacher_token_accuracy": 0.9984,
        "mean_episode_ce": 0.01,
        "factorial": {
            "original": {"items": 192, "joint_four_exact": 48},
            "rename": {"items": 192, "joint_four_exact": 48},
        },
    }


def _rows() -> list[dict]:
    rows = []
    for seed, branch in recovery.ENDPOINTS:
        rows.extend(
            {"seed": seed, "branch": branch, "event": "original_train_gate_row"}
            for _ in range(recovery.ORIGINAL_GATE_QUERIES_PER_ENDPOINT)
        )
        rows.extend(
            {"seed": seed, "branch": branch, "event": "length_score_row"}
            for _ in range(recovery.LENGTH_QUERIES_PER_ENDPOINT)
        )
    return rows


def test_frozen_repeated_scoring_denominators() -> None:
    assert len(recovery.ENDPOINTS) == 6
    assert recovery.ORIGINAL_GATE_QUERIES_PER_ENDPOINT == 960
    assert recovery.LENGTH_QUERIES_PER_ENDPOINT == 288
    assert recovery.TOTAL_QUERIES_PER_ENDPOINT == 1_248
    assert recovery.EXPECTED_TOTAL_QUERIES == 7_488
    assert recovery.CPU_LIMIT_SECONDS == 3_600
    assert (
        recovery.REPAIR_PLAN_SHA256
        == "4892cc62f197689e0437a104bd2e592696ed67863a7746d4a7cb36a3ca0bc965"
    )


def test_n150_preflight_binding_is_not_mistaken_for_its_result_report() -> None:
    n150_readiness = {"stage": "readiness"}
    n150_preflight = {"stage": "preflight"}
    source_binding = {"stage": "binding"}
    train_context = {"stage": "train_context"}
    schedule = {"stage": "schedule"}
    result_report = {"models": [{"seed": 11}]}

    unpacked = recovery.unpack_n150_preflight(
        (n150_readiness, n150_preflight, source_binding, train_context, schedule),
        result_report,
    )

    assert unpacked == (
        n150_readiness,
        n150_preflight,
        source_binding,
        result_report,
        train_context,
        schedule,
    )


def test_gate_parity_covers_integer_and_factorial_summaries() -> None:
    frozen = _gate_summary()
    recovered = {**frozen, "mean_episode_ce": 0.02, "teacher_token_accuracy": 0.997}
    recovery.validate_gate_parity(recovered, frozen)

    recovered["rename_free_by_kind"] = {**frozen["rename_free_by_kind"], "unknown": 42}
    with pytest.raises(recovery.Refuse, match="integer summaries differ"):
        recovery.validate_gate_parity(recovered, frozen)


def test_complete_itemwise_rows_require_each_endpoint_and_event() -> None:
    rows = _rows()
    coverage = recovery.validate_row_coverage(rows)
    assert coverage == {
        "endpoint_count": 6,
        "rows_per_endpoint": 1_248,
        "original_gate_rows_per_endpoint": 960,
        "length_rows_per_endpoint": 288,
        "total_rows": 7_488,
        "complete": True,
    }

    with pytest.raises(recovery.Refuse, match="denominator mismatch"):
        recovery.validate_row_coverage(rows[:-1])


def test_cli_invocation_occurs_after_all_restore_helpers() -> None:
    module = ast.parse(Path(recovery.__file__).read_text(encoding="utf-8"))
    guard = module.body[-1]
    assert isinstance(guard, ast.If)
    assert "__name__" in ast.unparse(guard.test)

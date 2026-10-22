from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from scripts.training import analyze_taiji_n1_plan162b_train_only_itemwise_attribution as analyzer


def _synthetic_gate_rows() -> list[dict]:
    rows = []
    for view in ("original", "rename"):
        for kind in analyzer.KINDS:
            for index in range(48):
                rows.append(
                    {
                        "evaluation_set": view,
                        "kind": kind,
                        "id": f"{kind}-{index:03d}",
                        "condition": None,
                        "case_id": None,
                        "free_exact": True,
                        "raw_hex": "61",
                        "stop_reason": "boundary",
                    }
                )
    for view in ("original", "rename"):
        for index in range(48):
            for condition in ("original", "direction_opposite", "turn_flip", "both"):
                rows.append(
                    {
                        "evaluation_set": f"factorial_{view}",
                        "kind": "transition",
                        "id": f"transition-{index:03d}",
                        "condition": condition,
                        "case_id": condition,
                        "free_exact": True,
                        "raw_hex": "6e6f727468",
                        "stop_reason": "boundary",
                    }
                )
    for view in ("pair", "rename_pair"):
        for index in range(96):
            rows.append(
                {
                    "evaluation_set": view,
                    "kind": "lookup",
                    "id": f"lookup-{index // 2:03d}",
                    "condition": None,
                    "case_id": f"pair-{index:03d}",
                    "free_exact": True,
                    "raw_hex": "6d6174657269616c",
                    "stop_reason": "boundary",
                }
            )
    return rows


def test_plan_digest_matches_source_constant() -> None:
    assert hashlib.sha256(analyzer.PLAN.read_bytes()).hexdigest() == analyzer.PLAN_SHA256


def test_gate_summary_recomputes_frozen_n132_requirements() -> None:
    summary = analyzer._summarize_gate_endpoint(_synthetic_gate_rows())
    assert summary["free_exact"] == 192
    assert summary["rename_free_exact"] == 192
    assert summary["pair_exact"] == 96
    assert summary["rename_pair_exact"] == 96
    assert summary["factorial"]["original"]["joint_four_exact"] == 48
    assert summary["factorial"]["rename"]["joint_four_exact"] == 48
    assert analyzer._sufficient(summary) is True


def test_gate_sufficiency_requires_each_original_frozen_condition() -> None:
    summary = analyzer._summarize_gate_endpoint(_synthetic_gate_rows())
    summary["rename_free_by_kind"]["transition"] = 39
    assert analyzer._sufficient(summary) is False
    summary["rename_free_by_kind"]["transition"] = 48
    summary["factorial"]["original"]["joint_four_exact"] = 43
    assert analyzer._sufficient(summary) is False


def test_endpoint_chunk_binding_recovers_overwritten_gate_arm() -> None:
    rows = []
    for seed, _arm in analyzer.ENDPOINTS:
        rows.extend(
            {"seed": seed, "arm": "position", "id": f"row-{i}"}
            for i in range(analyzer.GATE_ROWS_PER_ENDPOINT)
        )
    bound, overwritten = analyzer.bind_endpoint_chunks(
        rows,
        rows_per_endpoint=analyzer.GATE_ROWS_PER_ENDPOINT,
        event="original_train_gate_row",
    )
    assert len(bound) == 5_760
    assert overwritten == 5_760
    assert (
        Counter((row["seed"], row["experiment_arm"]) for row in bound)[(11, "surface_control")]
        == 960
    )
    assert (
        Counter((row["seed"], row["experiment_arm"]) for row in bound)[(13, "surface_augmented")]
        == 960
    )


def test_endpoint_chunk_binding_rejects_wrong_seed_order() -> None:
    rows = [{"seed": 12, "arm": "position"}] * (
        analyzer.GATE_ROWS_PER_ENDPOINT * len(analyzer.ENDPOINTS)
    )
    with pytest.raises(analyzer.Refuse, match="order/seed mismatch"):
        analyzer.bind_endpoint_chunks(
            rows, rows_per_endpoint=analyzer.GATE_ROWS_PER_ENDPOINT, event="original_train_gate_row"
        )


def test_paired_gate_summary_counts_regressions_and_output_changes() -> None:
    rows = [
        {
            "seed": 11,
            "experiment_arm": "surface_control",
            "evaluation_set": "original",
            "kind": "transition",
            "id": "train-002",
            "condition": None,
            "case_id": None,
            "free_exact": True,
            "raw_hex": "6e6f727468",
            "stop_reason": "boundary",
        },
        {
            "seed": 11,
            "experiment_arm": "surface_augmented",
            "evaluation_set": "original",
            "kind": "transition",
            "id": "train-002",
            "condition": None,
            "case_id": None,
            "free_exact": False,
            "raw_hex": "736f757468",
            "stop_reason": "boundary",
        },
    ]
    summary, errors = analyzer._paired_gate_rows(rows)
    assert summary[0]["control_only_correct"] == 1
    assert summary[0]["raw_output_changed"] == 1
    assert errors[0]["parent_id"] == "train-002"
    assert errors[0]["augmented_free_exact"] is False


def test_length_summary_pairs_by_endpoint_item_and_length() -> None:
    rows = []
    for arm, exact in (("surface_control", True), ("surface_augmented", False)):
        rows.append(
            {
                "seed": 11,
                "experiment_arm": arm,
                "context_length": 8,
                "kind": "unknown",
                "id": "train-003",
                "split": "train",
                "free_exact": exact,
            }
        )
    summary = analyzer._paired_length_rows(rows)
    assert summary == [
        {
            "seed": 11,
            "kind": "unknown",
            "context_length": 8,
            "items": 1,
            "control_exact": 1,
            "augmented_exact": 0,
            "control_only_correct": 1,
            "both_wrong": 0,
            "augmented_only_correct": 0,
        }
    ]


def test_all_outputs_are_repository_local() -> None:
    for path in (
        analyzer.PLAN,
        analyzer.N162B_CLOSEOUT,
        analyzer.N162B_RESULT,
        analyzer.PROGRESS,
        analyzer.SCHEDULE,
        analyzer.REPORT,
    ):
        assert Path(path).resolve().is_relative_to(analyzer.ROOT)

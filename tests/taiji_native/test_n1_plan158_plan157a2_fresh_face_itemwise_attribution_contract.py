from __future__ import annotations

import hashlib

import pytest

from scripts.training import (
    analyze_taiji_n1_plan158_plan157a2_fresh_face_itemwise_attribution as analyzer,
)


def test_plan_sha_and_frozen_diagnostic_denominators() -> None:
    assert hashlib.sha256(analyzer.PLAN.read_bytes()).hexdigest() == analyzer.PLAN_SHA256
    assert analyzer.EXPECTED_QUERIES == 2_880
    assert analyzer.PRIOR_ROWS + analyzer.NEW_ROWS == analyzer.EXPECTED_ROWS
    assert analyzer.CPU_LIMIT_SECONDS == 30.0


def test_paired_summary_counts_directional_candidate_parent_changes() -> None:
    rows = [
        {
            "seed": 11,
            "parent_id": "u0",
            "context_length": 8,
            "kind": "unknown",
            "branch": "unknown8_rebalanced",
            "free_exact": True,
        },
        {
            "seed": 11,
            "parent_id": "u0",
            "context_length": 8,
            "kind": "unknown",
            "branch": "length_mixed_parent",
            "free_exact": False,
        },
        {
            "seed": 11,
            "parent_id": "u1",
            "context_length": 8,
            "kind": "unknown",
            "branch": "unknown8_rebalanced",
            "free_exact": False,
        },
        {
            "seed": 11,
            "parent_id": "u1",
            "context_length": 8,
            "kind": "unknown",
            "branch": "length_mixed_parent",
            "free_exact": True,
        },
    ]
    assert analyzer.paired_summary(rows) == [
        {
            "seed": 11,
            "context_length": 8,
            "kind": "unknown",
            "pairs": 2,
            "both_correct": 0,
            "candidate_only_correct": 1,
            "parent_only_correct": 1,
            "both_wrong": 0,
            "candidate_exact": 1,
            "parent_exact": 1,
        }
    ]


def test_paired_summary_refuses_missing_or_duplicate_endpoint_rows() -> None:
    candidate = {
        "seed": 12,
        "parent_id": "transition-1",
        "context_length": 16,
        "kind": "transition",
        "branch": "unknown8_rebalanced",
        "free_exact": True,
    }
    with pytest.raises(analyzer.Refuse, match="incomplete"):
        analyzer.paired_summary([candidate])
    with pytest.raises(analyzer.Refuse, match="duplicate"):
        analyzer.paired_summary(
            [candidate, candidate, {**candidate, "branch": "length_mixed_parent"}]
        )


def test_nested_transition_summary_is_per_seed_and_arm() -> None:
    rows = []
    scores = {
        "unknown8_rebalanced": {8: False, 16: True, 32: True},
        "length_mixed_parent": {8: True, 16: False, 32: False},
    }
    for branch, by_length in scores.items():
        for length, exact in by_length.items():
            rows.append(
                {
                    "seed": 13,
                    "parent_id": "unknown-0",
                    "context_length": length,
                    "kind": "unknown",
                    "branch": branch,
                    "free_exact": exact,
                }
            )
    summary = analyzer.length_transition_summary(rows)
    candidate = next(
        row for row in summary
        if row["branch"] == "unknown8_rebalanced" and row["from_length"] == 8 and row["to_length"] == 16
    )
    parent = next(
        row for row in summary
        if row["branch"] == "length_mixed_parent" and row["from_length"] == 8 and row["to_length"] == 16
    )
    assert candidate["wrong_to_right"] == 1
    assert parent["right_to_wrong"] == 1


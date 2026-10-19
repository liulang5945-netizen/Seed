from __future__ import annotations

import hashlib

import pytest

from scripts.training import (
    analyze_taiji_n1_plan156_plan155a_train_score_itemwise_attribution as analyzer,
)


def test_plan_digest_matches_analyzer_binding() -> None:
    assert hashlib.sha256(analyzer.PLAN.read_bytes()).hexdigest() == analyzer.PLAN_SHA256


def test_paired_summary_separates_directional_error_changes() -> None:
    rows = [
        {
            "seed": 12,
            "evaluation_set": "original",
            "id": "train-1",
            "kind": "transition",
            "condition": None,
            "case_id": None,
            "branch": "unknown8_rebalanced",
            "free_exact": True,
        },
        {
            "seed": 12,
            "evaluation_set": "original",
            "id": "train-1",
            "kind": "transition",
            "condition": None,
            "case_id": None,
            "branch": "uniform_length_replay",
            "free_exact": False,
        },
        {
            "seed": 12,
            "evaluation_set": "original",
            "id": "train-2",
            "kind": "transition",
            "condition": None,
            "case_id": None,
            "branch": "unknown8_rebalanced",
            "free_exact": False,
        },
        {
            "seed": 12,
            "evaluation_set": "original",
            "id": "train-2",
            "kind": "transition",
            "condition": None,
            "case_id": None,
            "branch": "uniform_length_replay",
            "free_exact": True,
        },
    ]
    result = analyzer.paired_summary(
        rows,
        key_fields=analyzer.GATE_KEY_FIELDS,
        group_fields=analyzer.GATE_GROUP_FIELDS,
    )
    assert result == [
        {
            "seed": 12,
            "evaluation_set": "original",
            "kind": "transition",
            "condition": None,
            "items": 2,
            "both_exact": 0,
            "both_wrong": 0,
            "candidate_exact": 1,
            "candidate_only_correct": 1,
            "control_exact": 1,
            "control_only_correct": 1,
        }
    ]


def test_paired_summary_refuses_duplicate_or_unmatched_items() -> None:
    candidate = {
        "seed": 11,
        "evaluation_set": "original",
        "id": "train-0",
        "kind": "lookup",
        "branch": "unknown8_rebalanced",
        "free_exact": True,
    }
    with pytest.raises(analyzer.Refuse, match="duplicate paired"):
        analyzer.paired_summary(
            [candidate, candidate],
            key_fields=analyzer.GATE_KEY_FIELDS,
            group_fields=analyzer.GATE_GROUP_FIELDS,
        )
    with pytest.raises(analyzer.Refuse, match="coverage differs"):
        analyzer.paired_summary(
            [candidate],
            key_fields=analyzer.GATE_KEY_FIELDS,
            group_fields=analyzer.GATE_GROUP_FIELDS,
        )


def test_length_transition_summary_counts_wrong_right_directions() -> None:
    rows = []
    scores = {
        "unknown8_rebalanced": {8: False, 16: True, 32: True},
        "uniform_length_replay": {8: True, 16: False, 32: False},
    }
    for branch, by_length in scores.items():
        for length, exact in by_length.items():
            rows.append(
                {
                    "seed": 13,
                    "kind": "unknown",
                    "id": "train-u1",
                    "branch": branch,
                    "context_length": length,
                    "free_exact": exact,
                }
            )
    summary = analyzer._length_transition_summary(rows)
    candidate_short = next(
        row
        for row in summary
        if row["branch"] == "unknown8_rebalanced" and row["from_length"] == 8 and row["to_length"] == 16
    )
    control_short = next(
        row
        for row in summary
        if row["branch"] == "uniform_length_replay" and row["from_length"] == 8 and row["to_length"] == 16
    )
    assert candidate_short["wrong_to_right"] == 1
    assert control_short["right_to_wrong"] == 1


from __future__ import annotations

import pytest

from scripts.training import audit_taiji_n1_plan141_itemwise_retention_context_attribution as audit


def test_paired_counts_itemwise_correctness_transitions() -> None:
    result = audit._paired(
        {"a": True, "b": True, "c": False, "d": False},
        {"a": True, "b": False, "c": True, "d": False},
    )

    assert result["matched_items"] == 4
    assert result["both_correct"] == 1
    assert result["first_only_correct"] == 1
    assert result["second_only_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["second_minus_first"] == 0
    assert result["first_only_correct_ids"] == ["b"]
    assert result["second_only_correct_ids"] == ["c"]


def test_paired_rejects_different_item_universes() -> None:
    with pytest.raises(ValueError, match="identifiers differ"):
        audit._paired({"train-001": True}, {"train-002": True})


def test_score_map_rejects_duplicate_item_identifiers() -> None:
    with pytest.raises(ValueError, match="duplicate item identifiers"):
        audit._score_map(
            [{"id": "train-001", "free_exact": True}, {"id": "train-001", "free_exact": False}]
        )

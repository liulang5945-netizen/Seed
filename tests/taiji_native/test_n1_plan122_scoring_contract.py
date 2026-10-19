from __future__ import annotations

import pytest

from scripts.training import eval_taiji_n1_v12_after_n121 as evaluator


def test_one_shot_query_modes_require_owner_approval() -> None:
    with pytest.raises(PermissionError, match="owner approval"):
        evaluator.run_range_check(owner_approved=False)
    with pytest.raises(PermissionError, match="owner approval"):
        evaluator.run_confirmation(owner_approved=False)


def test_confirmation_requires_a_complete_passing_range_report() -> None:
    with pytest.raises(ValueError, match="range gate must pass"):
        evaluator._require_passing_range(
            {
                "complete": True,
                "status": "range_not_met",
                "range_gate": {"range_check_passed": False},
            }
        )


def test_confirmation_requires_all_180_sacrifice_rows_before_release() -> None:
    with pytest.raises(ValueError, match="range gate must pass"):
        evaluator._require_passing_range(
            {
                "complete": True,
                "status": "complete",
                "range_gate": {"range_check_passed": True},
                "model_queries": 179,
                "confirmation_queries": 0,
                "queried_subset": "sacrifice",
                "face_sha256": "face",
                "readiness_sha256": "readiness",
                "source_hashes_unchanged": True,
            }
        )

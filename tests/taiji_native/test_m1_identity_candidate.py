"""Regression coverage for the manually gated identity candidate review."""

from __future__ import annotations

from scripts.training.eval_taiji_m1_identity_candidate import run_review


def test_m1_identity_candidate_review_is_non_default_and_reversible() -> None:
    result = run_review(seeds=(11,))

    assert result["gate"]["passed"] is True
    assert result["candidate"]["default_replacement"] is False
    assert result["diagnostics"]["default_candidate_ready"] is False
    record = result["diagnostics"]["records"][0]
    assert record["schema"]["default_has_identity_payload"] is False
    assert record["schema"]["candidate_restore_exact"] is True
    assert record["checkpoint"]["default_rollback_exact"] is True

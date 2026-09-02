"""Regression test for the native identity organ's decoder boundary."""

from __future__ import annotations

from scripts.training.eval_taiji_m1_identity_shadow import run_shadow


def test_m1_identity_shadow_preserves_byte_boundary() -> None:
    result = run_shadow(seeds=(11,))

    assert result["all_records_pass"] is True
    record = result["records"][0]
    assert abs(record["b1"]["after_b2_mean_surprise_delta"]) <= 1e-12
    assert abs(record["b1"]["after_b5_retention_mean_surprise_delta"]) <= 1e-12
    assert record["provenance"]["final_action_owner"] == "ByteMotor"
    assert result["default_candidate_ready"] is False

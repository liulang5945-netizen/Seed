from __future__ import annotations

from scripts.training.eval_taiji_m1_identity_admission import run_admission


def test_m1_identity_admission_review_passes_one_seed_without_default_promotion() -> None:
    result = run_admission(seeds=(11,))

    assert result["all_records_pass"] is True
    assert result["recommended_gain"] == 32.0
    assert result["default_identity_organ_enabled"] is False
    assert result["default_candidate_ready"] is False
    assert result["records"][0]["route"]["cross_phase_slot_collisions"] == 0
    assert result["records"][0]["boundary"]["below_threshold_split"] is True


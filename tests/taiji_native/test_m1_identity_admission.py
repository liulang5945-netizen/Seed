from __future__ import annotations

from scripts.training.eval_taiji_m1_identity_admission import run_admission
from taiji import TaijiConfig


def test_m1_identity_admission_review_reports_the_live_organ_default() -> None:
    result = run_admission(seeds=(11,))

    assert result["all_records_pass"] is True
    assert result["recommended_gain"] == 32.0
    # M1-63 promoted the organ to the default path.  This review must mirror the
    # live default instead of pinning a stale one, so it stays a diagnostic
    # rather than a second, contradictory source of truth about the default.
    assert result["default_identity_organ_enabled"] is TaijiConfig().identity_organ_enabled
    assert result["default_candidate_ready"] is TaijiConfig().identity_organ_enabled
    assert result["records"][0]["route"]["cross_phase_slot_collisions"] == 0
    assert result["records"][0]["boundary"]["below_threshold_split"] is True


from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4r5_update_scale_pilot_aggregate_20260908.json"
)


def test_m4r5_half_scale_improves_pilot_but_strict_gate_blocks_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["diagnosis"]["half_scale_supported"] is False
    assert report["diagnosis"]["half_scale_improves_mean_gain"] is True
    assert report["diagnosis"]["half_scale_reduces_c_cycle2_degradation"] is True

    half = report["variants"]["scale_0p5"]
    assert half["c3_gain_positive_count"] == 3
    assert half["c_cycle3_degradation_count"] == 0
    assert half["c_cycle2_degradation_count"] == 1

    legacy = report["variants"]["scale_1p0"]
    assert legacy["c3_gain_positive_count"] == 2
    assert legacy["c_cycle2_degradation_count"] == 2

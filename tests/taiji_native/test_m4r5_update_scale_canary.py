from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4r5_update_scale_canary_seed11_20260908.json"
)


def test_m4r5_scale_canary_closes_owner_and_checkpoint_gates() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["configuration"]["fixed_capacity"] is True
    assert report["configuration"]["scales"] == [0.0, 0.5, 1.0]
    assert all(report["checks"].values())
    assert report["diagnosis"]["half_scale_candidate"] is True
    assert report["variants"]["scale_0p0"]["metrics"]["c3_holdout_gain_bpb"] == 0.0
    assert report["variants"]["scale_0p5"]["metrics"]["c3_holdout_gain_bpb"] > 0.0
    assert report["variants"]["scale_0p5"]["metrics"]["c_cycle2_delta_bpb"] <= 0.0
    assert report["variants"]["scale_0p5"]["metrics"]["c_cycle3_delta_bpb"] <= 0.0

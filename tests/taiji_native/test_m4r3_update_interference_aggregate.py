from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4r3_update_interference_aggregate_20260908.json"
)


def test_m4r3_three_seed_smoke_supports_joint_attribution_without_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["diagnosis"]["joint_attribution_supported"] is True

    joint = report["arms"]["joint"]
    assert joint["c3_gain_positive_count"] == 3
    assert joint["retention_gate_passed_all"] is True

    context = report["arms"]["context_only"]
    assert context["c3_gain_positive_count"] == 3
    assert context["retention_gate_passed_all"] is False

    readout = report["arms"]["readout_only"]
    assert readout["retention_gate_passed_all"] is True
    assert readout["c3_gain_positive_count"] == 2

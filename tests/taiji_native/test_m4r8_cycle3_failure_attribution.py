from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4r8_cycle3_failure_attribution_20260908.json"
)


def test_m4r8_withdraws_candidate_after_repeated_cycle3_failure() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["formal_candidate"] is False
    assert report["diagnosis"]["half_scale_c_cycle3_failure_count"] == 2
    assert report["diagnosis"]["half_scale_c_cycle3_failure_seeds"] == [11, 47]
    assert report["diagnosis"]["failure_isolated"] is False
    assert report["decision_boundary"]["training_allowed"] is False

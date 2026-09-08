from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4r3_update_interference_pilot_aggregate_20260908.json"
)


def test_m4r3_pilot_aggregate_blocks_formalization() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["diagnosis"]["joint_attribution_supported"] is False
    assert all(report["checks"].values())

    for arm in ("readout_only", "context_only", "joint"):
        assert report["arms"][arm]["retention_gate_passed_all"] is False

from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2] / "reports" / "taiji_m4r7_formal_preflight_20260908.json"
)


def test_m4r7_preflight_allows_bounded_formal_without_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["formal_allowed"] is True
    assert report["configuration"]["profile"] == "foundation"
    assert report["configuration"]["train_bytes"] == 65_536
    assert report["configuration"]["eval_bytes"] == 16_384
    assert report["checks"]["checkpoint_save_and_fresh_restore"] is True
    assert report["checks"]["record_disjoint_chain"] is True
    assert report["checks"]["foundation_budget_available"] is True
    assert report["checks"]["projected_cpu_within_bound"] is True

from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4r9_foundation_course_audit_20260908.json"
)


def test_m4r9_audits_course_before_any_new_training() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["training_allowed"] is False
    assert report["checks"]["record_disjoint_chain"] is True
    assert report["checks"]["distinct_phase_digests"] is True
    assert report["diagnosis"]["c3_degraded_seed_count"] == 2

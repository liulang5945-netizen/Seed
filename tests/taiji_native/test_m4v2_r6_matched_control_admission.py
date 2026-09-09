from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_r6_matched_control_admission_20260910.json"
)


def test_m4v2_r6_matched_control_admission_is_ready_but_not_promoted() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["report_format"] == "taiji-m4v2-r6-matched-control-admission-v1"
    assert report["admission"] == "ready_to_start_formal"
    assert report["blocking_failures"] == []
    assert report["can_start_r6_formal"] is True
    assert report["can_promote"] is False
    assert report["default_runtime_attached"] is False
    assert report["provider_attached"] is False
    assert report["mcp_attached"] is False
    assert report["client_attached"] is False
    assert report["cuda_used"] is False
    assert report["training_performed"] is False
    assert all(value is True for value in report["checks"].values())

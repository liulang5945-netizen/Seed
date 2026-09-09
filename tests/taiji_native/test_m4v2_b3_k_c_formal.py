from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_c_formal_20260910.json"
)


def test_c_entry_formal_closes_resource_and_candidate_quality_gates() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "c-entry-formal-execution"
    assert report["cell_count"] == 9
    assert report["training_performed"] is True
    assert report["sealed_test_scored"] is True
    assert report["technical_gate_passed"] is True
    assert report["resource_gate_passed"] is True
    assert report["candidate_quality_gate_passed"] is True
    assert report["formal_gate_passed"] is True
    assert report["candidate_beats_fixed_large_all"] is False
    assert report["candidate_beats_fixed_large_count"] == 0
    assert report["can_promote"] is False
    assert all(
        cell["resource"]["candidate"]["training"]["measurement_complete"]
        and cell["resource"]["fixed_large"]["training"]["measurement_complete"]
        and cell["resource"]["candidate"]["inference"]["measurement_complete"]
        and cell["resource"]["fixed_large"]["inference"]["measurement_complete"]
        for cell in report["cells"]
    )

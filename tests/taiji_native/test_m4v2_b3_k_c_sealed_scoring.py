from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_c_sealed_scoring_20260910.json"
)


def test_c_entry_sealed_scoring_canary_is_read_only_and_not_formal() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "c-entry-sealed-scoring-canary"
    assert report["cell_count"] == 9
    assert report["sealed_test_scored"] is True
    assert report["training_performed"] is False
    assert report["resource_measurement_complete"] is False
    assert report["formal_gate_passed"] is False
    assert report["can_start_formal"] is False
    assert report["can_promote"] is False
    assert all(
        cell["candidate_artifacts_read_only"]
        and cell["sealed_test_scored"]
        for cell in report["cells"]
    )

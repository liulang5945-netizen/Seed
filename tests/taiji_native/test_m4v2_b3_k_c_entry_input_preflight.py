from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_c_entry_input_preflight_20260910.json"
)


def test_c_entry_preflight_accepts_sealed_test_and_rejects_old_fixed_large_course() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "blocked"
    assert report["sealed_test"]["ready"] is True
    assert report["fixed_large_ready"] is False
    assert report["formal_input_ready"] is False
    assert report["can_start_formal"] is False
    assert report["can_promote"] is False
    assert report["required_course_episode_indexes"] == [0, 1, 2, 3, 4]
    assert all(
        cell["ready"] is False
        and cell["source_manifest_format_matches"] is False
        and cell["course_contract_digest_matches"] is False
        for cell in report["fixed_large_cells"]
    )

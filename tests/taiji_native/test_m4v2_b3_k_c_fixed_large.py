from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_c_fixed_large_build_20260910.json"
)


def test_c_entry_fixed_large_matrix_is_built_from_the_same_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["cell_count"] == 9
    assert report["sealed_test_scored"] is False
    assert report["can_start_formal"] is False
    assert report["can_promote"] is False
    assert all(
        cell["status"] == "passed"
        and cell["replica_updates_distinct"] is True
        and cell["ensemble_fresh_restore"] is True
        and cell["disk_ensemble_fresh_restore"] is True
        and cell["resource"]["measurement_complete"] is True
        and cell["resource"]["parameter_bytes"] > 0
        and len(cell["resource"]["checkpoint_write_paths"])
        == len(set(cell["resource"]["checkpoint_write_paths"]))
        and all(
            replica["prefit_checkpoint_gate"]["k1.semantic"]
            and replica["prefit_checkpoint_gate"]["k2.transition"]
            and replica["postfit_restore_gate"]["k1.semantic"]
            and replica["postfit_restore_gate"]["k2.transition"]
            for replica in cell["replica_reports"]
        )
        for cell in report["cells"]
    )

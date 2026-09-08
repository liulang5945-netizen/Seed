from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_m4r2_capacity_diagnosis_canary_closes_owner_and_rollback_gates() -> None:
    report_path = (
        PROJECT_ROOT / "reports" / "taiji_m4r2_capacity_diagnosis_canary_seed11_20260908.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert report["technical_gate_all_passed"] is True
    assert report["checks"]["zero_init_uniform"] is True
    assert report["checks"]["b_old_slot_unchanged"] is True
    assert report["checks"]["b_new_slot_changes"] is True
    assert report["checks"]["slot_detach"] is True
    assert report["checks"]["slot_rollback"] is True

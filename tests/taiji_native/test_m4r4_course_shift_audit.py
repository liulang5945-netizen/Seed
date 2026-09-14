from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2] / "reports" / "taiji_m4r4_course_shift_audit_20260908.json"
)


def test_m4r4_audit_closes_read_only_course_and_artifact_gates() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["configuration"]["training_performed"] is False
    assert set(report["configuration"]["arms"]) == {
        "readout_only",
        "context_only",
        "joint",
    }
    assert len(report["reports"]) == 3
    assert all(report["checks"].values())

    for seed_report in report["reports"]:
        assert len(seed_report["phases"]) == 3
        for phase in seed_report["phases"]:
            assert phase["frozen_holdout"]["read_only"] is True
        assert (
            seed_report["owner_updates"]["readout_only"]["active_predictive_readout"][
                "changed_scalars"
            ]
            == 12593
        )

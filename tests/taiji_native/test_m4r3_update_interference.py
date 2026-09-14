from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4r3_update_interference_canary_seed11_20260908.json"
)


def test_m4r3_seed11_smoke_contract() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["configuration"]["fixed_capacity"] is True
    assert report["configuration"]["gated_temporal_candidate"] is False
    assert set(report["variants"]) == {"readout_only", "context_only", "joint"}
    assert all(report["checks"].values())
    assert (
        report["variants"]["readout_only"]["active_readout_before"]
        != report["variants"]["readout_only"]["active_readout_after"]
    )
    assert (
        report["variants"]["context_only"]["owner_before"]["predictive_context"]
        != report["variants"]["context_only"]["owner_after"]["predictive_context"]
    )
    assert (
        report["variants"]["joint"]["owner_before"]["predictive_readout"]
        != report["variants"]["joint"]["owner_after"]["predictive_readout"]
    )

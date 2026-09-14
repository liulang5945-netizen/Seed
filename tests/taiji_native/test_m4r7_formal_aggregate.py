from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2] / "reports" / "taiji_m4r7_formal_aggregate_20260908.json"
)


def test_m4r7_formal_aggregate_requires_review_before_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["configuration"]["train_bytes"] == 65_536
    assert report["configuration"]["eval_bytes"] == 16_384
    assert report["checks"]["preflight_passed"] is True
    assert report["checks"]["all_scale_zero_controls_frozen"] is True
    assert report["checks"]["all_variants_checkpoint_round_trip"] is True

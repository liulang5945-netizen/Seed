from __future__ import annotations

import json
from pathlib import Path

from scripts.training.audit_taiji_m5_k_scorecard import build_scorecard


def test_m5_k_axis_scorecard_closes_evidence_without_promotion() -> None:
    root = Path(__file__).resolve().parents[2]
    k1 = json.loads(
        (root / "reports" / "taiji_m5_k1_skill_composition_formal_20260909.json").read_text(
            encoding="utf-8"
        )
    )
    k2 = json.loads(
        (root / "reports" / "taiji_m5_k2_multistep_formal_20260909.json").read_text(
            encoding="utf-8"
        )
    )

    scorecard = build_scorecard(k1, k2)

    assert scorecard["verdict"]["k_evidence_closed"] is True
    assert scorecard["verdict"]["promotion_gate"] is False
    assert scorecard["verdict"]["can_promote"] is False
    assert scorecard["evidence_gates"]["parent_retention_missing"] is True
    assert scorecard["scorecard"]["type"] == "scorecard"

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.training.aggregate_taiji_m4r1_consolidation import aggregate_reports

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = (
    PROJECT_ROOT / "reports" / "taiji_m4r1_cascade_formal_seed11_20260908.json",
    PROJECT_ROOT / "reports" / "taiji_m4r1_cascade_formal_seed29_20260908.json",
    PROJECT_ROOT / "reports" / "taiji_m4r1_cascade_formal_seed47_20260908.json",
)


def _load_reports() -> list[dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in REPORTS]


def test_m4r1_formal_aggregate_passes_technical_gate_but_not_promotion() -> None:
    payload = aggregate_reports(_load_reports())

    assert payload["status"] == "passed"
    assert payload["can_promote"] is False
    assert payload["consolidation_strength"] == 0.5
    aggregate = payload["aggregate"]
    assert aggregate["technical_gate_all_passed"] is True
    assert aggregate["retention_gate_passed"] is False
    assert aggregate["c3_holdout_gain_bpb"]["positive_seed_count"] == 3
    assert aggregate["c_cycle2_delta_bpb"]["degradation_seed_count"] == 3
    assert aggregate["c_cycle3_delta_bpb"]["degradation_seed_count"] == 3


def test_m4r1_aggregate_rejects_strength_drift() -> None:
    reports = _load_reports()
    tampered = copy.deepcopy(reports[0])
    tampered["consolidation_strength"] = 0.25
    reports[0] = tampered

    with pytest.raises(ValueError, match="consolidation strength mismatch"):
        aggregate_reports(reports)

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.training.aggregate_taiji_m4r0_cascade3 import aggregate_reports

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = (
    PROJECT_ROOT / "reports" / "taiji_m4r0_cascade3_seed11_20260907.json",
    PROJECT_ROOT / "reports" / "taiji_m4r0_cascade3_seed29_20260908.json",
    PROJECT_ROOT / "reports" / "taiji_m4r0_cascade3_seed47_20260908.json",
)


def _load_reports() -> list[dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in REPORTS]


def test_m4r0_three_cycle_aggregate_is_technical_pass_but_not_promotable() -> None:
    payload = aggregate_reports(_load_reports())

    assert payload["status"] == "passed"
    assert payload["can_promote"] is False
    aggregate = payload["aggregate"]
    assert aggregate["technical_gate_all_passed"] is True
    assert aggregate["c3_holdout_gain_bpb"]["positive_seed_count"] == 1
    assert aggregate["c_cycle3_delta_bpb"]["degradation_seed_count"] == 3


def test_m4r0_aggregate_rejects_record_overlap() -> None:
    reports = _load_reports()
    tampered = copy.deepcopy(reports[0])
    tampered["data_contract"]["phase_chain"]["overlap_counts"]["phase_c__vs__phase_c3"] = 1
    reports[0] = tampered

    with pytest.raises(ValueError, match="record overlap"):
        aggregate_reports(reports)

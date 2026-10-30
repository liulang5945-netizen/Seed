"""R4 preflight artifact: train-only data adapters, finite grads, zero-step restore."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/taiji_native_core_r4_preflight_20261030.json"


def test_r4_preflight_report():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "r4_preflight_passed"
    assert report["episodes"] == 4
    assert report["optimizer_updates"] == 0
    assert report["training_started"] is False
    assert report["query_read"] is False
    assert report["holdout_read"] is False
    assert report["train_only"] is True
    assert set(report["objects"]) == {"new_core_v1", "corrected_old_v1", "ordinary_v1"}
    for name, item in report["objects"].items():
        assert item["finite_loss"] is True, name
        assert item["finite_gradients"] is True, name
        assert item["scored_turns"] == 16, name
        assert item["target_bytes"] == 460, name
    groups = report["objects"]["new_core_v1"]["gradient_groups"]
    assert groups["attention"] is True
    assert groups["reader"] is True
    assert groups["decoder"] is True
    assert groups["content_encoder"] is False
    assert report["restore"]["in_process_equal"] is True
    assert report["restore"]["fresh_process_equal"] is True

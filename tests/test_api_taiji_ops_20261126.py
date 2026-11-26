"""Tests for Taiji N36/N37/N38 ops API and live training dashboard."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from api.routes_taiji_ops import (
    ClassifyChangeRequest,
    taiji_classify_change,
    taiji_n38_training,
    taiji_ops_dashboard,
    taiji_ops_overview,
)


def test_taiji_ops_handlers_and_app_registration() -> None:
    ov_data = taiji_ops_overview()
    assert ov_data["scope"] == "bounded_closed_vocab_and_multimodule_v1"
    assert ov_data["boundaries"]["check_reads"] == 0
    assert ov_data["boundaries"]["final_reads"] == 0
    assert "Archived" in ov_data["fsm_states"]

    tr = taiji_n38_training()
    assert tr["schema"] == "taiji-n38-training-telemetry-v1"

    c1 = taiji_classify_change(ClassifyChangeRequest(kind="ops_refactor"))
    assert c1["tier"] == "tier1_ops_only"

    c2 = taiji_classify_change(ClassifyChangeRequest(kind="ecl_calibration"))
    assert c2["allowed_in_shadow"] is True

    c3 = taiji_classify_change(ClassifyChangeRequest(kind="readout_class"))
    assert c3["reopen_research_required"] is True

    with pytest.raises(HTTPException) as exc_info:
        taiji_classify_change(ClassifyChangeRequest(kind="unknown_x"))
    assert exc_info.value.status_code == 400

    dash = taiji_ops_dashboard()
    assert "N38.1.01" in dash.body.decode("utf-8")

    app_src = Path("api/app.py").read_text(encoding="utf-8")
    assert "routes_taiji_ops" in app_src
    assert '"/taiji-ops"' in app_src
    assert '"/api/taiji/ops/overview"' in app_src

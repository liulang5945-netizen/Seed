"""R5 runner contracts: 15-run manifest, frozen unit costs, restore, approval gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/training/run_native_content_core_r5.py"

_spec = importlib.util.spec_from_file_location("run_native_content_core_r5", SCRIPT)
assert _spec is not None and _spec.loader is not None
R5 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R5
_spec.loader.exec_module(R5)


def _episodes(count: int = 4) -> list[dict]:
    payload = json.loads(
        (ROOT / "plans/manifests/native_core_r1_train_v1.json").read_text(encoding="utf-8")
    )
    return payload["episodes"][:count]


def test_run_specs_declare_q1_nine_and_q2_six():
    specs = R5.run_specs()
    assert len(specs) == 15
    assert len({spec.name for spec in specs}) == 15
    q1 = [spec for spec in specs if spec.block == "q1"]
    q2 = [spec for spec in specs if spec.block == "q2"]
    assert len(q1) == 9
    assert len(q2) == 6
    assert {spec.seed for spec in specs} == {0, 1, 2}
    assert {spec.object for spec in q1} == set(R5.OBJECTS)
    assert {(spec.credit_mode, spec.seed) for spec in q2} == {
        (mode, seed) for mode in ("credit_on", "credit_stop") for seed in (0, 1, 2)
    }


def test_unit_costs_come_from_the_frozen_pricing_report():
    costs = R5.unit_costs()
    report = json.loads(R5.PRICING_REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "r4_priced"
    assert costs == {
        name: record["cpu_seconds_per_update_median"] for name, record in report["objects"].items()
    }


def test_train_run_completes_restores_and_keeps_frozen_groups(tmp_path):
    spec = R5.RunSpecV1(
        name="q2_credit_stop_seed0",
        block="q2",
        object="new_core_v1",
        credit_mode="credit_stop",
        seed=0,
    )
    record = R5.train_run(spec, _episodes(), unit_cpu=5.45, updates=1, checkpoint_dir=tmp_path)
    assert record["steps_executed"] == 1
    assert record["restore_ok"] is True
    assert record["zero_restore"]["in_process_equal"] is True
    assert record["zero_restore"]["fresh_process_equal"] is True
    assert record["endpoint_restore"]["in_process_equal"] is True
    assert record["endpoint_restore"]["fresh_process_equal"] is True
    assert record["loss_all_finite"] is True
    assert record["frozen_parameters_unchanged"] is True
    assert "reader" not in record["last_update_gradient_group_norms"]
    assert "decoder" not in record["last_update_gradient_group_norms"]
    assert record["frozen_groups"] == list(R5.CREDIT_STOP_GROUPS)
    assert (tmp_path / "q2_credit_stop_seed0-zero.pt").exists()
    assert (tmp_path / "q2_credit_stop_seed0-endpoint.pt").exists()


def test_r5_training_requires_explicit_approval():
    with pytest.raises(SystemExit) as excinfo:
        R5._price("20261030", approve=False)
    assert "--approve-r5" in str(excinfo.value)

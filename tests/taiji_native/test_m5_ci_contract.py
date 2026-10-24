"""CI scope must preserve the measured legacy M5 failure and fail closed."""

from __future__ import annotations

import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

REQUIRED_CHECKS = (
    "beats_equal_width_trace_only_by_37_5pp",
    "recurrent_completion_is_causally_necessary",
    "recalls_outcome_and_provenance",
    "recalled_state_feeds_next_fabric_tick",
    "action_outcome_transaction_checkpoint_exact",
    "fixed_topology_no_event_slots",
)


@pytest.fixture(scope="module")
def observed_benchmark():
    # One actual tiny benchmark supplies every positive case; no stored model is loaded.
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/training"))
        verify = importlib.import_module("scripts.training.verify_taiji_m5_episodic_field")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report = verify.run_benchmark()
    finally:
        torch.set_num_threads(previous_threads)
    return verify, report


def test_ci_scope_preserves_the_measured_legacy_failure(observed_benchmark):
    verify, report = observed_benchmark
    original = deepcopy(report)
    scoped = verify.build_ci_contract_report(report)

    assert report == original
    assert report["status"] == "fail"
    assert not report["checks"]["one_shot_action_recall_at_least_87_5pct"]
    assert not report["checks"]["recalls_time_and_episode_codes"]
    assert scoped["scope"] == "current_m5_six_checks"
    assert set(scoped["checks"]) == set(REQUIRED_CHECKS)
    assert scoped["status"] == "pass"
    assert scoped["legacy_benchmark"] == original
    assert scoped["legacy_thresholds"] == {
        "action_accuracy_min": 0.875,
        "mean_time_code_cosine_min": 0.5,
        "episode_identity_accuracy_min": 0.75,
    }
    assert set(scoped["legacy_capability_failures"]) == {
        "one_shot_action_recall_at_least_87_5pct",
        "recalls_time_and_episode_codes",
    }


@pytest.mark.parametrize("name", REQUIRED_CHECKS)
def test_any_real_contract_failure_remains_a_failure(observed_benchmark, name):
    verify, observed = observed_benchmark
    report = deepcopy(observed)
    report["checks"][name] = False
    scoped = verify.build_ci_contract_report(report)
    assert scoped["checks"][name] is False
    assert scoped["status"] == "fail"
    assert scoped["legacy_benchmark"] == report


@pytest.mark.parametrize("name", REQUIRED_CHECKS)
@pytest.mark.parametrize("corruption", ["missing", "non_boolean"])
def test_contract_fields_fail_closed(observed_benchmark, name, corruption):
    verify, observed = observed_benchmark
    report = deepcopy(observed)
    if corruption == "missing":
        report["checks"].pop(name)
    else:
        report["checks"][name] = 1
    with pytest.raises(ValueError, match="checks"):
        verify.build_ci_contract_report(report)


@pytest.mark.parametrize("ci_contract", [False, True])
def test_cli_scope_and_exit_are_explicit(
    observed_benchmark, monkeypatch, capsys, tmp_path, ci_contract
):
    verify, observed = observed_benchmark
    output = tmp_path / "m5.json"
    argv = ["verify_taiji_m5_episodic_field.py", "--output", str(output)]
    if ci_contract:
        argv.append("--ci-contract")
    monkeypatch.setattr(sys, "argv", argv)
    # Reuse the real observation, including its negative capability result.
    monkeypatch.setattr(verify, "run_benchmark", lambda *, seed: deepcopy(observed))
    exit_code = verify.main()
    saved = json.loads(output.read_text(encoding="utf-8"))
    emitted = json.loads(capsys.readouterr().out.split("VERIFY_RESULT: ")[-1])

    if ci_contract:
        assert exit_code == 0
        assert emitted["name"] == "taiji_m5_current_contract"
        assert saved["scope"] == "current_m5_six_checks"
        assert saved["legacy_benchmark"] == observed
        assert saved["legacy_benchmark"]["status"] == "fail"
    else:
        assert exit_code == 1
        assert emitted["name"] == "taiji_m5_episodic_field"
        assert saved == observed

"""Live N2 audit guards: drift must fail without rewriting research outcomes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/audit_taiji_b0_n2_stop_reason_disposition.py"
REPORT = REPO / "reports/taiji_b0_n2_stop_reason_disposition_20260915.json"


@pytest.fixture(scope="module")
def audit():
    name = "_b0_n2_disposition_contract"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_current_review_surface_is_complete(audit, report):
    expected = {row["path"] for row in audit.EXPECTED_CONSUMERS}
    assert len(expected) == 17
    assert audit.live_consumers() == expected
    current = audit.disposition()
    assert current["review_checks_passed"] is True
    assert current["drift"]["added"] == current["drift"]["removed"] == []
    assert current == report
    assert report["classification_counts"] == {
        "record_only_files": 8,
        "judgement_or_mixed_files": 9,
    }


def test_windows_paths_and_self_audit_exclusions(audit, monkeypatch):
    paths = [r"scripts\training\example.py", *audit.SELF_AUDIT_PATHS]
    fake = SimpleNamespace(
        stop_reason_consumers=lambda: {"files": [{"path": path} for path in paths]}
    )
    monkeypatch.setattr(audit, "load_hardening", lambda: fake)
    assert audit.live_consumers() == {"scripts/training/example.py"}


def test_historical_inventory_is_not_rewritten(report):
    historic = json.loads(
        (REPO / "reports/taiji_b0_m4_hardening_20260913.json").read_text(encoding="utf-8")
    )
    assert historic["stop_reason_consumers"]["file_count"] == 11
    assert report["consumer_count_now"] == 17
    assert len(report["added_since_hardening_report"]) == 6


def test_all_reviewed_sites_have_markers(audit, report):
    assert len(audit.JUDGEMENT_SITES) == 10
    assert report["judgement_sites_intact"] is True
    assert report["missing_judgement_sites"] == []
    assert all(row["marker_present"] for row in report["judgement_sites"])
    kinds = {row["id"]: row["kind"] for row in report["judgement_sites"]}
    assert kinds["J1"] == kinds["J3"] == "replica_equality"
    assert kinds["J2"] == "prefix_predicate"
    assert kinds["J5"] == "substring_predicate"
    assert (
        kinds["J4"] == kinds["J6"] == kinds["J7"] == kinds["J9"] == kinds["J10"] == "test_assertion"
    )


@pytest.mark.parametrize("mutation", ["added", "removed", "missing_marker"])
def test_live_drift_is_a_failing_audit(audit, monkeypatch, mutation):
    expected = {row["path"] for row in audit.EXPECTED_CONSUMERS}
    if mutation == "added":
        expected.add("scripts/training/new_consumer.py")
    elif mutation == "removed":
        expected.remove(next(iter(sorted(expected))))
    else:
        monkeypatch.setattr(
            audit,
            "JUDGEMENT_SITES",
            ({**audit.JUDGEMENT_SITES[0], "marker": "N2_INTENTIONALLY_MISSING_MARKER"},),
        )
    monkeypatch.setattr(audit, "live_consumers", lambda: expected)
    result = audit.disposition()
    assert result["review_checks_passed"] is False
    assert (
        result["drift"]["added"] or result["drift"]["removed"] or result["missing_judgement_sites"]
    )


@pytest.mark.parametrize("passed, expected_code", [(True, 0), (False, 1)])
def test_cli_preserves_failed_evidence_but_exits_nonzero(
    audit, report, monkeypatch, tmp_path, passed, expected_code
):
    payload = {**report, "review_checks_passed": passed}
    monkeypatch.setattr(audit, "disposition", lambda: payload)
    output = tmp_path / "n2-evidence.json"
    assert audit.main(["--output", str(output)]) == expected_code
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_blocked_is_terminal_noncompletion_not_success_or_safety_proof(report):
    semantics = report["terminal_semantics"]
    assert semantics["implies_goal_reached"] is False
    assert semantics["implies_global_member_inability"] is False
    assert "terminal task non-completion" in semantics["all_members_blocked"]
    assert "cannot be inferred" in semantics["safety_status"]
    assert report["new_reason_carries_interception_prefix"] is False
    assert "not an exhaustive data-flow proof" in report["scope_limit"]
    assert report["status"] == "inventory_guard_for_frozen_preregistration"
    assert "M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md" in report["does_not_decide_n2"]
    assert report["does_not_decide_n2"]

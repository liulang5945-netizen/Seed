"""Contract guard for the step-10 closeout evidence card.

The card itself is a read-only recomputation, so the unit under test is the
verdict logic: every check must be able to fail. Each test tampers exactly one
field of the real reports and asserts the specific status that field owns.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "verify_taiji_closeout_10.py"


def _load(name: str, path: Path):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CLOSEOUT = _load("verify_taiji_closeout_10", SCRIPT)


def _inputs():
    probes = {
        sid: json.loads((REPO / relative).read_text(encoding="utf-8"))
        for sid, relative in CLOSEOUT.PROBE_REPORTS.items()
    }
    admission = json.loads((REPO / CLOSEOUT.ADMISSION_REPORT).read_text(encoding="utf-8"))
    card = json.loads((REPO / CLOSEOUT.CLOSEOUT_CARD).read_text(encoding="utf-8"))
    return probes, admission, card


def _status(probes, admission, card) -> str:
    return CLOSEOUT.status_from(CLOSEOUT.audit(probes, admission, card))


def test_real_reports_close_complete():
    probes, admission, card = _inputs()
    checks = CLOSEOUT.audit(probes, admission, card)
    assert CLOSEOUT.status_from(checks) == "closeout_10_evidence_complete"
    assert [check["id"] for check in checks] == list(CLOSEOUT.CHECK_ORDER)
    assert all(check["passed"] is True for check in checks)


def test_tampered_exit_is_named():
    probes, admission, card = _inputs()
    probes["10.4"]["status"] = "passed_10_4"
    assert _status(probes, admission, card) == "exit_not_recorded"


def test_tampered_sequence_breaks_mechanism():
    probes, admission, card = _inputs()
    row = probes["10.7"]["runs"][0]["held_out"][0]
    row["hit"] = not row["hit"]
    assert _status(probes, admission, card) == "mechanism_not_reproduced"


def test_missing_runs_fails_closed():
    probes, admission, card = _inputs()
    probes["10.7"].pop("runs")
    assert _status(probes, admission, card) == "mechanism_not_reproduced"


def test_tampered_identity_is_named():
    probes, admission, card = _inputs()
    probes["10.6"]["parent_model_digest"] = "0" * 64
    assert _status(probes, admission, card) == "identity_mismatch"


def test_capability_claim_is_named():
    probes, admission, card = _inputs()
    probes["10.5"]["capability_claimed"] = True
    assert _status(probes, admission, card) == "capability_claim_forbidden"


def test_weight_publication_is_named():
    probes, admission, card = _inputs()
    probes["10.7"]["weights_published"] = True
    assert _status(probes, admission, card) == "weights_published_forbidden"


def test_admission_regression_is_named():
    probes, admission, card = _inputs()
    admission["status"] = "admitted_to_l3_scope"
    assert _status(probes, admission, card) == "admission_regressed"


def test_budget_overrun_is_named():
    probes, admission, card = _inputs()
    probes["10.6"]["cpu_seconds"] = probes["10.6"]["budget"]["cpu_seconds"] + 1.0
    assert _status(probes, admission, card) == "budget_stop"


def test_retention_gate_is_checked():
    probes, admission, card = _inputs()
    fast = [run for run in probes["10.7"]["runs"] if abs(run["lr"] - 0.0001) <= 1e-12][0]
    fast["retention_drift"] = 0.001
    assert _status(probes, admission, card) == "mechanism_not_reproduced"


def test_loss_ratio_is_checked():
    probes, admission, card = _inputs()
    fast = copy.deepcopy(
        [run for run in probes["10.7"]["runs"] if abs(run["lr"] - 0.0001) <= 1e-12][0]
    )
    fast["last_loss"] = fast["first_loss"] * 0.95
    for run in probes["10.7"]["runs"]:
        if abs(run["lr"] - 0.0001) <= 1e-12:
            run["last_loss"] = fast["last_loss"]
    assert _status(probes, admission, card) == "mechanism_not_reproduced"

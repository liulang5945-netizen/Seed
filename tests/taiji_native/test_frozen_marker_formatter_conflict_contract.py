"""The formatter/audit-guard conflict must stay resolved, and provably so.

Background lives in ``scripts/training/audit_frozen_marker_formatter_conflict.py``:
``black`` reflowed the call carrying frozen judgement site J12, which made the N2 audit
lose its marker and report ``review_checks_passed = False``.  The mitigation was a
``[tool.black] extend-exclude`` entry -- and that entry was silently broken for a whole
round, because ``extend-exclude`` is one regex and its alternatives had been spread
over separate lines.

These tests therefore do two things: assert the conflict is currently handled, and
**falsify** the guard by handing it the broken pattern.  A test that only asserted
"passed is True" would stay green after somebody reintroduced the exact bug.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/audit_frozen_marker_formatter_conflict.py"
REPORT = REPO / "reports/taiji_frozen_marker_formatter_conflict.json"

#: The file whose reformatting broke J12, and the marker that must stay contiguous.
J12_TARGET = "scripts/training/run_taiji_unified_entry_evidence.py"
J12_MARKER = '"stop": "all_members_exhausted"})'

#: Frozen inventory size.  A new frozen marker is a review event, not a ratchet to widen.
FROZEN_MARKER_TOTAL = 12


@pytest.fixture(scope="module")
def guard():
    name = "_frozen_marker_formatter_conflict_contract"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def live(guard):
    return guard.classify(guard.discover_frozen_markers(), guard.black_mode())


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def _record(live, target, marker):
    for record in live["records"]:
        if record["target"] == target and record["marker"] == marker:
            return record
    raise AssertionError(f"no record for {target} :: {marker}")


def test_the_committed_report_matches_the_live_tree(guard, live, report):
    assert live["passed"] is True
    # black_version / black_line_length are environment metadata, not the contract:
    # CI may resolve a different black than a dev machine, and a version bump is not a
    # finding.  The claim under test is which markers exist and how each survived.
    semantic = ("records", "counts", "total_markers", "passed")
    assert {key: live[key] for key in semantic} == {key: report[key] for key in semantic}
    assert report["format"] == guard.FORMAT


def test_inventory_is_frozen_and_all_markers_are_located(live):
    assert live["total_markers"] == FROZEN_MARKER_TOTAL
    assert set(live["counts"]) <= {"ok", "fragile_but_excluded"}
    assert all(record["marker"] for record in live["records"])
    assert all(Path(REPO / record["target"]).is_file() for record in live["records"])


def test_j12_is_genuinely_fragile_and_the_exclusion_is_what_saves_it(guard, live):
    """The substantive claim: formatting really would still break J12.

    If this ever stops being true the exclusion is dead weight and should go, so the
    test pins both halves -- fragility and protection.
    """
    source = (REPO / J12_TARGET).read_text(encoding="utf-8")
    assert J12_MARKER in source, "the marker drifted; J12 needs re-review, not reformatting"

    formatted = guard.formatted_source(source, guard.black_mode())
    assert formatted is not None
    assert J12_MARKER not in formatted, (
        "black no longer reflows the J12 call; the extend-exclude entry is now dead "
        "weight and should be removed together with this assertion"
    )

    record = _record(live, J12_TARGET, J12_MARKER)
    assert record["status"] == "fragile_but_excluded"
    assert record["excluded_from_black"] is True


def test_broken_exclude_pattern_turns_the_guard_red(guard, live, monkeypatch):
    """The exact bug that hid for a round: alternatives written on separate lines.

    ``extend-exclude`` is a single regex, so a newline inside it makes the first branch
    require a trailing newline and match no real path.  The guard must notice.
    """
    broken = re.compile(
        "/(_libs|build|dist|node_modules|scripts/archive|direct-[^/]+)/"
        "\n^/scripts/training/run_taiji_unified_entry_evidence\\.py$"
    )
    monkeypatch.setattr(guard, "exclusion_patterns", lambda: [broken])
    assert guard.excluded_from_black(REPO / J12_TARGET) is False

    crippled = guard.classify(guard.discover_frozen_markers(), guard.black_mode())
    assert crippled["passed"] is False
    assert crippled["counts"].get("fragile_unprotected") == 1
    assert [failure["target"] for failure in crippled["failures"]] == [J12_TARGET]
    # The rest of the inventory is unaffected: the guard isolates, it does not scream.
    assert crippled["counts"].get("ok") == live["counts"].get("ok")


def test_a_missing_marker_is_drift_not_silence(guard, monkeypatch):
    """A frozen marker whose target no longer contains it must be a hard failure."""
    fake = [
        {
            "declared_in": "scripts/training/audit_taiji_b0_n2_stop_reason_disposition.py",
            "table": "JUDGEMENT_SITES",
            "target": J12_TARGET,
            "marker": "this_marker_does_not_exist_anywhere",
        }
    ]
    result = guard.classify(fake, guard.black_mode())
    assert result["passed"] is False
    assert result["records"][0]["status"] == "drift"


def test_pyproject_pattern_is_a_single_alternation(guard):
    """Guard the shape of the config itself, not just its effect.

    The failure mode was structural (a newline inside one regex), so a structural check
    fails fast and names the cause even when the effect check happens to pass.
    """
    raw = str(guard._black_config().get("extend-exclude", ""))
    assert (
        "\n" not in raw.strip()
    ), "extend-exclude is one regex; multi-line values silently break every branch"
    assert len(guard.exclusion_patterns()) >= 1

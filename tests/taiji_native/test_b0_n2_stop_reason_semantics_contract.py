"""Contract guard for the frozen N2 stop-reason semantics (WP-2).

``all_members_blocked`` is the terminal reason the audited rule (M4) introduces.
Its meaning has to be frozen *before* the runner changes, so every check here is
two-directional against archived evidence: it pins what the reason is allowed to
mean and what it must never be read as.  Written while M4 was still unimplemented and
expected to keep holding after it shipped -- a semantics guard that only passes before
landing is worth nothing once the rule is live.

Measured payload: ``reports/taiji_b0_structure_space_probe_wide_20260915.json``,
``reports/taiji_b0_structure_space_probe_m4landed_20260915.json`` and
``reports/taiji_b0_n2_stop_reason_disposition_20260915.json``.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
N2_REPORT = REPO / "reports" / "taiji_b0_n2_stop_reason_disposition_20260915.json"
STRUCTURE_REPORT = REPO / "reports" / "taiji_b0_structure_space_probe_wide_20260915.json"
COUNTERFACTUAL_SCRIPT = REPO / "scripts" / "training" / "probe_taiji_b0_m1_counterfactual.py"
PREREG = REPO / "plans" / "reference" / "M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md"

NEW_REASON = "all_members_blocked"
FROZEN_COUNTERPART = "all_members_exhausted"
INTERCEPTION_PREFIX = "contract_intercepted"
GOAL_REASON = "goal_reached"

#: The three cells where the audited rule is expected to hand off and gain, plus the
#: cell that hands off without changing the outcome -- so a handoff by itself is not
#: evidence of a gain, and the reason is not evidence of incapability either.
POSITIVE_CELLS = ["create__mismatch", "create__observation", "create__override"]
BLOCKED_BUT_NEUTRAL_CELL = "create__none"

LIVE_JUDGEMENT_KINDS = {"replica_equality", "prefix_predicate", "substring_predicate"}
TEST_ASSERTION_KIND = "test_assertion"

#: This file consumes stop reasons and is itself dispositioned as J8 by the scanner.
SELF_PATH = "tests/taiji_native/test_b0_n2_stop_reason_semantics_contract.py"


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    for entry in (str(path.parent), str(REPO)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def n2() -> dict:
    return json.loads(N2_REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def structure() -> dict:
    return json.loads(STRUCTURE_REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(structure: dict) -> dict[str, dict]:
    return {row["cell"]: row for row in structure["rows"]}


@pytest.fixture(scope="module")
def counterfactual() -> Any:
    return _load("_n2_semantics_counterfactual", COUNTERFACTUAL_SCRIPT)


@pytest.fixture(scope="module")
def frozen_gate(counterfactual: Any) -> Any:
    return counterfactual.load_frozen()


def _interceptions(counts: dict[str, int]) -> dict[str, int]:
    return {key: value for key, value in counts.items() if key.startswith(INTERCEPTION_PREFIX)}


# --------------------------------------------------------------------------- #
# I2: the new reason is a handoff terminal, never a contract interception
# --------------------------------------------------------------------------- #


def test_the_new_reason_carries_no_interception_prefix(n2: dict) -> None:
    assert n2["interception_prefix"] == INTERCEPTION_PREFIX
    assert n2["new_reason_carries_interception_prefix"] is False
    assert not n2["new_reason"].startswith(INTERCEPTION_PREFIX)
    assert not n2["frozen_counterpart"].startswith(INTERCEPTION_PREFIX)


def test_handoff_and_interception_stay_separate_categories(rows: dict[str, dict]) -> None:
    """Reverse direction: the reason never lands inside the interception bucket."""

    for cell, row in rows.items():
        for counts in (row["stop_reasons_frozen"], row["stop_reasons_audited"]):
            assert NEW_REASON not in _interceptions(counts), cell


# --------------------------------------------------------------------------- #
# I1 / I4: a blocked episode is a non-success terminal only the audited rule emits
# --------------------------------------------------------------------------- #


def _blocked_branch(source: str) -> str:
    """The ``if chosen is None:`` arm of the shipped rule, up to its terminating return."""

    marker = "        if chosen is None:"
    terminator = f'return finish("{NEW_REASON}")'
    assert marker in source and terminator in source, "the blocked arm is no longer in the gate"
    start = source.index(marker)
    end = source.index(terminator, start) + len(terminator)
    return source[start:end]


def test_the_blocked_branch_records_no_execution_and_no_goal(
    counterfactual: Any, frozen_gate: Any
) -> None:
    """The blocked branch is terminal, reports the step as not executed, never as goal attainment.

    Audit item B3: the first version grepped four strings out of ``M4_SELECTION``, a constant
    defined *inside the probe*, so it stayed green however the shipped gate was edited -- a
    "source-level pin" that never opened the source.  The constant is now required to be a
    verbatim substring of the live ``_member_episode``, which makes the two agree or the test
    red, and the properties are read off a slice of the real file.
    """

    source = inspect.getsource(frozen_gate._member_episode)
    assert (
        counterfactual.M4_SELECTION in source
    ), "the probe's idea of the selection block is no longer the code that ships"

    block = _blocked_branch(source)
    assert f'"stop": "{NEW_REASON}"' in block
    assert '"executed": False' in block
    assert f'return finish("{NEW_REASON}")' in block
    assert GOAL_REASON not in block


def test_the_new_reason_only_ever_applies_to_a_handoff_rule(
    frozen_gate: Any, rows: dict[str, dict]
) -> None:
    """The revision-0 claim ("the frozen rule cannot produce it") is now historical.

    What must hold across revisions instead: the reason is produced only by a rule that
    implements handoff, and the archived revision-0 evidence must show it never occurred
    under the priority-fallback rule.  Pinning the live source here after landing would
    assert a property of a rule that no longer ships.
    """

    source = inspect.getsource(frozen_gate._member_episode)
    shipped = 'return finish("all_members_blocked")' in source
    if shipped:
        # revision 1: the shipped rule can terminate this way, and the frozen column of
        # the archived revision-0 reports must stay clean (checked below).
        assert "if chosen is None:" in source
    else:  # revision 0: no handoff, so the reason cannot exist in the rule at all
        assert NEW_REASON not in source
    for cell, row in rows.items():
        assert row["stop_reasons_frozen"].get(NEW_REASON, 0) == 0, cell


def test_the_audited_rule_counts_blocked_episodes_apart_from_successes(
    rows: dict[str, dict],
) -> None:
    for cell in POSITIVE_CELLS:
        audited = rows[cell]["stop_reasons_audited"]
        assert audited.get(NEW_REASON, 0) > 0, cell
        assert audited.get(GOAL_REASON, 0) > 0, cell
        assert audited[GOAL_REASON] < audited[NEW_REASON], cell


def test_a_handoff_without_gain_still_terminates_honestly(rows: dict[str, dict]) -> None:
    """``create__none`` hands off and stays at zero gain: the reason proves neither a
    gain nor that any member is incapable."""

    row = rows[BLOCKED_BUT_NEUTRAL_CELL]
    assert row["stop_reasons_audited"].get(NEW_REASON, 0) > 0
    assert row["best_pair_gain_audited"] == 0.0
    assert row["interleaved_contexts_audited"] == 0


# --------------------------------------------------------------------------- #
# I5: the handoff must not crowd out contract interceptions
# --------------------------------------------------------------------------- #


def test_unrepaired_interceptions_are_identical_under_both_rules(rows: dict[str, dict]) -> None:
    """I5 (narrowed): only the interceptions the handoff does *not* touch must stay equal.

    ``contract_intercepted:preview_ValueError`` is unrelated to member selection, so any
    change there would mean the gain comparison is contaminated.  The language-assessment
    interception is the opposite case: the frozen rule accumulates it because the first
    member never lets anyone else create the file, and M4 repairs it -- so it must vanish,
    and where it does, that disappearance is mechanism evidence rather than pollution.
    """

    repaired = f"{INTERCEPTION_PREFIX}:language_assessment_unavailable"
    untouched = f"{INTERCEPTION_PREFIX}:preview_ValueError"
    for cell, row in rows.items():
        frozen = row["stop_reasons_frozen"]
        audited = row["stop_reasons_audited"]
        assert frozen.get(untouched, 0) == audited.get(untouched, 0), cell
        if cell == "create__observation":
            assert frozen.get(repaired, 0) > 0, cell
            assert audited.get(repaired, 0) == 0, cell
        else:
            assert frozen.get(repaired, 0) == 0, cell
            assert audited.get(repaired, 0) == 0, cell


# --------------------------------------------------------------------------- #
# I7: the two terminals are never folded into one another
# --------------------------------------------------------------------------- #


def test_the_two_terminal_reasons_are_never_folded(rows: dict[str, dict], n2: dict) -> None:
    assert n2["frozen_counterpart"] == FROZEN_COUNTERPART
    for cell, row in rows.items():
        assert sum(row["stop_reasons_frozen"].values()) == sum(
            row["stop_reasons_audited"].values()
        ), cell
    both = rows[POSITIVE_CELLS[0]]["stop_reasons_audited"]
    assert NEW_REASON in both and FROZEN_COUNTERPART in both


# --------------------------------------------------------------------------- #
# I3: many handoffs coexist with an attributed two-member gain
# --------------------------------------------------------------------------- #


def test_the_reason_coexists_with_an_attributed_two_member_gain(
    rows: dict[str, dict], structure: dict
) -> None:
    assert structure["verdict"]["m4_regresses_cells"] == []
    for cell in POSITIVE_CELLS:
        row = rows[cell]
        explained = row["handoff_explained_changes"]
        assert len(explained) == 1, cell
        entry = explained[0]
        assert entry["baseline_success_rate"] == 0.0, cell
        assert entry["audited_success_rate"] == 1.0, cell
        assert entry["delta"] > 0, cell
        assert row["unexplained_changes"] == [], cell


def test_every_jointly_required_context_shows_two_acting_members(rows: dict[str, dict]) -> None:
    for cell in POSITIVE_CELLS:
        row = rows[cell]
        contexts = row["handoff_explained_changes"][0]["per_context"]
        assert len(contexts) == len(row["contexts"]), cell
        for context in contexts:
            assert context["jointly_required"] is True, (cell, context["context"])
            assert len(context["acting_members"]) >= 2, (cell, context["context"])


# --------------------------------------------------------------------------- #
# The consumer inventory: re-dispositioned, and provably clean
# --------------------------------------------------------------------------- #


def test_the_live_surface_is_dispositioned_with_no_drift(n2: dict) -> None:
    """The scanner exits non-zero while a consumer is undispositioned; this pins the fix."""

    assert n2["drift"]["clean"] is True, n2["drift"]
    assert n2["drift"]["added"] == [] and n2["drift"]["removed"] == []
    assert n2["review_checks_passed"] is True
    assert n2["missing_judgement_sites"] == []
    assert n2["judgement_sites_intact"] is True


def test_this_file_is_itself_dispositioned_as_a_consumer(n2: dict) -> None:
    consumers = {row["path"]: row for row in n2["consumers"]}
    assert SELF_PATH in consumers
    assert consumers[SELF_PATH]["class"] == "judgement (test assertion)"
    sites = {site["id"]: site for site in n2["judgement_sites"]}
    registered = [site for site in sites.values() if site["path"] == SELF_PATH]
    assert len(registered) == 1
    assert registered[0]["marker_present"] is True


def test_every_consumer_carries_a_class_and_a_reason(n2: dict) -> None:
    assert len(n2["consumers"]) == n2["consumer_count"] == n2["consumer_count_now"]
    for consumer in n2["consumers"]:
        assert consumer["class"], consumer["path"]
        assert consumer["why"], consumer["path"]
    assert len({consumer["path"] for consumer in n2["consumers"]}) == n2["consumer_count"]


def test_the_inventory_separates_live_gates_from_test_assertions(n2: dict) -> None:
    live = [s for s in n2["judgement_sites"] if s["kind"] in LIVE_JUDGEMENT_KINDS]
    pinned = [s for s in n2["judgement_sites"] if s["kind"] == TEST_ASSERTION_KIND]
    assert (len(live), len(pinned)) == (4, 6)
    assert len(live) + len(pinned) == len(n2["judgement_sites"])
    assert all(site["marker_present"] is True for site in n2["judgement_sites"])


def test_the_scanner_excludes_only_itself_and_its_own_guard(n2: dict) -> None:
    excluded = {Path(path).name for path in n2["self_audit_paths_excluded"]}
    assert excluded == {
        "audit_taiji_b0_n2_stop_reason_disposition.py",
        "test_b0_n2_stop_reason_disposition_contract.py",
    }
    consumer_names = {Path(consumer["path"]).name for consumer in n2["consumers"]}
    assert consumer_names.isdisjoint(excluded)
    assert Path(SELF_PATH).name in consumer_names


def test_the_hardening_round_count_is_recorded_as_stale(n2: dict) -> None:
    assert n2["consumer_count_in_hardening_report"] == 11
    assert n2["count_was_stale"] is True
    added = set(n2["added_since_hardening_report"])
    assert len(added) == n2["consumer_count_now"] - 11
    assert SELF_PATH in added


# --------------------------------------------------------------------------- #
# The frozen document must match the evidence it cites
# --------------------------------------------------------------------------- #


def test_the_frozen_preregistration_states_the_measured_counts(n2: dict) -> None:
    text = PREREG.read_text(encoding="utf-8")
    assert "FROZEN" in text
    assert str(n2["consumer_count"]) in text, "the consumer count must match the live scan"
    assert str(len(n2["judgement_sites"])) in text, "the judgement-site count must match"
    assert "10 处判断点" in text and "17 个消费文件" in text

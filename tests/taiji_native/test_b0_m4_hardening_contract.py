"""Contract guard for the M4 hardening sweep (audit §6 risks 2, 3 and 4).

The sweep trains members for five seed offsets and runs seven surface measurements
(about 80 seconds), so it is a scripted instrument.  Pinned here are the parts that
decision D5 leans on, in **both directions** -- what must hold and what must be
refused:

* the scale bar the audit asked for (>= 12 contexts, >= 5 seeds) must actually be
  met, not merely argued to be unnecessary;
* the M4 gain must survive on **every** structural variant while the frozen rule
  gains on none, so widening the surface cannot have smuggled in an artifact;
* the sweep's own **limit** must stay disclosed: the six variants share one
  composition structure, so identical outcomes bound surface sensitivity only;
* each variant's contract prediction must agree with rule-independent scripted
  evidence -- including the ``.h``/``cpp`` prediction that the evidence overturned;
* the risk-2 file scan must not be padded with throwaway or non-source files.

The measured payload lives in ``reports/taiji_b0_m4_hardening_20260913.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE_SCRIPT = REPO / "scripts" / "training" / "audit_taiji_b0_m4_hardening.py"
MODULE_REPORT = REPO / "reports" / "taiji_b0_m4_hardening_20260913.json"

#: The audit's §6 remediation column states these numbers; they are the bar.
MIN_CONTEXTS = 12
MIN_SEED_OFFSETS = 5
MIN_VARIANTS = 6


def _load(name: str, path: Path):
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
def payload() -> dict:
    return json.loads(MODULE_REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hardening():
    return _load("_b0_m4_hardening_under_test", MODULE_SCRIPT)


# --------------------------------------------------------------------------- #
# Read-only discipline
# --------------------------------------------------------------------------- #


def test_sweep_is_read_only_and_m4_still_not_implemented(payload):
    assert payload["status"] == "draft_for_review"
    assert payload["rule_under_audit"] == "m4_failure_handoff"
    assert payload["frozen_attribute_intact"] is True

    source = MODULE_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("torch.save", "save_file", "from_checkpoint(", "optimizer"):
        assert forbidden not in source


def test_module_targets_exactly_the_decision_free_residual_risks(hardening):
    """Risks 1 and 5 belong to D5; silently claiming them would overstate the sweep."""

    doc = hardening.__doc__ or ""
    assert "Risk 2" in doc and "Risk 3" in doc and "Risk 4" in doc
    assert "Risks 1" in doc and "left to D5" in doc


# --------------------------------------------------------------------------- #
# Risk 3: the numeric bar, and the honest bound
# --------------------------------------------------------------------------- #


def test_scale_bar_from_the_audit_is_actually_met(payload):
    size = payload["effective_sample_size"]
    assert size["contexts"] >= MIN_CONTEXTS
    assert size["distinct_variants"] >= MIN_VARIANTS
    assert payload["audited_rule"]["context_count"] == size["contexts"]


def test_seed_bar_from_the_audit_is_actually_met(payload):
    assert len(payload["seed_sweep"]["offsets"]) >= MIN_SEED_OFFSETS


def test_variants_differ_in_surface_and_not_merely_by_index(hardening):
    specs = hardening.VARIANT_SPECS
    fingerprints = {(s["extension"], s["language"], s["base"], s["goal"]) for s in specs}
    assert len(fingerprints) == len(specs)
    assert len({s["extension"] for s in specs}) >= 3
    assert len({s["language"] for s in specs}) >= 2


def test_gain_survives_on_every_variant_and_on_none_under_the_frozen_rule(payload):
    audited, baseline = payload["audited_rule"], payload["baseline_rule"]
    assert audited["positive_variant_count"] == audited["distinct_variants"]
    assert [row["best_pair_gain"] for row in audited["per_variant"]] == [2.0] * len(
        audited["per_variant"]
    )
    assert baseline["positive_variant_count"] == 0
    assert baseline["pooled_best_pair_gain"] == 0.0
    assert baseline["interleaved_contexts"] == 0


def test_report_discloses_that_variants_are_surface_distinct_only(payload):
    """Six copies of one structural fact are one fact; the sweep must say so."""

    distinct = payload["effective_sample_size"]["outcome_distinctness"]
    assert distinct["variants_indistinguishable_in_outcome"] is True
    assert distinct["distinct_outcome_fingerprints"] == 1
    assert "surface sensitivity" in distinct["bounds"]
    assert "T1/T2/T3" in distinct["does_not_bound"]


def test_distinctness_diagnostic_is_not_a_constant(hardening):
    """The negative direction: differing variants must be reported as distinct."""

    audited = {
        "per_variant": [
            {
                "variant": "x",
                "best_pair": "member-a+member-c",
                "best_pair_gain": 2.0,
                "combination_only_contexts": 2,
                "interleaved_contexts": 2,
                "stop_reasons": {"goal_reached": 4},
            },
            {
                "variant": "y",
                "best_pair": "member-b+member-c",
                "best_pair_gain": 0.0,
                "combination_only_contexts": 0,
                "interleaved_contexts": 0,
                "stop_reasons": {"step_cap": 8},
            },
        ]
    }
    result = hardening.variant_outcome_distinctness(audited)
    assert result["distinct_outcome_fingerprints"] == 2
    assert result["variants_indistinguishable_in_outcome"] is False
    assert "outcome sensitivity" in result["bounds"]


# --------------------------------------------------------------------------- #
# Contract predictions are checked against evidence
# --------------------------------------------------------------------------- #


def test_every_contract_prediction_matches_scripted_evidence(payload):
    audit = payload["contract_expectation_audit"]
    assert audit["all_match"] is True
    assert audit["contradicted"] == []
    assert len(audit["rows"]) == payload["audited_rule"]["distinct_variants"]
    for row in audit["rows"]:
        assert row["expect_contract_ok"] == row["observed_scripted_admissible"]


def test_corrected_cpp_prediction_stays_corrected(payload):
    """``.h``/cpp was predicted to be intercepted and the evidence said otherwise.

    The wrong prediction must not creep back in: an explicit ``user_override`` plus an
    unambiguous extension gives the contract layer no ambiguity to reject, unlike the
    frozen block-3 family whose language evidence is absent.
    """

    rows = {row["variant"]: row for row in payload["contract_expectation_audit"]["rows"]}
    cpp = rows["h_cpp_header"]
    assert cpp["expect_contract_ok"] is True
    assert cpp["observed_scripted_admissible"] is True


def test_every_context_is_satisfiable_and_carries_no_blocked_step(payload):
    rows = payload["satisfiability"]
    assert len(rows) == payload["audited_rule"]["context_count"]
    assert all(row["forward_reaches_goal"] is True for row in rows)
    assert all(row["scripted_blocked_steps"] == [] for row in rows)


# --------------------------------------------------------------------------- #
# Risk 4: seeds, and no inert cells
# --------------------------------------------------------------------------- #


def test_seed_sweep_is_positive_on_every_offset(payload):
    sweep = payload["seed_sweep"]
    assert sweep["all_seeds_pooled_positive"] is True
    assert sweep["min_pooled_gain"] == sweep["max_pooled_gain"] == 2.0
    assert sweep["min_positive_variant_count"] == payload["audited_rule"]["distinct_variants"]
    assert all(row["intervention_reality_ok"] is True for row in sweep["rows"])


def test_no_inert_cells_under_either_rule(payload):
    """A zero-step cell is what made P5.2b a false success; it must not return."""

    for rule in ("baseline_rule", "audited_rule"):
        reality = payload[rule]["intervention_reality"]
        assert reality["interventions_happened"] is True
        assert reality["zero_step_intervention_episodes"] == 0


def test_feasibility_holds_on_all_three_references(payload):
    for row in payload["audited_rule"]["reference_requirements"]:
        assert row["feasible"] is True
        assert (
            row["available_combination_only_contexts"] >= row["required_combination_only_contexts"]
        )


# --------------------------------------------------------------------------- #
# Risk 2: stop-reason semantics
# --------------------------------------------------------------------------- #


def test_blocked_reason_distinguishes_m4_from_the_frozen_rule(payload):
    audited = payload["audited_rule"]["stop_reasons"]
    baseline = payload["baseline_rule"]["stop_reasons"]
    assert audited.get("all_members_blocked", 0) > 0
    assert "all_members_blocked" not in baseline
    assert baseline.get("step_cap", 0) > 0


def test_contract_interception_is_rule_independent(payload):
    """Interceptions come from members acting off-repertoire, not from M4.

    If M4 manufactured interceptions the gain comparison would be confounded.
    """

    audited = dict(payload["audited_rule"]["stop_reasons"])
    baseline = dict(payload["baseline_rule"]["stop_reasons"])
    intercepted = "contract_intercepted:preview_ValueError"
    assert audited[intercepted] == baseline[intercepted]


def test_stop_reason_scan_is_limited_to_real_sources(payload):
    scan = payload["stop_reason_consumers"]
    assert scan["file_count"] == len(scan["files"])
    assert scan["file_count"] > 0
    for row in scan["files"]:
        path = row["path"]
        assert path.endswith(".py")
        assert path.split("\\")[0] in {"scripts", "seed_platform", "taiji", "tests"}
        assert "_smoke" not in path
        assert not path.startswith("reports")


def test_scan_reports_new_reason_as_already_referenced(payload):
    assert payload["stop_reason_consumers"]["new_reason_already_referenced"] is True


# --------------------------------------------------------------------------- #
# Variant construction, without loading the frozen executor
# --------------------------------------------------------------------------- #


class _Record:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def test_variant_task_wires_both_demands(hardening):
    """Create *and* explicit language override -- the combination-only requirement."""

    class StubP52:
        ScriptedStep = _Record

    class StubP52a:
        Task = _Record

    class StubFrozen:
        p52 = StubP52
        p52a = StubP52a

    spec = {
        "label": "stub",
        "extension": ".py",
        "language": "python",
        "base": "x = {i}\n",
        "goal": "x = {i} + 1\n",
        "expect_contract_ok": True,
    }
    task = hardening.variant_tasks(StubFrozen(), spec, 7)

    assert task.kwargs["initial_files"] == {}
    assert task.kwargs["goal_files"] == {"hard_stub_7.py": "x = 7 + 1\n"}
    assert task.kwargs["goal_language"] == {"hard_stub_7.py": "python"}
    assert task.kwargs["requires_explicit_language_override"] is True

    kinds = [step.args[0] for step in task.kwargs["reference_steps"]]
    assert "workspace.create" in kinds
    override = next(s for s in task.kwargs["reference_steps"] if s.args[0] == "editor.set_language")
    assert override.args[1]["user_override"] is True
    assert override.args[1]["programming_language_id"] == "python"


def test_stop_reasons_by_variant_ignores_foreign_episodes(hardening):
    episodes = [
        {"task_id": "b0harden-py_python_fn-400", "stop_reason": "goal_reached"},
        {"task_id": "b0harden-py_python_fn-401", "stop_reason": "all_members_blocked"},
        {"task_id": "somewhere-else", "stop_reason": "goal_reached"},
    ]
    owner = {
        "b0harden-py_python_fn-400": "py_python_fn",
        "b0harden-py_python_fn-401": "py_python_fn",
    }
    rows = hardening.stop_reasons_by_variant(episodes, owner)
    assert list(rows) == ["py_python_fn"]
    assert rows["py_python_fn"] == {"all_members_blocked": 1, "goal_reached": 1}
    assert hardening.was_contract_intercepted(rows["py_python_fn"]) is False
    assert hardening.was_contract_intercepted({"contract_intercepted:preview_x": 2}) is True

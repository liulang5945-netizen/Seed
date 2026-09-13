"""Regression guard for the B0 measurement dictionary (route-B entry work).

Route A failed the transfer gate, but the post-route-A review found that the
*frozen gate itself* compared two numbers produced against **different
references**: the object used ``mean(P_ij - max(S_i, S_j))`` while control C2
used ``mean(max_i S_i - B)``, and both were carried in one field name
``mean_gain_vs_strongest_single`` before a single margin test consumed them.

This file pins the replacement dictionary -- a set of pure functions -- with
hand-calculated cases, so the estimands can no longer be conflated:

* ``fully_redundant``                  -- two members share one surface;
* ``coverage_complementary_no_synergy`` -- routing gains exist, synergy does not;
* ``jointly_required``                 -- the only regime supporting a
  same-reference collaboration claim;
* ``all_fail``                         -- an unsolvable task is neither evidence
  nor a violation;
* ``equal_outcome_different_cost``     -- cost must be frozen *before* the run.

Two properties matter more than the individual numbers:

1. the pair-internal reference is *weakly weaker* than the all-singleton
   reference (``max(S_i, S_j) <= max_i S_i``), so the two frozen fields are not
   two readings of one quantity -- a test demonstrates them diverging;
2. reachability is a property of the **task**, not of the representation: when
   no cell can beat the oracle, no model change can clear the gate.  The frozen
   route-C matrix is asserted to have that property, which is why route B has to
   start by redefining the task rather than by adding features.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = REPO / "scripts" / "training" / "audit_taiji_b0_measurement_reachability.py"
ROUTE_A_REPORT = (
    REPO / "reports" / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
)
ROUTE_C_REPORT = (
    REPO / "reports" / "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json"
)
P52B_REPORT = REPO / "reports" / "taiji_p5_2b_group_causal_corpora_20260913.json"


def _load(name: str, path: Path):
    scripts = str(path.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dictionary():
    return _load("_b0_measurement_dictionary", AUDIT_SCRIPT)


@pytest.fixture(scope="module")
def audit_payload(dictionary) -> Mapping[str, Any]:
    return dictionary.audit(ROUTE_A_REPORT, ROUTE_C_REPORT, P52B_REPORT)


# --------------------------------------------------------------------------- #
# Hand-calculated cases
# --------------------------------------------------------------------------- #


def test_fully_redundant_pair_buys_nothing(dictionary):
    """Two members with one identical surface: zero gain on every reference."""

    table = dictionary.case_fully_redundant()
    pair = ("member-a", "member-d")

    assert dictionary.policy_mean_gain_vs_pair_internal_oracle(table, pair) == 0.0
    assert dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair) == 0.0
    assert dictionary.policy_mean_gain_vs_baseline(table, pair) == 0.5
    # One shared block: the pair deviates from the additive expectation by -2 on
    # that context, which is exactly what "redundant" means arithmetically.
    assert dictionary.policy_mean_interaction(table, pair) == -0.5
    assert dictionary.contexts_beating_all_singleton_oracle(table, pair) == 0


def test_coverage_complementary_pair_gains_by_routing_only(dictionary):
    """The current P5.2c regime: coverage beats a fixed singleton, synergy is 0."""

    table = dictionary.case_coverage_complementary_no_synergy()
    pair = ("member-a", "member-b")

    pair_gain = dictionary.policy_mean_gain_vs_baseline(table, pair)
    member_gain = dictionary.policy_mean_gain_vs_baseline(table, ("member-a",))
    other_gain = dictionary.policy_mean_gain_vs_baseline(table, ("member-b",))

    # The pair covers two blocks, each fixed singleton covers one.
    assert pair_gain == 1.0
    assert member_gain == 0.5
    assert other_gain == 0.5
    assert pair_gain > member_gain

    # ... yet it never exceeds the per-context oracle, so no collaboration claim.
    assert dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair) == 0.0
    assert dictionary.policy_mean_gain_vs_pair_internal_oracle(table, pair) == 0.0
    assert dictionary.policy_mean_interaction(table, pair) == 0.0
    assert dictionary.contexts_beating_all_singleton_oracle(table, pair) == 0


def test_jointly_required_pair_is_the_only_collaboration_regime(dictionary):
    """Both singletons fail everywhere; only the pair succeeds."""

    table = dictionary.case_jointly_required()
    pair = ("member-a", "member-b")

    assert dictionary.policy_mean_interaction(table, pair) == 2.0
    assert dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair) == 2.0
    assert dictionary.policy_mean_gain_vs_pair_internal_oracle(table, pair) == 2.0
    assert dictionary.contexts_beating_all_singleton_oracle(table, pair) == len(table.contexts)

    # A singleton policy on the same task is worth exactly nothing.
    assert dictionary.policy_mean_gain_vs_baseline(table, ("member-a",)) == 0.0


def test_all_fail_task_scores_a_legitimate_zero(dictionary):
    """An unsolvable task must not crash, and must not look like a violation."""

    table = dictionary.case_all_fail()
    pair = ("member-a", "member-b")

    assert dictionary.policy_mean_gain_vs_baseline(table, pair) == 0.0
    assert dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair) == 0.0
    assert dictionary.policy_mean_interaction(table, pair) == 0.0
    assert dictionary.contexts_beating_all_singleton_oracle(table, pair) == 0
    # The blank reference itself is a well-defined number, not a missing value.
    assert dictionary.policy_mean_gain_vs_baseline(table, ()) == 0.0


def test_equal_outcome_different_cost_only_separates_with_a_frozen_weight(dictionary):
    """Cost is a knob unless its weight is fixed before the outcome is seen."""

    table = dictionary.case_equal_outcome_different_cost()
    pair = ("member-a", "member-b")
    singleton = ("member-a",)

    assert dictionary.policy_mean_outcome(table, pair) == dictionary.policy_mean_outcome(
        table, singleton
    )

    # Default weight is 0.0 on purpose: nothing is folded in implicitly.
    assert dictionary.policy_utility(table, pair) == dictionary.policy_utility(
        table, singleton
    )
    # With an explicit weight the two-member policy pays for its extra call.
    assert dictionary.policy_utility(table, pair, cost_weight=0.5) == 0.0
    assert dictionary.policy_utility(table, singleton, cost_weight=0.5) == 0.5
    assert dictionary.policy_utility(
        table, pair, cost_weight=0.5
    ) < dictionary.policy_utility(table, singleton, cost_weight=0.5)


# --------------------------------------------------------------------------- #
# The two frozen fields are not two readings of one quantity
# --------------------------------------------------------------------------- #


def _third_member_table(dictionary):
    """Three members, two contexts: the pair excludes the global strongest."""

    contexts = ("c0", "c1")
    return dictionary.ContextTable(
        contexts=contexts,
        baseline={"c0": -1.0, "c1": -1.0},
        singletons={
            "member-a": {"c0": 1.0, "c1": -1.0},
            "member-b": {"c0": -1.0, "c1": -1.0},
            "member-c": {"c0": 1.0, "c1": 1.0},
        },
        combinations={("member-a", "member-b"): {"c0": 1.0, "c1": -1.0}},
    )


def test_pair_internal_and_all_singleton_references_diverge(dictionary):
    """``max(S_i, S_j) <= max_i S_i`` -- the frozen fields are different estimands."""

    table = _third_member_table(dictionary)
    pair = ("member-a", "member-b")

    pair_internal = dictionary.policy_mean_gain_vs_pair_internal_oracle(table, pair)
    all_singleton = dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair)

    # c0: the pair ties its own best member -> 0 on both references.
    # c1: the pair fails while member-c succeeds -> 0 vs -2.
    assert pair_internal == 0.0
    assert all_singleton == -1.0
    assert pair_internal != all_singleton

    # The pair-internal reference is weakly weaker, never stronger: subtracting
    # it can only make a combination look better than the same-reference number.
    assert pair_internal >= all_singleton


def test_route_a_object_and_control_used_different_references(audit_payload):
    """Recompute the frozen route-A numbers and expose the mismatch."""

    recomputation = audit_payload["route_a_recomputation"]
    rows = {row["pair"]: row for row in recomputation["rows"]}

    # The reported object field is reproducible from the pair-internal formula...
    assert recomputation["object_reported"] == -0.5
    assert recomputation["object_recomputed_pair_internal"] == -0.5
    assert rows["member-a+member-c"]["recomputed_pair_internal_gain"] == -0.5

    # ... and C2 is reproducible from the all-singleton formula.
    assert recomputation["c2_reported"] == 1.5
    assert recomputation["c2_recomputed"] == 1.5
    assert recomputation["required_reported"] == recomputation["required_recomputed"]

    # Re-expressed against C2's reference the object is not -0.5 but -1.0, so the
    # reported margin shortfall was measured across two different scales.
    assert rows["member-a+member-c"]["recomputed_gain_vs_all_singleton_oracle"] == -1.0
    assert rows["member-b+member-d"]["recomputed_gain_vs_all_singleton_oracle"] == -0.5


# --------------------------------------------------------------------------- #
# Reachability is a property of the task, not of the representation
# --------------------------------------------------------------------------- #


def test_block_mean_reconstruction_matches_the_frozen_per_context_rows(audit_payload):
    """Every bound below rests on this reconstruction, so it is checked first."""

    cross_check = audit_payload["cross_check"]
    assert cross_check["all_match"] is True
    assert cross_check["checked_fields"] == 32
    assert cross_check["mismatches"] == []


def test_frozen_matrix_contains_no_combination_that_beats_the_oracle(audit_payload):
    """The reason route B must redefine the task instead of adding features."""

    for row in audit_payload["cells"]:
        assert row["contexts_beating_all_singleton_oracle"] == 0
        assert row["gain_vs_all_singleton_oracle"] <= 0.0

    reachability = audit_payload["reachability"]
    # Even an oracle with outcome access over every executed cell gains nothing
    # over the per-context singleton oracle -- there is no hidden pair capability.
    assert reachability["oracle_all_cell_gain_vs_baseline"] == 1.5
    assert reachability["structural_ceiling_current_matrix"] < 0.0
    # The one context where every singleton fails is the only place a new task
    # could create a same-reference gain.
    assert reachability["ceiling_if_block_becomes_solvable"] == 0.5
    assert (
        audit_payload["task_structure_facts"]["contexts_where_every_singleton_fails"]
        == ["p52a-validation-111"]
    )


def test_every_candidate_reference_is_self_consistently_reported(audit_payload):
    """Guard the reachability logic without freezing today's verdict."""

    reachability = audit_payload["reachability"]
    assert reachability["rows"], "the audit must report at least one candidate reference"

    for row in reachability["rows"]:
        assert row["required"] == pytest.approx(row["reference_value"] + reachability["margin"])
        assert row["reachable"] == (row["achievable_ceiling"] > row["required"])
        assert row["shortfall"] == pytest.approx(row["required"] - row["achievable_ceiling"])

    # An oracle reference must be marked non-deployable, or a gate could compare a
    # policy against something that had access to the outcome it is scored on.
    oracle_rows = [row for row in reachability["rows"] if row["reference"] == "all_singleton_oracle"]
    assert oracle_rows and oracle_rows[0]["reference_is_deployable"] is False
    deployable = [
        row for row in reachability["rows"] if row["reference"] != "all_singleton_oracle"
    ]
    assert deployable and all(row["reference_is_deployable"] for row in deployable)


def test_held_out_selection_spread_bounds_any_ranking_claim(audit_payload):
    """With two held-out pairs 0.5 apart, a perfect selector can only gain 0.5."""

    controls = audit_payload["reachability"]["deployable_controls"]
    gains = controls["held_out_pairs_gain_vs_baseline"]

    assert gains == {"member-a+member-c": 0.5, "member-b+member-d": 1.0}
    assert controls["held_out_selection_spread"] == 0.5

    # Route A's two held-out predictions were identical, so the selected object
    # was decided by tie-breaking, and it picked the weaker of the two.
    binding = audit_payload["prediction_binding_finding"]
    assert binding["prediction_distinctness"] == 2
    predictions = binding["held_out_predictions"]
    assert predictions["member-a+member-c"] == predictions["member-b+member-d"]
    assert binding["selected_object"] == ["member-a", "member-c"]
    assert gains["member-a+member-c"] < gains["member-b+member-d"]


def test_audit_is_read_only_and_does_not_touch_frozen_reports(audit_payload):
    """The audit adds fields; it never rewrites a historical number."""

    assert audit_payload["status"] == "draft_for_review"
    assert audit_payload["measurement_dictionary_version"].endswith("-draft")
    assert "does_not_change" in audit_payload

    frozen = json.loads(ROUTE_A_REPORT.read_text(encoding="utf-8"))
    assert frozen["outcome"] == "transfer_signal_constant"
    assert frozen["control_summary"]["required"] == 1.65
    assert frozen["growth_admitted"] is False
    assert frozen["can_promote"] is False

    p52b = json.loads(P52B_REPORT.read_text(encoding="utf-8"))
    assert audit_payload["p52b_state"]["groups"] == p52b["evaluator"]["groups"] == 0
    assert audit_payload["p52b_state"]["rejected"] == p52b["evaluator"]["rejected"] == 6


def test_hand_calculated_cases_cover_the_required_regimes(dictionary):
    """The five regimes the plan requires, each with its documented formula."""

    documented = {item["case"] for item in dictionary.CASE_DOCUMENTATION}
    implemented = set(dictionary.HAND_CALCULATED_CASES)
    assert documented == implemented
    assert documented == {
        "fully_redundant",
        "coverage_complementary_no_synergy",
        "jointly_required",
        "all_fail",
        "equal_outcome_different_cost",
    }

    for name in implemented:
        payload = dictionary.evaluate_hand_case(name)
        for field in (
            "pair_gain_vs_baseline",
            "pair_gain_vs_pair_internal_oracle",
            "pair_gain_vs_all_singleton_oracle",
            "pair_mean_interaction",
        ):
            assert isinstance(payload[field], float)


def test_reachability_table_accepts_an_explicit_solvable_context_set(dictionary):
    """The hypothetical ceiling must follow the contexts handed to it."""

    table = dictionary.case_jointly_required()
    assert dictionary.hypothetical_block_upper_bound(table, table.contexts) == 2.0
    assert dictionary.hypothetical_block_upper_bound(table, ()) == 0.0

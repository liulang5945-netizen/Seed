"""Contract guard for the WP-5a / B2 selection gate (negative result).

B2 asks the downstream question B1 could not: the selector picks one pair by
``argmax predicted_interaction``, that pair is executed on the frozen candidate
surface through the real contract, and its gain is compared against the frozen
reference and the deployable control.

The measured answer is negative, and that is exactly what has to stay pinned: the
selector picks ``member-b+member-c`` while the only pair that actually gains is
``member-a+member-c``, so the treatment arm produces zero task benefit while every
mechanical gate passes.  A silent edit that turns this green would be a fabricated
result, so the numbers below are asserted rather than described.

The gate itself takes ~31 s (748 real-contract episodes), so this file reads the
committed report; it never re-runs the experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports" / "taiji_b0_b2_selection_20260915.json"
PREREG = REPO / "plans" / "reference" / "M5_WP5A_B2_SELECTION_PREREGISTRATION_FROZEN_20260915.md"
RUNNER = REPO / "scripts" / "training" / "eval_taiji_b0_b2_selection_gate.py"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_preregistration_and_runner_exist(report):
    assert PREREG.is_file()
    assert RUNNER.is_file()
    assert report["preregistration"].endswith(
        "M5_WP5A_B2_SELECTION_PREREGISTRATION_FROZEN_20260915.md"
    )
    assert report["route_b_preregistration"].endswith(
        "M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md"
    )
    assert report["status"] == "completed"
    assert report.get("error") is None


def test_mechanical_gates_pass_and_benefit_gates_fail(report):
    """The whole point: a clean instrument reading zero task benefit."""

    gates = report["gates"]
    assert gates["G1_intervention_reality"] is True
    assert gates["G2_data_isolation"] is True
    assert gates["G3_ranking_stability"] is True
    assert gates["G6_budget_gate"] is True
    assert gates["G4_task_gate_H1"] is False
    assert gates["G5_collaboration_gate_H2"] is False
    assert report["outcome"] == "failed"
    assert report["experiment_passed"] is False
    assert report["outcome_detail"].startswith("no_task_benefit")
    # The frozen three-state list has no label for (G4 fail, G5 fail) with green
    # mechanical gates; the gap is disclosed rather than papered over.
    assert report["preregistration_gap"] is not None
    assert "no label" in report["preregistration_gap"]


def test_selector_picks_the_pair_that_does_not_gain(report):
    """The load-bearing negative finding, in one place."""

    selection = report["selection"]
    assert selection["selected_pair"] == "member-b+member-c"
    assert selection["context_conditioned"] is False
    assert selection["predictions"]["member-b+member-c"] == pytest.approx(0.141802, abs=1e-6)
    # The pair that actually gains is ranked below the selected pair.
    assert selection["predictions"]["member-a+member-c"] == pytest.approx(0.083829, abs=1e-6)
    assert (
        selection["predictions"]["member-a+member-c"]
        < selection["predictions"]["member-b+member-c"]
    )


def test_only_one_pair_gains_and_it_is_not_the_selected_one(report):
    gains = report["pooled"]["all_pair_gains"]
    assert gains["member-a+member-c"] == pytest.approx(2.0)
    for label, value in gains.items():
        if label != "member-a+member-c":
            assert value == pytest.approx(0.0), label
    assert report["pooled"]["selected_pair_gain_vs_all_singleton_oracle"] == pytest.approx(0.0)
    assert report["pooled"]["selected_pair_mean_outcome"] == pytest.approx(-1.0)
    # Every singleton fails on a combination-only surface: the control is a tie.
    assert set(report["pooled"]["singleton_mean_outcomes"].values()) == {-1.0}
    assert report["controls"]["best_fixed_singleton"]["tied_at_floor"] is True


def test_every_cell_is_judged_on_the_frozen_per_cell_criterion(report):
    """Route B section 2 derives required/k per cell (n=6), not on the pooled table."""

    assert len(report["cells"]) == 3
    for cell in report["cells"]:
        assert cell["contexts"] == 6
        assert cell["context_count"] == 6
        assert cell["required_combination_only_contexts"] == 5
        assert cell["available_combination_only_contexts"] == 6
        assert cell["ceiling_gain"] == pytest.approx(2.0)
        assert cell["max_clearable_reference"] == pytest.approx(1.85)
        assert cell["required"] == pytest.approx(1.65)
        assert cell["selected_pair_gain"] == pytest.approx(0.0)
        assert cell["selected_pair_meets_required"] is False
        assert cell["lesion"]["holds"] is False
        assert cell["gain_vs_all_singleton_oracle"]["member-a+member-c"] == pytest.approx(2.0)

    targets = report["estimation_targets"]
    assert targets["primary_reference"] == "all_singleton_oracle"
    assert targets["primary_reference_gain"] == pytest.approx(1.5)
    assert targets["required"] == pytest.approx(1.65)
    assert targets["margin"] == pytest.approx(0.15)
    assert targets["per_cell_required_combination_only_contexts"] == 5
    assert targets["per_cell_available_combination_only_contexts"] == 6
    # The pooled scale differs and is disclosed separately, never used as the criterion.
    assert targets["pooled_context_count"] == 18
    assert targets["pooled_required_combination_only_contexts"] == 15
    assert targets["risk_clause_R_ZS"]


def test_g5_reads_the_selected_pairs_own_interleaving(report):
    """A positive any-pair count must not be mistaken for the treatment arm's."""

    pooled = report["pooled"]
    assert pooled["selected_pair_interleaved_contexts"] == 0
    assert pooled["any_pair_interleaved_contexts"] == 18
    assert "SELECTED pair" in pooled["interleaved_note"]
    for cell in report["cells"]:
        assert cell["selected_pair_interleaved_contexts"] == 0
        assert cell["any_pair_interleaved_contexts"] == 6


def test_no_learning_fallback_and_controls_are_present(report):
    controls = report["controls"]
    assert set(controls) == {
        "selected_pair",
        "no_learning",
        "best_fixed_singleton",
        "best_observed_fixed_pair",
        "random_pair",
        "a_count",
        "additive_train_only",
    }
    assert controls["no_learning"]["pair"] == "member-a+member-b"
    assert controls["best_observed_fixed_pair"]["pair"] == "member-a+member-c"
    assert controls["best_observed_fixed_pair"]["deployable"] is False
    assert controls["best_fixed_singleton"]["deployable"] is True


def test_instrument_integrity_is_asserted_not_assumed(report):
    reality = report["intervention_reality"]
    assert reality["interventions_happened"] is True
    assert reality["zero_step_intervention_episodes"] == 0
    assert report["checkpoint"] == {"restore_matches_parent": True, "tamper_rejected": True}
    assert report["stability"] == {
        "renaming_stable": True,
        "order_permutation_stable": True,
    }
    # Candidate pair outcomes must never enter the fit corpus.
    corpus = report["corpus"]
    assert set(corpus["training_cells"]).isdisjoint(set(corpus["candidate_cells"]))
    assert corpus["episodes"]["candidate"] == 396
    # 8 fit contexts x 11 cells x 2 repeats; the candidate surface is the same shape.
    assert corpus["episodes"]["fit"] == 176
    assert corpus["episodes"]["indomain_holdout"] == 176
    assert corpus["fit_contexts"] == 8
    assert report["cost"]["episodes_total"] == 748


def test_budget_gate_is_measured_against_the_frozen_cap(report):
    assert report["record"]["wall_cap_seconds"] == pytest.approx(60.0)
    assert report["elapsed_seconds"] <= report["record"]["wall_cap_seconds"]


def test_no_collaboration_or_admission_claim_leaks_into_the_report(report):
    assert report["growth_admitted"] is False
    assert report["can_promote"] is False
    boundary = report["boundary"]
    assert "not collaboration evidence" in boundary
    assert "not one observed collaboration event" in boundary
    # The report must not claim per-cell routing: the representation is context-free.
    assert "not per-cell routing" in report["selection"]["context_conditioning_note"]

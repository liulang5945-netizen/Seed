"""Regression guard for the route-B task pre-check instrument.

The B0 audit established *that* the frozen gate is unreachable.  This file pins
*why*, because the reason determines what has to change:

1. **Arbitration cannot help.**  Any composition whose per-context outcome is one
   member's outcome is bounded by ``max(S_i, S_j) <= max_i S_i``.  The ceiling is
   therefore ``<= 0`` by construction, and a positive reading would mean the
   measurement is broken rather than that collaboration was found.
2. **Only combination-only solvable contexts can escape.**  On a context where
   every singleton fails, the oracle equals the blank, so a winning combination
   contributes ``success - B``.  ``k`` such contexts out of ``n`` give a ceiling of
   ``(success - B) * k / n``, which turns the D1 reference choice into a countable
   task-design obligation.
3. **The frozen composition rule is a priority fallback.**  The P5.2b mechanism
   executes the first member (in ``active_members`` order) whose action binds, so a
   pair reproduces one member's trajectory unless a member fails to bind
   mid-episode.  A test asserts that the frozen matrix contains **zero**
   interleaved trajectories, which is why the gate is blocked by the task and the
   mechanism -- not by the representation.
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
DICTIONARY_SCRIPT = REPO / "scripts" / "training" / "audit_taiji_b0_measurement_reachability.py"
PRECHECK_SCRIPT = REPO / "scripts" / "training" / "audit_taiji_b0_task_reachability_precheck.py"
ROUTE_A_REPORT = REPO / "reports" / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
ROUTE_C_REPORT = (
    REPO / "reports" / "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json"
)


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
    return _load("_b0_dictionary_for_precheck_tests", DICTIONARY_SCRIPT)


@pytest.fixture(scope="module")
def precheck():
    return _load("_b0_precheck_under_test", PRECHECK_SCRIPT)


@pytest.fixture(scope="module")
def frozen_table(dictionary):
    """The frozen route-C holdout surface, reconstructed as a context table."""

    route_c = json.loads(ROUTE_C_REPORT.read_text(encoding="utf-8"))
    success_matrix = route_c["success_matrix"]
    block_contexts = {block: [contexts[-1]] for block, contexts in success_matrix["blocks"].items()}
    return dictionary.table_from_success_matrix(success_matrix, block_contexts)


@pytest.fixture(scope="module")
def frozen_candidates(dictionary, frozen_table) -> Mapping[str, float]:
    route_a = json.loads(ROUTE_A_REPORT.read_text(encoding="utf-8"))
    return {
        "all_singleton_oracle": dictionary.oracle_all_singleton_gain(frozen_table),
        "best_fixed_singleton": dictionary.best_deployable_singleton(frozen_table)[1],
        "best_observed_fixed_pair": dictionary.best_deployable_pair(
            frozen_table, [tuple(pair) for pair in route_a["design"]["observed_pairs"]]
        )[1],
    }


def _three_member_table(dictionary, *, interleaved: bool):
    """Three members; the pair either ties its own best member or beats it."""

    contexts = ("c0", "c1")
    singletons = {
        "member-a": {"c0": -1.0, "c1": -1.0},
        "member-b": {"c0": -1.0, "c1": -1.0},
        "member-c": {"c0": 1.0, "c1": 1.0},
    }
    # c0 is combination-only solvable (every singleton fails).  When
    # ``interleaved`` the pair genuinely succeeds there; otherwise it fails too.
    pair = {"c0": 1.0 if interleaved else -1.0, "c1": -1.0}
    return dictionary.ContextTable(
        contexts=contexts,
        baseline={"c0": -1.0, "c1": -1.0},
        singletons=singletons,
        combinations={("member-a", "member-b"): pair},
    )


# --------------------------------------------------------------------------- #
# 1. The arbitration ceiling is provable, not empirical
# --------------------------------------------------------------------------- #


def test_arbitration_ceiling_is_never_positive_on_the_frozen_matrix(precheck, frozen_table):
    """``max(S_i, S_j) <= max_i S_i`` -- no arbitration mechanism can beat the oracle."""

    for pair in frozen_table.pair_cells():
        ceiling = precheck.arbitration_ceiling(frozen_table, pair)
        assert ceiling <= 0.0, f"{pair} reported a positive arbitration ceiling"

    assert precheck.arbitration_ceiling(frozen_table, ("member-a", "member-d")) == -1.0
    assert precheck.arbitration_ceiling(frozen_table, ("member-b", "member-c")) == -0.5


def test_interleaving_alone_is_not_enough_when_a_singleton_covers_the_context(precheck, dictionary):
    """Beating your own members is worthless if a third member already solves it."""

    table = _three_member_table(dictionary, interleaved=True)
    pair = ("member-a", "member-b")

    # On c0 the pair beats both of its members -- a genuine interleaved trajectory.
    assert precheck.arbitration_headroom(table, pair) < 0.0
    assert precheck.classify_pair_trajectory(table, pair)["contexts_interleaved"] == 1

    # But member-c succeeds on c0, so the all-singleton oracle still dominates:
    # the arbitration ceiling is strictly negative and there is nothing to claim.
    assert precheck.arbitration_ceiling(table, pair) == -2.0
    assert precheck.combination_only_solvable_contexts(table) == []
    assert precheck.combination_only_gain_potential(table) == 0.0


def test_interleaving_on_an_uncovered_context_is_the_escape_route(precheck, dictionary):
    """Drop the third member and the same interleaving becomes claimable."""

    contexts = ("c0", "c1")
    table = dictionary.ContextTable(
        contexts=contexts,
        baseline={"c0": -1.0, "c1": -1.0},
        singletons={
            # c0: both members fail -> combination-only solvable.
            # c1: member-b already succeeds -> no room for a collaboration claim.
            "member-a": {"c0": -1.0, "c1": -1.0},
            "member-b": {"c0": -1.0, "c1": 1.0},
        },
        combinations={("member-a", "member-b"): {"c0": 1.0, "c1": -1.0}},
    )

    assert precheck.combination_only_solvable_contexts(table) == ["c0"]
    # (success - B) on one of two contexts: the ceiling the task design must raise.
    assert precheck.combination_only_gain_potential(table) == 1.0
    assert precheck.arbitration_ceiling(table, ("member-a", "member-b")) == 0.0


def test_frozen_rule_headroom_separates_the_fixable_gap(precheck, frozen_table):
    """Half of the pairs lose one context to the frozen order; the rest lose none."""

    headroom = {
        "+".join(pair): precheck.arbitration_headroom(frozen_table, pair)
        for pair in frozen_table.pair_cells()
    }

    assert headroom == {
        "member-a+member-b": 0.5,
        "member-a+member-c": 0.5,
        "member-a+member-d": 0.0,
        "member-b+member-c": 0.0,
        "member-b+member-d": 0.0,
        "member-c+member-d": 0.5,
    }
    # The recoverable part is bounded well below every candidate threshold, so
    # fixing the order alone cannot clear the gate either.
    assert max(headroom.values()) == 0.5


# --------------------------------------------------------------------------- #
# 2. The escape route, in closed form
# --------------------------------------------------------------------------- #


def test_required_combination_only_contexts_matches_hand_computation(precheck):
    """``k > n * (reference + margin) / (success - B)``, rounded up."""

    cases = [
        # reference, margin, n, expected k
        (1.5, 0.15, 4, 4),  # 4 * 1.65 / 2 = 3.3  -> k = 4
        (1.0, 0.15, 4, 3),  # 4 * 1.15 / 2 = 2.3  -> k = 3
        (0.5, 0.15, 4, 2),  # 4 * 0.65 / 2 = 1.3  -> k = 2
        (0.0, 0.15, 4, 1),  # 4 * 0.15 / 2 = 0.3  -> k = 1
        (0.85, 0.15, 4, 3),  # exactly at the boundary: not strictly greater
        (0.84, 0.15, 4, 2),
    ]
    for reference, margin, n, expected in cases:
        assert (
            precheck.required_combination_only_contexts(
                reference_gain=reference, margin=margin, context_count=n
            )
            == expected
        ), (reference, margin, n)


def test_required_combination_only_contexts_reports_infeasible_above_the_ceiling(precheck):
    """When even ``k == n`` is not enough, the reference itself is the problem."""

    n = 4
    # (success - B) * n / n = 2.0 is the absolute ceiling of the mean gain.
    assert (
        precheck.required_combination_only_contexts(
            reference_gain=1.85, margin=0.15, context_count=n
        )
        == n + 1
    )
    assert (
        precheck.required_combination_only_contexts(
            reference_gain=1.84, margin=0.15, context_count=n
        )
        == n
    )


def test_max_clearable_reference_inverts_the_requirement(precheck):
    """The reported maximum is a supremum: the boundary case needs one more context."""

    supremum = precheck.max_clearable_reference(
        combination_only_contexts=2, context_count=4, margin=0.15
    )
    assert supremum == 0.85

    # Just below it the task shape suffices ...
    assert (
        precheck.required_combination_only_contexts(
            reference_gain=0.84, margin=0.15, context_count=4
        )
        == 2
    )
    # ... at the supremum it does not, because the comparison is strict.
    assert (
        precheck.required_combination_only_contexts(
            reference_gain=supremum, margin=0.15, context_count=4
        )
        == 3
    )


def test_combination_only_potential_uses_the_blank_reference(precheck, frozen_table):
    """The one context where every singleton fails is block 3, worth +2 of 4."""

    assert precheck.combination_only_solvable_contexts(frozen_table) == ["p52a-validation-111"]
    assert precheck.combination_only_gain_potential(frozen_table) == 0.5


# --------------------------------------------------------------------------- #
# 3. The frozen composition rule produced no interleaving at all
# --------------------------------------------------------------------------- #


def test_frozen_matrix_contains_zero_interleaved_trajectories(precheck, frozen_table):
    """Every pair reproduced some single member's outcome on every context."""

    for pair in frozen_table.pair_cells():
        trajectory = precheck.classify_pair_trajectory(frozen_table, pair)
        assert trajectory["contexts_interleaved"] == 0
        assert set(trajectory["kind_counts"]) <= {
            "both_members",
            "first_member_only",
            "second_member_only",
        }

    payload = precheck.precheck(
        frozen_table,
        margin=0.15,
        candidates={
            "all_singleton_oracle": 1.5,
            "best_fixed_singleton": 0.5,
            "best_observed_fixed_pair": 1.0,
        },
    )
    assert payload["verdict"]["contexts_interleaved"] == 0
    assert payload["verdict"]["arbitration_is_the_bottleneck"] is True


def test_three_pairs_already_reach_the_within_pair_optimum(precheck, frozen_table):
    """Routing is not the bottleneck: half the pairs hit max(S_i, S_j) everywhere."""

    at_optimum = {
        "+".join(pair): precheck.classify_pair_trajectory(frozen_table, pair)[
            "contexts_reaching_arbitration_optimum"
        ]
        for pair in frozen_table.pair_cells()
    }

    assert at_optimum == {
        "member-a+member-b": 3,
        "member-a+member-c": 3,
        "member-a+member-d": 4,
        "member-b+member-c": 4,
        "member-b+member-d": 4,
        "member-c+member-d": 3,
    }


def test_interleaved_trajectory_is_recognized_when_it_exists(precheck, dictionary):
    """A pair that beats both members on a context is classified as interleaved."""

    table = _three_member_table(dictionary, interleaved=True)
    trajectory = precheck.classify_pair_trajectory(table, ("member-a", "member-b"))

    assert trajectory["contexts_interleaved"] == 1
    assert trajectory["kind_counts"]["interleaved"] == 1
    assert trajectory["per_context"][0]["kind"] == "interleaved"
    assert trajectory["per_context"][1]["kind"] == "both_members"


# --------------------------------------------------------------------------- #
# 4. Reference requirements -- D1 as countable obligations
# --------------------------------------------------------------------------- #


def test_reference_requirements_on_the_frozen_matrix(precheck, frozen_table, frozen_candidates):
    """All three candidate references are infeasible with one available context."""

    rows = {
        row["reference"]: row
        for row in precheck.reference_requirements(
            frozen_table, margin=0.15, candidates=frozen_candidates
        )
    }

    assert frozen_candidates == {
        "all_singleton_oracle": 1.5,
        "best_fixed_singleton": 0.5,
        "best_observed_fixed_pair": 1.0,
    }
    assert rows["all_singleton_oracle"]["required_combination_only_contexts"] == 4
    assert rows["best_observed_fixed_pair"]["required_combination_only_contexts"] == 3
    assert rows["best_fixed_singleton"]["required_combination_only_contexts"] == 2

    for row in rows.values():
        assert row["available_combination_only_contexts"] == 1
        assert row["context_count"] == 4
        assert row["feasible"] is False
        # One available context of four: (success - B) * 1 / 4 = 0.5, less margin.
        assert row["max_clearable_reference"] == 0.35
        assert row["ceiling_gain"] == 0.5


def test_precheck_verdict_names_the_blocking_layer(precheck, frozen_table, frozen_candidates):
    """The verdict must blame the task and the mechanism, not the representation."""

    payload = precheck.precheck(frozen_table, margin=0.15, candidates=frozen_candidates)

    assert payload["status"] == "draft_for_review"
    assert payload["composition_rule"]["id"] == precheck.COMPOSITION_RULE_ID
    assert payload["verdict"]["arbitration_is_the_bottleneck"] is True
    assert "not by the representation" in payload["verdict"]["reading"]
    assert payload["context_count"] == 4
    assert payload["margin"] == 0.15


def test_required_combination_only_contexts_rejects_degenerate_inputs(precheck):
    """A zero-width outcome band would make the requirement undefined."""

    with pytest.raises(ValueError):
        precheck.required_combination_only_contexts(
            reference_gain=0.5,
            margin=0.15,
            context_count=4,
            success_outcome=1.0,
            blank_outcome=1.0,
        )
    with pytest.raises(ValueError):
        precheck.required_combination_only_contexts(
            reference_gain=0.5, margin=0.15, context_count=0
        )


def test_precheck_module_stays_read_only(precheck: Any) -> None:
    """The instrument must not carry a training or checkpoint-write surface."""

    source = PRECHECK_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("torch.save", "checkpoint(", "optimizer", "save_file", "shutil.rmtree"):
        assert forbidden not in source, f"pre-check must not contain {forbidden!r}"

    frozen = json.loads(ROUTE_A_REPORT.read_text(encoding="utf-8"))
    assert frozen["outcome"] == "transfer_signal_constant"
    assert frozen["growth_admitted"] is False
    assert frozen["can_promote"] is False

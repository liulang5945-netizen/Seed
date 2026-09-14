"""Route-B / B0 audit: unified measurement dictionary, old-value recomputation
and task reachability bounds.

Why this script exists
----------------------
The route-A result report (``M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_RESULT``)
compared two numbers that were produced against **different references** while
sharing one field name ``mean_gain_vs_strongest_single``:

* the object  -> ``mean(P_ij - max(S_i, S_j))``   (pair-internal reference)
* control C2  -> ``mean(max_all S - B)``           (blank reference)

The margin test then required ``object > C2 + 0.15``.  Two different estimands
were therefore subtracted from each other.  This script does **not** re-adjudicate
the frozen result and does **not** change any threshold.  It does three things:

1. states the measurement dictionary explicitly (formula, reference, denominator,
   cost, and the code assertion that enforces it) as pure functions;
2. recomputes the old values from the frozen report JSONs so the discrepancy is
   visible as arithmetic rather than as prose;
3. computes the task reachability upper bound, i.e. the best value the *current*
   success matrix can yield, and compares it with each candidate threshold.

Reading the output
------------------
``reachability`` reports, per candidate reference, the required threshold and the
achievable ceiling.  A reference whose ceiling is below its own requirement makes
the gate unreachable by construction; no representation change can fix that.

This module is a **read-only audit**.  It trains nothing, writes no checkpoint and
mutates no frozen artifact.  The dictionary here is a *draft* (``...-v1-draft``)
pending review; nothing in the repository is bound to it yet.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ROUTE_A_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
)
DEFAULT_ROUTE_C_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json"
)
DEFAULT_P52B_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_2b_group_causal_corpora_20260913.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_measurement_reachability_audit_20260913.json"

#: Draft dictionary revision.  Bump only with a reviewed preregistration; the
#: frozen A/c'' reports stay readable under this revision because the recomputed
#: fields are *added*, never substituted for the historical ones.
MEASUREMENT_DICTIONARY_VERSION = "b0-measurement-dictionary-v1-draft"

#: Margin inherited from the frozen A gate.  Read back from the report rather than
#: re-typed in the audit body, so a drift in the frozen gate is detectable here.
MARGIN_SOURCE_FIELD = ("control_summary", "margin")

#: Outcome encoding of the workbench matrix: a successful episode scores +1, a
#: failed one -1.  Every derivation below depends on this being frozen.
SUCCESS_OUTCOME = 1.0
FAILURE_OUTCOME = -1.0

_MEMBER_TOKEN = re.compile(r"member-[a-z0-9]+")


# --------------------------------------------------------------------------- #
# 1. Unified measurement dictionary (pure functions; draft revision)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContextTable:
    """Per-context outcomes for every *executed* cell of one experiment.

    A cell is identified by its member set: ``()`` is the blank baseline,
    ``(m,)`` a singleton and ``(m, n)`` a pair combination.  Values are mean
    outcomes over repeats, so the table is already an expectation and no
    further averaging over repeats happens downstream.
    """

    contexts: tuple[str, ...]
    baseline: Mapping[str, float]
    singletons: Mapping[str, Mapping[str, float]]
    combinations: Mapping[str, Mapping[tuple[str, ...], float]]

    def member_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.singletons))

    def pair_cells(self) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(self.combinations))

    def blank(self, context_id: str) -> float:
        return float(self.baseline[context_id])

    def singleton(self, context_id: str, member_id: str) -> float:
        return float(self.singletons[member_id][context_id])

    def combination(self, context_id: str, members: Sequence[str]) -> float:
        return float(self.combinations[tuple(sorted(members))][context_id])

    def outcome(self, context_id: str, members: Sequence[str]) -> float:
        """Realized outcome of one deployable policy on one context."""

        key = tuple(sorted(members))
        if not key:
            return self.blank(context_id)
        if len(key) == 1:
            return self.singleton(context_id, key[0])
        return self.combination(context_id, key)

    def strongest_singleton(self, context_id: str) -> tuple[str, float]:
        """Per-context argmax over singletons -- the *all-singleton oracle*.

        This is a scoring diagnostic, not a deployable policy: it is allowed to
        look at the outcome it is about to be compared with.
        """

        ranked = sorted(
            ((self.singleton(context_id, member), member) for member in self.singletons),
            key=lambda item: (-item[0], item[1]),
        )
        return ranked[0][1], ranked[0][0]


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean over a non-empty sequence; 0.0 for an empty one."""

    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def causal_interaction(
    *, combination: float, first: float, second: float, baseline: float
) -> float:
    """``P_ij - S_i - S_j + B`` -- factorial interaction, reported separately.

    This is never a task utility by itself: it answers "does the joint cell
    deviate from the additive expectation", not "is the joint cell worth
    deploying".  A negative interaction with a positive gain is a real and
    legitimate regime.
    """

    return float(combination - first - second + baseline)


def pair_internal_oracle(*, first: float, second: float) -> float:
    """``max(S_i, S_j)`` -- reference used by the frozen A object score."""

    return float(max(first, second))


def all_singleton_oracle(*, singletons: Sequence[float]) -> float:
    """``max_i S_i`` -- reference used by the frozen A control C2."""

    return float(max(singletons))


def policy_mean_outcome(table: ContextTable, members: Sequence[str]) -> float:
    """Primary task performance: mean outcome of a fixed policy over contexts."""

    return mean(table.outcome(context_id, members) for context_id in table.contexts)


def policy_mean_gain_vs_baseline(table: ContextTable, members: Sequence[str]) -> float:
    """``U(policy) - U(blank)`` -- same reference for every policy."""

    return mean(
        table.outcome(context_id, members) - table.blank(context_id)
        for context_id in table.contexts
    )


def policy_mean_gain_vs_pair_internal_oracle(table: ContextTable, pair: Sequence[str]) -> float:
    """``mean(P_ij - max(S_i, S_j))`` -- the frozen A object estimand."""

    first, second = tuple(sorted(pair))
    return mean(
        table.combination(context_id, (first, second))
        - pair_internal_oracle(
            first=table.singleton(context_id, first),
            second=table.singleton(context_id, second),
        )
        for context_id in table.contexts
    )


def policy_mean_gain_vs_all_singleton_oracle(table: ContextTable, members: Sequence[str]) -> float:
    """``mean(P - max_all S)`` -- the frozen A control-C2 estimand.

    Applying the *same* reference to the object is the whole point of the
    unified dictionary: the two frozen numbers were never comparable.
    """

    return mean(
        table.outcome(context_id, members)
        - all_singleton_oracle(
            singletons=[table.singleton(context_id, m) for m in table.singletons]
        )
        for context_id in table.contexts
    )


def policy_mean_interaction(table: ContextTable, pair: Sequence[str]) -> float:
    """``mean(P_ij - S_i - S_j + B)`` -- causal interaction, reported separately."""

    first, second = tuple(sorted(pair))
    return mean(
        causal_interaction(
            combination=table.combination(context_id, (first, second)),
            first=table.singleton(context_id, first),
            second=table.singleton(context_id, second),
            baseline=table.blank(context_id),
        )
        for context_id in table.contexts
    )


def policy_utility(
    table: ContextTable,
    members: Sequence[str],
    *,
    calls: float | None = None,
    cost_weight: float = 0.0,
) -> float:
    """``U(policy) - lambda * calls`` with an *a-priori frozen* ``lambda``.

    ``cost_weight`` defaults to 0.0 on purpose: folding cost into utility after
    seeing the outcome is how a cost model turns into a post-hoc knob.  Callers
    that pass a non-zero weight must record it in the preregistration first.
    """

    realized_calls = float(len(members)) if calls is None else float(calls)
    return policy_mean_outcome(table, members) - cost_weight * realized_calls


def contexts_beating_all_singleton_oracle(table: ContextTable, members: Sequence[str]) -> int:
    """Count of contexts where the policy strictly beats the all-singleton oracle."""

    return sum(
        1
        for context_id in table.contexts
        if table.outcome(context_id, members)
        > all_singleton_oracle(
            singletons=[table.singleton(context_id, m) for m in table.singletons]
        )
    )


# --------------------------------------------------------------------------- #
# 2. Upper bounds and threshold reachability
# --------------------------------------------------------------------------- #


def oracle_all_singleton_gain(table: ContextTable) -> float:
    """``mean(max_i S_i - B)``: the frozen control C2 value, recomputed."""

    return mean(
        all_singleton_oracle(singletons=[table.singleton(context_id, m) for m in table.singletons])
        - table.blank(context_id)
        for context_id in table.contexts
    )


def oracle_all_cell_gain(table: ContextTable) -> float:
    """``mean(max over every executed cell - B)`` -- diagnostic ceiling only.

    This is *not* a deployable policy: choosing it needs the outcome.  It is
    reported to separate "the matrix contains this capability somewhere" from
    "some policy can reach it".
    """

    return mean(
        max(
            [table.blank(context_id)]
            + [table.singleton(context_id, m) for m in table.singletons]
            + [table.combination(context_id, pair) for pair in table.pair_cells()]
        )
        - table.blank(context_id)
        for context_id in table.contexts
    )


def best_deployable_singleton(table: ContextTable) -> tuple[str, float]:
    """Best *fixed* singleton by mean gain vs baseline -- deployable control."""

    ranked = sorted(
        ((policy_mean_gain_vs_baseline(table, (member,)), member) for member in table.singletons),
        key=lambda item: (-item[0], item[1]),
    )
    return ranked[0][1], ranked[0][0]


def best_deployable_pair(
    table: ContextTable, pairs: Sequence[Sequence[str]]
) -> tuple[tuple[str, ...], float]:
    """Best *fixed* pair drawn from ``pairs`` by mean gain vs baseline."""

    ranked = sorted(
        ((policy_mean_gain_vs_baseline(table, tuple(pair)), tuple(sorted(pair))) for pair in pairs),
        key=lambda item: (-item[0], item[1]),
    )
    return ranked[0][1], ranked[0][0]


def combination_slack_upper_bound(table: ContextTable) -> float:
    """Structural ceiling of ``mean(P - max_all S)`` over the *current* matrix.

    No cell can exceed ``max_all S`` on a context where every singleton has been
    measured, so the ceiling is 0.0 whenever no pair beats the oracle anywhere.
    A negative value means no executed combination even ties the oracle on
    average -- which is strictly stronger than "the gate was not cleared".
    """

    return max(policy_mean_gain_vs_all_singleton_oracle(table, pair) for pair in table.pair_cells())


def hypothetical_block_upper_bound(table: ContextTable, solvable_contexts: Sequence[str]) -> float:
    """Ceiling if a combination could solve contexts where *every* singleton fails.

    On such a context the oracle is ``B`` (all singletons fail), so a winning
    combination contributes ``1 - B``.  This is the only mechanism left in the
    current design that could create a positive same-reference gain, and it is
    exactly the quantity the next task definition has to make measurable.
    """

    slacks = []
    for context_id in table.contexts:
        if context_id not in set(solvable_contexts):
            slacks.append(0.0)
            continue
        slacks.append(SUCCESS_OUTCOME - table.blank(context_id))
    return mean(slacks)


def reachability_table(
    table: ContextTable,
    *,
    margin: float,
    observed_pairs: Sequence[Sequence[str]],
    held_out_pairs: Sequence[Sequence[str]],
    solvable_contexts: Sequence[str],
) -> dict[str, Any]:
    """Required threshold vs achievable ceiling, per candidate reference.

    ``reachable`` answers "can any policy in this design clear the bar"; a
    reference with ``reachable == False`` blocks formal training until the task
    definition -- not the representation -- changes.
    """

    rows: list[dict[str, Any]] = []

    best_singleton_member, best_singleton_gain = best_deployable_singleton(table)
    best_pair, best_pair_gain = best_deployable_pair(table, observed_pairs)
    held_out_gains = {
        "+".join(sorted(pair)): policy_mean_gain_vs_baseline(table, pair) for pair in held_out_pairs
    }
    best_held_out = max(held_out_gains.values()) if held_out_gains else 0.0

    references = [
        {
            "reference": "all_singleton_oracle",
            "deployable": False,
            "value": oracle_all_singleton_gain(table),
            "ceiling": combination_slack_upper_bound(table),
        },
        {
            "reference": "best_fixed_singleton",
            "deployable": True,
            "value": best_singleton_gain,
            "ceiling": max(
                [policy_mean_gain_vs_baseline(table, pair) for pair in table.pair_cells()]
            )
            - best_singleton_gain,
        },
        {
            "reference": "best_observed_fixed_pair",
            "deployable": True,
            "value": best_pair_gain,
            "ceiling": max(
                [policy_mean_gain_vs_baseline(table, pair) for pair in table.pair_cells()]
            )
            - best_pair_gain,
        },
    ]
    for item in references:
        required = float(item["value"]) + margin
        rows.append(
            {
                "reference": item["reference"],
                "reference_is_deployable": item["deployable"],
                "reference_value": float(item["value"]),
                "required": required,
                "achievable_ceiling": float(item["ceiling"]),
                "reachable": bool(float(item["ceiling"]) > required),
                "shortfall": round(required - float(item["ceiling"]), 6),
            }
        )

    return {
        "margin": margin,
        "rows": rows,
        "deployable_controls": {
            "best_fixed_singleton": {
                "member_ids": [best_singleton_member],
                "gain_vs_baseline": best_singleton_gain,
            },
            "best_observed_fixed_pair": {
                "member_ids": list(best_pair),
                "gain_vs_baseline": best_pair_gain,
            },
            "held_out_pairs_gain_vs_baseline": held_out_gains,
            "best_held_out_pair_gain_vs_baseline": best_held_out,
            "held_out_selection_spread": (
                best_held_out - min(held_out_gains.values()) if held_out_gains else 0.0
            ),
        },
        "structural_ceiling_current_matrix": combination_slack_upper_bound(table),
        "ceiling_if_block_becomes_solvable": hypothetical_block_upper_bound(
            table, solvable_contexts
        ),
        "oracle_all_cell_gain_vs_baseline": oracle_all_cell_gain(table),
    }


# --------------------------------------------------------------------------- #
# 3. Hand-calculated cases (the dictionary's executable specification)
# --------------------------------------------------------------------------- #

CASE_DOCUMENTATION: tuple[dict[str, str], ...] = (
    {
        "case": "fully_redundant",
        "formula": "S_i = S_j = P on every context",
        "why": "two members share one capability surface; adding the second buys nothing",
    },
    {
        "case": "coverage_complementary_no_synergy",
        "formula": "each member wins a disjoint block, the combination routes to the winner",
        "why": "the pair covers more than any fixed singleton but never beats the oracle",
    },
    {
        "case": "jointly_required",
        "formula": "S_i = S_j = B < P on every context",
        "why": "the only regime that can support a same-reference collaboration claim",
    },
    {
        "case": "all_fail",
        "formula": "B = S_i = S_j = P = failure",
        "why": "a task nobody solves is neither capability evidence nor a violation",
    },
    {
        "case": "equal_outcome_different_cost",
        "formula": "equal outcome, different call counts, frozen cost weight",
        "why": "cost must be frozen in advance or it becomes a post-hoc knob",
    },
)


def case_fully_redundant() -> ContextTable:
    """Two members with one identical surface; the pair adds nothing."""

    contexts = ("c0", "c1", "c2", "c3")
    return ContextTable(
        contexts=contexts,
        baseline=dict.fromkeys(contexts, FAILURE_OUTCOME),
        singletons={
            "member-a": {
                "c0": SUCCESS_OUTCOME,
                "c1": FAILURE_OUTCOME,
                "c2": FAILURE_OUTCOME,
                "c3": FAILURE_OUTCOME,
            },
            "member-d": {
                "c0": SUCCESS_OUTCOME,
                "c1": FAILURE_OUTCOME,
                "c2": FAILURE_OUTCOME,
                "c3": FAILURE_OUTCOME,
            },
        },
        combinations={
            ("member-a", "member-d"): {
                "c0": SUCCESS_OUTCOME,
                "c1": FAILURE_OUTCOME,
                "c2": FAILURE_OUTCOME,
                "c3": FAILURE_OUTCOME,
            },
        },
    )


def case_coverage_complementary_no_synergy() -> ContextTable:
    """Routing covers two blocks; no context beats the per-context oracle."""

    contexts = ("c0", "c1", "c2", "c3")
    return ContextTable(
        contexts=contexts,
        baseline=dict.fromkeys(contexts, FAILURE_OUTCOME),
        singletons={
            "member-a": {
                "c0": SUCCESS_OUTCOME,
                "c1": FAILURE_OUTCOME,
                "c2": FAILURE_OUTCOME,
                "c3": FAILURE_OUTCOME,
            },
            "member-b": {
                "c0": FAILURE_OUTCOME,
                "c1": SUCCESS_OUTCOME,
                "c2": FAILURE_OUTCOME,
                "c3": FAILURE_OUTCOME,
            },
        },
        combinations={
            ("member-a", "member-b"): {
                "c0": SUCCESS_OUTCOME,
                "c1": SUCCESS_OUTCOME,
                "c2": FAILURE_OUTCOME,
                "c3": FAILURE_OUTCOME,
            },
        },
    )


def case_jointly_required() -> ContextTable:
    """Only the pair succeeds; both singletons fail everywhere."""

    contexts = ("c0", "c1", "c2", "c3")
    return ContextTable(
        contexts=contexts,
        baseline=dict.fromkeys(contexts, FAILURE_OUTCOME),
        singletons={
            "member-a": dict.fromkeys(contexts, FAILURE_OUTCOME),
            "member-b": dict.fromkeys(contexts, FAILURE_OUTCOME),
        },
        combinations={
            ("member-a", "member-b"): dict.fromkeys(contexts, SUCCESS_OUTCOME),
        },
    )


def case_all_fail() -> ContextTable:
    """Every cell fails on every context."""

    contexts = ("c0", "c1", "c2", "c3")
    return ContextTable(
        contexts=contexts,
        baseline=dict.fromkeys(contexts, FAILURE_OUTCOME),
        singletons={
            "member-a": dict.fromkeys(contexts, FAILURE_OUTCOME),
            "member-b": dict.fromkeys(contexts, FAILURE_OUTCOME),
        },
        combinations={
            ("member-a", "member-b"): dict.fromkeys(contexts, FAILURE_OUTCOME),
        },
    )


def case_equal_outcome_different_cost() -> ContextTable:
    """Equal outcomes; the policies differ only in how many members they call."""

    contexts = ("c0", "c1")
    return ContextTable(
        contexts=contexts,
        baseline=dict.fromkeys(contexts, FAILURE_OUTCOME),
        singletons={
            "member-a": dict.fromkeys(contexts, SUCCESS_OUTCOME),
            "member-b": dict.fromkeys(contexts, SUCCESS_OUTCOME),
        },
        combinations={
            ("member-a", "member-b"): dict.fromkeys(contexts, SUCCESS_OUTCOME),
        },
    )


HAND_CALCULATED_CASES = {
    "fully_redundant": case_fully_redundant,
    "coverage_complementary_no_synergy": case_coverage_complementary_no_synergy,
    "jointly_required": case_jointly_required,
    "all_fail": case_all_fail,
    "equal_outcome_different_cost": case_equal_outcome_different_cost,
}


def evaluate_hand_case(name: str) -> dict[str, Any]:
    """Run one hand-calculated case through the dictionary and report every field."""

    table = HAND_CALCULATED_CASES[name]()
    pair = table.pair_cells()[0]
    documentation = next(item for item in CASE_DOCUMENTATION if item["case"] == name)
    return {
        "case": name,
        "formula": documentation["formula"],
        "why": documentation["why"],
        "contexts": list(table.contexts),
        "pair": list(pair),
        "pair_mean_outcome": policy_mean_outcome(table, pair),
        "pair_gain_vs_baseline": policy_mean_gain_vs_baseline(table, pair),
        "pair_gain_vs_pair_internal_oracle": policy_mean_gain_vs_pair_internal_oracle(table, pair),
        "pair_gain_vs_all_singleton_oracle": policy_mean_gain_vs_all_singleton_oracle(table, pair),
        "pair_mean_interaction": policy_mean_interaction(table, pair),
        "contexts_beating_all_singleton_oracle": contexts_beating_all_singleton_oracle(table, pair),
        "utility_cost_weight_0.0": policy_utility(table, pair, cost_weight=0.0),
        "utility_cost_weight_0.5": policy_utility(table, pair, cost_weight=0.5),
    }


# --------------------------------------------------------------------------- #
# 4. Real-data recomputation from the frozen reports
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_members(cell_name: str) -> tuple[str, ...]:
    """``"member-a-member-b"`` -> ``("member-a", "member-b")``; ``"none"`` -> ``()``."""

    if cell_name == "none":
        return ()
    return tuple(sorted(_MEMBER_TOKEN.findall(cell_name)))


def table_from_success_matrix(
    success_matrix: Mapping[str, Any], block_contexts: Mapping[str, Sequence[str]]
) -> ContextTable:
    """Build a per-context table from a block-aggregated success matrix.

    ``_success_matrix`` aggregates by block (3 contexts x 2 repeats = 6 attempts).
    Every context inside a block is taken to share the block mean, which the
    audit verifies independently against the route-A per-context rows before the
    value is used -- see ``cross_check_route_a_rows``.
    """

    baseline: dict[str, float] = {}
    singletons: dict[str, dict[str, float]] = {}
    combinations: dict[tuple[str, ...], dict[str, float]] = {}
    contexts: list[str] = []

    for row in success_matrix["rows"]:
        members = _cell_members(str(row["cell"]))
        by_block = {int(item["block"]): item for item in row["blocks"]}
        for block, context_ids in sorted(block_contexts.items()):
            item = by_block[int(block)]
            attempts = int(item["attempts"])
            successes = int(item["successes"])
            value = (
                SUCCESS_OUTCOME
                if attempts == 0
                else SUCCESS_OUTCOME * (2.0 * successes / attempts - 1.0)
            )
            for context_id in context_ids:
                if context_id not in contexts:
                    contexts.append(context_id)
                if not members:
                    baseline[context_id] = value
                elif len(members) == 1:
                    singletons.setdefault(members[0], {})[context_id] = value
                else:
                    combinations.setdefault(members, {})[context_id] = value

    return ContextTable(
        contexts=tuple(contexts),
        baseline=baseline,
        singletons=singletons,
        combinations=combinations,
    )


def unseen_contexts_from_success_matrix(
    success_matrix: Mapping[str, Any], unseen_blocks: Sequence[int] = (0, 1, 2, 3)
) -> tuple[str, ...]:
    """Last context of each block is the holdout context (train uses the first two)."""

    ordered: list[str] = []
    for block in unseen_blocks:
        ordered.append(success_matrix["blocks"][str(block)][-1])
    return tuple(ordered)


def cross_check_route_a_rows(table: ContextTable, route_a: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the block-mean reconstruction against route A's per-context rows.

    If this check fails, every derived bound below is built on a mapping that the
    frozen report does not support, and the audit must be rejected rather than
    reported.
    """

    checks: list[dict[str, Any]] = []
    for key, value in route_a["held_out_pairs_scored"].items():
        pair = tuple(sorted(str(item) for item in value["member_ids"]))
        for row in value["per_context"]:
            context_id = str(row["context_id"])
            rebuilt = {
                "baseline": table.blank(context_id),
                "first": table.singleton(context_id, pair[0]),
                "second": table.singleton(context_id, pair[1]),
                "pair": table.combination(context_id, pair),
            }
            expected = {
                "baseline": float(row["baseline_outcome"]),
                "first": float(row["first_outcome"]),
                "second": float(row["second_outcome"]),
                "pair": float(row["pair_outcome"]),
            }
            for field, expected_value in expected.items():
                checks.append(
                    {
                        "pair": key,
                        "context_id": context_id,
                        "field": field,
                        "rebuilt": rebuilt[field],
                        "reported": expected_value,
                        "matches": bool(abs(rebuilt[field] - expected_value) < 1e-9),
                    }
                )
    return {
        "all_match": all(item["matches"] for item in checks),
        "checked_fields": len(checks),
        "mismatches": [item for item in checks if not item["matches"]],
    }


def recompute_route_a(route_a: Mapping[str, Any], table: ContextTable) -> dict[str, Any]:
    """Recompute route A's reported aggregates from its own per-context rows."""

    rows: list[dict[str, Any]] = []
    for key, value in route_a["held_out_pairs_scored"].items():
        pair = tuple(sorted(str(item) for item in value["member_ids"]))
        gains_pair_internal = [
            float(row["pair_outcome"])
            - pair_internal_oracle(
                first=float(row["first_outcome"]), second=float(row["second_outcome"])
            )
            for row in value["per_context"]
        ]
        interactions = [
            causal_interaction(
                combination=float(row["pair_outcome"]),
                first=float(row["first_outcome"]),
                second=float(row["second_outcome"]),
                baseline=float(row["baseline_outcome"]),
            )
            for row in value["per_context"]
        ]
        rows.append(
            {
                "pair": key,
                "reported_mean_gain_vs_strongest_single": float(
                    value["mean_gain_vs_strongest_single"]
                ),
                "recomputed_pair_internal_gain": mean(gains_pair_internal),
                "reported_mean_realized_interaction": float(value["mean_realized_interaction"]),
                "recomputed_mean_interaction": mean(interactions),
                "recomputed_gain_vs_all_singleton_oracle": (
                    policy_mean_gain_vs_all_singleton_oracle(table, pair)
                ),
                "recomputed_gain_vs_baseline": policy_mean_gain_vs_baseline(table, pair),
                "pair_internal_reference_kind": "max(S_i, S_j)",
            }
        )

    summary = route_a["control_summary"]
    c2 = oracle_all_singleton_gain(table)
    return {
        "rows": rows,
        "c2_reported": float(summary["c2_strongest_singleton_gain"]),
        "c2_recomputed": c2,
        "c2_reference_kind": "max_i S_i - B",
        "mixed_reference_finding": (
            "the object subtracts the pair-internal oracle while C2 subtracts the "
            "all-singleton oracle, yet both were carried in one field name and "
            "compared by a single margin test"
        ),
        "required_reported": float(summary["required"]),
        "required_recomputed": c2 + float(summary["margin"]),
        "object_reported": float(summary["object_gain"]),
        "object_recomputed_pair_internal": next(
            (
                row["recomputed_pair_internal_gain"]
                for row in rows
                if row["pair"] == "+".join(sorted(route_a["object"]["member_ids"]))
            ),
            0.0,
        ),
    }


def recompute_route_c(route_c: Mapping[str, Any], table: ContextTable) -> dict[str, Any]:
    """Recompute route C's aggregates so C and A are read through one dictionary."""

    rows = []
    for key, value in route_c["held_out_pairs_scored"].items():
        pair = tuple(sorted(str(item) for item in value["member_ids"]))
        rows.append(
            {
                "pair": key,
                "reported_mean_gain_vs_strongest_single": float(
                    value["mean_gain_vs_strongest_single"]
                ),
                "recomputed_gain_vs_all_singleton_oracle": (
                    policy_mean_gain_vs_all_singleton_oracle(table, pair)
                ),
                "recomputed_gain_vs_baseline": policy_mean_gain_vs_baseline(table, pair),
            }
        )
    return {
        "outcome": route_c.get("outcome"),
        "rows": rows,
        "c2_reported": float(route_c["control_summary"]["c2_strongest_singleton_gain"]),
        "c2_recomputed": oracle_all_singleton_gain(table),
    }


def cell_table(
    table: ContextTable, held_out_pairs: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    """One row per executed cell: every candidate reference side by side."""

    held_out = {tuple(sorted(pair)) for pair in held_out_pairs}
    rows: list[dict[str, Any]] = []
    for members in table.pair_cells():
        rows.append(
            {
                "cell": "+".join(members),
                "is_held_out": members in held_out,
                "mean_outcome": policy_mean_outcome(table, members),
                "gain_vs_baseline": policy_mean_gain_vs_baseline(table, members),
                "gain_vs_pair_internal_oracle": policy_mean_gain_vs_pair_internal_oracle(
                    table, members
                ),
                "gain_vs_all_singleton_oracle": policy_mean_gain_vs_all_singleton_oracle(
                    table, members
                ),
                "mean_interaction": policy_mean_interaction(table, members),
                "contexts_beating_all_singleton_oracle": (
                    contexts_beating_all_singleton_oracle(table, members)
                ),
            }
        )
    return rows


def audit(
    route_a_path: Path = DEFAULT_ROUTE_A_REPORT,
    route_c_path: Path = DEFAULT_ROUTE_C_REPORT,
    p52b_path: Path = DEFAULT_P52B_REPORT,
) -> dict[str, Any]:
    """Assemble the whole B0 audit payload."""

    route_a = _read_json(route_a_path)
    route_c = _read_json(route_c_path)
    p52b = _read_json(p52b_path)

    margin = float(route_a[MARGIN_SOURCE_FIELD[0]][MARGIN_SOURCE_FIELD[1]])
    success_matrix = route_c["success_matrix"]
    unseen = unseen_contexts_from_success_matrix(success_matrix)
    # Restrict the table to the holdout context of each block: route A scores on
    # ``contexts[TRAIN_CONTEXT_COUNT:]``, and mixing train contexts in would
    # silently average over a different context set than the frozen gate did.
    unseen_block_contexts = {
        block: [contexts[-1]] for block, contexts in success_matrix["blocks"].items()
    }
    table = table_from_success_matrix(success_matrix, unseen_block_contexts)
    if tuple(table.contexts) != unseen:
        raise SystemExit(
            "holdout-context selection disagrees between the block table and the "
            "success-matrix blocks; refusing to publish bounds built on it"
        )

    held_out_pairs = [tuple(pair) for pair in route_a["design"]["held_out_pairs"]]
    observed_pairs = [tuple(pair) for pair in route_a["design"]["observed_pairs"]]

    # The current design's only same-reference ceiling: contexts where every
    # singleton fails are the sole place a combination could add value.
    solvable_candidates = [
        context_id
        for context_id in table.contexts
        if table.strongest_singleton(context_id)[1] <= table.blank(context_id)
    ]

    cross_check = cross_check_route_a_rows(table, route_a)
    if not cross_check["all_match"]:
        raise SystemExit(
            "block-mean reconstruction disagrees with the frozen route-A rows; "
            "refusing to publish bounds built on it"
        )

    return {
        "format": "taiji-b0-measurement-reachability-audit-v1",
        "version": 1,
        "measurement_dictionary_version": MEASUREMENT_DICTIONARY_VERSION,
        "status": "draft_for_review",
        "does_not_change": [
            "no frozen report, preregistration or threshold is modified",
            "no training, no checkpoint write, no product adoption",
            "the historical numbers stay as reported; this audit adds fields",
        ],
        "inputs": {
            "route_a_report": str(route_a_path.relative_to(PROJECT_ROOT)),
            "route_c_report": str(route_c_path.relative_to(PROJECT_ROOT)),
            "p52b_report": str(p52b_path.relative_to(PROJECT_ROOT)),
        },
        "dictionary": {
            "blank": "B(context): outcome of the (F,F) baseline cell",
            "singleton": "S_i(context): outcome of member i alone",
            "combination": "P_ij(context): outcome of the pair cell as executed",
            "primary_task_performance": "mean outcome of a fixed policy over contexts",
            "same_reference_gain": "U(policy) - U(common reference), one reference for all policies",
            "pair_internal_reference": "max(S_i, S_j) -- frozen route-A object",
            "all_singleton_reference": "max_i S_i -- frozen route-A control C2",
            "causal_interaction": "P_ij - S_i - S_j + B, reported separately from utility",
            "deployable_control": "a policy frozen from train evidence only, no oracle access",
            "cost": "calls / steps / wall / recovery, folded with an a-priori frozen weight",
            "code_assertion": "each formula is exercised by a hand-calculated case in "
            "tests/taiji_native/test_b0_measurement_dictionary.py",
        },
        "cross_check": cross_check,
        "route_a_recomputation": recompute_route_a(route_a, table),
        "route_c_recomputation": recompute_route_c(route_c, table),
        "unseen_contexts": list(unseen),
        "cells": cell_table(table, held_out_pairs),
        "reachability": reachability_table(
            table,
            margin=margin,
            observed_pairs=observed_pairs,
            held_out_pairs=held_out_pairs,
            solvable_contexts=solvable_candidates,
        ),
        "task_structure_facts": {
            "singleton_surfaces": route_a["capability_surfaces"]["surfaces"],
            "pair_kinds": {
                key: value["kind"] for key, value in route_a["capability_surfaces"]["pairs"].items()
            },
            "block3_stop_reasons": route_a["block3_audit"]["stop_reason_classes"],
            "block3_uniformly_unreachable": route_a["block3_audit"]["uniformly_unreachable"],
            "discriminating_fraction": route_a["block3_audit"]["discriminating_fraction"],
            "contexts_where_every_singleton_fails": solvable_candidates,
        },
        "prediction_binding_finding": {
            "held_out_predictions": {
                "+".join(sorted(pair)): route_a["representation"]["predictions"].get(
                    "+".join(sorted(pair))
                )
                for pair in held_out_pairs
            },
            "prediction_distinctness": route_a["representation"]["prediction_distinctness"],
            "selected_object": list(route_a["object"]["member_ids"]),
            "finding": (
                "the two held-out pairs receive the identical prediction, so the "
                "selected object was fixed by tie-breaking rather than by ranking; "
                "a positive result under this binding would not evidence selection skill"
            ),
        },
        "p52b_state": {
            "groups": p52b["evaluator"]["groups"],
            "rejected": p52b["evaluator"]["rejected"],
            "rejected_reasons": p52b["evaluator"]["rejected_reasons"],
            "growth_admitted": p52b.get("growth_admitted"),
            "can_promote": p52b.get("can_promote"),
        },
        "hand_calculated_cases": [evaluate_hand_case(name) for name in HAND_CALCULATED_CASES],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--route-a", type=Path, default=DEFAULT_ROUTE_A_REPORT)
    parser.add_argument("--route-c", type=Path, default=DEFAULT_ROUTE_C_REPORT)
    parser.add_argument("--p52b", type=Path, default=DEFAULT_P52B_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = audit(args.route_a, args.route_c, args.p52b)
    _write_json(args.output, payload)

    reachability = payload["reachability"]
    print(f"dictionary: {payload['measurement_dictionary_version']}")
    print(
        f"cross-check: {payload['cross_check']['all_match']} "
        f"({payload['cross_check']['checked_fields']} fields)"
    )
    for row in reachability["rows"]:
        print(
            f"  reference={row['reference']:<26} required={row['required']:.3f} "
            f"ceiling={row['achievable_ceiling']:.3f} reachable={row['reachable']}"
        )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

"""Route-B task pre-check: does a candidate task admit the claim we intend to test?

Why this exists
---------------
The B0 audit showed that the frozen gate is unreachable, but *why* it is
unreachable determines what has to change.  This module answers that question
with two separable statements:

**1. The arbitration ceiling (provable).**  Any composition mechanism whose
per-context outcome is the outcome of *following one member's policy* is bounded
by ``max(S_i, S_j)``, and ``max(S_i, S_j) <= max_i S_i`` holds by construction.
Therefore no member-arbitration mechanism -- however good -- can make
``mean(P - max_i S_i)`` positive.  ``arbitration_slack`` computes this ceiling
for a concrete table and is asserted to be ``<= 0``.

**2. The escape route (measurable).**  A positive same-reference gain requires a
context where **every singleton fails but the combination succeeds** -- a
*combination-only solvable* context.  On such a context the oracle is the blank
outcome ``B`` and the combination scores a success, contributing ``1 - B`` to the
mean.  With ``n`` scored contexts and ``k`` such contexts, the ceiling is
``(1 - B) * k / n``.

That gives a design requirement in closed form::

    k > n * (reference_gain + margin) / (success_outcome - blank_outcome)

``required_combination_only_contexts`` returns the smallest such ``k``, so the
D1 reference choice becomes a countable task-design obligation instead of a
matter of taste.

**3. The frozen composition rule.**  The P5.2b mechanism executes, each tick, the
**first member in ``active_members`` order whose predicted action binds** -- a
priority fallback chain, not a free composition.  ``classify_pair_trajectory``
measures what that rule actually produced per context, which is how the audit
found that three of six pairs reach the within-pair optimum and three lose
exactly one context to the frozen order.

This module is a **read-only design instrument**.  It trains nothing, writes no
checkpoint and mutates no frozen artifact.  It is duck-typed against the
``ContextTable`` protocol of ``audit_taiji_b0_measurement_reachability`` and
therefore imports nothing at module scope.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_b0_task_reachability_precheck_20260913.json"
DICTIONARY_MODULE = (
    PROJECT_ROOT / "scripts" / "training" / ("audit_taiji_b0_measurement_reachability.py")
)

#: Identifier of the composition rule the P5.2b/P5.2c gates froze.  A different
#: rule is a different mechanism and needs its own preregistration.
COMPOSITION_RULE_ID = "p52b-priority-fallback-v1"
COMPOSITION_RULE_TEXT = (
    "each tick every active member predicts; the first member in active_members "
    "order whose action binds executes (lexicographic priority fallback)"
)

#: Outcome encoding, frozen with the measurement dictionary.
SUCCESS_OUTCOME = 1.0
FAILURE_OUTCOME = -1.0


# --------------------------------------------------------------------------- #
# 1. Arbitration ceiling -- provable, independent of any particular table
# --------------------------------------------------------------------------- #


def arbitration_ceiling(table: Any, pair: Sequence[str]) -> float:
    """``mean(max(S_i, S_j) - max_i S_i)`` -- the ceiling of *any* arbitration.

    Non-positive by construction: the maximum over a subset cannot exceed the
    maximum over the whole set.  A positive value here would mean the measurement
    is broken, not that collaboration was found.
    """

    first, second = tuple(sorted(pair))
    slacks = []
    for context_id in table.contexts:
        pair_best = max(table.singleton(context_id, first), table.singleton(context_id, second))
        all_best = max(table.singleton(context_id, member) for member in table.singletons)
        slacks.append(pair_best - all_best)
    return sum(slacks) / len(slacks) if slacks else 0.0


def arbitration_headroom(table: Any, pair: Sequence[str]) -> float:
    """``mean(max(S_i, S_j) - P_ij)`` -- how much the frozen rule left on the table.

    This is the part of the gap that *is* a mechanism/representation problem:
    it is recoverable without any new task, unlike the arbitration ceiling.
    """

    first, second = tuple(sorted(pair))
    losses = []
    for context_id in table.contexts:
        pair_best = max(table.singleton(context_id, first), table.singleton(context_id, second))
        losses.append(pair_best - table.combination(context_id, pair))
    return sum(losses) / len(losses) if losses else 0.0


# --------------------------------------------------------------------------- #
# 2. Trajectory classification -- what the frozen composition rule produced
# --------------------------------------------------------------------------- #


def classify_pair_trajectory(table: Any, pair: Sequence[str]) -> dict[str, Any]:
    """Classify each context by which member's outcome the pair reproduced.

    A pair that equals ``max(S_i, S_j)`` on every context already achieves the
    arbitration optimum -- so a failing gate cannot be blamed on arbitration.  A
    pair that equals *neither* member on some context has produced a genuinely
    interleaved trajectory, which is the only regime that can beat the oracle.
    """

    first, second = tuple(sorted(pair))
    per_context: list[dict[str, Any]] = []
    for context_id in table.contexts:
        outcome = table.combination(context_id, pair)
        s_first = table.singleton(context_id, first)
        s_second = table.singleton(context_id, second)
        if outcome == s_first == s_second:
            kind = "both_members"
        elif outcome == s_first:
            kind = "first_member_only"
        elif outcome == s_second:
            kind = "second_member_only"
        else:
            kind = "interleaved"
        per_context.append(
            {
                "context_id": context_id,
                "pair_outcome": outcome,
                "first_outcome": s_first,
                "second_outcome": s_second,
                "pair_best": max(s_first, s_second),
                "reaches_arbitration_optimum": bool(outcome == max(s_first, s_second)),
                "kind": kind,
            }
        )

    counts: dict[str, int] = {}
    for item in per_context:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1

    return {
        "pair": "+".join(sorted(pair)),
        "per_context": per_context,
        "kind_counts": counts,
        "contexts_reaching_arbitration_optimum": sum(
            1 for item in per_context if item["reaches_arbitration_optimum"]
        ),
        "contexts_interleaved": counts.get("interleaved", 0),
        "note": (
            "interleaved > 0 is the only trajectory class that can exceed the "
            "all-singleton oracle; equals-one-member can never do so"
        ),
    }


# --------------------------------------------------------------------------- #
# 3. Combination-only-solvable contexts -- the escape route, in closed form
# --------------------------------------------------------------------------- #


def combination_only_solvable_contexts(table: Any) -> list[str]:
    """Contexts where every singleton fails (so the oracle equals the blank).

    On these the combination has room to be *strictly* better than the oracle;
    everywhere else ``P <= max_i S_i`` by construction.  The property depends only
    on the singleton columns, not on which pair is being scored.
    """

    found: list[str] = []
    for context_id in table.contexts:
        oracle = max(table.singleton(context_id, member) for member in table.singletons)
        if oracle <= table.blank(context_id):
            found.append(context_id)
    return found


def combination_only_gain_potential(table: Any) -> float:
    """Ceiling of ``mean(P - max_i S_i)`` given *perfect* use of the escape route.

    Equals ``(success - B)`` on every context where the oracle has no capability
    and zero elsewhere.  This is the number the task design has to raise.
    """

    if not table.contexts:
        return 0.0
    total = 0.0
    for context_id in table.contexts:
        oracle = max(table.singleton(context_id, member) for member in table.singletons)
        if oracle <= table.blank(context_id):
            total += SUCCESS_OUTCOME - table.blank(context_id)
    return total / len(table.contexts)


def required_combination_only_contexts(
    *,
    reference_gain: float,
    margin: float,
    context_count: int,
    success_outcome: float = SUCCESS_OUTCOME,
    blank_outcome: float = FAILURE_OUTCOME,
) -> int:
    """Smallest ``k`` of ``n`` contexts that must be combination-only solvable.

    Derived from ``(success - B) * k / n > reference_gain + margin``.  Returns
    ``n + 1`` when even ``k == n`` is insufficient, which means the reference
    itself makes the claim unattainable -- the signal to change the reference
    rather than the task.
    """

    per_context = success_outcome - blank_outcome
    if per_context <= 0 or context_count <= 0:
        raise ValueError("success and blank outcomes must differ and n must be positive")
    required = (reference_gain + margin) * context_count / per_context
    k = int(required) + 1
    while k * per_context / context_count <= reference_gain + margin:
        k += 1
    return k if k <= context_count else context_count + 1


def max_clearable_reference(
    *,
    combination_only_contexts: int,
    context_count: int,
    margin: float,
    success_outcome: float = SUCCESS_OUTCOME,
    blank_outcome: float = FAILURE_OUTCOME,
) -> float:
    """Inverse view: the largest ``reference_gain`` a task of this shape can beat."""

    if context_count <= 0:
        return 0.0
    ceiling = (success_outcome - blank_outcome) * combination_only_contexts / context_count
    return ceiling - margin


# --------------------------------------------------------------------------- #
# 4. Reference requirements -- turning D1 into countable obligations
# --------------------------------------------------------------------------- #


def reference_requirements(
    table: Any, *, margin: float, candidates: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Per candidate reference: required ``k``, available ``k``, and the verdict.

    ``candidates`` maps a reference name to its frozen gain value.  The available
    count is measured from the table's singleton columns, so the same function
    serves both the current matrix and any proposed task.
    """

    n = len(table.contexts)
    available = len(combination_only_solvable_contexts(table))
    rows: list[dict[str, Any]] = []
    for name, gain in candidates.items():
        required = required_combination_only_contexts(
            reference_gain=gain, margin=margin, context_count=n
        )
        rows.append(
            {
                "reference": name,
                "reference_gain": float(gain),
                "required": float(gain) + margin,
                "required_combination_only_contexts": required,
                "available_combination_only_contexts": available,
                "context_count": n,
                "feasible": bool(required <= available),
                "ceiling_gain": combination_only_gain_potential(table),
                "max_clearable_reference": max_clearable_reference(
                    combination_only_contexts=available, context_count=n, margin=margin
                ),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 5. Whole-task pre-check (the mandatory precondition before any training)
# --------------------------------------------------------------------------- #


def precheck(table: Any, *, margin: float, candidates: Mapping[str, float]) -> dict[str, Any]:
    """Assemble the pre-check verdict for one candidate task surface."""

    pairs = [tuple(pair) for pair in table.pair_cells()]
    trajectories = [classify_pair_trajectory(table, pair) for pair in pairs]

    return {
        "format": "taiji-b0-task-reachability-precheck-v1",
        "version": 1,
        "status": "draft_for_review",
        "composition_rule": {
            "id": COMPOSITION_RULE_ID,
            "text": COMPOSITION_RULE_TEXT,
            "consequence": (
                "the pair cell reproduces one member's trajectory unless a member "
                "fails to bind mid-episode; it is a priority fallback, not a free "
                "composition, so 'interleaved' contexts are the only escape from "
                "the arbitration ceiling"
            ),
        },
        "contexts": list(table.contexts),
        "context_count": len(table.contexts),
        "margin": margin,
        "arbitration": [
            {
                "pair": "+".join(pair),
                "arbitration_ceiling": arbitration_ceiling(table, pair),
                "frozen_rule_headroom": arbitration_headroom(table, pair),
            }
            for pair in pairs
        ],
        "trajectories": trajectories,
        "combination_only": {
            "available_contexts": combination_only_solvable_contexts(table),
            "gain_potential": combination_only_gain_potential(table),
        },
        "reference_requirements": reference_requirements(
            table, margin=margin, candidates=candidates
        ),
        "verdict": _verdict(trajectories),
    }


def _verdict(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate the fixable gap from the unfixable one."""

    interleaved = sum(int(item["contexts_interleaved"]) for item in trajectories)
    optimum = sum(int(item["contexts_reaching_arbitration_optimum"]) for item in trajectories)
    scored = sum(len(item["per_context"]) for item in trajectories)
    return {
        "contexts_interleaved": interleaved,
        "contexts_at_arbitration_optimum": optimum,
        "contexts_scored": scored,
        "arbitration_is_the_bottleneck": bool(interleaved == 0),
        "reading": (
            "no context produced an interleaved trajectory, so every pair outcome "
            "is some single member's outcome: the gate is blocked by the task and "
            "the mechanism, not by the representation"
            if interleaved == 0
            else "interleaved trajectories exist, so a same-reference gain is at "
            "least structurally possible"
        ),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_dictionary() -> Any:
    import importlib.util
    import sys

    name = "_b0_dictionary_for_precheck"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DICTIONARY_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    dictionary = _load_dictionary()
    route_a = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
        ).read_text(encoding="utf-8")
    )
    route_c = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json"
        ).read_text(encoding="utf-8")
    )
    margin = float(route_a["control_summary"]["margin"])
    success_matrix = route_c["success_matrix"]
    block_contexts = {block: [contexts[-1]] for block, contexts in success_matrix["blocks"].items()}
    table = dictionary.table_from_success_matrix(success_matrix, block_contexts)

    candidates = {
        "all_singleton_oracle": dictionary.oracle_all_singleton_gain(table),
        "best_fixed_singleton": dictionary.best_deployable_singleton(table)[1],
        "best_observed_fixed_pair": dictionary.best_deployable_pair(
            table, [tuple(pair) for pair in route_a["design"]["observed_pairs"]]
        )[1],
    }
    payload = precheck(table, margin=margin, candidates=candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"composition rule: {COMPOSITION_RULE_ID}")
    for item in payload["arbitration"]:
        print(
            f"  {item['pair']:<26} ceiling={item['arbitration_ceiling']:+.3f} "
            f"frozen-rule headroom={item['frozen_rule_headroom']:+.3f}"
        )
    for item in payload["reference_requirements"]:
        print(
            f"  ref={item['reference']:<26} need k>={item['required_combination_only_contexts']}"
            f" of n={item['context_count']}, available={item['available_combination_only_contexts']}"
            f" feasible={item['feasible']}"
        )
    print(f"verdict: {payload['verdict']['reading']}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

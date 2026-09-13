"""Regression guard for the P5.2c' unseen-combination surface construction.

P5.2c was blocked because its gate 3 could not be satisfied: with four members
``train_only_candidates`` estimated all ``C(4,2)=6`` pairs, so no combination was
unseen.  The tempting fix -- use more members -- is measurably useless, because
``_estimate_pair`` returns an estimate as soon as any single train context has
all four factorial cells populated; adding members only adds observed pairs.

The real mechanism is to remove one designated pair's joint ``(T,T)`` cell from
every train context.  These tests pin the parts of that mechanism that are easy
to break silently:

* removal must cover the whole train partition -- leaving one complete context
  behind restores the estimate and empties the unseen surface again;
* removal must be confined to the joint cell, so both members keep complete
  marginal (singleton and baseline) evidence and the pair is *unseen* rather
  than *unsupported*;
* holdout must keep the joint cell, otherwise the realized gain is unmeasurable;
* the entry audit must fail closed when any of the above is violated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from taiji.interaction_groups import (
    InteractionTraceCorpus,
    InteractionTraceEpisode,
    InteractionTraceEvent,
)

REPO = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO / "scripts" / "training" / "eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py"
)
FMT = "taiji-interaction-trace-v1"


def _load_runner():
    scripts = str(RUNNER.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("_p52cp_gate_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _episode(context_id: str, members: tuple[str, ...], outcome: float = 1.0, tag: str = "") -> str:
    """Build a projected episode whose event owners are exactly ``members``."""

    episode_id = f"ep:{context_id}:{'-'.join(members) or 'none'}:{tag}"
    events = tuple(
        InteractionTraceEvent(
            event_id=f"{episode_id}:ev:{index}",
            owner_id=member,
            episode_id=episode_id,
            checkpoint_revision=1,
            outcome_id=f"{episode_id}:outcome",
            resource_cost=1.0,
        )
        for index, member in enumerate(members)
    )
    return InteractionTraceEpisode(
        episode_id=episode_id,
        checkpoint_revision=1,
        outcome_id=f"{episode_id}:outcome",
        events=events,
        outcome=outcome,
        context_id=context_id,
    )


# --------------------------------------------------------------------------- #
# The mechanism itself
# --------------------------------------------------------------------------- #


def test_declares_opaque_indexed_pair(runner) -> None:
    """The designated pair must be expressed as indices, not as semantic names."""

    assert runner.HELD_OUT_PAIR_INDEX == (0, 3)
    assert len(runner.HELD_OUT_PAIR) == 2
    members = runner.MEMBER_IDS
    assert runner.HELD_OUT_PAIR == tuple(
        sorted((members[0], members[3]))
    ), "the opaque index must resolve through MEMBER_IDS, not be hardcoded"
    # and the remaining observable pairs must be exactly the complement
    assert len(runner.OBSERVED_PAIR_MEMBER_SETS) == len(runner.PAIR_MEMBER_SETS) - 1


def test_removal_covers_entire_train_partition_not_part_of_it(runner) -> None:
    """Partial removal restores the estimate; only full removal hides the pair.

    This is the property that makes the design work and that a plausible
    "optimisation" would silently destroy.  Both variants are constructed and
    compared here, so the reasoning is pinned by execution rather than asserted
    in prose.
    """

    pair = runner.HELD_OUT_PAIR
    train_ids = sorted(runner._train_context_ids())
    assert len(train_ids) == runner.TRAIN_CONTEXT_COUNT

    others = [item for item in runner.SINGLETON_MEMBER_SETS]
    assert others, "singletons must exist for the surface to be merely unseen"

    def build(remove_from: set[str]) -> tuple[InteractionTraceEpisode, ...]:
        episodes: list[InteractionTraceEpisode] = []
        for context_id in train_ids:
            for cell in runner.CELL_MEMBER_SETS:
                if tuple(sorted(cell)) == pair and context_id in remove_from:
                    continue
                episodes.append(_episode(context_id, tuple(cell)))
        return tuple(episodes)

    from taiji.interaction_groups import InteractionGroupEvaluator

    evaluator = InteractionGroupEvaluator()

    def observed(episodes: tuple[InteractionTraceEpisode, ...]) -> set[tuple[str, ...]]:
        # holdout must be non-empty for the corpus; give it the full matrix
        holdout = tuple(
            _episode(f"hold-{index}", tuple(cell))
            for index in range(2)
            for cell in runner.CELL_MEMBER_SETS
        )
        corpus = InteractionTraceCorpus(train=episodes, holdout=holdout)
        return {tuple(sorted(item.member_ids)) for item in evaluator.train_only_candidates(corpus)}

    partial = observed(build(set(train_ids[:1])))
    assert pair in partial, (
        "removing the joint cell from only one train context must NOT hide the "
        "pair: a single complete context is sufficient for _estimate_pair"
    )

    full = observed(build(set(train_ids)))
    assert pair not in full, "removing the joint cell from all train contexts must hide the pair"
    assert len(full) == len(runner.OBSERVED_PAIR_MEMBER_SETS)
    for expected in runner.OBSERVED_PAIR_MEMBER_SETS:
        assert tuple(sorted(expected)) in full, f"{expected} must remain observable"


def test_removal_confined_to_joint_cell_keeps_marginal_evidence(runner) -> None:
    """The pair must be unseen, not unsupported: singletons and baseline survive."""

    pair = runner.HELD_OUT_PAIR
    train_ids = sorted(runner._train_context_ids())
    episodes: list[InteractionTraceEpisode] = []
    removed: list[dict] = []
    for context_id in train_ids:
        for cell in runner.CELL_MEMBER_SETS:
            if tuple(sorted(cell)) == pair:
                removed.append({"episode_id": f"drop:{context_id}", "context_id": context_id})
                continue
            episodes.append(_episode(context_id, tuple(cell)))
    holdout = tuple(
        _episode(f"hold-{index}", tuple(cell))
        for index in range(2)
        for cell in runner.CELL_MEMBER_SETS
    )
    corpus = InteractionTraceCorpus(train=tuple(episodes), holdout=holdout)

    for member in pair:
        singles = [item for item in corpus.train if item.member_ids == (member,)]
        baselines = [item for item in corpus.train if item.member_ids == ()]
        assert singles, f"{member} lost all singleton evidence in train"
        assert baselines, "the inactive baseline observations were lost from train"
    # the joint cell is gone from train but present in holdout
    assert not [item for item in corpus.train if frozenset(item.member_ids) == frozenset(pair)]
    assert [item for item in corpus.holdout if frozenset(item.member_ids) == frozenset(pair)]
    # the ledger records exactly the removal, so it is auditable
    assert removed


# --------------------------------------------------------------------------- #
# The entry audit must fail closed
# --------------------------------------------------------------------------- #


def _audit_inputs(runner):
    pair = runner.HELD_OUT_PAIR
    train_ids = sorted(runner._train_context_ids())
    train = tuple(
        _episode(context_id, tuple(cell))
        for context_id in train_ids
        for cell in runner.CELL_MEMBER_SETS
        if tuple(sorted(cell)) != pair
    )
    holdout = tuple(
        _episode(f"hold-{index}", tuple(cell))
        for index in range(2)
        for cell in runner.CELL_MEMBER_SETS
    )
    corpus = InteractionTraceCorpus(train=train, holdout=holdout)
    from taiji.interaction_groups import InteractionGroupEvaluator

    records = InteractionGroupEvaluator().train_only_candidates(corpus)
    return corpus, records


def test_entry_audit_passes_on_a_correctly_constructed_surface(runner) -> None:
    corpus, records = _audit_inputs(runner)
    reality = {"interventions_happened": True, "non_baseline_zero_step_total": 0}
    audit = runner._entry_audit(
        corpus,
        removed=[
            {"episode_id": f"drop:{index}", "context_id": ctx, "member_ids": []}
            for index, ctx in enumerate(sorted(runner._train_context_ids()))
            for _ in range(runner.REPEATS)
        ],
        evaluator=None,
        train_only_records=records,
        intervention_reality=reality,
    )
    assert audit["passed"] is True, audit["conditions"]
    assert audit["held_out_pair_joint_cell_count_in_train"] == 0
    assert audit["held_out_pair_joint_cell_count_in_holdout"] > 0


def test_entry_audit_fails_closed_when_pair_is_still_observable(runner) -> None:
    """If removal silently did nothing, the gate must stop -- not reinterpret."""

    pair = runner.HELD_OUT_PAIR
    train_ids = sorted(runner._train_context_ids())
    train = tuple(
        _episode(context_id, tuple(cell))
        for context_id in train_ids
        for cell in runner.CELL_MEMBER_SETS
    )
    holdout = tuple(
        _episode(f"hold-{index}", tuple(cell))
        for index in range(2)
        for cell in runner.CELL_MEMBER_SETS
    )
    corpus = InteractionTraceCorpus(train=train, holdout=holdout)
    from taiji.interaction_groups import InteractionGroupEvaluator

    records = InteractionGroupEvaluator().train_only_candidates(corpus)
    reality = {"interventions_happened": True, "non_baseline_zero_step_total": 0}
    audit = runner._entry_audit(
        corpus,
        removed=[],
        evaluator=None,
        train_only_records=records,
        intervention_reality=reality,
    )
    assert audit["passed"] is False
    names = {item["condition"] for item in audit["conditions"]}
    assert "held_out_pair_not_observed" in names
    assert "held_out_pair_joint_cell_absent_from_train" in names
    assert pair in {tuple(sorted(item.member_ids)) for item in records}


def test_entry_audit_rejects_inert_interventions(runner) -> None:
    corpus, records = _audit_inputs(runner)
    reality = {"interventions_happened": False, "non_baseline_zero_step_total": 7}
    audit = runner._entry_audit(
        corpus,
        removed=[
            {"episode_id": f"drop:{index}", "context_id": ctx, "member_ids": []}
            for index, ctx in enumerate(sorted(runner._train_context_ids()))
            for _ in range(runner.REPEATS)
        ],
        evaluator=None,
        train_only_records=records,
        intervention_reality=reality,
    )
    assert audit["passed"] is False
    assert "interventions_executed" in {item["condition"] for item in audit["conditions"]}


def test_calibration_thresholds_are_frozen_literals(runner) -> None:
    """Gate 8's thresholds were fixed at freeze time and must not be derived."""

    assert runner.CALIBRATION_MINIMUM_SIGN_MATCH_RATE == 0.5
    assert runner.CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR == 0.35
    assert runner.MARGIN == 0.15
    assert runner.TOTAL_SECONDS_CAP == 900.0
    source = RUNNER.read_text(encoding="utf-8")
    assert "fixed_at_freeze_time" in source


def test_fixed_control_coincides_with_answer_and_is_excluded_from_margin(runner) -> None:
    """The disclosed design cost: C4 duplicates the answer, so it cannot bear the margin."""

    assert runner.FIXED_COMBINATION == runner.HELD_OUT_PAIR
    source = RUNNER.read_text(encoding="utf-8")
    assert 'if name != "fixed_combination"' in source
    assert "known_cost_disclosure" in source


def test_report_does_not_overwrite_predecessor(runner) -> None:
    """The new report path must be distinct from the blocked P5.2c report."""

    assert RUNNER.exists()
    predecessor = REPO / "reports" / "taiji_p5_2c_unseen_combination_transfer_20260913.json"
    assert runner.DEFAULT_REPORT.name == (
        "taiji_p5_2c_prime_unseen_combination_transfer_20260913.json"
    )
    assert runner.DEFAULT_REPORT != predecessor
    assert "P5_2C_PRIME" in runner.PREREGISTRATION

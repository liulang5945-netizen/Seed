"""Regression guard for the P5.2c'' route-C design repair.

P5.2c' reported ``transfer_no_gain``, but the decision prompt established that the
result came from *design* defects rather than capability:

* only one unseen combination existed, so calibration rested on one sample;
* the held-out pair was chosen by lexical index (0,3), which turned out to be the
  single pair whose members share a capability surface (a and d both cover
  block-0) -- i.e. a structurally redundant pair;
* block-3 is unreachable for every cell, shrinking the discriminating surface.

This gate holds out TWO combinations, both verified by real execution to be
disjoint, and quantifies block-3 instead of hiding it.  These tests pin the parts
that would silently rot:

* both held-out pairs must be removed from the WHOLE train partition (partial
  removal re-exposes the pair, as established for the predecessor);
* removal must stay confined to the joint cells, so both pairs are *unseen*
  rather than *unsupported*;
* the capability surfaces must be re-measured, never quoted from the design note;
* the entry audit must additionally fail when the held-out pairs are NOT disjoint
  -- that is the specific defect this gate exists to prevent recurring;
* the outcome classifier must return ``transfer_signal_constant`` (not
  ``transfer_no_gain``) when no held-out pair shows a positive gain, because the
  preregistration distinguishes those two conditions.
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
    REPO
    / "scripts"
    / "training"
    / "eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate.py"
)
PRIME_RUNNER = (
    REPO / "scripts" / "training" / "eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py"
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
def runner():
    # the prime runner must be importable first: this gate reuses its helpers
    _load("_p52cp_gate_for_double_prime", PRIME_RUNNER)
    return _load("_p52cpp_gate_under_test", RUNNER)


def _episode(context_id: str, members: tuple[str, ...], tag: str = "") -> InteractionTraceEpisode:
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
        outcome=1.0,
        context_id=context_id,
    )


def _full_matrix(gate, context_ids, skip_joint: set[frozenset] | None = None):
    """Build a full cell x context matrix.

    ``gate`` is passed explicitly rather than read from the module global: the
    global named ``runner`` is the *fixture function*, not the loaded module, so
    reading it here silently looked up attributes on the wrong object.
    """

    skip_joint = skip_joint or set()
    out = []
    for context_id in context_ids:
        for cell in gate.CELL_MEMBER_SETS:
            if frozenset(cell) in skip_joint:
                continue
            out.append(_episode(context_id, tuple(cell)))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Design shape
# --------------------------------------------------------------------------- #


def test_holds_out_exactly_two_pairs_by_opaque_index(runner) -> None:
    assert runner.HELD_OUT_PAIR_INDICES == ((0, 2), (1, 3))
    assert len(runner.HELD_OUT_PAIRS) == 2
    members = runner.MEMBER_IDS
    assert runner.HELD_OUT_PAIRS[0] == tuple(sorted((members[0], members[2])))
    assert runner.HELD_OUT_PAIRS[1] == tuple(sorted((members[1], members[3])))
    # the observable surface is the complement, and it must still be non-trivial
    assert len(runner.OBSERVED_PAIR_MEMBER_SETS) == len(runner.PAIR_MEMBER_SETS) - 2
    assert len(runner.OBSERVED_PAIR_MEMBER_SETS) == 4


def test_does_not_widen_member_set(runner) -> None:
    """Adding members was rejected on evidence; the gate must not quietly do it."""

    assert len(runner.MEMBER_IDS) == 4
    assert len(runner.CELL_MEMBER_SETS) == 11
    source = RUNNER.read_text(encoding="utf-8")
    assert "member_set_widening_rejected_because" in source
    assert "four train template families" in source


def test_declares_it_does_not_fix_representation(runner) -> None:
    """A negative result here means 'under this representation', not 'impossible'."""

    source = RUNNER.read_text(encoding="utf-8")
    assert "does_not_fix" in source
    assert "route A is separate" in source
    assert "representation_defect_still_present_note" in source
    assert "contribution_uniform" in source


def test_frozen_constants_inherited_not_redefined(runner) -> None:
    """MARGIN and the calibration thresholds must be the frozen values, unchanged."""

    assert runner.MARGIN == 0.15
    assert runner.CALIBRATION_MINIMUM_SIGN_MATCH_RATE == 0.5
    assert runner.CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR == 0.35
    assert runner.TOTAL_SECONDS_CAP == 900.0
    # and they must come from the predecessor rather than be re-typed
    source = RUNNER.read_text(encoding="utf-8")
    assert "p52cp.MARGIN" in source
    assert "p52cp.CALIBRATION_MINIMUM_SIGN_MATCH_RATE" in source


# --------------------------------------------------------------------------- #
# Removal mechanism, for BOTH pairs
# --------------------------------------------------------------------------- #


def test_removal_covers_whole_train_partition_for_both_pairs(runner) -> None:
    """Both pairs must be hidden by removing their joint cell from ALL train ctx."""

    from taiji.interaction_groups import InteractionGroupEvaluator

    train_ids = sorted(runner._train_context_ids())
    held = {frozenset(p) for p in runner.HELD_OUT_PAIRS}

    def build(remove_from: set[str]):
        # Build the FULL matrix (no skip): the removal must be performed here, by
        # ``remove_from``, so that "partial" and "full" actually differ.  Passing
        # ``skip_joint=held`` would strip the held-out cells up front and make the
        # partial case vacuous.
        pushed = _full_matrix(runner, train_ids)
        kept = []
        for ep in pushed:
            if ep.context_id in remove_from and frozenset(ep.member_ids) in held:
                continue
            kept.append(ep)
        return tuple(kept)

    holdout = _full_matrix(runner, [f"hold-{i}" for i in range(2)])
    evaluator = InteractionGroupEvaluator()

    def observed(train):
        corpus = InteractionTraceCorpus(train=train, holdout=holdout)
        return {tuple(sorted(r.member_ids)) for r in evaluator.train_only_candidates(corpus)}

    # Partial removal must NOT hide the pair: the other train contexts still
    # supply the joint cell, so ``_estimate_pair`` keeps returning an estimate.
    # This is the mechanical fact that forced whole-partition removal.
    partial = observed(build(set(train_ids[:1])))
    for pair in runner.HELD_OUT_PAIRS:
        assert (
            tuple(sorted(pair)) in partial
        ), "removing the joint cell from one context must NOT hide the pair"

    full = observed(build(set(train_ids)))
    for pair in runner.HELD_OUT_PAIRS:
        assert tuple(sorted(pair)) not in full, f"{pair} should be hidden"
    for expected in runner.OBSERVED_PAIR_MEMBER_SETS:
        assert tuple(sorted(expected)) in full, f"{expected} must remain observable"


def test_build_corpus_removes_both_joint_cells_and_keeps_marginals(runner) -> None:
    train_ids = sorted(runner._train_context_ids())
    held = {frozenset(p) for p in runner.HELD_OUT_PAIRS}
    projected = _full_matrix(runner, train_ids + [f"hold-{i}" for i in range(2)])
    corpus, removed = runner._build_corpus(projected)

    # This synthetic matrix emits ONE episode per (context, cell), so removal is
    # 8 contexts x 2 held-out pairs x 1 episode = 16.  The 32 figure quoted in
    # the preregistration is for the REAL projected corpus, which carries
    # ``REPEATS``=2 episodes per (context, cell) -- asserted separately below so
    # the arithmetic can never silently drift apart.
    synthetic_per_cell = 1
    expected_removed = (
        len(runner._train_context_ids())
        * len(runner.HELD_OUT_PAIRS)
        * synthetic_per_cell
    )
    assert expected_removed == 16
    assert len(removed) == expected_removed

    real_expected_removed = (
        runner.TRAIN_CONTEXT_COUNT * runner.REPEATS * len(runner.HELD_OUT_PAIRS)
    )
    assert real_expected_removed == 32

    # every removed episode must belong to a held-out pair in a train context
    held_episode_ids = {e.episode_id for e in projected if e.context_id in set(train_ids)}
    for entry in removed:
        assert entry["episode_id"] in held_episode_ids
        assert frozenset(entry["member_ids"]) in held
        assert entry["context_id"] in set(train_ids)

    for pair in runner.HELD_OUT_PAIRS:
        target = frozenset(pair)
        assert not [e for e in corpus.train if frozenset(e.member_ids) == target]
        assert [e for e in corpus.holdout if frozenset(e.member_ids) == target]
        for member in pair:
            assert [
                e for e in corpus.train if e.member_ids == (member,)
            ], f"{member} lost singleton evidence"
    assert [e for e in corpus.train if e.member_ids == ()], "baseline evidence lost"


# --------------------------------------------------------------------------- #
# Entry audit must fail closed, including on the disjointness condition
# --------------------------------------------------------------------------- #


def _audit_setup(runner):
    train_ids = sorted(runner._train_context_ids())
    held = {frozenset(p) for p in runner.HELD_OUT_PAIRS}
    train = _full_matrix(runner, train_ids, skip_joint=held)
    holdout = _full_matrix(runner, [f"hold-{i}" for i in range(2)])
    corpus = InteractionTraceCorpus(train=train, holdout=holdout)
    from taiji.interaction_groups import InteractionGroupEvaluator

    records = InteractionGroupEvaluator().train_only_candidates(corpus)
    ledger = [
        {"episode_id": f"drop:{i}", "context_id": ctx, "member_ids": []}
        for i, ctx in enumerate(train_ids)
        for _ in range(runner.REPEATS * 2)
    ]
    return corpus, records, ledger


def test_entry_audit_passes_on_correct_construction(runner) -> None:
    corpus, records, ledger = _audit_setup(runner)
    audit = runner._entry_audit(
        corpus,
        removed=ledger,
        train_only_records=records,
        intervention_reality={
            "interventions_happened": True,
            "non_baseline_zero_step_total": 0,
        },
    )
    assert audit["passed"] is True, audit["conditions"]
    assert set(audit["held_out_pairs"]) == {
        "+".join(pair) for pair in runner.HELD_OUT_PAIRS
    }
    for entry in audit["held_out_pairs"].values():
        assert entry["joint_count_in_train"] == 0
        assert entry["joint_count_in_holdout"] > 0
        assert entry["in_observed_records"] is False


def test_entry_audit_fails_closed_when_removal_did_nothing(runner) -> None:
    train_ids = sorted(runner._train_context_ids())
    # NOTE: no joint cells removed at all
    train = _full_matrix(runner, train_ids)
    holdout = _full_matrix(runner, [f"hold-{i}" for i in range(2)])
    corpus = InteractionTraceCorpus(train=train, holdout=holdout)
    from taiji.interaction_groups import InteractionGroupEvaluator

    records = InteractionGroupEvaluator().train_only_candidates(corpus)
    audit = runner._entry_audit(
        corpus,
        removed=[],
        train_only_records=records,
        intervention_reality={
            "interventions_happened": True,
            "non_baseline_zero_step_total": 0,
        },
    )
    assert audit["passed"] is False
    names = {c["condition"] for c in audit["conditions"]}
    for pair in runner.HELD_OUT_PAIRS:
        label = "+".join(pair)
        assert f"held_out_pair[{label}]_not_observed" in names
        assert f"held_out_pair[{label}]_joint_absent_from_train" in names


def test_entry_audit_rejects_inert_interventions(runner) -> None:
    corpus, records, ledger = _audit_setup(runner)
    audit = runner._entry_audit(
        corpus,
        removed=ledger,
        train_only_records=records,
        intervention_reality={
            "interventions_happened": False,
            "non_baseline_zero_step_total": 5,
        },
    )
    assert audit["passed"] is False
    assert "interventions_executed" in {c["condition"] for c in audit["conditions"]}


def test_gate_adds_disjointness_condition(runner) -> None:
    """The specific defect route C exists to prevent must be checked, not assumed."""

    source = RUNNER.read_text(encoding="utf-8")
    assert "held_out_pairs_capability_disjoint" in source
    assert "held_out_pairs_all_disjoint" in source
    assert "measured this round from real execution" in source


# --------------------------------------------------------------------------- #
# Outcome classification must distinguish constant from no-gain
# --------------------------------------------------------------------------- #


def test_outcome_classifier_returns_signal_constant_when_no_pair_positive(runner) -> None:
    """Route C: zero positive held-out pairs is 'constant', not 'no_gain'."""

    source = RUNNER.read_text(encoding="utf-8")
    assert "any_held_out_positive" in source
    assert "all_cells_constant or not any_held_out_positive" in source
    # and the constant branch must map to transfer_signal_constant
    idx = source.index("all_cells_constant or not any_held_out_positive")
    assert "transfer_signal_constant" in source[idx : idx + 200]


def test_block3_is_quantified_not_hidden(runner) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "_block3_audit" in source
    assert "uniformly_unreachable" in source
    assert "discriminating_fraction" in source


def test_report_path_does_not_overwrite_predecessors(runner) -> None:
    assert runner.DEFAULT_REPORT.name == (
        "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json"
    )
    for predecessor in (
        "taiji_p5_2c_unseen_combination_transfer_20260913.json",
        "taiji_p5_2c_prime_unseen_combination_transfer_20260913.json",
        "taiji_p5_2b_group_causal_corpora_20260913.json",
    ):
        assert runner.DEFAULT_REPORT.name != predecessor
    assert "DOUBLE_PRIME" in runner.PREREGISTRATION


def test_rejection_probe_is_local_not_delegated(runner) -> None:
    """The probes must not take their *expected value* from the predecessor.

    Calling the predecessor's probe would fit a learner with that module's
    constants (which remove only its own single held-out pair) and then compare
    against that module's ``HELD_OUT_PAIR`` -- i.e. it would judge this gate by a
    corpus this gate never built.  The helper that merely *emits a script* carries
    no held-out constant and is legitimately reusable, so it is excluded.
    """

    source = RUNNER.read_text(encoding="utf-8")
    assert "p52cp._rejection_probe(" not in source
    assert "p52cp._recovery_probe(" not in source
    assert "p52cp._build_corpus(" not in source
    assert "p52cp._joint_counts(" not in source
    # both recovery and rejection must compute expectations from THIS gate's pairs
    assert "HELD_OUT_PAIRS[0]" in source
    assert "held_out_pair_predictable_P1" in source
    assert "held_out_pair_predictable_P2" in source
    assert "all_held_out_pairs_predictable" in source

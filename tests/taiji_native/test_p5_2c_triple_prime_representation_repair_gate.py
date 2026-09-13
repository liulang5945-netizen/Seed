"""Regression guard for the P5.2c''' route-A representation repair.

Route A replaces the pooled scalar member profile with a **capability surface**
plus three set-overlap pair features.  The predecessor P5.2c'' reported
``transfer_signal_constant``: every candidate pair received exactly the same
prediction, because ``contribution`` is a mean over all contexts (so it is
identical for all four members) and the old feature vector
``(1, (c1+c2)/2, c1*c2)`` therefore produced one single row for every pair --
a design matrix of rank 1 with 3 columns.

These tests pin the parts that would silently rot:

* ``MEMBER_EVIDENCE_VERSION`` 1 -> 2, and a version-1 profile must FAIL CLOSED
  rather than be silently defaulted to an empty surface;
* the surface is derived with a **strict** ``> 0`` on the singleton delta;
* ``_block_of`` must return ``None`` (not raise) for context ids without a
  numeric suffix, because the earlier P5.2 gates name contexts
  ``train-workbench-<family>``; raising there would break unrelated callers, and
  guessing a block would fabricate structure;
* the three added columns are algebraically dependent, so only *predictions*
  may be interpreted -- a test asserts the gate discloses this;
* the construction must be inherited verbatim from P5.2c'' so the comparison
  isolates the representation change;
* the classifier must be able to distinguish ``representation_repair_ineffective``
  from ``transfer_signal_constant``: if the repair did not take effect, the gain
  gate failing is NOT evidence about capability.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from taiji.interaction_group_transfer import (
    INTERACTION_GROUP_TRANSFER_MODEL_REVISION,
    MEMBER_EVIDENCE_VERSION,
    SURFACE_BLOCK_COUNT,
    InteractionGroupMemberEvidence,
    InteractionGroupTransferLearner,
    build_member_evidence,
)
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
    / "eval_taiji_p5_2c_triple_prime_representation_repair_gate.py"
)
DOUBLE_PRIME_RUNNER = (
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
    # the prime runner must be importable first: both later gates reuse helpers
    _load("_p52cp_gate_for_triple_prime", PRIME_RUNNER)
    _load("_p52cpp_gate_for_triple_prime", DOUBLE_PRIME_RUNNER)
    return _load("_p52cppp_gate_under_test", RUNNER)


DIGEST = hashlib.sha256(b"p5.2c-triple-prime-test").hexdigest()
DIGEST_B = hashlib.sha256(b"p5.2c-triple-prime-other").hexdigest()


def _profile(
    member_id: str,
    surface: tuple[int, ...] = (),
    contribution: float = 0.5,
    digest: str = DIGEST,
) -> InteractionGroupMemberEvidence:
    return InteractionGroupMemberEvidence(
        member_id=member_id,
        source_trace_digest=digest,
        checkpoint_revision=1,
        contribution=contribution,
        recovery_effect=0.0,
        resource_cost=0.1,
        observations=4,
        context_count=4,
        surface=surface,
    )


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


# --------------------------------------------------------------------------- #
# Version 2 profile: fail closed, never fabricate
# --------------------------------------------------------------------------- #


def test_member_evidence_version_bumped_to_two() -> None:
    assert MEMBER_EVIDENCE_VERSION == 2
    assert INTERACTION_GROUP_TRANSFER_MODEL_REVISION == 2
    assert SURFACE_BLOCK_COUNT == 4


def test_version_one_profile_must_fail_closed() -> None:
    """A v1 checkpoint has no surface; defaulting it to () would be a silent lie."""

    with pytest.raises(ValueError) as excinfo:
        InteractionGroupMemberEvidence(
            member_id="member-a",
            source_trace_digest=DIGEST,
            checkpoint_revision=1,
            contribution=0.5,
            recovery_effect=0.0,
            resource_cost=0.1,
            observations=4,
            context_count=4,
            version=1,
        )
    assert "version" in str(excinfo.value).lower()


def test_surface_must_be_sorted_deduplicated_and_in_range() -> None:
    _profile("member-a", surface=(0, 2))  # valid
    with pytest.raises(ValueError):
        _profile("member-a", surface=(2, 0))  # unsorted
    with pytest.raises(ValueError):
        _profile("member-a", surface=(0, 0))  # duplicated
    with pytest.raises(ValueError):
        _profile("member-a", surface=(SURFACE_BLOCK_COUNT,))  # out of range
    with pytest.raises(ValueError):
        _profile("member-a", surface=(-1,))  # negative


def test_surface_round_trips_through_payload() -> None:
    payload = _profile("member-b", surface=(1, 3)).to_payload()
    assert payload["surface"] == [1, 3]
    restored = InteractionGroupMemberEvidence.from_payload(payload)
    assert restored.surface == (1, 3)
    assert restored.surface_set == frozenset({1, 3})


# --------------------------------------------------------------------------- #
# Surface derivation: strict positivity, block = task_index % 4
# --------------------------------------------------------------------------- #


def test_block_of_returns_none_for_non_numeric_suffix() -> None:
    """Earlier gates name contexts ``train-workbench-<family>``.

    Raising would break unrelated callers that never had a block mapping;
    guessing would fabricate structure.  ``None`` is the only honest answer and
    the caller must then record NO surface hit.
    """

    from taiji.interaction_group_transfer import _block_of

    assert _block_of("train-workbench-ab") is None
    assert _block_of("holdout-workbench-lang") is None
    assert _block_of("no-suffix-here") is None
    # numeric suffixes must still map by block = index % 4
    assert _block_of("p52a-validation-100") == 0
    assert _block_of("p52a-validation-101") == 1
    assert _block_of("p52a-validation-102") == 2
    assert _block_of("p52a-validation-103") == 3
    assert _block_of("p52a-validation-104") == 0


def test_surface_uses_strict_positive_delta() -> None:
    """A zero-gain singleton must NOT be credited with that block."""

    def with_outcome(episode: InteractionTraceEpisode, outcome: float) -> InteractionTraceEpisode:
        return InteractionTraceEpisode(
            episode_id=episode.episode_id,
            checkpoint_revision=episode.checkpoint_revision,
            outcome_id=episode.outcome_id,
            events=episode.events,
            outcome=outcome,
            context_id=episode.context_id,
        )

    # block 0 (ctx-100): singleton 0.8 vs baseline 0.2 -> +0.6, a surface hit
    # block 1 (ctx-101): singleton 0.5 vs baseline 0.5 -> exactly 0.0, no hit
    train = (
        with_outcome(_episode("ctx-100", ()), 0.2),
        with_outcome(_episode("ctx-101", ()), 0.5),
        with_outcome(_episode("ctx-100", ("member-a",)), 0.8),
        with_outcome(_episode("ctx-101", ("member-a",)), 0.5),
    )
    profiles = build_member_evidence(
        train, source_trace_digest=DIGEST, checkpoint_revision=1
    )
    by_member = {profile.member_id: profile for profile in profiles}
    assert "member-a" in by_member
    assert by_member["member-a"].surface == (0,), (
        "a zero delta must not register a surface hit: "
        f"got {by_member['member-a'].surface}"
    )


def test_surface_records_every_block_with_strictly_positive_delta() -> None:
    """Positive deltas on two blocks must both appear, in ascending order."""

    def with_outcome(episode: InteractionTraceEpisode, outcome: float) -> InteractionTraceEpisode:
        return InteractionTraceEpisode(
            episode_id=episode.episode_id,
            checkpoint_revision=episode.checkpoint_revision,
            outcome_id=episode.outcome_id,
            events=episode.events,
            outcome=outcome,
            context_id=episode.context_id,
        )

    train = (
        with_outcome(_episode("ctx-100", ()), 0.0),
        with_outcome(_episode("ctx-102", ()), 0.0),
        with_outcome(_episode("ctx-100", ("member-a",)), 1.0),
        with_outcome(_episode("ctx-102", ("member-a",)), 1.0),
    )
    profiles = build_member_evidence(
        train, source_trace_digest=DIGEST, checkpoint_revision=1
    )
    by_member = {profile.member_id: profile for profile in profiles}
    assert by_member["member-a"].surface == (0, 2)


def test_build_member_evidence_skips_unmapped_contexts() -> None:
    """Contexts without a numeric suffix contribute no block, not a bogus one."""

    train = (
        _episode("train-workbench-ab", ()),
        _episode("train-workbench-ab", ("member-a",)),
    )
    profiles = build_member_evidence(
        train, source_trace_digest=DIGEST, checkpoint_revision=1
    )
    by_member = {profile.member_id: profile for profile in profiles}
    if "member-a" in by_member:
        assert by_member["member-a"].surface == ()


# --------------------------------------------------------------------------- #
# The representation itself: rank, distinctness, and honest coefficient caveat
# --------------------------------------------------------------------------- #


def _fitted_learner(profiles: tuple[InteractionGroupMemberEvidence, ...]):
    learner = InteractionGroupTransferLearner(
        ridge=0.1, minimum_utility=-1.0e9, maximum_uncertainty=1.0e9
    )
    learner.observe_members(profiles)
    return learner


ROUND_SURFACES = (("a", (0,)), ("b", (1,)), ("c", (2,)), ("d", (0,)))


def test_pair_features_have_six_columns_and_set_overlap_terms() -> None:
    profiles = tuple(_profile(f"member-{suffix}", surface) for suffix, surface in ROUND_SURFACES)
    learner = _fitted_learner(profiles)
    features = learner._pair_features(("member-a", "member-b"))
    assert len(features) == 6
    intercept, mean_contribution, product, union, overlap, symmetric = features
    assert intercept == 1.0
    assert mean_contribution == pytest.approx(0.5)
    assert product == pytest.approx(0.25)
    # disjoint surfaces: union 2/4, overlap 0, symmetric difference 2/4
    assert union == pytest.approx(2 / SURFACE_BLOCK_COUNT)
    assert overlap == pytest.approx(0.0)
    assert symmetric == pytest.approx(2 / SURFACE_BLOCK_COUNT)


def test_set_overlap_columns_are_algebraically_dependent() -> None:
    """|union| = |overlap| + |symmetric difference| always holds.

    This is the disclosed cost: the three surface columns are linearly
    dependent, so their individual coefficients are not separately
    interpretable.
    """

    profiles = tuple(_profile(f"member-{suffix}", surface) for suffix, surface in ROUND_SURFACES)
    learner = _fitted_learner(profiles)
    for left, right in (
        ("member-a", "member-b"),
        ("member-a", "member-d"),
        ("member-c", "member-d"),
    ):
        _, _, _, union, overlap, symmetric = learner._pair_features((left, right))
        assert union == pytest.approx(overlap + symmetric)


def test_learner_can_only_rank_redundant_pairs_lower() -> None:
    """The honest claim: redundant pairs are separable, complementary ones are not.

    All complementary pairs share ONE feature row (union=0.5, overlap=0,
    symmetric=0.5) because the columns count coverage but do not identify WHICH
    blocks.  Only the redundant (overlapping) pair differs.  Asserting this
    keeps a future positive result from being over-claimed as "the best
    complementary pair is identified".
    """

    profiles = tuple(_profile(f"member-{suffix}", surface) for suffix, surface in ROUND_SURFACES)
    learner = _fitted_learner(profiles)

    def surface_part(pair):
        return learner._pair_features(pair)[3:]

    # measured surfaces in this round: a={0} b={1} c={2} d={0}
    complementary = [
        surface_part(("member-a", "member-b")),
        surface_part(("member-a", "member-c")),
        surface_part(("member-b", "member-c")),
        surface_part(("member-b", "member-d")),
        surface_part(("member-c", "member-d")),
    ]
    for row in complementary:
        assert row == pytest.approx(complementary[0]), (
            "all disjoint pairs must share one feature row; if this ever fails the "
            "residual limitation changed and the preregistration section 10 must "
            "be revisited rather than the test relaxed"
        )
    redundant = surface_part(("member-a", "member-d"))
    assert redundant != pytest.approx(complementary[0])
    assert redundant[1] > 0.0, "a+d overlap must be visible"


def test_feature_rank_exceeds_one_under_uniform_contribution() -> None:
    """The exact predecessor defect: rank 1 with 3 columns and one row per pair.

    ``feature_rank()`` measures the design matrix of the *observed records*, so
    the learner must actually observe pair records; an observation-free learner
    correctly reports 0 rows.
    """

    from taiji.interaction_groups import InteractionGroupRecord

    profiles = tuple(_profile(f"member-{suffix}", surface) for suffix, surface in ROUND_SURFACES)
    learner = _fitted_learner(profiles)
    records = tuple(
        InteractionGroupRecord(
            group_id=f"group:{left}-{right}",
            member_ids=(left, right),
            source_trace_digest=DIGEST,
            checkpoint_revision=1,
            contribution=0.5,
            interaction=-0.5,
            uncertainty=0.0,
            resource_cost=0.2,
            owner_policy="train-only",
            status="candidate",
            method="factorial-counterfactual+lesion",
        )
        for left, right in (
            ("member-a", "member-b"),
            ("member-a", "member-d"),
            ("member-b", "member-c"),
            ("member-c", "member-d"),
        )
    )
    learner.observe_records(records)
    assert {profile.contribution for profile in profiles} == {0.5}, (
        "this test is only meaningful while contributions stay uniform"
    )
    assert learner.feature_rank() > 1, (
        "uniform contributions must no longer collapse the design to rank 1"
    )


# --------------------------------------------------------------------------- #
# Construction must be inherited verbatim: the comparison isolates the repair
# --------------------------------------------------------------------------- #


def test_construction_constants_inherited_from_predecessor(runner) -> None:
    assert runner.EXPECTED_CONSTRUCTION == {
        "train_episodes": 144,
        "holdout_episodes": 88,
        "removed_episode_count": 32,
        "observed_pair_count": 4,
    }
    assert runner.TRAIN_CONTEXT_COUNT == 8
    assert runner.CONTEXT_COUNT == 12
    assert runner.REPEATS == 2
    assert len(runner.HELD_OUT_PAIRS) == 2
    assert len(runner.MEMBER_IDS) == 4


def test_gate_reports_construction_equivalence(runner) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "construction_equivalence" in source
    assert "blocked_at_entry_audit" in source
    assert "EXPECTED_CONSTRUCTION" in source


def test_gate_holds_out_the_same_pairs_as_predecessor(runner) -> None:
    """A different held-out pair would change two things at once."""

    assert runner.HELD_OUT_PAIR_INDICES == ((0, 2), (1, 3))


# --------------------------------------------------------------------------- #
# Outcome classification: repair-ineffective must not be read as capability
# --------------------------------------------------------------------------- #


def test_outcome_classifier_separates_repair_ineffective_from_signal_constant(runner) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "representation_effective" in source
    assert "representation_repair_ineffective" in source
    # the ineffective branch must be tested BEFORE the capability branches
    ineffective_idx = source.index('outcome = "representation_repair_ineffective"')
    constant_idx = source.index('outcome = "transfer_signal_constant"')
    assert ineffective_idx < constant_idx, (
        "an ineffective repair has not been tested for capability, so it must be "
        "classified before transfer_signal_constant"
    )


def test_representation_gate_requires_all_three_positive_inputs(runner) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'representation["feature_rank"] > 1' in source
    assert 'representation["surface_columns_any_nonzero"]' in source
    assert 'representation["prediction_distinctness"] > 1' in source


def test_gate_declares_what_it_does_not_fix(runner) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "does_not_fix" in source
    assert "block-3 remains" in source
    assert "discriminating_fraction stays 0.75" in source


def test_gate_discloses_the_residual_limitation(runner) -> None:
    """Section 10 of the preregistration must not be quietly dropped.

    Asserted on the produced audit mapping rather than on the source text: a
    source-text match would pin line-wrap artefacts instead of the disclosure
    that actually reaches the report.
    """

    learner = InteractionGroupTransferLearner(
        ridge=0.1, minimum_utility=-1.0e9, maximum_uncertainty=1.0e9
    )
    profiles = tuple(_profile(f"member-{suffix}", surface) for suffix, surface in ROUND_SURFACES)
    learner.observe_members(profiles)
    audit = runner._representation_audit(
        learner, profiles, (("member-a", "member-b"), ("member-a", "member-d"))
    )
    assert "residual_limitation" in audit
    disclosure = audit["residual_limitation"]
    assert "do not identify WHICH blocks" in " ".join(disclosure.split())
    assert "redundant combinations are avoided" in " ".join(disclosure.split())
    # the disclosed cost must also be present: the surface columns are dependent
    assert "residual_rmse" in audit
    # feature_width is the fitted coefficient count, so it is zero until records
    # are observed; the six-column claim is covered by the _pair_features test.
    assert "surface_columns" in audit
    assert "surface_block_count" in audit

def test_report_path_does_not_overwrite_any_predecessor(runner) -> None:
    assert runner.DEFAULT_REPORT.name == (
        "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
    )
    for predecessor in (
        "taiji_p5_2c_unseen_combination_transfer_20260913.json",
        "taiji_p5_2c_prime_unseen_combination_transfer_20260913.json",
        "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json",
        "taiji_p5_2b_group_causal_corpora_20260913.json",
    ):
        assert runner.DEFAULT_REPORT.name != predecessor
    assert "TRIPLE_PRIME" in runner.PREREGISTRATION


# --------------------------------------------------------------------------- #
# Delegation: only predicate-pure helpers may be inherited
# --------------------------------------------------------------------------- #


def test_local_helpers_needed_for_this_gates_own_constants_are_not_delegated(runner) -> None:
    """Delegating a helper that reads the parent's globals judges the wrong corpus.

    ``_joint_counts`` reads ``_train_context_ids``; ``_entry_audit`` must see
    THIS gate's ``family_coverage``; ``_score_pair`` is a closure over the
    predecessor's ``object_index``.  All three are implemented locally.
    """

    source = RUNNER.read_text(encoding="utf-8")
    assert "p52cpp._joint_counts(" not in source
    assert "p52cpp._score_pair(" not in source
    assert "p52cpp._entry_audit(" not in source
    # and they must be defined here
    assert "def _joint_counts(" in source
    assert "def _score_pair(" in source
    assert "def _entry_audit(" in source


def test_lesion_control_zeroes_only_the_added_columns(runner) -> None:
    """C1 must isolate the repair, not weaken the learner overall."""

    profiles = tuple(_profile(f"member-{suffix}", surface) for suffix, surface in ROUND_SURFACES)
    learner = _fitted_learner(profiles)
    learner._coefficients = (0.4, 0.1, 0.05, 0.2, -0.3, 0.1)
    lesioned = runner._lesion_surface_columns(learner)
    assert lesioned._coefficients[:3] == (0.4, 0.1, 0.05)
    assert lesioned._coefficients[3:] == (0.0, 0.0, 0.0)


def test_entry_audit_requires_multi_block_pair_gain(runner) -> None:
    """The added condition: without complementary structure the repair is untestable."""

    train_ids = sorted(runner._train_context_ids())
    held = {frozenset(p) for p in runner.HELD_OUT_PAIRS}
    train = []
    holdout = []
    for context_id in train_ids:
        for cell in runner.CELL_MEMBER_SETS:
            if frozenset(cell) in held:
                continue
            train.append(_episode(context_id, tuple(cell)))
    for index in range(2):
        for cell in runner.CELL_MEMBER_SETS:
            holdout.append(_episode(f"hold-{index}", tuple(cell)))
    corpus = InteractionTraceCorpus(train=tuple(train), holdout=tuple(holdout))

    from taiji.interaction_groups import InteractionGroupEvaluator

    records = InteractionGroupEvaluator().train_only_candidates(corpus)
    ledger = [
        {"episode_id": f"drop:{i}", "context_id": ctx, "member_ids": []}
        for i, ctx in enumerate(train_ids)
        for _ in range(runner.REPEATS * 2)
    ]

    # A coverage table with positive gain on only ONE block per pair must fail.
    flat_coverage = {
        "rows": [
            {
                "context_id": f"c{i}",
                "block": 0,
                "member_ids": ["member-a", "member-b"],
                "realized_pair_gain": 1.0,
            }
            for i in range(3)
        ]
    }
    flat = runner._entry_audit(
        corpus,
        removed=ledger,
        train_only_records=records,
        intervention_reality={
            "interventions_happened": True,
            "non_baseline_zero_step_total": 0,
        },
        family_coverage=flat_coverage,
    )
    assert flat["passed"] is False
    assert "corpus_carries_multi_block_pair_gain" in {
        c["condition"] for c in flat["conditions"]
    }

    # A coverage table that spans two blocks must pass.
    spread_coverage = {
        "rows": [
            {
                "context_id": "c0",
                "block": 1,
                "member_ids": ["member-b", "member-c"],
                "realized_pair_gain": 1.0,
            },
            {
                "context_id": "c1",
                "block": 2,
                "member_ids": ["member-b", "member-c"],
                "realized_pair_gain": 1.0,
            },
        ]
    }
    spread = runner._entry_audit(
        corpus,
        removed=ledger,
        train_only_records=records,
        intervention_reality={
            "interventions_happened": True,
            "non_baseline_zero_step_total": 0,
        },
        family_coverage=spread_coverage,
    )
    assert spread["passed"] is True, spread["conditions"]
    assert spread["multi_block_pairs"] == ["member-b+member-c"]

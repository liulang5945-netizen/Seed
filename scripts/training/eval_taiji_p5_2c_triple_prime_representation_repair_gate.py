"""P5.2c''' representation-repair Gate: capability-surface profiles (route A).

Route A of the decision prompt.  P5.2c'' repaired the *instrument* and returned
``transfer_signal_constant``: two verified-disjoint held-out combinations, both
without gain.  The instrument is therefore sound, so the remaining explanation is
the representation:

* every member's ``contribution`` is the same pooled scalar (all 0.5), because
  ``build_member_evidence`` averages ``singleton - baseline`` across every
  context, which destroys the block structure;
* ``_pair_features`` was ``(1, (c1+c2)/2, c1*c2)``, so identical contributions
  made every pair row identical -- a rank-1 design for a 3-column model;
* measured directly: the fit returns coefficients ``(-0.375, 0, 0)`` and predicts
  the same value for all six pairs.

The underlying structure is in fact learnable.  Realized gain by block:

    a+b: {0:2}                 b+c: {1:2, 2:2}
    a+c: {0:2}                 b+d: {0:2, 1:2}
    a+d: {0:2}                 c+d: {2:2}

This gate changes ONE thing: the profile gains a capability ``surface`` (which
blocks the member succeeds on) and the pair features gain three set-overlap
columns.  Evidence construction, held-out pairs, controls, thresholds and the
budget are all inherited unchanged from P5.2c'' so the comparison is clean.

What this gate deliberately does NOT do
---------------------------------------
It does not add per-block indicator columns.  The three surface columns capture
*how many* blocks are covered and *how much* they overlap, not *which* blocks.
All four complementary pairs therefore share a single feature row:

    a+b {0,1} -> (0.5, 0.0, 0.5)
    a+c {0,2} -> (0.5, 0.0, 0.5)
    b+c {1,2} -> (0.5, 0.0, 0.5)
    b+d {0,1} -> (0.5, 0.0, 0.5)
    a+d {0}   -> (0.25, 0.25, 0.0)   <- only the redundant pair differs

So a positive result may only claim that the learner learned to avoid *redundant*
combinations -- never that it can rank complementary ones.  Preregistration
section 10 states this; it must not be dropped from the result report.

Preregistration: plans/reference/
  M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_PREREGISTRATION_20260913.md (frozen)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import eval_taiji_p5_2a_predictive_execution_gate as p52a  # noqa: E402
import eval_taiji_p5_2b_group_causal_corpora_gate as p52b  # noqa: E402
import eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate as p52cpp  # noqa: E402
import eval_taiji_p5_2c_prime_unseen_combination_transfer_gate as p52cp  # noqa: E402

from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.interaction_group_transfer import (  # noqa: E402
    INTERACTION_GROUP_TRANSFER_MODEL_REVISION,
    MEMBER_EVIDENCE_VERSION,
    SURFACE_BLOCK_COUNT,
    InteractionGroupTransferLearner,
    build_member_evidence,
)
from taiji.interaction_groups import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionTraceCorpus,
)

REPORT_FORMAT = "taiji-p5-2c-triple-prime-representation-repair-report-v1"
VERSION = 1
PREREGISTRATION = (
    "plans/reference/"
    "M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_PREREGISTRATION_20260913.md"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
)

# --------------------------------------------------------------------------- #
# Inherited constants.  Bound from the predecessor rather than re-typed, so the
# two gates cannot silently drift apart on thresholds or design.
# --------------------------------------------------------------------------- #

TOTAL_SECONDS_CAP = p52cpp.TOTAL_SECONDS_CAP
MARGIN = p52cpp.MARGIN
TRAIN_CONTEXT_COUNT = p52cpp.TRAIN_CONTEXT_COUNT
CONTEXT_COUNT = p52cpp.CONTEXT_COUNT
REPEATS = p52cpp.REPEATS
HELD_OUT_PAIR_INDICES = p52cpp.HELD_OUT_PAIR_INDICES
HELD_OUT_PAIRS = p52cpp.HELD_OUT_PAIRS
MEMBER_IDS = p52cpp.MEMBER_IDS
CELL_MEMBER_SETS = p52cpp.CELL_MEMBER_SETS
PAIR_MEMBER_SETS = p52cpp.PAIR_MEMBER_SETS
OBSERVED_PAIR_MEMBER_SETS = p52cpp.OBSERVED_PAIR_MEMBER_SETS
CALIBRATION_MINIMUM_SIGN_MATCH_RATE = p52cpp.CALIBRATION_MINIMUM_SIGN_MATCH_RATE
CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR = p52cpp.CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR
STATIC_CHECK_COMMANDS = p52cpp.STATIC_CHECK_COMMANDS
STATIC_CHECK_SCOPE = p52cpp.STATIC_CHECK_SCOPE
RANDOM_CONTROL_SEED = p52cpp.RANDOM_CONTROL_SEED

HELD_OUT_PAIR = HELD_OUT_PAIRS[0]
_held_out_sorted = {tuple(sorted(pair)) for pair in HELD_OUT_PAIRS}

#: Evidence-construction invariants inherited from P5.2c''.  The preregistration
#: requires this gate to reproduce the predecessor's corpus exactly; otherwise a
#: change in gain could not be attributed to the representation change alone.
EXPECTED_CONSTRUCTION = {
    "train_episodes": 144,
    "holdout_episodes": 88,
    "removed_episode_count": 32,
    "observed_pair_count": 4,
}

# Delegated, predicate-pure: these carry no gate-specific constants.
_build_corpus = p52cpp._build_corpus  # two-pair removal, inherited verbatim
_capability_surfaces = p52cpp._capability_surfaces
_block3_audit = p52cpp._block3_audit
_family_coverage = p52cpp._family_coverage
_intervention_reality = p52cpp._intervention_reality
_success_matrix = p52cpp._success_matrix
_select_sum_heuristic = p52cpp._select_sum_heuristic
_lesion_coefficients = p52cpp._lesion_coefficients
_order_candidate_sets = p52cpp._order_candidate_sets
_outcome_index = p52cpp._outcome_index
_cell_mean = p52cpp._cell_mean
_median = p52cpp._median
_write_json = p52cpp._write_json
_sha = p52cpp._sha
# Read-only accessor: names the train partition, carries no gate-specific
# constant and cannot change a decision, so reuse is safe here.
_train_context_ids = p52cp._train_context_ids


def _score_pair(
    pair: tuple[str, ...],
    index: dict[Any, list[float]],
    unseen_contexts: tuple[Any, ...],
) -> dict[str, Any]:
    """Per-context realized scores for one pair, on the unseen contexts.

    Reimplemented here (not delegated) because the predecessor's ``score`` is a
    closure over its own ``object_index`` and ``unseen_contexts``; delegating
    would evaluate contexts this gate never selected.
    """

    per_context: list[dict[str, Any]] = []
    for task in unseen_contexts:
        baseline = _cell_mean(index, task.task_id, ())
        members = [tuple(sorted((member,))) for member in pair]
        values = {member[0]: _cell_mean(index, task.task_id, member) for member in members}
        strongest_member = max(values, key=lambda key: (values[key], key))
        strongest_gain = values[strongest_member] - baseline
        pair_value = _cell_mean(index, task.task_id, tuple(sorted(pair)))
        first, second = members
        interaction = (
            pair_value
            - _cell_mean(index, task.task_id, first)
            - _cell_mean(index, task.task_id, second)
            + baseline
        )
        per_context.append(
            {
                "context_id": task.task_id,
                "member_ids": list(pair),
                "baseline_outcome": baseline,
                "first_outcome": _cell_mean(index, task.task_id, first),
                "second_outcome": _cell_mean(index, task.task_id, second),
                "pair_outcome": pair_value,
                "realized_interaction": interaction,
                "realized_pair_gain": pair_value - baseline,
                "realized_pair_gain_vs_strongest_single": pair_value - values[strongest_member],
                "strongest_singleton_member": strongest_member,
                "strongest_singleton_gain": strongest_gain,
            }
        )
    gains = [item["realized_pair_gain_vs_strongest_single"] for item in per_context]
    interactions = [item["realized_interaction"] for item in per_context]
    return {
        "member_ids": list(pair),
        "per_context": per_context,
        "mean_gain_vs_strongest_single": float(sum(gains) / len(gains)) if gains else 0.0,
        "mean_realized_interaction": (
            float(sum(interactions) / len(interactions)) if interactions else 0.0
        ),
        "contexts_beating_strongest_single": sum(1 for value in gains if value > 0.0),
        "contexts_scored": len(gains),
    }


def _joint_counts(corpus: InteractionTraceCorpus, pair: tuple[str, ...]) -> tuple[int, int]:
    """Joint-cell episode counts for one pair, split into train and holdout.

    Local, not delegated, because the predecessor's version reads its own
    ``_train_context_ids`` through its own parent module.
    """

    target = frozenset(pair)
    train_ids = p52cp._train_context_ids()
    in_train = sum(
        1
        for episode in corpus.train
        if episode.context_id in train_ids and frozenset(episode.member_ids) == target
    )
    in_holdout = sum(
        1
        for episode in corpus.holdout
        if episode.context_id not in train_ids and frozenset(episode.member_ids) == target
    )
    return in_train, in_holdout


def _singleton_counts(corpus: InteractionTraceCorpus, pair: tuple[str, ...]) -> dict[str, int]:
    """Marginal episode count per member, so "unseen" is not "unsupported"."""

    return {
        member: sum(1 for episode in corpus.train if episode.member_ids == (member,))
        for member in pair
    }


def _entry_audit(
    corpus: InteractionTraceCorpus,
    *,
    removed: list[dict[str, Any]],
    train_only_records: tuple[Any, ...],
    intervention_reality: dict[str, Any],
    family_coverage: dict[str, Any],
) -> dict[str, Any]:
    """Hard entry assertions, inherited from P5.2c'' plus one new condition.

    Reimplemented locally rather than delegated because the added
    complementary-power condition consumes this gate's own ``family_coverage``;
    delegating would have evaluated the predecessor's global instead, which is
    a measurement this gate never took.
    """

    observed_pairs = {tuple(sorted(item.member_ids)) for item in train_only_records}
    conditions: list[dict[str, Any]] = []

    def require(name: str, ok: bool, detail: str) -> None:
        if not ok:
            conditions.append({"condition": name, "detail": detail})

    per_pair: dict[str, Any] = {}
    for pair in HELD_OUT_PAIRS:
        label = "+".join(pair)
        train_joint, holdout_joint = _joint_counts(corpus, pair)
        singles = _singleton_counts(corpus, pair)
        per_pair[label] = {
            "member_ids": list(pair),
            "in_observed_records": tuple(sorted(pair)) in observed_pairs,
            "joint_count_in_train": train_joint,
            "joint_count_in_holdout": holdout_joint,
            "singleton_counts_in_train": singles,
        }
        require(
            f"held_out_pair[{label}]_not_observed",
            tuple(sorted(pair)) not in observed_pairs,
            f"{label} is still present in observed_records",
        )
        require(
            f"held_out_pair[{label}]_joint_absent_from_train",
            train_joint == 0,
            f"{label} joint cell occurs {train_joint} times in train",
        )
        require(
            f"held_out_pair[{label}]_joint_present_in_holdout",
            holdout_joint > 0,
            f"{label} joint cell absent from holdout, realized gain not measurable",
        )
        require(
            f"held_out_pair[{label}]_members_marginally_observed",
            all(count > 0 for count in singles.values()),
            f"{label} has a member with no singleton observation left in train, so the "
            "combination would be unsupported rather than unseen",
        )

    missing_observed = tuple(
        sorted(
            item for item in OBSERVED_PAIR_MEMBER_SETS if tuple(sorted(item)) not in observed_pairs
        )
    )
    require(
        "remaining_observed_pairs_intact",
        not missing_observed,
        f"these pairs should have stayed observable but were not estimated: {missing_observed}",
    )
    expected_removed = TRAIN_CONTEXT_COUNT * REPEATS * len(HELD_OUT_PAIRS)
    require(
        "removal_ledger_matches_expectation",
        len(removed) == expected_removed,
        f"removal ledger has {len(removed)} entries, expected {expected_removed}",
    )
    require(
        "interventions_executed",
        bool(intervention_reality["interventions_happened"]),
        f"{intervention_reality['non_baseline_zero_step_total']} non-baseline "
        "intervention episodes executed zero steps",
    )

    # ---- this round's own added entry condition -------------------------
    # The repair is only meaningful if the corpus actually carries pairs whose
    # joint gain is concentrated on more than one block of the training
    # partition.  If every pair were positive on at most one block, "learn to
    # avoid redundancy" and "learn anything at all" would be indistinguishable,
    # and the added set-overlap columns could not change the ranking.
    coverage_by_pair: dict[str, dict[int, int]] = {}
    for row in family_coverage.get("rows", ()):
        if float(row["realized_pair_gain"]) > 0.0:
            label = "+".join(row["member_ids"])
            counts = coverage_by_pair.setdefault(label, {})
            counts[int(row["block"])] = counts.get(int(row["block"]), 0) + 1
    multi_block_pairs = sorted(
        label for label, counts in coverage_by_pair.items() if len(counts) > 1
    )
    require(
        "corpus_carries_multi_block_pair_gain",
        bool(multi_block_pairs),
        "no pair shows positive joint gain on more than one block, so the added "
        "set-overlap columns have no structure to exploit; measured pair coverage: "
        f"{ {k: dict(sorted(v.items())) for k, v in sorted(coverage_by_pair.items())} }",
    )

    return {
        "conditions": conditions,
        "passed": not conditions,
        "held_out_pairs": per_pair,
        "observed_pairs": sorted(list(item) for item in observed_pairs),
        "missing_observed_pairs": [list(item) for item in missing_observed],
        "removed_episode_count": len(removed),
        "removed_episodes": removed,
        "train_episodes": len(corpus.train),
        "holdout_episodes": len(corpus.holdout),
        "pair_coverage_by_block": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(coverage_by_pair.items())
        },
        "multi_block_pairs": multi_block_pairs,
    }


def _fit_learner(corpus: InteractionTraceCorpus) -> tuple[Any, Any, Any]:
    """Fit the v2 surface-aware learner on the inherited corpus."""

    revision = next(iter(corpus.train_checkpoint_revisions))
    profiles = build_member_evidence(
        corpus.train,
        source_trace_digest=corpus.train_trace_digest,
        checkpoint_revision=revision,
    )
    learner = InteractionGroupTransferLearner(
        ridge=0.1, minimum_utility=-1.0e9, maximum_uncertainty=1.0e9
    )
    learner.observe_members(profiles)
    records = InteractionGroupEvaluator().train_only_candidates(corpus)
    learner.observe_records(records)
    return learner, profiles, records


def _lesion_surface_columns(
    learner: InteractionGroupTransferLearner,
) -> InteractionGroupTransferLearner:
    """Control C1: the same learner with the surface columns zeroed.

    This is the control that isolates the *repair* rather than overall learner
    strength.  It is rebuilt from the observed records as a 3-column fit so the
    surface columns cannot contribute by construction.
    """

    lesioned = InteractionGroupTransferLearner(
        ridge=learner.ridge,
        minimum_utility=-1.0e9,
        maximum_uncertainty=1.0e9,
    )
    lesioned.observe_members(learner.profiles)
    lesioned.observe_records(learner.observed_records)
    lesioned._coefficients = tuple(learner._coefficients[:3]) + (0.0, 0.0, 0.0)
    return lesioned


def _representation_audit(
    learner: InteractionGroupTransferLearner,
    profiles: Any,
    candidate_sets: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    """Gate 9(a): prove the repair engaged before judging gain.

    Without this, a failed gain gate is uninterpretable -- the fix might simply
    not have taken effect.  All three conditions must hold.
    """

    rank = learner.feature_rank()
    coefficients = tuple(float(value) for value in learner._coefficients)
    surface_columns = coefficients[-3:]
    predictions: dict[str, float | None] = {}
    for pair in candidate_sets:
        candidate = learner.candidate(pair, allow_observed=True)
        predictions["+".join(pair)] = (
            None if candidate is None else float(candidate.predicted_interaction)
        )
    distinct = len({round(value, 9) for value in predictions.values() if value is not None})
    return {
        "feature_rank": rank,
        "feature_width": len(coefficients),
        "coefficients": list(coefficients),
        "surface_columns": list(surface_columns),
        "surface_columns_any_nonzero": any(abs(value) > 1e-12 for value in surface_columns),
        "prediction_distinctness": distinct,
        "predictions": predictions,
        "surface_per_member": {item.member_id: list(item.surface) for item in profiles},
        "surface_empty_members": sorted(item.member_id for item in profiles if not item.surface),
        "member_evidence_version": MEMBER_EVIDENCE_VERSION,
        "model_revision": INTERACTION_GROUP_TRANSFER_MODEL_REVISION,
        "surface_block_count": SURFACE_BLOCK_COUNT,
        "residual_rmse": float(learner._residual_rmse),
        "baseline_comparison": (
            "predecessor measured rank 1, coefficients (-0.375, 0.0, 0.0) and "
            "prediction_distinctness 1; that is the state this gate repairs"
        ),
        "residual_limitation": (
            "the three surface columns count coverage and overlap but do not "
            "identify WHICH blocks; all four complementary pairs share one "
            "feature row, so a positive result permits only the claim that "
            "redundant combinations are avoided"
        ),
    }


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "growth_admitted": False,
        "can_promote": False,
        "preregistration": PREREGISTRATION,
        "route": "A (representation repair: capability-surface member profiles)",
        "does_not_fix": (
            "the measurement instrument and block-3 reachability. The instrument "
            "was repaired by P5.2c'' and is inherited unchanged; block-3 remains "
            "uniformly unreachable so discriminating_fraction stays 0.75."
        ),
        "static_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": list(STATIC_CHECK_COMMANDS),
            "executed_before_run": True,
        },
    }
    workspace_root = Path(p52cp.tempfile.mkdtemp(prefix="p52cppp-workbench-"))
    replica_root = Path(p52cp.tempfile.mkdtemp(prefix="p52cppp-replica-"))
    try:
        embedder = DocumentEmbedder()
        members = p52b._train_members(embedder)
        contexts = p52a._validation_tasks()[:CONTEXT_COUNT]
        cue_by_task = {task.task_id: embedder.embed([task.goal_text])[0] for task in contexts}

        matrix_episodes = p52cp._execute_matrix(workspace_root, contexts, members, cue_by_task)
        projected = p52cp._project(matrix_episodes)
        corpus, removed = _build_corpus(projected)

        intervention_reality = _intervention_reality(matrix_episodes, contexts)
        capability = _capability_surfaces(matrix_episodes, contexts)
        block3 = _block3_audit(matrix_episodes, contexts)

        train_only_records = InteractionGroupEvaluator().train_only_candidates(corpus)
        family_coverage = _family_coverage(matrix_episodes, contexts)

        entry_audit = _entry_audit(
            corpus,
            removed=removed,
            train_only_records=train_only_records,
            intervention_reality=intervention_reality,
            family_coverage=family_coverage,
        )
        entry_audit["conditions"].extend(
            []
            if capability["held_out_pairs_all_disjoint"]
            else [
                {
                    "condition": "held_out_pairs_capability_disjoint",
                    "detail": (
                        "held-out pairs are not all disjoint by measured capability "
                        f"surface: {capability['held_out_pair_kinds']}"
                    ),
                }
            ]
        )
        entry_audit["passed"] = not entry_audit["conditions"]

        # Construction-equivalence assertion: the preregistration forbids moving
        # the corpus, because a gain change must be attributable to the
        # representation change alone.
        construction = {
            "train_episodes": len(corpus.train),
            "holdout_episodes": len(corpus.holdout),
            "removed_episode_count": len(removed),
            "observed_pair_count": len(train_only_records),
        }
        construction_matches = bool(construction == EXPECTED_CONSTRUCTION)
        payload["construction_equivalence"] = {
            "expected": dict(EXPECTED_CONSTRUCTION),
            "actual": construction,
            "matches": construction_matches,
            "note": (
                "inherited from P5.2c'' verbatim; a mismatch means the comparison "
                "would not isolate the representation change"
            ),
        }

        if not entry_audit["passed"] or not construction_matches:
            payload.update(
                {
                    "status": "blocked",
                    "outcome": "blocked_at_entry_audit",
                    "entry_audit": entry_audit,
                    "capability_surfaces": capability,
                    "block3_audit": block3,
                    "intervention_reality": intervention_reality,
                    "matrix": {
                        "episodes": len(matrix_episodes),
                        "train_episodes": len(corpus.train),
                        "holdout_episodes": len(corpus.holdout),
                    },
                    "growth_admitted": False,
                    "can_promote": False,
                    "interpretation": (
                        "blocked: the inherited construction did not reproduce, or "
                        "the entry audit failed; repair the wiring, do not relax "
                        "the criterion"
                    ),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            _write_json(DEFAULT_REPORT, payload)
            return payload

        observed_pairs = {tuple(sorted(item.member_ids)) for item in train_only_records}
        candidate_sets_ordered = _order_candidate_sets(PAIR_MEMBER_SETS, RANDOM_CONTROL_SEED % 1000)

        learner, profiles, _ = _fit_learner(corpus)
        representation = _representation_audit(learner, profiles, candidate_sets_ordered)

        selected = learner.select(
            candidate_sets_ordered, resource_budget=10.0, unseen_only=True
        )
        if selected is None:
            payload.update(
                {
                    "status": "failed",
                    "outcome": "failed",
                    "entry_audit": entry_audit,
                    "representation": representation,
                    "error": (
                        "learner.select returned None although the entry audit "
                        "confirmed a non-empty unseen surface"
                    ),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            _write_json(DEFAULT_REPORT, payload)
            return payload

        _, candidate = selected
        selected_members = tuple(candidate.member_ids)
        selected_is_held_out = tuple(sorted(selected_members)) in _held_out_sorted

        parent_checkpoint = learner.checkpoint()
        object_index = _outcome_index(matrix_episodes)
        unseen_contexts = contexts[TRAIN_CONTEXT_COUNT:]

        def score(pair: tuple[str, ...]) -> dict[str, Any]:
            return _score_pair(pair, object_index, unseen_contexts)

        held_out_scores = {tuple(pair): score(pair) for pair in HELD_OUT_PAIRS}

        # ---- controls -------------------------------------------------------
        control_sets: dict[str, tuple[tuple[str, ...], ...]] = {}
        strongest_singleton = max(
            PAIR_MEMBER_SETS, key=lambda pair: score(pair)["mean_gain_vs_strongest_single"]
        )
        control_sets["strongest_leftover_pair"] = (strongest_singleton,)
        control_sets["fixed_combination"] = (tuple(sorted(HELD_OUT_PAIR)),)
        control_sets["no_learning"] = (
            candidate_sets_ordered[
                (RANDOM_CONTROL_SEED * 7) % len(candidate_sets_ordered)
            ],
        )
        control_sets["random_combination"] = (
            candidate_sets_ordered[
                (RANDOM_CONTROL_SEED * 13) % len(candidate_sets_ordered)
            ],
        )

        lesion = _lesion_surface_columns(learner)
        lesion_selected = lesion.select(
            candidate_sets_ordered, resource_budget=10.0, unseen_only=True
        )
        if lesion_selected is not None:
            control_sets["lesion_learner"] = (tuple(lesion_selected[1].member_ids),)

        # C5 ranks unseen pairs by summed profile contribution with no relation
        # fit, so it must be handed the member profiles, not the fitted learner.
        sum_heuristic = _select_sum_heuristic(
            profiles, candidate_sets_ordered, resource_budget=10.0
        )
        if sum_heuristic is not None:
            control_sets["train_only_simple_regression"] = (
                tuple(sum_heuristic[1].member_ids),
            )

        pair_control_scores = {name: score(value[0]) for name, value in control_sets.items()}

        # per-context strongest singleton (control C2)
        singleton_per_context = []
        for task in unseen_contexts:
            baseline = _cell_mean(object_index, task.task_id, ())
            best_member = None
            best_value = None
            for member in MEMBER_IDS:
                value = _cell_mean(object_index, task.task_id, (member,))
                if best_value is None or value > best_value:
                    best_value = value
                    best_member = member
            singleton_per_context.append(
                {"context_id": task.task_id, "strongest_singleton_gain": best_value - baseline,
                 "strongest_singleton_member": best_member}
            )
        singleton_gains = [item["strongest_singleton_gain"] for item in singleton_per_context]
        singleton_mean = float(sum(singleton_gains) / len(singleton_gains))

        control_scores: dict[str, Any] = {
            **pair_control_scores,
            "strongest_singleton": {
                "kind": "per_context_singleton",
                "member_ids": ["<per-context strongest singleton>"],
                "per_context": singleton_per_context,
                "mean_gain_vs_strongest_single": singleton_mean,
                "contexts_beating_strongest_single": 0,
                "contexts_scored": len(singleton_per_context),
            },
        }
        # C4 coincides with the correct answer by construction, so it is excluded
        # from the margin-bearing set.
        margin_controls = {
            name: value
            for name, value in control_scores.items()
            if name != "fixed_combination"
        }
        strongest_control = max(
            margin_controls.items(), key=lambda item: item[1]["mean_gain_vs_strongest_single"]
        )
        best_control_gain = strongest_control[1]["mean_gain_vs_strongest_single"]
        fixed_control_gain = control_scores["fixed_combination"]["mean_gain_vs_strongest_single"]

        object_score = held_out_scores[tuple(sorted(selected_members))]
        object_gain = object_score["mean_gain_vs_strongest_single"]
        margin_cleared = bool(object_gain > best_control_gain + MARGIN)
        collaboration_holds = bool(object_score["mean_realized_interaction"] > 0.0)
        any_held_out_positive = any(
            value["mean_gain_vs_strongest_single"] > 0.0 for value in held_out_scores.values()
        )

        held_out_summary = {
            "+".join(pair): {
                **value,
                "kind": capability["pairs"]["+".join(pair)]["kind"],
                "is_selected_object": tuple(sorted(pair)) == tuple(sorted(selected_members)),
            }
            for pair, value in held_out_scores.items()
        }

        # ---- calibration ----------------------------------------------------
        comparable = []
        for pair in PAIR_MEMBER_SETS:
            predicted = learner.candidate(pair, allow_observed=False)
            realized = score(pair)
            held_out = tuple(sorted(pair)) in _held_out_sorted
            if predicted is None:
                comparable.append(
                    {
                        "member_ids": list(pair),
                        "held_out": held_out,
                        "seen_in_train_evidence": True,
                        "predicted_interaction": None,
                        "realized_interaction": realized["mean_realized_interaction"],
                        "absolute_error": None,
                        "sign_matches": None,
                    }
                )
                continue
            error = abs(
                float(predicted.predicted_interaction)
                - float(realized["mean_realized_interaction"])
            )
            comparable.append(
                {
                    "member_ids": list(pair),
                    "held_out": held_out,
                    "seen_in_train_evidence": False,
                    "predicted_interaction": float(predicted.predicted_interaction),
                    "realized_interaction": realized["mean_realized_interaction"],
                    "absolute_error": error,
                    "sign_matches": (
                        (float(predicted.predicted_interaction) > 0.0)
                        == (realized["mean_realized_interaction"] > 0.0)
                    ),
                }
            )
        unseen_rows = [row for row in comparable if row["held_out"]]
        errors = [float(row["absolute_error"]) for row in unseen_rows]
        sign_hits = [bool(row["sign_matches"]) for row in unseen_rows]
        sign_match_rate = float(sum(sign_hits) / len(sign_hits)) if sign_hits else 0.0
        median_absolute_error = _median(errors)
        calibration = {
            "unseen_sample_count": len(unseen_rows),
            "comparable_pair_count": len(unseen_rows),
            "mean_absolute_error": (
                float(sum(errors) / len(errors)) if errors else None
            ),
            "median_absolute_error": median_absolute_error,
            "sign_match_rate": sign_match_rate,
            "small_sample_caveat": (
                "the unseen surface has cardinality 2 by construction; these pooled "
                "statistics rest on those two pairs and are not an independent sample"
            ),
            "thresholds": {
                "minimum_sign_match_rate": CALIBRATION_MINIMUM_SIGN_MATCH_RATE,
                "maximum_median_absolute_error": CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR,
                "fixed_at_freeze_time": True,
            },
            "rows": comparable,
        }
        calibration_ok = bool(
            unseen_rows
            and sign_match_rate >= CALIBRATION_MINIMUM_SIGN_MATCH_RATE
            and median_absolute_error is not None
            and median_absolute_error <= CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR
        )

        total_wall = time.perf_counter() - started

        # ---- mechanical checks ---------------------------------------------
        replica_episodes = p52cp._execute_matrix(
            replica_root, contexts, members, cue_by_task
        )

        # Inherited verbatim from P5.2c'': raw matrix episodes carry task_id but
        # no context_id, so episode_id is the identity that is available here.
        def surface_of(episodes: Any) -> list[Any]:
            return sorted(
                (item["episode_id"], item["success"], item["stop_reason"], item["resource_cost"])
                for item in episodes
            )

        replica_consistent = bool(surface_of(replica_episodes) == surface_of(matrix_episodes))
        recovery = p52cpp._recovery_probe(parent_checkpoint, candidate_sets_ordered)
        rejections = p52cpp._rejection_probe(
            corpus, train_only_records, profiles, candidate_sets_ordered
        )

        # Inherited verbatim from P5.2c'': these three facts live on the recorded
        # *steps*, not on the episode row.
        executed_entries = [
            step for episode in matrix_episodes for step in episode["steps"] if step.get("executed")
        ]
        provenance_ok = all(
            set(step.get("provenance", {}).values())
            <= {"goal_state", "world_state", "transaction_token", "goal_state+world_state"}
            for step in executed_entries
        )
        safety_violations = sum(
            1
            for episode in matrix_episodes
            for step in episode["steps"]
            if step.get("safety_violation")
        )

        prediction_binding = {
            "bound_before_scoring": True,
            "parent_checkpoint_digest": parent_checkpoint["checkpoint_digest"],
            "model_digest": learner.model_digest,
            "source_trace_digest": learner.source_trace_digest,
            "checkpoint_revision": learner.checkpoint_revision,
            "selected_group_id": candidate.group_id,
            "selected_member_ids": list(selected_members),
            "selected_is_a_held_out_pair": selected_is_held_out,
            "predicted_interaction": float(candidate.predicted_interaction),
            "uncertainty": float(candidate.uncertainty),
            "resource_cost": float(candidate.resource_cost),
            "support": int(candidate.support),
            "selection_utility": float(candidate.utility),
            "candidate_sets_considered": [
                list(item) for item in reversed(candidate_sets_ordered)
            ],
            "observed_pair_count": len(observed_pairs),
            "held_out_pairs": [list(pair) for pair in HELD_OUT_PAIRS],
        }

        evidence_admissible = bool(
            all(
                record.status == "candidate"
                and record.holdout_interaction is None
                and record.holdout_recovery_effect is None
                for record in train_only_records
            )
            and len(train_only_records) > 0
            and intervention_reality["interventions_happened"]
        )
        unseen_identity = bool(
            tuple(sorted(selected_members)) not in observed_pairs
            and all(member in {item.member_id for item in profiles} for member in selected_members)
            and bool(entry_audit["held_out_pairs"])
            and all(
                item["in_observed_records"] is False
                and item["joint_count_in_train"] == 0
                and item["joint_count_in_holdout"] > 0
                for item in entry_audit["held_out_pairs"].values()
            )
        )
        holdout_disjoint = bool(
            {item.context_id for item in corpus.train}.isdisjoint(
                {item.context_id for item in corpus.holdout}
            )
            and len(corpus.holdout)
            == (CONTEXT_COUNT - TRAIN_CONTEXT_COUNT) * REPEATS * len(CELL_MEMBER_SETS)
        )

        representation_effective = bool(
            representation["feature_rank"] > 1
            and representation["surface_columns_any_nonzero"]
            and representation["prediction_distinctness"] > 1
        )

        gates = {
            "static_checks": True,
            "evidence_admissibility": evidence_admissible,
            "unseen_combination_identity": unseen_identity,
            "unseen_context_holdout": holdout_disjoint,
            "label_opaqueness": True,
            "real_execution": bool(
                executed_entries
                and provenance_ok
                and safety_violations == 0
                and intervention_reality["interventions_happened"]
                and selected_is_held_out
            ),
            "rejection_recovery": bool(
                all(rejections.values()) and recovery["selection_reproduced"]
            ),
            "prediction_binding_and_calibration": bool(
                prediction_binding["bound_before_scoring"]
                and prediction_binding["parent_checkpoint_digest"]
                == parent_checkpoint["checkpoint_digest"]
                and calibration_ok
            ),
            "representation_and_transfer": bool(
                representation_effective
                and margin_cleared
                and collaboration_holds
                and any_held_out_positive
                and replica_consistent
                and total_wall <= TOTAL_SECONDS_CAP
            ),
        }
        mechanical = {
            key: value for key, value in gates.items() if key != "representation_and_transfer"
        }
        if not all(mechanical.values()):
            outcome = "failed"
        elif not all(gates.values()):
            if not representation_effective:
                outcome = "representation_repair_ineffective"
            elif not any_held_out_positive:
                outcome = "transfer_signal_constant"
            else:
                outcome = "transfer_no_gain"
        else:
            outcome = "unseen_combination_transfer_supported"

        payload.update(
            {
                "status": "completed",
                "outcome": outcome,
                "experiment_passed": bool(
                    outcome == "unseen_combination_transfer_supported"
                ),
                "design": {
                    "members": list(MEMBER_IDS),
                    "member_kind": "family-specialist procedural readouts (same instances as P5.2b)",
                    "composition": "dual-predict-select (P5.2b frozen revision)",
                    "contexts": CONTEXT_COUNT,
                    "train_contexts": TRAIN_CONTEXT_COUNT,
                    "unseen_contexts": CONTEXT_COUNT - TRAIN_CONTEXT_COUNT,
                    "repeats": REPEATS,
                    "cells": ["(F,F)", "(T,F)", "(F,T)", "(T,T)"],
                    "margin": MARGIN,
                    "held_out_pairs": [list(pair) for pair in HELD_OUT_PAIRS],
                    "held_out_pair_opaque_indices": [list(idx) for idx in HELD_OUT_PAIR_INDICES],
                    "structural_change": (
                        "capability-surface member profiles plus three set-overlap "
                        "pair features; evidence construction is inherited from "
                        "P5.2c'' verbatim so the comparison isolates the "
                        "representation change"
                    ),
                    "member_set_widening_rejected_because": (
                        "P5.2a has exactly four train template families, one per "
                        "member, so extra members would duplicate an existing "
                        "capability surface and add redundancy rather than "
                        "complementarity"
                    ),
                    "candidate_pairs": [list(item) for item in PAIR_MEMBER_SETS],
                    "observed_pairs": sorted(list(item) for item in observed_pairs),
                    "learner_observes_train_only_records": True,
                    "recovery_effect_disclosure": (
                        "not measured this gate; recorded as 0 without fabrication"
                    ),
                    "known_cost_disclosure": (
                        "the three surface columns are algebraically dependent "
                        "(|union| = |overlap| + |symmetric difference|), so single "
                        "coefficients must not be interpreted, only predictions. "
                        "P1* is structurally designated, so control C4 coincides "
                        "with the correct answer and is excluded from the "
                        "margin-bearing controls. All complementary pairs share one "
                        "feature row, so a pass claims avoidance of redundant "
                        "combinations only."
                    ),
                },
                "entry_audit": entry_audit,
                "capability_surfaces": capability,
                "block3_audit": block3,
                "representation": representation,
                "representation_gate_inputs": {
                    "feature_rank": representation["feature_rank"],
                    "surface_columns_any_nonzero": representation["surface_columns_any_nonzero"],
                    "prediction_distinctness": representation["prediction_distinctness"],
                    "representation_effective": representation_effective,
                },
                "prediction_binding": prediction_binding,
                "object": {"member_ids": list(selected_members), "score": object_score},
                "held_out_pairs_scored": held_out_summary,
                "controls": {
                    name: {
                        "member_ids": list(value[0]),
                        "kind": "pair_selection",
                        "score": pair_control_scores[name],
                    }
                    for name, value in control_sets.items()
                }
                | {
                    "strongest_singleton": control_scores["strongest_singleton"],
                },
                "control_summary": {
                    "strongest_control": strongest_control[0],
                    "strongest_control_gain": best_control_gain,
                    "object_gain": object_gain,
                    "margin": MARGIN,
                    "required": best_control_gain + MARGIN,
                    "margin_cleared": margin_cleared,
                    "collaboration_holds": collaboration_holds,
                    "any_held_out_positive": any_held_out_positive,
                    "fixed_combination_gain_excluded_from_margin": fixed_control_gain,
                    "c2_strongest_singleton_gain": singleton_mean,
                    "margin_bearing_controls": sorted(margin_controls),
                },
                "calibration": calibration,
                "family_coverage": _family_coverage(matrix_episodes, contexts),
                "success_matrix": _success_matrix(matrix_episodes, contexts),
                "matrix": {
                    "episodes": len(matrix_episodes),
                    "train_episodes": len(corpus.train),
                    "holdout_episodes": len(corpus.holdout),
                    "stop_reason_classes": dict(
                        sorted(
                            (p52cpp.Counter(
                                item["stop_reason"] for item in matrix_episodes
                                if item.get("stop_reason")
                            )).items()
                        )
                    ),
                },
                "intervention_reality": intervention_reality,
                "member_evidence": {
                    "profiles": len(profiles),
                    "member_ids": sorted(item.member_id for item in profiles),
                    "contributions": {item.member_id: item.contribution for item in profiles},
                    "surfaces": {item.member_id: list(item.surface) for item in profiles},
                    "contribution_uniform": len(
                        {round(float(item.contribution), 12) for item in profiles}
                    )
                    == 1,
                    "surfaces_all_equal": len({tuple(item.surface) for item in profiles}) == 1,
                    "note": (
                        "contributions may still be uniform; the surface is the "
                        "added discriminative signal this gate supplies"
                    ),
                },
                "recovery": recovery,
                "rejections": rejections,
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; representation_effective="
                    f"{representation_effective} (rank {representation['feature_rank']}, "
                    f"distinctness {representation['prediction_distinctness']}); "
                    f"object={'+'.join(selected_members)} gain={round(object_gain, 6)} vs "
                    f"strongest control {strongest_control[0]}="
                    f"{round(best_control_gain, 6)} (required "
                    f"{round(best_control_gain + MARGIN, 6)}); held_out_pairs="
                    f"{held_out_summary}"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    finally:
        p52cp.shutil.rmtree(workspace_root, ignore_errors=True)
        p52cp.shutil.rmtree(replica_root, ignore_errors=True)
    _write_json(DEFAULT_REPORT, payload)
    return payload


def main() -> int:
    result = run_gate()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "route": result.get("route"),
                "gates_failed": sorted(
                    key for key, value in (result.get("gates") or {}).items() if not value
                ),
                "entry_audit_passed": (result.get("entry_audit") or {}).get("passed"),
                "construction_equivalence": result.get("construction_equivalence"),
                "held_out_pair_kinds": (result.get("capability_surfaces") or {}).get(
                    "held_out_pair_kinds"
                ),
                "representation_gate_inputs": result.get("representation_gate_inputs"),
                "member_evidence": result.get("member_evidence"),
                "object": (result.get("object") or {}).get("member_ids"),
                "held_out_pairs_scored": result.get("held_out_pairs_scored"),
                "control_summary": result.get("control_summary"),
                "calibration": {
                    key: value
                    for key, value in (result.get("calibration") or {}).items()
                    if key
                    in (
                        "sign_match_rate",
                        "median_absolute_error",
                        "comparable_pair_count",
                        "unseen_sample_count",
                    )
                },
                "elapsed_seconds": result.get("elapsed_seconds"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

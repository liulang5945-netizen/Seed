"""P5.2c' unseen-combination transfer Gate.

This gate answers the question its predecessor (P5.2c) could not ask.  P5.2c was
blocked at entry because with four members all ``C(4,2)=6`` pairs are estimated
by ``train_only_candidates``, so the unseen-combination surface was empty.  The
entry audit then showed why widening the member set cannot help:

``taiji/interaction_groups.py`` enumerates every ``C(n,2)`` pair and calls
``_estimate_pair``; that helper returns an estimate as soon as **any single**
train context has all four factorial cells populated.  Adding members therefore
only adds more *observed* pairs.  Measured: n=4/5/6 all yield zero unseen pairs.

The only mechanism that produces a non-empty unseen surface is to remove the
joint ``(T,T)`` cell of one designated pair from **every** train context, so no
context can complete the factorial and ``_estimate_pair`` returns ``None``.
Removing it from only some contexts is not enough -- one surviving complete
context is sufficient to estimate the pair.

That removal happens in the evidence-construction layer only.  It shapes what
the learner may see; it never touches holdout execution or scoring.  The pair's
two members stay fully observed in train as singletons and as baselines, so the
learner holds complete *marginal* evidence and lacks only the *joint* effect.
That is precisely what "unseen combination" means.

Preregistration: plans/reference/
  M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md (frozen)

Design summary
--------------
* Same four family-specialist procedural readouts, same 12 contexts x 11 cells x
  2 repeats real contract execution as P5.2b.  No new member, scenario, corpus,
  or provider.
* ``P*`` is opaque here: lexical index (0,3) over ``MEMBER_IDS``, i.e. the pair
  that P5.2b admitted on the strength of empty-event episodes.
* The single structural change: drop ``P*``'s ``(T,T)`` from train contexts
  ``[0:8]`` (train 176 -> 160 episodes).  Holdout keeps it (88 episodes), so the
  realized gain is measured by real execution.
* MARGIN stays 0.15 (the P5.2a frozen constant); only the metric object changes.
* ``interventions_happened`` from the P5.2b repair is reused as a hard entry
  condition: any non-baseline cell that executed zero steps fails the gate
  outright rather than being scored.

Known cost, disclosed in the report and the preregistration
-----------------------------------------------------------
Because ``P*`` is structurally the *only* unseen combination, the candidate
surface has cardinality 1 and control C4 (fixed combination) coincides with the
correct answer.  This gate's pass therefore claims only "prediction-realization
consistency plus superiority over no-learning/random/singleton controls".  It
does **not** claim the learner can choose among several unseen combinations.

Member ids stay opaque; the semantic mapping lives only in this runner's config
section and never enters learner or evaluator inputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import eval_taiji_p5_2a_predictive_execution_gate as p52a  # noqa: E402
import eval_taiji_p5_2b_group_causal_corpora_gate as p52b  # noqa: E402

from instruments.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.interaction_group_learning import InteractionGroupSelection  # noqa: E402
from taiji.interaction_group_transfer import (  # noqa: E402
    InteractionGroupTransferCandidate,
    InteractionGroupTransferLearner,
    build_member_evidence,
)
from taiji.interaction_groups import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupRecord,
    InteractionTraceCorpus,
    InteractionTraceEpisode,
)
from taiji.procedural_memory import ProceduralSequenceLearner  # noqa: E402

REPORT_FORMAT = "taiji-p5-2c-prime-unseen-combination-transfer-report-v1"
VERSION = 1
PREREGISTRATION = (
    "plans/reference/M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_prime_unseen_combination_transfer_20260913.json"
)

TOTAL_SECONDS_CAP = 900.0
# inherited unchanged from the P5.2a frozen constant and the P5.2b comparison
# discipline; the measured object changes, the threshold does not.
MARGIN = 0.15
TRAIN_CONTEXT_COUNT = 8
CONTEXT_COUNT = 12
REPEATS = 2
TRACE_REVISION = 1
RANDOM_CONTROL_SEED = 20260913

# Gate 8 calibration thresholds, fixed at freeze time in the preregistration and
# never tuned against measurements.
CALIBRATION_MINIMUM_SIGN_MATCH_RATE = 0.5
CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR = 0.35

MEMBER_IDS = p52b.MEMBER_IDS
CELL_MEMBER_SETS = p52b.CELL_MEMBER_SETS
PAIR_MEMBER_SETS: tuple[tuple[str, ...], ...] = tuple(
    item for item in CELL_MEMBER_SETS if len(item) == 2
)
SINGLETON_MEMBER_SETS: tuple[tuple[str, ...], ...] = tuple(
    item for item in CELL_MEMBER_SETS if len(item) == 1
)

# The single held-out combination, expressed opaquely as lexical indices into
# MEMBER_IDS.  It is written as indices deliberately: the reader must not be able
# to reach the semantic family mapping from the object identity.
HELD_OUT_PAIR_INDEX: tuple[int, int] = (0, 3)
HELD_OUT_PAIR: tuple[str, ...] = tuple(
    sorted((MEMBER_IDS[HELD_OUT_PAIR_INDEX[0]], MEMBER_IDS[HELD_OUT_PAIR_INDEX[1]]))
)
# fixed combination control selects the designated pair; disclosed in the report
# as the reason gate 9 does not rest on "> C4" alone.
FIXED_COMBINATION: tuple[str, ...] = HELD_OUT_PAIR

OBSERVED_PAIR_MEMBER_SETS: tuple[tuple[str, ...], ...] = tuple(
    item for item in PAIR_MEMBER_SETS if tuple(sorted(item)) != HELD_OUT_PAIR
)

STATIC_CHECK_SCOPE = (
    "taiji/interaction_groups.py",
    "taiji/interaction_group_transfer.py",
    "taiji/interaction_group_learning.py",
    "seed_platform/workbench.py",
    "scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py",
    "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
    "scripts/training/eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py",
    "tests/taiji_native/test_intervention_reality_gate.py",
    "tests/taiji_native/test_p5_2c_prime_unseen_combination_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check scripts/training tests/taiji_native (and full repo)",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
    "python -m pytest tests/taiji_native (frozen baseline; existing debts registered "
    "in plans/active/roadmap/05_TECH_DEBT_REGISTER.md, no new failures)",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ids(members: tuple[str, ...]) -> str:
    return "-".join(members) if members else "none"


# --------------------------------------------------------------------------- #
# Matrix execution (delegates to the frozen P5.2b runner)
# --------------------------------------------------------------------------- #


def _execute_matrix(
    root: Path,
    contexts: tuple[p52a.Task, ...],
    members: dict[str, ProceduralSequenceLearner],
    cue_by_task: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Run the full frozen matrix.  Unmodified delegation: identical contract path."""

    return p52b._execute_matrix(root, contexts, members, cue_by_task)


def _project(episodes: list[dict[str, Any]]) -> tuple[InteractionTraceEpisode, ...]:
    return p52b._project(episodes)


# --------------------------------------------------------------------------- #
# The single structural change: remove P*'s joint cell from the train partition
# --------------------------------------------------------------------------- #


def _train_context_ids() -> frozenset[str]:
    return frozenset(f"p52a-validation-{100 + offset:03d}" for offset in range(TRAIN_CONTEXT_COUNT))


def _episode_is_held_out_joint(episode: InteractionTraceEpisode) -> bool:
    """True when this episode is P*'s (T,T) cell.

    The projected episode carries exactly the trace events of members that were
    genuinely called.  A ``(T,T)`` cell for ``P*`` is therefore identified by the
    event owner set being exactly ``P*``.  Using the event-derived ``member_ids``
    keeps the test on the same quantity ``_estimate_pair`` consumes.
    """

    return frozenset(episode.member_ids) == frozenset(HELD_OUT_PAIR)


def _build_corpus(
    projected: tuple[InteractionTraceEpisode, ...],
) -> tuple[InteractionTraceCorpus, list[dict[str, Any]]]:
    """Split into train/holdout and remove ``P*``'s joint cell from train only.

    Returns the corpus and the removal ledger, which is reported verbatim so the
    evidence construction is auditable rather than implied.
    """

    train_ids = _train_context_ids()
    removed: list[dict[str, Any]] = []
    train: list[InteractionTraceEpisode] = []
    holdout: list[InteractionTraceEpisode] = []
    for episode in projected:
        in_train = episode.context_id in train_ids
        if not in_train:
            holdout.append(episode)
            continue
        if _episode_is_held_out_joint(episode):
            removed.append(
                {
                    "episode_id": episode.episode_id,
                    "context_id": episode.context_id,
                    "member_ids": list(episode.member_ids),
                    "outcome": episode.outcome,
                }
            )
            continue
        train.append(episode)
    corpus = InteractionTraceCorpus(train=tuple(train), holdout=tuple(holdout))
    return corpus, removed


def _entry_audit(
    corpus: InteractionTraceCorpus,
    *,
    removed: list[dict[str, Any]],
    evaluator: InteractionGroupEvaluator,
    train_only_records: tuple[InteractionGroupRecord, ...],
    intervention_reality: dict[str, Any],
) -> dict[str, Any]:
    """Hard entry assertions required by preregistration sections 3.2 and 4.

    Any failure here means the removal action did not take effect (or the
    interventions were inert); the gate must stop rather than reinterpret.  The
    outcome in that case is ``blocked_at_entry_audit`` and the fix is to repair
    the wiring, never to relax the criterion.
    """

    observed_pairs = {tuple(sorted(item.member_ids)) for item in train_only_records}
    held_out_present = HELD_OUT_PAIR in observed_pairs
    missing_observed = tuple(
        sorted(
            item for item in OBSERVED_PAIR_MEMBER_SETS if tuple(sorted(item)) not in observed_pairs
        )
    )

    # direct count of P*'s joint cell in train, recomputed from the corpus itself
    train_joint = sum(
        1
        for episode in corpus.train
        if episode.context_id in _train_context_ids() and _episode_is_held_out_joint(episode)
    )
    holdout_joint = sum(
        1
        for episode in corpus.holdout
        if episode.context_id not in _train_context_ids() and _episode_is_held_out_joint(episode)
    )
    train_singleton = {
        member: sum(
            1
            for episode in corpus.train
            if episode.member_ids == (member,) and member in HELD_OUT_PAIR
        )
        for member in HELD_OUT_PAIR
    }

    conditions: list[dict[str, Any]] = []

    def require(name: str, ok: bool, detail: str) -> None:
        if not ok:
            conditions.append({"condition": name, "detail": detail})

    require(
        "held_out_pair_not_observed",
        not held_out_present,
        f"designated pair {'+'.join(HELD_OUT_PAIR)} is still present in observed_records",
    )
    require(
        "held_out_pair_joint_cell_absent_from_train",
        train_joint == 0,
        f"designated pair joint cell occurs {train_joint} times in the train partition",
    )
    require(
        "held_out_pair_joint_cell_present_in_holdout",
        holdout_joint > 0,
        "designated pair joint cell is absent from holdout, so its realized gain "
        "cannot be measured",
    )
    require(
        "remaining_observed_pairs_intact",
        not missing_observed,
        f"these pairs should have stayed observable but were not estimated: {missing_observed}",
    )
    require(
        "removal_ledger_matches_expectation",
        len(removed) == TRAIN_CONTEXT_COUNT * REPEATS,
        f"removal ledger has {len(removed)} entries, expected " f"{TRAIN_CONTEXT_COUNT * REPEATS}",
    )
    require(
        "held_out_members_still_marginally_observed",
        all(count > 0 for count in train_singleton.values()),
        "a held-out pair member has no singleton observation left in train, so the "
        "combination is not merely unseen but unsupported",
    )
    require(
        "interventions_executed",
        bool(intervention_reality["interventions_happened"]),
        f"{intervention_reality['non_baseline_zero_step_total']} non-baseline "
        "intervention episodes executed zero steps",
    )

    return {
        "conditions": conditions,
        "passed": not conditions,
        "observed_pairs": sorted(list(item) for item in observed_pairs),
        "held_out_pair": list(HELD_OUT_PAIR),
        "held_out_pair_in_observed_records": held_out_present,
        "held_out_pair_joint_cell_count_in_train": train_joint,
        "held_out_pair_joint_cell_count_in_holdout": holdout_joint,
        "held_out_pair_train_singleton_counts": train_singleton,
        "missing_observed_pairs": [list(item) for item in missing_observed],
        "removed_episodes": removed,
        "train_episodes": len(corpus.train),
        "holdout_episodes": len(corpus.holdout),
    }


# --------------------------------------------------------------------------- #
# Realized gains from measured factorial cells
# --------------------------------------------------------------------------- #


def _outcome_index(
    episodes: list[dict[str, Any]],
) -> dict[tuple[str, tuple[str, ...]], list[float]]:
    index: dict[tuple[str, tuple[str, ...]], list[float]] = {}
    for episode in episodes:
        key = (str(episode["task_id"]), tuple(episode["active_members"]))
        index.setdefault(key, []).append(1.0 if episode["success"] else -1.0)
    return index


def _cell_mean(
    index: dict[tuple[str, tuple[str, ...]], list[float]], task_id: str, members: tuple[str, ...]
) -> float:
    values = index.get((task_id, tuple(members)))
    if not values:
        raise AssertionError(f"missing measured cell {task_id}/{members}")
    return float(sum(values) / len(values))


def _realized(
    index: dict[tuple[str, tuple[str, ...]], list[float]],
    *,
    task_id: str,
    members: tuple[str, ...],
) -> dict[str, Any]:
    """Realized factorial decomposition for one pair in one unseen context."""

    first, second = members
    baseline = _cell_mean(index, task_id, ())
    first_only = _cell_mean(index, task_id, (first,))
    second_only = _cell_mean(index, task_id, (second,))
    pair = _cell_mean(index, task_id, members)
    strongest_singleton = max(first_only, second_only)
    strongest_member = first if first_only >= second_only else second
    return {
        "context_id": task_id,
        "member_ids": list(members),
        "baseline_outcome": baseline,
        "first_outcome": first_only,
        "second_outcome": second_only,
        "pair_outcome": pair,
        "realized_interaction": pair - first_only - second_only + baseline,
        "realized_pair_gain": pair - baseline,
        "realized_pair_gain_vs_strongest_single": pair - strongest_singleton,
        "strongest_singleton_member": strongest_member,
        "strongest_singleton_gain": strongest_singleton - baseline,
    }


def _cell_counts(
    index: dict[tuple[str, tuple[str, ...]], list[float]], task_id: str
) -> dict[str, int]:
    return {
        _ids(tuple(members)): len(index.get((task_id, tuple(members)), ()))
        for members in CELL_MEMBER_SETS
    }


# --------------------------------------------------------------------------- #
# Learners and controls
# --------------------------------------------------------------------------- #


def _fit_learner(
    corpus: InteractionTraceCorpus,
    *,
    ridge: float,
    minimum_utility: float,
    maximum_uncertainty: float,
    reverse_order: bool = False,
) -> tuple[
    InteractionGroupTransferLearner,
    tuple[Any, ...],
    tuple[InteractionGroupRecord, ...],
]:
    revision = next(iter(corpus.train_checkpoint_revisions))
    # canonical lineage: the corpus digest deliberately excludes holdout
    # outcomes, so profile and record lineage bind to one digest and cannot cross
    # it.
    profiles = build_member_evidence(
        corpus.train,
        source_trace_digest=corpus.train_trace_digest,
        checkpoint_revision=revision,
    )
    learner = InteractionGroupTransferLearner(
        ridge=ridge,
        minimum_utility=minimum_utility,
        maximum_uncertainty=maximum_uncertainty,
    )
    learner.observe_members(tuple(reversed(profiles)) if reverse_order else profiles)
    # train-only: every observed pair comes from train-only attribution, so the
    # learner cannot see a pair that was evaluated against holdout.
    evaluator = InteractionGroupEvaluator()
    candidates = evaluator.train_only_candidates(corpus)
    learner.observe_records(tuple(reversed(candidates)) if reverse_order else candidates)
    return learner, profiles, candidates


def _lesion_coefficients(
    learner: InteractionGroupTransferLearner,
) -> InteractionGroupTransferLearner:
    lesion = InteractionGroupTransferLearner.from_checkpoint(learner.checkpoint())
    object.__setattr__(lesion, "_coefficients", tuple(0.0 for _ in lesion._coefficients))
    return lesion


def _select_sum_heuristic(
    profiles: tuple[Any, ...],
    member_sets: tuple[tuple[str, ...], ...],
    *,
    resource_budget: float | None,
) -> tuple[InteractionGroupSelection, InteractionGroupTransferCandidate] | None:
    """C5: rank unseen pairs by summed profile contribution only (no relation fit)."""

    by_member = {item.member_id: item for item in profiles}
    scored: list[tuple[float, float, str, tuple[str, ...]]] = []
    for members in member_sets:
        if len(members) != 2 or any(member not in by_member for member in members):
            continue
        cost = sum(float(by_member[member].resource_cost) for member in members)
        if resource_budget is not None and cost > float(resource_budget):
            continue
        score = sum(float(by_member[member].contribution) for member in members)
        scored.append((score, cost, "-".join(members), members))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    score, cost, _, members = scored[0]
    digest = _sha(json.dumps({"members": list(members), "score": score}, sort_keys=True))
    candidate = InteractionGroupTransferCandidate(
        group_id="sum-heuristic:" + digest[:24],
        member_ids=members,
        source_trace_digest=str(by_member[members[0]].source_trace_digest),
        checkpoint_revision=int(by_member[members[0]].checkpoint_revision),
        predicted_interaction=float(score),
        uncertainty=0.0,
        resource_cost=float(cost),
        support=len(profiles),
        method="train-only-summed-member-contribution",
    )
    selection = InteractionGroupSelection(
        group_id=candidate.group_id,
        member_ids=candidate.member_ids,
        source_trace_digest=candidate.source_trace_digest,
        checkpoint_revision=candidate.checkpoint_revision,
        utility=float(candidate.predicted_interaction),
        resource_cost=float(candidate.resource_cost),
        observations=len(profiles),
    )
    return selection, candidate


def _order_candidate_sets(
    member_sets: tuple[tuple[str, ...], ...], seed: int
) -> tuple[tuple[str, ...], ...]:
    ordered = sorted(member_sets)
    if seed % 2:
        return tuple(reversed(ordered))
    return tuple(ordered)


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "growth_admitted": False,
        "can_promote": False,
        "preregistration": PREREGISTRATION,
        "static_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": list(STATIC_CHECK_COMMANDS),
            "executed_before_run": True,
        },
    }
    workspace_root = Path(tempfile.mkdtemp(prefix="p52cp-workbench-"))
    replica_root = Path(tempfile.mkdtemp(prefix="p52cp-replica-"))
    try:
        embedder = DocumentEmbedder()
        members = p52b._train_members(embedder)
        contexts = p52a._validation_tasks()[:CONTEXT_COUNT]
        cue_by_task = {task.task_id: embedder.embed([task.goal_text])[0] for task in contexts}

        matrix_episodes = _execute_matrix(workspace_root, contexts, members, cue_by_task)
        projected = _project(matrix_episodes)
        corpus, removed = _build_corpus(projected)

        # the P5.2b-repaired intervention reality check is a hard entry condition:
        # an intervention that produced no action did not happen, so its cell
        # cannot be evidence of anything.
        intervention_reality = _intervention_reality(matrix_episodes, contexts)

        revision = next(iter(corpus.train_checkpoint_revisions))
        train_trace_digest = corpus.train_trace_digest
        profiles = build_member_evidence(
            corpus.train,
            source_trace_digest=train_trace_digest,
            checkpoint_revision=revision,
        )
        evaluator = InteractionGroupEvaluator()
        train_only_records = evaluator.train_only_candidates(corpus)
        if any(
            record.source_trace_digest != train_trace_digest
            or int(record.checkpoint_revision) != int(revision)
            for record in train_only_records
        ):
            raise AssertionError("train-only records cross the train trace lineage")

        entry_audit = _entry_audit(
            corpus,
            removed=removed,
            evaluator=evaluator,
            train_only_records=train_only_records,
            intervention_reality=intervention_reality,
        )
        if not entry_audit["passed"]:
            payload.update(
                {
                    "status": "blocked",
                    "outcome": "blocked_at_entry_audit",
                    "entry_audit": entry_audit,
                    "intervention_reality": intervention_reality,
                    "matrix": {
                        "episodes": len(matrix_episodes),
                        "train_episodes": len(corpus.train),
                        "holdout_episodes": len(corpus.holdout),
                    },
                    "growth_admitted": False,
                    "can_promote": False,
                    "interpretation": (
                        "blocked: the train-partition removal did not produce the "
                        "preregistered unseen-combination surface, or the "
                        "interventions were inert; repair the wiring, do not relax "
                        "the criterion"
                    ),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(replica_root, ignore_errors=True)
            _write_json(DEFAULT_REPORT, payload)
            return payload

        observed_pairs = {tuple(sorted(item.member_ids)) for item in train_only_records}
        candidate_sets_ordered = _order_candidate_sets(PAIR_MEMBER_SETS, 11)

        # ---- learner: fit on train-only evidence, predict the unseen pair ----
        learner, _, _ = _fit_learner(
            corpus,
            ridge=0.1,
            minimum_utility=-1.0e9,
            maximum_uncertainty=1.0e9,
        )

        selected = learner.select(candidate_sets_ordered, resource_budget=10.0, unseen_only=True)
        if selected is None:
            payload.update(
                {
                    "status": "failed",
                    "outcome": "failed",
                    "entry_audit": entry_audit,
                    "intervention_reality": intervention_reality,
                    "error": (
                        "learner.select returned None although the entry audit "
                        "confirmed a non-empty unseen surface"
                    ),
                    "growth_admitted": False,
                    "can_promote": False,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(replica_root, ignore_errors=True)
            _write_json(DEFAULT_REPORT, payload)
            return payload
        selection, candidate = selected
        selected_members = tuple(candidate.member_ids)

        # preregistration section 3.4 step 3: the selection must be the
        # designated pair.  Selecting anything else is a mechanical failure, not
        # a scientific result.
        selected_is_designated = tuple(sorted(selected_members)) == HELD_OUT_PAIR

        # ---- pre-execution binding (recorded before any scoring) ----
        parent_checkpoint = learner.checkpoint()
        prediction_binding = {
            "bound_before_scoring": True,
            "parent_checkpoint_digest": parent_checkpoint["checkpoint_digest"],
            "model_digest": learner.model_digest,
            "source_trace_digest": candidate.source_trace_digest,
            "checkpoint_revision": candidate.checkpoint_revision,
            "selected_group_id": candidate.group_id,
            "selected_member_ids": list(selected_members),
            "selected_is_designated_unseen_pair": selected_is_designated,
            "predicted_interaction": candidate.predicted_interaction,
            "uncertainty": candidate.uncertainty,
            "resource_cost": candidate.resource_cost,
            "support": candidate.support,
            "selection_utility": selection.utility,
            "candidate_sets_considered": [list(item) for item in candidate_sets_ordered],
            "observed_pair_count": len(observed_pairs),
        }

        # ---- prediction for every candidate pair (full calibration surface) ----
        predictions: list[dict[str, Any]] = []
        for members_pair in PAIR_MEMBER_SETS:
            predicted = learner.candidate(members_pair, allow_observed=False)
            predictions.append(
                {
                    "member_ids": list(members_pair),
                    "seen_in_train_evidence": tuple(sorted(members_pair)) in observed_pairs,
                    "predicted": (
                        None
                        if predicted is None
                        else {
                            "group_id": predicted.group_id,
                            "predicted_interaction": predicted.predicted_interaction,
                            "uncertainty": predicted.uncertainty,
                            "resource_cost": predicted.resource_cost,
                            "support": predicted.support,
                        }
                    ),
                }
            )

        # ---- controls: same unseen context scoring, same contract path ----
        control_sets: dict[str, tuple[str, ...]] = {
            "no_learning": candidate_sets_ordered[0],
            "random_combination": candidate_sets_ordered[
                int(_sha(f"{RANDOM_CONTROL_SEED}:{sorted(candidate_sets_ordered)}")[:8], 16)
                % len(candidate_sets_ordered)
            ],
            "fixed_combination": FIXED_COMBINATION,
        }
        sum_heuristic = _select_sum_heuristic(
            profiles, candidate_sets_ordered, resource_budget=10.0
        )
        if sum_heuristic is None:
            raise AssertionError("sum-heuristic control returned no candidate")
        control_sets["train_only_simple_regression"] = tuple(sum_heuristic[1].member_ids)
        lesion = _lesion_coefficients(learner)
        lesion_selected = lesion.select(
            candidate_sets_ordered, resource_budget=10.0, unseen_only=True
        )
        if lesion_selected is None:
            raise AssertionError("lesion control returned no candidate")
        control_sets["lesion_learner"] = tuple(lesion_selected[1].member_ids)
        # C2 strongest_singleton is not a pair selection: it is scored directly
        # against the measured singleton cells for each context, which is exactly
        # what "the strongest single member" means for the comparison.

        # ---- real execution on unseen contexts for object + controls ----
        unseen_contexts = contexts[TRAIN_CONTEXT_COUNT:]
        object_index = _outcome_index(matrix_episodes)
        unseen_cells = {
            task.task_id: _cell_counts(object_index, task.task_id) for task in unseen_contexts
        }

        def score(member_sets: tuple[str, ...]) -> dict[str, Any]:
            per_context = [
                _realized(object_index, task_id=task.task_id, members=member_sets)
                for task in unseen_contexts
            ]
            gains = [item["realized_pair_gain_vs_strongest_single"] for item in per_context]
            interactions = [item["realized_interaction"] for item in per_context]
            return {
                "member_ids": list(member_sets),
                "per_context": per_context,
                "mean_gain_vs_strongest_single": float(sum(gains) / len(gains)),
                "mean_realized_interaction": float(sum(interactions) / len(interactions)),
                "contexts_beating_strongest_single": sum(1 for item in gains if item > 0.0),
                "contexts_scored": len(per_context),
            }

        def strongest_singleton_score() -> dict[str, Any]:
            """C2: per context, the best measured singleton's gain over baseline."""

            per_context: list[dict[str, Any]] = []
            for task in unseen_contexts:
                baseline = _cell_mean(object_index, task.task_id, ())
                best_member = None
                best_value = None
                for members_set in SINGLETON_MEMBER_SETS:
                    value = _cell_mean(object_index, task.task_id, tuple(members_set))
                    if best_value is None or value > best_value:
                        best_value = value
                        best_member = members_set[0]
                per_context.append(
                    {
                        "context_id": task.task_id,
                        "strongest_singleton_member": best_member,
                        "strongest_singleton_gain": float(best_value) - baseline,
                    }
                )
            gains = [item["strongest_singleton_gain"] for item in per_context]
            return {
                "member_ids": [],
                "per_context": per_context,
                "mean_gain_vs_strongest_single": 0.0,
                "mean_realized_interaction": None,
                "mean_strongest_singleton_gain": float(sum(gains) / len(gains)),
                "contexts_scored": len(per_context),
            }

        object_score = score(selected_members)
        # C1/C3/C4/C5/C6 are *pair selections*: each is scored with the same
        # factorial decomposition as the object.  C2 is not a pair at all -- the
        # strongest single member differs per context and has no joint cell -- so
        # it is scored separately and merged in below with the same comparison
        # quantity (gain over that context's own baseline).
        pair_control_scores = {name: score(value) for name, value in control_sets.items()}
        singleton_score = strongest_singleton_score()
        control_scores: dict[str, Any] = {
            **pair_control_scores,
            "strongest_singleton": {
                **singleton_score,
                "member_ids": ["<per-context strongest singleton>"],
                "mean_gain_vs_strongest_single": singleton_score["mean_strongest_singleton_gain"],
            },
        }

        # margin-bearing controls exclude C4 by construction (preregistration
        # section 3.5: C4 coincides with the answer, so it cannot bear the margin).
        margin_controls = {
            name: value for name, value in control_scores.items() if name != "fixed_combination"
        }
        strongest_control = max(
            margin_controls.items(),
            key=lambda item: item[1]["mean_gain_vs_strongest_single"],
        )
        fixed_control_gain = control_scores["fixed_combination"]["mean_gain_vs_strongest_single"]

        # ---- calibration: predicted vs realized on the candidate surface ----
        calibration_rows: list[dict[str, Any]] = []
        for entry in predictions:
            members_pair = tuple(entry["member_ids"])
            realized = score(members_pair)
            predicted_value = (
                None if entry["predicted"] is None else entry["predicted"]["predicted_interaction"]
            )
            calibration_rows.append(
                {
                    "member_ids": list(members_pair),
                    "seen_in_train_evidence": entry["seen_in_train_evidence"],
                    "predicted_interaction": predicted_value,
                    "realized_interaction": realized["mean_realized_interaction"],
                    "absolute_error": (
                        None
                        if predicted_value is None
                        else abs(float(predicted_value) - realized["mean_realized_interaction"])
                    ),
                    "sign_matches": (
                        None
                        if predicted_value is None
                        else (
                            (float(predicted_value) > 0.0)
                            == (realized["mean_realized_interaction"] > 0.0)
                        )
                    ),
                }
            )
        unseen_rows = [item for item in calibration_rows if not item["seen_in_train_evidence"]]
        comparable = [
            item
            for item in unseen_rows
            if item["predicted_interaction"] is not None
            and item["realized_interaction"] is not None
        ]
        errors = sorted(float(item["absolute_error"]) for item in comparable)
        sign_hits = [bool(item["sign_matches"]) for item in comparable]

        def _median(values: list[float]) -> float | None:
            if not values:
                return None
            middle = len(values) // 2
            if len(values) % 2:
                return float(values[middle])
            return float((values[middle - 1] + values[middle]) / 2.0)

        sign_match_rate = (
            float(sum(1 for item in sign_hits if item) / len(sign_hits)) if sign_hits else None
        )
        median_absolute_error = _median(errors)
        calibration_pooled = {
            "unseen_pair_count": len(unseen_rows),
            "comparable_pair_count": len(comparable),
            "mean_absolute_error": float(sum(errors) / len(errors)) if errors else None,
            "median_absolute_error": median_absolute_error,
            "sign_match_rate": sign_match_rate,
            "small_sample_caveat": (
                "the unseen surface has cardinality 1 by construction; these pooled "
                "statistics rest on that single pair and are not an independent sample"
            ),
            "thresholds": {
                "minimum_sign_match_rate": CALIBRATION_MINIMUM_SIGN_MATCH_RATE,
                "maximum_median_absolute_error": CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR,
                "fixed_at_freeze_time": True,
            },
            "rows": calibration_rows,
        }
        calibration_ok = bool(
            sign_match_rate is not None
            and median_absolute_error is not None
            and sign_match_rate >= CALIBRATION_MINIMUM_SIGN_MATCH_RATE
            and median_absolute_error <= CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR
        )

        # ---- family-coverage attribution ----
        coverage = _family_coverage(matrix_episodes, contexts)

        # ---- 11 x 4 success matrix over all contexts, per block ----
        success_matrix = _success_matrix(matrix_episodes, contexts)

        # ---- replica determinism ----
        replica_episodes = _execute_matrix(replica_root, contexts, members, cue_by_task)

        def surface(episodes: list[dict[str, Any]]) -> list[Any]:
            return sorted(
                (item["episode_id"], item["success"], item["stop_reason"], item["resource_cost"])
                for item in episodes
            )

        replica_consistent = bool(surface(replica_episodes) == surface(matrix_episodes))

        # ---- recovery: fresh process restore reproduces the selection ----
        recovery = _recovery_probe(parent_checkpoint, candidate_sets_ordered)
        rejections = _rejection_probe(corpus, train_only_records, profiles, candidate_sets_ordered)

        # ---- self-consistency: learner must reproduce the selection ----
        replay_learner, _, _ = _fit_learner(
            corpus,
            ridge=0.1,
            minimum_utility=-1.0e9,
            maximum_uncertainty=1.0e9,
            reverse_order=True,
        )
        replay_selected = replay_learner.select(
            tuple(reversed(candidate_sets_ordered)), resource_budget=10.0, unseen_only=True
        )
        replay_matches = bool(
            replay_selected is not None
            and tuple(replay_selected[1].member_ids) == selected_members
            and math.isclose(
                float(replay_selected[1].predicted_interaction),
                float(candidate.predicted_interaction),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

        # ---- execution provenance and safety ----
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
        stop_reasons: Counter[str] = Counter()
        for episode in matrix_episodes:
            stop_reasons[str(episode["stop_reason"]).split(":")[0]] += 1

        # ---- gate 9 quantities ----
        object_gain = object_score["mean_gain_vs_strongest_single"]
        best_control_gain = strongest_control[1]["mean_gain_vs_strongest_single"]
        margin_cleared = bool(object_gain > best_control_gain + MARGIN)
        collaboration_holds = bool(
            all(
                item["realized_pair_gain_vs_strongest_single"] >= 0.0
                for item in object_score["per_context"]
            )
            and object_score["contexts_beating_strongest_single"] > 0
        )

        all_cells_constant = bool(
            all(
                len({item["success"] for item in matrix_episodes if item["task_id"] == task_id})
                <= 1
                for task_id in {item["task_id"] for item in matrix_episodes}
            )
        )

        total_wall = time.perf_counter() - started

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
            and entry_audit["held_out_pair_joint_cell_count_in_train"] == 0
            and entry_audit["held_out_pair_joint_cell_count_in_holdout"] > 0
        )
        holdout_disjoint = bool(
            {item.context_id for item in corpus.train}.isdisjoint(
                {item.context_id for item in corpus.holdout}
            )
            and len(corpus.holdout)
            == (CONTEXT_COUNT - TRAIN_CONTEXT_COUNT) * REPEATS * len(CELL_MEMBER_SETS)
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
                and selected_is_designated
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
            "transfer_and_budget": bool(
                margin_cleared
                and collaboration_holds
                and replica_consistent
                and total_wall <= TOTAL_SECONDS_CAP
            ),
        }
        mechanical = {key: value for key, value in gates.items() if key != "transfer_and_budget"}
        if not all(mechanical.values()):
            outcome = "failed"
        elif not all(gates.values()):
            if all_cells_constant:
                outcome = "transfer_signal_constant"
            else:
                outcome = "transfer_no_gain"
        else:
            outcome = "unseen_combination_transfer_supported"

        payload.update(
            {
                "status": "completed",
                "outcome": outcome,
                "experiment_passed": bool(outcome == "unseen_combination_transfer_supported"),
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
                    "margin_provenance": (
                        "P5.2a frozen FROZEN_MARGIN constant, same value, new metric object"
                    ),
                    "held_out_pair": list(HELD_OUT_PAIR),
                    "held_out_pair_opaque_index": list(HELD_OUT_PAIR_INDEX),
                    "structural_change": (
                        "the designated pair's joint (T,T) cell is removed from every "
                        "train context; all its other cells and all singleton evidence "
                        "are retained, and holdout is untouched. See entry_audit."
                    ),
                    "candidate_pairs": [list(item) for item in PAIR_MEMBER_SETS],
                    "observed_pairs": sorted(list(item) for item in observed_pairs),
                    "learner_observes_train_only_records": True,
                    "recovery_effect_disclosure": (
                        "not measured this gate; recorded as 0 without fabrication"
                    ),
                    "semantic_mapping_disclosure": (
                        "member id -> training family mapping stored in runner config only"
                    ),
                    "sequence_generalization_claim": (
                        "none: gains are attributed to complementary family coverage only"
                    ),
                    "known_cost_disclosure": (
                        "the unseen surface has cardinality 1, so control C4 coincides "
                        "with the correct answer; the margin-bearing controls therefore "
                        "exclude C4, and a pass claims only prediction-realization "
                        "consistency plus superiority over no-learning/random/singleton/"
                        "regression/lesion controls. It does NOT claim that the learner "
                        "can choose among several unseen combinations."
                    ),
                },
                "entry_audit": entry_audit,
                "prediction_binding": prediction_binding,
                "prediction_surface": predictions,
                "object": {
                    "member_ids": list(selected_members),
                    "score": object_score,
                },
                "controls": {
                    name: {
                        "member_ids": (
                            list(control_sets[name])
                            if name in control_sets
                            else ["<per-context strongest singleton>"]
                        ),
                        "kind": (
                            "pair_selection" if name in control_sets else "per_context_singleton"
                        ),
                        "score": value,
                    }
                    for name, value in control_scores.items()
                },
                "control_summary": {
                    "strongest_control": strongest_control[0],
                    "strongest_control_member_ids": (
                        list(control_sets[strongest_control[0]])
                        if strongest_control[0] in control_sets
                        else ["<per-context strongest singleton>"]
                    ),
                    "strongest_control_gain": best_control_gain,
                    "object_gain": object_gain,
                    "margin": MARGIN,
                    "required": best_control_gain + MARGIN,
                    "margin_cleared": margin_cleared,
                    "collaboration_holds": collaboration_holds,
                    "fixed_combination_gain_excluded_from_margin": fixed_control_gain,
                    "c2_strongest_singleton_gain": singleton_score["mean_strongest_singleton_gain"],
                    "margin_bearing_controls": sorted(margin_controls),
                },
                "calibration": calibration_pooled,
                "family_coverage": coverage,
                "success_matrix": success_matrix,
                "matrix": {
                    "episodes": len(matrix_episodes),
                    "train_episodes": len(corpus.train),
                    "holdout_episodes": len(corpus.holdout),
                    "cells_per_unseen_context": {
                        task_id: dict(sorted(counts.items()))
                        for task_id, counts in unseen_cells.items()
                    },
                    "all_cells_constant": all_cells_constant,
                    "stop_reason_classes": dict(sorted(stop_reasons.items())),
                },
                "intervention_reality": intervention_reality,
                "evaluator": {
                    "train_only_records": len(train_only_records),
                    "observed_pairs": sorted(list(item) for item in observed_pairs),
                    "rejected_reasons": sorted(
                        Counter(
                            item.reason
                            for item in evaluator.evaluate(corpus).state.rejected_candidates or ()
                        ).items()
                    ),
                },
                "member_evidence": {
                    "profiles": len(profiles),
                    "member_ids": sorted(item.member_id for item in profiles),
                    "contributions": {item.member_id: item.contribution for item in profiles},
                },
                "recovery": recovery,
                "rejections": rejections,
                "self_consistency": {
                    "reverse_order_selection_matches": replay_matches,
                },
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; object={'+'.join(selected_members)} "
                    f"gain={round(object_gain, 6)} vs strongest control "
                    f"{strongest_control[0]}={round(best_control_gain, 6)} "
                    f"(required {round(best_control_gain + MARGIN, 6)}); "
                    f"calibration sign_rate={sign_match_rate}, "
                    f"median_abs_error={median_absolute_error}"
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
        shutil.rmtree(workspace_root, ignore_errors=True)
        shutil.rmtree(replica_root, ignore_errors=True)
    _write_json(DEFAULT_REPORT, payload)
    return payload


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def _success_matrix(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    block_of = {task.task_id: int(task.task_id.rsplit("-", 1)[1]) % 4 for task in contexts}
    totals: dict[tuple[str, int], int] = {}
    attempts: dict[tuple[str, int], int] = {}
    for episode in episodes:
        key = (_ids(tuple(episode["active_members"])), block_of[episode["task_id"]])
        attempts[key] = attempts.get(key, 0) + 1
        totals[key] = totals.get(key, 0) + (1 if episode["success"] else 0)
    rows: list[dict[str, Any]] = []
    for cell in CELL_MEMBER_SETS:
        label = _ids(cell)
        rows.append(
            {
                "cell": label,
                "blocks": [
                    {
                        "block": block,
                        "successes": totals.get((label, block), 0),
                        "attempts": attempts.get((label, block), 0),
                    }
                    for block in range(4)
                ],
                "total_successes": sum(totals.get((label, block), 0) for block in range(4)),
                "total_attempts": sum(attempts.get((label, block), 0) for block in range(4)),
            }
        )
    return {
        "blocks": {
            str(block): sorted(task.task_id for task in contexts if block_of[task.task_id] == block)
            for block in range(4)
        },
        "rows": rows,
    }


def _family_coverage(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    """Contingency table: pair gain vs covered-by-a-member-family context."""

    pairs: list[dict[str, Any]] = []
    block_of = {task.task_id: int(task.task_id.rsplit("-", 1)[1]) % 4 for task in contexts}
    index = _outcome_index(episodes)
    for task in contexts:
        block = block_of[task.task_id]
        for cell in PAIR_MEMBER_SETS:
            baseline = _cell_mean(index, task.task_id, ())
            pair = _cell_mean(index, task.task_id, tuple(cell))
            pairs.append(
                {
                    "context_id": task.task_id,
                    "block": block,
                    "member_ids": list(cell),
                    "realized_pair_gain": pair - baseline,
                }
            )
    covered = [item for item in pairs if item["realized_pair_gain"] > 0.0]
    uncovered = [item for item in pairs if item["realized_pair_gain"] <= 0.0]
    return {
        "pair_context_rows": len(pairs),
        "positive_gain_rows": len(covered),
        "nonpositive_gain_rows": len(uncovered),
        "positive_by_block": dict(sorted(Counter(item["block"] for item in covered).items())),
        "nonpositive_by_block": dict(sorted(Counter(item["block"] for item in uncovered).items())),
        "rows": pairs,
    }


def _recovery_probe_script() -> str:
    """Fresh-process restore probe: no in-process state may be reused."""

    return (
        "import json,sys;"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r});"
        "from taiji import InteractionGroupTransferLearner;"
        "payload=json.load(open(sys.argv[1],encoding='utf-8'));"
        "learner=InteractionGroupTransferLearner.from_checkpoint(payload);"
        "sets=tuple(tuple(item) for item in json.loads(sys.argv[2]));"
        "selected=learner.select(sets, resource_budget=10.0, unseen_only=True);"
        "print(json.dumps({'group_id':selected[1].group_id,"
        "'member_ids':list(selected[1].member_ids),"
        "'predicted_interaction':selected[1].predicted_interaction} if selected else None))"
    )


def _recovery_probe(
    parent_checkpoint: dict[str, Any], candidate_sets: tuple[tuple[str, ...], ...]
) -> dict[str, Any]:
    probe_script = _recovery_probe_script()
    checkpoint_path = Path(tempfile.mkdtemp(prefix="p52cp-recovery-")) / "learner.json"
    try:
        checkpoint_path.write_text(json.dumps(parent_checkpoint), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                probe_script,
                str(checkpoint_path),
                json.dumps([list(item) for item in candidate_sets]),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if completed.returncode != 0:
            return {
                "fresh_process": True,
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip()[-2000:],
                "selection_reproduced": False,
            }
        restored = json.loads(completed.stdout.strip().splitlines()[-1])
        digest_probe = InteractionGroupTransferLearner.from_checkpoint(parent_checkpoint)
        expected = None
        predicted = digest_probe.candidate(HELD_OUT_PAIR, allow_observed=False)
        if predicted is not None:
            expected = {
                "group_id": predicted.group_id,
                "member_ids": list(predicted.member_ids),
                "predicted_interaction": predicted.predicted_interaction,
            }
        return {
            "fresh_process": True,
            "returncode": completed.returncode,
            "restored_selection": restored,
            "in_process_selection": expected,
            "selection_reproduced": bool(restored == expected),
        }
    finally:
        shutil.rmtree(checkpoint_path.parent, ignore_errors=True)


def _rejection_probe(
    corpus: InteractionTraceCorpus,
    train_only_records: tuple[InteractionGroupRecord, ...],
    profiles: tuple[Any, ...],
    candidate_sets: tuple[tuple[str, ...], ...],
) -> dict[str, bool]:
    results: dict[str, bool] = {}

    learner, _, _ = _fit_learner(
        corpus,
        ridge=0.1,
        minimum_utility=-1.0e9,
        maximum_uncertainty=1.0e9,
    )

    holdout_record = dataclasses.replace(train_only_records[0], holdout_interaction=0.5)
    try:
        learner.observe_records((holdout_record,))
        results["holdout_record_rejected"] = False
    except ValueError:
        results["holdout_record_rejected"] = True

    terminal_record = dataclasses.replace(train_only_records[0], status="rejected")
    try:
        learner.observe_records((terminal_record,))
        results["terminal_record_rejected"] = False
    except ValueError:
        results["terminal_record_rejected"] = True

    tampered = dict(learner.checkpoint())
    tampered["checkpoint_digest"] = "0" * 64
    try:
        InteractionGroupTransferLearner.from_checkpoint(tampered)
        results["tampered_checkpoint_rejected"] = False
    except ValueError:
        results["tampered_checkpoint_rejected"] = True

    stale = dict(learner.checkpoint())
    stale["source_trace_digest"] = _sha("stale lineage probe")
    body = {key: value for key, value in stale.items() if key != "checkpoint_digest"}
    stale["checkpoint_digest"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":")))
    try:
        InteractionGroupTransferLearner.from_checkpoint(stale)
        results["stale_lineage_rejected"] = False
    except ValueError:
        results["stale_lineage_rejected"] = True

    known = sorted(item.member_id for item in profiles)[0]
    results["unknown_member_fails_closed"] = (
        learner.candidate((known, "member-unknown-opaque")) is None
    )
    # the preregistration requires that the five genuinely observed pairs cannot
    # be reselected under unseen_only=True; it also requires that the designated
    # pair *can* be.  Both directions are asserted, because checking only one
    # would accept a broken intermediate state.
    results["observed_pairs_not_reselectable"] = all(
        learner.candidate(members_pair, allow_observed=False) is None
        for members_pair in candidate_sets
        if tuple(sorted(members_pair))
        in {tuple(sorted(item.member_ids)) for item in train_only_records}
    )
    results["designated_pair_predictable"] = (
        learner.candidate(HELD_OUT_PAIR, allow_observed=False) is not None
    )
    return results


def _intervention_reality(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    """Assert every non-baseline intervention actually executed something.

    Reused from the P5.2b repair.  ``interventions_happened`` is false when any
    non-``(F,F)`` cell produced no action, which means the task was already
    satisfied before the intervention could take effect and the cell's recorded
    outcome is not attributable to its members.
    """

    per_context: dict[str, Any] = {}
    offenders: list[dict[str, Any]] = []
    for task in contexts:
        task_id = task.task_id
        cells = [item for item in episodes if item["task_id"] == task_id]
        zero_step = sum(1 for item in cells if not item["steps"])
        active_zero = sum(
            1 for item in cells if not item["steps"] and tuple(item["active_members"])
        )
        baseline_zero = sum(
            1 for item in cells if not item["steps"] and not tuple(item["active_members"])
        )
        per_context[task_id] = {
            "episodes": len(cells),
            "zero_step_episodes": zero_step,
            "non_baseline_zero_step_episodes": active_zero,
            "baseline_zero_step_episodes": baseline_zero,
            "successes": sum(1 for item in cells if item["success"]),
        }
        for item in cells:
            if not item["steps"] and tuple(item["active_members"]):
                offenders.append(
                    {
                        "context_id": task_id,
                        "member_ids": list(item["active_members"]),
                        "success": item["success"],
                        "stop_reason": item["stop_reason"],
                    }
                )
    total_non_baseline = sum(
        item["non_baseline_zero_step_episodes"] for item in per_context.values()
    )
    return {
        "per_context": per_context,
        "non_baseline_zero_step_total": total_non_baseline,
        "offenders": offenders[:40],
        "interventions_happened": bool(total_non_baseline == 0),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    result = run_gate()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "gates_failed": sorted(
                    key for key, value in (result.get("gates") or {}).items() if not value
                ),
                "entry_audit_passed": (result.get("entry_audit") or {}).get("passed"),
                "entry_audit_conditions": (result.get("entry_audit") or {}).get("conditions"),
                "object": (result.get("object") or {}).get("member_ids"),
                "control_summary": result.get("control_summary"),
                "calibration": {
                    key: value
                    for key, value in (result.get("calibration") or {}).items()
                    if key in ("sign_match_rate", "median_absolute_error", "comparable_pair_count")
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

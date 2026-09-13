"""P5.2c unseen-combination transfer Gate.

STATUS: BLOCKED at the entry audit (2026-09-13).  The gate wiring below is
complete and reproducible, but two preregistered conditions cannot be met
against the current member set:

1. Gate 3 (unseen_combination_identity) is structurally unsatisfiable: with
   four members, all C(4,2)=6 pairs are estimated by
   ``train_only_candidates``, so the unseen-combination surface is empty and
   ``select(..., unseen_only=True)`` returns None.  ``_pair_features`` supports
   pairs only, so there is no larger-cardinality fallback.
2. Gate 2's evidence is contaminated upstream: the P5.2a ``lang_confirm``
   template (contexts 100/104/108) has goal state == initial state, so 20 of 22
   episodes per context execute zero steps and still record success.  Those
   empty-event cells fabricate the P5.2b ``member-a+member-d`` interaction
   ``0.2222``.

See reports/M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT_20260913.md.  This runner is
retained because it is the instrument that exposed the defect and because the
fix (member-set widening plus task/criterion repair) will reuse it verbatim.
It is NOT authorized to produce a passing result; the report it writes records
the blocked outcome with ``growth_admitted=false`` and ``can_promote=false``.

Preregistration: plans/reference/M5_P5_2C_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md
(frozen; execution blocked).  The transfer learner must predict a signed joint
gain for a member combination it has never observed, using only
pre-intervention train evidence, and that prediction must be independently
confirmed by real Workbench execution on contexts the learner never saw.

Execution reuses P5.2b verbatim: the same four family-specialist procedural
readouts (real intervenable instances), the same 12 contexts x 11 cells x 2
repeats matrix, the same frozen dual-predict-select composition, and the same
policy -> approval -> execute contract path.  Split: contexts[0:8] carry all
train evidence (build_member_evidence + train_only_candidates, so every
observed pair comes from train-only attribution); contexts[8:12] are the
unseen contexts used for scoring.  Predictions are bound before execution and
realized gains are computed afterwards from the measured factorial cells.

Member ids stay opaque; the semantic mapping lives only in this runner's
config section and never enters learner/evaluator inputs.
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

from taiji.document_embedding import DocumentEmbedder  # noqa: E402
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

REPORT_FORMAT = "taiji-p5-2c-unseen-combination-transfer-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_P5_2C_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_2c_unseen_combination_transfer_20260913.json"

TOTAL_SECONDS_CAP = 900.0
# carried over from the P5.2a frozen constant and P5.2b comparison discipline;
# the metric object changes (holdout pair gain margin), the constant does not.
MARGIN = 0.15
TRAIN_CONTEXT_COUNT = 8
CONTEXT_COUNT = 12
REPEATS = 2
TRACE_REVISION = 1
RAYON_RANDOM_SEEDS = (11, 29, 47)
RANDOM_CONTROL_SEED = 20260913
MEMBER_IDS = p52b.MEMBER_IDS
CELL_MEMBER_SETS = p52b.CELL_MEMBER_SETS
PAIR_MEMBER_SETS: tuple[tuple[str, ...], ...] = tuple(
    item for item in CELL_MEMBER_SETS if len(item) == 2
)
SINGLETON_MEMBER_SETS: tuple[tuple[str, ...], ...] = tuple(
    item for item in CELL_MEMBER_SETS if len(item) == 1
)
FIXED_COMBINATION = tuple(sorted((MEMBER_IDS[0], MEMBER_IDS[3])))

STATIC_CHECK_SCOPE = (
    "seed_platform/workbench.py",
    "taiji/interaction_groups.py",
    "taiji/interaction_group_transfer.py",
    "taiji/interaction_group_learning.py",
    "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
    "scripts/training/eval_taiji_p5_2c_unseen_combination_transfer_gate.py",
)
STATIC_CHECK_COMMANDS = (
    "python -m py_compile <scope files>",
    "python -m ruff check scripts/training (and full repo)",
    "python -m mypy --follow-imports=silent seed taiji",
    "python -m black --no-cache --check <scope files>",
    "python -m pytest (baseline known debts, no new failures)",
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
    return p52b._execute_matrix(root, contexts, members, cue_by_task)


def _project(episodes: list[dict[str, Any]]) -> tuple[InteractionTraceEpisode, ...]:
    return p52b._project(episodes)


def _build_corpus(projected: tuple[InteractionTraceEpisode, ...]) -> InteractionTraceCorpus:
    train_contexts = {
        f"p52a-validation-{100 + offset:03d}" for offset in range(TRAIN_CONTEXT_COUNT)
    }
    return InteractionTraceCorpus(
        train=tuple(item for item in projected if item.context_id in train_contexts),
        holdout=tuple(item for item in projected if item.context_id not in train_contexts),
    )


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


def _pair_cells(
    index: dict[tuple[str, tuple[str, ...]], list[float]], task_id: str
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for members in CELL_MEMBER_SETS:
        counts[_ids(members)] = len(index.get((task_id, tuple(members)), ()))
    return counts


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
    # outcomes, so profile and record lineage bind to the same digest and
    # cannot cross it.
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
    # train-only: every observed pair comes from the train-only attribution, so
    # the learner cannot see a pair that was evaluated against holdout.
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


def _control_selection(
    name: str,
    member_sets: tuple[tuple[str, ...], ...],
) -> tuple[str, ...] | None:
    if name == "no_learning":
        return member_sets[0] if member_sets else None
    if name == "random_combination":
        digest = _sha(f"{RANDOM_CONTROL_SEED}:{sorted(member_sets)}")
        return member_sets[int(digest[:8], 16) % len(member_sets)]
    if name == "fixed_combination":
        return FIXED_COMBINATION
    raise ValueError(f"unsupported control {name}")


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
    workspace_root = Path(tempfile.mkdtemp(prefix="p52c-workbench-"))
    replica_root = Path(tempfile.mkdtemp(prefix="p52c-replica-"))
    try:
        embedder = DocumentEmbedder()
        members = p52b._train_members(embedder)
        contexts = p52a._validation_tasks()[:CONTEXT_COUNT]
        cue_by_task = {task.task_id: embedder.embed([task.goal_text])[0] for task in contexts}

        matrix_episodes = _execute_matrix(workspace_root, contexts, members, cue_by_task)
        projected = _project(matrix_episodes)
        corpus = _build_corpus(projected)

        # Entry-audit check the P5.2b gates lacked: an intervention that
        # produced no action at all did not happen, so a cell with an empty
        # event list cannot be evidence of anything.  Without this, a task
        # whose goal is already satisfied before the first tick records success
        # for every configuration and fabricates positive interaction.
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
        # the learner's observed pairs must be exactly the train-only candidates
        # and nothing else; asserted here so a silent lineage fallback cannot
        # widen the evidence surface.
        if any(
            record.source_trace_digest != train_trace_digest
            or int(record.checkpoint_revision) != int(revision)
            for record in train_only_records
        ):
            raise AssertionError("train-only records cross the train trace lineage")

        # ---- learner: fit on train-only evidence, predict unseen pairs ----
        learner, _, _ = _fit_learner(
            corpus,
            ridge=0.1,
            minimum_utility=-1.0e9,
            maximum_uncertainty=1.0e9,
        )
        observed_pairs = {frozenset(item.member_ids) for item in learner.observed_records}
        candidate_sets_ordered = _order_candidate_sets(PAIR_MEMBER_SETS, 11)
        # Gate 3: the unseen-combination surface must be non-empty.  With four
        # members all six pairs are estimated by train_only_candidates, so the
        # surface is empty and this gate cannot be satisfied; report that as a
        # structured blocking condition instead of an assertion failure.
        unseen_surface = [
            members_pair
            for members_pair in candidate_sets_ordered
            if frozenset(members_pair) not in observed_pairs
        ]
        blocking_conditions: list[dict[str, Any]] = []
        if not unseen_surface:
            blocking_conditions.append(
                {
                    "gate": "unseen_combination_identity",
                    "condition": "unseen_combination_surface_empty",
                    "detail": (
                        f"all {len(observed_pairs)} pairs over {len(MEMBER_IDS)} members were "
                        "estimated from train evidence, leaving no unseen combination to predict"
                    ),
                    "observed_pairs": sorted(sorted(item) for item in observed_pairs),
                    "possible_pairs": len(PAIR_MEMBER_SETS),
                }
            )
        if not intervention_reality["interventions_happened"]:
            blocking_conditions.append(
                {
                    "gate": "evidence_admissibility",
                    "condition": "intervention_did_not_execute",
                    "detail": (
                        f"{intervention_reality['non_baseline_zero_step_total']} non-baseline "
                        "intervention episodes executed zero steps yet recorded an outcome"
                    ),
                    "offenders": intervention_reality["offenders"][:12],
                }
            )

        selected = learner.select(candidate_sets_ordered, resource_budget=10.0, unseen_only=True)
        if selected is None:
            payload.update(
                {
                    "status": "blocked",
                    "outcome": "blocked_at_entry_audit",
                    "blocking_conditions": blocking_conditions,
                    "intervention_reality": intervention_reality,
                    "member_evidence": {
                        "profiles": len(profiles),
                        "member_ids": sorted(item.member_id for item in profiles),
                    },
                    "evaluator": {
                        "train_only_records": len(train_only_records),
                        "observed_pairs": sorted(sorted(item) for item in observed_pairs),
                    },
                    "matrix": {
                        "episodes": len(matrix_episodes),
                        "train_context_count": TRAIN_CONTEXT_COUNT,
                        "unseen_context_count": CONTEXT_COUNT - TRAIN_CONTEXT_COUNT,
                    },
                    "audit_report": "reports/M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT_20260913.md",
                    "growth_admitted": False,
                    "can_promote": False,
                    "interpretation": (
                        "blocked: the frozen P5.2c preregistration cannot be executed against the "
                        "current member set and trace projection; see blocking_conditions"
                    ),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            shutil.rmtree(workspace_root, ignore_errors=True)
            shutil.rmtree(replica_root, ignore_errors=True)
            _write_json(DEFAULT_REPORT, payload)
            return payload
        selection, candidate = selected
        selected_members = tuple(candidate.member_ids)

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
            "predicted_interaction": candidate.predicted_interaction,
            "predicted_gain_vs_strongest_single": candidate.predicted_interaction,
            "uncertainty": candidate.uncertainty,
            "resource_cost": candidate.resource_cost,
            "support": candidate.support,
            "selection_utility": selection.utility,
            "candidate_sets_considered": [list(item) for item in candidate_sets_ordered],
            "observed_pair_count": len(observed_pairs),
        }

        # ---- prediction for every unseen pair (full calibration surface) ----
        predictions: list[dict[str, Any]] = []
        for members_pair in PAIR_MEMBER_SETS:
            predicted = learner.candidate(members_pair, allow_observed=False)
            predictions.append(
                {
                    "member_ids": list(members_pair),
                    "seen_in_train_evidence": frozenset(members_pair) in observed_pairs,
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
        control_sets = {
            "no_learning": _control_selection("no_learning", candidate_sets_ordered),
            "random_combination": _control_selection("random_combination", candidate_sets_ordered),
            "fixed_combination": _control_selection("fixed_combination", candidate_sets_ordered),
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

        # ---- real execution on unseen contexts for object + controls ----
        unseen_contexts = contexts[TRAIN_CONTEXT_COUNT:]
        object_index = _outcome_index(matrix_episodes)
        object_cells = {
            task.task_id: _pair_cells(object_index, task.task_id) for task in unseen_contexts
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

        object_score = score(selected_members)
        control_scores = {name: score(value) for name, value in control_sets.items()}
        strongest_control = max(
            control_scores.items(),
            key=lambda item: item[1]["mean_gain_vs_strongest_single"],
        )

        # ---- calibration: predicted vs realized on every unseen pair ----
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
        errors = [float(item["absolute_error"]) for item in comparable]
        sign_hits = [bool(item["sign_matches"]) for item in comparable]
        calibration_pooled = {
            "unseen_pair_count": len(unseen_rows),
            "comparable_pair_count": len(comparable),
            "mean_absolute_error": float(sum(errors) / len(errors)) if errors else None,
            "sign_match_rate": (
                float(sum(1 for item in sign_hits if item) / len(sign_hits)) if sign_hits else None
            ),
            "rows": calibration_rows,
        }

        # ---- family-coverage attribution (preregistration section 3 constraint 2) ----
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

        # ---- self-consistency: learner must reproduce the train-only candidates ----
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
            reason = str(episode["stop_reason"])
            stop_reasons[reason.split(":")[0]] += 1

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

        matrix_constant = bool(
            all(len({value for value in counts.values()}) == 1 for counts in object_cells.values())
        )
        all_cells_constant = bool(
            all(
                len(set(item["success"] for item in matrix_episodes if item["task_id"] == task_id))
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
            # the entry audit addition: admissible evidence also requires that
            # the interventions behind it actually executed
            and intervention_reality["interventions_happened"]
        )
        unseen_identity = bool(
            frozenset(selected_members) not in observed_pairs
            and all(member in {item.member_id for item in profiles} for member in selected_members)
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
            "real_execution": bool(executed_entries and provenance_ok and safety_violations == 0),
            "rejection_recovery": bool(
                all(rejections.values()) and recovery["selection_reproduced"]
            ),
            "prediction_binding": bool(
                prediction_binding["bound_before_scoring"]
                and prediction_binding["parent_checkpoint_digest"]
                == parent_checkpoint["checkpoint_digest"]
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
            elif replay_matches and not margin_cleared:
                outcome = "transfer_no_gain"
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
                    "margin_provenance": "P5.2a frozen FROZEN_MARGIN constant, same value, new metric object",
                    "candidate_pairs": [list(item) for item in PAIR_MEMBER_SETS],
                    "learner_observes_train_only_records": True,
                    "recovery_effect_disclosure": "not measured this gate; recorded as 0 without fabrication",
                    "semantic_mapping_disclosure": "member id -> training family mapping stored in runner config only",
                    "sequence_generalization_claim": "none: gains are attributed to complementary family coverage only",
                },
                "prediction_binding": prediction_binding,
                "prediction_surface": predictions,
                "object": {
                    "member_ids": list(selected_members),
                    "score": object_score,
                },
                "controls": {
                    name: {"member_ids": list(control_sets[name]), "score": value}
                    for name, value in control_scores.items()
                },
                "control_summary": {
                    "strongest_control": strongest_control[0],
                    "strongest_control_member_ids": list(control_sets[strongest_control[0]]),
                    "strongest_control_gain": best_control_gain,
                    "object_gain": object_gain,
                    "margin": MARGIN,
                    "required": best_control_gain + MARGIN,
                    "margin_cleared": margin_cleared,
                    "collaboration_holds": collaboration_holds,
                },
                "calibration": calibration_pooled,
                "family_coverage": coverage,
                "success_matrix": success_matrix,
                "matrix": {
                    "episodes": len(matrix_episodes),
                    "cells_per_unseen_context": {
                        task_id: dict(sorted(counts.items()))
                        for task_id, counts in object_cells.items()
                    },
                    "train_context_cells_constant": matrix_constant,
                    "all_cells_constant": all_cells_constant,
                    "stop_reason_classes": dict(sorted(stop_reasons.items())),
                },
                "intervention_reality": intervention_reality,
                "blocking_conditions": blocking_conditions,
                "evaluator": {
                    "train_only_records": len(train_only_records),
                    "observed_pairs": sorted(sorted(item) for item in observed_pairs),
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
                    f"completed: outcome={outcome}; object={list(selected_members)} "
                    f"gain={round(object_gain, 6)} vs strongest control "
                    f"{strongest_control[0]}={round(best_control_gain, 6)} "
                    f"(required {round(best_control_gain + MARGIN, 6)}); "
                    f"unseen_pairs={len(unseen_rows)}"
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
    checkpoint_path = Path(tempfile.mkdtemp(prefix="p52c-recovery-")) / "learner.json"
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
        expected = None
        for members_pair in candidate_sets:
            digest_probe = InteractionGroupTransferLearner.from_checkpoint(parent_checkpoint)
            predicted = digest_probe.candidate(members_pair, allow_observed=False)
            if predicted is not None:
                expected = {
                    "group_id": predicted.group_id,
                    "member_ids": list(predicted.member_ids),
                    "predicted_interaction": predicted.predicted_interaction,
                }
                break
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
    results["observed_pair_not_reselected"] = all(
        learner.candidate(members_pair, allow_observed=False) is None
        for members_pair in candidate_sets
        if frozenset(members_pair)
        in {frozenset(item.member_ids) for item in learner.observed_records}
    )
    return results


def _intervention_reality(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    """Assert every non-baseline intervention actually executed something.

    Returns per-context zero-step counts.  ``interventions_happened`` is false
    when any non-``(F,F)`` cell produced no action, which means the task was
    already satisfied before the intervention could take effect and the cell's
    recorded outcome is not attributable to its members.
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
                "object": (result.get("object") or {}).get("member_ids"),
                "control_summary": result.get("control_summary"),
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

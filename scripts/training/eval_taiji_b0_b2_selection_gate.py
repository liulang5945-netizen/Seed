"""WP-5a / B2 selection gate: does the B1-driven selector's chosen combination
produce real task benefit once executed through the real contract?

Preregistration (frozen): plans/reference/M5_WP5A_B2_SELECTION_PREREGISTRATION_FROZEN_20260915.md
Estimation targets and G1-G6 are inherited from the frozen Route B preregistration
(``M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md``); this module freezes nothing
new and changes no threshold.

What B2 adds over B1
--------------------
B1 only asked whether the representation can *distinguish* candidate combinations.
B2 asks the downstream question: the selector picks one pair by
``argmax predicted_interaction``, that pair is then executed on the candidate
surface through the real contract, and its gain is compared against the frozen
reference and the deployable control.

Design, and why the arms are free
---------------------------------
The candidate surface is executed as one full factorial (3 cells x 6 contexts x
11 cells x 2 repeats = 396 episodes), so the selected pair, its two singleton
lesion arms, the blank baseline, every fixed pair and every singleton all come out
of the *same* execution pass.  Nothing is re-run to build a control, which means
no control can drift away from the treatment.

Honest limits carried into the report
-------------------------------------
* the representation is not conditioned on context, so all three candidate cells
  receive the *same* selected pair -- reported as such, never as per-cell routing;
* ``ceiling_gain`` is 2.0 whenever every context is clearable, so the gain axis has
  no headroom (risk clause R-ZS): the report always carries ``required`` /
  ``available`` / ``ceiling_gain`` next to any mean gain;
* discrimination and selection scores are not collaboration evidence: H2 lives in
  G5's real-execution gate, and a mean gain above the oracle is still not one
  observed collaboration event (L6).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import eval_taiji_b0_b1_representation_gate as b1  # noqa: E402
import probe_taiji_b0_handoff_feasibility as hfp  # noqa: E402
import probe_taiji_b0_structure_space as ssp  # noqa: E402

from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupTransferLearner,
    build_member_evidence,
)
from taiji.interaction_groups import InteractionTraceCorpus  # noqa: E402

REPORT_FORMAT = "taiji-b0-b2-selection-gate-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_WP5A_B2_SELECTION_PREREGISTRATION_FROZEN_20260915.md"
ROUTE_B_PREREGISTRATION = "plans/reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_b2_selection_20260915.json"
FROZEN_ROUTE_A_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_triple_prime_representation_repair_20260913.json"
)

#: Frozen by WP-5a section 2.6 (G6).  Exceeding it is recorded as a failure; the
#: cap is never raised after the fact.
WALL_CAP_SECONDS = 60.0

#: H2 primary reference (Route B section 2).  ``required`` is derived from the
#: frozen margin below, never hand-filled.
PRIMARY_REFERENCE = "all_singleton_oracle"
NO_LEARNING_FALLBACK_PAIR = ("member-a", "member-b")

#: The same three candidate cells B1 evaluated, so the surfaces stay comparable.
CANDIDATE_LABELS = ("create__observation", "create__override", "create__mismatch")

STATIC_CHECK_SCOPE = (
    "scripts/training/eval_taiji_b0_b2_selection_gate.py",
    "scripts/training/eval_taiji_b0_b1_representation_gate.py",
    "scripts/training/probe_taiji_b0_structure_space.py",
    "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
)


def _margin() -> float:
    """Frozen margin from the route-A control summary (never re-derived)."""

    return float(
        json.loads(FROZEN_ROUTE_A_REPORT.read_text(encoding="utf-8"))["control_summary"]["margin"]
    )


def _cell_of_task(task_id: str) -> str:
    """``b1cand-create__observation-600`` -> ``create__observation``."""

    return task_id.split("-", 1)[1].rsplit("-", 1)[0]


def _cell_tables(
    episodes: list[dict[str, Any]],
    candidate_tasks: tuple[Any, ...],
    member_ids: tuple[str, ...],
) -> dict[str, Any]:
    """One ``ContextTable`` per candidate cell, plus the pooled table."""

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_cell.setdefault(_cell_of_task(str(episode["task_id"])), []).append(episode)
    ids_by_cell: dict[str, list[str]] = {}
    for task in candidate_tasks:
        ids_by_cell.setdefault(_cell_of_task(task.task_id), []).append(task.task_id)

    tables: dict[str, Any] = {}
    for label, cell_episodes in sorted(by_cell.items()):
        outcomes = hfp.outcome_by_cell(cell_episodes)
        tables[label] = hfp.table_from_outcomes(outcomes, sorted(ids_by_cell[label]), member_ids)
    pooled_outcomes = hfp.outcome_by_cell(episodes)
    tables["_pooled"] = hfp.table_from_outcomes(
        pooled_outcomes, sorted(task.task_id for task in candidate_tasks), member_ids
    )
    return tables


def _lesion_attribution(table: Any, pair: tuple[str, ...]) -> dict[str, Any]:
    """Does the pair gain on contexts where *both* of its members alone fail?

    This is the G5 attribution test: a pair that wins where a member already wins
    is coverage, not collaboration.
    """

    gained = [c for c in table.contexts if table.combination(c, pair) > 0.0]
    attributed = [c for c in gained if all(table.singleton(c, member) <= 0.0 for member in pair)]
    return {
        "pair": "+".join(pair),
        "successful_contexts": len(gained),
        "attributed_contexts": len(attributed),
        "holds": bool(gained) and len(attributed) == len(gained),
        "unattributed_contexts": len(gained) - len(attributed),
    }


def _mean_outcome(table: Any, key: tuple[str, ...]) -> float:
    if not key:
        values = [table.blank(c) for c in table.contexts]
    elif len(key) == 1:
        values = [table.singleton(c, key[0]) for c in table.contexts]
    else:
        values = [table.combination(c, key) for c in table.contexts]
    return sum(values) / len(values) if values else 0.0


def _select_pair(
    learner: InteractionGroupTransferLearner, pairs: list[tuple[str, ...]]
) -> tuple[tuple[str, ...] | None, dict[str, Any]]:
    """Frozen selection protocol: argmax ``predicted_interaction`` among eligible pairs."""

    predictions = b1._predicted_map(learner, pairs)
    outcome = learner.select(pairs, unseen_only=False)
    if outcome is None:
        return None, {"predictions": predictions, "select_returned_none": True}
    selection, candidate = outcome
    return tuple(selection.member_ids), {
        "predictions": predictions,
        "select_returned_none": False,
        "group_id": selection.group_id,
        "member_ids": list(selection.member_ids),
        "utility": round(float(selection.utility), 9),
        "uncertainty": round(float(candidate.uncertainty), 9),
        "resource_cost": round(float(selection.resource_cost), 9),
        "observations": selection.observations,
    }


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "route_b_preregistration": ROUTE_B_PREREGISTRATION,
        "scope": (
            "B2 offline learned selection evaluated by real contract execution on the "
            "frozen candidate surface; selection score is a prior, never a benefit claim"
        ),
        "static_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": [
                "python -m py_compile <scope files>",
                "python -m ruff check .",
                "python -m black --no-cache --check <scope files>",
            ],
        },
    }
    try:
        frozen = ssp.load_frozen()
        counterfactual = ssp.load_counterfactual()
        margin = _margin()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        _grid, training_cells, candidate_cells = b1._cells()
        labels = tuple(cell["label"] for cell in candidate_cells)
        if labels != CANDIDATE_LABELS:
            raise SystemExit(f"candidate surface drifted from the frozen split: {labels}")

        # ---- members + fit corpus: byte-identical construction to B1
        member_start = time.perf_counter()
        embedder = frozen.DocumentEmbedder()
        members = frozen._train_members(embedder)
        member_seconds = time.perf_counter() - member_start

        fit_tasks = b1._tasks_for_cells(frozen, training_cells, (b1.FIT_CONTEXT_STEP,), "b1fit")
        indomain_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.INDOMAIN_HOLDOUT_STEP,), "b1ind"
        )
        candidate_tasks = b1._tasks_for_cells(frozen, candidate_cells, b1.CANDIDATE_STEPS, "b1cand")

        matrix_start = time.perf_counter()
        fit_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, fit_tasks
        )
        indomain_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, indomain_tasks
        )
        candidate_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, candidate_tasks
        )
        matrix_seconds = time.perf_counter() - matrix_start

        fit_ids = {task.task_id for task in fit_tasks}
        candidate_ids = {task.task_id for task in candidate_tasks}
        if fit_ids & candidate_ids:
            raise SystemExit("leakage: candidate contexts inside the fit corpus")

        # ---- official train-only fit surfaces (same call chain as B1)
        fit_start = time.perf_counter()
        projected_fit = frozen._project(fit_episodes)
        projected_indomain = frozen._project(indomain_episodes)
        corpus = InteractionTraceCorpus(train=projected_fit, holdout=projected_indomain)
        revision = next(iter(corpus.train_checkpoint_revisions))
        profiles = build_member_evidence(
            corpus.train,
            source_trace_digest=corpus.train_trace_digest,
            checkpoint_revision=revision,
        )
        evaluator = InteractionGroupEvaluator(
            InteractionGroupEvaluatorConfig(maximum_resource_cost=b1.FIT_RESOURCE_CAP)
        )
        records = evaluator.train_only_candidates(corpus)
        learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
        learner.observe_members(profiles)
        observed_records = learner.observe_records(records)
        fit_seconds = time.perf_counter() - fit_start

        member_ids = tuple(frozen.MEMBER_IDS)
        pairs = list(itertools.combinations(member_ids, 2))
        predictions = b1._predicted_map(learner, pairs)
        predictions_vary = len(set(round(value, 9) for value in predictions.values())) > 1
        feature_rank = learner.feature_rank()

        # ---- frozen selection protocol
        select_start = time.perf_counter()
        selected_pair, selection_detail = _select_pair(learner, pairs)
        select_seconds = time.perf_counter() - select_start
        if selected_pair is None:
            raise SystemExit("selector returned no candidate; B2 has no treatment arm")

        # ---- candidate-surface execution analysis
        tables = _cell_tables(candidate_episodes, candidate_tasks, member_ids)
        pooled = tables["_pooled"]
        precheck = hfp.load_precheck()
        dictionary = hfp.load_dictionary()
        # Frozen Route B section 2 derives ``required`` and the k-criterion PER CELL
        # (n = 6).  The pooled table (n = 18) is reported as a secondary diagnostic
        # only: its k scale differs even though ``required`` does not.
        reference_rows = precheck.reference_requirements(
            pooled, margin=margin, candidates=ssp.CANDIDATES_FOR_REVIEW
        )
        primary = next(row for row in reference_rows if row["reference"] == PRIMARY_REFERENCE)
        required = float(primary["required"])

        per_cell: list[dict[str, Any]] = []
        for label in labels:
            table = tables[label]
            gains = {
                "+".join(pair): dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair)
                for pair in table.pair_cells()
            }
            selected_trajectory = precheck.classify_pair_trajectory(table, selected_pair)
            any_pair_interleaved = sum(
                precheck.classify_pair_trajectory(table, pair)["contexts_interleaved"]
                for pair in table.pair_cells()
            )
            cell_references = precheck.reference_requirements(
                table, margin=margin, candidates=ssp.CANDIDATES_FOR_REVIEW
            )
            cell_primary = next(
                row for row in cell_references if row["reference"] == PRIMARY_REFERENCE
            )
            selected_gain = gains["+".join(selected_pair)]
            per_cell.append(
                {
                    "cell": label,
                    "contexts": len(table.contexts),
                    "gain_vs_all_singleton_oracle": {
                        key: round(value, 6) for key, value in sorted(gains.items())
                    },
                    "selected_pair_gain": round(selected_gain, 6),
                    "selected_pair_meets_required": bool(selected_gain >= required),
                    "required": required,
                    "required_combination_only_contexts": int(
                        cell_primary["required_combination_only_contexts"]
                    ),
                    "available_combination_only_contexts": int(
                        cell_primary["available_combination_only_contexts"]
                    ),
                    "context_count": int(cell_primary["context_count"]),
                    "ceiling_gain": round(float(cell_primary["ceiling_gain"]), 6),
                    "max_clearable_reference": round(
                        float(cell_primary["max_clearable_reference"]), 6
                    ),
                    "lesion": _lesion_attribution(table, selected_pair),
                    "selected_pair_interleaved_contexts": int(
                        selected_trajectory["contexts_interleaved"]
                    ),
                    "any_pair_interleaved_contexts": any_pair_interleaved,
                }
            )

        selected_gain_pooled = dictionary.policy_mean_gain_vs_all_singleton_oracle(
            pooled, selected_pair
        )
        pooled_lesion = _lesion_attribution(pooled, selected_pair)
        pooled_selected_interleaved = int(
            precheck.classify_pair_trajectory(pooled, selected_pair)["contexts_interleaved"]
        )
        pooled_any_interleaved = sum(
            precheck.classify_pair_trajectory(pooled, pair)["contexts_interleaved"]
            for pair in pooled.pair_cells()
        )

        # ---- controls: every arm is read out of the same factorial execution
        pooled_gains = {
            "+".join(pair): dictionary.policy_mean_gain_vs_all_singleton_oracle(pooled, pair)
            for pair in pooled.pair_cells()
        }
        singleton_outcomes = {member: _mean_outcome(pooled, (member,)) for member in member_ids}
        best_singleton = max(singleton_outcomes, key=lambda member: singleton_outcomes[member])
        random_generator = random.Random(b1.CALIBRATION_RANDOM_SEED)
        random_pair = tuple(sorted(random_generator.sample(list(member_ids), 2)))
        best_observed_pair = max(pooled_gains, key=lambda label: pooled_gains[label])
        additive = {
            "+".join(pair): (
                next(item.contribution for item in profiles if item.member_id == pair[0])
                + next(item.contribution for item in profiles if item.member_id == pair[1])
            )
            for pair in pairs
        }
        selected_utility = _mean_outcome(pooled, selected_pair)
        controls = {
            "selected_pair": {
                "pair": "+".join(selected_pair),
                "mean_outcome": round(selected_utility, 6),
                "gain_vs_all_singleton_oracle": round(selected_gain_pooled, 6),
            },
            "no_learning": {
                "pair": "+".join(NO_LEARNING_FALLBACK_PAIR),
                "gain_vs_all_singleton_oracle": round(
                    pooled_gains["+".join(NO_LEARNING_FALLBACK_PAIR)], 6
                ),
                "note": "unfitted representation has no candidate, so it falls back to a fixed pair",
            },
            "best_fixed_singleton": {
                "member": best_singleton,
                "mean_outcome": round(singleton_outcomes[best_singleton], 6),
                "deployable": True,
                "all_singleton_mean_outcomes": {
                    member: round(value, 6) for member, value in sorted(singleton_outcomes.items())
                },
                "tied_at_floor": bool(
                    len({round(value, 9) for value in singleton_outcomes.values()}) == 1
                ),
                "note": (
                    "H1/G4 control: deployable strongest-singleton routing. When every "
                    "singleton ties (all fail on a combination-only surface) the tie-break "
                    "is deterministic but arbitrary; disclosed so the control is not read "
                    "as a distinctive winner"
                ),
            },
            "best_observed_fixed_pair": {
                "pair": best_observed_pair,
                "gain_vs_all_singleton_oracle": round(pooled_gains[best_observed_pair], 6),
                "deployable": False,
                "note": "diagnostic only; it is not a deployable policy",
            },
            "random_pair": {
                "pair": "+".join(random_pair),
                "gain_vs_all_singleton_oracle": round(pooled_gains["+".join(random_pair)], 6),
            },
            "a_count": {"prediction_equal_for_all_pairs": True, "discriminates": False},
            "additive_train_only": {
                "ranking": {
                    key: round(value, 6)
                    for key, value in sorted(additive.items(), key=lambda item: -item[1])
                },
                "note": "train-only additive control; no pair-interaction term",
            },
        }

        # ---- G3: renaming + fit-order permutation must not move the selection
        rotation = {
            name: member_ids[(index + 1) % len(member_ids)] for index, name in enumerate(member_ids)
        }
        rotated_projected = tuple(
            dataclasses.replace(
                item,
                events=tuple(
                    dataclasses.replace(
                        event, owner_id=rotation.get(event.owner_id, event.owner_id)
                    )
                    for event in item.events
                ),
            )
            for item in projected_fit
        )
        rotated_corpus = InteractionTraceCorpus(train=rotated_projected, holdout=projected_indomain)
        rotated_profiles = build_member_evidence(
            rotated_corpus.train,
            source_trace_digest=rotated_corpus.train_trace_digest,
            checkpoint_revision=revision,
        )
        rotated_learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
        rotated_learner.observe_members(rotated_profiles)
        rotated_learner.observe_records(evaluator.train_only_candidates(rotated_corpus))
        rotated_pairs = list(itertools.combinations(sorted(rotation.values()), 2))
        rotated_selected, _ = _select_pair(rotated_learner, rotated_pairs)
        renaming_stable = bool(
            rotated_selected is not None
            and tuple(sorted(rotation[name] for name in selected_pair)) == tuple(rotated_selected)
        )

        shuffled = list(projected_fit)
        random_generator.shuffle(shuffled)
        shuffled_learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
        shuffled_learner.observe_members(profiles)
        shuffled_learner.observe_records(
            evaluator.train_only_candidates(
                InteractionTraceCorpus(train=tuple(shuffled), holdout=projected_indomain)
            )
        )
        shuffled_selected, _ = _select_pair(shuffled_learner, pairs)
        order_stable = bool(
            shuffled_selected is not None and tuple(shuffled_selected) == selected_pair
        )

        # ---- checkpoint + fresh-process restore + tamper
        checkpoint_payload = learner.checkpoint()
        ckpt_dir = Path(tempfile.mkdtemp(prefix="b2sel-"))
        ckpt_path = ckpt_dir / "transfer-learner.json"
        ckpt_path.write_text(json.dumps(checkpoint_payload, sort_keys=True), encoding="utf-8")
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(ckpt_path)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        restore_ok = False
        if child.returncode == 0:
            child_output = json.loads(child.stdout.strip().splitlines()[-1])
            restore_ok = tuple(child_output.get("selected_pair") or ()) == selected_pair
        tampered_payload = dict(checkpoint_payload)
        tampered_payload["coefficients"] = [
            value + 1.0 for value in tampered_payload.get("coefficients", ())
        ]
        tamper_rejected = False
        try:
            InteractionGroupTransferLearner.from_checkpoint(tampered_payload)
        except Exception:  # noqa: BLE001
            tamper_rejected = True
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        reality = frozen._intervention_reality(candidate_episodes)
        total_wall = time.perf_counter() - started

        # ---- gates (Route B section 5; never merged into one)
        g1 = bool(reality["interventions_happened"])
        g2 = bool(not (fit_ids & candidate_ids))
        g3 = bool(renaming_stable and order_stable)
        g4 = bool(selected_utility > singleton_outcomes[best_singleton])
        g5 = bool(
            all(row["selected_pair_meets_required"] for row in per_cell)
            and pooled_lesion["holds"]
            and pooled_selected_interleaved > 0
        )
        g6 = bool(total_wall <= WALL_CAP_SECONDS)
        gates = {
            "G1_intervention_reality": g1,
            "G2_data_isolation": g2,
            "G3_ranking_stability": g3,
            "G4_task_gate_H1": g4,
            "G5_collaboration_gate_H2": g5,
            "G6_budget_gate": g6,
        }

        mechanical_ok = g1 and g2 and g3 and g6
        representation_ok = bool(feature_rank > 1 and predictions_vary)
        if not mechanical_ok:
            outcome = "failed"
            outcome_detail = "mechanical_or_contract_gate_failed"
        elif not representation_ok:
            outcome = "representation_ineffective"
            outcome_detail = "representation_cannot_distinguish_candidates"
        elif g4 and g5:
            outcome = "collaboration_supported"
            outcome_detail = "G4_and_G5_passed_with_lesion_attribution"
        elif g4:
            outcome = "routing_only"
            outcome_detail = "G4_passed_G5_failed_no_collaboration_claim"
        else:
            # The frozen three-state list has no label for "mechanical gates pass
            # but the selected pair does not even beat the deployable control".
            # Recorded under ``failed`` with an explicit detail rather than
            # inventing a fifth state, and flagged as a preregistration gap.
            outcome = "failed"
            outcome_detail = "no_task_benefit: G4 and G5 both failed while mechanical gates passed"

        payload.update(
            {
                "status": "completed",
                "outcome": outcome,
                "outcome_detail": outcome_detail,
                "preregistration_gap": (
                    None
                    if outcome_detail
                    != "no_task_benefit: G4 and G5 both failed while mechanical gates passed"
                    else "frozen three-state list defines no label for (G4 fail, G5 fail) with "
                    "mechanical gates green; reported as failed + detail, needs a plan amendment"
                ),
                "experiment_passed": bool(outcome == "collaboration_supported"),
                "record": {
                    "commit": commit,
                    "rule_revision": frozen.RULE_REVISION,
                    "composition_rule": frozen.COMPOSITION_RULE,
                    "wall_cap_seconds": WALL_CAP_SECONDS,
                    "elapsed_seconds": round(total_wall, 3),
                },
                "corpus": {
                    "training_cells": [cell["label"] for cell in training_cells],
                    "candidate_cells": list(labels),
                    "fit_contexts": len(fit_ids),
                    "indomain_holdout_contexts": len(indomain_tasks),
                    "candidate_contexts": len(candidate_ids),
                    "episodes": {
                        "fit": len(fit_episodes),
                        "indomain_holdout": len(indomain_episodes),
                        "candidate": len(candidate_episodes),
                    },
                    "candidate_episode_budget": (
                        f"{len(labels)} cells x {len(b1.CANDIDATE_STEPS)} contexts x "
                        f"11 cells x {b1.REPEATS} repeats = {len(candidate_episodes)}"
                    ),
                    "fit_resource_cap_disclosure": (
                        f"evaluator maximum_resource_cost raised to {b1.FIT_RESOURCE_CAP} for the "
                        "fit corpus; candidate-cell outcomes are execution-only and never enter the fit"
                    ),
                },
                "selection": {
                    "protocol": "learner.select(6 pairs, unseen_only=False) -> argmax predicted_interaction",
                    "selected_pair": "+".join(selected_pair),
                    "predictions": {label: round(value, 6) for label, value in predictions.items()},
                    "detail": selection_detail,
                    "context_conditioned": False,
                    "context_conditioning_note": (
                        "the representation is not conditioned on context, so all three candidate "
                        "cells receive the same selected pair; this is reported as the current "
                        "boundary and is not per-cell routing"
                    ),
                    "seconds": round(select_seconds, 6),
                },
                "representation": {
                    "profiles": {
                        item.member_id: {
                            "contribution": round(item.contribution, 6),
                            "observations": item.observations,
                            "surface": list(item.surface),
                        }
                        for item in profiles
                    },
                    "train_only_candidate_records": len(observed_records),
                    "feature_rank": feature_rank,
                    "predictions_vary": predictions_vary,
                },
                "estimation_targets": {
                    "primary_reference": PRIMARY_REFERENCE,
                    "primary_reference_gain": float(primary["reference_gain"]),
                    "required": required,
                    "margin": margin,
                    "per_cell_derivation": (
                        "frozen Route B section 2 derives required/k per cell with n=6; "
                        "each cell entry carries its own reference_requirements"
                    ),
                    "per_cell_context_count": len(b1.CANDIDATE_STEPS),
                    "per_cell_required_combination_only_contexts": int(
                        per_cell[0]["required_combination_only_contexts"]
                    ),
                    "per_cell_available_combination_only_contexts": int(
                        per_cell[0]["available_combination_only_contexts"]
                    ),
                    "pooled_context_count": int(primary["context_count"]),
                    "pooled_required_combination_only_contexts": int(
                        primary["required_combination_only_contexts"]
                    ),
                    "pooled_available_combination_only_contexts": int(
                        primary["available_combination_only_contexts"]
                    ),
                    "pooled_feasible": bool(primary["feasible"]),
                    "pooled_all_references": reference_rows,
                    "risk_clause_R_ZS": (
                        "mean gains must always be reported with required/available/ceiling_gain"
                    ),
                },
                "cells": per_cell,
                "pooled": {
                    "selected_pair_gain_vs_all_singleton_oracle": round(selected_gain_pooled, 6),
                    "all_pair_gains": {
                        key: round(value, 6) for key, value in sorted(pooled_gains.items())
                    },
                    "singleton_mean_outcomes": {
                        member: round(value, 6)
                        for member, value in sorted(singleton_outcomes.items())
                    },
                    "lesion": pooled_lesion,
                    "selected_pair_interleaved_contexts": pooled_selected_interleaved,
                    "any_pair_interleaved_contexts": pooled_any_interleaved,
                    "interleaved_note": (
                        "the G5 criterion reads the SELECTED pair's own interleaving; the "
                        "any-pair sum is reported only so a positive number cannot be "
                        "mistaken for the treatment arm's interleaving"
                    ),
                    "selected_pair_mean_outcome": round(selected_utility, 6),
                },
                "controls": controls,
                "causal_interaction": {
                    "+".join(pair): round(
                        _mean_outcome(pooled, pair)
                        - _mean_outcome(pooled, (pair[0],))
                        - _mean_outcome(pooled, (pair[1],))
                        + _mean_outcome(pooled, ()),
                        6,
                    )
                    for pair in pairs
                },
                "stability": {
                    "renaming_stable": renaming_stable,
                    "order_permutation_stable": order_stable,
                },
                "checkpoint": {
                    "restore_matches_parent": restore_ok,
                    "tamper_rejected": tamper_rejected,
                },
                "cost": {
                    "member_training_seconds": round(member_seconds, 3),
                    "matrix_seconds": round(matrix_seconds, 3),
                    "fit_seconds": round(fit_seconds, 3),
                    "episodes_total": len(fit_episodes)
                    + len(indomain_episodes)
                    + len(candidate_episodes),
                },
                "intervention_reality": reality,
                "gates": gates,
                "growth_admitted": False,
                "can_promote": False,
                "boundary": (
                    "selection and discrimination scores are not collaboration evidence; "
                    "the load-bearing structural factor is still one (existence precondition); "
                    "a mean gain above the oracle is not one observed collaboration event"
                ),
                "elapsed_seconds": round(total_wall, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    b1._write_json(DEFAULT_REPORT, payload)
    return payload


def _child(ckpt_path: Path) -> int:
    """Fresh-process restore must reproduce the *selection*, not just the scores."""

    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    learner = InteractionGroupTransferLearner.from_checkpoint(payload)
    member_ids = tuple(sorted(profile.member_id for profile in learner.profiles))
    pairs = list(itertools.combinations(member_ids, 2))
    selected, detail = _select_pair(learner, pairs)
    print(json.dumps({"selected_pair": list(selected or ()), "predictions": detail["predictions"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.child is not None:
        return _child(args.child)
    result = run_gate()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "outcome_detail": result.get("outcome_detail"),
                "selected_pair": (result.get("selection") or {}).get("selected_pair"),
                "gates": result.get("gates"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("experiment_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

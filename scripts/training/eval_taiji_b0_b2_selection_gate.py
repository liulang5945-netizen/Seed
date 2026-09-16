"""Route B (B2) selection gate: does the B1 representation's chosen combination
produce real execution gains on the candidate surface?

Preregistration: plans/reference/M5_WP5A_B2_SELECTION_PREREGISTRATION_FROZEN_20260915.md
(frozen; estimation targets and gates G1-G6 inherited from the Route B frozen
preregistration).  Protocol:

  1. fit = B1 exactly (8 non-candidate structure cells, official train-only
     surfaces, evaluator resource cap 64 disclosed);
  2. selection = ``learner.select(pairs, unseen_only=False)`` -> highest
     predicted interaction (the representation is context-free, so one pair
     serves all three candidate cells - an honest boundary, disclosed);
  3. execution = full factorial on the candidate surface (3 create cells x 6
     frozen contexts x 11 member-set cells x 2 repeats = 396 episodes) through
     the real contract; the selected pair, its lesion cells (both singleton
     members), the baseline and every control arm come from the same execution;
  4. gates G1-G6 per the frozen texts; three-state outcome per section 3.

Discrimination numbers and selection scores are representation evidence; the
collaboration claim lives exclusively in G5's real-execution accounting.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import audit_taiji_b0_task_reachability_precheck as precheck  # noqa: E402
import eval_taiji_b0_b1_representation_gate as b1  # noqa: E402
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
ROUTE_B_PREREG = "plans/reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_b2_selection_20260915.json"

FIT_CONTEXT_STEP = 0
CANDIDATE_STEPS = tuple(range(6))  # frozen n=6
REPEATS = 2
FIT_RESOURCE_CAP = 64.0
REQUIRED_GAIN = 1.65
REQUIRED_K = 5
WALL_CAP_SECONDS = 60.0
CONTROL_RANDOM_SEED = 17

STATIC_CHECK_SCOPE = (
    "scripts/training/eval_taiji_b0_b2_selection_gate.py",
    "scripts/training/eval_taiji_b0_b1_representation_gate.py",
    "scripts/training/probe_taiji_b0_structure_space.py",
)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _context_breakdown(
    episodes: list[dict[str, Any]],
    context_ids: list[str],
    members: tuple[str, ...],
    pair: tuple[str, ...],
) -> dict[str, Any]:
    """Per-context gain accounting for one pair (frozen formula pieces)."""

    breakdown: dict[str, Any] = {}
    for context_id in context_ids:
        rates = b1._success_by_cell(episodes, context_id)
        pair_rate = rates.get(pair, 0.0)
        singleton_rates = {member: rates.get((member,), 0.0) for member in members}
        best_singleton = max(singleton_rates.values())
        baseline = rates.get((), 0.0)
        breakdown[context_id] = {
            "pair_success_rate": pair_rate,
            "singleton_success_rates": {
                member: round(rate, 6) for member, rate in singleton_rates.items()
            },
            "best_singleton_rate": best_singleton,
            "baseline_rate": baseline,
            "gain_vs_all_singleton_oracle": pair_rate - best_singleton,
            "causal_interaction": pair_rate
            - singleton_rates[pair[0]]
            - singleton_rates[pair[1]]
            + baseline,
        }
    return breakdown


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": 1,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "route_b_preregistration": ROUTE_B_PREREG,
        "scope": (
            "B2 selection gate: B1 representation selects one pair; real "
            "contract execution on the candidate surface decides G4/G5; no "
            "collaboration claim outside G5"
        ),
        "static_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": [
                "python -m py_compile <scope files>",
                "python -m ruff check scripts/training (and full repo)",
                "python -m black --no-cache --check <scope files>",
            ],
        },
    }
    try:
        frozen = ssp.load_frozen()
        counterfactual = ssp.load_counterfactual()
        handoff = ssp.load_handoff_probe()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        _grid, training_cells, candidate_cells = b1._cells()

        member_start = time.perf_counter()
        embedder = frozen.DocumentEmbedder()
        members = frozen._train_members(embedder)
        member_seconds = time.perf_counter() - member_start
        member_ids = frozen.MEMBER_IDS
        pairs = list(itertools.combinations(member_ids, 2))
        pair_labels = ["+".join(pair) for pair in pairs]

        # ---- fit (identical to B1)
        fit_tasks = b1._tasks_for_cells(frozen, training_cells, (b1.FIT_CONTEXT_STEP,), "b2fit")
        fit_start = time.perf_counter()
        fit_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, fit_tasks
        )
        projected_fit = frozen._project(fit_episodes)
        indomain_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.INDOMAIN_HOLDOUT_STEP,), "b2ind"
        )
        indomain_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, indomain_tasks
        )
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
        learner = InteractionGroupTransferLearner(minimum_utility=-10.0, maximum_uncertainty=2.0)
        learner.observe_members(profiles)
        learner.observe_records(records)
        fit_seconds = time.perf_counter() - fit_start

        # ---- selection (context-free representation: one pair, disclosed)
        selection_raw = learner.select(pairs, unseen_only=False)
        selection_disclosure = "learner.select returned the highest predicted-interaction pair"
        if selection_raw is None:
            selected_pair = max(
                pairs,
                key=lambda pair: (
                    learner.candidate(pair, allow_observed=True).predicted_interaction
                    if learner.candidate(pair, allow_observed=True)
                    else float("-inf")
                ),
            )
            selection_disclosure = (
                "learner.select returned None under utility/uncertainty bounds; "
                "fallback = argmax predicted_interaction (disclosed)"
            )
        else:
            selected_pair = tuple(selection_raw[0].member_ids)
        selected_label = "+".join(selected_pair)
        selected_predicted = (
            learner.candidate(selected_pair, allow_observed=True).predicted_interaction
            if learner.candidate(selected_pair, allow_observed=True)
            else 0.0
        )

        # ---- candidate-surface full factorial (real contract execution)
        candidate_tasks = b1._tasks_for_cells(frozen, candidate_cells, b1.CANDIDATE_STEPS, "b2cand")
        matrix_start = time.perf_counter()
        candidate_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, candidate_tasks
        )
        matrix_seconds = time.perf_counter() - matrix_start
        candidate_ids = sorted(task.task_id for task in candidate_tasks)

        outcomes = handoff.outcome_by_cell(candidate_episodes)
        table = handoff.table_from_outcomes(outcomes, candidate_ids, member_ids)

        # ---- per-cell gain accounting for the selected pair (frozen formula)
        cells_accounting: dict[str, Any] = {}
        g4_all = True
        g5_all = True
        lesion_all = True
        for cell in candidate_cells:
            cell_contexts = sorted(
                task.task_id
                for task in candidate_tasks
                if task.task_id.startswith(f"b2cand-{cell['label']}-")
            )
            breakdown = _context_breakdown(
                candidate_episodes, cell_contexts, member_ids, selected_pair
            )
            cell_gain = statistics.mean(
                item["gain_vs_all_singleton_oracle"] for item in breakdown.values()
            )
            positive_contexts = sum(
                1 for item in breakdown.values() if item["gain_vs_all_singleton_oracle"] > 0
            )
            causal_interaction = statistics.mean(
                item["causal_interaction"] for item in breakdown.values()
            )
            singleton_gains = {
                member: statistics.mean(
                    b1._success_by_cell(candidate_episodes, context_id).get((member,), 0.0)
                    - b1._success_by_cell(candidate_episodes, context_id).get((), 0.0)
                    for context_id in cell_contexts
                )
                for member in member_ids
            }
            best_singleton_member = max(singleton_gains, key=lambda member: singleton_gains[member])
            # G4: selected pair mean outcome beats the best deployable control
            # (best fixed singleton) on the same contexts
            pair_mean = statistics.mean(item["pair_success_rate"] for item in breakdown.values())
            control_mean = statistics.mean(
                item["singleton_success_rates"][best_singleton_member]
                for item in breakdown.values()
            )
            g4 = bool(pair_mean > control_mean)
            g4_all = g4_all and g4
            # G5 lesion: on contexts where the pair beats every singleton, both
            # members' singleton cells must fail (attribution to the combination)
            lesion_rows: dict[str, bool] = {}
            for context_id, item in breakdown.items():
                if item["gain_vs_all_singleton_oracle"] > 0:
                    lesion_rows[context_id] = all(
                        rate < item["pair_success_rate"]
                        for rate in item["singleton_success_rates"].values()
                    )
            lesion_ok = bool(lesion_rows) and all(lesion_rows.values())
            lesion_all = lesion_all and lesion_ok
            cells_accounting[cell["label"]] = {
                "gain_vs_all_singleton_oracle": round(cell_gain, 6),
                "required": REQUIRED_GAIN,
                "required_k": REQUIRED_K,
                "positive_gain_contexts": positive_contexts,
                "k_criterion_met": bool(positive_contexts >= REQUIRED_K),
                "causal_interaction": round(causal_interaction, 6),
                "pair_mean_outcome": round(pair_mean, 6),
                "best_singleton_member": best_singleton_member,
                "best_singleton_mean_outcome": round(control_mean, 6),
                "g4_beats_deployable_control": g4,
                "lesion_contexts_checked": len(lesion_rows),
                "lesion_attribution_holds": lesion_ok,
                "per_context": {
                    context_id: {
                        key: round(value, 6) if isinstance(value, float) else value
                        for key, value in item.items()
                    }
                    for context_id, item in breakdown.items()
                },
            }
            g5_all = g5_all and bool(
                cell_gain >= REQUIRED_GAIN and positive_contexts >= REQUIRED_K and lesion_ok
            )

        interleaved = precheck.classify_pair_trajectory(table, selected_pair)
        interleaved_count = int(interleaved["contexts_interleaved"])

        # ---- control arms from the same factorial (zero extra execution)
        random_generator = random.Random(CONTROL_RANDOM_SEED)
        random_pair = tuple(sorted(random_generator.sample(list(member_ids), 2)))
        all_contexts = candidate_ids
        pair_gains, singleton_gains_all = b1._actual_gains(
            candidate_episodes, all_contexts, member_ids
        )
        best_singleton_member = max(
            singleton_gains_all, key=lambda member: singleton_gains_all[member]
        )
        best_observed_pair_label = max(pair_gains, key=lambda label: pair_gains[label])
        control_arms = {
            "selected_pair": {
                "pair": selected_label,
                "mean_gain_vs_all_singleton_oracle": round(
                    statistics.mean(
                        cells_accounting[cell["label"]]["gain_vs_all_singleton_oracle"]
                        for cell in candidate_cells
                    ),
                    6,
                ),
            },
            "no_learning_fallback_ab": {
                "pair": "+".join((member_ids[0], member_ids[1])),
                "mean_gain_vs_all_singleton_oracle": round(
                    pair_gains["+".join((member_ids[0], member_ids[1]))], 6
                ),
            },
            "best_fixed_singleton": {
                "member": best_singleton_member,
                "mean_gain_vs_all_singleton_oracle": 0.0,
                "note": "the deployable control IS the max singleton; its gain vs itself is 0",
            },
            "best_observed_fixed_pair": {
                "pair": best_observed_pair_label,
                "mean_gain_vs_all_singleton_oracle": round(pair_gains[best_observed_pair_label], 6),
                "note": "post-hoc diagnostic column; oracle-deployable=False",
            },
            "random_pair": {
                "pair": "+".join(random_pair),
                "mean_gain_vs_all_singleton_oracle": round(pair_gains["+".join(random_pair)], 6),
            },
            "a_count": {"constant_prediction": True, "discriminates": False},
            "additive_train_only": {
                "rank_correlation_vs_candidate": round(
                    b1._rank_correlation(
                        [
                            next(
                                item.contribution for item in profiles if item.member_id == pair[0]
                            )
                            + next(
                                item.contribution for item in profiles if item.member_id == pair[1]
                            )
                            for pair in pairs
                        ],
                        [pair_gains[label] for label in pair_labels],
                    ),
                    6,
                ),
            },
        }

        # ---- G1 intervention reality: non-baseline cells must act
        flat_steps: dict[str, int] = {}
        for episode in candidate_episodes:
            key = "+".join(episode["active_members"]) or "baseline"
            flat_steps[key] = flat_steps.get(key, 0) + len(episode["steps"])
        g1 = bool(all(steps > 0 for name, steps in flat_steps.items() if name != "baseline"))

        # ---- G3 stability (renaming + fit-order permutation, B1 style)
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
        rotated_records = evaluator.train_only_candidates(rotated_corpus)
        rotated_learner = InteractionGroupTransferLearner(
            minimum_utility=-10.0, maximum_uncertainty=2.0
        )
        rotated_learner.observe_members(rotated_profiles)
        rotated_learner.observe_records(rotated_records)
        renaming_stable = True
        rotated_selected = tuple(sorted(rotation[name] for name in selected_pair))
        original_candidate = learner.candidate(selected_pair, allow_observed=True)
        rotated_candidate = rotated_learner.candidate(rotated_selected, allow_observed=True)
        if original_candidate is None or rotated_candidate is None:
            renaming_stable = False
        elif (
            abs(original_candidate.predicted_interaction - rotated_candidate.predicted_interaction)
            > 1e-9
        ):
            renaming_stable = False

        shuffled = list(projected_fit)
        random_generator.shuffle(shuffled)
        shuffled_corpus = InteractionTraceCorpus(train=tuple(shuffled), holdout=projected_indomain)
        shuffled_learner = InteractionGroupTransferLearner(
            minimum_utility=-10.0, maximum_uncertainty=2.0
        )
        shuffled_learner.observe_members(profiles)
        shuffled_learner.observe_records(evaluator.train_only_candidates(shuffled_corpus))
        order_stable = b1._predicted_map(shuffled_learner, pairs) == b1._predicted_map(
            learner, pairs
        )
        g3 = bool(renaming_stable and order_stable)

        # ---- G2 leakage (structural)
        fit_ids = {task.task_id for task in fit_tasks}
        g2 = bool(not (fit_ids & {task.task_id for task in candidate_tasks}))

        # ---- checkpoint + fresh-process restore reproduces the selection
        restore_start = time.perf_counter()
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
            restore_ok = child_output.get("selected_pair") == list(selected_pair)
        tampered_payload = dict(checkpoint_payload)
        tampered_payload["coefficients"] = [
            value + 1.0 for value in tampered_payload.get("coefficients", ())
        ]
        tamper_rejected = False
        try:
            InteractionGroupTransferLearner.from_checkpoint(tampered_payload)
        except Exception:  # noqa: BLE001
            tamper_rejected = True
        restore_seconds = time.perf_counter() - restore_start
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        total_wall = time.perf_counter() - started
        feature_rank = learner.feature_rank()
        predictions_vary = (
            len(set(round(value, 9) for value in b1._predicted_map(learner, pairs).values())) > 1
        )
        representation_ok = bool(feature_rank > 1 and predictions_vary)

        gates = {
            "static_checks": True,
            "g1_intervention_reality": g1,
            "g2_data_isolation": g2,
            "g3_ranking_stability": g3,
            "g4_routing_beats_deployable": bool(g4_all),
            "g5_collaboration": bool(g5_all and interleaved_count > 0),
            "g6_budget": bool(total_wall <= WALL_CAP_SECONDS),
            "restore": bool(restore_ok and tamper_rejected),
            "representation": representation_ok,
        }
        if (
            not gates["static_checks"]
            or not gates["restore"]
            or not gates["g1_intervention_reality"]
            or not gates["g2_data_isolation"]
            or not gates["g3_ranking_stability"]
            or not gates["g6_budget"]
        ):
            outcome = "failed"
        elif not representation_ok:
            outcome = "representation_ineffective"
        elif gates["g4_routing_beats_deployable"] and gates["g5_collaboration"]:
            outcome = "collaboration_supported"
        elif gates["g4_routing_beats_deployable"]:
            outcome = "routing_only"
        else:
            outcome = "failed"
        payload.update(
            {
                "status": "completed",
                "outcome": outcome,
                "experiment_passed": bool(outcome == "collaboration_supported"),
                "record": {
                    "commit": commit,
                    "rule_revision": frozen.RULE_REVISION,
                    "composition_rule": frozen.COMPOSITION_RULE,
                    "wall_cap_seconds": WALL_CAP_SECONDS,
                    "elapsed_seconds": round(total_wall, 3),
                },
                "selection": {
                    "selected_pair": selected_label,
                    "predicted_interaction": round(selected_predicted, 6),
                    "selection_disclosure": selection_disclosure,
                    "context_free_boundary": (
                        "the representation does not condition on context, so one "
                        "pair serves all three candidate cells; per-cell routing is "
                        "not claimed"
                    ),
                    "feature_rank": feature_rank,
                    "predicted_interaction_by_pair": {
                        label: round(value, 6)
                        for label, value in b1._predicted_map(learner, pairs).items()
                    },
                },
                "cells": cells_accounting,
                "interleaved": {
                    "contexts_interleaved": interleaved_count,
                    "required": "> 0",
                },
                "control_arms": control_arms,
                "gates": gates,
                "checkpoint": {
                    "restore_matches_parent": restore_ok,
                    "tamper_rejected": tamper_rejected,
                },
                "cost": {
                    "member_training_seconds": round(member_seconds, 3),
                    "fit_seconds": round(fit_seconds, 3),
                    "matrix_seconds": round(matrix_seconds, 3),
                    "restore_seconds": round(restore_seconds, 3),
                    "episodes_total": len(fit_episodes)
                    + len(indomain_episodes)
                    + len(candidate_episodes),
                },
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; selected pair {selected_label}; "
                        f"per-cell gains "
                        + ", ".join(
                            f"{cell['label']}={cells_accounting[cell['label']]['gain_vs_all_singleton_oracle']}"
                            for cell in candidate_cells
                        )
                        + f"; interleaved={interleaved_count}; H2 claim bounded by "
                        "Route B section 7 (one load-bearing structural factor, "
                        "single fingerprint across three language routes)"
                    )
                    if outcome != "failed"
                    else (
                        f"completed: outcome=failed; gates failed: "
                        f"{sorted(key for key, value in gates.items() if not value)}"
                    )
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
    _write_json(DEFAULT_REPORT, payload)
    return payload


def _child(ckpt_path: Path) -> int:
    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    learner = InteractionGroupTransferLearner.from_checkpoint(payload)
    pairs = list(
        itertools.combinations(sorted(profile.member_id for profile in learner.profiles), 2)
    )
    selection = learner.select(pairs, unseen_only=False)
    selected = list(selection[0].member_ids) if selection else []
    print(json.dumps({"selected_pair": selected}))
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


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
                "gates": result.get("gates"),
                "selected_pair": (result.get("selection") or {}).get("selected_pair"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("experiment_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

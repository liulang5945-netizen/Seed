"""Route B (B2-v4) collaboration judgment gate: dictionary-scale G5 accounting.

Preregistration: plans/reference/M5_B2V4_COLLABORATION_JUDGMENT_PREREGISTRATION_FROZEN_20260916.md
(frozen).  No mechanism, threshold, or criterion changes:

  - the selection procedure frozen by B2-v3 is deterministically
    re-instantiated; a drift gate requires the selections to match the v3
    report's recorded ``v3_selected_pairs``;
  - the same candidate-face factorial is freshly executed (reports are never
    overwritten or cross-referenced for numbers);
  - G4 stays on the deployable-control comparison (success-rate scale);
  - G5 is judged per cell on the measurement-dictionary plus/minus-one
    outcome scale -- the scale ``REQUIRED_GAIN=1.65`` is defined on and the
    scale the D5 counterfactual measured ``+2.000`` on:
    ``policy_mean_gain_vs_all_singleton_oracle >= 1.65``, positive contexts
    ``>= REQUIRED_K``, lesion attribution, ``interleaved > 0``;
  - a reversed-priority diagnostic column on create__override only (not
    gated; the frozen member order is the claim's deployment semantics);
  - ``all_members_blocked`` (the stop reason m4 introduced) is audited;
  - control arms disclose both scales side by side.
"""

from __future__ import annotations

import argparse
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

import eval_taiji_b0_b1_representation_gate as b1  # noqa: E402
import eval_taiji_b0_b2_selection_gate as b2  # noqa: E402
import eval_taiji_b0_b2v2_route_gate as b2v2  # noqa: E402
import eval_taiji_b0_b2v3_family_gate as b2v3  # noqa: E402
import probe_taiji_b0_structure_space as ssp  # noqa: E402

from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupTransferLearner,
    build_member_evidence,
)
from taiji.interaction_groups import InteractionTraceCorpus  # noqa: E402

REPORT_FORMAT = "taiji-b0-b2v4-collaboration-gate-report-v1"
VERSION = 1
PREREGISTRATION = (
    "plans/reference/M5_B2V4_COLLABORATION_JUDGMENT_PREREGISTRATION_FROZEN_20260916.md"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_b2v4_collaboration_20260916.json"
V3_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_b2v3_family_20260916.json"

LEGACY_PREFIX = "b2v4old"
FAMILY_PREFIX = "b2v4fam"
CANDIDATE_PREFIX = "b2v4cand"
REVERSED_PREFIX = "b2v4rev"
CONTROL_RANDOM_SEED = 17
WALL_CAP_SECONDS = 60.0

STATIC_CHECK_SCOPE = (
    "scripts/training/eval_taiji_b0_b2v4_collaboration_gate.py",
    "scripts/training/eval_taiji_b0_b2v3_family_gate.py",
    "scripts/training/eval_taiji_b0_b2v2_route_gate.py",
    "scripts/training/eval_taiji_b0_b2_selection_gate.py",
    "scripts/training/eval_taiji_b0_b1_representation_gate.py",
)


def _stop_reason_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        reason = str(episode.get("stop_reason", "?"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "scope": (
            "B2-v4 collaboration judgment gate; mechanism (m4_failure_handoff, "
            "rule_revision=1), thresholds (REQUIRED_GAIN=1.65, REQUIRED_K=5) and "
            "the candidate face are unchanged - G5 is judged per cell on the "
            "measurement-dictionary plus/minus-one outcome scale per the frozen "
            "D5 estimand, the B2-v3 selection procedure is re-instantiated under "
            "a drift gate, and both scales are disclosed for every control arm"
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
        dictionary = handoff.load_dictionary()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        _grid, training_cells, candidate_cells = b1._cells()
        family_cells = [cell for cell in candidate_cells if cell["label"] != "create__override"]
        override_cell = next(
            cell for cell in candidate_cells if cell["label"] == "create__override"
        )

        member_start = time.perf_counter()
        embedder = frozen.DocumentEmbedder()
        members = frozen._train_members(embedder)
        member_seconds = time.perf_counter() - member_start
        member_ids = frozen.MEMBER_IDS
        pairs = list(itertools.combinations(member_ids, 2))

        # ---- selection re-instantiation (B2-v3 frozen procedure)
        legacy_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.FIT_CONTEXT_STEP,), LEGACY_PREFIX
        )
        legacy_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, legacy_tasks
        )
        indomain_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.INDOMAIN_HOLDOUT_STEP,), "b2v4ind"
        )
        indomain_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, indomain_tasks
        )
        family_tasks = b1._tasks_for_cells(
            frozen, family_cells, (b2v3.FAMILY_FIT_STEP,), FAMILY_PREFIX
        )
        family_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, family_tasks
        )
        projected_fit = frozen._project(legacy_episodes)
        projected_indomain = frozen._project(indomain_episodes)

        route_means, per_cell_deltas = b2v3._route_deltas(
            legacy_episodes, training_cells, legacy_tasks, LEGACY_PREFIX, member_ids
        )
        contributions = {
            member: statistics.mean(
                per_cell_deltas[(member, cell["label"])]["delta"] for cell in training_cells
            )
            for member in member_ids
        }
        legacy_rows = b2v3._interaction_rows(
            legacy_episodes,
            training_cells,
            legacy_tasks,
            LEGACY_PREFIX,
            member_ids,
            pairs,
            "legacy_8cell",
        )
        family_rows = b2v3._interaction_rows(
            family_episodes,
            family_cells,
            family_tasks,
            FAMILY_PREFIX,
            member_ids,
            pairs,
            "family_step6",
        )

        selector = b2v3.FamilySelector(contributions)
        config_for_cell: dict[str, str] = {}
        for cell in candidate_cells:
            config_id = f"predict_{cell['label']}"
            config_for_cell[cell["label"]] = config_id
            fit_rows = legacy_rows + [row for row in family_rows if row["cell"] != cell["label"]]
            selector.fit(config_id, fit_rows, "create")
        v3_selected = {
            cell["label"]: selector.select(config_for_cell[cell["label"]], pairs)
            for cell in candidate_cells
        }

        # ---- drift gate against the frozen v3 report
        v3_recorded = json.loads(V3_REPORT.read_text(encoding="utf-8"))["representation"][
            "v3_selected_pairs"
        ]
        selection_drift = {
            cell_label: {
                "reinstated": "+".join(pair),
                "v3_recorded": v3_recorded.get(cell_label),
            }
            for cell_label, pair in v3_selected.items()
            if "+".join(pair) != v3_recorded.get(cell_label)
        }
        drift_gate = bool(not selection_drift)

        # ---- v1 control arm (library learner, unchanged)
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
        v1_learner = InteractionGroupTransferLearner(minimum_utility=-10.0, maximum_uncertainty=2.0)
        v1_learner.observe_members(profiles)
        v1_learner.observe_records(records)
        v1_selection = v1_learner.select(pairs, unseen_only=False)
        v1_pair = (
            tuple(v1_selection[0].member_ids)
            if v1_selection is not None
            else max(
                pairs,
                key=lambda pair: b1._predicted_map(v1_learner, [pair]).get(
                    "+".join(pair), float("-inf")
                ),
            )
        )

        # ---- v2 control arm (route-conditional ridge on legacy rows)
        legacy_fit_tuples = [
            (tuple(row["pair"].split("+")), row["route"], row["interaction"]) for row in legacy_rows
        ]
        v2_selector = b2v2.RouteSelector(route_means, contributions)
        v2_selector.fit(legacy_fit_tuples)
        v2_pair = v2_selector.select(pairs, "create")

        # ---- candidate-surface factorial (fresh execution)
        candidate_tasks = b1._tasks_for_cells(
            frozen, candidate_cells, b1.CANDIDATE_STEPS, CANDIDATE_PREFIX
        )
        matrix_start = time.perf_counter()
        candidate_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, candidate_tasks
        )
        matrix_seconds = time.perf_counter() - matrix_start
        candidate_ids = sorted(task.task_id for task in candidate_tasks)
        outcomes = handoff.outcome_by_cell(candidate_episodes)

        def dictionary_gain(pair: tuple[str, str], context_ids: list[str]) -> float:
            table = handoff.table_from_outcomes(outcomes, context_ids, member_ids)
            return float(dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair))

        # ---- per-cell accounting: G4 (rate scale) + G5 (dictionary scale)
        cells_accounting: dict[str, Any] = {}
        g4_all = True
        g5_all = True
        interleaved_counts: dict[str, int] = {}
        for cell in candidate_cells:
            pair = v3_selected[cell["label"]]
            cell_contexts = sorted(
                task.task_id
                for task in candidate_tasks
                if task.task_id.startswith(f"{CANDIDATE_PREFIX}-{cell['label']}-")
            )
            breakdown = b2._context_breakdown(candidate_episodes, cell_contexts, member_ids, pair)
            cell_gain_rate = statistics.mean(
                item["gain_vs_all_singleton_oracle"] for item in breakdown.values()
            )
            pair_mean = statistics.mean(item["pair_success_rate"] for item in breakdown.values())
            singleton_gains = {
                member: statistics.mean(
                    b1._success_by_cell(candidate_episodes, context_id).get((member,), 0.0)
                    - b1._success_by_cell(candidate_episodes, context_id).get((), 0.0)
                    for context_id in cell_contexts
                )
                for member in member_ids
            }
            best_singleton_member = max(singleton_gains, key=lambda member: singleton_gains[member])
            control_mean = statistics.mean(
                item["singleton_success_rates"][best_singleton_member]
                for item in breakdown.values()
            )
            g4 = bool(pair_mean > control_mean)
            g4_all = g4_all and g4

            table = handoff.table_from_outcomes(outcomes, cell_contexts, member_ids)
            dict_gain = float(dictionary.policy_mean_gain_vs_all_singleton_oracle(table, pair))
            deployable_reference = statistics.mean(
                float(table.singleton(context_id, best_singleton_member))
                for context_id in cell_contexts
            )
            per_context_gain = {
                context_id: float(table.outcome(context_id, pair))
                - float(
                    dictionary.all_singleton_oracle(
                        singletons=[
                            table.singleton(context_id, member)
                            for member in table.singletons
                        ]
                    )
                )
                for context_id in cell_contexts
            }
            positive_contexts = sum(1 for value in per_context_gain.values() if value > 0)
            lesion_ok = bool(positive_contexts) and all(
                float(table.singleton(context_id, member)) < float(table.outcome(context_id, pair))
                for context_id in cell_contexts
                if per_context_gain[context_id] > 0
                for member in member_ids
            )
            classified = b2v3.precheck_classify(table, pair)
            interleaved = int(classified["contexts_interleaved"])
            interleaved_counts.setdefault("+".join(pair), 0)
            interleaved_counts["+".join(pair)] = max(
                interleaved_counts["+".join(pair)], interleaved
            )
            cell_g5 = bool(
                dict_gain >= b2.REQUIRED_GAIN
                and positive_contexts >= b2.REQUIRED_K
                and lesion_ok
                and interleaved > 0
            )
            g5_all = g5_all and cell_g5
            cells_accounting[cell["label"]] = {
                "selected_pair": "+".join(pair),
                "g4_rate_scale": {
                    "gain_vs_all_singleton_oracle": round(cell_gain_rate, 6),
                    "pair_mean_outcome": round(pair_mean, 6),
                    "best_singleton_member": best_singleton_member,
                    "best_singleton_mean_outcome": round(control_mean, 6),
                    "g4_beats_deployable_control": g4,
                },
                "g5_dictionary_scale": {
                    "mean_gain_vs_all_singleton_oracle": round(dict_gain, 6),
                    "required": b2.REQUIRED_GAIN,
                    "positive_gain_contexts": positive_contexts,
                    "k_criterion_met": bool(positive_contexts >= b2.REQUIRED_K),
                    "lesion_attribution_holds": lesion_ok,
                    "contexts_interleaved": interleaved,
                    "best_deployable_singleton_outcome": round(deployable_reference, 6),
                    "g5_collaboration_criteria_met": cell_g5,
                },
            }
        interleaved_min = min(interleaved_counts.values()) if interleaved_counts else 0

        # ---- control arms, both scales
        random_generator = random.Random(CONTROL_RANDOM_SEED)
        random_pair = tuple(sorted(random_generator.sample(list(member_ids), 2)))
        pair_gains, singleton_gains_all = b1._actual_gains(
            candidate_episodes, candidate_ids, member_ids
        )
        best_observed_label = max(pair_gains, key=lambda label: pair_gains[label])

        def pooled_dictionary_gain(pair: tuple[str, str]) -> float:
            return statistics.mean(
                dictionary_gain(pair, contexts)
                for cell in candidate_cells
                for contexts in [
                    sorted(
                        task.task_id
                        for task in candidate_tasks
                        if task.task_id.startswith(f"{CANDIDATE_PREFIX}-{cell['label']}-")
                    )
                ]
            )

        uniform_selection = len({"+".join(pair) for pair in v3_selected.values()}) == 1

        def arm_entry(pair: tuple[str, str]) -> dict[str, Any]:
            return {
                "pair": "+".join(pair),
                "mean_gain_rate_scale": round(pair_gains["+".join(pair)], 6),
                "mean_gain_dictionary_scale": round(pooled_dictionary_gain(pair), 6),
            }

        control_arms = {
            "v3_selected_pairs": {
                cell_label: "+".join(pair) for cell_label, pair in v3_selected.items()
            },
            "v3_selected_pooled": (
                arm_entry(tuple(v3_selected[candidate_cells[0]["label"]]))
                if uniform_selection
                else {
                    "pair": None,
                    "note": "selections differ per cell; per-cell accounting is primary",
                }
            ),
            "v2_control_selection": arm_entry(v2_pair),
            "v1_control_selection": arm_entry(v1_pair),
            "no_learning_fallback_ab": arm_entry((member_ids[0], member_ids[1])),
            "best_fixed_singleton": {
                "member": max(singleton_gains_all, key=lambda member: singleton_gains_all[member]),
                "mean_gain_rate_scale": 0.0,
                "mean_gain_dictionary_scale": round(
                    statistics.mean(
                        float(
                            dictionary.policy_mean_gain_vs_all_singleton_oracle(
                                handoff.table_from_outcomes(
                                    outcomes,
                                    sorted(
                                        task.task_id
                                        for task in candidate_tasks
                                        if task.task_id.startswith(
                                            f"{CANDIDATE_PREFIX}-{cell['label']}-"
                                        )
                                    ),
                                    member_ids,
                                ),
                                (max(singleton_gains_all, key=lambda m: singleton_gains_all[m]),),
                            )
                        )
                        for cell in candidate_cells
                    ),
                    6,
                ),
            },
            "best_observed_fixed_pair": {
                "pair": best_observed_label,
                "mean_gain_rate_scale": round(pair_gains[best_observed_label], 6),
                "mean_gain_dictionary_scale": round(
                    pooled_dictionary_gain(tuple(best_observed_label.split("+"))),
                    6,
                ),
                "note": "post-hoc diagnostic column; oracle-deployable=False",
            },
            "random_pair": arm_entry(random_pair),
        }

        # ---- reversed-priority diagnostic (create__override only, not gated)
        reversed_cells = tuple(
            tuple(reversed(cell)) if len(cell) == 2 else cell for cell in frozen.CELL_MEMBER_SETS
        )
        override_tasks = b1._tasks_for_cells(
            frozen, [override_cell], b1.CANDIDATE_STEPS, REVERSED_PREFIX
        )
        reversed_episodes = counterfactual.execute_surface(
            frozen,
            frozen._member_episode,
            members,
            embedder,
            override_tasks,
            cell_member_sets=reversed_cells,
        )
        reversed_outcomes = handoff.outcome_by_cell(reversed_episodes)
        reversed_contexts = sorted(task.task_id for task in override_tasks)
        reversed_table = handoff.table_from_outcomes(
            reversed_outcomes, reversed_contexts, member_ids
        )
        reversed_gains = {
            "+".join(pair): float(
                dictionary.policy_mean_gain_vs_all_singleton_oracle(reversed_table, pair)
            )
            for pair in pairs
        }
        best_reversed = max(reversed_gains, key=lambda label: reversed_gains[label])
        reversed_interleaved = int(
            b2v3.precheck_classify(reversed_table, tuple(best_reversed.split("+")))[
                "contexts_interleaved"
            ]
        )
        reversed_diagnostic = {
            "scope": "create__override only; diagnostic, not gated",
            "best_pair": best_reversed,
            "mean_gain_dictionary_scale": round(reversed_gains[best_reversed], 6),
            "contexts_interleaved_best_pair": reversed_interleaved,
            "pair_gains_dictionary_scale": {
                label: round(value, 6) for label, value in sorted(reversed_gains.items())
            },
        }

        # ---- stop-reason audit (all_members_blocked is new with m4)
        stop_reasons = {
            "candidate_face": _stop_reason_counts(candidate_episodes),
            "reversed_diagnostic": _stop_reason_counts(reversed_episodes),
        }

        # ---- G1 intervention reality
        flat_steps: dict[str, int] = {}
        for episode in candidate_episodes:
            key = "+".join(episode["active_members"]) or "baseline"
            flat_steps[key] = flat_steps.get(key, 0) + len(episode["steps"])
        g1 = bool(all(steps > 0 for name, steps in flat_steps.items() if name != "baseline"))

        # ---- G2 leakage
        fit_ids = {task.task_id for task in legacy_tasks} | {task.task_id for task in family_tasks}
        g2 = bool(not (fit_ids & {task.task_id for task in candidate_tasks}))

        # ---- G3 stability per config (renaming + order permutation)
        rotation = {
            name: member_ids[(index + 1) % len(member_ids)] for index, name in enumerate(member_ids)
        }
        rotated_contributions = {rotation[member]: value for member, value in contributions.items()}
        rotated_legacy = [
            {
                "pair": "+".join(tuple(rotation[name] for name in row["pair"].split("+"))),
                "route": row["route"],
                "interaction": row["interaction"],
            }
            for row in legacy_rows
        ]
        rotated_family = [
            {
                "pair": "+".join(tuple(rotation[name] for name in row["pair"].split("+"))),
                "route": row["route"],
                "interaction": row["interaction"],
                "cell": row["cell"],
            }
            for row in family_rows
        ]
        renaming_stable = True
        order_stable = True
        stability_detail: dict[str, Any] = {}
        for cell in candidate_cells:
            config_id = config_for_cell[cell["label"]]
            rotated_selector = b2v3.FamilySelector(rotated_contributions)
            rotated_fit_rows = rotated_legacy + [
                row for row in rotated_family if row["cell"] != cell["label"]
            ]
            rotated_selector.fit(config_id, rotated_fit_rows, "create")
            rotated_selected = rotated_selector.select(
                config_id, [tuple(rotation[name] for name in pair) for pair in pairs]
            )
            expected = tuple(rotation[name] for name in v3_selected[cell["label"]])
            renaming_stable = renaming_stable and bool(rotated_selected == expected)

            shuffled_legacy = list(legacy_rows)
            shuffled_family = [row for row in family_rows if row["cell"] != cell["label"]]
            random_generator.shuffle(shuffled_legacy)
            shuffled_selector = b2v3.FamilySelector(contributions)
            shuffled_selector.fit(config_id, shuffled_legacy + shuffled_family, "create")
            order_stable = order_stable and all(
                abs(shuffled_selector.predict(config_id, pair) - selector.predict(config_id, pair))
                <= 1e-9
                for pair in pairs
            )
            stability_detail[config_id] = {
                "renaming_stable": rotated_selected == expected,
                "rotated_selected": "+".join(rotated_selected),
                "expected_rotated_selected": "+".join(expected),
            }
        g3 = bool(renaming_stable and order_stable)

        # ---- checkpoint + fresh-process restore + tamper
        restore_start = time.perf_counter()
        checkpoint_payload = selector.checkpoint()
        ckpt_dir = Path(tempfile.mkdtemp(prefix="b2v4-"))
        ckpt_path = ckpt_dir / "family-selector.json"
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
            restore_ok = child_output.get("selected_pairs") == {
                config_id: list(v3_selected[cell_label])
                for cell_label, config_id in config_for_cell.items()
            }
        tampered_payload = dict(checkpoint_payload)
        first_config = next(iter(tampered_payload.get("configs", {})))
        tampered_payload["configs"][first_config]["coefficients"] = [
            value + 1.0 for value in tampered_payload["configs"][first_config]["coefficients"]
        ]
        tamper_rejected = False
        try:
            b2v3.FamilySelector.from_checkpoint(tampered_payload)
        except Exception:  # noqa: BLE001
            tamper_rejected = True
        restore_seconds = time.perf_counter() - restore_start
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        total_wall = time.perf_counter() - started
        gates = {
            "static_checks": True,
            "selection_drift_vs_v3": drift_gate,
            "g1_intervention_reality": g1,
            "g2_data_isolation": g2,
            "g3_ranking_stability": g3,
            "g4_routing_beats_deployable": bool(g4_all),
            "g5_collaboration": bool(g5_all and interleaved_min > 0),
            "g6_budget": bool(total_wall <= WALL_CAP_SECONDS),
            "restore": bool(restore_ok and tamper_rejected),
        }
        mechanical = (
            "static_checks",
            "selection_drift_vs_v3",
            "g1_intervention_reality",
            "g2_data_isolation",
            "g3_ranking_stability",
            "g6_budget",
            "restore",
        )
        if any(not gates[name] for name in mechanical):
            outcome = "failed"
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
                    "reinstated_pairs": {
                        cell_label: "+".join(pair) for cell_label, pair in v3_selected.items()
                    },
                    "v3_recorded_pairs": v3_recorded,
                    "drift": selection_drift,
                },
                "cells": cells_accounting,
                "interleaved": {
                    "contexts_interleaved_per_pair": interleaved_counts,
                    "required": "> 0",
                },
                "control_arms": control_arms,
                "reversed_priority_diagnostic": reversed_diagnostic,
                "stop_reasons": stop_reasons,
                "gates": gates,
                "stability_detail": stability_detail,
                "checkpoint": {
                    "restore_matches_parent": restore_ok,
                    "tamper_rejected": tamper_rejected,
                },
                "cost": {
                    "member_training_seconds": round(member_seconds, 3),
                    "matrix_seconds": round(matrix_seconds, 3),
                    "restore_seconds": round(restore_seconds, 3),
                    "episodes_total": len(legacy_episodes)
                    + len(indomain_episodes)
                    + len(family_episodes)
                    + len(candidate_episodes)
                    + len(reversed_episodes),
                },
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; per-cell dictionary-scale G5: "
                        + ", ".join(
                            f"{cell['label']}="
                            f"{cells_accounting[cell['label']]['g5_dictionary_scale']['mean_gain_vs_all_singleton_oracle']}"
                            f"(k={cells_accounting[cell['label']]['g5_dictionary_scale']['positive_gain_contexts']},"
                            f"lesion={cells_accounting[cell['label']]['g5_dictionary_scale']['lesion_attribution_holds']},"
                            f"interleaved={cells_accounting[cell['label']]['g5_dictionary_scale']['contexts_interleaved']})"
                            for cell in candidate_cells
                        )
                        + "; claim scope: frozen mechanism + frozen v3 selection "
                        "procedure + create candidate face; growth_admitted=false "
                        "and can_promote=false remain in force"
                    )
                    if outcome != "failed"
                    else (
                        "completed: outcome=failed; failed gates "
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
    selector = b2v3.FamilySelector.from_checkpoint(payload)
    member_ids = sorted(selector.contributions)
    pairs = list(itertools.combinations(member_ids, 2))
    selected_pairs = {
        config_id: list(selector.select(config_id, pairs)) for config_id in sorted(selector.configs)
    }
    print(json.dumps({"selected_pairs": selected_pairs}))
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
                "selected_pairs": (result.get("selection") or {}).get("reinstated_pairs"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("experiment_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

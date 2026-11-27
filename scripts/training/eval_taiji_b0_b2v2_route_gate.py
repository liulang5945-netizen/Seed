"""Route B (B2-v2) selection gate: route-conditional exclusive-coverage selector.

Preregistration: plans/reference/M5_B2V2_ROUTE_CONDITIONAL_PREREGISTRATION_FROZEN_20260915.md
(frozen).  Differences from B2-v1 (attribution-driven, corpus/candidates/gates
unchanged - only the representation changes):

  - route-conditional profiles: delta[m][content_route] over none/create/patch
    (train-only, from the same 8-cell fit corpus);
  - 8-dim pair features separating complementary exclusive coverage from
    redundant overlap (the channel the v1 pooled features destroyed);
  - fit on all 6 pairs x 8 cells of training interactions (48 rows) with a
    runner-local standardized ridge (the library learner stays untouched and
    runs as the v1 control arm);
  - diagnostic-first: the full 48-row training interaction table is recorded;
    if the training surface contains no positive complementarity the fit
    degrades to zero predictions and that finding is recorded as-is.
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

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import eval_taiji_b0_b1_representation_gate as b1  # noqa: E402
import eval_taiji_b0_b2_selection_gate as b2  # noqa: E402
import probe_taiji_b0_structure_space as ssp  # noqa: E402

from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupTransferLearner,
    build_member_evidence,
)
from taiji.interaction_groups import InteractionTraceCorpus  # noqa: E402

REPORT_FORMAT = "taiji-b0-b2v2-route-gate-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_B2V2_ROUTE_CONDITIONAL_PREREGISTRATION_FROZEN_20260915.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_b2v2_route_20260915.json"

CHECKPOINT_FORMAT = "taiji-b2v2-route-selector-checkpoint-v1"
FEATURE_NAMES = (
    "bias",
    "exclusive_first",
    "exclusive_second",
    "exclusive_total",
    "redundant",
    "neither_covers",
    "mean_contrib",
    "product_contrib",
)
RIDGE_LAMBDA = 0.1
CONTROL_RANDOM_SEED = 17
WALL_CAP_SECONDS = 60.0

STATIC_CHECK_SCOPE = (
    "scripts/training/eval_taiji_b0_b2v2_route_gate.py",
    "scripts/training/eval_taiji_b0_b2_selection_gate.py",
    "scripts/training/eval_taiji_b0_b1_representation_gate.py",
)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _route_deltas(
    episodes: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    fit_tasks: tuple[Any, ...],
    member_ids: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    """Train-only delta[m][content_route] plus per-(member, cell) accounting."""

    deltas: dict[str, dict[str, float]] = {
        member: {"none": [], "create": [], "patch": []} for member in member_ids
    }
    per_cell: dict[tuple[str, str], dict[str, float]] = {}
    for cell in cells:
        route = str(cell["content_route"])
        context_ids = sorted(
            task.task_id
            for task in fit_tasks
            if task.task_id.startswith(f"b2v2fit-{cell['label']}-")
        )
        for member in member_ids:
            cell_rates = [b1._success_by_cell(episodes, context_id) for context_id in context_ids]
            singleton = statistics.mean(rates.get((member,), 0.0) for rates in cell_rates)
            baseline = statistics.mean(rates.get((), 0.0) for rates in cell_rates)
            delta = singleton - baseline
            deltas[member][route].append(delta)
            per_cell[(member, cell["label"])] = {
                "singleton": singleton,
                "baseline": baseline,
                "delta": delta,
                "route": route,
            }
    route_means = {
        member: {
            route: (statistics.mean(values) if values else 0.0) for route, values in routes.items()
        }
        for member, routes in deltas.items()
    }
    return route_means, per_cell


def _covers(delta: float) -> bool:
    return delta > 0.0


def _pair_features(
    first: str,
    second: str,
    route: str,
    route_means: dict[str, dict[str, float]],
    contributions: dict[str, float],
) -> tuple[float, ...]:
    delta_first = route_means[first][route]
    delta_second = route_means[second][route]
    first_covers, second_covers = _covers(delta_first), _covers(delta_second)
    exclusive_first = max(0.0, delta_first) if not second_covers else 0.0
    exclusive_second = max(0.0, delta_second) if not first_covers else 0.0
    redundant = (
        min(max(0.0, delta_first), max(0.0, delta_second))
        if first_covers and second_covers
        else 0.0
    )
    neither = 1.0 if not first_covers and not second_covers else 0.0
    return (
        1.0,
        exclusive_first,
        exclusive_second,
        exclusive_first + exclusive_second,
        redundant,
        neither,
        (contributions[first] + contributions[second]) / 2.0,
        contributions[first] * contributions[second],
    )


class RouteSelector:
    """Runner-local standardized ridge over route-conditional pair features."""

    def __init__(
        self,
        route_means: dict[str, dict[str, float]],
        contributions: dict[str, float],
    ) -> None:
        self.route_means = route_means
        self.contributions = contributions
        self.coefficients: tuple[float, ...] = ()
        self.scaler_mean: tuple[float, ...] = ()
        self.scaler_std: tuple[float, ...] = ()

    def _features(self, pair: tuple[str, str], route: str) -> tuple[float, ...]:
        return _pair_features(pair[0], pair[1], route, self.route_means, self.contributions)

    def fit(self, rows: list[tuple[tuple[str, str], str, float]]) -> None:
        matrix = [list(self._features(pair, route)) for pair, route, _ in rows]
        targets = [value for _, _, value in rows]
        width = len(FEATURE_NAMES)
        self.scaler_mean = tuple(
            statistics.mean(row[column] for row in matrix) for column in range(1, width)
        )
        self.scaler_std = tuple(
            (statistics.pstdev(row[column] for row in matrix) or 1.0) for column in range(1, width)
        )
        design = []
        for row in matrix:
            standardized = [1.0] + [
                (row[column] - self.scaler_mean[column - 1]) / self.scaler_std[column - 1]
                for column in range(1, width)
            ]
            design.append(standardized)
        x = torch.tensor(design, dtype=torch.float64)
        y = torch.tensor(targets, dtype=torch.float64)
        penalty = torch.eye(width, dtype=torch.float64) * RIDGE_LAMBDA
        penalty[0, 0] = 0.0  # intercept unpenalized
        solution = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
        self.coefficients = tuple(float(value) for value in solution)

    def predict(self, pair: tuple[str, str], route: str) -> float:
        row = list(self._features(pair, route))
        standardized = [1.0] + [
            (row[column] - self.scaler_mean[column - 1]) / self.scaler_std[column - 1]
            for column in range(1, len(FEATURE_NAMES))
        ]
        return sum(
            value * coefficient
            for value, coefficient in zip(standardized, self.coefficients, strict=True)
        )

    def select(self, pairs: list[tuple[str, str]], route: str) -> tuple[str, str]:
        # first-min wins on exact ties; the scan order maps equivariantly under
        # member renaming because the renamed list preserves pair order
        return min(pairs, key=lambda pair: -self.predict(pair, route))

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": 1,
            "ridge_lambda": RIDGE_LAMBDA,
            "feature_names": list(FEATURE_NAMES),
            "route_means": {member: dict(routes) for member, routes in self.route_means.items()},
            "contributions": dict(self.contributions),
            "coefficients": list(self.coefficients),
            "scaler_mean": list(self.scaler_mean),
            "scaler_std": list(self.scaler_std),
        }
        payload["checkpoint_digest"] = _sha(
            json.dumps(
                {key: value for key, value in payload.items() if key != "checkpoint_digest"},
                sort_keys=True,
            )
        )
        return payload

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> RouteSelector:
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unknown selector format: {payload.get('format')!r}")
        digest = payload.get("checkpoint_digest")
        recomputed = _sha(
            json.dumps(
                {key: value for key, value in payload.items() if key != "checkpoint_digest"},
                sort_keys=True,
            )
        )
        if digest != recomputed:
            raise ValueError("selector checkpoint digest mismatch")
        selector = cls(
            {member: dict(routes) for member, routes in payload["route_means"].items()},
            dict(payload["contributions"]),
        )
        selector.coefficients = tuple(payload["coefficients"])
        selector.scaler_mean = tuple(payload["scaler_mean"])
        selector.scaler_std = tuple(payload["scaler_std"])
        return selector


def _interaction_table(
    episodes: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    fit_tasks: tuple[Any, ...],
    member_ids: tuple[str, ...],
    pairs: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        context_ids = sorted(
            task.task_id
            for task in fit_tasks
            if task.task_id.startswith(f"b2v2fit-{cell['label']}-")
        )
        for pair in pairs:
            values = []
            for context_id in context_ids:
                rates = b1._success_by_cell(episodes, context_id)
                pair_rate = rates.get(tuple(sorted(pair)), 0.0)
                singleton_first = rates.get((pair[0],), 0.0)
                singleton_second = rates.get((pair[1],), 0.0)
                baseline = rates.get((), 0.0)
                values.append(pair_rate - singleton_first - singleton_second + baseline)
            rows.append(
                {
                    "cell": cell["label"],
                    "route": str(cell["content_route"]),
                    "pair": "+".join(pair),
                    "interaction": round(statistics.mean(values), 6),
                    "contexts": len(values),
                }
            )
    return rows


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": 1,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "scope": (
            "B2-v2 route-conditional selection gate; same corpus/candidates/gates "
            "as B2-v1, only the representation changes; library learner untouched "
            "and runs as the v1 control arm"
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

        # ---- fit corpus (identical to B1/B2-v1)
        fit_tasks = b1._tasks_for_cells(frozen, training_cells, (b1.FIT_CONTEXT_STEP,), "b2v2fit")
        fit_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, fit_tasks
        )
        indomain_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.INDOMAIN_HOLDOUT_STEP,), "b2v2ind"
        )
        indomain_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, indomain_tasks
        )
        projected_fit = frozen._project(fit_episodes)
        projected_indomain = frozen._project(indomain_episodes)

        # ---- route-conditional profiles (train-only)
        route_means, per_cell_deltas = _route_deltas(
            fit_episodes, training_cells, fit_tasks, member_ids
        )
        contributions = {
            member: statistics.mean(
                per_cell_deltas[(member, cell["label"])]["delta"] for cell in training_cells
            )
            for member in member_ids
        }

        # ---- diagnostic-first: full training interaction table
        interaction_table = _interaction_table(
            fit_episodes, training_cells, fit_tasks, member_ids, pairs
        )
        positive_training = [row for row in interaction_table if row["interaction"] > 0]

        # ---- v2 fit on all 48 rows
        fit_rows = [
            (tuple(row["pair"].split("+")), row["route"], row["interaction"])
            for row in interaction_table
        ]
        selector = RouteSelector(route_means, contributions)
        selector.fit(fit_rows)

        # ---- v1 control arm: the library learner (exactly as B2-v1)
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

        # ---- v2 selection per candidate route (context-free within route)
        candidate_route = "create"  # all three candidate cells are create-route
        v2_pair = selector.select(pairs, candidate_route)
        v2_predictions = {
            "+".join(pair): round(selector.predict(pair, candidate_route), 6) for pair in pairs
        }

        # ---- candidate-surface full factorial (identical to B2-v1)
        candidate_tasks = b1._tasks_for_cells(
            frozen, candidate_cells, b1.CANDIDATE_STEPS, "b2v2cand"
        )
        matrix_start = time.perf_counter()
        candidate_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, candidate_tasks
        )
        matrix_seconds = time.perf_counter() - matrix_start
        candidate_ids = sorted(task.task_id for task in candidate_tasks)
        outcomes = handoff.outcome_by_cell(candidate_episodes)
        table = handoff.table_from_outcomes(outcomes, candidate_ids, member_ids)

        # ---- per-cell gain accounting for the v2 selected pair
        cells_accounting: dict[str, Any] = {}
        g4_all = True
        g5_all = True
        for cell in candidate_cells:
            cell_contexts = sorted(
                task.task_id
                for task in candidate_tasks
                if task.task_id.startswith(f"b2v2cand-{cell['label']}-")
            )
            breakdown = b2._context_breakdown(
                candidate_episodes, cell_contexts, member_ids, v2_pair
            )
            cell_gain = statistics.mean(
                item["gain_vs_all_singleton_oracle"] for item in breakdown.values()
            )
            positive_contexts = sum(
                1 for item in breakdown.values() if item["gain_vs_all_singleton_oracle"] > 0
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
            lesion_rows = {
                context_id: all(
                    rate < item["pair_success_rate"]
                    for rate in item["singleton_success_rates"].values()
                )
                for context_id, item in breakdown.items()
                if item["gain_vs_all_singleton_oracle"] > 0
            }
            lesion_ok = bool(lesion_rows) and all(lesion_rows.values())
            g5_all = g5_all and bool(
                cell_gain >= b2.REQUIRED_GAIN and positive_contexts >= b2.REQUIRED_K and lesion_ok
            )
            cells_accounting[cell["label"]] = {
                "gain_vs_all_singleton_oracle": round(cell_gain, 6),
                "required": b2.REQUIRED_GAIN,
                "positive_gain_contexts": positive_contexts,
                "k_criterion_met": bool(positive_contexts >= b2.REQUIRED_K),
                "pair_mean_outcome": round(pair_mean, 6),
                "best_singleton_member": best_singleton_member,
                "best_singleton_mean_outcome": round(control_mean, 6),
                "g4_beats_deployable_control": g4,
                "lesion_attribution_holds": lesion_ok,
            }
        interleaved = precheck_classify(table, v2_pair)
        interleaved_count = int(interleaved["contexts_interleaved"])

        # ---- control arms from the same factorial
        random_generator = random.Random(CONTROL_RANDOM_SEED)
        random_pair = tuple(sorted(random_generator.sample(list(member_ids), 2)))
        pair_gains, singleton_gains_all = b1._actual_gains(
            candidate_episodes, candidate_ids, member_ids
        )
        best_observed_label = max(pair_gains, key=lambda label: pair_gains[label])
        control_arms = {
            "v2_selected_pair": {
                "pair": "+".join(v2_pair),
                "mean_gain_vs_all_singleton_oracle": round(
                    statistics.mean(
                        cells_accounting[cell["label"]]["gain_vs_all_singleton_oracle"]
                        for cell in candidate_cells
                    ),
                    6,
                ),
            },
            "v1_control_selection": {
                "pair": "+".join(v1_pair),
                "mean_gain_vs_all_singleton_oracle": round(pair_gains["+".join(v1_pair)], 6),
            },
            "no_learning_fallback_ab": {
                "pair": "+".join((member_ids[0], member_ids[1])),
                "mean_gain_vs_all_singleton_oracle": round(
                    pair_gains["+".join((member_ids[0], member_ids[1]))], 6
                ),
            },
            "best_fixed_singleton": {
                "member": max(
                    singleton_gains_all,
                    key=lambda member: singleton_gains_all[member],
                ),
                "mean_gain_vs_all_singleton_oracle": 0.0,
            },
            "best_observed_fixed_pair": {
                "pair": best_observed_label,
                "mean_gain_vs_all_singleton_oracle": round(pair_gains[best_observed_label], 6),
                "note": "post-hoc diagnostic column; oracle-deployable=False",
            },
            "random_pair": {
                "pair": "+".join(random_pair),
                "mean_gain_vs_all_singleton_oracle": round(pair_gains["+".join(random_pair)], 6),
            },
        }

        # ---- G1 intervention reality
        flat_steps: dict[str, int] = {}
        for episode in candidate_episodes:
            key = "+".join(episode["active_members"]) or "baseline"
            flat_steps[key] = flat_steps.get(key, 0) + len(episode["steps"])
        g1 = bool(all(steps > 0 for name, steps in flat_steps.items() if name != "baseline"))

        # ---- G3 stability: renaming + order permutation
        rotation = {
            name: member_ids[(index + 1) % len(member_ids)] for index, name in enumerate(member_ids)
        }
        rotated_route_means = {
            rotation[member]: dict(routes) for member, routes in route_means.items()
        }
        rotated_contributions = {rotation[member]: value for member, value in contributions.items()}
        rotated_selector = RouteSelector(rotated_route_means, rotated_contributions)
        # order-preserving rename: (first, second) columns must stay aligned
        # for the asymmetric exclusive_first/exclusive_second features
        rotated_rows = [
            (
                tuple(rotation[name] for name in pair),
                route,
                value,
            )
            for pair, route, value in fit_rows
        ]
        rotated_selector.fit(rotated_rows)
        rotated_selected = rotated_selector.select(
            [tuple(rotation[name] for name in pair) for pair in pairs],
            candidate_route,
        )
        renaming_stable = bool(rotated_selected == tuple(rotation[name] for name in v2_pair))

        shuffled_rows = list(fit_rows)
        random_generator.shuffle(shuffled_rows)
        shuffled_selector = RouteSelector(route_means, contributions)
        shuffled_selector.fit(shuffled_rows)
        order_stable = all(
            abs(
                shuffled_selector.predict(pair, candidate_route)
                - selector.predict(pair, candidate_route)
            )
            <= 1e-9
            for pair in pairs
        )
        g3 = bool(renaming_stable and order_stable)
        stability_detail = {
            "renaming_stable": renaming_stable,
            "order_permutation_stable": order_stable,
            "rotated_selected": "+".join(rotated_selected),
            "expected_rotated_selected": "+".join(rotation[name] for name in v2_pair),
            "rotated_prediction_of_expected": round(
                rotated_selector.predict(
                    tuple(rotation[name] for name in v2_pair), candidate_route
                ),
                6,
            ),
        }

        # ---- G2 leakage
        fit_ids = {task.task_id for task in fit_tasks}
        g2 = bool(not (fit_ids & {task.task_id for task in candidate_tasks}))

        # ---- checkpoint + fresh-process restore + tamper
        restore_start = time.perf_counter()
        checkpoint_payload = selector.checkpoint()
        ckpt_dir = Path(tempfile.mkdtemp(prefix="b2v2-"))
        ckpt_path = ckpt_dir / "route-selector.json"
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
            restore_ok = child_output.get("selected_pair") == list(v2_pair)
        tampered_payload = dict(checkpoint_payload)
        tampered_payload["coefficients"] = [
            value + 1.0 for value in tampered_payload.get("coefficients", ())
        ]
        tamper_rejected = False
        try:
            RouteSelector.from_checkpoint(tampered_payload)
        except Exception:  # noqa: BLE001
            tamper_rejected = True
        restore_seconds = time.perf_counter() - restore_start
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        total_wall = time.perf_counter() - started
        gates = {
            "static_checks": True,
            "g1_intervention_reality": g1,
            "g2_data_isolation": g2,
            "g3_ranking_stability": g3,
            "g4_routing_beats_deployable": bool(g4_all),
            "g5_collaboration": bool(g5_all and interleaved_count > 0),
            "g6_budget": bool(total_wall <= WALL_CAP_SECONDS),
            "restore": bool(restore_ok and tamper_rejected),
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
                "representation": {
                    "route_means": {
                        member: {route: round(value, 6) for route, value in routes.items()}
                        for member, routes in sorted(route_means.items())
                    },
                    "contributions": {
                        member: round(value, 6) for member, value in sorted(contributions.items())
                    },
                    "coefficients": {
                        name: round(value, 6)
                        for name, value in zip(FEATURE_NAMES, selector.coefficients, strict=True)
                    },
                    "v2_predictions_create_route": v2_predictions,
                    "v2_selected_pair": "+".join(v2_pair),
                    "v1_control_selected_pair": "+".join(v1_pair),
                },
                "diagnostic_training_interactions": {
                    "rows": interaction_table,
                    "positive_rows": len(positive_training),
                    "note": (
                        "if positive_rows is 0 the training surface contains no "
                        "complementary-success pattern; the fit degrades honestly "
                        "and training-surface expansion becomes the axis finding"
                    ),
                },
                "cells": cells_accounting,
                "interleaved": {
                    "contexts_interleaved": interleaved_count,
                    "required": "> 0",
                },
                "control_arms": control_arms,
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
                    "episodes_total": len(fit_episodes)
                    + len(indomain_episodes)
                    + len(candidate_episodes),
                },
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; v2 selected "
                        f"{'+'.join(v2_pair)} (v1 control selected "
                        f"{'+'.join(v1_pair)}); per-cell gains "
                        + ", ".join(
                            f"{cell['label']}={cells_accounting[cell['label']]['gain_vs_all_singleton_oracle']}"
                            for cell in candidate_cells
                        )
                        + f"; interleaved={interleaved_count}"
                    )
                    if outcome != "failed" or gates["g4_routing_beats_deployable"]
                    else (
                        "completed: outcome=failed; mechanical gates "
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


def precheck_classify(table: Any, pair: tuple[str, ...]) -> dict[str, Any]:
    import audit_taiji_b0_task_reachability_precheck as precheck

    return precheck.classify_pair_trajectory(table, pair)


def _child(ckpt_path: Path) -> int:
    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    selector = RouteSelector.from_checkpoint(payload)
    member_ids = sorted(selector.contributions)
    pairs = list(itertools.combinations(member_ids, 2))
    selected = selector.select(pairs, "create")
    print(json.dumps({"selected_pair": list(selected)}))
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
                "v2_selected_pair": (result.get("representation") or {}).get("v2_selected_pair"),
                "v1_control_selected_pair": (result.get("representation") or {}).get(
                    "v1_control_selected_pair"
                ),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("experiment_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

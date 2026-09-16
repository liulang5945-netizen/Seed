"""Route B (B2-v3) selection gate: family-transfer pair-record selector.

Preregistration: plans/reference/M5_B2V3_FAMILY_TRANSFER_PREREGISTRATION_FROZEN_20260916.md
(frozen).  Differences from B2-v2 (mechanism/checkpoints/candidate face/gates
unchanged - only the selection corpus and features change):

  - the fit corpus grows honestly inside the frozen contract: the 48 legacy
    8-cell rows are kept verbatim and 12 family rows (create__observation and
    create__mismatch at step 6, indices disjoint from the candidate steps 0-5)
    add the positive complementarity the training surface lacked;
  - leave-one-shape-out: the prediction for each candidate cell excludes that
    cell's own family rows, so every selection transfers across language
    routes inside the create family instead of looking itself up;
  - 4-dim pair-level family-profile features [1, fam[p][C], mean_contrib,
    product_contrib] replace the singleton-based v2 set whose mean_contrib
    channel misleads the create-family ranking;
  - the library learner (v1 arm) and the v2 route-conditional ridge (v2 arm)
    run unchanged on the old corpus so the v1/v2/v3 delta is attributable;
  - claim boundary: in-family (create-route) transfer to held-out contexts;
    cross-family routing and collaboration (G5) are out of claim.
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
import eval_taiji_b0_b2v2_route_gate as b2v2  # noqa: E402
import probe_taiji_b0_structure_space as ssp  # noqa: E402

from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupTransferLearner,
    build_member_evidence,
)
from taiji.interaction_groups import InteractionTraceCorpus  # noqa: E402

REPORT_FORMAT = "taiji-b0-b2v3-family-gate-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_B2V3_FAMILY_TRANSFER_PREREGISTRATION_FROZEN_20260916.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_b2v3_family_20260916.json"

CHECKPOINT_FORMAT = "taiji-b2v3-family-selector-checkpoint-v1"
FEATURE_NAMES = ("bias", "fam_route", "mean_contrib", "product_contrib")
RIDGE_LAMBDA = 0.1
FAMILY_FIT_STEP = 6
LEGACY_PREFIX = "b2v3old"
FAMILY_PREFIX = "b2v3fam"
CANDIDATE_PREFIX = "b2v3cand"
CONTROL_RANDOM_SEED = 17
WALL_CAP_SECONDS = 60.0

STATIC_CHECK_SCOPE = (
    "scripts/training/eval_taiji_b0_b2v3_family_gate.py",
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
    prefix: str,
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
            if task.task_id.startswith(f"{prefix}-{cell['label']}-")
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


def _interaction_rows(
    episodes: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    fit_tasks: tuple[Any, ...],
    prefix: str,
    member_ids: tuple[str, ...],
    pairs: list[tuple[str, str]],
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        context_ids = sorted(
            task.task_id
            for task in fit_tasks
            if task.task_id.startswith(f"{prefix}-{cell['label']}-")
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
                    "language_route": str(cell["language_route"]),
                    "pair": "+".join(pair),
                    "interaction": round(statistics.mean(values), 6),
                    "contexts": len(values),
                    "source": source,
                }
            )
    return rows


class FamilySelector:
    """Runner-local standardized ridge over pair-level family-profile features.

    One fit per leave-one-shape-out config; ``fam[p]`` is the pair's mean
    interaction over that config's candidate-route rows.
    """

    def __init__(self, contributions: dict[str, float]) -> None:
        self.contributions = contributions
        self.configs: dict[str, dict[str, Any]] = {}

    def _features(self, fam: float, pair: tuple[str, str]) -> tuple[float, ...]:
        return (
            1.0,
            fam,
            (self.contributions[pair[0]] + self.contributions[pair[1]]) / 2.0,
            self.contributions[pair[0]] * self.contributions[pair[1]],
        )

    def fit(self, config_id: str, rows: list[dict[str, Any]], route: str) -> None:
        per_pair: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if row["route"] == route:
                per_pair.setdefault(tuple(row["pair"].split("+")), []).append(
                    float(row["interaction"])
                )
        fam_table = {pair: statistics.mean(values) for pair, values in per_pair.items()}
        matrix = [
            list(
                self._features(
                    fam_table.get(tuple(row["pair"].split("+")), 0.0), tuple(row["pair"].split("+"))
                )
            )
            for row in rows
        ]
        targets = [float(row["interaction"]) for row in rows]
        width = len(FEATURE_NAMES)
        scaler_mean = tuple(
            statistics.mean(row[column] for row in matrix) for column in range(1, width)
        )
        scaler_std = tuple(
            (statistics.pstdev(row[column] for row in matrix) or 1.0) for column in range(1, width)
        )
        design = []
        for row in matrix:
            standardized = [1.0] + [
                (row[column] - scaler_mean[column - 1]) / scaler_std[column - 1]
                for column in range(1, width)
            ]
            design.append(standardized)
        x = torch.tensor(design, dtype=torch.float64)
        y = torch.tensor(targets, dtype=torch.float64)
        penalty = torch.eye(width, dtype=torch.float64) * RIDGE_LAMBDA
        penalty[0, 0] = 0.0  # intercept unpenalized
        solution = torch.linalg.solve(x.T @ x + penalty, x.T @ y)
        self.configs[config_id] = {
            "fam": fam_table,
            "coefficients": tuple(float(value) for value in solution),
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
        }

    def predict(self, config_id: str, pair: tuple[str, str]) -> float:
        config = self.configs[config_id]
        fam = config["fam"].get(pair, 0.0)
        row = list(self._features(fam, pair))
        standardized = [1.0] + [
            (row[column] - config["scaler_mean"][column - 1]) / config["scaler_std"][column - 1]
            for column in range(1, len(FEATURE_NAMES))
        ]
        return sum(
            value * coefficient
            for value, coefficient in zip(standardized, config["coefficients"], strict=True)
        )

    def select(self, config_id: str, pairs: list[tuple[str, str]]) -> tuple[str, str]:
        # first-min wins on exact ties; the scan order maps equivariantly under
        # member renaming because the renamed list preserves pair order
        return min(pairs, key=lambda pair: -self.predict(config_id, pair))

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": 1,
            "ridge_lambda": RIDGE_LAMBDA,
            "feature_names": list(FEATURE_NAMES),
            "contributions": dict(self.contributions),
            "configs": {
                config_id: {
                    "fam": {"+".join(pair): value for pair, value in config["fam"].items()},
                    "coefficients": list(config["coefficients"]),
                    "scaler_mean": list(config["scaler_mean"]),
                    "scaler_std": list(config["scaler_std"]),
                }
                for config_id, config in self.configs.items()
            },
        }
        payload["checkpoint_digest"] = _sha(
            json.dumps(
                {key: value for key, value in payload.items() if key != "checkpoint_digest"},
                sort_keys=True,
            )
        )
        return payload

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> FamilySelector:
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
        selector = cls(dict(payload["contributions"]))
        for config_id, config in payload["configs"].items():
            selector.configs[config_id] = {
                "fam": {tuple(pair.split("+")): value for pair, value in config["fam"].items()},
                "coefficients": tuple(config["coefficients"]),
                "scaler_mean": tuple(config["scaler_mean"]),
                "scaler_std": tuple(config["scaler_std"]),
            }
        return selector


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "scope": (
            "B2-v3 family-transfer selection gate; mechanism/checkpoints/candidate "
            "face/gates unchanged from B2-v1/v2 - only the selection corpus (legacy "
            "48 rows + create-family step-6 rows, leave-one-shape-out) and the "
            "pair-level family-profile features change; v1 library learner and v2 "
            "route-conditional ridge run unchanged as control arms"
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
        family_cells = [cell for cell in candidate_cells if cell["label"] != "create__override"]
        if [cell["label"] for cell in family_cells] != [
            "create__observation",
            "create__mismatch",
        ]:
            raise SystemExit("family fit cells do not match the frozen preregistration")

        member_start = time.perf_counter()
        embedder = frozen.DocumentEmbedder()
        members = frozen._train_members(embedder)
        member_seconds = time.perf_counter() - member_start
        member_ids = frozen.MEMBER_IDS
        pairs = list(itertools.combinations(member_ids, 2))

        # ---- legacy fit corpus (identical to B1/B2-v1/v2)
        legacy_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.FIT_CONTEXT_STEP,), LEGACY_PREFIX
        )
        legacy_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, legacy_tasks
        )
        indomain_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.INDOMAIN_HOLDOUT_STEP,), "b2v3ind"
        )
        indomain_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, indomain_tasks
        )
        projected_fit = frozen._project(legacy_episodes)
        projected_indomain = frozen._project(indomain_episodes)

        # ---- family fit rows: two create shapes at step 6 (indices disjoint
        # from the candidate steps 0-5)
        family_tasks = b1._tasks_for_cells(frozen, family_cells, (FAMILY_FIT_STEP,), FAMILY_PREFIX)
        family_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, family_tasks
        )

        # ---- singleton profiles (v2 definition: legacy 8-cell corpus only)
        route_means, per_cell_deltas = _route_deltas(
            legacy_episodes, training_cells, legacy_tasks, LEGACY_PREFIX, member_ids
        )
        contributions = {
            member: statistics.mean(
                per_cell_deltas[(member, cell["label"])]["delta"] for cell in training_cells
            )
            for member in member_ids
        }

        # ---- diagnostic-first: legacy 48 rows + 12 family rows
        legacy_rows = _interaction_rows(
            legacy_episodes,
            training_cells,
            legacy_tasks,
            LEGACY_PREFIX,
            member_ids,
            pairs,
            "legacy_8cell",
        )
        family_rows = _interaction_rows(
            family_episodes,
            family_cells,
            family_tasks,
            FAMILY_PREFIX,
            member_ids,
            pairs,
            "family_step6",
        )
        all_rows = legacy_rows + family_rows
        positive_family = [row for row in family_rows if row["interaction"] > 0]
        ac_family_ok = all(
            row["interaction"] > 0 for row in family_rows if row["pair"] == "member-a+member-c"
        )

        # ---- v3 leave-one-shape-out fits (one per candidate cell)
        selector = FamilySelector(contributions)
        config_for_cell: dict[str, str] = {}
        for cell in candidate_cells:
            config_id = f"predict_{cell['label']}"
            config_for_cell[cell["label"]] = config_id
            fit_rows = legacy_rows + [row for row in family_rows if row["cell"] != cell["label"]]
            selector.fit(config_id, fit_rows, "create")

        # ---- v1 control arm: the library learner (exactly as B2-v1/v2)
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

        # ---- v2 control arm: route-conditional ridge on the legacy 48 rows
        legacy_fit_tuples = [
            (tuple(row["pair"].split("+")), row["route"], row["interaction"]) for row in legacy_rows
        ]
        v2_selector = b2v2.RouteSelector(route_means, contributions)
        v2_selector.fit(legacy_fit_tuples)
        v2_pair = v2_selector.select(pairs, "create")

        # ---- v3 selection per candidate cell
        v3_selected: dict[str, tuple[str, str]] = {
            cell["label"]: selector.select(config_for_cell[cell["label"]], pairs)
            for cell in candidate_cells
        }
        v3_predictions = {
            config_id: {
                "+".join(pair): round(selector.predict(config_id, pair), 6) for pair in pairs
            }
            for config_id in config_for_cell.values()
        }
        distinct_selections = set(v3_selected.values())

        # ---- candidate-surface full factorial (identical to B2-v1/v2)
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
        table = handoff.table_from_outcomes(outcomes, candidate_ids, member_ids)

        # ---- per-cell gain accounting for each cell's own selected pair
        cells_accounting: dict[str, Any] = {}
        g4_all = True
        g5_all = True
        for cell in candidate_cells:
            v3_pair = v3_selected[cell["label"]]
            cell_contexts = sorted(
                task.task_id
                for task in candidate_tasks
                if task.task_id.startswith(f"{CANDIDATE_PREFIX}-{cell['label']}-")
            )
            breakdown = b2._context_breakdown(
                candidate_episodes, cell_contexts, member_ids, v3_pair
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
                "selected_pair": "+".join(v3_pair),
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
        interleaved_counts: dict[str, int] = {}
        for selected in sorted(distinct_selections):
            classified = precheck_classify(table, selected)
            interleaved_counts["+".join(selected)] = int(classified["contexts_interleaved"])
        interleaved_min = min(interleaved_counts.values()) if interleaved_counts else 0

        # ---- control arms from the same factorial
        random_generator = random.Random(CONTROL_RANDOM_SEED)
        random_pair = tuple(sorted(random_generator.sample(list(member_ids), 2)))
        pair_gains, singleton_gains_all = b1._actual_gains(
            candidate_episodes, candidate_ids, member_ids
        )
        best_observed_label = max(pair_gains, key=lambda label: pair_gains[label])
        pooled_v3_label = (
            "+".join(next(iter(distinct_selections))) if len(distinct_selections) == 1 else None
        )
        control_arms = {
            "v3_selected_pairs": {
                cell_label: "+".join(pair) for cell_label, pair in v3_selected.items()
            },
            "v3_selected_pooled": {
                "pair": pooled_v3_label,
                "mean_gain_vs_all_singleton_oracle": (
                    round(pair_gains[pooled_v3_label], 6) if pooled_v3_label else None
                ),
                "note": (
                    "pooled column defined only when all three cells select the "
                    "same pair; per-cell gains are the primary accounting"
                ),
            },
            "v2_control_selection": {
                "pair": "+".join(v2_pair),
                "mean_gain_vs_all_singleton_oracle": round(pair_gains["+".join(v2_pair)], 6),
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

        # ---- G2 leakage: legacy + family fit tasks disjoint from candidates
        fit_ids = {task.task_id for task in legacy_tasks} | {task.task_id for task in family_tasks}
        g2 = bool(not (fit_ids & {task.task_id for task in candidate_tasks}))

        # ---- G3 stability per config: renaming + order permutation
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
            rotated_selector = FamilySelector(rotated_contributions)
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
            shuffled_selector = FamilySelector(contributions)
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
        ckpt_dir = Path(tempfile.mkdtemp(prefix="b2v3-"))
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
            FamilySelector.from_checkpoint(tampered_payload)
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
            "g5_collaboration": bool(g5_all and interleaved_min > 0),
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
                    "contributions": {
                        member: round(value, 6) for member, value in sorted(contributions.items())
                    },
                    "family_profiles": {
                        config_id: {
                            "+".join(pair): round(value, 6)
                            for pair, value in sorted(selector.configs[config_id]["fam"].items())
                        }
                        for config_id in sorted(selector.configs)
                    },
                    "coefficients": {
                        config_id: {
                            name: round(value, 6)
                            for name, value in zip(
                                FEATURE_NAMES,
                                selector.configs[config_id]["coefficients"],
                                strict=True,
                            )
                        }
                        for config_id in sorted(selector.configs)
                    },
                    "v3_predictions": v3_predictions,
                    "v3_selected_pairs": {
                        cell_label: "+".join(pair) for cell_label, pair in v3_selected.items()
                    },
                    "v2_control_selected_pair": "+".join(v2_pair),
                    "v1_control_selected_pair": "+".join(v1_pair),
                },
                "diagnostic_training_interactions": {
                    "rows": all_rows,
                    "legacy_rows": len(legacy_rows),
                    "family_rows": len(family_rows),
                    "positive_family_rows": len(positive_family),
                    "ac_family_rows_positive": bool(ac_family_ok),
                    "note": (
                        "family rows are the create__observation/create__mismatch "
                        "step-6 pair records; if ac_family_rows_positive is false "
                        "the family-transfer assumption is falsified and the "
                        "diagnostic branch records it as-is"
                    ),
                },
                "cells": cells_accounting,
                "interleaved": {
                    "contexts_interleaved_per_pair": interleaved_counts,
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
                    "episodes_total": len(legacy_episodes)
                    + len(indomain_episodes)
                    + len(family_episodes)
                    + len(candidate_episodes),
                },
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; v3 selected "
                        f"{ {label: '+'.join(pair) for label, pair in v3_selected.items()} } "
                        f"(v2 control selected {'+'.join(v2_pair)}, v1 control selected "
                        f"{'+'.join(v1_pair)}); per-cell gains "
                        + ", ".join(
                            f"{cell['label']}={cells_accounting[cell['label']]['gain_vs_all_singleton_oracle']}"
                            for cell in candidate_cells
                        )
                        + f"; interleaved={interleaved_counts}; claim boundary: "
                        "in-family (create-route) transfer to held-out contexts; "
                        "cross-family routing and collaboration out of claim"
                    )
                    if outcome != "failed"
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
    selector = FamilySelector.from_checkpoint(payload)
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
                "v3_selected_pairs": (result.get("representation") or {}).get("v3_selected_pairs"),
                "v2_control_selected_pair": (result.get("representation") or {}).get(
                    "v2_control_selected_pair"
                ),
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

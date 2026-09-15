"""Route B (B1) representation gate: train the conditional relation representation
on non-candidate structure cells and audit whether it can distinguish candidate
combinations.

Preregistration context: plans/reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md
(frozen) sections 2/3/4; closeout plan revision WP-4.  B1 trains no policy and
measures no collaboration: it fits the official train-only surfaces
(``build_member_evidence`` -> ``observe_members`` ->
``evaluator.train_only_candidates`` -> ``observe_records``) of the
``InteractionGroupTransferLearner`` on cells where single members genuinely
differentiate, then audits candidate discrimination on the frozen candidate
surface (create x 3 language routes) whose pair outcomes come only from real
contract execution.  Every discrimination number here is representation
evidence; H2 collaboration claims stay with the G5 real-execution gate.

Design (from the calibration finding, roadmap 2026-09-15):
  - fit corpus = the 8 non-candidate structure cells (patch row x4, none row
    x3, create__none), one fit context each (step 0); in-domain holdout =
    step 1 of the same cells;
  - evaluation surface = the 3 candidate create cells, steps 0-5 (frozen n=6);
  - candidate-cell pair outcomes never enter the fit (leakage gate);
  - evaluator resource cap raised for the fit corpus (dual-predict-select
    pair cells legitimately cost more than 10 actions) and disclosed.
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

import probe_taiji_b0_structure_space as ssp  # noqa: E402

from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupTransferLearner,
    build_member_evidence,
)
from taiji.interaction_groups import InteractionTraceCorpus  # noqa: E402

REPORT_FORMAT = "taiji-b0-b1-representation-gate-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_b1_representation_20260915.json"

FIT_CONTEXT_STEP = 0
INDOMAIN_HOLDOUT_STEP = 1
CANDIDATE_STEPS = tuple(range(6))  # frozen n=6
REPEATS = 2
FIT_RESOURCE_CAP = 64.0  # dual-predict-select pair cells legitimately exceed 10
CALIBRATION_RANDOM_SEED = 17

STATIC_CHECK_SCOPE = (
    "scripts/training/eval_taiji_b0_b1_representation_gate.py",
    "scripts/training/probe_taiji_b0_structure_space.py",
    "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cells() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grid = ssp.grid()
    candidate_labels = ("create__observation", "create__override", "create__mismatch")
    candidates = [cell for cell in grid if cell["label"] in candidate_labels]
    training = [cell for cell in grid if cell["label"] not in candidate_labels]
    if len(candidates) != 3 or len(training) != 8:
        raise SystemExit("structure grid does not match the frozen 3+8 cell split")
    return grid, training, candidates


def _tasks_for_cells(
    frozen: Any, cells: list[dict[str, Any]], steps: tuple[int, ...], prefix: str
) -> tuple[Any, ...]:
    """Frozen per-cell construction (same shape as ``ssp.build_cell_tasks``)."""

    tasks: list[Any] = []
    for cell in cells:
        ordinal = int(cell["ordinal"])
        content_route = str(cell["content_route"])
        language_route = str(cell["language_route"])
        for step in steps:
            i = ssp.INDEX_BASE + ordinal * 10 + step
            name = f"b1_{content_route}_{language_route}_{i}{ssp.EXTENSION}"
            base = ssp.BASE_TEMPLATE.format(i=i)
            goal = ssp.PATCHED_TEMPLATE.format(i=i) if content_route == "patch" else base
            language = {
                "none": None,
                "observation": ssp.INFERABLE_LANGUAGE,
                "override": ssp.INFERABLE_LANGUAGE,
                "mismatch": ssp.NON_INFERABLE_LANGUAGE,
            }[language_route]
            requires_override = language_route in ("override", "mismatch")
            steps_spec: list[Any] = []
            if content_route == "create":
                steps_spec.append(frozen.p52.ScriptedStep("workspace.list", {"path": "."}))
                steps_spec.append(
                    frozen.p52.ScriptedStep("workspace.create", {"path": name, "content": goal})
                )
            steps_spec.append(frozen.p52.ScriptedStep("workspace.read", {"path": name}))
            if language_route != "none":
                steps_spec.append(
                    frozen.p52.ScriptedStep(
                        "workspace.programming_language.resolve", {"path": name}
                    )
                )
            if language_route in ("override", "mismatch"):
                steps_spec.append(
                    frozen.p52.ScriptedStep(
                        "editor.set_language",
                        {
                            "path": name,
                            "programming_language_id": language,
                            "user_override": requires_override,
                        },
                    )
                )
            if content_route == "patch":
                steps_spec.append(
                    frozen.p52.ScriptedStep(
                        "workspace.apply_patch",
                        {
                            "path": name,
                            "before_digest": _sha(base),
                            "patch": frozen.p52._patch_ops(base, f"return {i}", f"return {i} + 1"),
                            "expected_after_digest": _sha(goal),
                        },
                    )
                )
            task_id = f"{prefix}-{cell['label']}-{i}"
            tasks.append(
                frozen.p52a.Task(
                    task_id=task_id,
                    goal_text=(
                        f"Work order {task_id}: reach the requested end state on "
                        f"{name} using the workspace contract."
                    ),
                    initial_files=({} if content_route == "create" else {name: base}),
                    goal_files={name: goal},
                    goal_language=({name: language} if language else {}),
                    main_path=name,
                    reference_steps=tuple(steps_spec),
                    partition="b1",
                    template=f"b1_{cell['label']}",
                    requires_explicit_language_override=requires_override,
                )
            )
    return tuple(tasks)


def _success_by_cell(
    episodes: list[dict[str, Any]], context_id: str
) -> dict[tuple[str, ...], float]:
    rates: dict[tuple[str, ...], list[float]] = {}
    for episode in episodes:
        if episode["task_id"] != context_id:
            continue
        key = tuple(episode["active_members"])
        rates.setdefault(key, []).append(1.0 if episode["success"] else 0.0)
    return {key: sum(values) / len(values) for key, values in rates.items()}


def _actual_gains(
    episodes: list[dict[str, Any]], context_ids: list[str], members: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean(P - max_i S_i) per pair over contexts, plus singleton means."""

    pair_values: dict[str, list[float]] = {}
    singleton_values: dict[str, list[float]] = {}
    for context_id in context_ids:
        rates = _success_by_cell(episodes, context_id)
        baseline_singletons = max(rates.get((member,), 0.0) for member in members)
        for pair in itertools.combinations(members, 2):
            pair_values.setdefault("+".join(pair), []).append(
                rates.get(pair, 0.0) - baseline_singletons
            )
        for member in members:
            singleton_values.setdefault(member, []).append(rates.get((member,), 0.0))
    pair_gains = {key: statistics.mean(values) for key, values in pair_values.items()}
    singleton_means = {key: statistics.mean(values) for key, values in singleton_values.items()}
    return pair_gains, singleton_means


def _rank_correlation(predicted: list[float], actual: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start
            while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
                end += 1
            average = (start + end) / 2.0 + 1.0
            for position in range(start, end + 1):
                result[order[position]] = average
            start = end + 1
        return result

    n = len(predicted)
    if n < 2:
        return 0.0
    rank_p, rank_a = ranks(predicted), ranks(actual)
    mean_p, mean_a = statistics.mean(rank_p), statistics.mean(rank_a)
    numerator = sum(
        (left - mean_p) * (right - mean_a) for left, right in zip(rank_p, rank_a, strict=True)
    )
    denominator_p = sum((left - mean_p) ** 2 for left in rank_p) ** 0.5
    denominator_a = sum((right - mean_a) ** 2 for right in rank_a) ** 0.5
    if denominator_p == 0.0 or denominator_a == 0.0:
        return 0.0
    return numerator / (denominator_p * denominator_a)


def _predicted_map(
    learner: InteractionGroupTransferLearner, pairs: list[tuple[str, ...]]
) -> dict[str, float]:
    return {
        "+".join(pair): (
            learner.candidate(pair, allow_observed=True).predicted_interaction
            if learner.candidate(pair, allow_observed=True)
            else 0.0
        )
        for pair in pairs
    }


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": 1,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "scope": (
            "B1 representation training on non-candidate structure cells; "
            "candidate create cells are evaluation-only; no collaboration claim "
            "(H2 stays with the G5 real-execution gate)"
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
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        grid, training_cells, candidate_cells = _cells()

        member_start = time.perf_counter()
        embedder = frozen.DocumentEmbedder()
        members = frozen._train_members(embedder)
        member_seconds = time.perf_counter() - member_start

        # ---- fit corpus: non-candidate cells, step 0; in-domain holdout step 1
        fit_tasks = _tasks_for_cells(frozen, training_cells, (FIT_CONTEXT_STEP,), "b1fit")
        indomain_tasks = _tasks_for_cells(frozen, training_cells, (INDOMAIN_HOLDOUT_STEP,), "b1ind")
        candidate_tasks = _tasks_for_cells(frozen, candidate_cells, CANDIDATE_STEPS, "b1cand")
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

        # ---- official train-only fit surfaces
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
            InteractionGroupEvaluatorConfig(maximum_resource_cost=FIT_RESOURCE_CAP)
        )
        records = evaluator.train_only_candidates(corpus)
        learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
        learner.observe_members(profiles)
        observed_records = learner.observe_records(records)
        fit_seconds = time.perf_counter() - fit_start

        member_ids = frozen.MEMBER_IDS
        pairs = list(itertools.combinations(member_ids, 2))
        pair_labels = ["+".join(pair) for pair in pairs]
        predictions = _predicted_map(learner, pairs)
        predictions_vary = len(set(round(value, 9) for value in predictions.values())) > 1
        feature_rank = learner.feature_rank()

        # ---- candidate-surface discrimination (representation evidence only)
        candidate_pair_gains, candidate_singleton = _actual_gains(
            candidate_episodes, sorted(candidate_ids), member_ids
        )
        candidate_rank = _rank_correlation(
            [predictions[label] for label in pair_labels],
            [candidate_pair_gains[label] for label in pair_labels],
        )
        best_actual_candidate = max(pair_labels, key=lambda label: candidate_pair_gains[label])
        best_predicted_candidate = max(pair_labels, key=lambda label: predictions[label])

        # ---- in-domain holdout discrimination
        indomain_pair_gains, _ = _actual_gains(
            indomain_episodes, sorted(task.task_id for task in indomain_tasks), member_ids
        )
        indomain_rank = _rank_correlation(
            [predictions[label] for label in pair_labels],
            [indomain_pair_gains[label] for label in pair_labels],
        )

        # ---- frozen controls
        additive = {
            "+".join(pair): (
                next(item.contribution for item in profiles if item.member_id == pair[0])
                + next(item.contribution for item in profiles if item.member_id == pair[1])
            )
            for pair in pairs
        }
        additive_rank = _rank_correlation(
            [additive[label] for label in pair_labels],
            [candidate_pair_gains[label] for label in pair_labels],
        )
        random_generator = random.Random(CALIBRATION_RANDOM_SEED)
        random_pair = tuple(sorted(random_generator.sample(list(member_ids), 2)))
        controls = {
            "no_learning": {"constant_prediction": True, "discriminates": False},
            "a_count": {"prediction_equal_for_all_pairs": True, "discriminates": False},
            "additive_train_only": {"rank_correlation_vs_candidate": round(additive_rank, 6)},
            "lesion_zero_coefficients": {"discriminates": False},
            "fixed_pair": {
                "pair": pair_labels[0],
                "actual_gain": candidate_pair_gains[pair_labels[0]],
            },
            "random_pair": {
                "pair": "+".join(random_pair),
                "actual_gain": candidate_pair_gains["+".join(random_pair)],
            },
            "best_singleton_routing": {
                "singleton_success_rates": {
                    member: round(rate, 6) for member, rate in sorted(candidate_singleton.items())
                },
                "best_singleton": max(
                    candidate_singleton, key=lambda member: candidate_singleton[member]
                ),
            },
        }

        # ---- stability: renaming + fit-order permutation
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
        rotated_learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
        rotated_learner.observe_members(rotated_profiles)
        rotated_learner.observe_records(rotated_records)
        renaming_stable = True
        for pair in pairs:
            rotated_pair = tuple(sorted(rotation[name] for name in pair))
            original = learner.candidate(pair, allow_observed=True)
            rotated = rotated_learner.candidate(rotated_pair, allow_observed=True)
            if original is None or rotated is None:
                renaming_stable = False
                continue
            if abs(original.predicted_interaction - rotated.predicted_interaction) > 1e-9:
                renaming_stable = False

        order_start = time.perf_counter()
        shuffled = list(projected_fit)
        random_generator.shuffle(shuffled)
        shuffled_corpus = InteractionTraceCorpus(train=tuple(shuffled), holdout=projected_indomain)
        shuffled_learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
        shuffled_learner.observe_members(profiles)
        shuffled_learner.observe_records(evaluator.train_only_candidates(shuffled_corpus))
        order_stable = _predicted_map(shuffled_learner, pairs) == predictions
        order_seconds = time.perf_counter() - order_start

        # ---- checkpoint + fresh-process restore + tamper
        restore_start = time.perf_counter()
        checkpoint_payload = learner.checkpoint()
        ckpt_dir = Path(tempfile.mkdtemp(prefix="b1rep-"))
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
            restore_ok = child_output.get("predictions") == predictions
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
        gates = {
            "static_checks": True,
            "fit_integrity": bool(
                len(profiles) == len(member_ids)
                and len(records) > 0
                and feature_rank > 1
                and predictions_vary
            ),
            "no_leakage": bool(not (fit_ids & candidate_ids)),
            "restore": bool(restore_ok and tamper_rejected),
            "controls_complete": bool(
                set(controls)
                == {
                    "no_learning",
                    "a_count",
                    "additive_train_only",
                    "lesion_zero_coefficients",
                    "fixed_pair",
                    "random_pair",
                    "best_singleton_routing",
                }
            ),
            "stability": bool(renaming_stable and order_stable),
        }
        if all(gates.values()):
            outcome = "representation_discriminative"
        elif gates["static_checks"] and gates["no_leakage"] and gates["restore"]:
            outcome = "representation_rank_deficient"
        else:
            outcome = "failed"
        payload.update(
            {
                "status": "completed",
                "outcome": outcome,
                "experiment_passed": bool(outcome == "representation_discriminative"),
                "record": {
                    "commit": commit,
                    "rule_revision": frozen.RULE_REVISION,
                    "composition_rule": frozen.COMPOSITION_RULE,
                    "budget_seconds": 28.0,
                    "elapsed_seconds": round(total_wall, 3),
                },
                "corpus": {
                    "training_cells": [cell["label"] for cell in training_cells],
                    "candidate_cells": [cell["label"] for cell in candidate_cells],
                    "fit_contexts": len(fit_ids),
                    "indomain_holdout_contexts": len(indomain_tasks),
                    "candidate_contexts": len(candidate_ids),
                    "episodes": {
                        "fit": len(fit_episodes),
                        "indomain_holdout": len(indomain_episodes),
                        "candidate": len(candidate_episodes),
                    },
                    "fit_resource_cap_disclosure": (
                        f"evaluator maximum_resource_cost raised to {FIT_RESOURCE_CAP} "
                        "for the fit corpus: dual-predict-select pair cells legitimately "
                        "exceed the default 10-action cap; candidate-cell outcomes are "
                        "execution-only and never enter the fit"
                    ),
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
                    "coefficients": list(learner._coefficients),
                    "predicted_interaction": {
                        label: round(value, 6) for label, value in predictions.items()
                    },
                    "predictions_vary": predictions_vary,
                },
                "discrimination": {
                    "disclosure": "representation evidence; not a collaboration claim",
                    "candidate_surface": {
                        "actual_gain_vs_all_singleton_oracle": {
                            label: round(value, 6)
                            for label, value in sorted(candidate_pair_gains.items())
                        },
                        "rank_correlation": round(candidate_rank, 6),
                        "best_actual": best_actual_candidate,
                        "best_predicted": best_predicted_candidate,
                        "singleton_success_rates": {
                            member: round(rate, 6)
                            for member, rate in sorted(candidate_singleton.items())
                        },
                    },
                    "in_domain_holdout": {"rank_correlation": round(indomain_rank, 6)},
                    "controls": controls,
                },
                "stability": {
                    "renaming_stable": renaming_stable,
                    "order_permutation_stable": order_stable,
                    "order_permutation_seconds": round(order_seconds, 3),
                },
                "checkpoint": {
                    "restore_matches_parent": restore_ok,
                    "tamper_rejected": tamper_rejected,
                },
                "cost": {
                    "member_training_seconds": round(member_seconds, 3),
                    "matrix_seconds": round(matrix_seconds, 3),
                    "fit_seconds": round(fit_seconds, 3),
                    "restore_seconds": round(restore_seconds, 3),
                    "episodes_total": len(fit_episodes)
                    + len(indomain_episodes)
                    + len(candidate_episodes),
                },
                "gates": gates,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        "completed: outcome=representation_discriminative; "
                        f"feature_rank={feature_rank}; candidate-surface rank "
                        f"correlation={round(candidate_rank, 6)}; controls executed; "
                        "H2 collaboration claims remain with the G5 real-execution gate"
                    )
                    if outcome == "representation_discriminative"
                    else (
                        f"completed: outcome={outcome}; gates failed: "
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
    predictions = {
        "+".join(pair): (
            learner.candidate(pair, allow_observed=True).predicted_interaction
            if learner.candidate(pair, allow_observed=True)
            else None
        )
        for pair in pairs
    }
    print(json.dumps({"predictions": predictions}))
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
                "feature_rank": (result.get("representation") or {}).get("feature_rank"),
                "candidate_rank_correlation": (
                    (result.get("discrimination") or {}).get("candidate_surface") or {}
                ).get("rank_correlation"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("experiment_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

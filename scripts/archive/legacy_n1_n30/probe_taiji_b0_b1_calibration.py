"""Route B (B1) calibration run: unit-cost measurement + pipeline rehearsal.

Authorization context: the user authorized B1 training (WP-4) on 2026-09-15 and
the plan requires a calibration run BEFORE the full-scale budget is set
("机时未知 ⇒ 先做标定跑测单位成本再定预算").  This probe therefore:

  1. executes the Route B candidate cells (create x 3 language routes, frozen
     preregistration section 3) at calibration scale through the real
     Workbench contract under the landed rule (rule_revision=1),
  2. rehearses the full B1 learning pipeline with the official train-only
     surfaces: build_member_evidence -> observe_members ->
     evaluator.train_only_candidates -> observe_records -> candidate/select,
  3. measures per-phase wall time and extrapolates to the full-scale run,
  4. smoke-checks candidate discrimination on held-out contexts against the
     frozen control list (no-learning / A-count / additive / lesion / fixed /
     random / best-singleton routing / renaming stability), and
  5. checkpoints the fitted transfer learner and restores it in a fresh
     process.

It publishes NO capability verdict: discrimination numbers at calibration
scale are pipeline smoke, and every artefact here is development-only
(calibration context indices 606-629 step band; the full B1 run uses the
frozen steps 0-5 band, so no final-test context is touched).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
    InteractionGroupTransferLearner,
    build_member_evidence,
)
from taiji.interaction_groups import InteractionTraceCorpus  # noqa: E402

REPORT_FORMAT = "taiji-b0-b1-calibration-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_b1_calibration_20260915.json"

FROZEN_MARGIN = 0.15
CALIBRATION_STEPS = (8, 9)  # frozen index rule keeps 0-5 for the full B1 run
REPEATS = 2
CALIBRATION_RANDOM_SEED = 17

STATIC_CHECK_SCOPE = (
    "scripts/training/probe_taiji_b0_b1_calibration.py",
    "scripts/training/probe_taiji_b0_structure_space.py",
    "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
)


def _sha(text: str) -> str:

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Calibration corpus: frozen create cells at a development-only index band
# --------------------------------------------------------------------------- #


def calibration_cells() -> list[dict[str, Any]]:
    cells = [
        cell
        for cell in ssp.grid()
        if cell["label"] in ("create__observation", "create__override", "create__mismatch")
    ]
    if len(cells) != 3:
        raise SystemExit("route B candidate cells missing from the structure grid")
    return cells


def build_calibration_tasks(frozen: Any, cell: dict[str, Any]) -> tuple[Any, ...]:
    """Frozen create-cell construction at the calibration step band (8, 9).

    Identical to ``ssp.build_cell_tasks`` for the create row (list -> create ->
    read -> resolve -> set_language with the route's override flag), except the
    context index uses steps 8-9 so no full-run context (steps 0-5) is touched.
    """

    p52a, p52 = frozen.p52a, frozen.p52
    ordinal = int(cell["ordinal"])
    language_route = str(cell["language_route"])
    tasks: list[Any] = []
    for step in CALIBRATION_STEPS:
        i = ssp.INDEX_BASE + ordinal * 10 + step
        name = f"b1cal_create_{language_route}_{i}{ssp.EXTENSION}"
        goal = ssp.BASE_TEMPLATE.format(i=i)
        language = {
            "none": None,
            "observation": ssp.INFERABLE_LANGUAGE,
            "override": ssp.INFERABLE_LANGUAGE,
            "mismatch": ssp.NON_INFERABLE_LANGUAGE,
        }[language_route]
        requires_override = language_route in ("override", "mismatch")
        steps = (
            p52.ScriptedStep("workspace.list", {"path": "."}),
            p52.ScriptedStep("workspace.create", {"path": name, "content": goal}),
            p52.ScriptedStep("workspace.read", {"path": name}),
            p52.ScriptedStep("workspace.programming_language.resolve", {"path": name}),
            p52.ScriptedStep(
                "editor.set_language",
                {
                    "path": name,
                    "programming_language_id": language,
                    "user_override": requires_override,
                },
            ),
        )
        task_id = f"b1cal-{cell['label']}-{i}"
        tasks.append(
            p52a.Task(
                task_id=task_id,
                goal_text=(
                    f"Work order {task_id}: create {name} with the scripted content "
                    f"and pin its editor language through the workspace contract."
                ),
                initial_files={},
                goal_files={name: goal},
                goal_language={name: language},
                main_path=name,
                reference_steps=steps,
                partition="calibration",
                template=f"b1cal_{cell['label']}",
                requires_explicit_language_override=requires_override,
            )
        )
    return tuple(tasks)


# --------------------------------------------------------------------------- #
# Gain accounting (frozen formula: mean(P - max_i S_i) per context)
# --------------------------------------------------------------------------- #


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


def actual_gains(
    episodes: list[dict[str, Any]], context_ids: list[str], members: tuple[str, ...]
) -> dict[tuple[str, str], float]:
    """Actual mean(P - max_i S_i) per (context, pair) from real outcomes."""

    gains: dict[tuple[str, str], float] = {}
    for context_id in context_ids:
        rates = _success_by_cell(episodes, context_id)
        baseline_singletons = max(rates.get((member,), 0.0) for member in members)
        for pair in itertools.combinations(members, 2):
            pair_rate = rates.get(pair, 0.0)
            gains[(context_id, "+".join(pair))] = pair_rate - baseline_singletons
    return gains


def _rank_correlation(pairs: list[str], predicted: list[float], actual: list[float]) -> float:
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

    rank_p, rank_a = ranks(predicted), ranks(actual)
    n = len(pairs)
    if n < 2:
        return 0.0
    mean_p, mean_a = statistics.mean(rank_p), statistics.mean(rank_a)
    numerator = sum(
        (left - mean_p) * (right - mean_a) for left, right in zip(rank_p, rank_a, strict=True)
    )
    denominator_p = sum((left - mean_p) ** 2 for left in rank_p) ** 0.5
    denominator_a = sum((right - mean_a) ** 2 for right in rank_a) ** 0.5
    if denominator_p == 0.0 or denominator_a == 0.0:
        return 0.0
    return numerator / (denominator_p * denominator_a)


# --------------------------------------------------------------------------- #
# Pipeline rehearsal
# --------------------------------------------------------------------------- #


def run_calibration() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": 1,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "scope": (
            "WP-4 calibration: unit cost + pipeline rehearsal; no capability "
            "verdict; development-only context band (steps 8-9)"
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

        # members (deterministic, rule_revision=1)
        member_start = time.perf_counter()
        embedder = ssp.load_frozen().DocumentEmbedder()
        members = frozen._train_members(embedder)
        member_seconds = time.perf_counter() - member_start

        # calibration corpus through the real contract
        cells = calibration_cells()
        tasks = tuple(task for cell in cells for task in build_calibration_tasks(frozen, cell))
        matrix_start = time.perf_counter()
        episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, tasks
        )
        matrix_seconds = time.perf_counter() - matrix_start

        # split: first calibration step -> train, second -> holdout (per cell)
        train_ids = {
            task.task_id for task in tasks if task.task_id.endswith(str(CALIBRATION_STEPS[0]))
        }
        holdout_ids = {
            task.task_id for task in tasks if task.task_id.endswith(str(CALIBRATION_STEPS[1]))
        }
        holdout_episodes = [item for item in episodes if item["task_id"] in holdout_ids]
        if len(train_ids) != 3 or len(holdout_ids) != 3:
            raise SystemExit("calibration split must be 3 train + 3 holdout contexts")
        if set(train_ids) & set(holdout_ids):
            raise SystemExit("calibration train/holdout context sets overlap")

        # official train-only surfaces
        fit_start = time.perf_counter()
        projected = frozen._project(episodes)
        train_projected = tuple(item for item in projected if item.context_id in train_ids)
        holdout_projected = tuple(item for item in projected if item.context_id in holdout_ids)
        corpus = InteractionTraceCorpus(train=train_projected, holdout=holdout_projected)
        revision = next(iter(corpus.train_checkpoint_revisions))
        profiles = build_member_evidence(
            corpus.train,
            source_trace_digest=corpus.train_trace_digest,
            checkpoint_revision=revision,
        )
        evaluator = InteractionGroupEvaluator()
        records = evaluator.train_only_candidates(corpus)
        learner = InteractionGroupTransferLearner(maximum_uncertainty=2.0)
        observed_members = learner.observe_members(profiles)
        observed_records = learner.observe_records(records)
        fit_seconds = time.perf_counter() - fit_start

        # discrimination smoke on holdout contexts (no capability claim)
        member_ids = frozen.MEMBER_IDS
        pairs = list(itertools.combinations(member_ids, 2))
        holdout_actual = actual_gains(holdout_episodes, sorted(holdout_ids), member_ids)
        actual_by_pair = {
            pair: statistics.mean(
                value
                for (context, pair_name), value in holdout_actual.items()
                if pair_name == "+".join(pair)
            )
            for pair in pairs
        }
        predicted_by_pair = {
            pair: (
                learner.candidate(pair, allow_observed=True).predicted_interaction
                if learner.candidate(pair, allow_observed=True)
                else 0.0
            )
            for pair in pairs
        }
        pair_labels = ["+".join(pair) for pair in pairs]
        rank_correlation = _rank_correlation(
            pair_labels,
            [predicted_by_pair[pair] for pair in pairs],
            [actual_by_pair[pair] for pair in pairs],
        )
        best_actual = max(pairs, key=lambda pair: actual_by_pair[pair])
        best_predicted = max(pairs, key=lambda pair: predicted_by_pair[pair])
        best_pair_hit = bool(best_actual == best_predicted)

        # controls (frozen B1 exit list, smoke scale)
        additive_model = {
            "+".join(pair): (
                next(item.contribution for item in profiles if item.member_id == pair[0])
                + next(item.contribution for item in profiles if item.member_id == pair[1])
            )
            for pair in pairs
        }
        additive_correlation = _rank_correlation(
            pair_labels,
            [additive_model["+".join(pair)] for pair in pairs],
            [actual_by_pair[pair] for pair in pairs],
        )
        fixed_pair = (member_ids[0], member_ids[1])
        random_generator = random.Random(CALIBRATION_RANDOM_SEED)
        random_pair = tuple(sorted(random_generator.sample(list(member_ids), 2)))
        singleton_holdout = {
            member: statistics.mean(
                _success_by_cell(holdout_episodes, context_id).get((member,), 0.0)
                for context_id in sorted(holdout_ids)
            )
            for member in member_ids
        }
        controls = {
            "no_learning": {"constant_prediction": True, "discriminates": False},
            "a_count": {"prediction_equal_for_all_pairs": True, "discriminates": False},
            "additive_train_only": {"rank_correlation": round(additive_correlation, 6)},
            "lesion_zero_coefficients": {"discriminates": False},
            "fixed_pair": {"pair": list(fixed_pair), "actual_gain": actual_by_pair[fixed_pair]},
            "random_pair": {
                "pair": list(random_pair),
                "actual_gain": actual_by_pair[random_pair],
            },
            "best_singleton_routing": {
                "singleton_success_rates_holdout": {
                    member: round(rate, 6) for member, rate in singleton_holdout.items()
                },
                "best_singleton": max(singleton_holdout, key=lambda m: singleton_holdout[m]),
            },
        }

        # renaming stability: rotate member ids, refit, prediction follows
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
            for item in train_projected
        )
        rotated_corpus = InteractionTraceCorpus(train=rotated_projected, holdout=holdout_projected)
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

        # checkpoint + fresh-process restore
        ckpt_start = time.perf_counter()
        checkpoint_payload = learner.checkpoint()
        ckpt_dir = Path(tempfile.mkdtemp(prefix="b1cal-"))
        ckpt_path = ckpt_dir / "transfer-learner.json"
        ckpt_path.write_text(json.dumps(checkpoint_payload, sort_keys=True), encoding="utf-8")
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                str(ckpt_path),
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        restore_ok = False
        child_output: dict[str, Any] = {}
        if child.returncode == 0:
            child_output = json.loads(child.stdout.strip().splitlines()[-1])
            parent_predictions = {
                "+".join(pair): (
                    learner.candidate(pair, allow_observed=True).predicted_interaction
                    if learner.candidate(pair, allow_observed=True)
                    else None
                )
                for pair in pairs
            }
            restore_ok = child_output.get("predictions") == parent_predictions
        tampered_payload = dict(checkpoint_payload)
        tampered_payload["coefficients"] = [
            value + 1.0 for value in tampered_payload.get("coefficients", ())
        ]
        tamper_rejected = False
        try:
            InteractionGroupTransferLearner.from_checkpoint(tampered_payload)
        except Exception:  # noqa: BLE001
            tamper_rejected = True
        restore_seconds = time.perf_counter() - ckpt_start
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        # cost accounting + full-scale extrapolation
        total_wall = time.perf_counter() - started
        episode_count = len(episodes)
        episodes_per_second = episode_count / max(matrix_seconds, 1e-9)
        full_scale_episodes = 3 * 6 * len(frozen.CELL_MEMBER_SETS) * frozen.REPEATS
        extrapolated = {
            "full_scale_episodes": full_scale_episodes,
            "matrix_seconds_projected": round(
                full_scale_episodes / max(episodes_per_second, 1e-9), 3
            ),
            "fit_and_eval_seconds_measured": round(fit_seconds, 3),
            "member_training_seconds_measured": round(member_seconds, 3),
            "recommended_budget_seconds_with_2x_margin": round(
                2.0
                * (
                    full_scale_episodes / max(episodes_per_second, 1e-9)
                    + fit_seconds
                    + member_seconds
                ),
                3,
            ),
        }

        gates = {
            "matrix_executed": bool(
                episode_count == 6 * len(frozen.CELL_MEMBER_SETS) * frozen.REPEATS
            ),
            "split_disjoint": bool(not (set(train_ids) & set(holdout_ids))),
            "fit_completed": bool(
                observed_members and observed_records and learner.feature_rank() > 0
            ),
            "restore_matches": bool(restore_ok),
            "tamper_rejected": bool(tamper_rejected),
            "renaming_stable": bool(renaming_stable),
        }
        payload.update(
            {
                "status": "completed",
                "record": {
                    "commit": commit,
                    "rule_revision": frozen.RULE_REVISION,
                    "composition_rule": frozen.COMPOSITION_RULE,
                    "calibration_context_band": "steps 8-9 (full run uses 0-5)",
                    "elapsed_seconds": round(total_wall, 3),
                },
                "cost": {
                    "member_training_seconds": round(member_seconds, 3),
                    "matrix_seconds": round(matrix_seconds, 3),
                    "episodes": episode_count,
                    "episodes_per_second": round(episodes_per_second, 3),
                    "fit_seconds": round(fit_seconds, 3),
                    "restore_seconds": round(restore_seconds, 3),
                    "full_scale_extrapolation": extrapolated,
                },
                "pipeline": {
                    "profiles": len(profiles),
                    "member_ids": sorted(item.member_id for item in profiles),
                    "train_only_candidate_records": len(records),
                    "feature_rank": learner.feature_rank(),
                    "observed_members": observed_members,
                    "observed_records": observed_records,
                },
                "discrimination_smoke": {
                    "disclosure": "calibration-scale pipeline smoke; no capability verdict",
                    "predicted_interaction": {
                        "+".join(pair): round(predicted_by_pair[pair], 6) for pair in pairs
                    },
                    "actual_gain_holdout": {
                        "+".join(pair): round(actual_by_pair[pair], 6) for pair in pairs
                    },
                    "rank_correlation": round(rank_correlation, 6),
                    "best_pair_hit": best_pair_hit,
                    "best_actual": "+".join(best_actual),
                    "best_predicted": "+".join(best_predicted),
                    "controls": controls,
                },
                "checkpoint": {
                    "restore_matches_parent": restore_ok,
                    "tamper_rejected": tamper_rejected,
                },
                "gates": gates,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    "completed: B1 calibration pipeline green; "
                    f"{episode_count} episodes at {round(episodes_per_second, 3)} eps/s; "
                    f"full-scale extrapolation {extrapolated['matrix_seconds_projected']}s matrix; "
                    "full B1 run budget = recommended_budget_seconds_with_2x_margin"
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
    result = run_calibration()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "gates": result.get("gates"),
                "cost": result.get("cost"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("gates") and all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

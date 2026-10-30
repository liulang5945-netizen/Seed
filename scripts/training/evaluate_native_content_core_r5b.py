"""R5b evaluation: holdout/query behavior, mechanism, retention, gates.

Execution is gated on ``--approve-r5b`` (same approval as training).  The
runner consumes the frozen R5b training report and endpoint checkpoints, then
reads the query and holdout splits for the first time.  Sealed data is never
read.  The two baselines are NOT re-evaluated: their R5 evaluation report is
loaded by sha and their q1 vectors plus costs are reused exactly as declared in
the R5b contract section 4.

Declared scope (fixed before the first holdout read): all 72 holdout and 72
query histories, every declared answer turn; mechanism probes, retention and
receipt only for the three q1 candidate runs; Q2 compares credit_on vs
credit_stop on holdout answer_exact.  The verdict follows the R4 contract
4.1-4.6 with unchanged gate values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_content_eval import (  # noqa: E402
    bootstrap_diff,
    build_mechanism_case,
    exact_vector,
    expected_turns,
    mechanism_cases,
    receipt_ratio,
    replay_episode,
)
from taiji.native_content_evaluator import score_run  # noqa: E402
from taiji.native_content_training_v2 import R5BRunTrainer  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
QUERY = ROOT / "plans/manifests/native_core_r1_query_v1.json"
HOLDOUT = ROOT / "plans/manifests/native_core_r1_holdout_v1.json"
TRAINING_REPORT = ROOT / "reports/taiji_native_core_r5b_training_20261030.json"
R5_EVALUATION = ROOT / "reports/taiji_native_core_r5_evaluation_20261030.json"
R5_TRAINING = ROOT / "reports/taiji_native_core_r5_training_20261030.json"
REPORT = ROOT / "reports/taiji_native_core_r5b_evaluation_20261030.json"

MECHANISM_LIMIT = 128
RETENTION_EPISODES = 72
SAMPLE_SEED = 20261030
BOOTSTRAP_DRAWS = 1000
BEHAVIOR_GATES = {
    "answer_exact": 0.80,
    "current_query": 0.85,
    "previous_query": 0.75,
    "unknown_query": 0.85,
    "state_exact": 0.90,
}
MECHANISM_CUT_FLOOR = 0.50
MECHANISM_RESCUE_FLOOR = 0.90
MECHANISM_MIN_CASES = 30
RETENTION_EXACT_FLOOR = 0.85
RETENTION_GAP_CEILING = 0.10
RECEIPT_FLOOR = 0.95
COMPARISON_MARGIN = 0.05
QUERY_MARGIN = 0.03
Q2_MARGIN = 0.05
COST_RATIO_CEILING = 2.0
STATUS_BY_FAMILY = {
    "behavior": "FAIL_BEHAVIOR",
    "mechanism": "FAIL_MECHANISM",
    "retention": "FAIL_RETENTION",
    "comparison": "FAIL_BEHAVIOR",
    "cost": "FAIL_COST",
    "recovery": "FAIL_RECOVERY",
}


def _manifest(path: Path, split: str) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != split:
        raise ValueError(f"{path.name} is not the declared {split} split")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate_split(
    adapter: Any, model: Any, episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    predictions: dict[tuple[str, int], Any] = {}
    per_turn_results: dict[tuple[str, int], Any] = {}
    for episode in episodes:
        result = replay_episode(adapter, model, episode)
        predictions.update(result.predictions)
        for key in result.predictions:
            per_turn_results[key] = result
    expected = expected_turns(episodes)
    return {
        "expected": expected,
        "predictions": predictions,
        "per_turn_results": per_turn_results,
        "score": score_run(expected, predictions),
        "exact_vector": exact_vector(expected, predictions),
    }


def _mechanism(
    adapter: Any,
    model: Any,
    episodes: Sequence[Mapping[str, Any]],
    per_turn_results: Mapping[tuple[str, int], Any],
) -> dict[str, Any]:
    rows = mechanism_cases(episodes, per_turn_results, limit=MECHANISM_LIMIT, seed=SAMPLE_SEED)
    cases = []
    for row in rows:
        key = (str(row["episode"]["episode_id"]), int(row["turn_index"]))
        cases.append(
            build_mechanism_case(
                adapter,
                model,
                row,
                baseline_prediction=per_turn_results[key].predictions[key],
            )
        )
    if not cases:
        return {
            "cases": 0,
            "cut_drop": None,
            "rescue_gain": None,
            "relevant_change": [],
            "unrelated_change": [],
        }
    keys = [case.key for case in cases]
    expected = [
        turn for turn in expected_turns(episodes) if (turn.episode_id, turn.turn_index) in set(keys)
    ]
    score = score_run(
        expected,
        {case.key: case.baseline for case in cases},
        mechanisms=cases,
    )
    relevant_changes = [
        0.0 if case.baseline.produced == case.relevant_swap.produced else 1.0
        for case in cases
        if case.relevant_swap is not None
    ]
    unrelated_changes = [
        0.0 if case.baseline.produced == case.unrelated_swap.produced else 1.0
        for case in cases
        if case.unrelated_swap is not None
    ]
    return {
        "cases": len(cases),
        "cut_drop": score.get("cut_drop"),
        "rescue_gain": score.get("rescue_gain"),
        "relevant_change": relevant_changes,
        "unrelated_change": unrelated_changes,
    }


def _retention(
    adapter: Any, model: Any, train_episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    chooser = random.Random(SAMPLE_SEED)
    sample = sorted(chooser.sample(range(len(train_episodes)), RETENTION_EPISODES))
    result = _evaluate_split(adapter, model, [train_episodes[index] for index in sample])
    return {"episodes": len(sample), "score": result["score"]}


def _pooled_candidate_metrics(entries: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    vectors = [value for entry in entries for value in entry[f"{split}_exact_vector"]]
    categories: dict[str, float] = {}
    for category in ("current_query", "previous_query", "unknown_query"):
        rates = []
        for entry in entries:
            bucket = entry[split]["by_category"][category]
            rates.append(bucket["exact"] / bucket["turns"])
        categories[category] = _mean(rates)
    state_values = [entry[split]["state_exact"] for entry in entries]
    state = (
        None
        if any(value is None for value in state_values)
        else _mean([float(value) for value in state_values])
    )
    return {
        "answer_exact": _mean(vectors),
        "turns": len(vectors),
        **categories,
        "state_exact": state,
    }


def _families(
    pooled: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    retention: Mapping[str, Any],
    receipt: Mapping[str, Any],
    q1_diff: Mapping[str, Any],
    query_diff: Mapping[str, Any],
    q2_diff: Mapping[str, Any],
    cost_ratio: float | None,
    recovery_ok: bool,
) -> dict[str, Any]:
    behavior = {
        name: {
            "value": pooled.get(name),
            "floor": floor,
            "pass": pooled.get(name) is not None and pooled[name] >= floor,
        }
        for name, floor in BEHAVIOR_GATES.items()
    }
    change = None
    if mechanism.get("relevant_change") and mechanism.get("unrelated_change"):
        change = bootstrap_diff(
            mechanism["relevant_change"],
            mechanism["unrelated_change"],
            draws=BOOTSTRAP_DRAWS,
            seed=SAMPLE_SEED,
        )
    mechanism_gates = {
        "cut_drop": {
            "value": (
                None
                if mechanism.get("cases", 0) < MECHANISM_MIN_CASES
                else mechanism.get("cut_drop")
            ),
            "floor": 0.50,
            "pass": mechanism.get("cases", 0) >= MECHANISM_MIN_CASES
            and mechanism.get("cut_drop") is not None
            and float(mechanism["cut_drop"]) >= MECHANISM_CUT_FLOOR,
        },
        "rescue_gain": {
            "value": (
                None
                if mechanism.get("cases", 0) < MECHANISM_MIN_CASES
                else mechanism.get("rescue_gain")
            ),
            "floor": 0.90,
            "pass": mechanism.get("cases", 0) >= MECHANISM_MIN_CASES
            and mechanism.get("rescue_gain") is not None
            and float(mechanism["rescue_gain"]) >= MECHANISM_RESCUE_FLOOR,
        },
        "change_difference_ci_lower": {
            "value": None if change is None else change.get("lower_95"),
            "floor": 0.0,
            "pass": change is not None
            and change.get("lower_95") is not None
            and float(change["lower_95"]) >= 0.0,
        },
    }
    train_like = retention.get("answer_exact")
    holdout_exact = pooled.get("answer_exact")
    receipt_value = None if receipt.get("cases", 0) == 0 else receipt["exact"] / receipt["cases"]
    retention_gates = {
        "train_like_exact": {
            "value": train_like,
            "floor": RETENTION_EXACT_FLOOR,
            "pass": train_like is not None and train_like >= RETENTION_EXACT_FLOOR,
        },
        "holdout_minus_train_like": {
            "value": (
                None if train_like is None or holdout_exact is None else holdout_exact - train_like
            ),
            "ceiling": RETENTION_GAP_CEILING,
            "pass": train_like is not None
            and holdout_exact is not None
            and holdout_exact - train_like <= RETENTION_GAP_CEILING,
        },
        "receipt_exact": {
            "value": receipt_value,
            "floor": RECEIPT_FLOOR,
            "pass": receipt_value is not None and receipt_value >= RECEIPT_FLOOR,
        },
    }
    comparison_gates = {
        "holdout_margin": {
            "value": q1_diff.get("mean_diff"),
            "floor": COMPARISON_MARGIN,
            "pass": q1_diff.get("mean_diff") is not None
            and q1_diff["mean_diff"] >= COMPARISON_MARGIN,
        },
        "holdout_ci_lower": {
            "value": q1_diff.get("lower_95"),
            "floor": 0.0,
            "pass": q1_diff.get("lower_95") is not None and q1_diff["lower_95"] > 0,
        },
        "query_not_below": {
            "value": query_diff.get("mean_diff"),
            "floor": -QUERY_MARGIN,
            "pass": query_diff.get("mean_diff") is not None
            and query_diff["mean_diff"] >= -QUERY_MARGIN,
        },
        "q2_margin": {
            "value": q2_diff.get("mean_diff"),
            "floor": Q2_MARGIN,
            "pass": q2_diff.get("mean_diff") is not None and q2_diff["mean_diff"] >= Q2_MARGIN,
        },
        "q2_ci_lower": {
            "value": q2_diff.get("lower_95"),
            "floor": 0.0,
            "pass": q2_diff.get("lower_95") is not None and q2_diff["lower_95"] > 0,
        },
    }
    cost_value = {
        "value": cost_ratio,
        "ceiling": COST_RATIO_CEILING,
        "pass": cost_ratio is not None and cost_ratio <= COST_RATIO_CEILING,
    }
    return {
        "behavior": behavior,
        "mechanism": mechanism_gates,
        "retention": retention_gates,
        "comparison": comparison_gates,
        "cost": {"new_core_over_best_baseline": cost_value},
        "recovery": {"checkpoints": {"value": recovery_ok, "pass": recovery_ok}},
        "mechanism_change": change,
    }


def _verdict(families: Mapping[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    inconclusive = False
    for family, gates in families.items():
        if family == "mechanism_change":
            continue
        for name, gate in gates.items():
            value = gate.get("value")
            if value is None:
                inconclusive = True
            elif not gate.get("pass", False):
                failures.append(f"{family}.{name}")
    if failures:
        return STATUS_BY_FAMILY[failures[0].split(".")[0]], failures
    if inconclusive:
        return "INCONCLUSIVE", failures
    return "PASS_Q1", failures


def _evaluate(*, approve: bool, run_id: str) -> int:
    if not approve:
        raise SystemExit("R5b evaluation requires explicit user approval (--approve-r5b)")
    if REPORT.exists():
        raise SystemExit(f"R5b evaluation report is write-once: {REPORT}")
    training = json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))
    if training.get("status") != "r5b_training_complete":
        raise SystemExit("R5b evaluation requires a complete R5b training report")
    if len(training["runs"]) != 9 or training["read_boundaries"]["optimizer_updates"] != 9 * 512:
        raise SystemExit("R5b training report does not declare the frozen 9 x 512 exposure")
    r5_evaluation = json.loads(R5_EVALUATION.read_text(encoding="utf-8"))
    if r5_evaluation.get("status") == "ENGINEERING_FAILURE":
        raise SystemExit("baseline reuse requires a complete R5 evaluation report")
    r5_runs = r5_evaluation["runs"]

    holdout_episodes = list(_manifest(HOLDOUT, "holdout")["episodes"])
    query_episodes = list(_manifest(QUERY, "query")["episodes"])
    train_episodes = list(_manifest(TRAIN, "train")["episodes"])
    started_at = datetime.now(UTC)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()

    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    recovery_ok = True
    for name, record in training["runs"].items():
        checkpoint = ROOT / record["endpoint_restore"]["checkpoint"]
        if not checkpoint.exists():
            failures.append({"run": name, "error": "endpoint checkpoint missing"})
            break
        trainer = R5BRunTrainer.load(checkpoint)
        if trainer.probe_digest() != record["endpoint_restore"]["probe_digest"]:
            failures.append({"run": name, "error": "endpoint digest differs"})
            break
        adapter = trainer.adapter
        holdout = _evaluate_split(adapter, trainer.model, holdout_episodes)
        query = _evaluate_split(adapter, trainer.model, query_episodes)
        entry: dict[str, Any] = {
            "block": record["block"],
            "object": record["object"],
            "credit_mode": record["credit_mode"],
            "seed": record["seed"],
            "holdout": holdout["score"],
            "query": query["score"],
            "holdout_exact_vector": holdout["exact_vector"],
            "query_exact_vector": query["exact_vector"],
        }
        if record["block"] == "q1":
            receipt_exact, receipt_cases = receipt_ratio(
                holdout["expected"], holdout["predictions"]
            )
            entry["receipt"] = {"exact": receipt_exact, "cases": receipt_cases}
            entry["retention"] = _retention(adapter, trainer.model, train_episodes)
            entry["mechanism"] = _mechanism(
                adapter, trainer.model, holdout_episodes, holdout["per_turn_results"]
            )
        results[name] = entry

    families: dict[str, Any] = {}
    status = "ENGINEERING_FAILURE"
    failed: list[str] = []
    aggregate: dict[str, Any] = {}
    if not failures:
        candidate = [entry for entry in results.values() if entry["block"] == "q1"]
        credit_on = [
            entry
            for entry in results.values()
            if entry["block"] == "q2" and entry["credit_mode"] == "credit_on"
        ]
        credit_stop = [
            entry
            for entry in results.values()
            if entry["block"] == "q2" and entry["credit_mode"] == "credit_stop"
        ]

        def vector(entries: Sequence[Mapping[str, Any]], split: str) -> list[float]:
            return [value for entry in entries for value in entry[f"{split}_exact_vector"]]

        new_holdout = vector(candidate, "holdout")
        corrected_holdout = vector(
            [r5_runs[name] for name in r5_runs if r5_runs[name]["object"] == "corrected_old_v1"],
            "holdout",
        )
        ordinary_holdout = vector(
            [r5_runs[name] for name in r5_runs if r5_runs[name]["object"] == "ordinary_v1"],
            "holdout",
        )
        best_holdout = (
            corrected_holdout
            if _mean(corrected_holdout) >= _mean(ordinary_holdout)
            else ordinary_holdout
        )
        q1_diff = bootstrap_diff(new_holdout, best_holdout, draws=BOOTSTRAP_DRAWS, seed=SAMPLE_SEED)
        new_query = vector(candidate, "query")
        corrected_query = vector(
            [r5_runs[name] for name in r5_runs if r5_runs[name]["object"] == "corrected_old_v1"],
            "query",
        )
        ordinary_query = vector(
            [r5_runs[name] for name in r5_runs if r5_runs[name]["object"] == "ordinary_v1"],
            "query",
        )
        best_query = (
            corrected_query if _mean(corrected_query) >= _mean(ordinary_query) else ordinary_query
        )
        query_diff = bootstrap_diff(new_query, best_query, draws=BOOTSTRAP_DRAWS, seed=SAMPLE_SEED)
        q2_diff = bootstrap_diff(
            vector(credit_on, "holdout"),
            vector(credit_stop, "holdout"),
            draws=BOOTSTRAP_DRAWS,
            seed=SAMPLE_SEED,
        )

        mechanism_rows = [entry["mechanism"] for entry in candidate]
        mechanism = {
            "runs": len(mechanism_rows),
            "cases": sum(row["cases"] for row in mechanism_rows),
            "cut_drop": (
                None
                if any(row["cut_drop"] is None for row in mechanism_rows)
                else _mean([float(row["cut_drop"]) for row in mechanism_rows])
            ),
            "rescue_gain": (
                None
                if any(row["rescue_gain"] is None for row in mechanism_rows)
                else _mean([float(row["rescue_gain"]) for row in mechanism_rows])
            ),
            "relevant_change": [
                value for row in mechanism_rows for value in row["relevant_change"]
            ],
            "unrelated_change": [
                value for row in mechanism_rows for value in row["unrelated_change"]
            ],
        }
        retention_rows = [entry["retention"] for entry in candidate]
        retention = {
            "episodes": sum(row["episodes"] for row in retention_rows),
            "answer_exact": _mean([row["score"]["answer_exact"] for row in retention_rows]),
        }
        receipt = {
            "exact": sum(entry["receipt"]["exact"] for entry in candidate),
            "cases": sum(entry["receipt"]["cases"] for entry in candidate),
        }
        pooled = _pooled_candidate_metrics(candidate, "holdout")
        candidate_cost = sum(
            float(training["runs"][name]["cpu_seconds_total"])
            for name in training["runs"]
            if training["runs"][name]["block"] == "q1"
        )
        r5_costs = r5_evaluation["aggregate"]["cost_by_object"]
        baseline_cost = max(float(r5_costs["corrected_old_v1"]), float(r5_costs["ordinary_v1"]))
        cost_ratio = candidate_cost / baseline_cost
        families = _families(
            pooled,
            mechanism,
            retention,
            receipt,
            q1_diff,
            query_diff,
            q2_diff,
            cost_ratio,
            recovery_ok,
        )
        status, failed = _verdict(families)
        aggregate = {
            "q1_diff": q1_diff,
            "query_diff": query_diff,
            "q2_diff": q2_diff,
            "mechanism": mechanism,
            "retention": retention,
            "receipt": receipt,
            "pooled_candidate_holdout": pooled,
            "cost": {
                "candidate_q1_cpu_seconds": candidate_cost,
                "baseline_reused_from": "r5_evaluation.aggregate.cost_by_object",
                "baseline_cpu_seconds": baseline_cost,
                "ratio": cost_ratio,
            },
            "families": families,
            "failed_gates": failed,
        }

    total_wall = time.perf_counter() - started_wall
    total_cpu = time.process_time() - started_cpu
    report = {
        "format": "taiji-native-core-r5b-evaluation-v1",
        "version": 1,
        "date": "2026-10-30",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md",
        "training_report": {
            "path": TRAINING_REPORT.relative_to(ROOT).as_posix(),
            "sha256": _sha256(TRAINING_REPORT),
            "status": training["status"],
        },
        "baseline_reuse": {
            "r5_evaluation": {
                "path": R5_EVALUATION.relative_to(ROOT).as_posix(),
                "sha256": _sha256(R5_EVALUATION),
            },
            "declared": "R5b contract section 4: baselines are not retrained or re-evaluated",
        },
        "manifests": {
            "holdout": {
                "path": HOLDOUT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(HOLDOUT),
                "episodes": len(holdout_episodes),
            },
            "query": {
                "path": QUERY.relative_to(ROOT).as_posix(),
                "sha256": _sha256(QUERY),
                "episodes": len(query_episodes),
            },
            "train": {
                "path": TRAIN.relative_to(ROOT).as_posix(),
                "sha256": _sha256(TRAIN),
                "episodes": len(train_episodes),
            },
        },
        "protocol": {
            "run_id": run_id,
            "prediction_bytes": 48,
            "mechanism_limit_per_seed": MECHANISM_LIMIT,
            "retention_episodes": RETENTION_EPISODES,
            "sample_seed": SAMPLE_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "receipt_definition": "correct holdout turns whose selected value appears verbatim in the expected bytes",
            "mechanism_scope": "q1 candidate runs only (declared R5b protocol difference from R5)",
        },
        "runs": results,
        "aggregate": aggregate,
        "status": status,
        "engineering_failures": failures,
        "evaluation_cost": {"cpu_seconds": total_cpu, "wall_seconds": total_wall},
        "read_boundaries": {
            "query_split_read": True,
            "holdout_split_read": True,
            "sealed_read": False,
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    if not REPORT.exists():
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(REPORT)}, ensure_ascii=False))
    return 0 if status != "ENGINEERING_FAILURE" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261030")
    parser.add_argument(
        "--approve-r5b",
        action="store_true",
        help="required: the user approved R5b execution",
    )
    args = parser.parse_args()
    return _evaluate(approve=args.approve_r5b, run_id=args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())

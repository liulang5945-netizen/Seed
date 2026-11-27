"""R5 evaluation runner: holdout/query behavior, mechanism, retention, gates.

Execution is gated on ``--approve-r5`` (same approval as training).  The runner
consumes the frozen R5 training report and endpoint checkpoints, then reads the
query and holdout splits for the first time.  Sealed data is never read.

Declared evaluation scope (fixed before the first holdout read):

* all 72 holdout histories and all 72 query histories, every episode and every
  declared answer turn (fixed denominator, no filtering);
* mechanism probes only for ``new_core_v1``: 128 baseline-correct cases per
  seed, sampled with seed 20261030; cut/rescue/relevant swap/unrelated swap;
* retention sample: 72 train episodes sampled with seed 20261030 for the
  candidate only; receipt ratio over correct holdout turns with a readout;
* Q2 compares ``credit_on`` vs ``credit_stop`` on holdout answer_exact.

The verdict follows the R4 contract 4.1-4.6; negative results are recorded,
not rerun.
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
from taiji.native_content_training import NativeRunTrainerV1  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
QUERY = ROOT / "plans/manifests/native_core_r1_query_v1.json"
HOLDOUT = ROOT / "plans/manifests/native_core_r1_holdout_v1.json"
TRAINING_REPORT = ROOT / "reports/taiji_native_core_r5_training_20261030.json"
REPORT = ROOT / "reports/taiji_native_core_r5_evaluation_20261030.json"

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
            "floor": MECHANISM_CUT_FLOOR,
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
            "floor": MECHANISM_RESCUE_FLOOR,
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


def _price(run_id: str, *, approve: bool) -> int:
    if not approve:
        raise SystemExit("R5 evaluation requires explicit user approval (--approve-r5)")
    if REPORT.exists():
        raise SystemExit(f"R5 evaluation report is write-once: {REPORT}")
    training = json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))
    if training.get("status") != "r5_training_complete":
        raise SystemExit("R5 evaluation requires a complete R5 training report")
    if len(training["runs"]) != 15 or training["read_boundaries"]["optimizer_updates"] != 15 * 512:
        raise SystemExit("R5 training report does not declare the frozen 15 x 512 exposure")
    runs = training["runs"]
    holdout_episodes = list(_manifest(HOLDOUT, "holdout")["episodes"])
    query_episodes = list(_manifest(QUERY, "query")["episodes"])
    train_episodes = list(_manifest(TRAIN, "train")["episodes"])
    started_at = datetime.now(UTC)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()

    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    recovery_ok = True
    for name, record in runs.items():
        checkpoint = ROOT / record["endpoint_restore"]["checkpoint"]
        if not checkpoint.exists():
            failures.append({"run": name, "error": "endpoint checkpoint missing"})
            break
        trainer = NativeRunTrainerV1.load(checkpoint)
        digest = trainer.probe_digest()
        if digest != record["endpoint_restore"]["probe_digest"]:
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
        if adapter.name == "new_core_v1":
            receipt_exact, receipt_cases = receipt_ratio(
                holdout["expected"], holdout["predictions"]
            )
            entry["receipt"] = {"exact": receipt_exact, "cases": receipt_cases}
            entry["retention"] = _retention(adapter, trainer.model, train_episodes)
            entry["mechanism"] = _mechanism(
                adapter,
                trainer.model,
                holdout_episodes,
                holdout["per_turn_results"],
            )
        results[name] = entry

    families: dict[str, Any] = {}
    status = "ENGINEERING_FAILURE"
    failed: list[str] = []
    aggregate: dict[str, Any] = {}
    if not failures:
        candidate = [
            entry
            for entry in results.values()
            if entry["object"] == "new_core_v1" and entry["block"] == "q1"
        ]
        corrected = [entry for entry in results.values() if entry["object"] == "corrected_old_v1"]
        ordinary = [entry for entry in results.values() if entry["object"] == "ordinary_v1"]
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
        corrected_holdout = vector(corrected, "holdout")
        ordinary_holdout = vector(ordinary, "holdout")
        best_holdout = (
            corrected_holdout
            if _mean(corrected_holdout) >= _mean(ordinary_holdout)
            else ordinary_holdout
        )
        q1_diff = bootstrap_diff(new_holdout, best_holdout, draws=BOOTSTRAP_DRAWS, seed=SAMPLE_SEED)

        new_query = vector(candidate, "query")
        query_corrected = vector(corrected, "query")
        query_ordinary = vector(ordinary, "query")
        best_query = (
            query_corrected if _mean(query_corrected) >= _mean(query_ordinary) else query_ordinary
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
        cost_by_object = {
            object_name: sum(
                float(runs[name]["cpu_seconds_total"])
                for name in runs
                if runs[name]["object"] == object_name and runs[name]["block"] == "q1"
            )
            for object_name in ("new_core_v1", "corrected_old_v1", "ordinary_v1")
        }
        cost_ratio = cost_by_object["new_core_v1"] / max(
            cost_by_object["corrected_old_v1"], cost_by_object["ordinary_v1"]
        )
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
            "cost_by_object": cost_by_object,
            "families": families,
            "failed_gates": failed,
        }

    total_wall = time.perf_counter() - started_wall
    total_cpu = time.process_time() - started_cpu
    report = {
        "format": "taiji-native-core-r5-evaluation-v1",
        "version": 1,
        "date": "2026-10-30",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md",
        "training_report": {
            "path": TRAINING_REPORT.relative_to(ROOT).as_posix(),
            "sha256": _sha256(TRAINING_REPORT),
            "status": training["status"],
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
            "retention_parent": "none",
            "receipt_definition": "correct holdout turns whose selected value appears verbatim in the expected bytes",
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
        "--approve-r5",
        action="store_true",
        help="required: the user approved R5 execution",
    )
    args = parser.parse_args()
    return _price(args.run_id, approve=args.approve_r5)


if __name__ == "__main__":
    raise SystemExit(main())

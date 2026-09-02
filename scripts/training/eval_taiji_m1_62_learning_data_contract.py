"""Audit whether the native episodic path can learn an auditable key/value curriculum.

M1-62 deliberately stops before designing a new key/value learner.  It first
asks a cheaper question: is the training data itself learnable, partitioned and
free of bypass shortcuts?  Two contrasting curricula are built from the
evaluator-owned :class:`MemoryLearningCurriculum` contract -- a stable
``cue_key -> value`` course with repeated observations, and a negative control
whose keys deliberately carry conflicting values.  Only ``train`` examples are
ever written; holdout and retention stay read-only.  The identity organ is run
as an upper-bound reference arm and never becomes the default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_36_cue_curriculum import _config  # noqa: E402
from scripts.training.eval_taiji_m1_53_credit_identifiability import (  # noqa: E402
    _checkpoint_record,
    _probe,
)
from taiji import (  # noqa: E402
    MEMORY_LEARNING_PARTITIONS,
    DelayedMemoryTask,
    MemoryEpisode,
    MemoryLearningCurriculum,
    MemoryLearningExample,
    Taiji,
    TaijiConfig,
)

FORMAT = "taiji-native-m1-62-learning-data-contract-v1"
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m1_62_learning_data_contract_20260902.json"
)
SEEDS = (11, 29, 47)
ACTION_SYMBOLS = (48, 49)
OUTCOME_SYMBOLS = (43, 45)
COMBINATIONS = tuple(
    (action, outcome) for action in ACTION_SYMBOLS for outcome in OUTCOME_SYMBOLS
)
KEY_COUNT = 16
TRAIN_REPEATS = 4
CUE_START = 0
HOLDOUT_KEYS = range(0, KEY_COUNT // 2)
RETENTION_KEYS = range(KEY_COUNT // 2, KEY_COUNT)
ARMS = ("native_association", "identity_organ_reference")
ACTION_CHANCE = 1.0 / len(ACTION_SYMBOLS)
OUTCOME_CHANCE = 1.0 / len(OUTCOME_SYMBOLS)
READ_CHANNELS = ("native_recall", "decision_path")


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("m1-62 summary cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _arm_config(seed: int, arm: str) -> TaijiConfig:
    if arm not in ARMS:
        raise ValueError(f"unsupported m1-62 arm: {arm}")
    values = _config(seed).to_dict()
    values["identity_organ_enabled"] = arm == "identity_organ_reference"
    return TaijiConfig.from_dict(values)


def _curriculum(name: str, *, deterministic: bool) -> MemoryLearningCurriculum:
    train: list[MemoryLearningExample] = []
    for key in range(KEY_COUNT):
        for repeat in range(TRAIN_REPEATS):
            slot = key if deterministic else key + repeat
            action, outcome = COMBINATIONS[slot % len(COMBINATIONS)]
            index = key * TRAIN_REPEATS + repeat
            train.append(
                MemoryLearningExample(
                    example_id=f"m1-62-{name}-train-{index}",
                    partition="train",
                    cue_key=CUE_START + key,
                    action_value=action,
                    outcome_value=outcome,
                    role="repeated_observation",
                    episode_tag=f"m1-62-{name}-train-{index}",
                )
            )

    def queries(partition: str, keys: range) -> tuple[MemoryLearningExample, ...]:
        items: list[MemoryLearningExample] = []
        for key in keys:
            action, outcome = COMBINATIONS[key % len(COMBINATIONS)]
            items.append(
                MemoryLearningExample(
                    example_id=f"m1-62-{name}-{partition}-{key}",
                    partition=partition,
                    cue_key=CUE_START + key,
                    action_value=action,
                    outcome_value=outcome,
                    role="one_shot_experience",
                    episode_tag=f"m1-62-{name}-{partition}-{key}",
                )
            )
        return tuple(items)

    return MemoryLearningCurriculum(
        name=name,
        train=tuple(train),
        holdout=queries("holdout", HOLDOUT_KEYS),
        retention=queries("retention", RETENTION_KEYS),
        declared_key_value_deterministic=deterministic,
    )


def _episode(example: MemoryLearningExample) -> MemoryEpisode:
    return MemoryEpisode(
        memory_id=example.example_id,
        cue=example.cue_key,
        action=example.action_value,
        outcome=example.outcome_value,
    )


def _write_train(model: Taiji, curriculum: MemoryLearningCurriculum) -> int:
    written = 0
    for example in curriculum.train:
        if example.partition != "train":
            raise RuntimeError(
                f"m1-62 refuses to write a non-train partition: {example.partition}"
            )
        DelayedMemoryTask._write_episode(
            model, _episode(example), provenance=example.provenance
        )
        written += 1
    return written


def _decision_row(model: Taiji, example: MemoryLearningExample) -> dict[str, Any]:
    model.reset_dynamics(episode_id=f"m1-62-decision-{example.example_id}")
    model.observe(
        model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=True
    )
    step = model.observe(
        example.cue_key, learn=False, learn_motor=False, use_memory=True
    )
    probabilities = step.probabilities
    alternatives = [
        float(probabilities[action].item())
        for action in ACTION_SYMBOLS
        if action != example.action_value
    ]
    probability = float(probabilities[example.action_value].item())
    return {
        "query_id": example.example_id,
        "cue": int(example.cue_key),
        "action": int(example.action_value),
        "combination": example.value_key,
        "action_probability": probability,
        "action_margin": float(probability - max(alternatives)),
        "action_correct": int(
            max(
                ACTION_SYMBOLS,
                key=lambda value: float(probabilities[value].item()),
            )
            == example.action_value
        ),
        "identity_recall_used": bool(
            step.identity_recall is not None and step.identity_recall.used
        ),
    }


def _decision_probe(
    model: Taiji, examples: tuple[MemoryLearningExample, ...]
) -> dict[str, Any]:
    rows = [_decision_row(model, example) for example in examples]
    margins = [float(row["action_margin"]) for row in rows]
    return {
        "rows": rows,
        "summary": {
            "sample_count": len(rows),
            "action_accuracy": float(
                sum(int(row["action_correct"]) for row in rows) / len(rows)
            ),
            "action_margin": _summary(margins),
            "row_action_margin_min": float(min(margins)),
            "identity_recall_used_ratio": float(
                sum(int(row["identity_recall_used"]) for row in rows) / len(rows)
            ),
        },
    }


def _query_probe(
    model: Taiji, curriculum: MemoryLearningCurriculum, partition: str
) -> dict[str, Any]:
    examples = getattr(curriculum, partition)
    for example in examples:
        if example.partition != partition:
            raise RuntimeError("m1-62 query partition mismatch")
    return _probe(
        model,
        tuple(_episode(example) for example in examples),
        ACTION_SYMBOLS,
        OUTCOME_SYMBOLS,
        None,
    )


def _arm_record(
    curriculum: MemoryLearningCurriculum, seed: int, arm: str
) -> dict[str, Any]:
    started = time.perf_counter()
    config = _arm_config(seed, arm)
    model = Taiji(config, episode_id=f"m1-62-{curriculum.name}-{arm}-{seed}")
    preflight = _checkpoint_record(model)
    preflight["active_parameter_count"] = model.parameter_count()
    preflight["planned_active_parameter_count"] = (
        model.config.planned_active_parameter_count
    )
    preflight["parameter_count_matches_plan"] = (
        model.parameter_count() == model.config.planned_active_parameter_count
    )
    preflight["saveable_before_training"] = (
        preflight["same_process_digest_matches"]
        and preflight["fresh_process_digest_matches"]
        and preflight["parameter_count_matches_plan"]
    )
    written = _write_train(model, curriculum)
    holdout = _query_probe(model, curriculum, "holdout")
    retention = _query_probe(model, curriculum, "retention")
    holdout_decision = _decision_probe(model, curriculum.holdout)
    retention_decision = _decision_probe(model, curriculum.retention)
    trained = _checkpoint_record(model)
    trained["active_parameter_count"] = model.parameter_count()
    trained["planned_active_parameter_count"] = (
        model.config.planned_active_parameter_count
    )
    trained["parameter_count_matches_plan"] = (
        model.parameter_count() == model.config.planned_active_parameter_count
    )
    return {
        "seed": seed,
        "arm": arm,
        "identity_organ_enabled": bool(config.identity_organ_enabled),
        "train_examples_written": written,
        "partitions_written": ["train"],
        "holdout_updates": 0,
        "retention_updates": 0,
        "checkpoint_preflight": preflight,
        "checkpoint_after_train": trained,
        "holdout": holdout["summary"],
        "retention": retention["summary"],
        "holdout_decision": holdout_decision["summary"],
        "retention_decision": retention_decision["summary"],
        "row_action_margin_min": {
            "holdout": float(holdout["summary"]["all"]["action_margin"]["min"]),
            "retention": float(retention["summary"]["all"]["action_margin"]["min"]),
        },
        "row_outcome_margin_min": {
            "holdout": float(holdout["summary"]["all"]["outcome_margin"]["min"]),
            "retention": float(retention["summary"]["all"]["outcome_margin"]["min"]),
        },
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _arm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for partition in ("holdout", "retention"):
        rows = [record[partition]["all"] for record in records]
        summary[partition] = {
            "action_accuracy": _summary([float(row["action_accuracy"]) for row in rows]),
            "outcome_accuracy": _summary(
                [float(row["outcome_accuracy"]) for row in rows]
            ),
            "action_margin": _summary(
                [float(row["action_margin"]["mean"]) for row in rows]
            ),
            "outcome_margin": _summary(
                [float(row["outcome_margin"]["mean"]) for row in rows]
            ),
            "action_outcome_margin_gap": _summary(
                [float(row["action_outcome_margin_gap"]["mean"]) for row in rows]
            ),
        }
    for partition in ("holdout", "retention"):
        decisions = [record[f"{partition}_decision"] for record in records]
        summary[f"{partition}_decision"] = {
            "action_accuracy": _summary(
                [float(row["action_accuracy"]) for row in decisions]
            ),
            "action_margin": _summary(
                [float(row["action_margin"]["mean"]) for row in decisions]
            ),
            "row_action_margin_min": float(
                min(float(row["row_action_margin_min"]) for row in decisions)
            ),
            "identity_recall_used_ratio": _summary(
                [float(row["identity_recall_used_ratio"]) for row in decisions]
            ),
        }
    summary["row_action_margin_min"] = {
        partition: float(
            min(float(record["row_action_margin_min"][partition]) for record in records)
        )
        for partition in ("holdout", "retention")
    }
    summary["row_outcome_margin_min"] = {
        partition: float(
            min(float(record["row_outcome_margin_min"][partition]) for record in records)
        )
        for partition in ("holdout", "retention")
    }
    summary["positive_binding_all_seeds"] = all(
        summary[partition]["action_accuracy"]["min"] > ACTION_CHANCE
        and summary[partition]["outcome_accuracy"]["min"] > OUTCOME_CHANCE
        and summary["row_action_margin_min"][partition] > 0.0
        and summary["row_outcome_margin_min"][partition] > 0.0
        for partition in ("holdout", "retention")
    )
    summary["decision_positive_binding_all_seeds"] = all(
        summary[f"{partition}_decision"]["action_accuracy"]["min"] > ACTION_CHANCE
        and summary[f"{partition}_decision"]["row_action_margin_min"] > 0.0
        for partition in ("holdout", "retention")
    )
    summary["any_positive_binding_all_seeds"] = bool(
        summary["positive_binding_all_seeds"]
        or summary["decision_positive_binding_all_seeds"]
    )
    summary["checkpoint_preflight_all_seeds"] = all(
        record["checkpoint_preflight"]["saveable_before_training"]
        for record in records
    )
    summary["checkpoint_after_train_all_seeds"] = all(
        record["checkpoint_after_train"]["same_process_digest_matches"]
        and record["checkpoint_after_train"]["fresh_process_digest_matches"]
        and record["checkpoint_after_train"]["parameter_count_matches_plan"]
        for record in records
    )
    return summary


def _data_contract_verdict(courses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stable = courses["stable_key"]["audit"]
    negative = courses["conflicting_key"]["audit"]
    return {
        "stable_well_formed": bool(stable["well_formed"]),
        "stable_key_value_deterministic": bool(stable["observed_key_value_deterministic"]),
        "negative_control_declares_conflict": (
            not negative["declared_key_value_deterministic"]
            and not negative["observed_key_value_deterministic"]
            and bool(negative["declaration_matches_observation"])
        ),
        "no_bypass_field_leak": not (
            stable["leakage"]["any_bypass_field_leak"]
            or negative["leakage"]["any_bypass_field_leak"]
        ),
        "no_train_query_id_overlap": (
            stable["leakage"]["train_query_id_overlap"] == 0
            and negative["leakage"]["train_query_id_overlap"] == 0
        ),
        "all_partitions_factorial_complete": all(
            audit["partitions"][partition]["factorial_complete"]
            for audit in (stable, negative)
            for partition in MEMORY_LEARNING_PARTITIONS
        ),
        "train_keys_repeatedly_observed": (
            stable["partitions"]["train"]["observations_per_key"]["min"]
            >= float(TRAIN_REPEATS)
        ),
    }


def run(report_path: Path, seeds: tuple[int, ...]) -> dict[str, Any]:
    started = time.perf_counter()
    curricula = {
        "stable_key": _curriculum("stable_key", deterministic=True),
        "conflicting_key": _curriculum("conflicting_key", deterministic=False),
    }
    courses: dict[str, Any] = {}
    for name, curriculum in curricula.items():
        arms: dict[str, Any] = {}
        for arm in ARMS:
            records = [_arm_record(curriculum, seed, arm) for seed in seeds]
            arms[arm] = {"records": records, "summary": _arm_summary(records)}
        courses[name] = {
            "curriculum_digest": curriculum.digest,
            "audit": curriculum.audit(),
            "arms": arms,
        }

    data_contract = _data_contract_verdict(courses)
    native = courses["stable_key"]["arms"]["native_association"]["summary"]
    reference = courses["stable_key"]["arms"]["identity_organ_reference"]["summary"]
    negative_native = courses["conflicting_key"]["arms"]["native_association"]["summary"]
    negative_reference = courses["conflicting_key"]["arms"][
        "identity_organ_reference"
    ]["summary"]
    gates = {
        "data_contract_pass": all(data_contract.values()),
        "checkpoint_preflight_pass": all(
            course["arms"][arm]["summary"]["checkpoint_preflight_all_seeds"]
            for course in courses.values()
            for arm in ARMS
        ),
        "checkpoint_after_train_pass": all(
            course["arms"][arm]["summary"]["checkpoint_after_train_all_seeds"]
            for course in courses.values()
            for arm in ARMS
        ),
        "native_positive_binding_on_stable_keys": bool(
            native["any_positive_binding_all_seeds"]
        ),
        "native_stable_beats_conflicting_control": bool(
            native["holdout"]["action_accuracy"]["mean"]
            > negative_native["holdout"]["action_accuracy"]["mean"]
            and native["holdout_decision"]["action_accuracy"]["mean"]
            > negative_native["holdout_decision"]["action_accuracy"]["mean"]
            and native["row_action_margin_min"]["holdout"]
            > negative_native["row_action_margin_min"]["holdout"]
        ),
        "negative_control_rejected": bool(
            not negative_native["any_positive_binding_all_seeds"]
            and not negative_reference["any_positive_binding_all_seeds"]
        ),
        "identity_organ_reference_positive_binding": bool(
            reference["any_positive_binding_all_seeds"]
        ),
        "identity_organ_reference_beats_native": bool(
            reference["holdout_decision"]["action_accuracy"]["min"]
            > native["holdout_decision"]["action_accuracy"]["max"]
            and reference["holdout_decision"]["row_action_margin_min"]
            > native["holdout_decision"]["row_action_margin_min"]
        ),
    }
    blocking = [
        name
        for name in (
            "data_contract_pass",
            "checkpoint_preflight_pass",
            "checkpoint_after_train_pass",
            "negative_control_rejected",
            "native_positive_binding_on_stable_keys",
            "native_stable_beats_conflicting_control",
        )
        if not gates[name]
    ]
    result = {
        "format": FORMAT,
        "status": "gate_passed" if not blocking else "blocked",
        "blocking_gates": blocking,
        "seeds": list(seeds),
        "data_contract": data_contract,
        "gates": gates,
        "courses": courses,
        "prohibited_partitions_used": [],
        "holdout_updates": 0,
        "retention_updates": 0,
        "trace_only": True,
        "default_training_changed": False,
        "default_checkpoint_changed": False,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args()
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    result = run(args.report, seeds)
    print(
        json.dumps(
            {
                "format": result["format"],
                "status": result["status"],
                "report_path": str(args.report),
                "blocking_gates": result["blocking_gates"],
                "gates": result["gates"],
                "stable_key_digest": result["courses"]["stable_key"][
                    "curriculum_digest"
                ],
                "conflicting_key_digest": result["courses"]["conflicting_key"][
                    "curriculum_digest"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

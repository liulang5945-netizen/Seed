"""Audit F4 child checkpoints against the M0 promotion contract.

This command is intentionally a promotion audit, not another training entry
point.  It restores F4 checkpoints in a fresh process, runs their read-only
joint canary, and reports which M0 sample and ability gates are still missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_taiji_joint import (  # noqa: E402
    build_goal_corpus,
    build_memory_corpus,
    build_world_corpus,
)
from taiji import (  # noqa: E402
    FoundationManifest,
    FoundationTrainingDataset,
    JointTrainingRun,
)
from taiji.foundation_training import _code_revision  # noqa: E402

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_foundation_baseline_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_f5_promotion_20260901.json"
PROMOTION_FORMAT = "taiji-native-f5-promotion-v1"
PROMOTION_VERSION = 1


def _checkpoint_pair(value: list[str]) -> tuple[int, Path]:
    if len(value) != 2:
        raise ValueError("--checkpoint requires SEED PATH")
    seed = int(value[0])
    if seed <= 0:
        raise ValueError("checkpoint seed must be positive")
    path = Path(value[1])
    if not path.is_file():
        raise FileNotFoundError(path)
    return seed, path


def _coverage(
    manifest: FoundationManifest,
    *,
    dataset: FoundationTrainingDataset,
    memory_count: int,
    world_count: int,
    goal_count: int,
) -> dict[str, Any]:
    actual = {
        "b1_sequence_prediction": dataset.sample_counts,
        "b2_delayed_memory": {
            "train": int(memory_count),
            "holdout": int(memory_count),
            "retention": int(memory_count),
        },
        "b3_world_transition": {
            "train": int(world_count),
            "holdout": max(1, int(world_count) // 2),
            "retention": max(1, int(world_count) // 2),
        },
        "b4_goal_action": {
            "train": int(goal_count),
            "holdout": max(1, int(goal_count) // 2),
            "retention": max(1, int(goal_count) // 2),
        },
        # F4 does not contain an ordered phase-A -> phase-B continuation
        # record.  Zero is deliberate: it prevents a joint score from being
        # mistaken for continual-learning evidence.
        "b5_continual_learning": {"train": 0, "holdout": 0, "retention": 0},
    }
    required = {
        task.ability_id: {
            "train": task.minimum_train_units,
            "holdout": task.minimum_holdout_units,
            "retention": task.minimum_retention_units,
        }
        for task in manifest.tasks
    }
    records: dict[str, Any] = {}
    for ability_id, minimums in required.items():
        observed = actual[ability_id]
        records[ability_id] = {
            "observed": observed,
            "required": minimums,
            "meets_requirement": all(observed[key] >= minimums[key] for key in minimums),
        }
    return records


def _metric_comparison(
    parent: dict[str, float], final: dict[str, float]
) -> dict[str, bool]:
    return {
        "sequence_holdout_improved": final["sequence_holdout_bpb"] < parent["sequence_holdout_bpb"],
        "sequence_retention_improved": final["sequence_retention_bpb"]
        < parent["sequence_retention_bpb"],
        "memory_holdout_improved": final["memory_holdout_recall"] > parent["memory_holdout_recall"],
        "memory_retention_improved": final["memory_retention_recall"]
        > parent["memory_retention_recall"],
        "world_holdout_improved": final["world_holdout_error"] < parent["world_holdout_error"],
        "world_retention_improved": final["world_retention_error"]
        < parent["world_retention_error"],
        "goal_holdout_improved": final["goal_holdout_success"] > parent["goal_holdout_success"],
        "goal_retention_improved": final["goal_retention_success"] > parent["goal_retention_success"],
    }


def build_promotion_report(
    *,
    manifest: FoundationManifest,
    dataset: FoundationTrainingDataset,
    memory_count: int,
    world_count: int,
    goal_count: int,
    checkpoints: list[tuple[int, Path]],
) -> dict[str, Any]:
    memory_corpus = build_memory_corpus(count=memory_count)
    world_corpus = build_world_corpus(count=world_count)
    goal_corpus = build_goal_corpus(count=goal_count)
    checkpoint_canaries: list[dict[str, Any]] = []
    for seed, path in checkpoints:
        run = JointTrainingRun.from_checkpoint(
            path,
            dataset,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=path.parent,
            epochs=1,
        )
        evaluation = run.evaluate_only()
        final = {key: float(value) for key, value in evaluation["metrics"].items()}
        parent = {key: float(value) for key, value in run.parent_metrics.items()}
        checkpoint_canaries.append(
            {
                "seed": seed,
                "checkpoint": str(path),
                "checkpoint_digest": evaluation["checkpoint_digest"],
                "parent_metrics": parent,
                "final_metrics": final,
                "metric_comparison": _metric_comparison(parent, final),
                "checkpoint_read_only": bool(evaluation["checkpoint_read_only"]),
            }
        )

    coverage = _coverage(
        manifest,
        dataset=dataset,
        memory_count=memory_count,
        world_count=world_count,
        goal_count=goal_count,
    )
    coverage_missing = [
        f"{ability_id}:{partition}"
        for ability_id, record in coverage.items()
        if not record["meets_requirement"]
        for partition in ("train", "holdout", "retention")
        if record["observed"][partition] < record["required"][partition]
    ]
    metric_keys = tuple(
        key
        for canary in checkpoint_canaries
        for key, passed in canary["metric_comparison"].items()
        if not passed
    )
    failure_reasons = list(dict.fromkeys(coverage_missing))
    if "b5_continual_learning:train" in failure_reasons:
        failure_reasons.append("b5_continual_learning:not_evaluated_from_f4_checkpoint")
    if metric_keys:
        failure_reasons.append("joint_metric_regression")
    if not all(canary["checkpoint_read_only"] for canary in checkpoint_canaries):
        failure_reasons.append("checkpoint_read_only_failed")
    failure_reasons.append("m0_controls_not_recomputed_for_f4_child")
    return {
        "format": PROMOTION_FORMAT,
        "version": PROMOTION_VERSION,
        "status": "blocked" if failure_reasons else "passed",
        "can_promote": not failure_reasons,
        "manifest_digest": manifest.digest,
        "code_revision": _code_revision(),
        "dataset_digest": dataset.digest,
        "dataset_profile": dataset.profile,
        "checkpoint_count": len(checkpoint_canaries),
        "partition_seed": dataset.partition_seed,
        "coverage": coverage,
        "checkpoint_canaries": checkpoint_canaries,
        "failure_reasons": failure_reasons,
        "next_gate": "run full-coverage M0 five-ability matrix from a fresh continuation protocol",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", type=Path, default=[DEFAULT_CORPUS])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--partition-seed", type=int, default=11)
    parser.add_argument("--memory-count", type=int, default=64)
    parser.add_argument("--world-count", type=int, default=64)
    parser.add_argument("--goal-count", type=int, default=64)
    parser.add_argument("--checkpoint", action="append", nargs=2, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if any(int(value) <= 0 for value in (args.memory_count, args.world_count, args.goal_count)):
        raise ValueError("corpus counts must be positive")
    manifest = FoundationManifest.load(args.manifest)
    dataset = FoundationTrainingDataset.from_jsonl(
        args.corpus,
        profile="pilot",
        partition_seed=args.partition_seed,
    )
    checkpoints = [_checkpoint_pair(value) for value in args.checkpoint]
    seeds = [seed for seed, _path in checkpoints]
    if len(set(seeds)) != len(seeds):
        raise ValueError("checkpoint seeds must be unique")
    result = build_promotion_report(
        manifest=manifest,
        dataset=dataset,
        memory_count=args.memory_count,
        world_count=args.world_count,
        goal_count=args.goal_count,
        checkpoints=checkpoints,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(args.report)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

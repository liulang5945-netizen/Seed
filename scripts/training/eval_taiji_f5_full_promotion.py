"""Aggregate the M1 F5 full-coverage continuation Gate.

This is an audit-only command.  It does not train, mutate checkpoints, or
claim promotion from the joint score alone.  It validates the three full
continuations, measures old phase-A retention on the replay corpus, and keeps
controls that were not recomputed explicitly visible in the report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

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
    Taiji,
    TaijiConfig,
)
from taiji.foundation_training import (  # noqa: E402
    _code_revision,
    _world_learner_from_payload,
)
from taiji.internalization import content_digest  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_foundation_baseline_v1.json"
DEFAULT_PHASE_A_CORPUS = Path("data") / "simple_zh" / "dialogue_extended_clean.jsonl"
DEFAULT_FOUNDATION_CORPUS = Path("data") / "simple_zh" / "shared_core.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_f5_full_promotion_20260902.json"
PROMOTION_FORMAT = "taiji-native-f5-full-promotion-v1"
PROMOTION_VERSION = 1
SEEDS = (11, 29, 47)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    expected = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    if str(payload.get("checkpoint_digest", "")) != expected:
        raise ValueError(f"checkpoint digest mismatch: {path}")
    return dict(payload)


def _evaluate_checkpoint(
    path: Path,
    dataset: FoundationTrainingDataset,
    memory_corpus: Any,
    world_corpus: Any,
    goal_corpus: Any,
) -> dict[str, Any]:
    payload = _load_payload(path)
    model_payload = payload.get("model")
    learner_payload = payload.get("world_learner")
    parent_metrics = payload.get("parent_metrics")
    if not isinstance(model_payload, Mapping):
        raise ValueError(f"checkpoint is missing model payload: {path}")
    if not isinstance(learner_payload, Mapping):
        raise ValueError(f"checkpoint is missing world learner payload: {path}")
    if not isinstance(parent_metrics, Mapping):
        raise ValueError(f"checkpoint is missing parent metrics: {path}")

    model = Taiji(
        TaijiConfig.from_dict(dict(model_payload["config"])),
        episode_id=f"f5-audit-{path.stem}",
    )
    model.restore(dict(model_payload))
    run = JointTrainingRun(
        model,
        _world_learner_from_payload(learner_payload),
        dataset,
        memory_corpus,
        world_corpus,
        goal_corpus,
        output_dir=path.parent,
        model_tier=str(payload.get("model_tier", "joint")),
        epochs=1,
        chunk_bytes=int(payload["chunk_bytes"]),
        checkpoint_interval=int(payload["checkpoint_interval"]),
        metric_interval=int(payload.get("metric_interval", payload["checkpoint_interval"])),
        world_learning_rate=float(payload["world_learning_rate"]),
        world_repeats=int(payload["world_repeats"]),
        parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
        parent_metrics={key: float(value) for key, value in parent_metrics.items()},
        code_revision=str(payload.get("code_revision", _code_revision())),
    )
    result = run.evaluate_only()
    result["payload_checkpoint_digest"] = str(payload["checkpoint_digest"])
    result["checkpoint_path"] = str(path)
    return result


def _metric_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return set(left) == set(right) and all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-12)
        for key in left
    )


def _phase_a_transfer(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, Any]:
    transfer = {
        "sequence_holdout_bpb": float(before["sequence_holdout_bpb"])
        - float(after["sequence_holdout_bpb"]),
        "memory_holdout_recall": float(after["memory_holdout_recall"])
        - float(before["memory_holdout_recall"]),
        "world_holdout_error": float(before["world_holdout_error"])
        - float(after["world_holdout_error"]),
        "goal_holdout_success": float(after["goal_holdout_success"])
        - float(before["goal_holdout_success"]),
    }
    retained = all(value >= 0.0 for value in transfer.values())
    return {
        "before": dict(before),
        "after": dict(after),
        "transfer": transfer,
        "all_old_metrics_retained_or_improved": retained,
        "backward_transfer_score": min(transfer.values()),
    }


def _coverage(manifest: FoundationManifest, dataset: FoundationTrainingDataset) -> dict[str, Any]:
    actual = {
        "b1_sequence_prediction": dataset.sample_counts,
        "b2_delayed_memory": {"train": 1_000, "holdout": 1_000, "retention": 1_000},
        "b3_world_transition": {"train": 1_000, "holdout": 500, "retention": 500},
        "b4_goal_action": {"train": 1_000, "holdout": 500, "retention": 500},
    }
    required = {
        task.ability_id: {
            "train": task.minimum_train_units,
            "holdout": task.minimum_holdout_units,
            "retention": task.minimum_retention_units,
        }
        for task in manifest.tasks
        if task.ability_id != "b5_continual_learning"
    }
    records: dict[str, Any] = {}
    for ability_id, minimums in required.items():
        observed = actual[ability_id]
        records[ability_id] = {
            "observed": observed,
            "required": minimums,
            "meets_requirement": all(observed[key] >= minimums[key] for key in minimums),
        }
    records["b5_continual_learning"] = {
        "observed": {
            "phase_b_train": 1_000,
            "replay_bytes": 16_384,
            "dedicated_holdout_units": 0,
            "dedicated_retention_units": 0,
        },
        "required": {
            "train": manifest.task("b5_continual_learning").minimum_train_units,
            "holdout": manifest.task("b5_continual_learning").minimum_holdout_units,
            "retention": manifest.task("b5_continual_learning").minimum_retention_units,
        },
        "meets_requirement": False,
        "note": "F5 replay is real, but B5 dedicated task-unit partitions are not registered.",
    }
    return records


def build_full_promotion_report(
    *,
    manifest: FoundationManifest,
    phase_a_dataset: FoundationTrainingDataset,
    foundation_dataset: FoundationTrainingDataset,
    training_reports: Mapping[int, Path],
    eval_reports: Mapping[int, Path],
    source_checkpoints: Mapping[int, Path],
    final_checkpoints: Mapping[int, Path],
) -> dict[str, Any]:
    memory_corpus = build_memory_corpus(count=64)
    world_corpus = build_world_corpus(count=64)
    goal_corpus = build_goal_corpus(count=64)
    coverage = _coverage(manifest, foundation_dataset)
    seed_records: list[dict[str, Any]] = []
    failure_reasons: list[str] = []

    for seed in SEEDS:
        training = json.loads(training_reports[seed].read_text(encoding="utf-8"))
        independent = json.loads(eval_reports[seed].read_text(encoding="utf-8"))
        source_eval = _evaluate_checkpoint(
            source_checkpoints[seed], phase_a_dataset, memory_corpus, world_corpus, goal_corpus
        )
        final_eval = _evaluate_checkpoint(
            final_checkpoints[seed], phase_a_dataset, memory_corpus, world_corpus, goal_corpus
        )
        metrics_match = _metric_equal(training["final_metrics"], independent["metrics"])
        checkpoint_match = str(training["child_checkpoint_digest"]) == str(
            independent["checkpoint_digest"]
        )
        source_lineage_match = str(training["continuation_source_checkpoint_digest"]) == str(
            _load_payload(source_checkpoints[seed])["checkpoint_digest"]
        )
        replay_ok = (
            int(training.get("replay_epochs", 0)) >= 1
            and str(training.get("replay_dataset_digest")) == phase_a_dataset.digest
            and any(
                item.get("train_kind") == "replay"
                and int(item.get("replay_cursor", 0)) > 0
                for item in training.get("history", [])
            )
        )
        transfer = _phase_a_transfer(source_eval["metrics"], final_eval["metrics"])
        checks = {
            "training_completed": training.get("status") == "completed",
            "eval_only_completed": independent.get("status") == "evaluated",
            "checkpoint_digest_matches_eval": checkpoint_match,
            "metrics_match_eval": metrics_match,
            "checkpoint_read_only": bool(independent.get("checkpoint_read_only")),
            "source_lineage_matches": source_lineage_match,
            "holdout_updates_zero": int(training.get("holdout_updates", -1)) == 0,
            "world_transition_rejections_zero": int(
                training.get("world_transition_rejections", -1)
            )
            == 0,
            "replay_executed": replay_ok,
            "phase_a_old_metrics_retained": bool(
                transfer["all_old_metrics_retained_or_improved"]
            ),
        }
        seed_records.append(
            {
                "seed": seed,
                "training_report": str(training_reports[seed]),
                "eval_report": str(eval_reports[seed]),
                "training_final_metrics": training["final_metrics"],
                "eval_metrics": independent["metrics"],
                "checkpoint_digest": independent["checkpoint_digest"],
                "source_checkpoint_digest": _load_payload(source_checkpoints[seed])["checkpoint_digest"],
                "phase_a_backward_transfer": transfer,
                "checks": checks,
            }
        )
        failure_reasons.extend(
            f"seed_{seed}:{name}" for name, passed in checks.items() if not passed
        )

    if not all(record["meets_requirement"] for record in coverage.values()):
        failure_reasons.append("foundation_coverage_gate_failed")
    failure_reasons.extend(
        [
            "b5_dedicated_task_holdout_and_retention_not_registered",
            "m0_controls_not_recomputed_for_full_child",
            "missing_controls:random,simple_rule,hash_only",
            "no_replay_counterfactual_for_causal_replay_gain",
        ]
    )
    return {
        "format": PROMOTION_FORMAT,
        "version": PROMOTION_VERSION,
        "status": "blocked" if failure_reasons else "passed",
        "can_promote": not failure_reasons,
        "manifest_digest": manifest.digest,
        "dataset_digest": phase_a_dataset.digest,
        "dataset_sample_counts": phase_a_dataset.sample_counts,
        "phase_a_replay_dataset_digest": phase_a_dataset.digest,
        "code_revision": _code_revision(),
        "seeds": seed_records,
        "coverage": coverage,
        "controls": {
            "recomputed": ["frozen_parent", "credit_lesion"],
            "missing": ["random", "simple_rule", "hash_only", "full_m0_task_matrix"],
        },
        "failure_reasons": list(dict.fromkeys(failure_reasons)),
        "next_gate": (
            "repair memory retention/replay target, register a dedicated B5 task-unit split, "
            "and rerun full-child M0 controls before M1 promotion"
        ),
    }


def _parse_seed_path(value: list[str]) -> tuple[int, Path]:
    if len(value) != 2:
        raise ValueError("each --training-report/--eval-report/--source/--final needs SEED PATH")
    seed = int(value[0])
    if seed not in SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    path = Path(value[1])
    if not path.is_file():
        raise FileNotFoundError(path)
    return seed, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase-a-corpus", nargs="+", type=Path, default=[DEFAULT_PHASE_A_CORPUS])
    parser.add_argument(
        "--foundation-corpus", nargs="+", type=Path, default=[DEFAULT_FOUNDATION_CORPUS]
    )
    parser.add_argument("--phase-a-partition-seed", type=int, default=11)
    parser.add_argument("--training-report", action="append", nargs=2, required=True)
    parser.add_argument("--eval-report", action="append", nargs=2, required=True)
    parser.add_argument("--source", action="append", nargs=2, required=True)
    parser.add_argument("--final", dest="final_checkpoints", action="append", nargs=2, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    def indexed(values: list[list[str]]) -> dict[int, Path]:
        pairs = [_parse_seed_path(value) for value in values]
        result = {seed: path for seed, path in pairs}
        if set(result) != set(SEEDS):
            raise ValueError(f"expected exactly one path for seeds {SEEDS}")
        return result

    manifest = FoundationManifest.load(args.manifest)
    phase_a_dataset = FoundationTrainingDataset.from_jsonl(
        args.phase_a_corpus,
        profile="pilot",
        partition_seed=args.phase_a_partition_seed,
    )
    foundation_dataset = FoundationTrainingDataset.from_jsonl(
        args.foundation_corpus,
        profile="foundation",
        partition_seed=args.phase_a_partition_seed,
    )
    result = build_full_promotion_report(
        manifest=manifest,
        phase_a_dataset=phase_a_dataset,
        foundation_dataset=foundation_dataset,
        training_reports=indexed(args.training_report),
        eval_reports=indexed(args.eval_report),
        source_checkpoints=indexed(args.source),
        final_checkpoints=indexed(args.final_checkpoints),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(args.report)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["can_promote"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

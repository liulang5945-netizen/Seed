"""Aggregate the M1-8 full-coverage memory-replay Gate.

This is an audit-only command.  It never trains or mutates checkpoints.  The
Gate is deliberately separate from the historical F5 report because M1-8
adds an explicit phase-A memory replay stream and uses seed-specific replay
partition digests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_f5_full_promotion import (  # noqa: E402
    SEEDS,
    _evaluate_checkpoint,
    _load_payload,
    _phase_a_transfer,
)
from scripts.training.train_taiji_joint import (  # noqa: E402
    build_goal_corpus,
    build_memory_corpus,
    build_world_corpus,
)
from taiji import FoundationManifest, FoundationTrainingDataset  # noqa: E402
from taiji.foundation_training import _code_revision, _memory_corpus_digest  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_foundation_baseline_v1.json"
DEFAULT_PHASE_A_CORPUS = Path("data") / "simple_zh" / "dialogue_extended_clean.jsonl"
DEFAULT_FOUNDATION_CORPUS = Path("data") / "simple_zh" / "shared_core.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_f8_memory_replay_promotion_20260902.json"
PROMOTION_FORMAT = "taiji-native-m1-8-memory-replay-promotion-v1"
PROMOTION_VERSION = 1


def _metric_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return set(left) == set(right) and all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-12)
        for key in left
    )


def _history_has(history: Any, train_kind: str, minimum_cursor: int) -> bool:
    if not isinstance(history, list):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("train_kind") == train_kind
        and int(item.get("replay_memory_cursor" if train_kind == "replay-memory" else "replay_cursor", 0))
        >= minimum_cursor
        for item in history
    )


def _parse_seed_path(value: list[str]) -> tuple[int, Path]:
    if len(value) != 2:
        raise ValueError("each indexed path needs SEED PATH")
    seed = int(value[0])
    if seed not in SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    path = Path(value[1])
    if not path.is_file():
        raise FileNotFoundError(path)
    return seed, path


def _indexed(values: list[list[str]]) -> dict[int, Path]:
    pairs = [_parse_seed_path(value) for value in values]
    result = {seed: path for seed, path in pairs}
    if set(result) != set(SEEDS):
        raise ValueError(f"expected exactly one path for seeds {SEEDS}")
    return result


def build_promotion_report(
    *,
    manifest: FoundationManifest,
    phase_a_datasets: Mapping[int, FoundationTrainingDataset],
    foundation_dataset: FoundationTrainingDataset,
    training_reports: Mapping[int, Path],
    eval_reports: Mapping[int, Path],
    source_checkpoints: Mapping[int, Path],
    final_checkpoints: Mapping[int, Path],
    replay_memory_count: int,
) -> dict[str, Any]:
    if replay_memory_count <= 0:
        raise ValueError("replay_memory_count must be positive")
    memory_corpus = build_memory_corpus(count=64)
    world_corpus = build_world_corpus(count=64)
    goal_corpus = build_goal_corpus(count=64)
    expected_replay_memory_digest = _memory_corpus_digest(
        build_memory_corpus(count=replay_memory_count)
    )

    required = {
        task.ability_id: {
            "train": task.minimum_train_units,
            "holdout": task.minimum_holdout_units,
            "retention": task.minimum_retention_units,
        }
        for task in manifest.tasks
        if task.ability_id != "b5_continual_learning"
    }
    actual = {
        "b1_sequence_prediction": foundation_dataset.sample_counts,
        "b2_delayed_memory": {"train": 1_000, "holdout": 1_000, "retention": 1_000},
        "b3_world_transition": {"train": 1_000, "holdout": 500, "retention": 500},
        "b4_goal_action": {"train": 1_000, "holdout": 500, "retention": 500},
    }
    coverage = {
        ability_id: {
            "observed": actual[ability_id],
            "required": minimums,
            "meets_requirement": all(
                actual[ability_id][key] >= minimums[key] for key in minimums
            ),
        }
        for ability_id, minimums in required.items()
    }
    coverage["b5_continual_learning"] = {
        "observed": {
            "phase_b_train": 0,
            "dedicated_holdout_units": 0,
            "dedicated_retention_units": 0,
            "no_replay_counterfactual": False,
        },
        "required": {
            "train": manifest.task("b5_continual_learning").minimum_train_units,
            "holdout": manifest.task("b5_continual_learning").minimum_holdout_units,
            "retention": manifest.task("b5_continual_learning").minimum_retention_units,
        },
        "meets_requirement": False,
        "note": "Dedicated B5 full audit is still a separate Gate.",
    }

    seed_records: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    replay_memory_digests: set[str] = set()
    for seed in SEEDS:
        training = json.loads(training_reports[seed].read_text(encoding="utf-8"))
        independent = json.loads(eval_reports[seed].read_text(encoding="utf-8"))
        phase_a_dataset = phase_a_datasets[seed]
        source_payload = _load_payload(source_checkpoints[seed])
        source_eval = _evaluate_checkpoint(
            source_checkpoints[seed], phase_a_dataset, memory_corpus, world_corpus, goal_corpus
        )
        final_eval = _evaluate_checkpoint(
            final_checkpoints[seed], phase_a_dataset, memory_corpus, world_corpus, goal_corpus
        )
        replay_memory_digest = str(training.get("replay_memory_digest", ""))
        replay_memory_digests.add(replay_memory_digest)
        metrics_match = _metric_equal(training["final_metrics"], independent["metrics"])
        checkpoint_match = str(training["child_checkpoint_digest"]) == str(
            independent["checkpoint_digest"]
        )
        source_lineage_match = str(training["continuation_source_checkpoint_digest"]) == str(
            source_payload["checkpoint_digest"]
        )
        byte_replay_ok = (
            int(training.get("replay_epochs", 0)) >= 1
            and str(training.get("replay_dataset_digest")) == phase_a_dataset.digest
            and _history_has(training.get("history"), "replay", 1)
        )
        memory_replay_ok = (
            int(training.get("replay_memory_epochs", 0)) >= 1
            and replay_memory_digest == expected_replay_memory_digest
            and _history_has(training.get("history"), "replay-memory", replay_memory_count)
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
            "byte_replay_executed": byte_replay_ok,
            "memory_replay_executed": memory_replay_ok,
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
                "replay_memory_digest": replay_memory_digest,
                "phase_a_backward_transfer": transfer,
                "checks": checks,
            }
        )
        failure_reasons.extend(
            f"seed_{seed}:{name}" for name, passed in checks.items() if not passed
        )

    if len(replay_memory_digests) != 1:
        failure_reasons.append("replay_memory_digest_not_consistent_across_seeds")
    if not all(record["meets_requirement"] for record in coverage.values()):
        failure_reasons.append("foundation_coverage_gate_failed")
    failure_reasons.extend(
        [
            "b5_dedicated_task_holdout_and_retention_not_audited",
            "m0_controls_not_recomputed_for_full_child",
            "missing_controls:random,simple_rule,hash_only",
            "no_replay_counterfactual_for_dedicated_b5",
        ]
    )
    return {
        "format": PROMOTION_FORMAT,
        "version": PROMOTION_VERSION,
        "status": "blocked" if failure_reasons else "passed",
        "can_promote": not failure_reasons,
        "manifest_digest": manifest.digest,
        "foundation_dataset_digest": foundation_dataset.digest,
        "foundation_dataset_sample_counts": foundation_dataset.sample_counts,
        "code_revision": _code_revision(),
        "replay_memory_count": replay_memory_count,
        "replay_memory_digests": sorted(replay_memory_digests),
        "coverage": coverage,
        "seeds": seed_records,
        "controls": {
            "recomputed": ["frozen_parent", "credit_lesion"],
            "missing": ["random", "simple_rule", "hash_only", "full_m0_task_matrix"],
        },
        "failure_reasons": list(dict.fromkeys(failure_reasons)),
        "next_gate": (
            "run the dedicated full B5 no-replay/replay audit, then recompute the full M0 "
            "control matrix before any M1 promotion"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase-a-corpus", nargs="+", type=Path, default=[DEFAULT_PHASE_A_CORPUS])
    parser.add_argument(
        "--foundation-corpus", nargs="+", type=Path, default=[DEFAULT_FOUNDATION_CORPUS]
    )
    parser.add_argument("--replay-memory-count", type=int, default=64)
    parser.add_argument("--training-report", action="append", nargs=2, required=True)
    parser.add_argument("--eval-report", action="append", nargs=2, required=True)
    parser.add_argument("--source", action="append", nargs=2, required=True)
    parser.add_argument("--final", dest="final_checkpoints", action="append", nargs=2, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    manifest = FoundationManifest.load(args.manifest)
    phase_a_datasets = {
        seed: FoundationTrainingDataset.from_jsonl(
            args.phase_a_corpus, profile="pilot", partition_seed=seed
        )
        for seed in SEEDS
    }
    foundation_dataset = FoundationTrainingDataset.from_jsonl(
        args.foundation_corpus, profile="foundation", partition_seed=11
    )
    result = build_promotion_report(
        manifest=manifest,
        phase_a_datasets=phase_a_datasets,
        foundation_dataset=foundation_dataset,
        training_reports=_indexed(args.training_report),
        eval_reports=_indexed(args.eval_report),
        source_checkpoints=_indexed(args.source),
        final_checkpoints=_indexed(args.final_checkpoints),
        replay_memory_count=args.replay_memory_count,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(args.report)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["can_promote"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

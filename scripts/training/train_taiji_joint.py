"""Run the native Taiji F4 joint short-training course."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_taiji_memory import build_corpus as build_memory_corpus  # noqa: E402
from scripts.training.train_taiji_world_action import (  # noqa: E402
    build_goal_corpus,
    build_world_corpus,
    build_world_learner,
)
from taiji import (  # noqa: E402
    FoundationTrainingDataset,
    JointTrainingRun,
    Taiji,
    TaijiConfig,
)


def _config(seed: int) -> TaijiConfig:
    values = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_meta_dim=32,
        memory_readout_fan_in=32,
        memory_iterations=3,
    ).to_dict()
    values["seed"] = int(seed)
    return TaijiConfig.from_dict(values)


def _cold_start_action_organ(model: Taiji) -> None:
    with torch.no_grad():
        model.motor.synapses.edge_weight.zero_()
        model.motor.bias.zero_()
        model.motor.reward_baseline = 0.0
        model.motor.reward_updates = 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"],
    )
    parser.add_argument("--profile", choices=("smoke", "pilot", "foundation"), default="smoke")
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--partition-seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--chunk-bytes", type=int, default=1_024)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--metric-interval", type=int)
    parser.add_argument("--world-learning-rate", type=float, default=0.02)
    parser.add_argument("--world-repeats", type=int, default=8)
    parser.add_argument("--replay-corpus", nargs="+", type=Path)
    parser.add_argument("--replay-profile", choices=("smoke", "pilot", "foundation"), default="pilot")
    parser.add_argument("--replay-partition-seed", type=int, default=11)
    parser.add_argument("--replay-epochs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "taiji-m1-f4-joint",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--continue-from", dest="continue_from", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.resume is not None and args.continue_from is not None:
        parser.error("--resume and --continue-from are mutually exclusive")

    count = args.count if args.count is not None else {
        "smoke": 8,
        "pilot": 64,
        "foundation": 1_000,
    }[args.profile]
    checkpoint_interval = args.checkpoint_interval or {
        "smoke": 4,
        "pilot": 16,
        "foundation": 256,
    }[args.profile]
    dataset = FoundationTrainingDataset.from_jsonl(
        args.corpus,
        profile=args.profile,
        partition_seed=args.partition_seed,
    )
    replay_dataset = None
    if args.replay_corpus is not None:
        replay_dataset = FoundationTrainingDataset.from_jsonl(
            args.replay_corpus,
            profile=args.replay_profile,
            partition_seed=args.replay_partition_seed,
        )
    memory_corpus = build_memory_corpus(count=count)
    world_corpus = build_world_corpus(count=count)
    goal_corpus = build_goal_corpus(count=count)
    if args.continue_from is not None:
        run = JointTrainingRun.from_continuation_checkpoint(
            args.continue_from,
            dataset,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=args.output_dir,
            epochs=args.epochs,
            chunk_bytes=args.chunk_bytes,
            checkpoint_interval=checkpoint_interval,
            metric_interval=args.metric_interval,
            world_learning_rate=args.world_learning_rate,
            world_repeats=args.world_repeats,
            replay_dataset=replay_dataset,
            replay_epochs=args.replay_epochs,
        )
    elif args.resume is not None:
        run = JointTrainingRun.from_checkpoint(
            args.resume,
            dataset,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=args.output_dir,
            epochs=args.epochs,
            metric_interval=args.metric_interval,
            replay_dataset=replay_dataset,
            replay_epochs=args.replay_epochs,
        )
    else:
        model = Taiji(_config(args.seed), episode_id="joint-train")
        _cold_start_action_organ(model)
        run = JointTrainingRun(
            model,
            build_world_learner(world_corpus, seed=args.seed),
            dataset,
            memory_corpus,
            world_corpus,
            goal_corpus,
            output_dir=args.output_dir,
            model_tier="joint",
            epochs=args.epochs,
            chunk_bytes=args.chunk_bytes,
            checkpoint_interval=checkpoint_interval,
            metric_interval=args.metric_interval,
            world_learning_rate=args.world_learning_rate,
            world_repeats=args.world_repeats,
            replay_dataset=replay_dataset,
            replay_epochs=args.replay_epochs,
        )
    result: dict[str, Any]
    if args.eval_only:
        result = run.evaluate_only()
    else:
        result = run.run()
    report_path = args.report or args.output_dir / (
        "eval_report.json" if args.eval_only else "training_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

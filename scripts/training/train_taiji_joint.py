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

from scripts.training.eval_taiji_m1_64_foundation_memory import (  # noqa: E402
    build_foundation_delayed_memory_corpus,
)
from scripts.training.train_taiji_memory import build_corpus as build_memory_corpus  # noqa: E402
from scripts.training.train_taiji_world_action import (  # noqa: E402
    build_goal_corpus,
    build_world_corpus,
    build_world_learner,
)
from taiji import (  # noqa: E402
    JOINT_TRAINING_PHASES,
    DelayedMemoryCorpus,
    FoundationTrainingDataset,
    JointTrainingRun,
    Taiji,
    TaijiConfig,
)
from taiji.foundation_training import _code_revision  # noqa: E402


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


def _build_memory_course(course: str, *, count: int) -> DelayedMemoryCorpus:
    if course == "generic":
        return build_memory_corpus(count=count)
    if course == "foundation":
        if int(count) != 1_000:
            raise ValueError("the formal B2 memory course requires count=1000")
        return build_foundation_delayed_memory_corpus()
    raise ValueError(f"unsupported joint memory course: {course}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        "--phase-b-corpus",
        dest="corpus",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"],
    )
    parser.add_argument(
        "--profile",
        "--phase-b-profile",
        dest="profile",
        choices=("smoke", "pilot", "foundation"),
        default="smoke",
    )
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--partition-seed",
        "--phase-b-partition-seed",
        dest="partition_seed",
        type=int,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--chunk-bytes", type=int, default=1_024)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--metric-interval", type=int)
    parser.add_argument("--world-learning-rate", type=float, default=0.02)
    parser.add_argument("--world-repeats", type=int, default=8)
    parser.add_argument(
        "--memory-course",
        choices=("generic", "foundation"),
        default="generic",
        help=(
            "Select the content-addressed memory course. 'foundation' matches the formal "
            "B2 delayed/interference evaluator and requires --count 1000."
        ),
    )
    parser.add_argument(
        "--identity-growth-to",
        type=int,
        help=(
            "Append identity-organ slots before a continuation course. Existing slots "
            "and learned values are preserved; requires --continue-from."
        ),
    )
    parser.add_argument("--protected-corpus", nargs="+", type=Path)
    parser.add_argument(
        "--protected-profile",
        choices=("smoke", "pilot", "foundation"),
    )
    parser.add_argument("--protected-partition-seed", type=int)
    parser.add_argument(
        "--training-phases",
        nargs="+",
        choices=JOINT_TRAINING_PHASES,
        help=(
            "Explicit ordered course plan. M2 F5 requires --protected-corpus and "
            "must start with the sequence-only no-replay counterfactual."
        ),
    )
    parser.add_argument(
        "--freeze-sequence-fabric",
        action="store_true",
        help=(
            "Train F1's dedicated predictive readout and private temporal context "
            "while preserving shared fabric parameters; requires a sequence phase."
        ),
    )
    parser.add_argument("--replay-corpus", nargs="+", type=Path)
    parser.add_argument(
        "--replay-profile", choices=("smoke", "pilot", "foundation"), default="pilot"
    )
    parser.add_argument("--replay-partition-seed", type=int, default=11)
    parser.add_argument("--replay-epochs", type=int, default=1)
    parser.add_argument(
        "--replay-memory-count",
        type=int,
        help="Exact replay count; must equal the current memory course count.",
    )
    parser.add_argument("--replay-memory-epochs", type=int, default=1)
    parser.add_argument("--replay-memory-learning-scale", type=float)
    parser.add_argument(
        "--replay-memory-learning-targets",
        choices=("all", "association", "readout", "action_readout", "outcome_readout"),
        default="all",
    )
    parser.add_argument(
        "--memory-confidence-decay",
        type=float,
        default=0.0,
        help="Legacy lifetime confidence decay; native training defaults to 0.",
    )
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
    if args.identity_growth_to is not None and args.continue_from is None:
        parser.error("--identity-growth-to requires --continue-from")
    if args.identity_growth_to is not None and args.eval_only:
        parser.error("--identity-growth-to cannot be combined with --eval-only")
    if args.freeze_sequence_fabric and (
        args.training_phases is None or "sequence" not in args.training_phases
    ):
        parser.error("--freeze-sequence-fabric requires a sequence training phase")

    # ``None`` means preserve a resumed/continued course's stored semantics;
    # only the explicit flag creates the M2-2g predictor-only intervention.
    sequence_fabric_learning = False if args.freeze_sequence_fabric else None

    count = (
        args.count
        if args.count is not None
        else {
            "smoke": 8,
            "pilot": 64,
            "foundation": 1_000,
        }[args.profile]
    )
    checkpoint_interval = (
        args.checkpoint_interval
        or {
            "smoke": 4,
            "pilot": 16,
            "foundation": 256,
        }[args.profile]
    )
    if args.replay_memory_count is not None and args.replay_memory_count != count:
        parser.error(
            "--replay-memory-count must equal the current memory course count; "
            "partial or unrelated replay corpora are not accepted"
        )
    if args.protected_corpus is not None:
        if args.protected_profile is None or args.protected_partition_seed is None:
            parser.error(
                "--protected-corpus requires --protected-profile and " "--protected-partition-seed"
            )
        if args.partition_seed is None:
            parser.error("M2 F5 phase-B construction requires --phase-b-partition-seed")
        if args.training_phases is None:
            parser.error("M2 F5 requires an explicit --training-phases plan; begin with sequence")
        if args.replay_corpus is not None:
            parser.error(
                "with --protected-corpus, replay is the exact protected phase-A course; "
                "do not pass --replay-corpus"
            )
        protected_dataset = FoundationTrainingDataset.from_jsonl(
            args.protected_corpus,
            profile=args.protected_profile,
            partition_seed=args.protected_partition_seed,
        )
        dataset = FoundationTrainingDataset.from_jsonl(
            args.corpus,
            profile=args.profile,
            partition_seed=args.partition_seed,
            exclude_dataset=protected_dataset,
        )
        replay_dataset = protected_dataset if "replay" in args.training_phases else None
    else:
        protected_dataset = None
        dataset = FoundationTrainingDataset.from_jsonl(
            args.corpus,
            profile=args.profile,
            partition_seed=(args.partition_seed if args.partition_seed is not None else 11),
        )
        replay_dataset = None
        if args.replay_corpus is not None:
            replay_dataset = FoundationTrainingDataset.from_jsonl(
                args.replay_corpus,
                profile=args.replay_profile,
                partition_seed=args.replay_partition_seed,
            )
    if (
        args.training_phases is not None
        and "replay" in args.training_phases
        and replay_dataset is None
    ):
        parser.error("the replay phase requires --protected-corpus or --replay-corpus")
    try:
        memory_corpus = _build_memory_course(args.memory_course, count=count)
    except ValueError as exc:
        parser.error(str(exc))
    replay_memory_corpus = (
        _build_memory_course(args.memory_course, count=count)
        if args.replay_memory_count is not None
        else None
    )
    if (
        args.training_phases is not None
        and replay_memory_corpus is not None
        and "replay-memory" not in args.training_phases
    ):
        parser.error("--replay-memory-count requires the replay-memory phase")
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
            protected_dataset=protected_dataset,
            training_phases=args.training_phases,
            sequence_fabric_learning=sequence_fabric_learning,
            replay_dataset=replay_dataset,
            replay_epochs=args.replay_epochs,
            replay_memory_corpus=replay_memory_corpus,
            replay_memory_epochs=args.replay_memory_epochs,
            replay_memory_learning_scale=args.replay_memory_learning_scale,
            replay_memory_learning_targets=args.replay_memory_learning_targets,
            memory_confidence_decay=args.memory_confidence_decay,
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
            protected_dataset=protected_dataset,
            training_phases=args.training_phases,
            sequence_fabric_learning=sequence_fabric_learning,
            replay_dataset=replay_dataset,
            replay_epochs=args.replay_epochs,
            replay_memory_corpus=replay_memory_corpus,
            replay_memory_epochs=args.replay_memory_epochs,
            replay_memory_learning_scale=args.replay_memory_learning_scale,
            replay_memory_learning_targets=args.replay_memory_learning_targets,
            code_revision=_code_revision(),
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
            protected_dataset=protected_dataset,
            training_phases=args.training_phases,
            sequence_fabric_learning=(
                True if sequence_fabric_learning is None else sequence_fabric_learning
            ),
            replay_dataset=replay_dataset,
            replay_epochs=args.replay_epochs,
            replay_memory_corpus=replay_memory_corpus,
            replay_memory_epochs=args.replay_memory_epochs,
            replay_memory_learning_scale=args.replay_memory_learning_scale,
            replay_memory_learning_targets=args.replay_memory_learning_targets,
        )
    if args.identity_growth_to is not None:
        run.model.grow_identity_organ(args.identity_growth_to)
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
    result["memory_course"] = args.memory_course
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

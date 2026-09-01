"""Run the native Taiji world-transition and goal-action pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    GoalActionCorpus,
    GoalActionEpisode,
    Outcome,
    Taiji,
    TaijiConfig,
    WorldAction,
    WorldActionTrainingRun,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldObject,
    WorldSchema,
    WorldState,
    WorldTransitionCorpus,
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


def _world_case(case_id: str, position: float) -> WorldInterventionCase:
    before = WorldState(
        tick=0,
        latent=torch.zeros(1),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("target", attributes={"position": position}),
        ),
    )
    action = WorldAction(
        action_id=case_id,
        kind="push",
        tick=0,
        actor_id="agent",
        target_id="target",
        parameters={"amount": 1.0},
    )
    after = WorldState(
        tick=1,
        latent=torch.zeros(1),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("target", attributes={"position": position + 1.0}),
        ),
    )
    return WorldInterventionCase(
        case_id=case_id,
        initial=before,
        action=action,
        expected_state=after,
        expected_outcome=Outcome(
            intent_id=case_id,
            reward=1.0,
            success=True,
            tick=1,
        ),
    )


def build_world_corpus(*, count: int) -> WorldTransitionCorpus:
    if int(count) < 4:
        raise ValueError("F3 world corpus needs at least four transitions")
    return WorldTransitionCorpus(
        train=tuple(
            _world_case(f"m1-f3-world-train-{index}", float(index))
            for index in range(int(count))
        ),
        holdout=tuple(
            _world_case(f"m1-f3-world-holdout-{index}", float(count + index))
            for index in range(max(4, int(count) // 2))
        ),
        retention=tuple(
            _world_case(f"m1-f3-world-retention-{index}", float(2 * count + index))
            for index in range(max(4, int(count) // 2))
        ),
    )


def build_goal_corpus(*, count: int) -> GoalActionCorpus:
    if int(count) < 4 or int(count) % 2:
        raise ValueError("F3 goal corpus needs an even count of at least four episodes")

    def episode(prefix: str, index: int) -> GoalActionEpisode:
        return GoalActionEpisode(
            episode_id=f"m1-f3-goal-{prefix}-{index}",
            cue=65 + index % 2,
            preferred_action=48 + index % 2,
            alternate_action=49 - index % 2,
        )

    return GoalActionCorpus(
        train=tuple(episode("train", index) for index in range(int(count))),
        holdout=tuple(episode("holdout", index) for index in range(int(count) // 2)),
        retention=tuple(episode("retention", index) for index in range(int(count) // 2)),
    )


def build_world_learner(corpus: WorldTransitionCorpus, *, seed: int):
    all_cases = (*corpus.train, *corpus.holdout, *corpus.retention)
    schema = WorldSchema.from_corpus(
        WorldInterventionCorpus(train=all_cases, holdout=())
    )
    from taiji import WorldDynamicsLearner

    return WorldDynamicsLearner(schema, hidden_dim=32, seed=seed)


def _cold_start_action_organ(model: Taiji) -> None:
    with torch.no_grad():
        model.motor.synapses.edge_weight.zero_()
        model.motor.bias.zero_()
        model.motor.reward_baseline = 0.0
        model.motor.reward_updates = 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--world-learning-rate", type=float, default=0.02)
    parser.add_argument("--world-repeats", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "taiji-m1-f3-world-action",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    count = args.count if args.count is not None else (8 if args.profile == "smoke" else 64)
    checkpoint_interval = args.checkpoint_interval or (4 if args.profile == "smoke" else 16)
    world_corpus = build_world_corpus(count=count)
    goal_corpus = build_goal_corpus(count=count)
    if args.resume is not None:
        run = WorldActionTrainingRun.from_checkpoint(
            args.resume,
            world_corpus,
            goal_corpus,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
    else:
        model = Taiji(_config(args.seed), episode_id="world-action-train")
        _cold_start_action_organ(model)
        run = WorldActionTrainingRun(
            model,
            build_world_learner(world_corpus, seed=args.seed),
            world_corpus,
            goal_corpus,
            output_dir=args.output_dir,
            model_tier="world-action",
            epochs=args.epochs,
            checkpoint_interval=checkpoint_interval,
            world_learning_rate=args.world_learning_rate,
            world_repeats=args.world_repeats,
        )
    result = run.evaluate_only() if args.eval_only else run.run()
    report_path = args.report or args.output_dir / (
        "eval_report.json" if args.eval_only else "training_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(report_path)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

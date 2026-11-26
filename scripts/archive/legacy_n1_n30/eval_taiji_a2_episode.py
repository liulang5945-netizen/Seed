"""Evaluate multi-step Taiji world episodes and checkpoint recovery."""

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
    Outcome,
    WorldAction,
    WorldEpisode,
    WorldEpisodeCorpus,
    WorldEpisodeEvaluationConfig,
    WorldEpisodeEvaluator,
    WorldObject,
    WorldState,
    WorldTransition,
)

MANIFEST_FORMAT = "taiji-a2-world-episode-shift-v1"


def _initial() -> WorldState:
    return WorldState(
        tick=0,
        latent=torch.zeros(1),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": 0.0}),
            WorldObject("blue", attributes={"position": 0.0}),
        ),
    )


def _step(state: WorldState, target: str, amount: float, index: int) -> WorldTransition:
    updated = {obj.object_id: dict(obj.attributes) for obj in state.objects}
    updated[target]["position"] += amount
    after = WorldState(
        tick=state.tick + 1,
        latent=state.latent,
        objects=tuple(
            WorldObject(obj.object_id, attributes=updated[obj.object_id], tags=obj.tags)
            for obj in state.objects
        ),
    )
    action = WorldAction(
        action_id=f"move-{index}",
        kind="move",
        tick=state.tick,
        actor_id="agent",
        target_id=target,
        parameters={"step": amount},
    )
    return WorldTransition(
        before=state,
        action=action,
        after=after,
        outcome=Outcome(
            intent_id=action.action_id,
            reward=amount,
            success=amount > 0.0,
            tick=after.tick,
        ),
    )


def _episode(episode_id: str, steps: tuple[tuple[str, float], ...]) -> WorldEpisode:
    state = _initial()
    transitions = []
    for index, (target, amount) in enumerate(steps):
        transition = _step(state, target, amount, index)
        transitions.append(transition)
        state = transition.after
    return WorldEpisode(
        episode_id=episode_id, initial=transitions[0].before, transitions=tuple(transitions)
    )


def build_corpus() -> WorldEpisodeCorpus:
    return WorldEpisodeCorpus(
        train=(
            _episode("train-0", (("red", 1.0), ("blue", -1.0))),
            _episode("train-1", (("red", -1.0), ("blue", 1.0))),
            _episode("train-2", (("blue", 1.0), ("red", 1.0))),
        ),
        holdout=(
            _episode("unseen-episode-a", (("red", 1.0), ("blue", 1.0))),
            _episode("unseen-episode-b", (("blue", -1.0), ("red", -1.0))),
        ),
    )


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "objects": ["agent", "red", "blue"],
        "dynamics": "each move adds step to target.position and emits reward=step",
        "train": [
            {"episode_id": "train-0", "steps": [["red", 1.0], ["blue", -1.0]]},
            {"episode_id": "train-1", "steps": [["red", -1.0], ["blue", 1.0]]},
            {"episode_id": "train-2", "steps": [["blue", 1.0], ["red", 1.0]]},
        ],
        "holdout": [
            {"episode_id": "unseen-episode-a", "steps": [["red", 1.0], ["blue", 1.0]]},
            {"episode_id": "unseen-episode-b", "steps": [["blue", -1.0], ["red", -1.0]]},
        ],
        "constraints": [
            "episode IDs are disjoint and never enter the learned schema",
            "every transition tick is contiguous",
            "checkpoint is restored after the first transition before continuing",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_a2_episode_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_a2_episode_baseline_20260825.json",
    )
    args = parser.parse_args()

    report = WorldEpisodeEvaluator(
        WorldEpisodeEvaluationConfig(seeds=(11, 29, 47), hidden_dim=32, epochs=350)
    ).evaluate(build_corpus())
    report["benchmark"] = {
        "format": MANIFEST_FORMAT,
        "manifest": str(args.manifest.relative_to(PROJECT_ROOT)),
        "purpose": "multi-step rollout, cross-episode object persistence and checkpoint continuation",
    }
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["gate"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

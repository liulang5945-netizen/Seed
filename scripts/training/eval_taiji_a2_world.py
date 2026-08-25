"""Build and evaluate the first Taiji world-intervention benchmark."""

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
    WorldEvent,
    WorldInterventionCase,
    WorldInterventionCorpus,
    WorldInterventionEvaluationConfig,
    WorldInterventionEvaluator,
    WorldObject,
    WorldState,
)

MANIFEST_FORMAT = "taiji-a2-world-shift-v1"


def _state() -> WorldState:
    return WorldState(
        tick=0,
        latent=torch.zeros(2),
        objects=(
            WorldObject("agent", attributes={"energy": 1.0}),
            WorldObject("red", attributes={"position": 0.0}),
            WorldObject("blue", attributes={"position": 0.0}),
        ),
        events=(
            WorldEvent("context-a", "context", 0, subject_id="agent"),
            WorldEvent("context-b", "context", 0, subject_id="agent"),
        ),
    )


def _case(
    target: str,
    step: float,
    index: int,
    *,
    initial: WorldState | None = None,
) -> WorldInterventionCase:
    initial = _state() if initial is None else initial
    updated = {obj.object_id: dict(obj.attributes) for obj in initial.objects}
    updated[target]["position"] += step
    expected = WorldState(
        tick=1,
        latent=initial.latent,
        relations=initial.relations,
        objects=tuple(
            WorldObject(obj.object_id, attributes=updated[obj.object_id], tags=obj.tags)
            for obj in initial.objects
        ),
        events=initial.events,
        affordances=initial.affordances,
        uncertainty=initial.uncertainty,
    )
    action_id = f"move-{index}"
    action = WorldAction(
        action_id=action_id,
        kind="move",
        tick=0,
        actor_id="agent",
        target_id=target,
        parameters={"step": step},
    )
    return WorldInterventionCase(
        case_id=f"case-{index}",
        initial=initial,
        action=action,
        expected_state=expected,
        expected_outcome=Outcome(
            intent_id=action_id,
            reward=step,
            success=step > 0.0,
            tick=1,
        ),
    )


def _relation_case(target: str, enabled: float, index: int) -> WorldInterventionCase:
    initial = _state()
    relation = (target, "near", "blue" if target == "red" else "red")
    expected = WorldState(
        tick=1,
        latent=initial.latent,
        relations=(relation,) if enabled > 0.0 else (),
        objects=initial.objects,
        events=initial.events,
        affordances=initial.affordances,
        uncertainty=initial.uncertainty,
    )
    action_id = f"link-{index}"
    action = WorldAction(
        action_id=action_id,
        kind="link",
        tick=0,
        actor_id="agent",
        target_id=target,
        parameters={"enabled": enabled},
    )
    return WorldInterventionCase(
        case_id=f"relation-case-{index}",
        initial=initial,
        action=action,
        expected_state=expected,
        expected_outcome=Outcome(
            intent_id=action_id,
            reward=1.0 if enabled > 0.0 else -1.0,
            success=enabled > 0.0,
            tick=1,
        ),
    )


def build_corpus() -> WorldInterventionCorpus:
    train = []
    index = 0
    for target in ("red", "blue"):
        for step in (-2.0, -1.0, 1.0, 2.0):
            if (target, step) in (("red", 2.0), ("blue", -2.0)):
                continue
            train.append(_case(target, step, index))
            index += 1
    relation_train = []
    relation_index = 0
    for target in ("red", "blue"):
        for enabled in (-1.0, 1.0):
            relation_train.append(_relation_case(target, enabled, relation_index))
            relation_index += 1
    train.extend(relation_train)
    holdout = (
        _case("red", 2.0, 100),
        _case("blue", -2.0, 101),
        _relation_case("red", 2.0, 102),
        _relation_case("blue", -2.0, 103),
    )
    base = _state()
    shuffled = WorldState(
        tick=base.tick,
        latent=base.latent,
        relations=base.relations,
        objects=base.objects,
        events=tuple(reversed(base.events)),
        affordances=base.affordances,
        uncertainty=base.uncertainty,
    )
    time_shuffled = (_case("red", 2.0, 200, initial=shuffled),)
    return WorldInterventionCorpus(
        train=tuple(train),
        holdout=holdout,
        time_shuffled=time_shuffled,
    )


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "objects": ["agent", "red", "blue"],
        "state_attributes": {"agent": ["energy"], "red": ["position"], "blue": ["position"]},
        "action": {
            "kinds": ["move", "link"],
            "actor": "agent",
            "parameters": ["step", "enabled"],
            "intervention_rules": [
                "move: target.position += step",
                "link: add target.near.other_target when enabled > 0",
            ],
            "outcome_rules": [
                "move: reward = step; success = step > 0",
                "link: reward = 1 when enabled > 0 else -1",
            ],
        },
        "train": [
            {"target": target, "step": step}
            for target in ("red", "blue")
            for step in (-2.0, -1.0, 1.0, 2.0)
            if (target, step) not in (("red", 2.0), ("blue", -2.0))
        ],
        "holdout": [
            {"target": "red", "step": 2.0},
            {"target": "blue", "step": -2.0},
            {"target": "red", "enabled": 2.0},
            {"target": "blue", "enabled": -2.0},
        ],
        "time_shuffled": [
            {"target": "red", "step": 2.0, "event_order": "reversed"},
        ],
        "split_constraint": "holdout is unseen target-by-step composition, not a new object vocabulary",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_a2_world_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_a2_world_baseline_20260825.json",
    )
    args = parser.parse_args()

    corpus = build_corpus()
    report = WorldInterventionEvaluator(
        WorldInterventionEvaluationConfig(
            seeds=(11, 29, 47),
            hidden_dim=32,
            epochs=350,
            learning_rate=0.01,
        )
    ).evaluate(corpus)
    report["benchmark"] = {
        "format": MANIFEST_FORMAT,
        "manifest": str(args.manifest.relative_to(PROJECT_ROOT)),
        "purpose": "one-step structured state and action-outcome intervention prediction",
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

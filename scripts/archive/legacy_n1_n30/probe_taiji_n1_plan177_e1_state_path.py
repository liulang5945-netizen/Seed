"""Read-only probe: can the trained N177.6 producer parse E1 facts/queries?

This loads the frozen N177.6 correct/seed0 endpoint with its verified train
trace cache, switches the trainer to raw Adapter replay, then runs one public
E1 train episode through the real producer. It never loads the E1 holdout,
never calls the optimizer and never writes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
)
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
REPORT = ROOT / "reports/taiji_n1_plan177_e1_state_path_20261029.json"


def _pick_episode(entity: str, value: str):
    from taiji.joint_state_consumption_course import build_train_episodes, e1_families

    family = next(
        item for item in e1_families()["train"] if item.entity == entity and item.value == value
    )
    episode = next(
        item
        for item in build_train_episodes()
        if item.family_id == family.family_id and item.episode_id.endswith(":0")
    )
    return family, episode


def _parse_payload(parse) -> dict:
    return {
        "act": parse.act,
        "status": parse.status,
        "confidence": parse.confidence,
        "ambiguity": parse.ambiguity,
        "spans": [
            {
                "role": span.role,
                "text": span.text,
                "start_byte": span.start_byte,
                "end_byte": span.end_byte,
                "confidence": span.confidence,
            }
            for span in parse.spans
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe E1 state path on trained producer")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--trace-cache", type=Path, default=TRACE_CACHE)
    parser.add_argument("--entity", default="笔")
    parser.add_argument("--value", default="红色")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise ValueError("E1 state-path probe report is write-once")
    if not args.checkpoint.is_file() or not args.trace_cache.is_file():
        raise FileNotFoundError("probe requires the N177.6 endpoint and its train trace cache")

    from taiji.joint_state_consumption_course import e1_turn_text
    from taiji.joint_t1_trace_cache import JointT1TraceCacheV1
    from taiji.joint_t1_trainer import JointT1TrainerV1

    trace_cache = JointT1TraceCacheV1.from_payload(
        torch.load(args.trace_cache, map_location="cpu", weights_only=False)
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=trace_cache)
    trainer.input_trace_cache = None
    family, episode = _pick_episode(args.entity, args.value)
    adapter, joint = trainer._new_episode_state(episode.episode_id)

    cpu_start = time.process_time()
    wall_start = time.monotonic()
    turns = []
    with torch.no_grad():
        for turn in episode.turns:
            text, _ = e1_turn_text(turn)
            trace, joint = trainer._append_turn(adapter, joint, turn)
            parse = trainer.producer.predict(joint, trace)
            committed = False
            try:
                joint = trainer.producer.commit_to_joint_state(joint, trace, parse, trainer.model)
                committed = True
            except Exception as error:  # noqa: BLE001 - probe records the refusal verbatim
                turns.append(
                    {
                        "turn_index": turn.turn_index,
                        "text": text,
                        "parse": _parse_payload(parse),
                        "commit_error": type(error).__name__,
                        "commit_error_text": str(error),
                        "committed": False,
                    }
                )
                continue
            turns.append(
                {
                    "turn_index": turn.turn_index,
                    "text": text,
                    "parse": _parse_payload(parse),
                    "committed": committed,
                    "world_objects": len(joint.cognitive_state.world.objects),
                    "world_events": len(joint.cognitive_state.world.events),
                }
            )
    report = {
        "schema": "taiji-n177-e1-state-path-probe-v1",
        "status": "completed",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "trace_cache": str(args.trace_cache.relative_to(ROOT)),
        "family": {
            "family_id": family.family_id,
            "entity": family.entity,
            "value": family.value,
            "old_value": family.old_value,
        },
        "episode_id": episode.episode_id,
        "turns": turns,
        "model_calls": len(turns),
        "optimizer_updates": 0,
        "sealed_text_read": 0,
        "can_promote": False,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.monotonic() - wall_start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "episode": episode.episode_id}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

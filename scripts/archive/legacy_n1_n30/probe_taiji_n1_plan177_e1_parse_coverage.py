"""N177 E1/F 前置：量一遍 N177.6 producer 在 E1 train 家族上的解析覆盖率（只读）。

判据只有一个：对该家族的 train `:0` episode，producer 是否给出**恰好一个**实体/属性/值
各一段的事实，且该实体的当前值等于课程真值。逐家族登记成功/失败与解析出的值，
既给总数也给点名清单——覆盖率低时 F0b 的臂间差异会被"没解析出东西"主导。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.joint_state_consumption_course import (  # noqa: E402
    E1_ATTRIBUTE,
    build_train_episodes,
    e1_families,
)

CHECKPOINT = (
    ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
)
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
REPORT = ROOT / "reports/taiji_n1_plan177_e1_parse_coverage_20261029.json"


def parse_episode(trainer: Any, episode: Any) -> dict[str, Any]:
    adapter, joint = trainer._new_episode_state(episode.episode_id)
    facts: list[tuple[str, str, str, int]] = []
    turns = 0
    with torch.no_grad():
        for turn in episode.turns:
            turns += 1
            trace, joint = trainer._append_turn(adapter, joint, turn)
            parse = trainer.producer.predict(joint, trace)
            by_role: dict[str, list[str]] = {"entity": [], "attribute": [], "value": []}
            for span in getattr(parse, "spans", ()):
                role = getattr(span, "role", None)
                if role in by_role:
                    by_role[role].append(str(getattr(span, "text", "")))
            if all(len(by_role[role]) == 1 for role in by_role):
                facts.append(
                    (
                        by_role["entity"][0],
                        by_role["attribute"][0],
                        by_role["value"][0],
                        turn.turn_index,
                    )
                )
    current: dict[str, tuple[str, int]] = {}
    for entity, attribute, value, turn_index in facts:
        if attribute == E1_ATTRIBUTE:
            current[entity] = (value, turn_index)
    return {"turns": turns, "facts": facts, "current": current}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177 E1 train 解析覆盖率（只读）")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--trace-cache", type=Path, default=TRACE_CACHE)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.report.exists():
        raise ValueError("parse coverage report is write-once")

    from taiji.joint_t1_trace_cache import JointT1TraceCacheV1
    from taiji.joint_t1_trainer import JointT1TrainerV1

    cache = JointT1TraceCacheV1.from_payload(
        torch.load(args.trace_cache, map_location="cpu", weights_only=False)
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=cache)
    trainer.input_trace_cache = None
    episodes = {(item.family_id, item.episode_id): item for item in build_train_episodes()}
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    rows = []
    for family in e1_families()["train"]:
        episode = next(
            (
                item
                for (family_id, _), item in episodes.items()
                if family_id == family.family_id and item.episode_id.endswith(":0")
            ),
            None,
        )
        if episode is None:
            rows.append(
                {
                    "family_id": family.family_id,
                    "entity": family.entity,
                    "true_value": family.value,
                    "episode_id": None,
                    "parsed_value": None,
                    "covered": False,
                    "reason": "no :0 episode built",
                }
            )
            continue
        result = parse_episode(trainer, episode)
        parsed = result["current"].get(family.entity)
        rows.append(
            {
                "family_id": family.family_id,
                "entity": family.entity,
                "true_value": family.value,
                "episode_id": episode.episode_id,
                "turns": result["turns"],
                "single_role_facts": len(result["facts"]),
                "parsed_value": None if parsed is None else parsed[0],
                "parsed_turn": None if parsed is None else parsed[1],
                "covered": parsed is not None and parsed[0] == family.value,
            }
        )
    covered = [row for row in rows if row["covered"]]
    uncovered = [row for row in rows if not row["covered"]]
    report = {
        "format": "taiji-n177-e1-parse-coverage-v1",
        "date": "2026-10-29",
        "scope": "32 train families, one :0 episode each; read-only, no optimizer step",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "trace_cache": str(args.trace_cache.relative_to(ROOT)).replace("\\", "/"),
        "families": len(rows),
        "covered": len(covered),
        "coverage_rate": len(covered) / len(rows) if rows else 0.0,
        "covered_families": [row["family_id"] for row in covered],
        "uncovered_families": [
            {
                "family_id": row["family_id"],
                "entity": row["entity"],
                "true_value": row["true_value"],
                "parsed_value": row["parsed_value"],
                "single_role_facts": row["single_role_facts"],
            }
            for row in uncovered
        ],
        "rows": rows,
        "verdict": (
            "f0b_blocked_by_parse_coverage"
            if len(covered) < len(rows)
            else "parse_coverage_complete"
        ),
        "cost": {
            "process_cpu_seconds": time.process_time() - cpu_start,
            "wall_seconds": time.perf_counter() - wall_start,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    print(
        f"covered={report['covered']}/{report['families']} "
        f"rate={report['coverage_rate']:.3f} verdict={report['verdict']}"
    )
    print(f"cost={report['cost']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

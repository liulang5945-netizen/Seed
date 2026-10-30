"""N178 测价：parser 损失单训（加权采样）的单价与覆盖率方向（只读交付，5,000 CPU 秒上限）。

只做三件事：
1. 用 N177.6 的 correct/seed0 端点作父，按"含旧值轮的 episode、未覆盖实体的 episode"加权采样；
2. 跑 256 步（批 2）只对 parser 损失反传、只更新 producer，逐 step 记 CPU/wall；
3. 训练前后各量一次四格覆盖率（旧值/普通 × 已覆盖/未覆盖实体）。

不碰 V4 语言头、不读 sealed、不写耐久 checkpoint；触顶写 partial。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.evaluate_taiji_n1_plan177_f_f1_holdout import _replay  # noqa: E402
from taiji.joint_input_producer import ACT_NAMES  # noqa: E402
from taiji.joint_state_consumption_course import build_train_episodes, e1_families  # noqa: E402
from taiji.joint_t1_trace_cache import JointT1TraceCacheV1  # noqa: E402
from taiji.joint_t1_trainer import JointT1TrainerV1  # noqa: E402

PARENT = ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
REPORT = ROOT / "reports/taiji_n1_plan178_price_20261030.json"
COVERED = ("钥匙", "杯子", "书", "盒子")
CPU_CAP = 5000.0
UPDATES = 256
BATCH = 2


def _trainer() -> JointT1TrainerV1:
    cache = JointT1TraceCacheV1.from_payload(
        torch.load(TRACE_CACHE, map_location="cpu", weights_only=False)
    )
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=cache)
    # E1 课程不在 N177.6 的 trace cache 里：走原样 Adapter 重放，与 _replay 一致。
    trainer.input_trace_cache = None
    return trainer


def _weights(episodes: list[Any], entity_by_family: dict[str, str]) -> list[float]:
    weights = []
    for episode in episodes:
        text = "".join(
            bytes(turn.frame.payload).decode("utf-8", "replace") for turn in episode.turns
        )
        weight = 1.0
        if ("之前" in text) or ("以前" in text):
            weight += 2.0
        if entity_by_family[episode.family_id] not in COVERED:
            weight += 2.0
        weights.append(weight)
    return weights


def _coverage(
    trainer: JointT1TrainerV1, episodes: list[Any], entity_by_family: dict[str, str]
) -> dict[str, Any]:
    cells: collections.Counter = collections.Counter()
    query_act = ACT_NAMES.index("query")
    for episode in episodes:
        entity = entity_by_family[episode.family_id]
        rows = _replay(trainer, episode)
        for row in rows:
            # 分母锚在**课程标注的 act** 上，不用模型自己的 act 预测——否则模型把更多轮
            # 判成 query 就会把分母一起改大（实测 349→384），覆盖率随之虚高。
            if int(row["turn"].semantic_target.act_id) != query_act:
                continue
            text = bytes(row["turn"].frame.payload).decode("utf-8", "replace")
            old = ("之前" in text) or ("以前" in text)
            # 查询轮本就没有值 span；覆盖率按"能否给出实体"判，与解析缺口分解同口径。
            parsed = bool(row["spans"].get("entity"))
            cells[
                (
                    ("old" if old else "ordinary"),
                    ("covered" if entity in COVERED else "uncovered"),
                    parsed,
                )
            ] += 1
    out: dict[str, Any] = {}
    for template in ("old", "ordinary"):
        for coverage in ("covered", "uncovered"):
            hits = cells[(template, coverage, True)]
            total = hits + cells[(template, coverage, False)]
            out[f"{template}_{coverage}"] = {"parsed": hits, "total": total}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N178 parser 加压测价")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.report.exists():
        raise ValueError("N178 pricing report is write-once")

    torch.manual_seed(0)
    entity_by_family = {family.family_id: family.entity for family in e1_families()["train"]}
    episodes = list(build_train_episodes())
    weights = torch.tensor(_weights(episodes, entity_by_family), dtype=torch.float64)
    weights = weights / weights.sum()

    trainer = _trainer()
    cpu_start = time.process_time()
    coverage_before = _coverage(trainer, episodes, entity_by_family)
    baseline_total = sum(cell["total"] for cell in coverage_before.values())
    steps: list[dict[str, float]] = []
    status = "completed_updates"
    for step in range(args.updates):
        if time.process_time() - cpu_start >= CPU_CAP:
            status = "cpu_cap"
            break
        picks = torch.multinomial(weights, BATCH, replacement=True).tolist()
        step_cpu = time.process_time()
        parser_terms = []
        for index in picks:
            episode = episodes[index]
            adapter, joint = trainer._new_episode_state(episode.episode_id)
            turn_losses = []
            for turn in episode.turns:
                trace, joint = trainer._append_turn(adapter, joint, turn)
                semantic_loss, _ = trainer._semantic_loss(joint, trace, turn)
                turn_losses.append(semantic_loss)
            parser_terms.append(torch.stack(turn_losses).mean())
        loss = torch.stack(parser_terms).mean()
        trainer.producer_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.producer_optimizer.step()
        trainer.language_trainer.optimizer.zero_grad(set_to_none=True)
        steps.append(
            {
                "step": step + 1,
                "episodes": picks,
                "parser_loss": float(loss.detach()),
                "process_cpu_seconds": time.process_time() - step_cpu,
            }
        )
    endpoint = ROOT / "data/checkpoints/n1_plan178_price_20261030" / "parser_pressure_256.pt"
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.checkpoint(), endpoint)
    coverage_after = _coverage(trainer, episodes, entity_by_family)
    after_total = sum(cell["total"] for cell in coverage_after.values())
    cpu_total = time.process_time() - cpu_start
    per_update = sum(row["process_cpu_seconds"] for row in steps) / len(steps) if steps else None
    report = {
        "format": "taiji-n178-price-v1",
        "date": "2026-10-30",
        "scope": "parser-only loss on the existing train course, weighted sampling, no V4 update, no sealed read",
        "parent": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        "updates": len(steps),
        "batch": BATCH,
        "status": status,
        "cpu_cap": CPU_CAP,
        "cost": {
            "process_cpu_seconds": cpu_total,
            "cpu_per_update": per_update,
        },
        "first_parser_loss": steps[0]["parser_loss"] if steps else None,
        "last_parser_loss": steps[-1]["parser_loss"] if steps else None,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "endpoint": str(endpoint.relative_to(ROOT)).replace("\\", "/"),
        "coverage_denominators_stable": after_total == baseline_total,
        "coverage_denominator_before": baseline_total,
        "coverage_denominator_after": after_total,
        "projection": (
            None
            if per_update is None
            else {
                "one_arm_1280_updates_cpu": per_update * 1280,
                "with_1_3_margin": per_update * 1280 * 1.3,
            }
        ),
    }
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    print(
        json.dumps(
            {
                "updates": len(steps),
                "cpu_total": round(cpu_total, 1),
                "cpu_per_update": None if per_update is None else round(per_update, 2),
                "loss_first_last": [report["first_parser_loss"], report["last_parser_loss"]],
                "coverage_before": coverage_before,
                "coverage_after": coverage_after,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

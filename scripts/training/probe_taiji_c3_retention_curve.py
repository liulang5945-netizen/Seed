"""C3 保持时长曲线：产品基座上"间隔 N 步还能不能复述"（零训练，现成量具）。

预注册：`PLAN-A-24` §2e C3 行（"递归痕迹隔多少步还能复述？饱和点在哪？"）。
量具＝`taiji/foundation_tasks.py` 的 `DelayedMemoryTask` 写入/召回协议——
`_write_episode`（context→act→settle→outcome，产品 act/settle 写入路径）与
`_recall_accuracy`（reset→喂 context+cue→motor_probabilities 上对动作类 argmax）
**原样复用**，不重抄；新变的只有两件：跑在产品基座（`seed_beta.pt`，不是同 config 新建），
以及把"写入与查询之间的间隔 N"变成扫描变量。

**两条曲线**（对应"保持"在架构里的两个真实机制）：
1. **年龄曲线**（查询前 reset）：写入后先观察 N 个干扰符号（推进 tick、不学习），
   再按量具协议查询。字段里的 episode 不衰减，衰减的是 time_encoder 的时间码错位——
   N 越大，存入时的 tick 与查询时 tick 距离越远。这条回答" episode 年龄多大还能召回"。
2. **痕迹曲线**（查询前不 reset）：同一次写入后连续观察 N 个干扰符号（同一动力学 episode，
   递归痕迹自然衰减），然后直接喂 cue 语境并读字段级召回（`memory.recall` 的
   confidence 与 action_evidence 是否指回写入的动作）。这条回答"同一 episode 内
   递归痕迹隔多少步还能托住召回"。

**产出物**：能力—间隔曲线与饱和点（§2e 判断 2 的输入：决定"要不要更多记忆层"）。
**守卫**：N=0 的年龄曲线召回必须显著高于瞎猜（动作 4 选 1，chance=0.25；低于 0.5 即
量具在产品基座上不成立，退出码 2）；基座 sha 复核；全程 learn=False（零训练）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: 间隔扫描点（步）。1024 ≈ 情节场容量的量级（§1b：容量 1024）。
INTERVALS = (0, 16, 64, 256, 1024)
CHANCE = 0.25
N0_MIN = 0.5
SEED = 20260926


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_corpus(k: int) -> tuple[list[Any], list[Any], tuple[int, ...]]:
    """K 条绑定：不同语境文本、不同 cue 符号、动作 4 选 1。符号都在 0..255。"""
    from taiji.foundation_tasks import DelayedMemoryQuery, MemoryEpisode

    actions = tuple(ord(digit) for digit in "1234")
    episodes: list[Any] = []
    queries: list[Any] = []
    for index in range(k):
        context = tuple(f"背景{index}号。".encode())
        cue = 200 + index
        action = actions[index % len(actions)]
        episodes.append(
            MemoryEpisode(
                memory_id=f"c3-{index}",
                cue=cue,
                action=action,
                outcome=46,  # "." 字节：outcome 必须在传感器字母表（0..255）内
                context=context,
            )
        )
        queries.append(
            DelayedMemoryQuery(
                query_id=f"c3-q-{index}",
                cue=cue,
                expected_action=action,
                context=context,
            )
        )
    return episodes, queries, actions


def _interference(n: int) -> tuple[int, ...]:
    generator = torch.Generator().manual_seed(SEED)
    return tuple(int(value) for value in torch.randint(0, 256, (n,), generator=generator).tolist())


def _field_recall(
    substrate: Any, context: tuple[int, ...], cue: int, actions: tuple[int, ...]
) -> dict[str, Any]:
    """直接读字段：喂 context+cue（learn=False, use_memory=True），记 confidence 与证据指向。"""
    substrate.reset_dynamics(episode_id="c3-field-recall")
    substrate.observe(
        int(substrate.config.boundary_symbol), learn=False, readout="predictive", use_memory=True
    )
    for symbol in context:
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=True)
    substrate.observe(cue, learn=False, readout="predictive", use_memory=True)
    state = substrate._state
    _, recall = substrate.memory.recall(
        substrate.fabric.cortical_context(state.regions),
        state.memory,
        use_long_term=True,
    )
    evidence = recall.action_evidence
    return {
        "confidence": round(float(recall.confidence), 6),
        "evidence_argmax_is_action": (
            bool(int(evidence.argmax()) in actions) if evidence.numel() else False
        ),
    }


def run_interval(
    checkpoint: Path, episodes: list[Any], queries: list[Any], actions: tuple[int, ...], n: int
) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime
    from taiji.foundation_tasks import DelayedMemoryTask

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    for episode in episodes:
        DelayedMemoryTask._write_episode(substrate, episode)
    write_count = substrate.memory.write_count

    interference = _interference(n)
    for symbol in interference:
        substrate.observe(symbol, learn=False, learn_motor=False)

    accuracy_mem = DelayedMemoryTask._recall_accuracy(substrate, queries, actions, use_memory=True)
    accuracy_lesion = DelayedMemoryTask._recall_accuracy(
        substrate, queries, actions, use_memory=False
    )
    field_rows = [
        _field_recall(substrate, episode.context, episode.cue, actions) for episode in episodes
    ]
    return {
        "interval_steps": n,
        "write_count": write_count,
        "accuracy_use_memory": round(accuracy_mem, 4),
        "accuracy_lesion": round(accuracy_lesion, 4),
        "field_mean_confidence": round(
            sum(row["confidence"] for row in field_rows) / max(len(field_rows), 1), 6
        ),
        "field_confidences": [row["confidence"] for row in field_rows],
        "field_evidence_argmax_in_actions": sum(
            1 for row in field_rows if row["evidence_argmax_is_action"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--bindings", type=int, default=8)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    episodes, queries, actions = _build_corpus(args.bindings)

    curve = [run_interval(checkpoint, episodes, queries, actions, n) for n in INTERVALS]
    saturation = next(
        (row["interval_steps"] for row in curve if row["accuracy_use_memory"] <= CHANCE + 0.05),
        None,
    )
    report = {
        "format": "taiji-c3-retention-curve-v1",
        "prereg": "plans/reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md §2e C3",
        "checkpoint": args.checkpoint,
        "protocol": {
            "reused_instrument": "taiji.foundation_tasks.DelayedMemoryTask._write_episode/_recall_accuracy",
            "bindings": args.bindings,
            "action_classes": len(actions),
            "intervals": list(INTERVALS),
            "age_curve": "reset before query (field recall vs episode age via time code)",
            "lesion_arm": "use_memory=False per interval",
        },
        "curve": curve,
        "saturation_interval": saturation,
        "what_would_overturn": (
            "N=0 召回低于 chance ⇒ 量具在产品基座上不成立；若曲线在所有间隔都平且高于 N0_MIN ⇒ "
            "时间码错位不是保持瓶颈（瓶颈在别处，如容量或 cue 重合度）"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "episodes_written": all(row["write_count"] == args.bindings for row in curve),
        "n0_recall_above_half": curve[0]["accuracy_use_memory"] >= N0_MIN,
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values())
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "curve": [
                    {
                        "n": row["interval_steps"],
                        "mem": row["accuracy_use_memory"],
                        "lesion": row["accuracy_lesion"],
                        "confidence": row["field_mean_confidence"],
                    }
                    for row in curve
                ],
                "saturation": saturation,
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

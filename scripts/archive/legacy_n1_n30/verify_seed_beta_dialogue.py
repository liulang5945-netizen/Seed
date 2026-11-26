#!/usr/bin/env python3
"""公测（M2）门槛：Seed 对话能力量化评测面板。

固定题集 50 条（单轮 30 + 多轮 10 组 × 2 轮），对给定检查点测量公测
路线图 §1.1 的四项对话门槛：

判据 1（响应连贯性）：
    回复 UTF-8 有效解码率 >= 0.99（无替换字符、无半截多字节序列）。

判据 2（对话质量 / 可读性）：
    可读率 >= 0.60。可读 = 回复非空且长度 >= 4 字符、有效解码、
    CJK 字符与中文标点占比 >= 0.6、且无单字符重复主导（最大单字符
    占比 <= 0.5）。这是自动代理指标，公测发布前另做 50 条人工盲评。

判据 3（多轮上下文保持）：
    多轮组在第 1 轮植入一个事实（名字/颜色/城市/数字等），第 2 轮
    追问；回复包含植入关键词即计引用成功。引用率 >= 0.60。

判据 4（轮次结构）：
    回复不空、不越过下一轮 ``问：`` 标记（由运行时截断保证）、
    生成在预算内停止。有效率 >= 0.99。

约束：评测只读不写（不 learn、不落检查点）；序列化口径与
``api/seed_runtime.py`` 完全一致（问：/答： 标记 + boundary）。
输出 ``reports/seed_beta_dialogue_<date>.json``，失败以非零码退出。

运行：python -X utf8 -u scripts/training/verify_seed_beta_dialogue.py \\
      --checkpoint checkpoints/seed_corpus.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import _verify_emit  # noqa: E402
import torch  # noqa: E402

from seed import Seed  # noqa: E402

MAX_LENGTH = 256

# ---------------- 固定题集（不随版本漂移；改动必须记录理由） ----------------

SINGLE_TURN: tuple[str, ...] = (
    "你好，请介绍一下你自己。",
    "今天天气怎么样？",
    "1加1等于几？",
    "中国的首都是哪里？",
    "水的化学式是什么？",
    "一年有多少个月？",
    "请给我讲一个笑话。",
    "什么是人工智能？",
    "怎么学好英语？",
    "太阳从哪边升起？",
    "一个星期有几天？",
    "地球上最大的海洋是哪个？",
    "请你推荐一本书。",
    "为什么天空是蓝色的？",
    "怎么煮一碗面条？",
    "猫和狗哪个更适合当宠物？",
    "什么是光合作用？",
    "请你写一句祝福的话。",
    "飞机为什么能飞起来？",
    "冬天应该注意什么？",
    "怎样做才能早睡早起？",
    "月亮为什么会有圆缺？",
    "请解释一下什么是互联网。",
    "跑步前应该做什么准备？",
    "什么是数学？",
    "请你谈谈读书的好处。",
    "雨是怎么形成的？",
    "如何向朋友道歉？",
    "什么是友谊？",
    "请说一句鼓励人的话。",
)

# 多轮：第 1 轮植入事实，第 2 轮追问；关键词用于自动判定引用成功。
MULTI_TURN: tuple[tuple[str, str, str], ...] = (
    ("我叫小明，请记住我的名字。", "我叫什么名字？", "小明"),
    ("我最喜欢的颜色是蓝色。", "我最喜欢的颜色是什么？", "蓝色"),
    ("我住在北京。", "我住在哪个城市？", "北京"),
    ("我的幸运数字是7。", "我的幸运数字是几？", "7"),
    ("我养了一只叫雪球的小猫。", "我的小猫叫什么名字？", "雪球"),
    ("我最喜欢吃的水果是西瓜。", "我最喜欢吃的水果是什么？", "西瓜"),
    ("我的生日是三月五日。", "我的生日是哪一天？", "三月五日"),
    ("我是一名老师。", "我的职业是什么？", "老师"),
    ("我明天要去上海出差。", "我明天要去哪里？", "上海"),
    ("我最喜欢的运动是游泳。", "我最喜欢的运动是什么？", "游泳"),
)

THRESHOLDS = {
    "utf8_valid_rate": 0.99,
    "readable_rate": 0.60,
    "multi_turn_reference_rate": 0.60,
    "well_formed_rate": 0.99,
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ZH_PUNCT_RE = re.compile(r"[，。！？；：、“”‘’（）—…《》]")


def _serialize(prompt: str, history: list[tuple[str, str]]) -> str:
    parts = [f"问：{user}\n答：{assistant}" for user, assistant in history]
    parts.append(f"问：{prompt}\n答：")
    return "\n".join(parts)


def _generate_reply(model: Seed, prompt: str, history: list[tuple[str, str]]) -> tuple[bytes, str]:
    prefix = _serialize(prompt, history).encode("utf-8")
    raw = model.generate(prefix, MAX_LENGTH, stop_at_boundary=True, sample=False)
    answer = raw.decode("utf-8", errors="replace")
    for marker in ("\n问：", "问："):
        index = answer.find(marker)
        if index >= 0:
            answer = answer[:index]
    return raw, answer.strip()


def _is_utf8_clean(raw: bytes, decoded: str) -> bool:
    if "\ufffd" in decoded:
        return False
    try:
        raw.decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError:
        return False


def _is_readable(answer: str) -> bool:
    if len(answer) < 4 or "\ufffd" in answer:
        return False
    cjk = len(_CJK_RE.findall(answer)) + len(_ZH_PUNCT_RE.findall(answer))
    if cjk / len(answer) < 0.6:
        return False
    # 单字符重复主导（如 "的的的的..."）不算可读。
    counts: dict[str, int] = {}
    for char in answer:
        counts[char] = counts.get(char, 0) + 1
    return max(counts.values()) / len(answer) <= 0.5


def run_panel(checkpoint_path: Path) -> dict[str, object]:
    envelope = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Seed.from_checkpoint(envelope)

    replies: list[dict[str, object]] = []
    utf8_valid = 0
    readable = 0
    well_formed = 0
    total_turns = 0

    for prompt in SINGLE_TURN:
        raw, answer = _generate_reply(model, prompt, [])
        total_turns += 1
        clean = _is_utf8_clean(raw, answer)
        utf8_valid += int(clean)
        readable += int(_is_readable(answer))
        well_formed += int(bool(answer) and len(answer) <= MAX_LENGTH)
        replies.append({"kind": "single", "prompt": prompt, "answer": answer})

    reference_hits = 0
    for plant, probe, keyword in MULTI_TURN:
        history: list[tuple[str, str]] = []
        for turn_index, prompt in enumerate((plant, probe)):
            raw, answer = _generate_reply(model, prompt, history)
            total_turns += 1
            clean = _is_utf8_clean(raw, answer)
            utf8_valid += int(clean)
            readable += int(_is_readable(answer))
            well_formed += int(bool(answer) and len(answer) <= MAX_LENGTH)
            if turn_index == 1:
                hit = keyword in answer
                reference_hits += int(hit)
                replies.append(
                    {
                        "kind": "multi",
                        "plant": plant,
                        "probe": prompt,
                        "keyword": keyword,
                        "answer": answer,
                        "reference_hit": hit,
                    }
                )
            history.append((prompt, answer))

    checks = {
        "utf8_valid_rate_at_least_99pct": (
            utf8_valid / total_turns >= THRESHOLDS["utf8_valid_rate"]
        ),
        "readable_rate_at_least_60pct": (readable / total_turns >= THRESHOLDS["readable_rate"]),
        "multi_turn_reference_at_least_60pct": (
            reference_hits / len(MULTI_TURN) >= THRESHOLDS["multi_turn_reference_rate"]
        ),
        "well_formed_rate_at_least_99pct": (
            well_formed / total_turns >= THRESHOLDS["well_formed_rate"]
        ),
    }
    return {
        "benchmark": "seed_beta_dialogue",
        "checkpoint": str(checkpoint_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": THRESHOLDS,
        "metrics": {
            "total_turns": total_turns,
            "utf8_valid_rate": utf8_valid / total_turns,
            "readable_rate": readable / total_turns,
            "multi_turn_reference_rate": reference_hits / len(MULTI_TURN),
            "well_formed_rate": well_formed / total_turns,
        },
        "checks": checks,
        "replies": replies,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_panel(Path(args.checkpoint))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    out_path = args.output or (
        REPO / "reports" / f"seed_beta_dialogue_{time.strftime('%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"\n报告已写入 {out_path}", file=sys.stderr)
    return _verify_emit.emit_and_exit("seed_beta_dialogue", report)


if __name__ == "__main__":
    raise SystemExit(main())

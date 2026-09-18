"""Stage 0 只读证据：token 化演进线（VISION §3.1b）的两个前置数字。

本脚本是**只读分析**，不加载模型、不改动 taiji 任何状态或 digest。它回答两个
问题，作为 token 化线触发条件的第一手证据：

1. 距离压缩比 —— 若字节流可无损聚合为概念单元，序列的因果跨距被压缩多少倍。
   ``压缩比 = 平均字节步数 / 平均概念步数``。这是"token 化缓解 delay-probe
   长程信用稀释（distance>=32≈0）"的空间余地估计，不是能力承诺。
2. 聚类熵-容量拐点 —— 扫描容量候选 C，报覆盖率的边际收益与平均样本量，找
   "再加大也覆盖不了多少新内容"的拐点，作为词表初始容量与需求驱动扩容的锚。

诚实边界：概念单元用统计代理（ASCII 词 / CJK 簇 / 单符号），不是神经聚类本身；
覆盖率曲线是 Zipf 截断覆盖的代理，不冒充真实聚类的熵值。数字只用于"值不值得
预注册"的判定，不经本脚本授权任何实现。

用法：:
    python scripts/training/probe_taiji_stage0_token_evidence.py [--corpus PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ASCII 词 / 连续 CJK 簇 / 其余单符号。显式代理 token 化，不引入 BPE 或分词器。
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[一-\u9fff]+|[^\sA-Za-z0-9一-\u9fff]")

_CAPACITY_SCAN = (128, 256, 512, 1024, 2048, 4096)
_MARGINAL_GAIN_STOP = 0.10  # 覆盖率边际增益低于 10% 视为"拐点之后"
_OUT_DIR = Path(__file__).resolve().parents[2] / "reports"

_SAMPLE_CORPUS = (
    "Taiji is a symbol-by-symbol brain-like learner that updates local synapses online. "
    "It never calls loss.backward and keeps no attention matrix, yet it needs to hold "
    "enough context to predict the next byte across a long sentence boundary. "
    "The quick brown fox jumps over the lazy dog again and again while the model "
    "tries to aggregate repeated byte neighbours into reusable concept units. "
    "Neural clustering drifts prototypes online, freezes frequent anchors, and lets "
    "the long tail stay plastic so new words can still be learned later. "
    "在线神经聚类把常见字节邻接聚合成概念单元，高频锚冻结、长尾可塑，容量增长由任务需求驱动。"
    "概念单元的起源证据保留，每次扩容都写入成长史，受硬预算门控，不是无限词表。"
    "原型、强度、计数与首现字节都可审计，聚合边界随数据收敛后冻结。"
)


def _load_corpus_text(path: str | None) -> str:
    if not path:
        return _SAMPLE_CORPUS
    candidate = Path(path)
    if not candidate.is_file():
        raise SystemExit(f"--corpus 指定的文件不存在: {path}")
    raw = candidate.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _tokenize(text: str) -> tuple[int, list[str]]:
    """返回 (字节数, 代理概念单元列表)。"""
    return len(text.encode("utf-8")), _TOKEN_PATTERN.findall(text)


def _distance_compression(text: str, tokens: list[str]) -> dict[str, float]:
    byte_steps = len(text.encode("utf-8"))
    token_steps = max(1, len(tokens))
    compression = byte_steps / token_steps
    # 跨距的 90 分位：代表"这语料里典型需要跨越多少个单元"的尾部长度。
    # 用长短对冲：统计每 50 token 窗口内字节数，取 90 分位再除以 50 。
    per_token_bytes = [
        len("".join(window).encode("utf-8")) / max(1, len(window))
        for window in (tokens[i : i + 50] for i in range(0, max(1, len(tokens) - 49), 50))
    ]
    p90 = (
        sorted(per_token_bytes)[int(len(per_token_bytes) * 0.90) - 1]
        if per_token_bytes
        else compression
    )
    return {
        "byte_steps": float(byte_steps),
        "concept_steps": float(token_steps),
        "distance_compression_ratio": float(compression),
        "p90_bytes_per_50_concepts": float(p90),
    }


def _capacity_curve(tokens: list[str]) -> dict[str, object]:
    """Zipf 截断覆盖代理：容量 C 下 top-C 单元覆盖的 token 出现次数占比。"""
    counts = Counter(tokens)
    total = max(1, sum(counts.values()))
    ordered = sorted(counts.values(), reverse=True)
    curve: list[dict[str, float]] = []
    prev_gain: float | None = None
    for capacity in _CAPACITY_SCAN:
        covered = sum(ordered[:capacity])
        coverage = covered / total
        marginal = None
        if prev_gain is not None:
            marginal = max(0.0, coverage - prev_gain)
        distinct = len(ordered)
        avg_samples = covered / min(capacity, distinct)
        curve.append(
            {
                "capacity": float(capacity),
                "coverage": coverage,
                "marginal_gain": marginal if marginal is not None else 0.0,
                "avg_samples": avg_samples,
                "distinct_units": float(distinct),
            }
        )
        prev_gain = coverage
    # 拐点：首个边际增益低于 stop 阈值的容量。
    knee = next(
        (entry["capacity"] for entry in curve if entry["marginal_gain"] < _MARGINAL_GAIN_STOP),
        curve[-1]["capacity"],
    )
    return {
        "capacity_scan": curve,
        "marginal_gain_stop": _MARGINAL_GAIN_STOP,
        "knee_capacity": knee,
    }


def _report(
    text: str, tokens: list[str], compression: dict[str, float], curve: dict[str, object]
) -> dict[str, object]:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    return {
        "probe": "stage0_token_evidence",
        "created_at": now,
        "method_note": (
            "只读统计代理：距离压缩比为字节步/概念步；覆盖率曲线为 Zipf 截断覆盖，"
            "非真实聚类熵。仅用于 token 化线预注册判定。"
        ),
        "compression": compression,
        "capacity": curve,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default=None, help="语料文本或 utf-8 jsonl 路径；缺省用内置示例"
    )
    args = parser.parse_args(argv)

    text = _load_corpus_text(args.corpus)
    byte_steps, tokens = _tokenize(text)
    if not tokens:
        raise SystemExit("语料未产生任何概念单元，无法分析")
    compression = _distance_compression(text, tokens)
    curve = _capacity_curve(tokens)

    print(f"字节步数        : {compression['byte_steps']:.0f}")
    print(f"概念单元步数    : {compression['concept_steps']:.0f}")
    print(
        f"距离压缩比      : {compression['distance_compression_ratio']:.2f}x "
        f"(即概念级因果跨距缩短 {compression['distance_compression_ratio']:.2f} 倍)"
    )
    print(
        f"90 分位字节负担  : 每 50 概念单元约 {compression['p90_bytes_per_50_concepts']:.1f} 字节"
    )
    print("容量扫描（覆盖率 / 边际增益）:")
    for entry in curve["capacity_scan"]:  # type: ignore[union-attr]
        print(
            f"  C={entry['capacity']:>5}  覆盖={entry['coverage']:.3f}  "
            f"边际={entry['marginal_gain']:.3f}  平均样本/单元={entry['avg_samples']:.1f}"
        )
    print(f"拐点容量(边际<{curve['marginal_gain_stop']:.2f}) : {curve['knee_capacity']}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "probe_taiji_stage0_token_evidence.json"
    out_path.write_text(
        json.dumps(_report(text, tokens, compression, curve), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

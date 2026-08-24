#!/usr/bin/env python3
"""公测（M3）门槛：Seed 运行时性能基线与守卫。

测量公测路线图 §1.4 的三项交互性能门槛：

判据 1（运行时加载）：
    从磁盘加载检查点并构建模型 <= 30 秒。

判据 2（首字节延迟）：
    对话提示（含 ``问：/答：`` 序列化）到第一个生成字节 <= 2 秒
    （5 条固定提示取最大值）。

判据 3（生成吞吐）：
    256 字节回复的平均吞吐 >= 200 bytes/s（8 条固定提示）。

输出含基线快照（load_seconds / first_byte_seconds / bytes_per_second），
M1 大预算检查点替换后以此钉住性能守卫。只读不写。
输出 ``reports/seed_beta_perf_<date>.json``，失败以非零码退出。

运行：python -X utf8 -u scripts/training/verify_seed_beta_perf.py \\
      --checkpoint checkpoints/seed_corpus.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import _verify_emit  # noqa: E402

from seed import Seed  # noqa: E402

MAX_LENGTH = 256

THRESHOLDS = {
    "load_seconds": 30.0,
    "first_byte_seconds": 2.0,
    "bytes_per_second": 200.0,
}

LATENCY_PROMPTS = (
    "你好，请介绍一下你自己。",
    "今天天气怎么样？",
    "请给我讲一个笑话。",
    "什么是一次函数？",
    "请说一句鼓励人的话。",
)

THROUGHPUT_PROMPTS = (
    "中国的首都是哪里？",
    "水的化学式是什么？",
    "为什么天空是蓝色的？",
    "怎样煮一碗面条？",
    "请你推荐一本书。",
    "什么是光合作用？",
    "飞机为什么能飞起来？",
    "请解释一下什么是互联网。",
)


def _prefix(prompt: str) -> bytes:
    return f"问：{prompt}\n答：".encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)

    start = time.perf_counter()
    envelope = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Seed.from_checkpoint(envelope)
    load_seconds = time.perf_counter() - start

    first_byte_samples: List[float] = []
    for prompt in LATENCY_PROMPTS:
        start = time.perf_counter()
        model.generate(_prefix(prompt), 1, sample=False)
        first_byte_samples.append(time.perf_counter() - start)

    total_bytes = 0
    total_seconds = 0.0
    throughput_rows = []
    for prompt in THROUGHPUT_PROMPTS:
        start = time.perf_counter()
        raw = model.generate(_prefix(prompt), MAX_LENGTH, stop_at_boundary=True, sample=False)
        elapsed = time.perf_counter() - start
        total_bytes += len(raw)
        total_seconds += elapsed
        throughput_rows.append(
            {
                "prompt": prompt,
                "bytes": len(raw),
                "seconds": round(elapsed, 3),
                "bytes_per_second": round(len(raw) / max(elapsed, 1e-9), 1),
            }
        )
    bytes_per_second = total_bytes / max(total_seconds, 1e-9)

    metrics = {
        "load_seconds": round(load_seconds, 3),
        "first_byte_seconds_max": round(max(first_byte_samples), 3),
        "first_byte_seconds_all": [round(v, 3) for v in first_byte_samples],
        "bytes_per_second": round(bytes_per_second, 1),
        "total_generated_bytes": total_bytes,
        "throughput_rows": throughput_rows,
    }
    checks = {
        "load_within_30s": load_seconds <= THRESHOLDS["load_seconds"],
        "first_byte_within_2s": max(first_byte_samples) <= THRESHOLDS["first_byte_seconds"],
        "throughput_at_least_200_bps": bytes_per_second >= THRESHOLDS["bytes_per_second"],
    }
    report = {
        "benchmark": "seed_beta_perf",
        "checkpoint": str(checkpoint_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    out_path = args.output or (REPO / "reports" / f"seed_beta_perf_{time.strftime('%Y%m%d')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"\n报告已写入 {out_path}", file=sys.stderr)
    return _verify_emit.emit_and_exit("seed_beta_perf", report)


if __name__ == "__main__":
    raise SystemExit(main())

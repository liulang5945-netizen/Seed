"""B-3 收口前置：量 `infer_domain` 兜底路径的**非确定性**（零训练、秒级）。

PLAN-B-03 §4 声明的唯一行为变更 = 把 `_first_domain()`（`next(iter(set))`）换成确定性兜底。
本件是该变更的**前后对照证据**：同一棵树实现跑两遍（改前/改后）⇒ 哪几格随 set 迭代序翻转、
改后是否归零，全部由探针实测，不靠读代码断言。

度量面：黄金向量同一网格（10 个域集 × 33 条文本 = 330 格）。
每格在多个 `PYTHONHASHSEED` 子进程 × 多个键插入序下重复求值；观测到 >1 个不同输出 ⇒ 该格
**不是输入的确定函数**（即兜底可达且集内前缀不同）。

用法：
    python scripts/training/probe_taiji_cortex_domain_fallback_determinism.py --out reports/x.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN = PROJECT_ROOT / "reports" / "cortex_infer_domain_golden_20260925.json"
DEFAULT_RUNS = 12

DOMAIN_SETS = [
    [],
    ["zh"],
    ["en"],
    ["code"],
    ["math"],
    ["zh", "en"],
    ["zh", "code"],
    ["en", "math"],
    ["general"],
    ["zh_aug0_dialogue"],
]


def _orders(names: list[str]) -> list[tuple[str, ...]]:
    """同一域集的多种键插入序（≥2 个前缀时才有多序可言）。"""

    keys = [f"{n}_unit" for n in names]
    if len(keys) < 2:
        return [tuple(keys)]
    return list(itertools.islice(itertools.permutations(keys), 6))


def worker_grid() -> dict[str, dict[str, str]]:
    from neuroplex.brain import _cortex_helpers

    texts = json.loads(GOLDEN.read_text(encoding="utf-8"))["texts"]
    grid: dict[str, dict[str, str]] = {}
    for names in DOMAIN_SETS:
        key = "|".join(names) if names else "(empty)"
        cells: dict[str, str] = {}
        for text in texts:
            seen: set[str] = set()
            for order in _orders(names):
                seen.add(_cortex_helpers.infer_domain(set(order), text))
            # 同进程内多序若已分歧，任取一个仍不足以刻画 ⇒ 记 "!" 让父进程看见分歧
            cells[text] = seen.pop() if len(seen) == 1 else "!ambiguous"
        grid[key] = cells
    return grid


def run_parent(runs: int) -> dict[str, object]:
    distinct: dict[str, dict[str, list[str]]] = {}
    for seed in range(runs):
        env = {**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            capture_output=True,
            text=True,
            encoding="utf-8",  # 缺省按本机 locale（GBK）解码中文文本 ⇒ stdout 读崩
            env=env,
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        for key, cells in json.loads(proc.stdout).items():
            bucket = distinct.setdefault(key, {})
            for text, value in cells.items():
                bucket.setdefault(text, [])
                if value not in bucket[text]:
                    bucket[text].append(value)

    golden_grid = json.loads(GOLDEN.read_text(encoding="utf-8"))["grid"]
    unstable: list[dict[str, object]] = []
    for key, cells in distinct.items():
        names = [] if key == "(empty)" else key.split("|")
        for text, values in cells.items():
            if len(values) > 1:
                unstable.append(
                    {
                        "domain_set": key,
                        "text": text,
                        "observed": sorted(values),
                        "golden": golden_grid[key][text],
                        "deterministic_fallback": (
                            min(f"{n}_unit" for n in names).split("_")[0] if names else "general"
                        ),
                    }
                )
    return {
        "format": "cortex-domain-fallback-determinism-v1",
        "implementation_source": "neuroplex/brain/_cortex_helpers.py::infer_domain",
        "git_rev": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        ).stdout.strip(),
        "runs_per_cell": runs,
        "grid_shape": [
            len(DOMAIN_SETS),
            len(json.loads(GOLDEN.read_text(encoding="utf-8"))["texts"]),
        ],
        "cells_measured": sum(len(v) for v in distinct.values()),
        "non_deterministic_cells": unstable,
        "non_deterministic_count": len(unstable),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help="子进程模式：打印当前实现下的网格")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="哈希种子重复次数")
    ap.add_argument("--out", type=str, default="", help="报告落盘路径；留空只打印摘要")
    args = ap.parse_args()

    if args.worker:
        sys.stdout.buffer.write(json.dumps(worker_grid(), ensure_ascii=False).encode("utf-8"))
        return 0

    report = run_parent(args.runs)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        report["report"] = str(out)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("git_rev", "cells_measured", "non_deterministic_count", "report")
                if k in report
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

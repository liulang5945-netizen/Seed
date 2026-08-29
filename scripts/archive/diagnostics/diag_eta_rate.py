"""ETA 根因探针：测量逐字节 observe() 的吞吐是否随 tick 漂移。

不修改任何产品代码，只复刻 api/training/native.py 的模型构造方式，
按 PROGRESS_EVERY 的节奏打印「累计速率」与「窗口速率」，
用来判定 resume.py:138 的无平滑线性外推是否为主因。
"""

from __future__ import annotations

import time
from pathlib import Path

from seed import Seed, SeedConfig, iter_native_documents
from taiji.config import TaijiConfig

WINDOW = 500
ROUNDS = 24
DATASET = Path("data/simple_zh/shared_core.jsonl")


def main() -> None:
    config = SeedConfig(taiji=TaijiConfig.capacity_profile(2_000_000, seed=0))
    model = Seed(config, device="cpu", episode_id="diag-eta-rate")
    boundary = config.taiji.boundary_symbol

    def symbols():
        for text in iter_native_documents([DATASET]):
            yield boundary, 0
            for sym in text.encode("utf-8"):
                yield sym, 1

    started = time.perf_counter()
    ticks = 0
    window_start = started
    window_base = 0
    printed = 0

    print(f"{'tick':>10} {'累计速率':>12} {'窗口速率':>12} {'比值':>8}")
    for symbol, _ in symbols():
        model.observe(symbol, learn=True)
        ticks += 1
        if ticks % WINDOW == 0:
            now = time.perf_counter()
            cum = ticks / max(1e-6, now - started)
            win = (ticks - window_base) / max(1e-6, now - window_start)
            print(f"{ticks:>10,} {cum:>12.1f} {win:>12.1f} {win / cum:>8.3f}")
            window_start = now
            window_base = ticks
            printed += 1
            if printed >= ROUNDS:
                break


if __name__ == "__main__":
    main()

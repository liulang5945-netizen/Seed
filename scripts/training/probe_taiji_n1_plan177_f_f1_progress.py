"""F1 训练进度只读探针：对某个 run 的 `latest.pt` 在固定 16 条训练家族上算 loss/hit。

不 step、不写 checkpoint；只用来回答"这一步之后模型是否在学"，并把读数追加到
`data/checkpoints/n1_plan177_f_f1_20261029/progress.jsonl`（按策略忽略，机器本地）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.run_taiji_n1_plan177_f_f0b import (  # noqa: E402
    _Arm,
    chain_logits,
    load_trainer,
    train_items,
)

RUN_ROOT = ROOT / "data/checkpoints/n1_plan177_f_f1_20261029"
LOG = RUN_ROOT / "progress.jsonl"


def measure(arm: _Arm, trainer: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    losses = []
    hits = 0
    with torch.no_grad():
        for item in items:
            base = chain_logits(trainer, item)
            logits = base + arm.bias(arm.feature(item))
            target = torch.tensor([item["target_byte"]], dtype=torch.long)
            losses.append(float(F.cross_entropy(logits.reshape(1, -1), target)))
            hits += int(logits.argmax()) == item["target_byte"]
    return {
        "mean_loss": sum(losses) / len(losses),
        "hits": hits,
        "items": len(items),
        "hit_rate": hits / len(items),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F1 训练进度只读探针")
    parser.add_argument("--run", default="main_credit_on__seed_0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    run_root = RUN_ROOT / args.run
    checkpoint = args.checkpoint
    if checkpoint is None:
        latest = run_root / "latest.pt"
        checkpoint = latest if latest.is_file() else run_root / "zero.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arm = _Arm.from_checkpoint(payload)
    trainer = load_trainer()
    items = train_items()
    started = time.process_time()
    reading = measure(arm, trainer, items)
    reading.update(
        {
            "run": args.run,
            "checkpoint": str(checkpoint).replace("\\", "/"),
            "cursor": arm.cursor,
            "measured_at_unix": time.time(),
            "cpu_seconds": time.process_time() - started,
        }
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(reading, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(reading, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

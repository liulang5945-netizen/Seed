"""Train Seed on the raw-byte stream of the simple_zh corpus.

阶段 1 原生数据管线：复用 ``data/simple_zh/`` 既有语料（当前 canonical 为
dialogue_extended_clean），以 raw-byte 流喂入 ``Seed.observe``；
会话边界用 ``boundary_symbol``，对话结构沿用语料里的文本标记（问：/答：），
全程不引入 tokenizer。训练循环为分片流式多 epoch + 周期 ``checkpoint()``
落盘（seed-native-v1 信封），进度曲线逐条写入 ``reports/``。

用法（默认小预算冒烟）::

    python scripts/training/train_seed_corpus.py --smoke

正式训练（放大画像、限量符号数、周期落盘）::

    python scripts/training/train_seed_corpus.py \
        --parameter-budget 500000 --device auto \
        --epochs 1 --max-symbols 400000 \
        --checkpoint checkpoints/seed_corpus.pt \
        --progress reports/seed_corpus_progress.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from seed import Seed, SeedConfig, iter_native_documents  # noqa: E402
from seed.persistence import (  # noqa: E402
    atomic_save,
    attach_metadata,
    corpus_fingerprint,
)
from taiji import CapacityPolicy, TaijiConfig  # noqa: E402

DEFAULT_CORPUS = (
    # 2026-08-23 数据整理：canonical 对话语料仅此一文件（123090 条，
    # 已吸收 alpaca-zh/shared_core 内容）；旧声明中的 alpaca_zh_sft_clean /
    # class_a_chinese 已删除。
    "data/simple_zh/dialogue_extended_clean.jsonl",
)

# Fixed unseen probe: window statistics over the moving stream measure content
# difficulty, not model quality, so every progress entry also scores this
# constant byte string.  A monotone training run must drive its surprise down.
HOLDOUT_PROBE = (
    "水的沸点在标准大气压下是一百摄氏度。"
    "问：你好。\n答：你好，很高兴见到你。"
    "请解释一下牛顿第二定律和它的日常应用。"
).encode("utf-8")


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve a requested training device without silently ignoring CUDA."""

    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch build or machine has no available CUDA device"
        )
    return device


def load_capacity_policy(path: Path | str) -> CapacityPolicy:
    """Load an explicit structural search policy from JSON."""

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("capacity policy must be a JSON object")
    return CapacityPolicy.from_dict(payload)


def iter_corpus_symbols(
    paths: Sequence[Path | str],
    *,
    boundary: int = TaijiConfig().boundary_symbol,
) -> Iterator[int]:
    """Stream every corpus document as boundary-separated raw UTF-8 bytes.

    Every jsonl row contributes one session: one ``boundary_symbol`` followed by
    the document's UTF-8 bytes.  The dialogue structure already lives in the
    text (问：/答： markers), so no tokenizer and no structural re-encoding is
    needed -- the model sees exactly the bytes a reader would see.
    """

    for text in iter_native_documents(paths):
        yield boundary
        yield from text.encode("utf-8")


def run_training(
    *,
    corpus_paths: Sequence[Path | str],
    config: SeedConfig,
    epochs: int,
    checkpoint_path: Path | str,
    progress_path: Path | str,
    checkpoint_every: int,
    progress_every: int,
    max_symbols: Optional[int] = None,
    resume_checkpoint: Optional[Path | str] = None,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """Stream the corpus through ``Seed.observe`` with periodic persistence."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if checkpoint_every <= 0 or progress_every <= 0:
        raise ValueError("checkpoint/progress intervals must be positive")

    checkpoint_path = Path(checkpoint_path)
    progress_path = Path(progress_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    model = Seed(config, device=resolve_device(device), episode_id="seed-corpus")
    if resume_checkpoint is not None:
        model.restore(torch.load(resume_checkpoint, weights_only=False))
    boundary = config.taiji.boundary_symbol
    fingerprint = corpus_fingerprint(corpus_paths)

    def _persist() -> None:
        # 2026-08-23 M0：原子落盘 + 信封元数据，崩溃不产生半写文件。
        envelope = attach_metadata(
            model.checkpoint(),
            tick=model.tick,
            corpus_fingerprint=fingerprint,
            extra={"trainer": "train_seed_corpus"},
        )
        atomic_save(envelope, checkpoint_path)

    started = time.perf_counter()
    # 续训时以模型自身 tick 为基线：进度统计与检查点节奏（% checkpoint_every）
    # 与崩溃前对齐，避免计数器清零导致重复训练段与节奏错位。
    ticks = int(model.tick)
    base_ticks = ticks
    window_ticks = 0
    window_correct = 0
    window_surprise = 0.0

    def _flush(final: bool) -> None:
        if window_ticks <= 0 and not final:
            return
        entry = {
            "epoch": epoch,
            "ticks": ticks,
            "window_ticks": window_ticks,
            "online_accuracy": window_correct / max(1, window_ticks),
            "mean_surprise": window_surprise / max(1, window_ticks),
            "holdout_surprise": model.score_bytes(HOLDOUT_PROBE)["mean_surprise"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    for epoch in range(epochs):
        for symbol in iter_corpus_symbols(corpus_paths, boundary=boundary):
            step = model.observe(symbol, learn=True)
            ticks += 1
            if step.prior_prediction is not None:
                window_ticks += 1
                window_correct += int(step.prior_prediction == symbol)
                window_surprise += float(step.surprise)
            if ticks % progress_every == 0:
                _flush(final=False)
                window_ticks = 0
                window_correct = 0
                window_surprise = 0.0
            if ticks % checkpoint_every == 0:
                _persist()
            if max_symbols is not None and ticks >= base_ticks + max_symbols:
                _flush(final=True)
                _persist()
                return _summary(model, ticks)
    _flush(final=True)
    _persist()
    return _summary(model, ticks)


def _summary(model: Seed, ticks: int) -> Dict[str, float]:
    return {
        "ticks": float(ticks),
        "parameters": float(model.parameter_count()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        nargs="+",
        default=[str(PROJECT_ROOT / name) for name in DEFAULT_CORPUS],
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument(
        "--parameter-budget",
        type=int,
        default=None,
        help="自动规划不超过该数量的可学习参数；设置后替代 --scale",
    )
    parser.add_argument(
        "--capacity-policy",
        default=None,
        help="容量策略 JSON；可改变区域深度、比例与 fan-in 密度，需配合 --parameter-budget",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="训练设备：auto、cpu、cuda 或 cuda:N",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-symbols", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "checkpoints" / "seed_corpus.pt"),
    )
    parser.add_argument(
        "--progress",
        default=str(PROJECT_ROOT / "reports" / "seed_corpus_progress.jsonl"),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny default config and budget for a fast end-to-end run",
    )
    args = parser.parse_args()

    if args.smoke:
        config = SeedConfig()
        max_symbols = 5_000
    else:
        if args.parameter_budget is None:
            if args.capacity_policy is not None:
                parser.error("--capacity-policy requires --parameter-budget")
            taiji_config = TaijiConfig.training_profile(scale=args.scale, seed=args.seed)
        else:
            policy = (
                load_capacity_policy(args.capacity_policy)
                if args.capacity_policy is not None
                else None
            )
            taiji_config = TaijiConfig.capacity_profile(
                args.parameter_budget,
                policy=policy,
                seed=args.seed,
            )
        config = SeedConfig(taiji=taiji_config)
        max_symbols = args.max_symbols

    summary = run_training(
        corpus_paths=args.corpus,
        config=config,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        progress_path=args.progress,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        max_symbols=max_symbols,
        resume_checkpoint=args.resume,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

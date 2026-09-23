"""保号存档的守卫：`--keep-checkpoints` 必须真的多写快照，且**主路径行为不变**。

来历（2026-09-23 所有者问"为啥训练时不都记详细日志，这样每次都能得到曲线"）：
此前 `--checkpoint-every` 反复覆盖**同一个文件**，于是训练中途的状态直接消失——
后来才想到要量的指标（槽可分离性）**连事后补算都做不到**，只剩首尾两个端点。
修法是每次落盘额外写一份 `checkpoint_<tick>.pt`。

本文件钉住三件事：

1. 给了目录 ⇒ 每个落盘点都留下一份带 tick 的快照；
2. 不给目录（旧行为） ⇒ **一份快照都不写**，且主存档照常产出；
3. 快照**能读回来**、tick 对得上（不是写了些读不出的空壳）。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from seed import SeedConfig
from taiji import TaijiConfig

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "training" / "train_seed_corpus.py"


def _write_corpus(path: Path, rows: int = 40) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(json.dumps({"text": f"第{index}行：这是一段用来训练的中文文本。"}) + "\n")
    return path


def _config() -> SeedConfig:
    return SeedConfig(
        taiji=TaijiConfig(
            region_sizes=(16,),
            synapse_fan_in=4,
            motor_fan_in=8,
            memory_units=16,
            memory_fan_in=4,
            memory_readout_fan_in=8,
            memory_meta_dim=8,
            seed=11,
        )
    )


def _run(tmp_path: Path, name: str, *, history: bool) -> tuple[Path, Path | None]:
    import sys

    sys.path.insert(0, str(RUNNER.parent))
    from train_seed_corpus import run_training

    corpus = _write_corpus(tmp_path / f"{name}_corpus.jsonl")
    checkpoint = tmp_path / f"{name}.pt"
    history_dir = tmp_path / f"{name}.history" if history else None
    run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=checkpoint,
        progress_path=tmp_path / f"{name}_progress.jsonl",
        checkpoint_every=500,
        progress_every=500,
        max_symbols=1500,
        keep_history=history_dir,
    )
    return checkpoint, history_dir


def test_history_keeps_one_snapshot_per_checkpoint(tmp_path: Path) -> None:
    checkpoint, history_dir = _run(tmp_path, "with", history=True)
    assert checkpoint.is_file(), "主存档必须照常产出"
    assert history_dir is not None and history_dir.is_dir()
    snapshots = sorted(history_dir.glob("checkpoint_*.pt"))
    assert [p.name for p in snapshots] == [
        "checkpoint_000000000500.pt",
        "checkpoint_000000001000.pt",
        "checkpoint_000000001500.pt",
    ], [p.name for p in snapshots]


def test_history_snapshots_are_loadable_and_carry_their_tick(tmp_path: Path) -> None:
    _checkpoint, history_dir = _run(tmp_path, "load", history=True)
    assert history_dir is not None
    for path, expected in (
        (history_dir / "checkpoint_000000000500.pt", 500),
        (history_dir / "checkpoint_000000001500.pt", 1500),
    ):
        envelope = torch.load(path, map_location="cpu", weights_only=False)
        assert int(envelope["metadata"]["tick"]) == expected, path.name


def test_without_a_history_dir_nothing_extra_is_written(tmp_path: Path) -> None:
    """旧行为必须还在：不给目录就一份快照都不写。"""

    checkpoint, _ = _run(tmp_path, "without", history=False)
    assert checkpoint.is_file()
    assert not list(tmp_path.glob("*.history"))
    assert not list(tmp_path.glob("checkpoint_*.pt"))

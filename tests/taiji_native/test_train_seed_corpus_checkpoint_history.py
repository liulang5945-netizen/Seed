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


def _run_cadence(tmp_path: Path, name: str, *, cap: int | None) -> tuple[Path, Path]:
    """同一条链跑两遍（一遍不限、一遍设上限），用**第一遍的全名单**当第二遍的裁判。

    为什么非要两遍：直接断言"该留哪三枚"就是我手工抄一遍被守卫的东西，改了也不会红；
    而"不限那遍的全集"是生产者的实际落盘序列，拿它当集合基准，薄中间那步才有判别力。
    """

    import sys

    sys.path.insert(0, str(RUNNER.parent))
    from train_seed_corpus import exit_record_path, run_training

    corpus = _write_corpus(tmp_path / f"{name}_corpus.jsonl")
    checkpoint = tmp_path / f"{name}.pt"
    history_dir = tmp_path / f"{name}.history"
    progress = tmp_path / f"{name}_progress.jsonl"
    run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=checkpoint,
        progress_path=progress,
        checkpoint_every=250,
        progress_every=100_000,
        max_symbols=1500,
        keep_history=history_dir,
        keep_history_max=cap,
    )
    return history_dir, exit_record_path(progress)


def test_history_cap_thins_the_middle_but_never_the_ends(tmp_path: Path) -> None:
    """DEBT-G40 的负对照：6 个落盘点配 cap=3 ⇒ 目录只剩 3 枚，**首尾必留**，且删掉的数被自述。"""

    uncapped, _ = _run_cadence(tmp_path, "full", cap=None)
    every = sorted(p.name for p in uncapped.glob("checkpoint_*.pt"))
    assert len(every) == 6, every  # 先证明这一条链真的落了 6 次，否则"薄中间"无从谈起

    capped, record_path = _run_cadence(tmp_path, "capped", cap=3)
    kept = sorted(p.name for p in capped.glob("checkpoint_*.pt"))
    assert len(kept) == 3, kept
    assert set(kept) <= set(every), (kept, every)
    assert kept[0] == every[0], (kept, every)  # 首
    assert kept[-1] == every[-1], (kept, every)  # 尾

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["history_files"] == len(kept), record
    assert record["history_bytes"] == sum(p.stat().st_size for p in capped.glob("checkpoint_*.pt"))
    assert record["history_pruned"] == len(every) - len(kept), record


def test_exit_record_history_fields_are_measured_not_echoed(tmp_path: Path) -> None:
    """默认不设上限时：`history_pruned` 必须是 0，`history_files` 必须等于目录实际计数（不是配置回显）。"""

    history_dir, record_path = _run_cadence(tmp_path, "default", cap=None)
    actual = sorted(p.name for p in history_dir.glob("checkpoint_*.pt"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["history_files"] == len(actual) == 6, record
    assert record["history_pruned"] == 0, record
    assert record["checkpoint_sha256"] and len(record["checkpoint_sha256"]) == 64


def test_cap_without_history_dir_is_refused(tmp_path: Path) -> None:
    """没有快照可删却设了上限＝空承诺，必须响亮停（同 `--answer-source self` 那两条硬约束的口径）。"""

    import sys

    import pytest

    sys.path.insert(0, str(RUNNER.parent))
    from train_seed_corpus import run_training

    corpus = _write_corpus(tmp_path / "nope.jsonl")
    with pytest.raises(ValueError, match="keep_history_max"):
        run_training(
            corpus_paths=[corpus],
            config=_config(),
            epochs=1,
            checkpoint_path=tmp_path / "nope.pt",
            progress_path=tmp_path / "nope_progress.jsonl",
            checkpoint_every=500,
            progress_every=100_000,
            max_symbols=600,
            keep_history=None,
            keep_history_max=3,
        )

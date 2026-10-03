"""DEBT-G14 的守卫：长跑训练必须**自己说清**"我是因为什么退出的、预算吃满了没有"。

来历（2026-09-29 实测代价）：`output/a31_ding3_boundary/` 那枚件被当成"在飞/约 2.4 小时"引用了一整轮，
真相是它只吃到 `+750,000 ticks ＝ 预注册预算的 37.5%`——因为退出原因当时**不在档里**，
只能从 `progress.jsonl` 的末行形状反向推（而 `run.log` 是 0 字节的空文件，比没有更误导）。

本文件钉四件事（每件都对应一次真实的误读形状）：

1. **预算支**退出 ⇒ 记账写"max_symbols_reached"且 `reached_budget` 为真；
2. **耗尽支**退出 ⇒ 写"corpus_exhausted"、`budget_max_symbols` 为 `None`（不许把"没给预算"写成"吃满"）；
3. 记账**只出现在收尾那一行**，周期性行一字不动（否则下游按旧键解析的读数会漂）；
   且独立件 `*_exit.json` 与那一行同值同源；
4. `_summary()` 仍**只有浮点值**——别线的在飞工具按 `{k: float(v)}` 消费它
   （`scripts/training/train_p3b_aligned.py:240`），把字符串塞进返回值会当场炸它。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from seed import SeedConfig
from taiji import TaijiConfig

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "training" / "train_seed_corpus.py"
sys.path.insert(0, str(RUNNER.parent))

from train_seed_corpus import exit_record_path, run_training  # noqa: E402

LEGACY_KEYS = {
    "epoch",
    "ticks",
    "window_ticks",
    "online_accuracy",
    "mean_surprise",
    "holdout_surprise",
    "elapsed_seconds",
}


def _write_corpus(path: Path, rows: int) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(json.dumps({"text": f"第{index}行：这是一段用来训练的中文文本。"}) + "\n")
    return path


def _write_qa_corpus(path: Path, rows: int) -> Path:
    #: 分块档（`answer_chunking="per-answer"`）按设计**响亮拒绝**没有 `\\n答：` 接缝的行，
    #: 所以那一臂的语料必须是问答形状——这条拒绝本身是它的配对守卫，不是障碍。
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps({"text": f"问：第{index}问是什么。\n答：第{index}答是一段中文答案。"}) + "\n"
            )
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


def _entries(progress_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_budget_exit_records_why_it_stopped(tmp_path: Path) -> None:
    """预算支：给小预算 ⇒ 收尾行与独立件都必须说"max_symbols_reached／吃满了"。"""

    corpus = _write_corpus(tmp_path / "corpus.jsonl", 40)
    progress = tmp_path / "progress.jsonl"
    summary = run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=tmp_path / "seed.pt",
        progress_path=progress,
        checkpoint_every=100_000,
        progress_every=20,
        max_symbols=60,
    )
    final = _entries(progress)[-1]
    assert final["exit_reason"] == "max_symbols_reached", final
    assert final["budget_max_symbols"] == 60, final
    assert final["reached_budget"] is True, final
    assert final["ticks_at_exit"] == final["base_ticks"] + 60, final
    record = json.loads(exit_record_path(progress).read_text(encoding="utf-8"))
    assert {k: record[k] for k in final} == final, (record, final)
    assert record["corpus_fingerprint"] and record["checkpoint_path"].endswith("seed.pt")
    assert record["ticks"] == summary["ticks"]


def test_exit_record_names_the_bytes_it_wrote(tmp_path: Path) -> None:
    """DEBT-G39：退出记账必须自述**它写下的是哪些字节**，且那个哈希等于落盘件的当前哈希。

    来历：`--keep-checkpoints on` 下 `checkpoint.pt` 每几千 tick 重写一次，所以"件存在"不表示"跑完了"，
    而记账里没有哈希时，下游只能靠"跨 90 秒两次取样相同"这种间接办法绑字节（2026-10-03 一枚中途件
    因此被当成训后读数跑了一遍，靠探针自己的 `base_sha256_unchanged=false` 才被抓住）。
    这条守卫同时也是顺序守卫：三处退出点若还是"先 `_flush` 再 `_persist`"，记的就是**上一次**保存的字节。
    """

    corpus = _write_corpus(tmp_path / "sha.jsonl", 40)
    progress = tmp_path / "sha_progress.jsonl"
    checkpoint = tmp_path / "sha.pt"
    run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=checkpoint,
        progress_path=progress,
        checkpoint_every=100_000,
        progress_every=20,
        max_symbols=60,
    )
    record = json.loads(exit_record_path(progress).read_text(encoding="utf-8"))
    assert checkpoint.is_file(), "先落盘再写记账：退出时终件必须在场"
    on_disk = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert record["checkpoint_sha256"] == on_disk, record.get("checkpoint_sha256")
    assert len(record["checkpoint_sha256"]) == 64
    # 这条自述只属于独立件，不许漂进进度行（下一支测试钉的是进度行的键集）
    assert "checkpoint_sha256" not in _entries(progress)[-1]


def test_corpus_exhaustion_is_not_reported_as_budget(tmp_path: Path) -> None:
    """耗尽支：没给预算 ⇒ 必须写 `corpus_exhausted` 且 `budget_max_symbols=None`。"""

    corpus = _write_corpus(tmp_path / "tiny.jsonl", 2)
    progress = tmp_path / "p.jsonl"
    run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=tmp_path / "tiny.pt",
        progress_path=progress,
        checkpoint_every=100_000,
        progress_every=100_000,
    )
    final = _entries(progress)[-1]
    assert final["exit_reason"] == "corpus_exhausted", final
    assert final["budget_max_symbols"] is None, final
    assert final["reached_budget"] is False, final


def test_periodic_lines_keep_their_old_shape(tmp_path: Path) -> None:
    """加性：新键只许出现在收尾那一行——周期行必须还是那七个旧键。"""

    corpus = _write_corpus(tmp_path / "c2.jsonl", 40)
    progress = tmp_path / "p2.jsonl"
    run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=tmp_path / "c2.pt",
        progress_path=progress,
        checkpoint_every=100_000,
        progress_every=15,
        max_symbols=60,
    )
    entries = _entries(progress)
    assert len(entries) >= 2, entries
    assert set(entries[0]) == LEGACY_KEYS, set(entries[0])
    assert "exit_reason" not in entries[0]
    assert set(entries[-1]) == LEGACY_KEYS | {
        "exit_reason",
        "base_ticks",
        "budget_max_symbols",
        "ticks_at_exit",
        "reached_budget",
    }, set(entries[-1])


def test_per_answer_branch_is_accounted_too(tmp_path: Path) -> None:
    """两台预算支是**两处代码**（分块档每步跳 `len(chunk)+2`）⇒ 两条都得落记账。"""

    corpus = _write_qa_corpus(tmp_path / "pa.jsonl", 40)
    progress = tmp_path / "pa.jsonl"
    run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=tmp_path / "pa.pt",
        progress_path=progress,
        checkpoint_every=100_000,
        progress_every=1,
        max_symbols=300,
        answer_chunking="per-answer",
    )
    final = _entries(progress)[-1]
    assert final["exit_reason"] == "max_symbols_reached", final
    assert final["reached_budget"] is True, final
    assert exit_record_path(progress).is_file()


def test_summary_stays_float_only_for_the_p3b_consumer(tmp_path: Path) -> None:
    #: 别线在飞的工具按 `{key: float(value)}` 消费返回值 ⇒ 退出原因进了 summary 就会炸它。
    corpus = _write_corpus(tmp_path / "f.jsonl", 2)
    summary = run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=tmp_path / "f.pt",
        progress_path=tmp_path / "f.jsonl",
        checkpoint_every=100_000,
        progress_every=100_000,
    )
    assert set(summary) == {"ticks", "parameters"}, summary
    assert all(isinstance(value, float) for value in summary.values()), summary
    assert {key: float(value) for key, value in summary.items()} == summary


def test_exit_record_name_is_per_progress_file(tmp_path: Path) -> None:
    """两臂共用一个目录时不许互相覆盖记账件。"""

    first = exit_record_path(tmp_path / "arm_a_progress.jsonl")
    second = exit_record_path(tmp_path / "arm_b_progress.jsonl")
    assert first != second, (first, second)
    assert first.name == "arm_a_progress_exit.json", first.name

"""DEBT-G63 修法④ 的契约测：`history_*` 三键的 **None 与 0 语义分离**（回读发现语义已在场，这里补能为假的测）。

㊵-550 的回读结论：`output/n3a_control_x2/progress_exit.json`（`--keep-checkpoints off` 档）
现读 `history_files=None`、`history_bytes=None`、`history_pruned=None` ⇒
"没给保号存档目录"出版 `None`，而不是被 `len([])` 抹成 `0`。这个区分不是洁癖：
`0` 会说"目录里一枚都没留下"（存档功能坏了），`None` 说"这一档根本没有存档这回事"——
把它们混成 `0` 就等于给"存档静默失灵"造了一个看起来正常的读数（DEBT-G63 本体的同族）。

两支都走：①不给 `keep_history` ⇒ 三键都是 `None` 且**不是** int；
②给 ⇒ 三键都是 int，且实际目录里的枚数与 `history_files` 同数（生产者的现数不自述配置值）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.training import train_seed_corpus as trainer

KEYS = ("history_files", "history_bytes", "history_pruned")

DOCS = [
    "问：甲是什么？答：甲是一篇用来测保号存档的最小文字。",
    "问：乙呢？答：乙是第二篇文字，长度与甲接近但不完全相同。",
]


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps({"text": doc}) + "\n" for doc in DOCS),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _exit_record(tmp_path: Path, **over: Any) -> dict[str, Any]:
    from seed import SeedConfig

    progress = tmp_path / "progress.jsonl"
    trainer.run_training(
        corpus_paths=[_corpus(tmp_path)],
        config=SeedConfig(),
        epochs=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=progress,
        checkpoint_every=20,
        progress_every=1_000_000,
        max_symbols=120,
        readout="predictive",
        **over,
    )
    return json.loads(trainer.exit_record_path(progress).read_text(encoding="utf-8"))


def test_no_history_dir_publishes_none_not_zero(tmp_path: Path) -> None:
    record = _exit_record(tmp_path)
    for key in KEYS:
        assert record[key] is None, (key, record[key])
        #: 反面：不许被 `len([])`／`sum([])` 抹成整数零。
        assert not isinstance(record[key], int), key


def test_given_history_dir_publishes_live_counts(tmp_path: Path) -> None:
    history = tmp_path / "history"
    history.mkdir()
    record = _exit_record(tmp_path, keep_history=history, keep_history_max=None)
    kept = sorted(history.glob("checkpoint_*.pt"))
    for key in KEYS:
        assert isinstance(record[key], int), (key, record[key])
    assert record["history_files"] == len(kept), (record, kept)
    assert record["history_files"] >= 1
    assert record["history_bytes"] == sum(path.stat().st_size for path in kept)
    assert record["history_pruned"] == 0


def test_history_keys_are_named_exactly_once_in_the_record(tmp_path: Path) -> None:
    record = _exit_record(tmp_path)
    assert all(key in record for key in KEYS), sorted(record)
    #: 键名不许漂（下游按这三个名字取数）。
    assert len([key for key in record if key.startswith("history_")]) == 3


@pytest.mark.parametrize("key", KEYS)
def test_history_keys_belong_to_the_independent_record_only(tmp_path: Path, key: str) -> None:
    """这三键**只属于独立 exit 件**，不许漂进进度件收尾那一行（㊵-545 那条"同源同值"的边界）。

    我第一版把这条测写反了（断言"两张面同值"），当场红在 `KeyError`——红的是我的假设，
    不是行为：`train_seed_corpus.py` 的 `history_*` 从来只写进独立件，
    既有守卫 `test_exit_record_names_the_bytes_it_wrote` 也钉着"这类自述不进进度行"。
    按现行形状改写，而不是为了让测过而放宽。
    """
    from seed import SeedConfig

    progress = tmp_path / "progress.jsonl"
    trainer.run_training(
        corpus_paths=[_corpus(tmp_path)],
        config=SeedConfig(),
        epochs=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=progress,
        checkpoint_every=20,
        progress_every=1_000_000,
        max_symbols=120,
        readout="predictive",
    )
    lines = [json.loads(row) for row in progress.read_text(encoding="utf-8").splitlines() if row]
    face = json.loads(trainer.exit_record_path(progress).read_text(encoding="utf-8"))
    assert key in face, sorted(face)
    assert key not in lines[-1], sorted(lines[-1])

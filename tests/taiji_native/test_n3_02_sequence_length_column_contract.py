"""PLAN-N3-02 §8.7 的"序列长度"一列：由**发符处现数**，且必须能被判读器读到。

取法是冻着的（PLAN-N3-02 §4quater 那一行）："每篇文档的符号数分布（min/median/max），
可由 `iter_corpus_symbols` 在边界符处累加得到 ⇒ **不许用 `max_symbols` 参数值冒充实测分布**"。
这一格之前，判读器 `adjudicate_taiji_n3a_scaling_probe.py:36` 按 `("exit","sequence_length")`
取它而取不到 ⇒ 两臂的 `J_N3a` 全锁在 `ran_not_measured`（台账 08 ㊵-543）。

五支：①空档出版 `None` 而不是 0（DEBT-G63 那条同族纪律）；②默认支按篇现数、
逐篇字节可复算；③形状 B（K 封顶循环重用）也数得到，且重复篇逐条在场；
④收尾行出版这列、周期行不带这列，且它的 `max` 不等于符号预算值（防"参数冒充实测"）；
⑤把同一份产物直接喂给判读器 ⇒ §8.7 五项全为 `True`、`measurement_complete` 为真。
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.training import adjudicate_taiji_n3a_scaling_probe as judge  # noqa: E402
from scripts.training import train_seed_corpus as trainer  # noqa: E402

DOCS = [
    "问：甲是什么？答：甲是一个符号串。",
    "问：乙呢？答：乙是另一个符号串，比甲长一些。",
    "问：丙呢？答：丙是第三篇。",
]
BOUNDARY = 1


def _corpus(tmp_path: Path, docs: list[str] = DOCS) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps({"text": doc}) + "\n" for doc in docs),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _expected_lengths(docs: list[str], *, end_after_newline: bool = False) -> list[int]:
    #: 一篇的发符数＝1 个边界符＋正文 UTF-8 字节（`end_boundary_after_newline` 再 +1 个 0x0A）。
    return [1 + len(doc.encode("utf-8")) + (1 if end_after_newline else 0) for doc in docs]


def test_empty_window_publishes_none_not_zero() -> None:
    counters = trainer.DocumentStreamCounters()
    assert counters.sequence_length_stats() is None
    #: 反面：不许把"一篇都没喂"出版成 min=0／mean=0 这种看起来像读数的东西。
    stats = counters.sequence_length_stats()
    assert not isinstance(stats, dict)


def test_default_branch_counts_each_document(tmp_path: Path) -> None:
    counters = trainer.DocumentStreamCounters()
    symbols = list(
        trainer.iter_corpus_symbols([_corpus(tmp_path)], boundary=BOUNDARY, counters=counters)
    )
    expected = _expected_lengths(DOCS)
    stats = counters.sequence_length_stats()
    assert stats is not None
    assert stats["documents_counted"] == len(expected)
    assert stats["min"] == min(expected)
    assert stats["max"] == max(expected)
    assert stats["median"] == float(statistics.median(expected))
    assert stats["mean"] == round(sum(expected) / len(expected), 6)
    #: 交叉核对：逐篇计数之和必须等于流出去的符号数——两条独立取法不一致就停手。
    assert sum(expected) == len(symbols)


def test_capped_pool_branch_counts_revisits_too(tmp_path: Path) -> None:
    counters = trainer.DocumentStreamCounters()
    stream = trainer.iter_corpus_symbols(
        [_corpus(tmp_path)],
        boundary=BOUNDARY,
        max_unique_documents=2,
        counters=counters,
    )
    #: 取的是**符号数**（不是篇数）：K=2 ⇒ 池是前两条，取两整轮。
    #: `islice` 只拉够就走（不像自建包装器会多拉一个符号、把下一篇"进入"进去）——
    #: 计数点在进篇那一刻，多拉一个就会多出一篇，实测过：`5 != 4`。
    per_round = _expected_lengths(DOCS[:2])
    taken = list(itertools.islice(stream, sum(per_round) * 2))
    assert len(taken) == sum(per_round) * 2
    stats = counters.sequence_length_stats()
    assert stats is not None
    assert stats["documents_counted"] == 4
    assert counters.document_symbol_counts == per_round * 2


def test_exit_row_publishes_the_column_and_periodic_lines_do_not(tmp_path: Path) -> None:
    from seed import SeedConfig

    budget = 120
    progress = tmp_path / "progress.jsonl"
    trainer.run_training(
        corpus_paths=[_corpus(tmp_path)],
        config=SeedConfig(),
        epochs=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=progress,
        checkpoint_every=1_000_000,
        progress_every=10,
        max_symbols=budget,
        readout="predictive",
    )
    lines = [json.loads(row) for row in progress.read_text(encoding="utf-8").splitlines() if row]
    periodic, exit_row = lines[0], lines[-1]
    assert "sequence_length" not in periodic, sorted(periodic)
    stats = exit_row["sequence_length"]
    assert isinstance(stats, dict), stats
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["documents_counted"] >= 1
    #: 两条独立取法必须一致：逐篇计数的条数＝`document_visits`（同一生成器上的另一只计数器）。
    assert stats["documents_counted"] == exit_row["document_visits"], (stats, exit_row)
    #: 这一列必须是**实测分布**：`max` 恰等于符号预算就是这个数被参数冒充了的指纹。
    assert stats["max"] != budget, stats
    #: 独立 exit 件与收尾行同源同值（DEBT-G39 那条纪律的同一份 entry）。
    from_exit = json.loads(trainer.exit_record_path(progress).read_text(encoding="utf-8"))
    assert from_exit["sequence_length"] == stats


def test_adjudicator_reads_all_five_section_8_7_items(tmp_path: Path) -> None:
    from seed import SeedConfig

    progress = tmp_path / "progress.jsonl"
    trainer.run_training(
        corpus_paths=[_corpus(tmp_path)],
        config=SeedConfig(),
        epochs=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=progress,
        checkpoint_every=1_000_000,
        progress_every=10,
        max_symbols=120,
        readout="predictive",
    )
    payload = judge.judge_arm(progress, trainer.exit_record_path(progress))
    #: ㊵-543 那把锁的形状：五项里只有 `sequence_length` 为 False。
    completeness = payload["section_8_7_completeness"]
    assert all(completeness.values()), completeness
    assert payload["measurement_complete"] is True
    assert payload["exit_record"]["sequence_length"] is not None

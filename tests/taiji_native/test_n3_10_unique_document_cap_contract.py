"""PLAN-N3-10 形状 B（`--max-unique-documents`）的契约测。

这支旗标的全部意义是"把必然跟着变的重复率报出来，而不是假装它相等"，所以测的是**三轴自述能不能为假**：

* 默认关＝今天的形状逐字不变（取篇顺序、每篇前置边界符），且三轴退化成 `mean_revisits == 1.0`；
* 给 `K`＝只取前 K 篇成池、之后循环重用，`unique_documents` 必须停在 `K` 而 `document_visits` 继续涨；
* 篇池取空（语料为空）⇒ 响亮 `ValueError`，不是静默零符号；
* 三轴的键名被钉住——少一个键就等于把那一轴放回隐藏变量；
* 旗标名必须出现在 `--help` 里（"旗标被走到"的最低证据，不让它成为一个只存在于源码里的开关）。

语料夹具一律落 pytest 的 `tmp_path`，不碰仓内语料与产品件。
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/train_seed_corpus.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("n3_10_trainer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = _load_module()


def _corpus(tmp: Path, texts: list[str]) -> list[Path]:
    path = tmp / "corpus.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return [path]


def _symbols(tmp: Path, **kwargs: Any) -> tuple[list[int], Any]:
    counters = TRAINER.DocumentStreamCounters()
    stream = TRAINER.iter_corpus_symbols(
        _corpus(tmp, ["甲一", "乙二", "丙三", "丁四"]),
        boundary=1,
        counters=counters,
        **kwargs,
    )
    return list(itertools.islice(stream, 0, 200)), counters


def test_uncapped_stream_is_the_old_shape(tmp_path: Path) -> None:
    symbols, counters = _symbols(tmp_path)
    expected: list[int] = []
    for text in ["甲一", "乙二", "丙三", "丁四"]:
        expected += [1, *list(text.encode("utf-8"))]
    assert symbols == expected[: len(symbols)]
    #: 不限＝每篇恰好到一次 ⇒ 重复率这条轴自己退化成 1.0（不是缺键、也不是 0）。
    assert counters.as_dict() == {
        "unique_documents": 4,
        "document_visits": 4,
        "mean_revisits": 1.0,
    }


def test_capped_stream_cycles_the_pool_and_reports_revisits(tmp_path: Path) -> None:
    symbols, counters = _symbols(tmp_path, max_unique_documents=2)
    first: list[int] = []
    for text in ["甲一", "乙二"]:
        first += [1, *list(text.encode("utf-8"))]
    #: 200 个符号足够绕池好几圈 ⇒ 前两段必须完全等于"前两篇"的重复。
    assert symbols[: len(first) * 2] == first + first
    #: 池内唯一篇数停在 2，到达篇次继续涨 ⇒ 这正是"符号数同、数据不扩"必然带来的第二轴。
    assert counters.unique_documents == 2
    assert counters.document_visits > 2
    assert counters.as_dict()["mean_revisits"] == round(counters.document_visits / 2, 6)


def test_empty_corpus_with_a_cap_refuses_loudly(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    stream = TRAINER.iter_corpus_symbols(
        [empty], boundary=1, max_unique_documents=3, counters=TRAINER.DocumentStreamCounters()
    )
    try:
        next(stream)
    except ValueError as error:
        assert "语料空" in str(error)
    else:
        raise AssertionError("空语料配 K 必须响亮拒绝，不能静默产出零符号流")


def test_three_axis_keys_are_pinned() -> None:
    #: 少任何一轴，PLAN-N3-10 §2 的"不许把重复率当隐藏变量"就失效。
    assert list(TRAINER.DocumentStreamCounters().as_dict()) == [
        "unique_documents",
        "document_visits",
        "mean_revisits",
    ]


def test_flag_is_reachable_from_the_cli() -> None:
    proc = subprocess.run(  # noqa: S603 - 仓内固定脚本，参数不含用户输入
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "--max-unique-documents" in proc.stdout

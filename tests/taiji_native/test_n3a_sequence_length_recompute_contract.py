"""`recompute_taiji_n3a_sequence_length.py` 的契约测（PLAN-N3-13 的三条守卫都要能为假）。

这台仪器只干一件事：**不碰模型**，用同一个 `iter_corpus_symbols` 重放某臂当时那条取数流，
把 §8.7 第四项"序列长度"的实测分布补出来。它凭什么可以被当成"当时那条流"？就凭三条并立守卫：

① 语料指纹等于收尾件里那串（DEBT-G64 同族：跨语料就是换尺子）；
② `--max-symbols` 等于收尾件的 `budget_max_symbols`；
③ 重放出来的 `document_visits` 等于收尾件自述的篇数——计数器只有那条生成器会写，
   所以这一条是"我走的确实是同一条流"的直接证据。

外加一条形状守卫：流比预算短 ⇒ `stream_shorter_than_budget`（这档根本没吃满，
分布不能代表那条流）。六支各有反面，**不为假的那一侧都实测过**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.training import recompute_taiji_n3a_sequence_length as rec  # noqa: E402
from scripts.training import train_seed_corpus as trainer  # noqa: E402

DOCS = [
    "问：甲是什么？答：甲是一篇很短的文字。",
    "问：乙呢？答：乙是另一篇文字，比甲长一些，用来让中位数不是端点值。",
    "问：丙呢？答：丙是第三篇文字，长度介于甲与乙之间，用来把均值拉开。",
]


def _lengths() -> list[int]:
    #: 与训练器同一式子：1 个边界符＋正文 UTF-8 字节＋(旗标开时) 1 个 0x0A。
    return [1 + len(doc.encode("utf-8")) + 1 for doc in DOCS]


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps({"text": doc}) + "\n" for doc in DOCS),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _exit(corpus: Path, **over: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "document_visits": len(_lengths()),
        "budget_max_symbols": sum(_lengths()),
        "corpus_fingerprint": trainer.corpus_fingerprint([corpus]),
    }
    record.update(over)
    return record


def test_positive_replay_matches_hand_computed_lengths(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    lengths = _lengths()
    payload = rec.recompute(_exit(corpus), corpus, max_symbols=sum(lengths))
    assert payload["status"] == "ok", payload["checks"]
    assert all(bool(value) for value in payload["checks"].values())
    assert payload["symbols_taken"] == sum(lengths)
    stats = payload["sequence_length"]
    assert stats["documents_counted"] == 3
    assert stats["min"] == min(lengths)
    assert stats["max"] == max(lengths)
    assert stats["mean"] == round(sum(lengths) / 3, 6)
    assert payload["format"] == "taiji-n3a-sequence-length-recompute-v1"
    #: 两条独立取法必须一致：逐篇长度之和＝重放吃到的符号数。
    assert sum(lengths) == payload["symbols_taken"]


def test_other_corpus_fingerprint_is_rejected(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"text": DOCS[0]}) + "\n", encoding="utf-8", newline="\n")
    payload = rec.recompute(
        _exit(corpus, corpus_fingerprint=trainer.corpus_fingerprint([other])),
        corpus,
        max_symbols=sum(_lengths()),
    )
    assert payload["status"] == "guard_failed"
    assert payload["checks"]["corpus_fingerprint_matches_exit"] is False


def test_budget_mismatch_is_rejected(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    payload = rec.recompute(
        _exit(corpus, budget_max_symbols=sum(_lengths()) + 1),
        corpus,
        max_symbols=sum(_lengths()),
    )
    assert payload["status"] == "guard_failed"
    assert payload["checks"]["max_symbols_matches_exit_budget"] is False


def test_visit_count_mismatch_is_rejected(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    #: 收尾件自称 999 篇而重放只有 3 篇 ⇒ 不是同一条流，不许出版。
    payload = rec.recompute(_exit(corpus, document_visits=999), corpus, max_symbols=sum(_lengths()))
    assert payload["status"] == "guard_failed"
    assert payload["checks"]["replayed_visits_match_exit"] is False
    assert payload["replayed_document_visits"] == 3


def test_stream_shorter_than_budget_is_its_own_status(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    total = sum(_lengths())
    payload = rec.recompute(
        _exit(corpus, document_visits=3, budget_max_symbols=total + 500),
        corpus,
        max_symbols=total + 500,
    )
    assert payload["status"] == "stream_shorter_than_budget"
    assert payload["symbols_taken"] == total


def test_main_writes_the_sidecar_and_reports_rc(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    total = sum(_lengths())
    exit_path = tmp_path / "arm_exit.json"
    exit_path.write_text(
        json.dumps(_exit(corpus), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    good = tmp_path / "seq.json"
    rc = rec.main(
        [
            "--exit",
            str(exit_path),
            "--corpus",
            str(corpus),
            "--max-symbols",
            str(total),
            "--out",
            str(good),
        ]
    )
    assert rc == 0
    assert json.loads(good.read_text(encoding="utf-8"))["status"] == "ok"
    #: 落盘字节必须是 LF：这台仪器第一版没给 write_text 的 newline 参数，
    #: 在本机写出过 CRLF 件（实测两份件各 29 处行尾）；入库件一旦被翻成 CRLF，
    #: 就会红在按行尾取数的下游仪器上——所以这条守卫按字节量。
    assert b"\r\n" not in good.read_bytes(), good.read_bytes()[:40]

    bad = tmp_path / "seq_bad.json"
    rc = rec.main(
        [
            "--exit",
            str(exit_path),
            "--corpus",
            str(corpus),
            "--max-symbols",
            str(total + 7),
            "--out",
            str(bad),
        ]
    )
    assert rc == 2
    #: 守卫不齐也要**落件**（点名是哪条不齐），但判读器那边不认它。
    assert json.loads(bad.read_text(encoding="utf-8"))["status"] == "guard_failed"

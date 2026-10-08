"""N3 甲 §8.7 五项取数面的契约测（`audit_taiji_n3a_data_face.py`）。

两支都必须走到：
* **正支**＝合成小语料上五键齐、且"到达几篇 episode"要等于按消耗符号数推出来的那个数
  （钉住口径，防止 `reached` 变成永远等于总数的装饰）；
* **反支**＝缺 progress 文件 ⇒ rc=2 并点名缺哪一项——"没算出来"不许被读成"没问题"。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/audit_taiji_n3a_data_face.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("n3a_data_face", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _corpus(tmp_path: Path) -> Path:
    """三篇文档，字节长度分别 5／10／20（纯 ASCII，长度可预期）。"""

    docs = ["abcde", "0123456789", "x" * 20]
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps({"text": doc}) + "\n" for doc in docs), encoding="utf-8", newline="\n"
    )
    return path


def _progress(tmp_path: Path, consumed: int) -> Path:
    """造一份进度件：首行 ticks=consumed、window_ticks=consumed ⇒ base_ticks=0。"""

    path = tmp_path / "progress.jsonl"
    rows = [
        {"epoch": 0, "ticks": consumed, "window_ticks": consumed, "online_accuracy": 0.1},
        {"epoch": 0, "ticks": consumed, "window_ticks": 0, "online_accuracy": 0.2},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return path


def test_five_fields_present_and_episode_count_matches_consumed_symbols(tmp_path: Path) -> None:
    module = _load_module()
    corpus = _corpus(tmp_path)
    #: 消耗 6 个符号＝第一篇(5+边界1)刚好走完 ⇒ 只到达 1 篇 episode。
    progress = _progress(tmp_path, 6)

    face, rc = module.build_face([corpus], progress)

    assert rc == 0, face
    assert face["complete"] is True
    for key in (
        "corpus_available_bytes",
        "unique_episodes_reached",
        "weight_update_steps",
        "sequence_length_bytes",
        "holdout_disjointness",
    ):
        assert face.get(key) is not None, key
    assert face["corpus_documents_total"] == 3
    assert face["weight_update_steps"] == 6
    assert face["unique_episodes_reached"] == 1
    assert face["sequence_length_bytes"]["min"] == 5
    assert face["sequence_length_bytes"]["max"] == 20
    #: 探针窗口在这么小的 ASCII 语料里不该命中——命中了就得重新看这条独立性自证的口径。
    assert face["holdout_disjointness"]["probe_windows"] > 0
    assert face["holdout_disjointness"]["hit_rate"] == 0.0


def test_more_symbols_reach_more_documents(tmp_path: Path) -> None:
    """同一份语料，消耗到 17 个符号时只能到达 2 篇（第三篇 20+1 还没吃完）。"""

    module = _load_module()
    face, rc = module.build_face([_corpus(tmp_path)], _progress(tmp_path, 17))
    assert rc == 0
    assert face["unique_episodes_reached"] == 2
    assert face["episodes_total"] == 3


def test_missing_progress_is_a_loud_refusal(tmp_path: Path) -> None:
    module = _load_module()
    face, rc = module.build_face([_corpus(tmp_path)], tmp_path / "no_such_progress.jsonl")

    assert rc == 2
    assert face["complete"] is False
    assert "progress_rows" in face["missing"]


def test_missing_corpus_file_is_named(tmp_path: Path) -> None:
    module = _load_module()
    face, rc = module.build_face([tmp_path / "absent.jsonl"], _progress(tmp_path, 6))

    assert rc == 2
    assert "corpus_files" in face["missing"]


def test_probe_window_constant_is_documented() -> None:
    """disjointness 的窗口长度是这条自证的全部灵敏度所在——它必须写在件里且不为 0。"""

    module = _load_module()
    assert module.NGRAM_BYTES >= 8

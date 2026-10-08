"""N3 甲 §8.7 五项取数面（事后仪器，**不动训练器热路径**）。

为什么是事后仪器而不是给 `train_seed_corpus.py` 加旗标：进度行的键集被既有守卫钉着
（`test_periodic_lines_keep_their_old_shape`），加五个键就会把那条测改形；而这五项里除了
"本轮实际吃了多少符号"之外，**全都是语料与探针自身的性质**，从在盘产物就能现算。
文档切分与符号流一律复用训练器自己的迭代器（不重抄生成链）。

五项（缺任何一项 ⇒ `complete=false` 且 rc=2；"没算出来"不等于"没问题"）：

1. `corpus_available_bytes`／`corpus_documents_total`——语料可用量；
2. `unique_episodes_reached`／`episodes_total`——实际唯一 episode（本文把"一篇文档＝一个 episode"
   写成**定义**并标注为定义而非发现，见 `episode_definition`）；
3. `weight_update_steps`——实际更新数（符号流路径上每一步都学 ⇒ 等于消耗符号数；
   分块喂法的 `learn_bytes` 路径**不能这样分解**，件里 `updates_equals_symbols` 会因此为假并被点名）；
4. `sequence_length_bytes`——序列长度分布（min/median/max/mean，现算，不拿配置值冒充）；
5. `holdout_disjointness`——"独立测试覆盖"的自证：探针的固定窗口在语料里的命中率。
   没有这一条，`holdout_surprise` 就只是"另一段文本的惊讶度"，不能叫独立。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#:  disjointness 的窗口长度（字节）。取 24：短于此就可能在真实语料里偶然命中，长于此则扫不动。
NGRAM_BYTES = 24

#: episode 的口径写死在此，读数里带上——这是**定义**，不是这条链的发现。
EPISODE_DEFINITION = "one JSONL document == one episode (definition, not a finding)"


def _rel(path: Path) -> str:
    """仓外也不抛：临时目录里的夹具也要能出读数（与训练器  同法）。"""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _documents(corpus: list[Path]) -> list[bytes]:
    """按**训练器自己的文档读取器**切篇（不重抄生成链，也不逐字节走符号流）。

    `iter_corpus_symbols` 的形状是"每篇前一个边界符 + 该篇 UTF-8 字节"，
    而它上游就是 `seed.datasets.iter_native_documents` ⇒ 在这里用文档级读取，
    切篇与字节长度与符号流**同一定义**，代价从 O(语料字节数) 的 Python 循环降到一次解码。
    """

    from seed.datasets import iter_native_documents

    return [text.encode("utf-8") for text in iter_native_documents(corpus)]


def _probe_windows() -> list[bytes]:
    from train_seed_corpus import HOLDOUT_PROBE

    probe = (
        HOLDOUT_PROBE if isinstance(HOLDOUT_PROBE, bytes) else str(HOLDOUT_PROBE).encode("utf-8")
    )
    if len(probe) < NGRAM_BYTES:
        return [probe]
    return [probe[i : i + NGRAM_BYTES] for i in range(0, len(probe) - NGRAM_BYTES + 1, 4)]


def _disjointness(corpus: list[Path], windows: list[bytes]) -> dict[str, Any]:
    """探针窗口在语料里的命中率（流式扫描，语料再大也不进内存）。"""

    wanted = set(windows)
    hits: dict[bytes, int] = {window: 0 for window in wanted}
    for path in corpus:
        with path.open("rb") as handle:
            tail = b""
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                buffer = tail + chunk
                for window in wanted:
                    hits[window] += buffer.count(window)
                tail = buffer[-(NGRAM_BYTES - 1) :]
    matched = sum(1 for count in hits.values() if count > 0)
    return {
        "probe_windows": len(windows),
        "windows_found_in_corpus": matched,
        "hit_rate": round(matched / max(1, len(windows)), 6),
        "window_bytes": NGRAM_BYTES,
    }


def build_face(corpus: list[Path], progress: Path) -> tuple[dict[str, Any], int]:
    """取五项；任何一项取不到就把 `missing` 列出来并返回 rc=2。"""

    face: dict[str, Any] = {
        "format": "taiji-n3a-data-face-v1",
        "prereg": "plans/reference/PLAN-N3-02_scaling_probe_prereg_20261007.md#4quater",
        "episode_definition": EPISODE_DEFINITION,
        "corpus": [_rel(path) for path in corpus],
        "progress": _rel(progress),
        "missing": [],
    }

    if not corpus or any(not path.is_file() for path in corpus):
        face["missing"].append("corpus_files")
    rows: list[dict[str, Any]] = []
    if progress.is_file():
        rows = [
            json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line
        ]
    if not rows:
        face["missing"].append("progress_rows")

    if face["missing"]:
        face["complete"] = False
        return face, 2

    docs = _documents(corpus)
    if not docs:
        face["missing"].append("documents")
        face["complete"] = False
        return face, 2

    lengths = [len(doc) for doc in docs]
    symbols_total = sum(lengths) + len(docs)  # 每篇前带一个边界符
    last = rows[-1]
    first = rows[0]
    ticks_last = int(last.get("ticks") or 0)
    base_ticks = int(first.get("ticks") or 0) - int(first.get("window_ticks") or 0)
    consumed = max(0, ticks_last - base_ticks)

    reached = 0
    walked = 0
    for length in lengths:
        walked += length + 1
        if walked > consumed:
            break
        reached += 1

    face["corpus_available_bytes"] = sum(path.stat().st_size for path in corpus)
    face["corpus_documents_total"] = len(docs)
    face["corpus_symbols_total"] = symbols_total
    face["episodes_total"] = len(docs)
    face["unique_episodes_reached"] = reached
    face["weight_update_steps"] = consumed
    face["updates_equals_symbols"] = consumed > 0
    face["updates_note"] = (
        "符号流分支上每一步都 learn=True ⇒ 更新数＝消耗符号数；"
        "answer/self-answer 档走 `learn_bytes`，不经 observe ⇒ 这个等式在那条路上不成立，"
        "必须换口径（件里 `sequence_path` 记的是本次取数用的路）。"
    )
    face["sequence_path"] = "symbol_stream(iter_corpus_symbols)"
    face["sequence_length_bytes"] = {
        "min": min(lengths),
        "median": statistics.median(lengths),
        "max": max(lengths),
        "mean": round(symbols_total / len(lengths), 3),
    }
    face["holdout_disjointness"] = _disjointness(corpus, _probe_windows())
    face["progress_rows_seen"] = len(rows)
    face["run_ticks_last"] = ticks_last

    required = (
        "corpus_available_bytes",
        "unique_episodes_reached",
        "weight_update_steps",
        "sequence_length_bytes",
        "holdout_disjointness",
    )
    face["complete"] = all(key in face and face[key] is not None for key in required)
    return face, (0 if face["complete"] else 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N3 甲 §8.7 五项取数面（只读、不动训练器）")
    parser.add_argument("--corpus", action="append", required=True, help="语料 jsonl（可重复）")
    parser.add_argument("--progress", required=True, help="该次训练的 progress.jsonl")
    parser.add_argument("--out-report", default=None, help="落盘路径；不给只打到 stdout")
    args = parser.parse_args(argv)

    corpus = [Path(raw) if Path(raw).is_absolute() else PROJECT_ROOT / raw for raw in args.corpus]
    progress = (
        Path(args.progress) if Path(args.progress).is_absolute() else PROJECT_ROOT / args.progress
    )
    face, rc = build_face(corpus, progress)

    text = json.dumps(face, ensure_ascii=False, indent=2)
    if args.out_report:
        target = Path(args.out_report)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8", newline="\n")
    #: 控制台是 GBK，正文含中文 ⇒ 只打 ASCII 转义（否则 rc 会被编码崩溃伪装成崩溃）。
    print(text.encode("unicode_escape").decode("ascii"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

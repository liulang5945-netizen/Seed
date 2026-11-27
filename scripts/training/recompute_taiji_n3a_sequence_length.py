"""§8.7 第四项"序列长度"的**同源复算**入口（零训练跑量：只重放取数生成器，不碰模型）。

为什么要它：两臂在盘的收尾件是 ㊵-545 之前的代码写的，里面没有 `sequence_length` 这一列
⇒ 判读器按 PLAN-N3-02 §2 把两臂锁在 `ran_not_measured`。重跑两臂≈2.3 h/臂（改权重，要批），
而这一项本来就是**取数层的性质**（哪些篇、每篇多少符号），可以用同一个
`iter_corpus_symbols` 原样重放得到——㊵-494 立过这个先例（"事后只读仪器与当时算同一口径"）。

三条并立的在场性守卫（不齐就 rc=2，不出版数）：
① **同一份语料**：给定的 `--corpus` 的 `corpus_fingerprint` 必须等于收尾件里那串
   （DEBT-G64 同族：跨语料的对照会静默换尺子）；
② **同一条预算**：`--max-symbols` 必须等于收尾件的 `budget_max_symbols`；
③ **同一篇数**：重放出来的 `document_visits` 必须等于收尾件自述的 `document_visits`
   ——这一条是本件唯一的"我走的确实是当时那条流"的证据，因为计数器只有那条生成器会写。

已知不能判别的一件事（写在件里，不藏）：`end_boundary_after_newline` 这条旗标**不在**收尾件里
（收尾件只自述指纹／预算／篇数），本件默认按 CLI 默认值 True 复算（旗标 `--no-...` 是
`action="store_false"` 且未给 `default` ⇒ argparse 默认 True，`train_seed_corpus.py:1058-1066`），
而甲乙两臂冻过的命令串里都没有这个旗标 ⇒ 取默认。偏差的代价是每篇 ±1 符号，
上面第③条**分不出**这个 ±1（315 篇只挪 ~1 篇的切点），所以这里如实标为不可判别。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from train_seed_corpus import (  # noqa: E402
    DocumentStreamCounters,
    corpus_fingerprint,
    iter_corpus_symbols,
)

FORMAT = "taiji-n3a-sequence-length-recompute-v1"

CAVEATS = (
    "end_boundary_after_newline 不在收尾件的自述里；本件按 CLI 默认 True 复算"
    "（两臂冻过的命令串里没有该旗标），代价是每篇 ±1 符号，且篇数守卫分不出它。",
    "计数点在进篇那一刻 ⇒ 被预算切断的最后一篇按它自身的长度计入，"
    "与 `document_visits` 同数；吃到第几符号另有 `ticks_at_exit`。",
)


def recompute(
    exit_record: dict[str, Any],
    corpus: Path,
    *,
    max_symbols: int,
    max_unique_documents: int | None = None,
    end_boundary_after_newline: bool = True,
    boundary: int | None = None,
) -> dict[str, Any]:
    """重放取数流并出 §8.7 那一列；三条守卫任一不齐 ⇒ `status` 非 `ok` 且带上拒因。"""

    from taiji import TaijiConfig

    if boundary is None:
        #: 边界符取配置默认（与训练器同一处），它是每篇的 1 个符号，与取值无关。
        boundary = TaijiConfig().boundary_symbol

    expected_visits = exit_record.get("document_visits")
    expected_budget = exit_record.get("budget_max_symbols")
    fingerprint_here = corpus_fingerprint([corpus])
    checks = {
        "corpus_fingerprint_matches_exit": fingerprint_here
        == exit_record.get("corpus_fingerprint"),
        "max_symbols_matches_exit_budget": int(max_symbols) == expected_budget,
        "replayed_visits_match_exit": None,  # 跑完才填
    }

    counters = DocumentStreamCounters()
    stream = iter_corpus_symbols(
        [corpus],
        boundary=boundary,
        end_boundary_after_newline=end_boundary_after_newline,
        max_unique_documents=max_unique_documents,
        counters=counters,
    )
    #: `islice` 只拉够就走 ⇒ 取的是**符号数**（不是篇数）；两臂真件的 `ticks_at_exit` 就是它。
    taken = list(itertools.islice(stream, int(max_symbols)))
    checks["replayed_visits_match_exit"] = counters.document_visits == expected_visits

    status = "ok" if all(bool(value) for value in checks.values()) else "guard_failed"
    if status == "ok" and len(taken) != int(max_symbols):
        #: 流比预算短＝这档根本没吃满，复算的分布不能代表当时那条流。
        status = "stream_shorter_than_budget"

    return {
        "format": FORMAT,
        "status": status,
        "checks": checks,
        "arm_exit_document_visits": expected_visits,
        "replayed_document_visits": int(counters.document_visits),
        "replayed_unique_documents": int(counters.unique_documents),
        "symbols_taken": len(taken),
        "max_symbols": int(max_symbols),
        "max_unique_documents": max_unique_documents,
        "end_boundary_after_newline": bool(end_boundary_after_newline),
        "boundary_symbol": int(boundary),
        "corpus_fingerprint": fingerprint_here,
        "sequence_length": counters.sequence_length_stats(),
        "caveats": list(CAVEATS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="§8.7 序列长度一列的同源复算（零训练跑量）")
    parser.add_argument("--exit", required=True, help="某一臂的 progress_exit.json")
    parser.add_argument("--corpus", required=True, help="该臂冻过的语料路径")
    parser.add_argument("--max-symbols", required=True, type=int, help="该臂冻过的符号预算")
    parser.add_argument(
        "--max-unique-documents", type=int, default=None, help="形状 B 的封顶篇数（甲臂不给）"
    )
    parser.add_argument(
        "--no-end-boundary-after-newline",
        dest="end_boundary_after_newline",
        action="store_false",
        help="按旧形状复算（每篇正文后不补换行）；默认 True＝跟随训练器 CLI 默认。",
    )
    parser.add_argument("--out", required=True, help="复算件落盘路径")
    args = parser.parse_args(argv)

    exit_path = Path(args.exit)
    if not exit_path.is_file():
        print(
            json.dumps(
                {"status": "missing_exit_record", "exit": str(exit_path)}, ensure_ascii=False
            )
        )
        return 2
    corpus = Path(args.corpus)
    if not corpus.is_file():
        print(json.dumps({"status": "missing_corpus", "corpus": str(corpus)}, ensure_ascii=False))
        return 2
    payload = recompute(
        json.loads(exit_path.read_text(encoding="utf-8")),
        corpus,
        max_symbols=args.max_symbols,
        max_unique_documents=args.max_unique_documents,
        end_boundary_after_newline=args.end_boundary_after_newline,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": payload["status"], "out": str(out), **payload["checks"]}))
    return 0 if payload["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())

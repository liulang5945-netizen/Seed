"""PLAN-A-30 §3 丁路的**零训练归属检验**：训练分布上"该停的地方"模型到底会不会停。

`§2h` 把残留最坏那条定在 (B)「停止信号缺席」，但它只说了**模型自己走出来的那条分布**
（72 次生成 0 次自收）。据此就去花训练机时是不诚实的——先要分清两种可能：

  (B1) **信号根本没学到**：连"语料里真正结束的位置"都不给边界符以竞争力
       ⇒ 病在训练目标／频次那一层，丁（给结束以可学目标）才值得占机时；
  (B2) **信号学到了、但在自己的输出上失效**（exposure bias：训练时永远看着真文本，
       推理时看着自己写坏的东西）⇒ 加训同一个配方救不了，要动的是采样/课程侧。

分法：**不进模型自己的轨迹**，逐篇按**产品训练入口的同一套符号口径**喂真文本
（`Taiji.learn_bytes(..., include_boundary=True)` 用的就是 `sensor.symbols`＝每篇**前后各一个边界符**，
且每篇先 `reset_dynamics`），在每一步取"下一步的预测分布"，然后比两类位置上的边界符竞争力：

* **结束位**：语料里下一步真的是边界符的那些步（每篇恰好一个，就在该篇最后一个字节之后）；
* **轮结束位**：下一步是 `\n问：` 的那个换行字节——**产品口径下"这一答说完了"的实际编码**
  （`_serialize` 铺 `问：…\n答：…`，`chat()` 也按同一 marker 截断）；
* **对照位**：同一条喂入链上的其它步（下一步是真字节）。

读数三件，都带**乱序基线**该有的形态：边界概率、在合法候选集里的名次、以及"边界是不是 argmax"。
如果结束位的边界名次显著好于对照位、却仍然从不成为 argmax ⇒ 信号是有的、但被竞争压死（偏 B1 的弱式）；
如果两类位置的边界名次**一样差** ⇒ 信号根本没学到（强 B1）；
如果**结束位里有一批就是 argmax** ⇒ 训练分布上它会停，§2h 那条失效就归到 B2（自身轨迹上才失效）。

只读取，不训练：`learn=False`、基座 sha 逐位不变（件里有 `base_sha256_unchanged`）。

用法：
    python scripts/training/audit_taiji_a30_stop_signal_presence.py --docs 200
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_symbols(path: Path, docs: int) -> list[list[int]]:
    """按**训练脚本的口径**取每篇的符号序列（边界符 + 正文 UTF-8 字节）。"""

    from taiji import TaijiConfig

    boundary = int(TaijiConfig().boundary_symbol)
    chunks: list[list[int]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            content = payload["text"] if isinstance(payload, dict) and "text" in payload else text
            chunks.append([boundary, *content.encode("utf-8")])
            if len(chunks) >= docs:
                break
    return chunks


def _quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def at(quantile: float) -> float:
        return ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]

    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 6),
        "p10": round(at(0.10), 6),
        "median": round(at(0.50), 6),
        "p90": round(at(0.90), 6),
        "max": round(ordered[-1], 6),
    }


def audit(runtime: Any, chunks: list[list[int]], boundary: int, mask: bool) -> dict[str, Any]:
    """逐篇走**产品训练入口的同一套符号口径**（不改权重），按"下一符号是不是边界符"分两堆。

    口径不是自己定的：`Taiji.learn_bytes(..., include_boundary=True)` 喂的就是
    `sensor.symbols(data)`＝**前后各一个边界符**（`taiji/organs.py:62-64`），且每轮先
    `reset_dynamics`（`taiji/model.py:2863-2870`）⇒ "该停的位置"在训练面上**确实有目标**
    （每篇恰一个：最后一个正文字节之后就是收尾边界符）。本件要测的不是"有没有这个目标"，
    而是**训出来的基底在这个目标上到底赢不赢**。
    第一版把 120 篇首尾接成一条流、且不 reset，被产品自己的规则当场拒了
    （`readout changed inside an active dynamics episode; reset before switching`）——
    那是仪器没照产品口径走，不是模型的错；产品这条"一个 episode 不许换读出器"的规矩
    本身就是"每篇一个 episode"的证据，顺带记进件里。
    """

    from taiji.utf8_state import advance_utf8, utf8_allowed

    substrate = runtime.model.substrate
    rows: dict[str, list[dict[str, Any]]] = {"end": [], "turn_marker": [], "other": []}
    stream_symbols = 0
    #: 产品出口的"轮结束"编码其实是 `问：` 这几个字节（`_serialize` 铺的就是 `问：…\n答：…`，
    #: `chat()` 也按 `_TURN_MARKERS` 截）。训练面上每篇文档只有**一个**边界符目标、却有**多个**轮结束
    #: ⇒ "该停"这件事在语料里主要是靠 marker 教的。于是分三堆比：
    #: `end`＝下一步真是边界符；`turn_marker`＝下一步是 `\n问：` 的那个换行字节；`other`＝其余。
    #: 每一步同时给出**真下一符号**的概率与名次，两种停止编码才能放在同一把尺上比。
    for doc_index, chunk in enumerate(chunks):
        symbols = list(substrate.sensor.symbols(bytes(chunk[1:]), include_boundary=True))
        stream_symbols += len(symbols)
        substrate.reset_dynamics(episode_id=f"audit:stop-signal:{doc_index}")
        state = (0, 0)
        for position, symbol in enumerate(symbols):
            step = substrate.observe(
                symbol,
                learn=False,
                readout="predictive",
                use_memory=True,
                use_identity=False,
            )
            #: 掩码合法集要用**喂完这一步之后**的 UTF-8 态——产码环里也是这个次序
            #: （`advance_utf8` 在 `generated.append` 之后、下一轮取合法集之前）。
            state = (0, 0) if symbol == boundary else advance_utf8(state[0], state[1], symbol)
            if position + 1 >= len(symbols):
                break
            probabilities = step.probabilities.detach().cpu().float()
            vector = [float(value) for value in probabilities]
            legal = (
                set(utf8_allowed(state[0], state[1])) | {boundary}
                if mask
                else set(range(len(vector)))
            )
            best = max(vector[index] for index in legal)
            ranked = sorted(legal, key=lambda index: -vector[index])
            following = symbols[position + 1]
            #: `bytes()` 只收 0..255，而流里含边界符 256 ⇒ 第一版在这里炸
            #: （`ValueError: bytes must be in range(0, 256)`）；改成按整数序列比 marker 前缀
            #: `\n问` = 0x0A E9 97 AE。
            after_marker_lead = symbols[position + 1 : position + 5] == [0x0A, 0xE9, 0x97, 0xAE]
            kind = (
                "end"
                if following == boundary
                else "turn_marker" if following == 0x0A and after_marker_lead else "other"
            )
            rows[kind].append(
                {
                    "p_boundary": round(vector[boundary], 8),
                    "boundary_rank": ranked.index(boundary) + 1,
                    "boundary_is_argmax": bool(vector[boundary] >= best - 1e-9),
                    "legal_candidates": len(legal),
                    "p_next_true": round(vector[following], 8),
                    "next_true_is_argmax": bool(vector[following] >= best - 1e-9),
                }
            )
    doc_lengths = [len(chunk) - 1 for chunk in chunks]

    def _face(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        if not bucket:
            return {"n": 0}
        return {
            "n": len(bucket),
            "p_boundary": _quantiles([row["p_boundary"] for row in bucket]),
            "boundary_rank": _quantiles([float(row["boundary_rank"]) for row in bucket]),
            "boundary_is_argmax_count": sum(1 for row in bucket if row["boundary_is_argmax"]),
            "p_next_true": _quantiles([row["p_next_true"] for row in bucket]),
            "next_true_is_argmax_count": sum(1 for row in bucket if row["next_true_is_argmax"]),
        }

    #: 停止判据**不必是 argmax**。既然边界符在真结束位有名次/概率上的分离（只是从不胜出），
    #: 就可以问"多大的阈值能在真结束位收到多少、在别处误收多少"——这一张表给丁-4（校准式停止）定价，
    #: 用的还是模型自己学到的那个信号，不是产品替它改写内容。
    def _sweep() -> list[dict[str, Any]]:
        grid = [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2]
        ends = [row["p_boundary"] for row in rows["end"]]
        others = [row["p_boundary"] for row in rows["other"]]
        out: list[dict[str, Any]] = []
        for threshold in grid:
            recall = sum(1 for value in ends if value >= threshold) / len(ends) if ends else None
            false_rate = (
                sum(1 for value in others if value >= threshold) / len(others) if others else None
            )
            out.append(
                {
                    "threshold": threshold,
                    "true_stop_recall": round(recall, 4) if recall is not None else None,
                    "false_stop_rate_per_position": (
                        round(false_rate, 6) if false_rate is not None else None
                    ),
                }
            )
        return out

    return {
        "documents": len(chunks),
        "stream_symbols": stream_symbols,
        "faces": {kind: _face(bucket) for kind, bucket in rows.items()},
        "threshold_sweep": _sweep(),
        "mean_document_bytes": (
            round(sum(doc_lengths) / len(doc_lengths), 2) if doc_lengths else None
        ),
        "turn_markers_observed": len(rows["turn_marker"]),
        "end_positions": len(rows["end"]),
        "other_positions": len(rows["other"]),
        "end_rows_sample": rows["end"][:12],
        "turn_marker_rows_sample": rows["turn_marker"][:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--docs", type=int, default=200)
    parser.add_argument(
        "--mask",
        action="store_true",
        help="按产品解码口径只比较 UTF-8 合法候选（不给＝全 257 个符号上直接比，即教师强制面）",
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = PROJECT_ROOT / corpus

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    boundary = int(runtime.model.substrate.config.boundary_symbol)
    chunks = document_symbols(corpus, args.docs)
    result = audit(runtime, chunks, boundary, mask=args.mask)

    report = {
        "format": "taiji-a30-stop-signal-presence-v1",
        "prereg": "plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md §3 丁（零训练归属检验）",
        "question": "(B1) 停止信号没学到 还是 (B2) 学到了但在自身轨迹上失效",
        "checkpoint": args.checkpoint,
        "corpus": corpus.name,
        "corpus_bytes": corpus.stat().st_size,
        "documents": result["documents"],
        "generation_scope": "teacher_forced_on_corpus（不进模型自己的轨迹）",
        "decision_face": "utf8_masked_legal_set" if args.mask else "full_alphabet",
        "result": result,
        "instrument_guard": {
            "observe_calls_recorded": result["end_positions"] + result["other_positions"] > 0,
            "boundary_targets_expected": len(chunks),
            "boundary_targets_observed": result["end_positions"],
            "end_position_accounting_ok": result["end_positions"] == len(chunks),
            "turn_marker_bucket_populated": result["turn_markers_observed"] > 0,
            "turn_markers_observed": result["turn_markers_observed"],
            "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        },
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / f"reports/taiji_a30_stop_signal_presence_{args.docs}doc_20260928.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    summary = {
        kind: {
            "n": face.get("n"),
            "boundary_rank_median": (face.get("boundary_rank") or {}).get("median"),
            "boundary_p_median": (face.get("p_boundary") or {}).get("median"),
            "boundary_is_argmax": face.get("boundary_is_argmax_count"),
            "next_true_is_argmax": face.get("next_true_is_argmax_count"),
        }
        for kind, face in result["faces"].items()
    }
    print(
        json.dumps(
            {
                "docs": result["documents"],
                "stream_symbols": result["stream_symbols"],
                "mean_doc_bytes": result["mean_document_bytes"],
                "faces": summary,
                "threshold_sweep": result["threshold_sweep"],
                "guard": report["instrument_guard"],
                "mask": args.mask,
                "out": out.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

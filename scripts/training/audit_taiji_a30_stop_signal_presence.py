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
import hashlib
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def document_symbols(
    path: Path, docs: int, *, require_seam: bool = False, append_newline: bool = False
) -> dict[str, Any]:
    """按**训练脚本的口径**取每篇的符号序列（边界符 + 正文 UTF-8 字节）。

    `require_seam=True` 是给 §2s 那条新线索用的：默认语料**每篇只有一组问答**，
    所以"轮结束以 `\n问：` 字节表达"的位置天然稀少（400 篇才捞出 10 处）——
    直接把 docs 拉到几千要付几十万步的机器时间。改法是**先按文本筛**（读 jsonl 是秒级），
    只喂真正含轮接缝的那几篇：名次、概率、胜出率都照旧算，只是样本从"随机前 N 篇"
    变成"前 N 篇含接缝的"，**这一条改变必须在件里披露**（`selection` 字段），
    否则这就是另一口井里打的水却当成同一条河。
    """

    from taiji import TaijiConfig

    boundary = int(TaijiConfig().boundary_symbol)
    seam = chr(10).encode("utf-8") + "问：".encode()
    chunks: list[list[int]] = []
    lines_read = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            lines_read += 1
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            content = payload["text"] if isinstance(payload, dict) and "text" in payload else text
            raw = content.encode("utf-8")
            if require_seam and seam not in raw:
                continue
            if append_newline:
                #: 配方（`end_boundary_after_newline`）训的就是"正文后补一个换行再收尾"，
                #: 不加这一行时测面≠配方面，重训件在这里当然不显形（§2aj 当场发现的口径错配）。
                raw = raw + bytes([0x0A])
            chunks.append([boundary, *raw])
            if len(chunks) >= docs:
                break
    return {
        "chunks": chunks,
        "lines_read": lines_read,
        "selection": "only_documents_with_turn_seam" if require_seam else "first_n_documents",
    }


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
                    #: 位置身份：阈值停止**能不能用**取决于"若在 τ 处收笔，那一笔落在正文的第几格"，
                    #: 只看分离度会把"在答复 20% 处就误收"这种不可用的判成可用。加法字段，旧读数不受影响。
                    "doc_index": int(doc_index),
                    "position": int(position),
                    "p_boundary": round(vector[boundary], 8),
                    "boundary_rank": ranked.index(boundary) + 1,
                    "boundary_is_argmax": bool(vector[boundary] >= best - 1e-9),
                    "legal_candidates": len(legal),
                    "p_next_true": round(vector[following], 8),
                    "next_true_is_argmax": bool(vector[following] >= best - 1e-9),
                    #: **"是谁压过了边界符"——0/400 从不胜出这条已经量到，但胜者是谁没人记过。
                    #: 这一列把"停止没学会"与"停止学会了但被某个具体符号压制"分开，
                    #: 后者是有主人的（那个符号的训练分布），而丁-1..丁-4 都没碰过它。
                    "argmax_symbol": int(ranked[0]),
                    "p_argmax": round(vector[ranked[0]], 8),
                }
            )
    doc_lengths = [len(chunk) - 1 for chunk in chunks]

    def _face(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        if not bucket:
            return {"n": 0}
        #: 胜者 tally：只问"这些位置上谁在第一位"。ASCII 字节给字面提示，
        #: ≥128 的是 UTF-8 首字节（一个字节解不出一个汉字，件里就不假装解得出）。
        tally: dict[int, int] = {}
        for row in bucket:
            symbol = int(row.get("argmax_symbol", -1))
            tally[symbol] = tally.get(symbol, 0) + 1
        top = sorted(tally.items(), key=lambda item: -item[1])[:5]
        ratios = [
            row["p_argmax"] / row["p_boundary"]
            for row in bucket
            if row["p_boundary"] > 0 and "p_argmax" in row
        ]
        return {
            "n": len(bucket),
            "p_boundary": _quantiles([row["p_boundary"] for row in bucket]),
            "boundary_rank": _quantiles([float(row["boundary_rank"]) for row in bucket]),
            "boundary_is_argmax_count": sum(1 for row in bucket if row["boundary_is_argmax"]),
            "p_next_true": _quantiles([row["p_next_true"] for row in bucket]),
            "next_true_is_argmax_count": sum(1 for row in bucket if row["next_true_is_argmax"]),
            "argmax_winners_top5": [
                {
                    "symbol": symbol,
                    "count": count,
                    "ascii_hint": chr(symbol) if 32 <= symbol < 127 else None,
                }
                for symbol, count in top
            ],
            "p_argmax": _quantiles([row.get("p_argmax", 0.0) for row in bucket]),
            "median_ratio_argmax_over_boundary": (
                _quantiles([float(value) for value in ratios])["median"] if ratios else None
            ),
        }

    #: 停止判据**不必是 argmax**。既然边界符在真结束位有名次/概率上的分离（只是从不胜出），
    #: 就可以问"多大的阈值能在真结束位收到多少、在别处误收多少"——这一张表给丁-4（校准式停止）定价，
    #: 用的还是模型自己学到的那个信号，不是产品替它改写内容。
    def _sweep() -> list[dict[str, Any]]:
        grid = [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2]
        ends = [row["p_boundary"] for row in rows["end"]]
        others = [row["p_boundary"] for row in rows["other"]]
        #: "若在 τ 处收笔"这件事光看误率不够：一个分离度漂亮的阈值如果第一次误收就落在
        #: 正文 20% 处，产品在语义上就废了。这两列把"砍在第几格"量出来（按文档长度归一化）。
        other_by_doc: dict[int, list[tuple[int, float]]] = {}
        for row in rows["other"]:
            other_by_doc.setdefault(int(row["doc_index"]), []).append(
                (int(row["position"]), float(row["p_boundary"]))
            )
        out: list[dict[str, Any]] = []
        for threshold in grid:
            recall = sum(1 for value in ends if value >= threshold) / len(ends) if ends else None
            false_rate = (
                sum(1 for value in others if value >= threshold) / len(others) if others else None
            )
            fractions: list[float] = []
            for doc_index, entries in other_by_doc.items():
                fired = [position for position, value in entries if value >= threshold]
                length = doc_lengths[doc_index] if doc_index < len(doc_lengths) else 0
                if fired and length:
                    fractions.append(min(fired) / length)
            fractions.sort()
            out.append(
                {
                    "threshold": threshold,
                    "true_stop_recall": round(recall, 4) if recall is not None else None,
                    "false_stop_rate_per_position": (
                        round(false_rate, 6) if false_rate is not None else None
                    ),
                    "docs_with_false_fire": len(fractions),
                    "first_false_fire_fraction_of_doc_median": (
                        round(fractions[len(fractions) // 2], 4) if fractions else None
                    ),
                    "first_false_fire_fraction_of_doc_min": (
                        round(fractions[0], 4) if fractions else None
                    ),
                }
            )
        return out

    #: **比较式**停止判据的定价表：规则不是"p 超过绝对阈值"，而是"边界符离第一位只差 K 倍就收笔"。
    #: §2ak 那档量到自身轨迹上接缝的 `p_argmax/p_boundary` 中位 4.49、正文中间 4419——分离度在**比值**上
    #: 而不是在绝对值上，所以这张表才是下一档该读的东西（判据先于数写在 PLAN-A-30 §2am）。
    def _ratio_sweep() -> list[dict[str, Any]]:
        grid = [1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]

        def ratio(row: dict[str, Any]) -> float | None:
            if row["p_boundary"] <= 0:
                return None
            return row["p_argmax"] / row["p_boundary"]

        ends = [value for row in rows["end"] if (value := ratio(row)) is not None]
        others = [
            (int(row["doc_index"]), int(row["position"]), value)
            for row in rows["other"]
            if (value := ratio(row)) is not None
        ]
        other_by_doc: dict[int, list[tuple[int, float]]] = {}
        for doc_index, position, value in others:
            other_by_doc.setdefault(doc_index, []).append((position, value))
        out: list[dict[str, Any]] = []
        for cap in grid:
            recall = sum(1 for value in ends if value <= cap) / len(ends) if ends else None
            rate = (
                sum(1 for _, _, value in others if value <= cap) / len(others) if others else None
            )
            fractions: list[float] = []
            #: 带**位置条件**的误收篇数：产品若规定"正文走到 floor 比例之后才允许收笔"，同一个 K 下
            #: 还剩几篇会误收。这一列量的是那条规则的**代价**，本件不把它接进产品。
            floors = (0.0, 0.25, 0.5, 0.75)
            docs_by_floor = {floor: 0 for floor in floors}
            for doc_index, entries in other_by_doc.items():
                fired = [position for position, value in entries if value <= cap]
                length = doc_lengths[doc_index] if doc_index < len(doc_lengths) else 0
                if fired and length:
                    fractions.append(min(fired) / length)
                    #: 位置条件下的误收要按"**有没有允许窗口内的 firing**"数，不是"最早那次 firing 是否晚于窗口"：
                    #: 一条答复若在第 5% 和第 60% 都会触发，加了 floor=0.50 之后它**仍会在 60% 处被误收**。
                    #: 写成 `min(fired)/length >= floor` 会把这种条目当成"不误收"，从而**低报代价**（v7 初版即此错，
                    #: 已在 PLAN-A-30 §2an 留场更正，读数按本版重跑）。
                    for floor in floors:
                        if any(position / length >= floor for position in fired):
                            docs_by_floor[floor] += 1
            fractions.sort()
            out.append(
                {
                    "ratio_cap": cap,
                    "seam_hit_rate": round(recall, 4) if recall is not None else None,
                    "false_rate_per_position": round(rate, 6) if rate is not None else None,
                    "docs_with_false_fire": len(fractions),
                    "docs_with_false_fire_by_floor": {
                        f"{floor:.2f}": docs_by_floor[floor] for floor in floors
                    },
                    "first_false_fire_fraction_of_doc_median": (
                        round(fractions[len(fractions) // 2], 4) if fractions else None
                    ),
                    "first_false_fire_fraction_of_doc_min": (
                        round(fractions[0], 4) if fractions else None
                    ),
                }
            )
        return out

    return {
        "documents": len(chunks),
        "stream_symbols": stream_symbols,
        "faces": {kind: _face(bucket) for kind, bucket in rows.items()},
        "threshold_sweep": _sweep(),
        "ratio_sweep": _ratio_sweep(),
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
    parser.add_argument(
        "--require-seam",
        action="store_true",
        help="只取正文里真含 轮接缝（\n问：）的文档——把 marker 桶的样本量抬起来（件里披露 selection）",
    )
    parser.add_argument("--out-report", default=None)
    parser.add_argument(
        "--append-newline",
        action="store_true",
        help="每篇正文后补一个 0x0A 再收尾——把测面换成主线配方 `end_boundary_after_newline` 训的那张面",
    )
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = PROJECT_ROOT / corpus

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    boundary = int(runtime.model.substrate.config.boundary_symbol)
    sample = document_symbols(
        corpus, args.docs, require_seam=args.require_seam, append_newline=args.append_newline
    )
    chunks = sample["chunks"]
    result = audit(runtime, chunks, boundary, mask=args.mask)

    report = {
        "format": "taiji-a30-stop-signal-presence-v8",
        "format_note_v8": (
            "v8 **改语义不改算法**地修掉 v7 的 `docs_with_false_fire_by_floor` 一处低报：v7 数的是"
            "「最早那次触发是否晚于窗口」，而位置条件的正确问法是「窗口**之内**有没有触发」——"
            "一条答复若在第 5% 和第 60% 都会触发，加 floor=0.50 之后它仍会在 60% 处被误收，v7 把这种条目当成不误收。"
            "⇒ **v7 那三列 floor 数字一律低报代价、不可引用**（其余各列：`seam_hit_rate`、`false_rate_per_position`、"
            "`threshold_sweep`、`faces` 全部未动，与 v2–v7 同格可比）。本版起按 `any(position/length >= floor)` 计。"
        ),
        "format_note_v7": (
            "v7 只在 `ratio_sweep` 每行加一列 `docs_with_false_fire_by_floor`（floor∈{0.00,0.25,0.50,0.75}："
            "若规定正文走到 floor 比例之后才允许收笔，同一个 K 下还剩几篇误收）。其余各表算法一字未动 "
            "⇒ 与 v2–v6 同格可比。加它的理由见 PLAN-A-30 §2an：§2am 判死的是「位置局部」规则本身，"
            "这一列量的是「给规则加一条位置条件要花多少代价」，本件不把它接进产品。"
        ),
        "format_note_v6": (
            "v6 只加一张**加法**表 `ratio_sweep`：把停止判据从『p_boundary 超绝对阈值』换成『边界符离第一位只差 "
            "K 倍』（K∈{1.5,2,3,5,10,30,100}，`p_boundary<=0` 的格不入表）。列名换成 `seam_hit_rate` 以点名它量的"
            "是『接缝被这条比较式收到多少』；`threshold_sweep` 与 `faces` 的算法一字未动 ⇒ 与 v2–v5 同格可比。"
            "加它的理由见 §2ak：自身轨迹上接缝的 `p_argmax/p_boundary` 中位 4.49、正文中间 4419，"
            "分离度在比值上而不在绝对值上。"
        ),
        "format_note_v5": (
            "v5 只加两样且默认面逐位不变：新旗标 `--append-newline`（每篇正文后补一个 `0x0A` 再收尾，"
            "＝主线配方 `end_boundary_after_newline` 训的那张面）与一条自述字段 `documents_face`。"
            "加它的理由：§2aj 第一次跑这对读数时测面是『语料正文原样』，而配方改的是『正文后补换行再收尾』"
            "⇒ 测面≠配方面，重训件在这张面上本就不该显形，这不是模型的读数而是我的口径错配（已在 §2aj 留场）"
        ),
        "format_note_v4": (
            "v4 再加两处：每行带 `doc_index`/`position`（位置身份），阈值表带 `docs_with_false_fire` 与 "
            "`first_false_fire_fraction_of_doc_{median,min}`（若在 τ 收笔，第一次误收落在正文第几格）。"
            "`true_stop_recall` 与 `false_stop_rate_per_position` 两列的算法与语义**未动** "
            "⇒ v2/v3/v4 的阈值表同格可比。加位置列的理由：只看误率会把「分离度漂亮但第一笔就砍在答复 20% 处」"
            "的阈值当成可用。**同版修一处 v3 的崩点**：`docs_sha256` 原来对整条 chunk 取 `bytes()`，"
            "而 chunk 首元素是边界符 256 ⇒ `ValueError: bytes must be in range(0, 256)`，"
            "两枚件的 v3 读数因此一份都没落盘（这条例子在本件第 160 行的注释里写过，我自己踩了）。"
            "现改成正文部分 `chunk[1:]` 的指纹——边界符是喂入约定、不是文档内容。"
        ),
        "format_note": "v2 只在每格里**新增** `argmax_winners_top5`／`p_argmax`／`median_ratio_argmax_over_boundary` 三条与每行 `argmax_symbol`/`p_argmax`；旧字段语义与算法未动 ⇒ 与已入库的 v1 读数可直接同格比（v1 件里没有这几条，不是它们算出了 0）。v3 只再加一条 `docs_sha256`＝**实际吃进的那批文档字节的指纹**，用来把两枚检查点之间的配对机检起来（原来只能靠 `--docs` 参数相同来保证，那是口供不是检验）",
        "prereg": "plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md §3 丁（零训练归属检验）",
        "question": "(B1) 停止信号没学到 还是 (B2) 学到了但在自身轨迹上失效",
        "checkpoint": args.checkpoint,
        "corpus": corpus.name,
        "corpus_bytes": corpus.stat().st_size,
        "documents": result["documents"],
        "selection": sample["selection"],
        "lines_read_to_fill_sample": sample["lines_read"],
        "docs_sha256": hashlib.sha256(b"".join(bytes(chunk[1:]) for chunk in chunks)).hexdigest(),
        "generation_scope": "teacher_forced_on_corpus（不进模型自己的轨迹）",
        "decision_face": "utf8_masked_legal_set" if args.mask else "full_alphabet",
        "documents_face": (
            "corpus_body_plus_appended_newline" if args.append_newline else "corpus_body_as_stored"
        ),
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

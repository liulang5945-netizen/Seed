"""PLAN-A-30 §7-2「乙」的存在性证明：**只有"训练吃的答案是谁写的"这一个变量不同**，自身轨迹上的胜出会不会离开零。

动机（读数来自已入库件，不是推测）：落点配方在**教师强制面**把结束位胜出推到 239/300（对照件同面 0/300），
但在**模型自己写的正文**上是 `1/300`（`SPEC-A-24` L3 现值），raw 层自停 0/72（L2 现值）。
⇒ 瓶颈从"目标放哪一格"移到了"训练时见过谁写的文本"。本件就是定价这一手的**最小配对档**。

两臂唯一的差别（其余全同）：
* `corpus`＝今天训练面的形状：`问：{语料问句}\\n答：{语料答案}\\n`，每答收尾一次；
* `self`＝**乙 的形状**：同一批问句，把答案换成**底件自己生成的文本**（走产品出口 `chat(learn=False)`），每答同样收尾一次。

两臂都**经产品门面** `Seed.learn_bytes(..., include_end_boundary=True, reset=True)` 喂——
这顺带是 `DEBT-G13`（2026-09-29 由 `be4a8e58` 开）那条出口的**第一次产品路径实走**：
若转发没落地，这里会 TypeError 响亮失败，而不是静默走底层。

判读线（**先于数冻结，四支互斥**，与 §2ai/§2aq 同形）：
* `self − corpus ≥ 3` ⇒ **乙 的形状成立**（训练分布是那只缺失的手）；
* 两臂都 0 ⇒ **形状不是瓶颈**（换分布也学不出胜出），这条否证乙、把路留给丙或重训规模；
* 两臂都 ≤1 且至少一支为 1 ⇒ `not_resolved`（分辨率不足，不许写成"成立"或"不成立"）；
* 其余 ⇒ `not_resolved`。

**必须跟着读数走的三条边界**：①两臂喂入**字节数不相等**（自己写的答案长短不一），这是本档唯一没锁住的量，
件里 `trained_text_bytes_*` 逐臂报出；若结论与它同向，要按 §2y 的规矩点名"密度/体积没锁"。
②`self` 臂的答案是**底件（未训这形状前）**写的，训完之后它自己的输出会变 ⇒ 本档只证"吃过自己写过的文本"这一手的因果，
不证"收敛后的自我一致"（那是迭代档，另开）。③接缝是我强加的位置定义，不是模型选的。
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

from api.seed_runtime import SeedRuntime  # noqa: E402
from probe_taiji_a30_ding3_stop_target_pilot import (  # noqa: E402
    DEFAULT_CORPUS,
    facade_gap,
    read_groups,
    transfer_face,
)

BOUNDARY = 256


def _fingerprint(chunks: list[bytes]) -> str:
    return hashlib.sha256(b"\x00".join(chunks)).hexdigest()[:16]


def generate_self_answers(
    runtime: Any, pairs: list[tuple[str, str]], *, gen_max: int
) -> dict[str, str]:
    """用**底件**（训这形状之前）为每个问句写一份答案；同一份缓存喂给 `self` 与 `sized` 两臂。"""

    cache: dict[str, str] = {}
    for question, _ in pairs:
        if question in cache:
            continue
        cache[question] = str(
            runtime.chat(
                question,
                history=[],
                learn=False,
                max_length=gen_max,
                repetition_penalty=2.0,
            )
        )
    return cache


def _answer_stream(
    pairs: list[tuple[str, str]], *, mode: str, self_answers: dict[str, str]
) -> list[bytes]:
    """按"每答收尾一次"的形状产出喂入块。三臂只差**答案从哪来**：

    * `corpus`＝语料写的答案（今天训练面的形状）；
    * `self`＝底件自己写的答案（乙 的形状）；
    * `sized`＝语料答案按 `self` 的**字符长度**截短 ⇒ 体积与 `self` 同档、作者仍是语料。
      加这一档的理由：`self` 对 `corpus` 同时动了两个变量（作者＋喂入体积），
      而 `self` 对 `sized` 只差作者——没有中间档，读出任何方向都是对角线（本仓已有这条规矩）。
    """

    chunks: list[bytes] = []
    for question, corpus_answer in pairs:
        if mode == "self":
            answer = self_answers[question]
        elif mode == "sized":
            target = len(self_answers[question])
            answer = corpus_answer[:target] if target else corpus_answer
        else:
            answer = corpus_answer
        chunks.append(f"问：{question}\n答：{answer}\n".encode("utf-8"))
    return chunks


def train_arm(runtime: Any, chunks: list[bytes], *, epochs: int) -> dict[str, Any]:
    """经**产品门面**喂入；每块一次 `include_end_boundary=True` ＋ `reset=True`（＝每答收尾一次）。"""

    fed_bytes = 0
    feeds = 0
    for _ in range(epochs):
        for chunk in chunks:
            runtime.model.learn_bytes(
                chunk,
                epochs=1,
                include_boundary=False,
                include_end_boundary=True,
                reset=True,
                use_memory=False,
            )
            fed_bytes += len(chunk)
            feeds += 1
    # 问句指纹**从实际喂进的字节里取**（不是从参数里取）：这样"两臂只差答案作者"是被字节证明的。
    question_fp = _fingerprint(
        [c.decode("utf-8").partition("\n答：")[0].encode("utf-8") for c in chunks]
    )
    answer_fp = _fingerprint(
        [c.decode("utf-8").partition("\n答：")[2].encode("utf-8") for c in chunks]
    )
    return {
        "feeds": feeds,
        "trained_text_bytes": fed_bytes,
        "feed_fingerprint": _fingerprint(list(chunks) * epochs),
        "fed_question_fingerprint": question_fp,
        "fed_answer_fingerprint": answer_fp,
    }


def decide(self_wins: int, corpus_wins: int) -> str:
    """四支互斥；先于数冻结。"""

    if self_wins - corpus_wins >= 3:
        return "shape_holds（吃过自己写的文本 ⇒ 胜出离开零）"
    if self_wins == 0 and corpus_wins == 0:
        return "shape_not_the_bottleneck（换训练分布也不产出胜出）"
    if self_wins <= 1 and corpus_wins <= 1:
        return "not_resolved（两臂都 ≤1，分辨率不足）"
    return "not_resolved（差在 1–2 之间或非零但不足线）"


def read_volume_control(self_wins: int, corpus_wins: int, sized_wins: int) -> str:
    """体积对照的读法（**同样先于数冻结**）：主判据成立时，用它分清是"作者"还是"喂入体积"。"""

    if self_wins - corpus_wins < 3:
        return "not_applicable（主判据未成立，不必区分）"
    if self_wins - sized_wins >= 3:
        return "author_effect（与同体积的语料臂仍差 ≥3 ⇒ 是作者／分布，不是体积）"
    return "volume_confounded（同体积下差不出 3 ⇒ 主判据可能被体积解释，不许写成作者效应）"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="output/a26_p1/checkpoint.pt")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--exchanges", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gen-max", type=int, default=96, help="self 臂生成答案的字节预算")
    parser.add_argument("--positions", type=int, default=18, help="迁移档跑多少提问")
    parser.add_argument("--mask", action="store_true", default=True)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    base_path = PROJECT_ROOT / args.base
    sha_base = hashlib.sha256(base_path.read_bytes()).hexdigest()

    groups = read_groups(corpus, groups=args.groups, exchanges=args.exchanges)
    # 语料每行本身是一条「问：…\n答：…」⇒ 拆出 (问句, 语料答案) 对；拆不开就响亮停（不许静默少喂）
    marker = "\n答："
    pairs: list[tuple[str, str]] = []
    for chunk_group in groups["trained"]:
        for record in chunk_group:
            text = record.decode("utf-8")
            if marker not in text:
                raise RuntimeError(f"语料行里没有 {marker} 这个接缝，拆不出答案：{text[:24]!r}")
            question, _, answer = text.partition(marker)
            pairs.append((question.removeprefix("问：").strip(), answer.strip()))
    if not pairs:
        raise RuntimeError("读不到 (问句, 语料答案) 对——本档的配对前提不成立，拒绝跑")

    held_out = [
        record.decode("utf-8").partition(marker)[0].removeprefix("问：").strip()
        for chunk_group in groups["held_out"]
        for record in chunk_group
    ][: args.positions]
    if len(held_out) < args.positions:
        raise RuntimeError(
            f"held-out 只拆出 {len(held_out)} 条提问，不足 {args.positions}（拒绝把样本量降下来悄悄跑）"
        )

    # 生成一次，三臂共用同一份"自己写的答案"（`sized` 只用它的**长度**，不用它的内容）
    seed_runtime = SeedRuntime.load(base_path)
    self_answers = generate_self_answers(seed_runtime, pairs, gen_max=args.gen_max)
    del seed_runtime

    runs: dict[str, Any] = {}
    for arm in ("corpus", "self", "sized"):
        runtime = SeedRuntime.load(base_path)
        sha_at_load = hashlib.sha256(base_path.read_bytes()).hexdigest()[:16]
        gap = facade_gap(runtime)
        if not gap.get("edge_split_reachable_from_facade", False):
            raise RuntimeError(f"门面出口不可达（{gap}）——DEBT-G13 那行没落地，本档不走后门")
        chunks = _answer_stream(pairs, mode=arm, self_answers=self_answers)
        trained = train_arm(runtime, chunks, epochs=args.epochs)
        face = transfer_face(
            runtime, held_out, BOUNDARY, max_bytes=128, penalty=2.0, mask=args.mask
        )
        runs[arm] = {
            "train": trained,
            "transfer": face,
            "facade_reachable": bool(gap.get("edge_split_reachable_from_facade", False)),
            "facade_gap": {k: v for k, v in gap.items() if isinstance(v, (bool, str, int))},
            "base_sha": sha_at_load,
        }

    self_wins = runs["self"]["transfer"]["boundary_argmax_positions"]
    corpus_wins = runs["corpus"]["transfer"]["boundary_argmax_positions"]
    sized_wins = runs["sized"]["transfer"]["boundary_argmax_positions"]
    same_questions = (
        runs["self"]["train"]["fed_question_fingerprint"]
        == runs["corpus"]["train"]["fed_question_fingerprint"]
    )
    different_answers = (
        runs["self"]["train"]["fed_answer_fingerprint"]
        != runs["corpus"]["train"]["fed_answer_fingerprint"]
    )

    report = {
        "format": 1,
        "question": "乙：训练吃过模型自己写的答案，能否让自身轨迹接缝上的胜出离开零（唯一变量＝答案作者）",
        "prereg_criterion": "self − corpus ≥ 3 ⇒ 成立；两臂都 0 ⇒ 形状不是瓶颈；两臂都 ≤1 且有 1 ⇒ not_resolved",
        "base": args.base,
        "base_sha256": sha_base[:16],
        "corpus": str(corpus.relative_to(PROJECT_ROOT)),
        "groups": args.groups,
        "exchanges": args.exchanges,
        "epochs": args.epochs,
        "positions": len(held_out),
        "question_fingerprint": _fingerprint([q.encode("utf-8") for q in held_out]),
        "runs": runs,
        "instrument_guard": {
            "arms_share_the_same_questions": same_questions,
            "arms_differ_only_in_answer_author": different_answers,
            "facade_reachable_both_arms": all(bool(runs[arm]["facade_reachable"]) for arm in runs),
            "both_arms_same_base_sha": runs["self"]["base_sha"] == runs["corpus"]["base_sha"],
            "base_untouched_after_run": (
                hashlib.sha256(base_path.read_bytes()).hexdigest() == sha_base
            ),
            "heldout_disjoint_from_trained": bool(groups["heldout_disjoint_from_trained"]),
        },
        "unlocked_variable_disclosed": {
            "note": "两臂喂入字节数不等（自己写的答案长短不一）——这是本档唯一没锁住的量，逐臂见 trained_text_bytes",
            "corpus_bytes": runs["corpus"]["train"]["trained_text_bytes"],
            "self_bytes": runs["self"]["train"]["trained_text_bytes"],
        },
        "verdict": decide(self_wins, corpus_wins),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = PROJECT_ROOT / args.out_report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "self_wins": self_wins,
                "corpus_wins": corpus_wins,
                "verdict": report["verdict"],
                "bytes": report["unlocked_variable_disclosed"],
                "guards": report["instrument_guard"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

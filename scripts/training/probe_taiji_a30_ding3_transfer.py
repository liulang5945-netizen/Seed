"""PLAN-A-30 §2ai：**迁移档**——边界符在"模型自己写的答案之后"这一格上还赢不赢。

**为什么要这一档（§2ah 留下的那一问）**：落点配方在**教师强制**上把结束位的边界符推到
26/30（主线同档 231/300、263/300），但两枚装配的 raw 生成里它 **0 步**成为 argmax、
答复 72/72 吃满预算。两句话之间的缺口就是"迁移"：目标**学到了**，在模型自己的轨迹上**取不出**。
本件把这一格量成一个数，用来决定"还要不要往训练目标侧投资"。

**测法（复用 §2aa/§2ae 那两件里现成的函数，不重抄链）**：
`import` `probe_taiji_a30_ding3_stop_target_pilot` 的 `read_groups`／`question_of`／`transfer_face`，
同一批提问、同一个 `audit()`、同一张 mask 面，两枚检查点各测一次：

* **迁移面**＝`问：{提问}\n答：{它自己写的答案}` 之后再补一个换行 ⇒ 量这一格上边界符是否 argmax；
  （接缝是本件强加的，不是模型选的——读数必须带着这句。）
* **教师强制面（同件顺带量，做参照）**＝同样的位置但答案来自**语料** ⇒ 与 §2aa 同口径。

**判读线（先于数写死，看到数之后不许挪）**：设两枚件各 N 个迁移位置。
1. **迁移成立**：重训件 `boundary_argmax_positions` − base 件 **≥ 3**；
2. **迁移彻底不成立**（预期分支）：两枚件**都 ≤ 1**（＝教师强制那 26/30 的收益在自己轨迹上为零）
   ⇒ 结论落法：**目标侧放置这条投资线到此为止**，"停不下来"改登记为**分布／轨迹**问题，
   与 §2n（抬预算只买到更长）、§2v（prompt 回路放大退化）同一族；
3. 夹在中间（差 1–2）⇒ 记 `not_resolved`，不许写成"部分成立"。

**配对前提（机检，不靠我保证）**：两枚件吃的是**同一批提问**（件里存 `prompts_sha256`，两边必须相等）、
同一个预算/惩罚/mask、同一个 held-out 分组；`base_sha256_unchanged` 对**两枚**检查点分别成立（只读）。

**本件不动任何检查点、不改产品默认、不写 `checkpoints/`**：两枚件都只读；输出只有本报告 JSON。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
# 本机控制台是 GBK，判读串里的 ⇒ 会崩在写完报告之后的那句 print。
sys.stdout.reconfigure(encoding="utf-8")

from probe_taiji_a30_ding3_stop_target_pilot import (  # noqa: E402
    DEFAULT_CORPUS,
    question_of,
    read_groups,
    transfer_face,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(
    checkpoint: Path,
    prompts: list[str],
    *,
    boundary: int,
    mask: bool,
    max_bytes: int,
    penalty: float,
) -> dict[str, Any]:
    """一枚检查点：在它**实际吃到的那批提问**上量迁移面，并回带该批提问的指纹。"""

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    moved = transfer_face(
        runtime,
        prompts,
        boundary,
        max_bytes=max_bytes,
        penalty=penalty,
        mask=mask,
    )
    del runtime
    moved["prompts_sha256"] = hashlib.sha256("\n".join(prompts).encode()).hexdigest()
    return moved


def decide(retrain_hits: int, base_hits: int) -> str:
    """**互斥的四支**（§2ai 第 2 句那条重叠就在这里消掉；分支判的是同一对计数，不会两读）。

    线本身没改：迁移成立仍是 `重训 − base ≥ 3`。改的是"为零"那句的**触发条件**——
    原来写成"两臂都 ≤1"，于是 (1, 0) 这种数据会被 `if/elif` 的顺序判成比数据更硬的结论。
    """

    if retrain_hits - base_hits >= 3:
        return "迁移成立（重训件在自己写的答案之后能让边界符胜出，比 base 多 ≥3 格）"
    if retrain_hits == 0 and base_hits == 0:
        return (
            "迁移不成立：两枚件在自身轨迹上都是零胜出 ⇒ 目标放置这条线到此为止，"
            "'停不下来'改登记为分布／轨迹问题"
        )
    if retrain_hits <= 1 and base_hits <= 1:
        return (
            "not_resolved（两臂都 ≤1 但有一臂出过 1 次胜出：'为零'这句本档撑不起，"
            "要么扩样要么按 §2ai 第 2 句改判据——不再让 if/elif 顺序替我下更重的结论）"
        )
    return "not_resolved（差 1–2 格，不过 ≥3 线）"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", default="output/a31_ding3_boundary/checkpoint.pt")
    parser.add_argument("--base", default="output/a26_p1/checkpoint.pt")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--exchanges", type=int, default=3)
    parser.add_argument("--positions", type=int, default=12, help="迁移面跑多少个提问")
    parser.add_argument("--max-bytes", type=int, default=256)
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from taiji import TaijiConfig

    boundary = int(TaijiConfig().boundary_symbol)
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = PROJECT_ROOT / corpus
    retrain = PROJECT_ROOT / args.retrain
    base = PROJECT_ROOT / args.base
    sha_before = {"retrain": _sha(retrain), "base": _sha(base)}

    data = read_groups(corpus, groups=args.groups, exchanges=args.exchanges)
    prompts = [q for g in data["held_out"] if (q := question_of(g[0]))][: args.positions]
    if len(prompts) < args.positions:
        raise SystemExit(f"held-out 只拆出 {len(prompts)} 条提问，不足 {args.positions}")
    prompts_sha = hashlib.sha256("\n".join(prompts).encode()).hexdigest()

    runs = {}
    for label, ckpt in (("retrain", retrain), ("base", base)):
        runs[label] = run_one(
            ckpt,
            prompts,
            boundary=boundary,
            mask=args.mask,
            max_bytes=args.max_bytes,
            penalty=args.penalty,
        )
        runs[label]["checkpoint_sha256_before"] = sha_before[label]

    moved_diff = (
        runs["retrain"]["boundary_argmax_positions"] - runs["base"]["boundary_argmax_positions"]
    )
    verdict = decide(
        runs["retrain"]["boundary_argmax_positions"],
        runs["base"]["boundary_argmax_positions"],
    )

    report = {
        "format": "taiji-a30-ding3-transfer-v2",
        "format_note_v2": (
            "v2 只改**判读的分支结构**（消掉 §2ai 第 2 句登记的重叠：原来分支 2 写的是『两臂都 ≤1』，"
            "于是 (1, 0) 会被 if/elif 顺序判成『迁移不成立／收益为零』这种比数据更硬的结论）。"
            "四支现互斥：差 ≥3 ⇒ 成立；两臂都 0 ⇒ 不成立（双臂零胜出）；两臂都 ≤1 且有一臂 1 ⇒ not_resolved；"
            "其余 ⇒ not_resolved（差 1–2）。**线本身一字未动**，所以 v1 的三档读数（12/30/120 位置）"
            "仍可比——只是 (1, 0) 那种组合的标签会不同。"
        ),
        "prereg": "PLAN-A-30 §2ai（判读线先于数写在件 docstring）",
        "question": "边界符在'模型自己写的答案＋换行'这一格上赢不赢（教师强制 26/30 的收益能否迁移）",
        "retrain_checkpoint": args.retrain,
        "base_checkpoint": args.base,
        "corpus": corpus.name,
        "mask": bool(args.mask),
        "decision_face": "utf8_masked_legal_set" if args.mask else "full_alphabet",
        "positions": args.positions,
        "max_bytes": args.max_bytes,
        "repetition_penalty": args.penalty,
        "prompts_sha256": prompts_sha,
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in runs.items()},
        "rows": {k: v["rows"] for k, v in runs.items()},
        "delta_retrain_minus_base": {
            "boundary_argmax_positions": moved_diff,
            "answers_stopped_early": runs["retrain"]["answers_stopped_early"]
            - runs["base"]["answers_stopped_early"],
            "median_rank": (
                (runs["base"]["median_rank_over_positions"] or 0)
                - (runs["retrain"]["median_rank_over_positions"] or 0)
            ),
        },
        "verdict": verdict,
        "instrument_guard": {
            "same_prompts_both_arms": (
                runs["retrain"]["prompts_sha256"] == runs["base"]["prompts_sha256"]
            ),
            "both_arms_same_position_count": (
                runs["retrain"]["positions"] == runs["base"]["positions"] == len(prompts)
            ),
            "retrain_sha256_unchanged": _sha(retrain) == sha_before["retrain"],
            "base_sha256_unchanged": _sha(base) == sha_before["base"],
        },
        "operational_definition": (
            "迁移面接缝＝本件在'问：q\\n答：模型自己写的答案'之后补一个换行；"
            "接缝不是模型选的，所以这一格量的是'若它此刻该停，边界符赢不赢'，不是'它停没停'"
            "（后者见 §2ah 的 raw 计数：两枚件都 0/72 自停）。"
        ),
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / f"reports/taiji_a30_ding3_transfer_{args.positions}pos_20260929.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "positions": len(prompts),
                "retrain": {
                    k: runs["retrain"][k]
                    for k in (
                        "boundary_argmax_positions",
                        "answers_stopped_early",
                        "median_rank_over_positions",
                        "median_p_boundary_over_positions",
                    )
                },
                "base": {
                    k: runs["base"][k]
                    for k in (
                        "boundary_argmax_positions",
                        "answers_stopped_early",
                        "median_rank_over_positions",
                        "median_p_boundary_over_positions",
                    )
                },
                "verdict": verdict,
                "guard": report["instrument_guard"],
                "out": out.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PLAN-A-30 §2ak：**阈值停止在模型自己的轨迹上到底可不可用**（丁-4 的最后一档价）。

**为什么只差这一格**：§2ah 量的是"边界符在它自己写的文本里有没有**赢过 argmax**"（两枚装配都 0 步）；
§2ai 量的是"在它自己写的答案＋换行那一格，边界符赢不赢"（0/30），但**顺带读出**那一格上
`p_boundary` 中位 0.057149 对 base 0.000145（≈390×）、名次中位 7 对 20。
校准式停止**不需要赢过 argmax**，只需要"真结束位的 p 高于正文中间的 p"。这个分离度**没有任何一档量过**：
§2w 判死纯阈值用的是 base 件＋语料教师强制面，§2aj 补的是同一面的重训件，两处都不在自身轨迹上。

**测法（不重抄链）**：`chat()` 取它自己写的答案 ⇒ 拼 `问：{提问}\\n答：{它写的}` 再补一个 `0x0A` 作为
**一篇文档**喂给 `audit_taiji_a30_stop_signal_presence.audit()`——那个件按"下一符号是不是边界符"分堆，
所以这一格里：**end 桶＝那一篇的最后一格（接缝，本件强加的）**，**other 桶＝它自己正文里的每一格**
（问题部分＋答复部分的所有中间位）。阈值表 `threshold_sweep` 由同一个件给出，含
"若在 τ 收笔，第一次误收落在正文第几格"（`first_false_fire_fraction_of_doc_*`）——这一列是本轮才加的，
因为只看误率会把"分离度漂亮但第一笔就砍在答复 20% 处"的阈值当成可用。

**判读线（先于数写死，看到数之后不许挪）**：设 N＝迁移位置数（本件默认 24）。称某个网格阈值 τ 为
**可用可价点**，当且仅当它同时满足：
1. `true_stop_recall ≥ 0.5`（≥12/24 个真接缝的 p 过阈）；
2. `docs_with_false_fire ≤ N // 4`（≤6 篇在它自己正文里出现过误收）；
3. 若确有误收，`first_false_fire_fraction_of_doc_median ≥ 0.5`（要误收也收在正文后半，不至于把答复砍头）。

分支：
* 重训件存在可用可价点、base 不存在 ⇒ **阈值停止在自身轨迹上可用，且是这条配方买来的**：
  这才值得向 owner 提"产品出口加一条校准式停止"的一档（含 τ 的来路与回退面）；
* 两枚件都存在 ⇒ "可用"与配方无关，关于配方贡献记 `not_resolved`（同 §2aj 第 4 支）；
* 两枚件都不存在 ⇒ 丁-4 在**三面**（教师强制 §2w、生成 argmax §2ah/§2ai、自身轨迹阈值 §2ak）全部判死
  ⇒ 停止这条线整体收口，A 支线在"停不下来"上不再有新工具可试，剩下的方向都要 owner 裁；
* 分辨率线：N＝24 时 0.5 召回＝12 格，**±3 格以内的召回差不作结论**；分档只报"哪些 τ 过线"。

**本件不动权重**（`chat(learn=False)`、`audit()` 里 `observe(learn=False)`），两枚检查点只读，
机检 `*_sha256_unchanged`；两枚件吃同一批提问（`prompts_sha256` 必须相等）；
配对面另存 `docs_sha256`（＝实际喂进 audit 的文档字节指纹）。接缝仍是本件强加的，所以这一格量的还是
"若它此刻该停，边界符的概率高不高"，不是"它停没停"（后者见 §2ah：两枚件都 0/72 自停）。
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
# 本机控制台是 GBK，判读串里的 ⇒ 会崩在写完报告之后的那句 print。
sys.stdout.reconfigure(encoding="utf-8")

from audit_taiji_a30_stop_signal_presence import audit  # noqa: E402
from probe_taiji_a30_ding3_stop_target_pilot import (  # noqa: E402
    DEFAULT_CORPUS,
    question_of,
    read_groups,
)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def trajectory_documents(
    runtime: Any, prompts: list[str], *, max_bytes: int, penalty: float
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """把每条提问的**模型自答**铺成一篇文档（`[边界符] + 正文 + 换行`），并回带逐篇的字节账。"""

    chunks: list[list[int]] = []
    ledger: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        answer = runtime.chat(
            prompt, history=[], learn=False, max_length=max_bytes, repetition_penalty=penalty
        )
        prefix = f"问：{prompt}\n答：{answer}".encode() + bytes([0x0A])
        chunks.append([int(runtime.model.substrate.config.boundary_symbol), *prefix])
        ledger.append(
            {
                "doc_index": index,
                "prompt_bytes": len(prompt.encode()),
                "answer_bytes": len(answer.encode()),
                "document_bytes": len(prefix),
                "answer_stopped_early": len(answer.encode()) < max_bytes,
            }
        )
    return chunks, ledger


def priceable_points(sweep: list[dict[str, Any]], positions: int) -> list[float]:
    """按 §2ak 冻结的三条线筛出**可用可价点**（τ 网格只有 7 档，所以只报哪几档过线）。"""

    limit_docs = positions // 4
    hit: list[float] = []
    for row in sweep:
        late_enough = (
            row["docs_with_false_fire"] == 0
            or (row["first_false_fire_fraction_of_doc_median"] or 0.0) >= 0.5
        )
        if (
            (row["true_stop_recall"] or 0.0) >= 0.5
            and row["docs_with_false_fire"] <= limit_docs
            and late_enough
        ):
            hit.append(float(row["threshold"]))
    return hit


def priceable_ratio_points(sweep: list[dict[str, Any]], positions: int) -> list[float]:
    """§2am 那一档的**比较式**判据：规则＝"边界符离第一位只差 K 倍就收笔"，K 取网格值。

    冻结线（2026-09-29 写下，先于任何 ratio 读数）：与 §2ak 同形三条，同时满足才算**可比价点**——
    ① `seam_hit_rate ≥ 0.5`；② `docs_with_false_fire ≤ N//4`；③ 若确有误收，
    `first_false_fire_fraction_of_doc_median ≥ 0.5`。分支与 §2ak 一致：
    重训件有、base 没有 ⇒ 比较式停止在自身轨迹上可用且是配方买来的；两枚都有 ⇒ 配方贡献
    `not_resolved`；两枚都没有 ⇒ 连比较式也判死，停止这条线在产品侧没有未试过的形态了。
    """

    limit_docs = positions // 4
    hit: list[float] = []
    for row in sweep:
        late_enough = (
            row["docs_with_false_fire"] == 0
            or (row["first_false_fire_fraction_of_doc_median"] or 0.0) >= 0.5
        )
        if (
            (row["seam_hit_rate"] or 0.0) >= 0.5
            and row["docs_with_false_fire"] <= limit_docs
            and late_enough
        ):
            hit.append(float(row["ratio_cap"]))
    return hit


def priceable_ratio_with_floor(
    sweep: list[dict[str, Any]], positions: int
) -> list[tuple[float, float]]:
    """§2an：给比较式规则**加一条位置条件**之后的定价——产品若规定"正文走到 floor 比例之后才允许收笔"，
    还有哪些 (K, floor) 组合同时过线。

    冻结线（2026-09-30 写下，先于任何 `by_floor` 读数）：floor 网格只取 {0.25, 0.50}
    （0.00 就是 §2am 本身、已判死；0.75 不许取，因为那等于只允许在答复最后 1/4 收笔＝把结尾硬编码进产品）。
    称 (K, floor) 为**带条件可价点**，当且仅当同时满足：① `seam_hit_rate ≥ 0.5`；
    ② 该 floor 下 `docs_with_false_fire_by_floor ≤ N//4`。
    分支与前面各档同形：重训件有、base 没有 ⇒ 这条配方配一条温和的位置条件就能用（下一档才是产品的事）；
    两枚都有 ⇒ 位置条件是主角、配方贡献记 `not_resolved`；两枚都没有 ⇒ **"位置局部判据"这一族整族收口**，
    停止问题只剩两个形态：训练吃过自身轨迹（要机器时间），或者产品把结尾硬编码（那不是模型的能力）。
    """

    limit_docs = positions // 4
    hit: list[tuple[float, float]] = []
    for row in sweep:
        by_floor = row.get("docs_with_false_fire_by_floor") or {}
        if (row["seam_hit_rate"] or 0.0) < 0.5:
            continue
        for floor in (0.25, 0.50):
            if by_floor.get(f"{floor:.2f}", 10**9) <= limit_docs:
                hit.append((float(row["ratio_cap"]), floor))
    return hit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", default="output/a31_ding3_boundary/checkpoint.pt")
    parser.add_argument("--base", default="output/a26_p1/checkpoint.pt")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--groups", type=int, default=30)
    parser.add_argument("--exchanges", type=int, default=3)
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--max-bytes", type=int, default=256)
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = PROJECT_ROOT / corpus
    retrain = PROJECT_ROOT / args.retrain
    base = PROJECT_ROOT / args.base
    sha_before = {
        "retrain": _sha_bytes(retrain.read_bytes()),
        "base": _sha_bytes(base.read_bytes()),
    }

    data = read_groups(corpus, groups=args.groups, exchanges=args.exchanges)
    prompts = [q for g in data["held_out"] if (q := question_of(g[0]))][: args.positions]
    if len(prompts) < args.positions:
        raise SystemExit(f"held-out 只拆出 {len(prompts)} 条提问，不足 {args.positions}")
    prompts_sha = _sha_bytes("\n".join(prompts).encode())

    runs: dict[str, Any] = {}
    for label, checkpoint in (("retrain", retrain), ("base", base)):
        from api.seed_runtime import SeedRuntime

        runtime = SeedRuntime.load(checkpoint)
        boundary = int(runtime.model.substrate.config.boundary_symbol)
        chunks, ledger = trajectory_documents(
            runtime, prompts, max_bytes=args.max_bytes, penalty=args.penalty
        )
        result = audit(runtime, chunks, boundary, mask=args.mask)
        runs[label] = {
            "checkpoint_sha256_before": sha_before[label],
            "prompts_sha256": prompts_sha,
            "docs_sha256": _sha_bytes(b"".join(bytes(chunk[1:]) for chunk in chunks)),
            "positions": len(prompts),
            "stream_symbols": result["stream_symbols"],
            "end_positions": result["end_positions"],
            "other_positions": result["other_positions"],
            "faces": {kind: result["faces"][kind] for kind in ("end", "other", "turn_marker")},
            "threshold_sweep": result["threshold_sweep"],
            "ratio_sweep": result["ratio_sweep"],
            "priceable_thresholds": priceable_points(result["threshold_sweep"], len(prompts)),
            "priceable_ratio_thresholds": priceable_ratio_points(
                result["ratio_sweep"], len(prompts)
            ),
            "priceable_ratio_with_floor": priceable_ratio_with_floor(
                result["ratio_sweep"], len(prompts)
            ),
            "document_ledger": ledger,
        }
        del runtime

    retrain_hits = runs["retrain"]["priceable_thresholds"]
    base_hits = runs["base"]["priceable_thresholds"]
    if retrain_hits and not base_hits:
        verdict = (
            f"可用可价点存在且只属于重训件（τ∈{retrain_hits}）⇒ 阈值停止在它自己的轨迹上可用，"
            "这条配方买到了可用面；下一档才是产品出口（τ 的来路与回退面须 owner 裁）"
        )
    elif retrain_hits and base_hits:
        verdict = (
            f"两枚件都有可用可价点（重训 {retrain_hits}／base {base_hits}）"
            "⇒ 关于配方贡献记 not_resolved，可用性与这条配方无关"
        )
    else:
        verdict = (
            "两枚件都没有可用可价点 ⇒ 丁-4 三面判死（教师强制 §2w／生成 argmax §2ah-§2ai／自身轨迹阈值 §2ak）"
            "⇒ 停止这条线整体收口，剩下的方向都要 owner 裁"
        )

    retrain_ratio = runs["retrain"]["priceable_ratio_thresholds"]
    base_ratio = runs["base"]["priceable_ratio_thresholds"]
    if retrain_ratio and not base_ratio:
        verdict_ratio = f"可比价点只属于重训件（K∈{retrain_ratio}）⇒ 比较式停止在它自己的轨迹上可用，且是这条配方买来的"
    elif retrain_ratio and base_ratio:
        verdict_ratio = f"两枚件都有可比价点（重训 {retrain_ratio}／base {base_ratio}）⇒ 关于配方贡献记 not_resolved"
    else:
        verdict_ratio = (
            "两枚件都没有可比价点 ⇒ 连比较式（离第一位只差 K 倍）也判死，"
            "停止这条线在产品侧没有未试过的形态了，剩下的方向都要 owner 裁"
        )

    retrain_floor = runs["retrain"]["priceable_ratio_with_floor"]
    base_floor = runs["base"]["priceable_ratio_with_floor"]
    if retrain_floor and not base_floor:
        verdict_floor = (
            f"带条件可价点只属于重训件（(K,floor)∈{retrain_floor}）⇒ 这条配方配一条不超过半程的位置条件就能用；"
            "接不接进产品出口是 owner 的事，本件不动产品"
        )
    elif retrain_floor and base_floor:
        verdict_floor = (
            f"两枚件都有带条件可价点（重训 {retrain_floor}／base {base_floor}）⇒ 主角是那条位置条件，"
            "关于配方贡献记 not_resolved"
        )
    else:
        verdict_floor = (
            "两枚件都没有带条件可价点（floor 只允许取 0.25/0.50）⇒ 「位置局部判据」整族收口，"
            "停止问题只剩两个形态：训练吃过自身轨迹（要机器时间），或者产品把结尾硬编码（那不是模型的能力）"
        )

    report = {
        "format": "taiji-a30-ding3-trajectory-threshold-v3",
        "format_note_v3": (
            "v3 只在每臂加 `priceable_ratio_with_floor` 与一条 `verdict_ratio_with_floor`"
            "（判据＝§2an，写在 `priceable_ratio_with_floor` 的 docstring 里，先于数）；"
            "v1/v2 的表与判读字段一字未动 ⇒ 与已入库的 24pos／24pos_v2 读数同格可比"
        ),
        "format_note_v2": (
            "v2 只在每臂加 `ratio_sweep` 与 `priceable_ratio_thresholds`、并多一条 `verdict_ratio_face`"
            "（判据＝§2am，写在 `priceable_ratio_points` 的 docstring 里，先于数）；"
            "v1 的 `threshold_sweep`／`priceable_thresholds`／`verdict` 一字未动 ⇒ 与已入库的 24pos 读数同格可比"
        ),
        "prereg": "PLAN-A-30 §2ak（判读线先于数写在件 docstring）",
        "question": "边界符的概率在模型自己的正文与真接缝之间分不分得开，且误收砍在第几格（阈值停止可不可用）",
        "retrain_checkpoint": args.retrain,
        "base_checkpoint": args.base,
        "corpus": corpus.name,
        "mask": bool(args.mask),
        "decision_face": "utf8_masked_legal_set" if args.mask else "full_alphabet",
        "positions": args.positions,
        "max_bytes": args.max_bytes,
        "repetition_penalty": args.penalty,
        "false_fire_doc_limit": args.positions // 4,
        "prompts_sha256": prompts_sha,
        "runs": runs,
        "verdict": verdict,
        "verdict_ratio_face": verdict_ratio,
        "verdict_ratio_with_floor": verdict_floor,
        "instrument_guard": {
            "same_prompts_both_arms": runs["retrain"]["prompts_sha256"]
            == runs["base"]["prompts_sha256"],
            "end_bucket_is_the_imposed_seam": (
                runs["retrain"]["end_positions"] == runs["base"]["end_positions"] == args.positions
            ),
            "other_bucket_populated_on_own_text": (
                runs["retrain"]["other_positions"] > 0 and runs["base"]["other_positions"] > 0
            ),
            "retrain_sha256_unchanged": _sha_bytes(retrain.read_bytes()) == sha_before["retrain"],
            "base_sha256_unchanged": _sha_bytes(base.read_bytes()) == sha_before["base"],
        },
        "operational_definition": (
            "文档＝问：q\\n答：模型自己写的答案＋一个换行；接缝由本件强加，end 桶只有那一格，"
            "other 桶是它自己正文的每一格。所以这一格量的是'若它此刻该停，边界符的概率是否高于正文中间'，"
            "不是'它停没停'（后者＝§2ah 的 raw 计数，两枚件都 0/72 自停）。"
        ),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT
        / f"reports/taiji_a30_ding3_trajectory_threshold_{args.positions}pos_20260929.json"
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
                "positions": args.positions,
                "retrain_end_p_median": runs["retrain"]["faces"]["end"]["p_boundary"].get("median"),
                "base_end_p_median": runs["base"]["faces"]["end"]["p_boundary"].get("median"),
                "retrain_priceable": retrain_hits,
                "base_priceable": base_hits,
                "retrain_ratio_priceable": retrain_ratio,
                "base_ratio_priceable": base_ratio,
                "retrain_ratio_with_floor": retrain_floor,
                "base_ratio_with_floor": base_floor,
                "verdict": verdict,
                "verdict_ratio_face": verdict_ratio,
                "guard": report["instrument_guard"],
                "out": out.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

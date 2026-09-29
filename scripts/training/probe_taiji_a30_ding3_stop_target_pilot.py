"""PLAN-A-30 §3 丁-3 的**配对存在性证明**：不改解码、不改目标结构，只用现成 `learn_bytes`
的参数面把"结束目标"的**上下文形状**换掉，看结束位上的边界符会不会因此赢下 argmax。

机理已在码上（不是推理）：`taiji/organs.py:60-63` 让 `include_end_boundary=True` 在流尾加一个
边界符，而 `taiji/model.py:2865-2886` 对**流里的每一个符号**都 `observe(..., learn=True)`
⇒ 把"一答"作为一段喂进去并带上收尾边界，那一答的末尾就确实是一个**被学的目标**。
而默认语料每篇只有一组问答（§2u：123,090 行里只有 39 行含文档内接缝）⇒ 今天这个目标每篇只出现
一次，且永远落在"整篇结束"那一格：模型从未在"后面还得接着输入"的位置上练过收口。

两臂同底、同字节、同 epoch 数，**唯一差别＝同一个结束目标后面还有没有输入**：

* `granularity`：把 k 组问答串成**一段会话、一个 episode**，每答完一答就收尾一个边界符
  （首段 `include_start_boundary=True, reset=True`，其后各段 `include_start_boundary=False,
  reset=False`）⇒ 前 k−1 个结束目标**处在"后面还要继续"的上下文里**；
* `current`：同样的 k 组字节，但每篇各自 `learn_bytes`（前后各一个边界符、每篇一个新 episode）
  ＝今天训练面的形状 ⇒ 每个结束目标都恰好是 episode 的最后一格。

⇒ "只是多训了几步/多看了几个边界符"这条解释被这两臂挡住：两臂看到的**文本字节完全相同**、
边界符总数只差 k−1 个（件里逐臂报 `trained_symbols` 与 `boundary_targets`，不藏差）。

测量**复用 §2j/§2s 那一件**（`audit()`），不重抄链：held-out 的每组会话按"前缀截到第 j 答的末尾"
展开成 j 条流，于是 `audit` 的 `end` 桶正好就是两臂唯一差别的那批位置（每组 k 个）。
另带一条副作用读数（`other` 桶的 `p_next_true` 中位）——通用字节预测若被练坏，那里会露出来。

判读线（先看规则，后看数；n 由 `--groups × --exchanges` 定，样本量在件里）：
* `granularity` 的 `end.boundary_is_argmax_count ≥ current 的同名计数 + 3` ⇒ **机理成立**（存在性证明），
  丁-3 从"配方"升为"有配对证据的方案"，排机时才只欠规模；
* 两臂计数都不足 3 ⇒ `not_resolved`，不许写成"丁-3 无效"，只许写"这个规模的 n 不够判"；
* 副作用面掉一半以上 ⇒ 即便机理成立，也要如实写"它是拿通用预测换来的"。

基座只读：`base_sha256_unchanged` 为假则整件作废。本件**不写检查点、不改默认入口、不排训练机时**。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
#: 两套对照，各自只差**一个**变量：
#: * `shape`（v1 已入库那档）＝`granularity` 对 `current`，只差"同一个结束目标后面还有没有输入"；
#:   两臂的 `boundary_targets` 都是 `groups × exchanges × epochs`，**目标密度在这一档里没变过**。
#: * `all`＝再加 `sparse`（同样字节、同样分组，只在最后一答收尾）⇒ `granularity` 对 `sparse`
#:   只差**目标密度**（每组 k 个 vs 每组 1 个）。这是 §2x 留下的那一个未定价变量。
ARM_SETS = {
    "shape": ("granularity", "current"),
    "all": ("granularity", "sparse", "current"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_groups(path: Path, *, groups: int, exchanges: int) -> dict[str, Any]:
    """读 `groups` 组训练语料＋同样多组 held-out，每组 `exchanges` 条"问…答"。

    语料每行就是一组问答（`{"text": "问：…\\n答：…"}`）⇒ "串成一段会话"这件事在本件里
    是**按行号分组**造出来的，不是从真实多轮对话里来的。这条边界必须说：本件证的是
    *结束目标的上下文形状*这一个机理，不证明合成会话的内容连贯性。
    """

    need = groups * 2 * exchanges
    records: list[bytes] = []
    lines_read = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            lines_read += 1
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            content = payload["text"] if isinstance(payload, dict) and "text" in payload else text
            records.append(content.encode("utf-8"))
            if len(records) >= need:
                break
    if len(records) < need:
        raise SystemExit(f"corpus too short: need {need} records, read {len(records)}")
    trained = [records[i * exchanges : (i + 1) * exchanges] for i in range(groups)]
    held = [records[(groups + i) * exchanges : (groups + i + 1) * exchanges] for i in range(groups)]
    disjoint = all(any(a != b for a, b in zip(g, h, strict=False)) for g in trained for h in held)
    return {
        "trained": trained,
        "held_out": held,
        "lines_read": lines_read,
        "records_used": need,
        "heldout_disjoint_from_trained": disjoint,
    }


def prefix_chunks(groups: list[list[bytes]], boundary: int) -> list[list[int]]:
    """每组会话展开成 k 条"截到第 j 答末尾"的前缀流（`audit` 的输入形状：`[boundary, *正文]`）。

    这样 `audit` 的 `end` 桶里每一格都恰好是"一答刚说完"的那个位置——也就是两臂唯一差别所在。
    """

    chunks: list[list[int]] = []
    for group in groups:
        joined: list[int] = []
        for exchange in group:
            joined = [*joined, *exchange]
            chunks.append([boundary, *joined])
    return chunks


def train_arm(runtime: Any, groups: list[list[bytes]], *, arm: str, epochs: int) -> dict[str, Any]:
    """两臂各自吃下同样的字节，差别只在"一答之后还有没有输入"。

    训练入口取 **`runtime.model.substrate`（＝`Taiji`）**而不是 `runtime.model`（＝`Seed`）：
    `Seed.learn_bytes` 这层门面只转发了 `epochs/include_boundary/use_memory`
    （`seed/model.py:124-137`），**没有** `include_start_boundary`/`include_end_boundary`/`reset`
    的出口，而那三个正是"每答一个结束目标、且同一 episode 续喂"所必需的——
    件里 `facade_gap` 一条是**按签名机检**出来的，不是我读码的口供。
    """

    learner = runtime.model.substrate
    steps_before = int(runtime.model.tick)
    symbols = 0
    text_bytes = 0
    targets = 0
    observations = 0
    accuracy_sum = 0.0
    surprise_sum = 0.0
    started = time.perf_counter()
    for _ in range(epochs):
        for group in groups:
            feeds: list[tuple[bytes, dict[str, bool], int]] = []
            last = len(group) - 1
            if arm == "current":
                feeds = [(exchange, {"include_boundary": True}, 2) for exchange in group]
            elif arm == "sparse":
                #: **密度臂**：与 `granularity` 同样的字节、同样的 episode 分组与续喂方式，
                #: 但**只在最后一答**收尾一个边界符 ⇒ 每组的目标数从 k 降到 1（k＝`exchanges_per_group`）。
                #: 于是 `granularity` 对 `sparse` 是**只变目标密度**的单变量对照
                #: （§2x 那条读数里唯一没被定价的变量：两臂的 `boundary_targets` 都曾是 180）。
                for index, exchange in enumerate(group):
                    first = index == 0
                    final = index == last
                    feeds.append(
                        (
                            exchange,
                            {
                                "reset": first,
                                "include_start_boundary": first,
                                "include_end_boundary": final,
                            },
                            int(first) + int(final),
                        )
                    )
            else:
                for index, exchange in enumerate(group):
                    first = index == 0
                    feeds.append(
                        (
                            exchange,
                            {
                                "reset": first,
                                "include_start_boundary": first,
                                "include_end_boundary": True,
                            },
                            2 if first else 1,
                        )
                    )
            for exchange, kwargs, edges in feeds:
                result = learner.learn_bytes(exchange, epochs=1, **kwargs)
                count = int(result.get("observations", 0))
                observations += count
                accuracy_sum += float(result.get("online_accuracy", 0.0)) * max(1, count)
                surprise_sum += float(result.get("mean_surprise", 0.0)) * max(1, count)
                symbols += len(exchange) + edges
                text_bytes += len(exchange)
                #: `boundary_targets` 数的是**真带收尾边界的目标数**，不是喂入次数。
                #: v1 那份件里这一列写成了喂入次数（对 `granularity`／`current` 两臂恰好同值＝180，
                #: 所以 §2x 引的 180 不用改），但加进 `sparse` 臂之后它会把 60 报成 180——
                #: 冒烟就是这么暴露的（`targets_sparse` 与 `targets_granularity` 都报 6）。
                targets += (
                    1 if kwargs.get("include_boundary") or kwargs.get("include_end_boundary") else 0
                )
    divisor = max(1, observations)
    return {
        "arm": arm,
        "groups": len(groups),
        "exchanges_per_group": len(groups[0]) if groups else 0,
        "epochs": epochs,
        "trained_symbols": symbols,
        #: 跨臂的可比性口径＝**正文字节**（边沿边界符不算）：`granularity` 与 `sparse` 的
        #: `trained_symbols` 会因目标数不同而差几十个边界符，`trained_text_bytes` 必须逐臂相同。
        "trained_text_bytes": text_bytes,
        "boundary_targets": targets,
        #: 按喂入计划应当出现的收尾边界数（`sparse` 每组只有最后一答带收尾 ⇒ 是别的臂的 1/exchanges）。
        "expected_end_targets": len(groups) * epochs * (1 if arm == "sparse" else len(groups[0])),
        "learn_observations": observations,
        "train_online_accuracy": round(accuracy_sum / divisor, 6),
        "train_mean_surprise": round(surprise_sum / divisor, 6),
        #: `tick` 在这里**不是**证据：`learn_bytes` 每段都会按 `reset` 重记 dynamics，
        #: 冒烟实测两臂的 `tick` 增量都是**负**约 1.6e7（＝基座累计值被重记成从 0 起）——
        #: 这正是 §2m 记过的那条坑（"别拿 tick 增量当'被走到'"），本件改用 `learn_observations`。
        "tick_reading_delta": int(runtime.model.tick) - steps_before,
        "seconds": round(time.perf_counter() - started, 2),
    }


def measure(runtime: Any, chunks: list[list[int]], boundary: int, *, mask: bool) -> dict[str, Any]:
    from audit_taiji_a30_stop_signal_presence import audit

    return audit(runtime, chunks, boundary, mask)


def facade_gap(runtime: Any) -> dict[str, Any]:
    """机检"每答一个结束目标"这件事在**产品门面**上到底能不能表达（不靠我读码的口供）。"""

    import inspect

    wanted = ("include_start_boundary", "include_end_boundary", "reset")
    facade = inspect.signature(type(runtime.model).learn_bytes).parameters
    deep = inspect.signature(type(runtime.model.substrate).learn_bytes).parameters
    return {
        "facade_class": type(runtime.model).__name__,
        "substrate_class": type(runtime.model.substrate).__name__,
        "facade_learn_bytes_params": sorted(facade),
        "edge_split_required": list(wanted),
        "edge_split_reachable_from_facade": all(name in facade for name in wanted),
        "edge_split_present_on_substrate": all(name in deep for name in wanted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--groups", type=int, default=2, help="训练/held-out 各这么多组")
    parser.add_argument(
        "--exchanges", type=int, default=3, help="每组串这么多答（＝每组 end 位数）"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--arm-set",
        choices=tuple(ARM_SETS),
        default="shape",
        help="shape＝只差上下文形状（v1 那档）；all＝再加密度臂 sparse",
    )
    parser.add_argument(
        "--mask", action="store_true", help="在 UTF-8 合法集上量（默认关＝全字母表）"
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = PROJECT_ROOT / corpus

    from taiji import TaijiConfig

    boundary = int(TaijiConfig().boundary_symbol)
    data = read_groups(corpus, groups=args.groups, exchanges=args.exchanges)
    chunks = prefix_chunks(data["held_out"], boundary)
    expected_end_positions = args.groups * args.exchanges

    from api.seed_runtime import SeedRuntime

    def fresh() -> Any:
        return SeedRuntime.load(checkpoint)

    runtime = fresh()
    gap = facade_gap(runtime)
    before = measure(runtime, chunks, boundary, mask=args.mask)
    del runtime

    arms: list[dict[str, Any]] = []
    for arm in ARM_SETS[args.arm_set]:
        runtime = fresh()
        note = train_arm(runtime, data["trained"], arm=arm, epochs=args.epochs)
        face = measure(runtime, chunks, boundary, mask=args.mask)
        note["end"] = face["faces"]["end"]
        note["other"] = face["faces"]["other"]
        note["stream_symbols"] = face["stream_symbols"]
        arms.append(note)
        del runtime

    by_arm = {arm["arm"]: arm for arm in arms}
    pre_end = before["faces"]["end"]
    verdict = (
        "机理成立（granularity 在结束位上赢得 argmax 的次数比 current 多 ≥3）"
        if by_arm["granularity"]["end"]["boundary_is_argmax_count"]
        >= by_arm["current"]["end"]["boundary_is_argmax_count"] + 3
        else "not_resolved（两臂计数差 <3，或都不过 3 格）"
    )
    #: 密度判读只在 `--arm-set all` 下有对象；判读线与 shape 那条同规格（≥3 格才算成立），
    #: 先于数写在这里（§2x 留下的那一个未定价变量）。
    density_verdict = None
    delta_density = None
    if "sparse" in by_arm:
        g_end = by_arm["granularity"]["end"]
        s_end = by_arm["sparse"]["end"]
        delta_density = {
            "end_argmax_count": g_end["boundary_is_argmax_count"]
            - s_end["boundary_is_argmax_count"],
            "end_rank_median": round(
                (s_end["boundary_rank"]["median"] or 0) - (g_end["boundary_rank"]["median"] or 0),
                3,
            ),
            "end_p_boundary_median": round(
                (g_end["p_boundary"]["median"] or 0) - (s_end["p_boundary"]["median"] or 0), 6
            ),
            "targets_granularity": by_arm["granularity"]["boundary_targets"],
            "targets_sparse": by_arm["sparse"]["boundary_targets"],
        }
        density_verdict = (
            "密度成立（granularity 的结束位胜出数比 sparse 多 ≥3）"
            if g_end["boundary_is_argmax_count"] >= s_end["boundary_is_argmax_count"] + 3
            else "not_resolved（密度两臂差 <3，或都不过 3 格）"
        )
    report = {
        "format": "taiji-a30-ding3-stop-target-pilot-v1",
        "prereg": "PLAN-A-30 §3 丁-3 ＋本件 docstring 的判读线（先于数写下）",
        "question": "只用现成 learn_bytes 参数面把'结束目标的上下文形状'换掉，结束位上的边界符会不会开始赢",
        "checkpoint": args.checkpoint,
        "corpus": corpus.name,
        "mask": bool(args.mask),
        "decision_face": "utf8_masked_legal_set" if args.mask else "full_alphabet",
        "groups": args.groups,
        "exchanges_per_group": args.exchanges,
        "epochs": args.epochs,
        "end_positions_measured": expected_end_positions,
        "heldout_disjoint_from_trained": data["heldout_disjoint_from_trained"],
        "lines_read": data["lines_read"],
        "pre_pilot": {
            "end": pre_end,
            "other": before["faces"]["other"],
            "stream_symbols": before["stream_symbols"],
        },
        "arms": arms,
        "delta_vs_current": {
            "end_argmax_count": by_arm["granularity"]["end"]["boundary_is_argmax_count"]
            - by_arm["current"]["end"]["boundary_is_argmax_count"],
            "end_rank_median": round(
                (by_arm["current"]["end"]["boundary_rank"]["median"] or 0)
                - (by_arm["granularity"]["end"]["boundary_rank"]["median"] or 0),
                3,
            ),
            "end_p_boundary_median": round(
                (by_arm["granularity"]["end"]["p_boundary"]["median"] or 0)
                - (by_arm["current"]["end"]["p_boundary"]["median"] or 0),
                8,
            ),
            "other_p_next_true_median": round(
                (by_arm["granularity"]["other"]["p_next_true"]["median"] or 0)
                - (by_arm["current"]["other"]["p_next_true"]["median"] or 0),
                6,
            ),
        },
        "verdict": verdict,
        "arm_set": args.arm_set,
        "density_verdict": density_verdict,
        "delta_density": delta_density,
        "facade_gap": gap,
        "instrument_guard": {
            "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
            #: 两臂每篇的边沿符数差＝每组 (exchanges−1) 个**多出来的收尾边界**，再乘 epoch 数。
            #: 10g3x6e 实测 132,522 − 132,402 ＝ 120 ＝ 10×(3−1)×6 ⇒ 配对成立；
            #: 先前这条界漏乘 `epochs`（写成 10×2＝20），是守卫自己算式错，不是数据不匹配。
            "both_arms_ran_the_same_byte_count": abs(
                by_arm["granularity"]["trained_symbols"] - by_arm["current"]["trained_symbols"]
            )
            <= args.groups * (args.exchanges - 1) * args.epochs,
            "end_targets_match_feed_plan": all(
                arm["boundary_targets"] == arm["expected_end_targets"] for arm in arms
            ),
            "shape_pair_end_targets_equal": (
                by_arm["granularity"]["boundary_targets"] == by_arm["current"]["boundary_targets"]
            ),
            "density_ratio_is_exchanges": (
                "sparse" not in by_arm
                or by_arm["sparse"]["boundary_targets"] * args.exchanges
                == by_arm["granularity"]["boundary_targets"]
            ),
            "each_arm_actually_learned": all(arm["learn_observations"] > 0 for arm in arms),
            "end_positions_as_expected": all(
                arm["end"]["n"] == expected_end_positions for arm in arms
            )
            and pre_end["n"] == expected_end_positions,
            #: 三臂必须吃同样多的**正文字节**（边沿边界符允许差几十个）——这是 shape/density 两套
            #: 对照共同的配对前提；上一条 `both_arms_ran_the_same_byte_count` 只钉 granularity 与 current。
            "all_arms_same_text_bytes": len({arm["trained_text_bytes"] for arm in arms}) == 1,
            "audit_bucket_accounting_ok": all(
                arm["end"]["n"] + arm["other"]["n"] > 0 for arm in arms
            ),
        },
        "caveats": [
            "多轮会话是按语料行号分组**合成**的（默认语料每篇只有一组问答），本件只证'结束目标的上下文形状'这一个机理",
            "n ＝ groups × exchanges 个结束位；决策级规模需要另行排训练机时，本件不动默认入口、不写检查点",
        ],
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT
        / f"reports/taiji_a30_ding3_stop_target_pilot_{args.groups}g{args.exchanges}x{args.epochs}e_20260929.json"
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
                "end_positions": expected_end_positions,
                "pre_end": pre_end,
                "arms": [
                    {
                        "arm": arm["arm"],
                        "trained_symbols": arm["trained_symbols"],
                        "boundary_targets": arm["boundary_targets"],
                        "learn_observations": arm["learn_observations"],
                        "train_mean_surprise": arm["train_mean_surprise"],
                        "seconds": arm["seconds"],
                        "end": arm["end"],
                    }
                    for arm in arms
                ],
                "delta_vs_current": report["delta_vs_current"],
                "verdict": verdict,
                "density_verdict": density_verdict,
                "delta_density": delta_density,
                "guard": report["instrument_guard"],
                "out": out.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

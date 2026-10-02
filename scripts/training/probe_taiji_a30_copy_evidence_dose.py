"""复制回路加性证据的**剂量—响应**探针（零训练；缩放发生在仪器侧，不动产品源码）。

为什么要这一档：`taiji/model.py:2209-2228` 实读到——三条加性证据通道里 consolidated 前有
`consolidation_read_gain`(1.00)、memory 前有 `memory_read_gain`(3.00)、身份证据带
`identity_organ_evidence_gain`(16.00)，**只有复制回路是 `episodic_evidence + copy_circuit.evidence(...)`，
没有任何强度系数**。两端读数已在案（同一枚 (c) 件、同一批 300 篇：不挂回路 **186/300**、挂 seed-A **0/300**），
所以中间只补剂量档，不必再训件。

做法＝**接口级**包一层：把 `CopyCircuit.evidence` 的返回值乘 α，跑完撤掉实例属性回到类上的绑定方法。
刻意不碰实现行、不碰产品源码。

三条守卫（都设计成能为 false）：
* `unit_dose_bitwise_identical`——α=1 那档与"完全不包层"的那一趟必须**逐位相同**（否则实验连自己都没做对）；
* `wrapper_consumed_on_every_dose`＋每档调用次数——计数为 0 就说明补丁没被消费（"写了个不被走到的旋钮"是这类实验的经典假读数）；
* `checkpoint_untouched`——审计只读，落盘前后检查点 sha 必须相同。

α=0 只**报告**不硬失败：它是"算完再乘零"，与"根本没挂载"在这条审计路径上应当同值；若不同值，
正好暴露挂载还有第二条影响面，那是更值得知道的事。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402
from audit_taiji_a30_stop_signal_presence import audit, document_symbols  # noqa: E402


def _end_face_scalars(result: dict[str, Any]) -> dict[str, Any]:
    """只取比较用的标量列（不含调用计数），逐位比较时两边形状必须一致。"""

    end = result["faces"]["end"]
    return {
        "n": end["n"],
        "boundary_is_argmax_count": end["boundary_is_argmax_count"],
        "p_boundary_median": end["p_boundary"]["median"],
        "p_boundary_mean": end["p_boundary"]["mean"],
        "p_boundary_max": end["p_boundary"]["max"],
        "boundary_rank_median": end["boundary_rank"]["median"],
        "argmax_winners_top5": [
            (int(row["symbol"]), int(row["count"])) for row in end["argmax_winners_top5"]
        ],
        #: v1 加性：并取 `other` 面三条。理由见 `p_boundary_median_other` 的用途——它用来**独立证明**
        #: "下限确实改变了轨迹上的打分"。若接缝面不动而 other 面也不动，那说明下限根本没生效，
        #: 此时"接缝位置自己落在高相似度那一段"这条推论就不成立（这是那条推论的否证检验）。
        "other_n": result["faces"]["other"]["n"],
        "other_boundary_is_argmax_count": result["faces"]["other"]["boundary_is_argmax_count"],
        "other_p_boundary_median": result["faces"]["other"]["p_boundary"]["median"],
    }


def _make_scaled_evidence(
    original: Any,
    alpha: float,
    record: list[Any] | None = None,
    *,
    circuit: Any | None = None,
    scores: list[float] | None = None,
    floor_tau: float | None = None,
    ceiling_c: float | None = None,
) -> tuple[Any, list[int]]:
    """接口级包装：把 `evidence(**kwargs)` 的返回值乘 α，并数它**被消费**了几次。

    传 `record` 就顺手量注入的**硬度**（这条通道是 `gate * distribution`，`distribution` 是池化进
    257 维的软权重 ⇒ 量级全在学出来的那个标量 `gate` 上）。记的是每次调用的
    `(最大值, L1 和, 非零个数)`——用来判"加性证据"实际是不是当成硬值在用。
    """

    #: `calls` 三位：[被消费的调用数, 被**下限**静音的调用数, 被**上限**静音的调用数]。
    #: 后两位是 v12 补的——光有"包装器被走到"证明不了"过滤器开过枪"，
    #: 一个从未命中的 c 会给出与全剂量同值的读数却看不见自己是空的。
    calls = [0, 0, 0]

    def scaled(**kwargs: Any) -> Any:
        out_tensor = original(**kwargs)
        calls[0] += 1
        #: 相似度走**产品自己的那条算式**（`CopyCircuit._cosine` 的 docstring 明写它与 `best_match`
        #: 同一套），所以这里不另实现一份打分。但"算完再丢"与"根本不发"在**计数器副作用**上不等价
        #: （`_chosen_event` 的锁丢弃计数照旧会走），件里如实披露这一条。
        if (
            scores is not None or floor_tau is not None or ceiling_c is not None
        ) and circuit is not None:
            event = circuit.store.best_match(kwargs["cue"])
            score = None if event is None else float(circuit._cosine(kwargs["cue"], event.cue))
            if scores is not None:
                scores.append(-1.0 if score is None else score)
            if floor_tau is not None and (score is None or score < floor_tau):
                calls[1] += 1
                return out_tensor * 0.0
            #: 上限档（DEBT-G19 的第二种形状）：**越自信越要收着**——相似度高于 c 的步不发。
            #: 本轮读数显示有害的接缝步在 0.4332、有用的复述步在 0.1~0.3 ⇒ 只有这个方向可能两全。
            if ceiling_c is not None and score is not None and score > ceiling_c:
                calls[2] += 1
                return out_tensor * 0.0
        if record is not None:
            with torch.no_grad():
                record.append(
                    (
                        float(out_tensor.max()),
                        float(out_tensor.abs().sum()),
                        int(torch.count_nonzero(out_tensor)),
                    )
                )
        return out_tensor * alpha

    return scaled, calls


def _make_store_scoped_arm(store: Any) -> tuple[Any, list[int], set[int]]:
    """**资格档（检索侧）**：只让"本次对话里被告知的内容"有资格被取到。

    与 §第二十九/三十次停靠 的区别要说清：那两档动的是**发射时刻**（发不发、什么时候发），
    这一档动的是**候选集**（谁有资格被 `best_match` 取到）——DEBT-G19 早已量到整条通道的注入
    全由装载信封里残留的少量**跨题陈旧事件**驱动，所以"把陈旧事件请出候选集"是另一条独立的形状。

    只用公开接口：`events()` 读、`clear()` 清空、`record()` 把**保留**的事件写回
    （代价是 `event_id` 会重新编号，这一点必须在件里披露，不许假装没发生）。
    返回 `(reset 函数, 计数器 [清掉的陈旧条数, 重置后在库条数最小值, 最大值], 陈旧 id 集合)`。
    计数器第 2/3 位在 reset 之前先按"全部"算，重置后才反映真实候选数。
    """

    counters = [0, 0, 0]
    stale_ids = {int(event.event_id) for event in store.events()}
    stale = [event for event in store.events()]

    def reset() -> None:
        kept_live = [event for event in store.events() if int(event.event_id) not in stale_ids]
        store.clear()
        for event in kept_live:
            store.record(event.content, event.cue)
        # 陈旧那几条此刻已不在库里：把它们重新登记为"下一次要清掉的对象"没有意义
        # （reset 的语义＝每题开头只留本题被告知的内容），故这里只记本次清掉了几条。
        counters[0] = max(counters[0], len(stale))
        live = int(store.count)
        counters[1] = live if counters[1] == 0 else min(counters[1], live)
        counters[2] = max(counters[2], live)

    return reset, counters, stale_ids


def _make_window_armed_evidence(original: Any, window: int, position: Any) -> tuple[Any, list[int]]:
    """**资格档**：只在答复的前 `window` 步发证据，之后把这条通道静音（PLAN-A-30 §第三十次停靠）。

    `position()` 由**调用方**给出"现在走到答复的第几步"——这条实现刻意不知道各链怎么数步：
    L2 仪器用生成环内的 observe 序位（1 步＝1 字节），复述面用它自己的口径。
    这样两半共用同一副档（同一个静音规则），但刻度各自如实声明；各写一份就是两把尺子。
    计数器＝`[总调用, 窗内发出, 窗外静音]`。
    """

    counters = [0, 0, 0]

    def armed(**kwargs: Any) -> Any:
        out = original(**kwargs)
        counters[0] += 1
        if int(position()) >= window:
            counters[2] += 1
            return out * 0.0
        counters[1] += 1
        return out

    return armed, counters


def _make_content_armed_evidence(
    original: Any, kind: str, seed: int = 20261002
) -> tuple[Any, list[float]]:
    """**内容档 vs 硬度档**（PLAN-A-30 §第二十八次停靠）：接口级替换最终进 logits 的那个向量。

    * `permutation`：按固定种子置换元素位置 ⇒ 多重集与 L1/max 逐位守恒，唯一被毁掉的是
      "哪一维对应哪个符号"；
    * `frozen`：冻结到**第一条非零**证据。v13 曾把冻结源取在"第一次调用"，而那时 store 还是空的
      ⇒ 整档实为"把通道永久关闭"（自停 66/72 与不挂回路同值、`max_rel_l1_diff=1.0` 是指纹）——
      这一坑记在这儿，别再踩。

    返回 `(替换后的 callable, 计数器)`，计数器＝`[被走到, 原 L1 累加, 替换后 L1 累加,
    单次相对差最大值, 冻结发生在第几次调用, 冻结源自己的 L1]`。
    **这台仪器只有一副内容档**：`probe_taiji_a30_stop_failure.py`（自身轨迹面）与
    `score_taiji_r2_copy_circuit_chat_cap.py`（复述命中面）都从这里取，否则两半读的不是同一个操作。
    """

    counters = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rng = random.Random(seed)
    perm_cache: dict[int, Any] = {}
    frozen: list[Any] = []

    def armed(**kwargs: Any) -> Any:
        out = original(**kwargs)
        counters[0] += 1
        single = float(out.abs().sum())
        counters[1] += single
        if kind == "permutation":
            #: 置换表按**实际元素数**现取（不硬写词表尺寸；尺寸一变就响亮地重新洗牌而不是错位）。
            n = int(out.numel())
            if n not in perm_cache:
                order = list(range(n))
                rng.shuffle(order)
                perm_cache[n] = torch.tensor(order, dtype=torch.long)
            flat = out.reshape(-1)
            swapped = flat.index_select(0, perm_cache[n].to(flat.device)).reshape(out.shape)
        else:
            if not frozen and single > 0.0:
                frozen.append(out.clone())
                counters[4] = float(counters[0])
                counters[5] = single
            swapped = frozen[0] if frozen else out
        new = float(swapped.abs().sum())
        counters[2] += new
        #: 硬度守恒按**逐次相对差最大值**判，不按两趟累加之差：置换只改求和顺序，
        #: float32 下 1,881 次累加能差出 1e-4（n=1 冒烟实测），那是表示层噪声而不是"硬度变了"。
        if kind == "permutation" and single > 0.0:
            counters[3] = max(counters[3], abs(new - single) / single)
        return swapped

    return armed, counters


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--corpus", default="data/simple_zh/dialogue_extended_clean.jsonl")
    parser.add_argument("--docs", type=int, default=300)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--append-newline", action="store_true")
    parser.add_argument(
        "--doses", default="0.25,0.50,1.0", help="逗号分隔的 α；含 1.0 时自动做逐位锚点复检"
    )
    parser.add_argument("--out-report", required=True)
    parser.add_argument(
        "--tail-scores",
        type=int,
        default=0,
        help="按调用顺序留下最后 N 个相似度。**`--docs 1` 时最后一次调用就是那篇文档的接缝位置**"
        " ⇒ 用它把'接缝落在高相似度段'从推论升级成直接量到的数。",
    )
    parser.add_argument(
        "--relevance-ceiling-c",
        type=float,
        default=None,
        help="DEBT-G19 上限档：相似度**高于** c 的步把证据归零（与下限共用同一条 `_cosine` 与同一个包装器）",
    )
    parser.add_argument(
        "--record-scores",
        action="store_true",
        help="记录每一步被挑中告知的余弦相似度（τ 从这个分布里取，不拍脑袋定）",
    )
    parser.add_argument(
        "--relevance-floor-tau",
        type=float,
        default=None,
        help="DEBT-G19 修法①的模拟：相似度低于 τ ⇒ 证据归零（模拟 best_match 返回 None）",
    )
    parser.add_argument(
        "--record-magnitudes",
        action="store_true",
        help="顺手量这条通道的注入硬度（`(max, L1, nnz)` 三列分位数）——用来验'它是硬值还是软加性'",
    )
    args = parser.parse_args()

    out = Path(args.out_report)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    if out.exists():
        print(f"[拒绝落盘] {out} 已存在 ⇒ 不许覆盖既有证据件，换个文件名再跑", file=sys.stderr)
        return 2

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    corpus = Path(args.corpus)
    corpus = corpus if corpus.is_absolute() else PROJECT_ROOT / corpus

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:  # 没有可缩放的东西就响亮停，别静默给一排 0
        raise RuntimeError("copy_circuit 不在场 ⇒ 没有可缩放的证据通道，本档不成立")
    boundary = int(substrate.config.boundary_symbol)
    sample = document_symbols(corpus, args.docs, append_newline=args.append_newline)
    chunks = sample["chunks"]

    # 不包任何一层的那一趟：α=1 的逐位锚点。
    unpatched = _end_face_scalars(audit(runtime, chunks, boundary, mask=args.mask))

    doses: dict[str, Any] = {}
    unit_dose_identical: bool | None = None
    for alpha in [float(x) for x in args.doses.split(",") if x.strip()]:
        circuit = substrate.copy_circuit
        original_bound = circuit.evidence
        magnitudes: list[Any] | None = [] if args.record_magnitudes else None
        scores: list[float] | None = [] if args.record_scores else None
        scaled, calls = _make_scaled_evidence(
            original_bound,
            alpha,
            record=magnitudes,
            circuit=circuit,
            scores=scores,
            floor_tau=args.relevance_floor_tau,
            ceiling_c=args.relevance_ceiling_c,
        )
        circuit.evidence = scaled
        try:
            row = _end_face_scalars(audit(runtime, chunks, boundary, mask=args.mask))
        finally:
            del circuit.evidence  # 撤实例属性 ⇒ 下一档拿到的仍是类上的绑定方法
        assert circuit.evidence == original_bound, "补丁没撤干净 ⇒ 后面的档会叠乘"
        if alpha == 1.0:
            #: 逐位锚点**只在下限关闭时**才是"包层不扰动"的检验；下限开着时两趟**本就该不同**
            #: （接缝与 other 的差别正是那一枪的内容），所以那时把这条读成 false 是误读。
            #: 于是下限开着时报 `None`（不适用），而不是报 false。
            any_filter = (
                args.relevance_floor_tau is not None or args.relevance_ceiling_c is not None
            )
            unit_dose_identical = None if any_filter else row == unpatched
            if any_filter:
                row["anchor_not_applicable_reason"] = "floor is active ⇒ two passes should differ"
        row["evidence_calls"] = calls[0]
        #: "过滤器开过几枪"与"包装器被走到"是两件事：前者才支持"这一档确实动过路径"。
        row["silenced_by_floor_calls"] = calls[1]
        row["silenced_by_ceiling_calls"] = calls[2]
        row["silenced_share"] = round((calls[1] + calls[2]) / calls[0], 6) if calls[0] else None
        if scores:
            ordered = sorted(scores)
            row["picked_cosine"] = {
                "median": round(statistics.median(ordered), 4),
                "p10": round(ordered[int(0.10 * (len(ordered) - 1))], 4),
                "p25": round(ordered[int(0.25 * (len(ordered) - 1))], 4),
                "p75": round(ordered[int(0.75 * (len(ordered) - 1))], 4),
                "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
                "max": round(ordered[-1], 4),
                "share_below_0p3": round(sum(1 for x in ordered if 0 <= x < 0.3) / len(ordered), 4),
                "share_no_event": round(sum(1 for x in ordered if x < 0) / len(ordered), 4),
            }
            if args.tail_scores:
                row["picked_cosine_tail"] = [round(x, 4) for x in scores[-args.tail_scores :]]
            scores.clear()
        if magnitudes:
            row["injected_logit_max"] = {
                "median": round(statistics.median([m[0] for m in magnitudes]), 4),
                "p90": round(sorted(magnitudes)[int(0.9 * (len(magnitudes) - 1))][0], 4),
                "max": round(max(m[0] for m in magnitudes), 4),
            }
            row["injected_l1_median"] = round(statistics.median([m[1] for m in magnitudes]), 4)
            row["injected_nnz_median"] = statistics.median([m[2] for m in magnitudes])
            row["empty_evidence_share"] = round(
                sum(1 for m in magnitudes if m[2] == 0) / len(magnitudes), 4
            )
            magnitudes.clear()
        doses[str(alpha)] = row

    sha_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "format": "taiji-a30-copy-evidence-dose-v1",
        "question": "把复制回路的加性证据按 α 缩放／按相似度下限截断，语料接缝上的停止信号能不能回来",
        "relevance_floor_tau": args.relevance_floor_tau,
        "relevance_ceiling_c": args.relevance_ceiling_c,
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "circuit_sha256": hashlib.sha256((PROJECT_ROOT / args.circuit).read_bytes()).hexdigest()[
            :16
        ],
        "docs": args.docs,
        "documents_face": (
            "corpus_body_plus_appended_newline" if args.append_newline else "corpus_body"
        ),
        "decision_face": "utf8_masked_legal_set" if args.mask else "full_alphabet",
        "anchors_on_record": {
            "circuit_absent_same_checkpoint": "186/300（档 (c) 不挂回路，docs_sha256 同批）",
            "circuit_mounted_full_dose": "0/300（档 (c)＋seed-A）",
        },
        "unpatched_end_face": unpatched,
        "doses": doses,
        "instrument_guard": {
            "unit_dose_bitwise_identical": unit_dose_identical,
            "wrapper_consumed_on_every_dose": all(r["evidence_calls"] > 0 for r in doses.values()),
            "evidence_calls_by_dose": {k: v["evidence_calls"] for k, v in doses.items()},
            "checkpoint_untouched": sha_before == sha_after,
        },
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {"doses": report["doses"], "guard": report["instrument_guard"]}, ensure_ascii=False
        )[:1500]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A2.4 重测：复制回路接进 chat 协议后，CAP D+E 全量＋成句率（对照臂 vs 电路臂）。

预注册判据沿用 `M5_R2_A2_3_PREREG_20260925.md` §4-S2（D+E>0）＋§4-③（成句率不塌，
本次补上正式仪器）。两臂同协议同题集，唯一差别＝是否挂载并启用剪贴板。

**口径如实声明**：计分读**基底原始字节答案**（`generate_input`），不经语言器官表层——
当前产品态 `chat_enabled=false` 下器官把答案统一替换为占位句（2026-09-23 真机记录），
经器官计分测到的是器官不是模型。除表层外本件走的全是产品原语：
`_serialize`、`_record_told_history`（A2.4 接线本体）、`generate_input(reset=True)`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/cap0_eval_set_v2.json"
MAX_ANSWER_BYTES = 64


def _answer_raw(
    runtime: Any,
    prompt: str,
    history: list[tuple[str, str]],
    *,
    utf8_strict: bool = False,
    repetition_penalty: float = 0.0,
    max_bytes: int = MAX_ANSWER_BYTES,
) -> str:
    """产品装配下取基底原始答复（SPEC-A-21 那条冻结链）。

    `utf8_strict=True`＝把**产品表层链用的同一把解码掩码**（SPEC-R2-02，`SeedRuntime.chat` 里
    `generate_input(..., utf8_strict=True)`）套到这条评测链上——默认 `False` ⇒ 冻结读数逐位不变。
    为什么要有这一档：`PLAN-A-30` 要分「是掩码放大了电路的复读，还是语言器官」，
    而只有把掩码单独加在原始字节链上，才能把"掩码"与"器官"这两手分开量（不然只有两端可比）。
    `max_bytes`＝生成预算，默认 64（冻结面逐位不变）。**它是 `PLAN-A-30` §2h 量出来的一个口径缺陷**：
    产品 `chat()` 的预算是 256，而这条评测链一直是 64 ⇒ 跨链比较"表层比原始链差"时，
    链与预算两个变量同时动了。要分开就得能把预算拨到 256（`probe_taiji_a30_surface_repetition.py`
    的 `--max-bytes` 走的就是这一手）。
    """

    from api.seed_runtime import _TURN_MARKERS
    from taiji import InputFrame

    text = runtime._serialize(prompt, history)
    substrate = runtime.model.substrate
    circuit = substrate.copy_circuit
    if circuit is not None:
        runtime._record_told_history(circuit, history)
    frame = InputFrame(
        input_id=f"a24-eval:{substrate.tick}",
        modality="text",
        payload=text.encode("utf-8"),
        source="r2.a24.eval",
        timestamp=substrate.tick,
        provenance="external",
        confidence=1.0,
    )
    raw = runtime.model.generate_input(
        frame,
        max_bytes,
        stop_at_boundary=True,
        sample=False,
        utf8_strict=utf8_strict,
        repetition_penalty=repetition_penalty,
    )
    answer = raw.decode("utf-8", errors="replace")
    for marker in _TURN_MARKERS:
        index = answer.find(marker)
        if index >= 0:
            answer = answer[:index]
    return answer.strip()


def _quantiles_or_none(values: list[float]) -> dict[str, Any]:
    """`_quantiles` 的空输入版本：**没有样本要如实记 `n=0`**，不许让统计函数替仪器抛异常。"""

    if not values:
        return {"n": 0, "note": "no hits on this arm"}
    return _quantiles(values)


def _quantiles(values: list[float]) -> dict[str, Any]:
    """分位数摘要——用来让每条链报**自己**的相似度分布，而不是复用别条链的 τ。"""

    import statistics

    ordered = sorted(values)
    n = len(ordered)

    def at(q: float) -> float:
        return round(ordered[min(n - 1, int(q * (n - 1)))], 4)

    return {
        "n": n,
        "median": round(statistics.median(ordered), 4),
        "p25": at(0.25),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "max": round(ordered[-1], 4),
        "share_below_0p3": round(sum(1 for x in ordered if x < 0.3) / n, 4),
        "share_no_event": round(sum(1 for x in ordered if x < 0) / n, 4),
    }


def run_arm(
    checkpoint: Path,
    circuit_payload: str | None,
    *,
    evidence_utf8_gate: bool = False,
    evidence_alpha: float = 1.0,
    evidence_floor_tau: float | None = None,
    evidence_ceiling_c: float | None = None,
    record_scores: bool = False,
    evidence_content_arm: str | None = None,
    perm_seed: int = 20261002,
    evidence_window_steps: int | None = None,
    store_scope_conversation: bool = False,
    oracle_selector: bool = False,
    max_bytes: int = MAX_ANSWER_BYTES,
    limit: int | None = None,
) -> dict[str, Any]:
    """一臂：CAP 的 D+E 计分（基底原始字节）。

    `evidence_utf8_gate`（PLAN-A-25）：只在评测期把复制回路的加性证据按 UTF-8 位置状态门控
    ——默认 False ⇒ 与冻结链逐位相同；开启走 `Taiji.set_copy_evidence_utf8_gate` 运行时覆写。
    `max_bytes`＝生成预算，默认 64（冻结面逐位不变）；`PLAN-A-30` §2h/§2i 查出预算本身就是
    一个会动读数的变量（同链同装配把 64 拨到 256，命中 3→9、成句 13→6）⇒ 用它做同装配双预算档。
    `limit`＝只取前 N 道题（D/E 混排后取前 N，两臂用同一子集 ⇒ 配对成立；默认全量）。
    """

    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    if circuit_payload is not None:
        runtime.enable_copy_circuit(PROJECT_ROOT / circuit_payload)
    if evidence_utf8_gate:
        runtime.model.substrate.set_copy_evidence_utf8_gate(True)
    calls = [0]
    scores: list[float] = []
    if (
        evidence_alpha != 1.0
        or evidence_floor_tau is not None
        or evidence_ceiling_c is not None
        or record_scores
    ):
        # PLAN-A-30 §DEBT-G19 的剂量档：复用剂量探针那个接口级包装（不另写一份缩放）。
        from probe_taiji_a30_copy_evidence_dose import _make_scaled_evidence

        circuit = runtime.model.substrate.copy_circuit
        if circuit is None:
            raise RuntimeError("要求缩放证据但回路不在场 ⇒ 这一臂没有可缩放的通道")
        scaled, calls = _make_scaled_evidence(
            circuit.evidence,
            evidence_alpha,
            circuit=circuit,
            scores=scores if record_scores else None,
            floor_tau=evidence_floor_tau,
            ceiling_c=evidence_ceiling_c,
        )
        circuit.evidence = scaled
    #: v13/v14 那副内容档（PLAN-A-30 §第二十八次停靠）搬到剂量探针里**共用**：
    #: 轨迹面与复述面必须做同一个置换／冻结操作，否则两半读的不是同一件事。
    content_counters = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if evidence_content_arm is not None:
        from probe_taiji_a30_copy_evidence_dose import _make_content_armed_evidence

        circuit = runtime.model.substrate.copy_circuit
        if circuit is None:
            raise RuntimeError("要求内容分离档但回路不在场 ⇒ 没有可替换的证据通道")
        armed, content_counters = _make_content_armed_evidence(
            circuit.evidence, evidence_content_arm, perm_seed
        )
        circuit.evidence = armed
    #: v17 资格档接到复述链：**用与 L2 仪器同一个 `generation_loop_span`** 数环内步（1 步＝1 字节），
    #: 这样两半是同一把刻度、同一副档；不是把 K 换算成"多少次调用"那种近似。
    #: v18 检索侧资格档（与 L2 仪器共用同一份实现，别写第二把尺子）。
    store_counters = [0, 0, 0]
    store_reset = None
    if store_scope_conversation:
        from probe_taiji_a30_copy_evidence_dose import _make_store_scoped_arm

        scoped = runtime.model.substrate
        if scoped.copy_circuit is None:
            raise RuntimeError("要求检索侧资格档但回路不在场 ⇒ 没有可缩范围的存储")
        store_reset, store_counters, _stale = _make_store_scoped_arm(scoped.copy_circuit.store)

    window_counters = [0, 0, 0]
    loop_steps = [0]
    if evidence_window_steps is not None:
        import sys

        from probe_taiji_a30_copy_evidence_dose import _make_window_armed_evidence
        from probe_taiji_a30_stop_failure import generation_loop_span

        substrate = runtime.model.substrate
        if substrate.copy_circuit is None:
            raise RuntimeError("要求资格档但回路不在场 ⇒ 没有可静音的证据通道")
        loop_first, loop_last = generation_loop_span(type(substrate).generate)
        original_observe = substrate.observe

        def observing(symbol: Any, **kwargs: Any) -> Any:
            frame = sys._getframe(1)
            step = original_observe(symbol, **kwargs)
            if frame.f_code.co_name == "generate" and loop_first <= frame.f_lineno <= loop_last:
                loop_steps[0] += 1
            return step

        substrate.observe = observing  # type: ignore[method-assign]
        armed_window, window_counters = _make_window_armed_evidence(
            substrate.copy_circuit.evidence, evidence_window_steps, lambda: loop_steps[0]
        )
        substrate.copy_circuit.evidence = armed_window  # type: ignore[method-assign]

    #: v19 检索侧 **oracle** 档（与 L2 仪器共用 `_make_oracle_selector_arm`，别写第二把尺子）：
    #: 把 `store.best_match` 换成"内容含本题 `expected_contains` 的第一条事件"，找不到透传原实现。
    oracle_state = {"tokens": [], "calls": 0, "found": 0, "fell_through": 0}
    oracle_set_tokens = None
    if oracle_selector:
        from probe_taiji_a30_copy_evidence_dose import _make_oracle_selector_arm

        oracle_substrate = runtime.model.substrate
        if oracle_substrate.copy_circuit is None:
            raise RuntimeError("要求 oracle 选择档但回路不在场 ⇒ 没有可替换的 best_match")
        oracle_fn, oracle_state, oracle_set_tokens = _make_oracle_selector_arm(
            oracle_substrate.copy_circuit.store
        )
        oracle_substrate.copy_circuit.store.best_match = oracle_fn  # type: ignore[method-assign]

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = [
        {**item, "dimension": dim}
        for dim in ("D", "E")
        for item in manifest["dimensions"][dim]["items"]
        if item.get("expected_contains")
    ]
    if limit is not None:
        items = items[:limit]
    ngram = build_ngram_model()
    rows = []
    for item in items:
        history: list[tuple[str, str]] = []
        if store_reset is not None:
            store_reset()  # 每题开头：只留本题被告知的内容可被取到
        if oracle_set_tokens is not None:
            #: 标签来自题面（cap 面 40 条里 36 条带 `expected_contains`）；缺标签**响亮停下**，
            #: 不许静默透传成"现状选择器"再假装这一档生效了。
            expected = [str(tok) for tok in item.get("expected_contains") or []]
            if not expected:
                raise RuntimeError(f"{item['id']} 没有 expected_contains ⇒ oracle 档无标签可用")
            oracle_set_tokens([tok.encode("utf-8") for tok in expected])
        turns = list(item["turns"])
        answer = ""
        for index, turn in enumerate(turns):
            loop_steps[0] = 0  # 换一轮答复：步刻度从 0 重数（与 L2 仪器同一条规矩）
            answer = _answer_raw(runtime, turn, history, max_bytes=max_bytes)
            if index + 1 < len(turns):
                history.append((turn, answer))
        hit = any(token in answer for token in item["expected_contains"])
        #: K-定价档（PLAN-A-30 第三十次停靠·设计预备第 2 条）：**命中发生在答复的第几个字节**必须先量出来，
        #: 才谈得上"只在前 K 步发证据"这种资格档；`answer[:60]` 那个截断字段量不到这件事，故另存整条的长度
        #: 与首次命中的字节偏移。无命中时偏移记 `None`（不许记 0——0 会被读成"命中在开头"）。
        offsets = [
            answer.encode("utf-8").find(token.encode("utf-8"))
            for token in item["expected_contains"]
            if token.encode("utf-8") in answer.encode("utf-8")
        ]
        first_hit_offset = min(offsets) if offsets else None
        answer_bytes = len(answer.encode("utf-8"))
        #: `formed_full`＝**整条答复**过 `well_formed`（与下面那把只看 60 字符前缀的尺子不同，
        #: 见 `well_formed_scope`）。三者合起来才回答"回路买到的到底是词在场，还是一句能看的答案"
        #: ——`DEBT-G12` 要的那把联合判据。旧字段 `correct`／`well_formed_rate` 语义逐位不变。
        rows.append(
            {
                "id": item["id"],
                "dimension": item["dimension"],
                "hit": hit,
                "first_hit_offset_bytes": first_hit_offset,
                "answer_bytes": answer_bytes,
                "formed_full": bool(well_formed(answer, ngram)),
                "answer": answer[:60],
            }
        )
    return {
        "items": len(rows),
        "correct": sum(1 for row in rows if row["hit"]),
        "joint_hits": sum(1 for row in rows if row["hit"] and row["formed_full"]),
        "formed_full_texts": sum(1 for row in rows if row["formed_full"]),
        "well_formed_rate": round(
            sum(1 for row in rows if well_formed(row["answer"], ngram)) / len(rows), 4
        ),
        "rows": rows,
        #: 命中偏移的分位数与"答复长度"并排存 ⇒ 一眼看得出前 K 步这类窗口能不能同时盖住复述。
        #: 空命中要显式记 `n=0` 而不是把空列表交给 `_quantiles`——对照臂**结构性没有命中**，
        #: 那不是数据错误（本档第一次跑就红在 `no median for empty data`，是仪器的事不是被测的事）。
        "hit_offsets": _quantiles_or_none(
            [
                float(row["first_hit_offset_bytes"])
                for row in rows
                if row["first_hit_offset_bytes"] is not None
            ]
        ),
        "answer_bytes_quantiles": _quantiles_or_none(
            [float(row["answer_bytes"]) for row in rows if row["hit"]]
        ),
        #: 剂量档的"被走到"计数（`_make_scaled_evidence` 实际消费了几次）。
        "evidence_calls": calls[0],
        "picked_cosine": _quantiles(scores) if record_scores and scores else None,
        #: 内容档自述：被走到几次、硬度守恒到哪、冻结落在第几次调用（守恒只对置换档有意义）。
        #: oracle 档自述：被走到几次、其中多少次真找到了含标签的事件（选对率）、多少次透传。
        "oracle_arm": {
            "requested": bool(oracle_selector),
            "calls": oracle_state["calls"],
            "found": oracle_state["found"],
            "fell_through": oracle_state["fell_through"],
        },
        "store_scope_arm": {
            "requested": bool(store_scope_conversation),
            "stale_removed": store_counters[0],
            "events_min": store_counters[1],
            "events_max": store_counters[2],
        },
        "window_arm": {
            "steps": evidence_window_steps,
            "calls": window_counters[0],
            "emitted": window_counters[1],
            "silenced": window_counters[2],
        },
        "content_arm": {
            "kind": evidence_content_arm,
            "calls": content_counters[0],
            "max_rel_l1_diff": content_counters[3],
            "frozen_at_call": content_counters[4] or None,
            "frozen_l1": content_counters[5] or None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default="output/taiji_r2_copy_circuit/judge/circuit-final.pt")
    parser.add_argument("--out-report", default=None)
    parser.add_argument(
        "--copy-evidence-utf8-gate",
        action="store_true",
        help="PLAN-A-25：把复制回路的加性证据按 UTF-8 位置状态门控（默认关 ⇒ 与冻结链逐位相同）",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=MAX_ANSWER_BYTES,
        help="生成预算，默认 64＝冻结面；PLAN-A-30 §2h 用它跑同装配的双预算档",
    )
    parser.add_argument(
        "--relevance-ceiling-c",
        type=float,
        default=None,
        help="DEBT-G19 上限档：相似度**高于** c 的步不发（越自信越收着）；与接缝档共用同一个包装器与同一条 `_cosine`",
    )
    parser.add_argument(
        "--relevance-floor-tau",
        type=float,
        default=None,
        help="DEBT-G19 修法①的模拟：相似度低于 τ 的步把证据归零（与接缝档共用同一个包装器与同一条 `_cosine`）",
    )
    parser.add_argument(
        "--copy-evidence-alpha",
        type=float,
        default=1.0,
        help="DEBT-G19 剂量档：治疗臂把回路加性证据乘 α（默认 1.0 ⇒ 与已入库两臂档逐位可比）",
    )
    parser.add_argument(
        "--record-scores",
        action="store_true",
        help="量这条链**自己的**被挑中告知相似度分布（τ 必须按链路各取，跨链套用的后果见 PLAN-A-30 第十八次停靠）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只取前 N 道题（D/E 混排后取前 N，两臂同一子集）",
    )
    parser.add_argument(
        "--evidence-content-arm",
        choices=("permutation", "frozen"),
        default=None,
        help="内容档：与自身轨迹面共用剂量探针里那副 `_make_content_armed_evidence`"
        "（`permutation`＝整根置换、硬度逐位守恒；`frozen`＝冻结到第一条非零证据）。"
        "默认关 ⇒ 与已入库各档逐位可比。",
    )
    parser.add_argument(
        "--perm-seed",
        type=int,
        default=20261002,
        help="置换档的固定种子（写进件里，可复现）",
    )
    parser.add_argument(
        "--evidence-window-steps",
        type=int,
        default=None,
        help="v17 资格档：只在答复的前 K 步发回路证据，之后静音。刻度与 L2 仪器同源（同一个 `generation_loop_span`、1 步＝1 字节），故两半可用同一个 K。",
    )
    parser.add_argument(
        "--store-scope-conversation",
        action="store_true",
        help="v18 检索侧资格档：每题开头把装载信封里的陈旧事件请出候选集，与 L2 仪器共用同一份实现。",
    )
    parser.add_argument(
        "--oracle-selector",
        action="store_true",
        help="v19 检索侧 oracle 档：每题把 `best_match` 换成'内容含本题 expected_contains 的第一条事件'。"
        "与 L2 仪器共用同一份实现；停止面已实测 Δ=0，这档用在复述面验'选对能买多少'。",
    )
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    gate = bool(args.copy_evidence_utf8_gate)
    circuit_path = PROJECT_ROOT / args.circuit if args.circuit else None
    circuit_sha256 = (
        hashlib.sha256(circuit_path.read_bytes()).hexdigest()
        if circuit_path is not None and circuit_path.is_file()
        else None
    )
    control = run_arm(checkpoint, None, max_bytes=args.max_bytes, limit=args.limit)
    treated = run_arm(
        checkpoint,
        args.circuit,
        evidence_utf8_gate=gate,
        evidence_alpha=args.copy_evidence_alpha,
        evidence_floor_tau=args.relevance_floor_tau,
        evidence_ceiling_c=args.relevance_ceiling_c,
        record_scores=args.record_scores,
        evidence_window_steps=args.evidence_window_steps,
        store_scope_conversation=args.store_scope_conversation,
        oracle_selector=args.oracle_selector,
        evidence_content_arm=args.evidence_content_arm,
        perm_seed=args.perm_seed,
        max_bytes=args.max_bytes,
        limit=args.limit,
    )
    verdict = (
        "A2.4 重测通过（D+E>0 且成句率不塌于对照）"
        if (
            treated["correct"] > control["correct"]
            and treated["correct"] > 0
            and treated["well_formed_rate"] >= control["well_formed_rate"]
        )
        else "A2.4 重测未通过（如实记录）"
    )
    report = {
        "format": "taiji-r2-copy-circuit-chat-cap-v1",
        "prereg": "plans/reference/M5_R2_A2_3_PREREG_20260925.md §4-S2（判据沿用）",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        #: 复现性债的修法（PLAN-A-24 rev22）：光记**路径**不够——同名路径会被后来的长跑覆盖
        #: （实测：`output/taiji_r2_copy_circuit/judge/circuit-final.pt` 现在跑出 0/16，
        #: 而登记值是 2/16）。记下 payload 指纹，旧判读件才既验得了"当时用的是哪份权重"、
        #: 也不至于把覆盖后的新权重当成旧读数。
        "circuit_sha256": circuit_sha256,
        #: PLAN-A-25：门开/关必须落在件上，否则两份读数看起来像同一次实验。
        "copy_evidence_utf8_gate": gate,
        #: DEBT-G19 剂量档：治疗臂的证据乘数（1.0 ⇒ 与已入库两臂档逐位可比）。
        "copy_evidence_alpha": args.copy_evidence_alpha,
        "relevance_floor_tau": args.relevance_floor_tau,
        "relevance_ceiling_c": args.relevance_ceiling_c,
        "evidence_content_arm": args.evidence_content_arm,
        "evidence_window_steps": args.evidence_window_steps,
        "store_scope_conversation": bool(args.store_scope_conversation),
        "oracle_selector": bool(args.oracle_selector),
        "content_arm_note": "v1 加性字段：`content_arm` 逐臂自述被走到次数／守恒偏差／冻结点。"
        "格式串不动 ⇒ 与已入库各档同格可比（默认 None ⇒ 一次替换都没发生）。",
        #: 两条口径必须落在件上，否则这份读数会被当成"整条答复、预算 256"的那类去比：
        #: ①生成预算（`PLAN-A-30` §2h 实测同一链同一装配 64→256 会让命中 3→9、成句 13→6）；
        #: ②成句率量的是 `answer[:60]` **字符前缀**，不是整条答复（与 `probe_taiji_a30_*` 的
        #:    全文口径不同 ⇒ 两个"成句"不许互换）。
        "max_bytes": args.max_bytes,
        "item_limit": args.limit,
        "well_formed_scope": "answer[:60] 字符前缀（非整条答复）",
        "control_no_circuit": control,
        "treated_with_circuit": treated,
        "dose_guard": {
            "treated_consumed_scaler": bool(treated.get("evidence_calls")),
            "control_consumed_scaler": bool(control.get("evidence_calls")),
            "evidence_calls_by_arm": {
                "control": control.get("evidence_calls"),
                "treated": treated.get("evidence_calls"),
            },
        },
        "verdict": verdict,
    }
    print(
        json.dumps(
            {
                "control_correct": control["correct"],
                "treated_correct": treated["correct"],
                "control_well_formed": control["well_formed_rate"],
                "treated_well_formed": treated["well_formed_rate"],
                "items": control["items"],
                "verdict": verdict,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    out = (
        Path(args.out_report)
        if args.out_report
        else (PROJECT_ROOT / "reports" / "taiji_r2_copy_circuit_chat_cap_20260925.json")
    )
    if out.exists():
        from datetime import datetime

        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

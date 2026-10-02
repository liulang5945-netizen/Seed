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
    probe_store: bool = False,
    empty_store: bool = False,
    decoy: tuple[str, bytes, str] | None = None,
    relevance_gate: bool = False,
    probe_label_only: str | None = None,
    probe_checks: dict[str, int] | None = None,
    max_bytes: int = MAX_ANSWER_BYTES,
    limit: int | None = None,
    product_window_steps: int | None = None,
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
    #: 产品档（与 L2 探针 v20/v21 同名旗标）：门开在**产品代码**里，计步基与复位都在 `Taiji.generate()` 内。
    if product_window_steps is not None:
        if evidence_window_steps is not None:
            raise SystemExit("产品档与替身档不能同开——同开就分不出读数来自哪条路径")
        if product_window_steps <= 0:
            raise SystemExit("产品档的 K 必须是正整数")
        runtime.model.substrate.set_copy_evidence_window_steps(product_window_steps)
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

    #: 第四十次停靠·下一格：**挂回路但让库恒空**（把 `store.record` 换成 no-op）。
    #: 分开两件事——{q}回路在不在{Q}与{q}内容有没有进库{Q}。prompt 通道不受这档影响（告知文本本来就在提示里）。
    empty_state = {"calls": 0, "count_at_install": None, "store_obj": None}
    decoy_state = {"calls": 0, "swaps": 0, "no_event": 0}
    if decoy is not None:
        dy_circuit = runtime.model.substrate.copy_circuit
        if dy_circuit is None:
            raise RuntimeError("要求外来内容档但回路不在场 ⇒ 没有 best_match 可换")
        from probe_taiji_a30_copy_evidence_dose import _make_decoy_content_arm

        dy_fn, decoy_state = _make_decoy_content_arm(dy_circuit.store, decoy[1])
        dy_circuit.store.best_match = dy_fn  # type: ignore[method-assign]

    if empty_store:
        es_circuit = runtime.model.substrate.copy_circuit
        if es_circuit is None:
            raise RuntimeError("要求库恒空档但回路不在场 ⇒ 没有可空着的库")

        def no_record(*args: Any, **kwargs: Any) -> int:
            empty_state["calls"] += 1
            return 0

        es_circuit.store.record = no_record  # type: ignore[method-assign]
        #: 这里取的是**装机瞬间**的库内条数；答复时的条数由 `--probe-store` 那列独立给（别混成一列）。
        empty_state["count_at_install"] = int(es_circuit.store.count)
        empty_state["store_obj"] = es_circuit.store

    #: 第四十四次停靠：**装在最后一层**的相关性门——判的是"当下将被取到的那条内容有没有资格被说出来"，
    #: 所以必须盖在 decoy／oracle 之上，否则判的不是同一个事件。
    gate_box = {"q": ""}
    gate_state = {"asks": 0, "passed": 0, "blocked": 0, "no_event": 0}
    if relevance_gate:
        gated_circuit = runtime.model.substrate.copy_circuit
        if gated_circuit is None:
            raise RuntimeError("要求相关性门但回路不在场 ⇒ 没有可判定的事件")
        from probe_taiji_a30_copy_evidence_dose import _make_relevance_gate_arm

        gate_fn, gate_state = _make_relevance_gate_arm(gated_circuit.store, lambda: gate_box["q"])
        gated_circuit.store.best_match = gate_fn  # type: ignore[method-assign]

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = [
        {**item, "dimension": dim}
        for dim in ("D", "E")
        for item in manifest["dimensions"][dim]["items"]
        if item.get("expected_contains")
    ]
    decoy_item = None
    if decoy is not None and decoy[0] != "SYNTH":
        #: 合成探针（`SYNTH`）**故意不在题面里**——那正是它成立的前提（机检已在 main 里做完并拒过counter例）。
        by_id = {str(item.get("id")): item for item in items}
        decoy_item = by_id.get(decoy[0])
        if decoy_item is None:
            raise RuntimeError(f"decoy id {decoy[0]!r} 不在题面里 ⇒ 外来内容无从取")
    if limit is not None:
        items = items[:limit]
    ngram = build_ngram_model()
    rows = []
    for item in items:
        history: list[tuple[str, str]] = []
        #: 第四十次停靠·**纯读数**（不改任何行为）：跟踪"内容含本题标签的事件"何时进库、是否被 FIFO 挤掉。
        probe_store_obj = None
        if probe_store:
            probed = runtime.model.substrate
            if probed.copy_circuit is None:
                raise RuntimeError("要求库内快照读数但回路不在场 ⇒ 没有 store 可看")
            probe_store_obj = probed.copy_circuit.store
            probe_label_bytes = [
                str(token).encode("utf-8") for token in (item.get("expected_contains") or [])
            ]
            if not probe_label_bytes:
                raise RuntimeError(f"{item['id']} 没有 expected_contains ⇒ 库内快照无标签可找")
            probe_seen: set[int] = set()
            probe_present_at_answer = False
            probe_capacity = int(probe_store_obj.max_events)
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
            gate_box["q"] = str(turn)  # 本轮提问原文：门的判据来源（词法重叠，不是语义）
            loop_steps[0] = 0  # 换一轮答复：步刻度从 0 重数（与 L2 仪器同一条规矩）
            if probe_store_obj is not None:
                probe_live = {
                    int(event.event_id)
                    for event in probe_store_obj.events()
                    if any(token in bytes(event.content) for token in probe_label_bytes)
                }
                probe_seen |= probe_live
                probe_present_at_answer = bool(probe_live)
            answer = _answer_raw(runtime, turn, history, max_bytes=max_bytes)
            if index + 1 < len(turns):
                history.append((turn, answer))
        if probe_store_obj is not None:
            probe_still = {
                int(event.event_id)
                for event in probe_store_obj.events()
                if any(token in bytes(event.content) for token in probe_label_bytes)
            }
            probe_evicted = len(probe_seen - probe_still)
            probe_live_now = len(probe_still)
        own_labels = [str(tok) for tok in (item.get("expected_contains") or [])]
        decoy_label_in_answer = (
            decoy is not None
            and str(decoy[2]) in answer
            #: 排除集扩全：来源题 + **本题自身标签里含探针标签的**（上一格就栽在只排来源题）
            and str(item["id"]) != decoy[0]
            and not any(str(decoy[2]) in own for own in own_labels)
            and probe_label_only is None
        ) or (
            probe_label_only is not None
            and str(probe_label_only) in answer
            and not any(str(probe_label_only) in own for own in own_labels)
        )
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
                #: 第四十三次停靠：外来标签出现在本题答复里 ⇒ 通道把本轮没被告知的内容写了进去。
                "decoy_label_in_answer": (
                    decoy_label_in_answer
                    if (decoy is not None or probe_label_only is not None)
                    else None
                ),
                #: 第四十次停靠的三量（`None` ⇒ 本题没开读数，不是"假"）。
                "label_in_store_at_answer": (
                    probe_present_at_answer if probe_store_obj is not None else None
                ),
                "correct_events_seen": len(probe_seen) if probe_store_obj is not None else None,
                "correct_events_evicted": (probe_evicted if probe_store_obj is not None else None),
                "store_events_at_answer": (probe_live_now if probe_store_obj is not None else None),
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
        #: 库内快照的聚合：容量、"答复时库里有含标签事件"的题数、被 FIFO 挤掉的正确事件总数。
        "store_probe": (
            {
                "capacity": probe_capacity,
                "items_with_label_in_store": sum(
                    1 for row in rows if row["label_in_store_at_answer"]
                ),
                "items_total": len(rows),
                "correct_events_seen": sum(int(row["correct_events_seen"] or 0) for row in rows),
                "correct_events_evicted": sum(
                    int(row["correct_events_evicted"] or 0) for row in rows
                ),
            }
            if probe_store
            else None
        ),
        #: 库恒空档自述：`record` 被叫了几次（应为 >0）、结束时库里有几条（应为 0）。两数都能为假。
        #: 相关性门自述：判定次数、放行/拦截、库里本来没事件的次数（三者都能为假）。
        "label_probe": (
            {
                "probe_label": probe_label_only,
                "items_with_probe_label_in_answer": sum(
                    1 for row in rows if row["decoy_label_in_answer"]
                ),
                "items_total": len(rows),
                "manifest_occurrences_of_label": (probe_checks or {}).get(
                    "label_in_manifest_texts"
                ),
                "manifest_occurrences_of_sentence": (probe_checks or {}).get(
                    "sentence_in_manifest_texts"
                ),
                "items_owning_probe_label": (probe_checks or {}).get("items_owning_label"),
            }
            if probe_label_only is not None
            else None
        ),
        "relevance_gate": {"requested": bool(relevance_gate), **gate_state},
        "decoy_arm": (
            {
                "source_id": decoy[0],
                "source_turn": decoy_item["turns"][0] if decoy_item else None,
                "probe_label": decoy[2],
                "calls": decoy_state["calls"],
                "swaps": decoy_state["swaps"],
                "no_event": decoy_state["no_event"],
                "decoy_label_seen_in_other_answers": sum(
                    1 for row in rows if row["decoy_label_in_answer"]
                ),
            }
            if decoy is not None
            else None
        ),
        "empty_store_arm": (
            {
                "requested": True,
                "record_calls": empty_state["calls"],
                "store_count_at_install": empty_state["count_at_install"],
                "store_count_at_end": (
                    int(empty_state["store_obj"].count)
                    if empty_state["store_obj"] is not None
                    else None
                ),
            }
            if empty_store
            else None
        ),
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
    parser.add_argument(
        "--probe-store",
        action="store_true",
        help="第四十次停靠·纯读数：每题每轮答复前拍一次库内快照，跟踪『内容含本题标签的事件』"
        "何时进库、是否被 FIFO 挤掉。不改任何行为；D 命中必须仍与不开读数那趟相同（锚点检验）。",
    )
    parser.add_argument(
        "--empty-store",
        action="store_true",
        help="第四十次停靠·下一格：挂回路但把 `store.record` 换成 no-op，让库恒空——"
        "用来把『回路在不在』与『内容有没有进库』分开。prompt 通道不受影响。",
    )
    parser.add_argument(
        "--decoy-item",
        default=None,
        help="第四十三次停靠·外来内容档：指定题面里某个 ID，把它的**第一条用户轮原文**当作被取到的事件内容"
        "（cue 沿用真实事件 ⇒ 寻址面不动），再看它的 `expected_contains[0]` 是否出现在别的题的答复里。",
    )
    parser.add_argument(
        "--relevance-gate",
        action="store_true",
        help="第四十四次停靠：在 best_match 外面再加一层最小相关性门——被取到的事件内容"
        "与本轮提问共享 ≥2 字连续片段才放行，否则返回 None（evidence 遇 None 即精确零向量）。",
    )
    parser.add_argument(
        "--synthetic-decoy",
        default=None,
        help="第四十六次停靠：用**任何题面都不出现的合成句**当被取到的事件内容（配合 --synthetic-label）。"
        "机检不通过（该句或该标签在题面里出现过）就直接拒绝跑档。",
    )
    parser.add_argument(
        "--synthetic-label",
        default=None,
        help="合成 decoy 的探针标签（用于在答复里查外来内容是否被说出来）",
    )
    parser.add_argument(
        "--probe-label",
        default=None,
        help="只测不装：统计该标签在答复里出现的题数，作为污染测试的**基线件**（没有基线，decoy 件里的出现不算证据）。",
    )
    parser.add_argument(
        "--product-window-steps",
        type=int,
        default=None,
        help="产品档：调产品侧原生生命周期门（`Taiji.set_copy_evidence_window_steps`），不是本仪器的替身包装器。"
        "默认 None ⇒ 逐位不变；与 --evidence-window-steps 互斥。",
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
    #: DEBT-G21：与上面 `circuit_sha256` 同一个理由——同名路径会被后来的长跑覆盖，
    #: 所以底座也要按**内容**钉住，跨工件配对不能只比 `checkpoint` 这一列路径。
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]
    manifest_raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    probe_checks = {
        "label_in_manifest_texts": 0,
        "sentence_in_manifest_texts": 0,
        "items_owning_label": 0,
    }
    pool_all = [
        it
        for dim in ("D", "E")
        for it in manifest_raw["dimensions"][dim]["items"]
        if it.get("expected_contains")
    ]
    probe = args.probe_label or args.synthetic_label
    if probe:
        all_text = "".join("".join(str(turn) for turn in it.get("turns", [])) for it in pool_all)
        probe_checks["label_in_manifest_texts"] = all_text.count(str(probe))
        probe_checks["items_owning_label"] = sum(
            1 for it in pool_all if any(str(probe) in str(tok) for tok in it["expected_contains"])
        )
        if args.synthetic_decoy:
            probe_checks["sentence_in_manifest_texts"] = all_text.count(str(args.synthetic_decoy))
    decoy = None
    if args.synthetic_decoy:
        if not args.synthetic_label:
            raise SystemExit("--synthetic-decoy 需要同时给 --synthetic-label ⇒ 否则无从检测")
        if probe_checks["label_in_manifest_texts"] or probe_checks["items_owning_label"]:
            raise SystemExit(
                f"机检失败：探针标签在题面里出现 {probe_checks['label_in_manifest_texts']} 次、"
                f"被 {probe_checks['items_owning_label']} 题当作自身标签 ⇒ 这一对读数量不出污染，换探针"
            )
        if probe_checks["sentence_in_manifest_texts"]:
            raise SystemExit("机检失败：合成句本身在题面里出现过 ⇒ 不是外来内容")
        decoy = (
            "SYNTH",
            str(args.synthetic_decoy).encode("utf-8"),
            str(args.synthetic_label),
        )
        print("[decoy-syn] 机检通过：探针标签与合成句在全部题面里 0 次、无题以该标签为自身标签")
    if args.decoy_item:
        pool = [
            it
            for dim in ("D", "E")
            for it in manifest_raw["dimensions"][dim]["items"]
            if it.get("expected_contains")
        ]
        src = next((it for it in pool if str(it.get("id")) == args.decoy_item), None)
        if src is None:
            raise SystemExit(f"--decoy-item {args.decoy_item} 不在 manifest 里 ⇒ 响亮停")
        decoy = (
            args.decoy_item,
            str(src["turns"][0]).encode("utf-8"),
            str(src["expected_contains"][0]),
        )
        print(
            f"[decoy] 外来内容取自 {args.decoy_item}：{str(src['turns'][0])[:24]!r} 探针标签={src['expected_contains'][0]!r}"
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
        probe_store=args.probe_store,
        empty_store=args.empty_store,
        decoy=decoy,
        probe_label_only=args.probe_label,
        probe_checks=probe_checks,
        relevance_gate=args.relevance_gate,
        evidence_content_arm=args.evidence_content_arm,
        perm_seed=args.perm_seed,
        max_bytes=args.max_bytes,
        limit=args.limit,
        product_window_steps=args.product_window_steps,
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
        "checkpoint_sha256": checkpoint_sha256,
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
        "product_window_steps": args.product_window_steps,
        "store_scope_conversation": bool(args.store_scope_conversation),
        "oracle_selector": bool(args.oracle_selector),
        "probe_store": bool(args.probe_store),
        "decoy_item": args.decoy_item,
        "relevance_gate": bool(args.relevance_gate),
        "synthetic_decoy": args.synthetic_decoy,
        "synthetic_label": args.synthetic_label,
        "probe_label": args.probe_label,
        "empty_store": bool(args.empty_store),
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

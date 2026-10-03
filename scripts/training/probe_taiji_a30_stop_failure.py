"""PLAN-A-30 §2f 第 3 条的定位：为什么仍有一条答复是 256 字同字拖写（停不下来）。

`repetition_penalty=2.0` 在表层链上把成句从 12/72 抬到 71/72，但**最坏那条没消**：
seed-A 两段题面的 `max_longest_run` 都是 256（一条答复整段是同一个字）。
本探针要回答的是二选一，两者修法完全不同：

  (A) **掩码饿死**：那一步的合法后继集只剩 1 个候选 ⇒ 病在"合法集 × 证据集中"的交集，解码侧再动一手；
  (B) **模型不肯停**：合法集里候选不少、边界符概率也没被压成零，但每一步它就是比续字节低
      ⇒ 病在**停止信号本身**（没学会"这句说完了"），属训练目标/规模那一层，不是解码参数能治的。

做法（**不改产品源码**，实例级包装；**不重抄生成链**——跟的是产品 `chat()` 真实走过的步）：
把 `substrate.observe` 包一层，记下"这一步被喂进哪个符号"、"这一步返回的概率向量"，
以及"**这一步是从产品源码的哪一行被调的**"（`sys._getframe(1)` 的 `co_name` + `f_lineno`）。

段起点不靠猜。前两版的教训：
  v1 按"1 次边界符 + prompt 字节数"硬切 ⇒ 第 3 步就断（一轮里 `observe` 的调用比这多）；
  v2 改成"拿答复字节在偏移窗口里找吻合段"⇒ 对齐上了，但**自检仍全红**。
v2 全红的第一嫌疑就在仪器自己漏的一步：产品解码环是**先重复惩罚、后 UTF-8 掩码**
（`taiji/model.py` 里 `probabilities / (1.0 + repetition_penalty * counts)` 在 `masked_fill` 之前），
v2 只重放了掩码 ⇒ 拿"旧口径的 argmax"去对"新口径实际发出的字节"，每步都像错。
本版把惩罚按同一口径补进重放；**若自检仍红，则原因不在这条**，本件读数作废、另找。

现在按调用点切段：从 `Taiji.generate` 的源码行号自动推出**生成环的行号区间**
（不写死数字，源码移动它跟着移动），落在区间内的记录就是生成步，喂进符号的序列就是答复字节；
区间外、同属 `generate` 的那些就是边界符与提问的喂入。
逐步验「按产品口径重放（惩罚 → 掩码 → 取最大）后的最优符号 == 本步实际被喂的符号」；
不相等 ⇒ 该步计入 `argmax_mismatch_steps`，任一项红 ⇒ `all_items_reconstructed=False`，
本件读数一律作废（不拿来下结论），并把**前 5 条失败步**原样落进件里好一次跑看清。

用法：
    python scripts/training/probe_taiji_a30_stop_failure.py \
        --circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt --limit 24
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
TIE_TOLERANCE = 1e-9  # 与产品同为 float32，取等号按"并列即算命中"处理


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha(value) -> str | None:
    """回路 payload 的摘要：件里只记路径不够（`PLAN-A-24` rev22 那条复现性债）。"""

    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _sha256(path)


def _longest_same_char_run(text: str) -> int:
    best = run = 1
    for a, b in zip(text, text[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if text else 0


def generation_loop_span(generate: Any) -> tuple[int, int]:
    """从产品源码**自己**推出"生成环"的行号区间，避免把行号写死进仪器。"""

    import inspect

    lines, start = inspect.getsourcelines(generate)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("for ") and "range(length)" in stripped and stripped.endswith(":"):
            indent = len(line) - len(line.lstrip())
            end = index
            for following in range(index + 1, len(lines)):
                body = lines[following]
                if body.strip() and (len(body) - len(body.lstrip())) > indent:
                    end = following
            return start + index, start + end
    raise RuntimeError("generation loop (`for _ in range(length)`) not found in Taiji.generate")


def replay_step(
    probabilities: Any,
    byte: int,
    boundary: int,
    state: tuple[int, int],
    recent: list[int],
    penalty: float,
    window: int,
) -> dict[str, Any]:
    """按**产品解码环的口径**重放一步：先重复惩罚，后 UTF-8 掩码，再取最大。

    产码顺序照 `taiji/model.py::Taiji.generate` 的环体：惩罚只看已发字节的最后 `window` 个，
    掩码合法集 = `utf8_allowed(remaining, lead) | {boundary}`。
    """

    import torch

    from taiji.utf8_state import utf8_allowed

    probabilities = probabilities.detach().cpu().float()
    boundary_before_penalty = float(probabilities[boundary])
    if penalty > 0.0 and recent:
        counts = torch.zeros_like(probabilities)
        for symbol in recent[-window:]:
            counts[int(symbol)] += 1.0
        probabilities = probabilities / (1.0 + penalty * counts)
    legal = sorted(set(utf8_allowed(state[0], state[1])) | {boundary})
    vector = [float(value) for value in probabilities]
    best = max(vector[index] for index in legal)
    legal_ranked = sorted(legal, key=lambda index: -vector[index])
    return {
        "emitted_is_argmax": bool(vector[byte] >= best - TIE_TOLERANCE),
        "p_emitted": round(vector[byte], 6),
        "p_boundary": round(vector[boundary], 6),
        "p_boundary_before_penalty": round(boundary_before_penalty, 6),
        "boundary_rank_in_legal": legal_ranked.index(boundary) + 1,
        #: DEBT-G24：这一列**故意不含边界符**，而 `boundary_rank_in_legal` 是在含边界符的集合里排名
        #: ⇒ 名次可比它大 1。旧列不动（与 v6–v24 各件同格可比），补一列无歧义的分母。
        "legal_candidates": len(legal) - 1,
        "legal_candidates_including_boundary": len(legal),
        "ratio_best_over_boundary": (
            round(best / vector[boundary], 3) if vector[boundary] > 0 else None
        ),
        "boundary_is_argmax": bool(vector[boundary] >= best - TIE_TOLERANCE),
        "boundary_probability_is_zero": vector[boundary] == 0.0,
    }


def replay_surface_from_fed(
    fed: list[int],
    *,
    turn: str,
    history: list[tuple[str, str]],
    organ: Any,
    turn_markers: tuple[str, ...],
) -> dict[str, Any]:
    """从被拦截的生成字节重放**完整产品面链**：decode → marker 切割 → **同一个器官** emit。

    v5 及以前拿 raw 字节直接对比 `chat()` 返回值——当模型真早停（边界符胜出、`break` 在
    observe 之前）而 raw 短到器官拒收（decode 出 U+FFFD／不可读）时，产品会把占位句
    （`NativeReadableTextLanguageOrgan._fallback_text`）当作答复，逐位比较必然不等，
    早停这个 L2 要测的成功被记成面违规。深帧复现（§2bb-追加三）纠正了追加二的误诊：
    `V019` 首轮的 53 次 observe＝1 告知＋1 边界＋50 prompt＋**1 生成**——prompt 全部在案，
    1 字节是生成环的真实产量，而那张"完整成句"的答复头是器官模板，不是模型输出。
    ⇒ v6 把比较放到**同一表示层**：重放也走产品链的器官这一步。
    """

    from taiji import ExpressionPlan
    from taiji.utf8_state import trim_partial_tail

    native = trim_partial_tail(bytes(fed)).decode("utf-8", errors="replace")
    marker_at = min(
        (index for marker in turn_markers if (index := native.find(marker)) >= 0),
        default=None,
    )
    if marker_at is not None:
        native = native[:marker_at]
    history_payload = [
        {"user": user, "assistant": assistant} for user, assistant in history if user and assistant
    ]
    expression = ExpressionPlan(
        expression_id="chat:replay:expression",
        content_id="chat:replay:content",
        modality="text",
        channel="message",
        fields={
            "intent_kind": "chat_answer",
            "semantic_slots": {"prompt": turn, "history": history_payload},
            "native_prediction": native,
            "expected_outcome": "answer user in readable language",
        },
        provenance="seed.client.chat",
        tick=0,
    )
    emission = organ.emit(expression)
    return {
        "native_replay": native,
        "marker_at": marker_at,
        "replay_surface": emission.text_bytes.decode("utf-8", errors="strict"),
    }


def char_membership_of_run(text: str) -> list[bool]:
    """逐**字节**标出"这个字节所属的汉字处在一段同字连写里"。

    不能按字节直接比相等：一个汉字三字节，`我我` 的字节流是 `E6 88 91 E6 88 91`，
    逐字节比较会把"同一个字连着写"读成"不重复"，而把"同一字的中途续字节"读成"重复"。
    """

    chars = list(text)
    repeated = [
        bool(
            (index > 0 and chars[index] == chars[index - 1])
            or (index + 1 < len(chars) and chars[index] == chars[index + 1])
        )
        for index in range(len(chars))
    ]
    membership: list[bool] = []
    for index, char in enumerate(chars):
        membership.extend([repeated[index]] * len(char.encode("utf-8")))
    return membership



_FIXED_STEPS = (8, 16, 32, 64, 128)
_LF_BYTE = 10  # §73 读数：两底 118/118 次自停的"前一步"发的都是这个字节


def _lf_trace(group: list[dict]) -> dict[str, Any]:
    """v29（§73 末段已写死的判据）：这条生成里 LF 发过几次、最后一次在哪儿、停是不是紧跟其后。

    存在的理由：§73 只证明了"**停**总在 LF 之后"，而**拖写**是不是"根本没发 LF"还是"发了没接住"
    是两个不同的修法方向（定点 vs 训练侧），现有件里没这个量。
    """

    steps = [int(row["step"]) for row in group if int(row["emitted_byte"]) == _LF_BYTE]
    return {
        "lf_step_count": len(steps),
        "last_lf_step": max(steps) if steps else None,
        "generation_last_step": max(int(row["step"]) for row in group),
    }


def _utf8_byte_class(byte: int) -> str:
    """字节在 UTF-8 编码里的**位置类**——§72 判据要的那把尺（ASCII／首字节／续字节）。"""

    if byte < 0x80:
        return "ascii"
    if byte < 0xC0:
        return "continuation"
    if byte < 0xE0:
        return "lead2"
    if byte < 0xF0:
        return "lead3"
    if byte < 0xF8:
        return "lead4"
    return "invalid"


def _group_rows_by_generation(rows: list[dict]) -> list[list[dict]]:
    """把 item 内串接的逐步行按**每次生成**切开。

    核实依据（本文件 line ~562-582）：每次生成的行都带 `step`，且 `step` 在换答复时从 0 重数，
    而 `item_rows` 是把该 item 的 3 次生成**串接**后一起聚合的 ⇒ 现成的 `median_*` 列跨代混算，
    这就是它对"基底为什么不停"没有判别力的原因（§第五十九次停靠）。`step == 0` 即一代的开始。
    """

    #: v23 更正：v22 在这里用 `in_run` 过滤，而该字段是"该步是否落在**同字重复段**内"
    #: （由 `char_membership_of_run(answer)` 得出，见本文件 line 561/579），不是"是否在生成环内"。
    #: 后果实测过一次：72 次生成只剩 12／0 条，`ate_full_budget` 恒假（§第六十次停靠·资格前置判定）。
    #: `in_run` 仍保留，但作为第二维 `steps_in_repeat_run` 报告。
    groups: list[list[dict]] = []
    for row in rows:
        if not groups or row.get("step") == 0:
            groups.append([])
        groups[-1].append(row)
    return groups


def _endstep_probe_per_generation(
    rows: list[dict], max_length: int, terminals: list[dict | None]
) -> list[dict]:
    """四个无外部真值依赖的标量，按每次生成一条（§59 更正版；不存逐步大数组）。

    v27（DEBT-G26）：产品环体在 `argmax()==boundary` 时**先 break 再 observe**，所以停下那一步
    在 `rows` 里根本没有帧。`terminals` 是调用方在环外按同一口径补出的**终止决策行**，
    与生成分组一一对齐；对不上就抛错，不静默错配。
    """

    groups = _group_rows_by_generation(rows)
    if len(groups) != len(terminals):
        raise RuntimeError(
            f"终止决策行与生成分组数量不一致：groups={len(groups)} terminals={len(terminals)}"
        )
    out = []
    for group, terminal in zip(groups, terminals):
        if not group:
            continue
        peak = max(group, key=lambda row: row["p_boundary"])
        last_step = max(row["step"] for row in group)
        by_step = {int(row["step"]): row for row in group}
        last_row = by_step[last_step]
        if "emitted_byte" not in last_row:
            raise RuntimeError(
                "v28 要求每行自带 `emitted_byte`／`utf8_byte_class`（§72 的末步字面身份要用）"
                "⇒ 缺键说明仪器没走到写该键的那一步，读数不发表"
            )
        out.append(
            {
                "generation_steps": len(group),
                "steps_in_repeat_run": sum(1 for row in group if row.get("in_run")),
                "last_step": int(last_step),
                #: v28（§72 判读先于数）：**停之前那一步发的是哪个字节、当时在字的哪一段**。
                #: v29（§73 末段冻线）：这条生成里 LF 发过几次、最后一次在哪儿。
                "lf_trace_v29": _lf_trace(group),
                "tail_identity": {
                    "prev_step_emitted_byte": int(last_row["emitted_byte"]),
                    "prev_step_utf8_byte_class": str(last_row["utf8_byte_class"]),
                    "prev_step_utf8_state_before": [
                        int(last_row["utf8_state_before"][0]),
                        int(last_row["utf8_state_before"][1]),
                    ],
                    "prev_step_boundary_rank": int(last_row["boundary_rank_in_legal"]),
                    "prev_step_p_boundary": round(float(last_row["p_boundary"]), 6),
                },
                "p_boundary_max": round(float(peak["p_boundary"]), 6),
                "p_boundary_argmax_step": int(peak["step"]),
                "boundary_rank_at_peak_step": int(peak["boundary_rank_in_legal"]),
                "legal_candidates_at_peak_step": int(peak["legal_candidates"]),
                #: v27：`last_recorded_row` 是"最后一个**在案**步"（对自停的代＝停下前一步）。
                "last_recorded_row": {
                    "step": int(last_row["step"]),
                    "p_boundary": round(float(last_row["p_boundary"]), 6),
                    "boundary_rank_in_legal": int(last_row["boundary_rank_in_legal"]),
                    "legal_candidates_including_boundary": int(
                        last_row["legal_candidates_including_boundary"]
                    ),
                    "ratio_best_over_boundary": last_row["ratio_best_over_boundary"],
                },
                #: v27（DEBT-G26）：停下那一步的决策——只在该代是**边界自停**时才有。
                "terminal_decision": (
                    None
                    if terminal is None
                    else {
                        "step": int(terminal["step"]),
                        "p_boundary": round(float(terminal["p_boundary"]), 6),
                        "p_boundary_before_penalty": round(
                            float(terminal["p_boundary_before_penalty"]), 6
                        ),
                        "boundary_rank_in_legal": int(terminal["boundary_rank_in_legal"]),
                        "legal_candidates_including_boundary": int(
                            terminal["legal_candidates_including_boundary"]
                        ),
                        "ratio_best_over_boundary": terminal["ratio_best_over_boundary"],
                        "boundary_is_argmax": bool(terminal["boundary_is_argmax"]),
                        "terminal_utf8_state_before": [
                            int(terminal["utf8_state_before"][0]),
                            int(terminal["utf8_state_before"][1]),
                        ],
                        "terminal_over_recorded_peak": (
                            round(float(terminal["p_boundary"]) / float(peak["p_boundary"]), 4)
                            if peak["p_boundary"] > 0
                            else None
                        ),
                    }
                ),
                "terminal_decision_absent_reason": (
                    None
                    if terminal is not None
                    else ("ate_full_budget" if len(group) >= max_length else "no_loop_records")
                ),
                "peak_is_last_step": bool(peak["step"] == last_step),
                "ate_full_budget": bool(len(group) >= max_length),
                #: §第六十三次停靠·固定步位：长度只决定"能否走到那一步"，避开 §62 的存活偏置。
                "at_steps": [
                    {
                        "step": step_at,
                        "p_boundary": round(float(by_step[step_at]["p_boundary"]), 6),
                        "boundary_rank_in_legal": int(by_step[step_at]["boundary_rank_in_legal"]),
                        "legal_candidates": int(by_step[step_at]["legal_candidates"]),
                        "legal_candidates_including_boundary": int(
                            by_step[step_at]["legal_candidates_including_boundary"]
                        ),
                    }
                    for step_at in _FIXED_STEPS
                    if step_at in by_step
                ],
            }
        )
    return out


def _rank_buckets(rank: int | None) -> str:
    """把边界名次归到固定刻度（1／2／3／4-10／>10）——刻度先写死，避免事后挑分堆。"""

    if rank is None:
        return "none"
    if rank <= 3:
        return str(rank)
    if rank <= 10:
        return "4-10"
    return ">10"


def _terminal_summary_v27(per_item: list[dict]) -> dict[str, Any]:
    """件级「终止决策」汇总（DEBT-G26／§第七十次停靠）。

    存在理由：产品环体在边界胜出那一步 `break` 在 `observe` **之前**，所以那一格过去在件里
    根本不存在（§第六十六次停靠的 `peak_is_last_step` 因此是个不可能为假的量，已收回）。
    这里把补出的 `terminal_decision` 摊成可直接判读「新高型 vs 竞争型」的几列，
    并自带配对自证（`pairing_ok`）——配对不成立就是仪器分岔，读数不发表。
    """

    generations = [g for row in per_item for g in row["endstep_probe_v22"]]
    with_terminal = [g for g in generations if g["terminal_decision"] is not None]
    without_terminal = [g for g in generations if g["terminal_decision"] is None]
    ranks = [g["terminal_decision"]["boundary_rank_in_legal"] for g in with_terminal]
    above = sum(
        1
        for g in with_terminal
        if float(g["terminal_decision"]["p_boundary"]) > float(g["p_boundary_max"])
    )
    ratios = [
        g["terminal_decision"]["terminal_over_recorded_peak"]
        for g in with_terminal
        if g["terminal_decision"]["terminal_over_recorded_peak"] is not None
    ]
    stoppers = with_terminal
    eaters = [g for g in generations if g["ate_full_budget"]]

    def _median(values: list[Any]) -> Any:
        return sorted(values)[len(values) // 2] if values else None

    expected_stop = sum(row["generations_boundary_self_stop"] for row in per_item)
    expected_eat = sum(row["generations_eating_full_budget"] for row in per_item)
    return {
        "generations_total": len(generations),
        "with_terminal_row": len(with_terminal),
        "without_terminal_row": len(without_terminal),
        "absent_reasons": {
            reason: sum(1 for g in without_terminal if g["terminal_decision_absent_reason"] == reason)
            for reason in sorted({g["terminal_decision_absent_reason"] for g in without_terminal})
        },
        #: 资格前置 3：边界在终止那一步必须是掩码后第 1 名，否则重放与产品环分岔（面违规）。
        "terminal_rank_not_one_count": sum(1 for rank in ranks if rank != 1),
        "terminal_rank_values_seen": sorted(set(ranks)),
        "terminal_boundary_is_argmax_false_count": sum(
            1 for g in with_terminal if not g["terminal_decision"]["boundary_is_argmax"]
        ),
        "median_terminal_p_boundary": _median(
            [g["terminal_decision"]["p_boundary"] for g in with_terminal]
        ),
        "terminal_above_recorded_peak_count": above,
        "terminal_above_recorded_peak_share": (
            round(above / len(with_terminal), 4) if with_terminal else None
        ),
        "median_terminal_over_recorded_peak": _median(ratios),
        #: 第二问：末在案步的名次分布，自停组 vs 吃满组分开。
        "last_recorded_rank_buckets_for_stoppers": {
            bucket: sum(1 for g in stoppers if _rank_buckets(g["last_recorded_row"]["boundary_rank_in_legal"]) == bucket)
            for bucket in sorted({_rank_buckets(g["last_recorded_row"]["boundary_rank_in_legal"]) for g in stoppers})
        },
        "last_recorded_rank_buckets_for_eaters": {
            bucket: sum(1 for g in eaters if _rank_buckets(g["last_recorded_row"]["boundary_rank_in_legal"]) == bucket)
            for bucket in sorted({_rank_buckets(g["last_recorded_row"]["boundary_rank_in_legal"]) for g in eaters})
        },
        "last_recorded_median_rank_for_stoppers": _median(
            [g["last_recorded_row"]["boundary_rank_in_legal"] for g in stoppers]
        ),
        "last_recorded_median_rank_for_eaters": _median(
            [g["last_recorded_row"]["boundary_rank_in_legal"] for g in eaters]
        ),
        "expected_self_stop_from_item_counts": expected_stop,
        "expected_eat_from_item_counts": expected_eat,
        "pairing_ok": bool(
            len(with_terminal) == expected_stop and len(without_terminal) == expected_eat
        ),
    }


def _tail_identity_summary_v28(per_item: list[dict]) -> dict[str, Any]:
    """§72 判据要的四张分布表：自停组／吃满组各自的"末步字节"与"末步字节位置类"分布。

    分母各自算（两底、两组都不同数）；`cls_share` 指**该组内最高频位置类的占比**。
    """

    generations = [g for row in per_item for g in row["endstep_probe_v22"]]
    stoppers = [g for g in generations if g["terminal_decision"] is not None]
    eaters = [g for g in generations if g["ate_full_budget"]]
    #: 两组的定义式在真件里互斥（有终止行 ⇒ 环没吃满预算），但**这要靠读数成立，不靠约定**：
    #: 合成数据或未来的定义改动都会让两堆重叠／漏人，所以把两种异常都数出来。
    overlap = [g for g in generations if g["terminal_decision"] is not None and g["ate_full_budget"]]
    neither = [g for g in generations if g["terminal_decision"] is None and not g["ate_full_budget"]]

    def _dist(items: list[dict], picker) -> dict[str, Any]:
        total = len(items)
        if not total:
            return {"n": 0, "top": [], "distinct": 0, "top1_share": None}
        counts: dict[Any, int] = {}
        for item in items:
            key = picker(item)
            counts[key] = counts.get(key, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return {
            "n": total,
            "top": [
                {"key": key, "count": count, "share": round(count / total, 4)}
                for key, count in ranked[:5]
            ],
            "distinct": len(counts),
            "top1_share": round(ranked[0][1] / total, 4),
        }

    byte_of = lambda g: g["tail_identity"]["prev_step_emitted_byte"]  # noqa: E731
    class_of = lambda g: g["tail_identity"]["prev_step_utf8_byte_class"]  # noqa: E731
    return {
        "stoppers_prev_step_bytes": _dist(stoppers, byte_of),
        "eaters_prev_step_bytes": _dist(eaters, byte_of),
        "stoppers_prev_step_byte_classes": _dist(stoppers, class_of),
        "eaters_prev_step_byte_classes": _dist(eaters, class_of),
        "stoppers_top1_class_share": _dist(stoppers, class_of)["top1_share"],
        "eaters_top1_class_share": _dist(eaters, class_of)["top1_share"],
        "class_share_stoppers_minus_eaters": (
            round(
                _dist(stoppers, class_of)["top1_share"] - _dist(eaters, class_of)["top1_share"],
                4,
            )
            if stoppers and eaters
            else None
        ),
        "tail_identity_present_for_all_generations": all(
            g.get("tail_identity") for g in generations
        ),
        #: 分组健康度两条（真件里都必须为 0）
        "stoppers_that_also_ate_full_budget": len(overlap),
        "generations_neither_stop_nor_eater": len(neither),
    }


def _lf_followthrough_summary_v29(per_item: list[dict]) -> dict[str, Any]:
    """§73 末段冻线的那把尺：**拖写的生成里有多少发过 LF**（发了没接住 vs 根本没发）。"""

    generations = [g for row in per_item for g in row["endstep_probe_v22"]]
    stoppers = [g for g in generations if g["terminal_decision"] is not None]
    eaters = [g for g in generations if g["ate_full_budget"]]

    def _pack(items: list[dict]) -> dict[str, Any]:
        total = len(items)
        with_lf = [g for g in items if g["lf_trace_v29"]["lf_step_count"] >= 1]
        return {
            "n": total,
            "with_lf": len(with_lf),
            "with_lf_share": round(len(with_lf) / total, 4) if total else None,
            "lf_count_median": (
                sorted(g["lf_trace_v29"]["lf_step_count"] for g in items)[total // 2] if total else None
            ),
        }

    stopped_after_lf = [
        g for g in stoppers if g["lf_trace_v29"]["last_lf_step"] == g["lf_trace_v29"]["generation_last_step"]
    ]
    return {
        "stoppers": _pack(stoppers),
        "eaters": _pack(eaters),
        "stoppers_last_lf_is_last_step": len(stopped_after_lf),
        "stoppers_last_lf_is_last_step_share": (
            round(len(stopped_after_lf) / len(stoppers), 4) if stoppers else None
        ),
        #: 拖写组里"最后一次 LF 落在预算的哪一段"——分母小的那底要如实看到自己的分母。
        "eaters_last_lf_position_buckets": {
            bucket: sum(
                1
                for g in eaters
                if g["lf_trace_v29"]["last_lf_step"] is not None
                and _lf_bucket(g["lf_trace_v29"]["last_lf_step"], g["lf_trace_v29"]["generation_last_step"])
                == bucket
            )
            for bucket in sorted(
                {
                    _lf_bucket(g["lf_trace_v29"]["last_lf_step"], g["lf_trace_v29"]["generation_last_step"])
                    for g in eaters
                    if g["lf_trace_v29"]["last_lf_step"] is not None
                }
            )
        },
    }


def _lf_bucket(last_lf: int, last_step: int) -> str:
    """最后一次 LF 相对该代末尾的位置（四分堆，刻度先写死）。"""

    if last_step <= 0:
        return "degenerate"
    ratio = (last_step - last_lf) / last_step
    if ratio <= 0.02:
        return "at-the-end(<=2%)"
    if ratio <= 0.25:
        return "near-end(<=25%)"
    return "far-from-end(>25%)"


def _position_histogram(positions: list[int]) -> dict[str, int]:
    """把"边界符胜出所在的步序"分堆——给资格档定价用：**停止决定落在答复的哪一段**。

    堆界取 2 的幂（16/32/64/128）是刻意的：窗口 K 只能落在这些刻度上，才不会出现"事后挑一个刚好过线的 K"。
    """

    buckets = {"0-15": 0, "16-31": 0, "32-63": 0, "64-127": 0, "128+": 0}
    for position in positions:
        if position < 16:
            buckets["0-15"] += 1
        elif position < 32:
            buckets["16-31"] += 1
        elif position < 64:
            buckets["32-63"] += 1
        elif position < 128:
            buckets["64-127"] += 1
        else:
            buckets["128+"] += 1
    buckets["total"] = len(positions)
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--circuit",
        default=None,
        help="不给＝不挂回路，即**今日出厂那面**（丁-2 的判据要用它，不能只在挂回路面上说事）",
    )
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument(
        "--ids",
        default=None,
        help="只跑题面里这些 id（逗号分隔）——长预算档用它点住最坏那几条，不必整批重跑",
    )
    parser.add_argument(
        "--relevance-ceiling-c",
        type=float,
        default=None,
        help="v11：相似度高于 c 的步把回路证据归零（自信度衰减档）；与接缝／复述两档共用同一个包装器"
        "与同一条 `_cosine`，所以三半里的 c 是同一个定义",
    )
    parser.add_argument(
        "--copy-evidence-alpha",
        type=float,
        default=1.0,
        help="v9：把回路加性证据整体乘 α（DEBT-G19 剂量档；1.0 ⇒ 与 v4–v8 逐位可比）",
    )
    parser.add_argument(
        "--no-copy-evidence-gate",
        action="store_true",
        help="v8：挂上回路之后把**那条加性证据的 UTF-8 位置门**显式关掉——用来拆'挂载回路'这一个动作里"
        "捆绑着的两样东西（回路的 prompt／证据通道 vs 那条门）。它与**回写门槛①**无关：后者按'工件与"
        "检查点同目录'解析，而本仪器走 `learn=False`，那张面上根本不跑。",
    )
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--penalty-window", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out-report", default=None)
    parser.add_argument(
        "--evidence-content-arm",
        choices=("permutation", "frozen"),
        default=None,
        help="v13 **内容／硬度的分离档**（owner 2026-10-02 裁：先追'轨迹面由什么在管'，不立项产品改动）："
        "`permutation`＝把证据向量的 257 维按固定种子**置换**（多重集不变 ⇒ 硬度逐位相同、内容身份毁掉）；"
        "`frozen`＝每次调用都返回**第一次**那一条（硬度同分布、内容不再跟着 cue 走）。默认关 ⇒ 与 v12 逐位可比。",
    )
    parser.add_argument(
        "--perm-seed",
        type=int,
        default=20261002,
        help="置换档的固定种子（写进件里，可复现）",
    )
    parser.add_argument(
        "--oracle-selector",
        action="store_true",
        help="v19 检索侧 oracle 档：把 `store.best_match` 换成'内容含本题 expected_contains 的第一条事件'，"
        "找不到则透传原实现。用来把'选对了还拖不拖写'从'内容身份/发射时刻'里单独摘出来。"
        "X 面 104 题全部自带标签（机检 100%），故这档在停止面可构造；默认关 ⇒ 与 v18 逐位可比。",
    )
    parser.add_argument(
        "--store-scope-conversation",
        action="store_true",
        help="v18 资格档（检索侧）：每题开头把**装载信封里带来的陈旧事件**请出候选集，只留本次对话被告知的内容可被取到。动的是候选集，不是发射时刻——与窗口档是两条独立的形状。",
    )
    parser.add_argument(
        "--evidence-window-steps",
        type=int,
        default=None,
        help="v17 资格档：只在答复的前 K 步发复制回路证据，之后把这条通道静音。"
        "K 由 §第三十次停靠·定价档按先写死的规则取（最小的 2 的幂、覆盖 ≥90%% 命中偏移 ⇒ K=64），"
        "不许事后挑。刻度＝本条链的环内步数（1 步＝1 字节），每轮换答复即清零。",
    )
    parser.add_argument(
        "--product-window-steps",
        type=int,
        default=None,
        help="v20 产品档：不开仪器替身，直接调产品侧的原生生命周期门 "
        "`Taiji.set_copy_evidence_window_steps(K)`。owner 2026-10-02 裁「立项进产品」后的验收面——"
        "这一档测的是产品代码里的门，不是本仪器 monkeypatch 出来的等价物。与 --evidence-window-steps 互斥。",
    )
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    all_items = json.loads(manifest.read_text(encoding="utf-8"))["dimensions"]["X"]["items"]
    if args.ids:
        wanted = [part.strip() for part in args.ids.split(",") if part.strip()]
        known = {item["id"] for item in all_items}
        missing = [name for name in wanted if name not in known]
        if missing:
            parser.error(f"ids not in manifest: {missing}")
        items = [item for item in all_items if item["id"] in wanted]
    else:
        items = all_items[: args.limit]

    from api.seed_runtime import _TURN_MARKERS, SeedRuntime
    from taiji.utf8_state import advance_utf8, trim_partial_tail

    runtime = SeedRuntime.load(checkpoint)
    if args.circuit:
        runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
    substrate = runtime.model.substrate
    #: 三位一体来自剂量探针那一个包装器：[被走到, 被下限静音, 被上限静音]（全链，含 prompt 侧）。
    alpha_calls = [0, 0, 0]
    #: 资格档的刻度：本条生成链已经走过的**环内步数**（1 步＝喂进 1 字节），每轮清零。
    loop_steps = [0]
    window_counters = [0, 0, 0]
    #: v12：只在**生成环内**归因的那一对计数；没装过滤器时保持 [0, 0]。
    loop_silenced = [0, 0]
    if args.copy_evidence_alpha != 1.0 or args.relevance_ceiling_c is not None:
        # v9／v11：接口级包装，复用剂量探针里那一个包装器 ⇒ 三半（接缝／复述／自停）量的是同一个乘数、
        # 同一条 `_cosine`、同一个 c 定义，不是三口各自造的尺子。
        from probe_taiji_a30_copy_evidence_dose import _make_scaled_evidence

        if substrate.copy_circuit is None:
            raise RuntimeError("要求缩放或按相似度截断证据，但回路不在场 ⇒ 没有可包装的通道")
        scaled, alpha_calls = _make_scaled_evidence(
            substrate.copy_circuit.evidence,
            args.copy_evidence_alpha,
            circuit=substrate.copy_circuit,
            ceiling_c=args.relevance_ceiling_c,
        )
        substrate.copy_circuit.evidence = scaled
    if args.no_copy_evidence_gate:
        # v8：显式置 False ⇒ `copy_evidence_utf8_gate_effective` 应随之为 False（守卫会抓"没走到"那种）。
        substrate.set_copy_evidence_utf8_gate(False)
    boundary = int(substrate.config.boundary_symbol)
    loop_first, loop_last = generation_loop_span(type(substrate).generate)

    records: list[dict[str, Any]] = []
    original_observe = substrate.observe

    def observing(symbol: Any, **kwargs: Any) -> Any:
        frame = sys._getframe(1)
        step = original_observe(symbol, **kwargs)
        in_loop = frame.f_code.co_name == "generate" and loop_first <= frame.f_lineno <= loop_last
        records.append(
            {
                "symbol": int(symbol),
                "probabilities": step.probabilities.detach().cpu().clone(),
                "caller": f"{frame.f_code.co_name}:{frame.f_lineno}",
                "in_generation_loop": in_loop,
            }
        )
        #: 资格档的刻度：只在**生成环内**数步（1 步＝喂进 1 字节），prompt 侧与回写侧不计。
        if in_loop:
            loop_steps[0] += 1
        return step

    substrate.observe = observing  # type: ignore[method-assign]

    #: v12：**"包装器被走到"（`alpha_calls[0]`）与"过滤器在 L2 这条链上开过枪"是两件事**——
    #: 全链计数把 prompt 侧（`record_told_history`）的调用一起算了，而那一边不改写自身轨迹那一列。
    #: 归因法＝取这次调用**之前**最后一条在案帧的 `in_generation_loop`（`records` 每轮清空，
    #: 所以"上一帧在环内"＝正走在生成环的两步之间）；每轮第一步的前一帧是 prompt ⇒ 不算，属**少计**。
    def _observe_silencing(inner: Any) -> tuple[Any, list[int]]:
        counters = [0, 0]  # [环内的 evidence 调用数, 其中被上限静音的数]

        def wrapped(**kwargs: Any) -> Any:
            prev = records[-1] if records else None
            silenced_before = alpha_calls[2]
            out = inner(**kwargs)
            if prev is not None and prev["in_generation_loop"]:
                counters[0] += 1
                if alpha_calls[2] > silenced_before:
                    counters[1] += 1
            return out

        return wrapped, counters

    if args.copy_evidence_alpha != 1.0 or args.relevance_ceiling_c is not None:
        scaled, loop_silenced = _observe_silencing(substrate.copy_circuit.evidence)
        substrate.copy_circuit.evidence = scaled  # type: ignore[method-assign]

    #: v13（owner 2026-10-02 裁："不立项产品改动，先追轨迹面由什么在管"）：**内容档 vs 硬度档**。
    #: v15 起这副档**搬到剂量探针里共用**（`_make_content_armed_evidence`）——自身轨迹面与复述命中面
    #: 必须做同一个操作，否则两半读的不是同一件事。装在**最外层**：替换的是最终进 logits 的那个向量。
    content_guard = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if args.evidence_content_arm is not None:
        if substrate.copy_circuit is None:
            raise RuntimeError("要求内容分离档但回路不在场 ⇒ 没有可替换的证据通道")
        from probe_taiji_a30_copy_evidence_dose import _make_content_armed_evidence

        armed, content_guard = _make_content_armed_evidence(
            substrate.copy_circuit.evidence, args.evidence_content_arm, args.perm_seed
        )
        substrate.copy_circuit.evidence = armed  # type: ignore[method-assign]

    #: v17 **资格档**：只在答复的前 K 步发证据，之后把这条通道静音。K 不是挑出来的——
    #: 由 §第三十次停靠·定价档按先写死的规则取（最小的 2 的幂、覆盖 ≥90% 命中偏移 ⇒ K=64）。
    #: 刻度由本仪器自己声明：`loop_steps` 是**环内 observe 的序位**，1 步＝1 字节。
    #: v18：检索侧资格档装在**存储**上（与发射侧包装器正交），只用公开接口 `events()/clear()/record()`；
    #: `reset` 在每题开头调一次。代价：保留事件的 `event_id` 会重新编号（件里披露）。
    oracle_state = {"tokens": [], "calls": 0, "found": 0, "fell_through": 0}
    oracle_set_tokens = None
    if args.oracle_selector:
        if substrate.copy_circuit is None:
            raise RuntimeError("要求 oracle 选择档但回路不在场 ⇒ 没有可替换的 best_match")
        from probe_taiji_a30_copy_evidence_dose import _make_oracle_selector_arm

        oracle_fn, oracle_state, oracle_set_tokens = _make_oracle_selector_arm(
            substrate.copy_circuit.store
        )
        substrate.copy_circuit.store.best_match = oracle_fn  # type: ignore[method-assign]

    store_reset = None
    store_counters = [0, 0, 0]
    if args.store_scope_conversation:
        if substrate.copy_circuit is None:
            raise RuntimeError("要求检索侧资格档但回路不在场 ⇒ 没有可缩范围的存储")
        from probe_taiji_a30_copy_evidence_dose import _make_store_scoped_arm

        store_reset, store_counters, _stale_ids = _make_store_scoped_arm(
            substrate.copy_circuit.store
        )

    if args.product_window_steps is not None and args.evidence_window_steps is not None:
        raise SystemExit("v20：产品门与仪器替身档不能同开——同开就分不出读数是哪条路径给的")
    if args.product_window_steps is not None:
        if args.product_window_steps <= 0:
            raise SystemExit("v20：产品门的 K 必须是正整数")
        #: 走**产品原生门**（不是本仪器的替身包装器）：这是 owner 立项后的验收面。
        substrate.set_copy_evidence_window_steps(args.product_window_steps)
    if args.evidence_window_steps is not None:
        if substrate.copy_circuit is None:
            raise RuntimeError("要求资格档但回路不在场 ⇒ 没有可静音的证据通道")
        from probe_taiji_a30_copy_evidence_dose import _make_window_armed_evidence

        armed_window, window_counters = _make_window_armed_evidence(
            substrate.copy_circuit.evidence, args.evidence_window_steps, lambda: loop_steps[0]
        )
        substrate.copy_circuit.evidence = armed_window  # type: ignore[method-assign]

    per_item: list[dict[str, Any]] = []
    offenders: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    caller_totals: dict[str, int] = {}
    for item in items:
        history: list[tuple[str, str]] = []
        if oracle_set_tokens is not None:
            #: 标签来自题面本身；缺标签**响亮停下**，不许静默透传成"现状选择器"那样伪装成生效。
            expected = [str(tok) for tok in item.get("expected_contains") or []]
            if not expected:
                raise RuntimeError(f"{item['id']} 没有 expected_contains ⇒ oracle 档无标签可用")
            oracle_set_tokens([tok.encode("utf-8") for tok in expected])
        if store_reset is not None:
            store_reset()
        item_rows: list[dict[str, Any]] = []
        terminal_rows: list[dict | None] = []
        surface_checks: list[dict[str, Any]] = []
        worst_run = 0
        for index, turn in enumerate([str(t) for t in item["turns"]]):
            records.clear()  # 换一条答复：在案帧与资格档的步数都从 0 重数
            loop_steps[0] = 0
            if args.product_window_steps is not None:
                #: v20 修：这台仪器**自己驱动生成环**（不走 `Taiji.generate()`），所以产品门的
                #: "一趟一复位"必须由调用方在换答复处显式做。漏掉它的后果实测过：K 在第一趟之后
                #: 永久静音，`emitted=64／silenced=276`，读数 66/72 只是"等于不挂回路"而不是增益。
                substrate.reset_copy_evidence_window()
            answer = runtime.chat(
                turn,
                history=history,
                learn=False,
                max_length=args.max_length,
                repetition_penalty=args.penalty,
            )
            loop_records = [record for record in records if record["in_generation_loop"]]
            pre_records = [
                record
                for record in records
                if record["caller"].startswith("generate:") and not record["in_generation_loop"]
            ]
            for record in records:
                caller_totals[record["caller"]] = caller_totals.get(record["caller"], 0) + 1
            fed = [int(record["symbol"]) for record in loop_records]
            raw = trim_partial_tail(bytes(fed)).decode("utf-8", errors="replace")
            #: 产品出口其实有**两种**收口：边界的 `stop_at_boundary`（模型自己说完了），
            #: 以及生成串里再次出现 `问：` 时的**事后截断**（`chat()` 里那段 marker 切割）。
            #: 两手都必须分开记，因为"预算决定长度"这句话很容易被说过头——
            #: 实测这条面上两者都不出现（72 次生成：边界 0 次、`问：` 0 次 ⇒ 72/72 吃满预算）。
            #: （默认语料每篇只有一个 `问：` 与一个 `答：` ⇒ 两种停止编码在训练里的**频次其实相等**，
            #: 谁先出现是分布问题，不能靠推理定。）
            marker_at = min(
                (index for marker in _TURN_MARKERS if (index := raw.find(marker)) >= 0),
                default=None,
            )
            raw = (raw[:marker_at] if marker_at is not None else raw).strip()
            #: v6：重放走**完整产品面链**（同一个器官实例），与 `chat()` 返回值在同一表示层比较。
            replay = replay_surface_from_fed(
                fed,
                turn=turn,
                history=history,
                organ=runtime._chat_organ,
                turn_markers=_TURN_MARKERS,
            )
            surface_checks.append(
                {
                    "fed_bytes": len(fed),
                    "surface_is_replayed_raw": bool(replay["replay_surface"] == answer),
                    "turn_marker_fired": marker_at is not None,
                    "cut_by_marker_bytes": marker_at,
                    "answer_head": answer[:24],
                    # v6：早停样本（fed 极短）经器官重放后与答复逐位相等，不再记成面违规。
                    # 深帧复现纠正追加二：prompt 的 50 字节全部在案，1 字节是生成环的真实产量
                    # （边界符胜出后 break），那张"完整成句"答复是器官占位句模板。
                    "mismatch_reason": (
                        None
                        if replay["replay_surface"] == answer
                        else "surface_differs_from_replay"
                    ),
                    "replay_surface_head": replay["replay_surface"][:24],
                }
            )
            if not pre_records or not fed:
                surface_checks[-1]["replay_skipped"] = (
                    "no-pre-generation-observe" if not pre_records else "no-generated-bytes"
                )
                continue
            state = (0, 0)
            run_membership = char_membership_of_run(bytes(fed).decode("utf-8", errors="replace"))
            for position, record in enumerate(loop_records):
                basis = (
                    pre_records[-1]["probabilities"]
                    if position == 0
                    else loop_records[position - 1]["probabilities"]
                )
                byte = int(record["symbol"])
                row = replay_step(
                    basis,
                    byte,
                    boundary,
                    state,
                    fed[:position],
                    args.penalty,
                    args.penalty_window,
                )
                row["step"] = position
                #: v28（§72）：末步**字面身份**要能回答"停之前发的是哪个字节、当时在字的哪一段"。
                row["emitted_byte"] = byte
                row["utf8_state_before"] = [state[0], state[1]]
                row["utf8_byte_class"] = _utf8_byte_class(byte)
                row["in_run"] = bool(position < len(run_membership) and run_membership[position])
                if not row["emitted_is_argmax"]:
                    failure_examples.append({"id": item["id"], **row})
                item_rows.append(row)
                state = advance_utf8(state[0], state[1], byte)
            #: v27（DEBT-G26）：环体是 `argmax()==boundary → break → observe`，所以**停下那一步不产生帧**；
            #: 但每个 record 携带的是"喂完该字节之后"的下一步分布 ⇒ 最后一个在案 record 的概率
            #: 恰好就是终止决策的 logits。按同一 `replay_step` 口径补一行，不塞进 `item_rows`
            #: （既有聚合列 `steps_boundary_is_argmax` 等的语义因此一字不动，与 v22–v26 各件同格可比）。
            terminal_rows.append(
                None
                if len(loop_records) >= args.max_length
                else {
                    **replay_step(
                        loop_records[-1]["probabilities"],
                        boundary,
                        boundary,
                        state,
                        fed,
                        args.penalty,
                        args.penalty_window,
                    ),
                    "step": len(loop_records),
                    "utf8_state_before": [state[0], state[1]],
                }
            )
            worst_run = max(worst_run, _longest_same_char_run(answer))
            if index + 1 < len(item["turns"]):
                history.append((turn, answer))
        legal_sizes = [row["legal_candidates"] for row in item_rows]
        ranks = [row["boundary_rank_in_legal"] for row in item_rows]
        ratios = [
            row["ratio_best_over_boundary"] for row in item_rows if row["ratio_best_over_boundary"]
        ]
        summary = {
            "id": item["id"],
            "answer_run": worst_run,
            "steps": len(item_rows),
            "generations": len(surface_checks),
            "generations_cut_by_turn_marker": sum(
                1 for check in surface_checks if check["turn_marker_fired"]
            ),
            # v6 更正（第三处仪器缺陷，§2bb-追加四）：`eating_full_budget` 的旧定义
            # （非 marker 切割数）把**边界符自停**（fed < 预算＝生成环唯一的提前出口，
            # `break` 在 observe 之前）全数误标成"吃满预算"。真终止的两种形态都必须点数：
            "generations_boundary_self_stop": sum(
                1 for check in surface_checks if check["fed_bytes"] < args.max_length
            ),
            "generations_eating_full_budget": sum(
                1 for check in surface_checks if check["fed_bytes"] >= args.max_length
            ),
            "argmax_mismatch_steps": sum(1 for row in item_rows if not row["emitted_is_argmax"]),
            "surface_matches_replayed_raw": all(
                check["surface_is_replayed_raw"] for check in surface_checks
            ),
            "surface_checks": surface_checks,
            "steps_with_single_legal_candidate": sum(1 for size in legal_sizes if size <= 1),
            "steps_with_zero_boundary_probability": sum(
                1 for row in item_rows if row["boundary_probability_is_zero"]
            ),
            "steps_boundary_is_argmax": sum(1 for row in item_rows if row["boundary_is_argmax"]),
            "median_boundary_rank_in_legal": sorted(ranks)[len(ranks) // 2] if ranks else None,
            "median_ratio_best_over_boundary": (
                sorted(ratios)[len(ratios) // 2] if ratios else None
            ),
            "run_steps": sum(1 for row in item_rows if row["in_run"]),
            #: v16（第三十次停靠·设计预备第 2 条）：**边界符胜出发生在答复的第几步**——
            #: "只在前 K 步发证据"这种资格档必须先量出停止决定落在哪一段，才谈得上选 K；
            #: 量不到就那一档不跑。位置按**在环内的序位**数（0 起），跨轮不累加。
            #: v17：**改成从 `fed_bytes` 取**。v16 按"在案行的 `boundary_is_argmax`"数位置，
            #: 恒等于 0——边界符胜出那一步 `break` 发生在 `observe` 之前（这条就写在本文件 v6 说明里），
            #: 那一步根本不入案。自停的生成其 `fed_bytes` 就是停止发生的字节位置，是同一件事的正确刻度。
            "endstep_probe_v22": _endstep_probe_per_generation(
                item_rows, args.max_length, terminal_rows
            ),
            "run_boundary_win_positions": [
                int(check["fed_bytes"])
                for check in surface_checks
                if check["fed_bytes"] < args.max_length
            ],
        }
        per_item.append(summary)
        if worst_run >= 20 and item_rows:
            run_rows = [row for row in item_rows if row["in_run"]]
            run_ratios = [
                row["ratio_best_over_boundary"]
                for row in run_rows
                if row["ratio_best_over_boundary"]
            ]
            offenders.append(
                {
                    **summary,
                    "run_legal_hist": {
                        str(size): sum(1 for row in run_rows if row["legal_candidates"] == size)
                        for size in sorted({row["legal_candidates"] for row in run_rows})
                    },
                    "run_boundary_rank_hist": {
                        str(rank): sum(
                            1 for row in run_rows if row["boundary_rank_in_legal"] == rank
                        )
                        for rank in sorted({row["boundary_rank_in_legal"] for row in run_rows})
                    },
                    "run_ratio_min": min(run_ratios) if run_ratios else None,
                    "run_ratio_median": (
                        sorted(run_ratios)[len(run_ratios) // 2] if run_ratios else None
                    ),
                    "run_rows_sample": run_rows[:12],
                }
            )

    substrate.observe = original_observe  # type: ignore[method-assign]
    report = {
        "format": "taiji-a30-stop-failure-v29",
        "format_note_v29": "v29（2026-10-03）：§第七十三次停靠 读出"
        "『两底 118/118 次自停的前一步都是字节 10（LF）』之后，那一跳的**修法方向**取决于一个还没量过的量："
        "拖写的生成是**根本没发 LF**，还是**发了没接住**。本版按 §73 末段冻线补 `lf_trace_v29`"
        "（每代 `lf_step_count`／`last_lf_step`／`generation_last_step`）与件级 `lf_followthrough_summary_v29`"
        "（自停组与吃满组各自的『发过 LF 的占比』、末 LF 相对该代末尾的四分堆）。"
        "既有列一字未动 ⇒ 与 v27/v28 各件同格可比。",
        "format_note_v28": "v28（2026-10-03）：按 §第七十二次停靠 的预注册加**末步字面身份**——逐步行补 "
        "`emitted_byte`／`utf8_byte_class`／`utf8_state_before` 三个键，每代补 `tail_identity`"
        "（停之前那一步发的是哪个字节、当时在字的哪一段、该步的边界名次与概率），件级补 "
        "`tail_identity_summary_v28`（自停组 vs 吃满组各自的字节分布与位置类分布，分母各自算）；"
        "终止行另带 `terminal_utf8_state_before`。理由：§71 把问题移到了『那一跳为什么不发生』，"
        "而现有件里只有 24 字符的 `answer_head`、**没有字节尾串**，所以字面身份必须重放补记；"
        "缺 `emitted_byte` 直接抛错（不静默交空）。既有列一字未动 ⇒ 与 v27 各件同格可比。",
        "format_note_v19": "v19 加性多一条检索侧 **oracle** 档：`--oracle-selector` 把 `store.best_match` 换成『内容含本题 `expected_contains` 的第一条事件』，找不到则透传原实现，用来把『选对了还拖不拖写』从『内容身份』与『发射时刻』里单独摘出来验。缺标签时响亮停下而非静默透传（那会伪装成生效）；自述 `oracle_calls`／`oracle_found`／`oracle_fell_through`。默认关 ⇒ 与 v18 逐位可比。",
        "format_note_v18": "v18 **加性**多一条**检索侧**资格档：`--store-scope-conversation` 在每题开头把装载信封带来的陈旧事件请出候选集，只留本次对话被告知的内容可被 `best_match` 取到（只用公开接口 `events()/clear()/record()`；代价是 `event_id` 重新编号，已在件里披露）。它与窗口档正交：一个动候选集、一个动发射时刻。默认关 ⇒ 与 v17 逐位可比。",
        "format_note_v27": "v27（2026-10-03）：DEBT-G26——产品环体在 `argmax()==boundary` 时**先 break 再 observe**"
        "（`taiji/model.py:3238-3240`），而逐帧记录装在 observe 包装里 ⇒ **停下那一步过去在件里根本没有行**，"
        "`peak_is_last_step` 因此是个不可能为假的量（§第六十六次停靠的第②条据此写出的『尖峰与停下脱钩』已收回，见 §六十九）。"
        "本版按生成补两样：`terminal_decision`（用该代最后一个在案 record 携带的**下一步** logits 走同一个 `replay_step`，"
        "字节＝边界符、UTF-8 状态＝答复喂完后的状态；只在边界自停的代出现）与 `last_recorded_row`（最后一个在案步）；"
        "件级新增 `terminal_decision_summary_v27`（含 `terminal_rank_not_one_count` 面违规计数与 `pairing_ok` 配对自证）。"
        "**只加字段、不塞进 `item_rows`** ⇒ 既有聚合列语义一字不动，与 v22–v26 各件同格可比。",
        "format_note_v26": "v26（2026-10-03）：DEBT-G25——产品门的 `product_window_stats` 现在带全程累计三键"
        "（emitted_steps_total／silenced_steps_total／steps_seen_total）；旧三键语义不变（末趟），"
        "因为每趟复位使末趟在短答复生成上结构不可能静音，只看末趟会把成功的档自我否证。",
        "format_note_v25": "v25（2026-10-03）：DEBT-G24——`legal_candidates` 不含边界符而 `boundary_rank_in_legal` 在含边界符的集合里排名，"
        "两者放在一起会让名次比分母大 1（§64 实测到）。旧列保留以便与历史件同格比较，"
        "新增 `legal_candidates_including_boundary` 作无歧义分母；守卫钉住两条不等式。",
        "format_note_v24": "v24（2026-10-03）：按 §第六十三次停靠的预注册加 `endstep_probe_v22[*].at_steps`——"
        "固定步位 8/16/32/64/128 上各记 p_boundary 与边界名次与合法候选数（只存 5 个点，不存整条序列）。"
        "动机：§62 证明'按生成长度分组比峰值'条件在存活上（早停必然短），固定步位才把长度变成'能否走到'而非分组变量。",
        "format_note_v23": "v23（2026-10-02）：`endstep_probe_v22` 的分组改用全部逐步行（`step` 归零即换代），"
        "并把'是否吃满预算'的定义收到 `generation_steps >= max_length`；"
        "`in_run` 降级为第二维 `steps_in_repeat_run`。v22 那两件因分组误用不发表，见 §第六十次停靠。",
        "format_note_v22": "v22（2026-10-02）：§第五十九次停靠的四个标量按**每次生成**入案 `endstep_probe_v22`——"
        "核实过 item_rows 是把一个 item 的 3 次生成串接后再取中位数，故现成 median_* 列跨代混算、"
        "对'基底为什么不停'没有判别力。四标量不依赖外部真值：p_boundary 峰值与其步位、该步的边界名次与合法候选数、"
        "峰值是否落在该代最后一步、该代是否吃满预算。既有键不动 ⇒ 与 v17–v21 各档同格可比。",
        "format_note_v21": "v21（2026-10-02）：产品门的计步基改成**只数答复相**——`Taiji.generate()` 在 prompt 喂完后"
        "复位计数器。v20 那一版的 K 会被 prompt 段吃光（`emitted=64／silenced=276`＝`84＋256`），"
        "读数 66/72 与不挂回路逐列同值却不是增益。产品档与替身档仍互斥；默认 None ⇒ 逐位不变。",
        "format_note_v20": "v20（2026-10-02）：owner 裁「立项进产品」后加**产品档** `--product-window-steps`——"
        "它调的是产品侧原生生命周期门（`Taiji.set_copy_evidence_window_steps`，默认 None ⇒ 逐位不变），"
        "与本仪器 v17 那副 monkeypatch 替身互斥；件里新增 `product_window_steps`／`product_window_stats` "
        "与两条守卫（fired／两侧都报）。加字段不动既有键 ⇒ 与 v17–v19 各档同格可比。",
        "format_note_v17": "v17 两件事：① 加**资格档** `--evidence-window-steps`（前 K 步发、之后静音；档本身住在剂量探针里与复述面共用，刻度由本仪器声明为环内步数，守卫 `window_both_sides_seen` 要求两侧都出现过）；② 修 v16 那列结构上恒为 0 的 `run_boundary_win_positions`——边界符胜出那一步 `break` 在 `observe` 之前、不入案，正确刻度是自停生成的 `fed_bytes`（已在件里，无需重跑即可读出）。其余字段与判据一字未动。",
        "format_note_v16": "v16 加性只多两列**停止决定的位置**信息：每题 `run_boundary_win_positions` 与件级 `boundary_win_position_hist`。用途是给『只在前 K 步发证据』这一族资格档**定价**——设计预备第 2 条要求 K 只能从链上先量到的分布里取，不许事后挑刚好过线的那个。判据、计数与生成路径一字未动，故与 v13/v15 同格可比（锚点 23/49/6 与 total_steps 就是这条可比性的检验）。",
        "format_note_v15": "v15 **只是把 v13/v14 那副内容档搬到剂量探针里与复述面共用**（`_make_content_armed_evidence`）：轨迹面与复述面必须做**同一个**置换／冻结操作，各写一份就是两把尺子。字段、算法、默认关闭时的逐位行为一字未动 ⇒ 与 v13/v14 同格可比（锚点档的 23/49/6 就是这条可比性的检验）。",
        "format_note_v14": "v14 **只修 `frozen` 档的冻结源**（仪器缺陷，不是新测量）：v13 取'第一次调用'，而第一次调用时 store 仍为空 ⇒ 冻结到的是**精确零向量**，那一档实际测的是'把通道永久关掉'（现场证据：自停 66/72 与不挂回路那件同值、`content_arm_max_rel_l1_diff=1.0`）。现冻结到**第一条非零**证据并披露冻结点与它的 L1。`permutation` 档与其余字段一字未动 ⇒ v13 的置换档读数继续可比。",
        "format_note_v13": "v13 **加性**多一格 owner 裁定后要的那把分离尺：`--evidence-content-arm` "
        "（`permutation`＝把证据向量按固定种子置换，多重集与 L1/max 逐位不变，只毁掉"
        "'哪一维对应哪个符号'；`frozen`＝每次返回第一次那一条，内容不再跟着 cue 走），"
        "配自述 `evidence_content_arm`／`perm_seed` 与守卫 `content_arm_consumed`／"
        "`content_arm_magnitude_preserved`（置换档的**前提**：硬度不守恒整档作废，因为那时两个臂"
        "差的不止内容）。默认关 ⇒ 与 v12 逐位可比；其余字段与算法一字未动。"
        "加它的理由：上限档把环内静音推到 8.25% 之后真自停仍 23→25，'相似度轴'这一族在轨迹面上"
        "已经排除得差不多了 ⇒ 下一个要分开的是'听内容'还是'只要有一条非零向量在加'。",
        "format_note_v12": "v12 **加性**只补一位被 v11 现场暴露出来的空档：`evidence_alpha_calls`／"
        "`relevance_ceiling_consumed` 数的是'包装器被走到'，不是'过滤器开过枪'——一个从未命中的 c "
        "会给出与全剂量同值的读数却看不见自己是空的，而全链计数还把 prompt 侧（"
        "`record_told_history`）一起算了，那一边不改写自身轨迹那一列。现按**生成环内**归因，件里存 "
        "`relevance_ceiling_silenced_calls`／`_share`／`evidence_calls_in_generation_loop`，"
        "守卫 `relevance_ceiling_fired`＋`ceiling_fire_count_reported`；v11 那把松尺子 "
        "`relevance_ceiling_consumed` **退役**（它能在'一次都没命中'时报 true）。"
        "其余字段与算法一字未动 ⇒ 与 v4–v11 同格可比；包装器与 `_cosine` 仍复用剂量探针那一份，"
        "c 的定义不变。",
        "format_note_v11": "v11 **加性**多一个旗标 `--relevance-ceiling-c`（自信度衰减：相似度高于 c 的步不发）"
        "与两条自述（`relevance_ceiling_c`／守卫 `relevance_ceiling_consumed`），其余字段与算法一字未动"
        " ⇒ 与 v4–v10 同格可比。加它的理由：DEBT-G19 三条修法里只有上限这条还没在**自身轨迹面**上被测过，"
        "而 L2 才是晋升判据用的那一列；包装器与 `_cosine` 都复用剂量探针那一份，三半里的 c 是同一个定义。",
        "format_note_v10": "v10 **加性**只补一条被点名过两次的缺口：件里此前只记 `circuit_sha256`、不记**检查点自身**的 sha ⇒ "
        "与旧档（如 09-30 那枚被作废的 quarter v4 件）只能按'同路径＋mtime 早于那次跑'配对，那是**路径级**不是 sha 级。"
        "现补 `checkpoint_sha256`（跑前那一次读盘，与 `base_sha256_unchanged` 共用同一趟哈希，不多读一遍 12MB）"
        "＋守卫 `checkpoint_sha_recorded`。其余字段与算法一字未动 ⇒ 与 v4–v9 同格可比。",
        "format_note_v9": "v9 **加性**多一个旗标 `--copy-evidence-alpha`（把回路的加性证据整体乘 α）与两条自述"
        "（`copy_evidence_alpha`／守卫 `evidence_alpha_consumed`），其余字段与算法一字未动 ⇒ 与 v4–v8 同格可比。"
        "加它的理由：DEBT-G19 查出这条通道是三条加性证据里唯一没有强度系数的那一条；剂量档要与 v8 的"
        "'门开关'档用**同一台仪器、同一个乘数来源**，否则又是一口井里的水。",
        "format_note_v8": "v8 **加性**多一个旗标 `--no-copy-evidence-gate`（挂完回路后把那条加性证据的 UTF-8 "
        "位置门显式关掉）与两条自述（`copy_evidence_gate_off_requested`／守卫 `evidence_gate_flag_honored`），"
        "其余字段与算法一字未动 ⇒ 与 v4–v7 同格可比。加它的理由：v7 之前把'挂载回路'当成一个变量，其实它一次"
        "开两样东西（回路的 prompt／证据通道 ＋ 那条门），拆开来才允许写'是回路的通道造成'。",
        "format_note_v7": "v7 **加性**多存两条回写门槛自述（`surface_gate_state`／`write_back_gate_last_reason`），"
        "其余字段与算法一字未动 ⇒ 与 v4–v6 同格可比。加它的理由：门槛①（回写放行）**不是**由挂载回路武装的——"
        "它按`seed/surface_gate.py:150-156` 的'工件与检查点同目录'规则在 `load()` 里解析，"
        "`enable_copy_circuit` 之后仍是 `disarmed:artifact_absent`（2026-10-02 运行时实测：候选基底挂上 seed-A 后 "
        "`last_write_back_gate=(True, 'factory_face_or_gate_disarmed')`）。不报这一位，就会把'挂上回路'误读成'同时"
        "把回写门槛也开上了'——我 2026-10-02 早上就按那个误读写过一句保守的错账（PLAN-A-30 第九次停靠·更正）。",
        "format_note_v6": "v6 **换比较的表示层＋更正早停计数**（诊断更正，见 §2bb-追加三/追加四）："
        "（一）v5 的逐位比较是『raw 重放 vs `chat()` 返回值』，而早停样本（边界符胜出即 break，fed 极短）的返回值是"
        "**器官占位句**（`NativeReadableTextLanguageOrgan._fallback_text`）——V019 首轮 1 字节生成"
        "对应 44 字占位句，逐位比较必然不等，L2 要测的成功被记成面违规。v6 的重放改走"
        "**完整产品面链**（`replay_surface_from_fed`：decode → marker 切割 → **同一个器官实例** emit）"
        "再与返回值比较；`replay_tiny_feed` 这一分类随之退役（深帧复现证明 prompt 的 50 字节全部在案，"
        "追加二『没复现 prompt』的误诊被纠正）。"
        "（二）`generations_eating_full_budget` 的旧定义（`generations − marker 切割数`）把**边界符自停**"
        "（fed < 预算＝生成环唯一提前出口）误标成吃满预算——v4/v5 的『早停 1/72、5/72』数的其实是"
        "marker 切割；真正的边界自停（对照件 0/72、self/quarter 各 19/72）从未被点数。v6 起"
        " `eating_full_budget` 按 `fed ≥ max_length` 判，新增 `generations_boundary_self_stop`。"
        "`all_surfaces_are_replayed_raw` 与 `replay_suspect_generations` 的**公式不变**"
        "（后者改数 `surface_differs_from_replay`），严格性不降：真正被产品链改过的面仍然红。",
        "format_note_v5": "v5 **加性**：每条 `surface_checks` 多一列 `mismatch_reason`"
        "（`replay_tiny_feed`＝回放侧只归到 <8 个符号，属**仪器归类缺陷**；`surface_differs_from_replay`＝真的面不一致），"
        "聚合里多一条 `replay_suspect_generations`。**严格守卫 `all_surfaces_are_replayed_raw` 一字未动**"
        "——放宽它等于挪门柱；加这一列只是为了让'仪器坏了'与'被测量不符'能被分开读"
        "（§2bb-追加二 复现 `V019` 得到的就是前者）。其余各表算法与语义未动 ⇒ 与 v4 同格可比。",
        "format_note_v4": "v4 **加性**多存三条装配自述（`mount_route`／`copy_circuit_present_after_load`／"
        "`copy_evidence_utf8_gate`），其余字段与算法一字未动 ⇒ 与 v3 同格可比。加它的理由：`--circuit` 不给 **不等于**"
        '"出厂无回路面"——带回路的信封在 `SeedRuntime.load` 里会自动挂载，所以这张面是按命令行猜出来的。',
        "prereg": "plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md §2f 第 3 条",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "circuit_sha256": _payload_sha(args.circuit),
        #: v10：跨档配对应到 sha 级（见 format_note_v10）。
        "checkpoint_sha256": sha_before[:16],
        # 取数面按这三条判，不按命令行猜（v4）。
        "mount_route": (
            "enable_copy_circuit"
            if args.circuit
            else (
                "envelope_auto_mount"
                if getattr(runtime.model.substrate, "copy_circuit", None) is not None
                else "none"
            )
        ),
        "copy_circuit_present_after_load": (
            getattr(runtime.model.substrate, "copy_circuit", None) is not None
        ),
        # 有效值不是 config 那一位：`taiji/config.py:236` 默认 False，而 restore 的自动挂载分支
        # 会 `set_copy_evidence_utf8_gate(True)`（owner 裁定 (b)，见 `taiji/model.py:3497` 那段注释）。
        # 只报 config 会把"门是开的"读成"门是关的"，故这里报**有效值**并同带两个成分。
        "copy_evidence_utf8_gate_effective": bool(
            runtime.model.substrate.config.copy_evidence_utf8_gate
            if getattr(runtime.model.substrate, "_copy_evidence_utf8_gate_override", None) is None
            else bool(runtime.model.substrate._copy_evidence_utf8_gate_override)
        ),
        "copy_evidence_utf8_gate_config": bool(
            runtime.model.substrate.config.copy_evidence_utf8_gate
        ),
        "copy_evidence_utf8_gate_override": getattr(
            runtime.model.substrate, "_copy_evidence_utf8_gate_override", None
        ),
        # v7：门槛①（回写放行）与上面那条"证据 UTF-8 位置门"是**两条不同的门**，一起报才不会互相顶名。
        "copy_evidence_gate_off_requested": bool(args.no_copy_evidence_gate),
        "copy_evidence_alpha": args.copy_evidence_alpha,
        "relevance_ceiling_c": args.relevance_ceiling_c,
        #: v13 自述：这一档替换的是**内容身份**，硬度分布由守卫逐项验，不是靠注释声明。
        "evidence_window_steps": args.evidence_window_steps,
        #: v20：产品原生门的自述（None ⇒ 这条面从未被走）。
        "product_window_steps": args.product_window_steps,
        "product_window_stats": substrate.copy_evidence_window_stats(),
        "store_scope_conversation": bool(args.store_scope_conversation),
        "oracle_selector": bool(args.oracle_selector),
        "evidence_content_arm": args.evidence_content_arm,
        "perm_seed": args.perm_seed if args.evidence_content_arm == "permutation" else None,
        "surface_gate_state": runtime.surface_gate_state,
        "write_back_gate_last_reason": (
            str(runtime.last_write_back_gate[1]) if runtime.last_write_back_gate else None
        ),
        "repetition_penalty": args.penalty,
        "repetition_window": args.penalty_window,
        "max_length": args.max_length,
        "generation_loop_lines": [loop_first, loop_last],
        "items": len(items),
        #: v16 聚合：全部在案步里边界符胜出的位置分布（按 0–15／16–31／32–63／64–127／128＋ 分堆）。
        "boundary_win_position_hist": _position_histogram(
            [pos for row in per_item for pos in row["run_boundary_win_positions"]]
        ),
        #: v27（DEBT-G26）：把"停下那一步"补成可见行后的件级判读列（新高型 vs 竞争型）。
        "terminal_decision_summary_v27": _terminal_summary_v27(per_item),
        #: v28（§72）：末步字面身份的四张分布表（自停组 vs 吃满组，各自分母）。
        "tail_identity_summary_v28": _tail_identity_summary_v28(per_item),
        #: v29（§73 末段冻线）：拖写的生成是"没发 LF"还是"发了没接住"。
        "lf_followthrough_summary_v29": _lf_followthrough_summary_v29(per_item),
        "instrument_guard": {
            "observe_calls_recorded": bool(records),
            # v7 自述守卫：这条面必须**报出**回写门槛状态（ None／缺键都算仪器没走到，红）。
            "surface_gate_state_reported": runtime.surface_gate_state is not None,
            # v9：乘数必须**被消费**——传了非 1.0 的 α 而计数为 0，就是补丁没走到（假档）。
            "evidence_alpha_consumed": (args.copy_evidence_alpha == 1.0) or alpha_calls[0] > 0,
            "evidence_alpha_calls": alpha_calls[0],
            #: v12：退役 v11 那把松尺子（`relevance_ceiling_consumed` 只看全链调用数 ⇒ 一个从未命中
            #: 的 c 也能报 true）。换成**在生成环内**归因的一对：开过几枪、占环内调用的多少。
            "relevance_ceiling_fired": (args.relevance_ceiling_c is None or loop_silenced[1] > 0),
            "relevance_ceiling_silenced_calls": loop_silenced[1],
            "relevance_ceiling_silenced_share": (
                round(loop_silenced[1] / loop_silenced[0], 6) if loop_silenced[0] else None
            ),
            "evidence_calls_in_generation_loop": loop_silenced[0],
            "ceiling_fire_count_reported": args.relevance_ceiling_c is None or loop_silenced[0] > 0,
            #: v13：内容档必须**被走到**；置换档的硬度守恒是这一档的**前提**——不守恒时两个臂差的
            #: 就不只是"内容"，整档作废。`frozen` 档是故意换掉内容的分布 ⇒ 守恒项对它不作判，报 `null`。
            "content_arm_consumed": args.evidence_content_arm is None or content_guard[0] > 0,
            #: v17：资格档必须**既发过也静音过**——静音数为 0 说明窗口没起作用（等于没这档），
            #: 发出数为 0 说明窗口关得太早（整条通道恒零，那是另一档的读数，不是资格档）。
            "window_arm_consumed": args.evidence_window_steps is None or window_counters[0] > 0,
            #: v20：产品门也必须**自证开过枪**——steps_seen=0 ⇒ 门根本没被走到（假档）。
            "product_window_fired": args.product_window_steps is None
            or substrate.copy_evidence_window_stats()["steps_seen"] > 0,
            "product_window_emitted_and_silenced_reported": args.product_window_steps is None
            or all(
                key in substrate.copy_evidence_window_stats()
                for key in ("emitted_steps", "silenced_steps")
            ),
            #: v18：检索侧档必须**被走到**（清掉的陈旧条数 > 0），并把重置后的候选集大小上下界存进件里。
            "store_scope_consumed": (not args.store_scope_conversation) or store_counters[0] > 0,
            #: v19：oracle 档必须被走到，且"选对率"如实披露（fell_through 高 ⇒ 库里根本没有正确事件）。
            "oracle_consumed": (not args.oracle_selector) or oracle_state["calls"] > 0,
            "oracle_calls": oracle_state["calls"],
            "oracle_found": oracle_state["found"],
            "oracle_fell_through": oracle_state["fell_through"],
            "store_scope_stale_removed": store_counters[0],
            "store_scope_events_min": store_counters[1],
            "store_scope_events_max": store_counters[2],
            "window_emitted_calls": window_counters[1],
            "window_silenced_calls": window_counters[2],
            "window_both_sides_seen": args.evidence_window_steps is None
            or (window_counters[1] > 0 and window_counters[2] > 0),
            "content_arm_magnitude_preserved": (
                None if args.evidence_content_arm != "permutation" else content_guard[3] <= 1e-5
            ),
            "content_arm_max_rel_l1_diff": content_guard[3],
            "content_arm_frozen_at_call": content_guard[4] or None,
            "content_arm_frozen_l1": content_guard[5] or None,
            "content_arm_calls": content_guard[0],
            "content_arm_l1_original_sum": round(content_guard[1], 6),
            "content_arm_l1_replaced_sum": round(content_guard[2], 6),
            #: 全链总量留在件里，与环内量并排——两数之比就是"prompt 侧占了多少"。
            "relevance_ceiling_silenced_calls_all_chains": alpha_calls[2],
            # v8：旗标必须**被走到**——传了 `--no-copy-evidence-gate` 却仍报出有效值为真，就是仪器没生效。
            "evidence_gate_flag_honored": (not args.no_copy_evidence_gate)
            or not bool(
                substrate.config.copy_evidence_utf8_gate
                if getattr(substrate, "_copy_evidence_utf8_gate_override", None) is None
                else substrate._copy_evidence_utf8_gate_override
            ),
            "observe_calls_by_caller": dict(
                sorted(caller_totals.items(), key=lambda pair: -pair[1])[:12]
            ),
            "all_items_reconstructed": not any(row["argmax_mismatch_steps"] for row in per_item),
            "items_failing_reconstruction": [
                row["id"] for row in per_item if row["argmax_mismatch_steps"]
            ],
            "total_argmax_mismatch_steps": sum(row["argmax_mismatch_steps"] for row in per_item),
            "all_surfaces_are_replayed_raw": all(
                row["surface_matches_replayed_raw"] for row in per_item
            ),
            # v6：`replay_tiny_feed` 退役（见 format_note_v6）——可疑生成改数真面不一致。
            "replay_suspect_generations": sum(
                1
                for row in per_item
                for check in row["surface_checks"]
                if check.get("mismatch_reason") == "surface_differs_from_replay"
            ),
            "items_whose_surface_is_not_model_bytes": [
                row["id"] for row in per_item if not row["surface_matches_replayed_raw"]
            ],
            "total_steps": sum(row["steps"] for row in per_item),
            "generations": sum(row["generations"] for row in per_item),
            "generations_cut_by_turn_marker": sum(
                row["generations_cut_by_turn_marker"] for row in per_item
            ),
            # v6 更正：两列都改由 fed 字节直接判定（见 per_item 同名字段的注释）。
            "generations_boundary_self_stop": sum(
                row["generations_boundary_self_stop"] for row in per_item
            ),
            "generations_eating_full_budget": sum(
                row["generations_eating_full_budget"] for row in per_item
            ),
            "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
            #: v10 守卫：这个字段必须真的报出来（空串/缺哈希都算仪器没走到）。
            "checkpoint_sha_recorded": bool(sha_before) and len(sha_before) == 64,
        },
        "failure_examples": failure_examples[:5],
        "offender_count": len(offenders),
        "offenders": offenders,
        "per_item": per_item,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / f"reports/taiji_a30_stop_failure_20260928_ml{args.max_length}.json"
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
                "items": report["items"],
                "guard": report["instrument_guard"],
                "failure_examples": report["failure_examples"],
                "offender_count": report["offender_count"],
                "offenders": [
                    {
                        "id": offender["id"],
                        "run": offender["answer_run"],
                        "steps": offender["steps"],
                        "run_steps": offender["run_steps"],
                        "starved": offender["steps_with_single_legal_candidate"],
                        "boundary_zero_steps": offender["steps_with_zero_boundary_probability"],
                        "boundary_argmax_steps": offender["steps_boundary_is_argmax"],
                        "run_legal_hist": offender["run_legal_hist"],
                        "run_boundary_rank_hist": offender["run_boundary_rank_hist"],
                        "run_ratio_min": offender["run_ratio_min"],
                        "run_ratio_median": offender["run_ratio_median"],
                    }
                    for offender in offenders
                ],
                "out": out.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

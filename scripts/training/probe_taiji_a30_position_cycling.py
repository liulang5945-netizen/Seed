"""PLAN-A-30 判读 2 的逐 step 诊断：电路有没有"这段已经说过"的状态。

读码得到的机理（`taiji/copy_circuit.py:278-304`）：复制证据每一步都把**整条告知的所有位置**
做 softmax，位置上的推进信号只有两样——F1 语境投影 `query_state`，以及
`_successor_bonus`：**内容里前一个字节等于 `prev_byte` 的那些位置**各加一份 `copy_induce_bias`。
⇒ "下一步该说哪个字节"完全由**上一个字节**决定，档里没有"已经说到哪儿"的状态。
于是一旦某个字节在同一条告知里出现多次（CJK 三字节序列里极常见），后继位置就有多个候选，
其中包含**靠前的那些**——复述完一遍后再从中间/开头重走，就是循环。

本探针把这条机理**量出来**而不是宣布它：对每题每一步记
`argmax 位置 / 该位置质量 / 后继候选个数（歧义度）/ 是否倒退（比上一步的 argmax 靠前）`，
再对答案做**最小周期检测**（复读的单位是不是告知内容的连续子串），
并按"这条答案有没有复读"做**同臂内对照**（本仓规矩：涨点/遮点都要有能区分的对照）。

可推翻条件（预登记）：若"无复读"那一堆的倒退率与"复读"那堆同高，或复读答案的位置轨迹
**单调前进从不倒退** ⇒ 判读 2 降级，病回到键/选择侧（`PLAN-A-27` 战线二那条链）。

用法：
    python scripts/training/probe_taiji_a30_position_cycling.py \
        --circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt --limit 24
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimal_period(text: str) -> dict[str, Any]:
    """答案是不是"某个单位重复 k 遍"：返回最小周期长度、单位与重复次数。

    取**字符**级周期（不是字节级），因为 CJK 一字三字节，按字节算周期会把"同一个字重复"
    报成周期 3——那与"三个字组成的短语循环"在可读性上是完全不同的东西。
    """
    n = len(text)
    if n < 4:
        return {"period": n, "repeats": 1, "unit": text}
    for size in range(1, n // 2 + 1):
        if n % size:
            continue
        unit = text[:size]
        if unit * (n // size) == text:
            return {"period": size, "repeats": n // size, "unit": unit}
    return {"period": n, "repeats": 1, "unit": text}


def best_partial_period(text: str) -> dict[str, Any]:
    """严格周期之外再量一次"尾巴上在循环"：取最长后缀里重复三遍以上的最短单位。

    真实表层答复几乎不会是整串严格周期（开头常有几个杂字节），只看 `minimal_period`
    会把"复读 20 遍"判成"无周期"。
    """
    n = len(text)
    for size in range(1, max(2, n // 3) + 1):
        tail = text[-size * 3 :] if n >= size * 3 else ""
        if len(tail) == size * 3 and tail[:size] == tail[size : 2 * size] == tail[2 * size :]:
            repeats = 1
            index = n - size
            while index - size >= 0 and text[index - size : index] == text[:size]:
                repeats += 1
                index -= size
            return {"period": size, "repeats": repeats, "unit": text[:size], "cyclic_tail": True}
    return {"period": 0, "repeats": 0, "unit": "", "cyclic_tail": False}


def wrap_events(positions: list[int], length: int) -> int:
    """锯齿signature：走完一遍告知后又跳回开头 ⇒ "说完再来一遍"的直接形态。

    判据取**同一条事件内**的相邻两步：前一步已到尾部（≥75%），后一步回到头部（≤25%）。
    """
    if length <= 1:
        return 0
    high = 0.75 * (length - 1)
    low = 0.25 * (length - 1)
    return sum(1 for a, b in zip(positions, positions[1:], strict=False) if a >= high and b <= low)


def chance_wraps(positions: list[int], length: int, draws: int = 8) -> float:
    """同一串位置的**乱序基线**——不回绕就比随机还高，那"锯齿"就是错觉。

    固定种子，件里可复现。这一步不是装饰：首版我只量"倒退率"，n=2 时复读组 0.3056
    对无复读组 0.3333 ——**没有基线的比率没法判读**。
    """
    if length <= 1 or len(positions) < 3:
        return 0.0
    import random

    generator = random.Random(20260928)
    total = 0
    for _ in range(draws):
        shuffled = list(positions)
        generator.shuffle(shuffled)
        total += wrap_events(shuffled, length)
    return round(total / draws, 4)


def analyze_steps(records: list[dict[str, Any]]) -> dict[str, Any]:
    """位置轨迹：回绕（锯齿）率、歧义度、质量集中度，外加乱序基线。"""
    if not records:
        return {"steps": 0}
    #: 只有"确实选到了事件并算出位置"的步子才进轨迹；其余如实报数——
    #: 如果这一堆占多数，本诊断的结论就不成立（要的是"有多少步可归因"，不是默认全可归因）。
    positioned = [r for r in records if r.get("argmax_position") is not None]
    ambiguity = [r["successor_candidates"] for r in positioned]
    masses = [r["mass_at_argmax"] for r in positioned]
    regress = sum(
        1
        for a, b in zip(positioned, positioned[1:], strict=False)
        if a["event_id"] == b["event_id"] and b["argmax_position"] < a["argmax_position"]
    )
    wraps = 0
    chance = 0.0
    for _event_id, group in _by_event(positioned).items():
        length = max((g.get("content_chars", 0) for g in group), default=0)
        positions = [g["argmax_position"] for g in group]
        wraps += wrap_events(positions, length)
        chance += chance_wraps(positions, length)
    return {
        "steps": len(records),
        "positioned_steps": len(positioned),
        "regressions": regress,
        "regression_rate": round(regress / max(len(positioned) - 1, 1), 4),
        "wrap_events": wraps,
        "wrap_chance_baseline": round(chance, 4),
        "mean_successor_candidates": (
            round(sum(ambiguity) / len(ambiguity), 4) if ambiguity else 0.0
        ),
        "steps_with_multiple_successors": sum(1 for a in ambiguity if a > 1),
        "mean_mass_at_argmax": round(sum(masses) / len(masses), 4) if masses else 0.0,
        "distinct_events_used": len({r["event_id"] for r in records if r["event_id"] is not None}),
    }


def _by_event(records: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["event_id"], []).append(record)
    return grouped


def unit_in_content(unit: str, contents: list[str]) -> bool:
    return any(unit and unit in content for content in contents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    items = json.loads(manifest.read_text(encoding="utf-8"))["dimensions"]["X"]["items"][
        : args.limit
    ]

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
    circuit = substrate.copy_circuit
    assert circuit is not None, "治疗臂必须挂上回路"

    #: 实例级包装（不改产品源码，同 `probe_taiji_a27_walked_ledger` 的做法）：
    #: 原函数照调，返回值原样返回；诊断量另算一次 `_position_weights`（只读、确定性）。
    trace: list[dict[str, Any]] = []
    original = circuit.evidence

    def spy(**kwargs: Any) -> Any:
        out = original(**kwargs)
        cue = kwargs.get("cue")
        f1 = kwargs.get("f1_context")
        prev_byte = kwargs.get("prev_byte")
        record: dict[str, Any] = {
            "step": len(trace),
            "prev_byte": None if prev_byte is None else int(prev_byte),
            "event_id": None,
            "argmax_position": None,
            "mass_at_argmax": 0.0,
            "successor_candidates": 0,
        }
        try:
            event = circuit._chosen_event(cue) if cue is not None else None
            if event is not None and f1 is not None:
                weights = circuit._position_weights(event, f1, prev_byte)
                position = int(torch.argmax(weights).item())
                codes = list(event.content)
                record.update(
                    {
                        "event_id": int(event.event_id),
                        "argmax_position": position,
                        "mass_at_argmax": round(float(weights[position].item()), 6),
                        "successor_candidates": int(
                            sum(
                                1
                                for i in range(1, len(codes))
                                if prev_byte is not None and int(codes[i - 1]) == int(prev_byte)
                            )
                        ),
                        "content_chars": len(codes),
                    }
                )
        except Exception as exc:  # 诊断不得改变生成 ⇒ 出错就记一条空样本并计数
            record["error"] = f"{type(exc).__name__}: {exc}"
        trace.append(record)
        return out

    circuit.evidence = spy  # type: ignore[method-assign]

    lock_dropped_before = int(circuit.selection_lock_dropped)
    texts: list[dict[str, Any]] = []
    for item in items:
        history: list[tuple[str, str]] = []
        marks: list[tuple[int, int]] = []
        for index, turn in enumerate([str(t) for t in item["turns"]]):
            start = len(trace)
            answer = runtime.chat(turn, history=history, learn=False)
            marks.append((start, len(trace)))
            texts.append({"item": item["id"], "text": answer, "span": (start, len(trace))})
            if index + 1 < len(item["turns"]):
                history.append((turn, answer))

    circuit.evidence = original  # type: ignore[method-assign]
    lock_dropped_after = int(circuit.selection_lock_dropped)

    contents = [
        str(event.content.decode("utf-8", errors="replace")) for event in circuit.store.events()
    ]
    #: 每题自己的对话文本——用来把"复述记忆"与"回显输入"分开（两者都表现为循环）
    turn_texts = {str(item["id"]): [str(turn) for turn in item["turns"]] for item in items}
    looped = [t for t in texts if best_partial_period(t["text"])["cyclic_tail"]]
    other = [t for t in texts if not best_partial_period(t["text"])["cyclic_tail"]]

    def group_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
        stats = [analyze_steps(trace[s:e]) for s, e in (r["span"] for r in rows)]
        steps = sum(s.get("steps", 0) for s in stats)
        positioned = sum(s.get("positioned_steps", 0) for s in stats)
        if not steps:
            return {"texts": len(rows), "steps": 0}
        regress = sum(s["regressions"] for s in stats)
        wraps = sum(s["wrap_events"] for s in stats)
        chance = sum(s["wrap_chance_baseline"] for s in stats)
        multi = sum(s["steps_with_multiple_successors"] for s in stats)
        periods = [best_partial_period(r["text"]) for r in rows]
        return {
            "texts": len(rows),
            "steps": steps,
            "positioned_steps": positioned,
            "regression_rate": round(regress / max(positioned - len(stats), 1), 4),
            #: 回绕数**必须与同一批位置的乱序基线一起报**。首版只量了"倒退率"，n=2 时
            #: 复读组 0.3056 对无复读组 0.3333 ——没有基线、组间又无差，那个数没法判读。
            "wrap_events": wraps,
            "wrap_chance_baseline": round(chance, 2),
            "multi_successor_share": round(multi / max(positioned, 1), 4),
            "mean_mass_at_argmax": round(
                sum(s["mean_mass_at_argmax"] * s["positioned_steps"] for s in stats)
                / max(positioned, 1),
                4,
            ),
            "mean_period": round(sum(p["period"] for p in periods) / len(rows), 2),
            "mean_repeats": round(sum(p["repeats"] for p in periods) / len(rows), 2),
            #: 复读单位的**出处**分三类：在告知库内容里／在本题对话文本里（输入回显）／都不在。
            #: 只量"是不是循环"分不开"复述记忆"与"回显输入"——那两条的修法完全不同。
            "unit_in_told_content": sum(1 for p in periods if unit_in_content(p["unit"], contents)),
            "unit_in_dialog_text": sum(
                1
                for r, p in zip(rows, periods, strict=True)
                if unit_in_content(p["unit"], turn_texts.get(r["item"], []))
            ),
            "unit_nowhere": sum(
                1
                for r, p in zip(rows, periods, strict=True)
                if p["cyclic_tail"]
                and not unit_in_content(p["unit"], contents)
                and not unit_in_content(p["unit"], turn_texts.get(r["item"], []))
            ),
        }

    report = {
        "format": "taiji-a30-position-cycling-v1",
        "prereg": "plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md 判读 2",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "items": len(items),
        "texts": len(texts),
        "evidence_calls": len(trace),
        "instrument_guard": {
            "evidence_called": len(trace) > 0,
            "lock_dropped_unchanged": lock_dropped_before == lock_dropped_after,
            "diagnostic_errors": sum(1 for r in trace if "error" in r),
            "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        },
        "texts_with_cyclic_tail": len(looped),
        "cyclic_share": round(len(looped) / max(len(texts), 1), 4),
        #: 前 12 步原样贴出来——归因覆盖度（`positioned_steps` 对不对）只有看得见步面才判得了，
        #: 也防"仪器自称算了但其实每步都没算出来"这类假读数。
        "trace_sample": trace[:12],
        "group_looped": group_report(looped),
        "group_other": group_report(other),
        "top_periods": Counter(best_partial_period(t["text"])["unit"] for t in looped).most_common(
            8
        ),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports/taiji_a30_position_cycling_20260928.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("top_periods",)}, ensure_ascii=False
        )
    )
    print("top_periods:", report["top_periods"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

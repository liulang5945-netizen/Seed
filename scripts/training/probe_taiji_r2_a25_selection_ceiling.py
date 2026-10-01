"""A2.5 前置的第二把尺（零训练）：**事件选择的天花板到底值不值得去抢**。

§9 的诊断只证明了"键现在不可分"，但**没证明存在一个可分的键**——我自己造的字符重叠 oracle
同样只有 7/12，说明参照选错了。于是换一把不依赖键的尺：**直接把选择做对**（只把正确那条
告知放进历史，干扰轮全部拿掉＝oracle 事件注入），看端到端能答到多少。

两臂同一次运行、同基座、同电路、同题集，唯一差别＝历史里给几条告知：
* ``as-is``   按评价集原样把**所有**前置用户轮入历史（今天线上的行为）；
* ``oracle``  只把**含可复制答案词**的那条告知入历史，其余干扰轮丢弃（选择被替它做对）。

判读（先冻结）：
* ``oracle_strict − asis_strict ≥ 3 题`` ⇒ 事件选择确实是瓶颈，A2.5 值得开（给它加可学参数）；
* 差 ``< 3 题`` ⇒ 选择不是当前瓶颈，**A2.5 不该开**，钱花在发射/表层那一侧。
  用 ≥3 题这条线与 §9 修订后的表层判据同源：n=12-16 的题集上，1-2 题的差不算方向。

纪律：计分链复用 A2.3b 已落地的产品原语与严格口径（`_answer_raw`／`copyable_tokens`），
不另起一套；`checkpoints/` 只读、跑前后 sha256 复核；只新增一份报告，不覆写。
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

MIN_GAP_ITEMS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _told_index_with_copyable(item: dict[str, Any], tokens: tuple[str, ...]) -> int | None:
    """找出**含可复制答案词**的那条前置告知轮下标（oracle 要留下的那一条）。"""
    turns = [str(turn) for turn in (item.get("turns") or [])]
    for index, turn in enumerate(turns[:-1]):
        if any(token in turn for token in tokens):
            return index
    return None


def _arm(runtime: Any, items: list[tuple[str, dict[str, Any]]], *, oracle: bool) -> dict[str, Any]:
    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw
    from score_taiji_r2_copy_strict_cap import copyable_tokens

    rows = []
    strict_hits = 0
    loose_hits = 0
    for item_id, item in items:
        turns = [str(turn) for turn in (item.get("turns") or [])]
        ask = turns[-1]
        tokens = copyable_tokens(item)
        if not tokens:
            continue
        keep = [index for index in range(len(turns) - 1)]
        if oracle:
            told = _told_index_with_copyable(item, tokens)
            keep = [] if told is None else [told]
        history: list[tuple[str, str]] = []
        for index in keep:
            reply = _answer_raw(runtime, turns[index], list(history))
            history.append((turns[index], reply))
        answer = _answer_raw(runtime, ask, list(history))
        loose = any(token in answer for token in item.get("expected_contains") or [])
        strict = any(token in answer for token in tokens)
        loose_hits += int(loose)
        strict_hits += int(strict)
        rows.append(
            {
                "id": item_id,
                "history_tells": [turns[index] for index in keep],
                "answer": answer[:60],
                "hit_loose": loose,
                "hit_strict": strict,
                "copyable_tokens": list(tokens),
            }
        )
    return {
        "items": len(rows),
        "correct_loose": loose_hits,
        "correct_strict": strict_hits,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--circuit", default="output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from score_taiji_r2_copy_strict_cap import copyable_tokens, load_items

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)

    items = [
        (item_id, item)
        for item_id, item in sorted(load_items().items())
        if item["dimension"] == "D" and copyable_tokens(item)
    ]

    def _run_arm(arm_name: str, *, mount: bool, oracle: bool) -> dict[str, Any]:
        """每臂独立加载 runtime——避免上一臂的入库/动力学残留串味。"""
        loaded = SeedRuntime.load(checkpoint)
        if mount:
            loaded.enable_copy_circuit(PROJECT_ROOT / args.circuit)
        result = _arm(loaded, items, oracle=oracle)
        result["mounted"] = mount
        return result

    report: dict[str, Any] = {
        "format": "taiji-r2-a25-selection-ceiling-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §9",
        "reading_rule": (
            f"oracle_strict - asis_strict >= {MIN_GAP_ITEMS} 题 ⇒ 事件选择是瓶颈，A2.5 值得开；"
            f"差 < {MIN_GAP_ITEMS} ⇒ 选择不是瓶颈，A2.5 不该开"
        ),
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "arms": {
            "asis_with_circuit": _run_arm("asis", mount=True, oracle=False),
            "oracle_with_circuit": _run_arm("oracle", mount=True, oracle=True),
            "asis_no_circuit": _run_arm("asis", mount=False, oracle=False),
        },
    }
    asis = report["arms"]["asis_with_circuit"]
    oracle = report["arms"]["oracle_with_circuit"]
    gap = oracle["correct_strict"] - asis["correct_strict"]
    report["gap_strict_oracle_minus_asis"] = gap
    report["verdict"] = (
        "事件选择是当前瓶颈 ⇒ A2.5 值得开"
        if gap >= MIN_GAP_ITEMS
        else "选择不是当前瓶颈 ⇒ A2.5 不该开（钱在发射/表层那一侧）"
    )
    report["base_sha256_unchanged"] = _sha256(checkpoint) == sha_before

    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports/taiji_r2_a25_selection_ceiling_20260925.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "items": asis["items"],
                "asis_strict": asis["correct_strict"],
                "oracle_strict": oracle["correct_strict"],
                "gap": gap,
                "base_unchanged": report["base_sha256_unchanged"],
                "out": out.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

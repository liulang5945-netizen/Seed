"""§91 的反事实收口规则计算器：在**已入库的逐步读数**上算"若当时有这条规则会在第几步停"。

来历（2026-10-03，PLAN-A-30 第九十一次停靠）：判据先于数写死，所以这台计算器必须在读到
任何 §91 读数**之前**存在——它不许在读到数之后再被改动语义。

规则（原样照 §91，不许调）：对每次生成，按步序看 `lf_margins_v34.per_step`；某条＝"这一步之前发过
`0x0A`"，若该步 `ratio_best_over_boundary ≤ R` ⇒ 反事实停在**那一步**（边界符胜出）；没有这样的步 ⇒ 该代仍算未停。
阈值网格先定 `R ∈ (1.05, 1.20, 1.50, 2.00)`，**四个各报一条，不许四选一当结论**。

两条内建的低报口径（§91 明令要如实写成下界）：
* `per_step` 每件只存**前 6 次** LF ⇒ 第 7 次以后的机会看不见，所以 `generations_with_lf_steps_gt_6` 要单独报；
* 更早退出**不改变**退出点之前的任何一步（生成环自回归于已发字节），所以前缀截断是精确的、不需要重放。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

#: §91 先写的网格；顺序与理由一起冻结（"几乎必须已赢"→"差一倍也算"）。
R_GRID = (1.05, 1.20, 1.50, 2.00)
R_GRID_REASON = (
    'R=1.05：只有"胜出者几乎已与边界同概率"才算，最保守的一档',
    "R=1.20：装机底拖写的比值中位实测在 1.212 附近，这一档刚好把它切在边缘",
    "R=1.50：甲判读线（耦合改动门槛）用的同一个倍数",
    'R=2.00：容忍"差一倍也算会停"，上界侧',
)
PER_STEP_CAP = 6  # 探针 v34 里 `pairs[:6]` 的常量，此处只做一致性核对。


def iter_generations(report: dict[str, Any]) -> list[dict[str, Any]]:
    """摊平成一份生成交集，并把题面身份挂上（逐代字典里**没有** `id`，它在 `per_item` 层）。"""

    out: list[dict[str, Any]] = []
    for item in report.get("per_item", []):
        for index, generation in enumerate(item.get("endstep_probe_v22", [])):
            generation["item_id"] = item.get("id")
            generation["generation_index"] = index
            out.append(generation)
    return out


def counterfactual_stop(generation: dict[str, Any], threshold: float) -> int | None:
    """返回反事实停止的步序；该代没有"LF 之后且比值 ≤R 的一步" ⇒ None。"""

    margins = generation.get("lf_margins_v34") or {}
    best: int | None = None
    for row in margins.get("per_step", []):
        ratio = row.get("ratio_best_over_boundary")
        if not ratio:  # None 或 0 都不算——0 不会是"胜出者/边界"的合法值。
            continue
        step = int(row["step"])
        if float(ratio) <= threshold and (best is None or step < best):
            best = step
    return best


def _quantiles_or_none(values: list[float]) -> dict[str, Any]:
    #: §91 的代价面在对照情形下可能整体为空 ⇒ 必须如实记 n=0，不许抛异常。
    if not values:
        return {"n": 0, "median": None, "p10": None}
    ordered = sorted(values)
    index = max(0, int(0.10 * (len(ordered) - 1)))
    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 4),
        "p10": round(ordered[index], 4),
    }


def summarize(generations: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    denominator = len(generations)
    actual_stops = [g for g in generations if g.get("terminal_decision") is not None]
    fired: list[dict[str, Any]] = []
    new_stops = 0
    for generation in generations:
        step = counterfactual_stop(generation, threshold)
        if step is None:
            continue
        original = int(generation["generation_steps"])
        fired.append(
            {
                "id": generation.get("item_id"),
                "generation_index": generation.get("generation_index"),
                "stop_step": step,
                "original_steps": original,
                "retained_share": round(step / original, 4) if original else None,
                "already_self_stopped": generation.get("terminal_decision") is not None,
            }
        )
        if generation.get("terminal_decision") is None:
            new_stops += 1
    truncation = [
        g
        for g in generations
        if int((g.get("lf_margins_v34") or {}).get("lf_steps", 0)) > PER_STEP_CAP
    ]
    return {
        "R": threshold,
        "denominator_generations": denominator,
        "actual_self_stop_count": len(actual_stops),
        "actual_self_stop_rate": round(len(actual_stops) / denominator, 4) if denominator else None,
        "counterfactual_stop_count": len(fired),
        "counterfactual_stop_rate": round(len(fired) / denominator, 4) if denominator else None,
        "delta_pp": (
            round(100.0 * (len(fired) - len(actual_stops)) / denominator, 2)
            if denominator
            else None
        ),
        "net_new_stops": new_stops,
        "counterfactual_stops_that_are_already_real": len(fired) - new_stops,
        "retained_share_of_cut": _quantiles_or_none(
            [float(item["retained_share"]) for item in fired if item["retained_share"] is not None]
        ),
        "truncated_generations_lf_steps_gt_6": len(truncation),
        "examples": fired[:5],
    }


def precondition_failures(report: dict[str, Any]) -> list[str]:
    """§91 的三条前置（不满足就**整档不发表**，但要把缺的那条响亮报出来）。"""

    failures: list[str] = []
    generations = iter_generations(report)
    if "v34" not in str(report.get("format", "")):
        failures.append("format_is_not_v34")
    margins = report.get("lf_margin_summary_v34")
    if not isinstance(margins, dict):
        failures.append("lf_margin_summary_v34_absent")
        return failures
    if margins.get("eaters_dynamic_range", {}).get("usable") is not True:
        failures.append("eaters_ruler_not_usable")
    if int(margins.get("eaters_with_lf_next", 0)) < 20:
        failures.append("eaters_with_lf_next_below_20")
    terminal = report.get("terminal_decision_summary_v27", {})
    if terminal.get("pairing_ok") is not True:
        failures.append("pairing_not_ok")
    if int(terminal.get("terminal_rank_not_one_count", -1)) != 0:
        failures.append("terminal_rank_not_one")
    #: ②原文："per_step 里 step 与比值都非空"——缺读数的件不能拿来算反事实。
    for generation in generations:
        for row in (generation.get("lf_margins_v34") or {}).get("per_step", []):
            if row.get("step") is None or row.get("ratio_best_over_boundary") is None:
                failures.append("per_step_has_missing_readout")
                return failures
    #: `argmax_mismatch_steps` 住在 `per_item` 层（件级没有这个键），按题面求和。
    if (
        sum(int(item.get("argmax_mismatch_steps", 0) or 0) for item in report.get("per_item", []))
        != 0
    ):
        failures.append("argmax_mismatch_nonzero")
    return failures


def run(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    generations = iter_generations(report)
    failures = precondition_failures(report)
    #: 前置不齐 ⇒ **一条数都不许出**（实测踩过：第一版照算，等于让退化尺子下的读数能被引用）。
    grid = (
        [summarize(generations, value) for value in R_GRID]
        if (generations and not failures)
        else []
    )
    payload: dict[str, Any] = {
        "report": report_path.name,
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "product_window_steps": report.get("product_window_steps"),
        "circuit": report.get("circuit"),
        "published": not failures,
        "precondition_failures": failures,
        "grid": grid,
        "grid_reason": list(R_GRID_REASON),
        "per_step_cap": PER_STEP_CAP,
    }
    #: 判据的**措辞**（甲/乙/丙）由 §91 负责；这里只算，不判。
    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="append", required=True, help="v34 报告件路径（可重复）")
    args = parser.parse_args(argv)
    results = [run(Path(path)) for path in args.report]
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    #: rc 只编码"前置是否齐"：**前置不齐 ⇒ 整档不发表**（rc=2），齐 ⇒ 0。
    return 0 if all(item["published"] for item in results) else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

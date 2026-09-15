"""把 CAP-0 报告里**待人工复核**的条目导成可填写的评分表（只读报告，不评测、不改报告）。

为什么现在需要它：[P3b 预注册](../../plans/reference/M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md)
§3 的 **J3** 要求"B 经人工复核达标"，而 B/G 两维每题都是 `score = null` +
`pending_human_review = true`（机检预判只是线索，不是分数）——**没有任何仪器能替人打这个分**。
双臂 campaign 要跑约 54 小时，若等它跑完再准备复核，用户的工时就成了主线串行的一段；
本工具把复核提前到训练期间并行做。

复核量的控制（写在这里以免事后放宽）：一份报告 B+G 共 40 题。逐阶段全复不现实，
所以默认按"**只导与上一份不同的题**"（`--only-changed-vs`）出表：
两臂在同一 tick 的答案、以及与 P3a 基线的答案做对比，相同即跳过。
**跳过的题仍要在表尾列出题号**，否则"没复核"会被读成"复核过且没变"。

产出只是**空表**（含 0/1/2 评分位与备注位）。填好的表才是证据；本工具不产生分数、
不判定 J3、也不改写任何报告。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PENDING_DIMENSIONS = ("B", "G")
#: 评分量表来自冻结评价自身的 scoring_discipline（自由回答用 0/1/2 并保留原始回答）。
RUBRIC = "0 = 不可用／答非所问；1 = 部分可用但有明显缺陷；2 = 达到该题意图"
#: 默认输出是一个**新**工作表名，绝不指向已封存的报告。
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "p3b_review_worksheet_20260916.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _items(report: dict[str, Any], dimension: str) -> dict[str, dict[str, Any]]:
    block = (report.get("dimensions") or {}).get(dimension)
    if not block:
        return {}
    return {str(item.get("id")): item for item in block.get("items") or []}


def pending_items(report: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(维度, 题号, 条目)：只取真的等人打分的题。"""

    out: list[tuple[str, str, dict[str, Any]]] = []
    for dimension in PENDING_DIMENSIONS:
        for item_id, item in sorted(_items(report, dimension).items()):
            if item.get("pending_human_review"):
                out.append((dimension, item_id, item))
    return out


def answers_by_id(report: dict[str, Any]) -> dict[str, str]:
    return {
        f"{dimension}/{item_id}": str(item.get("raw_last_output"))
        for dimension, item_id, item in _all_items(report)
    }


def _all_items(report: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    out: list[tuple[str, str, dict[str, Any]]] = []
    for dimension in PENDING_DIMENSIONS:
        for item_id, item in sorted(_items(report, dimension).items()):
            out.append((dimension, item_id, item))
    return out


def build_worksheet(
    report: dict[str, Any],
    *,
    label: str,
    changed_vs: dict[str, Any] | None = None,
) -> str:
    """Markdown 工作表；`changed_vs` 给出时只导答案确实发生变化的题。"""

    pending = pending_items(report)
    previous = answers_by_id(changed_vs) if changed_vs is not None else {}
    ticks = sorted(
        {str(item.get("tick")) for _, _, item in _all_items(report) if item.get("tick") is not None}
    )
    per_dimension = "；".join(
        f"{dimension}：{sum(1 for dimension_name, _, _ in pending if dimension_name == dimension)} 题"
        for dimension in PENDING_DIMENSIONS
    )
    lines: list[str] = [
        f"# CAP-0 人工复核工作表 · {label}",
        "",
        f"- 来源检查点：{report.get('checkpoint', '?')}（tick {'、'.join(ticks) or '?'}）",
        f"- 待复核维度：{'、'.join(PENDING_DIMENSIONS)}（其余维度有机检分，或整维记 not_executed）",
        f"- 每维待复核题数：{per_dimension}",
        f"- 量表：{RUBRIC}",
        "- 纪律：**未复核的题不得记为通过**；跳过的题号在表尾列出，空白评分位等于没复核。",
        "",
    ]
    if changed_vs is not None:
        lines += ["- 本表只导**答案相对上一份发生变化**的题（表尾列出被跳过的题号）。", ""]

    skipped: list[str] = []
    counted = 0
    for dimension, item_id, item in pending:
        key = f"{dimension}/{item_id}"
        if previous and previous.get(key) == str(item.get("raw_last_output")):
            skipped.append(key)
            continue
        counted += 1
        precheck = item.get("machine_precheck") or {}
        turns = item.get("turns") or [{}]
        prompts = " / ".join(str(turn.get("prompt", "")) for turn in turns)
        lines += [
            f"### {item_id} · {item.get('family', '?')}",
            "",
            f"- 提问：{prompts}",
            f"- 原始输出：{item.get('raw_last_output', '')}",
            f"- 机检预判：{precheck.get('machine_verdict', '?')}（{precheck.get('reason', '-')}）",
            "- **评分（0/1/2）**：",
            "- 备注：",
            "",
        ]
    lines += [
        "## 汇总",
        "",
        f"- 本题表需打分：**{counted}** 题",
        f"- 待复核总数（含被跳过的）：**{len(pending)}** 题",
        f"- 因「答案与上一份相同」而跳过：{len(skipped)} 题",
    ]
    if skipped:
        lines.append(f"- 跳过题号（**不等于已复核**）：{', '.join(skipped)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 CAP-0 待人工复核条目的评分表（只读）")
    parser.add_argument("--report", type=Path, required=True, help="一份 eval_taiji_cap0 报告")
    parser.add_argument("--label", default=None, help="写在表头的名字，缺省用文件名")
    parser.add_argument(
        "--changed-vs", type=Path, default=None, help="上一份报告，用于只导变化的题"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = _load(args.report)
    changed = _load(args.changed_vs) if args.changed_vs else None
    text = build_worksheet(report, label=args.label or args.report.name, changed_vs=changed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    counted = len(pending_items(report))
    print(
        json.dumps(
            {
                "event": "p3b_review_worksheet",
                "output": str(args.output.relative_to(PROJECT_ROOT)),
                "pending_items": counted,
                "dimensions": list(PENDING_DIMENSIONS),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

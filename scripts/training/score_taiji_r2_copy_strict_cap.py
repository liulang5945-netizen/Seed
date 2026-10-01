"""A2.3b 严格计分：把「真命中」定义成机器可检的**可复制实体词命中**（2026-09-25）。

**为什么需要本件**：既有 CAP 计分口径是 `any(token in answer for token in expected_contains)`，
而 `expected_contains` 里混着通用词（D03 的「是」「符合」、E 维的「一个」）。A2.4 读数里对照臂
（根本没装电路）在 D+E 上拿到 2 个命中，经明细核对全是这类通用词——**总分把仪器没测到的东西
也算成了能力**。复制通道的产物在结构上只能是「更早的用户轮里逐字出现过的字节串」，
所以严格口径＝**期望词必须在该题更早的用户轮原文里逐字出现**，否则该词不计入命中。

派生后果（本件实测，不在注释里断言）：
* E 维（算术/推导）单轮、答案不在任何告知文本里 ⇒ **E 维严格可命中数恒 0**——E 不是复制题，
  装电路也判不到；CAP D+E>0 这条冻结判据里真正承载复制能力的只有 D 维。
* 单轮题（turns 只有一轮）同样不可复制 ⇒ 严格集自动排除，无需人工挑。

两种用法：
* `--report <判决件> [--report ...]`：**零训练复算**已落地报告（只读，产新报告）；
* `--checkpoint/--circuit` 跑臂是 A2.3b 判读件的事，本件只负责口径与复算，不重跑模型。

纪律：不覆写既有判决件；输出文件已存在则加时间戳。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = PROJECT_ROOT / "plans/manifests/cap0_eval_set_v2.json"
#: 判决件行内答案字段的候选名（不同批次命名不同，按存在取用）。
_TEXT_KEYS = ("answer", "text")


def load_items(manifest_path: Path = MANIFEST) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: dict[str, dict[str, Any]] = {}
    for dimension, block in manifest["dimensions"].items():
        for item in block.get("items", []):
            if item.get("expected_contains"):
                items[str(item["id"])] = {**item, "dimension": dimension}
    return items


def copyable_tokens(item: dict[str, Any]) -> tuple[str, ...]:
    """该题**可能**由复制通道产出的期望词＝在更早的用户轮原文里逐字出现的期望词。

    最后一轮是提问轮，它不在"告知内容"里；单轮题没有更早轮 ⇒ 无可复制词。
    """
    turns = [str(turn) for turn in (item.get("turns") or [])]
    if len(turns) < 2:
        return ()
    prior = "".join(turns[:-1])
    return tuple(
        token for token in (item.get("expected_contains") or []) if token and token in prior
    )


def row_text(row: dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        if isinstance(row.get(key), str):
            return str(row[key])
    return ""


def annotate(arm: dict[str, Any], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """给一份报告的某个臂补严格读数（原 `hit` 字段保留，便于对账）。"""
    rows = []
    strict_hits = 0
    eligible = 0
    for row in arm.get("rows", []):
        item = items.get(str(row.get("id")))
        text = row_text(row)
        if item is None:
            rows.append({**row, "copyable_tokens": None, "hit_strict": None})
            continue
        tokens = copyable_tokens(item)
        eligible += int(bool(tokens))
        hit_strict = any(token in text for token in tokens)
        strict_hits += int(hit_strict)
        rows.append(
            {
                **row,
                "dimension": item["dimension"],
                "copyable_tokens": list(tokens),
                "hit_loose": bool(row.get("hit")),
                "hit_strict": hit_strict,
            }
        )
    return {
        "items": arm.get("items", len(rows)),
        "correct_loose": sum(1 for row in rows if row.get("hit_loose")),
        "correct_strict": strict_hits,
        "strict_eligible_items": eligible,
        "rows": rows,
    }


def recompute(report_path: Path, items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    arms = {
        key: annotate(value, items)
        for key, value in report.items()
        if isinstance(value, dict) and isinstance(value.get("rows"), list)
    }
    # 两臂同题时，"电路没参与"的题＝两臂答案逐字节相同——严格读数要把它单独计出，
    # 否则对照臂的通用词命中会被当成电路的功劳（A2.4 明细里 E20 就是这个形状）。
    keys = list(arms)
    inactive: dict[str, int] = {}
    if len(keys) == 2:
        left, right = keys
        left_rows = {row["id"]: row_text(row) for row in arms[left]["rows"] if "id" in row}
        right_rows = {row["id"]: row_text(row) for row in arms[right]["rows"] if "id" in row}
        shared = set(left_rows) & set(right_rows)
        inactive["identical_answer_arms"] = sum(
            1 for item_id in shared if left_rows[item_id] == right_rows[item_id]
        )
        strict_only_right = sum(
            1 for row in arms[right]["rows"] if row.get("hit_strict") and not row.get("hit_loose")
        )
        inactive["strict_hit_not_loose"] = strict_only_right
    return {
        "source_report": (
            report_path.relative_to(PROJECT_ROOT).as_posix()
            if report_path.is_relative_to(PROJECT_ROOT)
            else str(report_path)
        ),
        "source_format": report.get("format"),
        "source_prereg": report.get("prereg"),
        "criterion": report.get("verdict"),
        "arms": arms,
        "extra": inactive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=None,
        help="要复算的判决件（可多次给出）；默认复算 A2.3/A2.4 两份已落地报告",
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    items = load_items()
    defaults = [
        PROJECT_ROOT / "reports/taiji_r2_copy_circuit_cap_spotcheck_20260925.json",
        PROJECT_ROOT / "reports/taiji_r2_copy_circuit_chat_cap_20260925.json",
    ]
    paths = [Path(path) for path in args.report] if args.report else defaults
    report = {
        "format": "taiji-r2-copy-strict-cap-recompute-v1",
        "definition": (
            "hit_strict ⇔ 某期望词既在更早用户轮原文里逐字出现（可复制），又出现在答案里；"
            "单轮题与通用词自动排除。"
        ),
        "manifest": MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "strict_eligible_by_dimension": {
            dimension: sum(
                1
                for item_id, item in items.items()
                if item["dimension"] == dimension and copyable_tokens(item)
            )
            for dimension in sorted({item["dimension"] for item in items.values()})
        },
        "reports": [recompute(path, items) for path in paths],
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else (PROJECT_ROOT / "reports/taiji_r2_copy_strict_cap_recompute_20260925.json")
    )
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "strict_eligible_by_dimension": report["strict_eligible_by_dimension"],
        "arms": {
            f"{entry['source_report']}:{arm_name}": {
                "loose": arm["correct_loose"],
                "strict": arm["correct_strict"],
                "eligible": arm["strict_eligible_items"],
            }
            for entry in report["reports"]
            for arm_name, arm in entry["arms"].items()
        },
        "out": (
            out.relative_to(PROJECT_ROOT).as_posix()
            if out.is_relative_to(PROJECT_ROOT)
            else str(out)
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

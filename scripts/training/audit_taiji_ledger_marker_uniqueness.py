"""DEBT-G120 修法① 的普查仪：数 08 册的台账圈号（㊵-N），重号即红。

为什么现在才有这台仪器：全仓对「圈号」「重号」两个词命中 0，读 08 册的在册文件共 4 枚也都不数圈号
（㊵-742 枚举取证）。而 08 是全仓引用的锚点面——一句「㊵-587 已登记」在今天指向两条不同的行。

三枚读数分开，只有一枚是红支：
* `duplicates`——同一个圈号出现在多行：**红**（rc=1）。这是唯一没有解释空间的违例。
* `nonmonotonic_runs`——号码相对行序回落的次数：**只出版**。改号修正本身就会造成回落，
  拿它开红会把「按 owner 裁定重编号」这件事变成违例。
* `rows_counted`——参与计数的行：**两档并列**（㊵-1095 升 v2 的缘由，见下）。注意历史行有**不带日期**的形态
  （`【㊵-587 **09 规划第四版定稿…**】`），所以不能只数「【2026」。

〔㊵-1095 升版本〕首版**只认「【」起手的行**，而 08 册从 2026-10-17 起改用 `### ㊵-N · 日期 · 标题` 的
标题形登记：那 125 行**整档不在门的眼前**，于是四枚重号（1090×2、1091×3 那一族）入库时这台仪器报的是
`rows_counted=594, duplicates={}`＝**绿**。"有门守着"是假的（同 [[guard-must-be-able-to-fail]]）。
v2 把两档一起数，并按"每行第一枚圈号"取号，所以标题正文里引用的历史号仍不算登记。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: 与 `eval_taiji_cap0_baseline.py`／引用在册普查同一条约定（㊵-790、㊵-795）：本仪器既被契约册
#: 按文件路径加载，也被起成真子进程，脚本自己那层目录要先上 `sys.path`，统一落盘助手才解析得到。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

PLAYBOOK = "plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md"
REPORT_FORMAT = "taiji-ledger-marker-uniqueness-v2"
ROW_RE = re.compile("^" + chr(0x3010))
#: 标题形登记（2026-10-17 起 08 册的主形）：`### ㊵-1095 · 2026-10-18 · 标题`。
HEAD_RE = re.compile("^### " + chr(0x32B5) + r"-(\d+)(?:\s|$)")
MARK_RE = re.compile(chr(0x32B5) + r"-(\d+)")


def scan(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").split("\n")
    numbered: list[tuple[int, int]] = []
    faces = {"bracket_rows": 0, "heading_rows": 0}
    for index, line in enumerate(lines, start=1):
        heading = HEAD_RE.match(line)
        if not heading and not ROW_RE.match(line):
            continue
        faces["heading_rows" if heading else "bracket_rows"] += 1
        #: 两档同一条规则：只取**每行第一枚**圈号，标题里追加的"引用他号"注记不算登记。
        match = MARK_RE.search(line)
        if match:
            numbered.append((index, int(match.group(1))))
    if not numbered:
        raise ValueError(
            "零行台账：这台普查仪什么都没数到。先确认读的是哪份册（"
            f"{PLAYBOOK}），零命中不是「圈号已规范化」"
        )
    counts = collections.Counter(number for _, number in numbered)
    duplicates = {str(n): c for n, c in sorted(counts.items()) if c > 1}
    numbers = [number for _, number in numbered]
    nonmonotonic = sum(1 for a, b in zip(numbers, numbers[1:], strict=False) if b < a)
    return {
        "format": REPORT_FORMAT,
        "playbook": PLAYBOOK,
        **faces,
        "rows_counted": len(numbered),
        "distinct_markers": len(counts),
        "max_marker": max(numbers),
        "duplicates": duplicates,
        "duplicate_lines": {
            str(n): [index for index, number in numbered if number == n]
            for n in sorted({int(k) for k in duplicates})
        },
        "nonmonotonic_runs": nonmonotonic,
        "verdicts": {"ledger_markers_unique": not duplicates},
        "reading_limit": (
            "只认每行**第一枚**圈号；行内引用别处历史号（正文里的 ㊵-123）不计入，"
            "所以这台仪器量的是「这一条登记在哪个号下」，不是「这条提到了谁」。"
            "v2 起两档并列计数（`bracket_rows` 中括号形＋`heading_rows` 标题形）；"
            "首版只数中括号形，会把整档标题形登记看成隐形——四枚重号就是这样入库的（㊵-1095）。"
        ),
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="08 册台账圈号唯一性普查（DEBT-G120 修法①）")
    parser.add_argument("--playbook", default=PLAYBOOK)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.playbook)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        report = scan(path)
    except ValueError as exc:
        print(json.dumps({"format": REPORT_FORMAT, "status": "refuse_scan", "detail": str(exc)}))
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(
            json.dumps(
                {
                    "format": REPORT_FORMAT,
                    "status": "refuse_overwrite",
                    "path": str(out),
                }
            )
        )
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)
    print(json.dumps({k: report[k] for k in ("rows_counted", "duplicates", "nonmonotonic_runs")}))
    return 1 if report["duplicates"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

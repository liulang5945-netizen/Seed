"""DEBT-G120 修法① 的普查仪：数 08 册的台账圈号（㊵-N），重号即红。

为什么现在才有这台仪器：全仓对「圈号」「重号」两个词命中 0，读 08 册的在册文件共 4 枚也都不数圈号
（㊵-742 枚举取证）。而 08 是全仓引用的锚点面——一句「㊵-587 已登记」在今天指向两条不同的行。

三枚读数分开，只有一枚是红支：
* `duplicates`——同一个圈号出现在多行：**红**（rc=1）。这是唯一没有解释空间的违例。
* `nonmonotonic_runs`——号码相对行序回落的次数：**只出版**。改号修正本身就会造成回落，
  拿它开红会把「按 owner 裁定重编号」这件事变成违例。
* `rows_counted`——参与计数的行：以「【」开头的行；注意历史行有**不带日期**的形态
  （`【㊵-587 **09 规划第四版定稿…**】`），所以不能只数「【2026」。
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
REPORT_FORMAT = "taiji-ledger-marker-uniqueness-v1"
ROW_RE = re.compile("^" + chr(0x3010))
MARK_RE = re.compile(chr(0x32B5) + r"-(\d+)")


def scan(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").split("\n")
    numbered: list[tuple[int, int]] = []
    for index, line in enumerate(lines, start=1):
        if not ROW_RE.match(line):
            continue
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
        "bracket_rows": sum(1 for line in lines if ROW_RE.match(line)),
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
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)
    print(json.dumps({k: report[k] for k in ("rows_counted", "duplicates", "nonmonotonic_runs")}))
    return 1 if report["duplicates"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""DEBT-G101 修法② 的普查仪：只出版债表的**列形分布**，不替 owner 决定合并方向。

G101 记的是：表头声明四列，而现行行里既有五列的（多出的那列实际承载「状态」但表头没这个名字），
也有四列的，还有行内含额外竖线（把一列切成两列）。**任何按列取值的脚本会在同一张表上拿到两种行。**
这条到今天只有"登记＋一次人工计数"，没有人再数过——而每加一行债，分布就变一次。

本件只做三件事：①数每行的竖线与切分数；②把它和表头声明的列数对照；③出版最大债号，
供"债表现至 DEBT-Gxx"这类句子回查。**它不下"该并成几列"的结论**（那要 owner 裁，
且批量改历史行的列形超出我的半径）。
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
REPORT_FORMAT = "taiji-debt-table-column-shape-v1"
DEBT = "plans/active/roadmap/05_TECH_DEBT_REGISTER.md"
ROW_RE = re.compile(r"^\|\s*DEBT-G(\d+)\s*\|")
ID_RE = re.compile(r"DEBT-G(\d+)")


def measure(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").split("\n")
    rows = [ln for ln in lines if ROW_RE.match(ln)]
    if not rows:
        raise ValueError(
            "零行债：这台普查仪什么都没数到。先确认读的是哪张表（"
            f"{DEBT}），不许把零命中读成「债表已规范化」"
        )
    header = next(
        (ln for ln in lines if ln.startswith("| 编号") or ln.startswith("| DEBT 编号")),
        None,
    )
    if header is None:
        raise ValueError("找不到债表表头行（以 `| 编号` 起）⇒ 无法对照声明列数")
    declared = header.rstrip().count("|") - 1
    split_counts = collections.Counter(ln.count("|") - 1 for ln in rows)
    closed = sum(1 for ln in rows if ln.rstrip().endswith("|"))
    ids = sorted(int(m.group(1)) for m in (ID_RE.search(ln) for ln in rows) if m)
    return {
        "format": REPORT_FORMAT,
        "debt": "DEBT-G101 修法②",
        "table": DEBT,
        "header_declared_cells": declared,
        "rows": len(rows),
        "cells_per_row": {str(k): v for k, v in sorted(split_counts.items())},
        "rows_matching_declaration": split_counts.get(declared, 0),
        "rows_with_more_cells": sum(v for k, v in split_counts.items() if k > declared),
        "rows_with_fewer_cells": sum(v for k, v in split_counts.items() if k < declared),
        "rows_closed_with_pipe": closed,
        "rows_not_closed": len(rows) - closed,
        "max_debt_id": ids[-1],
        "distinct_ids": len(set(ids)),
        "id_duplicates": {str(n): c for n, c in collections.Counter(ids).items() if c > 1},
        "declared_is_the_majority_shape": split_counts.most_common(1)[0][0] == declared,
        "scope": "只出版分布；列形要不要并成一型、往哪边并，归 owner（批量改他人历史行超出实施半径）",
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=DEBT)
    parser.add_argument(
        "--out-report",
        default="reports/taiji_debt_table_column_shape_20261011.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = measure(PROJECT_ROOT / args.table)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" is not cosmetic: without it this audit writes CRLF into a tracked
    # `reports/` artifact on Windows, and no gate in the family notices (DEBT-G115).
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "header_declared_cells": report["header_declared_cells"],
                "cells_per_row": report["cells_per_row"],
                "max_debt_id": report["max_debt_id"],
                "declared_is_the_majority_shape": report["declared_is_the_majority_shape"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

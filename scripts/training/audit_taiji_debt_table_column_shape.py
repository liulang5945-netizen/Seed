r"""DEBT-G101 修法② 的普查仪 ＋ DEBT-G119 修法① 的红支：数债表的列形分布，并在列形不统一时判红。

G101 记的是：表头声明四列，而现行行里既有五列的（多出的那列实际承载「状态」但表头没这个名字），
也有四列的，还有行内含额外竖线（把一列切成两列）。**任何按列取值的脚本会在同一张表上拿到两种行。**
这条到今天只有"登记＋一次人工计数"，没有人再数过——而每加一行债，分布就变一次。
**七裁⑶ 已裁「统一所有历史行到同一列形」**（台账 ㊵-745），所以本件从"只出版分布"升成"守一型"：
目标列数由 `--expect-cells` 给（缺省 5），并加了两把尺——原始竖线数与转义敏感数（转义后的竖线是内容）。

本件做四件事：①数每行的竖线与切分数（两把尺都出版）；②把它和表头声明的列数对照；
③出版最大债号与重号，供"债表现至 DEBT-Gxx"这类句子回查；④合取失败即 rc=1，但**分级开**（㊵-748）：
「编号唯一」「行数＝distinct 数」现在就红；「每行的转义敏感列数都是目标列」「表头声明目标列」在传 `--enforce-shape` 之后才红（债表并成一型之后再常开）。
**它仍然不下"该并成几列"的结论**——那一半已由 owner 裁掉，本件只是执行裁定后的形状守门。
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
#: 真正的列分隔符＝没被反斜杠转义过的竖线；`\|` 是格子里的内容（markdown 亦如此解析）。
SPLIT_RE = re.compile(r"(?<!\\)\|")


def _cells(line: str) -> int:
    """按分隔符数出这一行的列数（转义敏感）。

    行尾省略收尾竖线的行（markdown 允许）不能按"首尾各一段"去算，否则会把最后一格少计一列——
    ㊵-748 第一版就因此把 7 行历史行少算一列，连带让重排器丢了一整格内容。
    """

    parts = SPLIT_RE.split(line)
    return max(0, len(parts) - (2 if line.rstrip().endswith("|") else 1))


def measure(path: Path, *, expected_cells: int = 5) -> dict[str, object]:
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
    #: 转义敏感的那把尺（㊵-748）：`\|` 在单元格里是内容而不是分隔符，markdown 也这样解析。
    #: 两把尺都出版——只看原始竖线数会把"内容里带竖线"与"列数真的多"混成一谈。
    aware_counts = collections.Counter(_cells(ln) for ln in rows)
    #: `原始竖线数 > 转义敏感数` 不是违例，而是"格子里有被正确转义的竖线"的记号（转义本来就是内容）。
    #: 违例只有一种定义：转义敏感列数不等于目标列数——那才是"某一格被竖线切开"。
    escaped_cells = sum(1 for ln in rows if (ln.count("|") - 1) > _cells(ln))
    closed = sum(1 for ln in rows if ln.rstrip().endswith("|"))
    ids = sorted(int(m.group(1)) for m in (ID_RE.search(ln) for ln in rows) if m)
    duplicates = {str(n): c for n, c in collections.Counter(ids).items() if c > 1}
    not_expected = sum(v for k, v in aware_counts.items() if k != expected_cells)
    return {
        "format": REPORT_FORMAT,
        "debt": "DEBT-G101 修法②／DEBT-G119 修法①",
        "table": DEBT,
        "header_declared_cells": declared,
        "expected_cells": int(expected_cells),
        "rows": len(rows),
        "cells_per_row": {str(k): v for k, v in sorted(split_counts.items())},
        "cells_per_row_escape_aware": {str(k): v for k, v in sorted(aware_counts.items())},
        "rows_with_escaped_pipes_in_cells": escaped_cells,
        "rows_matching_declaration": split_counts.get(declared, 0),
        "rows_with_more_cells": sum(v for k, v in split_counts.items() if k > declared),
        "rows_with_fewer_cells": sum(v for k, v in split_counts.items() if k < declared),
        "rows_closed_with_pipe": closed,
        "rows_not_closed": len(rows) - closed,
        "max_debt_id": ids[-1],
        "distinct_ids": len(set(ids)),
        "id_duplicates": duplicates,
        "declared_is_the_majority_shape": split_counts.most_common(1)[0][0] == declared,
        "verdicts": {
            "ids_unique": not duplicates,
            "row_count_matches_distinct_ids": len(set(ids)) == len(rows),
            "all_rows_are_the_expected_shape": not_expected == 0,
            "header_declares_the_expected_shape": declared == expected_cells,
        },
        "rows_not_expected_cells": not_expected,
        "scope": (
            "七裁⑶ 已裁「统一所有历史行到同一列形」，本件因此从「只出版分布」升成「守一型」——"
            "但红是**分级开**的：编号唯一性与行数＝distinct 现在就守；形状与表头那两条要等债表真的"
            "并成一型（㊵-748 实测仍有 41 行旧四列形与 3 行需人工归列）才用 --enforce-shape 打开。"
            "转义敏感列数才是判据面：格子里带被转义的竖线是内容，不是违例。"
        ),
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=DEBT)
    parser.add_argument(
        "--out-report",
        default="reports/taiji_debt_table_column_shape_20261011.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expect-cells",
        type=int,
        default=5,
        help="统一列形的目标列数（七裁⑶ 定的是 5）；负数或零响亮拒绝",
    )
    parser.add_argument(
        "--enforce-shape",
        action="store_true",
        help="把「每行都是目标列」与「表头声明目标列」也变成失败条件（债表并成一型之后才该开）",
    )
    args = parser.parse_args(argv)
    if int(args.expect_cells) <= 0:
        print("REJECT expect_cells must be positive", file=sys.stderr)
        return 2
    try:
        report = measure(PROJECT_ROOT / args.table, expected_cells=int(args.expect_cells))
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
    #: DEBT-G119 的红支：这台仪器以前第 108 行是无条件 `return 0`，
    #: 于是"我加行时造出的重复编号"只躺在读数里。现在三条合取任一为假就 rc=1，
    #: 而**读数件照样先落盘**——红也要留证据，不许红了就什么都不出版。
    #: 分级红：编号类两条现在就守（它们今天为真，破了就是有人加错了行）；
    #: 形状/表头那两条要等债表真的并成一型才开启（㊵-748 实测：现在还有 41 行是旧四列形、
    #: 3 行的格子里有裸竖线需要人工归列），否则这道门会把每一批提交都堵死。
    enforced = {"ids_unique", "row_count_matches_distinct_ids"}
    if args.enforce_shape:
        enforced |= {"all_rows_are_the_expected_shape", "header_declares_the_expected_shape"}
    failing = sorted(
        name for name, ok in report["verdicts"].items() if ok is not True and name in enforced
    )
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "header_declared_cells": report["header_declared_cells"],
                "cells_per_row": report["cells_per_row"],
                "cells_per_row_escape_aware": report["cells_per_row_escape_aware"],
                "rows_not_expected_cells": report["rows_not_expected_cells"],
                "rows_with_escaped_pipes_in_cells": report["rows_with_escaped_pipes_in_cells"],
                "max_debt_id": report["max_debt_id"],
                "declared_is_the_majority_shape": report["declared_is_the_majority_shape"],
                "failing_verdicts": failing,
            },
            ensure_ascii=True,
        )
    )
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""预注册"出口句含糊用词"扫描门（DEBT-G46 修法③的落地件；只读，不改任何判据文本）。

为什么要有这台仪器（缺陷本体）：PLAN-N1-01 的三出口写成"J-S1a 过 ⇒ …；不过 ⇒ …；
**不过但 F1/F3 大幅改善** ⇒ 登记'读取可修但不充分'"，而"大幅改善"没有冻结数值 ⇒ 实测落进
含糊区时**两个出口都说得通**，判据本身无法裁决（㊵-485⑤a 已为此登记 DEBT-G46；
㊵-487⑦ 又抓到第二类实例"恰好跌破 1 项"无定义）。

三条硬规矩：
* **只报不修**：判据文本一经冻结不追改，所以默认**不**扫全库——扫谁由命令行点名（新预注册落地前自扫）；
* **能为 false 两面都测**：已知含糊件必须命中，合成合规件必须零命中；
* **没有判据段就响亮拒绝**（rc=2）：一份没有可机检判据的预注册，"零命中"不等于"干净"。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 含糊出口词——来自 DEBT-G46 修法③点名的四个。
VAGUE_WORDS = ("明显", "大幅", "显著", "足够")

#: 一条出口句要算"被数值钉住"的判据：同行出现比较算符且紧邻数字。
NUMERIC_PIN = re.compile(r"(?:[≥≤><]=?\s*-?\d|[-+]?\d(?:\.\d+)?\s*(?:[%％]?)\s*(?:对|倍|＞|>))")

#: 判据/出口段的行首标记（行内含这些词才算判据句，别拿散文段当判据扫）。
CRITERION_MARKERS = ("出口", "判据", "J-", "J－")

#: DEBT-G54 修法②的守卫：一份 `PLAN-*` **必须**至少有一节标题含下列任一词，否则"零命中"只是因为
#: 作者换了叫法（"约定/口径/处置"），门对整份件隐形 ⇒ 把"隐形"从静默变成 rc=2 的响亮失败。
#: 只认**二级及以下**标题：文档大标题里顺带出现"判据"两个字不算一节判据段
#: （写这条的测时我自己就撞过一次——件名里写着"不肯说判据"就把标记式扫描糊过去了）。
CRITERION_HEADING_WORDS = ("判据", "出口", "验收")
HEADING_LINE = re.compile(r"^#{2,6} \s*(.*)$")


def criterion_headings(text: str) -> list[int]:
    """二级及以下标题里含判据/出口/验收的行号；空表 ⇒ 这份预注册对现行扫描面不可见。"""

    return [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if (m := HEADING_LINE.match(line.strip()))
        and any(word in m.group(1) for word in CRITERION_HEADING_WORDS)
    ]


def scan_text(text: str) -> dict[str, Any]:
    """返回逐行命中表；`criterion_lines` 为 0 时调用方必须拒判。"""

    lines = text.splitlines()
    criterion_lines = [
        (number, line)
        for number, line in enumerate(lines, start=1)
        if any(marker in line for marker in CRITERION_MARKERS)
    ]
    hits: list[dict[str, Any]] = []
    for number, line in criterion_lines:
        found = [word for word in VAGUE_WORDS if word in line]
        if found and not NUMERIC_PIN.search(line):
            hits.append(
                {
                    "line": number,
                    "words": found,
                    "excerpt": line.strip()[:160],
                }
            )
    return {
        "criterion_lines": len(criterion_lines),
        "vague_word_lines_unpinned": len(hits),
        "hits": hits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预注册出口句含糊用词扫描（DEBT-G46 修法③）")
    parser.add_argument("--doc", action="append", required=True, help="要扫的预注册件（可重复）")
    parser.add_argument("--out-report", default=None, help="落盘路径；不给只打到 stdout")
    parser.add_argument(
        "--min-criterion-lines",
        type=int,
        default=0,
        help="DEBT-G54 修法③：判据行的**期望下限**，扫到的行数低于它就单独出一档 rc=3"
        "（`criterion_surface_too_thin`）。默认 0＝不分档（向后兼容既有调用）。",
    )
    args = parser.parse_args(argv)

    results: list[dict[str, Any]] = []
    #: rc 分档：2＝这份件不可扫（缺件/无判据段/判据面隐形），1＝扫到含糊用词，
    #: 3＝扫得到但覆盖面薄（本该扫到 200 行却只扫到 2 行那种），0＝干净。
    #: 多份件时按"严重度秩"取最大，**不许后一份把前一份的失败洗成 0**。
    rank = {0: 0, 1: 1, 3: 2, 2: 3}
    rc = 0

    def worse(current: int, new: int) -> int:
        return new if rank[new] > rank[current] else current

    for raw in args.doc:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        entry: dict[str, Any] = {"doc": str(path), "status": "ok"}
        entry["criterion_lines_floor"] = int(args.min_criterion_lines)
        if not path.is_file():
            #: 缺件不是"零命中"——点名缺失并把 rc 定成响亮拒绝。
            entry["status"] = "missing_doc"
            rc = worse(rc, 2)
            results.append(entry)
            continue
        raw_text = path.read_text(encoding="utf-8")
        scan = scan_text(raw_text)
        entry.update(scan)
        if path.name.startswith("PLAN-"):
            #: 存在性检查走**标题**，不走"某行有没有写过判据这三个字"。
            headings = criterion_headings(raw_text)
            entry["criterion_headings"] = len(headings)
            if not headings:
                entry["status"] = "invisible_criterion_surface"
                rc = worse(rc, 2)
                results.append(entry)
                continue
        if scan["criterion_lines"] == 0:
            entry["status"] = "no_criterion_section"
            rc = worse(rc, 2)
        elif scan["vague_word_lines_unpinned"]:
            entry["status"] = "ambiguous_exit_wording"
            rc = worse(rc, 1)
        elif int(args.min_criterion_lines) > 0 and scan["criterion_lines"] < int(
            args.min_criterion_lines
        ):
            entry["status"] = "criterion_surface_too_thin"
            rc = worse(rc, 3)
        results.append(entry)

    payload = {
        "format": "taiji-prereg-exit-wording-audit-v1",
        "debt": "DEBT-G46 修法③",
        "vague_words": list(VAGUE_WORDS),
        "results": results,
        "rc": rc,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_report:
        target = Path(args.out_report)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8", newline="\n")
    #: Windows 控制台是 GBK：正文含中文，直接把中文打到 stdout 会在判完之后
    #: 抛 UnicodeEncodeError，把 rc 伪装成崩溃（㊵-484⑤ 同族自伤）⇒ 只打 ASCII 转义。
    print(text.encode("unicode_escape").decode("ascii"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

r"""#79（七裁⑶）的批量重排器：把债表每一行统一成同一列形，缺的列只留空标记、不代填内容。

owner 裁的是「统一所有历史行到同一列形」。这张表今天的真实分布是 122 行里五列 80、四列 41、三列 1，
另有若干行的格子里带**未转义**的竖线（那是 `Tests 1 failed | 487 passed` 这类命令输出，
markdown 会把它们当列分隔符，所以那几行**在渲染上已经断了**）。

规矩（按顺序对每一行 `| DEBT-Gnn | … |` 施加，默认只干跑，带 --apply 才写）：
1. 列数多于目标 ⇒ 只转义**落在反引号码段内**的竖线（转义后的竖线在格子里是内容不是分隔符）；
   转义完仍多于目标列 ⇒ 整件拒写。我不猜列边界，也不把任何人的文字挪到别的列里。
2. 正好少一列 ⇒ 末尾追加一枚「状态未登记」标记格；少两列 ⇒ 追加「修法未登记」＋「状态未登记」。
   标记里写明是哪一格的裁定、且不代填内容——留空是事实，填空是我编。
3. 一行只有在「原有各格仍在原位、去转义后逐字节相等」时才被接受，否则整件拒写。

本文件的源码里刻意不手写反斜杠字面量：这一层的传输会把两个反斜杠收成一个，
一个错误的分隔符正则会让整份分析说谎（㊵-748 实测撞过一次）。转义串一律由 chr(92) 显式拼出来。
"""

import argparse
import json
import re
from pathlib import Path

B = chr(92)
ESCAPED_PIPE = B + "|"
SPLIT_RE = re.compile("(?<!" + B + B + ")" + B + "|")
ROW_RE = re.compile("^[|]" + B + "s*DEBT-G([0-9]+)" + B + "s*[|]")
TARGET_CELLS = 5
STATUS_MARKER = "状态未登记（原文按四列登记，㊵-748 统一列形时留空不代填）"
FIX_MARKER = "修法未登记（原文只有三列，㊵-748 统一列形时留空不代填）"
NEW_HEADER = "| 编号 | 缺陷或事项 | 影响 | 修法或处置约束 | 状态 |"
NEW_SEPARATOR = "|---|---|---|---|---|"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEBT = "plans/active/roadmap/05_TECH_DEBT_REGISTER.md"


def merge_last_cell_into_previous(line: str) -> str:
    """把"最后一格"并入倒数第二格：只把它前面那枚分隔符转义成内容，别的一个字节不动。"""

    positions = [
        position
        for position, ch in enumerate(line)
        if ch == "|" and (position == 0 or line[position - 1] != B)
    ]
    if len(positions) < 3:
        return line
    target = positions[-2]
    return line[:target] + ESCAPED_PIPE + line[target + 1 :]


def cells_of(line: str) -> list[str]:
    """按分隔符切格；行尾没有竖线的行（markdown 允许省略收尾竖线）不能把最后一格切掉。

    ㊵-748 的第一版就是在这里用 `[1:-1]` 无条件丢掉最后一段，于是 7 行"收尾竖线省略了"的历史行
    被重排时整格内容消失（DEBT-G22 少了 341 个字符）。现在：行尾有竖线才丢尾部空段。
    """

    parts = SPLIT_RE.split(line)
    if line.rstrip().endswith("|"):
        return parts[1:-1]
    return parts[1:]


def code_span_flags(text: str) -> list[bool]:
    inside = False
    flags: list[bool] = []
    for ch in text:
        if ch == "`":
            inside = not inside
            flags.append(False)
            continue
        flags.append(inside)
    return flags


def escape_pipes_inside_code_spans(line: str) -> str:
    flags = code_span_flags(line)
    pieces = []
    for position, ch in enumerate(line):
        if ch == "|" and flags[position] and 0 < position < len(line) - 1:
            pieces.append(ESCAPED_PIPE)
            continue
        pieces.append(ch)
    return "".join(pieces)


def transform(lines: list[str]) -> tuple[list[str], dict[str, int], list[str]]:
    out = list(lines)
    stats = {
        "rows": 0,
        "kept_five": 0,
        "escaped": 0,
        "appended_status": 0,
        "appended_two": 0,
        "closed_missing_tail": 0,
        "merged_trailing_note": 0,
    }
    problems: list[str] = []
    for index, line in enumerate(lines):
        if not ROW_RE.match(line):
            continue
        stats["rows"] += 1
        work = line
        if not work.rstrip().endswith("|"):
            #: 先把省略掉的收尾竖线补回来（纯格式，不动任何字节顺序），否则下面所有判断都建立在错切的格子上。
            work = work.rstrip() + " |"
            stats["closed_missing_tail"] += 1
        if len(cells_of(work)) > TARGET_CELLS:
            candidate = escape_pipes_inside_code_spans(work)
            if len(cells_of(candidate)) <= TARGET_CELLS:
                work = candidate
                stats["escaped"] += 1
            #: 转义码段内竖线还不够的那几行不在这里判死——交给下面的"末格并入"规则；
            #: 两条都救不动时才由下面的 else 报问题并拒写整件（宁可整件不动，也不猜列边界）。
        original = cells_of(work)
        if len(original) == TARGET_CELLS + 1 and original[-1].lstrip().startswith(("【", "**")):
            #: ㊵-748 逐行点验过的五行（G41／G42／G43／G105／G106）：作者把后来的带日期追加注
            #: 用一枚裸竖线接在行尾，于是渲染出第六格。那一格在语义上是「状态」列的延续，
            #: 所以这里只做一件事：把它前面那枚竖线转义掉（字节级插入一个反斜杠，别的一个字节都不动）。
            work = merge_last_cell_into_previous(work)
            stats["merged_trailing_note"] += 1
            original = cells_of(work)
        if len(original) == TARGET_CELLS:
            stats["kept_five"] += 1
            out[index] = work
            continue
        if len(original) == TARGET_CELLS - 1:
            added = [STATUS_MARKER]
            stats["appended_status"] += 1
        elif len(original) == TARGET_CELLS - 2:
            added = [FIX_MARKER, STATUS_MARKER]
            stats["appended_two"] += 1
        else:
            problems.append(f"line {index + 1}: {len(original)} 列，超出规则覆盖")
            continue
        #: 关键：只做**行尾追加**，绝不重新排版整行。第一版在这里用
        #: `rebuild_row(cells + added)` 按解析结果重铺整行，结果把 DEBT-G22 的一整格弄丢了
        #: （919 字符变 578，「已修与实测」那格消失）——解析器看不懂的转义组合会在重铺时被当成分隔符吃掉。
        #: 行尾追加是纯字节插入，不经过解析，因此不可能弄丢它没读过的东西。
        if not work.rstrip().endswith("|"):
            problems.append(f"line {index + 1}: 行尾不是竖线，无法安全追加格子")
            continue
        head = work.rstrip()[:-1].rstrip()
        merged = head + "".join(f" | {marker}" for marker in added) + " |"
        check = cells_of(merged)
        if len(check) != TARGET_CELLS:
            problems.append(f"line {index + 1}: 追加后 {len(check)} 列")
            continue
        #: 无损性由构造保证：新行 = 原行去掉行尾竖线与尾随空格后的那段字节 ＋ 只有标记的尾巴。
        #: 所以这里只查三件事：前缀确实是原行（不是重排出来的）、追加的部分只可能是标记、列数到位。
        if not work.rstrip().startswith(head) or merged[: len(head)] != head:
            problems.append(f"line {index + 1}: 追加后的前缀不再是原行的字节")
            continue
        if (
            merged[len(head) :].replace(" | " + STATUS_MARKER, "").replace(" | " + FIX_MARKER, "")
            != " |"
        ):
            problems.append(f"line {index + 1}: 尾巴里出现了标记以外的文字")
            continue
        out[index] = merged
    return out, stats, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", default=DEBT)
    parser.add_argument("--apply", action="store_true", help="缺省只干跑（不写盘）")
    parser.add_argument("--out-report", default=None, help="可选：把统计与问题清单落成 JSON")
    args = parser.parse_args(argv)
    path = PROJECT_ROOT / args.table
    lines = path.read_text(encoding="utf-8").split("\n")
    header_index = next((i for i, l in enumerate(lines) if l.startswith("| 编号")), None)
    if header_index is None:
        raise ValueError("找不到债表表头行（以 `| 编号` 起）⇒ 无法对照声明列数")
    if not lines[header_index + 1].startswith("|-"):
        raise ValueError("表头下一行不是分隔行，拒绝猜测表格边界")

    out, stats, problems = transform(lines)
    collateral = [
        i + 1
        for i, (before, after) in enumerate(zip(lines, out, strict=False))
        if not ROW_RE.match(before) and before != after
    ]
    payload = {
        "table": args.table,
        "target_cells": TARGET_CELLS,
        "stats": stats,
        "problems": problems,
        "collateral_non_row_lines_changed": collateral,
        "writable": not problems and not collateral and len(out) == len(lines),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    if args.out_report:
        report = PROJECT_ROOT / args.out_report
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    if not payload["writable"]:
        return 2
    if not args.apply:
        return 0
    out[header_index] = NEW_HEADER
    out[header_index + 1] = NEW_SEPARATOR
    path.write_bytes("\n".join(out).encode("utf-8"))
    raw = path.read_bytes()
    print("WROTE", args.table, "bytes", len(raw), "CR", raw.count(b"\r"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

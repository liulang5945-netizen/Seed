r"""DEBT-G125 修法①：把 ㊵-751 用过的债表字节守恒判据做成入库仪器（自带三档负对照的契约册见
`tests/taiji_native/test_n6_08_debt_table_bytes_verifier_contract.py`）。

背景：`unify_taiji_debt_table_columns.py` 的第一版把整行按解析结果重铺，实测吃掉 `DEBT-G22`
一整格（919 字符变 578，「已修与实测」那格消失），而**工具自己的断言当时是错的却全绿**——
那次是靠一份临时写在仓外的独立复核抓到的。本件就是那份复核的入库版：判定权属于读数，不属于台账文本。

判据形状（逐行、纯字节、不经任何切格解析器）：
1. 旧行的每一个字符都必须**按顺序**出现在新行里（先把旧行行尾空白剥掉，那不算内容）；
2. 新行相对旧行只允许两种多出——在一枚 `|` 之前插入一枚反斜杠（转义），以及旧行读完之后的尾巴；
3. 尾巴只许由空白、竖线和那两条已登记的留空标记组成；
4. 非 DEBT 行只允许表头与分隔行被整体替换。

一件重要的否定式：这里**不做整体反转义**。HEAD 里本来就有些格子写着 `\\|`，
一把反转义会把作者原有的转义误报成"仪器改了字节"（2026-10-12 第一次复核就在 `DEBT-G22` 上这样误报过）。

默认落点：本件**没有默认写靶**，`--out-report` 不给就只印 stdout（DEBT-G35／G50 那一族的教训）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = "plans/active/roadmap/05_TECH_DEBT_REGISTER.md"
#: 本文件的源码里刻意不手写反斜杠字面量：分隔符正则错一枚就会让整份分析说谎（㊵-748 撞过）。
B = chr(92)
ROW_RE = re.compile("^[|]" + B + "s*DEBT-G([0-9]+)" + B + "s*[|]")
#: 这两条文字与 `unify_taiji_debt_table_columns.py` 里的标记同源；改一边必须改另一边。
STATUS_MARKER = "状态未登记（原文按四列登记，㊵-748 统一列形时留空不代填）"
FIX_MARKER = "修法未登记（原文只有三列，㊵-748 统一列形时留空不代填）"
MARKERS = (FIX_MARKER, STATUS_MARKER)
FORMAT = "taiji-debt-table-bytes-verification-v2"


def show(text: str) -> str:
    """把不可编码字形降级成转义形式：拒绝消息要走 stdout，GBK 控制台上抛错会掩盖 rc（DEBT-G81）。"""

    return text.encode("ascii", "backslashreplace").decode("ascii")


def align(old: str, new: str) -> tuple[int, str]:
    """返回 (旧行读完时新行的位置, 失败原因)；原因为空串表示这一行的旧字节按序在场。"""

    source = old.rstrip()
    i = j = 0
    while i < len(source):
        if j >= len(new):
            return -1, f"new ran out with {len(source) - i} bytes of old unmatched"
        if source[i] == new[j]:
            i += 1
            j += 1
            continue
        if new[j] == B and source[i] == "|" and j + 1 < len(new) and new[j + 1] == "|":
            j += 1
            continue
        return (
            -1,
            f"first diff old[{i}]={show(source[i : i + 24])!r} new[{j}]={show(new[j : j + 24])!r}",
        )
    return j, ""


def tail_is_only_markers(tail: str) -> bool:
    rest = tail
    while True:
        matched = False
        for marker in MARKERS:
            if marker in rest:
                rest = rest.replace(marker, "", 1)
                matched = True
                break
        if not matched:
            break
    return re.fullmatch(r"[\s|]*", rest) is not None


def compare(old_text: str, new_text: str, *, table: str, strict: bool = False) -> dict[str, object]:
    """按**债号**对齐，而不是按行号对齐。

    这条修正是这台仪器自己撞出来的：它首跑的对象是"HEAD 与工作树"，而那一轮之间正常新增了
    两行债（DEBT-G124／G125）——按行号 zip 会让其后每一行都错位，195 条假红，
    于是这道门会把**每一次合法加行**都堵死（与 `--enforce-shape` 当初不能早开是同一族理由）。
    现在的语义是"重写之前已存在的每一行，字节是否按序在场"：加行是 `rows_added`，
    删行才是红（那是真丢数据），行号移动不影响判定。
    """

    old_lines = old_text.split("\n")
    new_lines = new_text.split("\n")
    #: 按类别分桶记问题，而不是把消息拼成一个列表再去匹配关键词——判定若靠"问题文本里有没有某枚词"，
    #: 换个措辞就会让整条判据隐形（同一族教训见 DEBT-G81③）。
    byte_problems: list[str] = []
    tail_problems: list[str] = []
    collateral_problems: list[str] = []
    lost_ids: list[str] = []
    old_rows: dict[str, str] = {}
    new_rows: dict[str, str] = {}
    for line in old_lines:
        match = ROW_RE.match(line)
        if match is not None:
            old_rows.setdefault(match.group(1), line)
    for line in new_lines:
        match = ROW_RE.match(line)
        if match is not None:
            new_rows.setdefault(match.group(1), line)
    matched = rewritten = appended = closed = escaped_only = unchanged = 0
    for identifier, old in old_rows.items():
        new = new_rows.get(identifier)
        if new is None:
            lost_ids.append(identifier)
            continue
        matched += 1
        if old == new:
            unchanged += 1
            continue
        rewritten += 1
        if not strict:
            #: 缺档只禁"删除与替换"——数据丢失的定义就是这个（㊵-747 那次 919 字符变 578）。
            #: 纯插入必须放过：本仓的常规动作就是往状态格里**就地追加一条带日期的注**，
            #: 那既不是丢字节，也不该被一台守门仪器堵死。
            source = old.rstrip()
            target = new.rstrip()
            cursor = 0
            for ch in source:
                found = target.find(ch, cursor)
                if found < 0:
                    byte_problems.append(
                        f"DEBT-G{identifier}: old bytes are not a subsequence of the new row "
                        f"(stalled at new offset {cursor}) -> {show(source[:40])!r}"
                    )
                    break
                cursor = found + 1
            continue
        position, why = align(old, new)
        if position < 0:
            byte_problems.append(f"DEBT-G{identifier}: {why}")
            continue
        tail = new[position:]
        if not tail_is_only_markers(tail):
            tail_problems.append(
                f"DEBT-G{identifier}: tail holds text beyond the markers -> {show(tail[:80])!r}"
            )
            continue
        if any(marker in tail for marker in MARKERS):
            appended += 1
        elif "|" in tail:
            closed += 1
        else:
            escaped_only += 1

    #: 非表格行：允许"插入"（新加一段说明），不允许"删除或改写"（那是别人的字被吞）。
    #: v2（DEBT-G135 修法②）：比较前先各自剥掉**尾随空行**——那是写盘工具的产物不是作者的字，
    #: 但剥掉的枚数照样出版，并用一条**只朝累积方向为假**的判据看着它（清理是另一笔提交，不算违规）。
    def _tail_blanks(seq):
        count = 0
        while count < len(seq) and seq[len(seq) - 1 - count].strip() == "":
            count += 1
        return count

    old_blanks = _tail_blanks(old_lines)
    new_blanks = _tail_blanks(new_lines)
    old_prose = [
        line for line in old_lines[: len(old_lines) - old_blanks] if ROW_RE.match(line) is None
    ]
    new_prose = [
        line for line in new_lines[: len(new_lines) - new_blanks] if ROW_RE.match(line) is None
    ]
    cursor = 0
    for line in old_prose:
        while cursor < len(new_prose) and new_prose[cursor] != line:
            cursor += 1
        if cursor == len(new_prose):
            collateral_problems.append(f"prose line missing in the new face -> {show(line[:70])!r}")
            break
        cursor += 1
    rows_added = len(set(new_rows) - set(old_rows))
    if not old_rows:
        raise ValueError(
            f"zero rows in {table}: this instrument measured nothing; "
            "do not read a zero hit as 'the register is normalized'"
        )
    return {
        "format": FORMAT,
        "table": table,
        #: 两档语义必须写进读数件：缺档 `no_deletion` 只禁删除与替换（可当每批的守门），
        #: `strict_rewrite` 才是"只许转义插入＋尾巴留空标记追加"（验证并形器那一次重写用）。
        "profile": "strict_rewrite" if strict else "no_deletion",
        "rows": matched,
        "rows_added": rows_added,
        "rows_lost": len(lost_ids),
        "lost_debt_ids": lost_ids,
        "rows_rewritten": rewritten,
        "rows_unchanged": unchanged,
        "lines_old": len(old_lines),
        "lines_new": len(new_lines),
        "rewrite_kinds": {
            "appended_marker": appended,
            "closed_tail": closed,
            "escape_only": escaped_only,
        },
        "allowed_markers": list(MARKERS),
        "trailing_blank_lines": {"old": old_blanks, "new": new_blanks},
        "problems": byte_problems + tail_problems + collateral_problems,
        "verdicts": {
            "no_rows_lost": not lost_ids,
            "all_row_bytes_preserved_in_order": not byte_problems,
            "tails_are_only_registered_markers": not tail_problems,
            "non_row_lines_untouched": not collateral_problems,
            #: 尾随空行**变多**才红（那是写盘幂等缺失的指纹）；变少＝清理，属另一笔提交。
            "file_tail_not_grown": new_blanks <= old_blanks,
        },
    }


def load_old(*, table: str, old_ref: str, old_file: str | None) -> str:
    if old_file is not None:
        return Path(old_file).read_bytes().decode("utf-8")
    proc = subprocess.run(
        ["git", "show", f"{old_ref}:{table}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    )
    return proc.stdout.decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    #: DEBT-G81 修法①：本件的拒绝消息走 stdout，未守卫时 GBK 会把 rc=1 的形状盖成 traceback。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Verify a markdown table rewrite preserved every original byte in order."
    )
    parser.add_argument(
        "--table", default=DEFAULT_TABLE, help="register path, relative to the repo"
    )
    parser.add_argument(
        "--new", default=None, help="rewritten file; defaults to the working tree copy"
    )
    parser.add_argument("--old-ref", default="HEAD", help="git ref to compare against")
    parser.add_argument(
        "--old-file", default=None, help="explicit old file (used by the contract book)"
    )
    parser.add_argument("--out-report", default=None, help="optional; stdout only when omitted")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict-rewrite",
        action="store_true",
        help="strict profile: only backslash insertions before a pipe and registered marker "
        "tails are allowed; use it to verify one unifier rewrite. Default profile only "
        "forbids deleted or substituted bytes.",
    )
    args = parser.parse_args(argv)

    new_path = Path(args.new) if args.new else PROJECT_ROOT / args.table
    try:
        old_text = load_old(table=args.table, old_ref=args.old_ref, old_file=args.old_file)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(show(f"cannot read the old face: {exc}"), file=sys.stderr)
        return 2
    try:
        report = compare(
            old_text,
            new_path.read_bytes().decode("utf-8"),
            table=args.table,
            strict=args.strict_rewrite,
        )
    except ValueError as exc:
        print(show(str(exc)), file=sys.stderr)
        return 2

    if args.out_report:
        out = Path(args.out_report)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        if out.exists() and not args.overwrite:
            print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    failing = sorted(name for name, ok in report["verdicts"].items() if ok is not True)
    print(
        json.dumps(
            {
                "profile": report["profile"],
                "rows": report["rows"],
                "rows_added": report["rows_added"],
                "rows_lost": report["rows_lost"],
                "rows_rewritten": report["rows_rewritten"],
                "rows_unchanged": report["rows_unchanged"],
                "lines_old": report["lines_old"],
                "lines_new": report["lines_new"],
                "rewrite_kinds": report["rewrite_kinds"],
                "problem_count": len(report["problems"]),
                "failing_verdicts": failing,
            },
            ensure_ascii=True,
        )
    )
    for problem in report["problems"][:12]:
        print("  PROBLEM", show(problem))
    return 1 if failing or report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

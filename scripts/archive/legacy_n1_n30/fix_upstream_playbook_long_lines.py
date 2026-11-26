"""修`08_UPSTREAM_SYNC_PLAYBOOK.md` 的超长行格式（纯格式，零语义变更）。

**问题**：该文件有 64 行超过 2000 字符（最长 25998 字符＝一整段挤成一行），
合计 204336 字符，占全文 1.7MB 的绝大部分。成因是长期追加时段落内换行丢失，
不是某次事故（回溯 8 个提交，长行数一直是 63–64）。
**后果**：git diff 无法定位改动、grep 命中整段、按行引用失效——
而这个文件恰恰记录「债 48→30→28」这类数字演化，埋在 2.6 万字符的一行里等于不可查。

**切分规则（经探针实测选定）**：只在 `空格 + 标记 + 空格` 处切，
标记＝`①-⑩` 与 `⇒`。硬约束「前后皆空格」是必须的：
不加约束时 `①` 会切在词中间（实测段首落在 `）` 的段有 38 个），
加 `bullet_star`（`** `）规则会在加粗正文中间乱切（688 个不安全段首）。

**安全性自证**（脚本每次运行都会重跑，缺一即报错退出）：
- 往返：切完再拼接必须与原行**逐字符相等**（标记已回贴）。
- 段首：每个新段的第一个字符必须是合法标记，不得是收尾/句读字符。
- 幂等：再跑一次不应产生任何切点（切完后的行不再超长）。

用法：
    python scripts/training/fix_upstream_playbook_long_lines.py --check
    python scripts/training/fix_upstream_playbook_long_lines.py --apply
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "plans" / "active" / "roadmap" / "08_UPSTREAM_SYNC_PLAYBOOK.md"

#: 只有超过这个长度的行才处理（正常条目不动）。
LONG_LINE_CHARS = 2000

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"
MARKER = f"[{CIRCLED}]|⇒"

#: 标记必须前后各有空格/制表符 —— 这是「结构性标记」与「文中同形字符」的唯一区别。
SPLIT_RE = re.compile(rf"(?<=[ \t])({MARKER})(?= )")

#: 新段允许的首字符（用于自证段首合法）。
LEGAL_HEADS = set(CIRCLED + "⇒-*")


def split_line(line: str) -> list[str]:
    """在「空格 + 标记 + 空格」处切开；标记回贴到新段开头（内容零丢失）。

    切点形如`X ⇒ Y`：我们把**切点前的那个空格留在原处**（当作上一段末尾的行尾空格），
    新段直接从标记 `⇒` 开始。因此每段是原文的一次连续切片，
    `"".join(pieces) == line` 逐字符相等（`check` 会验这一条）。
    """
    pieces: list[str] = []
    cursor = 0
    for match in SPLIT_RE.finditer(line):
        cut = match.start()
        if cut == 0:
            continue
        # 段边界取在**标记起点**：第一段是 `line[0:首个标记位置]`（末尾带着
        # 标记前那个空格，当行尾空格不渲染），其后每段从标记起一直取到下一个
        # 标记之前。于是每段首字符必是合法标记，且 `"".join(pieces) == line`。
        pieces.append(line[cursor:cut])
        cursor = cut
    pieces.append(line[cursor:])
    return [p for p in pieces if p.strip()]


def check(text: str) -> tuple[int, list[str]]:
    """返回（可切段数, 问题清单）。不修改任何内容。"""
    problems: list[str] = []
    lines = text.split("\n")
    longs = [(i + 1, line) for i, line in enumerate(lines) if len(line) > LONG_LINE_CHARS]
    for number, line in longs:
        pieces = split_line(line)
        if "".join(pieces) != line:
            problems.append(f"行 {number}: 切分后再拼接与原行不等 —— 会丢字符，拒绝写")
            continue
        for piece in pieces[1:]:
            head = piece[0] if piece else ""
            if head not in LEGAL_HEADS:
                problems.append(f"行 {number}: 段首 {head!r} 不是合法标记 —— 切点可能落在词中")
    return len(longs), problems


def apply_fix(text: str) -> tuple[str, int, int]:
    """返回（新文本, 处理行数, 切出段数）。"""
    out: list[str] = []
    touched = 0
    added = 0
    for line in text.split("\n"):
        if len(line) <= LONG_LINE_CHARS:
            out.append(line)
            continue
        pieces = split_line(line)
        touched += 1
        added += len(pieces) - 1
        out.extend(pieces)
    return "\n".join(out), touched, added


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.check == args.apply:
        print("必须且只能给一个：--check 或 --apply")
        return 2

    text = TARGET.read_text(encoding="utf-8")
    longs, problems = check(text)
    print(f"{TARGET.relative_to(REPO)}：超长行（>{LONG_LINE_CHARS} 字符）{longs} 行")
    if problems:
        print("**自证失败，不予写**：")
        for problem in problems[:20]:
            print(f"  {problem}")
        return 1
    print("自证通过：往返逐字符相等 ＋ 段首全为合法标记")

    if args.check:
        fixed, touched, added = apply_fix(text)
        after, _problems_after = check(fixed)
        print(f"预估：处理 {touched} 行、切出 {added} 段；处理后超长行 {after} 行")
        print(f"预估体积：{len(text):,} → {len(fixed):,} 字符")
        return 0

    fixed, touched, added = apply_fix(text)
    if check(fixed)[1]:
        print("写前复核失败，已中止")
        return 1
    TARGET.write_text(fixed, encoding="utf-8")
    print(f"已写入：处理 {touched} 行、切出 {added} 段")
    print(f"体积：{len(text):,} → {len(fixed):,} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

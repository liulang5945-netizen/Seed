"""仪器自己的用法屏必须能打出来（2026-10-04 实测这条会崩）。

来历：本会话给下一次停靠写交接时立了一条规矩——"写进交接的命令至少空手跑过一次 `--help`"。
照这条规矩跑 `probe_taiji_a30_stop_failure.py --help` 时当场炸：
`ValueError: unsupported format character '？' (0xff09)`——argparse 会对 `help=` 做 `%` 插值，
而某条帮助文字里写了一个字面 `%`（`机检 100%）`）。同文件另一条帮助写的是 `%%`（正确的 escape），
所以这不是"风格不一致"而是**有一整条旗标的说明在用法屏上永远读不到**，
读它的人只能去猜旗标名。

本守卫钉三件事，缺一不可：
1. `--help` 退出码为 0（不是 Traceback）；
2. 帮助输出里**每个已声明的旗标名都出现**——只有 rc=0 不代表整屏打印成功；
3. 全仓 `add_argument` 的帮助文字里不许出现裸 `%`（防同类回归在别的仪器上复发）。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE = PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py"


def _help_output() -> str:
    run = subprocess.run(
        [sys.executable, str(PROBE), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert run.returncode == 0, f"--help 退出码 {run.returncode}：{run.stderr[-300:]}"
    return run.stdout


def test_probe_help_exits_zero_and_lists_every_flag() -> None:
    text = _help_output()
    source = PROBE.read_text(encoding="utf-8")
    declared = sorted(set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source)))
    assert declared, "一个旗标都没扫到 ⇒ 扫描方式过期了，本守卫失去覆盖面"
    missing = [flag for flag in declared if flag not in text]
    assert not missing, f"这些旗标没出现在用法屏上：{missing}（共声明 {len(declared)} 个）"


def test_no_bare_percent_in_any_add_argument_help() -> None:
    """只查 `help=` 这个关键字的值——argparse 仅对帮助文字做 `%` 插值。

    第一版我用正则扫 `add_argument(...)` 里的所有字符串字面量，结果把 `default=` 里的
    `{datetime.now(UTC):%Y%m%d}` 也算成 offender（那是 f-string 已经插值完的日期格式，不会崩）。
    假阳性会淹掉真问题，所以改成按 AST 取 `help` 关键字的常量值。
    """

    import ast

    offenders: list[str] = []
    for path in sorted((PROJECT_ROOT / "scripts" / "training").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            for kw in node.keywords:
                if kw.arg != "help" or not isinstance(kw.value, ast.Constant):
                    continue
                value = kw.value.value
                if isinstance(value, str) and re.search(r"(?<!%)%(?![%s])", value):
                    flag = node.args[0].value if node.args else "?"
                    offenders.append(f"{path.name}:{node.lineno} {flag} ⇒ {value[:70]}")
    assert not offenders, (
        "帮助文字里有裸 %（argparse 会做 % 插值 ⇒ 这条旗标的说明在用法屏上永远读不到）：\n  "
        + "\n  ".join(offenders)
    )

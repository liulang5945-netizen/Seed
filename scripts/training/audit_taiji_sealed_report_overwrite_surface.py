"""DEBT-G35／G50 的暴露面读数：哪些入库仪器的默认写靶直接落在**已入库**的报告件上。

G35 的处方写着「训练仪器的 `--out` 写靶必须按 sha 后缀命名或写前把旧件改名，让『覆盖』变『并存』」，
G50 又写着「与 DEBT-G35 的拒绝面需一起设计以免互相覆盖」——两行到今天只有散文，**没有暴露面计数**。
本件把那一面做成读数：扫版本控制里的 `scripts/**/*.py`，用 AST 取 argparse 的写靶旗标默认值，
只对**已 tracked 的路径**计数，并按"写前会不会拒绝"分**三类**出版。

为什么是三分类而不是"有／无护栏"两分类
--------------------------------------
起草时我原本按「有 `--overwrite` 旗标 **且** 有 `exists()` 检查」算"有护栏"，
但实读 `scripts/training/calibrate_taiji_r2_readout_retrain.py:215` 是另一种合法形状：

    if report_path.exists():
        parser.error(f"{report_path} already exists; this instrument never overwrites a report")

它**没有** `--overwrite` 旗标，却是最严格的那种"永不覆写"。用两分类会把它误算进暴露面——
那正是"仪器把自己的错报成事实"的形状。所以分三档，且每档都要给出**命中的证据原文**：

* `refusal_without_override_flag`：写前拒绝，但不给覆写口（最严，`calibrate_taiji_r2` 那一形）；
* `refusal_with_override_flag`：写前拒绝＋`--overwrite`／`--force` 显式口子（本轮新落三枚仪器那一形）；
* `no_refusal_found`：既没拒绝也没旗标 ⇒ 才算**暴露面**。

`exists()` 单独出现不算护栏：`verify_core_reasoning_coverage.py:153`／`:167` 的 `exists()` 是**读**检查。
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FORMAT = "taiji-sealed-report-overwrite-surface-v1"
WRITE_TARGET_FLAGS = {"out", "report", "arm-report", "out-report"}
OVERRIDE_FLAGS = ("--overwrite", "--force")
REFUSAL_CALLS = ("error", "SystemExit", "exit", "_exit", "raise", "return")
CATEGORY_REFUSE_ONLY = "refusal_without_override_flag"
CATEGORY_REFUSE_WITH_FLAG = "refusal_with_override_flag"
CATEGORY_NONE = "no_refusal_found"


def _tracked(root: Path) -> list[str]:
    return subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()


def _write_flag_defaults(tree: ast.Module) -> list[tuple[list[str], str]]:
    found: list[tuple[list[str], str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", "") != "add_argument":
            continue
        names = [
            a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        if not names or names[0].lstrip("-") not in WRITE_TARGET_FLAGS:
            continue
        default = None
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                default = kw.value.value
        if isinstance(default, str):
            found.append((names, default))
    return found


def _declares_override_flag(tree: ast.Module) -> bool:
    """同一文件里是否声明了 `--overwrite`／`--force`（拒绝覆写要能被显式解除）。"""

    return any(
        isinstance(node.args[0], ast.Constant) and node.args[0].value in OVERRIDE_FLAGS
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_argument"
        and node.args
    )


def _negated_positions(test_node: ast.expr) -> list[ast.expr]:
    """返回被 `not` 直接作用的 `.exists()` 调用（读侧检查的形状）。"""

    negated: list[ast.expr] = []
    for node in ast.walk(test_node):
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Call)
            and isinstance(node.operand.func, ast.Attribute)
            and node.operand.func.attr == "exists"
        ):
            negated.append(node.operand)
    return negated


def _exists_guard_bodies(tree: ast.Module) -> list[str]:
    """找 `if <...>.exists():` 且体内调用拒绝出口（parser.error／SystemExit／…）的那些体原文。"""

    bodies: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        #: 测试条件常写成 `if path.exists() and not args.overwrite:`（`BoolOp`），
        #: 只认裸 `Call` 会把**自己家的护栏**漏掉、误报成暴露面（沙箱实测抓到这一形）。
        exists_calls = [
            sub
            for sub in ast.walk(node.test)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "exists"
        ]
        if not exists_calls:
            continue
        #: `if not path.exists(): raise` 是**读侧存在性检查**（输入没准备好），
        #: 不是"写前拒绝覆写"。把它当护栏会把真暴露面漏掉（实跑 `verify_core_reasoning_coverage` 抓到）。
        negated = _negated_positions(node.test)
        if exists_calls and len(negated) == len(exists_calls):
            #: 全部 exists() 都是被 `not` 作用的读侧检查 ⇒ 不构成"写前拒绝覆写"的护栏。
            continue
        rendered = "\n".join(ast.unparse(child) for child in node.body)
        if any(token in rendered for token in REFUSAL_CALLS):
            bodies.append(rendered)
    return bodies


def classify(tree: ast.Module, source: str) -> tuple[str, list[str]]:
    bodies = _exists_guard_bodies(tree)
    if not bodies:
        return CATEGORY_NONE, []
    evidence = bodies[:2]
    if _declares_override_flag(tree):
        return CATEGORY_REFUSE_WITH_FLAG, evidence
    return CATEGORY_REFUSE_ONLY, evidence


def scan(root: Path, tracked_set: set[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rel in sorted(
        p for p in sorted(tracked_set) if p.startswith("scripts/") and p.endswith(".py")
    ):
        source = (root / rel).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        category, evidence = classify(tree, source)
        for names, default in _write_flag_defaults(tree):
            if default in tracked_set:
                rows.append(
                    {
                        "file": rel,
                        "flags": names,
                        "default_target": default,
                        "category": category,
                        "refusal_evidence": evidence,
                    }
                )
    return rows


def build_report(root: Path) -> dict[str, object]:
    tracked = set(_tracked(root))
    rows = scan(root, tracked)
    if not rows:
        raise ValueError(
            "零命中：没有一枚仪器的默认写靶落在已入库件上。先确认扫描面（tracked 的 scripts/**/*.py），"
            "不许把零命中读成「这面已经干净」"
        )
    counts = {
        category: sum(1 for row in rows if row["category"] == category)
        for category in (CATEGORY_REFUSE_ONLY, CATEGORY_REFUSE_WITH_FLAG, CATEGORY_NONE)
    }
    return {
        "format": REPORT_FORMAT,
        "debts": ["DEBT-G35", "DEBT-G50"],
        "face": "tracked 的 scripts/**/*.py；写靶只认 argparse 第一个参数名在名单里的 default 字符串",
        "script_files_scanned": len(
            [p for p in tracked if p.startswith("scripts/") and p.endswith(".py")]
        ),
        "default_targets_on_tracked_files": len(rows),
        "distinct_files": len({row["file"] for row in rows}),
        "category_counts": counts,
        "exposure_targets": counts[CATEGORY_NONE],
        "exposure_files": len({row["file"] for row in rows if row["category"] == CATEGORY_NONE}),
        "rows": rows,
        "categories": {
            CATEGORY_REFUSE_ONLY: "写前拒绝覆写、但不给覆写口（最严；`calibrate_taiji_r2` 那一形）",
            CATEGORY_REFUSE_WITH_FLAG: "写前拒绝＋`--overwrite`／`--force` 显式口子",
            CATEGORY_NONE: "既没写前拒绝也没旗标 ⇒ 计入暴露面",
        },
        "scope_note": "只读普查：不改别人名下仪器的默认落点（改了会让既有引用失效，属别线半径）",
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="扫描根（契约测指向临时仓）")
    parser.add_argument(
        "--out-report",
        default="reports/taiji_sealed_report_overwrite_surface_20261011.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(Path(args.root))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = Path(args.root) / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "script_files_scanned": report["script_files_scanned"],
                "default_targets_on_tracked_files": report["default_targets_on_tracked_files"],
                "distinct_files": report["distinct_files"],
                "category_counts": report["category_counts"],
                "exposure_targets": report["exposure_targets"],
                "exposure_files": report["exposure_files"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

2026-10-13（DEBT-G132／㊵-774）升版 v2：加**第二张面** `literal_assignment`＝源码里硬编的字面落盘路径
（`out = R / "reports/x.json"` 且 `out` 被当写靶用过）。v1 只认 argparse 的 `default`，于是
`measure_taiji_n2_06_write_contract_values.py` 那种"把落盘靶硬编成一枚已入库读数件"的仪器**整面看不见**——
它既不报命中也不报暴露，而我差点就照着"暴露面里没有它"去跑它、覆盖掉那份 20261011 的在册件。
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
#: 契约册（`test_n5_25`）按文件路径加载本件、也被起成真子进程，脚本自己那层目录先上 `sys.path`，
#: 统一落盘助手才解析得到（同㊵-790／802 那条例外约定）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

#: ㊵-790：写靶识别面**变宽**（统一落盘助手的两个入口也算写入）⇒ 按"识别边界变了就升版"的规矩走 v3。
REPORT_FORMAT = "taiji-sealed-report-overwrite-surface-v3"
WRITE_TARGET_FLAGS = {"out", "report", "arm-report", "out-report"}
#: 字面路径里出现这个前缀才算"落盘到读数件面"（第二张面的判据之一）。
REPORT_MARKER = "reports/"
OVERRIDE_FLAGS = ("--overwrite", "--force")
CATEGORY_REFUSE_ONLY = "refusal_without_override_flag"
CATEGORY_REFUSE_WITH_FLAG = "refusal_with_override_flag"
CATEGORY_NONE = "no_refusal_found"


#: 统一落盘助手的两个写入口（`scripts/training/_taiji_report_io.py`）。它们与 `write_text` 等价地
#: "把一个变量当写靶用过"——不认它们，修法③ 每迁一台仪器就从本件面上少一个写靶。
SHARED_WRITER_CALLS = ("write_text_report", "write_json_report")


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


def _body_has_refusal(body: list[ast.stmt]) -> bool:
    """拒绝出口只认**节点形状**：`x.error(...)`／`raise`／`sys.exit`／`return <整数>`。

    v1 用的是"渲染成文本后找子串"，于是两处假护栏把暴露面算少了（2026-10-13 现查，DEBT-G132）：
    `fam.read_text(encoding='utf-8', errors='replace')` 里的 `errors=` 命中了 `error`，
    而函数中途的 `return 'bypass_confirmed（…）'` 命中了 `return`。安全面宁可多报暴露，
    不能把"其实会覆写"读成"有护栏"（[[guard-must-be-able-to-fail]]）。
    """

    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("error", "exit", "_exit"):
                return True
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
            #: `isinstance(True, int)` 为真，但 `return True` 不是拒绝出口。
            and not isinstance(node.value.value, bool)
        ):
            return True
    return False


def _exists_guard_bodies(tree: ast.Module) -> list[str]:
    """找 `if <...>.exists():` 且体内有拒绝出口（`parser.error`／`raise`／`exit`／`return <int>`）的那些体原文。"""

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
        if _body_has_refusal(node.body):
            bodies.append("\n".join(ast.unparse(child) for child in node.body))
    return bodies


def _base_name(expr: ast.expr) -> str | None:
    """`target.write_text(...)`／`Path(target).write_text(...)`／`open(target, "w")` 里的宿主变量名。"""

    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.args:
        inner = expr.args[0]
        return inner.id if isinstance(inner, ast.Name) else None
    return None


def _literal_target_names(tree: ast.Module) -> dict[str, list[str]]:
    """变量名 → 它绑定的 `reports/` **字面**路径（含 `R / "reports/x.json"` 这种拼接形）。

    这是第二张面：写靶不走 argparse 默认值，而是硬编在源码里。2026-10-13 发现本器整面漏着它
    （`measure_taiji_n2_06_write_contract_values.py` 就是把落盘路径硬编成**一枚已入库读数件**，
    所以它既不在命中里也不在暴露面里 ⇒ DEBT-G132）。
    """

    bound: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        literals = [
            sub.value
            for sub in ast.walk(node.value)
            if isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and REPORT_MARKER in sub.value
        ]
        if literals:
            bound.setdefault(target.id, []).extend(literals)
    return bound


def _written_names(tree: ast.Module) -> set[str]:
    """被**当写靶用过**的变量名。只绑过字面路径而没往里写的不算（那是读数件常量或读检查）。

    ㊵-790：统一落盘助手（DEBT-G115 修法③，`_taiji_report_io.py`）把写入从 `x.write_text(...)`
    收进 `report_io.write_text_report(x, ...)`——**如果这里只认 `write_text`，迁移一台仪器就从本件的
    识别面上抹掉一个写靶**，暴露面会被自己主导的收口动作悄悄缩小。故助手那两个名字同样算写入。
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = getattr(func, "attr", None) or (func.id if isinstance(func, ast.Name) else "")
        if called in SHARED_WRITER_CALLS and node.args:
            #: 助手的第一个参数就是写靶；`report_io` 这个接收者本身不是靶。
            name = _base_name(node.args[0])
            if name:
                names.add(name)
            continue
        if isinstance(func, ast.Attribute) and func.attr in ("write_text", "write_bytes"):
            name = _base_name(func.value)
            if name:
                names.add(name)
        elif isinstance(func, ast.Name) and func.id == "open" and node.args:
            mode = node.args[1] if len(node.args) > 1 else None
            writing = (
                isinstance(mode, ast.Constant)
                and isinstance(mode.value, str)
                and any(char in mode.value for char in "wax+")
            )
            if writing:
                name = _base_name(node.args[0])
                if name:
                    names.add(name)
    return names


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
                        "face": "argparse_default",
                        "flags": names,
                        "default_target": default,
                        "category": category,
                        "refusal_evidence": evidence,
                    }
                )
        written = _written_names(tree)
        for name, literals in sorted(_literal_target_names(tree).items()):
            if name not in written:
                #: 只绑过字面路径、没往里写过 ⇒ 那是读数件常量或读侧检查，不是落盘靶。
                continue
            for literal in sorted(set(literals)):
                if literal in tracked_set:
                    rows.append(
                        {
                            "file": rel,
                            "face": "literal_assignment",
                            "flags": [name],
                            "default_target": literal,
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
    face_counts = {
        face: sum(1 for row in rows if row["face"] == face)
        for face in ("argparse_default", "literal_assignment")
    }
    return {
        "format": REPORT_FORMAT,
        "debts": ["DEBT-G35", "DEBT-G50", "DEBT-G132"],
        "face": (
            "tracked 的 scripts/**/*.py；两张面并立——`argparse_default`＝写靶旗标的默认值，"
            "`literal_assignment`＝源码里硬编的字面路径且该变量被当写靶用过"
            "（v2 新增这一面：v1 只认 argparse 默认值，把硬编靶整面漏掉）"
        ),
        "script_files_scanned": len(
            [p for p in tracked if p.startswith("scripts/") and p.endswith(".py")]
        ),
        "default_targets_on_tracked_files": len(rows),
        "face_counts": face_counts,
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
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "script_files_scanned": report["script_files_scanned"],
                "default_targets_on_tracked_files": report["default_targets_on_tracked_files"],
                "distinct_files": report["distinct_files"],
                "face_counts": report["face_counts"],
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

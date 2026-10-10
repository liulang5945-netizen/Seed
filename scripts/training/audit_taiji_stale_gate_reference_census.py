""""读已封存件冒充现行绿"普查仪：把判据里对 `["gate"]` 的取下标按**取路**分两档。

来历（2026-10-10，DEBT-G91）：p2-11 那枚 IDE 语言链门红了三天（㊵-662／DEBT-G87），而仓里零道门跟着红——
因为引用它的三条判据读的都是 `reports/*_20260831.json` 里的 `gate.passed` 字段，
**封存件永远绿**。这类判据字面读起来像"现在也绿"，实际只证明"文件里写着绿"。

本件不裁判任何行为，只回答一件事：*这条判据的绿是当场跑出来的，还是从一件里读出来的*。

两档定义（按取路，不按名字猜）：
* `sealed_artifact` —— 该赋值里对 `["gate"]` 的取下标，其被下标对象链上出现 `read_text`／`json.loads`，
  **且**同一赋值里能找出 `.json` 字面量或 `"reports"` 字面量；
* `live_object` —— 有 `["gate"]` 下标但取路不是读文件（典型是判读函数当场返回的 dict）。

读数面必须能证自己扫到了东西：`files_scanned == 0` 或 glob 根不存在 ⇒ rc=2（零命中要连范围一起报）。
输出只出 ASCII（`ensure_ascii=True`），win32 GBK 控制台上中文与 `⇒` 会把一次正常普查砸成 `UnicodeEncodeError`。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAT = "taiji-stale-gate-reference-census-v1"
GATE_KEY = "gate"
READ_CALLS = ("read_text", "loads")


def _display(path: Path) -> str:
    """仓内文件显示成相对路径，仓外（例如测里的 tmp_path 面）原样显示。

    `relative_to` 对仓外路径会抛 ValueError——那正是本器第一版在合成面上踩到的错，
    由契约测抓出（见 tests/taiji_native/test_n6_01_stale_gate_reference_census_contract.py）。
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _gate_subscripts(tree: ast.AST) -> list[ast.Subscript]:
    found: list[ast.Subscript] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and slice_node.value == GATE_KEY:
            found.append(node)
    return found


def _reads_a_file(subscript: ast.Subscript) -> bool:
    chain = ast.dump(subscript.value)
    return any(call in chain for call in READ_CALLS)


def _json_literals(container: ast.AST) -> list[str]:
    names = [
        node.value
        for node in ast.walk(container)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.endswith(".json")
    ]
    reports_flag = any(
        isinstance(node, ast.Constant) and node.value == "reports" for node in ast.walk(container)
    )
    return sorted(set(names)) + (["<reports-dir-constant>"] if reports_flag else [])


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _comprehension_iter(subscript: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    """向上找包含该下标的**推导式**，取它的 `iter`（`all(read_text … for name in (…json…))` 那两处的件名
    就住在 iter 里，不在下标自己的子树里）。只往上看一层不够：下标 → Compare → GeneratorExp。"""
    iters: list[ast.AST] = []
    node = parents.get(subscript)
    while node is not None:
        if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            for comprehension in node.generators:
                iters.append(comprehension.iter)
            break
        if isinstance(node, (ast.Assign, ast.Return)):
            break  #: 不在推导式里就只看下标自身那条链，不拿同一条赋值里的别的件名凑证据。
        node = parents.get(node)
    return iters


def scan_file(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [{"file": str(path), "status": "unparsed"}]
    parents = _parent_map(tree)
    rows: list[dict[str, Any]] = []
    for subscript in _gate_subscripts(tree):
        reads_file = _reads_a_file(subscript)
        scopes: list[ast.AST] = [subscript.value] if reads_file else []
        scopes += _comprehension_iter(subscript, parents) if reads_file else []
        literals = sorted({name for scope in scopes for name in _json_literals(scope)})
        rows.append(
            {
                "file": _display(path),
                "line": getattr(subscript, "lineno", 0),
                "bucket": "sealed_artifact" if literals else "live_object",
                "reads_file_chain": reads_file,
                "artifact_names": literals,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="gate 判据取路普查（封存件 vs 当场跑）")
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        metavar="DIR",
        help="相对仓根的扫描目录；不给则用 scripts/training",
    )
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args(argv)

    roots = [str(root) for root in (args.scan_root or ["scripts/training"])]
    missing_roots = [root for root in roots if not (PROJECT_ROOT / root).is_dir()]
    if missing_roots:
        print(json.dumps({"format": FORMAT, "status": "refuse_missing_scan_root", "missing": missing_roots}))
        return 2

    files = sorted(
        path
        for root in roots
        for path in (PROJECT_ROOT / root).rglob("*.py")
        if path.is_file()
    )
    rows: list[dict[str, Any]] = []
    unparsed: list[str] = []
    for path in files:
        for row in scan_file(path):
            if row.get("status") == "unparsed":
                unparsed.append(row["file"])
            else:
                rows.append(row)

    sealed = [row for row in rows if row["bucket"] == "sealed_artifact"]
    payload: dict[str, Any] = {
        "format": FORMAT,
        "status": "measured",
        "scan_roots": roots,
        "pattern": 'Subscript(slice==\"gate\") ∧ 被下标对象链含 read_text／json.loads ∧ 同赋值内有 .json 字面量或 \"reports\" 常量',
        "files_scanned": len(files),
        "gate_subscript_sites": len(rows),
        "sealed_artifact_sites": len(sealed),
        "live_object_sites": len(rows) - len(sealed),
        "unparsed_file_count": len(unparsed),
        "sealed_reads": sealed,
        "all_sites": rows,
        "reading_limit": (
            "本器只答『这条绿是从件里读的还是当场跑的』，不证明该门今天真跑会绿，"
            "也不裁判判据本身对不对（裁判在读数件与测里）"
        ),
    }
    if not files:
        #: glob 写窄会让"零违例"变成假绿 ⇒ 宁可红。
        payload["status"] = "refuse_nothing_scanned"
        print(json.dumps(payload, ensure_ascii=True))
        return 2
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    if args.out_report is not None:
        out_path = args.out_report if args.out_report.is_absolute() else PROJECT_ROOT / args.out_report
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

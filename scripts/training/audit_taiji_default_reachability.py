"""默认位可达性分析器：给定"产品面第 3 层读取点"，判它在**默认装配**下会不会被走到。

为什么要有这一台
----------------
㊵-699 的教训是：「代码里存在消费者」与「默认链上有人会走到它」是两句。
`audit_taiji_write_only_key_face.py` 出版了点位，但可达性那一列当时要求**人工逐枚登记**——
手登记就是把算术交给我自己（见『判据的算术一旦手算就是债』）。本件把它变成可复算读数。

三路判定（每路都能为假，且各自点名取法）
----------------------------------------
1. **`reachable_at_default`**＝该读取点所在函数在 tracked 代码里有调用者，且**至少一个调用者
   不在旗标分支里**（旗标＝同文件 `argparse` 里 `action="store_true"` 或 `default=False` 的目标名）。
2. **`flag_gated`**＝所有调用者都落在某个旗标的真分支里 ⇒ 默认位不可达。
   这一路的"为假"侧＝去掉那层 `if` 后必须翻成 `reachable_at_default`（契约测钉这一支）。
3. **`unreachable_in_tracked_code`**＝tracked 代码里**找不到任何调用者**。
   这是一个**独立出口**，不许被读成"可达"，也不许被读成"不可达"：静态面看不见动态派发与外部调用，
   它只说"在这张扫描面上没人调用它"。

已知看不见什么（如实出版，不当成 0）
------------------------------------
* 通过属性名转发（`getattr(obj, name)`）、回调、注册表分派、跨语言入口。
* 同名函数不同类（本件**按名字**匹配调用点，与 ㊵-699 的"名字会伪装轴"同族风险）⇒ 判"可达"侧偏保守：
  只要有一个非门控调用者就算可达；判"flag_gated"侧要求**全部**调用者都被门控。
* 继承带来的调用者（子类里 `self.method()` 的名字仍是同一函数名，能命中；反向的改名命中不到）。

零算力：只读版本控制里的源码文本。
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FORMAT = "taiji-default-reachability-v1"
VERDICTS = (
    "reachable_at_default",
    "reachable_in_default_training_run",
    "flag_gated",
    "no_default_face_caller",
    "reachability_indeterminate",
    "unreachable_in_tracked_code",
)


def tracked_python(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [root / rel for rel in out]


def face_of(rel: str) -> str:
    """调用者住在哪张面。

    **只有 `product` 与 `runner` 两张面参与默认装配判定**：`scripts/training/` 下除了跑道器
    本身，其余是 eval_/probe_/audit_/measure_ 一类**仪器**——仪器调用一个函数不证明产品在默认
    装配下会走到它（那正是 ㊵-699 与 DEBT-G106 反复付过学费的地方）。
    """

    if rel.startswith("taiji/") or rel.startswith("seed/"):
        return "product"
    if rel == "scripts/training/train_seed_corpus.py":
        return "runner"
    if rel.startswith("scripts/"):
        return "instrument"
    if rel.startswith("tests/"):
        return "tests"
    return "other"


def _flag_dests(tree: ast.Module) -> set[str]:
    """同文件 argparse 里默认关的旗标目标名（`--n5-shadow` → `n5_shadow`）。"""

    dests: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        actions = {
            keyword.value.value
            for keyword in node.keywords
            if keyword.arg == "action" and isinstance(keyword.value, ast.Constant)
        }
        defaults = [
            keyword.value
            for keyword in node.keywords
            if keyword.arg == "default"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
        ]
        if "store_true" not in actions and not defaults:
            continue
        positional = [
            a for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        for opt in positional:
            dests.add(opt.value.lstrip("-").replace("-", "_"))
        for keyword in node.keywords:
            if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                dests.add(str(keyword.value.value))
    return dests


def _gate_flags_in_test(test: ast.expr) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(test):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _gating_flags(enclosing_ifs: list[ast.If], flag_dests: set[str]) -> set[str]:
    gating: set[str] = set()
    for stmt in enclosing_ifs:
        gating |= _gate_flags_in_test(stmt.test) & flag_dests
    return gating


def _contains(node: ast.AST, lineno: int) -> bool:
    return node.lineno <= lineno <= (node.end_lineno or node.lineno)


def _if_stack(fn: ast.FunctionDef, lineno: int) -> list[ast.If]:
    """函数体内**把 lineno 包在分支里**的 If（由外到内；目标落在 `test` 本身的不算门控）。"""

    stack = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(_contains(stmt, lineno) for group in (node.body, node.orelse) for stmt in group)
    ]
    return sorted(stack, key=lambda node: node.lineno)


def index_callers(root: Path) -> dict[str, list[dict[str, object]]]:
    """`函数名 -> 调用点清单`，每个调用点带面、所在函数、以及**包住它的旗标名**。"""

    by_name: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in tracked_python(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        face = face_of(rel)
        flags = _flag_dests(tree)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            name = getattr(call.func, "attr", None) or getattr(call.func, "id", None)
            if not name:
                continue
            owner = max(
                (
                    fn
                    for fn in functions
                    if fn.lineno <= call.lineno <= (fn.end_lineno or fn.lineno)
                ),
                key=lambda fn: fn.lineno,
                default=None,
            )
            gating = _gating_flags(_if_stack(owner, call.lineno), flags) if owner else set()
            by_name[str(name)].append(
                {
                    "file": rel,
                    "face": face,
                    "line": call.lineno,
                    "function": owner.name if owner else "<module>",
                    "gating_flags": sorted(gating),
                }
            )
    return dict(by_name)


def classify(
    root: Path,
    sites: list[dict[str, object]],
    callers: dict[str, list[dict[str, object]]] | None = None,
) -> list[dict[str, object]]:
    """按**调用者的面**分层判可达性。

    测试文件里的调用**不**证明默认装配可达（那正是 `audit_taiji_write_only_key_face`
    把"只有测试在读"单列一个出口的理由）；评测／诊断脚本同理——它们是仪器，不是产品装配。
    """

    callers = callers if callers is not None else index_callers(root)
    rows: list[dict[str, object]] = []
    for site in sites:
        function = str(site["function"])
        rel = str(site["file"])
        found = [
            c
            for c in callers.get(function, [])
            if not (c["file"] == rel and c["line"] == site["line"])
        ]
        relevant = [c for c in found if c["face"] in {"product", "runner"}]
        product_ungated = [c for c in relevant if c["face"] == "product" and not c["gating_flags"]]
        runner_ungated = [c for c in relevant if c["face"] == "runner" and not c["gating_flags"]]
        gated = [c for c in relevant if c["gating_flags"]]
        if not found:
            verdict = "unreachable_in_tracked_code"
        elif not relevant:
            verdict = "no_default_face_caller"
        elif product_ungated:
            verdict = "reachable_at_default"
        elif runner_ungated:
            verdict = "reachable_in_default_training_run"
        elif len(gated) == len(relevant):
            verdict = "flag_gated"
        else:
            verdict = "reachability_indeterminate"
        rows.append(
            {
                "site": f"{rel}:{site['line']}@{function}",
                "caller_count": len(found),
                "default_face_callers": len(relevant),
                "product_ungated_callers": len(product_ungated),
                "runner_ungated_callers": len(runner_ungated),
                "gated_callers": len(gated),
                "gating_flags": sorted({f for c in found for f in c["gating_flags"]}),
                "verdict": verdict,
                "caller_faces": dict(
                    sorted(
                        {
                            face: sum(1 for c in found if c["face"] == face)
                            for face in {str(c["face"]) for c in found}
                        }.items()
                    )
                ),
                "evidence": [
                    f"{c['file']}:{c['line']}@{c['function']} gated={('|'.join(c['gating_flags'])) or '-'}"
                    for c in found[:6]
                ],
            }
        )
    return rows


def build_report(root: Path, sites_by_key: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    callers = index_callers(root)
    per_key = {key: classify(root, sites, callers) for key, sites in sites_by_key.items()}
    tally: dict[str, int] = defaultdict(int)
    for rows in per_key.values():
        for row in rows:
            tally[str(row["verdict"])] += 1
    unknown = [
        f"{key}@{row['site']}"
        for key, rows in per_key.items()
        for row in rows
        if row["verdict"] == "unreachable_in_tracked_code"
    ]
    indeterminate = [
        f"{key}@{row['site']}"
        for key, rows in per_key.items()
        for row in rows
        if row["verdict"] == "reachability_indeterminate"
    ]
    stray = [
        f"{key}@{row['site']}"
        for key, rows in per_key.items()
        for row in rows
        if row["verdict"] not in VERDICTS
    ]
    if stray:
        raise ValueError(f"可达性判级词不在声明的词典里（仪器自己走歪了）：{stray}")
    return {
        "format": REPORT_FORMAT,
        "verdict_vocabulary": list(VERDICTS),
        "scanned_files": len(tracked_python(root)),
        "per_key": per_key,
        "tally": dict(tally),
        "no_caller_sites": unknown,
        "indeterminate_sites": indeterminate,
        "limits": [
            "调用点按**函数名**匹配：动态派发（getattr／注册表／回调）与改名后的调用看不见；"
            "判定刻意不对称——产品面有一个非门控调用者就算 reachable_at_default，"
            "全部调用者都被门控才判 flag_gated",
            "**测试与仪器里的调用者不参与默认装配判定**：只有它们在读的点位单列成 "
            "no_default_face_caller（与 audit_taiji_write_only_key_face 的 "
            "read_only_outside_product 同族），既不读成「可达」也不读成「没人用」",
            "旗标默认值只从**同文件**的 add_argument 取（跨文件传参的开关链不在静态面内）；"
            "混合门控与非门控调用者的点位判 reachability_indeterminate 并单列，不并成任何一边",
        ],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--from-report",
        type=Path,
        default=PROJECT_ROOT / "reports/taiji_write_only_key_face_20261011.json",
        help="读取 `audit_taiji_write_only_key_face` 的件，取其 action_read_sites 作输入",
    )
    parser.add_argument("--out-report", default="reports/taiji_default_reachability_20261011.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report_path = Path(args.from_report)
    if not report_path.is_file():
        print(f"输入件不存在，先跑 audit_taiji_write_only_key_face：{report_path}", file=sys.stderr)
        return 2
    face = json.loads(report_path.read_text(encoding="utf-8"))
    sites_by_key: dict[str, list[dict[str, object]]] = {}
    for row in face["rows"]:
        parsed: list[dict[str, object]] = []
        for site in row["action_read_sites"]:
            location, _, function = site.partition("@")
            file_name, _, line = location.partition(":")
            parsed.append({"file": file_name, "line": int(line), "function": function})
        sites_by_key[str(row["key"])] = parsed
    try:
        out_report = build_report(Path(args.root), sites_by_key)
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
    out.write_text(
        json.dumps(out_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "scanned_files": out_report["scanned_files"],
                "tally": out_report["tally"],
                "no_caller_sites": len(out_report["no_caller_sites"]),
                "indeterminate_sites": len(out_report["indeterminate_sites"]),
                "out": out.as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

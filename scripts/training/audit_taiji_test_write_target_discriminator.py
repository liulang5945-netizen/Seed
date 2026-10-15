"""DEBT-G137／PLAN-G138 的实施格仪器：把 `tests/` 调用面里的（工具，旗标）对按**工具自己的数据流**判读／写靶角色。

为什么不是按旗标名（`PLAN-G138` §1 冻的就是这个）：`--report` 在本仓一名两职——它既是评测器的写靶，
也是复算仪的读源；按名字归类必然造出假暴露面或漏掉真暴露面。本器把角色只从**被调工具自己的 AST** 取，
测试侧只贡献"哪些（工具，旗标）对真的被调用过"这一张域。

口径（逐条对应出版键）：
* 域＝tracked 测的调用面里出现过的（工具，旗标）三元组；`unmatched` 单列并逐枚点名，
  `every_seen_pair_resolves_in_tool` 要求两者相等（J-G138-1 的第一支契约测形状）。
* 角色＝`write`／`read`／`both`／`unresolved`；`both` 与 `unresolved` 不入写靶桶（J-G138-1／J-G138-4）。
* 旗标字面量整串匹配 `^--[A-Za-z0-9][A-Za-z0-9_-]*$`；同时出版"放宽成前缀匹配会多捞什么"，
  于是 G-G138-1 这条守卫能为假。
* 所有桶名都带 `test_side_` 前缀：`37／36／22／21` 那组在册口径归
  `audit_taiji_sealed_report_overwrite_surface.py`，本器不顺带动它（J-G138-3）。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _entry in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import _taiji_report_io as report_io  # noqa: E402

REPORT_FORMAT = "taiji-test-write-target-discriminator-v1"
FLAG_STRICT = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9_-]*$")
FLAG_PREFIX_LOOSE = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9_-]*")
WRITE_RECEIVER_METHODS = {"write_text", "write_bytes"}
READ_RECEIVER_METHODS = {"read_text", "read_bytes"}
WRITE_ARG_FUNCS = {"write_text_report", "write_json_report", "dump"}
READ_ARG_FUNCS = {"load", "loads"}
ARGS_NAMES = {"args", "parsed", "ns"}


def tracked_python(root: Path) -> list[str]:
    """The scan face comes from version control, not from a disk glob."""

    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line for line in result.stdout.splitlines() if line]


def parse(rel: str, root: Path) -> ast.Module | None:
    try:
        return ast.parse((root / rel).read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def flag_table(tree: ast.Module) -> dict[str, str]:
    """flag string -> argparse dest, for every `add_argument` call in this module."""

    table: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        longs = [
            a.value
            for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--")
        ]
        if not longs:
            continue
        dest = next(
            (
                kw.value.value
                for kw in node.keywords
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant)
            ),
            None,
        )
        table[longs[0]] = dest or longs[0][2:].replace("-", "_")
    return table


def name_ids(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def unwrap(node: ast.expr) -> list[ast.expr]:
    """Follow the shapes the frozen plan requires: plain, ``Path(...)`` and ternary."""

    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Path":
        return list(node.args)
    if isinstance(node, ast.IfExp):
        return [node.body, node.orelse]
    return [node]


def is_args_dest(node: ast.expr, dest: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == dest
        and isinstance(node.value, ast.Name)
        and node.value.id in ARGS_NAMES
    )


def target_names(target: ast.expr) -> set[str]:
    out: set[str] = set()
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            out |= target_names(element)
    return out


def function_params(tree: ast.Module, func: str) -> dict[str, list[str]]:
    """Same-module signature: positional parameters with the bound-method slot dropped, plus keywords."""

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            args = node.args
            positional = [a.arg for a in [*args.posonlyargs, *args.args]]
            if positional and positional[0] in {"self", "cls"}:
                positional = positional[1:]
            keywords = {a.arg for a in [*args.kwonlyargs, *args.args, *args.posonlyargs]}
            return {"positional": positional, "keyword": sorted(keywords)}
    return {}


def carries(node: ast.expr, dest: str, watched: set[str]) -> bool:
    """Does this expression carry ``args.<dest>`` or an already-bound name?  Subtree-wide, so the
    for-tuple and container shapes in PLAN-G138 §1 are seen, not just a top-level unwrap."""

    return any(is_args_dest(sub, dest) or bool(name_ids(sub) & watched) for sub in ast.walk(node))


def bound_names(tree: ast.Module, dest: str) -> set[str]:
    """Names that transitively carry ``args.<dest>``: assignments (incl. ``Path(...)`` and ternary),
    for-tuple elements fed by such a container, and one hop of same-module function parameters."""

    seeds = {dest}
    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                sources = [s for value in unwrap(node.value) for s in unwrap(value)]
                if any(is_args_dest(s, dest) or name_ids(s) & (names | seeds) for s in sources):
                    for target in node.targets:
                        for tid in target_names(target):
                            if tid not in names:
                                names.add(tid)
                                changed = True
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                if carries(node.iter, dest, names | seeds):
                    for tid in target_names(node.target):
                        if tid not in names:
                            names.add(tid)
                            changed = True
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                params = function_params(tree, node.func.id)
                if not params:
                    continue
                positional = params.get("positional", [])
                keywords = set(params.get("keyword", []))
                fed: set[str] = set()
                for index, arg in enumerate(node.args):
                    if carries(arg, dest, names | seeds) and index < len(positional):
                        fed.add(positional[index])
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    if carries(kw.value, dest, names | seeds) and kw.arg in keywords:
                        fed.add(kw.arg)
                for tid in fed:
                    if tid not in names:
                        names.add(tid)
                        changed = True
    return names


def dest_role(tree: ast.Module, dest: str) -> tuple[str, list[str]]:
    names = bound_names(tree, dest)
    roles: list[str] = []
    evidence: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        method = getattr(node.func, "attr", None)
        called = getattr(node.func, "id", "")
        subject: ast.expr | None = None
        if method in WRITE_RECEIVER_METHODS | READ_RECEIVER_METHODS:
            subject = node.func.value  # type: ignore[attr-defined]
        elif called in WRITE_ARG_FUNCS | READ_ARG_FUNCS:
            subject = node.args[0] if node.args else None
        if subject is None:
            continue
        if not (name_ids(subject) & names) and not is_args_dest(subject, dest):
            continue
        if method in WRITE_RECEIVER_METHODS or called in WRITE_ARG_FUNCS:
            roles.append("write")
            evidence.append(f"{'write '}{method or called}@{node.lineno}")
        else:
            roles.append("read")
            evidence.append(f"{'read '}{method or called}@{node.lineno}")
    if "write" in roles and "read" in roles:
        return "both", evidence
    if roles:
        return roles[0], evidence
    return "unresolved", evidence


def string_or_bound(element: ast.expr, constants: dict[str, str]) -> str | None:
    if isinstance(element, ast.Constant) and isinstance(element.value, str):
        return element.value
    if isinstance(element, ast.Name) and element.id in constants:
        return constants[element.id]
    return None


def module_string_constants(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for tid in target_names(node.targets[0] if node.targets else ast.Name(id="")):
                out[tid] = node.value.value
    return out


def call_site_pairs(tree: ast.Module) -> list[tuple[str, str]]:
    """(tool path, flag) pairs taken from list/tuple literals carrying both a ``.py`` path and flags."""

    constants = module_string_constants(tree)
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        strings = [
            s for element in node.elts if (s := string_or_bound(element, constants)) is not None
        ]
        tools = [s for s in strings if s.endswith(".py")]
        flags = [s for s in strings if FLAG_STRICT.match(s)]
        pairs.extend((tool, flag) for tool in tools for flag in flags)
    return pairs


def loose_only_literals(tree: ast.Module) -> set[str]:
    """G-G138-1 的负对照面：前缀匹配会把这些当旗标，整串匹配把它们挡在门外。"""

    hits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if FLAG_PREFIX_LOOSE.match(text) and not FLAG_STRICT.match(text):
                hits.add(text)
    return hits


def legacy_heuristic_candidates(root: Path, tests: list[str]) -> list[dict[str, str]]:
    """㊵-807 那批"同一调用里同时出现写靶旗标与 `reports/` 路径"的启发式候选——加回键就是为了它能被复算。

    这是**旧口径**（按旗标名猜角色），保留它只为两件事：让当时那个"6 枚"可复现，
    以及给 J-G138-2 的经验复核提供逐枚实跑的名单。分类一律不采用它。
    """

    write_flagish = re.compile(r"^--(out|output|report|dest|path)", re.I)
    out: list[dict[str, str]] = []
    for rel in tests:
        tree = parse(rel, root)
        if tree is None:
            continue
        constants = module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            strings = [
                s for element in node.elts if (s := string_or_bound(element, constants)) is not None
            ]
            paths = [s for s in strings if "reports" in s and s.endswith(".json")]
            flags = [s for s in strings if write_flagish.match(s)]
            if paths and flags:
                out.append({"test": rel, "report_path": paths[0], "flag": flags[0]})
                break
    return out


def build_report(root: Path, observed_writes: set[str] | None = None) -> dict[str, Any]:
    tracked = tracked_python(root)
    tests = [rel for rel in tracked if rel.startswith("tests/")]
    tools = [rel for rel in tracked if rel.startswith("scripts/")]

    tool_tables: dict[str, dict[str, str]] = {}
    tool_trees: dict[str, ast.Module] = {}
    for rel in tools:
        tree = parse(rel, root)
        if tree is None:
            continue
        table = flag_table(tree)
        if not table:
            continue
        tool_trees[rel] = tree
        tool_tables[rel] = table

    by_basename: dict[str, list[str]] = {}
    for rel in tool_tables:
        by_basename.setdefault(Path(rel).name, []).append(rel)

    pairs: dict[tuple[str, str, str], str] = {}
    downgraded: list[dict[str, Any]] = []
    unresolved_named: list[dict[str, Any]] = []
    unmatched: list[dict[str, str]] = []
    unresolvable_tool_strings: list[dict[str, str]] = []
    seen = 0
    loose_only: set[str] = set()
    for rel in tests:
        tree = parse(rel, root)
        if tree is None:
            continue
        loose_only |= loose_only_literals(tree)
        for tool_ref, flag in call_site_pairs(tree):
            candidates = by_basename.get(Path(tool_ref).name, [])
            if not candidates:
                # Not a (tool, flag) pair at all: the string is a glob, a fixture name, or a path that
                # is not a tracked tool. It stays out of the domain but is published so a reader can audit it.
                unresolvable_tool_strings.append({"test": rel, "tool": tool_ref, "flag": flag})
                continue
            seen += 1
            tool_rel = candidates[0]
            table = tool_tables[tool_rel]
            if flag not in table:
                unmatched.append(
                    {"test": rel, "tool": tool_rel, "flag": flag, "why": "flag_not_declared"}
                )
                continue
            role, evidence = dest_role(tool_trees[tool_rel], table[flag])
            label = f"{rel} :: {tool_rel} {flag}"
            if observed_writes and label in observed_writes and role != "write":
                # J-G138-2：观测是仲裁者。分类说它不写，实跑却改到了 reports/ ⇒ 降级并点名。
                downgraded.append(
                    {"pair": label, "role_by_ast": role, "role_after_observation": "unresolved"}
                )
                role, evidence = "unresolved", [*evidence, "observed_write_on_reports_face"]
            pairs[(rel, tool_rel, flag)] = role
            if role in {"unresolved", "both"}:
                unresolved_named.append(
                    {
                        "test": rel,
                        "tool": tool_rel,
                        "flag": flag,
                        "role": role,
                        "evidence": evidence[:3],
                    }
                )

    tally = {"write": 0, "read": 0, "both": 0, "unresolved": 0}
    for role in pairs.values():
        tally[role] += 1
    write_pairs = sorted(
        f"{test} :: {tool} {flag}" for (test, tool, flag), role in pairs.items() if role == "write"
    )
    # One entry per pair: a test may pass the same flag in two call sites, and the role is the tool's,
    # so the named list must not repeat it.
    seen_keys: set[tuple[str, str, str]] = set()
    unresolved_unique: list[dict[str, Any]] = []
    for item in unresolved_named:
        key = (item["test"], item["tool"], item["flag"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unresolved_unique.append(item)

    return {
        "format": REPORT_FORMAT,
        "debt": "DEBT-G137 第二半／PLAN-G138",
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "face": "tracked tests/ 的调用面（git ls-files，不按磁盘 glob）",
        "tracked_tests_total": len(tests),
        "tracked_tools_with_add_argument": len(tool_tables),
        "pairs_seen_in_call_sites": seen,
        "tool_strings_not_in_face_named": unresolvable_tool_strings[:40],
        "legacy_heuristic_candidates": legacy_heuristic_candidates(root, tests),
        "legacy_heuristic_candidate_total": len(legacy_heuristic_candidates(root, tests)),
        "observation_downgraded_pairs": downgraded,
        "observation_downgraded_total": len(downgraded),
        "tool_strings_not_in_face_total": len(unresolvable_tool_strings),
        "domain_pairs": len(pairs),
        "test_side_role_tally": tally,
        "test_side_write_pairs": write_pairs,
        "test_side_not_write_buckets_named": unresolved_unique[:120],
        "test_side_not_write_buckets_total": len(unresolved_unique),
        "pairs_unmatched_named": unmatched[:60],
        "pairs_unmatched_total": len(unmatched),
        "flag_literals_rejected_by_strict_regex": sorted(loose_only)[:25],
        "flag_literals_rejected_total": len(loose_only),
        "verdicts": {
            "face_is_not_empty": len(pairs) > 0,
            "every_seen_pair_resolves_in_tool": len(unmatched) == 0,
            "write_bucket_excludes_unresolved_and_both": len(write_pairs) == tally["write"]
            and not (
                set(write_pairs)
                & {f"{item['test']} :: {item['tool']} {item['flag']}" for item in unresolved_unique}
            ),
        },
        "reading_limit": (
            "本器只回答：**在 tracked 测的调用面里**哪些（工具，旗标）对可判为写靶。"
            "它不是 `tests/` 写靶的全集，也不是覆写风险清点；`37／36／22／21` 那组在册口径属 "
            "`audit_taiji_sealed_report_overwrite_surface.py`，本器不顺带动它（J-G138-3）。"
            "`unresolved` 与 `both` 不入写靶桶并逐枚点名（J-N5 系列的同族纪律）。"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tests/ 侧（工具，旗标）对的读／写判别式")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="git 仓根（契约测指向临时仓）")
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖自己名下的读数件")
    parser.add_argument(
        "--observed-writes",
        type=Path,
        default=None,
        help="J-G138-2 的观测清单：实跑确证改到 reports/ 的（测 :: 工具 旗标）对，逐条降级为 unresolved",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    observed: set[str] = set()
    if args.observed_writes is not None:
        observed_path = args.observed_writes
        if not observed_path.is_absolute():
            observed_path = root / observed_path
        observed = {str(line) for line in json.loads(observed_path.read_text(encoding="utf-8"))}

    out = args.out_report
    if not out.is_absolute():
        out = root / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    report = build_report(root, observed)
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)
    verdicts = report["verdicts"]
    if not all(verdicts.values()):
        print(
            "DISCRIMINATOR_REFUSED failing_verdicts="
            + ",".join(k for k, value in verdicts.items() if not value),
            file=sys.stderr,
        )
        return 1
    tally = report["test_side_role_tally"]
    print(
        "DISCRIMINATOR_OK "
        f"seen={report['pairs_seen_in_call_sites']} "
        f"resolved={report['domain_pairs']} "
        f"write={tally['write']} read={tally['read']} "
        f"both={tally['both']} unresolved={tally['unresolved']} "
        f"rejected={report['flag_literals_rejected_total']} "
        f"unmatched={report['pairs_unmatched_total']}"
    )
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    raise SystemExit(main())

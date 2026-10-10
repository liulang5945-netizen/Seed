"""控制台**可编码字形**普查（DEBT-G81 修法③，静态、零算力、只读）。

**为什么要有它**：㊵-648 那次实测里，一台 fail-closed 的仪器在 owner 的控制台上**说不出拒绝的话**——
`UnicodeEncodeError: 'gbk' codec can't encode character '\\u21d2'` 把设计好的 rc=2 降级成 rc=1＋traceback。
症状不在判据上而在**输出面**上，所以 capsys／重定向到 UTF-8 文件的任何测都看不见它。

**取法（这一条是本器唯一的实质主张）**：按"消息字符串定义处"取，不按 `print` 行取。
理由有实测出处——当天崩的那条写作 `print("  - " + row)`，字形住在被调用函数里的 f-string 常量中，
`grep 'print(.*⇒'` 那种取法只抓到 4 处／3 枚文件，是**下界**。本器分两档出版：

* `stdout_print_glyphs`：字形字面出现在**绑向 stdout** 的输出调用子树内 ⇒ 会崩的那一面，本器的判定只看它；
* `stderr_print_glyphs`：同上但落 stderr ⇒ **不崩**，只把字形印成字面量 `\\u21d2`，登记成"难看面"；
* `message_pool_glyphs`：字形出现在同文件**其它**字符串常量里（除 docstring）——它可能经赋值／拼接流向输出，
  本器不做数据流分析，所以这一档是**保守上界**，单独计数，不许代答前两者。

**为什么必须分 sink**（同一支脚本两种落点的实测，2026-10-10）：`sys.stderr.errors=backslashreplace`，
`sys.stdout.errors=surrogateescape`——后者只对孤立代理字符宽容，所以同一个 `⇒` 在 stderr 上照常出版、
在 stdout 上直接 `UnicodeEncodeError`＋rc=1。把两档相加会得到一张**错的面**：本仓真正会崩的只有 stdout 那一面。

**豁免**：文件里存在 `sys.stdout|stderr.reconfigure(encoding="utf-8")` 的调用 ⇒ 该件不计为缺陷
（这是 `eval_taiji_artifact_consumption_policy.py:336-338` 的既有写法，不是本器发明的口径）。
豁免只在 `--codec` 与真实控制台同源时才成立——本器默认 `gbk`：本机 `locale.getpreferredencoding()` 现读
`cp936`，而 `codecs.lookup("cp936").name` 就是 `gbk`（同一枚 codec，实测），且当天那条报错原文写的是
`'gbk' codec can't encode`。换机器要换参数，不许拿"绿"代答"这台机器的控制台"。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCAN_DIRS = ("scripts", "taiji", "seed", "api")
DEFAULT_CODEC = "gbk"
OUTPUT_SINKS = ("write", "writelines", "print")
SINK_RECEIVER_TOKENS = ("stdout", "stderr")
GUARD_METHOD = "reconfigure"
GUARD_ENCODING = "utf-8"


def _unencodable(text: str, codec: str) -> list[str]:
    """返回 `text` 里不能按 `codec` 编码的字符（去重、按码点排序）。"""

    bad: set[str] = set()
    for ch in text:
        try:
            ch.encode(codec)
        except UnicodeEncodeError:
            bad.add(ch)
    return sorted(bad, key=ord)


def _docstrings(tree: ast.AST) -> set[int]:
    """docstring 不是消息——它永远不会流向控制台，把它算进来会造出大量假阳性。"""

    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                out.add(id(body[0].value))
    return out


def _has_console_guard(tree: ast.AST, source: str) -> bool:
    """只认**真的**把输出流钉成 UTF-8 的调用：接收者要提到 stdout／stderr，encoding 要等于 utf-8。"""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != GUARD_METHOD:
            continue
        receiver = ast.get_source_segment(source, node.func.value) or ""
        if not any(token in receiver for token in SINK_RECEIVER_TOKENS):
            continue
        for kw in node.keywords:
            value = kw.value
            if (
                kw.arg == "encoding"
                and isinstance(value, ast.Constant)
                and value.value == GUARD_ENCODING
            ):
                return True
    return False


def _print_calls(tree: ast.AST, source: str) -> list[ast.Call]:
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            out.append(node)
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in OUTPUT_SINKS:
            receiver = ast.get_source_segment(source, node.func.value) or ""
            if any(token in receiver for token in SINK_RECEIVER_TOKENS):
                out.append(node)
    return out


def _sink_of(call: ast.Call, source: str) -> str:
    """把一支输出调用归到 `stdout`／`stderr`——**只有 stdout 会崩**，这两档不能合并。

    实测出处（2026-10-10，同一支脚本两种落点）：`sys.stderr.errors=backslashreplace` ⇒
    写到 stderr 的 `⇒` 变成字面量 `\\u21d2` 照常出版（难看但不崩）；
    `sys.stdout.errors=surrogateescape` 只对孤立代理字符宽容 ⇒ 同一个字形在 stdout 上
    直接 `UnicodeEncodeError`＋rc=1。`print()` 默认落 stdout，除非带 `file=`。
    """

    if isinstance(call.func, ast.Attribute):
        receiver = ast.get_source_segment(source, call.func.value) or ""
        if "stderr" in receiver:
            return "stderr"
        return "stdout"
    for kw in call.keywords:
        if kw.arg == "file":
            text = ast.get_source_segment(source, kw.value) or ""
            return "stderr" if "stderr" in text else "other"
    return "stdout"


def _constants_in(node: ast.AST) -> list[ast.Constant]:
    return [
        one
        for one in ast.walk(node)
        if isinstance(one, ast.Constant) and isinstance(one.value, str)
    ]


def _file_reading(path: Path, relative: str, source: str, codec: str) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(path))
    docstrings = _docstrings(tree)
    guarded = _has_console_guard(tree, source)
    console_calls = [one for one in _print_calls(tree, source) if _sink_of(one, source) != "other"]
    stdout_ids: set[int] = set()
    stderr_ids: set[int] = set()
    for call in console_calls:
        sink = _sink_of(call, source)
        bucket = stdout_ids if sink == "stdout" else stderr_ids
        bucket.update(id(one) for one in _constants_in(call))

    rows: dict[str, list[dict[str, Any]]] = {
        "stdout_print_glyphs": [],
        "stderr_print_glyphs": [],
        "message_pool_glyphs": [],
    }
    for holder in _constants_in(tree):
        if id(holder) in docstrings:
            continue
        glyphs = _unencodable(str(holder.value), codec)
        if not glyphs:
            continue
        row = {
            "line": int(holder.lineno),
            "glyphs": [f"U{ord(ch):04X}" for ch in glyphs],
        }
        if id(holder) in stdout_ids:
            rows["stdout_print_glyphs"].append(row)
        elif id(holder) in stderr_ids:
            rows["stderr_print_glyphs"].append(row)
        else:
            rows["message_pool_glyphs"].append(row)
    return {
        "file": relative,
        "prints": bool(console_calls),
        "console_guard": guarded,
        **rows,
    }


def _files(root: Path) -> list[Path]:
    out: list[Path] = []
    for name in SCAN_DIRS:
        base = root / name
        if not base.is_dir():
            continue
        out.extend(p for p in sorted(base.rglob("*.py")) if "__pycache__" not in p.parts)
    return out


def audit(root: Path, codec: str) -> dict[str, Any]:
    files = _files(root)
    if not files:
        raise ValueError(f"扫面为空：{root} 下 {'/'.join(SCAN_DIRS)} 里没有 .py ⇒ 零命中不算读数")
    readings: list[dict[str, Any]] = []
    #: 一枚解析不了的件不该让整门死掉，也不许被静默跳过——`unparsed_files` 随件出版，
    #: 且 `census_covered_every_scanned_file` 为 false 时本器不算"全清"。
    unparsed: list[dict[str, str]] = []
    for path in files:
        relative = str(path.relative_to(root)).replace("\\", "/")
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        try:
            readings.append(_file_reading(path, relative, source, codec))
        except SyntaxError as error:
            unparsed.append({"file": relative, "error": str(error)})
    unguarded = [one for one in readings if not one["console_guard"]]
    crash_files = [one for one in unguarded if one["stdout_print_glyphs"]]
    escape_files = [one for one in unguarded if one["stderr_print_glyphs"]]
    pool_files = [one for one in unguarded if one["message_pool_glyphs"]]
    total_stdout = sum(len(one["stdout_print_glyphs"]) for one in readings)
    total_stderr = sum(len(one["stderr_print_glyphs"]) for one in readings)
    total_pool = sum(len(one["message_pool_glyphs"]) for one in readings)
    return {
        "format": "taiji-console-glyph-census-v1",
        "codec": codec,
        "scan_dirs": list(SCAN_DIRS),
        "files_scanned": len(files),
        "files_parsed": len(readings),
        "unparsed_file_count": len(unparsed),
        "unparsed_files": unparsed,
        "files_printing": sum(1 for one in readings if one["prints"]),
        "files_with_console_guard": sum(1 for one in readings if one["console_guard"]),
        "constant_count_stdout_print": total_stdout,
        "constant_count_stderr_print": total_stderr,
        "constant_count_message_pool": total_pool,
        #: 三档各自成面，**不许相加**：stdout＝会崩，stderr＝只转义，pool＝没做数据流的保守上界。
        "crash_face_file_count": len(crash_files),
        "escape_face_file_count": len(escape_files),
        "unclassified_pool_face_file_count": len(pool_files),
        "crash_face": [
            {
                "file": one["file"],
                "stdout_lines": [row["line"] for row in one["stdout_print_glyphs"]],
                "stderr_lines": [row["line"] for row in one["stderr_print_glyphs"]],
                "pool_lines": [row["line"] for row in one["message_pool_glyphs"]],
                "glyphs": sorted(
                    {
                        g
                        for key in (
                            "stdout_print_glyphs",
                            "stderr_print_glyphs",
                            "message_pool_glyphs",
                        )
                        for row in one[key]
                        for g in row["glyphs"]
                    }
                ),
            }
            for one in crash_files
        ],
        "readings": readings,
        "verdicts": {
            "no_unguarded_stdout_glyph_print": not crash_files,
            "census_covered_every_scanned_file": not unparsed,
        },
        "reading_limit": (
            "static AST only: stdout face is literal constants inside a stdout-bound print (a "
            "message assembled at runtime is NOT counted there), pool is an upper bound with no "
            "data-flow analysis, and the exemption tracks a source-level reconfigure call rather "
            "than a verified console. stderr is reported separately because it escapes instead of "
            "raising (measured: sys.stderr.errors=backslashreplace, sys.stdout.errors=surrogateescape)"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="console-glyph encodability census (static, zero-compute)"
    )
    parser.add_argument("--out-report", type=Path, required=True, help="where to write the JSON")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="tree to scan")
    parser.add_argument(
        "--codec",
        default=DEFAULT_CODEC,
        help="console codec to test against (default: the locale codec on this machine)",
    )
    args = parser.parse_args(argv)

    #: 本器自己的输出也必须在这控制台上说得出话——正文一律 ensure_ascii，字形只以 U+码点出现。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = audit(args.root, args.codec)
    except (FileNotFoundError, ValueError, SyntaxError) as error:
        args.out_report.write_text(
            json.dumps(
                {
                    "format": "taiji-console-glyph-census-v1",
                    "status": "census_failed",
                    "error": str(error),
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps({"status": "census_failed"}, ensure_ascii=True))
        return 2
    args.out_report.write_text(
        json.dumps({"status": "ok", **payload}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "files_scanned": payload["files_scanned"],
                "files_printing": payload["files_printing"],
                "files_with_console_guard": payload["files_with_console_guard"],
                "crash_face_file_count": payload["crash_face_file_count"],
                "escape_face_file_count": payload["escape_face_file_count"],
                "unclassified_pool_face_file_count": payload["unclassified_pool_face_file_count"],
                "constant_count_stdout_print": payload["constant_count_stdout_print"],
                "constant_count_message_pool": payload["constant_count_message_pool"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

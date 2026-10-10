"""控制台**可编码字形**普查（DEBT-G81 修法③，静态、零算力、只读）。

**为什么要有它**：㊵-648 那次实测里，一台 fail-closed 的仪器在 owner 的控制台上**说不出拒绝的话**——
`UnicodeEncodeError: 'gbk' codec can't encode character '\\u21d2'` 把设计好的 rc=2 降级成 rc=1＋traceback。
症状不在判据上而在**输出面**上，所以 capsys／重定向到 UTF-8 文件的任何测都看不见它。

**取法（这一条是本器唯一的实质主张）**：按"消息字符串定义处"取，不按 `print` 行取。
理由有实测出处——当天崩的那条写作 `print("  - " + row)`，字形住在被调用函数里的 f-string 常量中，
`grep 'print(.*⇒'` 那种取法只抓到 4 处／3 枚文件，是**下界**。本器分两档出版：

* `stdout_print_glyphs`：字形字面出现在**绑向 stdout** 的输出调用子树内 ⇒ 会崩的一面；
* `argparse_surface_glyphs`：字形住在 `ArgumentParser(description=/epilog=)` 或 `add_argument(help=)`
  的字符串常量里——**这一档是被一支既有守卫逼出来的**（㊵-651）：`tests/taiji_native/test_a30_probe_help_runs.py`
  长红，红因是 `probe_taiji_a30_stop_failure.py --help` 在 GBK 上崩，而那枚 `⇒` 住在
  `ArgumentParser(description=__doc__)`（:1056）所引的**模块 docstring** 里 ⇒ docstring 算不算消息，
  取决于它有没有被喂进用法屏，不能一律排除；
* `stderr_print_glyphs`：同上但落 stderr ⇒ **不崩**，只把字形印成字面量 `\\u21d2`，登记成"难看面"；
* `message_pool_glyphs`：字形出现在同文件**其它**字符串常量里——它可能经赋值／拼接流向输出，
  本器不做数据流分析，所以这一档是**保守上界**（里面同时装着异常消息与只进文件的消息），
  单独计数，不许代答前三者。

**两档分开判，不并成一个"崩溃面"**：`crash_face` 只看 `stdout_print_glyphs`（正常跑就会崩）；
用法屏单列 `help_face`——它只在 `--help` 这条路写 stdout，而 `parser.error` 的 usage/help 落 **stderr**
（⇒ 转义不崩）。㊵-651 的第一版把两档并起来，读数从 46 枚跳到 108 枚：那是一张更吓人也更错的面，
已按实测拆开。

**为什么必须分 sink**（同一支脚本两种落点的实测，2026-10-10）：`sys.stderr.errors=backslashreplace`，
`sys.stdout.errors=surrogateescape`——后者只对孤立代理字符宽容，所以同一个 `⇒` 在 stderr 上照常出版、
在 stdout 上直接 `UnicodeEncodeError`＋rc=1。把两档相加会得到一张**错的面**：真正会崩的只有写 stdout 那两面。

**豁免**：文件里存在 `sys.stdout|stderr.reconfigure(encoding="utf-8")` 的调用 ⇒ 该件不计为缺陷
（这是 `eval_taiji_artifact_consumption_policy.py:336-338` 的既有写法，不是本器发明的口径）。
豁免只在 `--codec` 与真实控制台同源时才成立——本器默认 `gbk`：本机 `locale.getpreferredencoding()` 现读
`cp936`，而 `codecs.lookup("cp936").name` 就是 `gbk`（同一枚 codec，实测），且当天那条报错原文写的是
`'gbk' codec can't encode`。换机器要换参数，不许拿"绿"代答"这台机器的控制台"。

**`live`／`archived` 分档（㊵-668，DEBT-G81 的收尾更正；形状＝追加键，旧键一字未动）**：
标题数 `crash_face_file_count` 原先把**死码与活码混在一张面上**数，读出来是 43 枚，而按目录分组现读
**42 枚住 `scripts/archive/`、1 枚住 `scripts/training/utils.py`（是一枚库）** ⇒ 那 42 枚永远不会在这台
机器上被跑到，给它们灌守卫是把死码刷绿的白工；真正该排的是活的那 1 枚。分档不删读数、不相加、不改判：
本器**不出版**"两档之和＝整面"这类键（同一批行对象分两次数，结构上不可能为 false ⇒ 不配当证据，
见 ㊵-618 对 `shadow_materialized` 的更正），等式只由测来钉：`tests/taiji_native/
test_n5_17_console_glyph_census_contract.py` 里按 `bucket` 字段对 `crash_face` 行重数一遍再比计数。
两条旧判定（`no_unguarded_stdout_glyph_print`／`no_unguarded_argparse_help_glyph`）**照旧按整面判**，
新增的 `no_unguarded_live_stdout_glyph_print` 只是把同一件事按活码面再问一遍——旧判定为 false 时
不许拿它代答"已清"。
`scripts/legacy/` **故意不划进归档档**：本器只能按路径名分，没有任何一枚读数证明它没被活码 import；
把未证的目录划成"死码"就是给自己造豁免（对照 `--codec` 那段：豁免必须有出处）。
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
ARGPARSE_CONSTRUCTOR = "ArgumentParser"
ADD_ARGUMENT_METHOD = "add_argument"
DUAL_DOC_NAME = "__doc__"
ARGPARSE_KWARGS = ("description", "epilog", "help")
#: ㊵-668：只有这一枚前缀算"归档＝这台机器跑不到"。`scripts/legacy/` 不在列——
#: 没有任何读数证明它没被活码 import，按名字给它豁免就是自造豁免。
ARCHIVED_PREFIXES = ("scripts/archive/",)


def _bucket(relative: str) -> str:
    return "archived" if relative.startswith(ARCHIVED_PREFIXES) else "live"


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
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
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


def _parser_names(node: ast.AST) -> bool:
    """`argparse.ArgumentParser(...)`／`ArgumentParser(...)` 两种写法都算构造用法屏。"""

    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == ARGPARSE_CONSTRUCTOR
    return isinstance(func, ast.Name) and func.id == ARGPARSE_CONSTRUCTOR


def _argparse_constants(tree: ast.AST) -> set[int]:
    """由 **argparse 库自己**打到 stdout 的常量：`description=`／`epilog=`／每条 `help=`。

    这一档不是补装饰——㊵-651 是一支**既有守卫**抓住的形状：`probe_taiji_a30_stop_failure.py --help`
    在 GBK 控制台上崩，而它的 `⇒` 既不在 print 子树里、也不是我原先排除的"无害 docstring"，
    它住在 `ArgumentParser(description=__doc__)`（:1056）所引的模块 docstring 里。
    ⇒ 结论：**docstring 是否算消息，取决于它有没有被喂进用法屏**，所以这里要把
    `__doc__` 那一条线单独接进来，而不是把 docstring 一律排除。
    """

    found: set[int] = set()
    module_doc: ast.Constant | None = None
    body = getattr(tree, "body", [])
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        module_doc = body[0].value

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _parser_names(node):
            pairs = ("description", "epilog")
        elif isinstance(node.func, ast.Attribute) and node.func.attr == ADD_ARGUMENT_METHOD:
            pairs = ("help",)
        else:
            continue
        for kw in node.keywords:
            if kw.arg not in pairs:
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                found.add(id(kw.value))
            elif (
                isinstance(kw.value, ast.Name)
                and kw.value.id == DUAL_DOC_NAME
                and module_doc is not None
            ):
                found.add(id(module_doc))
    return found


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
    argparse_ids = _argparse_constants(tree)

    rows: dict[str, list[dict[str, Any]]] = {
        "stdout_print_glyphs": [],
        "stderr_print_glyphs": [],
        "argparse_surface_glyphs": [],
        "message_pool_glyphs": [],
    }
    for holder in _constants_in(tree):
        #: docstring 只有在**没被喂进用法屏**时才是"不是消息"（`description=__doc__` 那条例外）。
        if id(holder) in docstrings and id(holder) not in argparse_ids:
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
        elif id(holder) in argparse_ids:
            rows["argparse_surface_glyphs"].append(row)
        else:
            rows["message_pool_glyphs"].append(row)
    return {
        "file": relative,
        "bucket": _bucket(relative),
        "prints": bool(console_calls),
        "has_argparse": any(_parser_names(one) for one in ast.walk(tree)),
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
    #: `crash_face` 只算 print 到 stdout（正常跑就会崩的那一面）；用法屏单列一档，
    #: 因为它只在 `--help` 这条路写 stdout——`parser.error` 打的 usage/help 落 **stderr**，
    #: 而 stderr 是 backslashreplace ⇒ 不崩。把两档并起来会把数从 46 抬到 108，
    #: 那是一张"更吓人也更错"的面（㊵-651 的第一版就并错了，见本件 docstring）。
    crash_files = [one for one in unguarded if one["stdout_print_glyphs"]]
    help_files = [one for one in unguarded if one["argparse_surface_glyphs"]]
    escape_files = [one for one in unguarded if one["stderr_print_glyphs"]]
    pool_files = [one for one in unguarded if one["message_pool_glyphs"]]
    #: ㊵-668：同一档的 live／archived 拆分。用**同一批行对象**数，不用第二次筛选条件，
    #: 否则"分区求和＝整面"这条自证就只是我把同一个数抄了两遍。
    crash_live = [one for one in crash_files if one["bucket"] == "live"]
    crash_archived = [one for one in crash_files if one["bucket"] == "archived"]
    help_live = [one for one in help_files if one["bucket"] == "live"]
    help_archived = [one for one in help_files if one["bucket"] == "archived"]
    total_stdout = sum(len(one["stdout_print_glyphs"]) for one in readings)
    total_stderr = sum(len(one["stderr_print_glyphs"]) for one in readings)
    total_argparse = sum(len(one["argparse_surface_glyphs"]) for one in readings)
    total_pool = sum(len(one["message_pool_glyphs"]) for one in readings)
    return {
        "format": "taiji-console-glyph-census-v3",
        "codec": codec,
        "scan_dirs": list(SCAN_DIRS),
        "files_scanned": len(files),
        "files_parsed": len(readings),
        "unparsed_file_count": len(unparsed),
        "unparsed_files": unparsed,
        "files_printing": sum(1 for one in readings if one["prints"]),
        "files_with_argparse": sum(1 for one in readings if one["has_argparse"]),
        "files_with_console_guard": sum(1 for one in readings if one["console_guard"]),
        "constant_count_stdout_print": total_stdout,
        "constant_count_stderr_print": total_stderr,
        "constant_count_argparse_surface": total_argparse,
        "constant_count_message_pool": total_pool,
        #: 四档各自成面，**不许相加**：stdout 与用法屏＝会崩，stderr＝只转义，pool＝保守上界。
        "crash_face_file_count": len(crash_files),
        "help_face_file_count": len(help_files),
        "escape_face_file_count": len(escape_files),
        "unclassified_pool_face_file_count": len(pool_files),
        #: ㊵-668 追加键：标题数把死码与活码混着数，这两组把同一张面按"这台机器跑不跑得到"分开。
        "crash_face_live_file_count": len(crash_live),
        "crash_face_archived_file_count": len(crash_archived),
        "help_face_live_file_count": len(help_live),
        "help_face_archived_file_count": len(help_archived),
        "archived_prefixes": list(ARCHIVED_PREFIXES),
        "crash_face_live": [one["file"] for one in crash_live],
        "help_face_live": [one["file"] for one in help_live],
        "crash_face": [
            {
                "file": one["file"],
                "bucket": one["bucket"],
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
        "help_face": [
            {
                "file": one["file"],
                "bucket": one["bucket"],
                "argparse_lines": [row["line"] for row in one["argparse_surface_glyphs"]],
                "glyphs": sorted(
                    {g for row in one["argparse_surface_glyphs"] for g in row["glyphs"]}
                ),
            }
            for one in help_files
        ],
        "readings": readings,
        "verdicts": {
            "no_unguarded_stdout_glyph_print": not crash_files,
            "no_unguarded_argparse_help_glyph": not help_files,
            "census_covered_every_scanned_file": not unparsed,
            #: 追加判定：整面旧判定**不因此翻绿**，不许拿这一枚代答"已清"。
            "no_unguarded_live_stdout_glyph_print": not crash_live,
        },
        "reading_limit": (
            "static AST only. `crash_face` counts literals inside a stdout-bound print/write call "
            "— that is the face that raises on a normal run. `help_face` counts literals argparse "
            "itself prints (description=/epilog=/help=, plus the module docstring when passed as "
            "__doc__): it raises only on the `--help` path, because `parser.error` writes usage to "
            "stderr and stderr escapes rather than raising (measured: "
            "sys.stderr.errors=backslashreplace, sys.stdout.errors=surrogateescape). message_pool "
            "is an upper bound with no data-flow analysis, so it also holds exception messages and "
            "file-only text and must not be read as a defect count. The exemption tracks a "
            "source-level reconfigure call, not a verified console. The live/archived split (added "
            "2026-10-10) is a path-prefix bucket over the SAME row set, so it relocates a file "
            "between two lists without changing any face: scripts/archive is 'not reachable on this "
            "machine', nothing else is exempted (scripts/legacy is counted live because no reading "
            "shows it unimported), and the pre-existing whole-face verdicts stay false when the "
            "archived files stay unguarded — the live-only verdict must not be read as 'cleared'."
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
                    "format": "taiji-console-glyph-census-v3",
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
                "crash_face_live_file_count": payload["crash_face_live_file_count"],
                "crash_face_archived_file_count": payload["crash_face_archived_file_count"],
                "help_face_live_file_count": payload["help_face_live_file_count"],
                "help_face_archived_file_count": payload["help_face_archived_file_count"],
                "help_face_file_count": payload["help_face_file_count"],
                "escape_face_file_count": payload["escape_face_file_count"],
                "unclassified_pool_face_file_count": payload["unclassified_pool_face_file_count"],
                "constant_count_stdout_print": payload["constant_count_stdout_print"],
                "constant_count_argparse_surface": payload["constant_count_argparse_surface"],
                "constant_count_message_pool": payload["constant_count_message_pool"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

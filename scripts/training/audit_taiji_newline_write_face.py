"""写靶行尾普查仪（DEBT-G115 修法① 的前置取证；㊵-762 落库）。

债表原句给的是**文件级粗筛上界 82 枚**（"文件里出现 `reports/` 且 `write_text(` 处数多于 `newline=` 处数"）。
那把尺量的是"这个文件提没提过 reports/"，量的不是"这一处写入落到哪"。本仪器把判定改到**调用点**上，
并明确分成三桶，各自说清自己能证明什么：

* **A 桶＝能静态认出写靶、且靶是 `reports/` 字面路径、却没带 `newline=` 的调用**。这一桶是**唯一**能直接
  当门用的一桶：基线 0，任何新增就红。现读＝**0 处／0 枚文件**（㊵-762，扫 1632 个已跟踪 `.py`）。
* **B 桶＝写靶是个变量/表达式，静态读不出落点**（含 `x.write_text(payload)` 里 `x` 只是个名字的那种）。
  这一桶**不判**、只出版：现读 **977 处／688 枚文件**——那才是"先逐枚现读确认再动"这句话的真实工作量，
  也是我不许任何人（包括我自己）拿 A 桶的 0 去说"已经没有待办"的理由。
* **C 桶＝写靶认得出但不含 `reports/`**（临时件、`output/`、别人目录）：只出总数。

取法上有个坑必须写在件里：`x.write_text(data)` 的 **args[0] 是内容、路径是接收者 `x`**。
第一版探针按 args[0] 取靶，于是把 payload 里那句 `"evidence": "reports/…json"` 当成了写靶，
造出一枚**假阳性**（点名了 `run_taiji_unified_entry_evidence.py`，而它那一处写的根本不是 `reports/`）。
本仪器按接收者取靶，并对 `open(...)` 的文本档/二进制档分开处理。

零命中一律响亮失败：面为空说明 `--face`/`--root` 指错了地方，那不是"已规范化"（DEBT-G91／G58 同族）。
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: ㊵-765：读出面加了 `d_face`（待办收窄那一桶）⇒ 按"面变了就升版"的规矩走 v2；
#: 同名件的 v1 那份不删，它的历史在 git 里，引用时要点名是哪一版。
#: ㊵-789：读出面再加 `helper_face`（DEBT-G115 修法③ 的统一落盘助手被采用了几处）⇒ 同样是面变宽，走 v3。
#: 助手侧的写入口天生带行尾参数，所以它不参与"缺 `newline=`"这一判定，只出版被采用的次数——
#: 队列的可度量性靠这一个数，不靠我的叙述。
HELPER_WRITE_NAMES = ("write_text_report", "write_json_report")
REPORT_FORMAT = "taiji-newline-write-face-v3"
DEBT = "DEBT-G115"
TARGET_MARKER = "reports/"
TEXT_WRITE_METHODS = ("write_text", "writelines")
#: A 桶基线＝0（现读，㊵-762）。要往下调只能是"改了码"，要往上加就是门该红的时候。
A_BASELINE = 0
#: 已知 parse 不动的三枚归档脚本（`scripts/archive/legacy_*`，现读 ㊵-762）。
#: 它们是本仪器**看不到**的面：A 桶那句"0 处"是"在扫得到的面上 0 处"，不是"全仓 0 处"。
#: 判据写成子集——第四枚解析不动就必须红，而谁把它们修到能 parse 只是让基线缩小，不为红。
PARSE_BASELINE = (
    "scripts/archive/legacy_convert_dense_model_format.py",
    "scripts/archive/legacy_expand_lifeform_data.py",
    "scripts/archive/legacy_fill_general_corpus.py",
)


def _string_fragments(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    return [
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    ]


def _is_binary_open(node: ast.Call) -> bool:
    mode = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    return isinstance(mode, str) and "b" in mode


def _open_is_text_write(node: ast.Call) -> bool:
    mode = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    if mode is None:
        # `open(p)` 的默认档是 "r"＝只读，**不是**写入。第一版这里返回 True，
        # 于是把一堆读件的地方算成"缺 newline= 的写靶"（虚高），而注释写的恰恰是另一件。
        return False
    return isinstance(mode, str) and any(token in mode for token in ("w", "a", "+"))


def target_fragments(node: ast.Call) -> tuple[list[str], bool]:
    """这一处写入的**落点**表达式里能认出什么。（片段, 是否只认得出名字）"""

    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in TEXT_WRITE_METHODS:
        # 关键：args[0] 是**内容**，路径是接收者。取错这一层就会把 payload 里的路径串当成写靶。
        fragments = _string_fragments(func.value)
        return fragments, not fragments
    if isinstance(func, ast.Attribute) and func.attr == "open":
        fragments = _string_fragments(node.args[0] if node.args else None)
        return fragments, not fragments
    if isinstance(func, ast.Name) and func.id == "open":
        fragments = _string_fragments(node.args[0] if node.args else None)
        return fragments, not fragments
    return [], True


def has_reports_default(tree: ast.AST) -> bool:
    """这一枚文件里有没有"argparse 的 `default=` 字符串含 `reports/`"。

    这是 D 桶的前提：写靶是变量时静态读不出落点，但**如果这台仪器的 `--out-report` 默认值就落在
    `reports/`，那它的写靶变量极可能就是那枚件**。D 桶不判、不结清任何东西，它只把债表里
    "先逐枚现读确认它真写 reports/"这句话的**待办面从 1210 处收窄到可数的一小撮文件**。
    """

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "default":
            for text in _string_fragments(node.value):
                if TARGET_MARKER in text:
                    return True
    return False


def classify_source(source: str) -> dict[str, list[int]]:
    tree = ast.parse(source)
    buckets: dict[str, list[int]] = {"a": [], "b": [], "c": [], "helper": []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = getattr(node.func, "attr", None) or getattr(node.func, "id", None) or ""
        if called in HELPER_WRITE_NAMES:
            #: 走助手的写入：行尾由助手负责，这里只数"被采用了几处"。
            buckets["helper"].append(node.lineno)
            continue
        if called in TEXT_WRITE_METHODS:
            pass
        elif called == "open":
            if _is_binary_open(node) or not _open_is_text_write(node):
                continue
        else:
            continue
        if any(kw.arg == "newline" for kw in node.keywords):
            continue
        fragments, unresolved = target_fragments(node)
        joined = " ".join(fragments)
        if unresolved:
            buckets["b"].append(node.lineno)
        elif TARGET_MARKER in joined:
            buckets["a"].append(node.lineno)
        else:
            buckets["c"].append(node.lineno)
    return buckets


def tracked_python(root: Path, face: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "--", face],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed rc={proc.returncode}: {proc.stderr.strip()[:120]}")
    return sorted(line for line in proc.stdout.splitlines() if line.endswith(".py"))


def scan(root: Path, face: str) -> dict[str, Any]:
    paths = tracked_python(root, face)
    if not paths:
        raise ValueError(f"零个已跟踪 .py 落在 {face!r}：先确认 --face/--root 指的是哪一面")
    a: dict[str, list[int]] = {}
    b: dict[str, list[int]] = {}
    d: dict[str, list[int]] = {}
    helper: dict[str, list[int]] = {}
    c_sites = 0
    unparsed: list[str] = []
    for name in paths:
        try:
            source = (root / name).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            unparsed.append(name)
            continue
        buckets = classify_source(source)
        if buckets["a"]:
            a[name] = buckets["a"]
        if buckets["b"]:
            b[name] = buckets["b"]
            if has_reports_default(tree):
                d[name] = buckets["b"]
        c_sites += len(buckets["c"])
        if buckets["helper"]:
            helper[name] = buckets["helper"]
    a_total = sum(len(lines) for lines in a.values())
    verdicts = {
        "face_is_not_empty": bool(paths),
        "no_unparseable_file_beyond_the_known_three": set(unparsed) <= set(PARSE_BASELINE),
        "a_face_at_or_below_baseline": a_total <= A_BASELINE,
    }
    return {
        "format": REPORT_FORMAT,
        "debt": DEBT,
        "face": face,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tracked_python_total": len(paths),
        "unparsed_files": unparsed,
        "parse_baseline": list(PARSE_BASELINE),
        "a_face": {"sites": a_total, "files": {name: a[name] for name in sorted(a)}},
        "b_face": {
            "sites": sum(len(lines) for lines in b.values()),
            "files": len(b),
            "note": "写靶是变量/表达式，静态读不出落点——这一桶不判，是修法① 的真实待办面",
        },
        "d_face": {
            "sites": sum(len(lines) for lines in d.values()),
            "files": {name: d[name] for name in sorted(d)},
            "note": (
                "D ⊆ B：写靶静态读不出，但这一枚文件的 argparse 默认落点自己就指向 reports/——"
                "修法① 那句「先逐枚现读确认它真写 reports/」的待办面是这一撮，不是整个 B 桶。"
                "D 不入 verdict：它要人（或下一批）逐枚把接收者对齐到那枚默认值。"
            ),
        },
        "c_face": {
            "sites": c_sites,
            "note": "写靶认得出但不含 reports/（临时件、output/、他人目录）",
        },
        "helper_face": {
            "sites": sum(len(lines) for lines in helper.values()),
            "files": {name: helper[name] for name in sorted(helper)},
            "module": "scripts/training/_taiji_report_io.py",
            "note": (
                "经统一落盘助手（DEBT-G115 修法③）写出去的调用。与 D 同理**不入 verdict**："
                "它数的是「已经被收口到一处」的次数，把它算红等于宣称其余写靶都已收口，"
                "而 B 桶的逐枚现读正是没做的事。"
            ),
        },
        "verdicts": verdicts,
        "verdict": "holds" if all(verdicts.values()) else "not_holds",
        "failing_verdicts": sorted(key for key, ok in verdicts.items() if not ok),
        "reading_limit": (
            "A 桶＝能证明写靶是 reports/ 字面路径且缺 newline= 的调用，现读 0 处；"
            "这**不等于**全仓写靶都带上了 newline=。B 桶是写靶为变量／表达式、静态读不出落点的那批，"
            "把它们说成已处理需要逐枚现读——修法① 的真实待办面就是 B 桶的规模。"
            "债表粗筛那句「82 枚文件」与本件不是同一把尺：它数的是提过 reports/ 的文件，"
            "本件数的是这一处写入落到哪。另有三枚 `scripts/archive/legacy_*` parse 不动"
            "（在 `parse_baseline` 里点名），所以 A 桶那句是「扫得到的面上 0 处」，不是「全仓 0 处」。"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按调用点普查文本写靶是否带 newline=（A 桶可判、B 桶不可判，分开出版）"
    )
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="git 仓根（契约测指向临时仓）")
    parser.add_argument("--face", default=".", help="按版本控制取的面（默认整仓）")
    parser.add_argument(
        "--out-report", required=True, help="落盘路径（已存在则拒覆盖 rc=2，除非 --overwrite）"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    root = Path(args.root).resolve()
    try:
        report = scan(root, args.face)
    except (RuntimeError, ValueError) as error:
        print(f"REJECT {error}", file=sys.stderr)
        return 2
    report["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 3)

    out = Path(args.out_report)
    if not out.is_absolute():
        out = root / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    summary = {
        "verdict": report["verdict"],
        "failing": report["failing_verdicts"],
        "tracked_python_total": report["tracked_python_total"],
        "a_sites": report["a_face"]["sites"],
        "a_files": len(report["a_face"]["files"]),
        "b_sites": report["b_face"]["sites"],
        "d_sites": report["d_face"]["sites"],
        "d_files": len(report["d_face"]["files"]),
        "c_sites": report["c_face"]["sites"],
        "out": out.as_posix(),
    }
    sys.stdout.buffer.write(json.dumps(summary, ensure_ascii=True).encode("ascii") + b"\n")
    return 0 if report["verdict"] == "holds" else 1


if __name__ == "__main__":
    raise SystemExit(main())

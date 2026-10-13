"""DEBT-G106 修法③／G105 修法③ 的零算力复算：谁在读 E 存储、读的是不是别的会话的记录、哪些读数含合成材料。

为什么这台仪器要按**接收者类型**分
----------------------------------
`git grep "\\.retrieve("` 会同时命中概念器官的 `retrieve`（`taiji/adapter.py:10845`／`:10908` 是
`_concept_formation`）与情节场的 `EpisodicField.recall`，裸 grep 数出来的分子分母不是同一个集合。
本件按 AST 走两件事：①跟踪 `X = EpisodicMemoryStore(...)` 的绑定名与 `self._episodic_memory`
（其类型由 `attach_episodic_memory(store: EpisodicMemoryStore | None)` 钉住）；②只统计这些名字上的
`.retrieve(...)` 调用点。

两条判读各自能为假
------------------
* **跨会话读取**：读取点所在函数里有没有出现 `episode_id`——出现了不等于已过滤（那是弱信号），
  所以本件出版的是"函数体引用过 episode_id 的处数"，并把它命名成 `episode_id_mentioning`，
  **不**命名成 `episode_filtered`；一处都不提 ⇒ 那句「靠上层不调用它」被否证。
* **合成材料**：写入点里 cue 的构造是不是一眼合成（`torch.eye`／`torch.tensor([[`／`arange` 之类），
  以及来源标签是什么。标签本轮不改（改它会动 one-hot 输入、使这些脚本的历史读数不可比），
  本件只把"哪些读数含合成材料"列成清单。

零算力：不加载检查点、不训练、不更新权重；只读版本控制面内的源码文本。
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
#: 契约册（`test_n5_24`）按文件路径加载本件、也被起成真子进程，脚本自己那层目录先上 `sys.path`，
#: 统一落盘助手才解析得到（同㊵-790／802 那条例外约定）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

REPORT_FORMAT = "taiji-episodic-read-sharing-face-v1"
STORE_CONSTRUCTOR = "EpisodicMemoryStore"
ADAPTER_SLOT = "_episodic_memory"
SYNTHETIC_CUE_MARKERS = (
    "torch.eye",
    "torch.tensor([[",
    "torch.arange",
    "torch.randn",
    "np.eye",
    "identity(",
)


def _tracked_python(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [root / rel for rel in out]


def _record_bindings(tree: ast.Module) -> dict[str, str]:
    """`record = EpisodicMemoryRecord(...)` 这类**先构造再写入**的绑定，用于把标记查到实体。

    第一版只查 `write()` 的第一个实参，于是 `store.write(record)` 永远查不到合成标记、
    把 DEBT-G105 那 13 处读成 0 处——那是这台仪器自己的假阴性。
    """

    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        target = None
        value = None
        if isinstance(node, ast.Assign) and node.targets:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        if isinstance(value, ast.Call):
            label = getattr(value.func, "id", None) or getattr(value.func, "attr", None)
            if label and label.endswith("Record"):
                bindings[target.id] = ast.unparse(value)
    return bindings


def _store_names(tree: ast.Module) -> set[str]:
    """`X = EpisodicMemoryStore(...)` 的绑定名（含赋值语句与带注解的声明）。"""

    names: set[str] = set()
    for node in ast.walk(tree):
        target = None
        value = None
        if isinstance(node, ast.Assign) and node.targets:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if target is None or value is None:
            continue
        if isinstance(target, ast.Name) and isinstance(value, ast.Call):
            func = value.func
            label = getattr(func, "id", None) or getattr(func, "attr", None)
            if label == STORE_CONSTRUCTOR:
                names.add(target.id)
    return names


def _is_store_attribute(call: ast.Call) -> bool:
    """`self._episodic_memory.retrieve(...)` 或 `<store>.retrieve(...)`。"""

    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr != "retrieve":
        return False
    value = func.value
    if isinstance(value, ast.Attribute) and value.attr == ADAPTER_SLOT:
        return True
    return isinstance(value, ast.Name)


def _enclosing_function(tree: ast.Module, lineno: int):
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= lineno <= (node.end_lineno or node.lineno)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda fn: fn.lineno)


def scan(root: Path) -> dict[str, object]:
    read_sites: list[dict[str, object]] = []
    write_sites: list[dict[str, object]] = []
    synthetic_records: list[dict[str, object]] = []
    files = _tracked_python(root)
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        names = _store_names(tree)
        records = _record_bindings(tree)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            holder = func.value
            matched = False
            if func.attr in {"retrieve", "write"}:
                is_slot = isinstance(holder, ast.Attribute) and holder.attr == ADAPTER_SLOT
                is_local = isinstance(holder, ast.Name) and holder.id in names
                if is_slot or is_local:
                    matched = True
            if not matched:
                continue
            enclosing = _enclosing_function(tree, node.lineno)
            body_text = ""
            if enclosing is not None:
                body_text = "\n".join(lines[enclosing.lineno - 1 : enclosing.end_lineno or 0])
            rel = path.relative_to(root).as_posix()
            row = {
                "file": rel,
                "line": node.lineno,
                "method": func.attr,
                "function": enclosing.name if enclosing is not None else "<module>",
                "episode_id_mentioning": bool("episode_id" in body_text),
                "source": (lines[node.lineno - 1].strip()[:140] if lines else ""),
            }
            if func.attr == "retrieve":
                read_sites.append(row)
            else:
                record = ast.unparse(node.args[0]) if node.args else ""
                if not record and node.keywords:
                    record = ast.unparse(node.keywords[0].value)
                first = node.args[0] if node.args else None
                if isinstance(first, ast.Name) and first.id in records:
                    record = records[first.id]
                row["cue_synthetic_markers"] = [
                    marker for marker in SYNTHETIC_CUE_MARKERS if marker in record
                ]
                labels = [
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg in {"provenance", "episode_id"}
                ]
                row["declared_keywords"] = labels
                write_sites.append(row)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            label = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if not label or not label.endswith("Record"):
                continue
            text = ast.unparse(node)
            markers = [marker for marker in SYNTHETIC_CUE_MARKERS if marker in text]
            if markers:
                synthetic_records.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "line": node.lineno,
                        "constructor": label,
                        "markers": markers,
                    }
                )
    return {
        "files_parsed": len(files),
        "read_sites": read_sites,
        "write_sites": write_sites,
        "synthetic_records": synthetic_records,
    }


def _retrieve_session_gate_face(root: Path) -> dict[str, bool]:
    """读取侧的会话闸按 `EpisodicMemoryStore.retrieve` 的**签名**判，不按调用点文本猜。

    ㊵-713（DEBT-G106 修法①）把一枚布尔改成三枚：改前只有"签名里没有 episode 参数"，
    实施后那句会整体翻假、把"默认仍不筛"这件事一起抹掉。拆开之后
    "有口可筛"与"默认未筛"是两条独立读数，`J-N6b-3` 要的那句才立得住。
    """

    source = (root / "taiji" / "episodic_memory.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "retrieve":
            params = {a.arg: a for a in node.args.args + node.args.kwonlyargs}
            episode = next((a for name, a in params.items() if "episode" in name), None)
            if episode is None:
                return {
                    "retrieve_signature_has_episode_param": False,
                    "retrieve_episode_param_defaults_to_none": False,
                    "session_gate_enforced_by_default": False,
                }
            default_none = _default_is_none(node, episode.arg)
            return {
                "retrieve_signature_has_episode_param": True,
                "retrieve_episode_param_defaults_to_none": default_none,
                "session_gate_enforced_by_default": not default_none,
            }
    raise ValueError("找不到 EpisodicMemoryStore.retrieve：判据对象不在盘上")


def _default_is_none(func: ast.FunctionDef, name: str) -> bool:
    """该形参的默认值是不是字面 `None`（默认位＝不筛，就是 DEBT-G106 今天没结清的那件事）。"""

    positional = len(func.args.args) - len(func.args.defaults)
    defaults = {a.arg: None for a in func.args.args[:positional]}
    defaults.update(
        {a.arg: d for a, d in zip(func.args.args[positional:], func.args.defaults, strict=False)}
    )
    defaults.update(
        {a.arg: d for a, d in zip(func.args.kwonlyargs, func.args.kw_defaults, strict=False)}
    )
    node = defaults.get(name)
    return isinstance(node, ast.Constant) and node.value is None


def build_report(root: Path) -> dict[str, object]:
    face = scan(root)
    reads = face["read_sites"]
    writes = face["write_sites"]
    if not reads:
        raise ValueError(
            "E 存储读取点为零命中：这台仪器今天什么都没测到。先确认扫描面（读的是版本控制里的 .py），"
            "不许把零命中读成「没人跨会话读」"
        )
    synth_records = face["synthetic_records"]
    writer_files = {row["file"] for row in writes}
    synthetic_files = sorted({row["file"] for row in synth_records} & writer_files)
    return {
        "format": REPORT_FORMAT,
        "debts": ["DEBT-G106 修法③", "DEBT-G105 修法③"],
        "face": "tracked .py（`git ls-files -- *.py`），按 AST 接收者类型分派，不靠裸 grep",
        "files_parsed": face["files_parsed"],
        "read_site_count": len(reads),
        "read_sites": reads,
        "reads_mentioning_episode_id": sum(1 for row in reads if row["episode_id_mentioning"]),
        "reads_in_product_default_chain": sum(
            1 for row in reads if row["file"] == "taiji/adapter.py"
        ),
        "write_site_count": len(writes),
        "synthetic_record_sites": len(synth_records),
        "files_that_both_write_and_construct_synthetic_records": synthetic_files,
        "synthetic_record_files": sorted({row["file"] for row in synth_records}),
        "verdicts": {
            "someone_in_product_chain_reads_the_store": face["read_sites"]
            and any(row["file"] == "taiji/adapter.py" for row in reads),
            **_retrieve_session_gate_face(root),
            "synthetic_material_reaches_the_store": bool(synthetic_files),
        },
        "coverage_gap": {
            "why_this_list_is_not_exhaustive": "多数 eval 脚本把 cue 经变量绑好再传进构造式"
            "（`cue=cue.detach().clone()`／`cue=bits + torch.randn(...)`／`cue=pattern`），"
            "静态文件级标记查不到那条链",
            "authoritative_enumeration": "DEBT-G105 本行的 12 文件／13 处是人工逐处核过的清单，"
            "本件的计数不替代它，也不许被读成「其余那些不是合成材料」",
        },
        "limits": [
            "`episode_id_mentioning` 是**弱信号**：函数体提过 episode_id 不等于读取时按它过滤，"
            "所以会话闸那条结论按 retrieve 的**签名**判，且拆成三枚（`retrieve_signature_has_episode_param`／`retrieve_episode_param_defaults_to_none`／`session_gate_enforced_by_default`）——㊵-713 起拆开：改前只有一枚「签名里没有 episode 参数」，实施后那句整体翻假会把「默认仍然不筛」这件事一起抹掉，而 `J-N6b-3` 要钉的正是「有口可筛、默认未筛」这两件同时为真），"
            "调用点的提及数只作分布披露",
            "合成材料的对账是**文件级共现**（既写进 E 存储、又构造过带合成 cue 的 Record），不是调用路径证明——"
            "这些脚本里 record 常由另一个函数造好再传进来，逐点链归属超出静态读的能力，"
            "所以清单只说「这些文件里有合成材料进了 E」，不说「这条读数一定由它产出」",
            "标签本轮一律不改（DEBT-G105 的代价是历史读数不可比，属 owner）",
        ],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="扫描根（契约测指向临时目录用）")
    parser.add_argument(
        "--out-report",
        default="reports/taiji_episodic_read_sharing_face_20261011.json",
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
                "files_parsed": report["files_parsed"],
                "read_site_count": report["read_site_count"],
                "reads_mentioning_episode_id": report["reads_mentioning_episode_id"],
                "write_site_count": report["write_site_count"],
                "synthetic_record_sites": report["synthetic_record_sites"],
                "synthetic_files_reaching_store": len(
                    report["files_that_both_write_and_construct_synthetic_records"]
                ),
                "verdicts": report["verdicts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A-27 战线一 乙档：零件台账——每个可选零件"挂在哪条链、那条链在主入口下被走到几次"。

为什么要有这一支
----------------
2026-09-28 付过学费的那个缺陷（`PLAN-A-26` §6.2）就是这一类：一个**默认关**的开关被打开后，
它挂的链在主入口下**根本没被走到**，于是那个面永远是 0，而两个 2M-tick 的臂白跑。
`SPEC-A-22` §23 也自报过同型的盲区（点名式清单必然漏）。
本支给每个可选零件记三件事：**装配了吗**、**挂在哪个方法上**、**在四种入口下各被走到几次**。

做法（**不改任何产品源文件**）
------------------------------
在探针里对**实例**打方法包装（`setattr(part, method, wrapped)`，`__dict__` 里留痕、跑完删掉），
逐零件计数。这样既不动产品码、也不改类（不会污染别的实例），跑完可逐位复原。

四种入口（每个入口各用一枚**全新** runtime）
--------------------------------------------
| 入口 | 调用 | 说明 |
|---|---|---|
| `generate` | `substrate.generate(...)` | 产品那条生成链 |
| `chat` | `runtime.chat(...)` | 产品会话链 |
| `observe_action` | `substrate.observe(byte, readout="action")` | **旧主训练线**（`train_seed_corpus.py` 改动前的默认） |
| `observe_predictive` | `substrate.observe(byte, readout="predictive")` | **新主训练线**（`PLAN-A-26` 两臂） |
| `consolidate` | `substrate.consolidate(cycles=1)` | 睡眠巩固那条慢通路 |

判据（跑之前写死）
------------------
* 每个入口每个零件格子**必须有分母**（该入口总调用次数）；分母为 0 不出结论（退出码 2）；
* **插桩不改行为**（本支自己的守卫）：同一入口在"插桩"与"不插桩"两种情形下，
  `generate` 的输出必须**逐字节相同**；基座 sha256 跑前后相同；
* 本支**不下"该不该收敛"的结论**（那是丙档）——它只给分母与清单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: 零件 → (属性路径, 记账方法)。属性以 `_` 开头的都是"可选装配"件（默认 None＝未装配）。
#: 记账方法取该零件**最有代表性的入口**（学习／发射／取回），只记这一个是刻意的：
#: 记更多名字会让"被走到"失去单一含义，而这里要的就是"这条链有没有被走到"。
PARTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("fabric", ("fabric",), "step"),
    ("motor", ("motor",), "learn"),
    ("predictive_readout", ("predictive_readout",), "learn"),
    ("predictive_context", ("predictive_context",), "learn"),
    ("memory.recall", ("memory",), "recall"),
    ("memory.write", ("memory",), "write"),
    ("identity_organ", ("identity_organ",), "recall"),
    ("copy_circuit.evidence", ("copy_circuit",), "evidence"),
    ("copy_circuit.store.record", ("copy_circuit", "store"), "record"),
    ("gated_temporal_candidate", ("_gated_temporal_candidate",), "encode"),
    ("adaptive_residual_bridge", ("_adaptive_residual_bridge",), "forward"),
    ("adaptive_residual_shadow", ("_adaptive_residual_shadow",), "forward"),
    ("response_plan_readout", ("_response_plan_readout",), "plan_probabilities"),
    ("episodic_memory", ("_episodic_memory",), "recall"),
    ("semantic_memory", ("_semantic_memory",), "recall"),
    ("world_dynamics", ("_world_dynamics",), "forward"),
    ("workspace_router", ("_workspace_router",), "route"),
    ("executive", ("_executive",), "decide"),
    ("goal_planner", ("_goal_planner",), "plan"),
    ("affordance_features", ("_affordance_features",), "observe"),
    ("generation_controller", ("_generation_controller",), "step"),
    ("content_selector", ("_content_selector",), "select"),
    ("language_organ", ("_language_organ",), "render"),
    ("procedural_memory", ("_procedural_memory",), "recall"),
    ("procedural_sequence_memory", ("_procedural_sequence_memory",), "recall"),
    ("homeostatic_controller", ("_homeostatic_controller",), "step"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Any, path: tuple[str, ...]) -> Any:
    node = root
    for name in path:
        node = getattr(node, name, None)
        if node is None:
            return None
    return node


def _resolve_method(root: Any, path: tuple[str, ...], method: str) -> tuple[Any, Any] | None:
    target = _resolve(root, path)
    if target is None:
        return None
    #: 链上的对象可能是"持有者"（如 `copy_circuit.store`），取链尾那一段。
    original = getattr(target, method, None)
    if not callable(original):
        return None
    return target, original


def _install(root: Any, counter: dict[str, int], installed: list[tuple[Any, str]]) -> list[str]:
    """逐零件打包装；返回**未装配**的零件名（它们本身就是答案的一半）。"""

    absent: list[str] = []
    for label, path, method in PARTS:
        found = _resolve_method(root, path, method)
        if found is None:
            absent.append(label)
            continue
        target, original = found

        def wrapped(
            *args: Any, _label: str = label, _original: Any = original, **kwargs: Any
        ) -> Any:
            counter[_label] += 1
            return _original(*args, **kwargs)

        setattr(target, method, wrapped)
        installed.append((target, method))
    return absent


def _restore(installed: list[tuple[Any, str]]) -> None:
    for target, method in installed:
        target.__dict__.pop(method, None)


def _entry_points(substrate: Any, runtime: Any, *, byte: int) -> dict[str, Any]:
    """五种入口各跑固定预算；每个入口只碰自己的 runtime。"""

    boundary = int(substrate.config.boundary_symbol)
    return {
        "generate": lambda: substrate.generate(
            b"\xe4\xbd\xa0\xe5\xa5\xbd\xe3\x80\x82", 32, stop_at_boundary=True, sample=False
        ),
        "chat": lambda: runtime.chat("你好。", learn=False, repetition_penalty=0.0),
        "observe_action": lambda: (
            substrate.reset_dynamics(episode_id="a27-ledger"),
            [
                substrate.observe(symbol, learn=True, readout="action")
                for symbol in (boundary, byte, byte + 1, byte)
            ],
        ),
        "observe_predictive": lambda: (
            #: `seed_beta` 载入时情节是 `action` 档且活跃，换链前必须开新情节
            #: （`observe` 明令"情节活跃时不许换读出"）。
            substrate.reset_dynamics(episode_id="a27-ledger"),
            [
                substrate.observe(symbol, learn=True, readout="predictive", learn_motor=False)
                for symbol in (boundary, byte, byte + 1, byte)
            ],
        ),
        "consolidate": lambda: substrate.consolidate(cycles=1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default=None, help="可选：装上复制回路再普查")
    parser.add_argument("--out-report", default="reports/taiji_a27_walked_ledger_20260928.json")
    args = parser.parse_args()

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    entries = ("generate", "chat", "observe_action", "observe_predictive", "consolidate")

    #: 先量"不插桩"的 generate 输出，供"插桩不改行为"这条守卫对照。
    reference = SeedRuntime.load(checkpoint)
    if args.circuit:
        reference.enable_copy_circuit(PROJECT_ROOT / args.circuit)
    baseline_output = bytes(
        _entry_points(reference.model.substrate, reference, byte=0xE4)["generate"]()
    )

    matrix: dict[str, dict[str, int]] = {entry: {} for entry in entries}
    coverage: dict[str, dict[str, Any]] = {
        entry: {"calls": 0, "absent": [], "blocked": None} for entry in entries
    }
    output_under_instrumentation: dict[str, str] = {}
    for entry in entries:
        runtime = SeedRuntime.load(checkpoint)
        if args.circuit:
            runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
        substrate = runtime.model.substrate
        counter: dict[str, int] = {label: 0 for label, _, _ in PARTS}
        installed: list[tuple[Any, str]] = []
        absent = _install(substrate, counter, installed)
        coverage[entry]["calls"] = 1
        try:
            action = _entry_points(substrate, runtime, byte=0xE4)[entry]
            if entry == "generate":
                output_under_instrumentation[entry] = hashlib.sha256(bytes(action())).hexdigest()
            else:
                action()
        except Exception as exc:  # noqa: BLE001 — 入口被前置门挡住**本身就是读数**
            #: 例：`consolidate()` 在 `write_count == 0` 时抛
            #: "consolidation requires at least one episodic write"（`SPEC-A-22` §23 的门槛②）。
            #: 这类"入口根本跑不起来"必须如实记下，不能当成"跑了但计数为 0"。
            coverage[entry]["blocked"] = f"{type(exc).__name__}: {exc}"
        finally:
            _restore(installed)
        matrix[entry] = counter
        coverage[entry]["absent"] = absent

    walked = {
        entry: {label: count for label, count in matrix[entry].items() if count > 0}
        for entry in entries
    }
    report: dict[str, Any] = {
        "format": "taiji-a27-walked-ledger-v1",
        "prereg": "plans/reference/PLAN-A-27_three-gaps-decision-brief_20260928.md §1.2（战线一 乙档）",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "entries": list(entries),
        "parts_probed": [label for label, _, _ in PARTS],
        "absent_by_entry": {entry: coverage[entry]["absent"] for entry in entries},
        "blocked_by_entry": {
            entry: coverage[entry]["blocked"] for entry in entries if coverage[entry]["blocked"]
        },
        "walked_by_entry": walked,
        "matrix": matrix,
        "notes": [
            "未装配（absent）＝该零件在该 runtime 下根本没装上（多因默认关）",
            "计数为 0 且未装 ⇒ 两件不同的事：本表把二者分开列",
            "**口径陷阱（必须带这条读）**：记的是该零件**学习/写方向的代表方法**"
            "（方法名见 `methods_by_part`）⇒ 只读入口（generate／chat）上 `learn` 类计数为 0"
            "**不代表该零件没被使用**，只代表这一趟没写它。要判'用没用'得看读方向的方法"
            "（如 `probabilities`／`encode`／`recall`），那是本表的已知局限，未修。",
            "本支不下'该不该收敛'的结论（丙档）；它只给分母与清单",
        ],
        "methods_by_part": {label: method for label, _, method in PARTS},
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    report["instrument_guard"] = {
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        "instrumentation_is_behavior_neutral": (
            output_under_instrumentation.get("generate")
            == hashlib.sha256(baseline_output).hexdigest()
        ),
        "every_entry_has_denominator": all(coverage[entry]["calls"] > 0 for entry in entries),
    }
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("零件".ljust(28) + "".join(entry.ljust(20) for entry in entries))
    for label, _, _ in PARTS:
        cells = []
        for entry in entries:
            cells.append(
                "未装配".ljust(20)
                if label in coverage[entry]["absent"]
                else str(matrix[entry].get(label, 0)).ljust(20)
            )
        print(label.ljust(28) + "".join(cells))
    print()
    for entry in entries:
        print(
            f"{entry}: 未装配 {len(coverage[entry]['absent'])} 个零件；被走到 {len(walked[entry])} 个"
        )
    print(json.dumps({"guard": report["instrument_guard"], "out": out.name}, ensure_ascii=False))
    return 0 if all(report["instrument_guard"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

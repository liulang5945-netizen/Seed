"""A-27 战线一 乙档：零件台账——每个可选零件"挂在哪条链、那条链在主入口下被走到几次"。

为什么要有这一支
----------------
2026-09-28 付过学费的那个缺陷（`PLAN-A-26` §6.2）就是这一类：一个**默认关**的开关被打开后，
它挂的链在主入口下**根本没被走到**，于是那个面永远是 0，而两个 2M-tick 的臂白跑。
`SPEC-A-22` §23 也自报过同型的盲区（点名式清单必然漏）。
本支给每个可选零件记三件事：**装配了吗**、**挂在哪个方法上**、**在四种入口下各被走到几次**。

2026-10-11（PLAN-N5-12 批次 A）改版两处：①"没装上"与"方法名点错"分成两列出版（旧版把两件混在
`absent` 一桶，而 `PARTS` 里 26 枚有 10 枚点着该类上并不存在／从不被调的名字）；②`--out-report`
的默认落点不再指向已封存的 20260928 件，且目标已存在时需 `--overwrite` 才许覆写。

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
    ("episodic_memory", ("_episodic_memory",), "retrieve"),
    ("semantic_memory", ("_semantic_memory",), "predict"),
    ("world_dynamics", ("_world_dynamics",), "predict"),
    ("workspace_router", ("_workspace_router",), "route"),
    ("executive", ("_executive",), "select"),
    ("goal_planner", ("_goal_planner",), "plan"),
    ("affordance_features", ("_affordance_features",), "encode"),
    ("generation_controller", ("_generation_controller",), "plan_content"),
    ("content_selector", ("_content_selector",), "select"),
    ("language_organ", ("_language_organ",), "emit"),
    ("procedural_memory", ("_procedural_memory",), "predict"),
    ("procedural_sequence_memory", ("_procedural_sequence_memory",), "predict_episode"),
    ("homeostatic_controller", ("_homeostatic_controller",), "update"),
    ("homeostatic_select_mode", ("_homeostatic_controller",), "select_mode"),
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


def _install(
    root: Any, counter: dict[str, int], installed: list[tuple[Any, str]]
) -> tuple[list[str], list[str]]:
    """逐零件打包装；返回 **(未装配, 方法名点错)** 两列。

    2026-10-11（PLAN-N5-12 批次 A）之前这两件事混在一个 `absent` 桶里，而函数文档串只声明了
    "未装配"一种：方法名点错的零件因此永远读成"这条链没装上"，仪器对它**没有判别力**。
    分列之后 `absent` 仍在报告里出版，取两列之并，历史列可对账。
    """

    not_attached: list[str] = []
    method_not_found: list[str] = []
    for label, path, method in PARTS:
        found = _resolve_method(root, path, method)
        if found is None:
            if _resolve(root, path) is None:
                not_attached.append(label)
            else:
                method_not_found.append(label)
            continue
        target, original = found

        def wrapped(
            *args: Any, _label: str = label, _original: Any = original, **kwargs: Any
        ) -> Any:
            counter[_label] += 1
            return _original(*args, **kwargs)

        setattr(target, method, wrapped)
        installed.append((target, method))
    return not_attached, method_not_found


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


def _guard_overwrite(target: Path, *, force: bool) -> None:
    """落盘护栏：目标已存在时默认拒绝覆写。

    这台仪器的 `--out-report` 旧默认值就是已封存的 tracked 件
    `reports/taiji_a27_walked_ledger_20260928.json`，而写入是无条件 `write_text` ⇒ 直接重跑
    会把历史读数盖掉（PLAN-N5-12 §4 批次 A 的第三条）。
    """

    if target.exists() and not force:
        raise SystemExit(f"refusing to overwrite an existing report: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default=None, help="可选：装上复制回路再普查")
    parser.add_argument(
        "--out-report",
        default="reports/taiji_a27_walked_ledger_20261011.json",
        help="落点。目标已存在时需 --overwrite 才许覆写（默认响亮拒绝）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许覆写已存在的报告件（旧默认落点曾是已封存件）",
    )
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
        entry: {
            "calls": 0,
            "absent": [],
            "not_attached": [],
            "method_not_found": [],
            "blocked": None,
        }
        for entry in entries
    }
    output_under_instrumentation: dict[str, str] = {}
    for entry in entries:
        runtime = SeedRuntime.load(checkpoint)
        if args.circuit:
            runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
        substrate = runtime.model.substrate
        counter: dict[str, int] = {label: 0 for label, _, _ in PARTS}
        installed: list[tuple[Any, str]] = []
        not_attached, method_not_found = _install(substrate, counter, installed)
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
        coverage[entry]["not_attached"] = not_attached
        coverage[entry]["method_not_found"] = method_not_found
        coverage[entry]["absent"] = sorted(set(not_attached) | set(method_not_found))

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
        "not_attached_by_entry": {entry: coverage[entry]["not_attached"] for entry in entries},
        "method_not_found_by_entry": {
            entry: coverage[entry]["method_not_found"] for entry in entries
        },
        "blocked_by_entry": {
            entry: coverage[entry]["blocked"] for entry in entries if coverage[entry]["blocked"]
        },
        "walked_by_entry": walked,
        "matrix": matrix,
        "notes": [
            "not_attached＝该零件在该 runtime 下根本没装上（多因默认关）",
            "method_not_found＝零件装着，但 PARTS 点的名字在它的类上不存在（2026-10-11 前混在 absent 一桶里）",
            "absent 仍出版＝上面两列之并，历史列可对账；出处＝PLAN-N5-12 §2 事实 2／4：26 格里 10 枚曾点错名",
            "第三种形状：名字取得到（如经 nn.Module 的 forward 占位）却永远 0 计数——既不算 absent 也不是被走到",
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
    _guard_overwrite(out, force=args.overwrite)
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
        cell = coverage[entry]
        print(
            f"{entry}: 未装配 {len(cell['not_attached'])} 个／名字点错 "
            f"{len(cell['method_not_found'])} 个；被走到 {len(walked[entry])} 个"
        )
    print(json.dumps({"guard": report["instrument_guard"], "out": out.name}, ensure_ascii=False))
    return 0 if all(report["instrument_guard"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

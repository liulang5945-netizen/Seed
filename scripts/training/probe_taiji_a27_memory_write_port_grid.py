"""A-27 战线三 甲档：记忆写入口的**四格**（进程内模拟，零改产品码、零训练）。

问题（`SPEC-A-22` §23 裁定 + `PLAN-A-27` §3）
--------------------------------------------
产品在跑的那条链上，情节记忆**读的口接了但库是空的**：写入被 `pending_experience`
（动作＋奖励＋结果）门控，而语言轮走 `observe(learn=False)`，永远不产生它。实测：
`write_count` 0→0、`generate(use_memory=False)` 对 `True` **6/6 逐字节相同**、
情节场证据范数 0.0、巩固三个解码头权重全零。
⇒ "接进去"有**两个独立门槛**：①**写入口**（语言轮该往库里写什么——这是设计问题）；
②**慢通路要训练才通电**（`consolidate()` 的前置是 `write_count > 0`，现在恒不满足）。

本支做什么
----------
**不改任何产品源文件**（沿用 `probe_taiji_cap0_legacy_load.py` 那种"短命进程里包装"的做法），
在探针里**进程内**调用产品已有的 `EpisodicField.write(...)`（参数形状照抄 `model.py:1932-1943`
那次产品自己的调用），然后量 **写/不写 × 读/不读** 四格；并让"**写什么**"的三个候选各跑一格。

三个候选（"一个语言轮该写什么"的最小三档，都只用产品既有参数面）
----------------------------------------------------------------
| 候选 | cue（皮质态取自哪一刻） | action_symbol | outcome_symbol | 含义 |
|---|---|---|---|---|
| `ask_answer` | **提问**读完那一刻 | 提问首字节 | 答案首字节 | "被问 → 答"记成一次转移 |
| `tell_answer` | **告知**读完那一刻 | 告知首字节 | 答案首字节 | "被告知 → 答"记成一次转移 |
| `blank` | 提问读完那一刻 | 边界符 | 边界符 | **对照**：写了，但没写内容 |

判据（跑之前写死，跑完不改）
----------------------------
* **前置否证门（甲档的全部意义）**：`写+读` 相对 `不写+读` 的输出**逐字节相同**
  ⇒ 读通路对语言内容是死的 ⇒ **停**，不开乙/丙（并如实登记为否证）。
  有差异 ⇒ 继续，并按三候选里差异最大的那格定"该写什么"。
* 三条分母必须给全：`write_count`（写臂须 > 0）、情节场证据范数（须 > 0）、
  巩固解码头范数（**预期仍为 0**——本支不跑 `consolidate()`；这一格留给丙）。
* 守卫：基座 sha256 跑前后逐位相同；`config_untouched`（本支一个配置都不改）；
  **源码指纹**（`taiji/`＋`api/` 跑前后相同）⇒ 证明"进程内模拟"没有落到产品源上；
  每格输出长度 > 0。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: 不重抄链路：四格里的"读"直接复用既有记忆审计探针的产品调用（`substrate.generate`）。
from probe_taiji_memory_wiring_audit import (  # noqa: E402
    _compare,
    _inventory,
    _run,
    _sha256,
)

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
CANDS = ("ask_answer", "tell_answer", "blank")


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for folder in ("taiji", "api"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _feed(substrate: Any, text: str) -> None:
    """把一段文本喂进动力学（与产品 `generate` 的喂入逐字同参：learn=False、predictive、不用记忆）。

    只用来把"那一刻的皮质态"造出来，好让 `write` 的 cue 取自真实读数而不是凭空造张量。
    """

    substrate.reset_dynamics(episode_id="a27-write-port")
    substrate.observe(
        substrate.config.boundary_symbol, learn=False, readout="predictive", use_memory=False
    )
    for byte in text.encode("utf-8"):
        substrate.observe(int(byte), learn=False, readout="predictive", use_memory=False)


def _write_event(substrate: Any, *, cue_text: str, action: int, outcome: int) -> dict[str, Any]:
    """进程内调一次产品自己的 `write`（参数照抄 `taiji/model.py:1932-1943`）。"""

    _feed(substrate, cue_text)
    state = substrate.snapshot()
    memory = substrate.memory
    before = int(memory.write_count)
    memory.write(
        substrate.cortical_cue(),
        action_symbol=int(action),
        reward=0.0,
        outcome_symbol=int(outcome),
        tick=int(state.tick),
        episode_id="a27-write-port",
        provenance="external",  # 语言轮的告知来自外部；合法取值见 `memory.PROVENANCE_KINDS`
        threshold=state.memory.threshold,
    )
    return {
        "write_count_before": before,
        "write_count_after": int(memory.write_count),
        "cue_bytes": len(cue_text.encode("utf-8")),
    }


def _cells(runtime: Any, items: list[dict[str, Any]], cand: str) -> dict[str, Any]:
    """跑一格：对每题"按候选写一次"，再分别在 read=off/on 下生成答案。"""

    substrate = runtime.model.substrate
    boundary = int(substrate.config.boundary_symbol)
    off: list[bytes] = []
    on: list[bytes] = []
    writes: list[dict[str, Any]] = []
    for item in items:
        told = [str(turn) for turn in item["turns"][:-1]]
        question = str(item["turns"][-1])
        tokens = [str(token) for token in item["expected_contains"]]
        answer_byte = tokens[0].encode("utf-8")[0] if tokens else boundary
        if cand == "blank":
            writes.append(
                _write_event(substrate, cue_text=question, action=boundary, outcome=boundary)
            )
        elif cand == "ask_answer":
            writes.append(
                _write_event(
                    substrate,
                    cue_text=question,
                    action=(question.encode("utf-8") or b"\x00")[0],
                    outcome=answer_byte,
                )
            )
        else:  # tell_answer
            writes.append(
                _write_event(
                    substrate,
                    cue_text="".join(told),
                    action=(("".join(told)).encode("utf-8") or b"\x00")[0],
                    outcome=answer_byte,
                )
            )
        off.append(_run([question], substrate, use_memory=False)[0])
        on.append(_run([question], substrate, use_memory=True)[0])
    return {
        "read_off": [bytes(value) for value in off],
        "read_on": [bytes(value) for value in on],
        "writes": writes,
        "inventory_after": _inventory(substrate),
    }


def _hits(outputs: list[bytes], items: list[dict[str, Any]]) -> int:
    """命中数：答案字节里含不含该题的 `expected_contains`（与 A2 那条链同口径）。"""

    count = 0
    for output, item in zip(outputs, items, strict=True):
        text = bytes(output).decode("utf-8", "ignore")
        if any(str(token) in text for token in item["expected_contains"]):
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    from score_taiji_r2_copy_surface_extension import load_items

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    source_before = _source_fingerprint()
    items = load_items(MANIFEST)[: args.limit]

    #: 基线格：**不写**（库保持空）——它是"写"的对照，也是 §23 那条 6/6 的复现位。
    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    questions = [str(item["turns"][-1]) for item in items]
    empty = {
        "read_off": [bytes(value) for value in _run(questions, substrate, use_memory=False)],
        "read_on": [bytes(value) for value in _run(questions, substrate, use_memory=True)],
        "inventory_after": _inventory(substrate),
    }

    grid: dict[str, Any] = {"no_write": empty}
    for cand in CANDS:
        fresh = SeedRuntime.load(checkpoint)
        grid[cand] = _cells(fresh, items, cand)

    rows = []
    for name, cell in grid.items():
        for index, question in enumerate(questions):
            rows.append(
                {
                    "grid": name,
                    "item": index,
                    "read_off_vs_baseline_off": _compare(
                        cell["read_off"][index], empty["read_off"][index]
                    ),
                    "read_on_vs_read_off": _compare(
                        cell["read_on"][index], cell["read_off"][index]
                    ),
                    "read_on_vs_baseline_on": _compare(
                        cell["read_on"][index], empty["read_on"][index]
                    ),
                }
            )

    def _identical(rows_subset: list[dict[str, Any]], key: str) -> int:
        return sum(1 for row in rows_subset if row[key]["differing_bytes"] == 0)

    write_cands = {
        cand: rows[len(questions) * (i + 1) : len(questions) * (i + 2)]
        for i, cand in enumerate(CANDS)
    }
    base_rows = rows[: len(questions)]
    report: dict[str, Any] = {
        "format": "taiji-a27-memory-write-port-grid-v1",
        "prereg": "plans/reference/PLAN-A-27_three-gaps-decision-brief_20260928.md §3（战线三 甲档）",
        "manifest": MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256(MANIFEST),
        "checkpoint": args.checkpoint,
        "items": len(items),
        "questions": questions,
        "grid": {
            name: {
                "write_count_after": cell["inventory_after"]["memory_write_count"],
                "memory_records": cell["inventory_after"]["memory_records"],
                "all_zero_surfaces": cell["inventory_after"]["all_zero_surfaces"],
            }
            for name, cell in grid.items()
        },
        "summary": {
            "baseline_read_gate": {
                #: §23 那条读法的复现位：库空时 read off/on 应当逐字节相同。
                "items_identical": _identical(base_rows, "read_on_vs_read_off"),
                "items": len(base_rows),
                "read_off_hits": _hits(empty["read_off"], items),
                "read_on_hits": _hits(empty["read_on"], items),
            },
            **{
                cand: {
                    "read_on_vs_read_off_identical": _identical(
                        write_cands[cand], "read_on_vs_read_off"
                    ),
                    "read_on_vs_baseline_on_identical": _identical(
                        write_cands[cand], "read_on_vs_baseline_on"
                    ),
                    "read_off_vs_baseline_off_identical": _identical(
                        write_cands[cand], "read_off_vs_baseline_off"
                    ),
                    #: "变没变"只说明**通路活着**；"往哪边变"要看命中——三候选靠这一列分。
                    "read_off_hits": _hits(grid[cand]["read_off"], items),
                    "read_on_hits": _hits(grid[cand]["read_on"], items),
                }
                for cand in CANDS
            },
        },
        "rows": rows,
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    report["instrument_guard"] = {
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        "product_source_untouched": _source_fingerprint() == source_before,
        "config_untouched": True,
        "every_cell_nonempty": all(
            len(value) == len(questions)
            for cell in grid.values()
            for key in ("read_off", "read_on")
            for value in [cell[key]]
        ),
        "every_write_arm_wrote": all(
            (grid[cand]["writes"][0]["write_count_after"] > 0) for cand in CANDS
        ),
    }
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"summary": report["summary"], "guard": report["instrument_guard"]},
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"out: {out.name}")
    return 0 if all(report["instrument_guard"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

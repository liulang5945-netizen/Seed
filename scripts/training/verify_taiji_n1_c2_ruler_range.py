"""C2 那把尺有没有动态范围——把"先量尺再量能力"从散文变成可复算的读数（DEBT-G157 的③）。

owner 于 2026-10-17 弹窗裁**甲＝先修尺**（08 ㊵-1100 登记）。本件不重跑任何实验：它只读两枚
**已封存**的 C2 报告——

* `final`：七臂各 16 条的 `strict_hits`／`raw_strict_hits`；
* `placement`：换一次剂量干预后的 `raw_changes_from_tail`（oracle／wrong-episode／zero-replay）。

两件事合起来才是判据：**尺有没有范围**看"是否存在任一非零严格命中"，尤其看上界诊断臂
`oracle-correct`（PLAN-N1-07 §表定义它由保密计分标签指定正确 episode、按原序回放，"不得计作选择能力"
⇒ 它是这把尺的**上界**）；**机制有没有位移**看剂量臂是否大于对照臂。上界臂为 0 时，"0/16"只描述尺，
不描述能力 ⇒ `can_be_cited_as_capability_falsification` 必须为 `false`。

`models_loaded=0`、`model_queries=0`、`parameter_updates=0`、`writes_any_face=false`；写靶一次性。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "reports/taiji_n1_c2_final_20261015_r2.json"
PLACEMENT = ROOT / "reports/taiji_n1_c2_padding_placement_final_20261015.json"
FORMAT = "taiji-n1-c2-ruler-range-v1"
#: 上界诊断臂的名字是本判决的支点：它改名或不在场，本件必须响亮拒绝而不是猜。
UPPER_BOUND_ARM = "oracle-correct"
CONTROL_ARM = "zero-replay"


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不落盘。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, expect_status: str) -> dict:
    require(path.is_file(), f"sealed report missing: {path}")
    entry = json.loads(path.read_text(encoding="utf-8"))
    require(
        entry.get("status") == expect_status,
        f"{path.name} must be {expect_status}, got {entry.get('status')!r}",
    )
    summary = entry.get("summary")
    require(isinstance(summary, dict) and bool(summary), f"{path.name} has no summary block")
    return entry


def arm_field(summary: dict, arm: str, key: str, path: Path) -> int:
    block = summary.get(arm)
    require(isinstance(block, dict), f"{path.name} lacks arm {arm!r}")
    value = block.get(key)
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{path.name}.{arm}.{key} not an int",
    )
    return value


def measure(final: dict, placement: dict) -> dict:
    final_summary, place_summary = final["summary"], placement["summary"]
    #: 尺的范围：任一臂非零严格命中即有范围；上界臂单独点名出版（它 0 时"0/16"只描述尺）。
    strict = {
        arm: arm_field(final_summary, arm, "strict_hits", FINAL) for arm in sorted(final_summary)
    }
    require(
        UPPER_BOUND_ARM in strict, f"{FINAL.name} lacks the upper-bound arm {UPPER_BOUND_ARM!r}"
    )
    for arm, block in final_summary.items():
        count = block.get("count")
        require(isinstance(count, int), f"{FINAL.name}.{arm}.count missing")
        require(
            strict[arm] <= count,
            f"{FINAL.name}.{arm}: strict_hits {strict[arm]} exceeds count {count}",
        )
    has_range = any(value > 0 for value in strict.values())
    #: 机制位移：剂量臂必须严格大于对照臂，中间臂只披露不参与判定。
    require(UPPER_BOUND_ARM in place_summary, f"{PLACEMENT.name} lacks {UPPER_BOUND_ARM!r}")
    require(CONTROL_ARM in place_summary, f"{PLACEMENT.name} lacks {CONTROL_ARM!r}")
    dose = arm_field(place_summary, UPPER_BOUND_ARM, "raw_changes_from_tail", PLACEMENT)
    control = arm_field(place_summary, CONTROL_ARM, "raw_changes_from_tail", PLACEMENT)
    moved = dose > control
    if has_range and strict.get(UPPER_BOUND_ARM, 0) > 0:
        status = "ruler_has_range_and_upper_bound_moves"
        falsifiable = True
    elif has_range:
        status = "ruler_has_range_upper_bound_still_zero"
        falsifiable = True
    elif moved:
        #: 今天这一支：尺取不到非零值，机制侧却有位移 ⇒ "0/16" 不能当能力否证。
        status = "ruler_no_range_mechanism_moves"
        falsifiable = False
    else:
        status = "ruler_no_range_and_no_mechanism_move"
        falsifiable = False
    return {
        "format": FORMAT,
        "status": status,
        "strict_hits_by_arm": strict,
        "upper_bound_arm": UPPER_BOUND_ARM,
        "upper_bound_strict_hits": strict[UPPER_BOUND_ARM],
        "ruler_has_dynamic_range": has_range,
        "placement_raw_changes": {
            arm: arm_field(place_summary, arm, "raw_changes_from_tail", PLACEMENT)
            for arm in sorted(place_summary)
        },
        "dose_arm": UPPER_BOUND_ARM,
        "control_arm": CONTROL_ARM,
        "mechanism_moves": moved,
        "can_be_cited_as_capability_falsification": falsifiable,
        "falsification_blocked_because": (
            "无（这把尺判得动）"
            if falsifiable
            else (
                "上界诊断臂 "
                + UPPER_BOUND_ARM
                + " 的严格命中为 "
                + str(strict[UPPER_BOUND_ARM])
                + "，尺在该面取不到非零值；0/16 只描述尺，不描述能力"
            )
        ),
        "forbidden_sentence": (
            "七臂严格命中各 0/16 证明 C2 能力不通过" if not falsifiable else "无（本件判得动）"
        ),
        "reading_limit": (
            "本件只判『这把尺有没有范围』与『换剂量后生成是否位移』，不判能力、不判信用、"
            "不替 PLAN-N-01 的阶梯改口径；改判据或改阶梯归 owner。"
        ),
        "models_loaded": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "writes_any_face": False,
        "inputs": {
            str(FINAL): sha256(FINAL),
            str(PLACEMENT): sha256(PLACEMENT),
        },
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final-report", type=Path, default=FINAL)
    parser.add_argument("--placement-report", type=Path, default=PLACEMENT)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    final_path, place_path = arguments.final_report, arguments.placement_report
    if not final_path.is_absolute():
        final_path = ROOT / final_path
    if not place_path.is_absolute():
        place_path = ROOT / place_path
    out = arguments.out_report
    if not out.is_absolute():
        out = ROOT / out
    try:
        report = measure(load(final_path, "complete"), load(place_path, "complete"))
        require(out.parent.is_dir(), f"write target directory missing: {out.parent}")
        require(not out.exists() or arguments.overwrite, f"report already exists: {out}")
    except Refuse as refusal:
        print(f"REFUSE: {refusal}")
        return 2
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "ruler_has_dynamic_range": report["ruler_has_dynamic_range"],
                "mechanism_moves": report["mechanism_moves"],
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

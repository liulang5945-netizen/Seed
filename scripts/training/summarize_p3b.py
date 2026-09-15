"""Print the P3b two-arm progress table from the campaign reports (read-only).

The campaign writes one row per scored checkpoint; the claim lives in the **difference between
arms**, because both arms share start, budget, objective and chain, while only treatment sees the
dialogue-dense subset.  This script exists so monitoring never needs an ad-hoc one-liner that
silently redefines "improved".

Usage::

    python -X utf8 scripts/training/summarize_p3b.py
    python -X utf8 scripts/training/summarize_p3b.py --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARMS = ("treatment", "control")
MECHANISED = ("C", "D", "E")
#: 修订 §4 预声明的读法：每维 20 题 => 单题翻转 = 0.05 是噪声，主效应要到 **0.15（三题）**
#: 且方向在 **两个连续共同检查点** 一致才称为效应；否则只能记 not detected at this resolution。
MAIN_EFFECT_RESOLUTION = 0.15
CONSECUTIVE_CONFIRMATIONS = 2
#: 阶段报告里这三维整维是 not_executed（07 §5：不记 0 也不记通过），所以 J4 的 A/H 分支
#: 在本链路上判不了 -- 缺字段不是通过，汇总器必须把它显式说出来。
NOT_EXECUTED_DIMENSIONS = ("A", "F", "H")


def load_arm(arm: str, root: Path | None = None) -> dict[str, Any]:
    base = root or PROJECT_ROOT
    path = base / "reports" / f"taiji_p3b_campaign_{arm}_20260915.json"
    if not path.exists():
        return {"arm": arm, "stages": [], "status": "missing", "campaign_stop": None}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_stage(report: dict[str, Any]) -> dict[str, Any] | None:
    stages = report.get("stages") or []
    return stages[-1] if stages else None


def tick_table(treatment: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    """Match stages by tick and report treatment, control and their difference per dimension."""

    def by_tick(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
        return {int(row["tick"]): row for row in report.get("stages") or []}

    treated, controlled = by_tick(treatment), by_tick(control)
    rows: list[dict[str, Any]] = []
    for tick in sorted(set(treated) | set(controlled)):
        left = treated.get(tick)
        right = controlled.get(tick)
        row: dict[str, Any] = {
            "tick": tick,
            "treatment": None if left is None else left["scores"],
            "control": None if right is None else right["scores"],
            "treatment_minus_control": None,
            "chain_ok": bool(
                left and right and left["chain_matches_p3a"] and right["chain_matches_p3a"]
            ),
        }
        if left and right:
            row["treatment_minus_control"] = {
                key: (
                    None
                    if left["scores"].get(key) is None or right["scores"].get(key) is None
                    else round(left["scores"][key] - right["scores"][key], 4)
                )
                for key in MECHANISED
            }
        rows.append(row)
    return rows


def main_effect_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-declared resolution to the tick-matched arm differences.

    Only ticks where **both** arms have a score count; a tick missing one arm is not a zero, it is
    no data.  A dimension is called an effect only when ``CONSECUTIVE_CONFIRMATIONS`` matched
    checkpoints in a row reach ``MAIN_EFFECT_RESOLUTION`` with the same sign -- anything less is
    reported as noise, in both directions.
    """

    out: dict[str, Any] = {}
    for key in MECHANISED:
        series = [
            (row["tick"], row["treatment_minus_control"][key])
            for row in rows
            if row["treatment_minus_control"] is not None
            and row["treatment_minus_control"].get(key) is not None
        ]
        best = streak = 0
        confirmed_sign: float | None = None
        previous: float | None = None
        for _, delta in series:
            reached = abs(delta) >= MAIN_EFFECT_RESOLUTION
            same_sign = reached and previous is not None and delta * previous > 0
            streak = (streak + 1) if same_sign else (1 if reached else 0)
            best = max(best, streak)
            if streak >= CONSECUTIVE_CONFIRMATIONS:
                confirmed_sign = delta
            previous = delta if reached else None
        last = series[-1][1] if series else None
        direction = (
            None if confirmed_sign is None else ("positive" if confirmed_sign > 0 else "negative")
        )
        if not series:
            call = "no_matched_checkpoint"
        elif direction is not None:
            call = f"effect_{direction}"
        elif best == 1:
            call = "above_resolution_single_point"
        else:
            call = "not_detected_at_this_resolution"
        out[key] = {
            "matched_ticks": len(series),
            "latest_delta": last,
            "longest_confirmed_run": best,
            "confirmed_direction": direction,
            "verdict": call,
        }
    out["unjudged"] = {
        "dimensions_not_executed_by_this_runner": list(NOT_EXECUTED_DIMENSIONS),
        "consequence": "J4's A/H branch has no data in stage reports; it must be reported untested",
        "resolution": MAIN_EFFECT_RESOLUTION,
        "consecutive_confirmations": CONSECUTIVE_CONFIRMATIONS,
    }
    return out


def arm_headline(report: dict[str, Any]) -> dict[str, Any]:
    """Only the current key is read here.

    A campaign started before the rename writes its stop verdict under the field's **previous**
    name; such artifacts survive only under a ``discarded_*`` name and are read by a one-off
    command.  The rename exists so the earlier token stays reserved for episode termination
    reasons (the N2 audit surface) -- reading both keys here would put that token back into this
    file.  See 03 §「P3b 双臂 campaign」的键名说明。
    """

    stage = latest_stage(report)
    criteria = report.get("criteria") or {}
    return {
        "arm": report.get("arm"),
        "status": report.get("status"),
        "campaign_stop": report.get("campaign_stop"),
        "stages": len(report.get("stages") or []),
        "latest_tick": None if stage is None else stage["tick"],
        "latest_scores": None if stage is None else stage["scores"],
        "deltas_vs_p3a": None if stage is None else stage["deltas_vs_p3a"],
        "criteria_verdict": criteria.get("verdict"),
        "protected_unchanged": report.get("protected_checkpoints_unchanged"),
    }


#: DEBT-I6: stage reports are written non-atomically, and the driver reuses any existing file on
#: resume.  Nothing in the running campaign notices a truncated report, so the monitor checks the
#: files themselves.  An empty problem list means the stage is intact.
REQUIRED_SURFACE_FIELDS = ("eval_set", "eval_set_format", "eval_set_frozen_on", "declared_mode")
ITEMS_PER_MECHANISED_DIMENSION = 20


def stage_integrity(arm: str, row: dict[str, Any]) -> list[str]:
    """What is wrong with this stage's report on disk; ``[]`` means it is intact."""

    path = PROJECT_ROOT / "reports" / "p3b_stages" / arm / str(row.get("report"))
    if not path.exists():
        return ["missing_report"]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["torn_json"]
    problems = [f"no_{field}" for field in REQUIRED_SURFACE_FIELDS if field not in report]
    dimensions = report.get("dimensions") or {}
    for key in MECHANISED:
        items = (dimensions.get(key) or {}).get("items")
        if not isinstance(items, list):
            problems.append(f"{key}:no_items")
        elif len(items) != ITEMS_PER_MECHANISED_DIMENSION:
            problems.append(f"{key}:items={len(items)}")
    if report.get("trained_during_eval") is not False:
        problems.append("trained_during_eval_not_false")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3b 双臂进度汇总（只读）")
    parser.add_argument("--json", action="store_true", help="输出机器可读汇总")
    args = parser.parse_args(argv)

    reports = {arm: load_arm(arm) for arm in ARMS}
    headlines = {arm: arm_headline(reports[arm]) for arm in ARMS}
    rows = tick_table(reports["treatment"], reports["control"])
    effect = main_effect_verdict(rows)
    integrity = {
        arm: [
            {"tick": row.get("tick"), "problems": stage_integrity(arm, row)}
            for row in (reports[arm].get("stages") or [])
        ]
        for arm in ARMS
    }
    if args.json:
        print(
            json.dumps(
                {
                    "arms": headlines,
                    "stages": rows,
                    "main_effect": effect,
                    "integrity": integrity,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0
    for arm in ARMS:
        print(f"== {arm} ==")
        for key, value in headlines[arm].items():
            print(f"  {key}: {value}")
    print("== tick 对照（treatment - control）==")
    for row in rows:
        delta = row["treatment_minus_control"]
        printable = "n/a" if delta is None else ", ".join(f"{k}={delta[k]:+}" for k in MECHANISED)
        print(f"  tick {row['tick']}: {printable}  chain_ok={row['chain_ok']}")
    unjudged = effect["unjudged"]
    print(
        f"== 主效应（>= {unjudged['resolution']} 且连续 "
        f"{unjudged['consecutive_confirmations']} 个共同检查点同向）=="
    )
    for key in MECHANISED:
        block = effect[key]
        print(
            f"  {key}: {block['verdict']}  latest={block['latest_delta']} "
            f"run={block['longest_confirmed_run']} matched={block['matched_ticks']}"
        )
    print(f"  A/F/H: {unjudged['consequence']}")
    broken = [
        f"  {arm} tick {item['tick']}: {', '.join(item['problems'])}"
        for arm in ARMS
        for item in integrity[arm]
        if item["problems"]
    ]
    print("== 阶段报告完整性（DEBT-I6 监视）==")
    print("\n".join(broken) if broken else "  all recorded stage reports intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

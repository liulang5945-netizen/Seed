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


def arm_headline(report: dict[str, Any]) -> dict[str, Any]:
    """Only the current key is read here.

    The two campaigns started at 2026-09-15 16:25 write their stop verdict under the field's
    **previous** name, because their code was loaded before the rename.  That rename exists so
    the earlier token stays reserved for episode termination reasons (the N2 audit surface);
    reading those in-flight artifacts is a one-off monitoring command, not a second concept
    living in this file.  See 03 §「P3b 双臂 campaign」的键名说明。
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3b 双臂进度汇总（只读）")
    parser.add_argument("--json", action="store_true", help="输出机器可读汇总")
    args = parser.parse_args(argv)

    reports = {arm: load_arm(arm) for arm in ARMS}
    headlines = {arm: arm_headline(reports[arm]) for arm in ARMS}
    rows = tick_table(reports["treatment"], reports["control"])
    if args.json:
        print(json.dumps({"arms": headlines, "stages": rows}, ensure_ascii=True, indent=2))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

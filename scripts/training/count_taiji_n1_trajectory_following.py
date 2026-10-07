"""Count the frozen attribution readings of PLAN-N1-02 (§3 F1–F5) from a v43 artifact.

预注册把判别规则先冻了，本脚本只做两件事：把逐预测步行（`surface_checks[].trajectory_follow_v43`）
按生成行的群体归属分组（被困 never-LF 拖写代＝主群；全体拖写代与提前停止代＝对照），再按冻结
定义算 F1/F2/F1c/F3/F4/F5 并套用 H-A 的三带判别。任何对不上都拒绝出版（rc=2）：面指纹不符、
行为读数不复现基线（eaters 247／never-LF 236／stoppers 41）、轨迹行总数 ≠ 环内步总数、
surface_checks 与 endstep 行的轮数或步数失配——宁可拒绝，不出一个看着合理的数。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: v37 基线在库件的行为读数（PLAN-N1-02 §4 G1′ 行为同一性守卫）。
BASELINE = {"eaters": 247, "never_lf": 236, "stoppers": 41}


def _classify_generations(report: dict[str, Any]) -> list[dict[str, Any]]:
    """把每条生成行标上群体归属，并把它轮内的轨迹行带上；任何失配都是拒绝，不是零。"""
    max_length = report.get("max_length")
    if not isinstance(max_length, (int, float)):
        raise ValueError("件里没有 max_length ⇒ 不判")
    generations: list[dict[str, Any]] = []
    for item in report.get("per_item", []):
        checks = item.get("surface_checks", [])
        endsteps = item.get("endstep_probe_v22", [])
        if len(checks) != len(endsteps):
            raise ValueError(f"{item.get('id')}: 轮数 surface_checks≠endstep ⇒ 不判")
        for check, endstep in zip(checks, endsteps, strict=True):
            if check.get("fed_bytes") != endstep.get("generation_steps"):
                raise ValueError(f"{item.get('id')}: fed_bytes≠generation_steps ⇒ 不判")
            rows = check.get("trajectory_follow_v43")
            if not isinstance(rows, list):
                raise ValueError(f"{item.get('id')}: 轮内缺 trajectory_follow_v43 ⇒ 不判")
            eater = bool(endstep["generation_steps"] >= max_length)
            lf_count = endstep.get("lf_trace_v29", {}).get("lf_step_count")
            if lf_count is None:
                raise ValueError(f"{item.get('id')}: 生成行缺 lf_step_count ⇒ 不判")
            generations.append(
                {
                    "item": str(item.get("id")),
                    "group": (
                        "never_lf_eater"
                        if eater and lf_count == 0
                        else ("with_lf_eater" if eater else "stopper")
                    ),
                    "rows": rows,
                    "steps": int(endstep["generation_steps"]),
                }
            )
    return generations


def _group_readings(generations: list[dict[str, Any]]) -> dict[str, Any]:
    """冻结读数（PLAN-N1-02 §3）；资格集按读数各自的定义，事件缺席行一律排除在判别集外。"""
    rows = [row for gen in generations for row in gen["rows"]]
    absent = sum(1 for row in rows if row.get("event_absent"))
    eligible = [row for row in rows if not row.get("event_absent")]
    follow_rows = [row for row in eligible if row.get("follow") is not None]
    if not follow_rows:
        raise ValueError("判别集为空 ⇒ 不判")
    follows = [bool(row["follow"]) for row in follow_rows]
    mass_hits = [bool(row["mass_hit"]) for row in follow_rows]
    chances = [float(row["succ_mult"]) for row in follow_rows if row.get("succ_mult") is not None]
    if len(chances) != len(follow_rows):
        raise ValueError("succ_mult 缺行 ⇒ F1c 资格集与 F1 不一致，不判")
    last_rows = eligible
    last_hits = [bool(row["emitted_next"] == row["last_byte"]) for row in last_rows]
    gates = sorted(float(row["gate"]) for row in follow_rows)
    succ_ws = sorted(
        float(row["succ_w"]) for row in follow_rows if row.get("succ_w") is not None
    )
    f1 = sum(follows) / len(follows)
    f2 = sum(mass_hits) / len(mass_hits)
    f1c = sum(chances) / len(chances)
    f3 = sum(last_hits) / len(last_rows)
    f4_median = gates[len(gates) // 2]
    f4_p90 = gates[min(len(gates) - 1, int(len(gates) * 0.9))]
    f5_median = succ_ws[len(succ_ws) // 2] if succ_ws else None
    return {
        "prediction_steps": len(rows),
        "event_absent_steps": absent,
        "eligible_steps": len(eligible),
        "F1_follow_rate": round(f1, 6),
        "F2_mass_hit_rate": round(f2, 6),
        "F1c_chance_baseline": round(f1c, 6),
        "F2_minus_F1": round(f2 - f1, 6),
        "F1_minus_chance": round(f1 - f1c, 6),
        "F3_last_byte_rate": round(f3, 6),
        "F4_gate_median": round(f4_median, 4),
        "F4_gate_p90": round(f4_p90, 4),
        "F5_succ_weight_median": None if f5_median is None else round(f5_median, 6),
    }


def _ha_verdict(group: dict[str, Any]) -> str:
    """冻结三带判别（PLAN-N1-02 §3）：成立 / 非主因 / 中间带。只对主群（never-LF 拖写代）下判。"""
    gap = group["F2_minus_F1"]
    above_chance = group["F1_minus_chance"]
    if gap >= 0.20 and above_chance <= 0.10:
        return "confirmed"
    if gap <= 0.10 or above_chance >= 0.20:
        return "not_primary"
    return "middle_band"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True, help="v43 停止面件（--trajectory-following 档）")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args(argv)

    path = Path(args.report)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    entry: dict[str, Any] = {"report": path.name}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        entry.update({"status": "unreadable", "error": str(error)[:200]})
        print(json.dumps({"results": [entry], "rc": 2}, ensure_ascii=False, indent=2))
        return 2

    try:
        if report.get("format") != "taiji-a30-stop-failure-v43":
            raise ValueError(f"format={report.get('format')} 不是 v43 轨迹档 ⇒ 不判")
        if report.get("trajectory_following") is not True:
            raise ValueError("件不是 --trajectory-following 档 ⇒ 不判")
        if report.get("items_sha256") != "0541a3f4568a9c5b":
            raise ValueError("题面指纹不符（须与 v37 基线同面）⇒ 不判")
        generations = _classify_generations(report)
        groups: dict[str, list[dict[str, Any]]] = {
            "never_lf_eaters": [],
            "all_eaters": [],
            "stoppers": [],
        }
        for gen in generations:
            if gen["group"] in ("never_lf_eater", "with_lf_eater"):
                groups["all_eaters"].append(gen)
            if gen["group"] == "never_lf_eater":
                groups["never_lf_eaters"].append(gen)
            if gen["group"] == "stopper":
                groups["stoppers"].append(gen)
        behavior = {
            "eaters": sum(1 for gen in generations if gen["group"] != "stopper"),
            "never_lf": sum(1 for gen in generations if gen["group"] == "never_lf_eater"),
            "stoppers": sum(1 for gen in generations if gen["group"] == "stopper"),
        }
        if behavior != BASELINE:
            raise ValueError(f"行为读数 {behavior} 不复现 v37 基线 {BASELINE} ⇒ 仪器改了行为，整件作废")
        total_rows = sum(len(gen["rows"]) for gen in generations)
        total_steps = sum(gen["steps"] for gen in generations)
        if total_rows != total_steps:
            raise ValueError(f"轨迹行 {total_rows} ≠ 环内步 {total_steps} ⇒ 1:1 破了，不判")
        readings = {name: _group_readings(gens) for name, gens in groups.items()}
        entry.update(
            {
                "status": "ok",
                "behavior_baseline_check": behavior,
                "trajectory_rows_total": total_rows,
                "group_readings": readings,
                "HA_verdict_on_primary_group": _ha_verdict(readings["never_lf_eaters"]),
            }
        )
    except ValueError as error:
        entry.update({"status": "refused", "error": str(error)[:300]})
        print(json.dumps({"results": [entry], "rc": 2}, ensure_ascii=False, indent=2))
        return 2

    payload = {"format": "taiji-n1-trajectory-following-v1", "results": [entry], "rc": 0}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_report:
        out = Path(args.out_report)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

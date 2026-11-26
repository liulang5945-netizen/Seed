"""N177.6 评价报告的**验收器**（只读）：给终评再加一把独立尺子。

评价器自己会算门；本器不重算任何数，只核对"这份报告是否真的回答了预注册的问题"：

* 15 臂是否都在 `fully_scored_holdout_arms` 里（缺臂的报告不得当结论读）；
* SPEC-N1-177.5 v2 §绝对任务线与学习门要求的**键**是否逐条在场（阈值本身钉在评价器代码里，
  本器只核对每个门都有对应读数，不替它重算）；
* 两条对照（no_update／parser_permuted）是否各有三 seed 配对；
* 状态写入消融、F0 保持三 seed、归因五件是否齐全；
* holdout 只读一次的记账：`evaluation_attempts == 1`、无 superseded 报告。

任一不齐 ⇒ rc=1 并点名第一处缺口。它不判能力，只判"报告是否可读作结论"。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports/taiji_n1_plan177_6_t1_pilot_20261027.json"
DEFAULT_STATE = ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/run_state.json"
ARMS = 15

#: SPEC §绝对任务线（逐条照抄）。
ABSOLUTE_GATES = {
    "f0_answer_exact": 0.85,
    "f0_valid_utf8": 0.99,
    "f0_stop_marker": 0.95,
    "f1_entity_span_exact": 0.90,
    "f1_attribute_span_exact": 0.90,
    "f1_value_span_exact": 0.90,
    "f1_post_correction_state_exact": 0.90,
    "f1_post_correction_answer_exact": 0.90,
    "f1_unrelated_keep_exact": 0.95,
}
#: 误写率是"≤ 5%"，方向与上面相反，单独列。
FALSE_WRITE_MAX = 0.05
#: SPEC §主学习门。
LEARNING_MEDIAN_MIN = 0.10
LEARNING_MIN_SEEDS_POSITIVE = 2
LEARNING_WORST_ALLOWED = -0.05


def verify(report: dict[str, Any], state: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    if report.get("format") != "taiji-n177-6-t1-evaluation-v1":
        problems.append(f"format={report.get('format')!r}")
    if report.get("arms_completed") != ARMS:
        problems.append(f"arms_completed={report.get('arms_completed')!r} (need {ARMS})")
    #: 真实报告里这个字段是**计数**（int）；早期版本曾是列表。两种形状都要能读，别因为形状判"缺口"。
    scored = report.get("fully_scored_holdout_arms")
    scored_count = scored if isinstance(scored, int) else len(scored or [])
    if scored_count != ARMS:
        problems.append(f"fully_scored_holdout_arms={scored_count} (need {ARMS})")
    if report.get("sealed_reads") not in (None, 0, 1):
        problems.append(f"sealed_reads={report.get('sealed_reads')!r}")
    if state.get("evaluation_attempts") != 1:
        problems.append(
            f"evaluation_attempts={state.get('evaluation_attempts')!r} (holdout must be read once)"
        )
    if state.get("evaluation_report_superseded") is not None:
        problems.append("evaluation_report_superseded is set: a previous report was overwritten")

    #: SPEC：均值裁决之外还要"另列逐 seed 读数" ⇒ 逐臂面必须在件里，且带各门要用的列。
    per_level_keys = {
        "F0": ("answer_exact_rate", "valid_utf8_rate", "stop_marker_rate"),
        "F1": (
            "answer_exact_rate",
            "f1_state_query_score",
            "entity_span_exact_rate",
            "attribute_span_exact_rate",
            "value_span_exact_rate",
            "f1_post_correction_state_exact",
            "f1_post_correction_answer_exact",
            "f1_unrelated_keep_exact",
            "f1_false_write_rate",
        ),
    }
    splits = report.get("query_and_holdout")
    if not isinstance(splits, list) or {entry.get("split") for entry in splits} != {
        "query",
        "holdout",
    }:
        problems.append("query_and_holdout must carry exactly the query and holdout splits")
    else:
        holdout = next(entry for entry in splits if entry.get("split") == "holdout")
        arms = holdout.get("arms") or []
        if len(arms) != ARMS:
            problems.append(f"holdout arms={len(arms)} (per-seed readouts need {ARMS})")
        for arm in arms:
            by_level = arm.get("by_level") or {}
            for level, keys in per_level_keys.items():
                macro = (by_level.get(level) or {}).get("family_macro") or {}
                for key in keys:
                    if key not in macro:
                        problems.append(
                            f"holdout {arm.get('mode')}/seed{arm.get('seed')} {level} missing {key}"
                        )
                        break
    gates = report.get("absolute_gates") or {}
    means = report.get("absolute_means") or {}
    for name in ABSOLUTE_GATES:
        if name not in gates:
            problems.append(f"absolute_gates missing {name}")
        if name not in means:
            problems.append(f"absolute_means missing {name} (gate would have no reading)")
    if "f1_false_write_rate" not in gates:
        problems.append("absolute_gates missing f1_false_write_rate")
    if "f1_false_write_rate" not in means:
        problems.append("absolute_means missing f1_false_write_rate")

    thresholds = report.get("thresholds")
    if thresholds is not None:
        for name, expected in ABSOLUTE_GATES.items():
            if thresholds.get(name) not in (None, expected):
                problems.append(f"threshold drift on {name}: {thresholds.get(name)} != {expected}")
    if report.get("f1_false_write_threshold") not in (None, FALSE_WRITE_MAX):
        problems.append("false-write threshold is not the frozen 5%")

    contrasts = report.get("learning_contrasts") or {}
    for control in ("no_update", "parser_permuted"):
        item = contrasts.get(control)
        if not isinstance(item, dict):
            problems.append(f"learning_contrasts missing {control}")
            continue
        pairs = item.get("paired_seed_deltas") or []
        if len(pairs) != 3:
            problems.append(f"{control}: paired_seed_deltas={len(pairs)} (need 3 seeds)")
        for key in ("median_delta", "positive_seed_count", "minimum_delta", "passed"):
            if key not in item:
                problems.append(f"{control}: missing {key}")
        #: 从逐 seed 差值**独立重算**聚合字段与门，别只信报告里那三个数。
        values = [float(pair["delta"]) for pair in pairs]
        if values:
            if len(values) != 3:
                continue
            recomputed_median = statistics.median(values)
            recomputed_positive = sum(1 for value in values if value > 0.0)
            recomputed_min = min(values)
            if (
                item.get("median_delta") is None
                or abs(item["median_delta"] - recomputed_median) > 1e-12
            ):
                problems.append(f"{control}: median_delta disagrees with paired deltas")
            if item.get("positive_seed_count") != recomputed_positive:
                problems.append(f"{control}: positive_seed_count disagrees with paired deltas")
            if (
                item.get("minimum_delta") is None
                or abs(item["minimum_delta"] - recomputed_min) > 1e-12
            ):
                problems.append(f"{control}: minimum_delta disagrees with paired deltas")
            expected_passed = (
                recomputed_median >= LEARNING_MEDIAN_MIN
                and recomputed_positive >= LEARNING_MIN_SEEDS_POSITIVE
                and recomputed_min >= LEARNING_WORST_ALLOWED
            )
            if item.get("passed") is not expected_passed:
                problems.append(
                    f"{control}: passed flag disagrees with the frozen gate on the paired deltas"
                )
    if report.get("state_write_diagnostic_required") is False:
        problems.append("state-write diagnostic was declared not required")
    if not report.get("state_write_diagnostics"):
        problems.append("state_write_diagnostics is empty (the causal gate would be unreadable)")
    diagnostics = report.get("state_write_diagnostics") or []
    for row in diagnostics:
        pairs = row.get("paired_family_deltas") or []
        if row.get("paired_family_count") != len(pairs):
            problems.append(
                "state_write_diagnostics: paired_family_count != len(paired_family_deltas)"
            )
        for pair in pairs:
            if abs((pair["write_enabled"] - pair["write_disabled"]) - pair["delta"]) > 1e-12:
                problems.append(
                    "state_write_diagnostics: a delta does not equal write_enabled - write_disabled"
                )
                break
    retention = report.get("f0_retention") or []
    if len(retention) != 3:
        problems.append(f"f0_retention={len(retention)} (need 3 seeds)")
    attribution = report.get("architecture_attribution") or {}
    for key in (
        "information_path",
        "semantic_feature_bridge",
        "credit_path",
        "state_lifecycle",
        "conclusion",
    ):
        if key not in attribution:
            problems.append(f"architecture_attribution missing {key}")
    if report.get("parameters_updated_during_scoring") is not False:
        problems.append("parameters_updated_during_scoring must be false")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177.6 评价报告验收（只读）")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--run-state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args(argv)
    if not args.report.is_file() or not args.run_state.is_file():
        print(f"MISSING_INPUTS report={args.report.is_file()} state={args.run_state.is_file()}")
        return 2
    report = json.loads(args.report.read_text(encoding="utf-8"))
    state = json.loads(args.run_state.read_text(encoding="utf-8"))
    problems = verify(report, state)
    if problems:
        for problem in problems:
            print("PROBLEM", problem)
        print(f"VERDICT not_publishable problems={len(problems)}")
        return 1
    print(
        f"VERDICT publishable arms={report['arms_completed']} "
        f"median_delta(no_update)={((report.get('learning_contrasts') or {}).get('no_update') or {}).get('median_delta')} "
        f"median_delta(parser_permuted)={((report.get('learning_contrasts') or {}).get('parser_permuted') or {}).get('median_delta')} "
        f"attempts={state.get('evaluation_attempts')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

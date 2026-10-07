"""Freeze the R4 growth thresholds from a v1 pressure face (PLAN-N3-01 步骤一).

面是 `train_seed_corpus.py --pressure-record` 跑出来的 JSONL：首行 `kind=face` 自述挂载与配方，
其余行 `kind=pressure` 是每 tick 的原生观测（五信号＋合成 `pressure`＋`should_propose`）。

本脚本只做两件**先于数冻结**的事（判据文本在 `plans/reference/PLAN-N3-01_r4_hooks_prereg_20261007.md` §1）：
① 按分位取分布；② 把 `minimum_*` 六个阈值各自定在"实测 p90 向上取整到 0.05 格"，`required_pressure_steps` 取 3。
任何一项不符都拒绝出版（rc=2）而不是凑一个数：没有 face 行、没有观测行、观测行数低于样本下限、
`pressure` 与五信号的加权和不自洽（说明记录器或生成链被动过）、缺控制面所以口径对照做不了。
口径对照是预注册里那条**否证**：`pressure` 的分位若与 `1 − online_accuracy` 差过 0.1，判"这套信号与
accuracy 口径不同源"，此时**不冻阈值**、只把两侧读数如实登记。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: `AdaptiveResidualGrowthPressure.pressure` 的权重（taiji/adaptive_residual_growth.py:83-88）。
#: 这里**不重算生成链**，只用来核验记录件自洽：加权和与记下来的 `pressure` 必须逐位对得上。
PRESSURE_WEIGHTS = {
    "residual_error": 0.30,
    "fast_slow_conflict": 0.25,
    "activity_saturation": 0.20,
    "utility_gap": 0.25,
}

FIELDS = ("pressure", *PRESSURE_WEIGHTS, "resource_state")

#: 样本下限：低于这个观测数就认为分布不可用（4000 步的面正常应有数千行）。
MIN_OBSERVATIONS = 500

#: 覆盖率下限：观测数必须覆盖面内 tick 数的这个比例，否则判为被截断（见 _load_face 的注释）。
MIN_COVERAGE = 0.95

#: 口径对照的容差（PLAN-N3-01 §1 冻结规则里"差 >0.1 ⇒ 不冻"的那个 0.1）。
CALIBER_TOLERANCE = 0.10

GRID = 0.05


def _percentile(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * fraction))]


def _freeze(value: float) -> float:
    """p90 向上取整到 0.05 格，封顶 1.0（`_unit` 校验要求阈值落在 [0,1]）。"""
    return round(min(1.0, math.ceil(value / GRID) * GRID), 4)


def _load_face(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    header: dict[str, Any] | None = None
    tail: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        kind = record.get("kind")
        if kind == "face":
            if header is not None:
                raise ValueError("件里有两条 face 行 ⇒ 面不唯一，不判")
            header = record
        elif kind == "tail":
            tail = record
        elif kind == "pressure":
            rows.append(record)
        else:
            raise ValueError(f"不认识的行 kind={kind!r} ⇒ 不判")
    if header is None:
        raise ValueError("缺 face 行 ⇒ 这张面没有自述，不判")
    if tail is None:
        raise ValueError("缺 tail 行 ⇒ 面没写到收尾自述（跑中断掉或未走完整预算），不判")
    if int(tail.get("records_written", -1)) != len(rows):
        raise ValueError(
            f"tail 自述记了 {tail.get('records_written')} 条，实际 {len(rows)} 条 ⇒ 面不自洽，不判"
        )
    if not rows:
        raise ValueError("零压强观测 ⇒ 正是预注册要防的静默空转（读出链或喂法不对），不判")
    if len(rows) < MIN_OBSERVATIONS:
        raise ValueError(f"观测只有 {len(rows)} 条，低于样本下限 {MIN_OBSERVATIONS} ⇒ 不判")
    #: 覆盖率守卫：面内 tick 数与观测数必须同阶。第一版栽过一次——周期性 holdout 探针之后
    #: 压强支不再产出观测，而进度行照涨，分布只剩最前面一小段且看不出来（2026-10-08 实测）。
    first_tick = int(rows[0]["tick"])
    span = int(tail["ticks_at_close"]) - first_tick + 1
    coverage = len(rows) / span if span > 0 else 0.0
    if coverage < MIN_COVERAGE:
        raise ValueError(
            f"观测只覆盖面内 tick 的 {coverage:.1%}（{len(rows)}/{span}）⇒ 分布被截断，不判"
        )
    return header, rows, tail


def _online_accuracy_last(progress_path: Path) -> float:
    last: dict[str, Any] | None = None
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = json.loads(line)
    if last is None or "online_accuracy" not in last:
        raise ValueError(f"控制面 {progress_path.name} 里没有 online_accuracy ⇒ 口径对照做不了")
    return float(last["online_accuracy"])


def summarize(pressure_path: Path, control_progress_path: Path | None) -> dict[str, Any]:
    header, rows, tail = _load_face(pressure_path)

    #: 自洽核验：记下来的 `pressure` 必须等于五信号的加权和（同 `pressure` 属性的式子）。
    worst_delta = 0.0
    for row in rows:
        recomputed = sum(weight * float(row[field]) for field, weight in PRESSURE_WEIGHTS.items())
        worst_delta = max(worst_delta, abs(recomputed - float(row["pressure"])))
    if worst_delta > 1e-6:
        raise ValueError(f"pressure 与五信号加权和最大偏差 {worst_delta:.3e} ⇒ 记录件不自洽，不判")

    distributions: dict[str, dict[str, float]] = {}
    for field in FIELDS:
        values = sorted(float(row[field]) for row in rows)
        distributions[field] = {
            "p50": round(_percentile(values, 0.50), 6),
            "p90": round(_percentile(values, 0.90), 6),
            "p99": round(_percentile(values, 0.99), 6),
            "max": round(values[-1], 6),
            "mean": round(sum(values) / len(values), 6),
        }

    frozen = {
        f"minimum_{field}": _freeze(distributions[field]["p90"])
        for field in FIELDS
        if field != "pressure"
    }
    frozen["minimum_pressure"] = _freeze(distributions["pressure"]["p90"])
    frozen["required_pressure_steps"] = 3

    propose_share = sum(1 for row in rows if row.get("decision_should_propose")) / len(rows)

    #: 派生算式（不是新观测量）：当 `fast_slow_conflict` 与 `activity_saturation` 整场恒零时，
    #: `utility_gap = 0.5 × residual_error`，于是 `pressure = 0.30r + 0.25(0.5r) = 0.425r ≤ 0.425`。
    #: 这一支的用处：把"两个信号没参与"从一条观察升成一条**上限**，让"默认阈够不够得着"变成算术问题。
    both_zero = all(
        distributions[field]["max"] == 0.0
        for field in ("fast_slow_conflict", "activity_saturation")
    )
    ceiling = (
        round(PRESSURE_WEIGHTS["residual_error"] + PRESSURE_WEIGHTS["utility_gap"] * 0.5, 6)
        if both_zero
        else None
    )

    #: 09 的现行口径推得值（accuracy 0.59 ⇒ 残差阈 ≥0.4）在这里**复算**而不是沿用。
    caliber: dict[str, Any]
    if control_progress_path is None:
        caliber = {"status": "missing_control_face", "verdict": "not_frozen"}
        verdict = "not_frozen_caliber_unchecked"
    else:
        accuracy = _online_accuracy_last(control_progress_path)
        one_minus = 1.0 - accuracy
        residual_p90 = distributions["residual_error"]["p90"]
        delta = abs(residual_p90 - one_minus)
        caliber = {
            "status": "measured",
            "control_online_accuracy_last_line": round(accuracy, 6),
            "one_minus_accuracy": round(one_minus, 6),
            "residual_error_p90": round(residual_p90, 6),
            "abs_delta": round(delta, 6),
            "tolerance": CALIBER_TOLERANCE,
        }
        verdict = "frozen" if delta <= CALIBER_TOLERANCE else "not_frozen_caliber_mismatch"

    return {
        "format": "taiji-n3-pressure-thresholds-v1",
        "face": {
            "pressure_file": pressure_path.name,
            "observations": len(rows),
            "bridge_digest": header.get("bridge", {}).get("bridge_digest"),
            "bridge_gate_at_mount": header.get("bridge", {}).get("gate"),
            "trigger_parent_digest": header.get("growth", {}).get("parent_checkpoint_digest"),
            "policy_at_mount": header.get("policy"),
            "readout": header.get("readout"),
            "seed": header.get("seed"),
            "ticks_at_close": int(tail["ticks_at_close"]),
            "coverage_of_face_ticks": round(
                len(rows) / (int(tail["ticks_at_close"]) - int(rows[0]["tick"]) + 1), 6
            ),
        },
        "pressure_identity_max_abs_delta": worst_delta,
        "distributions": distributions,
        "two_signals_zero_throughout": both_zero,
        "derived_pressure_ceiling_with_both_zero": ceiling,
        "frozen_thresholds": frozen,
        "default_policy_should_propose_share": round(propose_share, 6),
        "caliber_check": caliber,
        "threshold_verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pressure", required=True, help="`--pressure-record` 跑出的 JSONL")
    parser.add_argument(
        "--control-progress",
        default=None,
        help="同参不开旗标那一臂的进度 JSONL（口径对照用；缺则阈值不冻）",
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args(argv)

    try:
        pressure_path = Path(args.pressure)
        pressure_path = (
            pressure_path if pressure_path.is_absolute() else PROJECT_ROOT / pressure_path
        )
        control: Path | None = None
        if args.control_progress:
            control = Path(args.control_progress)
            control = control if control.is_absolute() else PROJECT_ROOT / control
        payload = summarize(pressure_path, control)
        rc = 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        payload = {
            "format": "taiji-n3-pressure-thresholds-v1",
            "status": "refused",
            "error": str(error)[:300],
        }
        rc = 2
    else:
        payload["status"] = "ok"
        payload["rc"] = rc
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_report and rc == 0:
        out = Path(args.out_report)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    #: 控制台那一份走 ASCII 转义：拒绝理由里有"⇒"这类字形，GBK 终端上 `print` 会在**已经判完**之后
    #: 抛 UnicodeEncodeError（本机 2026-10-08 实测），把 rc=2 的响亮拒绝伪装成 rc=1 的崩溃。
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())

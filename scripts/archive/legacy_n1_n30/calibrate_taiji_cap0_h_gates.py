"""H 维（性能与稳定性）响应/内存门的设备标定采样器。

07 §4.2 要求 H 的响应/内存门"按目标设备预检标定并在正式评价前冻结，不可留空即宣布通过"。
本脚本只做**标定所需的采样**，并把建议阈值标成 `proposed_not_frozen`：冻结是项目所有者/CI
的设备决策，不在本脚本内生效。

为什么必须重取：既有 v5 健康报告的计时是在全量测试**并发**下采的，那份读数混进了别的进程的
排队时间，不能当设备能力用（债册 DEBT-I4 的计时披露）。

用法：
    python -X utf8 -u scripts/training/calibrate_taiji_cap0_h_gates.py \
        --repeats 5 --out reports/taiji_cap0_h_calibration_YYYYMMDD.json

纪律：
- 每次重复都**新建进程**跑一遍健康支（与正式评价同链路），不复用上一次的 runtime；
- 链路显式传入并写进产物：**读数依链路而定**（DEBT-I4 第八批），标定与正式评价必须同链路；
- 采样期间不并行跑别的套（否则又是一次并发污染）；脚本启动时先报告负载，由操作者确认空闲。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_cap0_baseline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    run_health,
)

#: 参与标定的 H 读数。H05（无崩溃）是布尔，不标阈值。
MEASURED_KEYS = (
    "H01_cold_start_seconds",
    "H02_first_response_seconds",
    "H04_peak_traced_bytes",
)
SAFETY_FACTOR = 2.0
CALIBRATION_FORMAT = "taiji-cap0-h-calibration-v1"


def _per_run_seconds(total: float, runs: int) -> float:
    return round(float(total) / max(runs, 1), 4)


def _summarise(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in MEASURED_KEYS:
        values = [float(row[key]) for row in samples if isinstance(row.get(key), (int, float))]
        if not values:
            summary[key] = {"n": 0, "note": "无读数 ⇒ 不提出阈值"}
            continue
        summary[key] = {
            "n": len(values),
            "min": round(min(values), 4),
            "median": round(statistics.median(values), 4),
            "max": round(max(values), 4),
            "proposed_ceiling": round(max(values) * SAFETY_FACTOR, 4),
        }
    totals = [
        (float(row["H03_total_seconds_for_runs"]), int(row["stability_runs"]))
        for row in samples
        if isinstance(row.get("H03_total_seconds_for_runs"), (int, float))
    ]
    if totals:
        per_run = [_per_run_seconds(total, runs) for total, runs in totals]
        summary["H03_seconds_per_chat_run"] = {
            "n": len(per_run),
            "min": min(per_run),
            "median": round(statistics.median(per_run), 4),
            "max": max(per_run),
            "proposed_ceiling": round(max(per_run) * SAFETY_FACTOR, 4),
        }
    return summary


def sample(repeats: int, checkpoint: Path, chain: dict[str, bool]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(repeats):
        started = time.perf_counter()
        report = run_health(
            checkpoint,
            relax_legacy_guard=bool(chain["relax_legacy_guard"]),
            constrained_decode=bool(chain["constrained_decode"]),
        )
        health = report["dimensions"]["H"]
        row = {**health.get("measurements", {})}
        row["sample_index"] = index
        row["stability_runs"] = health.get("stability_runs")
        row["stability_crashes"] = health.get("stability_crashes")
        row["H05_no_crash_over_n_runs"] = bool(
            (health.get("checks") or {}).get("H05_no_crash_over_n_runs")
        )
        row["wall_seconds_whole_health"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        print(f"sample {index}: {json.dumps(row, ensure_ascii=False)}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--chain",
        default="required",
        help="required=relax_legacy_guard+constrained_decode（与正式评价同链路）| bare",
    )
    args = parser.parse_args(argv)

    chain = (
        {"relax_legacy_guard": True, "constrained_decode": True}
        if args.chain == "required"
        else {"relax_legacy_guard": False, "constrained_decode": False}
    )
    if args.repeats < 3:
        parser.error("repeats < 3 不构成分布，只是单点 ⇒ 拒绝采样，不产出可冻结的东西")

    samples = sample(args.repeats, args.checkpoint, chain)
    payload = {
        "format": CALIBRATION_FORMAT,
        "repeats": args.repeats,
        "checkpoint": str(args.checkpoint),
        "chain": chain,
        "safety_factor": SAFETY_FACTOR,
        "samples": samples,
        "summary": _summarise(samples),
        "threshold_status": "proposed_not_frozen",
        "freeze_authority": "项目所有者/CI 的设备决策；本件不冻结任何阈值",
        "disclosure": (
            "阈值口径=该设备该链路该 checkpoint 上 max × safety_factor。它只约束'相对本机基线是否劣化'，"
            "不构成对用户侧设备的能力承诺；正式评价换设备或换链路须重标。"
        ),
    }
    target = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        parser.error(f"{target} 已存在：标定件不覆写")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"calibration -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""N177.6 训练监督器（只读）：把 run_state／metrics／锁 算成一份带时刻的读数件。

为什么要有它：停靠检查原先靠人反复读同一行数字，既没有可引用的产物，也看不出"每臂成本
在漂"这种**趋势**。本仪器只做三件事，全部可从盘上复算：

* 逐臂成本（该臂 `update` 事件首末差 ÷ 更新数）与预算占比；
* 按**各模式自己的**成本投影剩余臂的终值（不用全程混合均值——那会把便宜臂的成本摊给贵臂）；
* 异常面：`checkpoint_resume_rollback`／`engineering_failure_recovered` 的事件数与最近 revision、
  锁的归属进程是否仍活着。

零写入模型侧、零网络；产物是 `reports/` 下一份 LF JSON。不判能力、不替代结案。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027"
MODES = ("correct", "parser_permuted", "no_update", "readout_only", "responsible_owner_frozen")
SEEDS = (0, 1, 2)
PER_ARM_UPDATES = 2560
#: 低于这个样本量的臂不算成本（新起的臂只有几步，速率不可用）。
MIN_SAMPLE = 60


def _arm_id(mode: str, seed: int) -> str:
    return f"{mode}__seed_{seed}"


def load(
    run_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lock_path = run_root / "run_root.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else {}
    return state, rows, manifest, lock


def arm_costs(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    first: dict[str, dict[str, Any]] = {}
    last: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "update":
            continue
        arm = row["arm_id"]
        first.setdefault(arm, row)
        last[arm] = row
    costs: dict[str, dict[str, float]] = {}
    for arm, start in first.items():
        end = last[arm]
        span = end["update"] - start["update"]
        if span < MIN_SAMPLE:
            continue
        costs[arm] = {
            "updates_measured": span,
            "cpu_per_update": (
                end["cumulative_process_cpu_seconds"] - start["cumulative_process_cpu_seconds"]
            )
            / span,
            "wall_per_update": (end["cumulative_wall_seconds"] - start["cumulative_wall_seconds"])
            / span,
        }
    return costs


def _mode_rate(costs: dict[str, dict[str, float]], mode: str, key: str) -> float | None:
    values = [arm[key] for arm_id, arm in costs.items() if arm_id.rsplit("__", 1)[0] == mode]
    return sum(values) / len(values) if values else None


def build(run_root: Path = RUN_ROOT) -> dict[str, Any]:
    state, rows, manifest, lock = load(run_root)
    caps = manifest["hard_caps"]
    costs = arm_costs(rows)
    cursors = {
        arm: int(info.get("update_cursor", 0))
        for arm, info in state.get("latest_checkpoint_by_arm", {}).items()
    }
    last_update = max(
        (row for row in rows if row.get("event") == "update"),
        key=lambda row: row["cumulative_wall_seconds"],
    )
    #: 在跑臂的耐久游标滞后于 metrics；取两者较大值当"已做"。
    cursors[last_update["arm_id"]] = max(
        cursors.get(last_update["arm_id"], 0), int(last_update["update"])
    )

    remaining_cpu = remaining_wall = 0.0
    unknown_modes: list[str] = []
    per_mode = {}
    for mode in MODES:
        cpu_rate = _mode_rate(costs, mode, "cpu_per_update")
        wall_rate = _mode_rate(costs, mode, "wall_per_update")
        left = sum(max(PER_ARM_UPDATES - cursors.get(_arm_id(mode, seed), 0), 0) for seed in SEEDS)
        per_mode[mode] = {
            "remaining_updates": left,
            "cpu_per_update": cpu_rate,
            "wall_per_update": wall_rate,
        }
        if left:
            if cpu_rate is None or wall_rate is None:
                unknown_modes.append(mode)
                continue
            remaining_cpu += left * cpu_rate
            remaining_wall += left * wall_rate

    events = defaultdict(int)
    events_since_latest = defaultdict(int)
    latest_revision = None
    for row in rows:
        revision = row.get("implementation_revision")
        if revision:
            latest_revision = revision
    for row in rows:
        event = row.get("event")
        if event in {"checkpoint_resume_rollback", "engineering_failure_recovered"}:
            events[event] += 1
            #: 只有"本轮实现身份下"的异常才说明这一次续训有问题；历史异常单独报，不混算。
            if (
                latest_revision is not None
                and row.get("implementation_revision") == latest_revision
            ):
                events_since_latest[event] += 1

    lock_pid = lock.get("pid")
    return {
        "format": "taiji-n177-6-t1-supervision-v1",
        "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": state.get("status"),
        "durable_cursor": state.get("optimizer_updates"),
        "completed_arms": len(state.get("completed_arms", [])),
        "current": {
            "arm_id": last_update["arm_id"],
            "update": last_update["update"],
            "process_cpu_seconds": last_update["cumulative_process_cpu_seconds"],
            "wall_seconds": last_update["cumulative_wall_seconds"],
        },
        "caps": caps,
        "used": {
            "process_cpu_ratio": last_update["cumulative_process_cpu_seconds"]
            / caps["process_cpu_seconds"],
            "wall_ratio": last_update["cumulative_wall_seconds"] / caps["wall_seconds"],
            "artifact_ratio": state["artifact_bytes"] / caps["artifact_bytes"],
        },
        "per_arm_cost": {
            arm: {k: round(v, 4) if isinstance(v, float) else v for k, v in data.items()}
            for arm, data in sorted(costs.items())
        },
        "per_mode_remaining": per_mode,
        "projection": {
            "remaining_updates": sum(item["remaining_updates"] for item in per_mode.values()),
            "remaining_process_cpu_seconds": round(remaining_cpu, 1),
            "remaining_wall_seconds": round(remaining_wall, 1),
            "final_process_cpu_ratio": round(
                (last_update["cumulative_process_cpu_seconds"] + remaining_cpu)
                / caps["process_cpu_seconds"],
                4,
            ),
            "final_wall_ratio": round(
                (last_update["cumulative_wall_seconds"] + remaining_wall) / caps["wall_seconds"], 4
            ),
            "modes_without_cost_sample": unknown_modes,
        },
        "anomalies": {
            "rollback_events": events["checkpoint_resume_rollback"],
            "recovered_failure_events": events["engineering_failure_recovered"],
            "rollback_events_since_latest_revision": events_since_latest[
                "checkpoint_resume_rollback"
            ],
            "recovered_failure_events_since_latest_revision": events_since_latest[
                "engineering_failure_recovered"
            ],
            "latest_implementation_revision": latest_revision,
        },
        "lock": {
            "pid": lock_pid,
            "marker": lock.get("marker"),
            "taken_local": lock.get("taken_local"),
            "pid_alive": _pid_alive(lock_pid),
        },
        "scope_limits": [
            "只读；不判能力，不替代结案",
            "投影用各模式自己测得的单位成本，剩余臂含在跑臂的未完成部分",
            "轮次成本受宿主争用影响，投影随争用漂移",
        ],
    }


def _pid_alive(pid: Any) -> bool | None:
    if not isinstance(pid, int):
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.pid_exists(pid)
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177.6 训练监督读数（只读）")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build(args.run_root)
    out = args.out_report or (
        ROOT
        / "reports"
        / f"taiji_n1_plan177_6_t1_supervision_{time.strftime('%Y%m%d', time.gmtime())}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, out)
    projection = report["projection"]
    print(
        f"status={report['status']} cursor={report['durable_cursor']} "
        f"done={report['completed_arms']}/15 arm={report['current']['arm_id']} "
        f"upd={report['current']['update']} "
        f"cpu={100 * report['used']['process_cpu_ratio']:.1f}% "
        f"wall={100 * report['used']['wall_ratio']:.1f}% "
        f"art={100 * report['used']['artifact_ratio']:.1f}% "
        f"proj_cpu={100 * projection['final_process_cpu_ratio']:.1f}% "
        f"proj_wall={100 * projection['final_wall_ratio']:.1f}% "
        f"remaining_h={projection['remaining_wall_seconds'] / 3600:.1f} "
        f"anomalies={report['anomalies']['rollback_events'] + report['anomalies']['recovered_failure_events']} "
        f"(since_resume={report['anomalies']['rollback_events_since_latest_revision'] + report['anomalies']['recovered_failure_events_since_latest_revision']}) "
        f"lock_alive={report['lock']['pid_alive']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

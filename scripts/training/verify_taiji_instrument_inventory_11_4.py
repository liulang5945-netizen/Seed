"""11.4: inventory the candidate's instrument facets, naming what is measured and what is not.

The card asks for source/scope, missing weights, raw generation, repeat runs, interrupt
recovery, peak memory and p50/p95. This reads the existing cards and marks each facet
`measured`, `partial` or `unmeasured` -- "未测记未测" rather than leaving a blank that
looks like zero.

Reads JSON only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUDGET_CPU_SECONDS = 60
BUDGET_WALL_SECONDS = 120

CARDS = {
    "11.1a": "reports/taiji_core_candidate_manifest_11_1_20261101.json",
    "11.2a": "reports/taiji_core_task_entry_11_2_20261101.json",
    "11.2b": "reports/taiji_core_task_entry_roundtrip_11_2b_20261101.json",
    "11.3a": "reports/taiji_core_task_suite_11_3_20261101.json",
    "11.3b": "reports/taiji_core_abstain_11_3b_20261101.json",
    "11.4b": "reports/taiji_core_device_profile_11_4b_20261101.json",
    "11.4c": "reports/taiji_core_interrupt_recovery_11_4c_20261101.json",
    "11.7": "reports/taiji_core_instrument_11_7_v2_20261101.json",
    "10.20": "reports/taiji_core_capability_closeout_10_20_20261101.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.4 card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    cards = {
        name: json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for name, relative in CARDS.items()
    }

    facets = [
        {
            "facet": "来源/manifest",
            "state": "measured",
            "source": f"11.1a:modules={cards['11.1a']['modules_bound']}",
        },
        {
            "facet": "缺权重/未发布",
            "state": "measured",
            "source": "11.1a:weights 未发布（manifest 只绑摘要与来源）",
        },
        {
            "facet": "原始生成",
            "state": "measured",
            "source": f"11.3b:rows={len(cards['11.3b']['rows'])} 原始 top/熵",
        },
        {
            "facet": "重复运行",
            "state": "measured",
            "source": (
                f"11.7:deterministic={cards['11.7']['deterministic_outputs']}; "
                f"11.2a:two_sessions={cards['11.2a']['two_sessions_identical']}"
            ),
        },
        {
            "facet": "中断恢复",
            "state": (
                "measured"
                if cards["11.4c"]["status"] == "interrupt_recovery_verified"
                else "partial"
            ),
            "source": (
                f"11.4c:hard_exit=os._exit, answers_identical="
                f"{cards['11.4c']['answers_identical']}（子进程写快照后硬退，"
                "另一进程续跑）"
            ),
        },
        {
            "facet": "峰值内存",
            "state": "measured",
            "source": (
                "11.7:start/after_build/after_calls = "
                f"{cards['11.7']['peak_working_set_start_bytes']}/"
                f"{cards['11.7']['peak_working_set_after_build_bytes']}/"
                f"{cards['11.7']['peak_working_set_after_calls_bytes']}"
            ),
        },
        {
            "facet": "p50/p95",
            "state": "measured",
            "source": (
                f"11.7:p50={round(cards['11.7']['p50_seconds'], 4)}/"
                f"p95={round(cards['11.7']['p95_seconds'], 4)}"
            ),
        },
        {
            "facet": "真实性A",
            "state": "unmeasured",
            "source": "未做真实语料/真实入口的真实性读数",
        },
        {
            "facet": "代表机制F",
            "state": "measured",
            "source": (f"10.20:{cards['10.20']['status']}（F 的代表能力未成立）"),
        },
        {
            "facet": "设备H",
            "state": (
                "measured" if cards["11.4b"]["status"] == "device_profile_measured" else "partial"
            ),
            "source": (
                "11.4b:cpu="
                f"{cards['11.4b']['profile']['cpu_logical']} logical/"
                f"{cards['11.4b']['profile']['torch_threads']} torch threads, "
                f"cuda={cards['11.4b']['profile']['cuda_available']}, "
                f"ram_kib={cards['11.4b']['profile']['total_memory_kib']}"
            ),
        },
    ]
    missing = [row["facet"] for row in facets if row["state"] != "measured"]
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = (
        "budget_stop"
        if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
        else "instrument_facets_inventoried"
    )

    report = {
        "schema": "taiji-instrument-inventory-11-4-v1",
        "step": "11.4",
        "date": "2026-11-01",
        "status": status,
        "facets": facets,
        "measured": [row["facet"] for row in facets if row["state"] == "measured"],
        "partial_or_unmeasured": missing,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Inventory only: it says which instrument facets have a card behind them and "
            "which do not. A measured facet is not a capability reading, and the two "
            "unmeasured ones stay unmeasured until someone records them."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (*CARDS.values(),)
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "measured": len(report["measured"]),
                "partial_or_unmeasured": missing,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""11.3c: is any confidence threshold able to separate the two task groups?

11.3b's gate fired on everything. This reads that card and checks whether *any* cut on
top probability or entropy could have separated the five answerable items from the two
that need clarify/abstain. If the ranges overlap, no threshold exists and the channel
cannot be calibrated on this candidate at all.

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

SOURCE = "reports/taiji_core_abstain_11_3b_20261101.json"
BUDGET_CPU_SECONDS = 60
BUDGET_WALL_SECONDS = 120


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.3c card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    source = json.loads((ROOT / SOURCE).read_text(encoding="utf-8"))
    rows = source["rows"]
    scorable = [row for row in rows if row["scored"]]
    unscorable = [row for row in rows if not row["scored"]]
    scorable_p = [row["top_probability"] for row in scorable]
    unscorable_p = [row["top_probability"] for row in unscorable]
    scorable_h = [row["entropy"] for row in scorable]
    unscorable_h = [row["entropy"] for row in unscorable]

    # A threshold exists only if one group's whole range sits strictly on one side.
    p_separable = max(scorable_p) < min(unscorable_p) or max(unscorable_p) < min(scorable_p)
    h_separable = max(scorable_h) < min(unscorable_h) or max(unscorable_h) < min(scorable_h)
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif p_separable or h_separable:
        status = "confidence_separable"
    else:
        status = "confidence_not_separable"

    report = {
        "schema": "taiji-confidence-separable-11-3c-v1",
        "step": "11.3c",
        "date": "2026-11-01",
        "status": status,
        "source": SOURCE,
        "scorable_range_top_p": [min(scorable_p), max(scorable_p)],
        "unscorable_range_top_p": [min(unscorable_p), max(unscorable_p)],
        "scorable_range_entropy": [min(scorable_h), max(scorable_h)],
        "unscorable_range_entropy": [min(unscorable_h), max(unscorable_h)],
        "uniform_entropy_reference": 5.549,
        "p_separable": p_separable,
        "entropy_separable": h_separable,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Separability of a signal, checked on seven prompts: ranges overlap, so no "
            "threshold can split the groups. It does not show the model never has useful "
            "confidence elsewhere, only that this channel cannot be calibrated here."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                SOURCE,
                "scripts/training/verify_taiji_confidence_separable_11_3c.py",
            )
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
                "scorable_p": report["scorable_range_top_p"],
                "unscorable_p": report["unscorable_range_top_p"],
                "entropy": report["scorable_range_entropy"],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

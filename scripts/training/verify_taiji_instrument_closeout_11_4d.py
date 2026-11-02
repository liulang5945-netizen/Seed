"""11.4d: close the instrument work without letting "surfaces complete" read as "usable".

Checks on the inventory v3 and the capability closeout:

* nine of ten instrument facets have a card behind them;
* the one missing facet is realism A, named explicitly;
* the representative mechanism F reading is negative;
* therefore the candidate is not usable, whatever the instrument says.

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

INVENTORY = "reports/taiji_core_instrument_inventory_11_4_v3_20261101.json"
CAPABILITY = "reports/taiji_core_capability_closeout_10_20_20261101.json"
BUDGET_CPU_SECONDS = 60
BUDGET_WALL_SECONDS = 120


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.4d card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    inventory = json.loads((ROOT / INVENTORY).read_text(encoding="utf-8"))
    capability = json.loads((ROOT / CAPABILITY).read_text(encoding="utf-8"))
    measured = inventory["measured"]
    missing = inventory["partial_or_unmeasured"]
    mechanism_negative = capability["status"] == "capability_line_closed_negative"
    surfaces_ok = len(measured) == 9 and missing == ["真实性A"]

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif not surfaces_ok:
        status = "instrument_surfaces_incomplete"
    elif not mechanism_negative:
        status = "mechanism_reading_missing"
    else:
        status = "instrument_closed_pending_realism_and_capability"

    report = {
        "schema": "taiji-instrument-closeout-11-4d-v1",
        "step": "11.4d",
        "date": "2026-11-01",
        "status": status,
        "measured_facets": len(measured),
        "measured": measured,
        "unmeasured": missing,
        "mechanism_status": capability["status"],
        "candidate_usable": False,
        "usable_reason": "representative mechanism F is negative (10.20) and realism A is unmeasured",
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Closeout of the instrument work only: nine of ten facets have a card, realism A "
            "does not, and F is negative. The candidate is therefore not usable, and 11 stays "
            "not admitted; nothing here changes the capability line."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                INVENTORY,
                CAPABILITY,
                "scripts/training/verify_taiji_instrument_closeout_11_4d.py",
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
                "measured": len(measured),
                "unmeasured": missing,
                "mechanism": capability["status"],
                "candidate_usable": False,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

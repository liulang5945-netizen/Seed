"""Record the 22.6.0 fixture: build the schedule, assess it, write one receipt.

This card is deliberately arithmetic-only. It never constructs a model, never
steps the world, and never reads a sealed partition, so the receipt's cost is the
cost of hashing and summing small integer tables.

Refusals are hard: the manifest must be frozen, the report is create-once, and a
missing or altered source pin stops the run before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/manifests/taiji_u2_22_6_0_fixture_20261115.json"
REPORT = ROOT / "reports/taiji_u2_22_6_0_fixture_20261115.json"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def refuse(reason: str) -> int:
    print(json.dumps({"verdict": "refused", "reason": reason}, ensure_ascii=False))
    return 2


def guard(contract: dict) -> str | None:
    if contract.get("status") != "frozen":
        return "contract_not_frozen"
    pins = contract.get("source_pins") or {}
    for name, pin in pins.items():
        path = ROOT / pins_path(name)
        if not path.exists() or sha_of(path) != pin:
            return f"source_pin_mismatch:{name}"
    if REPORT.exists():
        return "report_already_exists"
    return None


def pins_path(name: str) -> str:
    return {
        "module": "taiji/u2_drive_task.py",
        "test": "tests/taiji_native/test_u2_drive_task.py",
    }[name]


def run(contract: dict) -> int:
    sys.path.insert(0, str(ROOT))
    from taiji.u2_drive_task import assess_fixture, build_schedule, schedule_digest

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    rows = []
    for seed in contract["seeds"]:
        schedule = build_schedule(int(seed))
        verdict = assess_fixture(schedule)
        rebuilt = build_schedule(int(seed))
        verdict["deterministic"] = rebuilt["digest"] == schedule["digest"]
        verdict["digest"] = schedule_digest(schedule)
        verdict["seed"] = int(seed)
        rows.append(verdict)

    all_valid = all(row["valid"] and row["deterministic"] for row in rows)
    worst = sorted({failure for row in rows for failure in row["failures"]})
    report = {
        "schema": "taiji-u2-22-6-0-fixture-report-v1",
        "card": "22.6.0",
        "date": contract["date"],
        "verdict": "fixture_valid" if all_valid else "fixture_invalid",
        "failures": worst,
        "seeds": rows,
        "criteria_sha256": contract["criteria_sha256"],
        "parent": contract["parent"],
        "discipline": contract["discipline"],
        "not_measured_here": [
            "policy choice quality",
            "gain/risk/cost predictor",
            "net utility against fixed and reactive controls",
        ],
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {"verdict": report["verdict"], "failures": worst, "seeds": len(rows)},
            ensure_ascii=False,
        )
    )
    return 0 if all_valid else 5


def main() -> int:
    if not MANIFEST.exists():
        return refuse("manifest_missing")
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reason = guard(contract)
    if reason:
        return refuse(reason)
    return run(contract)


if __name__ == "__main__":
    raise SystemExit(main())

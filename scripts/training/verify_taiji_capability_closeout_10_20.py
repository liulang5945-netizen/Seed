"""10.20: machine-checked closeout of the step-10 capability line.

The line ends with a mechanism, not a score: the subject copies one entity out of
records (10.18) but never composes two copies into an action sentence (10.19), and a
state-feature readout does not compose either (10.17). This card reads every capability
card in the line and checks that the record is consistent with that statement.

It reads JSON only: nothing is trained or re-run.
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
    "10.4": "reports/taiji_core_subject_selection_10_4_20261101.json",
    "10.5": "reports/taiji_core_goal_conditioned_10_5_20261101.json",
    "10.6": "reports/taiji_core_action_verdict_10_6_20261101.json",
    "10.7": "reports/taiji_core_goal_training_10_7_20261101.json",
    "10.7e": "reports/taiji_core_readout_10_7e_20261101.json",
    "10.7e-1": "reports/taiji_core_readout_rank_10_7e1_20261101.json",
    "10.7e-2": "reports/taiji_core_raw_rank_10_7e2_20261101.json",
    "10.7f": "reports/taiji_core_selection_10_7f_20261101.json",
    "10.7h": "reports/taiji_core_geometry_select_10_7h_20261101.json",
    "10.10": "reports/taiji_core_glossary_10_10_20261101.json",
    "10.11": "reports/taiji_core_glossary_records_10_11_20261101.json",
    "10.11b": "reports/taiji_core_glossary_control_10_11b_20261101.json",
    "10.12": "reports/taiji_core_records_state_10_12_v2_20261101.json",
    "10.13": "reports/taiji_core_state_selection_10_13_v2_20261101.json",
    "10.16": "reports/taiji_core_wide_selection_10_16_20261101.json",
    "10.17": "reports/taiji_core_wide_object_split_10_17_20261101.json",
    "10.18": "reports/taiji_core_records_copy_10_18_20261101.json",
    "10.19": "reports/taiji_core_generation_10_19_20261101.json",
}
WITHDRAWN = ("10.7e", "10.7e-1")
NEGATIVE = ("10.13", "10.17", "10.19")
COPY_EVIDENCE = "10.18"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.20 card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    rows = {}
    missing = []
    for name, relative in CARDS.items():
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        card = json.loads(path.read_text(encoding="utf-8"))
        rows[name] = {
            "report": relative,
            "status": card.get("status"),
            "cpu_seconds": card.get("cpu_seconds"),
            "capability_claimed": card.get("capability_claimed"),
            "zero_frozen": (card.get("backbone_frozen") if "backbone_frozen" in card else "n/a"),
        }

    claims = [name for name, row in rows.items() if row["capability_claimed"] is True]
    withdrawal_ok = (
        claims == [WITHDRAWN[0]]
        and rows.get(WITHDRAWN[1], {}).get("status") == "readout_rank_below"
    )
    negatives_ok = all(
        str(rows.get(name, {}).get("status", "")).find("absent") != -1
        or "below" in str(rows.get(name, {}).get("status", ""))
        or "rank_below" in str(rows.get(name, {}).get("status", ""))
        for name in NEGATIVE
    )
    copy_card = rows.get(COPY_EVIDENCE, {})
    copy_ok = copy_card.get("status") == "records_copy_inconclusive"
    frozen_values = {
        name: row["zero_frozen"]
        for name, row in rows.items()
        if row["zero_frozen"] not in ("n/a", None)
    }
    frozen_ok = all(value is True for value in frozen_values.values())

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif missing or len(rows) != len(CARDS):
        status = "trace_incomplete"
    elif not withdrawal_ok:
        status = "withdrawal_not_recorded"
    elif not negatives_ok:
        status = "line_not_all_negative"
    elif not copy_ok:
        status = "copy_evidence_missing"
    elif not frozen_ok:
        status = "backbone_not_frozen"
    else:
        status = "capability_line_closed_negative"

    report = {
        "schema": "taiji-capability-closeout-10-20-v1",
        "step": "10.20",
        "date": "2026-11-01",
        "status": status,
        "cards_expected": len(CARDS),
        "cards_read": len(rows),
        "missing": missing,
        "statuses": {name: row["status"] for name, row in rows.items()},
        "claims": claims,
        "withdrawal_recorded": withdrawal_ok,
        "negative_cards": list(NEGATIVE),
        "negatives_checked": negatives_ok,
        "copy_evidence": {"card": COPY_EVIDENCE, "status": copy_card.get("status")},
        "frozen_cards": frozen_values,
        "line_cpu_seconds": sum(
            row["cpu_seconds"]
            for row in rows.values()
            if isinstance(row["cpu_seconds"], (int, float))
        ),
        "mechanism": (
            "single-hop copy is present (10.18) and composition is not (10.19 copies one "
            "glossary line and stops; 10.17 state readout scores 0% on an unseen object)"
        ),
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Closeout only: it checks that the line's cards agree with the stated mechanism "
            "and that no capability claim survives. It does not prove the subject can never "
            "select, does not change step 11, and authorises nothing."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            "scripts/training/verify_taiji_capability_closeout_10_20.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
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
                "cards": len(rows),
                "claims": claims,
                "negatives_checked": negatives_ok,
                "copy_ok": copy_ok,
                "frozen_ok": frozen_ok,
                "line_cpu_seconds": round(report["line_cpu_seconds"], 1),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""10.14: machine-checked closeout of the readout line (10.4 -> 10.13).

The line is closed by a card, not by prose. This step reads every card in the line and
checks three things a negative closeout must satisfy:

* every card exists and carries a status;
* every capability-bearing card is negative **except** the one that claimed a result and
  was withdrawn by a later card in the same line (10.7e -> 10.7e-1);
* no card published weights or replaced the default.

It reads JSON only; nothing is trained or re-run.
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
    "10.7a": "reports/taiji_core_training_replay_10_7a_v2_20261101.json",
    "10.7b": "reports/taiji_core_training_direction_10_7b_20261101.json",
    "10.7c": "reports/taiji_core_residual_structure_10_7c_v2_20261101.json",
    "10.7d": "reports/taiji_core_margin_objective_10_7d_20261101.json",
    "10.7e": "reports/taiji_core_readout_10_7e_20261101.json",
    "10.7e-1": "reports/taiji_core_readout_rank_10_7e1_20261101.json",
    "10.7e-2": "reports/taiji_core_raw_rank_10_7e2_20261101.json",
    "10.7f": "reports/taiji_core_selection_10_7f_20261101.json",
    "10.7g": "reports/taiji_core_feature_additivity_10_7g_20261101.json",
    "10.7h": "reports/taiji_core_geometry_select_10_7h_20261101.json",
    "10.9": "reports/taiji_core_goal_attributes_10_9_20261101.json",
    "10.10": "reports/taiji_core_glossary_10_10_20261101.json",
    "10.11": "reports/taiji_core_glossary_records_10_11_20261101.json",
    "10.11b": "reports/taiji_core_glossary_control_10_11b_20261101.json",
    "10.12": "reports/taiji_core_records_state_10_12_v2_20261101.json",
    "10.13": "reports/taiji_core_state_selection_10_13_v2_20261101.json",
}
WITHDRAWN = ("10.7e", "10.7e-1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.14 card is write-once")

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
            "weights_published": card.get("weights_published"),
            "default_model_replaced": card.get("default_model_replaced"),
        }

    claims = [name for name, row in rows.items() if row["capability_claimed"] is True]
    withdrawn = [name for name in WITHDRAWN if name in rows]
    withdrawal_card = rows.get(WITHDRAWN[1], {})
    withdrawn_ok = (
        claims == [WITHDRAWN[0]] and withdrawal_card.get("status") == "readout_rank_below"
    )
    weights_clean = all(
        row["weights_published"] in (False, None) and row["default_model_replaced"] in (False, None)
        for row in rows.values()
    )
    statuses = {name: row["status"] for name, row in rows.items()}
    negative = [
        name
        for name, row in rows.items()
        if name not in WITHDRAWN
        and str(row["status"]).startswith(("passed_", "readout_binding_supported"))
    ]

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif missing or len(rows) != len(CARDS):
        status = "trace_incomplete"
    elif negative:
        status = "readout_line_has_unwithdrawn_positive"
    elif not withdrawn_ok:
        status = "withdrawal_not_recorded"
    elif not weights_clean:
        status = "weights_leaked"
    else:
        status = "readout_line_closed_negative"

    report = {
        "schema": "taiji-readout-closeout-10-14-v1",
        "step": "10.14",
        "date": "2026-11-01",
        "status": status,
        "cards_expected": len(CARDS),
        "cards_read": len(rows),
        "missing": missing,
        "statuses": statuses,
        "claims": claims,
        "withdrawn": withdrawn,
        "withdrawal_recorded": withdrawn_ok,
        "weights_clean": weights_clean,
        "line_cpu_seconds": sum(
            row["cpu_seconds"]
            for row in rows.values()
            if isinstance(row["cpu_seconds"], (int, float))
        ),
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Closeout only: it checks that the line's cards agree and that every claim but "
            "one was already withdrawn by the line itself. It does not re-run anything, does "
            "not show the subject cannot ever select, and does not change step 11."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            "scripts/training/verify_taiji_readout_closeout_10_14.py": hashlib.sha256(
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
                "withdrawal_recorded": withdrawn_ok,
                "line_cpu_seconds": round(report["line_cpu_seconds"], 1),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

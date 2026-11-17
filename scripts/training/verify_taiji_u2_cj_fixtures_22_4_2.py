"""22.4.2.4 zero-training C/D/J fixture certification.

A/B/unknown oracle readings are cited from the existing temporal report (32/32).
C uses acquisition_fixture (inspect/wait utility), J uses joint_fixture (18
public-oracle checks across three sessions), D (capacity pressure) has no frozen
definition yet and is reported as a blocking gap. No model forward, no optimizer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.u2_task_episode import acquisition_fixture, joint_fixture

REPORT = ROOT / "reports/taiji_u2_cj_fixture_certification_20261117.json"
SEEDS = (124, 125, 126)
AXES = (-0.55, 0.55)
ORACLE_MIN = 0.95


def certify_c() -> dict:
    rows = []
    correct = 0
    for seed in SEEDS:
        for initial_x in AXES:
            for known in (True, False):
                fixture = acquisition_fixture(seed, initial_x, known)
                best = fixture["best"]
                oracle_best = max(fixture["utilities"], key=fixture["utilities"].get)
                answer_ok = fixture["answers"][best] == fixture["truth"]
                oracle_ok = best == oracle_best
                correct += int(answer_ok and oracle_ok)
                rows.append(
                    {
                        "seed": seed,
                        "initial_x": initial_x,
                        "known": known,
                        "best": best,
                        "oracle_best": oracle_best,
                        "answers": fixture["answers"],
                        "truth": fixture["truth"],
                        "utilities": fixture["utilities"],
                        "oracle_ok": oracle_ok,
                        "answer_ok": answer_ok,
                    }
                )
    return {
        "rows": rows,
        "correct": correct,
        "denominator": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
    }


def certify_j() -> dict:
    rows = []
    correct = 0
    denominator = 0
    for seed in SEEDS:
        fixture = joint_fixture(seed)
        checks = fixture["checks"]
        hits = sum(int(item["answer"] == item["expected"]) for item in checks)
        correct += hits
        denominator += len(checks)
        rows.append(
            {
                "seed": seed,
                "sessions": fixture["sessions"],
                "rounds": fixture["rounds"],
                "correct": hits,
                "denominator": len(checks),
                "mismatches": [item for item in checks if item["answer"] != item["expected"]],
            }
        )
    return {
        "rows": rows,
        "correct": correct,
        "denominator": denominator,
        "accuracy": correct / denominator if denominator else 0.0,
    }


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    c = certify_c()
    j = certify_j()
    d = {
        "defined": False,
        "reason": "capacity-pressure task has no frozen definition in the 22.4.2 spec/plan; D is a blocking gap",
    }
    c_ok = c["accuracy"] >= ORACLE_MIN
    j_ok = j["accuracy"] >= ORACLE_MIN
    if not c_ok or not j_ok:
        verdict = "fixture_invalid"
    elif not d["defined"]:
        verdict = "fixture_invalid_d_undefined"
    else:
        verdict = "c_d_j_fixtures_valid"
    report = {
        "schema": "taiji-u2-22-4-2-cdj-fixture-certification-v1",
        "card": "22.4.2.4",
        "phase": "zero_training_fixture_certification",
        "verdict": verdict,
        "c_acquisition": c,
        "j_joint_oracle": j,
        "d_capacity_pressure": d,
        "a_b_unknown_reference": {
            "report": "reports/taiji_u2_temporal_world_22_4_2_20261112.json",
            "groups": {
                "A_initial_current": "32/32",
                "B_previous": "32/32",
                "U_unknown": "32/32",
            },
            "blank_previous_information_ceiling": 0.5,
            "wrong_time_accuracy": 0.5,
            "note": "cited, not rerun; the existing report has 0 model forwards and 0 optimizer steps",
        },
        "gate": {
            "oracle_min": ORACLE_MIN,
            "c_ok": c_ok,
            "j_ok": j_ok,
            "d_defined": d["defined"],
            "all_required_families": c_ok and j_ok and d["defined"],
        },
        "discipline": {
            "model_forwards": 0,
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    if REPORT.exists():
        raise FileExistsError(str(REPORT))
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        "CDJ verdict="
        + verdict
        + " C="
        + str(c["correct"])
        + "/"
        + str(c["denominator"])
        + " J="
        + str(j["correct"])
        + "/"
        + str(j["denominator"])
        + " D="
        + str(d["defined"])
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

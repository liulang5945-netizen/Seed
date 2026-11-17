"""22.4.2.4 zero-training C/D/J fixture certification.

A/B/unknown oracle readings are cited from the existing temporal report (32/32).
C uses acquisition_fixture (inspect/wait utility), J uses joint_fixture (18
public-oracle checks across three sessions), D (capacity pressure) uses two
three-body sub-worlds for six entity observations at capacity 4, with occlusion
and same-appearance/different-dynamics arms. No model forward, no optimizer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.u2_task_capacity import capacity_arms
from taiji.u2_task_contract import request_text
from taiji.u2_task_episode import acquisition_fixture, joint_fixture, public_oracle

REPORT = ROOT / "reports/taiji_u2_cdj_fixture_certification_20261117_v2.json"
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


def certify_d() -> dict:
    rows = []
    standard_correct = standard_denominator = 0
    occluded_correct = occluded_denominator = 0
    blank_correct = 0
    dynamics_ok = True
    sample = None
    for seed in SEEDS:
        arms = capacity_arms(seed)
        sample = sample or arms["standard"]
        standard = arms["standard"]
        standard_hits = sum(int(item["answer"] == item["expected"]) for item in standard["checks"])
        standard_correct += standard_hits
        standard_denominator += len(standard["checks"])
        blank_correct += sum(
            int(
                public_oracle(request_text(item["entity"], item["temporal"]), []).decode()
                == item["expected"]
            )
            for item in standard["checks"]
        )
        occluded = arms["occlusion"]
        occluded_hits = sum(int(item["answer"] == item["expected"]) for item in occluded["checks"])
        occluded_correct += occluded_hits
        occluded_denominator += len(occluded["checks"])
        dynamics = arms["dynamics"]
        baseline_hits = sum(
            int(item["answer"] == item["expected"]) for item in dynamics["baseline_checks"]
        )
        altered_hits = sum(
            int(item["answer"] == item["expected"]) for item in dynamics["altered_checks"]
        )
        seed_ok = (
            dynamics["first_frames_equal"]
            and dynamics["last_answers_differ"]
            and baseline_hits == len(dynamics["baseline_checks"])
            and altered_hits == len(dynamics["altered_checks"])
        )
        dynamics_ok = dynamics_ok and seed_ok
        rows.append(
            {
                "seed": seed,
                "standard": {"correct": standard_hits, "denominator": len(standard["checks"])},
                "occlusion": {
                    "correct": occluded_hits,
                    "denominator": len(occluded["checks"]),
                    "occluded_entity": occluded["occluded_entity"],
                },
                "dynamics": {
                    "baseline_correct": baseline_hits,
                    "altered_correct": altered_hits,
                    "first_frames_equal": dynamics["first_frames_equal"],
                    "last_answers_differ": dynamics["last_answers_differ"],
                    "altered_mass": dynamics["altered_mass"],
                    "altered_drag": dynamics["altered_drag"],
                    "seed_ok": seed_ok,
                },
            }
        )
    return {
        "rows": rows,
        "standard_correct": standard_correct,
        "standard_denominator": standard_denominator,
        "standard_accuracy": (
            standard_correct / standard_denominator if standard_denominator else 0.0
        ),
        "occlusion_correct": occluded_correct,
        "occlusion_denominator": occluded_denominator,
        "occlusion_accuracy": (
            occluded_correct / occluded_denominator if occluded_denominator else 0.0
        ),
        "blank_correct": blank_correct,
        "blank_accuracy": blank_correct / standard_denominator if standard_denominator else 0.0,
        "dynamics_ok": dynamics_ok,
        "capacity": sample["capacity"],
        "entities": sample["entities"],
        "worlds": sample["worlds"],
        "world_body_limit": sample["world_body_limit"],
        "frames": sample["frames"],
        "rounds": sample["rounds"],
        "observed_facts": sample["observed_facts"],
        "pressure_ratio": sample["pressure_ratio"],
    }


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    c = certify_c()
    j = certify_j()
    d = certify_d()
    d_ok = (
        d["standard_accuracy"] >= ORACLE_MIN
        and d["occlusion_accuracy"] >= ORACLE_MIN
        and d["blank_accuracy"] <= 0.60
        and d["dynamics_ok"]
    )
    c_ok = c["accuracy"] >= ORACLE_MIN
    j_ok = j["accuracy"] >= ORACLE_MIN
    verdict = "fixture_invalid" if not c_ok or not j_ok or not d_ok else "c_d_j_fixtures_valid"
    report = {
        "schema": "taiji-u2-22-4-2-cdj-fixture-certification-v1",
        "card": "22.4.2.4",
        "phase": "zero_training_fixture_certification",
        "verdict": verdict,
        "c_acquisition": c,
        "j_joint_oracle": j,
        "d_capacity_pressure": {**d, "defined": True},
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
            "d_ok": d_ok,
            "all_required_families": c_ok and j_ok and d_ok,
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
        + " Dstd="
        + str(d["standard_correct"])
        + "/"
        + str(d["standard_denominator"])
        + " Doccl="
        + str(d["occlusion_correct"])
        + "/"
        + str(d["occlusion_denominator"])
        + " blank="
        + str(round(d["blank_accuracy"], 4))
        + " dyn="
        + str(d["dynamics_ok"])
        + " cpu="
        + str(report["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

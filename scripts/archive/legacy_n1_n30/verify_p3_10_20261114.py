"""Guard for the P3.10 dose curve.

Recomputes every dose accuracy from its bundle, re-derives the frozen verdict,
checks the collapse fact and the discipline counters, and proves the guard rings
with in-memory and real-file mutations. No training, no optimizer step.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import evaluate, load_learner
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_10_execution_20261114.json"
REPORT = ROOT / "reports/taiji_u2_p3_10_curve_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_10_guard_20261114.json"
GATE = 0.80
FLAT = 0.05


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def verdict_for(zero: float, accuracy: dict[str, float]) -> str:
    if zero < GATE <= accuracy.get("256", -1.0):
        return "dose_curve_measured"
    if accuracy.get("256", -1.0) < GATE <= accuracy.get("1844", -1.0):
        return "gate_dose_infeasible"
    if accuracy.get("1844", 0.0) < GATE and accuracy["1844"] - accuracy.get("64", 0.0) < FLAT:
        return "recipe_scoped_flat"
    return "inconclusive_curve"


def recompute(contract: dict, report: dict) -> dict:
    data = atomic_load(Path(contract["data_root"]))
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    out = {}
    for dose in report["doses"]:
        model, _, _ = load_learner(Path(contract["run_root"]) / f"dose-{dose['update']:04d}.bin")
        scored = evaluate(model, data["fit"])
        outputs = [row["output_hex"] for row in scored["rows"]]
        counts = collections.Counter(outputs)
        modal_bytes = counts.most_common(1)[0][0]
        out[str(dose["update"])] = {
            "accuracy": scored["accuracy"],
            "correct": scored["correct"],
            "distinct_outputs": len(counts),
            "modal_count": counts.most_common(1)[0][1],
            "modal_bytes": modal_bytes,
            "modal_target_matches": sum(
                1 for row in scored["rows"] if row["target"].encode().hex() == modal_bytes
            ),
        }
    return out


def evaluate_checks(contract: dict, report: dict, recomputed: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_frozen"] = contract.get("status") == "frozen"
    checks["authority_present"] = bool(contract.get("training_authority"))
    checks["caps_set"] = bool(contract.get("cpu_cap")) and bool(contract.get("wall_cap"))
    checks["criteria_sha_matches"] = (
        hashlib.sha256(canonical(contract["criteria"])).hexdigest() == contract["criteria_sha256"]
    )
    accuracy = {key: value["accuracy"] for key, value in recomputed.items()}
    zero = report["zero"]["accuracy"]
    checks["verdict_matches_rule"] = report["verdict"] == verdict_for(zero, accuracy)
    checks["dose_accuracy_matches_recompute"] = all(
        dose["accuracy"] == accuracy[str(dose["update"])] for dose in report["doses"]
    )
    checks["dose_correct_matches_recompute"] = all(
        dose["correct"] == recomputed[str(dose["update"])]["correct"] for dose in report["doses"]
    )
    checks["restore_next_loss_equal"] = (
        report["restore"]["next_loss_equal"] is True and report["restore"]["returncode"] == 0
    )
    checks["ledger_recompute_flag"] = report["dose_recompute_matches_ledger"] is True
    final = recomputed["1844"]
    checks["collapse_fact_consistent"] = (
        final["distinct_outputs"] == 1
        and final["modal_count"] == report["doses"][-1]["denominator"]
        and final["modal_target_matches"] == final["correct"]
        and final["correct"] == report["doses"][-1]["correct"]
    )
    discipline = report["discipline"]
    checks["no_capability_claims"] = discipline["capability_claims"] == 0
    checks["no_sealed_reads"] = (
        discipline["check_reads"] == 0
        and discipline["final_reads"] == 0
        and discipline["confirm_reads"] == 0
    )
    checks["wall_within_cap"] = (
        report["cost"]["wall_seconds_from_artifacts"] <= contract["wall_cap"]
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-drills", action="store_true")
    args = parser.parse_args()
    started, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    recomputed = recompute(contract, report)
    checks = evaluate_checks(contract, report, recomputed)

    drills: dict[str, bool] = {}
    real: dict = {"performed": False}
    if not args.skip_drills:
        mutated = copy.deepcopy(report)
        mutated["verdict"] = "dose_curve_measured"
        drills["tamper_verdict"] = not evaluate_checks(contract, mutated, recomputed)[
            "verdict_matches_rule"
        ]
        mutated = copy.deepcopy(report)
        mutated["doses"][-1]["accuracy"] = 0.5
        drills["tamper_accuracy"] = not evaluate_checks(contract, mutated, recomputed)[
            "dose_accuracy_matches_recompute"
        ]
        mutated = copy.deepcopy(report)
        mutated["discipline"]["capability_claims"] = 1
        drills["claim_capability"] = not evaluate_checks(contract, mutated, recomputed)[
            "no_capability_claims"
        ]
        mutated = copy.deepcopy(report)
        mutated["restore"]["next_loss_equal"] = False
        drills["deny_restore"] = not evaluate_checks(contract, mutated, recomputed)[
            "restore_next_loss_equal"
        ]
        original = REPORT.read_bytes()
        before = hashlib.sha256(original).hexdigest()
        tampered = copy.deepcopy(report)
        tampered["doses"][-1]["correct"] = 16
        REPORT.write_text(
            json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        try:
            red = evaluate_checks(
                contract, json.loads(REPORT.read_text(encoding="utf-8")), recomputed
            )
        finally:
            REPORT.write_bytes(original)
        after = hashlib.sha256(REPORT.read_bytes()).hexdigest()
        green = evaluate_checks(
            contract, json.loads(REPORT.read_text(encoding="utf-8")), recomputed
        )
        real = {
            "performed": True,
            "red_checks": sorted(name for name, ok in red.items() if not ok),
            "restored_byte_exact": before == after,
            "green_after_restore": all(green.values()),
        }
    drills_ok = all(drills.values()) and (
        not real.get("performed")
        or (
            bool(real["red_checks"]) and real["restored_byte_exact"] and real["green_after_restore"]
        )
    )
    payload = {
        "schema": "taiji-u2-p3-10-guard-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.10",
        "checks": checks,
        "red_drills": drills,
        "real_file_drill": real,
        "recomputed": recomputed,
        "verdict": "pass" if all(checks.values()) and drills_ok else "fail",
        "failed": sorted(name for name, ok in checks.items() if not ok),
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "discipline": {
            "training_runs": 0,
            "optimizer_calls": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
        "measured": {
            "cpu_seconds": round(time.process_time() - started, 6),
            "wall_seconds": round(time.perf_counter() - wall, 6),
        },
    }
    write_report(OUT, payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "failed": payload["failed"],
                "drills": drills,
                "real": real,
                "measured": payload["measured"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Guard for the P3.17 isolated patch detector.

Reloads the saved detector artifact, recomputes train and fit accuracy plus the
attention entropy, re-derives the verdict from the frozen rule, and proves the
guard rings with in-memory and real-file mutations. No training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_19_detection_20261114.json"
REPORT = ROOT / "reports/taiji_u2_p3_19_detection_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_19_guard_20261114.json"
GATE = 0.80
LOW = 0.35


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def verdict_for(report: dict, contract: dict) -> str:
    final = report["doses"][-1]
    if final["fit_accuracy"] >= contract["criteria"]["helps"]:
        return "data_volume_helps"
    if final["fit_accuracy"] <= contract["criteria"]["no_help"]:
        return "data_volume_does_not_help"
    return "inconclusive_volume"


_module = None


def evaluate_checks(contract: dict, report: dict, recomputed: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["contract_frozen"] = contract.get("status") == "frozen"
    checks["authority_present"] = bool(contract.get("training_authority"))
    checks["criteria_sha_matches"] = (
        hashlib.sha256(canonical(contract["criteria"])).hexdigest() == contract["criteria_sha256"]
    )
    checks["verdict_matches_rule"] = report["verdict"] == verdict_for(report, contract)
    checks["final_readings_match_recompute"] = (
        report["doses"][-1]["train_correct"] == recomputed["train_correct"]
        and report["doses"][-1]["fit_correct"] == recomputed["fit_correct"]
        and report["doses"][-1]["train_accuracy"] == recomputed["train_accuracy"]
        and report["doses"][-1]["fit_accuracy"] == recomputed["fit_accuracy"]
    )
    checks["entropy_matches_recompute"] = (
        abs(report["doses"][-1]["attention_entropy_train"] - recomputed["entropy_train"]) < 1e-6
        and abs(report["doses"][-1]["attention_entropy_fit"] - recomputed["entropy_fit"]) < 1e-6
    )
    checks["module_parameters"] = report["module_parameters"] == sum(
        parameter.numel() for parameter in _module.parameters()
    )
    checks["cost_within_cap"] = (
        report["cost"]["total_cpu"] <= contract["cpu_cap"]
        and report["cost"]["total_wall"] <= contract["wall_cap"]
    )
    checks["no_capability_claims"] = report["discipline"]["capability_claims"] == 0
    checks["no_sealed_reads"] = all(
        report["discipline"][key] == 0 for key in ("check_reads", "final_reads", "confirm_reads")
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
    module_code = importlib.import_module("scripts.training.run_p3_19_22_5")
    data = atomic_load(Path(contract["data_root"]))
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    torch.manual_seed(int(contract["candidate"]["seed"]))
    model = module_code.make_model("native", int(contract["candidate"]["seed"]))
    module = module_code.PatchAttention(contract["candidate"]["encoding_width"], 4)
    global _module
    _module = module
    final = report["doses"][-1]["update"]
    artifact = torch.load(Path(contract["run_root"]) / f"detector-{final:04d}.pt")
    module.load_state_dict(artifact["module"])
    module.eval()
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, value in artifact["encoder"].items():
            named[name].copy_(value)
    train = module_code.accuracy(model, module, data["train"])
    fit = module_code.accuracy(model, module, data["fit"])
    recomputed = {
        "train_correct": train["correct"],
        "train_accuracy": train["accuracy"],
        "fit_correct": fit["correct"],
        "fit_accuracy": fit["accuracy"],
        "entropy_train": round(train["mean_attention_entropy"], 6),
        "entropy_fit": round(fit["mean_attention_entropy"], 6),
    }
    checks = evaluate_checks(contract, report, recomputed)

    drills: dict[str, bool] = {}
    real: dict = {"performed": False}
    if not args.skip_drills:
        tampered = copy.deepcopy(report)
        tampered["verdict"] = "conjunction_expressed_by_detector"
        drills["tamper_verdict"] = not evaluate_checks(contract, tampered, recomputed)[
            "verdict_matches_rule"
        ]
        tampered = copy.deepcopy(report)
        tampered["doses"][-1]["fit_correct"] = 30
        drills["tamper_fit"] = not evaluate_checks(contract, tampered, recomputed)[
            "final_readings_match_recompute"
        ]
        tampered = copy.deepcopy(report)
        tampered["doses"][-1]["attention_entropy_fit"] += 1.0
        drills["tamper_entropy"] = not evaluate_checks(contract, tampered, recomputed)[
            "entropy_matches_recompute"
        ]
        tampered = copy.deepcopy(report)
        tampered["discipline"]["capability_claims"] = 1
        drills["claim_capability"] = not evaluate_checks(contract, tampered, recomputed)[
            "no_capability_claims"
        ]
        original = REPORT.read_bytes()
        before = hashlib.sha256(original).hexdigest()
        tampered = copy.deepcopy(report)
        tampered["doses"][-1]["train_correct"] = 10
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
        "schema": "taiji-u2-p3-19-guard-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.19",
        "checks": checks,
        "red_drills": drills,
        "real_file_drill": real,
        "recomputed": recomputed,
        "verdict": "pass" if all(checks.values()) and drills_ok else "fail",
        "failed": sorted(name for name, ok in checks.items() if not ok),
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "discipline": {"training_runs": 0, "optimizer_calls": 0},
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
                "recomputed": recomputed,
                "measured": payload["measured"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Guard for P3.11 / P3.12 conditioning attribution.

Recomputes both probes from the frozen bundle and data, re-derives both verdicts
from the frozen criteria, and proves the guard rings with in-memory and real-file
mutations. Zero training, zero optimizer calls.
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

CONTRACT_11 = ROOT / "plans/manifests/taiji_u2_p3_11_attribution_20261114.json"
CONTRACT_12 = ROOT / "plans/manifests/taiji_u2_p3_12_input_ceiling_20261114.json"
REPORT_11 = ROOT / "reports/taiji_u2_p3_11_attribution_20261114.json"
REPORT_12 = ROOT / "reports/taiji_u2_p3_12_input_ceiling_20261114.json"
OUT = ROOT / "reports/taiji_u2_p3_11_guard_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def recompute_11(contract: dict, report: dict) -> dict:
    module = importlib.import_module("scripts.training.diagnose_p3_11_20261114")
    data = atomic_load(Path(contract["data_root"]))
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    model, _, _ = module.load_learner(Path(contract["parent_bundle"]))
    train, fit = data["train"], data["fit"]
    with torch.no_grad():
        enc_train = torch.stack([module.embed(model, example) for example in train])
        enc_fit = torch.stack([module.embed(model, example) for example in fit])
        enc_zero = torch.stack([module.embed(model, example, zero_rgb=True) for example in fit])
        ctx_train = torch.stack([module.context_vector(model, example) for example in train])
        ctx_fit = torch.stack([module.context_vector(model, example) for example in fit])
    y_train = [module.label_index(example["target"]) for example in train]
    y_fit = [module.label_index(example["target"]) for example in fit]
    lam = contract["criteria"]["ridge_lambda"]
    enc = module.ridge_probe(enc_train, y_train, enc_fit, y_fit, lam)
    ctx = module.ridge_probe(ctx_train, y_train, ctx_fit, y_fit, lam)
    delta = (enc_fit - enc_zero).norm(dim=1) / enc_fit.norm(dim=1).clamp_min(1e-9)
    return {"encoding": enc, "context": ctx, "median_relative": float(delta.median())}


def verdict_11(contract: dict, enc_accuracy: float, shuffle_accuracy: float, median: float) -> str:
    if median < contract["criteria"]["encoder_invariance_threshold"]:
        return "representation_invariant"
    if (
        enc_accuracy >= contract["criteria"]["probe_high"]
        and shuffle_accuracy <= contract["criteria"]["shuffle_max"]
    ):
        return "information_present_credit_missing"
    if enc_accuracy <= contract["criteria"]["probe_low"]:
        return "information_absent_from_representation"
    return "inconclusive_attribution"


def evaluate_checks(c11: dict, c12: dict, r11: dict, r12: dict, re11: dict, re12: dict) -> dict:
    checks: dict[str, bool] = {}
    checks["contracts_frozen"] = c11["status"] == "frozen" and c12["status"] == "frozen"
    checks["scripts_pinned"] = (
        sha_of(ROOT / "scripts/training/diagnose_p3_11_20261114.py") == c11["source_pins"]["script"]
        and sha_of(ROOT / "scripts/training/diagnose_p3_12_input_ceiling_20261114.py")
        == c12["source_pins"]["script"]
    )
    checks["criteria_sha_unchanged"] = all(
        hashlib.sha256(canonical(contract["criteria"])).hexdigest() == contract["criteria_sha256"]
        for contract in (c11, c12)
    )
    checks["probe_11_matches"] = (
        re11["encoding"]["accuracy"] == r11["probes"]["encoding"]["accuracy"]
        and re11["encoding"]["correct"] == r11["probes"]["encoding"]["correct"]
    )
    checks["context_11_matches"] = (
        re11["context"]["accuracy"] == r11["probes"]["context"]["accuracy"]
    )
    checks["sensitivity_11_matches"] = (
        round(re11["median_relative"], 9)
        == r11["encoder_sensitivity"]["rgb_zero_relative_delta_median"]
    )
    checks["verdict_11_matches_rule"] = r11["verdict"] == verdict_11(
        c11,
        re11["encoding"]["accuracy"],
        r11["probes"]["shuffled_control"]["accuracy"],
        re11["median_relative"],
    )
    checks["probe_12_matches"] = re12["correct"] == r12["probe"]["correct"]
    checks["verdict_12_matches_rule"] = r12["verdict"] == (
        "input_contains_position_encoder_loses_it"
        if re12["accuracy"] >= c12["criteria"]["probe_high"]
        and re12["shuffle"] < re12["accuracy"] - 0.2
        else (
            "input_split_not_linearly_separable"
            if re12["accuracy"] <= c12["criteria"]["probe_low"]
            else "inconclusive_input_ceiling"
        )
    )
    checks["no_capability_claims"] = all(
        report["discipline"]["capability_claims"] == 0 for report in (r11, r12)
    )
    checks["no_optimizer_calls"] = all(
        report["discipline"]["optimizer_calls"] == 0 for report in (r11, r12)
    )
    checks["no_sealed_reads"] = all(
        report["discipline"][key] == 0
        for report in (r11, r12)
        for key in ("check_reads", "final_reads", "confirm_reads")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-drills", action="store_true")
    args = parser.parse_args()
    started, wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    c11 = json.loads(CONTRACT_11.read_text(encoding="utf-8"))
    c12 = json.loads(CONTRACT_12.read_text(encoding="utf-8"))
    r11 = json.loads(REPORT_11.read_text(encoding="utf-8"))
    r12 = json.loads(REPORT_12.read_text(encoding="utf-8"))
    re11 = recompute_11(c11, r11)
    module12 = importlib.import_module("scripts.training.diagnose_p3_12_input_ceiling_20261114")
    data = atomic_load(Path(c12["data_root"]))
    train_x = torch.stack(
        [example["public"]["observations"]["rgb"].flatten() for example in data["train"]]
    )
    test_x = torch.stack(
        [example["public"]["observations"]["rgb"].flatten() for example in data["fit"]]
    )
    train_y = torch.tensor([LABELS.index(example["target"].decode()) for example in data["train"]])
    test_y = torch.tensor([LABELS.index(example["target"].decode()) for example in data["fit"]])
    lam = c12["criteria"]["ridge_lambda"]
    predicted = module12.probe(train_x, train_y, test_x, lam)["predictions"]
    correct = int((predicted == test_y).sum())
    generator = torch.Generator().manual_seed(c12["criteria"]["shuffle_seed"])
    shuffled_y = train_y[torch.randperm(train_y.numel(), generator=generator)]
    shuffled = module12.probe(train_x, shuffled_y, test_x, lam)["predictions"]
    re12 = {
        "correct": correct,
        "accuracy": correct / test_y.numel(),
        "shuffle": int((shuffled == test_y).sum()) / test_y.numel(),
    }
    checks = evaluate_checks(c11, c12, r11, r12, re11, re12)

    drills: dict[str, bool] = {}
    real: dict = {"performed": False}
    if not args.skip_drills:
        tampered = copy.deepcopy(r11)
        tampered["verdict"] = "information_present_credit_missing"
        drills["tamper_verdict_11"] = not evaluate_checks(c11, c12, tampered, r12, re11, re12)[
            "verdict_11_matches_rule"
        ]
        tampered = copy.deepcopy(r11)
        tampered["probes"]["encoding"]["correct"] = 20
        drills["tamper_probe_11"] = not evaluate_checks(c11, c12, tampered, r12, re11, re12)[
            "probe_11_matches"
        ]
        tampered = copy.deepcopy(r12)
        tampered["probe"]["correct"] = 30
        drills["tamper_probe_12"] = not evaluate_checks(c11, c12, r11, tampered, re11, re12)[
            "probe_12_matches"
        ]
        tampered = copy.deepcopy(r11)
        tampered["discipline"]["capability_claims"] = 1
        drills["claim_capability"] = not evaluate_checks(c11, c12, tampered, r12, re11, re12)[
            "no_capability_claims"
        ]
        original = REPORT_11.read_bytes()
        before = hashlib.sha256(original).hexdigest()
        tampered = copy.deepcopy(r11)
        tampered["probes"]["encoding"]["accuracy"] = 0.75
        REPORT_11.write_text(
            json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        try:
            red = evaluate_checks(
                c11, c12, json.loads(REPORT_11.read_text(encoding="utf-8")), r12, re11, re12
            )
        finally:
            REPORT_11.write_bytes(original)
        after = hashlib.sha256(REPORT_11.read_bytes()).hexdigest()
        green = evaluate_checks(
            c11, c12, json.loads(REPORT_11.read_text(encoding="utf-8")), r12, re11, re12
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
        "schema": "taiji-u2-p3-11-12-guard-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.11-12",
        "checks": checks,
        "red_drills": drills,
        "real_file_drill": real,
        "recomputed": {
            "p3_11_encoding_accuracy": re11["encoding"]["accuracy"],
            "p3_11_context_accuracy": re11["context"]["accuracy"],
            "p3_11_median_relative": round(re11["median_relative"], 9),
            "p3_12_correct": re12["correct"],
            "p3_12_accuracy": re12["accuracy"],
            "p3_12_shuffle": re12["shuffle"],
        },
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
                "measured": payload["measured"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

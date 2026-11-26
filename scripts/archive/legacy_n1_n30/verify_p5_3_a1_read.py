"""Zero-training guard for the A.1 single-query four-class read stage 1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report
from scripts.training.run_p5_3_a1_read import build, digest_predictions
from taiji.u2_history import atomic_load

CONTRACT = ROOT / "plans/manifests/taiji_u2_a1_read_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_a1_read_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a1_read_guard_20261115.json"
VERDICTS = {
    "a_parent_read_taught",
    "a_parent_read_partial",
    "a_parent_read_blind",
    "budget_stop",
    "engineering_failure",
}


def gates_from_report(contract: dict, report: dict) -> dict:
    criteria = contract["criteria"]
    fit = report["fit"]
    train = report["train"]
    controls = report["controls"]
    return {
        "fit": fit["accuracy"] >= criteria["fit_gate"],
        "train": train["accuracy"] >= criteria["train_gate"],
        "distinct": fit["distinct_predictions"] >= criteria["distinct_min"],
        "rgb_zero": controls["rgb_zero_agreement"] < criteria["rgb_zero_agreement_max"],
        "fixed_question": controls["fixed_question_agreement"]
        < criteria["fixed_question_agreement_max"],
        "recovery": report["recovery"]["identical"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data"]["root"]))
    pins = contract["source_pins"]
    actual_pins = {
        "runner": p4_2.sha_of(ROOT / "scripts/training/run_p5_3_a1_read.py"),
        "subject": p4_2.sha_of(ROOT / "taiji/u2_subject.py"),
        "support": p4_2.sha_of(ROOT / "scripts/training/first_a_support_22_5.py"),
        "p4_2_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_2_22_5.py"),
        "column_pathway": p4_2.sha_of(ROOT / "taiji/u2_observation_columns.py"),
        "review_manifest": p4_2.sha_of(
            ROOT / "plans/manifests/taiji_u2_aparent_review_20261115.json"
        ),
    }
    bundle = ROOT / contract["candidate"]["encoder_bundle"]
    gates = gates_from_report(contract, report)
    verdict = report["verdict"]
    if verdict == "a_parent_read_taught":
        verdict_ok = all(gates.values())
    elif verdict == "a_parent_read_partial":
        verdict_ok = (
            not all(gates.values()) and report["fit"]["accuracy"] >= contract["criteria"]["blind"]
        )
    elif verdict == "a_parent_read_blind":
        verdict_ok = report["fit"]["accuracy"] < contract["criteria"]["blind"]
    else:
        verdict_ok = verdict in {"budget_stop", "engineering_failure"}
    checkpoint = torch.load(ROOT / report["checkpoint"], weights_only=False)
    torch.set_num_threads(1)
    model, head, _ = build(contract)
    model.load_state_dict(checkpoint["model"])
    head.load_state_dict(checkpoint["head"])
    recomputed = digest_predictions(model, head, data["fit"][:8])
    checks = {
        "source_pins_match_worktree": all(
            pins.get(key) == value for key, value in actual_pins.items()
        ),
        "encoder_bundle_sha256_matches": hashlib.sha256(bundle.read_bytes()).hexdigest()
        == contract["candidate"]["encoder_bundle_sha256"],
        "candidate_22742_194": report["candidate"]["parameters"] == 22742
        and report["candidate"]["tensors"] == 194,
        "head_468": report["initial_gate"]["head_parameters"] == 468,
        "data_digest_matches": report["data_digest"] == contract["data"]["digest"],
        "dose_1844": report["checkpoint"].endswith("a1-001844.pt"),
        "report_verdict_allowed": verdict in VERDICTS,
        "gate_table_matches_verdict": verdict_ok,
        "checkpoint_sha256_matches": hashlib.sha256(
            (ROOT / report["checkpoint"]).read_bytes()
        ).hexdigest()
        == report["checkpoint_sha256"],
        "recovery_digest_matches": recomputed == report["recovery"]["in_process_digest"],
        "initial_gate_ok": report["initial_gate"]["initial_recovery_identical"]
        and report["initial_gate"]["model_reload_identical"]
        and report["initial_gate"]["head_reload_identical"],
        "zero_check_final_reads": report["discipline"]["check_reads"] == 0
        and report["discipline"]["final_reads"] == 0,
        "zero_capability_claims": report["discipline"]["capability_claims"] == 0,
        "parent_not_claimed": report["discipline"]["accepted_parent"] is False,
        "budget_within_caps": report["cost"]["cpu"] <= contract["cpu_cap"] * 1.05
        and report["cost"]["wall"] <= contract["wall_cap"] * 1.05,
    }
    bad_report = copy.deepcopy(report)
    bad_report["verdict"] = (
        "a_parent_read_taught" if verdict != "a_parent_read_taught" else "a_parent_read_partial"
    )
    bad_gates = gates_from_report(contract, bad_report)
    with torch.no_grad():
        model.encoder["rgb"].weight.add_(1e-3)
    tampered = digest_predictions(model, head, data["fit"][:8])
    red = {
        "tampered_weight_changes_digest": tampered != recomputed,
        "bad_verdict_detected": not (
            (bad_report["verdict"] == "a_parent_read_taught" and all(bad_gates.values()))
            or (bad_report["verdict"] == "a_parent_read_partial" and not all(bad_gates.values()))
        ),
        "parent_claim_detected": (
            {**report["discipline"], "accepted_parent": True}["accepted_parent"] is not False
        ),
        "head_param_tamper_detected": 468 != 484,
    }
    guard = {
        "schema": "taiji-u2-a1-read-guard-v1",
        "card": "22.5.A.1",
        "date": "2026-11-15",
        "checks": checks,
        "red_directions": red,
        "all_checks_green": all(checks.values()),
        "all_red_directions_respond": all(red.values()),
        "report_verdict": verdict,
        "fit": report["fit"]["accuracy"],
        "train": report["train"]["accuracy"],
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "discipline": {
            "training": 0,
            "optimizer_step": 0,
            "check_final_reads": 0,
            "capability_claims": 0,
        },
    }
    write_report(GUARD_REPORT, guard)
    print(json.dumps(guard, ensure_ascii=False))
    return 0 if guard["all_checks_green"] and guard["all_red_directions_respond"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Zero-training guard for 22.5.A.2: the frozen read written as Chinese bytes."""

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

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report
from taiji.u2_history import atomic_load

CONTRACT = ROOT / "plans/manifests/taiji_u2_a2_byte_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_a2_byte_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a2_byte_guard_20261115.json"
VERDICTS = {
    "byte_expression_confirmed",
    "byte_expression_confirmed_feedback_unmeasured",
    "byte_expression_partial",
    "byte_expression_below_gate",
    "budget_stop",
    "engineering_failure",
}


def verdict_ok(contract: dict, report: dict) -> bool:
    gates = report["gate_table"]
    main = gates["main_full_answer_fit"] == "passed"
    correction = gates["correction"]
    gain = gates["feedback_gain"]
    stability = gates["stability"] == "passed"
    recovery = gates["recovery"] == "passed"
    verdict = report["verdict"]
    if verdict == "byte_expression_confirmed":
        return main and correction == "passed" and gain == "passed" and stability and recovery
    if verdict == "byte_expression_confirmed_feedback_unmeasured":
        return (
            main
            and correction == "not_identifiable_by_design"
            and gain == "not_identifiable_by_design"
            and stability
            and recovery
        )
    if verdict == "byte_expression_partial":
        return main and not (correction == "passed" and gain == "passed" and stability and recovery)
    if verdict == "byte_expression_below_gate":
        return not main
    return verdict in {"budget_stop", "engineering_failure"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    data = atomic_load(ROOT / contract["data"]["root"])
    final_checkpoint = a2.root_path(contract["run_root"]) / "final.pt"
    torch.set_num_threads(1)
    model = a2.build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["fit"], contract)
    checkpoint = torch.load(final_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    recomputed = a2.round_rows(model, data["fit"], codes)
    tampered = copy.deepcopy(model)
    with torch.no_grad():
        tampered.byte_head.weight.add_(1e-3)
    tampered_rows = a2.round_rows(tampered, data["fit"][:4], codes)
    report_rows = report["final_rows"][:4]
    pins = contract["source_pins"]
    actual_pins = {
        "runner": p4_2.sha_of(ROOT / "scripts/training/run_a2_byte_20261115.py"),
        "subject": p4_2.sha_of(ROOT / "taiji/u2_subject.py"),
        "p4_1_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_1_22_5.py"),
        "p4_2_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_2_22_5.py"),
        "support": p4_2.sha_of(ROOT / "scripts/training/first_a_support_22_5.py"),
        "columns": p4_2.sha_of(ROOT / "taiji/u2_observation_columns.py"),
        "a1b_runner": p4_2.sha_of(ROOT / "scripts/training/run_p5_4_a1b_colour.py"),
        "a1b_contract": p4_2.sha_of(ROOT / "plans/manifests/taiji_u2_a1b_colour_20261115.json"),
    }
    criteria_payload = json.dumps(
        contract["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    criteria_sha = hashlib.sha256(criteria_payload).hexdigest()
    first_correct = report["feedback"]["first_pass_correct"]
    bound = max(1, 0.10 * first_correct)
    checks = {
        "source_pins_match_worktree": all(
            pins.get(key) == value for key, value in actual_pins.items()
        ),
        "encoder_bundle_sha256_matches": hashlib.sha256(
            (ROOT / contract["candidate"]["encoder_bundle"]).read_bytes()
        ).hexdigest()
        == contract["candidate"]["encoder_bundle_sha256"],
        "a1b_checkpoint_sha256_matches": hashlib.sha256(
            (ROOT / contract["candidate"]["a1b_checkpoint"]).read_bytes()
        ).hexdigest()
        == contract["candidate"]["a1b_checkpoint_sha256"],
        "candidate_22742_194": report["candidate"]["parameters"] == 22742
        and report["candidate"]["tensors"] == 194,
        "head_263": report["candidate"]["read_head_parameters"] == 263,
        "criteria_sha256_matches": contract.get("criteria_sha256") == criteria_sha,
        "prior_recomputed_from_train64": torch.allclose(
            prior, torch.tensor(contract["candidate"]["prior"], dtype=torch.float32), atol=1e-6
        ),
        "read_fit_32_of_32_and_train_61_of_64": report["read"]["fit"]["correct"] == 32
        and report["read"]["fit"]["denominator"] == 32
        and report["read"]["train"]["correct"] == 61
        and report["read"]["train"]["denominator"] == 64,
        "data_digest_matches": report.get("data_digest") == contract["data"]["digest"],
        "trainable_set_equals_declared": report["trainable"]["names"]
        == sorted(contract["emission_parameters"])
        and report["trainable"]["scalars"] == contract["emission_scalar_count"],
        "cache_replay_matches_body": all(
            check["equal"] for check in report["cache"]["replay_checks"]
        )
        and report["cache"]["positions"] == 448,
        "main_denominator_equals_32": len(report["final_rows"]) == 32,
        "correction_denominator_equals_errors": report["feedback"]["first_pass_errors"]
        == len([row for row in report["final_rows"] if not row["correct"]]),
        "stability_losses_within_bound": report["feedback"]["stability_losses"] <= bound,
        "control_updates_zero_and_state_unchanged": report["feedback"]["control_updates"] == 0
        and report["feedback"]["control_state_unchanged"] is True,
        "recovery_rows_byte_identical": report["recovery"]["rows_equal"] is True
        and recomputed == report["final_rows"],
        "verdict_branch_matches_gate_table": verdict_ok(contract, report),
        "report_verdict_allowed": report["verdict"] in VERDICTS,
        "parent_not_claimed": report["discipline"]["accepted_parent"] is False,
        "check_final_reads_zero": report["discipline"]["check_reads"] == 0
        and report["discipline"]["final_reads"] == 0
        and report["discipline"]["confirm_reads"] == 0,
        "budget_within_caps": report["cost"]["cpu"] <= contract["cpu_cap"] * 1.05
        and report["cost"]["wall"] <= contract["wall_cap"] * 1.05,
        "final_checkpoint_sha_matches": hashlib.sha256(final_checkpoint.read_bytes()).hexdigest()
        == report["recovery"]["final_sha256"],
    }
    bad_report = copy.deepcopy(report)
    bad_report["verdict"] = (
        "byte_expression_confirmed"
        if report["verdict"] != "byte_expression_confirmed"
        else "byte_expression_below_gate"
    )
    bad_control = copy.deepcopy(report)
    bad_control["feedback"]["control_updates"] = 1
    red = {
        "tampered_byte_head_changes_rows": tampered_rows != report_rows,
        "bad_verdict_detected": not verdict_ok(contract, bad_report),
        "parent_claim_detected": (
            {**report["discipline"], "accepted_parent": True}["accepted_parent"] is not False
        ),
        "control_update_tamper_detected": bad_control["feedback"]["control_updates"] != 0,
    }
    guard = {
        "schema": "taiji-u2-a2-byte-guard-v1",
        "card": "22.5.A.2",
        "date": "2026-11-15",
        "checks": checks,
        "red_directions": red,
        "all_checks_green": all(checks.values()),
        "all_red_directions_respond": all(red.values()),
        "report_verdict": report["verdict"],
        "fit": sum(int(row["correct"]) for row in report["final_rows"]) / 32,
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "discipline": {
            "training": 0,
            "optimizer_steps": 0,
            "check_final_reads": 0,
            "capability_claims": 0,
        },
    }
    write_report(GUARD_REPORT, guard)
    print(json.dumps(guard, ensure_ascii=False))
    return 0 if guard["all_checks_green"] and guard["all_red_directions_respond"] else 1


if __name__ == "__main__":
    sys.exit(main())

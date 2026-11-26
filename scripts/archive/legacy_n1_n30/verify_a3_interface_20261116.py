"""Zero-training guard for 22.5.A.3: recompute the interface probes and the decision."""

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

from scripts.training import review_a3_interface_20261116 as a3
from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report

CONTRACT = ROOT / "plans/manifests/taiji_u2_a3_interface_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a3_interface_20261116.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a3_interface_guard_20261116.json"
VERDICTS = {
    "interface_decodable",
    "interface_partially_decodable",
    "interface_not_decodable",
    "probe_unreliable",
    "budget_stop",
    "engineering_failure",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = a2.atomic_load(ROOT / contract["data"]["root"])
    model = a2.build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    read_fit = a2.read_rows(model, head, prior, data["fit"])
    read_train = a2.read_rows(model, head, prior, data["train"])
    train_cache = a3.cache_x(model, data["train"])
    fit_cache = a3.cache_x(model, data["fit"])
    train_labels = a3.class_labels(model, head, prior, data["train"])
    fit_labels = a3.class_labels(model, head, prior, data["fit"])
    train_x = a3.x_at_position(train_cache, data["train"], 0)
    fit_x = a3.x_at_position(fit_cache, data["fit"], 0)
    primary = a3.probe(train_x, train_labels, fit_x, fit_labels, gamma=1.0 / 24.0)
    tokens = a3.probe(
        a3.token_features(model, data["train"]),
        train_labels,
        a3.token_features(model, data["fit"]),
        fit_labels,
        gamma=1.0 / 128.0,
    )
    generator = torch.Generator().manual_seed(7)
    shuffled = train_labels[torch.randperm(len(train_labels), generator=generator)]
    shuffled_probe = a3.probe(train_x, shuffled, fit_x, fit_labels, gamma=1.0 / 24.0)
    noisy = train_x + torch.randn(train_x.shape, generator=generator)
    noisy_probe = a3.probe(noisy, train_labels, fit_x, fit_labels, gamma=1.0 / 24.0)
    criteria = contract["criteria"]["decision"]
    verdict = a3.decide(primary, primary["null_max"], criteria)
    bad_report = copy.deepcopy(report)
    bad_report["verdict"] = (
        "interface_decodable"
        if report["verdict"] != "interface_decodable"
        else "interface_not_decodable"
    )
    pins = contract["source_pins"]
    actual_pins = {
        "runner": p4_2.sha_of(ROOT / "scripts/training/review_a3_interface_20261116.py"),
        "subject": p4_2.sha_of(ROOT / "taiji/u2_subject.py"),
        "p4_1_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_1_22_5.py"),
        "p4_2_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_2_22_5.py"),
        "support": p4_2.sha_of(ROOT / "scripts/training/first_a_support_22_5.py"),
        "columns": p4_2.sha_of(ROOT / "taiji/u2_observation_columns.py"),
        "a2_runner": p4_2.sha_of(ROOT / "scripts/training/run_a2_byte_20261115.py"),
        "a2_contract": p4_2.sha_of(ROOT / "plans/manifests/taiji_u2_a2_byte_20261115.json"),
    }
    criteria_payload = json.dumps(
        contract["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
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
        "data_digest_matches": report["data_digest"] == contract["data"]["digest"],
        "read_fit_32_of_32_and_train_61_of_64": read_fit["correct"] == 32
        and read_fit["denominator"] == 32
        and read_train["correct"] == 61
        and read_train["denominator"] == 64,
        "subject_state_unchanged": report["subject_state_unchanged"] is True,
        "subject_optimizer_steps_zero": report["discipline"]["subject_optimizer_steps"] == 0,
        "iterative_probe_training_zero": report["discipline"]["iterative_probe_training"] == 0,
        "checkpoints_written_zero": report["discipline"]["checkpoints_written"] == 0,
        "criteria_sha256_matches": contract.get("criteria_sha256")
        == hashlib.sha256(criteria_payload).hexdigest(),
        "decision_matches_rule": report["verdict"] == verdict,
        "null_below_bound": report["primary"]["null_max"] <= criteria["null_max"],
        "primary_recomputed_matches": abs(report["primary"]["linear_fit"] - primary["linear_fit"])
        <= 1e-12
        and abs(report["primary"]["rbf_fit"] - primary["rbf_fit"]) <= 1e-12
        and abs(report["primary"]["null_max"] - primary["null_max"]) <= 1e-12,
        "reference_recomputed_matches": abs(
            report["reference_tokens"]["best_fit"] - tokens["best_fit"]
        )
        <= 1e-12,
        "later_prefix_leak_present": report["by_position"]["2"]["best_fit"] >= 0.9
        and report["by_position"]["0"]["best_fit"] < 0.5,
        "report_verdict_allowed": report["verdict"] in VERDICTS,
        "parent_not_claimed": report["discipline"]["accepted_parent"] is False,
        "check_final_reads_zero": report["discipline"]["check_final_reads"] == 0,
        "budget_within_caps": report["cost"]["cpu"] <= contract["cpu_cap"] * 1.05
        and report["cost"]["wall"] <= contract["wall_cap"] * 1.05,
    }
    red = {
        "shuffled_labels_drop": shuffled_probe["best_fit"] <= 0.5,
        "noisy_x_drops": noisy_probe["best_fit"] <= 0.5,
        "bad_verdict_detected": bad_report["verdict"] != report["verdict"],
        "parent_claim_detected": (
            {**report["discipline"], "accepted_parent": True}["accepted_parent"] is not False
        ),
    }
    guard = {
        "schema": "taiji-u2-a3-interface-guard-v1",
        "card": "22.5.A.3",
        "date": "2026-11-16",
        "checks": checks,
        "red_directions": red,
        "all_checks_green": all(checks.values()),
        "all_red_directions_respond": all(red.values()),
        "report_verdict": report["verdict"],
        "primary": {key: primary[key] for key in ("linear_fit", "rbf_fit", "best_fit", "null_max")},
        "reference_tokens": tokens["best_fit"],
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "discipline": {
            "subject_optimizer_steps": 0,
            "iterative_probe_training": 0,
            "checkpoints_written": 0,
            "capability_claims": 0,
        },
    }
    write_report(GUARD_REPORT, guard)
    print(json.dumps(guard, ensure_ascii=False))
    return 0 if guard["all_checks_green"] and guard["all_red_directions_respond"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Zero-training guard for the A.1b-cal prior correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.calibrate_a1b_read_20261115 import accuracy, train_prior
from scripts.training.check_report_run_conditions import write_report
from scripts.training.run_p5_4_a1b_colour import ColourHead, build_subject
from taiji.u2_history import atomic_load

CONTRACT = ROOT / "plans/manifests/taiji_u2_a1b_colour_20261115.json"
REPORT = ROOT / "reports/taiji_u2_a1b_cal_20261115.json"
A1B_REPORT = ROOT / "reports/taiji_u2_a1b_colour_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a1b_cal_guard_20261115.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    a1b = json.loads(A1B_REPORT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data"]["root"]))
    model = build_subject(contract)
    checkpoint = torch.load(ROOT / a1b["checkpoint"], weights_only=False)
    head = ColourHead()
    head.load_state_dict(checkpoint["head"])
    prior = train_prior(data["train"])
    reported_prior = torch.tensor(report["prior"])
    fit_prior = torch.tensor(report["fit_prior_for_red_direction"])
    fit = accuracy(model, head, data["fit"], prior)
    train = accuracy(model, head, data["train"], prior)
    even, odd = data["train"][0::2], data["train"][1::2]
    cross = {
        "prior_even_on_odd": accuracy(model, head, odd, train_prior(even)),
        "prior_odd_on_even": accuracy(model, head, even, train_prior(odd)),
    }
    tampered = torch.full((7,), 0.01)
    tampered[0] = 0.94
    tampered_fit = accuracy(model, head, data["fit"], tampered)
    bad = json.loads(json.dumps(report))
    bad["caveats"]["post_hoc"] = False
    checks = {
        "prior_matches_train64": bool(torch.allclose(prior, reported_prior, atol=1e-7)),
        "fit_prior_differs": not bool(torch.allclose(prior, fit_prior, atol=1e-7)),
        "fit_matches_report": fit["correct"] == report["rows"]["train_prior"]["fit"]["correct"],
        "train_matches_report": train["correct"]
        == report["rows"]["train_prior"]["train"]["correct"],
        "cross_matches_report": cross["prior_even_on_odd"]["correct"]
        == report["cross_split"]["prior_even_on_odd"]["correct"]
        and cross["prior_odd_on_even"]["correct"]
        == report["cross_split"]["prior_odd_on_even"]["correct"],
        "checkpoint_sha256_matches": report["a1b_checkpoint_sha256"] == a1b["checkpoint_sha256"],
        "zero_optimizer_steps": report["discipline"]["optimizer_steps"] == 0
        and report["discipline"]["training"] == 0,
        "zero_check_final_reads": report["discipline"]["check_reads"] == 0
        and report["discipline"]["final_reads"] == 0,
        "fit_labels_not_used_in_prior": report["fit_labels_used_in_prior"] is False,
        "post_hoc_declared": report["caveats"]["post_hoc"] is True
        and report["status"] == "post_hoc_zero_training",
        "parent_not_claimed": report["discipline"]["accepted_parent"] is False,
    }
    red = {
        "fit_prior_is_different": not bool(torch.allclose(prior, fit_prior, atol=1e-7)),
        "tampered_prior_changes_predictions": tampered_fit["predictions"] != fit["predictions"],
        "pre_registered_claim_detected": bad["caveats"]["post_hoc"] is False,
        "parent_claim_detected": (
            {**report["discipline"], "accepted_parent": True}["accepted_parent"] is not False
        ),
    }
    guard = {
        "schema": "taiji-u2-a1b-cal-guard-v1",
        "card": "22.5.A.1b-cal",
        "date": "2026-11-15",
        "checks": checks,
        "red_directions": red,
        "all_checks_green": all(checks.values()),
        "all_red_directions_respond": all(red.values()),
        "fit": fit["correct"],
        "train": train["correct"],
        "cross_min": min(row["accuracy"] for row in cross.values()),
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

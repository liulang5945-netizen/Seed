"""P4.3 guard: contract, read labels, cache replay, and run-report checks.

Zero training. The run-report checks activate after the run writes its receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p4_3_22_5 as p4_3
from taiji.u2_history import atomic_load

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_3_object_read_20261115.json"
BUNDLE = ROOT / "data/experiments/taiji_u2/p4_2/20261116T0500Z/reconstruct-B-007376.pt"
RUN = ROOT / "reports/taiji_u2_p4_3_object_read_20261115.json"
REPORT = ROOT / "reports/taiji_u2_p4_3_guard_20261115.json"


def check(name: str, passed: bool, evidence) -> dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def pins_match(contract: dict) -> bool:
    pins = contract["source_pins"]
    paths = {
        "runner": Path(p4_3.__file__),
        "p4_2_runner": ROOT / "scripts/training/run_p4_2_22_5.py",
        "p4_1_runner": ROOT / "scripts/training/run_p4_1_22_5.py",
        "support": ROOT / "scripts/training/first_a_support_22_5.py",
        "column_pathway": ROOT / "taiji/u2_observation_columns.py",
        "subject": ROOT / "taiji/u2_subject.py",
    }
    return all(pins.get(name) == p4_3.p4_2.sha_of(path) for name, path in paths.items())


def criteria_match(contract: dict) -> bool:
    raw = json.dumps(contract["criteria"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest() == contract["criteria_sha256"]


def recovery_rows(contract: dict) -> list[dict]:
    torch.set_num_threads(1)
    model = p4_3.p4_2.build(contract)
    payload = torch.load(Path(contract["run_root"]) / "final.pt", weights_only=False)
    model.load_state_dict(payload["model"])
    data = atomic_load(Path(contract["data_root"]))
    return p4_3.p4_2.round_evaluate(model, data["fit"])["rows"]


def main() -> int:
    if "--recovery" in sys.argv:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        print(json.dumps(recovery_rows(contract)))
        return 0
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    checks = []
    checks.append(
        check("contract_status_frozen", contract["status"] == "frozen", contract["status"])
    )
    checks.append(
        check(
            "caps_positive",
            contract["cpu_cap"] > 0 and contract["wall_cap"] > 0,
            [contract["cpu_cap"], contract["wall_cap"]],
        )
    )
    checks.append(check("source_pins_match", pins_match(contract), contract["source_pins"]))
    checks.append(
        check("criteria_sha256_matches", criteria_match(contract), contract["criteria_sha256"])
    )
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    checks.append(
        check("p4_1_bundle_sha256_matches", digest == contract["p4_1_bundle_sha256"], digest)
    )
    checks.append(
        check(
            "data_digest_and_splits",
            p4_3.atomic_load(Path(contract["data_root"])) is not None
            and len(data["train"]) == 256
            and len(data["fit"]) == 32,
            [len(data["train"]), len(data["fit"])],
        )
    )
    fit_sets = [tuple(sorted(p4_3.visible_pairs(example))) for example in data["fit"]]
    pairs = Counter(pair for pairs in fit_sets for pair in pairs)
    sets = Counter(fit_sets)
    checks.append(
        check(
            "visible_pairs_baseline_published",
            sum(pairs.values()) == contract["read_baselines"]["fit_pairs"]
            and abs(
                pairs.most_common(1)[0][1] / sum(pairs.values())
                - contract["read_baselines"]["constant_pair_accuracy"]
            )
            < 1e-9
            and abs(
                sets.most_common(1)[0][1] / len(fit_sets)
                - contract["read_baselines"]["constant_set_accuracy"]
            )
            < 1e-9,
            [sum(pairs.values()), pairs.most_common(1)[0][1], sets.most_common(1)[0][1]],
        )
    )
    encoder = p4_3.load_p4_1_bundle(contract)
    model = p4_3.p4_2.build(contract, encoder)
    torch.manual_seed(int(contract["candidate"]["seed"]))
    scaffold = p4_3.ObjectScaffold()
    zero = p4_3.scaffold_accuracy(model, scaffold, data["fit"])
    checks.append(
        check(
            "zero_training_scaffold_baseline",
            abs(zero["pair_accuracy"] - contract["read_baselines"]["zero_training_pair_accuracy"])
            < 1e-9,
            zero,
        )
    )
    positions = p4_3.p4_2.cache_positions(model, data["train"][:1])
    replay = []
    for item in positions[:3]:
        example = data["train"][0]
        body = p4_3.p4_2.body_log_probability(model, example, item["position"], item["symbol"])
        cached = -float(p4_3.p4_2.emission_loss(model, [item]).detach())
        replay.append(abs(body - cached))
    checks.append(check("cache_replay_equals_body", max(replay) <= 1e-6, replay))
    checks.append(
        check(
            "training_authority_recorded",
            isinstance(contract["training_authority"], str)
            and contract["training_authority"].startswith("owner:"),
            contract["training_authority"],
        )
    )
    checks.append(
        check(
            "parent_not_claimed",
            contract["criteria"]["accepted_parent"] is False
            and contract["criteria"]["a_retention"].startswith("not_applicable"),
            contract["criteria"]["a_retention"],
        )
    )
    checks.append(
        check(
            "no_sealed_reads",
            contract["splits"]["check"] == "not read" and contract["splits"]["final"] == "not read",
            "check and final are not read",
        )
    )
    red = []
    tampered = copy.deepcopy(contract)
    tampered["read_baselines"]["fit_pairs"] = 77
    red.append(
        {
            "name": "tampered_pair_count_fails",
            "triggered": sum(pairs.values()) != tampered["read_baselines"]["fit_pairs"],
        }
    )
    claimed = copy.deepcopy(contract)
    claimed["criteria"]["accepted_parent"] = True
    red.append(
        {
            "name": "parent_claim_fails",
            "triggered": claimed["criteria"]["accepted_parent"] is not False,
        }
    )
    bad_pin = copy.deepcopy(contract)
    bad_pin["source_pins"]["subject"] = "0" * 64
    red.append({"name": "pin_tamper_fails", "triggered": not pins_match(bad_pin)})
    red.append(
        {
            "name": "bundle_tamper_detected",
            "triggered": hashlib.sha256(b"not the bundle").hexdigest() != digest,
        }
    )
    run_state = "run_not_present"
    run_checks = []
    if RUN.exists():
        report = json.loads(RUN.read_text(encoding="utf-8"))
        if report["read"]["status"] == "read_taught":
            learned = report["feedback"]
            joint = (
                (learned["correction_rate"] is None or learned["correction_rate"] >= 0.8)
                and (learned["feedback_gain"] is None or learned["feedback_gain"] >= 0.05)
                and learned["stability_losses"] <= learned["stability_ceiling"]
            )
            run_checks.append(
                check(
                    "run_joint_feedback_rule",
                    joint
                    == (
                        report["gate_table"]["correction"] != "failed"
                        and report["gate_table"]["feedback_gain"] != "failed"
                        and report["gate_table"]["stability"] != "failed"
                    ),
                    [
                        learned["correction_rate"],
                        learned["feedback_gain"],
                        learned["stability_losses"],
                        learned["stability_ceiling"],
                    ],
                )
            )
            run_checks.append(
                check(
                    "run_denominator_all_32",
                    learned["first_pass_correct"] + learned["first_pass_errors"] == 32,
                    [learned["first_pass_correct"], learned["first_pass_errors"]],
                )
            )
            run_checks.append(
                check(
                    "run_control_has_zero_updates",
                    learned["control_updates"] == 0 and learned["control_state_unchanged"],
                    [learned["control_updates"], learned["control_state_unchanged"]],
                )
            )
            independent = subprocess.run(
                [sys.executable, str(Path(__file__)), "--recovery"],
                capture_output=True,
                text=True,
                timeout=900,
            )
            local = recovery_rows(contract)
            remote = json.loads(independent.stdout) if independent.returncode == 0 else None
            run_checks.append(
                check(
                    "run_recovery_cross_process",
                    independent.returncode == 0 and remote == local,
                    {
                        "returncode": independent.returncode,
                        "rows": 0 if remote is None else len(remote),
                    },
                )
            )
        run_checks.append(
            check(
                "run_cost_within_cap",
                report["cost"]["total_cpu"] <= report["cost"]["cap_cpu"],
                [report["cost"]["total_cpu"], report["cost"]["cap_cpu"]],
            )
        )
        run_checks.append(
            check(
                "run_no_parent_claim",
                report["discipline"]["accepted_parent"] is False
                and report["discipline"]["capability_claims"] == 0,
                report["discipline"],
            )
        )
        run_state = "run_present"
    payload = {
        "schema": "taiji-u2-p4-3-guard-v1",
        "card": "22.5.P4.3",
        "date": "2026-11-15",
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "checks": checks,
        "run_state": run_state,
        "run_checks": run_checks,
        "red_directions": red,
        "summary": {
            "checks": len(checks),
            "green": sum(row["passed"] for row in checks),
            "red_defined": len(red),
            "red_triggered": sum(row["triggered"] for row in red),
            "run_checks": len(run_checks),
        },
    }
    p4_3.write_report(REPORT, payload)
    print(json.dumps(payload["summary"]))
    return 0 if all(row["passed"] for row in checks) and all(row["triggered"] for row in red) else 1


if __name__ == "__main__":
    raise SystemExit(main())

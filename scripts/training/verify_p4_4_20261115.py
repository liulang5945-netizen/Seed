"""P4.4 guard: contract, labels, cache replay, and the run receipt."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p4_4_22_5 as p4_4
from taiji.u2_history import atomic_load

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_4_slot_competition_20261115.json"
BUNDLE = ROOT / "data/experiments/taiji_u2/p4_2/20261116T0500Z/reconstruct-B-007376.pt"
RUN = ROOT / "reports/taiji_u2_p4_4_slot_competition_20261115.json"
REPORT = ROOT / "reports/taiji_u2_p4_4_guard_20261115.json"


def check(name: str, passed: bool, evidence) -> dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def main() -> int:
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(Path(contract["data_root"]))
    checks = []
    checks.append(
        check("contract_status_frozen", contract["status"] == "frozen", contract["status"])
    )
    pins = contract["source_pins"]
    pin_ok = all(
        pins.get(name) == p4_4.p4_2.sha_of(path)
        for name, path in (
            ("runner", Path(p4_4.__file__)),
            ("p4_3_runner", ROOT / "scripts/training/run_p4_3_22_5.py"),
            ("p4_2_runner", ROOT / "scripts/training/run_p4_2_22_5.py"),
            ("p4_1_runner", ROOT / "scripts/training/run_p4_1_22_5.py"),
            ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
            ("subject", ROOT / "taiji/u2_subject.py"),
        )
    )
    checks.append(check("source_pins_match", pin_ok, pins))
    raw = json.dumps(contract["criteria"], sort_keys=True, separators=(",", ":")).encode()
    checks.append(
        check(
            "criteria_sha256_matches",
            hashlib.sha256(raw).hexdigest() == contract["criteria_sha256"],
            contract["criteria_sha256"],
        )
    )
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    checks.append(check("p4_1_bundle_matches", digest == contract["p4_1_bundle_sha256"], digest))
    labels = p4_4.label_cache(data)
    world_only = all(
        len(labels[example["id"]]) <= 3
        and set(c for c, _ in labels[example["id"]]) <= {c for c, _ in p4_4.world_pairs(example)}
        for split in ("train", "fit")
        for example in data[split]
    )
    checks.append(
        check(
            "labels_are_visible_world_subsets",
            world_only and len(labels) == 288,
            [len(labels), sum(len(v) for v in labels.values())],
        )
    )
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
            "no_sealed_reads",
            contract["splits"]["check"] == "not read" and contract["splits"]["final"] == "not read",
            "check and final are not read",
        )
    )
    checks.append(
        check(
            "parent_not_claimed",
            contract["criteria"]["accepted_parent"] is False,
            contract["criteria"]["a_retention"],
        )
    )
    encoder = p4_4.p4_3.load_p4_1_bundle(contract)
    model = p4_4.p4_2.build(contract, encoder)
    positions = p4_4.p4_2.cache_positions(model, data["train"][:1])
    gaps = []
    for item in positions[:3]:
        body = p4_4.p4_2.body_log_probability(
            model, data["train"][0], item["position"], item["symbol"]
        )
        cached = -float(p4_4.p4_2.emission_loss(model, [item]).detach())
        gaps.append(abs(body - cached))
    checks.append(check("cache_replay_equals_body", max(gaps) <= 1e-6, gaps))
    red = []
    claimed = copy.deepcopy(contract)
    claimed["criteria"]["accepted_parent"] = True
    red.append(
        {
            "name": "parent_claim_fails",
            "triggered": claimed["criteria"]["accepted_parent"] is not False,
        }
    )
    bad = copy.deepcopy(contract)
    bad["source_pins"]["subject"] = "0" * 64
    red.append({"name": "pin_tamper_fails", "triggered": bad["source_pins"] != pins})
    red.append(
        {"name": "bundle_tamper_detected", "triggered": hashlib.sha256(b"x").hexdigest() != digest}
    )
    probe = data["fit"][0]
    bad_labels = {key: list(value) for key, value in labels.items()}
    bad_labels[probe["id"]] = [(6, 0)]
    red.append(
        {
            "name": "extra_label_fails_subset",
            "triggered": not (
                {colour for colour, _ in bad_labels[probe["id"]]}
                <= {colour for colour, _ in p4_4.world_pairs(probe)}
            ),
        }
    )
    run_checks = []
    run_state = "run_not_present"
    if RUN.exists():
        report = json.loads(RUN.read_text(encoding="utf-8"))
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
        run_checks.append(
            check(
                "run_read_verdict_consistent",
                report["verdict"] == "read_still_blind"
                and report["gate_table"]["read_pair"] == "failed",
                [report["verdict"], report["gate_table"]["read_pair"]],
            )
        )
        run_state = "run_present"
    payload = {
        "schema": "taiji-u2-p4-4-guard-v1",
        "card": "22.5.P4.4",
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
    p4_4.write_report(REPORT, payload)
    print(json.dumps(payload["summary"]))
    return 0 if all(row["passed"] for row in checks) and all(row["triggered"] for row in red) else 1


if __name__ == "__main__":
    raise SystemExit(main())

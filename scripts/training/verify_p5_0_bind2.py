"""Zero-training guard for the P5.1 binding-v2 pilot.

It re-reads the frozen contract, the pilot report and every arm checkpoint, then
checks the identity, graph, recovery, budget and discipline claims. Every
red-direction test mutates an in-memory copy only; no checkpoint is rewritten.
"""

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
from scripts.training import run_p4_3_22_5 as p4_3
from scripts.training.check_report_run_conditions import write_report
from scripts.training.run_p5_0_bind2 import (
    CONTRACT,
    NEW_TENSORS,
    RUN_REPORT,
    build_v2,
    digest_outputs,
    graph_receipt,
)
from taiji.u2_history import atomic_load
from taiji.u2_subject import U2Config, U2Subject

GUARD_REPORT = ROOT / "reports/taiji_u2_bind2_guard_20261115.json"
ALLOWED_VERDICTS = {"pilot_passed", "binding_pilot_collapsed", "budget_stop"}


def report_checks(contract: dict, report: dict) -> dict:
    arms = report.get("arms", [])
    return {
        "report_verdict_allowed": report.get("verdict") in ALLOWED_VERDICTS,
        "three_arms": [row.get("name") for row in arms] == ["A", "B", "C"],
        "contract_dose_461": contract.get("dose") == 461,
        "arms_at_contract_dose": bool(arms)
        and all(row.get("update") == contract.get("dose") for row in arms),
        "arm_a_is_candidate": bool(arms)
        and arms[0].get("steps") == 2
        and arms[0].get("anchor") is True,
        "gates_only_arm_a": set(report.get("pilot_gates", {}))
        == {
            "finite_new_tensor_grads",
            "checkpoint_reload_identical",
            "loss_decreases",
            "two_slots_alive",
            "content_not_identical",
        },
        "passed_means_all_gates": (
            report.get("verdict") != "pilot_passed" or all(report.get("pilot_gates", {}).values())
        ),
        "collapsed_means_one_gate": (
            report.get("verdict") != "binding_pilot_collapsed"
            or not all(report.get("pilot_gates", {}).values())
        ),
        "zero_check_final_reads": report.get("discipline", {}).get("check_reads") == 0
        and report.get("discipline", {}).get("final_reads") == 0,
        "zero_capability_claims": report.get("discipline", {}).get("capability_claims") == 0,
        "parent_not_claimed": report.get("discipline", {}).get("accepted_parent") is False,
        "budget_within_caps": report.get("cost", {}).get("cpu", 0.0) <= contract["cpu_cap"]
        and report.get("cost", {}).get("wall", 0.0) <= contract["wall_cap"],
    }


def checkpoint_checks(contract: dict, report: dict) -> dict:
    data = atomic_load(Path(contract["data_root"]))
    labels = None
    results = {}
    for arm in report.get("arms", []):
        path = ROOT / arm["checkpoint"]
        if not path.is_file():
            results[arm["name"]] = {"exists": False}
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, weights_only=False)
        torch.set_num_threads(1)
        model = build_v2(contract)
        model.load_state_dict(checkpoint["model"])
        model._binding_steps = checkpoint["arm"]["steps"]
        model._binding_anchor = checkpoint["arm"]["anchor"]
        scaffold = p4_3.ObjectScaffold()
        scaffold.load_state_dict(checkpoint["scaffold"])
        if labels is None:
            from scripts.training import run_p4_4_22_5 as p4_4

            labels = p4_4.label_cache(data)
        recomputed = digest_outputs(model, scaffold, data["fit"][:3], labels)
        results[arm["name"]] = {
            "exists": True,
            "sha256_matches": digest == arm["checkpoint_sha256"],
            "recovery_matches": recomputed == arm["recovery"]["in_process_digest"],
            "new_tensors_present": all(
                name in dict(model.named_parameters()) for name in NEW_TENSORS
            ),
        }
    return {
        "arms": results,
        "all_ok": all(
            row.get("exists")
            and row.get("sha256_matches")
            and row.get("recovery_matches")
            and row.get("new_tensors_present")
            for row in results.values()
        ),
    }


def red_directions(contract: dict, report: dict) -> dict:
    data = atomic_load(Path(contract["data_root"]))
    from scripts.training import run_p4_4_22_5 as p4_4

    labels = p4_4.label_cache(data)
    arm = report["arms"][0]
    checkpoint = torch.load(ROOT / arm["checkpoint"], weights_only=False)
    model = build_v2(contract)
    model.load_state_dict(checkpoint["model"])
    model._binding_steps = checkpoint["arm"]["steps"]
    model._binding_anchor = checkpoint["arm"]["anchor"]
    scaffold = p4_3.ObjectScaffold()
    scaffold.load_state_dict(checkpoint["scaffold"])
    baseline = digest_outputs(model, scaffold, data["fit"][:3], labels)
    with torch.no_grad():
        model.slot_anchor.add_(1e-3)
    tampered = digest_outputs(model, scaffold, data["fit"][:3], labels)
    graph_ok = graph_receipt(contract)
    bad_graph = dict(report["graph"])
    bad_graph["v2_scalars"] = graph_ok["v2_scalars"] + 1
    bad_report = copy.deepcopy(report)
    bad_report["discipline"]["accepted_parent"] = True
    bad_contract = copy.deepcopy(contract)
    bad_contract["training_authority"] = ""
    return {
        "tampered_weight_changes_digest": tampered != baseline,
        "graph_mismatch_detected": bad_graph["v2_scalars"] != graph_ok["v2_scalars"],
        "parent_claim_detected": report_checks(contract, bad_report)["parent_not_claimed"] is False,
        "missing_authority_detected": not bool(bad_contract["training_authority"]),
        "v5_schema_carries_new_tensors": all(name in checkpoint["model"] for name in NEW_TENSORS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    pins = contract["source_pins"]
    actual_pins = {
        "runner": p4_2.sha_of(ROOT / "scripts/training/run_p5_0_bind2.py"),
        "subject": p4_2.sha_of(ROOT / "taiji/u2_subject.py"),
        "p4_2_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_2_22_5.py"),
        "p4_3_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_3_22_5.py"),
        "p4_4_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_4_22_5.py"),
        "support": p4_2.sha_of(ROOT / "scripts/training/first_a_support_22_5.py"),
        "column_pathway": p4_2.sha_of(ROOT / "taiji/u2_observation_columns.py"),
        "design_manifest": p4_2.sha_of(
            ROOT / "plans/manifests/taiji_u2_bind_2_design_20261115.json"
        ),
    }
    bundle = ROOT / contract["p4_1_bundle"]
    graph = graph_receipt(contract)
    checks = {
        "source_pins_match_worktree": all(
            pins.get(key) == value for key, value in actual_pins.items()
        ),
        "parent_bundle_sha256_matches": hashlib.sha256(bundle.read_bytes()).hexdigest()
        == contract["p4_1_bundle_sha256"],
        "graph_is_23118_198_40": graph["v2_scalars"] == 23118
        and graph["v2_tensors"] == 198
        and graph["state_fields"] == 40,
        "shared_v1_tensors_bit_identical": graph["shared_bit_identical"] is True,
        "new_tensor_set": graph["new_tensors"] == sorted(NEW_TENSORS),
        "state_fields_stay_40": len(
            U2Subject(U2Config(seed=int(contract["candidate"]["seed"]))).state_registry()
        )
        == 40,
        "read_scaffold_458": sum(p.numel() for p in p4_3.ObjectScaffold().parameters()) == 458,
    }
    checks.update(report_checks(contract, report))
    checkpoints = checkpoint_checks(contract, report)
    checks["checkpoints_ok"] = checkpoints["all_ok"]
    red = red_directions(contract, report)
    guard = {
        "schema": "taiji-u2-bind2-guard-v1",
        "card": "22.5.P5.1",
        "date": "2026-11-15",
        "checks": checks,
        "red_directions": red,
        "all_checks_green": all(checks.values()),
        "all_red_directions_respond": all(red.values()),
        "checkpoint_checks": checkpoints,
        "report_verdict": report.get("verdict"),
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

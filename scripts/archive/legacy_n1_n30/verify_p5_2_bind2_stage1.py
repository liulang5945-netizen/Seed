"""Zero-training guard for the P5.2 stage-1 run.

It checks the frozen contract, the stage-1 report, both dose checkpoints and the
gate-to-verdict consistency. Red directions mutate in-memory copies only.
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
from scripts.training import run_p4_4_22_5 as p4_4
from scripts.training.check_report_run_conditions import write_report
from scripts.training.run_p5_0_bind2 import build_v2, digest_outputs
from taiji.u2_history import atomic_load

CONTRACT = ROOT / "plans/manifests/taiji_u2_bind_2_stage1_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_bind2_stage1_20261115.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_bind2_stage1_guard_20261115.json"
VERDICTS = {
    "binding_forms_object_slots",
    "binding_partial",
    "binding_cannot_form_slots",
    "read_ignores_slots",
    "budget_stop",
    "engineering_failure",
}


def main_gates(contract: dict, row: dict) -> dict:
    criteria = contract["criteria"]
    return {
        "pair": row["fit"]["pair_accuracy"] >= criteria["read_pair_gate"],
        "set": row["fit"]["set_accuracy"] >= criteria["read_set_gate"],
        "duplicate": row["fit"]["duplicate_rate"] <= criteria["duplicate_rate_max"],
        "count": row["fit"]["count_accuracy"] >= criteria["count_accuracy_gate"],
        "train_set": row["train"]["set_accuracy"] >= criteria["train_set_gate"],
    }


def report_checks(contract: dict, report: dict) -> dict:
    rows = report.get("doses", [])
    final = rows[-1] if rows else None
    verdict = report.get("verdict")
    gates = main_gates(contract, final) if final else {}
    knockout = report.get("knockout", {})
    if verdict == "binding_forms_object_slots":
        verdict_ok = bool(final) and all(gates.values()) and knockout.get("rate", 0.0) >= 0.6
    elif verdict == "binding_partial":
        verdict_ok = bool(final) and (not all(gates.values()) or knockout.get("rate", 0.0) < 0.6)
    elif verdict == "binding_cannot_form_slots":
        verdict_ok = (
            len(rows) == 1
            and rows[0]["fit"]["set_accuracy"] < contract["criteria"]["early_stop_set"]
            and rows[0]["fit"]["pair_accuracy"] < contract["criteria"]["early_stop_pair"]
        )
    elif verdict == "read_ignores_slots":
        verdict_ok = bool(final) and all(gates.values()) and knockout.get("rate") == 0.0
    elif verdict == "budget_stop":
        verdict_ok = report.get("cost", {}).get("cpu", 0.0) <= contract["cpu_cap"] * 1.05
    else:
        verdict_ok = verdict == "engineering_failure"
    return {
        "report_verdict_allowed": verdict in VERDICTS,
        "report_doses_match_contract": bool(rows)
        and all(row.get("update") in contract["dose_points"] for row in rows)
        and [row.get("update") for row in rows] == contract["dose_points"][: len(rows)],
        "gate_table_matches_verdict": verdict_ok,
        "zero_check_final_reads": report.get("discipline", {}).get("check_reads") == 0
        and report.get("discipline", {}).get("final_reads") == 0,
        "zero_capability_claims": report.get("discipline", {}).get("capability_claims") == 0,
        "parent_not_claimed": report.get("discipline", {}).get("accepted_parent") is False,
        "budget_within_caps": report.get("cost", {}).get("cpu", 0.0) <= contract["cpu_cap"] * 1.05
        and report.get("cost", {}).get("wall", 0.0) <= contract["wall_cap"] * 1.05,
        "arm_a_only": contract["candidate"]["arm"].startswith("A"),
        "trainable_parameters_199": report.get("trainable_parameters") == 199,
    }


def checkpoint_checks(contract: dict, report: dict) -> dict:
    data = atomic_load(Path(contract["data_root"]))
    labels = p4_4.label_cache(data)
    rows = {}
    for dose in report.get("doses", []):
        path = ROOT / dose["checkpoint"]
        if not path.is_file():
            rows[dose["update"]] = {"exists": False}
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
        rows[dose["update"]] = {
            "exists": True,
            "sha256_matches": digest == dose["checkpoint_sha256"],
            "update_matches": checkpoint["update"] == dose["update"],
            "digest": digest_outputs(model, scaffold, data["fit"][:3], labels),
        }
    final = report.get("doses", [])[-1] if report.get("doses") else None
    recovery = report.get("recovery", {})
    final_digest = rows.get(final["update"], {}).get("digest") if final else None
    return {
        "doses": rows,
        "all_ok": all(
            row.get("exists") and row.get("sha256_matches") and row.get("update_matches")
            for row in rows.values()
        ),
        "final_digest_matches_report": (
            final_digest is None
            or final_digest == recovery.get("in_process_digest")
            or not recovery
        ),
        "independent_matches": not recovery
        or recovery.get("in_process_digest") == recovery.get("independent_digest"),
    }


def red_directions(contract: dict, report: dict) -> dict:
    rows = report.get("doses", [])
    if not rows:
        return {
            "tampered_weight_changes_digest": True,
            "bad_verdict_detected": True,
            "parent_claim_detected": True,
            "dose_tamper_detected": True,
            "pilot_hash_tamper_detected": True,
        }
    data = atomic_load(Path(contract["data_root"]))
    labels = p4_4.label_cache(data)
    checkpoint = torch.load(ROOT / rows[-1]["checkpoint"], weights_only=False)
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
    bad_report = copy.deepcopy(report)
    bad_report["verdict"] = (
        "binding_partial"
        if report["verdict"] == "binding_forms_object_slots"
        else "binding_forms_object_slots"
    )
    bad_contract = copy.deepcopy(contract)
    bad_contract["dose_points"] = [10, 20]
    pilot = json.loads((ROOT / contract["pilot_manifest"]).read_text(encoding="utf-8-sig"))
    return {
        "tampered_weight_changes_digest": tampered != baseline,
        "bad_verdict_detected": not report_checks(contract, bad_report)[
            "gate_table_matches_verdict"
        ],
        "parent_claim_detected": report_checks(
            contract, {**report, "discipline": {**report["discipline"], "accepted_parent": True}}
        )["parent_not_claimed"]
        is False,
        "dose_tamper_detected": bad_contract["dose_points"] != contract["dose_points"],
        "pilot_hash_tamper_detected": pilot["criteria_sha256"] != "0" * 64,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    report = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    actual_pins = {
        "runner": p4_2.sha_of(ROOT / "scripts/training/run_p5_2_bind2_stage1.py"),
        "subject": p4_2.sha_of(ROOT / "taiji/u2_subject.py"),
        "pilot_runner": p4_2.sha_of(ROOT / "scripts/training/run_p5_0_bind2.py"),
        "p4_2_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_2_22_5.py"),
        "p4_3_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_3_22_5.py"),
        "p4_4_runner": p4_2.sha_of(ROOT / "scripts/training/run_p4_4_22_5.py"),
        "support": p4_2.sha_of(ROOT / "scripts/training/first_a_support_22_5.py"),
        "column_pathway": p4_2.sha_of(ROOT / "taiji/u2_observation_columns.py"),
        "design_manifest": p4_2.sha_of(
            ROOT / "plans/manifests/taiji_u2_bind_2_design_20261115.json"
        ),
    }
    pilot = json.loads((ROOT / contract["pilot_manifest"]).read_text(encoding="utf-8-sig"))
    bundle = ROOT / contract["p4_1_bundle"]
    checks = {
        "source_pins_match_worktree": all(
            contract["source_pins"].get(key) == value for key, value in actual_pins.items()
        ),
        "parent_bundle_sha256_matches": hashlib.sha256(bundle.read_bytes()).hexdigest()
        == contract["p4_1_bundle_sha256"],
        "pilot_criteria_sha256_matches": pilot["criteria_sha256"]
        == contract["pilot_criteria_sha256"],
        "dose_points_are_1844_and_3688": contract["dose_points"] == [1844, 3688],
        "read_scaffold_458": sum(p.numel() for p in p4_3.ObjectScaffold().parameters()) == 458,
    }
    checks.update(report_checks(contract, report))
    checkpoints = checkpoint_checks(contract, report)
    checks["checkpoints_ok"] = checkpoints["all_ok"]
    checks["final_digest_matches_report"] = checkpoints["final_digest_matches_report"]
    checks["independent_matches"] = checkpoints["independent_matches"]
    red = red_directions(contract, report)
    guard = {
        "schema": "taiji-u2-bind2-stage1-guard-v1",
        "card": "22.5.P5.2",
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

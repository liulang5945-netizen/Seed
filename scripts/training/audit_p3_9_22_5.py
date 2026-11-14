"""Independent re-adjudication of the P3.9 stage receipts.

Reads only committed-style JSON receipts and the frozen manifest; no model,
no forward pass, no optimizer step, no sealed face. Red directions are checked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_9_prereg_20261114.json"
STAGE_NAMES = (
    "taiji_u2_p3_9_1_price_20261114.json",
    "taiji_u2_p3_9_1_graph_20261114.json",
    "taiji_u2_p3_9_2_restore_20261114.json",
    "taiji_u2_p3_9_3_behavior_20261114.json",
    "taiji_u2_p3_9_4_closeout_20261114.json",
)
OUTPUT = REPORTS / "taiji_u2_p3_9_audit_20261114.json"


def sha_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def recompute_behavior(criteria: dict, receipt: dict) -> str:
    rules = criteria["behavior"]
    band = rules["cpu_ratio_band"]
    ci = receipt["paired_bootstrap"]
    delta = ci["delta"]
    if receipt["measured"]["cpu_seconds"] > rules["cap_cpu"]:
        return "comparison_not_admitted"
    if not band[0] <= receipt["cpu_ratio"] <= band[1]:
        return "compute_parity_unavailable"
    if not receipt["dynamic_range_present"]:
        return "budget_cannot_express"
    if ci["ci_lower"] > 0 and delta >= rules["min_meaningful_effect"]:
        return "b_superior_in_window"
    if ci["ci_upper"] < 0 and -delta >= rules["min_meaningful_effect"]:
        return "a_superior_in_window"
    return "inconclusive_within_window"


def adjudicate(manifest: dict, receipts: dict[str, dict]) -> dict:
    criteria = manifest["criteria"]
    checks: dict[str, bool] = {}
    uncovered: list[str] = []
    checks["criteria_sha_matches"] = (
        hashlib.sha256(canonical(manifest["criteria"])).hexdigest() == manifest["criteria_sha256"]
    )
    checks["price_sha_matches"] = sha_of(REPORTS / STAGE_NAMES[0]) == manifest["price"]["sha256"]
    for name in STAGE_NAMES[1:4]:
        checks[f"criteria_sha_in_{name[8:20]}"] = (
            receipts[name].get("criteria_sha256") == manifest["criteria_sha256"]
        )

    graph = receipts[STAGE_NAMES[1]]
    checks["graph_all_checks_green"] = (
        len(graph["checks"]) == 19 and all(graph["checks"].values()) and not graph["uncovered"]
    )
    checks["graph_verdict"] = graph["verdict"] == criteria["graph"]["verdict_pass"]
    restore = receipts[STAGE_NAMES[2]]
    checks["restore_all_checks_green"] = (
        all(restore["checks"].values()) and not restore["uncovered"]
    )
    checks["restore_count"] = (
        len(restore["restore_receipts"]) >= criteria["restore"]["required_restores"] * 2
    )
    checks["restore_independent_processes"] = all(
        row["pid_differs"] and row["digest_matches"] for row in restore["restore_receipts"]
    )
    checks["restore_verdict"] = restore["verdict"] == criteria["restore"]["verdict_pass"]

    behavior = receipts[STAGE_NAMES[3]]
    recomputed = recompute_behavior(criteria, behavior)
    checks["behavior_verdict_matches_decision_rule"] = recomputed == behavior["verdict"]
    checks["behavior_equal_exposure"] = (
        behavior["sample_exposure"]["updates_per_arm"] == criteria["behavior"]["paired_updates"]
        and behavior["arms"]["A"]["updates"] == behavior["arms"]["B"]["updates"]
    )
    checks["behavior_within_budget"] = (
        behavior["measured"]["cpu_seconds"] <= criteria["behavior"]["cap_cpu"]
    )
    checks["behavior_no_capability_claim"] = behavior["discipline"]["capability_claims"] == 0
    checks["behavior_retention_not_signed"] = "not signed" in behavior["retention"]

    closeout = receipts[STAGE_NAMES[4]]
    checks["closeout_stage_sha_matches"] = all(
        row["sha256"] == sha_of(REPORTS / pathlib.Path(row["report"]).name)
        for row in closeout["stages"]
    )
    checks["closeout_five_ledgers"] = set(closeout["five_ledgers"]) == set(
        criteria["closeout"]["ledgers"]
    )
    checks["closeout_no_capability_change"] = closeout["capability_ledger"]["change"] == "none"
    checks["closeout_handoff_gated"] = closeout["handoff_to_formal_ab_qualification"] == (
        behavior["verdict"] in ("b_superior_in_window", "a_superior_in_window")
    )
    checks["no_p3_8_budget_transfer"] = criteria["budget"]["p3_8_budget_transferred"] is False
    checks["execution_hashes_still_null"] = (
        manifest["execution_runner_sha256"] is None and manifest["execution_report_sha256"] is None
    )
    uncovered = [key for key, value in checks.items() if not value]
    verdict = "audit_agrees_with_receipts" if not uncovered else "audit_disagrees"
    return {
        "checks": checks,
        "uncovered": uncovered,
        "audited_behavior_verdict": recomputed,
        "audited_verdict": verdict,
    }


def red_drills(manifest: dict, receipts: dict[str, dict]) -> dict[str, bool]:
    results = {}
    planted = copy.deepcopy(receipts[STAGE_NAMES[3]])
    planted["cpu_ratio"] = 1.0
    results["parity_repaired_changes_verdict"] = (
        adjudicate(manifest, {**receipts, STAGE_NAMES[3]: planted})["audited_behavior_verdict"]
        != adjudicate(manifest, receipts)["audited_behavior_verdict"]
    )
    planted = copy.deepcopy(receipts[STAGE_NAMES[3]])
    planted["cpu_ratio"] = 1.0
    planted["paired_bootstrap"]["ci_lower"] = 0.1
    planted["paired_bootstrap"]["delta"] = 0.2
    results["strong_b_effect_changes_verdict"] = (
        adjudicate(manifest, {**receipts, STAGE_NAMES[3]: planted})["audited_behavior_verdict"]
        == "b_superior_in_window"
    )
    planted = copy.deepcopy(receipts[STAGE_NAMES[1]])
    planted["checks"]["A_float64_output"] = False
    results["graph_gap_fires"] = (
        adjudicate(manifest, {**receipts, STAGE_NAMES[1]: planted})["audited_verdict"]
        == "audit_disagrees"
    )
    planted = copy.deepcopy(receipts[STAGE_NAMES[2]])
    planted["restore_receipts"][0]["digest_matches"] = False
    results["restore_gap_fires"] = (
        adjudicate(manifest, {**receipts, STAGE_NAMES[2]: planted})["audited_verdict"]
        == "audit_disagrees"
    )
    planted = copy.deepcopy(receipts[STAGE_NAMES[4]])
    planted["stages"][0]["sha256"] = "0" * 64
    results["closeout_sha_gap_fires"] = (
        adjudicate(manifest, {**receipts, STAGE_NAMES[4]: planted})["audited_verdict"]
        == "audit_disagrees"
    )
    return results


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    receipts = {name: load(name) for name in STAGE_NAMES}
    result = adjudicate(manifest, receipts)
    drills = red_drills(manifest, receipts)
    passed = result["audited_verdict"] == "audit_agrees_with_receipts" and all(drills.values())
    receipt = {
        "schema": "taiji-u2-p3-9-audit-v1",
        "date": "2026-11-14",
        "criteria_sha256": manifest["criteria_sha256"],
        "audited_verdict": result["audited_verdict"],
        "audited_behavior_verdict": result["audited_behavior_verdict"],
        "checks": result["checks"],
        "uncovered": result["uncovered"],
        "red_drills": drills,
        "stage_sha256": {name: sha_of(REPORTS / name) for name in STAGE_NAMES},
        "passed": passed,
        "measured": {
            "cpu_seconds": round(time.process_time() - started_cpu, 4),
            "wall_seconds": round(time.monotonic() - started_wall, 4),
        },
    }
    OUTPUT.write_bytes(json.dumps(receipt, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "audited_verdict": result["audited_verdict"],
                "checks_green": sum(result["checks"].values()),
                "checks_total": len(result["checks"]),
                "red_drills": drills,
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

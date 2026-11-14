"""Independent re-adjudication of the P3.8.3 comparison receipt."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_8_3_prereg_20261114.json"
MANIFEST = ROOT / "plans/manifests/taiji_u2_p3_8_3_recipe_comparison_20261114.json"
RECEIPT = ROOT / "reports/taiji_u2_p3_8_3_recipe_comparison_20261114.json"
PREFLIGHT = ROOT / "reports/taiji_u2_p3_8_3_preflight_20261114.json"
OUTPUT = ROOT / "reports/taiji_u2_p3_8_3_recipe_comparison_audit_20261114.json"
RUNNER = ROOT / "scripts/training/run_p3_8_3_recipe_comparison_22_5.py"


def canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def adjudicate(criteria: dict, receipt: dict, declared_sha: str) -> dict:
    gates = criteria["gates"]
    arms = receipt["arms"]
    checks: dict[str, bool] = {}
    uncovered: list[str] = []
    checks["criteria_sha_matches"] = canonical_sha(criteria) == declared_sha
    checks["criteria_unchanged_after_reading"] = bool(
        receipt["discipline"]["criteria_sha_unchanged_after_reading"]
    )
    checks["two_arms_present"] = set(arms) == {"A", "B"}
    checks["both_completed_128_updates"] = all(
        arms[key]["update_count"] == criteria["dose"]["paired_updates"] for key in arms
    )
    checks["both_arms_full_capsule_restores"] = all(
        len(arms[key]["restores"]) == 2
        and all(item["digest_matches"] for item in arms[key]["restores"])
        for key in arms
    )
    checks["preflight_passed_before_run"] = (
        json.loads(PREFLIGHT.read_text(encoding="utf-8"))["verdict"] == "preflight_passed"
    )
    checks["within_runner_budget"] = (
        receipt["measured"]["cpu_seconds"] <= criteria["budget"]["runner_cpu_cap"]
        and receipt["measured"]["wall_seconds"] <= criteria["budget"]["runner_wall_cap"]
    )
    checks["within_per_arm_budget"] = all(
        arms[key]["cpu_seconds"] <= criteria["budget"]["per_arm_cpu_cap"] for key in arms
    )
    checks["cpu_parity_within_band"] = (
        receipt["cpu_ratio"] is not None and receipt["cpu_ratio"] <= 2.0
    )
    checks["data_rights_respected"] = all(
        receipt["discipline"][key] == 0 for key in ("check_reads", "final_reads", "confirm_reads")
    )
    checks["bootstrap_complete"] = (
        receipt["paired_bootstrap"] is not None
        and receipt["paired_bootstrap"]["resamples"] == gates["paired_bootstrap_resamples"]
        and receipt["paired_bootstrap"]["paired_units"] == 360
    )
    runner_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    checks["dual_runner_hashes_match_bytes"] = (
        MANIFEST.exists()
        and json.loads(MANIFEST.read_text(encoding="utf-8"))["runner_sha256"] == runner_sha
        and json.loads(MANIFEST.read_text(encoding="utf-8"))["submitted_runner_sha256"]
        == runner_sha
    )

    a_before, a_after = arms["A"]["before"], arms["A"]["after"]
    dynamic = any(
        a_after[key] > 0.5 + 1e-9 or a_after[key] != a_before[key]
        for key in ("accuracy", "first_turn_accuracy")
    )
    if not dynamic:
        verdict = "budget_cannot_express"
    elif (
        receipt["paired_bootstrap"]["observed_delta"] >= gates["minimum_meaningful_gain"]
        and receipt["paired_bootstrap"]["ci_lower"] > 0
    ):
        verdict = "recipe_gain"
    elif (
        receipt["paired_bootstrap"]["observed_delta"] <= 0
        or receipt["paired_bootstrap"]["ci_lower"] <= 0
    ):
        verdict = "recipe_no_gain"
    else:
        verdict = "inconclusive"
    checks["all_faces_and_controls_reported"] = all(
        all(key in arms[arm]["controls"] for key in ("without_rgb", "without_questions"))
        for arm in arms
    )
    if not checks["all_faces_and_controls_reported"]:
        uncovered.append("missing control reading")
    return {
        "checks": checks,
        "uncovered": uncovered,
        "a_arm_dynamic_range": dynamic,
        "audited_verdict": verdict,
    }


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    criteria = contract["criteria"]
    declared_sha = contract["criteria_sha256"]
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    result = adjudicate(criteria, receipt, declared_sha)
    agrees = result["audited_verdict"] == receipt["verdict"]
    # Red direction: giving arm A a dynamic range must change the verdict away
    # from budget_cannot_express, so the guard is not constant-true.
    planted = json.loads(json.dumps(receipt))
    planted["arms"]["A"]["after"]["accuracy"] = 0.6
    planted["arms"]["A"]["after"]["first_turn_accuracy"] = 0.6
    planted["paired_bootstrap"]["observed_delta"] = 0.0
    planted["paired_bootstrap"]["ci_lower"] = 0.0
    red = adjudicate(criteria, planted, declared_sha)["audited_verdict"]
    report = {
        "schema": "taiji-u2-p3-8-3-comparison-audit-v1",
        "date": "2026-11-14",
        "criteria_sha256": contract["criteria_sha256"],
        "receipt_sha256": hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "agrees_with_receipt": agrees,
        "red_direction_verdict_with_dynamic_arm_a": red,
        "red_guard_fires": red != "budget_cannot_express",
        "note": "Independent read-only re-adjudication; the receipt is not modified.",
        **result,
    }
    temporary = OUTPUT.with_suffix(".json.writing")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(OUTPUT)
    print(
        json.dumps(
            {
                "receipt_verdict": receipt["verdict"],
                "audited_verdict": result["audited_verdict"],
                "agrees": agrees,
                "checks_failed": [k for k, v in result["checks"].items() if not v],
                "red_guard_fires": report["red_guard_fires"],
            },
            ensure_ascii=False,
        )
    )
    assert agrees and report["red_guard_fires"] and not result["uncovered"]
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Independent audit of the P3.8.2b proposal receipt against the frozen v2 contract.

The original receipt is left untouched. This audit applies the contract's
decision_rule literally and names every uncovered item, so an over-claimed
verdict cannot be quoted as an engineering closeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONTRACT = ROOT / "plans/manifests/taiji_u2_native_credit_transfer_proposals_v2_20261114.json"
RECEIPT = ROOT / "reports/taiji_u2_native_credit_transfer_proposals_20261114.json"
DUTY = ROOT / "reports/taiji_u2_native_credit_transfer_duty_v2_20261114.json"
OUTPUT = ROOT / "reports/taiji_u2_native_credit_transfer_proposals_audit_20261114.json"


def canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def audit(contract: dict, receipt: dict, duty: dict) -> dict:
    criteria = contract["criteria"]
    rows = receipt["proposals"]
    counters = receipt["counters"]
    checks: dict[str, bool] = {}
    uncovered: list[str] = []

    checks["criteria_sha_matches"] = canonical_sha(criteria) == contract["criteria_sha256"]
    checks["criteria_unchanged_after_reading"] = bool(
        receipt["discipline"]["criteria_sha_unchanged_after_reading"]
    )
    checks["duty_verified_before_proposals"] = duty["verdict"] == "duty_verified"
    checks["six_real_adam_calls"] = counters["real_adam_calls"] == 6
    checks["four_normal_plus_two_fault"] = (
        sum(1 for row in rows if row.get("kind") == "normal") == 4
        and sum(1 for row in rows if row.get("kind") == "fault_before_commit") == 2
    )
    checks["clock_consistency"] = (
        counters["proposal_clock_u"] == 6
        and 0 <= counters["parameter_version_v"] <= counters["proposal_clock_u"]
    )
    checks["fault_losses_restored"] = all(
        row.get("loss_restored") is True for row in rows if row.get("kind") == "fault_before_commit"
    )
    checks["within_budget"] = (
        receipt["measured"]["cpu_seconds"] <= criteria["budget"]["cpu_cap"]
        and receipt["measured"]["wall_seconds"] <= criteria["budget"]["wall_cap"]
    )
    checks["data_rights_respected"] = all(
        receipt["discipline"][key] == 0
        for key in ("dev_reads", "check_reads", "final_reads", "confirm_reads")
    )

    # Contract requires one restore before and one after each accepted commit.
    accepted = counters["accepted_commits"]
    required_restores = 2 * accepted
    actual_restores = len(receipt["restore_checks"])
    checks["restore_coverage_complete"] = actual_restores >= required_restores
    if not checks["restore_coverage_complete"]:
        uncovered.append(
            f"restore coverage {actual_restores}/{required_restores}: the contract requires "
            "one independent-process restore before and one after each accepted commit; "
            "only the first-non-empty-moment restore was recorded"
        )

    # Atomic commit must re-encode legal original history under the new theta.
    has_commit_receipts = any("commit_receipt" in row or "history_reencoded" in row for row in rows)
    checks["atomic_commit_and_history_reencoding_evidenced"] = has_commit_receipts
    if not has_commit_receipts:
        uncovered.append(
            "no per-commit receipt for the parameter-migration transaction or re-encoding "
            "of legal original history under the new theta (card section 2)"
        )

    # Headless equivalence needs a before/after measurement, not a boolean claim.
    headless = receipt.get("headless") or {}
    has_headless_pair = "before_head_digest" in headless and "after_head_digest" in headless
    checks["headless_equivalence_measured"] = has_headless_pair
    if not has_headless_pair:
        uncovered.append(
            "headless equivalence is a declared boolean plus a single digest, not a "
            "before/after body distribution comparison"
        )

    # Fault rollback needs measured parameter digests, not a hardcoded flag.
    fault_rows = [row for row in rows if row.get("kind") == "fault_before_commit"]
    measured_fault_digest = all(
        "digest_before" in row and "digest_after" in row for row in fault_rows
    )
    checks["fault_parameter_digest_measured"] = measured_fault_digest
    if not measured_fault_digest:
        uncovered.append(
            "parameter_digest_restored is a hardcoded True; no digest_before/digest_after pair "
            "was measured for the fault rollback"
        )

    positive = [
        "criteria_sha_matches",
        "criteria_unchanged_after_reading",
        "duty_verified_before_proposals",
        "six_real_adam_calls",
        "four_normal_plus_two_fault",
        "clock_consistency",
        "fault_losses_restored",
        "within_budget",
        "data_rights_respected",
    ]
    all_positive = all(checks[key] for key in positive)
    full_coverage = not uncovered
    verdict = (
        "engineering_transfer_verified"
        if all_positive and full_coverage
        else ("inconclusive_coverage_gaps" if all_positive else "engineering_failure")
    )
    return {
        "checks": checks,
        "uncovered": uncovered,
        "required_restores": required_restores,
        "actual_restores": actual_restores,
        "audited_verdict": verdict,
        "original_verdict": receipt["verdict"],
        "quotation_rule": (
            "The six proposal readings are valid and retained. The original "
            "engineering_transfer_verified verdict must NOT be quoted as an engineering "
            "closeout: three contract requirements are unevidenced. P3.8.3 behaviour "
            "comparison is not admitted."
        ),
    }


def main() -> int:
    global RECEIPT, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", default=str(RECEIPT))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    RECEIPT = Path(args.receipt)
    OUTPUT = Path(args.output)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    duty = json.loads(DUTY.read_text(encoding="utf-8"))
    result = audit(contract, receipt, duty)
    report = {
        "schema": "taiji-u2-p382b-proposals-audit-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.8.2b",
        "contract": CONTRACT.name,
        "criteria_sha256": contract["criteria_sha256"],
        "receipt_sha256": hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
        "duty_sha256": hashlib.sha256(DUTY.read_bytes()).hexdigest(),
        **result,
        "note": "Independent read-only re-adjudication. The original receipt is not modified.",
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
                "original_verdict": result["original_verdict"],
                "audited_verdict": result["audited_verdict"],
                "uncovered": len(result["uncovered"]),
                "restores": f"{result['actual_restores']}/{result['required_restores']}",
            },
            ensure_ascii=False,
        )
    )
    # Red direction: filling every gap and stripping every evidence item must
    # produce different verdicts, so the audit cannot be a constant-true guard.
    planted = json.loads(json.dumps(receipt))
    planted["restore_checks"] = (planted["restore_checks"] or [{"digest_matches": True}]) * 8
    planted["headless"] = {
        **(planted.get("headless") or {}),
        "before_head_digest": "x",
        "after_head_digest": "x",
    }
    for row in planted["proposals"]:
        if row.get("kind") == "normal":
            row["commit_receipt"] = {"status": "committed"}
        if row.get("kind") == "fault_before_commit":
            row["digest_before"] = row["digest_after"] = "x"
    filled = audit(contract, planted, duty)
    stripped = json.loads(json.dumps(receipt))
    stripped["restore_checks"] = []
    stripped["headless"] = {}
    for row in stripped["proposals"]:
        row.pop("commit_receipt", None)
        row.pop("digest_before", None)
        row.pop("digest_after", None)
    emptied = audit(contract, stripped, duty)
    print(
        json.dumps(
            {
                "red_filled": filled["audited_verdict"],
                "red_stripped": emptied["audited_verdict"],
            },
            ensure_ascii=False,
        )
    )
    assert filled["audited_verdict"] != emptied["audited_verdict"], "audit guard is decorative"
    return 0


if __name__ == "__main__":
    sys.exit(main())

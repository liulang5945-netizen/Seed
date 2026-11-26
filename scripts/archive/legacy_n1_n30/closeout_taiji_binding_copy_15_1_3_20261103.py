#!/usr/bin/env python
"""15.1.3 closeout receipt for N15.1 (read-only).

Consumes existing reports, manifests and product hashes only. No model forward,
no optimizer step, no confirm read, no change to any 1.1 product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = "reports/taiji_binding_copy_15_1_3_closeout_20261103.json"
MANIFEST = "plans/manifests/taiji_binding_copy_15_1_3_closeout_20261103.json"
STAGE_CPU_SECONDS = 300.0
STAGE_WALL_SECONDS = 600.0
STAGE_ADAM_CALLS = 0
STAGE_PRODUCT_BYTES = 100 * 1024**2

COMPARE_REPORT = "reports/taiji_binding_copy_15_1_compare_20261102.json"
COMPARE_MANIFEST = "plans/manifests/taiji_binding_copy_15_1_compare_20261102.json"
CLOSEOUT_REPORT = "reports/taiji_binding_copy_15_1_closeout_20261103.json"
CLOSEOUT_MANIFEST = "plans/manifests/taiji_binding_copy_15_1_closeout_20261103.json"
PRICE_REPORT = "reports/taiji_binding_copy_15_1_price_20261102.json"
GRAPH_REPORT = "reports/taiji_binding_copy_15_1_graph_20261102.json"
PREFLIGHT_REPORT = "reports/taiji_binding_copy_15_1_preflight_final_20261102.json"
ATTRIBUTION_REPORT = "reports/taiji_binding_copy_15_1_attribution_20261103.json"
ATTRIBUTION_BUNDLES = "reports/taiji_binding_copy_15_1_attribution_bundles_20261103.json"
PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_SHA256 = "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="closeout", choices=("closeout",))
    parser.parse_args()

    compare = read_json(COMPARE_REPORT)
    closeout = read_json(CLOSEOUT_REPORT)
    price = read_json(PRICE_REPORT)
    graph = read_json(GRAPH_REPORT)
    preflight = read_json(PREFLIGHT_REPORT)
    attribution = read_json(ATTRIBUTION_REPORT)
    bundles = read_json(ATTRIBUTION_BUNDLES)
    closeout_manifest = read_json(CLOSEOUT_MANIFEST)

    retained = closeout_manifest["retained_products"]
    bundle_checks = []
    for entry in retained:
        path = ROOT / entry["path"]
        exists = path.exists()
        actual_sha = sha256_file(path) if exists else None
        bundle_checks.append(
            {
                "path": entry["path"],
                "exists": bool(exists),
                "expected_sha256": entry["sha256"],
                "sha256_matches": bool(exists and actual_sha == entry["sha256"]),
                "bytes_matches": bool(exists and path.stat().st_size == entry["bytes"]),
            }
        )

    alpha_rows = {f"{r['seed']}-{r['arm']}": r for r in bundles["rows"]}
    comparable = {
        f"{c['seed']}-{c['arm']}": c
        for c in compare["configs"]
        if f"{c['seed']}-{c['arm']}" in alpha_rows
    }
    bundled_alpha = all(
        abs(alpha_rows[key]["alpha_record_top1"] - 0.0) == 0.0 for key in comparable
    )

    report = {
        "schema": "taiji-binding-copy-closeout-v1",
        "step": "15.1.3",
        "date": "2026-11-03",
        "status": "closed_negative_with_followup_candidate",
        "verdict_source": "15.1.1 native_byte_task_failed",
        "item_1_candidate_scope": {
            "qualified_parent": None,
            "reason": "no arm passed the task gate",
            "cannot_carry": [
                "unseen output names",
                "new descriptions",
                "both-new combinations",
            ],
            "can_carry_with_caveat": "template-plus-memorised-name answers only (0.5625 overall, 0.1125 unseen combination)",
        },
        "item_2_component_contribution": {
            "content_path_necessary": True,
            "content_drops": closeout["mechanism_gate"]["content_necessity"],
            "content_path_necessary_reading": "removing records drops completions from 0.4167 to 0.0; wrong record 0.375/0.25; wrong position 0.4167",
            "binding_not_established": True,
            "gamma_in_use": False,
            "gamma_evidence": "forcing gamma=0 leaves completions unchanged (drop 0.0), so the copy path is not engaged",
            "record_selection_accuracy": 0.0,
            "record_selection_denominator": 240,
            "curriculum_inert_by_construction": True,
            "curriculum_evidence": "greedy self-prefix equals the teacher prefix on 8/8 train probes after training",
        },
        "item_3_restore_and_cost": {
            "identity_rescue_equal": 8,
            "identity_rescue_denominator": 8,
            "identity_rescue_exact": True,
            "parent_bundle_sha": PARENT_SHA256,
            "parent_unchanged": sha256_file(ROOT / PARENT_BUNDLE) == PARENT_SHA256,
            "total_cpu_seconds": closeout["cost"]["cpu_seconds"],
            "total_wall_seconds": closeout["cost"]["wall_seconds"],
            "adam_calls": closeout["cost"]["adam_calls"],
            "stage_totals": {
                "0a_cpu": read_json("reports/taiji_binding_copy_15_1_data_20261102.json")["cost"][
                    "stage_cpu_seconds"
                ],
                "0b_cpu": read_json("reports/taiji_binding_copy_15_1_graph_20261102.json")["cost"][
                    "total_cpu_seconds"
                ],
                "0c_cpu": price["cost"]["cpu_seconds"],
                "0d_cpu": preflight["cost"]["cpu_seconds"],
                "attribution_probe_cpu": attribution["cost"]["cpu_seconds"],
            },
        },
        "item_4_next_task_fit": {
            "fit": False,
            "reason": "no qualified parent; the current candidate cannot carry cross-session experience or feedback tasks",
            "next_candidate": "plans/reference/SPEC-N16-01_SELECTION_LOCKED_BINDING_DRAFT_20261103.md",
            "method": "A: explicit record selection with selection-locked copy",
            "training_started": True,
            "training_note": "N16.1 is a separate ruled package; this receipt does not consume its results",
        },
        "explicit_registrations": {
            "1.2": "not_released",
            "1.2_reason": "task gate not satisfied",
            "confirm": "sealed_and_unread",
            "14_7": "not_released",
            "N15_2": "not_started",
            "step_12": "not_started",
            "step_13": "not_started",
            "no_additional_seed": True,
            "no_additional_steps": True,
            "no_additional_dose": True,
            "no_head_swap_retry": True,
        },
        "less_change_first_rationale": {
            "rule": "prefer the least-change qualified option before complex structures",
            "why_not_B_only_selection_supervision": "B keeps the soft record mixture and the self-referential copy posterior, so the 15.1 shortcut (memorised pooled content) survives; expected ceiling is lower",
            "why_not_C_curriculum_only": "C was falsified: the R1 curriculum is inert by construction and the decode gate never opens, so a curriculum-only change cannot separate causes",
            "why_A": "A changes exactly three interfaces: the record-level alpha source, the copy-gate floor and the curriculum trigger; it makes selection a separately measurable quantity and keeps the dose unchanged",
            "three_interfaces": [
                "record-level alpha is replaced by a one-hot from the selection head while the model's byte betas are kept",
                "gamma is floored at 0.2 instead of being free to collapse",
                "the self-prefix curriculum triggers on per-batch teacher-forced completion instead of fixed update indices",
            ],
        },
        "alpha_recheck_consistent": bool(bundled_alpha),
        "retained_bundle_checks": bundle_checks,
        "retained_bundles_all_match": all(
            b["exists"] and b["sha256_matches"] and b["bytes_matches"] for b in bundle_checks
        ),
        "confirm_note": {
            "spec": closeout["limits"],
            "materialised": False,
            "read": False,
        },
        "evidence": {
            "compare_report": COMPARE_REPORT,
            "compare_manifest": COMPARE_MANIFEST,
            "closeout_report": CLOSEOUT_REPORT,
            "closeout_manifest": CLOSEOUT_MANIFEST,
            "attribution_report": ATTRIBUTION_REPORT,
            "attribution_bundles": ATTRIBUTION_BUNDLES,
            "price_report": PRICE_REPORT,
            "graph_report": GRAPH_REPORT,
            "preflight_report": PREFLIGHT_REPORT,
        },
        "accounting": {"actual_adam_calls": 0, "effective_adam_calls": 0},
        "cost": {
            "cpu_seconds": time.process_time() - STARTED_CPU,
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "adam_calls": 0,
            "product_bytes": 0,
        },
        "budget": {
            "cpu_seconds": STAGE_CPU_SECONDS,
            "wall_seconds": STAGE_WALL_SECONDS,
            "adam_calls": STAGE_ADAM_CALLS,
            "product_bytes": STAGE_PRODUCT_BYTES,
        },
        "limits": (
            "Read-only closeout. No rerun, no re-evaluation, no confirm read, no product change."
        ),
    }

    checks = {
        "no_forward_no_adam": True,
        "stage_budget": report["cost"]["cpu_seconds"] <= STAGE_CPU_SECONDS
        and report["cost"]["wall_seconds"] <= STAGE_WALL_SECONDS,
        "product_budget": report["cost"]["product_bytes"] <= STAGE_PRODUCT_BYTES,
        "parent_unchanged": report["item_3_restore_and_cost"]["parent_unchanged"],
        "retained_bundles_match": report["retained_bundles_all_match"],
        "alpha_recheck_consistent": report["alpha_recheck_consistent"],
        "verdict_consistent": compare["status"] == "native_byte_task_failed"
        and closeout["status"] == "native_byte_task_failed",
        "runtime_adam_recovered": preflight["cost"]["adam_calls"] == 24
        and price["cost"]["adam_calls"] == 12
        and graph["cost"]["adam_calls"] == 12,
        "evidence_files_present": all(
            Path(p).exists()
            for p in (
                COMPARE_REPORT,
                COMPARE_MANIFEST,
                CLOSEOUT_REPORT,
                CLOSEOUT_MANIFEST,
                PRICE_REPORT,
                GRAPH_REPORT,
                PREFLIGHT_REPORT,
                ATTRIBUTION_REPORT,
                ATTRIBUTION_BUNDLES,
            )
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    report["checks"] = checks
    report["failed"] = failed
    report["status"] = (
        "closed_negative_with_followup_candidate" if not failed else "engineering_failure"
    )

    Path(REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    inputs = [
        COMPARE_REPORT,
        COMPARE_MANIFEST,
        CLOSEOUT_REPORT,
        CLOSEOUT_MANIFEST,
        PRICE_REPORT,
        GRAPH_REPORT,
        PREFLIGHT_REPORT,
        ATTRIBUTION_REPORT,
        ATTRIBUTION_BUNDLES,
    ]
    manifest = {
        "schema": "taiji-binding-copy-manifest-v1",
        "step": "15.1.3",
        "status": report["status"],
        "report": REPORT,
        "inputs": {path: sha256_file(ROOT / path) for path in inputs},
        "retained_products": retained,
        "checks": checks,
        "cost": report["cost"],
        "limits": report["limits"],
    }
    Path(MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"15.1.3 status: {report['status']}")
    print(f"  failed: {failed}")
    print(f"  cpu {report['cost']['cpu_seconds']:.2f}/{STAGE_CPU_SECONDS}, adam 0")
    print(f"  retained bundles match: {report['retained_bundles_all_match']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

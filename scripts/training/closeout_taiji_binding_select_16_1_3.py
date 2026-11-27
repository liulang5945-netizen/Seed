#!/usr/bin/env python
"""N16.1.3 global closeout (read-only, 0 updates, 0 forward).

Consumes the frozen comparison report, the corrected attribution and the
comparison manifest.  It writes the four-item ledger with corrected
denominators, the defect index and mandatory next-contract self-checks, the
terminology errata, the architecture diagnosis with refuted alternatives, and
the admission assessment for the 17 routing candidate.

No model is imported or loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COMPARE = "reports/taiji_binding_select_16_1_compare_20261103.json"
ATTRIBUTION = "reports/taiji_binding_select_16_1_attribution_20261103.json"
COMPARE_MANIFEST = "plans/manifests/taiji_binding_select_16_1_compare_20261103.json"
CANDIDATE_17 = "plans/reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md"
OUT = "reports/taiji_binding_select_16_1_3_closeout_20261103.json"
MANIFEST = "plans/manifests/taiji_binding_select_16_1_3_closeout_20261103.json"

STAGE_CPU_SECONDS = 300.0
STAGE_WALL_SECONDS = 600.0
STAGE_ADAM_CALLS = 0
STAGE_PRODUCT_BYTES = 100 * 1024**2

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CloseoutStop(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="closeout", choices=("closeout",))
    parser.parse_args()

    compare = json.loads((ROOT / COMPARE).read_text(encoding="utf-8"))
    attribution = json.loads((ROOT / ATTRIBUTION).read_text(encoding="utf-8"))
    manifest_in = json.loads((ROOT / COMPARE_MANIFEST).read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}
    checks["compare_is_negative"] = compare["status"] == "native_byte_task_failed"
    checks["attribution_passed"] = (
        attribution["status"] == "instrument_partial_fault_true_negative"
        and not attribution["failed"]
    )
    checks["task_gate_failed"] = attribution["task_pass"] is False
    checks["mechanism_gate_failed"] = attribution["mechanism_pass"] is False
    checks["four_configs"] = len(compare["configs"]) == 4
    checks["full_dose"] = compare["accounting"]["actual_adam_calls"] == 1200
    checks["zero_confirm"] = compare["accounting"]["consumed_confirm_examples"] == 0
    checks["reverse_pair_withdrawn"] = all(
        item["reversed_paired"]["status"] == "withdrawn_index_cache_no_stored_raw"
        for item in attribution["readings"].values()
    )
    checks["four_defects_registered"] = len(attribution["instrument_faults"]) >= 3 and bool(
        attribution["contract_fault"]
    )
    bundle_checks = []
    for item in manifest_in["products"]["bundles"]:
        path = Path(item["path"])
        bundle_checks.append(
            path.exists()
            and path.stat().st_size == item["bytes"]
            and sha256_file(path) == item["sha256"]
        )
    checks["bundles_intact"] = all(bundle_checks)
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise CloseoutStop(f"closeout checks failed: {failed}")

    headline = attribution["corrected_headline"]
    readings = attribution["readings"]
    overall = {seed: headline["overall"][seed] for seed in ("28031", "28032")}
    new_desc = {seed: headline["new_description"][seed] for seed in ("28031", "28032")}
    unseen = {seed: headline["unseen_output_name"][seed] for seed in ("28031", "28032")}
    selection = {seed: headline["selection"][seed] for seed in ("28031", "28032")}
    gamma = {seed: headline["gamma"][seed] for seed in ("28031", "28032")}
    content = {seed: headline["content"][seed] for seed in ("28031", "28032")}
    increment = compare["contrast_A_minus_B"]
    self_prefix = attribution["contract_fault"]["self_prefix_triggers"]

    recomputed = [
        "subset denominators/rates: 60/60/15/36/108 (was 240)",
        "selection-by-subset grouped by (world,index)",
        "content and counterfactual true side looked up by (world,index)",
        "object_exact conditional on a correct record selection",
        "copy weight 1-gamma from the model gamma mean",
        "reverse accuracy from stored per-row scores (paired rate withdrawn)",
        "per-config cost sums and bundle sha verification",
    ]
    preserved = [
        "overall completion over the 240 dev rows",
        "selection top-1 over the 240 dev rows",
        "model/used gamma means and step0 gamma",
        "identity rescue 8/8 and teacher-forced NLL",
        "per-row score dictionaries and bundle bytes/sha",
        "raw reverse-pair flag (kept as evidence, withdrawn as a gate)",
        "1.1 wall/cpu timing and parent bundle identity b75a83a0...",
    ]

    four_items = {
        "1_reliable_range": {
            "qualified_parent": False,
            "reason": (
                "The 14.6 A_full step300 parent can carry the same-distribution template and "
                "memorized names/places (unlocked control 0.3375/0.2833 overall) but not "
                "unseen output names (0/60 both seeds), not new descriptions (3/60 and 1/60), "
                "not both-new rows (0/15), and not description-conditioned record selection "
                "(0/60 selection on new descriptions)."
            ),
            "supported_surface": (
                "in-distribution template + memorized entity names through the generation "
                "branch; content is used when records are present"
            ),
            "unsupported_surface": (
                "description-compositional generalization, entity-start copy, unseen names, "
                "and the cross-session experience/feedback task"
            ),
        },
        "2_mechanism_contribution": {
            "content_necessity": {
                "28031": content["28031"],
                "28032": content["28032"],
            },
            "selection_decode_binding": "not established",
            "selection": selection,
            "selection_new_description": {
                seed: headline["selection_new_description"][seed] for seed in ("28031", "28032")
            },
            "copy_weight": {
                seed: round(gamma[seed]["copy_weight_mean"], 4) for seed in ("28031", "28032")
            },
            "object_exact_given_correct_selection": {
                seed: round(
                    readings[f"{seed}-A-locked"]["given_correct_selection"]["object_exact"], 4
                )
                for seed in ("28031", "28032")
            },
            "gamma_floor": {
                "reading": (
                    "gamma is the generation weight; the 0.2 clamp is a generation floor, not a "
                    "copy floor.  The floor was active on 14.6%/23.0% of dev steps and changing "
                    "it to 0 moved completion by 0.0/+0.0167, so it does not carry the score."
                )
            },
            "curriculum": {
                "A_triggers": {seed: self_prefix[seed]["A"] for seed in ("28031", "28032")},
                "B_triggers": {seed: self_prefix[seed]["B"] for seed in ("28031", "28032")},
                "note": (
                    "A never reaches the 0.5 batch completion threshold; B fires 127/121 times "
                    "with self-imitation labels, so A-B is confounded."
                ),
            },
            "increment": {
                "pooled_mean": increment["pooled_mean"],
                "bootstrap_lower": {
                    seed: increment["per_seed"][seed]["bootstrap"]["lower"]
                    for seed in ("28031", "28032")
                },
                "usable_as_clean_mechanism_test": False,
            },
        },
        "3_recovery_and_old_behavior": {
            "identity_rescue": {
                seed: readings[f"{seed}-A-locked"]["identity_rescue"] for seed in ("28031", "28032")
            },
            "parent_bundle_sha256": manifest_in["parent_sha256"],
            "parent_unchanged": True,
            "total_cost_cpu_seconds": compare["cost_detail"]["total_cpu_seconds"],
            "total_cost_wall_seconds": compare["cost_detail"]["total_wall_seconds"],
            "actual_adam_calls": compare["accounting"]["actual_adam_calls"],
            "consumed_train_examples": compare["accounting"]["consumed_train_examples"],
            "consumed_confirm_examples": 0,
            "old_behavior": (
                "Unlocked B keeps the memorization shortcut (0.3375/0.2833) with 0/60 unseen "
                "names; locked A removes the shortcut (0.0500/0.0167) without installing a "
                "correct copy.  Rescue is bit-exact 8/8 in both arms."
            ),
        },
        "4_next_task_applicability": {
            "current_parent_can_carry_cross_session_experience_feedback": False,
            "reason": (
                "The parent cannot even carry description-compositional selection or entity-start "
                "alignment, so it cannot carry a cross-session experience/feedback task that "
                "requires revising which record/content is consumed from accumulated history."
            ),
            "next_candidate": "17 stepwise selection routing (design only, not launched)",
        },
    }

    defects = [
        {
            "id": "D16-1",
            "symptom": "subset rates divided by 240 instead of the subset length",
            "impact": (
                "new-description completion printed 0.0125/0.0042 instead of 3/60=0.05 and "
                "1/60=0.0167; subset gates were evaluated on the wrong scale (pass/fail "
                "unchanged because the counts are small)"
            ),
            "action": "recomputed read-only with 60/60/15/36/108; raw card preserved",
            "next_contract_self_check": "S1 subset denominators from subset length; all-correct oracle = 1 per non-empty subset",
            "status": "corrected_read_only",
        },
        {
            "id": "D16-2",
            "symptom": "true-side caches keyed by the world-local index (20 values for 240 rows)",
            "impact": (
                "content-necessity and counterfactual true sides and selection-by-subset "
                "grouping used another world's row; reverse pairing used the wrong true_raw"
            ),
            "action": (
                "recomputed content/counterfactual/selection by (world,index); reverse pairing "
                "withdrawn because the raw bytes were not stored"
            ),
            "next_contract_self_check": "S2 all cross-world caches keyed by (split,world,index); store raw bytes for paired metrics",
            "status": "corrected_read_only",
        },
        {
            "id": "D16-3",
            "symptom": "gamma read as a copy gate",
            "impact": (
                "gamma 0.8212/0.7471 was reported as copy use, but gamma multiplies the "
                "generated branch; the copy weight is 0.1788/0.2529 and the 0.2 clamp is a "
                "generation floor"
            ),
            "action": "errata written; report gamma and 1-gamma; floor-removal sensitivity reported",
            "next_contract_self_check": "S3 gamma semantic unit test (gamma=0 copy-dominant, gamma=1 generation-dominant; 0.2 is a generation floor)",
            "status": "corrected_read_only",
        },
        {
            "id": "D16-4",
            "symptom": "self-prefix bytes passed as the target labels",
            "impact": (
                "when the accuracy trigger fired, the answer NLL trained self-imitation; B fired "
                "127/121 updates, A fired 0, so A-B mixes the lock with a different curriculum"
            ),
            "action": (
                "registered as a contract fault; A's result is unaffected because it never "
                "triggered; B's result is reported as self-imitation; A-B is not used as a clean "
                "mechanism test"
            ),
            "next_contract_self_check": "S4 input_prefix and target_bytes separate; labels always from the teacher target",
            "status": "registered_contract_fault",
        },
    ]

    mandatory_self_checks = [
        {
            "id": "S1",
            "check": "every subset rate uses its own denominator; a stored all-correct oracle scores 1.0 on each non-empty subset; an empty subset is reported not evaluable",
        },
        {
            "id": "S2",
            "check": "every cross-world cache/join is keyed by (split,world,index); paired metrics store raw bytes and EOS and normalize byte sequences before comparison",
        },
        {
            "id": "S3",
            "check": "gamma semantic unit test: with alpha one-hot, gamma=0 must be copy-dominant and gamma=1 generation-dominant; the 0.2 clamp is asserted on the generation weight and both gamma and 1-gamma are reported",
        },
        {
            "id": "S4",
            "check": "the self-prefix branch changes only input_prefix; target_bytes stay the original 28 labels; a wrong-prefix counterexample must keep the same 28 target IDs and the same target NLL and be distinguishable from self-imitation",
        },
        {
            "id": "S5",
            "check": "score-card preflight re-scores a stored card with the new aggregator: denominators, per-subset oracle=1, paired rate from stored raw; failures block the formal run",
        },
        {
            "id": "S6",
            "check": "if two arms use different curricula, the A-B contrast is reported as confounded; equal curricula or an independent mechanism probe is required before a mechanism claim",
        },
    ]

    errata = [
        {
            "term": "gamma",
            "old_reading": "copy gate; gamma>=0.20 means copy in use",
            "corrected_reading": "generation weight; copy weight is 1-gamma; gamma>=0.20 is a generation floor",
            "source": "taiji/binding_copy.py answer_step probability formula",
        },
        {
            "term": "source_top1 / alpha",
            "old_reading": "entity-stage record selection",
            "corrected_reading": "first-output-step max-byte record (15.1) / record-level alpha (16.1); entity-stage selection needs its own probe",
            "source": "evaluate_source_top1 and selection_report",
        },
        {
            "term": "reverse paired",
            "old_reading": "a task gate at 0.95",
            "corrected_reading": "paired bytes+EOS agreement, not reversed accuracy; withdrawn in 16.1.1 because the true_raw cache was world-local and raw bytes were not stored",
            "source": "D16-2",
        },
        {
            "term": "unseen entity",
            "old_reading": "unseen output name",
            "corrected_reading": "new-description axis in 15.1; unseen output name is a separate 60-row subset",
            "source": "15.1 post-closeout audit",
        },
    ]

    diagnosis = {
        "breakpoint_1": {
            "name": "description matcher lacks compositional generalization",
            "evidence": {
                "selection_new_description": {
                    seed: headline["selection_new_description"][seed] for seed in ("28031", "28032")
                },
                "selection_seen": {
                    seed: readings[f"{seed}-A-locked"]["selection_by_subset"]["seen"]["rate"]
                    for seed in ("28031", "28032")
                },
                "selection_unseen_name": {
                    seed: readings[f"{seed}-A-locked"]["selection_by_subset"]["unseen_output_name"][
                        "rate"
                    ]
                    for seed in ("28031", "28032")
                },
                "head": "mean byte embedding; byte-count permutations are indistinguishable; both arms train the same head",
            },
        },
        "breakpoint_2": {
            "name": "entity-start / beta alignment missing",
            "evidence": {
                "copy_weight": {
                    seed: round(gamma[seed]["copy_weight_mean"], 4) for seed in ("28031", "28032")
                },
                "object_exact_given_correct_selection": {
                    seed: round(
                        readings[f"{seed}-A-locked"]["given_correct_selection"]["object_exact"], 4
                    )
                    for seed in ("28031", "28032")
                },
                "completed_given_correct_selection": {
                    seed: round(
                        readings[f"{seed}-A-locked"]["given_correct_selection"]["completed"], 4
                    )
                    for seed in ("28031", "28032")
                },
                "first_divergence_at_byte_3": {
                    seed: round(readings[f"{seed}-A-locked"]["first_divergence_3"], 4)
                    for seed in ("28031", "28032")
                },
                "teacher_forced_nll": {
                    seed: round(readings[f"{seed}-A-locked"]["nll"], 4)
                    for seed in ("28031", "28032")
                },
            },
        },
        "refuted_alternatives": {
            "dose": "loss/NLL fell (1.525/1.26) and gamma rose to 0.82/0.75, yet new descriptions stay 0/60 and overall 0.05/0.0167",
            "seed": "both seeds show 0/60 new-description selection and 0/60 unseen names; no seed rescues the task",
            "steps": "300 updates converged; the failure is a structural 0/60, not a mid-training snapshot",
            "learning_rate_or_optimizer": "the same failure appears in both arms/seeds; clip/loss traces do not show a recoverable optimization gap (if '利润率' refers to another quantity, register it explicitly)",
            "gamma_floor": "model gamma is 0.82/0.75, far above 0.2; removing the floor changes completion by 0.0/+0.0167, so the floor is not the binding constraint",
            "capacity": "25,602 head parameters fail a simple description match; the mean embedding is order-blind, so more parameters do not add string matching",
            "data_or_labels": "keys are descriptions and labels are asserted unique per row; new descriptions are unseen by construction, not mislabeled",
        },
        "conclusion": (
            "Two independent architecture breaks: compositional description matching and "
            "entity-start copy alignment.  Fixing one alone leaves the other as the binding "
            "ceiling; no single small change clears the task."
        ),
    }

    candidate = {
        "reference": CANDIDATE_17,
        "design_unchanged": True,
        "differences_from_16_1": [
            "shared ordered record encoder + attention pool for the selection head (37,120 params) instead of mean byte embeddings",
            "per-output-step object/place/null routing instead of one record choice for the whole sentence",
            "final-Q locking: selected record only; other record mass zero; null means zero copy mass",
            "smooth generation floor gamma_eff = 0.2 + 0.8*gamma_raw instead of a clamp with zero gradient below 0.2",
            "straight-through task credit for the hard routing choice, plus a stop-credit control arm",
            "input_prefix and target_bytes separated, so the self-prefix branch keeps the original targets",
            "three arms A_joint_lock / B_stop_lock / C_unlocked and a separate 240-row confirm set",
        ],
        "ceiling": [
            "addresses both breakpoints in one graph: order-aware description matching plus phase-specific object/place routing and entity-start labels",
            "removes the memorization shortcut from the mechanism contrast (C unlocked vs A/B locked)",
            "repairs the curriculum target so the self-prefix factor is interpretable",
            "the smooth floor keeps a gradient where 16.1's clamp was dead",
        ],
        "risks": [
            "1,800 formal updates (3 arms x 2 seeds x 300); extrapolating 16.1's 11.19 CPU/update gives about 20,150 CPU before extra evaluation, likely crossing the 20,000 line; a pre-run resource ruling after 17.3 pricing is required",
            "new routing/phase labels, final-Q invariants and ST-VJP add engineering surface; 0b/0d must prove them before 1.1",
            "the shared encoder's generalization to new descriptions is still unproven; it is the main scientific risk",
            "ST credit is an approximation; A-B only supports the extra task credit if both seeds' intervals clear zero",
            "the separate confirm set requires its own frozen generation and access proof",
        ],
        "measured_budget_basis": {
            "16_1_actual_cpu_seconds": compare["cost_detail"]["total_cpu_seconds"],
            "16_1_updates": 1200,
            "cpu_per_update": round(compare["cost_detail"]["total_cpu_seconds"] / 1200, 3),
            "17_formal_updates": 1800,
            "17_projected_cpu_seconds": round(
                compare["cost_detail"]["total_cpu_seconds"] / 1200 * 1800, 1
            ),
            "observation_line_cpu_seconds": 20000.0,
            "verdict": "likely over the line before extra evaluation; resource ruling required",
        },
        "smaller_alternatives": {
            "only_description_encoder": (
                "fixes the 0/60 new-description selection but leaves beta/entity alignment and "
                "place routing; the ceiling stays near object_exact 0.29/0.05 even with correct "
                "selection, so it cannot reach 0.80"
            ),
            "only_entity_alignment": (
                "fixes the copy start but leaves selection at 0/60 on new descriptions and no "
                "place routing; the ceiling stays near a 1/9-1/3 random record choice"
            ),
            "only_label_separation_or_curriculum": (
                "fixes B's self-imitation confound but neither breakpoint; it only makes the "
                "A-B contrast interpretable"
            ),
            "minimal_sufficient_set": (
                "ordered description encoder + per-step object/place/null routing + phase labels "
                "+ separated self-prefix target + a task gradient for the hard choice (ST or "
                "equivalent).  This set is inherently multi-part; no single small change is "
                "sufficient."
            ),
        },
        "least_change_rule": (
            "The least-change rule prefers the smallest change that clears the gate.  Here no "
            "single small change clears both independent breakpoints, so the minimal sufficient "
            "set is multi-part.  Candidate 17 contains that set plus a diagnostic third arm; it "
            "is the higher-ceiling primary.  Its cost is the binding risk, not its design: if the "
            "owner fixes the budget, a two-arm A/B version of the same design is the smaller "
            "variant (drop C), not a substitute for the design."
        ),
        "admission_recommendation": (
            "17 remains the primary on ceiling; admission should be conditional on 17.3 pricing "
            "and an owner resource ruling, or on an explicit two-arm A/B scope with the same "
            "graph.  Do not launch it in this closeout."
        ),
    }

    release = {
        "N16.1.2": "not_released",
        "confirm": "sealed",
        "14.7": "not_released",
        "N15.2": "not_started",
        "12_13": "not_started",
        "extra_seeds_steps_dose": False,
        "rerun": False,
    }

    products = {
        "retained_bundles": manifest_in["products"]["bundles"],
        "retained_bundle_bytes": manifest_in["products"]["bundle_bytes"],
        "deleted_in_1_1": manifest_in["products"]["deleted"],
        "deleted_in_1_1_bytes": manifest_in["products"]["deleted_bytes"],
        "deleted_in_1_3": [],
        "deleted_in_1_3_bytes": 0,
    }

    report = {
        "schema": "taiji-binding-select-closeout-v1",
        "step": "N16.1.3",
        "date": "2026-11-03",
        "status": "closed_negative_with_followup_candidate",
        "verdict": "native_byte_task_failed",
        "release": release,
        "checks": checks,
        "failed": failed,
        "four_items": four_items,
        "corrected_headline": {
            "overall": overall,
            "new_description": new_desc,
            "unseen_output_name": unseen,
            "selection": selection,
        },
        "recomputed_vs_preserved": {
            "recomputed_read_only": recomputed,
            "preserved_original": preserved,
            "superseded_fields": [
                "compare card subset rates (240 denominator)",
                "compare card content/counterfactual true side",
                "compare card selection-by-subset grouping",
                "compare card reverse-pair rate (withdrawn)",
                "compare card gamma-as-copy-gate reading",
            ],
        },
        "defects": defects,
        "mandatory_self_checks": mandatory_self_checks,
        "terminology_errata": errata,
        "architecture_diagnosis": diagnosis,
        "candidate_17_admission": candidate,
        "accounting": {
            "updates": 0,
            "forward_calls": 0,
            "adam_calls": 0,
            "confirm_examples": 0,
            "source_reports_read": [COMPARE, ATTRIBUTION, COMPARE_MANIFEST],
        },
        "cost": {
            "cpu_seconds": time.process_time() - STARTED_CPU,
            "wall_seconds": time.perf_counter() - STARTED_WALL,
            "adam_calls": 0,
            "forward_calls": 0,
        },
        "budget": {
            "cpu_seconds": STAGE_CPU_SECONDS,
            "wall_seconds": STAGE_WALL_SECONDS,
            "adam_calls": STAGE_ADAM_CALLS,
            "product_bytes": STAGE_PRODUCT_BYTES,
        },
        "products": products,
        "source_hashes": {
            COMPARE: sha256_file(ROOT / COMPARE),
            ATTRIBUTION: sha256_file(ROOT / ATTRIBUTION),
            COMPARE_MANIFEST: sha256_file(ROOT / COMPARE_MANIFEST),
            CANDIDATE_17: sha256_file(ROOT / CANDIDATE_17),
            "scripts/training/closeout_taiji_binding_select_16_1_3.py": sha256_file(
                ROOT / "scripts/training/closeout_taiji_binding_select_16_1_3.py"
            ),
        },
        "next_step": (
            "owner ruling on candidate 17 (design unchanged; admission conditional on 17.3 "
            "pricing and a resource ruling, or a two-arm A/B scope); do not launch training"
        ),
        "limits": (
            "Read-only closeout of the frozen 1.1 card: 0 updates, 0 forward, 0 confirm.  It "
            "does not re-run, change the dose/gates or release 1.2/14.7/N15.2/12/13."
        ),
    }

    cpu = report["cost"]["cpu_seconds"]
    wall = report["cost"]["wall_seconds"]
    if cpu > STAGE_CPU_SECONDS or wall > STAGE_WALL_SECONDS:
        raise CloseoutStop(f"stage budget exceeded: {cpu:.2f}/{wall:.2f}")
    out_path = ROOT / OUT
    out_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if len(out_text.encode("utf-8")) > STAGE_PRODUCT_BYTES:
        raise CloseoutStop("product budget exceeded")
    out_path.write_text(out_text, encoding="utf-8")

    manifest = {
        "schema": "taiji-binding-select-manifest-v1",
        "step": "N16.1.3",
        "status": report["status"],
        "report": OUT,
        "compare_report": COMPARE,
        "attribution_report": ATTRIBUTION,
        "checks": checks,
        "failed": failed,
        "release": release,
        "defects": [item["id"] for item in defects],
        "mandatory_self_checks": [item["id"] for item in mandatory_self_checks],
        "candidate_17_admission": candidate["admission_recommendation"],
        "cost": report["cost"],
        "products": products,
        "source_hashes": report["source_hashes"],
        "limits": report["limits"],
    }
    manifest_path = ROOT / MANIFEST
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    total_product = len(out_text.encode("utf-8")) + len(manifest_text.encode("utf-8"))
    print(f"N16.1.3 status: {report['status']}")
    print(f"  defects: {[item['id'] for item in defects]}")
    print(f"  checks: {len(checks)}/{len(checks)} passed")
    print(
        f"  cost: {cpu:.2f}/{STAGE_CPU_SECONDS} CPU, {wall:.2f}/{STAGE_WALL_SECONDS} wall, 0 Adam, 0 forward"
    )
    print(f"  products: {total_product} bytes (cap {STAGE_PRODUCT_BYTES})")
    print(f"  next: {report['next_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

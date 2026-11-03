#!/usr/bin/env python
"""N16.1.1 read-only attribution and correction (0 updates, 0 forward).

Consumes only the frozen comparison report.  It applies the (split, world,
index) key discipline to the selection/content/counterfactual baselines,
recomputes the subset denominators, registers the gamma-direction and
self-prefix label faults, withdraws the reverse-pair reading that cannot be
rebuilt without rerunning deterministic generations, and records the single
next step.  No model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REPORT = "reports/taiji_binding_select_16_1_compare_20261103.json"
MANIFEST = "plans/manifests/taiji_binding_select_16_1_attribution_20261103.json"
OUT = "reports/taiji_binding_select_16_1_attribution_20261103.json"

SEEDS = (28031, 28032)
ARMS = ("A-locked", "B-select")
GATE_SELECTION = 0.90
GATE_UNSEEN = 0.80
GATE_NEW_DESCRIPTION = 0.80
GATE_OVERALL = 0.80
GATE_REVERSED = 0.95
GATE_CONTENT_DROP = 0.10
GATE_GAMMA = 0.20
GATE_INCREMENT = 0.10
TRAIN_UPDATES = 300
PRICE_CPU_SECONDS = 8390.9375
PRE_RUN_PROJECTION = 14300.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def per_config(entry: dict) -> dict:
    step = entry["step300"]
    rows = step["rows"]
    by_key = {(row["world"], row["index"]): row for row in rows}
    selection = step["selection"]["rows"]

    subsets = {}
    for name in ("new_description", "unseen_output_name", "both_new", "unseen_combination"):
        n = sum(1 for row in rows if row["flags"][name])
        completed = sum(1 for row in rows if row["flags"][name] and row["score"]["completed"])
        subsets[name] = {
            "completed": completed,
            "n": n,
            "rate": completed / n if n else None,
        }
    seen = [row for row in rows if not any(row["flags"].values())]
    subsets["seen"] = {
        "completed": sum(1 for row in seen if row["score"]["completed"]),
        "n": len(seen),
        "rate": (sum(1 for row in seen if row["score"]["completed"]) / len(seen) if seen else None),
    }

    selection_by_subset: dict[str, dict] = {}
    for item in selection:
        flags = by_key[(item["world"], item["index"])]["flags"]
        keys = [
            name
            for name in ("new_description", "unseen_output_name", "both_new", "unseen_combination")
            if flags[name]
        ]
        if not any(flags.values()):
            keys = ["seen"]
        for name in keys:
            bucket = selection_by_subset.setdefault(name, {"correct": 0, "n": 0})
            bucket["correct"] += int(item["correct"])
            bucket["n"] += 1
    for _name, bucket in selection_by_subset.items():
        bucket["rate"] = bucket["correct"] / bucket["n"] if bucket["n"] else None

    total_selection = {
        "correct": sum(1 for item in selection if item["correct"]),
        "n": len(selection),
        "top1": sum(1 for item in selection if item["correct"]) / len(selection),
    }

    interventions = step["content_interventions"]
    content = {"applicable": bool(interventions.get("applicable"))}
    if content["applicable"]:
        targets = interventions["rows"]
        true_completed = sum(
            1 for item in targets if by_key[(item["world"], item["index"])]["score"]["completed"]
        )
        intervention_rates = {
            name: sum(1 for item in targets if item[f"{name}_completed"]) / len(targets)
            for name in ("no_source", "wrong_record", "wrong_position")
        }
        content.update(
            {
                "n": len(targets),
                "true_completed": true_completed,
                "true_rate": true_completed / len(targets),
                "intervention_rates": intervention_rates,
                "drops": {
                    name: true_completed / len(targets) - rate
                    for name, rate in intervention_rates.items()
                },
                "degenerate": true_completed == 0,
            }
        )
    else:
        content["reason"] = interventions.get("reason")

    counterfactual = step["counterfactual"]
    cf_true = sum(
        1
        for item in counterfactual["rows"]
        if by_key[(item["world"], item["index"])]["score"]["completed"]
    )
    cf_both = sum(
        1
        for item in counterfactual["rows"]
        if item["counterfactual_completed"]
        and by_key[(item["world"], item["index"])]["score"]["completed"]
    )
    counterfactual_corrected = {
        "n": len(counterfactual["rows"]),
        "true_completed": cf_true,
        "both_correct": cf_both / len(counterfactual["rows"]),
        "counterfactual_completed": sum(
            1 for item in counterfactual["rows"] if item["counterfactual_completed"]
        ),
    }

    selection_by_key = {(item["world"], item["index"]): item["correct"] for item in selection}
    correct_rows = [row for row in rows if selection_by_key.get((row["world"], row["index"]))]
    given_correct = {
        "n": len(correct_rows),
        "object_exact": (
            sum(1 for row in correct_rows if row["score"]["object_exact"]) / len(correct_rows)
            if correct_rows
            else 0.0
        ),
        "completed": (
            sum(1 for row in correct_rows if row["score"]["completed"]) / len(correct_rows)
            if correct_rows
            else 0.0
        ),
    }

    gamma = step["gamma"]
    gamma_report = {
        "gamma_model_mean": gamma["gamma_model_mean"],
        "gamma_used_mean": gamma["gamma_used_mean"],
        "copy_weight_mean": 1.0 - gamma["gamma_model_mean"],
        "floored_step_fraction": gamma["floored_step_fraction"],
        "examples_with_any_floored": gamma["examples_with_any_floored"],
        "step0_model_mean": entry["step0"]["gamma"]["gamma_model_mean"],
        "direction": (
            "gamma multiplies the generated branch: P = [gamma + (1-gamma)*q_null]*P_gen "
            "+ (1-gamma)*P_copy.  gamma is the generation weight; the 0.2 clamp is a "
            "generation floor and does not force copying."
        ),
    }

    return {
        "seed": entry["seed"],
        "arm": entry["arm"],
        "overall": step["overall"]["completed"],
        "subsets": subsets,
        "selection": total_selection,
        "selection_by_subset": selection_by_subset,
        "object_exact": sum(1 for row in rows if row["score"]["object_exact"]) / len(rows),
        "first_divergence_3": sum(1 for row in rows if row["score"]["first_divergence"] == 3)
        / len(rows),
        "given_correct_selection": given_correct,
        "alpha_top1": step["alpha_reading"]["alpha_top1"],
        "byte_top1": step["alpha_reading"]["byte_top1"],
        "gamma": gamma_report,
        "reversed_accuracy": step["reversed"]["accuracy"],
        "reversed_paired": {
            "stored": step["reversed"]["paired_rate"],
            "status": "withdrawn_index_cache_no_stored_raw",
        },
        "content": content,
        "counterfactual": counterfactual_corrected,
        "identity_rescue": [
            step["identity_rescue"]["equal"],
            step["identity_rescue"]["denominator"],
        ],
        "nll": step["nll"]["mean_nll"],
        "curriculum": {
            "rate_max": entry["train"]["curriculum_rate_max"],
            "self_prefix_updates": entry["train"]["self_prefix_updates"],
        },
        "train": {
            "first_loss": entry["train"]["first_loss"],
            "last_loss": entry["train"]["last_loss"],
            "last20": entry["train"]["loss_last20_mean"],
        },
        "sensitivity": entry["sensitivity"],
        "step0": {
            "overall": entry["step0"]["overall"]["completed"],
            "selection_top1": entry["step0"]["selection"]["top1"],
        },
        "cost": entry["cost"],
        "updates": entry["updates"],
        "resumed_from": entry["resumed_from"],
    }


def audit(report: dict) -> dict:
    entries = {(entry["seed"], entry["arm"]): entry for entry in report["configs"]}
    readings = {(seed, arm): per_config(entries[(seed, arm)]) for seed in SEEDS for arm in ARMS}

    checks: dict[str, bool] = {}
    checks["report_is_negative"] = report["status"] == "native_byte_task_failed"
    checks["config_grid"] = len(entries) == 4
    checks["full_dose"] = all(
        item["updates"] == TRAIN_UPDATES and item["cost"]["adam_calls"] == TRAIN_UPDATES
        for item in readings.values()
    )
    checks["precheck_passed"] = report["precheck"]["status"] == "passed" and all(
        report["precheck"]["checks"].values()
    )
    checks["precheck_instrument_parity"] = bool(
        report["precheck"]["checks"].get("instrument_parity")
    )
    checks["no_confirm"] = report["accounting"]["consumed_confirm_examples"] == 0
    checks["step0_equal_across_arms"] = all(
        readings[(seed, "A-locked")]["step0"]["overall"]
        == readings[(seed, "B-select")]["step0"]["overall"]
        for seed in SEEDS
    )
    checks["selection_symmetry"] = all(
        abs(
            readings[(seed, "A-locked")]["selection"]["top1"]
            - readings[(seed, "B-select")]["selection"]["top1"]
        )
        <= 0.05
        for seed in SEEDS
    )
    checks["lock_changes_decode"] = all(
        readings[(seed, "B-select")]["alpha_top1"] == 0.0
        and readings[(seed, "A-locked")]["overall"] < 0.10
        and readings[(seed, "B-select")]["overall"] > 0.25
        for seed in SEEDS
    )
    checks["no_rerun"] = all(item["resumed_from"] == 0 for item in readings.values())
    checks["wall_matches_cpu"] = all(
        abs(item["cost"]["wall_seconds"] - item["cost"]["cpu_seconds"])
        / max(item["cost"]["cpu_seconds"], 1.0)
        <= 0.25
        for item in readings.values()
    )
    checks["under_watch_line"] = (
        report["cost_detail"]["total_cpu_seconds"] < 20000.0
        and report["cost_detail"]["total_wall_seconds"] < 30000.0
    )
    checks["primary_task_readings_cache_free"] = all(
        item["subsets"]["new_description"]["n"] == 60
        and item["subsets"]["unseen_output_name"]["n"] == 60
        and item["subsets"]["both_new"]["n"] == 15
        and item["subsets"]["unseen_combination"]["n"] == 36
        and item["subsets"]["seen"]["n"] == 108
        for item in readings.values()
    )
    checks["selection_denominators_240"] = all(
        item["selection"]["n"] == 240 for item in readings.values()
    )

    # Task gate with corrected denominators.  The reverse-pair gate is withdrawn:
    # the stored pairing used a local-index true_raw cache (20 values for 240
    # rows), and the raw bytes were not stored, so it cannot be rebuilt without
    # rerunning deterministic generations.
    task = {}
    for seed in SEEDS:
        item = readings[(seed, "A-locked")]
        task[str(seed)] = {
            "overall": {
                "value": item["overall"],
                "threshold": GATE_OVERALL,
                "pass": item["overall"] >= GATE_OVERALL,
            },
            "unseen_output_name": {
                "value": item["subsets"]["unseen_output_name"]["rate"],
                "count": item["subsets"]["unseen_output_name"]["completed"],
                "n": item["subsets"]["unseen_output_name"]["n"],
                "threshold": GATE_UNSEEN,
                "pass": item["subsets"]["unseen_output_name"]["rate"] >= GATE_UNSEEN,
            },
            "new_description": {
                "value": item["subsets"]["new_description"]["rate"],
                "count": item["subsets"]["new_description"]["completed"],
                "n": item["subsets"]["new_description"]["n"],
                "threshold": GATE_NEW_DESCRIPTION,
                "pass": item["subsets"]["new_description"]["rate"] >= GATE_NEW_DESCRIPTION,
            },
            "reversed_paired": {
                "status": "withdrawn",
                "reason": "local-index true_raw cache; raw bytes not stored",
                "pass": None,
            },
        }

    mechanism = {}
    for seed in SEEDS:
        item = readings[(seed, "A-locked")]
        content = item["content"]
        content_pass = {}
        if content["applicable"]:
            for name, drop in content["drops"].items():
                content_pass[name] = {
                    "drop": drop,
                    "pass": drop >= GATE_CONTENT_DROP,
                }
            content_evaluable = not content["degenerate"]
        else:
            content_pass = {}
            content_evaluable = False
        mechanism[str(seed)] = {
            "selection": {
                "value": item["selection"]["top1"],
                "threshold": GATE_SELECTION,
                "pass": item["selection"]["top1"] >= GATE_SELECTION,
            },
            "gamma_raw": {
                "value": item["gamma"]["gamma_model_mean"],
                "threshold": GATE_GAMMA,
                "pass": item["gamma"]["gamma_model_mean"] >= GATE_GAMMA,
                "intent_reading": (
                    "passes as a generation-weight gate; copy_weight_mean is "
                    f"{item['gamma']['copy_weight_mean']:.4f}, so it does not support "
                    "'copy in use'"
                ),
            },
            "content": {
                "evaluable": content_evaluable,
                "true_rate": content.get("true_rate"),
                "drops": content_pass,
                "pass": (
                    content_evaluable and all(entry["pass"] for entry in content_pass.values())
                ),
            },
            "identity_rescue": {
                "equal": item["identity_rescue"][0],
                "n": item["identity_rescue"][1],
                "pass": item["identity_rescue"][0] == item["identity_rescue"][1] == 8,
            },
        }

    increment = report["contrast_A_minus_B"]
    self_prefix = {
        str(seed): {
            "A": readings[(seed, "A-locked")]["curriculum"]["self_prefix_updates"],
            "B": readings[(seed, "B-select")]["curriculum"]["self_prefix_updates"],
        }
        for seed in SEEDS
    }
    mechanism["increment"] = {
        "pooled_mean": increment["pooled_mean"],
        "bootstrap_lower": {
            key: stat["bootstrap"]["lower"] for key, stat in increment["per_seed"].items()
        },
        "pass": False,
        "reason": (
            "negative and confounded: the unlocked arm ran the accuracy-triggered "
            "self-prefix 127/121 updates with self-imitation labels, so A-B mixes the "
            "lock with a different curriculum target"
        ),
    }

    task_pass = all(
        gate["pass"]
        for seed_gates in task.values()
        for name, gate in seed_gates.items()
        if name != "reversed_paired"
    )
    mechanism_pass = (
        all(
            gate["pass"]
            for seed_gates in mechanism.values()
            for name, gate in seed_gates.items()
            if name in ("selection", "content", "identity_rescue")
        )
        and mechanism["increment"]["pass"]
    )
    verdict = "native_byte_task_failed"
    if not checks["precheck_passed"]:
        verdict = "engineering_failure"

    withdrawn = [
        "subset rates with a 240 denominator (15.1-style aggregation fault)",
        "content-necessity true side keyed by the local index cache",
        "counterfactual true side keyed by the local index cache",
        "selection-by-subset grouping keyed by the local index cache",
        "reverse-pair rate keyed by the local index cache (not rebuildable without stored raw)",
        "gamma read as a 'copy gate'; gamma is the generation weight",
    ]
    return {
        "schema": "taiji-binding-select-attribution-corrected-v1",
        "step": "N16.1.1-attribution",
        "date": "2026-11-03",
        "status": (
            "instrument_partial_fault_true_negative"
            if not any(name for name, ok in checks.items() if not ok)
            else "attribution_inconclusive"
        ),
        "checks": checks,
        "failed": [name for name, ok in checks.items() if not ok],
        "readings": {
            str(seed) + "-" + arm: readings[(seed, arm)] for seed in SEEDS for arm in ARMS
        },
        "gates": {"task": task, "mechanism": mechanism},
        "task_pass": task_pass,
        "mechanism_pass": mechanism_pass,
        "verdict": verdict,
        "instrument_faults": {
            "subset_denominators": (
                "the runner's summarize() divided every subset by 240 instead of its own "
                "n; corrected values are 0.05/0.0167 for new descriptions, not "
                "0.0125/0.0042"
            ),
            "index_cache": (
                "true_gen/flags were keyed by the world-local index (20 values for 240 "
                "rows); the content and counterfactual true sides are recomputed from "
                "the stored per-row scores with (world, index), and the reverse-pair "
                "rate is withdrawn"
            ),
            "gamma_direction": (
                "gamma multiplies the generated branch, so the 0.2 clamp is a generation "
                "floor; the copy weight at step300 is 0.1788/0.2529 in A and 0.9384/0.9608 "
                "in B, but B's alpha is null so its copy mass is zero.  The floor does not "
                "carry the score (removing it changes completion by 0.0/+0.0167)"
            ),
            "reversed_pair": (
                "stored raw bytes are absent, so the correct paired rate cannot be "
                "rebuilt without rerunning 240+240 deterministic generations; the reading "
                "is withdrawn, not replaced"
            ),
        },
        "contract_fault": {
            "self_prefix_labels": (
                "batch_objective passes the self-generated prefix as target_prefix to "
                "locked_loss/score_example, so labels become the model's own bytes; the "
                "curriculum is self-imitation, not correction.  A never triggered (0 "
                "updates); B triggered 127/121 updates.  The A-B contrast is therefore "
                "confounded and is not used as a clean mechanism test."
            ),
            "self_prefix_triggers": self_prefix,
            "affected_reading": "A-B increment",
        },
        "corrected_headline": {
            "overall": {str(seed): readings[(seed, "A-locked")]["overall"] for seed in SEEDS},
            "new_description": {
                str(seed): readings[(seed, "A-locked")]["subsets"]["new_description"]
                for seed in SEEDS
            },
            "unseen_output_name": {
                str(seed): readings[(seed, "A-locked")]["subsets"]["unseen_output_name"]
                for seed in SEEDS
            },
            "selection": {
                str(seed): readings[(seed, "A-locked")]["selection"]["top1"] for seed in SEEDS
            },
            "selection_new_description": {
                str(seed): readings[(seed, "A-locked")]["selection_by_subset"].get(
                    "new_description"
                )
                for seed in SEEDS
            },
            "content": {str(seed): readings[(seed, "A-locked")]["content"] for seed in SEEDS},
            "gamma": {str(seed): readings[(seed, "A-locked")]["gamma"] for seed in SEEDS},
            "reversed_accuracy": {
                str(seed): readings[(seed, "A-locked")]["reversed_accuracy"] for seed in SEEDS
            },
        },
        "withdrawn": withdrawn,
        "architecture_diagnosis": (
            "Three independent breaks remain: (1) the selection head's mean byte "
            "embedding gives 0/60 on new descriptions in both arms (seen ~0.43-0.61); "
            "(2) the locked arm's copy weight is only 0.18-0.25 and the entity start is "
            "still wrong (object_exact 0.25/0.09, first divergence at byte 3 in "
            "50%/63%); (3) the gamma floor is a generation floor, not a copy floor, so "
            "it cannot force the missing copy.  The unlocked arm retains a memorized "
            "generation shortcut (0.3375/0.2833 overall, 0 on unseen names) whose "
            "curriculum was self-imitation.  This is an interface/representation "
            "problem, not dose or seed."
        ),
        "next_step": {
            "single": (
                "N16.1.3 全局收束（只读：用本修正读数、弃用反序配对、登记仪器/合同"
                "缺陷），再由 owner 裁定是否准入 17 号逐步选择路由候选"
            ),
            "candidate_reference": (
                "plans/reference/DESIGN-POST-N16_SELECTION_ROUTING_20261103.md"
            ),
            "candidate_state": "design only; not launched",
            "why_not_launch": (
                "17 requires the 16.1.3 score/attribution calibration and an owner ruling; "
                "the current 1.1 card does not release 1.2 or any new training."
            ),
        },
        "cost": report["cost_detail"],
        "post_run_changes": {
            "script": "scripts/training/verify_taiji_binding_select_16_1_compare.py",
            "ran_sha256": report["source_hashes"].get(
                "scripts/training/verify_taiji_binding_select_16_1_compare.py"
            ),
            "now_sha256": sha256_file(
                ROOT / "scripts/training/verify_taiji_binding_select_16_1_compare.py"
            ),
            "changes": (
                "aggregation only: subset denominators, (world,index) keys for the true "
                "side, stored true raw for reverse pairing, selection rows carry flags, "
                "copy_weight_mean reported.  The training recipe is unchanged; the "
                "self-prefix label behavior is left frozen and annotated."
            ),
        },
        "price_deviation": {
            "frozen_price_cpu_seconds": PRICE_CPU_SECONDS,
            "actual_cpu_seconds": report["cost_detail"]["total_cpu_seconds"],
            "ratio_to_0c_price": report["cost_detail"]["total_cpu_seconds"] / PRICE_CPU_SECONDS,
            "pre_run_projection_cpu_seconds": PRE_RUN_PROJECTION,
            "actual_vs_pre_run_projection": report["cost_detail"]["total_cpu_seconds"]
            / PRE_RUN_PROJECTION,
            "attribution": (
                "0c did not price the per-batch teacher-forced curriculum probe (a second "
                "full forward over the same 28-step prefix; copy/joint ~16.5 CPU vs 8.3 "
                "priced) nor the extra reversed/counterfactual/intervention/sensitivity "
                "generations.  The pre-run projection was ~14,300 CPU; actual 13,438.1 is "
                "94%.  No rerun, no duplicate config, wall==cpu, single thread, under the "
                "20,000/30,000 observation line."
            ),
        },
        "limits": (
            "Read-only correction of the frozen report: 0 updates, 0 forward, 0 confirm.  "
            "It does not re-run the comparison, change the eligible dose or release 1.2."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="attribution", choices=("attribution",))
    parser.parse_args()
    report = json.loads((ROOT / REPORT).read_text(encoding="utf-8"))
    result = audit(report)
    Path(ROOT / OUT).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "taiji-binding-select-manifest-v1",
        "step": "N16.1.1-attribution",
        "status": result["status"],
        "report": OUT,
        "source_report": REPORT,
        "source_report_sha256": sha256_file(ROOT / REPORT),
        "checks": result["checks"],
        "failed": result["failed"],
        "verdict": result["verdict"],
        "task_pass": result["task_pass"],
        "mechanism_pass": result["mechanism_pass"],
        "withdrawn": result["withdrawn"],
        "instrument_faults": result["instrument_faults"],
        "contract_fault": result["contract_fault"],
        "next_step": result["next_step"],
        "cost": {"cpu_seconds": 0.0, "wall_seconds": 0.0, "adam_calls": 0, "forward_calls": 0},
        "limits": result["limits"],
    }
    Path(ROOT / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"attribution: {result['status']}")
    print(f"  failed: {result['failed']}")
    print(f"  task_pass {result['task_pass']}, mechanism_pass {result['mechanism_pass']}")
    print(f"  next: {result['next_step']['single']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Machine-computed closeout arithmetic for U.1r1 (no hand-carried means or ratios).

Reads three published reports and emits the comparison table quoted by the ledger.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_u1r1_position_20261116 as r1

BASE = ROOT / "reports/taiji_u2_u1_unit_read_run_20261116.json"
REV = ROOT / "reports/taiji_u2_u1r1_position_run_20261116.json"
PROBE = ROOT / "reports/taiji_u2_u1r1_attention_collapse_20261116.json"
REBUILD = ROOT / "reports/taiji_u2_u1r1_rebuild_20261116.json"
REPORT = ROOT / "reports/taiji_u2_u1r1_closeout_20261116.json"
FIELDS = (
    "before_fit_accuracy",
    "fit_accuracy",
    "before_train_accuracy",
    "train_accuracy",
    "rgb_zero_agreement",
    "fixed_question_agreement",
    "attention_entropy_mean",
    "final_loss",
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def face(report: dict[str, Any]) -> dict[str, Any]:
    arms = report["arms"]
    out: dict[str, Any] = {"seeds": [int(a["seed"]) for a in arms]}
    for field in FIELDS:
        values = [float(a[field]) for a in arms if a.get(field) is not None]
        if not values:
            continue
        out[field] = {
            "per_seed": values,
            "mean": round(mean(values), 6),
            "min": min(values),
            "max": max(values),
        }
    out["fit_distinct_per_seed"] = [int(a["fit_distinct"]) for a in arms]
    out["train_distinct_per_seed"] = [int(a["train_distinct"]) for a in arms]
    out["cpu_seconds_total"] = round(sum(float(a["cpu_seconds"]) for a in arms), 4)
    out["cost"] = report["cost"]
    out["verdict"] = report["verdict"]
    out["seed_pass"] = report["seed_pass"]
    return out


def main() -> int:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    rev = json.loads(REV.read_text(encoding="utf-8"))
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    rebuild = json.loads(REBUILD.read_text(encoding="utf-8"))
    recover = json.loads(r1.RECOVER_REPORT.read_text(encoding="utf-8"))
    verify = json.loads(r1.VERIFY_REPORT.read_text(encoding="utf-8"))
    counts = {row["partition"]: int(row["count"]) for row in rebuild["dataset"]}
    hit_lengths = {len(arm["fit_hits"]) for arm in base["arms"] + rev["arms"]}
    if hit_lengths != {counts["fit"]}:
        raise ValueError("fit_denominator_mismatch_between_runs_and_rebuild")
    contract = json.loads(r1.CONTRACT.read_text(encoding="utf-8"))
    criteria = contract["criteria"]
    b, r = face(base), face(rev)
    deltas = {
        field: {
            "mean_delta_rev_minus_base": round(r[field]["mean"] - b[field]["mean"], 6),
            "per_seed_delta": [
                round(rv - bv, 6)
                for rv, bv in zip(r[field]["per_seed"], b[field]["per_seed"], strict=True)
            ],
        }
        for field in FIELDS
        if field in b and field in r
    }
    margins = {
        "fit_gate": float(criteria["fit_gate"]),
        "train_gate": float(criteria["train_gate"]),
        "distinct_min": int(criteria["distinct_min"]),
        "control_agreement_max": float(criteria["control_agreement_max"]),
        "best_fit_accuracy": r["fit_accuracy"]["max"],
        "best_train_accuracy": r["train_accuracy"]["max"],
        "fit_shortfall_at_best_seed": round(
            float(criteria["fit_gate"]) - r["fit_accuracy"]["max"], 6
        ),
        "train_shortfall_at_best_seed": round(
            float(criteria["train_gate"]) - r["train_accuracy"]["max"], 6
        ),
        "seeds_meeting_fit_gate": sum(
            1 for v in r["fit_accuracy"]["per_seed"] if v >= float(criteria["fit_gate"])
        ),
        "seeds_meeting_train_gate": sum(
            1 for v in r["train_accuracy"]["per_seed"] if v >= float(criteria["train_gate"])
        ),
        "seeds_meeting_distinct_min": sum(
            1 for v in r["fit_distinct_per_seed"] if v >= int(criteria["distinct_min"])
        ),
        "seeds_meeting_both_controls_strictly_below_max": sum(
            1
            for rgb, fixed in zip(
                r["rgb_zero_agreement"]["per_seed"],
                r["fixed_question_agreement"]["per_seed"],
                strict=True,
            )
            if rgb < float(criteria["control_agreement_max"])
            and fixed < float(criteria["control_agreement_max"])
        ),
        "chance_cross_entropy": round(math.log(len(r1.LABELS)), 6),
        "seeds_with_final_loss_above_chance": sum(
            1 for v in r["final_loss"]["per_seed"] if v > math.log(len(r1.LABELS))
        ),
        "seeds_with_fit_worse_than_before": sum(
            1
            for after, before in zip(
                r["fit_accuracy"]["per_seed"],
                r["before_fit_accuracy"]["per_seed"],
                strict=True,
            )
            if after < before
        ),
    }
    probe_rows = []
    for row in probe["seeds"]:
        arms = row["arms"]
        credit = row["task_credit"]["credit"]
        head = credit["position_head.weight"]
        probe_rows.append(
            {
                "seed": row["seed"],
                "question_distinct": row["question_distinct"],
                "trained_logit_range_mean": arms["trained"]["logit_range_mean"],
                "init_logit_range_mean": arms["init"]["logit_range_mean"],
                "forced_logit_range_mean": arms["trained_forced_query"]["logit_range_mean"],
                "trained_entropy_fraction_of_ceiling": arms["trained"][
                    "entropy_fraction_of_uniform_ceiling"
                ],
                "forced_entropy_fraction_of_ceiling": arms["trained_forced_query"][
                    "entropy_fraction_of_uniform_ceiling"
                ],
                "trained_effective_columns_mean": arms["trained"]["effective_columns_mean"],
                "forced_effective_columns_mean": arms["trained_forced_query"][
                    "effective_columns_mean"
                ],
                "trained_max_attention_mean": arms["trained"]["max_attention_mean"],
                "forced_max_attention_mean": arms["trained_forced_query"]["max_attention_mean"],
                "uniform_max_attention": round(1.0 / probe["columns_T"], 6),
                "w_q_frobenius_init": row["w_q_frobenius_init"],
                "w_q_frobenius_trained": row["w_q_frobenius_trained"],
                "w_q_norm_relative_change": round(
                    (row["w_q_frobenius_trained"] - row["w_q_frobenius_init"])
                    / row["w_q_frobenius_init"],
                    6,
                ),
                "w_q_forced_over_trained_norm": round(
                    row["w_q_frobenius_forced"] / row["w_q_frobenius_trained"], 4
                ),
                "credit_w_q": credit["W_q"],
                "credit_position_head": head,
                "credit_units_p": credit["units.p"],
                "credit_ratio_w_q_over_position_head": round(credit["W_q"] / head, 6),
                "credit_ratio_w_q_over_units_p": round(credit["W_q"] / credit["units.p"], 6),
                "credit_head_over_w_q": round(head / credit["W_q"], 4),
                "credit_units_p_over_w_q": round(credit["units.p"] / credit["W_q"], 4),
                "fit_accuracy_trained": arms["trained"]["fit_accuracy"],
                "fit_accuracy_forced_query": arms["trained_forced_query"]["fit_accuracy"],
                "fit_accuracy_zero_query": arms["trained_zero_query"]["fit_accuracy"],
                "fit_delta_zero_query_minus_trained": round(
                    arms["trained_zero_query"]["fit_accuracy"] - arms["trained"]["fit_accuracy"], 6
                ),
                "controls": row["controls"],
            }
        )
    derived = {
        "cost_delta_rev_minus_base": {
            "cpu": round(float(rev["cost"]["cpu"]) - float(base["cost"]["cpu"]), 4),
            "wall": round(float(rev["cost"]["wall"]) - float(base["cost"]["wall"]), 4),
        },
        "credit_head_over_w_q_min": min(row["credit_head_over_w_q"] for row in probe_rows),
        "credit_head_over_w_q_max": max(row["credit_head_over_w_q"] for row in probe_rows),
        "w_q_forced_over_trained_norm_min": min(
            row["w_q_forced_over_trained_norm"] for row in probe_rows
        ),
        "w_q_forced_over_trained_norm_max": max(
            row["w_q_forced_over_trained_norm"] for row in probe_rows
        ),
        "w_q_norm_relative_change_max_abs": max(
            abs(row["w_q_norm_relative_change"]) for row in probe_rows
        ),
        "trained_logit_range_min": min(row["trained_logit_range_mean"] for row in probe_rows),
        "trained_logit_range_max": max(row["trained_logit_range_mean"] for row in probe_rows),
        "forced_logit_range_min": min(row["forced_logit_range_mean"] for row in probe_rows),
        "forced_logit_range_max": max(row["forced_logit_range_mean"] for row in probe_rows),
        "trained_entropy_fraction_min": min(
            row["trained_entropy_fraction_of_ceiling"] for row in probe_rows
        ),
        "trained_entropy_fraction_max": max(
            row["trained_entropy_fraction_of_ceiling"] for row in probe_rows
        ),
        "trained_effective_columns_min": min(
            row["trained_effective_columns_mean"] for row in probe_rows
        ),
        "forced_effective_columns_min": min(
            row["forced_effective_columns_mean"] for row in probe_rows
        ),
        "forced_effective_columns_max": max(
            row["forced_effective_columns_mean"] for row in probe_rows
        ),
        "probe_controls_true": sum(
            1 for row in probe_rows for value in row["controls"].values() if value
        ),
        "probe_controls_total": sum(len(row["controls"]) for row in probe_rows),
        "zero_query_fit_delta_per_seed": [
            row["fit_delta_zero_query_minus_trained"] for row in probe_rows
        ],
        "zero_query_never_worse_seeds": sum(
            1 for row in probe_rows if row["fit_delta_zero_query_minus_trained"] >= 0
        ),
        "zero_query_strictly_better_seeds": sum(
            1 for row in probe_rows if row["fit_delta_zero_query_minus_trained"] > 0
        ),
        "zero_query_fit_delta_max": max(
            row["fit_delta_zero_query_minus_trained"] for row in probe_rows
        ),
    }
    payload = {
        "schema": "taiji-u2-u1r1-closeout-v1",
        "card": "22.5.U.1r1",
        "phase": "closeout_arithmetic",
        "inputs": {
            "base_run": {"path": str(BASE.relative_to(ROOT)), "verdict": base["verdict"]},
            "revision_run": {"path": str(REV.relative_to(ROOT)), "verdict": rev["verdict"]},
            "mechanism_probe": {
                "path": str(PROBE.relative_to(ROOT)),
                "classification": probe["classification"],
            },
        },
        "criteria_sha256": contract["criteria_sha256"],
        "same_face": {
            "seeds_base": b["seeds"],
            "seeds_revision": r["seeds"],
            "updates_per_seed": int(criteria["updates_per_seed"]),
            "learning_rate": float(criteria["learning_rate"]),
            "fit_denominator": counts["fit"],
            "train_denominator": counts["train"],
            "label_count": len(r1.LABELS),
        },
        "base_mean_pool": b,
        "revision_attention": r,
        "deltas_revision_minus_base": deltas,
        "gate_margins": margins,
        "mechanism_probe_rows": probe_rows,
        "derived_ranges": derived,
        "probe_cost": probe["cost"],
        "recover": {
            "verdict": recover["verdict"],
            "checks_true": sum(1 for v in recover["checks"].values() if v),
            "checks_total": len(recover["checks"]),
            "uncovered": recover["uncovered"],
            "cost": recover["cost"],
        },
        "verify": {
            "verdict": verify["verdict"],
            "checks_true": sum(1 for v in verify["checks"].values() if v),
            "checks_total": len(verify["checks"]),
            "uncovered": verify["uncovered"],
            "cost": verify["cost"],
        },
        "criteria_sha256_all_match": all(
            report["criteria_sha256"] == contract["criteria_sha256"]
            for report in (rev, probe, rebuild, recover, verify)
        ),
        "base_criteria_sha256": base.get("criteria_sha256"),
        "base_is_a_different_card": base.get("card"),
        "revision_cost": rev["cost"],
        "cap": {
            "cpu": float(contract["plan"]["cap_cpu"]),
            "wall": float(contract["plan"]["cap_wall"]),
        },
        "capability_scope": "U.1 read prerequisite only; no correction/retention/parent claim",
        "discipline": {
            "optimizer_steps": 0,
            "training_updates": 0,
            "task_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
        },
    }
    r1.write_report(REPORT, payload)
    print(
        "CLOSEOUT_ARITHMETIC"
        f" fit_mean_base={b['fit_accuracy']['mean']}"
        f" fit_mean_rev={r['fit_accuracy']['mean']}"
        f" delta={deltas['fit_accuracy']['mean_delta_rev_minus_base']}"
        f" class={probe['classification']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

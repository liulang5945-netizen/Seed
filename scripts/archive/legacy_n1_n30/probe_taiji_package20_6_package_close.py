"""20.6: close package 20.  Bookkeeping only, no model and no optimizer.

The package plan already fixed how this ends: 20.4 compares two seeds x two
arms, 20.5 opens only if 20.4 selects a single identity, and 20.6 closes the
package whatever the result, with no automatic 20.x patch chain.  This runner
therefore does exactly four things:

* amend the 20.4d report with the corrected eight-cell variance decomposition
  (the original block used only the four diagonal cells, where the model seed
  and the evaluation batch seed coincide, so it charged seed variance to the
  arm; the pre-registered criteria and verdict are left untouched);
* read every package-20 evidence file and write the package close report,
  including the section 10 branch-by-branch check, the three attribution
  classes plus the competing explanations, the cost ledger, and the capability
  ledger;
* write the close manifest;
* append one fsync'd progress row per stage.

Nothing is trained, no optimizer or Adam step is taken, no probe is added, no
evaluation is rerun, and no confirm world is read.  The only writes are the
close report, its manifest, the progress log, and the 20.4d variance
amendment.  The report uses write-then-rename, and both runner hashes are
recorded (``runner_sha256`` produced the bytes; ``submitted_runner_sha256`` is
filled after the commit).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MANIFESTS = ROOT / "plans/manifests"
REPORT = REPORTS / "taiji_package20_6_package_close_20261107.json"
MANIFEST = MANIFESTS / "taiji_package20_6_package_close_20261107.json"
PROGRESS = REPORTS / "taiji_package20_6_package_close_20261107_progress.jsonl"
R4D = REPORTS / "taiji_package20_4d_shared_eval_20261106.json"
M4D = MANIFESTS / "taiji_package20_4d_shared_eval_20261106.json"

CPU_CAP = 1500.0
CPU_GUARD = 1480.0
WALL_CAP = 1800.0

START_CPU = time.process_time()
START_WALL = time.time()

EVIDENCE = {
    "20_0_plan_review": REPORTS / "taiji_package20_plan_review_20261105.json",
    "20_1": REPORTS / "taiji_package20_1_rule_learning_fixture_graph_20261105.json",
    "20_2": REPORTS / "taiji_package20_2_credit_recovery_20261105.json",
    "20_2r": REPORTS / "taiji_package20_2r_f_leverage_20261105.json",
    "20_2s": REPORTS / "taiji_package20_2s_f_information_20261105.json",
    "20_3a": REPORTS / "taiji_package20_3a_addressed_store_20261105.json",
    "20_3b": REPORTS / "taiji_package20_3b_f_authority_20261105.json",
    "20_4": REPORTS / "taiji_package20_4_pricing_freeze_20261105.json",
    "20_4a": REPORTS / "taiji_package20_4a_unit_cost_20261105.json",
    "20_4b": REPORTS / "taiji_package20_4b_graph_optimization_20261105.json",
    "20_4c": REPORTS / "taiji_package20_4c_formal_comparison_20261105.json",
    "20_4c_r2": REPORTS / "taiji_package20_4c_r2_formal_comparison_20261105.json",
    "20_4d": R4D,
    "20_4d_preregistration": REPORTS / "taiji_package20_4d_preregistration_20261106.json",
    "20_2t_manifest": MANIFESTS / "taiji_package20_2t_write_granularity_20261105.json",
}


class BudgetStop(RuntimeError):
    """Raised when the CPU or wall guard fires."""

    def __init__(self, verdict: str, detail: str) -> None:
        super().__init__(detail)
        self.verdict = verdict
        self.detail = detail


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def guard(stage: str) -> None:
    cpu, wall = cost()
    if cpu > CPU_GUARD:
        raise BudgetStop(
            "cpu_cap_reached", f"CPU cap reached at {stage}: {cpu:.1f} of {CPU_CAP:.0f}"
        )
    if wall > WALL_CAP:
        raise BudgetStop("wall_cap_reached", f"wall cap reached at {stage}: {wall:.1f}")


def append_progress(row: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    return sha_bytes(Path(path).read_bytes())


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 9),
        "min": round(min(values), 9),
        "max": round(max(values), 9),
    }


def variance_decomposition(report: dict) -> dict:
    """Two-way split of the full eight-cell checkerboard.

    model = arm x training seed (four models, nested in arm);
    batch = evaluation seed (20261105 / 20261106);
    one observation per cell.  The original block used only the diagonal four
    cells, where model seed == batch seed, so it could not separate the two.
    """
    cells: list[dict] = []
    for entry in report["checkerboard"].values():
        arm = entry["arm"]
        seed = int(entry["seed"])
        for batch, reading in entry["by_batch"].items():
            value = reading["decision_accuracy"]
            if value is None:
                continue
            cells.append({"arm": arm, "model_seed": seed, "batch_seed": int(batch), "value": value})
    values = [cell["value"] for cell in cells]
    grand = sum(values) / len(values)
    arms = sorted({cell["arm"] for cell in cells})
    models = sorted({(cell["arm"], cell["model_seed"]) for cell in cells})
    batches = sorted({cell["batch_seed"] for cell in cells})
    arm_means = {arm: sum(c["value"] for c in cells if c["arm"] == arm) / 4.0 for arm in arms}
    model_means = {
        model: sum(c["value"] for c in cells if (c["arm"], c["model_seed"]) == model) / 2.0
        for model in models
    }
    batch_means = {
        batch: sum(c["value"] for c in cells if c["batch_seed"] == batch) / 4.0 for batch in batches
    }
    ss_total = sum((value - grand) ** 2 for value in values)
    ss_arm = 4.0 * sum((arm_means[arm] - grand) ** 2 for arm in arms)
    ss_model = 2.0 * sum((model_means[model] - arm_means[model[0]]) ** 2 for model in models)
    ss_batch = 4.0 * sum((batch_means[batch] - grand) ** 2 for batch in batches)
    ss_interaction = ss_total - ss_arm - ss_model - ss_batch
    share = {
        "model_within_arm": round(ss_model / ss_total, 6),
        "arm": round(ss_arm / ss_total, 6),
        "batch": round(ss_batch / ss_total, 6),
        "interaction": round(ss_interaction / ss_total, 6),
    }
    return {
        "cells": len(cells),
        "grand_mean": round(grand, 9),
        "arm_means": {key: round(value, 9) for key, value in arm_means.items()},
        "model_means": {
            f"{arm}|{seed}": round(value, 9) for (arm, seed), value in model_means.items()
        },
        "batch_means": {str(key): round(value, 9) for key, value in batch_means.items()},
        "sum_of_squares": {
            "total": round(ss_total, 9),
            "model_within_arm": round(ss_model, 9),
            "arm": round(ss_arm, 9),
            "batch": round(ss_batch, 9),
            "interaction": round(ss_interaction, 9),
        },
        "share": share,
    }


def amend_20_4d(runner_sha: str) -> dict:
    existing = load(M4D)
    recorded = existing.get("amendment", {}).get("report_sha256_before")
    report_sha_before = recorded or file_sha(R4D)
    report = load(R4D)
    decomposition = variance_decomposition(report)
    report["variance_correction"] = {
        "status": "amended",
        "amended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "amended_evidence": "20_6",
        "amended_by_runner_sha256": runner_sha,
        "report_sha256_before": report_sha_before,
        "reason": (
            "the original variance block used only the four diagonal cells, where the "
            "evaluation batch seed equals the model seed by construction; a two-way split "
            "of those four cells therefore charges model-seed variance to the arm"
        ),
        "original_block": (
            "variance.decision_accuracy stays in this file unchanged for audit; it is "
            "confounded and must not be used together with or against the block below"
        ),
        "method": (
            "two-way sum-of-squares on the full eight-cell checkerboard: arm main effect, "
            "model seed within arm, evaluation batch main effect, residual interaction; "
            "one observation per cell, no model fitted"
        ),
        "decomposition": decomposition,
        "effect_on_preregistered_criteria": (
            "none: the criteria were frozen before the first world was scored and the "
            "verdict arm_indistinguishable rests on the shared-batch paired deltas, not on "
            "this table"
        ),
        "verdict_unchanged": True,
    }
    write_json(R4D, report)
    manifest = load(M4D)
    manifest["report_sha256"] = file_sha(R4D)
    manifest["amendment"] = {
        "evidence": "20_6",
        "variance_correction": True,
        "runner_sha256": runner_sha,
        "report_sha256_before": report_sha_before,
        "criteria_unchanged": True,
        "verdict_unchanged": True,
    }
    write_json(M4D, manifest)
    return {
        "decomposition": decomposition,
        "report_sha256_before": report_sha_before,
        "report_sha256_after": file_sha(R4D),
    }


def cost_rows() -> tuple[list[dict], dict]:
    review = load(EVIDENCE["20_0_plan_review"])
    r1 = load(EVIDENCE["20_1"])
    r2 = load(EVIDENCE["20_2"])
    r2r = load(EVIDENCE["20_2r"])
    r2s = load(EVIDENCE["20_2s"])
    r3a = load(EVIDENCE["20_3a"])
    r3b = load(EVIDENCE["20_3b"])
    r4 = load(EVIDENCE["20_4"])
    r4a = load(EVIDENCE["20_4a"])
    r4b = load(EVIDENCE["20_4b"])
    r4c = load(EVIDENCE["20_4c"])
    r4cr2 = load(EVIDENCE["20_4c_r2"])
    r4d = load(EVIDENCE["20_4d"])

    rows: list[dict] = []

    def add(
        stage: str,
        kind: str,
        date: str,
        cpu,
        wall,
        updates,
        adam,
        probes: int,
        verdict: str,
        source: str,
        note: str | None = None,
        estimate: bool = False,
        lower_bound=None,
    ) -> None:
        rows.append(
            {
                "stage": stage,
                "kind": kind,
                "date": date,
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "updates": updates,
                "adam_steps": adam,
                "f_probe_evaluations": probes,
                "verdict": verdict,
                "source": source,
                "note": note,
                "estimate": estimate,
                "cpu_seconds_lower_bound": lower_bound,
            }
        )

    add(
        "20.0",
        "planned",
        "2026-11-05",
        review.get("static_audit_cpu_seconds"),
        None,
        0,
        0,
        0,
        "plan_review_passed",
        rel(EVIDENCE["20_0_plan_review"]),
        "static audit only; wall not recorded",
    )
    add(
        "20.1",
        "planned",
        "2026-11-05",
        r1["cost"]["cpu_seconds"],
        r1["cost"]["wall_seconds"],
        0,
        0,
        0,
        r1.get("verdict"),
        rel(EVIDENCE["20_1"]),
    )
    add(
        "20.2",
        "planned",
        "2026-11-05",
        r2["cost"]["cpu_seconds"],
        r2["cost"]["wall_seconds"],
        0,
        0,
        0,
        r2.get("verdict"),
        rel(EVIDENCE["20_2"]),
        "0 Adam steps; 6 backward calls",
    )
    add(
        "20.2r",
        "diagnostic",
        "2026-11-05",
        r2r["cost"]["cpu_seconds"],
        r2r["cost"]["wall_seconds"],
        0,
        0,
        110,
        r2r.get("verdict"),
        rel(EVIDENCE["20_2r"]),
        "22 correction turns x 5 counterfactual arms; 23 backward calls",
    )
    add(
        "20.2s",
        "diagnostic",
        "2026-11-05",
        r2s["cost"]["cpu_seconds"],
        r2s["cost"]["wall_seconds"],
        0,
        0,
        0,
        r2s.get("verdict"),
        rel(EVIDENCE["20_2s"]),
        "closed-form 22-fold readout; not a counterfactual F probe",
    )
    add(
        "20.2s_failed_attempt",
        "diagnostic",
        "2026-11-05",
        None,
        None,
        0,
        0,
        0,
        "discarded",
        rel(EVIDENCE["20_2s"]),
        "exact CPU missing; the report records a lower bound of 25.75 CPU",
        lower_bound=25.75,
    )
    add(
        "20.2t",
        "diagnostic",
        "2026-11-05",
        None,
        None,
        0,
        0,
        0,
        "design_only",
        rel(EVIDENCE["20_2t_manifest"]),
        "manifest only; no report and no runner, so CPU is missing, not zero",
    )
    add(
        "20.3a",
        "diagnostic",
        "2026-11-05",
        r3a["cost"]["cpu_seconds"],
        r3a["cost"]["wall_seconds"],
        0,
        0,
        27,
        r3a.get("verdict"),
        rel(EVIDENCE["20_3a"]),
        "5 counterfactual arms on one turn plus a 22-turn leave-one-out information probe",
    )
    add(
        "20.3b",
        "diagnostic",
        "2026-11-05",
        r3b["cost"]["cpu_seconds"],
        r3b["cost"]["wall_seconds"],
        0,
        0,
        14,
        r3b.get("verdict"),
        rel(EVIDENCE["20_3b"]),
        "5 gain levels + 4 ridge values + 4 rank settings + 1 oracle replacement",
    )
    add(
        "20.3b_discarded_attempts",
        "diagnostic",
        "2026-11-05",
        None,
        None,
        None,
        0,
        0,
        "discarded",
        rel(EVIDENCE["20_3b"]),
        "two discarded attempts; their CPU is not recorded (missing, not zero)",
    )
    add(
        "20.4",
        "planned",
        "2026-11-05",
        r4["cost"]["cpu_seconds"],
        r4["cost"]["wall_seconds"],
        12,
        12,
        0,
        r4.get("verdict"),
        rel(EVIDENCE["20_4"]),
        "6 measured pricing updates + 3 preflight saved + 3 preflight resumed",
    )
    add(
        "20.4_pre_probe_calibration",
        "planned",
        "2026-11-05",
        620.0,
        None,
        3,
        3,
        0,
        "calibration",
        rel(EVIDENCE["20_4"]),
        "declared estimate, not a measured process cost; excluded from measured total",
        estimate=True,
    )
    add(
        "20.4a",
        "diagnostic",
        "2026-11-05",
        r4a["cost"]["total_cpu_seconds"],
        r4a["cost"]["wall_seconds"],
        20,
        20,
        0,
        r4a.get("verdict"),
        rel(EVIDENCE["20_4a"]),
        "total 784.188 = 727.891 earlier pass + 56.297 this pass; earlier wall missing",
    )
    add(
        "20.4b",
        "diagnostic",
        "2026-11-05",
        r4b["cost"]["cpu_seconds"],
        r4b["cost"]["wall_seconds"],
        6,
        6,
        0,
        r4b.get("verdict"),
        rel(EVIDENCE["20_4b"]),
        "6 calibration updates (1 equivalence reference + 5 timed)",
    )
    add(
        "20.4b_engineering_reruns",
        "diagnostic",
        "2026-11-05",
        1700.0,
        None,
        None,
        None,
        0,
        "engineering_reruns",
        "plans/active/roadmap/03_CURRENT_EXECUTION.md",
        "about 1,700 CPU declared in the execution log; per-run exact values missing",
        estimate=True,
    )
    add(
        "20.4c",
        "formal",
        "2026-11-05",
        r4c["cost"]["cpu_seconds"],
        r4c["cost"]["wall_seconds"],
        25,
        25,
        5,
        r4c.get("verdict"),
        rel(EVIDENCE["20_4c"]),
        "aborted on the absolute memory floor after 25 updates; readings kept as history",
    )
    add(
        "20.4c_r2_pre_restart",
        "formal",
        "2026-11-06",
        r4cr2["cost"]["prior_breakdown"]["20_4c_r2_pre_restart"],
        None,
        275,
        275,
        0,
        "carried_forward",
        rel(EVIDENCE["20_4c_r2"]),
        "CF1|20261105 complete + B_EH|20261105 at 75; wall missing",
    )
    add(
        "20.4c_r2_post_restart",
        "formal",
        "2026-11-06",
        r4cr2["cost"]["cpu_seconds"],
        r4cr2["cost"]["wall_seconds"],
        525,
        525,
        80,
        r4cr2.get("verdict"),
        rel(EVIDENCE["20_4c_r2"]),
        "B_EH|20261105 125 + CF1|20261106 200 + B_EH|20261106 200; 4 probe steps x 5 arms x 4 runs",
    )
    add(
        "20.4d",
        "diagnostic",
        "2026-11-06",
        r4d["cost"]["cpu_seconds"],
        r4d["cost"]["wall_seconds"],
        0,
        0,
        0,
        r4d.get("verdict_label"),
        rel(EVIDENCE["20_4d"]),
        "8-cell cross evaluation, 0 training, 0 optimizer step",
    )

    measured = sum(
        row["cpu_seconds"] for row in rows if row["cpu_seconds"] is not None and not row["estimate"]
    )
    estimates = sum(
        row["cpu_seconds"] for row in rows if row["cpu_seconds"] is not None and row["estimate"]
    )
    missing = [row["stage"] for row in rows if row["cpu_seconds"] is None]
    lower_bound = measured + sum(row["cpu_seconds_lower_bound"] or 0.0 for row in rows)
    walls = [row["wall_seconds"] for row in rows if row["wall_seconds"] is not None]
    wall_missing = [row["stage"] for row in rows if row["wall_seconds"] is None]
    totals = {
        "measured_cpu_seconds": round(measured, 3),
        "declared_estimate_cpu_seconds": round(estimates, 3),
        "lower_bound_including_missing_lower_bounds": round(lower_bound, 3),
        "measured_plus_estimates": round(measured + estimates, 3),
        "missing_cpu_stages": missing,
        "measured_wall_seconds": round(sum(walls), 3),
        "wall_missing_stages": wall_missing,
        "proposal_b_cpu_seconds": 24000.0,
        "magnitude_gate_cpu_seconds": 48000.0,
        "measured_vs_b": round(measured / 24000.0, 4),
        "measured_vs_gate": round(measured / 48000.0, 4),
        "measured_minus_gate": round(measured - 48000.0, 3),
        "measured_plus_estimates_vs_gate": round((measured + estimates) / 48000.0, 4),
        "measured_plus_estimates_minus_gate": round(measured + estimates - 48000.0, 3),
    }
    return rows, totals


def section10_branch_check() -> list[dict]:
    return [
        {
            "branch": "A高于强捷径，A/B接近，纠正/保持/恢复成立",
            "selected": False,
            "evidence": (
                "best cell decision accuracy is 0.240196 (CF1|20261106 on batch 20261105); "
                "all eight cells have full-byte accuracy 0.0; template bits are 0.0 except "
                "CF1|20261106 0.007541.  This is far below a strong shortcut, not close to it"
            ),
        },
        {
            "branch": "A/B都低于强捷径，信息和信用门已过",
            "selected": True,
            "evidence": (
                "both arms are far below the strong-shortcut reference; the task-side "
                "information gate (five anchors identify the rule uniquely, 20.1) and the "
                "credit gate (dL/dF norm 0.0688, share 0.00447, 20.2) passed; 20.3a's "
                "readout-information gate failure is itself representation evidence, not "
                "task information shortage"
            ),
        },
        {
            "branch": "A明显优于独立B，cut/permute退化，保持与独立确认成立",
            "selected": False,
            "evidence": (
                "the shared-batch paired delta reverses sign across seeds: -0.202206 for "
                "model seed 20261105 and +0.146078 for 20261106; pooled -0.028064 with CI "
                "[-0.064706, +0.010539]; no confirm was released"
            ),
        },
        {
            "branch": "只有teacher-forcing好、持续轨迹差",
            "selected": False,
            "evidence": (
                "20.3b replaced F with oracle F and still got 0/22 decisions and 0 byte "
                "difference; the teacher-forced face is not good either, so this branch "
                "does not describe the result"
            ),
        },
        {
            "branch": "source/时间/反馈泄漏或保存不完整",
            "selected": False,
            "evidence": (
                "20.1 static leak probe found 0 forbidden imports; 20.4c_r2 resume_verified "
                "ok for CF1 and B_EH at step 26, all checkpoint shas match, determinism 16/16"
            ),
        },
        {
            "branch": "未完成工作即触价/时间/信息上限",
            "selected": False,
            "evidence": (
                "20.4c_r2 completed all 800 formal updates within the authorized 52,000 CPU "
                "(42,831.453 charged); 20.4d completed 8/8 cells within 6,000 CPU.  The "
                "package total exceeding the 48,000 magnitude gate is a cost fact reported "
                "in the ledger, not a work-stopping budget stop"
            ),
        },
    ]


def attribution() -> dict:
    return {
        "representation": {
            "established": [
                (
                    "per-query correction is not represented at window level: 20.2s found "
                    "only 2 distinct F vectors across 22 correction surfaces; 20.2t's write "
                    "audit found writes only on release (10 rows), no per-turn write"
                ),
                (
                    "the content-addressed store makes 22 distinct F vectors but its read "
                    "keys are near-collinear: median top-1 cosine 0.9850, median top1-top2 "
                    "margin 0.00317, 22/22 below 0.05; F leverage 8.7e-6 and byte diff 0 "
                    "(20.3a)"
                ),
                (
                    "the raw F state does carry window-level information: a closed-form "
                    "readout reaches 15/22 (0.6818) on correction surfaces (20.2s), but the "
                    "window-identity control also matches, so the information is not "
                    "query-specific and the verdict there is inconclusive"
                ),
            ],
            "not_established": [
                "whether a per-query separable write/read representation would gain authority (not tested)",
                "whether a wider or differently factored F state would help (not tested)",
                "that the representation cannot work (not proven; do not read the negative as impossibility)",
            ],
        },
        "composition": {
            "established": [
                (
                    "credit reaches F: dL/dF norm 0.0688, share 0.00447, F influences loss; "
                    "save/load and crash recovery pass (20.2)"
                ),
                (
                    "the current composition writes only the 10 feedback rows at two "
                    "releases, so 22 correction surfaces collapse to 2 F vectors (20.2s/20.2t)"
                ),
                (
                    "the CF1-vs-B_EH switch has no consistent effect: pooled delta -0.028064, "
                    "CI [-0.064706, +0.010539], sign reversal across seeds (20.4d)"
                ),
                (
                    "even a perfect F replacement does not change the output: oracle F gives "
                    "0/22 decisions and 0 byte difference (20.3b)"
                ),
            ],
            "not_established": [
                "whether per-turn or per-query composition would help (not tested)",
                "whether the v1-to-v2 version correction is learnable with another write schedule (not tested)",
                "that composition is the only bottleneck (not proven)",
            ],
        },
        "optimization": {
            "established": [
                (
                    "200 updates x 4 runs completed; dev decision accuracy is 0.022304, "
                    "0.210539, 0.237500, 0.055392 against a 0.0 baseline; final cumulative "
                    "training readings are 0.035588, 0.096765, 0.086520, 0.050833"
                ),
                (
                    "F influence under training is unstable and seed-dependent: CF1|20261105 "
                    "leverage 0.025847 -> 0.005852 -> 0.001595 -> 0.019882; CF1|20261106 "
                    "0.024830 -> 0.001313 -> 3.39e-5 -> 1.49e-6 with byte diff 0 at step 200"
                ),
                "full-byte accuracy is 0.0 in all eight cells; template bits are near 0",
                (
                    "engineering discipline held: checkpoint integrity ok, resume_verified ok, "
                    "post-restart calibration 35.849 s/update = 0.63x baseline, determinism 16/16"
                ),
            ],
            "not_established": [
                "whether more updates would help (not tested; section 10 says stop dose escalation)",
                "whether another optimizer, learning rate or clip would help (not tested)",
                "that optimization cannot solve it (not proven)",
            ],
        },
        "competing_explanations": [
            {
                "name": "seed and initialization dominance",
                "status": "established",
                "evidence": (
                    "corrected eight-cell decomposition: model seed 99.40% of total SS, arm "
                    "0.28%, evaluation batch 0.008%, interaction 0.32%; same model across "
                    "batches moves at most 2.06pp; arm means 0.129106 vs 0.139338"
                ),
            },
            {
                "name": "Task1 carried by structure, not by trained weights",
                "status": "established within the tested scope",
                "evidence": (
                    "20.1 shows the structural copy/selection path; 20.2 shows credit reaches "
                    "F; 20.4d shows the F switch adds no consistent gain, and B_EH without F "
                    "is not worse.  The structural task is not evidence of trained capability"
                ),
            },
            {
                "name": "measurement environment rather than mechanism",
                "status": "partly established for cost, not for accuracy",
                "evidence": (
                    "unit price moved 0.61x-1.18x baseline with host memory pressure; "
                    "determinism is 16/16 and all diagonal cells matched the reused weights "
                    "digest, so accuracy differences are not explained by the environment"
                ),
            },
            {
                "name": "pre-registered F gate failure",
                "status": "established",
                "evidence": (
                    "pooled point gap 0.028064 < 0.05 and paired lower bound -0.064706 < 0; "
                    "per seed 20261106 passes the point and lower-bound clauses but 20261105 "
                    "reverses, so sign agreement fails; cut/permute were diagnostics only, not "
                    "part of the formal 2x2"
                ),
            },
            {
                "name": "not proven is not refuted",
                "status": "boundary statement",
                "evidence": (
                    "the tested candidate failed; the architecture, other write schedules, "
                    "other readouts and higher dose were not tested and are not declared "
                    "impossible"
                ),
            },
        ],
    }


def capability_ledger() -> dict:
    return {
        "added": [
            (
                "a bounded 2 seed x 2 arm comparison harness with per-25-update checkpoints, "
                "restart calibration, resume verification and 16/16 determinism"
            ),
            (
                "a pre-registered negative result: the current F candidate is not established, "
                "the arms are indistinguishable (pooled delta -0.028064, CI contains 0), and "
                "the evaluation batch is not dominant (ratio 0.7336)"
            ),
            (
                "a corrected variance fact: model seed explains 99.40% of the eight-cell "
                "decision-accuracy sum of squares, arm 0.28%, batch 0.008%"
            ),
            (
                "a cost calibration: 56.906 CPU/update from 20.4b; the formal comparison charged "
                "42,831.453 CPU for 800 updates; 20.4d added 4,026.438 CPU for 0 training"
            ),
        ],
        "not_established": [
            "feedback-conditioned adaptation and version correction",
            "a net extra contribution of F beyond the independent B_EH arm",
            "post-learning retention on the Task1 face (not measured in package 20)",
            "the confirm face (sealed; 0 reads in the whole package)",
            "any claim that the architecture cannot learn this task, or that other mechanisms cannot",
        ],
        "explicit_parallel": {
            "task1_structural": (
                "Task1 is carried by the structure; within the tested scope the trained weights "
                "contribute 0 to the structural task.  This is an established statement about "
                "the tested candidate, not a claim that training is useless everywhere"
            ),
            "feedback_learning": (
                "feedback learning is not established.  This is a negative result for the "
                "current write/read/optimization candidate, not a proof of impossibility"
            ),
            "do_not_merge": (
                "the two statements must stay separate: the structural success of Task1 does "
                "not sign feedback learning, and the failure of feedback learning does not "
                "retract the structural Task1 result"
            ),
        },
    }


def diagnostic_chain() -> list[dict]:
    items = [
        ("20.2r", "cut/permute leverage diagnostics"),
        ("20.2s", "window-level F information and closed-form readout"),
        ("20.2t", "write-granularity audit (manifest only)"),
        ("20.3a", "content-addressed store and read-key separability"),
        ("20.3b", "F authority and oracle replacement"),
        ("20.4a", "unit-cost decomposition"),
        ("20.4b", "graph-level executor optimization and equivalence"),
        ("20.4d", "shared-batch cross evaluation and variance separation"),
    ]
    return [
        {
            "id": item,
            "role": role,
            "classification": "diagnostic_non_planned",
            "conclusion": "merged_into_20_6",
        }
        for item, role in items
    ]


def build_report(runner_sha: str, amendment: dict) -> dict:
    r4cr2 = load(EVIDENCE["20_4c_r2"])
    rows, totals = cost_rows()
    cpu_now, wall_now = cost()
    rows.append(
        {
            "stage": "20.6",
            "kind": "close",
            "date": "2026-11-07",
            "cpu_seconds": round(cpu_now, 3),
            "wall_seconds": round(wall_now, 3),
            "updates": 0,
            "adam_steps": 0,
            "f_probe_evaluations": 0,
            "verdict": "package20_closed",
            "source": rel(REPORT),
            "note": "this bookkeeping runner; measured at report build time",
            "estimate": False,
            "cpu_seconds_lower_bound": None,
        }
    )
    measured = totals["measured_cpu_seconds"] + round(cpu_now, 3)
    totals["measured_cpu_seconds"] = round(measured, 3)
    totals["lower_bound_including_missing_lower_bounds"] = round(
        totals["lower_bound_including_missing_lower_bounds"] + cpu_now, 3
    )
    totals["measured_plus_estimates"] = round(totals["measured_plus_estimates"] + cpu_now, 3)
    totals["measured_wall_seconds"] = round(totals["measured_wall_seconds"] + wall_now, 3)
    totals["measured_vs_b"] = round(measured / 24000.0, 4)
    totals["measured_vs_gate"] = round(measured / 48000.0, 4)
    totals["measured_minus_gate"] = round(measured - 48000.0, 3)
    totals["measured_plus_estimates_vs_gate"] = round(
        totals["measured_plus_estimates"] / 48000.0, 4
    )
    totals["measured_plus_estimates_minus_gate"] = round(
        totals["measured_plus_estimates"] - 48000.0, 3
    )
    runs = r4cr2["runs"]
    final_readings = {
        key: {
            "decision_accuracy": run["readings"][-1].get("decision_accuracy"),
            "template_bit_accuracy": run["readings"][-1].get("template_bit_accuracy"),
            "content_bit_accuracy": run["readings"][-1].get("content_bit_accuracy"),
            "correction_accuracy": run["readings"][-1].get("correction_accuracy"),
            "post_feedback_accuracy": run["readings"][-1].get("post_feedback_accuracy"),
        }
        for key, run in runs.items()
    }
    evidence_shas = {name: file_sha(path) for name, path in EVIDENCE.items()}
    return {
        "schema": "taiji-rule-learning-20-6-package-close-v1",
        "evidence": "20_6",
        "date": "2026-11-07",
        "status": "completed",
        "verdict": "package20_closed_current_learning_candidate_not_established",
        "verdict_label": "package20_closed",
        "detail": (
            "section 10 row 2 selected: both arms are far below a strong shortcut, the "
            "task-side information and credit gates passed, the current learning candidate "
            "is not successful, and the extra contribution of F is not established.  20.5 "
            "admission is void, 20.5 is not executed, confirm stays sealed"
        ),
        "package_status": {
            "package": 20,
            "state": "closed",
            "planned_steps_completed": ["20.1", "20.2", "20.3", "20.4", "20.6"],
            "planned_step_not_executed": ["20.5"],
            "reason_20_5": (
                "20.4 did not select a single arm identity; the plan gates 20.5 on that "
                "selection, so 20.5 is not executed"
            ),
            "confirm": {
                "released": False,
                "read_record": 0,
                "indices": [240, 319],
                "materialised": False,
            },
            "automatic_patch_chain": False,
        },
        "discipline": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "new_probes": 0,
            "evaluation_reruns": 0,
            "confirm_reads": 0,
            "checkpoints_written": 0,
            "frozen_modules_touched": False,
            "reads": [rel(path) for path in EVIDENCE.values()],
            "write_then_rename": True,
            "progress_append_fsync": True,
            "cpu_cap": CPU_CAP,
            "cpu_seconds_this_runner": round(cpu_now, 3),
            "wall_seconds_this_runner": round(wall_now, 3),
        },
        "section10_branch_check": section10_branch_check(),
        "closure_findings": {
            "two_seed_arm_reversal": {
                "seed_20261105_delta_CF1_minus_B_EH": -0.202206,
                "seed_20261105_ci": [-0.222304, -0.182843],
                "seed_20261106_delta_CF1_minus_B_EH": 0.146078,
                "seed_20261106_ci": [0.098775, 0.191176],
                "pooled_delta": -0.028064,
                "pooled_ci": [-0.064706, 0.010539],
                "sign_agreement": False,
                "source": rel(EVIDENCE["20_4d"]) + " -> paired_shared_batch",
            },
            "f_net_contribution_gate": {
                "point_gap_threshold": 0.05,
                "paired_lower_bound_threshold": 0.0,
                "pooled_point_gap": 0.028064,
                "pooled_lower_bound": -0.064706,
                "established": False,
                "why": (
                    "the pooled point gap is below 0.05 and the paired lower bound is below "
                    "0; per seed, 20261106 passes both clauses but 20261105 reverses, so "
                    "sign agreement fails; cut and permute were diagnostics, not part of the "
                    "formal 2x2"
                ),
            },
            "absolute_level": {
                "best_decision_cell": 0.240196,
                "all_eight_full_byte_accuracy": 0.0,
                "template_bit_accuracy_range": [0.0, 0.007541],
                "strong_shortcut_reference": 0.636364,
                "strong_shortcut_source": "7/11 label-multiset Bayes reference, plan section 3.3",
                "caveat": (
                    "the 0.636364 reference is on the 22-query set while the decision cells "
                    "cover all 51 turns; the directly comparable fact is that every cell has "
                    "zero whole-sentence accuracy"
                ),
            },
            "correction_metrics": {
                "cf1_20261105": {
                    "correction_accuracy": None,
                    "post_feedback_accuracy": None,
                    "status": "missing",
                },
                "cf1_20261106": {
                    "correction_accuracy": 0.264773,
                    "post_feedback_accuracy": 0.268519,
                    "status": "measured",
                },
                "b_eh_20261105": {
                    "correction_accuracy": 0.25,
                    "post_feedback_accuracy": 0.248148,
                    "status": "measured",
                },
                "b_eh_20261106": {
                    "correction_accuracy": 0.096591,
                    "post_feedback_accuracy": 0.080093,
                    "status": "measured",
                },
            },
            "final_training_readings": final_readings,
            "confirm": "sealed; 0 reads; not materialised",
        },
        "attribution": attribution(),
        "cost_ledger": {
            "unit": "CPU seconds unless stated",
            "rows": rows,
            "totals": totals,
            "wasted_and_rerun": {
                "fully_wasted": [
                    {
                        "stage": "20.4c",
                        "cpu_seconds": 2514.203,
                        "updates": 25,
                        "reason": "aborted on the old absolute memory floor; readings kept as history",
                    },
                    {
                        "stage": "20.2s_failed_attempt",
                        "cpu_seconds": None,
                        "lower_bound": 25.75,
                        "reason": "exact CPU missing; lower bound from partial progress",
                    },
                    {
                        "stage": "20.3b_discarded_attempts",
                        "cpu_seconds": None,
                        "reason": "two discarded attempts; CPU missing",
                    },
                ],
                "carried_forward_not_wasted": [
                    {
                        "stage": "20.4c_r2_pre_restart",
                        "cpu_seconds": 16321.281,
                        "updates": 275,
                        "reason": "CF1|20261105 complete and B_EH|20261105 at 75; resumed from checkpoint",
                    },
                    {
                        "stage": "20.4a_earlier_pass",
                        "cpu_seconds": 727.891,
                        "reason": "10 calibration samples reused by the delivered pass",
                    },
                ],
                "engineering_reruns_declared": [
                    {
                        "stage": "20.4b",
                        "cpu_seconds": 1700.0,
                        "exact": False,
                        "reason": "four engineering reruns declared in the execution log; per-run values missing",
                    }
                ],
            },
            "missing_policy": (
                "missing values stay null and are listed by stage; they are not replaced by 0.  "
                "The measured total is therefore a lower bound of the true spend"
            ),
        },
        "training_ledger": {
            "formal_updates": 800,
            "aborted_updates": 25,
            "total_training_updates": 825,
            "adam_steps": 825,
            "formal_runs": 4,
            "updates_per_run": 200,
            "engineering_updates_known": {
                "20_4": 12,
                "20_4a": 20,
                "20_4b": 6,
                "note": "known optimizer updates outside the formal comparison; discarded or "
                "retried runs add an unrecorded amount, so the exact total is missing",
            },
            "calibration_observations": 15,
            "f_probe_events": 17,
            "f_probe_arm_evaluations": 85,
            "f_probe_detail": {
                "20_4c": 5,
                "20_4c_r2": 80,
                "rule": "five counterfactual levels per event, 0 optimizer steps",
            },
        },
        "capability_ledger": capability_ledger(),
        "known_defects": [
            {
                "item": "CF1|20261105 correction-surface and post-feedback first-answer readings",
                "value": "missing",
                "reason": (
                    "the run completed before the split keys existed and was reused under "
                    "run_completed; its accumulator was computed by the old accuracy() and "
                    "cannot be back-filled without rerunning the dev evaluation, which 20.6 "
                    "forbids.  The missing cell is the diagonal CF1|20261105 on batch "
                    "20261105; the shared-batch paired delta is unaffected"
                ),
                "source": rel(EVIDENCE["20_4c_r2"]) + " -> runs.CF1|20261105.dev_eval",
            },
            {
                "item": "20.4d variance block",
                "value": "amended",
                "reason": (
                    "the original block used only the four diagonal cells and charged model-seed "
                    "variance to the arm; the corrected eight-cell split is recorded in the "
                    "20.4d report under variance_correction.  Criteria and verdict are unchanged"
                ),
                "source": rel(R4D) + " -> variance_correction",
            },
            {
                "item": "20.2t",
                "value": "missing",
                "reason": "manifest only; no report and no runner, so cost and probe counts are missing, not zero",
                "source": rel(EVIDENCE["20_2t_manifest"]),
            },
            {
                "item": "20.3b discarded attempts",
                "value": "missing",
                "reason": "two discarded attempts have no recorded CPU",
                "source": rel(EVIDENCE["20_3b"]),
            },
            {
                "item": "20.4b rejected first pricing measurement",
                "value": "partly missing",
                "reason": (
                    "the report keeps both unit readings (40.291 and 56.906) but not the exact "
                    "cost of every engineering rerun; the execution log gives an aggregate of "
                    "about 1,700 CPU, recorded as an estimate"
                ),
                "source": "plans/active/roadmap/03_CURRENT_EXECUTION.md",
            },
            {
                "item": "post-learning retention on the Task1 face",
                "value": "missing",
                "reason": "package 20 did not run the retention face; this is not a failed gate",
                "source": "plan section 8, 20.4 post-evaluation controls",
            },
        ],
        "diagnostic_chain": diagnostic_chain(),
        "plans_sync": {
            "status": "scheduled",
            "targets": [
                "plans/active/roadmap/03_CURRENT_EXECUTION.md",
                "plans/MAINLINE.md",
                "plans/MASTER-PLAN.md",
                "plans/reference/ROADMAP-NEXT_20261104.md",
                "plans/reference/INVARIANT-LEDGER_20261103.md",
                "plans/reference/CAPABILITY-CARD_20261103.md",
            ],
            "required_state": (
                "20.2r/2s/2t, 20.3a/3b and 20.4a/4b/4d are marked diagnostic, non-planned, "
                "conclusions merged into 20.6; the queue head becomes 'package 20 closed' plus "
                "the single next question"
            ),
        },
        "next_question": {
            "question": (
                "200 updates raised content bits from 0 to 0.25-0.46 while template bits stayed "
                "near 0 and whole-sentence accuracy stayed 0 in every cell: does this task fail "
                "first at whole-sentence decoding/readout, or at feedback write/read authority?"
            ),
            "basis": [
                "all eight cells have full-byte accuracy 0.0; template bits are 0.0 except 0.007541",
                "content bits rise to 0.251470-0.453922 while decision bits stay 0.016422-0.240196",
                "20.3b oracle F replacement changes 0 bytes and answers 0/22",
                "20.4d arm effect 0.28% of SS versus model seed 99.40%",
            ],
            "not_a_training_plan": True,
            "not_a_price": True,
            "not_claiming_impossible": True,
            "note": (
                "this is the single queue-head question; it does not prescribe a training run, "
                "a budget or a package number, and it does not turn 'not proven' into 'impossible'"
            ),
        },
        "evidence_sha256": evidence_shas,
        "variance_decomposition_used": amendment["decomposition"],
        "amendment_of_20_4d": {
            "report_sha256_before": amendment["report_sha256_before"],
            "report_sha256_after": amendment["report_sha256_after"],
            "criteria_unchanged": True,
            "verdict_unchanged": True,
        },
        "script": {
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": None,
            "frozen_module_sha256": {
                "taiji/rule_learning_20_1_net.py": "8b5e95ef8aa14aad4dac0cf45b64b4505e30e1aac22b8c1d8a492f4a114de4bb",
                "taiji/living_core.py": "144a4760a1b7973d59d17f3edb59932e5e824289e63d570d50349dde1fbf8246",
            },
        },
        "cost": {
            "cpu_seconds": round(cpu_now, 3),
            "wall_seconds": round(wall_now, 3),
            "cpu_cap": CPU_CAP,
        },
    }


def main() -> int:
    guard("start")
    runner_sha = file_sha(Path(__file__).resolve())
    append_progress(
        {
            "stage": "start",
            "evidence": "20_6",
            "runner_sha256": runner_sha,
            "cpu": round(cost()[0], 3),
        }
    )

    amendment = amend_20_4d(runner_sha)
    append_progress(
        {
            "stage": "amend_20_4d",
            "status": "amended",
            "variance_share": amendment["decomposition"]["share"],
            "cpu": round(cost()[0], 3),
        }
    )
    guard("after_amendment")

    report = build_report(runner_sha, amendment)
    guard("before_report_write")
    write_json(REPORT, report)
    append_progress(
        {
            "stage": "report_written",
            "path": rel(REPORT),
            "sha256": file_sha(REPORT),
            "cpu": round(cost()[0], 3),
        }
    )

    manifest = {
        "schema": "taiji-rule-learning-20-6-package-close-manifest-v1",
        "evidence": "20_6",
        "verdict": report["verdict"],
        "verdict_label": report["verdict_label"],
        "package": 20,
        "package_state": "closed",
        "report_path": rel(REPORT),
        "report_sha256": file_sha(REPORT),
        "runner_sha256": runner_sha,
        "submitted_runner_sha256": None,
        "cpu_seconds": report["cost"]["cpu_seconds"],
        "cpu_cap": CPU_CAP,
        "training_steps": 0,
        "adam_steps": 0,
        "confirm_reads": 0,
        "completed": True,
    }
    write_json(MANIFEST, manifest)
    append_progress(
        {
            "stage": "manifest_written",
            "path": rel(MANIFEST),
            "sha256": file_sha(MANIFEST),
            "cpu": round(cost()[0], 3),
        }
    )
    cpu, wall = cost()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "measured_cpu_seconds": report["cost_ledger"]["totals"]["measured_cpu_seconds"],
                "runner_cpu_seconds": round(cpu, 3),
                "runner_wall_seconds": round(wall, 3),
                "variance_share": amendment["decomposition"]["share"],
                "report": rel(REPORT),
                "manifest": rel(MANIFEST),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

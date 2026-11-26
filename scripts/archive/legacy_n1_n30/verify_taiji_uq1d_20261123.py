"""UQ-1d: multi-seed calibration + Clopper-Pearson selective risk control.

Same subject/recipe as UQ-1.  Calibration and test splits each pool 4 generator seeds
(1024 samples) to stabilise the distribution.  On each task's calibration split choose
the largest confidence prefix k (k/n >= 0.30) whose exact one-sided 95% Clopper-Pearson
upper bound on the prefix risk is <= alpha_t (p24 0.10, p27 0.04); apply tau to the test
split.  Permuted control shuffles only the correctness sequence (breaking the
confidence-correctness association).  Single seed training, <=300 CPU, zero check/final.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
from scipy.stats import beta

from scripts.training.verify_taiji_24s_f1c_20261122 import capsule_state_diff
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_uq1b_20261123 import (
    build_subject,
    evaluate_policy,
    logits_at,
    train,
)

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_uq1d_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_uq1d_20261123.json"
SEED = 20261120
FIT_OFFSET = 10_000_000


def p24_multi(offsets: list[int], per: int) -> dict:
    ticks, labels = [], []
    for off in offsets:
        b = _make_p24_stream(per, SEED + off + FIT_OFFSET)
        ticks.append(b["ticks"])
        labels.append(b["val_labels"])
    return {"ticks": torch.cat(ticks), "labels": torch.cat(labels)}


def p27_multi(offsets: list[int], per: int) -> dict:
    ticks, labels = [], []
    for off in offsets:
        b = _make_p27_physics_batch(per, SEED + off + FIT_OFFSET)
        ticks.append(b["ticks"])
        labels.append(b["labels"])
    return {"ticks": torch.cat(ticks), "labels": torch.cat(labels)}


def cp_upper(errors: int, k: int, level: float) -> float:
    if k <= 0 or errors >= k:
        return 1.0
    return float(beta.ppf(level, errors + 1, k - errors))


def conformal_threshold(
    conf: torch.Tensor,
    correct: torch.Tensor,
    alpha: float,
    min_calib_coverage: float,
    level: float,
) -> dict:
    n = conf.numel()
    order = torch.argsort(conf, descending=True)
    c = correct[order]
    errs = (~c).float()
    cum = torch.cumsum(errs, 0)
    ok = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        k = i + 1
        if k / n < min_calib_coverage:
            continue
        ok[i] = cp_upper(int(cum[i]), k, level) <= alpha
    if not bool(ok.any()):
        return {"tau": float("inf"), "calib_coverage": 0.0, "calib_ucb": None, "k": 0}
    k = int(ok.nonzero()[-1]) + 1
    return {
        "tau": float(conf[order[k - 1]]),
        "calib_coverage": k / n,
        "calib_ucb": cp_upper(int(cum[k - 1]), k, level),
        "k": k,
    }


def main() -> int:
    started = time.process_time()
    crit = json.loads(CRITERIA.read_text(encoding="utf-8"))
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    digest = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == crit["criteria_sha256"], "criteria_sha256_mismatch"
    torch.set_num_threads(1)
    p = crit["protocol"]

    subj = build_subject()
    o = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    train(subj, o, "p24", p["p24_steps"])
    p24_calib = p24_multi(p["p24_calib_offsets"], p["per_seed_count"])
    p24_test = p24_multi(p["p24_test_offsets"], p["per_seed_count"])
    p24_after_p24 = float(
        (logits_at(subj, "p24", p24_test["ticks"]).argmax(-1) == p24_test["labels"]).float().mean()
    )
    for x in subj.graph.cross_channel_mix.parameters():
        x.requires_grad_(False)
    for x in subj.phase_heads["p24u"].parameters():
        x.requires_grad_(False)
    for x in subj.phase_heads["p27u"].parameters():
        x.requires_grad_(True)
    o27 = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    train(subj, o27, "p27", p["p27_steps"])
    p27_calib = p27_multi(p["p27_calib_offsets"], p["per_seed_count"])
    p27_test = p27_multi(p["p27_test_offsets"], p["per_seed_count"])
    p24_retention = float(
        (logits_at(subj, "p24", p24_test["ticks"]).argmax(-1) == p24_test["labels"]).float().mean()
    )

    tasks = {}
    for task, calib, test in (("p24", p24_calib, p24_test), ("p27", p27_calib, p27_test)):
        alpha = p["alpha"][task]
        cal_prob = torch.softmax(logits_at(subj, task, calib["ticks"]), dim=-1)
        test_prob = torch.softmax(logits_at(subj, task, test["ticks"]), dim=-1)
        cal_conf, cal_pred = cal_prob.max(dim=-1)
        test_conf, test_pred = test_prob.max(dim=-1)
        cal_correct = cal_pred == calib["labels"]
        test_correct = test_pred == test["labels"]
        policy = conformal_threshold(
            cal_conf, cal_correct, alpha, p["min_calib_coverage"], p["cp_level"]
        )
        test_policy = evaluate_policy(test_conf, test_correct, policy["tau"])
        perm_index = torch.randperm(
            cal_conf.numel(), generator=torch.Generator().manual_seed(SEED + 91)
        )
        perm_correct = cal_correct[perm_index]
        perm = conformal_threshold(
            cal_conf, perm_correct, alpha, p["min_calib_coverage"], p["cp_level"]
        )
        test_perm = evaluate_policy(test_conf, test_correct, perm["tau"])
        always = evaluate_policy(test_conf, test_correct, -1.0)
        tasks[task] = {
            "alpha": alpha,
            "n_calib": int(cal_conf.numel()),
            "n_test": int(test_conf.numel()),
            "test_acc": float(test_correct.float().mean()),
            "calib_acc": float(cal_correct.float().mean()),
            "conformal": {
                "tau": policy["tau"],
                "calib_coverage": policy["calib_coverage"],
                "calib_ucb": policy["calib_ucb"],
                **test_policy,
            },
            "permuted": {
                "tau": perm["tau"],
                "calib_coverage": perm["calib_coverage"],
                **test_perm,
                "satisfies": (test_perm["risk"] <= alpha + p["risk_tolerance"])
                and (test_perm["coverage"] >= p["coverage_min"]),
            },
            "always_answer": {"risk": 1.0 - float(test_correct.float().mean()), **always},
        }

    capsule_diff = capsule_state_diff(subj, p24_test["ticks"][:32], torch.ones(subj.n_units))
    retention_drop = max(0.0, p24_after_p24 - p24_retention)
    gates = {
        "G0": tasks["p24"]["test_acc"] >= crit["gates"]["G0_p24_acc_min"]
        and tasks["p27"]["test_acc"] >= crit["gates"]["G0_p27_acc_min"],
        "G1": all(
            tasks[t]["conformal"]["risk"] <= tasks[t]["alpha"] + p["risk_tolerance"] for t in tasks
        ),
        "G2": all(tasks[t]["conformal"]["coverage"] >= p["coverage_min"] for t in tasks),
        "G3": all(not tasks[t]["permuted"]["satisfies"] for t in tasks),
        "G4": any(
            tasks[t]["always_answer"]["risk"] > tasks[t]["alpha"] + p["risk_tolerance"]
            for t in tasks
        ),
        "G5": capsule_diff <= crit["gates"]["G5_capsule_diff_max"]
        and retention_drop <= crit["gates"]["G5_retention_drop_max"],
    }
    if not gates["G0"]:
        verdict = "uq1d_baseline_insufficient"
    elif not (gates["G1"] and gates["G2"]):
        verdict = "uq1d_risk_unmet"
    elif not gates["G3"]:
        verdict = "uq1d_feedback_weak"
    elif not gates["G4"]:
        verdict = "uq1d_calibration_optional"
    elif all(gates.values()):
        verdict = "uq1d_verified"
    else:
        verdict = "uq1d_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-uq1d-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "training": {"p24_acc_after_p24": p24_after_p24, "p24_acc_after_p27": p24_retention},
        "tasks": tasks,
        "capsule_state_diff": capsule_diff,
        "retention_drop": retention_drop,
        "gates": gates,
        "verdict": verdict,
        "run_conditions": {"threads": torch.get_num_threads()},
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": cost},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "tasks": {
                    k: {
                        "alpha": v["alpha"],
                        "test_acc": round(v["test_acc"], 4),
                        "conformal": {
                            "tau": round(v["conformal"]["tau"], 4),
                            "calib_cov": round(v["conformal"]["calib_coverage"], 4),
                            "calib_ucb": (
                                round(v["conformal"]["calib_ucb"], 4)
                                if v["conformal"]["calib_ucb"] is not None
                                else None
                            ),
                            "test_cov": round(v["conformal"]["coverage"], 4),
                            "test_risk": round(v["conformal"]["risk"], 4),
                        },
                        "permuted": {
                            "cov": round(v["permuted"]["coverage"], 4),
                            "risk": round(v["permuted"]["risk"], 4),
                            "satisfies": v["permuted"]["satisfies"],
                        },
                        "always_risk": round(v["always_answer"]["risk"], 4),
                    }
                    for k, v in tasks.items()
                },
                "capsule_diff": capsule_diff,
                "retention_drop": round(retention_drop, 4),
                "cpu": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

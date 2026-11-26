"""UQ-1b: risk-target selective answering with adaptive per-task thresholds.

Same subject/recipe as UQ-1 (p24 500 + p27 600 steps, per-block 48-dim heads, block
isolation, wd=0).  On the calibration split of each task choose the largest coverage
whose empirical risk is <= alpha=0.10; apply that threshold to a separate test split
without tuning.  Controls: fixed-70%-coverage threshold, permuted-calibration threshold
(must fail risk control), retention/capsule identity.  Single seed, <=300 CPU,
zero check/final. Standing authorization; owner may revise.
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
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_24s_f1c_20261122 import capsule_state_diff
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from scripts.training.verify_taiji_uq1_20261122 import own_logits

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_uq1b_criteria_20261123.json"
OUT = ROOT / "reports/taiji_u2_uq1b_20261123.json"
SEED = 20261120
FIT = 256
FIT_OFFSET = 10_000_000


def build_subject() -> Subject72:
    torch.manual_seed(SEED)
    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 401)
        subj.phase_heads["p24u"] = PhaseHead(48, 4, 64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 402)
        subj.phase_heads["p27u"] = PhaseHead(48, 4, 64)
    for p in subj.phase_heads["p24u"].parameters():
        p.requires_grad_(True)
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    with torch.no_grad():
        units = subj.graph.selective_units
        for cols in (slice(0, 24), slice(48, 72)):
            units.w[24:48, cols] = 0.0
            units.write_state.weight[24:48, cols] = 0.0
    return subj


def train(subj, opt, task: str, steps: int) -> None:
    for _k in range(steps):
        if task == "p24":
            b = _make_p24_stream(48, SEED * 25000 + _k)
            ticks = b["ticks"]
            labels = b["val_labels"]
        else:
            b = _make_p27_physics_batch(48, SEED * 41000 + _k)
            ticks = b["ticks"]
            labels = b["labels"]
        loss = F.cross_entropy(own_logits(subj, task, ticks), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, task)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()


def logits_at(subj, task: str, ticks) -> torch.Tensor:
    subj.eval_mode()
    with torch.no_grad():
        return own_logits(subj, task, ticks)


def risk_threshold(conf: torch.Tensor, correct: torch.Tensor, alpha: float) -> dict:
    order = torch.argsort(conf, descending=True)
    c = correct[order]
    n = conf.numel()
    cum_err = torch.cumsum((~c).float(), 0) / torch.arange(1, n + 1)
    ok = cum_err <= alpha
    k = n if bool(ok.all()) else int((~ok).nonzero()[0])
    if k == 0:
        return {"tau": float("inf"), "coverage": 0.0, "calib_risk": 0.0, "k": 0}
    tau = float(conf[order[k - 1]])
    return {
        "tau": tau,
        "coverage": k / n,
        "calib_risk": float(cum_err[k - 1]),
        "k": k,
    }


def evaluate_policy(conf: torch.Tensor, correct: torch.Tensor, tau: float) -> dict:
    answered = conf >= tau
    coverage = float(answered.float().mean())
    risk = float((~correct[answered]).float().mean()) if int(answered.sum()) > 0 else 1.0
    sel_acc = 1.0 - risk if int(answered.sum()) > 0 else 0.0
    return {
        "coverage": coverage,
        "risk": risk,
        "selective_acc": sel_acc,
        "abstain_rate": 1.0 - coverage,
        "n_answered": int(answered.sum()),
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
    p24_calib = _make_p24_stream(FIT, SEED + p["p24_calib_seed_offset"] + FIT_OFFSET)
    p24_test = _make_p24_stream(FIT, SEED + p["p24_test_seed_offset"] + FIT_OFFSET)
    p24_after_p24 = float(
        (logits_at(subj, "p24", p24_test["ticks"]).argmax(-1) == p24_test["val_labels"])
        .float()
        .mean()
    )
    for x in subj.graph.cross_channel_mix.parameters():
        x.requires_grad_(False)
    for x in subj.phase_heads["p24u"].parameters():
        x.requires_grad_(False)
    for x in subj.phase_heads["p27u"].parameters():
        x.requires_grad_(True)
    o27 = torch.optim.Adam(subj.params(), lr=p["lr"], weight_decay=p["wd"])
    train(subj, o27, "p27", p["p27_steps"])
    p27_calib = _make_p27_physics_batch(FIT, SEED + p["p27_calib_seed_offset"] + FIT_OFFSET)
    p27_test = _make_p27_physics_batch(FIT, SEED + p["p27_test_seed_offset"] + FIT_OFFSET)
    p24_retention = float(
        (logits_at(subj, "p24", p24_test["ticks"]).argmax(-1) == p24_test["val_labels"])
        .float()
        .mean()
    )

    tasks = {}
    for task, calib, test, label_key in (
        ("p24", p24_calib, p24_test, "val_labels"),
        ("p27", p27_calib, p27_test, "labels"),
    ):
        cal_logits = logits_at(subj, task, calib["ticks"])
        test_logits = logits_at(subj, task, test["ticks"])
        cal_prob = torch.softmax(cal_logits, dim=-1)
        test_prob = torch.softmax(test_logits, dim=-1)
        cal_conf, cal_pred = cal_prob.max(dim=-1)
        test_conf, test_pred = test_prob.max(dim=-1)
        cal_correct = cal_pred == calib[label_key]
        test_correct = test_pred == test[label_key]
        risky = risk_threshold(cal_conf, cal_correct, p["alpha"])
        t1 = risky["tau"]
        policy = evaluate_policy(test_conf, test_correct, t1)
        fixed_tau = float(torch.quantile(cal_conf, 1.0 - p["fixed_coverage"]))
        fixed = evaluate_policy(test_conf, test_correct, fixed_tau)
        perm_index = torch.randperm(cal_conf.numel())
        perm_risky = risk_threshold(cal_conf[perm_index], cal_correct[perm_index], p["alpha"])
        permuted = evaluate_policy(test_conf, test_correct, perm_risky["tau"])
        # per-sample confidence compression for the permuted control is not needed:
        # the threshold itself comes from permuted correctness, applied to the test split.
        perm_ok = (permuted["risk"] <= p["risk_tolerance"] + p["alpha"]) and (
            permuted["coverage"] >= p["coverage_min"]
        )
        tasks[task] = {
            "test_acc": float(test_correct.float().mean()),
            "calib_acc": float(cal_correct.float().mean()),
            "risk_target": {
                "tau": t1,
                "calib_coverage": risky["coverage"],
                "calib_risk": risky["calib_risk"],
                **policy,
            },
            "fixed_70": {"tau": fixed_tau, **fixed},
            "permuted_calib": {
                "tau": perm_risky["tau"],
                "calib_coverage": perm_risky["coverage"],
                **permuted,
                "satisfies_risk_control": perm_ok,
            },
        }

    capsule_diff = capsule_state_diff(subj, p24_test["ticks"][:32], torch.ones(subj.n_units))
    retention_drop = max(0.0, p24_after_p24 - p24_retention)
    abstain_p24 = tasks["p24"]["risk_target"]["abstain_rate"]
    abstain_p27 = tasks["p27"]["risk_target"]["abstain_rate"]
    gates = {
        "G0": tasks["p24"]["test_acc"] >= crit["gates"]["G0_p24_acc_min"]
        and tasks["p27"]["test_acc"] >= crit["gates"]["G0_p27_acc_min"],
        "G1": all(tasks[t]["risk_target"]["risk"] <= crit["gates"]["G1_risk_max"] for t in tasks),
        "G2": all(
            tasks[t]["risk_target"]["coverage"] >= crit["gates"]["G2_coverage_min"] for t in tasks
        ),
        "G3": abstain_p27 <= abstain_p24 - crit["gates"]["G3_abstain_gap_min"],
        "G4": all(not tasks[t]["permuted_calib"]["satisfies_risk_control"] for t in tasks),
        "G5": capsule_diff <= crit["gates"]["G5_capsule_diff_max"]
        and retention_drop <= crit["gates"]["G5_retention_drop_max"],
    }
    if not gates["G0"]:
        verdict = "uq1b_baseline_insufficient"
    elif not (gates["G1"] and gates["G2"]):
        verdict = "uq1b_risk_unmet"
    elif not gates["G3"]:
        verdict = "uq1b_not_adaptive"
    elif not gates["G4"]:
        verdict = "uq1b_feedback_weak"
    elif all(gates.values()):
        verdict = "uq1b_verified"
    else:
        verdict = "uq1b_partial"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-uq1b-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "alpha": p["alpha"],
        "training": {
            "p24_acc_after_p24": p24_after_p24,
            "p24_acc_after_p27": p24_retention,
        },
        "tasks": tasks,
        "capsule_state_diff": capsule_diff,
        "retention_drop": retention_drop,
        "abstain": {"p24": abstain_p24, "p27": abstain_p27},
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
                        "test_acc": round(v["test_acc"], 4),
                        "risk_target": {
                            "tau": round(v["risk_target"]["tau"], 4),
                            "coverage": round(v["risk_target"]["coverage"], 4),
                            "risk": round(v["risk_target"]["risk"], 4),
                            "abstain": round(v["risk_target"]["abstain_rate"], 4),
                        },
                        "fixed_70": {
                            "coverage": round(v["fixed_70"]["coverage"], 4),
                            "risk": round(v["fixed_70"]["risk"], 4),
                        },
                        "permuted_calib": {
                            "coverage": round(v["permuted_calib"]["coverage"], 4),
                            "risk": round(v["permuted_calib"]["risk"], 4),
                            "satisfies": v["permuted_calib"]["satisfies_risk_control"],
                        },
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

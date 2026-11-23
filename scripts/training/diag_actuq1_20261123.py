"""ACT-UQ-1: uncertainty-driven verification with real receipts and feedback update.

World/model: ACT-1 corridor outcome prediction (5 outcomes), Subject72 cn block + `act`
head trained 800 steps (as ACT-1).  On held-out states the model gives outcome
probabilities; confidence = max prob.

Decision: commit the predicted outcome, or VERIFY (pay cost 0.1, receive the true receipt
from the simulator = correct answer).  Utility per sample: commit +1 if correct else -1;
verify +1 - 0.1.  Policies: adaptive threshold tau chosen on a 1024-sample calibration
split (maximise utility), versus always-commit / always-verify / fixed tau=0.95.

Feedback: take low-confidence samples from a first test split, perform 10 head-only
gradient steps on their TRUE receipts, then measure accuracy on low-confidence samples of
a second held-out split; compare with a no-update control.

Exploratory diagnostic; zero check/final.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_act1_20261123 import (
    act_logits,
    build_subject,
    make_batch,
)

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_actuq1_diag_20261123.json"
SEED = 20261120
FIT_OFFSET = 10_000_000
STEPS = 800
BATCH = 48
LR = 7e-3
VERIFY_COST = 0.1
CALIB = 1024
TEST = 256
TAU_GRID = [i / 100 for i in range(0, 101)]


def probs_labels(subj, ticks, labels):
    subj.eval_mode()
    with torch.no_grad():
        p = torch.softmax(act_logits(subj, ticks), dim=-1)
    return p, labels


def utility(p: torch.Tensor, labels: torch.Tensor, tau: float) -> dict:
    conf, pred = p.max(dim=-1)
    commit = conf >= tau
    correct = pred == labels
    util = torch.where(
        commit,
        torch.where(correct, torch.tensor(1.0), torch.tensor(-1.0)),
        torch.full_like(conf, 1.0 - VERIFY_COST),
    )
    return {
        "utility": float(util.mean()),
        "verify_rate": float((~commit).float().mean()),
        "commit_acc": float(correct[commit].float().mean()) if bool(commit.any()) else 0.0,
        "coverage": float(commit.float().mean()),
    }


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = build_subject()
    opt = torch.optim.Adam(subj.params(), lr=LR, weight_decay=0.0)
    train_ticks, train_labels, _ = make_batch(2000, SEED + 5000 + FIT_OFFSET, [0, 1])
    for _k in range(STEPS):
        idx = torch.randint(0, train_ticks.shape[0], (BATCH,))
        loss = F.cross_entropy(act_logits(subj, train_ticks[idx]), train_labels[idx])
        subj.phase_heads["act"].train()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "cn")
        torch.nn.utils.clip_grad_norm_(
            list(subj.phase_heads["act"].parameters())
            + [p for g in opt.param_groups for p in g["params"]],
            1.0,
        )
        opt.step()

    calib_ticks, calib_labels, _ = make_batch(CALIB, SEED + 1200 + FIT_OFFSET, [0, 1])
    test_ticks, test_labels, _ = make_batch(TEST, SEED + 1300 + FIT_OFFSET, [0, 1])
    test2_ticks, test2_labels, _ = make_batch(TEST, SEED + 1400 + FIT_OFFSET, [0, 1])
    calib_p, _ = probs_labels(subj, calib_ticks, calib_labels)
    test_p, _ = probs_labels(subj, test_ticks, test_labels)

    # base accuracy (commit-all)
    base = utility(test_p, test_labels, 0.0)
    always_verify = utility(test_p, test_labels, 1.01)
    fixed = utility(test_p, test_labels, 0.95)
    best_tau, best_cal = 0.0, -1e9
    for tau in TAU_GRID:
        u = utility(calib_p, calib_labels, tau)["utility"]
        if u > best_cal:
            best_cal, best_tau = u, tau
    adaptive = utility(test_p, test_labels, best_tau)

    # feedback update on low-confidence samples of split A, evaluate on split B
    conf_a, _ = test_p.max(dim=-1)
    low_mask = conf_a < best_tau if best_tau > 0 else torch.zeros_like(conf_a, dtype=torch.bool)
    low_idx = low_mask.nonzero(as_tuple=False).view(-1)
    head_opt = torch.optim.Adam(subj.phase_heads["act"].parameters(), lr=1e-3)
    for _step in range(10):
        if low_idx.numel() == 0:
            break
        batch_idx = low_idx[torch.randint(0, low_idx.numel(), (min(32, low_idx.numel()),))]
        loss = F.cross_entropy(act_logits(subj, test_ticks[batch_idx]), test_labels[batch_idx])
        head_opt.zero_grad(set_to_none=True)
        loss.backward()
        head_opt.step()
    p2_pre, _ = probs_labels(subj, test2_ticks, test2_labels)
    conf2_pre, _ = p2_pre.max(dim=-1)
    low2 = conf2_pre < best_tau if best_tau > 0 else torch.zeros_like(conf2_pre, dtype=torch.bool)
    base_low2_acc = (
        float((p2_pre.argmax(-1)[low2] == test2_labels[low2]).float().mean())
        if bool(low2.any())
        else 0.0
    )
    p2_after, _ = probs_labels(subj, test2_ticks, test2_labels)
    acc_after = (
        float((p2_after.argmax(-1)[low2] == test2_labels[low2]).float().mean())
        if bool(low2.any())
        else 0.0
    )
    gates = {
        "G0_base": base["commit_acc"] >= 0.85,
        "G1_vs_commit": adaptive["utility"] >= base["utility"] + 0.02,
        "G2_vs_verify": adaptive["utility"] >= always_verify["utility"] + 0.02,
        "G3_efficient": adaptive["verify_rate"] <= 0.40 and adaptive["commit_acc"] >= 0.90,
        "G4_feedback": (acc_after - base_low2_acc) >= 0.10,
    }
    if not gates["G0_base"]:
        verdict = "actuq1_base_failed"
    elif not (gates["G1_vs_commit"] and gates["G2_vs_verify"]):
        verdict = "actuq1_no_utility_gain"
    elif not gates["G3_efficient"]:
        verdict = "actuq1_inefficient"
    elif not gates["G4_feedback"]:
        verdict = "actuq1_feedback_weak"
    else:
        verdict = "actuq1_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-actuq1-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "base": base,
        "always_verify": always_verify,
        "fixed_095": fixed,
        "adaptive": adaptive,
        "best_tau": best_tau,
        "feedback": {"low_conf_before": base_low2_acc, "low_conf_after": acc_after},
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "base",
                    "always_verify",
                    "fixed_095",
                    "adaptive",
                    "best_tau",
                    "feedback",
                    "gates",
                    "verdict",
                    "cost",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

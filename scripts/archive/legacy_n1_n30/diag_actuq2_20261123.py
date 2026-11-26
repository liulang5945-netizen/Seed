"""ACT-UQ-2: uncertainty-driven verification on a NON-saturated task (P24, base ~0.79).

Model: UQ-1 recipe p24 block 500 steps (block 24:48 + dedicated 48-dim head).  The model
predicts 4-way P24 answers with confidence = max prob.
Decision: commit vs VERIFY (cost 0.1, receive the true receipt = correct answer).
Utility: commit +1/-1 (correct/wrong), verify +0.9.
Policies: adaptive tau on 1024 calibration samples (max utility), always-commit,
always-verify, fixed 0.95.
Feedback: 10 head-only steps on verified low-confidence receipts of split A; measure on
the same low-confidence subset of split B (pre/post).
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
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream
from scripts.training.verify_taiji_uq1_20261122 import own_logits
from scripts.training.verify_taiji_uq1b_20261123 import build_subject, train

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_actuq2_diag_20261123.json"
SEED = 20261120
FIT_OFFSET = 10_000_000
P24_STEPS = 500
VERIFY_COST = 0.1
CALIB = 1024
TEST = 256
TAU_GRID = [i / 100 for i in range(0, 101)]


def probs_labels(subj, task: str, ticks, labels):
    subj.eval_mode()
    with torch.no_grad():
        p = torch.softmax(own_logits(subj, task, ticks), dim=-1)
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
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    train(subj, opt, "p24", P24_STEPS)

    calib = _make_p24_stream(CALIB, SEED + 1200 + FIT_OFFSET)
    test = _make_p24_stream(TEST, SEED + 1300 + FIT_OFFSET)
    test2 = _make_p24_stream(TEST, SEED + 1400 + FIT_OFFSET)
    calib_p, _ = probs_labels(subj, "p24", calib["ticks"], calib["val_labels"])
    test_p, _ = probs_labels(subj, "p24", test["ticks"], test["val_labels"])

    base = utility(test_p, test["val_labels"], 0.0)
    always_verify = utility(test_p, test["val_labels"], 1.01)
    fixed = utility(test_p, test["val_labels"], 0.95)
    best_tau, best_cal = 0.0, -1e9
    for tau in TAU_GRID:
        u = utility(calib_p, calib["val_labels"], tau)["utility"]
        if u > best_cal:
            best_cal, best_tau = u, tau
    adaptive = utility(test_p, test["val_labels"], best_tau)

    p2_pre, _ = probs_labels(subj, "p24", test2["ticks"], test2["val_labels"])
    conf2, _ = p2_pre.max(dim=-1)
    low2 = conf2 < best_tau if best_tau > 0 else torch.zeros_like(conf2, dtype=torch.bool)
    base_low2 = (
        float((p2_pre.argmax(-1)[low2] == test2["val_labels"][low2]).float().mean())
        if bool(low2.any())
        else 0.0
    )

    # feedback on verified (low-confidence) receipts of split A, evaluated on split B
    conf_a, _ = test_p.max(dim=-1)
    low_a = conf_a < best_tau if best_tau > 0 else torch.zeros_like(conf_a, dtype=torch.bool)
    low_idx = low_a.nonzero(as_tuple=False).view(-1)
    head_opt = torch.optim.Adam(subj.phase_heads["p24u"].parameters(), lr=1e-3)
    for _step in range(10):
        if low_idx.numel() == 0:
            break
        batch_idx = low_idx[torch.randint(0, low_idx.numel(), (min(32, low_idx.numel()),))]
        loss = F.cross_entropy(
            own_logits(subj, "p24", test["ticks"][batch_idx]), test["val_labels"][batch_idx]
        )
        head_opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "p24")
        head_opt.step()

    p2_after, _ = probs_labels(subj, "p24", test2["ticks"], test2["val_labels"])
    after_low2 = (
        float((p2_after.argmax(-1)[low2] == test2["val_labels"][low2]).float().mean())
        if bool(low2.any())
        else 0.0
    )

    gates = {
        "G0_base": base["commit_acc"] >= 0.70,
        "G1_vs_commit": adaptive["utility"] >= base["utility"] + 0.02,
        "G2_vs_verify": adaptive["utility"] >= always_verify["utility"] + 0.02,
        "G3_efficient": adaptive["verify_rate"] <= 0.40 and adaptive["commit_acc"] >= 0.90,
        "G4_feedback": (after_low2 - base_low2) >= 0.10,
    }
    if not gates["G0_base"]:
        verdict = "actuq2_base_failed"
    elif not (gates["G1_vs_commit"] and gates["G2_vs_verify"]):
        verdict = "actuq2_no_utility_gain"
    elif not gates["G3_efficient"]:
        verdict = "actuq2_inefficient"
    elif not gates["G4_feedback"]:
        verdict = "actuq2_feedback_weak"
    else:
        verdict = "actuq2_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-actuq2-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "base": base,
        "always_verify": always_verify,
        "fixed_095": fixed,
        "adaptive": adaptive,
        "best_tau": best_tau,
        "feedback": {
            "low_conf_before": base_low2,
            "low_conf_after": after_low2,
            "n_low": int(low2.sum()),
        },
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

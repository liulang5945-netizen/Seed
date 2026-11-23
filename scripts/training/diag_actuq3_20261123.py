"""ACT-UQ-3: strengthening receipt-based feedback update on the P24 non-saturated task.

Same setup as ACT-UQ-2 (p24 500 steps, adaptive tau on calibration).  On the low-
confidence samples of test split A (conf < tau*) we compare feedback arms:
  HEAD10     head-only, 10 steps, lr 1e-3 (ACT-UQ-2 reproduction)
  HEAD100    head-only, 100 steps, lr 3e-3
  UNIT50     unit+head, 50 steps, lr 1e-3
  RECAL      no model update; recalibrate tau on calibration + received receipts
Evaluation: accuracy on the fixed low-confidence subset of held-out split B, and overall
test-B utility with the policy tau.
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
OUT = ROOT / "reports/taiji_actuq3_diag_20261123.json"
SEED = 20261120
FIT_OFFSET = 10_000_000
P24_STEPS = 500
VERIFY_COST = 0.1
CALIB = 1024
TEST = 256
TAU_GRID = [i / 100 for i in range(0, 101)]


def probs(subj, ticks):
    subj.eval_mode()
    with torch.no_grad():
        return torch.softmax(own_logits(subj, "p24", ticks), dim=-1)


def policy_utility(p, labels, tau):
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
    }


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = build_subject()
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    train(subj, opt, "p24", P24_STEPS)

    calib = _make_p24_stream(CALIB, SEED + 1200 + FIT_OFFSET)
    a = _make_p24_stream(TEST, SEED + 1300 + FIT_OFFSET)
    b = _make_p24_stream(TEST, SEED + 1400 + FIT_OFFSET)
    calib_p = probs(subj, calib["ticks"])
    a_p = probs(subj, a["ticks"])
    b_p_pre = probs(subj, b["ticks"])
    best_tau = max(
        TAU_GRID, key=lambda t: policy_utility(calib_p, calib["val_labels"], t)["utility"]
    )
    conf_a, _ = a_p.max(dim=-1)
    low_a = (conf_a < best_tau).nonzero(as_tuple=False).view(-1)
    conf_b, _ = b_p_pre.max(dim=-1)
    low_b = conf_b < best_tau
    acc_b_low_pre = (
        float((b_p_pre.argmax(-1)[low_b] == b["val_labels"][low_b]).float().mean())
        if bool(low_b.any())
        else 0.0
    )

    def update(steps: int, lr: float, units: bool):
        params = list(subj.phase_heads["p24u"].parameters())
        if units:
            params += [p for p in subj.graph.selective_units.parameters() if p.requires_grad]
        opt_fb = torch.optim.Adam(params, lr=lr)
        for _ in range(steps):
            if low_a.numel() == 0:
                break
            idx = low_a[torch.randint(0, low_a.numel(), (min(32, low_a.numel()),))]
            loss = F.cross_entropy(own_logits(subj, "p24", a["ticks"][idx]), a["val_labels"][idx])
            opt_fb.zero_grad(set_to_none=True)
            loss.backward()
            mask_grads(subj, "p24")
            opt_fb.step()

    results = {}
    for name, steps, lr, units in (
        ("HEAD10", 10, 1e-3, False),
        ("HEAD100", 100, 3e-3, False),
        ("UNIT50", 50, 1e-3, True),
    ):
        trial = subj
        update(steps, lr, units)
        p_b = probs(trial, b["ticks"])
        acc_after = (
            float((p_b.argmax(-1)[low_b] == b["val_labels"][low_b]).float().mean())
            if bool(low_b.any())
            else 0.0
        )
        results[name] = {
            "low_before": acc_b_low_pre,
            "low_after": acc_after,
            "gain": acc_after - acc_b_low_pre,
            "utility": policy_utility(p_b, b["val_labels"], best_tau)["utility"],
        }
        # reload: rebuild subject for the next arm to keep arms independent
        subj = build_subject()
        opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
        train(subj, opt, "p24", P24_STEPS)

    # RECAL arm: no model update; recalibrate tau using calibration + A receipts
    combo_conf = torch.cat([calib_p, a_p], dim=0)
    combo_labels = torch.cat([calib["val_labels"], a["val_labels"]], dim=0)
    recal_tau = max(TAU_GRID, key=lambda t: policy_utility(combo_conf, combo_labels, t)["utility"])
    p_b = probs(subj, b["ticks"])
    results["RECAL"] = {
        "tau": recal_tau,
        "utility": policy_utility(p_b, b["val_labels"], recal_tau)["utility"],
    }

    gains = [results[k]["gain"] for k in ("HEAD10", "HEAD100", "UNIT50")]
    utilities = [results[k]["utility"] for k in ("HEAD10", "HEAD100", "UNIT50", "RECAL")]
    gates = {
        "G0_base": policy_utility(b_p_pre, b["val_labels"], 0.0)["commit_acc"] >= 0.70,
        "G1_gain": max(gains) >= 0.10,
        "G2_utility": max(utilities) >= 0.95,
        "G3_verify": policy_utility(b_p_pre, b["val_labels"], best_tau)["verify_rate"] <= 0.40,
    }
    if not gates["G0_base"]:
        verdict = "actuq3_base_failed"
    elif not gates["G1_gain"]:
        verdict = "actuq3_feedback_weak"
    elif not gates["G2_utility"]:
        verdict = "actuq3_utility_low"
    elif not gates["G3_verify"]:
        verdict = "actuq3_inefficient"
    else:
        verdict = "actuq3_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-actuq3-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "best_tau": best_tau,
        "low_before": acc_b_low_pre,
        "arms": results,
        "gates": gates,
        "verdict": verdict,
        "cost": {"cpu": cost},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: payload[k] for k in ("best_tau", "low_before", "arms", "gates", "verdict", "cost")},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ACT-UQ-5: cross the receipt-feedback gate (UNIT600, full low-confidence replay).

Seeds 20261120/21/22; per seed: p24 block 500 steps; adaptive tau; low-confidence subset of
split A; arms (copies of the trained subject):
  UNIT600  unit+head, 600 steps, FULL low-confidence replay (batches of 64 over all low
           samples, cycled), lr 2e-3
  UNIT300  unit+head, 300 steps, random 64-sample batches (ACT-UQ-4 reproduction)
  HEAD300  head-only, 300 steps
  NOFB     no update control
Evaluate the same low-confidence subset of split B and the policy utility.
Gates: mean UNIT600 gain >= 0.10 with 3/3 seeds positive; UNIT600 > HEAD300 + 0.03;
min utility >= 0.95; budget <= 700 CPU.  Exploratory diagnostic; zero check/final.
"""

from __future__ import annotations

import copy
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
OUT = ROOT / "reports/taiji_actuq5_diag_20261123.json"
FIT_OFFSET = 10_000_000
P24_STEPS = 500
VERIFY_COST = 0.1
CALIB = 1024
TEST = 256
TAU_GRID = [i / 100 for i in range(0, 101)]
SEEDS = (20261120, 20261121, 20261122)


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


def feedback(subj, a, low_idx, steps: int, units: bool, full: bool) -> None:
    params = list(subj.phase_heads["p24u"].parameters())
    if units:
        params += [p for p in subj.graph.selective_units.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=2e-3)
    cycle = 0
    for _ in range(steps):
        if low_idx.numel() == 0:
            break
        if full:
            start = (cycle * 64) % low_idx.numel()
            idx = low_idx[start : start + 64]
            cycle += 1
            if idx.numel() == 0:
                continue
        else:
            idx = low_idx[torch.randint(0, low_idx.numel(), (min(64, low_idx.numel()),))]
        loss = F.cross_entropy(own_logits(subj, "p24", a["ticks"][idx]), a["val_labels"][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, "p24")
        opt.step()


def low_acc(p, labels, mask) -> float:
    return float((p.argmax(-1)[mask] == labels[mask]).float().mean()) if bool(mask.any()) else 0.0


def run_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    subj = build_subject()
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    train(subj, opt, "p24", P24_STEPS)
    calib = _make_p24_stream(CALIB, seed + 1200 + FIT_OFFSET)
    a = _make_p24_stream(TEST, seed + 1300 + FIT_OFFSET)
    b = _make_p24_stream(TEST, seed + 1400 + FIT_OFFSET)
    calib_p = probs(subj, calib["ticks"])
    a_p = probs(subj, a["ticks"])
    b_p_pre = probs(subj, b["ticks"])
    tau = max(TAU_GRID, key=lambda t: policy_utility(calib_p, calib["val_labels"], t)["utility"])
    conf_a, _ = a_p.max(dim=-1)
    low_a = (conf_a < tau).nonzero(as_tuple=False).view(-1)
    conf_b, _ = b_p_pre.max(dim=-1)
    low_b = conf_b < tau
    before = low_acc(b_p_pre, b["val_labels"], low_b)
    arms = {}
    for name, steps, units, full in (
        ("UNIT600", 600, True, True),
        ("UNIT300", 300, True, False),
        ("HEAD300", 300, False, False),
        ("NOFB", 0, False, False),
    ):
        trial = copy.deepcopy(subj)
        if steps:
            feedback(trial, a, low_a, steps, units, full)
        p_b = probs(trial, b["ticks"])
        after = low_acc(p_b, b["val_labels"], low_b)
        arms[name] = {
            "low_after": after,
            "gain": after - before,
            "utility": policy_utility(p_b, b["val_labels"], tau)["utility"],
        }
    return {
        "tau": tau,
        "low_before": before,
        "base_commit_acc": policy_utility(b_p_pre, b["val_labels"], 0.0)["commit_acc"],
        "arms": arms,
    }


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    rows = {}
    for seed in SEEDS:
        rows[str(seed)] = run_seed(seed)
        print("seed", seed, "done", flush=True)
    unit600 = [rows[str(s)]["arms"]["UNIT600"]["gain"] for s in SEEDS]
    unit300 = [rows[str(s)]["arms"]["UNIT300"]["gain"] for s in SEEDS]
    head300 = [rows[str(s)]["arms"]["HEAD300"]["gain"] for s in SEEDS]
    utilities = [rows[str(s)]["arms"]["UNIT600"]["utility"] for s in SEEDS]
    mean600 = sum(unit600) / len(unit600)
    mean300 = sum(unit300) / len(unit300)
    meanhead = sum(head300) / len(head300)
    positive = sum(1 for g in unit600 if g >= 0.05)
    gates = {
        "G0_base": all(rows[str(s)]["base_commit_acc"] >= 0.70 for s in SEEDS),
        "G1_gain": mean600 >= 0.10 and positive >= 3,
        "G2_vs_head": mean600 >= meanhead + 0.03,
        "G3_utility": min(utilities) >= 0.95,
        "G4_budget": (time.process_time() - started) <= 700.0,
    }
    if not gates["G0_base"]:
        verdict = "actuq5_base_failed"
    elif not gates["G1_gain"]:
        verdict = "actuq5_feedback_weak"
    elif not gates["G2_vs_head"]:
        verdict = "actuq5_head_sufficient"
    elif not gates["G3_utility"]:
        verdict = "actuq5_utility_low"
    elif not gates["G4_budget"]:
        verdict = "budget_stop"
    else:
        verdict = "actuq5_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-actuq5-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": SEEDS,
        "rows": rows,
        "mean_unit600": mean600,
        "mean_unit300": mean300,
        "mean_head300": meanhead,
        "positive_seeds": positive,
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
                    "mean_unit600",
                    "mean_unit300",
                    "mean_head300",
                    "positive_seeds",
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

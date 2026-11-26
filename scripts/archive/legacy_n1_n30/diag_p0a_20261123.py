"""P0-A diagnostic: why newborn units contribute little (owner steering 2026-11-23).

Pre-check b: train a probe to convergence on the FROZEN old p24 representation of the
revision stream.  If the probe is already near the target accuracy, the task does not
require a new representation (pressure absent) and no growth mechanism can demonstrate
newborn contribution on it.

Pre-check a: random-birth newborn units + newborn-only readout, budget curve
8/50/200/800 interactions on the revision stream.  If accuracy reaches >=0.75 at some
budget the issue is speed; if it stays near chance at 800 the newborn units lack
information access.

Mechanism A: allow lateral connections w[new, old] / write_state[new, old] gradients
while keeping w[old, new] frozen (old functionality bitwise protected), 200 interactions.

Exploratory diagnostic only: no capability claim; zero check/final.
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

from scripts.training.verify_taiji_24s_f1_20261122 import (
    BLOCKS,
    FIT,
    FIT_OFFSET,
    SEED,
    base_step,
    make_revision_stream,
)
from scripts.training.verify_taiji_24s_f1c_20261122 import (
    OLD_N,
    P24_OLD,
    _forward_states,
    mask_newborn_grads,
    newborn_params,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0a_diag_20261123.json"
NEWBORN = 24
BASE_STEPS = 1200
BASE_BATCH = 48
BASE_LR = 0.007
BASE_WD = 1e-4
GRAPH_LR = 0.01
HEAD_LR = 0.02
BUDGETS = (8, 50, 200, 800)
TARGET = 0.75
PRESSURE_MARGIN = 0.20


def build_base():
    torch.manual_seed(SEED)
    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in BLOCKS.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(True)
    subj.train_mode()
    opt = torch.optim.Adam(subj.params(), lr=BASE_LR, weight_decay=BASE_WD)
    for k in range(BASE_STEPS):
        b = _make_p24_stream(BASE_BATCH, SEED * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        base_step(opt, loss, subj)
    return subj


def grow_random(base_subj):
    subj = copy.deepcopy(base_subj)
    new_ids = subj.graph.grow_units_neutral_(2)
    newn = subj.graph.selective_units.n_units
    subj.n_units = newn
    for name in ("cn", "p24", "p27"):
        pad = torch.zeros(newn)
        pad[:OLD_N] = subj.masks[name]
        subj.masks[name] = pad
    rev_mask = torch.zeros(newn)
    rev_mask[P24_OLD] = 1.0
    rev_mask[OLD_N:] = 1.0
    old_mask = torch.zeros(newn)
    old_mask[P24_OLD] = 1.0
    subj.graph.set_task_route("p24", P24_OLD + new_ids)
    return subj, rev_mask, old_mask


def make_rev_head():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 1017)
        head = PhaseHead(2 * NEWBORN, 5, 64)
    return head


def rev_logits(subj, ticks, rev_mask, head):
    z, a = _forward_states(subj, ticks, rev_mask)
    return head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1))


def eval_rev(subj, ticks, labels, rev_mask, head) -> float:
    subj.eval_mode()
    with torch.no_grad():
        pred = rev_logits(subj, ticks, rev_mask, head)[:, :4].argmax(-1)
    return float((pred == labels).float().mean())


def lateral_mask_grads(subj, old_n: int) -> None:
    """Keep old rows/params frozen but allow newborn -> old (columns) gradients."""
    for name, p in subj.graph.selective_units.named_parameters():
        if p.grad is None:
            continue
        if name in ("w", "write_state.weight", "write_in.weight"):
            p.grad[:old_n, :] = 0.0
        elif name == "p":
            p.grad[:, :old_n] = 0.0
        else:
            p.grad[:old_n] = 0.0


def train_newborn(subj, head, rev_mask, budget: int, lateral: bool) -> float:
    head_opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    graph_opt = torch.optim.Adam(newborn_params(subj), lr=GRAPH_LR)
    for i in range(budget):
        b = make_revision_stream(96, SEED * 77000 + i)
        head.train()
        logits = rev_logits(subj, b["ticks"], rev_mask, head)
        loss = F.cross_entropy(logits[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            logits[:, 4], torch.zeros_like(logits[:, 4])
        )
        head_opt.zero_grad(set_to_none=True)
        graph_opt.zero_grad(set_to_none=True)
        loss.backward()
        if lateral:
            lateral_mask_grads(subj, OLD_N)
        else:
            mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + newborn_params(subj), 1.0)
        graph_opt.step()
        head_opt.step()
    probe = make_revision_stream(FIT, SEED + 444 + FIT_OFFSET)
    return eval_rev(subj, probe["ticks"], probe["val_labels"], rev_mask, head)


def probe_pressure(base_subj):
    pre_mask = torch.zeros(OLD_N)
    for i in P24_OLD:
        pre_mask[i] = 1.0
    train = make_revision_stream(1024, SEED * 66000 + 1)
    test = make_revision_stream(256, SEED + 444 + FIT_OFFSET)
    base_subj.eval_mode()
    with torch.no_grad():
        z_tr, a_tr = _forward_states(base_subj, train["ticks"], pre_mask)
        z_te, a_te = _forward_states(base_subj, test["ticks"], pre_mask)
    x_tr = torch.cat(
        [z_tr[:, P24_OLD[0] : P24_OLD[-1] + 1], a_tr[:, P24_OLD[0] : P24_OLD[-1] + 1]], dim=-1
    )
    x_te = torch.cat(
        [z_te[:, P24_OLD[0] : P24_OLD[-1] + 1], a_te[:, P24_OLD[0] : P24_OLD[-1] + 1]], dim=-1
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 2101)
        probe = torch.nn.Sequential(
            torch.nn.Linear(x_tr.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)
        )
    opt = torch.optim.Adam(probe.parameters(), lr=7e-3)
    n = x_tr.shape[0]
    for _k in range(1500):
        idx = torch.randint(0, n, (64,))
        loss = F.cross_entropy(probe(x_tr[idx]), train["val_labels"][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = float((probe(x_tr).argmax(-1) == train["val_labels"]).float().mean())
        test_acc = float((probe(x_te).argmax(-1) == test["val_labels"]).float().mean())
    return {"train_acc": train_acc, "test_acc": test_acc, "feature_dim": int(x_tr.shape[-1])}


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    pre_mask = torch.zeros(OLD_N)
    for i in P24_OLD:
        pre_mask[i] = 1.0
    base_eval = float(
        (
            base.logits("p24", make_revision_stream(256, SEED + 444 + FIT_OFFSET)["ticks"])[
                :, :4
            ].argmax(-1)
            == make_revision_stream(256, SEED + 444 + FIT_OFFSET)["val_labels"]
        )
        .float()
        .mean()
    )
    pressure = probe_pressure(base)
    grown, rev_mask, old_mask = grow_random(base)

    curve = {}
    for budget in BUDGETS:
        arm = copy.deepcopy(grown)
        head = make_rev_head()
        curve[budget] = train_newborn(arm, head, rev_mask, budget, lateral=False)

    lateral = {}
    for budget in (200, 800):
        arm = copy.deepcopy(grown)
        head = make_rev_head()
        lateral[budget] = train_newborn(arm, head, rev_mask, budget, lateral=True)

    pr = pressure["test_acc"]
    pressure_absent = pr > TARGET - PRESSURE_MARGIN
    best_budget = max(curve, key=lambda k: curve[k])
    speed_issue = curve[best_budget] >= TARGET
    lateral_gain = lateral[200] - curve[200]
    verdict = (
        "p0a_pressure_absent"
        if pressure_absent
        else ("p0a_speed_issue" if speed_issue else "p0a_info_issue")
    )
    payload = {
        "schema": "taiji-p0a-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "base_rev_acc_random_head": base_eval,
        "probe": pressure,
        "probe_interval": {"target": TARGET, "margin": PRESSURE_MARGIN},
        "budget_curve": {str(k): v for k, v in curve.items()},
        "lateral_curve": {str(k): v for k, v in lateral.items()},
        "lateral_gain_at_200": lateral_gain,
        "verdict": verdict,
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "base_rev_acc_random_head",
                    "probe",
                    "budget_curve",
                    "lateral_curve",
                    "lateral_gain_at_200",
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

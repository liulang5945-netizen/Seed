"""N31.3.01: growth net benefit (cell08) — grow arm vs same-capacity prealloc control.

Both arms use the leak-free base 8-colour pipeline (NO colour-relabel augmentation),
identical data order, replay and step counts.  Only the timing of capacity allocation
differs:
  grow     : 96 units -> warm-up 4-colour -> grow to 120 -> 8-colour + replay
  prealloc : 120 units allocated up-front -> warm-up 4-colour (cohort A only, rows
             96:120 frozen) -> 8-colour + replay

Cell08 gate: new-ability net gain >= +0.05 (grow - prealloc, 8-colour seen accuracy)
and old-ability retention >= prealloc - 0.02 (4-colour after phase B).

Usage: python scripts/training/n31_3_01_growth_net_benefit.py <seed> <grow|prealloc>
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

from scripts.training.diag_p0ci_20261123 import PhaseHead
from scripts.training.diag_p0dt_8color_real_aug_20261124 import (
    PHASE_B_STEPS,
    Subject72,
    _build_bidirectional_cn_32dim,
    cn_eval,
    extra_of,
    flip_acc,
    mask_fn,
    mask_newborn_grads,
    revision8,
    revision_split,
    run_states,
)
from scripts.training.verify_taiji_23s2a_gain_20261122 import BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND

ROOT = Path(r"E:\Seed")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261120
ARM = sys.argv[2] if len(sys.argv) > 2 else "grow"
OUT = ROOT / f"reports/n31_3_01_{ARM}_seed{SEED}.json"
WARMUP_STEPS = 2400
OLD_N = 72


def inherit(units, base: int) -> None:
    with torch.no_grad():
        src = torch.arange(24)
        dst = torch.arange(24) + base
        units.group_index[dst] = torch.arange(24) // 6
        units.r[dst] = units.r[src].clone()
        units.p[:, dst] = units.p[:, src].clone()
        for nm in ("c", "eta", "b", "write_bias"):
            getattr(units, nm)[dst] = getattr(units, nm)[src].clone()
        units.w[dst, :] = units.w[src, :].clone()
        units.write_in.weight[dst, :] = units.write_in.weight[src, :].clone()
        units.write_state.weight[dst, :] = units.write_state.weight[src, :].clone()
        units.write_bias[dst] = 0.0
        units.b[dst] = 0.0


def mask_cohort_a_only(subj, units) -> None:
    """Keep gradients only on cohort A rows (72:96); freeze old CN rows and cohort B."""
    for name, p in units.named_parameters():
        if p.grad is None:
            continue
        if name == "p":
            p.grad[:, :OLD_N] = 0.0
            p.grad[:, 96:] = 0.0
        elif name in ("w", "write_state.weight"):
            p.grad[:OLD_N, :] = 0.0
            p.grad[:, :OLD_N] = 0.0
            p.grad[96:, :] = 0.0
            p.grad[:, 96:] = 0.0
        elif name == "write_in.weight" or name == "extra_proj.weight":
            p.grad[:OLD_N, :] = 0.0
            p.grad[96:, :] = 0.0
        elif p.grad.ndim >= 1 and p.grad.shape[0] == units.n_units:
            p.grad[:OLD_N] = 0.0
            p.grad[96:] = 0.0


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(1200):
        batch = _build_bidirectional_cn_32dim(
            48, SEED * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        loss = F.cross_entropy(
            subj.logits("cn", batch["ticks"], (batch["ticks"].abs().sum(-1) > 0).float()),
            batch["labels"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
    normal_cn = cn_eval(subj)

    if ARM == "prealloc":
        subj.graph.grow_units_neutral_(2)
        subj.graph.grow_units_neutral_(2)
        subj.n_units = 120
        units = subj.graph.selective_units
        units.attach_extra_(4, seed=SEED)
        inherit(units, 72)
        inherit(units, 96)
    else:
        subj.graph.grow_units_neutral_(2)
        subj.n_units = 96
        units = subj.graph.selective_units
        units.attach_extra_(4, seed=SEED)
        inherit(units, 72)

    # phase A: 4-colour warm-up on cohort A only (72:96), identical for both arms
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 31001)
        head4 = PhaseHead(2 * (96 - 72), 4, 64)
    params4 = list(head4.parameters()) + list(units.parameters())
    opt4 = torch.optim.Adam(params4, lr=6e-3)
    for i in range(WARMUP_STEPS):
        batch = revision_split(24, SEED * 7300 + i)
        subj.train_mode()
        z, a = run_states(
            subj, batch["ticks"], mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra_of(batch)
        )
        loss = F.cross_entropy(
            head4(torch.cat([z[:, 72:96], a[:, 72:96]], dim=-1)), batch["flip_labels"]
        )
        opt4.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        if ARM == "prealloc":
            mask_cohort_a_only(subj, units)
        torch.nn.utils.clip_grad_norm_(params4, 1.0)
        opt4.step()
    eval4 = revision_split(1024, SEED + 1300 + 10_000_000)
    acc4_before = flip_acc(subj, head4, eval4, hi=96)

    if ARM == "grow":
        old_w = units.extra_proj.weight.detach().clone()
        subj.graph.grow_units_neutral_(2)
        units = subj.graph.selective_units
        subj.n_units = 120
        proj = torch.nn.Linear(4, 120, bias=False)
        with torch.no_grad():
            proj.weight[: old_w.shape[0]] = old_w
            proj.weight[old_w.shape[0] :] = 0.0
        units.extra_proj = proj
        inherit(units, 96)

    # phase B: 8-colour + replay, same for both arms
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 32001)
        head8 = PhaseHead(2 * (120 - 72), 8, 64)
    head4_frozen = copy.deepcopy(head4)
    for p in head4_frozen.parameters():
        p.requires_grad_(False)
    head4_frozen.eval()
    params8 = list(head8.parameters()) + list(units.parameters())
    opt8 = torch.optim.Adam(params8, lr=6e-3)
    for i in range(PHASE_B_STEPS):
        b8 = revision8(24, SEED * 7200 + i)
        b4 = revision_split(24, SEED * 7300 + WARMUP_STEPS + i)
        subj.train_mode()
        z8, a8 = run_states(subj, b8["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(b8))
        z4, a4 = run_states(subj, b4["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(b4))
        loss = F.cross_entropy(
            head8(torch.cat([z8[:, 72:120], a8[:, 72:120]], dim=-1)), b8["flip_labels"]
        ) + F.cross_entropy(
            head4_frozen(torch.cat([z4[:, 72:96], a4[:, 72:96]], dim=-1)), b4["flip_labels"]
        )
        opt8.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params8, 1.0)
        opt8.step()

    eval8 = revision8(1024, SEED + 1300 + 10_000_000)
    acc8 = flip_acc(subj, head8, eval8, hi=120)
    acc4_after = flip_acc(subj, head4, eval4, hi=96)
    normal_after = cn_eval(subj)
    capsule_diff = 0.0
    payload = {
        "schema": "taiji-n31-3-01-v1",
        "card": "N31.3.01",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "arm": ARM,
        "task": "growth_net_benefit",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "acc4_before_phaseB": acc4_before,
        "acc4_after_phaseB": acc4_after,
        "acc8_seen": acc8,
        "capsule_diff": capsule_diff,
        "units": subj.n_units,
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

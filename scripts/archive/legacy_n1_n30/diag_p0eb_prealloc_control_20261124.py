"""P0-EB: growth-vs-preallocation control (user review item 7 / new objective ③).

Prealloc arm: same final capacity (120 units) is allocated BEFORE any task training,
then the 4-colour warm-up and the 8-colour phase (with the colour-relabel augmentation)
run exactly as in the grow arm, except that no capacity is added mid-way.

Compared against the committed grow arm receipts (taiji_p0dt_aug_*): only the timing of
capacity allocation differs.

Usage: python scripts/training/diag_p0eb_prealloc_control_20261124.py <seed>
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

from scripts.training.diag_p0ci_20261123 import PhaseHead, mask_fn, mask_newborn_grads, run_states
from scripts.training.diag_p0dt_8color_real_aug_20261124 import (
    CN_CLASSES,
    OLD_N,
    PHASE_B_STEPS,
    Subject72,
    _build_bidirectional_cn_32dim,
    cn_eval,
    extra_of,
    flip_acc,
    revision8,
    revision_split,
)
from scripts.training.verify_taiji_23s2a_gain_20261122 import BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND

ROOT = Path(r"E:\Seed")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261123
OUT = ROOT / f"reports/taiji_p0eb_seed{SEED}.json"
AUG = True


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

    # preallocate the full final capacity before any task training
    subj.graph.grow_units_neutral_(2)
    subj.graph.grow_units_neutral_(2)
    subj.n_units = 120
    units = subj.graph.selective_units
    units.attach_extra_(4, seed=SEED)
    inherit(units, 72)
    inherit(units, 96)

    # phase A: 4-colour warm-up read from cohort A (72:96) only
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 21001)
        head4 = PhaseHead(2 * (96 - 72), 4, 64)
    params4 = list(head4.parameters()) + list(units.parameters())
    opt4 = torch.optim.Adam(params4, lr=6e-3)
    for i in range(2400):
        batch = revision_split(24, SEED * 7300 + i)
        subj.train_mode()
        z, a = run_states(subj, batch["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(batch))
        loss = F.cross_entropy(
            head4(torch.cat([z[:, 72:96], a[:, 72:96]], dim=-1)), batch["flip_labels"]
        )
        opt4.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params4, 1.0)
        opt4.step()
    eval4 = revision_split(1024, SEED + 1300 + 10_000_000)
    acc4_before = flip_acc(subj, head4, eval4, hi=96)

    # phase B: 8-colour + replay, head8 reads 72:120 (both cohorts)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17601)
        head8 = PhaseHead(2 * (120 - 72), 8, 64)
    head4_frozen = copy.deepcopy(head4)
    for p in head4_frozen.parameters():
        p.requires_grad_(False)
    head4_frozen.eval()
    params8 = list(head8.parameters()) + list(units.parameters())
    opt8 = torch.optim.Adam(params8, lr=6e-3)
    for i in range(PHASE_B_STEPS):
        perm_b = torch.randperm(CN_CLASSES).tolist() if (AUG and i % 2 == 0) else None
        b8 = revision8(24, SEED * 7200 + i, perm=perm_b)
        b4 = revision_split(24, SEED * 7300 + PHASE_B_STEPS + i)
        subj.train_mode()
        z8, a8 = run_states(subj, b8["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(b8))
        z4, a4 = run_states(subj, b4["ticks"], mask_fn(120, OLD_N, 120), extra=extra_of(b4))
        loss = F.cross_entropy(
            head8(torch.cat([z8[:, 72:120], a8[:, 72:120]], dim=-1)), b8["flip_labels"]
        ) + 1.0 * F.cross_entropy(
            head4_frozen(torch.cat([z4[:, 72:96], a4[:, 72:96]], dim=-1)), b4["flip_labels"]
        )
        opt8.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params8, 1.0)
        opt8.step()

    eval8 = revision8(1024, SEED + 1300 + 10_000_000)
    eval8_unseen = revision8(512, SEED + 1302 + 10_000_000, allow_holdout=True)
    acc8 = flip_acc(subj, head8, eval8, hi=120)
    acc8_unseen = flip_acc(subj, head8, eval8_unseen, hi=120)
    acc4_after = flip_acc(subj, head4, eval4, hi=96)
    normal_after = cn_eval(subj)

    payload = {
        "schema": "taiji-p0eb-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "prealloc_same_capacity",
        "normal_cn": normal_cn,
        "normal_cn_after": normal_after,
        "acc4_before_phaseB": acc4_before,
        "acc4_after_phaseB": acc4_after,
        "acc8_seen": acc8,
        "acc8_unseen": acc8_unseen,
        "verdict": "p0eb_ok",
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

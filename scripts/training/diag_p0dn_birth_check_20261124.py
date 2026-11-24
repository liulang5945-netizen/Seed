"""P0-DN: corrected re-check of the P0-CS "growth breaks old ability" claim.

The birth-neutrality regression test shows that growth + the manual edits used by
P0-CS keep old-cohort states bitwise identical (|dz|=0).  This script therefore
re-runs the CS scenario with the corrected evaluation: train M on cohort A
(72:96), grow cohort B, then evaluate M on the SAME cohort A features
(hi=96) instead of the mixed/full feature vector used in CS.

Exploratory; zero check/final.
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

from scripts.training.diag_p0ci_20261123 import (
    OLD_N,
    PhaseHead,
    _build_bidirectional_cn_32dim,
    cn_eval,
    extra_of,
    mask_fn,
    multimodal_batch,
    run_states,
)
from scripts.training.verify_taiji_23s2a_gain_20261122 import BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20261120
OUT = ROOT / f"reports/taiji_p0dn_seed{SEED}.json"


def grow_and_inherit(subj, base: int) -> None:
    old_w = subj.graph.selective_units.extra_proj.weight.detach().clone()
    subj.graph.grow_units_neutral_(2)
    units = subj.graph.selective_units
    subj.n_units = units.n_units
    proj = torch.nn.Linear(8, subj.n_units, bias=False)
    with torch.no_grad():
        proj.weight[: old_w.shape[0]] = old_w
        proj.weight[old_w.shape[0] :] = 0.0
    units.extra_proj = proj
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


def acc_cohort(subj, head, batch, lo: int, hi: int) -> float:
    head.eval()
    subj.eval_mode()
    with torch.no_grad():
        z, a = run_states(
            subj, batch["ticks"], mask_fn(subj.n_units, OLD_N, subj.n_units), extra=extra_of(batch)
        )
        feats = torch.cat([z[:, lo:hi], a[:, lo:hi]], dim=-1)
        return float((head(feats).argmax(-1) == batch["labels"]).float().mean())


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

    subj.graph.grow_units_neutral_(2)
    subj.n_units = 96
    subj.graph.selective_units.attach_extra_(8, seed=SEED)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 19901)
        head = PhaseHead(2 * (96 - 72), 4, 64)
    params = list(head.parameters()) + list(subj.graph.selective_units.parameters())
    opt2 = torch.optim.Adam(params, lr=6e-3)
    for i in range(2400):
        batch = multimodal_batch(32, SEED * 9300 + i)
        subj.train_mode()
        z, a = run_states(subj, batch["ticks"], mask_fn(96, OLD_N, 96), extra=extra_of(batch))
        feats = torch.cat([z[:, 72:96], a[:, 72:96]], dim=-1)
        loss = F.cross_entropy(head(feats), batch["labels"])
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in subj.graph.selective_units.named_parameters():
            if p.grad is None:
                continue
            if name == "p":
                p.grad[:, :72] = 0.0
            elif name in ("w", "write_state.weight"):
                p.grad[:72, :] = 0.0
                p.grad[:, :72] = 0.0
            elif name == "write_in.weight":
                p.grad[:72, :] = 0.0
            elif (
                p.grad.ndim >= 1
                and p.grad.shape[0] == 96
                or p.grad.ndim >= 2
                and p.grad.shape[0] == 96
            ):
                p.grad[:72] = 0.0
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt2.step()
    eval_batch = multimodal_batch(1024, SEED + 2400 + 10_000_000)
    m_before = acc_cohort(subj, head, eval_batch, 72, 96)

    grow_and_inherit(subj, base=96)  # 96 -> 120, manual edits as in P0-CS
    m_after = acc_cohort(subj, head, eval_batch, 72, 96)
    payload = {
        "schema": "taiji-p0dn-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "task": "birth_neutral_corrected_cs",
        "normal_cn": normal_cn,
        "m_before_growth": m_before,
        "m_after_growth_correct_eval": m_after,
        "m_after_growth_delta": m_after - m_before,
        "units": subj.n_units,
        "verdict": "p0dn_verified" if abs(m_after - m_before) <= 1e-9 else "p0dn_growth_perturbed",
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""P0-Q: which input channel groups does the CN-trained base actually transmit?

Build Subject72, train only the cn block (800 steps, standard Chinese 8-colour task).
Then probe CN-state decodability of a 4-way one-hot flag injected into candidate channel
groups (32:36, 40:44) at a fixed tick, versus channels 0:4 (char space, positive control).
A group with probe accuracy <= 0.55 is effectively suppressed by the base encoding -> a
good place for a NEW evidence-source channel.  Exploratory diagnostic; zero check/final.
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

from scripts.training.verify_taiji_23s2a_gain_20261122 import cn_split
from scripts.training.verify_taiji_24s_f1c_20261122 import _forward_states
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0q_diag_20261123.json"
SEED = 20261120
CN_STEPS = 800
BATCH = 48
GROUPS = {"char_0_4": (0, 4), "grp_32_36": (32, 36), "grp_40_44": (40, 44)}


def flag_batch(n: int, seed: int, group: tuple[int, int]) -> dict:
    base = cn_split(n, seed, "seen")
    ticks = base["ticks"].clone()
    g = torch.Generator().manual_seed(seed + 31337)
    flags = torch.randint(0, 4, (n,), generator=g)
    lo, hi = group
    ticks[:, :, lo:hi] = 0.0
    for i in range(n):
        ticks[i, :, lo + int(flags[i])] = 1.0
    return {"ticks": ticks, "flags": flags}


def states_masked(subj, ticks, mask) -> torch.Tensor:
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, ticks, mask)
    out = torch.cat([z, a], dim=-1)
    return out


def probe(feat: torch.Tensor, labels: torch.Tensor) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 31415)
        net = torch.nn.Sequential(
            torch.nn.Linear(feat.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)
        )
    opt = torch.optim.Adam(net.parameters(), lr=7e-3)
    for _k in range(1500):
        idx = torch.randint(0, feat.shape[0], (64,))
        loss = F.cross_entropy(net(feat[idx]), labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        train = float((net(feat).argmax(-1) == labels).float().mean())
    return {"train": train}


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(CN_STEPS):
        batch = cn_split(BATCH, SEED * 6000 + step, "train")
        loss = F.cross_entropy(
            subj.logits("cn", batch["ticks"], (batch["ticks"].abs().sum(-1) > 0).float()),
            batch["labels"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
    cn_mask = torch.zeros(72)
    cn_mask[:24] = 1.0

    results = {}
    for name, group in GROUPS.items():
        tr = flag_batch(512, SEED + 1400 + 10_000_000, group)
        feat = states_masked(subj, tr["ticks"], cn_mask)
        # probe on the cn-block states only (first 24 units of z and a = 48 dims)
        feat_cn = torch.cat([feat[:, :24], feat[:, 72 : 72 + 24]], dim=-1)
        results[name] = probe(feat_cn, tr["flags"])
    payload = {
        "schema": "taiji-p0q-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "cn_steps": CN_STEPS,
        "probes": results,
        "rule": "group probe <= 0.55 => suppressed (usable for a new source channel)",
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"probes": results, "cost": payload["cost"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

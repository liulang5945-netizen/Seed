"""P0-W: held-out re-test of channel-group suppression on the converged CN base.

Train cn block with the stage1b recipe (1200 steps, normal accuracy via logits).  Inject a
4-way one-hot flag into candidate channel groups (0:4 control, 32:36, 40:44) in three
placements (every tick / only tick 0 / only last tick), then probe CN-state decodability
with a HELD-OUT split (75% train / 25% test).  A suppressed group must have test <= 0.55.
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

from scripts.training.verify_taiji_23s2a_gain_20261122 import BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
from scripts.training.verify_taiji_24s_f1c_20261122 import _forward_states
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    FIT_OFFSET,
    _build_bidirectional_cn_32dim,
)
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0w_diag_20261123.json"
SEED = 20261120
CN_STEPS = 1200
BATCH = 48
GROUPS = {"char_0_4": (0, 4), "grp_32_36": (32, 36), "grp_40_44": (40, 44)}
PLACEMENTS = {"all_ticks": None, "tick0": 0, "last_tick": 13}


def flag_batch(n: int, seed: int, group: tuple[int, int], placement: int | None) -> dict:
    base = _build_bidirectional_cn_32dim(n, seed, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)
    ticks = base["ticks"].clone()
    g = torch.Generator().manual_seed(seed + 31337)
    flags = torch.randint(0, 4, (n,), generator=g)
    lo, hi = group
    ticks[:, :, lo:hi] = 0.0
    ticks_of_interest = range(ticks.shape[1]) if placement is None else [placement]
    for i in range(n):
        for tick in ticks_of_interest:
            ticks[i, tick, lo + int(flags[i])] = 1.0
    return {"ticks": ticks, "flags": flags}


def cn_states48(subj, ticks) -> torch.Tensor:
    mask = torch.zeros(72)
    mask[:24] = 1.0
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, :24], a[:, :24]], dim=-1)


def probe(feat: torch.Tensor, labels: torch.Tensor) -> dict:
    n_train = int(feat.shape[0] * 0.75)
    tr_x, tr_y = feat[:n_train], labels[:n_train]
    te_x, te_y = feat[n_train:], labels[n_train:]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 17401)
        net = torch.nn.Sequential(
            torch.nn.Linear(feat.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)
        )
    opt = torch.optim.Adam(net.parameters(), lr=7e-3)
    for _ in range(1500):
        idx = torch.randint(0, tr_x.shape[0], (64,))
        loss = F.cross_entropy(net(tr_x[idx]), tr_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return {
            "train": float((net(tr_x).argmax(-1) == tr_y).float().mean()),
            "test": float((net(te_x).argmax(-1) == te_y).float().mean()),
        }


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    subj = Subject72(SEED)
    subj.set_masks(True)
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=1e-4)
    subj.train_mode()
    for step in range(CN_STEPS):
        batch = _build_bidirectional_cn_32dim(
            BATCH, SEED * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        loss = F.cross_entropy(
            subj.logits("cn", batch["ticks"], (batch["ticks"].abs().sum(-1) > 0).float()),
            batch["labels"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
    fit = _build_bidirectional_cn_32dim(
        256, SEED + 700 + FIT_OFFSET, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    )
    subj.eval_mode()
    with torch.no_grad():
        cn_acc = float(
            (
                subj.logits("cn", fit["ticks"], (fit["ticks"].abs().sum(-1) > 0).float()).argmax(-1)
                == fit["labels"]
            )
            .float()
            .mean()
        )
    results = {}
    for name, group in GROUPS.items():
        for placement_name, placement in PLACEMENTS.items():
            tr = flag_batch(512, SEED + 1500 + 10_000_000, group, placement)
            results[f"{name}@{placement_name}"] = probe(cn_states48(subj, tr["ticks"]), tr["flags"])
    payload = {
        "schema": "taiji-p0w-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "cn_acc": cn_acc,
        "results": results,
        "rule": "group test <= 0.55 => suppressed on held-out",
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"cn_acc": cn_acc, "results": results, "cost": payload["cost"]}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

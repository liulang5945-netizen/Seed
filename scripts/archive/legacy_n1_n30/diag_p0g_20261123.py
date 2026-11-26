"""P0-G: seek a genuinely pressured task (both probes must be low).

Candidate tasks (14-tick streams, 48 channels):
  PARITY14  channel 0 carries one random bit per tick; label = parity of the 14 bits.
  DELAY4    channel 4:8 carries a random 4-way class ONLY at tick 0; ticks 1..13 carry
            random distractor values in channels 4:8; label = the tick-0 class.
  XORPAIR   channels 32,33 carry two bits per sample (persistent); label = (b1 XOR b2,
            b1 AND b2) mapped to 4 classes (non-linearly separable in 2-D input space).

Probes (train to convergence, 1500 steps, MLP 64 hidden):
  * old p24 representation (mask P24_OLD) on the p24-trained base;
  * birth newborn-unit states after growth (random units, no training).

A task is "pressured" when BOTH probes have test accuracy <= 0.55 (target 0.75 - 0.20).
If a pressured task exists, run a newborn-unit training budget curve {200, 800} to check
learnability.  Exploratory diagnostic; zero check/final.
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

from scripts.training.diag_p0b_20261123 import (
    OLD_N,
    SEED,
    build_base,
    grow_random,
)
from scripts.training.verify_taiji_24s_f1c_20261122 import (
    _forward_states,
    mask_newborn_grads,
    newborn_params,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0g_diag_20261123.json"
TARGET = 0.75
MARGIN = 0.20
NEWBORN = 24
HEAD_LR = 0.02
GRAPH_LR = 0.01


def parity_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n, 14), generator=g).float()
    ticks = torch.zeros(n, 14, 48)
    ticks[:, :, 0] = bits
    labels = (bits.sum(dim=1) % 2).long() * 2  # 0 or 2 (4-way head shares labels with XOR task)
    return ticks, labels, bits


def delay_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    cls = torch.randint(0, 4, (n,), generator=g)
    ticks = torch.zeros(n, 14, 48)
    ticks[:, 1:, 4:8] = torch.rand(n, 13, 4, generator=g)
    ticks[:, 0, 4:8] = 0.0
    for i in range(n):
        ticks[i, 0, 4 + int(cls[i])] = 1.0
    return ticks, cls, None


def xor_pair_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    b1 = torch.randint(0, 2, (n,), generator=g)
    b2 = torch.randint(0, 2, (n,), generator=g)
    ticks = torch.zeros(n, 14, 48)
    ticks[:, :, 32] = b1.float().unsqueeze(1)
    ticks[:, :, 33] = b2.float().unsqueeze(1)
    label = (b1 ^ b2) * 2 + (b1 & b2)
    return ticks, label, None


TASKS = {"PARITY14": parity_batch, "DELAY4": delay_batch, "XORPAIR": xor_pair_batch}


def states_old(subj, ticks):
    mask = torch.zeros(OLD_N)
    mask[24:48] = 1.0
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, 24:48], a[:, 24:48]], dim=-1)


def states_new(subj, ticks):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def probe(feature_fn, task_fn, subj) -> dict:
    train_x, train_y, _ = task_fn(1024, SEED * 77000 + 11)
    test_x, test_y, _ = task_fn(256, SEED + 900 + 10_000_000)
    subj.eval_mode()
    with torch.no_grad():
        f_tr = feature_fn(subj, train_x)
        f_te = feature_fn(subj, test_x)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 9101)
        net = torch.nn.Sequential(
            torch.nn.Linear(f_tr.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)
        )
    opt = torch.optim.Adam(net.parameters(), lr=7e-3)
    for _k in range(1500):
        idx = torch.randint(0, f_tr.shape[0], (64,))
        loss = F.cross_entropy(net(f_tr[idx]), train_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        tr = float((net(f_tr).argmax(-1) == train_y).float().mean())
        te = float((net(f_te).argmax(-1) == test_y).float().mean())
    return {"train": tr, "test": te}


def train_units(subj, task_fn, interactions: int) -> float:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 9201)
        head = PhaseHead(2 * NEWBORN, 4, 64)
    opt = torch.optim.Adam(list(head.parameters()) + list(newborn_params(subj)), lr=GRAPH_LR)
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    for i in range(interactions):
        ticks, labels, _ = task_fn(96, SEED * 93000 + i)
        z, a = _forward_states(subj, ticks, mask)
        logits = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1))
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(newborn_params(subj)), 1.0)
        opt.step()
    ticks, labels, _ = task_fn(256, SEED + 911 + 10_000_000)
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, ticks, mask)
        pred = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)).argmax(-1)
    return float((pred == labels).float().mean())


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    grown = grow_random(base)
    results = {}
    for name, task_fn in TASKS.items():
        results[name] = {
            "old_probe": probe(states_old, task_fn, base),
            "birth_probe": probe(states_new, task_fn, grown),
        }
    pressured = [
        name
        for name, row in results.items()
        if row["old_probe"]["test"] <= TARGET - MARGIN
        and row["birth_probe"]["test"] <= TARGET - MARGIN
    ]
    learnability = {}
    for name in pressured:
        subj = copy.deepcopy(grown)
        learnability[name] = {
            "units_200": train_units(copy.deepcopy(subj), TASKS[name], 200),
            "units_800": train_units(copy.deepcopy(subj), TASKS[name], 800),
        }
    payload = {
        "schema": "taiji-p0g-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "probes": results,
        "pressure_rule": {"target": TARGET, "margin": MARGIN},
        "pressured_tasks": pressured,
        "learnability": learnability,
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: payload[k] for k in ("probes", "pressured_tasks", "learnability", "cost")},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

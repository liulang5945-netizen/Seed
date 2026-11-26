"""P0-N: multi-seed replication of the developmental schedule on the qualified pressure task.

Seeds 20261120/21/22; per seed: base p24 1200 steps, 24 random-birth newborn units with
write_in/r zeroed (P0-H pressure), fixed random adapter routing.
Arms (150 labelled-replay interactions each):
  DEV   write_bias=0; 5x lr for the first 50 interactions, promote, then 1x lr.
  CTRL  1x lr throughout.
Checkpoints 5/10/20/50/150; report interactions-to-0.99 (itos), final acc, transfer/reset
for seed 0 only (cost).  Exploratory diagnostic; zero check/final.
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
    build_base,
    eval_p24,
    grow_random,
    sensor_batch,
)
from scripts.training.diag_p0c_20261123 import make_adapter, sensor_with_adapter
from scripts.training.diag_p0h_20261123 import zero_newborn_input
from scripts.training.verify_taiji_24s_f1c_20261122 import (
    _forward_states,
    mask_newborn_grads,
    newborn_params,
)

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0n_diag_20261123.json"
NEWBORN = 24
BASE_LR = 0.01
DEV_MULT = 5.0
DEV_PHASE = 50
TOTAL = 150
CHECKPOINTS = (5, 10, 20, 50, 150)
SEEDS = (20261120, 20261121, 20261122)
FAST_LR = 0.02


def newborn_states(subj, ticks):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def linear(seed, in_dim=2 * NEWBORN):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 14101)
        return torch.nn.Linear(in_dim, 4)


def eval_acc(head, subj, adapter, seed, n=256) -> float:
    ticks, labels = sensor_batch(n, seed + 958 + 10_000_000)
    ticks = sensor_with_adapter(ticks, adapter)
    subj.eval_mode()
    with torch.no_grad():
        pred = head(newborn_states(subj, ticks)).argmax(-1)
    return float((pred == labels).float().mean())


def train_head(head, subj, adapter, seed, interactions, offset) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=FAST_LR)
    for i in range(interactions):
        ticks, labels = sensor_batch(96, seed * 99700 + offset + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def reset_write_bias(subj) -> None:
    units = subj.graph.selective_units
    with torch.no_grad():
        for name in ("write_bias", "b"):
            if hasattr(units, name):
                getattr(units, name)[OLD_N:] = 0.0


def run_arm(grown, adapter, seed, dev: bool):
    subj = copy.deepcopy(grown)
    if dev:
        reset_write_bias(subj)
    head = linear(seed)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt = torch.optim.Adam(params, lr=BASE_LR)
    curve = {}
    for i in range(1, TOTAL + 1):
        lr = BASE_LR * (DEV_MULT if (dev and i <= DEV_PHASE) else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        ticks, labels = sensor_batch(96, seed * 99701 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if dev and i == DEV_PHASE:
            subj.graph.selective_units.promote_immature_to_mature_()
        if i in CHECKPOINTS:
            curve[str(i)] = eval_acc(head, subj, adapter, seed)
    final = curve[str(TOTAL)]
    itos = next((x for x in CHECKPOINTS if curve[str(x)] >= 0.99), 10**9)
    return {"curve": curve, "final": final, "itos": itos, "subj": subj}


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    rows = {}
    seed0_dev = None
    for seed in SEEDS:
        torch.manual_seed(seed)
        base = build_base()
        p24_before = eval_p24(base)
        grown = grow_random(base)
        zero_newborn_input(grown)
        adapter = make_adapter()
        for p in adapter.parameters():
            p.requires_grad_(False)
        dev = run_arm(copy.deepcopy(grown), adapter, seed, dev=True)
        ctrl = run_arm(copy.deepcopy(grown), adapter, seed, dev=False)
        p24_after = eval_p24(dev["subj"])
        rows[str(seed)] = {
            "dev": {k: dev[k] for k in ("curve", "final", "itos")},
            "ctrl": {k: ctrl[k] for k in ("curve", "final", "itos")},
            "p24_before": p24_before,
            "p24_after": p24_after,
        }
        if seed0_dev is None:
            seed0_dev = dev

    itos_dev = [rows[str(s)]["dev"]["itos"] for s in SEEDS]
    itos_ctrl = [rows[str(s)]["ctrl"]["itos"] for s in SEEDS]
    faster = sum(1 for a, b in zip(itos_dev, itos_ctrl, strict=True) if a < b)
    mean_dev = sum(itos_dev) / len(itos_dev)
    mean_ctrl = sum(itos_ctrl) / len(itos_ctrl)
    # transfer/reset on seed-0 dev subject (cost control)
    seed0_subj = seed0_dev["subj"]
    for p in newborn_params(seed0_subj):
        p.requires_grad_(False)
    head = linear(SEEDS[0])
    adapter0 = make_adapter()
    train_head(head, seed0_subj, adapter0, SEEDS[0], 8, 9000)
    transfer = eval_acc(head, seed0_subj, adapter0, SEEDS[0])
    subj_reset = copy.deepcopy(seed0_subj)
    zero_newborn_input(subj_reset)
    head2 = linear(SEEDS[0])
    train_head(head2, subj_reset, adapter0, SEEDS[0], 8, 9100)
    reset_acc = eval_acc(head2, subj_reset, adapter0, SEEDS[0])
    p24_retention_ok = all(
        (rows[str(s)]["p24_before"] - rows[str(s)]["p24_after"]) <= 0.01 for s in SEEDS
    )
    gates = {
        "G0_faster_majority": faster >= 2,
        "G1_mean_faster": mean_ctrl - mean_dev >= 5.0,
        "G2_finals": all(rows[str(s)]["dev"]["final"] >= 0.90 for s in SEEDS)
        and all(rows[str(s)]["ctrl"]["final"] >= 0.90 for s in SEEDS),
        "G3_transfer": transfer >= 0.75,
        "G4_reset_drop": (transfer - reset_acc) >= 0.15,
        "G5_retention": p24_retention_ok,
    }
    if not gates["G0_faster_majority"] and not gates["G1_mean_faster"]:
        verdict = "p0n_no_acceleration"
    elif not gates["G2_finals"]:
        verdict = "p0n_final_failed"
    elif not gates["G3_transfer"]:
        verdict = "p0n_no_transfer"
    elif not gates["G4_reset_drop"]:
        verdict = "p0n_connections_not_needed"
    elif not gates["G5_retention"]:
        verdict = "p0n_retention_breach"
    else:
        verdict = "p0n_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0n-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": SEEDS,
        "rows": rows,
        "itos_dev": itos_dev,
        "itos_ctrl": itos_ctrl,
        "faster_seeds": faster,
        "mean_itos_dev": mean_dev,
        "mean_itos_ctrl": mean_ctrl,
        "transfer": transfer,
        "reset": reset_acc,
        "reset_drop": transfer - reset_acc,
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
                    "itos_dev",
                    "itos_ctrl",
                    "faster_seeds",
                    "mean_itos_dev",
                    "mean_itos_ctrl",
                    "transfer",
                    "reset",
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

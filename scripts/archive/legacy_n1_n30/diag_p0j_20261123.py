"""P0-J: developmental plasticity scheduling on the P0-H/P0-I qualified pressure task.

Pressure task: 4-way sensor routed into channels 0:8; newborn write_in/r zeroed at birth.
Arms (200 labelled-replay interactions each):
  DEV  write_bias=0 at birth; first 50 interactions at 5x graph lr (immature), then
       promote_immature_to_mature_ and 150 interactions at 1x lr.
  CTRL 200 interactions at 1x lr, no promotion, no write_bias reset.
Checkpoints at 5/10/20/50/200 interactions (shared readout).
Transfer: readout reset to a fresh linear head, 8 interactions (units frozen).
Reset: newborn write_in/r zeroed; fresh linear head, 8 interactions.
Exploratory diagnostic; zero check/final.
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
OUT = ROOT / "reports/taiji_p0j_diag_20261123.json"
NEWBORN = 24
BASE_LR = 0.01
DEV_MULT = 5.0
DEV_PHASE = 50
TOTAL = 200
FAST_LR = 0.02
CHECKPOINTS = (5, 10, 20, 50, 200)
TARGET = 0.75


def newborn_states(subj, ticks):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def linear(in_dim=2 * NEWBORN):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 11101)
        return torch.nn.Linear(in_dim, 4)


def eval_acc(head, subj, adapter, n=256) -> float:
    ticks, labels = sensor_batch(n, SEED + 954 + 10_000_000)
    ticks = sensor_with_adapter(ticks, adapter)
    subj.eval_mode()
    with torch.no_grad():
        pred = head(newborn_states(subj, ticks)).argmax(-1)
    return float((pred == labels).float().mean())


def train_head(head, subj, adapter, interactions: int, offset: int) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=FAST_LR)
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99400 + offset + i)
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


def run_arm(grown, adapter, dev: bool):
    subj = copy.deepcopy(grown)
    if dev:
        reset_write_bias(subj)
    head = linear()
    params = list(head.parameters()) + list(newborn_params(subj))
    curve = {}
    opt = torch.optim.Adam(params, lr=BASE_LR)
    for i in range(1, TOTAL + 1):
        mult = DEV_MULT if (dev and i <= DEV_PHASE) else 1.0
        for g in opt.param_groups:
            g["lr"] = BASE_LR * mult
        ticks, labels = sensor_batch(96, SEED * 99401 + i)
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
            curve[str(i)] = eval_acc(head, subj, adapter)
    final = eval_acc(head, subj, adapter)
    for p in newborn_params(subj):
        p.requires_grad_(False)
    transfer_head = linear()
    train_head(transfer_head, subj, adapter, 8, offset=5000)
    transfer = eval_acc(transfer_head, subj, adapter)
    subj_reset = copy.deepcopy(subj)
    zero_newborn_input(subj_reset)
    reset_head = linear()
    train_head(reset_head, subj_reset, adapter, 8, offset=6000)
    reset_acc = eval_acc(reset_head, subj_reset, adapter)
    return {
        "curve": curve,
        "final": final,
        "transfer": transfer,
        "reset": reset_acc,
        "reset_drop": transfer - reset_acc,
        "subj": subj,
    }


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    p24_before = eval_p24(base)
    grown = grow_random(base)
    zero_newborn_input(grown)
    adapter = make_adapter()
    for p in adapter.parameters():
        p.requires_grad_(False)

    dev = run_arm(copy.deepcopy(grown), adapter, dev=True)
    ctrl = run_arm(copy.deepcopy(grown), adapter, dev=False)
    p24_after = eval_p24(dev["subj"])

    early_idx = "20"
    early_gain = dev["curve"].get(early_idx, 0.0) - ctrl["curve"].get(early_idx, 0.0)
    gates = {
        "G0_early_gain": early_gain >= 0.10,
        "G1_final": dev["final"] >= 0.90 and ctrl["final"] >= 0.90,
        "G2_transfer": dev["transfer"] >= 0.75,
        "G3_reset_drop": dev["reset_drop"] >= 0.15,
        "G4_retention": (p24_before - p24_after) <= 0.01,
    }
    if not gates["G0_early_gain"]:
        verdict = "p0j_no_early_advantage"
    elif not gates["G1_final"]:
        verdict = "p0j_final_failed"
    elif not gates["G2_transfer"]:
        verdict = "p0j_no_transfer"
    elif not gates["G3_reset_drop"]:
        verdict = "p0j_connections_not_needed"
    elif not gates["G4_retention"]:
        verdict = "p0j_retention_breach"
    else:
        verdict = "p0j_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0j-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "dev": {k: dev[k] for k in ("curve", "final", "transfer", "reset", "reset_drop")},
        "ctrl": {k: ctrl[k] for k in ("curve", "final", "transfer", "reset", "reset_drop")},
        "early_gain_at_20": early_gain,
        "p24_before": p24_before,
        "p24_after": p24_after,
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
                    "dev",
                    "ctrl",
                    "early_gain_at_20",
                    "p24_before",
                    "p24_after",
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

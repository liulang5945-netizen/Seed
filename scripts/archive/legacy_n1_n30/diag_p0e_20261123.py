"""P0-E: developmental plasticity scheduling on top of the P0-D two-timescale path.

Same pressure task as P0-D.  Two consolidation arms (200 labelled-replay interactions each,
linear readout):

  DEV  immature newborn units (write_bias set to 0 at birth) trained for 50 interactions
       at 5x graph lr, then promoted to mature and trained for 150 interactions at 1x lr.
  CTRL 200 interactions at 1x lr, no promotion, no write_bias reset.

Metrics: accuracy at 50 interactions (early), final accuracy, transfer (readout reset to a
fresh linear head, 8 interactions), reset of newborn units to birth snapshot (drop), old
p24 retention.  Exploratory diagnostic; zero check/final.
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
    zero_new_sensor_wiring,
)
from scripts.training.diag_p0c_20261123 import make_adapter, sensor_with_adapter
from scripts.training.verify_taiji_24s_f1c_20261122 import mask_newborn_grads, newborn_params
from scripts.training.verify_taiji_24s_f1e_20261122 import budding_snapshot, load_snapshot

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0e_diag_20261123.json"
NEWBORN = 24
SLEEP_LR = 0.01
DEV_MULT = 5.0
DEV_PHASE = 50
TOTAL = 200
FAST_LR = 0.02


def newborn_states(subj, ticks):
    from scripts.training.verify_taiji_24s_f1c_20261122 import _forward_states

    new_mask = torch.zeros(subj.n_units)
    new_mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, new_mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def linear_head():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 7101)
        return torch.nn.Linear(2 * NEWBORN, 4)


def train_head(head, subj, adapter, interactions: int, lr: float = FAST_LR) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99100 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def eval_head(head, subj, adapter, n=256) -> float:
    ticks, labels = sensor_batch(n, SEED + 893 + 10_000_000)
    ticks = sensor_with_adapter(ticks, adapter)
    subj.eval_mode()
    with torch.no_grad():
        pred = head(newborn_states(subj, ticks)).argmax(-1)
    return float((pred == labels).float().mean())


def consolidate(subj, head, adapter, interactions: int, lr_mult: float, offset: int) -> None:
    params = list(newborn_params(subj)) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=SLEEP_LR * lr_mult)
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99101 + offset + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


def reset_write_bias(subj) -> None:
    units = subj.graph.selective_units
    with torch.no_grad():
        for name in ("write_bias", "b"):
            if hasattr(units, name):
                getattr(units, name)[OLD_N:] = 0.0


def run_arm(grown, birth, adapter, dev: bool, offset: int):
    subj = copy.deepcopy(grown)
    if dev:
        reset_write_bias(subj)
    head = linear_head()
    early = 0.0
    if dev:
        consolidate(subj, head, adapter, DEV_PHASE, DEV_MULT, offset)
        early = eval_head(head, subj, adapter)
        subj.graph.selective_units.promote_immature_to_mature_()
        consolidate(subj, head, adapter, TOTAL - DEV_PHASE, 1.0, offset + 1000)
    else:
        consolidate(subj, head, adapter, DEV_PHASE, 1.0, offset)
        early = eval_head(head, subj, adapter)
        consolidate(subj, head, adapter, TOTAL - DEV_PHASE, 1.0, offset + 1000)
    final = eval_head(head, subj, adapter)
    # transfer: readout reset, units frozen
    for p in newborn_params(subj):
        p.requires_grad_(False)
    head2 = linear_head()
    train_head(head2, subj, adapter, 8)
    transfer = eval_head(head2, subj, adapter)
    # reset units to birth snapshot
    subj_reset = copy.deepcopy(subj)
    load_snapshot(subj_reset, birth)
    head3 = linear_head()
    train_head(head3, subj_reset, adapter, 8)
    reset_acc = eval_head(head3, subj_reset, adapter)
    return {
        "early_50": early,
        "final": final,
        "transfer": transfer,
        "reset": reset_acc,
        "reset_drop": transfer - reset_acc,
        "subj": subj,
        "early_only": None,
    }


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    p24_before = eval_p24(base)
    zero_new_sensor_wiring(base)
    grown = grow_random(base)
    birth = budding_snapshot(grown)
    adapter = make_adapter()
    for p in adapter.parameters():
        p.requires_grad_(False)

    dev = run_arm(copy.deepcopy(grown), birth, adapter, dev=True, offset=0)
    ctrl = run_arm(copy.deepcopy(grown), birth, adapter, dev=False, offset=0)
    p24_after = eval_p24(dev["subj"])

    early_gain = dev["early_50"] - ctrl["early_50"]
    gates = {
        "G0_early_gain": early_gain >= 0.05,
        "G1_final": dev["final"] >= 0.90,
        "G2_transfer": dev["transfer"] >= 0.75,
        "G3_reset_drop": dev["reset_drop"] >= 0.15,
        "G4_retention": (p24_before - p24_after) <= 0.01,
    }
    if not gates["G0_early_gain"] and not gates["G1_final"]:
        verdict = "p0e_no_advantage"
    elif not gates["G1_final"]:
        verdict = "p0e_dev_failed"
    elif not gates["G2_transfer"]:
        verdict = "p0e_no_transfer"
    elif not gates["G3_reset_drop"]:
        verdict = "p0e_units_not_needed"
    elif not gates["G4_retention"]:
        verdict = "p0e_retention_breach"
    elif all(gates.values()):
        verdict = "p0e_verified"
    else:
        verdict = "p0e_partial"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0e-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "dev": {k: dev[k] for k in ("early_50", "final", "transfer", "reset", "reset_drop")},
        "ctrl": {k: ctrl[k] for k in ("early_50", "final", "transfer", "reset", "reset_drop")},
        "early_gain": early_gain,
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
                    "early_gain",
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

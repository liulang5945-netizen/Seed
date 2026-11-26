"""P0-D: fast/slow two-timescale consolidation and transfer on the P0-B pressure task.

Setup (all from P0-B/P0-C): base p24 trained, old write_in on new sensor 32:36 zeroed, 24
random-birth newborn units, fixed random adapter routing the sensor into channels 0:8.

Phases:
  1 FAST        fresh LINEAR readout trained 8 interactions on newborn-unit states.
  2 SLEEP       labelled replay consolidation: newborn units + readout trained 200
                interactions (real labels, not distillation).
  3 TRANSFER    readout reset to a fresh LINEAR head trained 8 interactions; accuracy
                >= 0.75 shows the function moved from the fast pathway into the units.
  4 RESET       newborn units reset to the random-birth snapshot, fresh LINEAR readout
                trained 8 interactions; a drop >= 0.15 shows the units carry the function.

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
    zero_new_sensor_wiring,
)
from scripts.training.diag_p0c_20261123 import make_adapter, sensor_with_adapter
from scripts.training.verify_taiji_24s_f1c_20261122 import (
    _forward_states,
    mask_newborn_grads,
    newborn_params,
)
from scripts.training.verify_taiji_24s_f1e_20261122 import budding_snapshot, load_snapshot

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0d_diag_20261123.json"
NEWBORN = 24
FAST_LR = 0.02
SLEEP_LR = 0.01
INTERACTIONS_SLEEP = 200
TARGET = 0.75


def linear_head():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 6101)
        head = torch.nn.Linear(2 * NEWBORN, 4)
    return head


def newborn_states(subj, ticks):
    new_mask = torch.zeros(subj.n_units)
    new_mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, new_mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def train_head(head, subj, adapter, interactions: int) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=FAST_LR)
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99000 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def eval_head(head, subj, adapter, n=256) -> float:
    ticks, labels = sensor_batch(n, SEED + 892 + 10_000_000)
    ticks = sensor_with_adapter(ticks, adapter)
    subj.eval_mode()
    with torch.no_grad():
        pred = head(newborn_states(subj, ticks)).argmax(-1)
    return float((pred == labels).float().mean())


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

    # phase 1: fast linear readout
    subj = copy.deepcopy(grown)
    head = linear_head()
    train_head(head, subj, adapter, 8)
    acc_fast = eval_head(head, subj, adapter)

    # phase 2: sleep consolidation (labelled replay, units + readout)
    params = list(newborn_params(subj)) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=SLEEP_LR)
    for i in range(INTERACTIONS_SLEEP):
        ticks, labels = sensor_batch(96, SEED * 99001 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    acc_sleep = eval_head(head, subj, adapter)

    # phase 3: transfer test (readout reset to a fresh linear head)
    for p in newborn_params(subj):
        p.requires_grad_(False)
    head2 = linear_head()
    train_head(head2, subj, adapter, 8)
    acc_transfer = eval_head(head2, subj, adapter)

    # phase 4: reset newborn units to birth snapshot
    subj_reset = copy.deepcopy(subj)
    load_snapshot(subj_reset, birth)
    head3 = linear_head()
    train_head(head3, subj_reset, adapter, 8)
    acc_reset = eval_head(head3, subj_reset, adapter)

    p24_after = eval_p24(subj)
    drop_reset = acc_transfer - acc_reset
    gates = {
        "G0": acc_fast >= 0.75,
        "G1": acc_sleep >= 0.90,
        "G2": acc_transfer >= 0.75,
        "G3": drop_reset >= 0.15,
        "G4": (p24_before - p24_after) <= 0.01,
    }
    if not gates["G0"]:
        verdict = "p0d_fast_failed"
    elif not gates["G1"]:
        verdict = "p0d_sleep_failed"
    elif not gates["G2"]:
        verdict = "p0d_no_transfer"
    elif not gates["G3"]:
        verdict = "p0d_units_not_needed"
    elif not gates["G4"]:
        verdict = "p0d_retention_breach"
    else:
        verdict = "p0d_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0d-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "phases": {
            "fast_8": acc_fast,
            "sleep_200": acc_sleep,
            "transfer_readout_reset": acc_transfer,
            "reset_units": acc_reset,
        },
        "reset_drop": drop_reset,
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
                    "phases",
                    "reset_drop",
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

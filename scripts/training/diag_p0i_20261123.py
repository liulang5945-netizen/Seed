"""P0-I: fast/slow two-timescale on the P0-H qualified pressure task.

Pressure task (P0-H): 4-way sensor routed by a fixed random adapter into channels 0:8; at
birth the newborn units' write_in/r are zeroed (they cannot read any input), while base
features carry the routed class.

  FAST      linear readout on the CURRENT full-unit features (old units still read the
            routed code) trained 8 interactions.
  SLOW      newborn units learn new input connections + newborn-only readout, 200
            labelled-replay interactions.
  TRANSFER  drop the fast readout; a fresh linear head on newborn states trained 8
            interactions must retain >= 0.75 (function moved into the newborn units).
  RESET     zero the learned write_in/r; fresh linear head 8 interactions; drop >= 0.15.

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
OUT = ROOT / "reports/taiji_p0i_diag_20261123.json"
NEWBORN = 24
FAST_LR = 0.02
SLOW_LR = 0.01
SLOW_INTERACTIONS = 200
TARGET = 0.75


def full_states(subj, ticks):
    mask = torch.ones(subj.n_units)
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z, a], dim=-1)


def newborn_states(subj, ticks):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def linear(in_dim):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 10101)
        return torch.nn.Linear(in_dim, 4)


def train_linear(head, feat_fn, subj, adapter, interactions: int) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=FAST_LR)
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99300 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(feat_fn(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def eval_linear(head, feat_fn, subj, adapter, n=256) -> float:
    ticks, labels = sensor_batch(n, SEED + 953 + 10_000_000)
    ticks = sensor_with_adapter(ticks, adapter)
    subj.eval_mode()
    with torch.no_grad():
        pred = head(feat_fn(subj, ticks)).argmax(-1)
    return float((pred == labels).float().mean())


def train_slow(subj, adapter, interactions: int) -> None:
    head = linear(2 * NEWBORN)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt = torch.optim.Adam(params, lr=SLOW_LR)
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99301 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        z, a = _forward_states(subj, ticks, mask)
        loss = F.cross_entropy(head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


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

    # FAST: direct linear readout on current full-unit features
    fast_head = linear(2 * int(grown.n_units))
    train_linear(fast_head, full_states, grown, adapter, 8)
    acc_fast = eval_linear(fast_head, full_states, grown, adapter)

    # SLOW: newborn units learn new input connections
    subj = copy.deepcopy(grown)
    train_slow(subj, adapter, SLOW_INTERACTIONS)
    acc_slow = eval_linear(linear(2 * NEWBORN), newborn_states, subj, adapter)
    # retrain a fresh newborn readout for the transfer test
    transfer_head = linear(2 * NEWBORN)
    train_linear(transfer_head, newborn_states, subj, adapter, 8)
    acc_transfer = eval_linear(transfer_head, newborn_states, subj, adapter)

    # RESET learned input connections
    subj_reset = copy.deepcopy(subj)
    zero_newborn_input(subj_reset)
    reset_head = linear(2 * NEWBORN)
    train_linear(reset_head, newborn_states, subj_reset, adapter, 8)
    acc_reset = eval_linear(reset_head, newborn_states, subj_reset, adapter)

    p24_after = eval_p24(subj)
    gates = {
        "G0_fast": acc_fast >= 0.75,
        "G1_slow": acc_slow >= 0.75,
        "G2_transfer": acc_transfer >= 0.75,
        "G3_reset_drop": (acc_transfer - acc_reset) >= 0.15,
        "G4_retention": (p24_before - p24_after) <= 0.01,
    }
    if not gates["G0_fast"]:
        verdict = "p0i_fast_failed"
    elif not gates["G1_slow"]:
        verdict = "p0i_slow_failed"
    elif not gates["G2_transfer"]:
        verdict = "p0i_no_transfer"
    elif not gates["G3_reset_drop"]:
        verdict = "p0i_connections_not_needed"
    elif not gates["G4_retention"]:
        verdict = "p0i_retention_breach"
    else:
        verdict = "p0i_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0i-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "phases": {
            "fast_8": acc_fast,
            "slow_200": acc_slow,
            "transfer": acc_transfer,
            "reset": acc_reset,
        },
        "reset_drop": acc_transfer - acc_reset,
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

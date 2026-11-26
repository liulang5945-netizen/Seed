"""P0-M: developmental schedule sweep on the qualified pressure task (P0-H/P0-I).

Grid: lr multiplier {2,5,10} x immature phase {20,50,100} interactions, plus a 1x control.
Total 150 labelled-replay interactions per config; checkpoints at 5/10/20/50/100/150.
For the fastest config (fewest interactions to reach >=0.99): transfer (readout reset to a
fresh linear head, 8 interactions) and reset (write_in/r zeroed, fresh head) checks.
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
OUT = ROOT / "reports/taiji_p0m_diag_20261123.json"
NEWBORN = 24
BASE_LR = 0.01
TOTAL = 150
CHECKPOINTS = (5, 10, 20, 50, 100, 150)
MULTS = (2.0, 5.0, 10.0)
PHASES = (20, 50, 100)
CTRL = "1x"
FAST_LR = 0.02


def newborn_states(subj, ticks):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def linear(in_dim=2 * NEWBORN):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 13101)
        return torch.nn.Linear(in_dim, 4)


def eval_acc(head, subj, adapter, n=256) -> float:
    ticks, labels = sensor_batch(n, SEED + 957 + 10_000_000)
    ticks = sensor_with_adapter(ticks, adapter)
    subj.eval_mode()
    with torch.no_grad():
        pred = head(newborn_states(subj, ticks)).argmax(-1)
    return float((pred == labels).float().mean())


def train_head(head, subj, adapter, interactions: int, offset: int) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=FAST_LR)
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99600 + offset + i)
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


def run_config(grown, adapter, mult: float, phase: int):
    subj = copy.deepcopy(grown)
    if mult > 1.0:
        reset_write_bias(subj)
    head = linear()
    params = list(head.parameters()) + list(newborn_params(subj))
    opt = torch.optim.Adam(params, lr=BASE_LR)
    curve = {}
    for i in range(1, TOTAL + 1):
        lr = BASE_LR * (mult if (mult > 1.0 and i <= phase) else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        ticks, labels = sensor_batch(96, SEED * 99601 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if mult > 1.0 and i == phase:
            subj.graph.selective_units.promote_immature_to_mature_()
        if i in CHECKPOINTS:
            curve[str(i)] = eval_acc(head, subj, adapter)
    return {"curve": curve, "final": curve[str(TOTAL)], "subj": subj}


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

    results = {}
    ctrl = run_config(copy.deepcopy(grown), adapter, 1.0, 0)
    results[CTRL] = {k: ctrl[k] for k in ("curve", "final")}
    best_key = None
    best_itos = 10**9
    for mult in MULTS:
        for phase in PHASES:
            key = f"{mult:g}x_p{phase}"
            res = run_config(copy.deepcopy(grown), adapter, mult, phase)
            results[key] = {k: res[k] for k in ("curve", "final")}
            itos = next((int(c) for c in checkpoints_order() if res["curve"][c] >= 0.99), 10**9)
            if itos < best_itos:
                best_itos = itos
                best_key = key
    # transfer/reset for the fastest non-control config
    best_res = run_config(copy.deepcopy(grown), adapter, *parse(best_key))
    subj = best_res["subj"]
    for p in newborn_params(subj):
        p.requires_grad_(False)
    transfer_head = linear()
    train_head(transfer_head, subj, adapter, 8, offset=7000)
    transfer = eval_acc(transfer_head, subj, adapter)
    subj_reset = copy.deepcopy(subj)
    zero_newborn_input(subj_reset)
    reset_head = linear()
    train_head(reset_head, subj_reset, adapter, 8, offset=8000)
    reset_acc = eval_acc(reset_head, subj_reset, adapter)
    p24_after = eval_p24(subj)

    early_gain = results[best_key]["curve"]["20"] - results[CTRL]["curve"]["20"]
    gates = {
        "G0_fast_enough": best_itos <= 20,
        "G1_early_gain": early_gain >= 0.10,
        "G2_final": results[best_key]["final"] >= 0.90,
        "G3_transfer": transfer >= 0.75,
        "G4_reset_drop": (transfer - reset_acc) >= 0.15,
        "G5_retention": (p24_before - p24_after) <= 0.01,
    }
    if not gates["G0_fast_enough"]:
        verdict = "p0m_slow_best"
    elif not (gates["G1_early_gain"] or gates["G2_final"]):
        verdict = "p0m_no_acceleration"
    elif not gates["G3_transfer"]:
        verdict = "p0m_no_transfer"
    elif not gates["G4_reset_drop"]:
        verdict = "p0m_connections_not_needed"
    elif not gates["G5_retention"]:
        verdict = "p0m_retention_breach"
    else:
        verdict = "p0m_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0m-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "results": results,
        "best": {"config": best_key, "interactions_to_099": best_itos},
        "transfer": transfer,
        "reset": reset_acc,
        "reset_drop": transfer - reset_acc,
        "p24_before": p24_before,
        "p24_after": p24_after,
        "early_gain_at_20": early_gain,
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
                    "results",
                    "best",
                    "transfer",
                    "reset",
                    "reset_drop",
                    "early_gain_at_20",
                    "gates",
                    "verdict",
                    "cost",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


def checkpoints_order():
    return [str(c) for c in CHECKPOINTS]


def parse(key: str):
    mult, phase = key.split("_p")
    return float(mult.rstrip("x")), int(phase)


if __name__ == "__main__":
    raise SystemExit(main())

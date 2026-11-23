"""P0-H: pressure task with a learnable NEW INPUT CONNECTION.

Task: 4-way class carried by sensor channels 32:36 (persistent one-hot), routed by a fixed
random adapter into channels 0:8 (the base encoding transmits those channels), label=class.

Pressure construction: at birth the newborn units' `write_in` rows are set to ZERO (they
cannot read any input, not even the routed code) while old units keep their p24 weights.
Thus birth newborn-unit states carry no class information (birth probe ~ chance), yet the
information is present in the base features and a newborn unit can learn a new input
connection (write_in) during training.

Arms:
  birth probe (MLP to convergence)          -> must be <= 0.55 (pressure)
  train newborn units 200 / 800 interactions -> must reach >= 0.75 / >= 0.90
  reset write_in to zero after training      -> drop >= 0.15 (new connections carry it)
  old p24 retention                          -> drop <= 0.01

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
from scripts.training.verify_taiji_24s_f1c_20261122 import (
    _forward_states,
    mask_newborn_grads,
    newborn_params,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0h_diag_20261123.json"
NEWBORN = 24
TARGET = 0.75
MARGIN = 0.20
GRAPH_LR = 0.01
BUDGETS = (200, 800)


def newborn_states(subj, ticks):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def zero_newborn_input(subj) -> None:
    units = subj.graph.selective_units
    with torch.no_grad():
        units.write_in.weight[OLD_N:] = 0.0
        if hasattr(units, "r"):
            units.r[OLD_N:] = 0.0


def probe_birth(subj, adapter) -> dict:
    train_x, train_y = sensor_batch(1024, SEED * 66010 + 1)
    test_x, test_y = sensor_batch(256, SEED + 951 + 10_000_000)
    train_x = sensor_with_adapter(train_x, adapter)
    test_x = sensor_with_adapter(test_x, adapter)
    subj.eval_mode()
    with torch.no_grad():
        f_tr = newborn_states(subj, train_x)
        f_te = newborn_states(subj, test_x)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 9601)
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


def train_units(subj, adapter, interactions: int) -> float:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 9701)
        head = PhaseHead(2 * NEWBORN, 4, 64)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt = torch.optim.Adam(params, lr=GRAPH_LR)
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99000 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        z, a = _forward_states(subj, ticks, mask)
        logits = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1))
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    test_x, test_y = sensor_batch(256, SEED + 952 + 10_000_000)
    test_x = sensor_with_adapter(test_x, adapter)
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, test_x, mask)
        pred = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)).argmax(-1)
    return float((pred == test_y).float().mean())


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

    birth_probe = probe_birth(grown, adapter)
    results = {}
    for budget in BUDGETS:
        subj = copy.deepcopy(grown)
        acc = train_units(subj, adapter, budget)
        results[str(budget)] = acc
        if budget == BUDGETS[-1]:
            final_subj = subj
            # reset the learned input connections to the birth (zero) state
            units = subj.graph.selective_units
            with torch.no_grad():
                units.write_in.weight[OLD_N:] = 0.0
                if hasattr(units, "r"):
                    units.r[OLD_N:] = 0.0
            results["reset_input"] = train_units(copy.deepcopy(subj), adapter, 0)
    p24_after = eval_p24(final_subj)
    gates = {
        "G0_birth_pressure": birth_probe["test"] <= TARGET - MARGIN,
        "G1_learn_200": results["200"] >= 0.75,
        "G2_learn_800": results["800"] >= 0.90,
        "G3_reset_drop": (results["800"] - results["reset_input"]) >= 0.15,
        "G4_retention": (p24_before - p24_after) <= 0.01,
    }
    if not gates["G0_birth_pressure"]:
        verdict = "p0h_not_pressured"
    elif not gates["G1_learn_200"]:
        verdict = "p0h_not_learnable"
    elif not gates["G2_learn_800"]:
        verdict = "p0h_slow_learning"
    elif not gates["G3_reset_drop"]:
        verdict = "p0h_connections_not_needed"
    elif not gates["G4_retention"]:
        verdict = "p0h_retention_breach"
    else:
        verdict = "p0h_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0h-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "birth_probe": birth_probe,
        "probe_rule": {"target": TARGET, "margin": MARGIN},
        "learn": results,
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
                    "birth_probe",
                    "learn",
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

"""P0-B: construct and validate a pressure task for newborn-unit plasticity.

Task: a NEW sensor channel group (channels 32:36, one-hot class, persistent across the
14 ticks, label = 4-way class) that the old p24 cohort was never wired to: after base
training the old units' write_in columns for these channels are zeroed (simulating a
sensor that did not exist during training).  Old functionality is checked for retention.

Pre-check b: probe on the frozen old p24 representation must be <= 0.55 (target 0.75).
Pre-check a: random-birth newborn units + newborn-only readout, budget curve
{8, 50, 200} interactions; learnability = reaching >= 0.75.

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

from scripts.training.verify_taiji_24s_f1_20261122 import SEED, base_step
from scripts.training.verify_taiji_24s_f1c_20261122 import (
    OLD_N,
    P24_OLD,
    _forward_states,
    mask_newborn_grads,
    newborn_params,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0b_diag_20261123.json"
NEWBORN = 24
BASE_STEPS = 1200
BASE_BATCH = 48
BASE_LR = 0.007
BASE_WD = 1e-4
HEAD_LR = 0.02
GRAPH_LR = 0.01
SENSOR_CH = (32, 36)
BUDGETS = (8, 50, 200)
TARGET = 0.75
PRESSURE_MARGIN = 0.20


def build_base():
    torch.manual_seed(SEED)
    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(True)
    subj.train_mode()
    opt = torch.optim.Adam(subj.params(), lr=BASE_LR, weight_decay=BASE_WD)
    for k in range(BASE_STEPS):
        b = _make_p24_stream(BASE_BATCH, SEED * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        base_step(opt, loss, subj)
    return subj


def sensor_batch(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(n, 14, 48)
    labels = torch.randint(0, 4, (n,), generator=g)
    for i in range(n):
        ticks[i, :, SENSOR_CH[0] + int(labels[i])] = 1.0
    return ticks, labels


def p24_test_batch():
    return _make_p24_stream(256, SEED + 212 + 10_000_000)


def eval_p24(subj) -> float:
    b = p24_test_batch()
    mask = torch.zeros(subj.n_units)
    mask[P24_OLD] = 1.0
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, b["ticks"], mask)
        lg = subj.phase_heads["p24"](torch.cat([z[:, :OLD_N], a[:, :OLD_N]], dim=-1))
    return float((lg[:, :4].argmax(-1) == b["val_labels"]).float().mean())


def zero_new_sensor_wiring(subj) -> None:
    units = subj.graph.selective_units
    with torch.no_grad():
        units.write_in.weight[:OLD_N, SENSOR_CH[0] : SENSOR_CH[1]] = 0.0


def grow_random(base_subj):
    subj = copy.deepcopy(base_subj)
    subj.graph.grow_units_neutral_(2)
    newn = subj.graph.selective_units.n_units
    subj.n_units = newn
    for name in ("cn", "p24", "p27"):
        pad = torch.zeros(newn)
        pad[:OLD_N] = subj.masks[name]
        subj.masks[name] = pad
    return subj


def probe_old(base) -> dict:
    mask = torch.zeros(OLD_N)
    mask[P24_OLD] = 1.0
    train_x, train_y = sensor_batch(1024, SEED * 66000 + 3)
    test_x, test_y = sensor_batch(256, SEED + 777 + 10_000_000)
    base.eval_mode()
    with torch.no_grad():
        z_tr, a_tr = _forward_states(base, train_x, mask)
        z_te, a_te = _forward_states(base, test_x, mask)
    x_tr = torch.cat(
        [z_tr[:, P24_OLD[0] : P24_OLD[-1] + 1], a_tr[:, P24_OLD[0] : P24_OLD[-1] + 1]], -1
    )
    x_te = torch.cat(
        [z_te[:, P24_OLD[0] : P24_OLD[-1] + 1], a_te[:, P24_OLD[0] : P24_OLD[-1] + 1]], -1
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 3101)
        probe = torch.nn.Sequential(
            torch.nn.Linear(x_tr.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)
        )
    opt = torch.optim.Adam(probe.parameters(), lr=7e-3)
    for _k in range(1500):
        idx = torch.randint(0, x_tr.shape[0], (64,))
        loss = F.cross_entropy(probe(x_tr[idx]), train_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr = float((probe(x_tr).argmax(-1) == train_y).float().mean())
        te = float((probe(x_te).argmax(-1) == test_y).float().mean())
    return {"train_acc": tr, "test_acc": te}


def probe_newborn_states(grown) -> dict:
    new_mask = torch.zeros(grown.n_units)
    new_mask[OLD_N:] = 1.0
    train_x, train_y = sensor_batch(1024, SEED * 66000 + 5)
    test_x, test_y = sensor_batch(256, SEED + 779 + 10_000_000)
    grown.eval_mode()
    with torch.no_grad():
        z_tr, a_tr = _forward_states(grown, train_x, new_mask)
        z_te, a_te = _forward_states(grown, test_x, new_mask)
    x_tr = torch.cat([z_tr[:, OLD_N:], a_tr[:, OLD_N:]], -1)
    x_te = torch.cat([z_te[:, OLD_N:], a_te[:, OLD_N:]], -1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 3201)
        probe = torch.nn.Sequential(
            torch.nn.Linear(x_tr.shape[-1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)
        )
    opt = torch.optim.Adam(probe.parameters(), lr=7e-3)
    for _k in range(1500):
        idx = torch.randint(0, x_tr.shape[0], (64,))
        loss = F.cross_entropy(probe(x_tr[idx]), train_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr = float((probe(x_tr).argmax(-1) == train_y).float().mean())
        te = float((probe(x_te).argmax(-1) == test_y).float().mean())
    return {"train_acc": tr, "test_acc": te}


def train_newborn_budget(grown, budget: int) -> float:
    subj = copy.deepcopy(grown)
    new_mask = torch.zeros(subj.n_units)
    new_mask[OLD_N:] = 1.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 4101)
        head = PhaseHead(2 * NEWBORN, 4, 64)
    head_opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    graph_opt = torch.optim.Adam(newborn_params(subj), lr=GRAPH_LR)
    for i in range(budget):
        ticks, labels = sensor_batch(96, SEED * 88000 + i)
        head.train()
        z, a = _forward_states(subj, ticks, new_mask)
        logits = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1))
        loss = F.cross_entropy(logits, labels)
        head_opt.zero_grad(set_to_none=True)
        graph_opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + newborn_params(subj), 1.0)
        graph_opt.step()
        head_opt.step()
    test_x, test_y = sensor_batch(256, SEED + 888 + 10_000_000)
    subj.eval_mode()
    with torch.no_grad():
        z, a = _forward_states(subj, test_x, new_mask)
        pred = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)).argmax(-1)
    return float((pred == test_y).float().mean())


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    p24_before = eval_p24(base)
    zero_new_sensor_wiring(base)
    p24_after_wiring = eval_p24(base)
    probe = probe_old(base)
    grown = grow_random(base)
    newborn_probe = probe_newborn_states(grown)
    curve = {str(b): train_newborn_budget(copy.deepcopy(grown), b) for b in BUDGETS}
    pressure = probe["test_acc"] > TARGET - PRESSURE_MARGIN  # True => pressure absent
    learnable = max(curve.values()) >= TARGET
    if not pressure:
        verdict = "p0b_pressure_present" if learnable else "p0b_pressure_but_not_learnable"
    else:
        verdict = "p0b_no_pressure"
    payload = {
        "schema": "taiji-p0b-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "p24_before_wiring": p24_before,
        "p24_after_wiring_zero": p24_after_wiring,
        "old_p24_retention_drop": p24_before - p24_after_wiring,
        "probe": probe,
        "newborn_state_probe": newborn_probe,
        "probe_rule": {"target": TARGET, "margin": PRESSURE_MARGIN},
        "budget_curve": curve,
        "pressure_present": (not pressure),
        "learnable": learnable,
        "verdict": verdict,
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "p24_before_wiring",
                    "p24_after_wiring_zero",
                    "old_p24_retention_drop",
                    "probe",
                    "budget_curve",
                    "pressure_present",
                    "learnable",
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

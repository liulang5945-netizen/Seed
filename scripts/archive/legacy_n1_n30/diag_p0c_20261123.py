"""P0-C: mechanism B (trainable input adapter for newborn units) on the P0-B pressure task.

Pressure task (from P0-B): new sensor channels 32:36 carry a persistent one-hot class, the
old p24 cohort was never wired to them (write_in columns zeroed after base training), and
the trained base encoding ignores those channels.  Mechanism B gives the newborn units a
trainable adapter that projects the raw sensor class into a channel group the base
encoding does transmit (channels 0:8), trained jointly with the newborn units and the
newborn-only readout.

Arms (same base, 200 interactions each):
  B     adapter + newborn units + head   (adapter trainable)
  N     no adapter (sensor ticks as-is)  (reproduces P0-B: chance)
  F     frozen random adapter + newborn units + head (adapter must learn)

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
from scripts.training.verify_taiji_24s_f1c_20261122 import (
    _forward_states,
    mask_newborn_grads,
    newborn_params,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0c_diag_20261123.json"
NEWBORN = 24
HEAD_LR = 0.02
GRAPH_LR = 0.01
ADAPTER_LR = 0.02
INTERACTIONS = 200
ADAPTER_CH = (0, 8)
TARGET = 0.75


def make_adapter():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 5101)
        adapter = torch.nn.Linear(4, ADAPTER_CH[1] - ADAPTER_CH[0])
    return adapter


def apply_adapter(adapter, onehot: torch.Tensor) -> torch.Tensor:
    code = adapter(onehot)  # (n, 8)
    return code.unsqueeze(1)


def sensor_with_adapter(ticks: torch.Tensor, adapter) -> torch.Tensor:
    with torch.no_grad():
        cls = ticks[:, 0, 32:36]
    code = apply_adapter(adapter, cls)
    out = ticks.clone()
    out[:, :, ADAPTER_CH[0] : ADAPTER_CH[1]] = code.expand(-1, ticks.shape[1], -1)
    return out


def eval_adapter(subj, head, adapter, use_adapter: bool, n=256) -> float:
    ticks, labels = sensor_batch(n, SEED + 891 + 10_000_000)
    new_mask = torch.zeros(subj.n_units)
    new_mask[OLD_N:] = 1.0
    subj.eval_mode()
    with torch.no_grad():
        t = sensor_with_adapter(ticks, adapter) if use_adapter else ticks
        z, a = _forward_states(subj, t, new_mask)
        pred = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)).argmax(-1)
    return float((pred == labels).float().mean())


def train_arm(grown, use_adapter: bool, freeze_adapter: bool) -> dict:
    subj = copy.deepcopy(grown)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 5102)
        head = PhaseHead(2 * NEWBORN, 4, 64)
    adapter = make_adapter() if use_adapter else None
    if adapter is not None and freeze_adapter:
        for p in adapter.parameters():
            p.requires_grad_(False)
    head_opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    graph_params = list(newborn_params(subj))
    adapter_params = (
        list(adapter.parameters()) if (adapter is not None and not freeze_adapter) else []
    )
    graph_opt = torch.optim.Adam(graph_params + adapter_params, lr=GRAPH_LR)
    new_mask = torch.zeros(subj.n_units)
    new_mask[OLD_N:] = 1.0
    for i in range(INTERACTIONS):
        ticks, labels = sensor_batch(96, SEED * 98000 + i)
        if adapter is not None:
            ticks = sensor_with_adapter(ticks, adapter)
        head.train()
        z, a = _forward_states(subj, ticks, new_mask)
        logits = head(torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1))
        loss = F.cross_entropy(logits, labels)
        head_opt.zero_grad(set_to_none=True)
        graph_opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + graph_params + adapter_params, 1.0)
        graph_opt.step()
        head_opt.step()
    acc = eval_adapter(subj, head, adapter, use_adapter)
    return {"acc": acc, "subj": subj, "head": head, "adapter": adapter}


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    p24_before = eval_p24(base)
    zero_new_sensor_wiring(base)
    grown = grow_random(base)

    b = train_arm(copy.deepcopy(grown), use_adapter=True, freeze_adapter=False)
    n = train_arm(copy.deepcopy(grown), use_adapter=False, freeze_adapter=False)
    f = train_arm(copy.deepcopy(grown), use_adapter=True, freeze_adapter=True)

    subj_b = b["subj"]
    p24_after = eval_p24(subj_b)
    units = subj_b.graph.selective_units
    units.set_unit_lifecycle_(range(OLD_N, subj_b.n_units), "dormant")
    lesion_acc = eval_adapter(subj_b, b["head"], b["adapter"], True)
    units.set_unit_lifecycle_(range(OLD_N, subj_b.n_units), "mature")

    gates = {
        "G0": b["acc"] >= 0.90,
        "G1": (b["acc"] - n["acc"]) >= 0.40,
        "G2": f["acc"] <= 0.50,
        "G3": (p24_before - p24_after) <= 0.01,
        "G4": (b["acc"] - lesion_acc) >= 0.40,
    }
    if not gates["G0"]:
        verdict = "p0c_adapter_failed"
    elif not gates["G1"]:
        verdict = "p0c_no_adapter_gain"
    elif not gates["G2"]:
        verdict = "p0c_adapter_not_learned"
    elif not gates["G3"]:
        verdict = "p0c_retention_breach"
    elif not gates["G4"]:
        verdict = "p0c_no_unit_causality"
    else:
        verdict = "p0c_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0c-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "interactions": INTERACTIONS,
        "base_p24": p24_before,
        "p24_after": p24_after,
        "retention_drop": p24_before - p24_after,
        "arms": {"adapter": b["acc"], "no_adapter": n["acc"], "frozen_adapter": f["acc"]},
        "lesion_newborn_dormant": lesion_acc,
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
                    "base_p24",
                    "p24_after",
                    "retention_drop",
                    "arms",
                    "lesion_newborn_dormant",
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

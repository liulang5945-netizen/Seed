"""P0-L: lifecycle retention of the NEWLY LEARNED input connections (P0-H task).

Setup: P0-H qualified pressure task (sensor routed to 0:8, newborn write_in/r zeroed at
birth); train the newborn units + readout for 200 labelled-replay interactions (DEV
schedule from P0-J optional).  Then:
  DORMANT   newborn units set dormant -> accuracy must collapse.
  WAKE      mature again -> accuracy must be restored bitwise (same readout).
  CAPSULE   capsule_v2 save/load into a fresh graph -> forward diff <= 1e-6 and same acc.
  RESET     write_in/r zeroed -> drop >= 0.15 (learned connections carry the function).
Old p24 retention checked throughout.  Exploratory diagnostic; zero check/final.
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
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from taiji.u2_selective_units import (
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
)

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0l_diag_20261123.json"
NEWBORN = 24
SLOW_LR = 0.01
INTERACTIONS = 200


def newborn_states(subj, ticks):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def linear(in_dim=2 * NEWBORN):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 12101)
        return torch.nn.Linear(in_dim, 4)


def graph_states(graph, ticks, mask):
    gates = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], mask.numel())
    out = graph(_channels_seq(ticks), gates, graph.initial_state(batch_size=ticks.shape[0]))
    return out["next_state"]["z"], out["next_state"]["a"]


def eval_acc(head, subj, adapter, n=256) -> float:
    ticks, labels = sensor_batch(n, SEED + 955 + 10_000_000)
    ticks = sensor_with_adapter(ticks, adapter)
    subj.eval_mode()
    with torch.no_grad():
        pred = head(newborn_states(subj, ticks)).argmax(-1)
    return float((pred == labels).float().mean())


def train(subj, adapter) -> torch.nn.Linear:
    head = linear()
    params = list(head.parameters()) + list(newborn_params(subj))
    opt = torch.optim.Adam(params, lr=SLOW_LR)
    for i in range(INTERACTIONS):
        ticks, labels = sensor_batch(96, SEED * 99500 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return head


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

    subj = copy.deepcopy(grown)
    head = train(subj, adapter)
    acc_trained = eval_acc(head, subj, adapter)

    units = subj.graph.selective_units
    units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "dormant")
    acc_dormant = eval_acc(head, subj, adapter)
    units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "mature")
    acc_wake = eval_acc(head, subj, adapter)

    # capsule round trip
    cap = build_capsule_v2(subj.graph)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subj.graph.base_graph), subj.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(cap))
    probe_ticks = sensor_with_adapter(sensor_batch(64, SEED + 956 + 10_000_000)[0], adapter)
    mask = torch.ones(subj.n_units)
    with torch.no_grad():
        z0, a0 = graph_states(subj.graph, probe_ticks, mask)
        z1, a1 = graph_states(fresh, probe_ticks, mask)
    capsule_diff = float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))
    restored = copy.deepcopy(subj)
    apply_capsule_v2(restored.graph, copy.deepcopy(cap))
    acc_restored = eval_acc(head, restored, adapter)

    # reset learned connections on the trained subject
    zero_newborn_input(subj)
    acc_reset = eval_acc(head, subj, adapter)

    p24_after = eval_p24(restored)
    gates = {
        "G0_trained": acc_trained >= 0.90,
        "G1_dormant": acc_dormant <= 0.40,
        "G2_wake": abs(acc_wake - acc_trained) <= 1e-9,
        "G3_capsule": capsule_diff <= 1e-6 and abs(acc_restored - acc_trained) <= 1e-9,
        "G4_reset_drop": (acc_trained - acc_reset) >= 0.15,
        "G5_retention": (p24_before - p24_after) <= 0.01,
    }
    if not gates["G0_trained"]:
        verdict = "p0l_train_failed"
    elif not gates["G1_dormant"]:
        verdict = "p0l_dormancy_ineffective"
    elif not gates["G2_wake"]:
        verdict = "p0l_wake_not_restored"
    elif not gates["G3_capsule"]:
        verdict = "p0l_capsule_mismatch"
    elif not gates["G4_reset_drop"]:
        verdict = "p0l_connections_not_needed"
    elif not gates["G5_retention"]:
        verdict = "p0l_retention_breach"
    else:
        verdict = "p0l_verified"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-p0l-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "trained": acc_trained,
        "dormant": acc_dormant,
        "wake": acc_wake,
        "capsule_diff": capsule_diff,
        "restored": acc_restored,
        "reset": acc_reset,
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
                    "trained",
                    "dormant",
                    "wake",
                    "capsule_diff",
                    "restored",
                    "reset",
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

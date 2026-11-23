"""P0-F: control — can a linear/MLP readout on BIRTH (untrained) newborn units solve the
P0-B pressure task once the sensor is routed (P0-C)?  If yes, the pressure task only tests
routing, not unit plasticity, and P0-D's reset drop needs re-examination.

Arms: birth units (never trained) + linear readout at 8/50/200 interactions; + MLP readout
at 200.  Exploratory diagnostic; zero check/final.
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
    grow_random,
    sensor_batch,
    zero_new_sensor_wiring,
)
from scripts.training.diag_p0c_20261123 import make_adapter, sensor_with_adapter

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_p0f_diag_20261123.json"


def newborn_states(subj, ticks):
    from scripts.training.verify_taiji_24s_f1c_20261122 import _forward_states

    new_mask = torch.zeros(subj.n_units)
    new_mask[OLD_N:] = 1.0
    z, a = _forward_states(subj, ticks, new_mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def eval_head(head, subj, adapter, n=256) -> float:
    ticks, labels = sensor_batch(n, SEED + 894 + 10_000_000)
    ticks = sensor_with_adapter(ticks, adapter)
    subj.eval_mode()
    with torch.no_grad():
        pred = head(newborn_states(subj, ticks)).argmax(-1)
    return float((pred == labels).float().mean())


def train(head, subj, adapter, interactions: int, lr=0.02):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for i in range(interactions):
        ticks, labels = sensor_batch(96, SEED * 99200 + i)
        ticks = sensor_with_adapter(ticks, adapter)
        loss = F.cross_entropy(head(newborn_states(subj, ticks)), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    zero_new_sensor_wiring(base)
    grown = grow_random(base)
    adapter = make_adapter()
    for p in adapter.parameters():
        p.requires_grad_(False)
    subj = copy.deepcopy(grown)  # birth units, never trained

    results = {}
    for n_int in (8, 50, 200):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED + 8101)
            head = torch.nn.Linear(2 * 24, 4)
        train(head, subj, adapter, n_int)
        results[f"linear_{n_int}"] = eval_head(head, subj, adapter)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 8102)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 24, 64), torch.nn.Tanh(), torch.nn.Linear(64, 4)
        )
    train(mlp, subj, adapter, 200)
    results["mlp_200"] = eval_head(mlp, subj, adapter)

    payload = {
        "schema": "taiji-p0f-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "birth_units": results,
        "interpretation": (
            "linear_200 >= 0.90 => the pressure task is linearly decodable from birth units "
            "once routed; unit plasticity is not required for it"
        ),
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"birth_units": results, "cost": payload["cost"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

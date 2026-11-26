"""Smoke test for the unit-level raw-input adapter (`attach_extra_`).

Checks:
  E0  attaching the adapter with zero init does not change forward states (equivalence).
  E1  a 40-interaction training run on the P0-H pressure task, where the newborn units are
      blind (`write_in`/`r` zeroed) and the 4-way class arrives ONLY through the raw
      adapter (`extra`), learns the task (acc >= 0.75 at 40 interactions).
  E2  zeroing the adapter weights after training collapses accuracy (drop >= 0.15), i.e.
      the learned adapter carries the function.
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

from scripts.training.diag_p0b_20261123 import OLD_N, SEED, build_base, grow_random, sensor_batch
from scripts.training.diag_p0h_20261123 import zero_newborn_input
from scripts.training.verify_taiji_24s_f1c_20261122 import mask_newborn_grads, newborn_params
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_adapter_smoke_20261123.json"
NEWBORN = 24


def states_extra(subj, ticks, mask, extra):
    gates = mask.view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], mask.numel())
    out = subj.graph(
        _channels_seq(ticks),
        gates,
        subj.graph.initial_state(batch_size=ticks.shape[0]),
        extra=extra,
    )
    return out["next_state"]["z"], out["next_state"]["a"]


def newborn_states(subj, ticks, extra):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    z, a = states_extra(subj, ticks, mask, extra)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    grown = grow_random(base)
    zero_newborn_input(grown)
    subj = copy.deepcopy(grown)
    subj.graph.selective_units.attach_extra_(4, seed=SEED)

    ticks, labels = sensor_batch(64, SEED + 500)
    extra = ticks[:, :, 32:36]
    mask_all = torch.ones(subj.n_units)
    with torch.no_grad():
        z0, a0 = states_extra(subj, ticks, mask_all, None)
        z1, a1 = states_extra(subj, ticks, mask_all, extra)
    diff_zero_init = float(max((z0 - z1).abs().max(), (a0 - a1).abs().max()))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 16001)
        head = PhaseHead(2 * NEWBORN, 4, 64)
    params = list(head.parameters()) + list(newborn_params(subj))
    opt = torch.optim.Adam(params, lr=0.02)
    for i in range(40):
        b_ticks, b_labels = sensor_batch(96, SEED * 99800 + i)
        b_extra = b_ticks[:, :, 32:36]
        logits = head(newborn_states(subj, b_ticks, b_extra))
        loss = F.cross_entropy(logits, b_labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_newborn_grads(subj, OLD_N)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    test_ticks, test_labels = sensor_batch(256, SEED + 16002 + 10_000_000)
    subj.eval_mode()
    with torch.no_grad():
        acc = float(
            (
                head(newborn_states(subj, test_ticks, test_ticks[:, :, 32:36])).argmax(-1)
                == test_labels
            )
            .float()
            .mean()
        )
    with torch.no_grad():
        proj = subj.graph.selective_units.extra_proj
        proj.weight[OLD_N:] = 0.0
        reset_acc = float(
            (
                head(newborn_states(subj, test_ticks, test_ticks[:, :, 32:36])).argmax(-1)
                == test_labels
            )
            .float()
            .mean()
        )

    gates = {
        "E0_equivalence": diff_zero_init <= 1e-7,
        "E1_learn": acc >= 0.75,
        "E2_adapter_carries": (acc - reset_acc) >= 0.15,
    }
    verdict = "adapter_smoke_verified" if all(gates.values()) else "adapter_smoke_failed"
    cost = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-adapter-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "zero_init_forward_diff": diff_zero_init,
        "train_40_acc": acc,
        "adapter_zeroed_acc": reset_acc,
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
                    "zero_init_forward_diff",
                    "train_40_acc",
                    "adapter_zeroed_acc",
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

"""Debug: is the dormant->mature lifecycle round trip exact on a grown subject?"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch

from scripts.training.diag_p0b_20261123 import OLD_N, SEED, build_base, grow_random, sensor_batch
from scripts.training.verify_taiji_24s_f1c_20261122 import _forward_states

ROOT = Path(r"E:\Seed")
OUT = ROOT / "reports/taiji_lifecycle_debug_20261123.json"


def newborn_states(subj, ticks):
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    with torch.no_grad():
        z, a = _forward_states(subj, ticks, mask)
    return torch.cat([z[:, OLD_N:], a[:, OLD_N:]], dim=-1)


def main() -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    base = build_base()
    subj = grow_random(base)
    subj.graph.selective_units.attach_extra_(4, seed=SEED)
    units = subj.graph.selective_units
    ticks, _ = sensor_batch(32, SEED + 500)
    extra = ticks[:, :, 32:36]
    mask = torch.zeros(subj.n_units)
    mask[OLD_N:] = 1.0
    gates = mask.view(1, 1, -1).expand(32, 14, subj.n_units)

    def fwd():
        with torch.no_grad():
            out = subj.graph(
                __import__(
                    "scripts.training.verify_taiji_algorithm_holistic_20261121",
                    fromlist=["_channels_seq"],
                )._channels_seq(ticks),
                gates,
                subj.graph.initial_state(batch_size=32),
                extra=extra,
            )
            return out["next_state"]["z"], out["next_state"]["a"]

    z0, a0 = fwd()
    lifecycle_before = units.unit_lifecycle.clone()
    state_before = float(torch.cat([z0[:, OLD_N:], a0[:, OLD_N:]], dim=-1).abs().sum())

    units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "dormant")
    z1, a1 = fwd()
    state_dormant = float(torch.cat([z1[:, OLD_N:], a1[:, OLD_N:]], dim=-1).abs().sum())

    units.set_unit_lifecycle_(range(OLD_N, subj.n_units), "mature")
    z2, a2 = fwd()
    state_wake = float(torch.cat([z2[:, OLD_N:], a2[:, OLD_N:]], dim=-1).abs().sum())
    lifecycle_after = units.unit_lifecycle.clone()

    payload = {
        "schema": "taiji-lifecycle-debug-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "n_units": int(subj.n_units),
        "units_n_units": int(units.n_units),
        "state_abs_sum_before": state_before,
        "state_abs_sum_dormant": state_dormant,
        "state_abs_sum_wake": state_wake,
        "wake_equals_before": abs(state_wake - state_before) <= 1e-9,
        "lifecycle_newborn_before": lifecycle_before[OLD_N:].tolist(),
        "lifecycle_newborn_after": lifecycle_after[OLD_N:].tolist(),
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

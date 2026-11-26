"""Zero-training price pass for the 22.5.0 A-parent formal package.

The readout line closed with `readout_gate_earned`, so the formal package runs
under a new identity (pixel-salience, d=32) at a new dose (2000 updates).  The
frozen 22.4.4 price (0.54375 CPU/update) predates d=32, so this measures the
current per-example cost directly on a throwaway model (threads=1, one-off
backward calls, no kept optimizer state), times the v2 fixture generation, and
prices the 3-seed package including the FLOPs-parity baseline leg scaled from
the frozen refreeze contract unit.  Zero formal training.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from scripts.training.run_a_gate_pixel_20261117 import (  # noqa: E402
    FIRST_WEIGHT,
    masked_loss,
)
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_a_gate_package_price_20261119.json"
SEEDS = 3
DOSE = 2000
BATCH = 8
PARITY_REFREEZE_CPU = 417.6  # frozen contract, 2671 baseline updates x 3 seeds
PARITY_REFREEZE_UPDATES_PER_SEED = 2671
NEW_PARITY_UPDATES_PER_SEED = 20874  # 2000 x flops_ratio 10.436519
CHECK_WORKER_CPU = 118.0  # measured 2026-11-19: 113.6/110.5/119.9
FORMAL_LEG_CPU = 150.0  # preflight + save/load roundtrips + fit/recovery evals


def timed(count, fn) -> float:
    start = time.process_time()
    for _ in range(count):
        fn()
    return (time.process_time() - start) / count


def main() -> int:
    torch.set_num_threads(1)
    model = (
        U2Subject(replace(U2Config.small(), seed=20261110, readout="pixel-salience", d=32))
        .metadata_candidate()
        .request_candidate()
    )
    warm = generate_v2("train", 8)
    fixture_start = time.process_time()
    generate_v2("train", 64)
    fixture64_cpu = time.process_time() - fixture_start
    for example in warm:  # autograd/graph warmup, untimed
        masked_loss(model, [example], FIRST_WEIGHT, False, None).backward()
        model.zero_grad(set_to_none=True)
    per_example = timed(
        30,
        lambda: (
            masked_loss(model, [warm[0]], FIRST_WEIGHT, False, None).backward(),
            model.zero_grad(set_to_none=True),
        ),
    )
    per_update = per_example * BATCH
    fixture768 = fixture64_cpu * (768 / 64)
    train_leg = DOSE * per_update
    fixture_leg = 9 * fixture768  # 8 workers + main digest regen per leg
    u2_arm_per_seed = train_leg + fixture_leg + CHECK_WORKER_CPU + FORMAL_LEG_CPU
    u2_arm = SEEDS * u2_arm_per_seed
    parity = PARITY_REFREEZE_CPU * (NEW_PARITY_UPDATES_PER_SEED / PARITY_REFREEZE_UPDATES_PER_SEED)
    package = u2_arm + parity
    payload = {
        "schema": "taiji-u2-22-5-a-gate-package-price-v1",
        "card": "22.5.0 formal package pricing",
        "protocol": {
            "dose_per_seed": DOSE,
            "batch": BATCH,
            "seeds": SEEDS,
            "identity": "pixel-salience + d=32 (readout_gate_earned)",
        },
        "measured": {
            "per_example_cpu": round(per_example, 6),
            "per_update_cpu": round(per_update, 6),
            "fixture64_cpu": round(fixture64_cpu, 4),
            "fixture768_cpu_estimated": round(fixture768, 4),
            "threads": 1,
        },
        "sourced": {
            "check_worker_cpu": CHECK_WORKER_CPU,
            "formal_leg_cpu_allowance": FORMAL_LEG_CPU,
            "parity_refreeze_cpu": PARITY_REFREEZE_CPU,
            "new_parity_updates_per_seed": NEW_PARITY_UPDATES_PER_SEED,
        },
        "price": {
            "train_leg_cpu_per_seed": round(train_leg, 2),
            "fixture_leg_cpu_per_seed": round(fixture_leg, 2),
            "u2_arm_cpu_per_seed": round(u2_arm_per_seed, 2),
            "u2_arm_cpu_total": round(u2_arm, 2),
            "parity_baseline_cpu": round(parity, 2),
            "package_cpu_total": round(package, 2),
            "cap_cpu_proposed": round(package * 1.25, 2),
            "u2_updates_total": DOSE * SEEDS,
            "stage_22_5_frozen_capacity": {"updates": 9808, "cpu": 16000.0},
        },
        "formal_training_authority": False,
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps(payload["measured"], ensure_ascii=False))
    print(json.dumps(payload["price"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

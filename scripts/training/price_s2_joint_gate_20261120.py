"""Zero-training price pass for the 22.5.1 joint gate (S2).

Measures one full organ-training arm (256 steps, batch 2, cached parent
log-probabilities), one full decode-matrix measurement (A/B/D/multi/supersede
on 24 items), and the independent-process recovery allowance on throwaway
models.  Then prices the pre-declared 3 child seeds x 3 training arms + cuts +
overwrite probe layout.  Zero formal training, zero check/final reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from scripts.training.run_s2_joint_gate_20261120 import measure, train_arm  # noqa: E402
from scripts.training.s0_parent_loader_20261119 import (  # noqa: E402
    CHECKPOINTS,
    load_historical,
)

OUT = ROOT / "reports/taiji_u2_22_5_1_joint_gate_price_20261120.json"
PARENT_SEED = 20261112
CHILD_SEEDS = (20261110, 20261111, 20261112)
STEPS = 256
BATCH = 2
FEEDBACK_EXAMPLES = 64
FIT_EXAMPLES = 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.process_time()
    fit = generate_v2("fit", FIT_EXAMPLES)
    feedback = generate_v2("train", FEEDBACK_EXAMPLES)
    parent, _payload, _audit = load_historical(CHECKPOINTS[PARENT_SEED])

    begin = time.process_time()
    child, info = train_arm(parent, CHILD_SEEDS[0], "main", feedback)
    arm_cpu = time.process_time() - begin
    begin = time.process_time()
    reading = measure(child, parent, fit)
    measure_cpu = time.process_time() - begin

    n = reading["n"]
    eval_cpu_arm = measure_cpu
    run_per_seed = 3 * arm_cpu + 7 * eval_cpu_arm + 0.5 * eval_cpu_arm
    preflight_cpu = 5.0
    run_total = len(CHILD_SEEDS) * (run_per_seed + preflight_cpu)
    verify_total = 0.1 * run_total + 3.0
    recover_total = len(CHILD_SEEDS) * 4.0
    engineering = 15.0
    package = run_total + verify_total + recover_total + engineering
    cap = package * 1.25
    payload = {
        "schema": "taiji-u2-22-5-1-joint-gate-price-v1",
        "card": "22.5.1 joint gate (fit face) S2 pricing",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "measured": {
            "arm_train_cpu_256_steps_batch2": round(arm_cpu, 3),
            "measure_cpu_24_items": round(measure_cpu, 3),
            "items_per_face": n,
            "organ_mix_after_arm": info.get("organ_mix"),
        },
        "protocol": {
            "parent_seed": PARENT_SEED,
            "parent_checkpoint": str(CHECKPOINTS[PARENT_SEED].relative_to(ROOT)),
            "child_seeds": list(CHILD_SEEDS),
            "feedback_examples": FEEDBACK_EXAMPLES,
            "fit_examples": FIT_EXAMPLES,
            "steps": STEPS,
            "batch_worlds": BATCH,
            "arms": ["main", "frozen_update", "permuted_feedback"],
            "extra_measures": [
                "cut:record_organ",
                "cut:fast",
                "cut:record_injection",
                "overwrite_probe",
            ],
            "fit_face_only": True,
            "check_final_reads": 0,
        },
        "price": {
            "arm_train_cpu": round(arm_cpu, 2),
            "measure_cpu": round(measure_cpu, 2),
            "run_cpu_per_seed": round(run_per_seed + preflight_cpu, 2),
            "run_cpu_total": round(run_total, 2),
            "verify_cpu_total": round(verify_total, 2),
            "recover_cpu_total": round(recover_total, 2),
            "engineering_allowance": engineering,
            "package_cpu": round(package, 2),
            "cap_cpu_proposed": round(cap, 2),
            "wall_minutes_proposed": round(cap / 60 + 5, 1),
        },
        "formal_training_authority": None,
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(json.dumps(payload["price"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

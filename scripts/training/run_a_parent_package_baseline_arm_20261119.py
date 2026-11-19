"""22.5.0 A-parent formal package: FLOPs-parity baseline arm per seed.

Trains the frozen Plan C identity behind the A-task adapter at the
FLOPs-matched dose (168,318 example-updates/seed; measured ratio 10.5199 over
the U2 arm's 2,000 x batch-8 example-steps, FlopCounterMode on identical
examples, prereg D-20261119-03) with the identity's frozen optimizer recipe
(Adam lr 0.001), then reads fit and the 192-world check face with the same
`evaluate()` instrument.  Aggregate-only check release, per the exposure
discipline of the package prereg.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.adapter_a_task_baseline_20261119 import ATaskBaselineAdapter  # noqa: E402
from scripts.training.first_a_support_22_5 import evaluate  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from scripts.training.run_a_gate_c_20261117 import _check_examples  # noqa: E402

WORLDS = 768
DOSE = 168_318  # measured FLOPs parity: 2000 x 8 x 10.5199
LR = 0.001  # frozen identity recipe


def main(seed: int) -> int:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    torch.set_num_threads(1)
    adapter = ATaskBaselineAdapter(seed)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    train = generate_v2("train", WORLDS)
    order = torch.randperm(WORLDS, generator=torch.Generator().manual_seed(seed + 9)).tolist()
    cursor = 0
    last = 0.0
    for step in range(DOSE):
        example = train[order[cursor]]
        optimizer.zero_grad(set_to_none=True)
        loss = adapter.task_loss(example["public"], example["target"])
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        cursor = (cursor + 1) % WORLDS
        if (step + 1) % 50_000 == 0:
            print(f"seed {seed} step {step + 1}/{DOSE} loss {round(last, 4)}", flush=True)
    fit = evaluate(adapter, generate_v2("fit", 32))
    check_examples = _check_examples()
    check_rows = evaluate(adapter, check_examples)
    per_label: dict[str, dict[str, int]] = {}
    for row in check_rows["rows"]:
        bucket = per_label.setdefault(row["target"], {"correct": 0, "denominator": 0})
        bucket["denominator"] += 1
        bucket["correct"] += int(row["correct"])
    report = {
        "schema": "taiji-u2-22-5-a-parent-baseline-arm-v1",
        "card": "22.5.0 A-parent formal package",
        "seed": seed,
        "identity": adapter.architecture_id,
        "flops_parity": {
            "dose": DOSE,
            "ratio_u2_over_adapter": 10.5199,
            "method": "FlopCounterMode per-example on identical examples",
        },
        "optimizer": {"type": "Adam", "lr": LR, "betas": [0.9, 0.999], "eps": 1e-8},
        "last_train_loss": round(last, 6),
        "fit": {"correct": fit["correct"], "denominator": fit["denominator"]},
        "check": {
            "correct": check_rows["correct"],
            "denominator": check_rows["denominator"],
            "accuracy": check_rows["accuracy"],
            "valid_utf8": sum(int(row["valid_utf8"]) for row in check_rows["rows"]),
            "eos": sum(int(row["eos"]) for row in check_rows["rows"]),
            "per_label": per_label,
            "aggregate_only": True,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.perf_counter() - started_wall, 4),
            "threads": 1,
        },
    }
    out = ROOT / f"reports/taiji_u2_a_parent_package_baseline_seed{seed}_20261119.json"
    out.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {
                "seed": seed,
                "fit": fit["correct"],
                "check": check_rows["correct"],
                "check_accuracy": round(check_rows["accuracy"], 4),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1])))

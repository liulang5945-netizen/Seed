"""22.5.0 A-parent formal package: U2 arm per seed (D-20261119-03 authorization).

One seed per invocation: preflight (initial checkpoint roundtrip + gradient
equality on price examples), training via the byte-identical validated parallel
runner (2000 updates, d=32, pixel-salience, lr 5e-3 + warmup + clip, batch 8),
recovery (independent reloads reproduce the subprocess reading), then the
aggregate-only check worker.  Zero read of any partition beyond the frozen
rights: train (via the runner), fit, price, and the 192-world check face.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    load_learner,
    save_learner,
)
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from scripts.training.run_a_gate_pixel_20261117 import (  # noqa: E402
    FIRST_WEIGHT,
    evaluate,
    masked_loss,
)
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

WORLDS = 768
LR = 0.005
DOSE = int(os.environ.get("TAIJI_U2_DOSE", "2000"))
BATCH = 8


def gradients(model, examples):
    model.zero_grad(set_to_none=True)
    loss = masked_loss(model, examples, FIRST_WEIGHT, False, None)
    loss.backward()
    return float(loss.detach()), {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def same_gradient(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(torch.equal(left, right))


def main(seed: int) -> int:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    torch.manual_seed(seed)
    config = replace(U2Config.small(), seed=seed, readout="pixel-salience", d=32)
    model = U2Subject(config).metadata_candidate().request_candidate()
    order = torch.randperm(WORLDS, generator=torch.Generator().manual_seed(seed + 9)).tolist()
    directory = ROOT / "output" / f"a_parent_package_20261119_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    initial = directory / "initial-segment.bin"
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    train_digest = generate_v2("train", WORLDS)
    save_learner(initial, model, optimizer, "native", 0, 0, order, value_digest(train_digest))
    del train_digest
    restored, _, _ = load_learner(initial)
    price = generate_v2("price", 4)
    _, first = gradients(model, price[:2])
    _, second = gradients(restored, price[:2])
    preflight_equal = all(same_gradient(first[name], second[name]) for name in first)
    del model, restored, first, second
    command = [
        sys.executable,
        "-m",
        "scripts.training.run_a_gate_pixel_parallel_20261118",
        str(DOSE),
        "8",
        str(BATCH),
        str(initial),
    ]
    environment = {
        **os.environ,
        "TAIJI_A_FIXTURE": "v2",
        "TAIJI_A_EOS_MASK": "0",
        "TAIJI_D": "32",
        "TAIJI_LR": str(LR),
        "TAIJI_A_SEED": str(seed),
        "PYTHONUTF8": "1",
    }
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"parallel runner failed: {completed.stderr[-2000:]}")
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    final = ROOT / payload["checkpoint"]
    fit = generate_v2("fit", 32)
    reload_one, _, _ = load_learner(final)
    reading_one = evaluate(reload_one, fit)
    reload_two, _, _ = load_learner(final)
    reading_two = evaluate(reload_two, fit)
    recovery_equal = (
        reading_one["rows"] == reading_two["rows"]
        and [row["output"] for row in reading_one["rows"]] == payload["outputs"]
    )
    check_out = directory / "check.json"
    worker = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.training.run_a_gate_c_20261117",
            "--check-worker",
            "--checkpoint",
            str(final),
            "--check-out",
            str(check_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if worker.returncode != 0:
        raise RuntimeError(f"check worker failed: {worker.stderr[-2000:]}")
    check = json.loads(check_out.read_text(encoding="utf-8"))
    report = {
        "schema": "taiji-u2-22-5-a-parent-u2-arm-v1",
        "card": "22.5.0 A-parent formal package",
        "seed": seed,
        "preflight_equal": preflight_equal,
        "train": {
            "updates": DOSE,
            "batch": BATCH,
            "lr": LR,
            "fit_correct": payload["fit_correct"],
            "fit_denominator": payload["fit_denominator"],
            "delta_spread": payload["delta_spread"],
            "wall_seconds": payload["wall_seconds"],
            "checkpoint": payload["checkpoint"],
        },
        "recovery_equal": recovery_equal,
        "check": check,
        "cost": {
            "cpu": round(time.process_time() - started_cpu + check["self_cpu"], 4),
            "wall": round(time.perf_counter() - started_wall, 4),
        },
    }
    out = ROOT / f"reports/taiji_u2_a_parent_package_u2_seed{seed}_20261119.json"
    out.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {
                "seed": seed,
                "preflight_equal": preflight_equal,
                "fit_correct": payload["fit_correct"],
                "recovery_equal": recovery_equal,
                "check_correct": check["correct"],
                "check_accuracy": check["accuracy"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1])))

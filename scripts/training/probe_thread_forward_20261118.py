"""Does thread-level parallelism buy anything on the per-example forward?

The per-example path is thousands of small torch ops, so the GIL is released for
most of the wall time; four threads may therefore overlap.  This measures the
forward of four examples sequentially versus a thread pool, and the backward of
the summed loss (single-threaded by construction, reported for accounting).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (
    accept,
    generate_examples,
    output,
    prepare,
)  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_thread_probe_20261118.json"


def per_example_loss(model, example) -> torch.Tensor:
    state = prepare(model, example["public"], example["id"])
    terms = []
    for index, symbol in enumerate([*example["target"], 256]):
        probability, _, details = output(model, state)
        if index == 0:
            masked = probability.clone()
            masked[256] = -torch.inf
            probability = masked - torch.logsumexp(masked, -1)
        terms.append(-probability[symbol] * (2.0 if index == 0 else 1.0))
        if index < 2:
            state = accept(model, state, symbol, details)
    return torch.stack(terms).mean()


def main() -> int:
    model = (
        U2Subject(replace(U2Config.small(), seed=20261110, readout="pixel-salience"))
        .metadata_candidate()
        .request_candidate()
    )
    batch = generate_examples("train", 4)
    result = {"threads": torch.get_num_threads()}

    started = time.process_time()
    sequential = [per_example_loss(model, example) for example in batch]
    result["forward_sequential"] = round(time.process_time() - started, 4)

    with ThreadPoolExecutor(max_workers=4) as pool:
        started = time.process_time()
        threaded = list(pool.map(lambda e: per_example_loss(model, e), batch))
        result["forward_threaded"] = round(time.process_time() - started, 4)

    result["loss_sequential"] = [round(float(value.detach()), 6) for value in sequential]
    result["loss_threaded"] = [round(float(value.detach()), 6) for value in threaded]

    model.zero_grad(set_to_none=True)
    started = time.process_time()
    (sum(sequential) / len(sequential)).backward()
    result["forward_plus_backward"] = round(time.process_time() - started, 4)

    result["speedup_forward"] = round(
        result["forward_sequential"] / max(result["forward_threaded"], 1e-9), 2
    )
    OUT.write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

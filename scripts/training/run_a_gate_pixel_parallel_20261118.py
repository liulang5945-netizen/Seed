"""Process-level data parallelism: workers compute per-example gradients, main steps.

Threads do not help (0.82x, GIL) and inductor does not fuse the dict-driven path,
so the remaining lever is to run examples in separate processes.  Each worker holds
a model copy and receives the current flat parameter vector; it returns the
gradient for its shard.  The main process keeps the authoritative model, an Adam
optimizer and the batch order, so optimiser semantics are unchanged.

No model code is modified and the bundle format is untouched.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.nn.utils import parameters_to_vector, vector_to_parameters

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    generate_examples,
    load_learner,
    save_learner,
)
from scripts.training.run_a_gate_pixel_20261117 import (  # noqa: E402
    FIRST_WEIGHT,
    SEED,
    delta_spread,
    evaluate,
    masked_loss,
)
from scripts.training.run_a_gate_pixel_segment_20261117 import probe  # noqa: E402

if os.environ.get("TAIJI_A_FIXTURE") == "v2":  # arm selector: three-body train split
    from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: F811
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_parallel_speed_20261118.json"
WORLDS = 768
RUN_SEED = int(os.environ.get("TAIJI_A_SEED", SEED))
MASK_EOS = os.environ.get("TAIJI_A_EOS_MASK", "1") == "1"
BYTE_WEIGHT = {0xE4: 2.0, 0xE6: 2.0} if os.environ.get("TAIJI_A_BYTE_BALANCE") == "1" else None
WORKER: dict = {}


def _init(workers: int) -> None:
    torch.set_num_threads(1)
    config = replace(U2Config.small(), seed=RUN_SEED, readout="pixel-salience")
    WORKER["model"] = U2Subject(config).metadata_candidate().request_candidate()
    WORKER["train"] = generate_examples("train", WORLDS)
    WORKER["workers"] = workers


def _grad(task) -> tuple[float, torch.Tensor]:
    flat, indices = task
    model = WORKER["model"]
    vector_to_parameters(flat, model.parameters())
    model.zero_grad(set_to_none=True)
    train = WORKER["train"]
    loss = torch.stack(
        [masked_loss(model, [train[i]], FIRST_WEIGHT, MASK_EOS, BYTE_WEIGHT) for i in indices]
    ).mean()
    loss.backward()
    pieces = [
        parameter.grad.flatten() if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in model.parameters()
    ]
    return float(loss.detach()), torch.cat([piece.flatten() for piece in pieces])


def main(argv: list[str] | None = None) -> int:
    updates = int(argv[0]) if argv else 100
    workers = int(argv[1]) if len(argv) > 1 else 4
    batch = int(argv[2]) if len(argv) > 2 else 4
    config = replace(U2Config.small(), seed=RUN_SEED, readout="pixel-salience")
    model = U2Subject(config).metadata_candidate().request_candidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    order = torch.randperm(WORLDS, generator=torch.Generator().manual_seed(RUN_SEED + 9)).tolist()
    cursor = 0
    resume = Path(argv[3]) if len(argv) > 3 else None
    if resume is not None:
        model, optimizer, loaded = load_learner(resume)
        order = [int(index) for index in loaded.get("order", order)]
        cursor = int(loaded.get(cursor, 0))
    flat = parameters_to_vector(model.parameters()).detach()
    wall = time.perf_counter()
    with mp.Pool(workers, initializer=_init, initargs=(workers,)) as pool:
        for _ in range(updates):
            shards = [[order[(cursor + offset) % WORLDS]] for offset in range(batch)]
            cursor = (cursor + batch) % WORLDS
            results = pool.map(_grad, [(flat, shard) for shard in shards])
            gradient = torch.stack([row[1] for row in results]).mean(0)
            optimizer.zero_grad(set_to_none=True)
            offset = 0
            for parameter in model.parameters():
                size = parameter.numel()
                parameter.grad = gradient[offset : offset + size].view_as(parameter).clone()
                offset += size
            optimizer.step()
            flat = parameters_to_vector(model.parameters()).detach()
    elapsed = time.perf_counter() - wall
    fit = generate_examples("fit", 32)
    result = evaluate(model, fit)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory = ROOT / "output" / f"a_pixel_parallel_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    train_for_digest = generate_examples("train", WORLDS)
    checkpoint = directory / "segment.bin"
    save_learner(
        checkpoint,
        model,
        optimizer,
        "native",
        updates,
        updates * batch,
        order,
        value_digest(train_for_digest),
    )
    payload = {
        "schema": "taiji-u2-parallel-speed-v1",
        "card": "22.5.0.A",
        "fixture": os.environ.get("TAIJI_A_FIXTURE", "v1"),
        "eos_masked_in_loss": MASK_EOS,
        "byte_weight": BYTE_WEIGHT,
        "updates": updates,
        "seed": RUN_SEED,
        "workers": workers,
        "batch": batch,
        "wall_seconds": round(elapsed, 2),
        "updates_per_second": round(updates / elapsed, 4),
        "seconds_per_update": round(elapsed / updates, 4),
        "torch_threads": torch.get_num_threads(),
        "fit_correct": result["correct"],
        "fit_denominator": result["denominator"],
        "outputs": [row["output"] for row in result["rows"]],
        "delta_spread": round(delta_spread(model, fit), 6),
        "probe": probe(model),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
    }
    if not OUT.exists():
        OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

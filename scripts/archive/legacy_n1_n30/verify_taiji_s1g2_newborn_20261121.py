"""S1-G2: correct-layout newborn integration after growth (0 check, <=600 CPU).

S1-G proved grow_units_neutral_ is birth-neutral and capsule-exact once heads are remapped
correctly.  This card trains the newborn units through the p27 head with the correct layout:
zero-init the new head columns (birth-neutral), stage-1 temp head over the newborn slice with
masked gradients, stage-2 joint training, then lesion = drop with the newborn columns zeroed.
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
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r_restructure_20261121 import SubjectR  # noqa: E402

OUT = ROOT / "reports/taiji_u2_s1g2_newborn_20261121.json"
CAP_CPU = 600.0
SEEDS = (20261120, 20261121)
BATCH = 48
FIT_OFFSET = 10_000_000
STEPS_BASE = 250
STEPS_STAGE1 = 250
STEPS_STAGE2 = 250


def _remap(head: nn.Module, old_units: int, new_units: int) -> None:
    old_linear = head.net[0]
    width = old_linear.out_features
    new_linear = nn.Linear(2 * new_units, width)
    with torch.no_grad():
        new_linear.weight.zero_()
        new_linear.bias.copy_(old_linear.bias)
        new_linear.weight[:, :old_units].copy_(old_linear.weight[:, :old_units])
        new_linear.weight[:, new_units : new_units + old_units].copy_(
            old_linear.weight[:, old_units : 2 * old_units]
        )
    head.net[0] = new_linear


def _mask_new_rows(graph, old_units: int) -> None:
    for _name, parameter in graph.selective_units.named_parameters():
        if parameter.grad is None or parameter.ndim == 0:
            continue
        if parameter.shape[0] == graph.selective_units.n_units:
            parameter.grad[:old_units] = 0.0


def run_seed(seed: int) -> dict:
    torch.set_num_threads(1)
    subject = SubjectR(seed)
    base = _make_p27_physics_batch(400, seed + 91)
    for head in subject.phase_heads.values():
        head.train()
    optimizer = torch.optim.Adam(subject.params(), lr=6e-3)
    rng = torch.Generator().manual_seed(seed + 1001)
    for _ in range(STEPS_BASE):
        idx = torch.randint(0, base["ticks"].shape[0], (BATCH,), generator=rng)
        loss = F.cross_entropy(subject.logits("p27", base["ticks"][idx]), base["labels"][idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(subject.params(), 1.0)
        optimizer.step()
    for head in subject.phase_heads.values():
        head.eval()
    fit = _make_p27_physics_batch(128, seed + 92 + FIT_OFFSET)
    with torch.no_grad():
        base_acc = float(
            (subject.logits("p27", fit["ticks"]).argmax(-1) == fit["labels"]).float().mean()
        )
        before_logits = subject.logits("p27", fit["ticks"])

    old_units = subject.graph.selective_units.n_units
    for _ in range(2):
        before = subject.graph.selective_units.n_units
        subject.graph.grow_units_neutral_(1)
        subject.n_units = subject.graph.selective_units.n_units
        for head in subject.phase_heads.values():
            _remap(head, before, subject.n_units)
    new_units = subject.graph.selective_units.n_units
    head = subject.phase_heads["p27"]
    with torch.no_grad():
        head.net[0].weight[:, old_units:new_units] = 0.0
        head.net[0].weight[:, new_units + old_units : 2 * new_units] = 0.0
    for head_ in subject.phase_heads.values():
        head_.eval()
    with torch.no_grad():
        birth = subject.logits("p27", fit["ticks"])
    birth_diff = float((birth - before_logits).abs().max())

    # stage 1: temp head over newborn slice, masked gradients keep mature units frozen
    temp = nn.Sequential(nn.Linear(2 * (new_units - old_units), 64), nn.Tanh(), nn.Linear(64, 4))
    for head_ in subject.phase_heads.values():
        head_.train()
    params = [p for p in subject.graph.selective_units.parameters() if p.requires_grad] + list(
        temp.parameters()
    )
    opt1 = torch.optim.Adam(params, lr=4e-3)
    rng = torch.Generator().manual_seed(seed + 1101)
    for _ in range(STEPS_STAGE1):
        batch = _make_p27_physics_batch(
            BATCH, seed * 51000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        z, a = subject.state_of(batch["ticks"])
        x = torch.cat([z, a], dim=-1)[:, 2 * old_units :]
        loss = F.cross_entropy(temp(x), batch["labels"])
        opt1.zero_grad(set_to_none=True)
        loss.backward()
        _mask_new_rows(subject.graph, old_units)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt1.step()

    # stage 2: zero-init p27 newborn cols, train newborn units + p27 head
    for p in subject.graph.parameters():
        p.requires_grad_(False)
    for p in subject.graph.selective_units.parameters():
        p.requires_grad_(True)
    for p in head.parameters():
        p.requires_grad_(True)
    trainable = [p for p in subject.graph.selective_units.parameters() if p.requires_grad]
    trainable += [p for p in head.parameters() if p.requires_grad]
    opt2 = torch.optim.Adam(trainable, lr=4e-3)
    for _ in range(STEPS_STAGE2):
        batch = _make_p27_physics_batch(
            BATCH, seed * 52000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        loss = F.cross_entropy(subject.logits("p27", batch["ticks"]), batch["labels"])
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        _mask_new_rows(subject.graph, old_units)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt2.step()

    for head_ in subject.phase_heads.values():
        head_.eval()
    with torch.no_grad():
        final = subject.logits("p27", fit["ticks"])
        final_acc = float((final.argmax(-1) == fit["labels"]).float().mean())
        weight = head.net[0].weight.clone()
        head.net[0].weight[:, old_units:new_units] = 0.0
        head.net[0].weight[:, new_units + old_units : 2 * new_units] = 0.0
        les = head(torch.cat(subject.state_of(fit["ticks"]), dim=-1)).argmax(-1)
        lesion_drop = final_acc - float((les == fit["labels"]).float().mean())
        head.net[0].weight.copy_(weight)
    return {
        "seed": seed,
        "base_acc": base_acc,
        "final_acc": final_acc,
        "birth_diff": birth_diff,
        "lesion_drop": lesion_drop,
        "units": [old_units, new_units],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    rows = [run_seed(seed) for seed in SEEDS]
    gates = {
        "birth_neutral": max(row["birth_diff"] for row in rows) <= 1e-6,
        "integration": min(row["lesion_drop"] for row in rows) >= 0.20,
    }
    verdict = "s1g2_verified" if all(gates.values()) else "s1g2_partial"
    payload = {
        "schema": "taiji-u2-s1g2-newborn-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "growth": "48 -> 64",
            "stages": [STEPS_BASE, STEPS_STAGE1, STEPS_STAGE2],
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "unit core untouched; this is an experiment-level integration test after correct head remap",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "s1g2_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())

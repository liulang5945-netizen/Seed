"""Minimal trainability differential: known-good SubjectR.logits vs the custom masked path (0 check).

Fixes the stage-1b implementation failure by isolating why CN does not learn in the custom
loop.  Fixed batch, 100 steps, no check/final, <=100 CPU.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import (  # noqa: E402
    _channels_seq,
    _clip_and_step,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead  # noqa: E402
from scripts.training.verify_taiji_s1r_restructure_20261121 import SubjectR  # noqa: E402

OUT = ROOT / "reports/taiji_u2_cn_min_unit_20261121.json"
CAP_CPU = 100.0


def data(seed: int) -> dict[str, torch.Tensor]:
    return _build_bidirectional_cn_32dim(48, seed, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)


def known_good(seed: int, steps: int = 100) -> list[float]:
    torch.set_num_threads(1)
    subject = SubjectR(seed)
    batch = data(seed)
    sm = (batch["ticks"].abs().sum(-1) > 0).float()
    opt = torch.optim.Adam(subject.params(), lr=6e-3, weight_decay=1e-4)
    losses = []
    for _ in range(steps):
        logits = subject.logits("cn", batch["ticks"], step_mask=sm)
        loss = F.cross_entropy(logits, batch["labels"])
        _clip_and_step(opt, subject.graph, loss)
        losses.append(float(loss.detach()))
    acc = float(
        (subject.logits("cn", batch["ticks"], step_mask=sm).argmax(-1) == batch["labels"])
        .float()
        .mean()
    )
    return losses, acc


def custom(seed: int, steps: int = 100) -> tuple[list[float], float, dict]:
    torch.set_num_threads(1)
    subject = SubjectR(seed)
    subject.phase_heads["cn"] = PhaseHead(2 * subject.n_units, 8, 128)
    batch = data(seed)
    sm = (batch["ticks"].abs().sum(-1) > 0).float()
    n = subject.n_units
    mask = torch.ones(1, 1, n)
    gates = mask.expand(batch["ticks"].shape[0], batch["ticks"].shape[1], n)
    opt = torch.optim.Adam(subject.params(), lr=6e-3, weight_decay=1e-4)
    losses = []
    grads = []
    for _ in range(steps):
        out = subject.graph(
            _channels_seq(batch["ticks"]),
            gates,
            subject.graph.initial_state(batch_size=batch["ticks"].shape[0]),
            step_mask=sm,
        )
        z, a = out["next_state"]["z"], out["next_state"]["a"]
        head = subject.phase_heads["cn"]
        logits = head(torch.cat([z, a], dim=-1))
        loss = F.cross_entropy(logits, batch["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(float(subject.graph.selective_units.r.grad.abs().mean()))
        opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        out = subject.graph(
            _channels_seq(batch["ticks"]),
            gates,
            subject.graph.initial_state(batch_size=batch["ticks"].shape[0]),
            step_mask=sm,
        )
        z, a = out["next_state"]["z"], out["next_state"]["a"]
        logits = subject.phase_heads["cn"](torch.cat([z, a], dim=-1))
        acc = float((logits.argmax(-1) == batch["labels"]).float().mean())
    return losses, acc, {"grad_mean": grads[:3], "gates_shape": list(gates.shape)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    seed = 20261120
    good_losses, good_acc = known_good(seed)
    bad_losses, bad_acc, info = custom(seed)
    payload = {
        "schema": "taiji-u2-cn-min-unit-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {"threads": 1, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "known_good": {"first": good_losses[:3], "last": good_losses[-3:], "acc": good_acc},
        "custom": {"first": bad_losses[:3], "last": bad_losses[-3:], "acc": bad_acc, **info},
        "verdict": "cn_min_unit_diagnostic",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

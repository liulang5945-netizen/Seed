"""S1 gap diagnosis: P-B capacity vs optimization, and why grown units stay unused (0 check).

Two seeds, <=300 CPU, no check/final.  Three probes:
  D1  dedicated P24 training on the S1 unified subject (30 units, 13-dim readout) -> train/fit.
  D2  dedicated S0 P24 graph (48 units, 5-dim readout) -> fit (task-learnability control).
  D3  readout-only training after growth: freeze all but the selective readout, grow 30->54,
      train readout on P-D, measure lesion drop and new-unit weight mass.
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
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import (  # noqa: E402
    _channels_seq,
    _clip_and_step,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    _train_p24_graph,
)
from scripts.training.verify_taiji_s1_lifeline_20261121 import (  # noqa: E402
    FIT_OFFSET,
    Subject,
    _p24_batch,
)

OUT = ROOT / "reports/taiji_u2_s1_gap_diag_20261121.json"
CAP_CPU = 300.0
SEEDS = (20261120, 20261121)
STEPS_MAIN = 900
STEPS_READOUT = 300
BATCH = 48
FIT = 160


def diag_unified_subject(seed: int) -> dict:
    subject = Subject(seed)
    train = _p24_batch(seed + 211, 320)
    fit = _p24_batch(seed + 212 + FIT_OFFSET, FIT)
    rng = torch.Generator().manual_seed(seed + 701)
    for _ in range(STEPS_MAIN):
        idx = torch.randint(0, train["ticks"].shape[0], (BATCH,), generator=rng)
        out = subject.forward(train["ticks"][idx])
        loss = F.cross_entropy(out["readout"][:, :4], train["val_labels"][idx])
        loss = loss + F.binary_cross_entropy_with_logits(
            out["readout"][:, 8], train["conflict_labels"][idx]
        )
        subject.train_step(loss)
    with torch.no_grad():
        tr = subject.forward(train["ticks"])["readout"]
        fit_out = subject.forward(fit["ticks"])["readout"]
    return {
        "train_acc": float((tr[:, :4].argmax(-1) == train["val_labels"]).float().mean()),
        "fit_acc": float((fit_out[:, :4].argmax(-1) == fit["val_labels"]).float().mean()),
        "conflict_fit": float(
            ((fit_out[:, 8] > 0).float() == fit["conflict_labels"]).float().mean()
        ),
    }


def diag_s0_graph(seed: int) -> dict:
    base, _ = train_joint_parent(seed)
    graph = _train_p24_graph(base, seed + 211, steps=1000)
    fit = _p24_batch(seed + 212 + FIT_OFFSET, FIT)
    with torch.no_grad():
        out = graph(
            _channels_seq(fit["ticks"]),
            torch.ones(FIT, 14, graph.selective_spec.n_units),
            graph.initial_state(batch_size=FIT),
        )["readout"]
    return {"fit_acc": float((out[:, :4].argmax(-1) == fit["val_labels"]).float().mean())}


def diag_readout_after_growth(seed: int) -> dict:
    subject = Subject(seed)
    old_units = subject.graph.selective_units.n_units
    for parameter in subject.graph.parameters():
        parameter.requires_grad_(False)
    subject.graph.grow_units_neutral_(2)
    subject.graph.grow_units_neutral_(2)
    subject.n_units = subject.graph.selective_units.n_units
    readout_params = [
        parameter
        for name, parameter in subject.graph.named_parameters()
        if "selective_readout" in name
    ]
    for parameter in readout_params:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(readout_params, lr=5e-3)
    fit = _p24_batch(seed + 212 + FIT_OFFSET, FIT)
    rng = torch.Generator().manual_seed(seed + 702)
    for _ in range(STEPS_READOUT):
        idx = torch.randint(0, fit["ticks"].shape[0], (BATCH,), generator=rng)
        out = subject.forward(fit["ticks"][idx])
        loss = F.cross_entropy(out["readout"][:, :4], fit["val_labels"][idx])
        _clip_and_step(optimizer, subject.graph, loss)
    with torch.no_grad():
        full = subject.forward(fit["ticks"])["readout"][:, :4].argmax(-1)
        gates = torch.ones(FIT, 14, subject.n_units)
        gates[..., old_units:] = 0.0
        les = subject.forward(fit["ticks"], gates=gates)["readout"][:, :4].argmax(-1)
        weights = {
            name: tensor.detach()
            for name, tensor in subject.graph.named_parameters()
            if "selective_readout" in name
        }
        mass = {}
        for name, tensor in weights.items():
            if tensor.ndim == 2 and tensor.shape[1] == subject.n_units:
                mass[name] = {
                    "new_abs_mean": float(tensor[:, old_units:].abs().mean()),
                    "old_abs_mean": float(tensor[:, :old_units].abs().mean()),
                }
            else:
                mass[name] = {"shape": list(tensor.shape)}
    return {
        "fit_acc": float((full == fit["val_labels"]).float().mean()),
        "lesion_drop": float((full == fit["val_labels"]).float().mean())
        - float((les == fit["val_labels"]).float().mean()),
        "readout_mass": mass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = []
    for seed in SEEDS:
        rows.append(
            {
                "seed": seed,
                "unified_subject_p24": diag_unified_subject(seed),
                "s0_p24_graph": diag_s0_graph(seed),
                "readout_after_growth": diag_readout_after_growth(seed),
            }
        )
    means = {
        key: sum(row[key][sub] for row in rows) / len(rows)
        for key, sub in (
            ("unified_subject_p24", "fit_acc"),
            ("unified_subject_p24", "train_acc"),
            ("s0_p24_graph", "fit_acc"),
            ("readout_after_growth", "lesion_drop"),
        )
    }
    payload = {
        "schema": "taiji-u2-s1-gap-diag-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {"seeds": list(SEEDS), "steps": STEPS_MAIN, "readout_steps": STEPS_READOUT},
        "rows": rows,
        "means": means,
        "verdict": "s1_gap_diagnostic",
        "disclosure": "zero check/final; only fit; diagnosis only, not a capability card",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": payload["verdict"], "means": means, "cost": payload["cost"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

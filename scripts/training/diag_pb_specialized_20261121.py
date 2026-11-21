"""P-B specialized-subject probe before the 23-S narrow contract (0 check, <=600 CPU).

Diagnosis showed P-B fails inside the unified 13-dim readout (fit 0.51-0.58) while the S0
48-unit/5-dim graph reaches 0.91-1.0.  This probe trains a dedicated 48-unit subject with a
p24-only head (128 wide) to see whether the 0.95 gate is reachable before it is written into
the narrow contract.  Zero check/final.
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
from scripts.training.verify_taiji_cognitive_loop_20261121 import _make_p24_stream  # noqa: E402
from scripts.training.verify_taiji_s1r_restructure_20261121 import SubjectR  # noqa: E402

OUT = ROOT / "reports/taiji_u2_pb_specialized_20261121.json"
CAP_CPU = 600.0
SEEDS = (20261120, 20261121, 20261122)
BATCH = 48
FIT_OFFSET = 10_000_000
STEPS = 2000


def run_seed(seed: int) -> dict:
    torch.set_num_threads(1)
    subject = SubjectR(seed)
    train_eval = _make_p24_stream(480, seed + 213 + FIT_OFFSET)
    fit = _make_p24_stream(160, seed + 212 + FIT_OFFSET)
    for head in subject.phase_heads.values():
        head.train()
    optimizer = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=1e-4)
    for step in range(STEPS):
        batch = _make_p24_stream(BATCH, seed * 25000 + step)
        logits = subject.logits("p24", batch["ticks"])
        loss = F.cross_entropy(logits[:, :4], batch["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], batch["conflict_labels"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(subject.params(), 1.0)
        optimizer.step()
    for head in subject.phase_heads.values():
        head.eval()
    with torch.no_grad():
        train_logits = subject.logits("p24", train_eval["ticks"])
        fit_logits = subject.logits("p24", fit["ticks"])
        train_acc = float(
            (train_logits[:, :4].argmax(-1) == train_eval["val_labels"]).float().mean()
        )
        overall = float((fit_logits[:, :4].argmax(-1) == fit["val_labels"]).float().mean())
        conflict_acc = float(
            ((fit_logits[:, 4] > 0).float() == fit["conflict_labels"]).float().mean()
        )
    return {
        "seed": seed,
        "train_acc": train_acc,
        "fit_overall": overall,
        "conflict_acc": conflict_acc,
        "units": subject.n_units,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    rows = [run_seed(seed) for seed in SEEDS]
    means = {
        "train_acc": sum(row["train_acc"] for row in rows) / len(rows),
        "fit_overall": sum(row["fit_overall"] for row in rows) / len(rows),
        "conflict_acc": sum(row["conflict_acc"] for row in rows) / len(rows),
    }
    verdict = (
        "pb_specialized_reachable"
        if min(row["fit_overall"] for row in rows) >= 0.95
        else "pb_specialized_partial"
    )
    payload = {
        "schema": "taiji-u2-pb-specialized-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {"seeds": list(SEEDS), "units": 48, "head": "p24-only 128", "steps": STEPS},
        "rows": rows,
        "means": means,
        "verdict": verdict,
        "disclosure": "diagnostic for the 23-S narrow contract; not a capability card",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "means": means, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

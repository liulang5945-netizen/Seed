"""Recover the first P4.2 attempt: phase-1 accuracies and the failure receipt.

The run died at the first phase-2 dose evaluation because ``round_evaluate`` was
called with ``budget_guard`` while its parameter was named ``guard``. No report
was written. This script reads the two phase-1 checkpoints (zero training) and
writes the engineering-failure receipt with the recovered scaffold readings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_2_byte_path_20261115.json"
RUN_ROOT = ROOT / "data/experiments/taiji_u2/p4_2/20261116T0000Z"
REPORT = ROOT / "reports/taiji_u2_p4_2_engineering_failure_20261116T0000Z.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-observed", type=float, default=1395.609375)
    parser.add_argument("--cpu-estimated", type=float, default=1880.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data = p4_2.atomic_load(Path(contract["data_root"]))
    rows = []
    for update in (1844, 3688):
        path = RUN_ROOT / f"phase1-{update:06d}.pt"
        payload = torch.load(path, weights_only=False)
        model = p4_2.build(contract)
        model.load_state_dict(payload["model"])
        scaffold = torch.nn.Linear(24, 4)
        scaffold.load_state_dict(payload["scaffold"])
        train = p4_2.scaffold_accuracy(model, scaffold, data["train"][:32])
        fit = p4_2.scaffold_accuracy(model, scaffold, data["fit"])
        rows.append(
            {
                "update": update,
                "train_accuracy": train["accuracy"],
                "train_correct": train["correct"],
                "fit_accuracy": fit["accuracy"],
                "fit_correct": fit["correct"],
                "checkpoint": str(path),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    report = {
        "schema": "taiji-u2-p4-2-engineering-failure-v1",
        "card": "22.5.P4.2",
        "date": "2026-11-15",
        "attempt": 1,
        "run_root": str(RUN_ROOT),
        "error": "TypeError: round_evaluate() got an unexpected keyword argument 'budget_guard'",
        "where": "run_p4_2_22_5.py line 780, first phase-2 dose evaluation",
        "cause": "the helper parameter was named 'guard' while every call site passed 'budget_guard'",
        "state": {
            "stage0_reconstruction": "completed, signature accepted",
            "stage1": "converged at 3688 steps and broke early",
            "stage2": "cache built and replay check passed; died at the first dose evaluation",
            "run_report_written": False,
        },
        "recovered_phase1": rows,
        "kept_checkpoints": [str(path) for path in sorted(RUN_ROOT.glob("*.pt"))],
        "cost": {
            "cpu_observed_at_last_check": args.cpu_observed,
            "cpu_estimated_final": args.cpu_estimated,
            "basis": "last process_time reading at 04:19:45 plus the measured one-core rate to the crash time",
            "note": "the process died before writing a receipt, so the final figure is an estimate, not a measurement",
        },
        "discipline": {
            "training_resumed": False,
            "checkpoints_deleted": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
    }
    write_report(REPORT, report)
    print(json.dumps({"phase1": rows, "cpu": report["cost"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

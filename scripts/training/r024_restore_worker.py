"""R024 restore worker CLI: runs the worker job in a fresh interpreter process.

The job logic lives in taiji.state_migration.run_worker_job so that tests can
launch it via multiprocessing spawn (a genuinely separate interpreter). This CLI
wraps the same function for standalone/audit use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from taiji.state_migration import run_worker_job


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mode", choices=("microstep", "update", "resolve"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path)
    parser.add_argument(
        "--current-journal",
        type=Path,
        help="JSON file with the host's current journal entries (resolve guard)",
    )
    args = parser.parse_args()
    current_entries = (
        json.loads(args.current_journal.read_text(encoding="utf-8"))
        if args.current_journal is not None
        else []
    )
    result = run_worker_job(
        {
            "bundle": str(args.bundle),
            "mode": args.mode,
            "out": None,
            "out_bundle": str(args.out_bundle) if args.out_bundle is not None else None,
            "current_entries": current_entries,
            "fee_per_step": 1e-3,
        }
    )
    if args.out is not None:
        torch.save(result, args.out)
        print(
            json.dumps(
                {
                    "mode": result["mode"],
                    "resolved": result.get("resolved"),
                    "reward_delta": result.get("reward_delta"),
                    "consumed": result.get("consumed"),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

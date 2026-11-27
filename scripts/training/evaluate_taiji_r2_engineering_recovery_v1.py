"""Versioned recovery entry; initially disabled pending one protocol decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.training import (
    evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1 as worker,
)
from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as frozen
from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter
from scripts.training import taiji_r2_evaluation_recovery_v1 as provider

MODULE = "scripts.training.evaluate_taiji_r2_engineering_recovery_v1"


def worker_factory(**kwargs: Any) -> Any:
    command = adapter.extension_worker_command(kwargs["attempt_path"], kwargs["ledger_path"])
    command[command.index("-m") + 1] = MODULE
    return frozen.SupervisedWorker(**kwargs, command=command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ledger-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--job-name", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            if not args.attempt_path or not args.ledger_path or not args.job_name:
                raise ValueError("private worker requires attempt, ledger and Job identity")
            return worker.worker_loop(
                args.attempt_path.resolve(),
                args.ledger_path.resolve(),
                args.job_name,
                preflight_fn=provider.preflight,
                evaluation_paths_fn=provider.paths,
            )
        if args.attempt_path or args.ledger_path or args.job_name:
            raise ValueError("private worker paths are not public evaluation arguments")
        provider.require_authorization(json.loads(provider.CONTRACT.read_text(encoding="utf-8")))
        if args.execute:
            result = frozen.execute(
                paths=provider.paths(),
                preflight_fn=provider.preflight,
                worker_factory=worker_factory,
                budget_factory=provider.budget_factory,
            )
            print(json.dumps({"status": result.get("status"), "can_promote": False}))
            return 0 if result.get("complete") is True else 1
        preflight = provider.preflight()
        paths = provider.paths()
        adapter.write_extension_readiness(paths["readiness"], preflight)
        adapter.write_endpoint_manifest(paths["endpoint_manifest"], preflight)
        print(
            json.dumps(
                {"status": "ready", "expected_generation_outputs": 156, "can_promote": False}
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps({"status": "refused", "error": str(exc), "can_promote": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

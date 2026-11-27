"""Versioned continuation-extension paired-evaluation entry (CC-1.2).

``--validate-only`` freezes the extension endpoint manifest and the frozen
evaluator's readiness envelope.  ``--execute`` runs the same one-shot,
four-endpoint evaluation through a versioned worker, and ``--worker`` is that
worker's private IPC loop.  The frozen recovery-v2 evaluator is imported as a
library; its module globals are never patched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import (
    evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1 as worker,  # noqa: E402
)
from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as recovery  # noqa: E402
from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter  # noqa: E402


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _worker_factory(**kwargs: Any) -> Any:
    command = adapter.extension_worker_command(kwargs["attempt_path"], kwargs["ledger_path"])
    return recovery.SupervisedWorker(**kwargs, command=command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ledger-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--job-name", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.attempt_path is None or args.ledger_path is None or not args.job_name:
            _emit(
                {
                    "status": "refused",
                    "error": "private --worker mode requires attempt, ledger and job name",
                    "can_promote": False,
                }
            )
            return 2
        return worker.worker_loop(
            args.attempt_path.resolve(), args.ledger_path.resolve(), args.job_name
        )

    if args.attempt_path is not None or args.ledger_path is not None or args.job_name:
        _emit(
            {
                "status": "refused",
                "error": "attempt/ledger paths are only valid in private worker mode",
                "can_promote": False,
            }
        )
        return 2

    try:
        if args.execute:
            result = recovery.execute(
                paths=adapter.extension_evaluation_paths(),
                preflight_fn=adapter.extension_preflight,
                worker_factory=_worker_factory,
            )
            _emit({"status": result.get("status"), "can_promote": False})
            return 0
        preflight = adapter.extension_preflight()
        paths = adapter.extension_evaluation_paths()
        readiness = adapter.write_extension_readiness(paths["readiness"], preflight)
        manifest = adapter.write_endpoint_manifest(paths["endpoint_manifest"], preflight)
    except BaseException as exc:
        _emit(
            {
                "status": "refused",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "can_promote": False,
            }
        )
        return 2
    _emit(
        {
            "status": "ready",
            "endpoint_schema": manifest["schema"],
            "endpoint_manifest_sha256": manifest["manifest_sha256"],
            "endpoint_order": manifest["endpoint_order"],
            "expected_generation_outputs": readiness["expected_generation_outputs"],
            "can_promote": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Versioned continuation-extension paired-evaluation entry (CC-1.2).

``--validate-only`` freezes the extension endpoint manifest and the frozen
evaluator's readiness envelope.  ``--execute`` / ``--worker`` stay refused until
the versioned extension worker is wired; this entry never monkeypatches the
frozen recovery-v2 evaluator.
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

from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter  # noqa: E402


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


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

    if args.worker or args.execute:
        _emit(
            {
                "status": "refused",
                "error": "extension worker provider is not wired; run --validate-only",
                "can_promote": False,
            }
        )
        return 2

    try:
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

"""C6 P1 wiring verification: the workbench capability snapshot, end to end.

Zero product change on top of the P1 wiring itself: it exercises the real chain
``GET /api/workbench/capabilities`` -> ``WorkbenchCapabilityAdapter.project``
(which constructs ``workbench_artifact`` corpus units, so the corpus-side kind
whitelist is exercised) -> ``workbench_training_record`` (no-prose template) ->
``seed.datasets.inspect_native_dataset``, and judges the Z1 gate -- units > 0
and native_trainable -- against the live runtime rather than a prototype.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed.datasets import inspect_native_dataset  # noqa: E402
from seed_platform.evolution_adapters import (  # noqa: E402
    WORKBENCH_QUESTION,
    WorkbenchCapabilityAdapter,
    workbench_training_record,
)

DEFAULT_URL = "http://127.0.0.1:8000"
CAPABILITIES_PATH = "/api/workbench/capabilities"


def _load_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    if args.snapshot is not None:
        raw = Path(args.snapshot).read_text(encoding="utf-8")
    else:
        url = f"{args.url.rstrip('/')}{CAPABILITIES_PATH}"
        with urllib.request.urlopen(url, timeout=args.timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise SystemExit("workbench capabilities snapshot is not a JSON object")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="runtime root to read")
    parser.add_argument("--snapshot", help="path to a captured snapshot instead of reading it")
    parser.add_argument("--out", default="output/c6_p1", help="where to write the rendered corpus")
    parser.add_argument("--report", required=True, help="where to write the JSON reading")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="seconds to wait for the endpoint"
    )
    args = parser.parse_args()

    snapshot = _load_snapshot(args)
    declared = len([item for item in snapshot.get("capabilities", []) if isinstance(item, dict)])

    projection = WorkbenchCapabilityAdapter().project(snapshot)
    records = [workbench_training_record(dict(unit.content)) for unit in projection.corpus]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "c6-p1-workbench-no-prose.jsonl"
    corpus_path.write_text(
        "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records),
        encoding="utf-8",
    )
    inspected = inspect_native_dataset(corpus_path)

    reading = {
        "probe": "taiji_c6_p1_wiring",
        "establishes": (
            "the P1 chain works end to end against the live runtime: the snapshot projects "
            "through WorkbenchCapabilityAdapter (so the corpus-side workbench_artifact kind "
            "is accepted), every declared capability renders one no-prose 问/答 record, and "
            "the rendered file is native_trainable with units > 0 -- the Z1 gate, re-judged "
            "against the wired implementation"
        ),
        "z1_gate": {
            "units": len(records),
            "native_trainable": inspected.native_trainable,
            "pass": len(records) > 0 and inspected.native_trainable,
        },
        "snapshot": {
            "snapshot_id": snapshot.get("snapshot_id"),
            "revision": snapshot.get("revision"),
            "declared_capabilities": declared,
            "projected_units": len(projection.corpus),
        },
        "render": {
            "question": WORKBENCH_QUESTION,
            "sample": records[0]["text"] if records else "",
            "invalid_records": inspected.invalid_records,
            "documents": inspected.documents,
        },
        "corpus_path": str(corpus_path),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(reading, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reading["z1_gate"], ensure_ascii=False))
    return 0 if reading["z1_gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

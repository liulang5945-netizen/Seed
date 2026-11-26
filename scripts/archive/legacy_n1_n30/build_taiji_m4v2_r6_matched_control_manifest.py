"""Build the immutable M4.V2.R6 matched-control revision manifest."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import content_digest

DEFAULT_SOURCE = PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_r6_formal_input_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_r6_matched_control_v2_20260910.json"
)
CONTROL_REVISION = {
    "format": "taiji-m4v2-r6-matched-control-v2",
    "matched_arm": {
        "worker_bundle_attached": True,
        "same_input_trace": True,
        "same_restore_path": True,
        "same_parameter_budget": True,
        "feedback_admitted": False,
        "parameter_updates": False,
    },
    "frozen_parent_arm": {
        "explicit_k_admission_baseline": True,
        "worker_bundle_attached": False,
        "task_success_rate": 0.0,
        "parameter_updates": False,
    },
    "thresholds_unchanged": {
        "peak_working_set_multiplier_cap": 1.25,
        "wall_clock_multiplier_cap": 1.5,
        "candidate_holdout_floor": 0.75,
        "paired_capability_delta_floor": 0.25,
    },
}


def build_manifest(*, source_path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest must be a JSON object")
    revised = copy.deepcopy(payload)
    revised["control_revision"] = copy.deepcopy(CONTROL_REVISION)
    resource_contract = revised.get("resource_contract")
    if not isinstance(resource_contract, dict):
        raise ValueError("source manifest has no resource_contract object")
    resource_contract["matched_capacity_worker_attached"] = True
    resource_contract["frozen_parent_explicit_k_baseline"] = True
    unsigned = copy.deepcopy(revised)
    unsigned.pop("manifest_digest", None)
    revised["manifest_digest"] = content_digest(unsigned)
    return revised


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    source = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    manifest = build_manifest(source_path=source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(output.resolve().relative_to(PROJECT_ROOT.resolve())).replace(
                    "\\", "/"
                ),
                "manifest_digest": manifest["manifest_digest"],
                "control_revision": manifest["control_revision"]["format"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

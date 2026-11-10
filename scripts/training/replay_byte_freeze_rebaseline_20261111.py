"""One-off replay driver for the 2026-11-11 byte-freeze rebaseline.

R58/R59/R60/R60b/R61/R62/R63 and the stop-boundary runner hard-code their
manifest/report/progress paths.  After the six frozen taiji modules were
re-annotated, the original paths must not be written again, so this driver
rebinds the module-level path globals and re-runs a step into new
``*_rebaseline_20261111`` files.  Nothing else about a runner is touched.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "scripts" / "training"
for entry in (ROOT, TRAINING):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

STEPS = {
    "r58": "probe_taiji_stop_shape_r58_20261109",
    "r59": "probe_taiji_stop_shape_r59_20261109",
    "r60": "probe_taiji_object_length_r60_20261109",
    "r60b": "probe_taiji_object_length_dense_r60b_20261110",
    "r61": "probe_taiji_object_edge_r61_20261109",
    "r62": "probe_taiji_object_edge_intervention_r62_20261109",
    "r63": "probe_taiji_object_edge_char_r63_20261110",
    "stop_boundary": "probe_taiji_stop_boundary_20261109",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=sorted(STEPS))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    module = importlib.import_module(STEPS[args.step])
    suffix = "_rebaseline_20261111"
    module.MANIFEST = ROOT / "plans" / "manifests" / f"{module.MANIFEST.stem}{suffix}.json"
    module.REPORT = ROOT / "reports" / f"{module.REPORT.stem}{suffix}.json"
    module.PROGRESS = ROOT / "reports" / f"{module.PROGRESS.stem}{suffix}.jsonl"
    print(
        f"[replay] {args.step} -> {module.MANIFEST.name} / {module.REPORT.name}",
        flush=True,
    )
    sys.argv = ["replay", *(["--freeze"] if args.freeze else [])]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())

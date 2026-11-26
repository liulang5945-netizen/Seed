"""Recover this morning's 12 seed-probe readings from their checkpoints.

`output/_e5_restart_protocol.sh` ran 12 x 300-update probes (3 base seeds x 4
offsets) but only printed the readings to a terminal that is gone; the
checkpoints are the durable record.  This loads each `a_pixel_parallel_*
/segment.bin`, reports the seed, the v2-fixture fit decode, the delta spread
(the protocol's own pass criterion) and the v1-row linear probe, and writes
one consolidated report.  Zero optimizer steps on the subject; the probe's
own linear head is trained per checkpoint exactly like the runner does.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    evaluate,
    load_learner,
)
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from scripts.training.run_a_gate_pixel_20261117 import delta_spread  # noqa: E402
from scripts.training.run_a_gate_pixel_segment_20261117 import probe  # noqa: E402

OUT = ROOT / "reports/taiji_u2_a_gate_seed_probe_readout_20261119.json"


def main() -> int:
    torch.set_num_threads(4)
    pattern = sys.argv[1] if len(sys.argv) > 1 else "output/a_pixel_parallel_20261119*"
    fit = generate_v2("fit", 32)
    rows = []
    for checkpoint in sorted(glob.glob(str(ROOT / pattern / "segment.bin"))):
        label = Path(checkpoint).parent.name
        try:
            model, _, loaded = load_learner(Path(checkpoint))
        except Exception as error:  # noqa: BLE001 - report and continue
            print(json.dumps({"run": label, "load_error": repr(error)}), flush=True)
            continue
        result = evaluate(model, fit)
        row = {
            "run": label,
            "seed": loaded["description"]["config"]["seed"],
            "d": loaded["description"]["config"]["d"],
            "updates": loaded["update"],
            "fit_correct": result["correct"],
            "fit_denominator": result["denominator"],
            "delta_spread": round(delta_spread(model, fit), 6),
            "probe": probe(model),
        }
        rows.append(row)
        print(json.dumps({key: row[key] for key in row if key != "probe"}), flush=True)
    rows.sort(key=lambda row: (row["seed"], row["updates"]))
    OUT.write_bytes(
        json.dumps(
            {"schema": "taiji-a-gate-seed-sweep-readout-v1", "rows": rows},
            ensure_ascii=False,
            indent=2,
        ).encode()
        + b"\n"
    )
    print(f"wrote {OUT.relative_to(ROOT).as_posix()} with {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

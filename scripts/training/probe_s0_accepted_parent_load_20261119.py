"""Zero-training loadability/equivalence probe for the accepted 22.5.0 A parents.

S1's rebuild question has two layers: can the accepted v1 parent checkpoint be
loaded by the current code with its frozen identity, and does it reproduce the
behaviour recorded at acceptance?  This probe answers the load/reproduce layer
without any optimizer step and without reading check or final partitions.

The read-only loader lives in ``s0_parent_loader_20261119`` and reports the
source-pin mismatch set for disclosure; it refuses non-native schemas and any
non-v1 readout identity, and it cannot train.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from scripts.training.s0_parent_loader_20261119 import (  # noqa: E402
    ARM_REPORTS,
    CHECKPOINTS,
    load_historical,
)

OUT = ROOT / "reports/taiji_u2_s0_accepted_parent_load_20261119.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    from scripts.training.run_a_gate_pixel_20261117 import evaluate

    fit = generate_v2("fit", 32)
    rows = []
    for seed, checkpoint in CHECKPOINTS.items():
        arm = json.loads(ARM_REPORTS[seed].read_text(encoding="utf-8"))
        expected = int(arm["train"]["fit_correct"])
        expected_denominator = int(arm["train"]["fit_denominator"])
        try:
            model, payload, audit = load_historical(checkpoint)
            reading = evaluate(model, fit)
            reproduced = reading["correct"] == expected
            row = {
                "seed": seed,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "recorded_fit_correct": expected,
                "recorded_fit_denominator": expected_denominator,
                "loaded_fit_correct": reading["correct"],
                "loaded_fit_denominator": len(fit),
                "fit_rows_equal": reading["correct"] == expected,
                "update": payload.get("update"),
                "revision": payload.get("revision"),
                **audit,
                "verdict": ("load_reproduces_recorded_fit" if reproduced else "load_fit_mismatch"),
            }
        except Exception as error:  # noqa: BLE001 - report every refusal
            row = {
                "seed": seed,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "verdict": "load_refused",
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
    all_equivalent = all(row["verdict"] == "load_reproduces_recorded_fit" for row in rows)
    payload = {
        "schema": "taiji-u2-s0-accepted-parent-load-v1",
        "card": "22.5.1/S0 decision support (option 2 factual base)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "scope": (
            "read-only load + recorded-fit reproduction on the v2 fixture fit 32; "
            "zero optimizer steps, zero check/final reads, no capability signing"
        ),
        "fixture": "first_a_support_22_5_v2.generate_examples('fit', 32)",
        "rows": rows,
        "all_load_equivalent": all_equivalent,
        "verdict": (
            "accepted_parent_load_equivalent"
            if all_equivalent
            else "accepted_parent_load_not_equivalent"
        ),
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "rows": [
                    {
                        "seed": row["seed"],
                        "verdict": row["verdict"],
                        "recorded": row.get("recorded_fit_correct"),
                        "loaded": row.get("loaded_fit_correct"),
                        "pin_mismatches": len(row.get("source_pin_mismatches", [])),
                    }
                    for row in rows
                ],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if all_equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Zero-training loadability/equivalence probe for the accepted 22.5.0 A parents.

S1's rebuild question has two layers: can the accepted v1 parent checkpoint be
loaded by the current code with its frozen identity, and does it reproduce the
behaviour recorded at acceptance?  This probe answers the load/reproduce layer
without any optimizer step and without reading check or final partitions.

`first_a_support_22_5.load_learner` is fail-closed on live source pins, which
S0 changed after acceptance; the probe therefore uses a read-only historical
loader that (a) asserts schema, architecture and the v1 readout identity,
(b) compares every recorded pin against the current file and reports the
mismatch set, (c) requires a strict state-dict load, and (d) reproduces the
recorded fit reading with the same instrument and fixture version used at
acceptance.  It cannot be used to train or to sign anything: no optimizer step,
no check/final read, and every divergence is reported as `not_equivalent`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from taiji.u2_history import atomic_load, value_digest  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_s0_accepted_parent_load_20261119.json"
ARM_REPORTS = {
    20261110: ROOT / "reports/taiji_u2_a_parent_package_u2_seed20261110_20261119.json",
    20261111: ROOT / "reports/taiji_u2_a_parent_package_u2_seed20261111_20261119.json",
    20261112: ROOT / "reports/taiji_u2_a_parent_package_u2_seed20261112_20261119.json",
}
CHECKPOINTS = {
    20261110: ROOT / "output/a_pixel_parallel_20261119T063705Z/segment.bin",
    20261111: ROOT / "output/a_pixel_parallel_20261119T065049Z/segment.bin",
    20261112: ROOT / "output/a_pixel_parallel_20261119T070447Z/segment.bin",
}
V1_IDENTITY = "pixel-salience"


def current_pins(payload: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    status: dict[str, bool] = {}
    mismatched: list[str] = []
    for name, recorded in payload["source_pins"].items():
        path = ROOT / name
        if not path.is_file():
            status[name] = False
            mismatched.append(name)
            continue
        actual = hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()
        status[name] = actual == recorded
        if actual != recorded:
            mismatched.append(name)
    return status, mismatched


def load_historical(path: Path) -> tuple[U2Subject, dict[str, Any], dict[str, Any]]:
    """Read-only loader for the accepted v1 identity; returns (model, payload, audit)."""
    if not path.is_file():
        raise FileNotFoundError(f"accepted_parent_checkpoint_missing:{path}")
    payload = atomic_load(path)
    if payload.get("schema") != "taiji-first-A-learner-v1":
        raise ValueError("unsupported_first_A_checkpoint")
    description = payload["description"]
    if description.get("architecture") != "native":
        raise ValueError("unexpected_first_A_architecture")
    config = description["config"]
    if config.get("readout") != V1_IDENTITY:
        raise ValueError(f"unexpected_readout_identity:{config.get('readout')}")
    pin_status, mismatched = current_pins(payload)
    model = U2Subject(U2Config(**config))
    first = next(iter(payload["parameters"].values()))
    model = model.to(dtype=first.dtype, device=first.device)
    model.load_state_dict(payload["parameters"])  # strict: identity must match
    model.parameter_revision = payload["revision"]
    audit = {
        "schema": payload["schema"],
        "identity": config.get("readout"),
        "config": {
            key: config[key]
            for key in ("d", "readout", "record_metadata", "separate_requests", "seed")
            if key in config
        },
        "parameters_digest": value_digest(payload["parameters"]),
        "source_pin_matches": pin_status,
        "source_pin_mismatches": mismatched,
        "strict_load": True,
    }
    return model, payload, audit


def decode_fit(model: U2Subject, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scripts.training.run_a_gate_pixel_20261117 import evaluate

    return evaluate(model, examples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    fit = generate_v2("fit", 32)
    rows = []
    for seed, checkpoint in CHECKPOINTS.items():
        arm = json.loads(ARM_REPORTS[seed].read_text(encoding="utf-8"))
        expected = int(arm["train"]["fit_correct"])
        expected_denominator = int(arm["train"]["fit_denominator"])
        try:
            model, payload, audit = load_historical(checkpoint)
            reading = decode_fit(model, fit)
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

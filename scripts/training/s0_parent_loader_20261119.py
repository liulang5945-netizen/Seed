"""Read-only historical loader for the accepted 22.5.0 v1 A parents.

The accepted parents were saved by ``first_a_support_22_5.save_learner`` with
live source pins.  S0 changed ``taiji/u2_subject.py`` after acceptance, so the
frozen ``load_learner`` refuses them by design.  This module provides the
narrow, read-only counterpart used by diagnostics and the S0 localization card:
it asserts schema/architecture/readout identity, requires a STRICT state-dict
load, records the source-pin mismatch set for disclosure, and never trains,
writes back, or touches check/final partitions.

It is a diagnostic instrument, not a re-admission path: S1's rebuild check is
still the only thing that can establish generation-chain reproducibility.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from taiji.u2_history import atomic_load, value_digest
from taiji.u2_subject import U2Config, U2Subject

ROOT = Path(__file__).resolve().parents[2]
V1_IDENTITY = "pixel-salience"
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


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def source_pin_mismatches(payload: dict[str, Any]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for name, recorded in payload.get("source_pins", {}).items():
        path = ROOT / name
        status[name] = path.is_file() and sha_of(path) == recorded
    return status


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
    pins = source_pin_mismatches(payload)
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
        "source_pin_matches": pins,
        "source_pin_mismatches": sorted(name for name, ok in pins.items() if not ok),
        "strict_load": True,
    }
    return model, payload, audit

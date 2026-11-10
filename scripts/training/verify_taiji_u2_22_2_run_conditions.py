"""Attach a fresh, condition-bound zero-update round-trip to the frozen 22.2 run.

The earlier report is preserved. This check does not reconstruct or repeat its
two disposable Adam updates, and does not certify its trained-state values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STARTED = time.monotonic()

import torch

from scripts.training.check_report_run_conditions import write_report


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    target = args.report.resolve()
    if target.parent != ROOT / "reports" or target.exists():
        parser.error("A report must be a new file directly inside reports/.")
    torch.set_num_threads(1)
    conditions = {
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "dtype": "torch.float64",
        "python": sys.version,
        "torch": torch.__version__,
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    contract_path = ROOT / "plans/manifests/taiji_u2_22_2_contract_20261110.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for name, expected in contract["source_pins"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen reference source changed: {name}")
    script = ROOT / "scripts/training/verify_taiji_u2_subject_22_2.py"
    spec = importlib.util.spec_from_file_location("u2_condition_bound_roundtrip", script)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    parent = ROOT / ".seed_test_tmp"
    parent.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="u2_22_2_conditions_", dir=parent))
    if not directory.resolve().is_relative_to(ROOT.resolve() / ".seed_test_tmp"):
        raise ValueError("Unexpected temporary verification directory")
    try:
        reference.WORK = directory
        result = reference.engineering_run(contract, execute=False)
        assert result["optimizer_steps"] == 0
        report = {
            "schema": "taiji-u2-22.2-condition-bound-roundtrip-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "verdict": "passed_condition_bound_pre_update_roundtrip_only",
            "run_conditions": conditions,
            "frozen_contract_sha256": digest(contract_path),
            "frozen_base_report": {
                "path": "reports/taiji_u2_subject_22_2_20261110.json",
                "canonical_sha256": digest(ROOT / "reports/taiji_u2_subject_22_2_20261110.json"),
                "runtime_conditions_complete_in_original_report": False,
                "optimizer_steps_in_original_run": 2,
                "trained_state_equality_machine_verified_here": False,
            },
            "zero_update_checks": result["bundle_before_update"],
            "parameter_tensors": len(result["parameters"]),
            "state_fields": result["state_fields"],
            "optimizer_steps": 0,
            "existing_checkpoints_loaded": 0,
            "cpu_seconds": time.process_time(),
            "wall_seconds": time.monotonic() - STARTED,
            "semantic_alignment": "inconsistent_fixture_not_evaluated",
            "source_sha256": digest(Path(__file__)),
        }
        temporary = directory / "condition_report.writing"
        write_report(temporary, report)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(temporary.read_text(encoding="utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.unlink()
        print(
            json.dumps(
                {
                    "verdict": report["verdict"],
                    "optimizer_steps": 0,
                    "cpu_seconds": report["cpu_seconds"],
                }
            )
        )
    finally:
        for name in ("before_updates.pt", "writer.lock"):
            (directory / name).unlink(missing_ok=True)
        if not any(directory.iterdir()):
            directory.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

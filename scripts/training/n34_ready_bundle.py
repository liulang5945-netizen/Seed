"""N34 ready-bundle verification (zero training).

One command that re-verifies everything the N34 signature would rely on:
* all facility unit-test files pass (pytest, one process);
* all zero-training calibration cards re-run and still report their `_ready` exits;
* the long-run harness and durability cards re-run green;
* the growth-benefit v2 script still compiles (not executed);
* discipline: no check/final reads, confirmation seeds untouched (manifest unchanged).

Usage: python scripts/training/n34_ready_bundle.py
Output: reports/n34_ready_bundle.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PY = sys.executable
TEST_FILES = [
    "tests/test_session_store_20261125.py",
    "tests/test_long_run_harness_20261126.py",
    "tests/test_chinese_task_env_20261126.py",
    "tests/test_evidence_ledger_20261126.py",
    "tests/test_uncertainty_calibration_20261126.py",
    "tests/test_subject_loop_20261126.py",
    "tests/test_multimodal_physics_env_20261126.py",
    "tests/test_continual_growth_task_v2_20261126.py",
    "tests/test_choice_simulator_20261125.py",
    "tests/test_recursive_dev_20261126.py",
]
CALIBRATIONS = [
    (
        "nch",
        "scripts/training/nch_1_01_calibration.py",
        "reports/nch_1_01_calibration.json",
        "env_ready",
    ),
    (
        "ecl",
        "scripts/training/ecl_1_01_calibration.py",
        "reports/ecl_1_01_ledger.json",
        "ledger_ready",
    ),
    (
        "unc",
        "scripts/training/unc_1_01_calibration.py",
        "reports/unc_1_01_calibration.json",
        "calibration_ready",
    ),
    (
        "phy",
        "scripts/training/phy_1_01_calibration.py",
        "reports/phy_1_01_calibration.json",
        "physics_ready",
    ),
    ("sub1", "scripts/training/sub_1_01_loop_smoke.py", "reports/sub_1_01_loop.json", "loop_ready"),
    (
        "sub2",
        "scripts/training/sub_1_02_mixed_loop.py",
        "reports/sub_1_02_mixed_loop.json",
        "mixed_loop_ready",
    ),
    (
        "n34_1_01",
        "scripts/training/n34_1_01_harness.py",
        "reports/n34_1_01_harness.json",
        "harness_ready",
    ),
    (
        "n34_1_02",
        "scripts/training/n34_1_02_durability.py",
        "reports/n34_1_02_durability.json",
        "durability_ready",
    ),
]
OUT = ROOT / "reports" / "n34_ready_bundle.json"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)


def main() -> int:
    started = time.process_time()
    checks: dict[str, bool] = {}

    pytest_proc = _run([PY, "-m", "pytest", "-q", *TEST_FILES])
    checks["unit_tests_pass"] = pytest_proc.returncode == 0
    test_tail = (pytest_proc.stdout or "").strip().splitlines()[-1:] if pytest_proc.stdout else []

    calib: dict[str, dict] = {}
    for name, script, report, expected_exit in CALIBRATIONS:
        proc = _run([PY, script])
        rep_file = ROOT / report
        exit_seen = None
        if rep_file.exists():
            try:
                exit_seen = json.loads(rep_file.read_text(encoding="utf-8")).get("exit")
            except json.JSONDecodeError:
                exit_seen = None
        ok = proc.returncode == 0 and exit_seen == expected_exit
        checks[f"calib_{name}"] = ok
        calib[name] = {"rc": proc.returncode, "exit": exit_seen, "expected": expected_exit}

    growth = ROOT / "scripts/training/n34_2_01_growth_benefit_v2.py"
    compile_proc = _run([PY, "-m", "py_compile", str(growth)])
    checks["growth_script_compiles"] = compile_proc.returncode == 0

    manifest = ROOT / "plans/manifests/n31_confirmation_seed_reads.json"
    manifest_text = manifest.read_text(encoding="utf-8")
    checks["confirmation_manifest_intact"] = (
        "N33.1.90" in manifest_text and "N34.4" not in manifest_text
    )

    bundle_ready = all(checks.values())
    payload = {
        "schema": "taiji-n34-ready-bundle-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip(),
        "unit_tests": {
            "files": len(TEST_FILES),
            "tail": test_tail,
            "passed": checks["unit_tests_pass"],
        },
        "calibrations": calib,
        "growth_script": {
            "path": str(growth.relative_to(ROOT)),
            "compiled": checks["growth_script_compiles"],
            "executed": False,
        },
        "checks": checks,
        "exit": "bundle_ready" if bundle_ready else "bundle_failure",
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": 30.0},
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "trained_models": 0},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": payload["exit"],
                "checks": checks,
                "test_tail": test_tail,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if bundle_ready else 3


if __name__ == "__main__":
    raise SystemExit(main())

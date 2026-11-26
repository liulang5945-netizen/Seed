"""Cross-session lifecycle probe: dormant/retire persistence + wake in session 3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402

OUT = ROOT / "reports/taiji_u2_multi_session_lifecycle_20261120.json"
WORKER = ROOT / "scripts/training/s2_multi_session_lifecycle_worker_20261120.py"
OUTPUT = ROOT / "output/s2_multi_session_lifecycle"
CAP_CPU = 240.0
SEEDS = (0, 1, 2)
SESSIONS = (1, 2, 3)
RETENTION_MAX = 0.02


def run_sessions(seed: int) -> list[dict]:
    receipts = []
    previous = ""
    for session in SESSIONS:
        capsule_path = OUTPUT / f"seed{seed}_session{session}.pt"
        receipt_path = OUTPUT / f"seed{seed}_session{session}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(WORKER),
                "--seed",
                str(seed),
                "--session",
                str(session),
                "--in-capsule",
                previous,
                "--out-capsule",
                str(capsule_path),
                "--receipt",
                str(receipt_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"worker_failed seed={seed} session={session}: {completed.stderr[-500:]}"
            )
        receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
        previous = str(capsule_path)
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    child_cpu = 0.0
    for seed in SEEDS:
        receipts = run_sessions(seed)
        child_cpu += sum(receipt["cpu"] for receipt in receipts)
        first, second, third = receipts
        expected = second["birth"]["lifecycles"]
        persisted = all(third["lifecycles"].get(unit) == state for unit, state in expected.items())
        inheritance = (
            second["load_digest"] == first["save_digest"]
            and third["load_digest"] == second["save_digest"]
        )
        wake = third["wake"] or {}
        dormant_zero = all(value == 0.0 for value in (wake.get("dormant_before") or {}).values())
        wake_activity = float(wake.get("dormant_after_wake", 0.0))
        wake_changes = float(wake.get("logits_change", 0.0))
        params_unchanged = third["before_wake_digest"] == third["after_wake_digest"]
        retired = wake.get("retired", [])
        retired_zero = all(value == 0.0 for value in (wake.get("retired_activity") or {}).values())
        accuracies = [first["accuracy"], second["accuracy"], third["accuracy"]]
        retention = max(
            first["accuracy"] - second["accuracy"],
            second["accuracy"] - third["accuracy"],
        )
        results.append(
            {
                "seed": seed,
                "accuracies": accuracies,
                "inheritance_exact": inheritance,
                "lifecycle_persisted": persisted,
                "neonatal_states": expected,
                "wake": {
                    "dormant_before_zero": dormant_zero,
                    "activity_after": wake_activity,
                    "logits_change": wake_changes,
                    "params_unchanged": params_unchanged,
                    "retired": retired,
                    "retired_zero_activity": retired_zero,
                },
                "retention_drop": round(retention, 4),
            }
        )
    gates = {
        "inheritance_exact": all(result["inheritance_exact"] for result in results),
        "lifecycle_persisted": all(result["lifecycle_persisted"] for result in results),
        "dormant_zero_before_wake": all(
            result["wake"]["dormant_before_zero"] for result in results
        ),
        "wake_restores_activity": all(result["wake"]["activity_after"] > 0.0 for result in results),
        "wake_changes_readout": all(result["wake"]["logits_change"] > 0.0 for result in results),
        "wake_keeps_parameters": all(result["wake"]["params_unchanged"] for result in results),
        "retired_zero_activity": all(result["wake"]["retired_zero_activity"] for result in results),
        "retention_within_0_02": all(
            result["retention_drop"] <= RETENTION_MAX for result in results
        ),
    }
    verdict = (
        "multi_session_lifecycle_ready" if all(gates.values()) else "multi_session_lifecycle_absent"
    )
    payload = {
        "schema": "taiji-u2-multi-session-lifecycle-v1",
        "card": "cross-session dormant/retire persistence and wake probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": len(SESSIONS),
            "birth_session": 2,
            "wake_session": 3,
            "retention_max": RETENTION_MAX,
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": 400 * len(SESSIONS) * len(SEEDS),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started, 2),
            "children_cpu": round(child_cpu, 2),
        },
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cost": payload["cost"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "multi_session_lifecycle_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

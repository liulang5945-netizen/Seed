"""Multi-session birth/interitance probe: 30 -> 33 units across three processes.

Verifies that a structural birth inside session 2 is (a) neutral at birth, (b)
inherited bitwise into session 3, (c) keeps unit identities/lineage, and (d) the
newborn units remain present in the final capsule.  Three seeds; zero check.
"""

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

OUT = ROOT / "reports/taiji_u2_multi_session_birth_20261120.json"
WORKER = ROOT / "scripts/training/s2_multi_session_birth_worker_20261120.py"
OUTPUT = ROOT / "output/s2_multi_session_birth"
CAP_CPU = 240.0
SEEDS = (0, 1, 2)
SESSIONS = (1, 2, 3)
RETENTION_MAX = 0.02
NEUTRAL_MAX = 0.02


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
        second = receipts[1]
        third = receipts[2]
        birth = second.get("birth") or {}
        inheritance = (
            second["load_digest"] == receipts[0]["save_digest"]
            and third["load_digest"] == second["save_digest"]
        )
        units = [receipt["units"] for receipt in receipts]
        newborn_ids = second.get("newborn_ids", [])
        newborns_inherited = bool(newborn_ids) and all(
            unit_id in third["unit_ids"] for unit_id in newborn_ids
        )
        accuracies = [receipt["accuracy"] for receipt in receipts]
        retention = max(
            receipts[0]["accuracy"] - receipts[1]["accuracy"],
            receipts[1]["accuracy"] - receipts[2]["accuracy"],
        )
        results.append(
            {
                "seed": seed,
                "accuracies": accuracies,
                "units": units,
                "inheritance_exact": inheritance,
                "newborns": newborn_ids,
                "newborns_inherited": newborns_inherited,
                "neutral_drop": birth.get("neutral_drop", None),
                "retention_drop": round(retention, 4),
            }
        )
    gates = {
        "inheritance_exact": all(result["inheritance_exact"] for result in results),
        "newborns_inherited": all(result["newborns_inherited"] for result in results),
        "birth_neutral": all(
            result["neutral_drop"] is not None and result["neutral_drop"] <= NEUTRAL_MAX
            for result in results
        ),
        "retention_within_0_02": all(
            result["retention_drop"] <= RETENTION_MAX for result in results
        ),
    }
    verdict = "multi_session_birth_ready" if all(gates.values()) else "multi_session_birth_absent"
    payload = {
        "schema": "taiji-u2-multi-session-birth-v1",
        "card": "cross-session structural birth inheritance probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": len(SESSIONS),
            "birth_session": 2,
            "birth_count": 3,
            "retention_max": RETENTION_MAX,
            "neutral_max": NEUTRAL_MAX,
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
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "multi_session_birth_ready" else 1


def torch_threads() -> int:
    import torch  # noqa: PLC0415

    return torch.get_num_threads()


if __name__ == "__main__":
    raise SystemExit(main())

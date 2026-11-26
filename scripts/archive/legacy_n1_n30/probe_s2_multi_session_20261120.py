"""Multi-session continuity feasibility: three cross-process sessions (zero check).

Runs the worker three times per seed in separate processes; each later session
loads the previous capsule into a fresh graph, continues training, and saves a new
capsule.  Checks: parameter inheritance digest matches the previous save; unit
identities/lineage preserved; fit accuracy retention across sessions within 0.02.

Zero check; cap 180 CPU.  This is a lifetimes-axis feasibility probe only, not the
22.8 multi-session acceptance card.
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

OUT = ROOT / "reports/taiji_u2_multi_session_20261120.json"
WORKER = ROOT / "scripts/training/s2_multi_session_worker_20261120.py"
OUTPUT = ROOT / "output/s2_multi_session"
CAP_CPU = 180.0
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
                f"worker_failed seed={seed} session={session}: {completed.stderr[-400:]}"
            )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["capsule"] = str(capsule_path.relative_to(ROOT))
        receipts.append(receipt)
        previous = str(capsule_path)
    return receipts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch_threads = 1
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    child_cpus = []
    for seed in SEEDS:
        receipts = run_sessions(seed)
        child_cpus.append([receipt.get("cpu", 0.0) for receipt in receipts])
        inheritance = all(
            receipts[index]["load_digest"] == receipts[index - 1]["save_digest"]
            for index in range(1, len(receipts))
        )
        identities = all(
            receipts[index]["unit_ids"] == receipts[0]["unit_ids"]
            and receipts[index]["n_units_graph"] == receipts[0]["n_units_graph"]
            and receipts[index]["identity_digest"] == receipts[0]["identity_digest"]
            for index in range(1, len(receipts))
        )
        accuracies = [receipt["accuracy"] for receipt in receipts]
        retention = max(accuracies[0] - value for value in accuracies[1:])
        results.append(
            {
                "seed": seed,
                "accuracies": accuracies,
                "inheritance_exact": inheritance,
                "identities_preserved": identities,
                "retention_drop": round(retention, 4),
            }
        )
    gates = {
        "inheritance_exact": all(result["inheritance_exact"] for result in results),
        "identities_preserved": all(result["identities_preserved"] for result in results),
        "retention_within_0_02": all(
            result["retention_drop"] <= RETENTION_MAX for result in results
        ),
    }
    verdict = (
        "multi_session_continuity_ready"
        if all(gates.values())
        else "multi_session_continuity_absent"
    )
    payload = {
        "schema": "taiji-u2-multi-session-v1",
        "card": "three cross-process sessions continuity feasibility",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch_threads,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "sessions": len(SESSIONS),
            "seeds": list(SEEDS),
            "steps_per_session": 400,
            "worker": "scripts/training/s2_multi_session_worker_20261120.py",
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
            "children_cpu": round(sum(result for receipts in child_cpus for result in receipts), 2),
        },
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "multi_session_continuity_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

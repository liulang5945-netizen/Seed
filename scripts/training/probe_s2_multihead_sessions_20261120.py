"""Cross-session multi-task-head probe: three heads, three sessions, exact retention."""

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

OUT = ROOT / "reports/taiji_u2_multihead_sessions_20261120.json"
WORKER = ROOT / "scripts/training/s2_multihead_worker_20261120.py"
OUTPUT = ROOT / "output/s2_multihead_sessions"
CAP_CPU = 180.0
SEEDS = (0, 1, 2)
SESSIONS = (1, 2, 3)
RETENTION_MAX = 1e-9
NEW_TASK_MIN = 0.60


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
        inheritance = all(
            receipts[index]["load_digest"] == receipts[index - 1]["save_digest"] for index in (1, 2)
        )
        old_retention = first["accuracies"]["old"] - third["accuracies"]["old"]
        new_retention = second["accuracies"]["new"] - third["accuracies"]["new"]
        results.append(
            {
                "seed": seed,
                "old_after_session3": third["accuracies"]["old"],
                "new_after_session3": third["accuracies"]["new"],
                "direction_after_session3": third["accuracies"]["direction"],
                "inheritance_exact": inheritance,
                "old_retention_drop": round(old_retention, 6),
                "new_retention_drop": round(new_retention, 6),
            }
        )
    gates = {
        "inheritance_exact": all(result["inheritance_exact"] for result in results),
        "old_exact": all(result["old_retention_drop"] <= RETENTION_MAX for result in results),
        "new_exact": all(result["new_retention_drop"] <= RETENTION_MAX for result in results),
        "third_task_learnable": all(
            result["direction_after_session3"] >= NEW_TASK_MIN for result in results
        ),
    }
    verdict = "multihead_sessions_ready" if all(gates.values()) else "multihead_sessions_absent"
    payload = {
        "schema": "taiji-u2-multihead-sessions-v1",
        "card": "cross-session three-head isolation (joint24, action bins, push direction)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": len(SESSIONS),
            "heads": 3,
            "head_hidden": 16,
            "task_a": "joint24",
            "task_b": "action bins (3 classes)",
            "task_c": "push direction (2 classes)",
            "retention_max": RETENTION_MAX,
            "third_task_min": NEW_TASK_MIN,
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (1200 + 600 * 2) * len(SEEDS),
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
    return 0 if verdict == "multihead_sessions_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())

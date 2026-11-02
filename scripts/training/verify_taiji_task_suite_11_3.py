"""11.3: run the fixed-answer Chinese suite through the entry and score it exactly.

One session, tasks in order (the follow-up and correction depend on the material item).
Exact match after stripping whitespace and terminal punctuation. The clarify and unknown
items are recorded but kept out of the denominator, and the exclusion is reported.

Nothing is trained; this is the suite's floor reading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest  # noqa: E402
from taiji.candidate_session import CandidateSession  # noqa: E402
from taiji.native_task_suite_11_3 import TASKS, normalize  # noqa: E402

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
MAX_BYTES = 16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.3 card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    bundle = ROOT / BUNDLE
    manifest = build_manifest(ROOT, bundle)

    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        session = CandidateSession(ROOT, manifest_path, bundle, "frozen_eval")
        rows = []
        hits = 0
        scored = 0
        for task in TASKS:
            session.observe(task["prompt"])
            response = session.respond(MAX_BYTES)
            text = response["text"]
            if task["scored"]:
                scored += 1
                hit = normalize(text) == normalize(task["expected"])
                hits += int(hit)
            else:
                hit = None
            rows.append(
                {
                    "id": task["id"],
                    "category": task["category"],
                    "prompt": task["prompt"],
                    "expected": task["expected"],
                    "answer": text,
                    "bytes": len(response["raw"]["steps"]),
                    "scored": task["scored"],
                    "hit": hit,
                    "note": task.get("note"),
                }
            )

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    else:
        status = "task_suite_floor_measured"

    report = {
        "schema": "taiji-task-suite-11-3-v1",
        "step": "11.3",
        "date": "2026-11-01",
        "status": status,
        "candidate_id": manifest["candidate_id"],
        "rubric": "exact match after stripping whitespace and terminal punctuation",
        "tasks": len(TASKS),
        "scored_tasks": scored,
        "excluded_tasks": len(TASKS) - scored,
        "exclusion_reason": "clarify and unknown have no model-driven channel in this entry",
        "exact_hits": hits,
        "exact_rate": hits / scored if scored else None,
        "max_bytes": MAX_BYTES,
        "rows": rows,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Floor reading only: a byte-level generative candidate answering a fixed-answer "
            "suite through the raw-first entry. Two items are excluded because the entry has "
            "no clarify/abstain channel; that exclusion is itself a gap in the entry, not a "
            "result about the candidate."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_task_suite_11_3.py",
                "taiji/candidate_session.py",
                "scripts/training/verify_taiji_task_suite_11_3.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "exact_hits": hits,
                "scored": scored,
                "excluded": len(TASKS) - scored,
                "rate": round(hits / scored, 3) if scored else None,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

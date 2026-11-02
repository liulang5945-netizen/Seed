"""11.4c: hard-exit after a checkpoint, then resume in a fresh process.

11.2b proved the snapshot round-trip inside one process. This runs the same idea across
a real process death:

* child A loads the candidate, observes P1, writes the snapshot, then calls ``os._exit(0)``
  -- no cleanup, no interpreter shutdown;
* child B is a fresh process that loads that snapshot, observes P2 and prints the answer;
* the parent does the same two observations in one session and compares.

A match means the checkpoint is complete at the moment of a hard exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest  # noqa: E402
from taiji.candidate_session import CandidateSession  # noqa: E402

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
PROMPT_ONE = "目标是让圆口的器皿到四方木台。"
PROMPT_TWO = "圆口的器皿在木桌。"

CHILD_A = """
import os, sys
from pathlib import Path
sys.path.insert(0, {root!r})
from taiji.candidate_session import CandidateSession
session = CandidateSession(
    Path({root!r}), Path(sys.argv[1]), Path(sys.argv[2]), "frozen_eval"
)
session.observe({prompt_one!r})
session.save(Path(sys.argv[3]))
os._exit(0)
"""

CHILD_B = """
import json, sys
from pathlib import Path
sys.path.insert(0, {root!r})
from taiji.candidate_session import CandidateSession
session = CandidateSession(
    Path({root!r}), Path(sys.argv[1]), Path(sys.argv[2]), "frozen_eval"
)
session.load(Path(sys.argv[3]))
session.observe({prompt_two!r})
print(json.dumps(session.answer()["raw"]))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.4c card is write-once")

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
        snapshot = Path(directory) / "snapshot.pt"

        parent = CandidateSession(ROOT, manifest_path, bundle, "frozen_eval")
        parent.observe(PROMPT_ONE)
        parent_digest = parent.transcript[-1]["state"]
        parent.observe(PROMPT_TWO)
        parent_answer = parent.answer()["raw"]

        child_a = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD_A.format(root=str(ROOT), prompt_one=PROMPT_ONE),
                str(manifest_path),
                str(bundle),
                str(snapshot),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        snapshot_written = snapshot.exists() and snapshot.stat().st_size > 0

        child_b = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD_B.format(root=str(ROOT), prompt_two=PROMPT_TWO),
                str(manifest_path),
                str(bundle),
                str(snapshot),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        child_answer = None
        if child_b.returncode == 0 and child_b.stdout.strip():
            child_answer = json.loads(child_b.stdout.strip().splitlines()[-1])

    same_answer = child_answer == parent_answer
    both_ran = child_a.returncode == 0 and child_b.returncode == 0
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif not (both_ran and snapshot_written):
        status = "interrupt_failed"
    elif not same_answer:
        status = "interrupt_diverged"
    else:
        status = "interrupt_recovery_verified"

    report = {
        "schema": "taiji-interrupt-11-4c-v1",
        "step": "11.4c",
        "date": "2026-11-01",
        "status": status,
        "candidate_id": manifest["candidate_id"],
        "child_a_returncode": child_a.returncode,
        "child_b_returncode": child_b.returncode,
        "child_a_stderr_tail": (child_a.stderr or "").strip().splitlines()[-1:] or [],
        "child_b_stderr_tail": (child_b.stderr or "").strip().splitlines()[-1:] or [],
        "snapshot_written": snapshot_written,
        "hard_exit_used": "os._exit(0)",
        "answers_identical": same_answer,
        "parent_answer": parent_answer,
        "child_answer": child_answer,
        "parent_state_after_observe": parent_digest,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Interrupt semantics only: a child wrote a checkpoint and died without cleanup; "
            "a fresh process resumed it and produced the same answer. It does not test "
            "mid-computation crashes or OS-level power loss."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/candidate_session.py",
                "scripts/training/verify_taiji_interrupt_recovery_11_4c.py",
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
                "child_a_rc": child_a.returncode,
                "child_b_rc": child_b.returncode,
                "snapshot": snapshot_written,
                "answers_identical": same_answer,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

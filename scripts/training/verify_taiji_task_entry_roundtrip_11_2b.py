"""11.2b: the remaining entry calls -- save / load / outcome / act.

Checks on the 09.8 candidate:

* a snapshot saved after one observation reproduces the same answer, and a session
  restored from it continues bit-identically through a second observation;
* ``outcome`` records the result without touching the state (frozen mode never learns);
* ``act`` is refused with ``no_action_channel`` -- this candidate has no action path;
* a snapshot carrying another candidate's id is refused.

Nothing is trained.
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

BUNDLE = "output/taiji_core_09/subject_bundle.pt"
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
PROMPT_ONE = "目标是让圆口的器皿到四方木台。"
PROMPT_TWO = "圆口的器皿在木桌。"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.2b card is write-once")

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

        first = CandidateSession(ROOT, manifest_path, bundle, "frozen_eval")
        first.observe(PROMPT_ONE)
        answer_one = first.answer()["raw"]
        first.save(snapshot)

        second = CandidateSession(ROOT, manifest_path, bundle, "frozen_eval")
        second.load(snapshot)
        answer_one_restored = second.answer()["raw"]
        second.observe(PROMPT_TWO)
        answer_two_restored = second.answer()["raw"]

        first.observe(PROMPT_TWO)
        answer_two = first.answer()["raw"]

        outcome = first.outcome("intent-1", "success", "owner")
        act = first.act()

        foreign = torch.load(snapshot, map_location="cpu", weights_only=True)
        foreign["candidate_id"] = "0" * 64
        foreign_path = Path(directory) / "foreign.pt"
        torch.save(foreign, foreign_path)
        foreign_refused = False
        try:
            second.load(foreign_path)
        except ValueError:
            foreign_refused = True

    same_first = answer_one == answer_one_restored
    same_second = answer_two == answer_two_restored
    outcome_ok = outcome["state_unchanged"] is True
    act_ok = act["refused"] == "no_action_channel"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif not (same_first and same_second):
        status = "snapshot_mismatch"
    elif not foreign_refused:
        status = "foreign_snapshot_accepted"
    elif not outcome_ok:
        status = "outcome_mutates_state"
    elif not act_ok:
        status = "act_not_refused"
    else:
        status = "entry_roundtrip_verified"

    report = {
        "schema": "taiji-task-entry-11-2b-v1",
        "step": "11.2b",
        "date": "2026-11-01",
        "status": status,
        "candidate_id": manifest["candidate_id"],
        "answer_after_load_identical": same_first,
        "continuation_identical": same_second,
        "outcome_state_unchanged": outcome_ok,
        "outcome_result": outcome["result"],
        "act_refused": act["refused"],
        "foreign_snapshot_refused": foreign_refused,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Entry round-trip only: snapshots restore the state bit-identically, outcomes are "
            "logged without learning, and act is refused because there is no action path. It "
            "does not test answer quality and does not pretend the entry is complete."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/candidate_session.py",
                "scripts/training/verify_taiji_task_entry_roundtrip_11_2b.py",
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
                "same_first": same_first,
                "same_second": same_second,
                "outcome_ok": outcome_ok,
                "act_ok": act_ok,
                "foreign_refused": foreign_refused,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

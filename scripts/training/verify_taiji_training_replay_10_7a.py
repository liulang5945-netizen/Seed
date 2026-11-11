"""10.7a replay: reload the saved 10.7 weights and reproduce their published readings.

The training card reports what each arm answered on the six held-out samples. This
step rebuilds the same subject from its declared seed, loads each saved
``state_dict`` with ``strict=True`` and recomputes the same twelve verdicts. It
trains nothing, writes no weights and publishes nothing.

Two outcomes are distinguished on purpose: bit-identical replay is the strong
claim, and a tolerance match is a weaker one that must be named as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.goal_conditioned_training import accuracy, samples  # noqa: E402

CARD = "reports/taiji_core_goal_training_10_7_20261101.json"
TRAIN_STATES = 3
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
TOLERANCE = 1e-9


def split_samples() -> tuple[list[dict], list[dict]]:
    rows = samples()
    per_state = len(rows) // 6
    cut = TRAIN_STATES * per_state
    return rows[:cut], rows[cut:]


def compare_arm(stored: dict, replay: dict) -> dict:
    stored_rows = stored.get("held_out") or []
    replay_rows = replay.get("rows") or []
    deltas = []
    present = True
    hits_match = len(stored_rows) == len(replay_rows) and bool(replay_rows)
    for old, new in zip(stored_rows, replay_rows):
        for key in ("correct_verdict", "other_verdict"):
            if not isinstance(old.get(key), (int, float)) or not isinstance(
                new.get(key), (int, float)
            ):
                hits_match = False
                present = False
                continue
            deltas.append(abs(new[key] - old[key]))
        if old.get("hit") != new.get("hit"):
            hits_match = False
    return {
        "rows": len(replay_rows),
        "stored_rows": len(stored_rows),
        "hits": replay.get("hits"),
        "stored_hits": stored.get("held_out_hits"),
        "hits_match": hits_match,
        "max_abs_delta": max(deltas) if deltas else None,
        "bitwise_equal": present and bool(deltas) and all(delta == 0.0 for delta in deltas),
    }


def status_from(*, files_ok: bool, replay_ok: bool, tolerance_ok: bool, budget_ok: bool) -> str:
    if not budget_ok:
        return "budget_stop"
    if not files_ok:
        return "engineering_failure"
    if replay_ok:
        return "weights_replay_bitwise"
    if tolerance_ok:
        return "weights_replay_within_tolerance"
    return "weights_replay_mismatch"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7a replay card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    card = json.loads((ROOT / CARD).read_text(encoding="utf-8"))
    expected_identity = card.get("parent_model_digest")
    expected_steps = card.get("declared", {}).get("steps")

    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    _train_rows, held_rows = split_samples()

    arms, files_ok = [], True
    for run in card.get("runs") or []:
        relative = run.get("weights_path")
        path = ROOT / relative if relative else None
        exists = path is not None and path.exists()
        payload = torch.load(path, map_location="cpu", weights_only=True) if exists else {}
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        bytes_match = exists and path.stat().st_size == run.get("weights_bytes")
        sized = exists and bytes_match
        files_ok = files_ok and sized
        if sized:
            subject.load_state_dict(state, strict=True)
            replay = accuracy(subject, held_rows, provider)
        else:
            replay = {"hits": None, "rows": []}
        row = {
            "lr": run.get("lr"),
            "steps": expected_steps,
            "weights_path": relative,
            "weights_bytes": run.get("weights_bytes"),
            "file_bytes": path.stat().st_size if exists else None,
            "file_bytes_match": bytes_match,
            "weights_sha256": (hashlib.sha256(path.read_bytes()).hexdigest() if exists else None),
            "load_strict": sized,
        }
        row.update(compare_arm(run, replay))
        arms.append(row)

    replay_ok = files_ok and all(
        arm["hits_match"] and arm["rows"] == 6 and arm["bitwise_equal"] for arm in arms
    )
    tolerance_ok = files_ok and all(
        arm["hits_match"]
        and arm["rows"] == 6
        and isinstance(arm["max_abs_delta"], (int, float))
        and arm["max_abs_delta"] <= TOLERANCE
        for arm in arms
    )

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = status_from(
        files_ok=files_ok,
        replay_ok=replay_ok,
        tolerance_ok=tolerance_ok,
        budget_ok=cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
    )

    report = {
        "schema": "taiji-training-replay-10-7a-v1",
        "step": "10.7a",
        "date": "2026-11-01",
        "status": status,
        "parent_model_digest": expected_identity,
        "rebuilt_digest": digest,
        "identity_match": digest == expected_identity,
        "steps": expected_steps,
        "held_out_samples": len(held_rows),
        "tolerance": TOLERANCE,
        "arms": arms,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "weights_published": False,
        "limits": (
            "Replay check only: it reloads already-saved weights and recomputes twelve verdicts. "
            "It trains nothing, changes no default and adds no capability evidence. A bitwise "
            "replay shows the saved artifact reproduces its published reading; it does not show "
            "that the arm learned goal-conditioned selection."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                CARD,
                "scripts/training/verify_taiji_training_replay_10_7a.py",
                "taiji/goal_conditioned_training.py",
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
                "identity_match": report["identity_match"],
                "arms": [
                    {
                        "lr": arm["lr"],
                        "hits": arm["hits"],
                        "stored_hits": arm["stored_hits"],
                        "bitwise": arm["bitwise_equal"],
                        "max_abs_delta": arm["max_abs_delta"],
                        "bytes_match": arm["file_bytes_match"],
                    }
                    for arm in arms
                ],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

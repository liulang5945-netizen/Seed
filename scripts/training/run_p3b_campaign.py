"""P3b campaign driver: train one arm, score every checkpoint, and honour the frozen stop rules.

One process owns one arm.  The trainer (``train_p3b_aligned.py --arm …``) is a child; each
checkpoint it writes is scored with the **same chain P3a was scored on** (relaxed legacy guard +
UTF-8 constrained decode) through ``eval_taiji_cap0_baseline.py``.  The campaign record is rewritten
atomically after every stage, so an interrupted 48-hour run loses nothing.

Two arms, one difference (decided 2026-09-15, ceiling option):
* ``--arm treatment`` trains on the dialogue-dense subset (P3b 预注册 §2 data row);
* ``--arm control`` trains on the raw ``simple_zh`` stream the 16M-tick state was trained on,
  with the **same** start checkpoint, budget, objective and chain.
Without the control, a gain could equally be "more training" or "the fresh identity organ the
guard attaches", and neither would say anything about H-P3b (data distribution).

Stop conditions (§4; the definitions are written into the report so they are not re-litigated):
1. budget exhausted (child exits on its own);
2. three consecutive checkpoints with no C/D/E improvement over the P3a baseline;
3. regression -- a mechanised dimension below its P3a value, or B/G pending-review counts up --
   then the child is terminated and the start checkpoint stays untouched.

Usage::

    python -X utf8 -u scripts/training/run_p3b_campaign.py --arm treatment --budget-tier 48h
    python -X utf8 -u scripts/training/run_p3b_campaign.py --arm control   --budget-tier 48h
    python -X utf8 -u scripts/training/run_p3b_campaign.py --arm treatment \\
        --max-symbols 12000 --checkpoint-every 6000      # plumbing smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_p3b_aligned import arm_paths  # noqa: E402

TRAINER = PROJECT_ROOT / "scripts" / "training" / "train_p3b_aligned.py"
EVALUATOR = PROJECT_ROOT / "scripts" / "training" / "eval_taiji_cap0_baseline.py"
CRITERIA = PROJECT_ROOT / "scripts" / "training" / "check_p3b_criteria.py"
P3A_BASELINE = PROJECT_ROOT / "reports" / "taiji_cap0_baseline_constrained_20260915.json"
REQUIRED_CHAIN = {"relax_legacy_guard": True, "constrained_decode": True}
MECHANISED = ("C", "D", "E")
PENDING_DIMS = ("B", "G")
STALL_LIMIT = 3
#: 20 items per dimension, so one flipped item is 0.05; two flipped items is the noise floor.
REGRESSION_MARGIN = 0.10
PROTECTED_CHECKPOINTS = (
    PROJECT_ROOT / "checkpoints" / "seed_corpus.pt",
    PROJECT_ROOT / "checkpoints" / "seed_beta.pt",
)


def campaign_paths(arm: str) -> tuple[Path, Path, Path]:
    """(working checkpoint, stage dir, campaign report).  The checkpoint is the trainer's own
    arm path, so the two processes can never disagree about which file is live."""

    checkpoint = arm_paths(arm)[0]
    stage_dir = PROJECT_ROOT / "reports" / "p3b_stages" / arm
    report = PROJECT_ROOT / "reports" / f"taiji_p3b_campaign_{arm}_20260915.json"
    return checkpoint, stage_dir, report


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalised(report: dict[str, Any], key: str) -> float | None:
    tally = report.get("dimensions", {}).get(key, {}).get("tally") or {}
    value = tally.get("machine_normalised")
    return float(value) if isinstance(value, (int, float)) else None


def _pending(report: dict[str, Any], key: str) -> int | None:
    tally = report.get("dimensions", {}).get(key, {}).get("tally") or {}
    value = tally.get("pending_human_review_items")
    return int(value) if isinstance(value, int) else None


def _protected_state() -> dict[str, list[int]]:
    state: dict[str, list[int]] = {}
    for path in PROTECTED_CHECKPOINTS:
        try:
            stat = os.stat(path)
        except OSError:
            state[path.name] = [-1, -1]
            continue
        state[path.name] = [stat.st_size, int(stat.st_mtime)]
    return state


def _write(report_path: Path, record: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)


def _file_state(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_size, int(stat.st_mtime))


def _latest_tick(checkpoint: Path) -> int:
    """Tick recorded in the checkpoint envelope; 0 while a write is still landing."""

    import torch

    try:
        envelope = torch.load(checkpoint, weights_only=False)
    except Exception:
        return 0
    return int((envelope.get("metadata") or {}).get("tick", 0))


def _snapshot(checkpoint: Path, tick: int) -> Path:
    """Freeze the live checkpoint before scoring it.

    One CAP-0 stage costs ~205 s and the trainer overwrites its checkpoint while the stage is
    running, so scoring the live file could mix two training states across dimensions.
    """

    directory = checkpoint.parent / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / f"{checkpoint.stem}_tick_{tick}.pt"
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        temporary.write_bytes(checkpoint.read_bytes())
        temporary.replace(frozen)
    return frozen


def _evaluate(checkpoint: Path, stage_path: Path) -> dict[str, Any]:
    """Score one checkpoint on the frozen eval set with the P3a chain (fresh process per stage)."""

    if stage_path.exists():
        return _load(stage_path)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        str(EVALUATOR),
        "--checkpoint",
        str(checkpoint),
        "--report",
        str(stage_path),
        "--relax-legacy-guard",
        "--constrained-decode",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not stage_path.exists():
        tail = (completed.stderr or completed.stdout or "")[-500:]
        raise SystemExit(f"stage evaluation failed (rc={completed.returncode}): {tail}")
    report = _load(stage_path)
    report["_stage_seconds"] = round(time.perf_counter() - started, 1)
    return report


def _stage_row(
    tick: int, report: dict[str, Any], baseline: dict[str, Any], stage_name: str
) -> dict[str, Any]:
    scores = {key: _normalised(report, key) for key in MECHANISED}
    deltas = {
        key: (
            None
            if scores[key] is None or _normalised(baseline, key) is None
            else round(scores[key] - _normalised(baseline, key), 4)
        )
        for key in MECHANISED
    }
    pending_delta = {
        key: (
            None
            if _pending(report, key) is None or _pending(baseline, key) is None
            else _pending(report, key) - _pending(baseline, key)
        )
        for key in PENDING_DIMS
    }
    numeric = [value for value in deltas.values() if value is not None]
    return {
        "tick": tick,
        "report": stage_name,
        "stage_seconds": report.get("_stage_seconds"),
        "chain": report.get("chain"),
        "chain_matches_p3a": report.get("chain") == REQUIRED_CHAIN,
        "trained_during_eval": report.get("trained_during_eval"),
        "scores": scores,
        "deltas_vs_p3a": deltas,
        "pending_delta_vs_p3a": pending_delta,
        "improved": bool(numeric) and max(numeric) > 0,
        "all_three_strictly_higher": bool(len(numeric) == 3)
        and all(value > 0 for value in numeric),
        "regressed": bool(numeric) and min(numeric) < 0,
        "pending_grew": any(value is not None and value > 0 for value in pending_delta.values()),
    }


def _stall_streak(stages: list[dict[str, Any]]) -> int:
    streak = 0
    for stage in reversed(stages):
        if stage["improved"]:
            break
        streak += 1
    return streak


def _regression_kind(stages: list[dict[str, Any]]) -> str | None:
    """Noise is not regression: one item of twenty is 0.05 of a dimension.

    ``material``  -- some mechanised dimension sits at or below ``baseline - 0.10`` (two items).
    ``persistent`` -- the same dimension is below its P3a value at two consecutive checkpoints.
    """

    if not stages:
        return None
    last = stages[-1]["deltas_vs_p3a"]
    if any(value is not None and value <= -REGRESSION_MARGIN for value in last.values()):
        return "material"
    if len(stages) < 2:
        return None
    previous = stages[-2]["deltas_vs_p3a"]
    for key in MECHANISED:
        if (last.get(key) or 0) < 0 and (previous.get(key) or 0) < 0:
            return "persistent"
    return None


def run(
    *,
    arm: str,
    budget_args: list[str],
    checkpoint_every: int,
    poll_seconds: int,
    device: str,
) -> dict[str, Any]:
    checkpoint, stage_dir, report_path = campaign_paths(arm)
    baseline = _load(P3A_BASELINE)
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        str(TRAINER),
        "--arm",
        arm,
        *budget_args,
        "--checkpoint-every",
        str(checkpoint_every),
        "--device",
        device,
    ]
    print(
        json.dumps({"event": "p3b_campaign_launch", "arm": arm, "pid": os.getpid()}),
        flush=True,
    )
    child = subprocess.Popen(command, cwd=str(PROJECT_ROOT))
    record: dict[str, Any] = {
        "format": "taiji-p3b-campaign-v1",
        "status": "running",
        "arm": arm,
        "campaign_stop": None,
        "companion_arm": "control" if arm == "treatment" else "treatment",
        "preregistration": (
            "plans/reference/M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md"
        ),
        "why_two_arms": (
            "the relaxed guard attaches a fresh identity organ and any budget adds training, "
            "so a single arm cannot separate data distribution from those two effects"
        ),
        "p3a_baseline": P3A_BASELINE.name,
        "baseline_scores": {key: _normalised(baseline, key) for key in MECHANISED},
        "stop_definitions": {
            "improved": "max(C/D/E delta vs P3a) > 0",
            "stall": f"{STALL_LIMIT} consecutive checkpoints with improved=false",
            "regression": (
                f"material: any mechanised delta <= -{REGRESSION_MARGIN} (two items of twenty); "
                "persistent: the same dimension below P3a at two consecutive checkpoints; "
                "or B/G pending-review count above P3a"
            ),
            "stage_scoring": (
                "the live checkpoint is copied to checkpoints/p3b/snapshots before scoring, "
                "because a stage costs ~205 s while the trainer overwrites the file"
            ),
            "chain_required": REQUIRED_CHAIN,
        },
        "budget_args": budget_args,
        "checkpoint_every": checkpoint_every,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "stages": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trainer_pid": child.pid,
        "protected_checkpoints_at_start": _protected_state(),
    }
    _write(report_path, record)

    last_state = _file_state(checkpoint)
    campaign_stop: str | None = None
    while True:
        state = _file_state(checkpoint)
        alive = child.poll() is None
        if state != last_state and state[0] > 0:
            time.sleep(2)  # let the atomic replace land before reading it back
            last_state = state
            tick = _latest_tick(checkpoint)
            if tick <= 0:
                time.sleep(poll_seconds)
                continue
            stage_path = stage_dir / f"cap0_tick_{tick}.json"
            frozen = _snapshot(checkpoint, tick)
            scored = _evaluate(frozen, stage_path)
            row = _stage_row(tick, scored, baseline, stage_path.name)
            row["snapshot"] = frozen.name
            record["stages"].append(row)
            record["stall_streak"] = _stall_streak(record["stages"])
            print(
                json.dumps({"event": "p3b_stage", "arm": arm, **row}, ensure_ascii=True),
                flush=True,
            )
            kind = _regression_kind(record["stages"])
            row["regression_kind"] = kind
            if kind is not None or row["pending_grew"]:
                campaign_stop = "regressed"
            elif record["stall_streak"] >= STALL_LIMIT:
                campaign_stop = "stalled"
            if campaign_stop and alive:
                child.terminate()
                record["terminated_trainer_pid"] = child.pid
            _write(report_path, record)
        if not alive or campaign_stop:
            break
        time.sleep(poll_seconds)

    record["trainer_exit_code"] = child.wait()
    record["campaign_stop"] = campaign_stop or (
        "budget_exhausted" if record["trainer_exit_code"] == 0 else "trainer_failed"
    )
    if record["stages"]:
        scored_stages = [stage for stage in record["stages"] if stage["scores"]["D"] is not None]
        best = max(scored_stages, key=lambda stage: stage["scores"]["D"] or 0.0, default=None)
        record["best_stage_tick"] = None if best is None else best["tick"]
        record["best_stage"] = None if best is None else best["report"]
        criteria_report = PROJECT_ROOT / "reports" / f"taiji_p3b_criteria_check_{arm}_20260915.json"
        subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(CRITERIA),
                "--baseline",
                str(P3A_BASELINE),
                "--candidate",
                str(stage_dir / record["stages"][-1]["report"]),
                "--output",
                str(criteria_report),
            ],
            check=False,
        )
        if criteria_report.exists():
            record["criteria"] = _load(criteria_report)
    end_state = _protected_state()
    record["protected_checkpoints_at_end"] = end_state
    record["protected_checkpoints_unchanged"] = (
        end_state == record["protected_checkpoints_at_start"]
    )
    record["status"] = "completed"
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    record["boundary"] = [
        "scores are only comparable inside this chain (relaxed guard + constrained decode)",
        "J3 would need C/E >= 0.70 and D >= 0.80; J2 needs all three mechanised dims above P3a",
        "treatment-minus-control is the data-distribution effect; either arm alone is not",
        "a negative result attributes the gap to the architecture layer (separate agenda)",
    ]
    _write(report_path, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3b campaign（训练 + 逐检查点评测 + 停止条件）")
    parser.add_argument("--arm", choices=("treatment", "control"), required=True)
    parser.add_argument("--budget-tier", choices=("16h", "48h", "custom"), default="48h")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1_000_000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    budget_args = ["--budget-tier", args.budget_tier]
    if args.budget_tier == "custom":
        if not args.max_symbols:
            parser.error("--max-symbols is required with --budget-tier custom")
        budget_args += ["--max-symbols", str(args.max_symbols)]
    elif args.max_symbols:
        parser.error("--max-symbols only applies with --budget-tier custom")

    record = run(
        arm=args.arm,
        budget_args=budget_args,
        checkpoint_every=args.checkpoint_every,
        poll_seconds=args.poll_seconds,
        device=args.device,
    )
    protected_ok = bool(record.get("protected_checkpoints_unchanged", False))
    print(
        json.dumps(
            {
                "event": "p3b_campaign_done",
                "arm": record["arm"],
                "campaign_stop": record["campaign_stop"],
                "stages": len(record["stages"]),
                "verdict": (record.get("criteria") or {}).get("verdict"),
                "protected_checkpoints_unchanged": protected_ok,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    if not protected_ok:
        print("FAIL: a protected checkpoint changed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

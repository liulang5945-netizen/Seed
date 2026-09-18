"""P3b campaign driver: train one arm, score every checkpoint, and honour the frozen stop rules.

One process owns one arm.  The trainer (``train_p3b_aligned.py --arm …``) is a child; each
checkpoint it writes is scored with the **same chain P3a was scored on** (relaxed legacy guard +
UTF-8 constrained decode) through ``eval_taiji_cap0_baseline.py``.  The campaign record is rewritten
atomically after every stage, so an interrupted 48-hour run loses nothing.

Two arms, one difference (decided 2026-09-15, ceiling option; novelty-matched per the amendment):
* ``--arm treatment`` trains on dialogue-dense rows (``build_p3b_arm_corpus.py --rule dialogue``);
* ``--arm control`` trains on **every** row of the *same* source window (``--rule all``),
  with the **same** start checkpoint, budget, objective and chain.

The 16M-tick state had already consumed the first 11,199,800 symbols of ``simple_zh_texts.jsonl``,
so an unslliced stream would spend ~23% of its budget on replay and the arms would differ in data
novelty as well as distribution.  Both corpora therefore start at the same unseen row.
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
import hashlib
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
#: The health sample taken from the *same* checkpoint as P3A_BASELINE (DEBT-I4).  Pairing is
#: checked by the criteria checker, so this name has to stay tied to that report.
#: The P3a reference health report **on the required chain** (v3/v4 were taken bare, so the
#: criteria checker's chain-parity guard refuses them; see ``judge_health``).
P3A_HEALTH = PROJECT_ROOT / "reports" / "taiji_cap0_health_v5_seedbeta_constrained_20260918.json"
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


def _file_state(path: Path) -> tuple[int, str]:
    """(size, sha256).

    mtime was not good enough: during this campaign a test re-saved the product default
    checkpoint, and size+mtime flagged it as damage while a same-size *content* change by a real
    writer would have looked equally alarming or equally innocent either way.  A hash answers the
    only question that matters -- did the bytes change.
    """

    if not path.exists():
        return (0, "")
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return (stat.st_size, digest.hexdigest())


def _latest_tick(checkpoint: Path) -> int:
    """Tick recorded in the checkpoint envelope; 0 while a write is still landing."""

    import torch

    try:
        envelope = torch.load(checkpoint, weights_only=False)
    except Exception:
        return 0
    return int((envelope.get("metadata") or {}).get("tick", 0))


def _snapshot(checkpoint: Path, tick: int) -> tuple[Path, int]:
    """Freeze the live checkpoint before scoring it, and **trust the copy over the caller**.

    One CAP-0 stage costs ~205 s and the trainer overwrites its checkpoint while the stage is
    running, so scoring the live file could mix two training states across dimensions.

    The caller learned ``tick`` from the live file a moment ago; the trainer may have atomic-saved
    again before these bytes were taken.  Naming the copy after the older tick would then score a
    later state and label it as an earlier one, which no later reading could detect -- so the
    envelope inside the copy decides, and a mismatch renames the file to the truth.
    """

    directory = checkpoint.parent / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / f"{checkpoint.stem}_tick_{tick}.pt"
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        temporary.write_bytes(checkpoint.read_bytes())
        temporary.replace(frozen)
    actual = _latest_tick(frozen)
    if actual <= 0:
        raise SystemExit(
            f"snapshot {frozen.name} carries no readable tick; refusing to score a torn copy"
        )
    if actual != tick:
        corrected = directory / f"{checkpoint.stem}_tick_{actual}.pt"
        frozen.replace(corrected)
        return corrected, actual
    return frozen, actual


def _evaluate(checkpoint: Path, stage_path: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    """Score one checkpoint on the frozen eval set with the P3a chain (fresh process per stage).

    A stage report already on disk is reused only after ``_reuse_defects`` clears it (DEBT-I6).
    Re-scoring is allowed only while the snapshot that stage was scored from still exists: the
    live checkpoint has moved past that tick, so a replacement produced from it would carry an
    older tick's label over a newer training state -- a mix no later reading could detect.
    """

    if stage_path.exists():
        try:
            stored = _load(stage_path)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            defects = [f"unreadable ({type(error).__name__})"]
        else:
            defects = _reuse_defects(stored, baseline)
        if not defects:
            return stored
        if not checkpoint.exists():
            raise SystemExit(
                f"stage report {stage_path.name} is unusable ({'; '.join(defects)}) and its "
                f"snapshot {checkpoint.name} is gone: refusing to re-score, because only the "
                "snapshot proves which training state this tick had"
            )
        preserved = stage_path.with_name(f"{stage_path.stem}.unusable")
        if preserved.exists():
            preserved = preserved.with_name(f"{preserved.name}.{os.getpid()}")
        stage_path.replace(preserved)
        print(
            f"re-scoring {stage_path.name} from {checkpoint.name}; "
            f"kept {preserved.name} ({'; '.join(defects)})",
            flush=True,
        )
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


def _evaluate_health(checkpoint: Path, health_path: Path) -> dict[str, Any]:
    """07 §4.2's A/H boolean clause for one stage: re-run the health probe in a fresh process.

    Runs on **the same chain as the scores** -- the two flags are not interchangeable here: A05b
    (does the emitted answer move under weight ablation) measures False bare and True constrained,
    because the readable-surface gate only rejects the raw bytes when they are undecodable.  The
    criteria checker refuses a health report whose ``chain`` is not the required one.
    Cheap next to a CAP-0 stage (measured 17.2 s before A05, ~25.6 s with it, against 345.7 s), and
    without it J4's A/H branch stays ``untested`` forever -- see DEBT-I4.  No reuse shortcut here:
    a health report is a property of the file it was taken from, and re-running costs seconds.
    """

    health_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        str(EVALUATOR),
        "--health",
        "--checkpoint",
        str(checkpoint),
        "--health-report",
        str(health_path),
        "--relax-legacy-guard",
        "--constrained-decode",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not health_path.exists():
        tail = (completed.stderr or completed.stdout or "")[-500:]
        raise SystemExit(f"stage health probe failed (rc={completed.returncode}): {tail}")
    health = _load(health_path)
    health["_health_seconds"] = round(time.perf_counter() - started, 1)
    return health


#: A stage is only comparable with the P3a baseline if the frozen evaluation surface is the
#: same surface -- not merely "the same script".  Verified equal on the plumbing smoke
#: (same manifest, format, frozen date, mode, and C/D/E item order; only ``checkpoint`` differs).
EVAL_SURFACE_FIELDS = ("eval_set", "eval_set_format", "eval_set_frozen_on", "declared_mode")


#: The stop rules as written into every campaign record.  A module constant, not an inline
#: literal, so a contract test can assert its **structure** instead of grepping words that also
#: appear in identifiers and docstrings.
STOP_DEFINITIONS: dict[str, Any] = {
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
    "stage_reuse": (
        "an existing stage report is re-read only after _reuse_defects clears it "
        "(evaluation surface, chain, trained_during_eval); a defective one is re-scored if its "
        "snapshot survives and the run refuses to continue if it does not"
    ),
    "health_per_stage": (
        "every stage also re-runs `--health` (measured 17.2 s before A05, plus 8.4 s for the A05 "
        "ablation arms now inside it = ~25.6 s, against a 345.7 s CAP-0 stage) so J4's A/H boolean "
        "clause is judged rather than left untested (DEBT-I4); the criteria call pairs each health "
        "report with its own evaluation report and refuses on mismatch"
    ),
    "comparability": (
        "a stage must reproduce the P3a evaluation surface exactly ("
        + ", ".join(EVAL_SURFACE_FIELDS)
        + ", plus C/D/E item id order); "
        "drift stops the run, because two numbers from different surfaces are not an effect"
    ),
    "chain_required": REQUIRED_CHAIN,
}


def _surface_drift(stage: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    drift = [field for field in EVAL_SURFACE_FIELDS if stage.get(field) != baseline.get(field)]
    for key in MECHANISED:
        scored = [
            item.get("id") for item in stage.get("dimensions", {}).get(key, {}).get("items", [])
        ]
        frozen = [
            item.get("id") for item in baseline.get("dimensions", {}).get(key, {}).get("items", [])
        ]
        if scored != frozen:
            drift.append(f"{key}:item_ids")
    return drift


def _reuse_defects(report: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Why an on-disk stage report may **not** be trusted (DEBT-I6).  Empty list means reusable.

    Item counts come from the baseline rather than a literal, so the check cannot rot into a fossil
    if the frozen eval set is ever re-issued -- ``_surface_drift`` already compares item ids.
    """

    defects = _surface_drift(report, baseline)
    if report.get("chain") != REQUIRED_CHAIN:
        defects.append(f"chain {report.get('chain')!r} is not the P3a chain {REQUIRED_CHAIN}")
    if report.get("trained_during_eval") is not False:
        defects.append(f"trained_during_eval is {report.get('trained_during_eval')!r}")
    return defects


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
    child = subprocess.Popen(command, cwd=str(PROJECT_ROOT))
    print(
        json.dumps(
            {
                "event": "p3b_campaign_launch",
                "arm": arm,
                "driver_pid": os.getpid(),
                "trainer_pid": child.pid,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
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
            "so a single arm cannot separate data distribution from those two effects; "
            "both arm corpora are sliced past the 11,199,800 symbols the start checkpoint had "
            "already seen, so the arms differ in row selection and not in data novelty either"
        ),
        "p3a_baseline": P3A_BASELINE.name,
        "baseline_scores": {key: _normalised(baseline, key) for key in MECHANISED},
        "stop_definitions": STOP_DEFINITIONS,
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
            requested = _latest_tick(checkpoint)
            if requested <= 0:
                time.sleep(poll_seconds)
                continue
            frozen, tick = _snapshot(checkpoint, requested)
            stage_path = stage_dir / f"cap0_tick_{tick}.json"
            scored = _evaluate(frozen, stage_path, baseline)
            health_path = stage_dir / f"health_tick_{tick}.json"
            health = _evaluate_health(frozen, health_path)
            row = _stage_row(tick, scored, baseline, stage_path.name)
            row["snapshot"] = frozen.name
            row["health_report"] = health_path.name
            row["health_checks"] = health["dimensions"]["A"]["checks"]
            row["health_seconds"] = health.get("_health_seconds")
            row["tick_corrected_from"] = None if tick == requested else requested
            row["eval_surface_drift"] = _surface_drift(scored, baseline)
            record["stages"].append(row)
            record["stall_streak"] = _stall_streak(record["stages"])
            print(
                json.dumps({"event": "p3b_stage", "arm": arm, **row}, ensure_ascii=True),
                flush=True,
            )
            kind = _regression_kind(record["stages"])
            row["regression_kind"] = kind
            if row["eval_surface_drift"]:
                campaign_stop = "eval_surface_drift"
            elif kind is not None or row["pending_grew"]:
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
                "--baseline-health",
                str(P3A_HEALTH),
                "--candidate-health",
                str(stage_dir / record["stages"][-1]["health_report"]),
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

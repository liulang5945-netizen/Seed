"""对N1-162B真实训练步做单/多线程吞吐基准，不写checkpoint。"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan162_question_surface_schedule_and_face as builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan155_short_unknown_context_length_training as n155
from scripts.training import run_taiji_n1_plan162b_question_surface_matched_continuation as n162
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150

PLAN = ROOT / "plans/reference/PLAN-N1-162G_training_thread_scaling_benchmark_20261020.md"
PLAN_SHA256 = "a6ff0f26ae3548c706e0e99c464a496d14dde298a75aa116943c9e5624c26ad9"
PRIOR_RESULT = ROOT / "reports/taiji_n1_plan162b_question_surface_result_20261020.json"
PRIOR_RESULT_SHA256 = "19dfaeb2086abaa503ee5150dba3bbfc9756d0892e8b6b54222fbfc4ecba2c31"
SCHEDULE_MANIFEST = ROOT / "plans/manifests/n1_plan162_v34_question_surface_schedule_20261020.json"
SCHEDULE_MANIFEST_SHA256 = "225b5d6e55df9cec3b62a977e2a34173a9dc39b9c54d32d5258c2451e9e99fd1"
SCHEDULE_ROWS_SHA256 = "bf6cc77a9f569b3d9dca151d455bfbe6ccf875715d17f2a123d94dfc2c88c84c"
CHECKPOINT = (
    ROOT / "data/checkpoints/n1_plan162b_question_surface_20261020/s11_surface_control_final.pt"
)
CHECKPOINT_SHA256 = "6976af645eff7213aa4d210d0919451c37e430bf3821b3c1e0c1360f17acd913"
CHECKPOINT_FINGERPRINT = "0df7ba460cf9022fdba4e20902df754b255f6e187c2fcc2dfc7b754c825e9890"
ATTEMPT = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_20261020.json"

THREAD_COUNTS = (1, 2, 4)
TRIAL_ORDER = ((1, 1), (2, 1), (4, 1), (4, 2), (2, 2), (1, 2))
WARMUP_UPDATES = 32
MEASURED_UPDATES = 256
SCHEDULE_ROWS = WARMUP_UPDATES + MEASURED_UPDATES
CPU_LIMIT_SECONDS = 900.0


class Refuse(RuntimeError):
    """冻结来源或吞吐基准合同不满足。"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"write-once PLAN-N1-162G output exists: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _check_budget(started_cpu: float) -> None:
    used = time.process_time() - started_cpu
    if used > CPU_LIMIT_SECONDS:
        raise TimeoutError(f"PLAN-N1-162G CPU budget exhausted: {used:.3f}/{CPU_LIMIT_SECONDS:.3f}")


def _verify_sources() -> tuple[dict, dict, dict, dict[str, str]]:
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-162G plan SHA changed")
    if sha256(PRIOR_RESULT) != PRIOR_RESULT_SHA256:
        raise Refuse("N1-162B result SHA changed")
    if sha256(SCHEDULE_MANIFEST) != SCHEDULE_MANIFEST_SHA256:
        raise Refuse("N1-162 schedule manifest SHA changed")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise Refuse("N1-162B benchmark checkpoint SHA changed")

    prior = read_json(PRIOR_RESULT)
    if (
        prior.get("complete") is not True
        or prior.get("training_complete") is not True
        or prior.get("training_updates") != 55_296
        or prior.get("source_hashes_unchanged") is not True
        or prior.get("can_promote") is not False
    ):
        raise Refuse("N1-162B training source result is incomplete or changed")
    matches = [
        row
        for row in prior.get("models", [])
        if row.get("seed") == 11 and row.get("arm") == "surface_control"
    ]
    if len(matches) != 1:
        raise Refuse("N1-162B seed11 surface_control endpoint is ambiguous")
    endpoint = matches[0]
    restore = endpoint.get("final_independent_restore", {})
    if (
        endpoint.get("final_checkpoint_path") != str(CHECKPOINT.resolve())
        or endpoint.get("final_checkpoint_sha256") != CHECKPOINT_SHA256
        or endpoint.get("final_bundle_fingerprint") != CHECKPOINT_FINGERPRINT
        or endpoint.get("final_history_count") != 79_872
        or endpoint.get("optimizer_step_values") != [61_440]
        or restore.get("complete") is not True
        or restore.get("equal") is not True
    ):
        raise Refuse("N1-162B benchmark endpoint restore receipt differs")

    runtime, source_hashes, _exclusions, _candidates = builder._source_context()
    base_schedule = runtime["n155_schedule"]
    schedule = builder.build_schedule(base_schedule)
    manifest = read_json(SCHEDULE_MANIFEST)
    builder.validate_schedule(manifest)
    if schedule["manifest"]["schedule_sha256"] != SCHEDULE_ROWS_SHA256:
        raise Refuse("N1-162 live reconstructed schedule digest differs")
    if manifest.get("schedule_sha256") != SCHEDULE_ROWS_SHA256:
        raise Refuse("N1-162 frozen schedule rows digest differs")
    if len(schedule["rows"]) != 9_216 or len(schedule["slots"]) != 9_216:
        raise Refuse("N1-162 frozen schedule row count differs")

    required_hashes = {
        str(PLAN.resolve()): PLAN_SHA256,
        str(PRIOR_RESULT.resolve()): PRIOR_RESULT_SHA256,
        str(SCHEDULE_MANIFEST.resolve()): SCHEDULE_MANIFEST_SHA256,
        str(CHECKPOINT.resolve()): CHECKPOINT_SHA256,
        str(Path(n162.__file__).resolve()): sha256(Path(n162.__file__).resolve()),
        str(Path(n132.__file__).resolve()): sha256(Path(n132.__file__).resolve()),
        str(Path(n155.__file__).resolve()): sha256(Path(n155.__file__).resolve()),
        str(Path(n150.__file__).resolve()): sha256(Path(n150.__file__).resolve()),
        str(Path(builder.__file__).resolve()): sha256(Path(builder.__file__).resolve()),
    }
    source_hashes.update(required_hashes)
    if not n150.verify_hashes(source_hashes):
        raise Refuse("PLAN-N1-162G source/checkpoint hashes changed during readiness")
    return runtime, schedule, endpoint, dict(sorted(source_hashes.items()))


def _prepare_step(slot: dict, row: dict, schedule: dict) -> tuple[dict, dict | None]:
    if row.get("control_template_id") != "original":
        raise Refuse("benchmark control schedule is not the frozen original question surface")
    item = n150._apply_context(
        slot["item"], slot["item_key"], int(row["context_length"]), schedule["views"]
    )
    second = None
    if slot.get("second") is not None:
        second = n150._with_observations(
            slot["second"], item["observations"], int(row["context_length"])
        )
    return item, second


def _optimizer_step_values(optimizer) -> set[int]:
    values = set()
    for state in optimizer.state.values():
        if "step" in state:
            values.add(int(state["step"]))
    return values


def _summarize_trials(trials: list[dict]) -> tuple[dict, int, int]:
    by_threads = {}
    for threads in THREAD_COUNTS:
        rows = [row for row in trials if int(row["threads"]) == threads]
        if len(rows) != 2:
            raise Refuse(f"thread count {threads} has {len(rows)} repeats, expected 2")
        rates = [float(row["updates_per_second"]) for row in rows]
        by_threads[str(threads)] = {
            "rates_updates_per_second": rates,
            "median_updates_per_second": round(statistics.median(rates), 6),
            "median_cpu_seconds_per_update": round(
                statistics.median(float(row["cpu_seconds_per_update"]) for row in rows), 6
            ),
        }
    baseline = float(by_threads["1"]["median_updates_per_second"])
    for threads in (2, 4):
        rates = by_threads[str(threads)]["rates_updates_per_second"]
        by_threads[str(threads)]["speedup_vs_single_thread"] = round(
            float(by_threads[str(threads)]["median_updates_per_second"]) / baseline, 4
        )
        by_threads[str(threads)]["stable_gain_10_percent"] = all(
            rate >= baseline * 1.10 for rate in rates
        )
    winner = max(
        THREAD_COUNTS, key=lambda value: float(by_threads[str(value)]["median_updates_per_second"])
    )
    stable_candidates = [
        threads for threads in (2, 4) if by_threads[str(threads)]["stable_gain_10_percent"]
    ]
    stable_winner = (
        max(
            stable_candidates,
            key=lambda value: float(by_threads[str(value)]["median_updates_per_second"]),
        )
        if stable_candidates
        else 1
    )
    return by_threads, winner, stable_winner


def _run_trial(
    threads: int, repeat: int, runtime: dict, schedule: dict, endpoint: dict, started_cpu: float
) -> dict:
    torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise Refuse(f"PyTorch did not apply requested thread count {threads}")
    model, optimizer, payload, fingerprint = n155._restore_checkpoint(CHECKPOINT)
    if (
        fingerprint != CHECKPOINT_FINGERPRINT
        or payload.get("bundle_fingerprint") != CHECKPOINT_FINGERPRINT
        or len(model._observations) != endpoint["final_history_count"]
        or _optimizer_step_values(optimizer) != {61_440}
    ):
        raise Refuse(f"independent benchmark restore differs: threads={threads} repeat={repeat}")

    random.seed(162_000 + threads * 10 + repeat)
    torch.manual_seed(162_000 + threads * 10 + repeat)
    start_history = len(model._observations)
    start_wall = time.perf_counter()
    start_process_cpu = time.process_time()
    slot_rows = schedule["rows"][:SCHEDULE_ROWS]
    for index, row in enumerate(slot_rows):
        _check_budget(started_cpu)
        slot = schedule["slots"][index]
        item, second = _prepare_step(slot, row, schedule)
        n162._training_step(
            model,
            optimizer,
            item,
            second,
            f"n1-plan162g-thread{threads}-repeat{repeat}-step{index}",
            runtime["n155_bound"]["g"],
        )
        if index + 1 == WARMUP_UPDATES:
            start_wall = time.perf_counter()
            start_process_cpu = time.process_time()
        if index + 1 == WARMUP_UPDATES + MEASURED_UPDATES:
            elapsed_wall = time.perf_counter() - start_wall
            elapsed_cpu = time.process_time() - start_process_cpu

    if len(model._observations) != start_history + SCHEDULE_ROWS:
        raise Refuse("ephemeral benchmark update history count differs")
    if _optimizer_step_values(optimizer) != {61_440 + SCHEDULE_ROWS}:
        raise Refuse("ephemeral benchmark optimizer update count differs")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise Refuse("benchmark modified the source checkpoint")
    _check_budget(started_cpu)

    row = {
        "trial": 0,
        "threads": threads,
        "repeat": repeat,
        "warmup_updates": WARMUP_UPDATES,
        "measured_updates": MEASURED_UPDATES,
        "wall_seconds": round(elapsed_wall, 6),
        "process_cpu_seconds": round(elapsed_cpu, 6),
        "effective_cpu_cores": round(elapsed_cpu / elapsed_wall, 4),
        "updates_per_second": round(MEASURED_UPDATES / elapsed_wall, 6),
        "cpu_seconds_per_update": round(elapsed_cpu / MEASURED_UPDATES, 6),
        "restore_fingerprint": fingerprint,
        "start_history_count": start_history,
        "end_history_count": len(model._observations),
        "end_optimizer_step": 61_440 + SCHEDULE_ROWS,
        "checkpoint_sha256_after": sha256(CHECKPOINT),
    }
    del model, optimizer, payload
    gc.collect()
    return row


def execute() -> dict:
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-162G outputs are write-once")
    started_cpu = time.process_time()
    write_new(
        ATTEMPT,
        {
            "schema": "taiji-n1-plan162g-thread-scaling-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "trial_order": [
                {"threads": threads, "repeat": repeat} for threads, repeat in TRIAL_ORDER
            ],
            "updates_per_trial": SCHEDULE_ROWS,
            "parameter_updates_in_memory": len(TRIAL_ORDER) * SCHEDULE_ROWS,
            "checkpoint_writes": 0,
            "can_promote": False,
        },
    )
    progress_path = PROGRESS.open("x", encoding="utf-8")
    trials: list[dict] = []
    source_hashes: dict[str, str] = {}
    errors: list[str] = []
    try:
        runtime, schedule, endpoint, source_hashes = _verify_sources()
        for trial_index, (threads, repeat) in enumerate(TRIAL_ORDER, start=1):
            row = _run_trial(threads, repeat, runtime, schedule, endpoint, started_cpu)
            row["trial"] = trial_index
            trials.append(row)
            progress_path.write(
                json.dumps({"event": "trial_complete", **row}, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            progress_path.flush()

        by_threads, winner, stable_winner = _summarize_trials(trials)
        if not n150.verify_hashes(source_hashes):
            raise Refuse("source hashes changed during PLAN-N1-162G benchmark")
        if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
            raise Refuse("benchmark source checkpoint SHA changed at completion")
        if len(trials) != len(TRIAL_ORDER):
            raise Refuse("PLAN-N1-162G did not complete all six trials")

        total_updates = len(trials) * SCHEDULE_ROWS
        total_cpu = time.process_time() - started_cpu
        _check_budget(started_cpu)
        report = {
            "schema": "taiji-n1-plan162g-training-thread-scaling-v1",
            "status": "complete",
            "complete": True,
            "plan_sha256": PLAN_SHA256,
            "prior_result_sha256": PRIOR_RESULT_SHA256,
            "schedule_manifest_sha256": SCHEDULE_MANIFEST_SHA256,
            "schedule_rows_sha256": SCHEDULE_ROWS_SHA256,
            "checkpoint_path": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_sha256_unchanged": True,
            "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
            "endpoint_restore_receipt_reused": endpoint["final_independent_restore"],
            "live_independent_restores": len(trials),
            "thread_counts": list(THREAD_COUNTS),
            "cpu_model": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "trial_order": [
                {"threads": threads, "repeat": repeat} for threads, repeat in TRIAL_ORDER
            ],
            "warmup_updates_per_trial": WARMUP_UPDATES,
            "measured_updates_per_trial": MEASURED_UPDATES,
            "parameter_updates_in_memory": total_updates,
            "checkpoint_writes": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "evaluation_queries": 0,
            "cpu_seconds": round(total_cpu, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "trials": trials,
            "summary_by_threads": by_threads,
            "best_median_thread_count": winner,
            "stable_recommendation_thread_count": stable_winner,
            "source_hashes": source_hashes,
            "source_hashes_unchanged": True,
            "errors": errors,
            "can_promote": False,
            "summary_zh": (
                f"单线程真实训练步CPU吞吐基准完成：线程1/2/4各2次，"
                f"最佳中位吞吐为{winner}线程，按冻结10%重复收益规则建议{stable_winner}线程；"
                f"CPU{total_cpu:.3f}/{CPU_LIMIT_SECONDS:.0f}秒，{total_updates}次更新仅发生于内存副本，0 checkpoint写入，can_promote=false。"
            ),
        }
        write_new(RESULT, report)
        return report
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        failure = {
            "schema": "taiji-n1-plan162g-training-thread-scaling-v1",
            "status": "failed",
            "complete": False,
            "plan_sha256": PLAN_SHA256,
            "cpu_seconds": round(time.process_time() - started_cpu, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "completed_trials": trials,
            "source_hashes": source_hashes,
            "source_hashes_unchanged": bool(source_hashes) and n150.verify_hashes(source_hashes),
            "errors": errors,
            "checkpoint_writes": 0,
            "can_promote": False,
        }
        if not RESULT.exists():
            write_new(RESULT, failure)
        raise
    finally:
        progress_path.close()


def main() -> int:
    torch.set_num_threads(1)
    try:
        result = execute()
    except Exception as exc:
        print(f"PLAN-N1-162G refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "cpu_seconds",
                    "parameter_updates_in_memory",
                    "summary_by_threads",
                    "best_median_thread_count",
                    "stable_recommendation_thread_count",
                    "summary_zh",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

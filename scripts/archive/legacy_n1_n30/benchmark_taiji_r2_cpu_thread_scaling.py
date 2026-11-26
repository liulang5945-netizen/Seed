"""Measure intra-op thread scaling on copies of one frozen R2 train-only state."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import LanguageAlignmentTrainer, LanguageEpisodeCorpus
from taiji.internalization import content_digest

PLAN = ROOT / "plans/reference/PLAN-R2-THREAD-SCALING-DIAGNOSTIC_20261021.md"
CHECKPOINT = (
    ROOT
    / "data/checkpoints/r2_daily_dialogue_warmstart_matched_20261021/warm_start/full_turn_zero.pt"
)
EPISODES = ROOT / "data/conversation/OASST2/taiji_r2_dialogue_full_turn_pretraining_episodes.jsonl"
MANIFEST = ROOT / "data/conversation/OASST2/taiji_r2_dialogue_full_turn_pretraining_manifest.json"
REPORT_ROOT = ROOT / "reports"
RESULT = ROOT / "reports/taiji_r2_daily_dialogue_cpu_thread_scaling_20261021.json"
ATTEMPT = ROOT / "reports/taiji_r2_daily_dialogue_cpu_thread_scaling_20261021_attempt.json"
PROGRESS = ROOT / "reports/taiji_r2_daily_dialogue_cpu_thread_scaling_20261021_progress.jsonl"
THREAD_ORDER = (1, 8, 2, 4, 4, 2, 8, 1)
TRIAL_EPISODES = 16
CPU_LIMIT_SECONDS = 900.0
TRIAL_CPU_RESERVE_SECONDS = 300.0


class Refuse(RuntimeError):
    """A frozen R2 thread-scaling benchmark precondition failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"write-once benchmark output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_progress(stream, value: dict[str, Any]) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()


def source_hashes() -> dict[str, str]:
    paths = (
        PLAN,
        Path(__file__).resolve(),
        CHECKPOINT,
        EPISODES,
        MANIFEST,
        ROOT / "taiji/language_alignment.py",
        ROOT / "taiji/model.py",
        ROOT / "taiji/response_plan_target.py",
    )
    return {str(path.resolve()): sha256(path) for path in paths}


def resource_sample() -> dict[str, Any]:
    cpu = psutil.cpu_percent(interval=1.0)
    memory_gib = psutil.virtual_memory().available / 1024**3
    if cpu > 85.0:
        raise Refuse(f"host CPU is above benchmark start ceiling: {cpu:.1f}%")
    if memory_gib < 8.0:
        raise Refuse(f"available memory is below 8 GiB: {memory_gib:.2f}")
    return {
        "logical_cpus": os.cpu_count(),
        "cpu_percent": cpu,
        "available_memory_gib": round(memory_gib, 3),
    }


def _trial_fits_cpu_budget(cpu_used: float) -> bool:
    """Reserve enough CPU for the slowest observed 16-episode trial."""
    return float(cpu_used) + TRIAL_CPU_RESERVE_SECONDS <= CPU_LIMIT_SECONDS


def execute() -> dict[str, Any]:
    for path in (PLAN, CHECKPOINT, EPISODES, MANIFEST):
        if not path.is_file():
            raise Refuse(f"required benchmark source is missing: {path}")
    if any(path.exists() for path in (RESULT, ATTEMPT, PROGRESS)):
        raise FileExistsError("R2 thread-scaling benchmark is write-once")

    resource_before = resource_sample()
    sources_before = source_hashes()
    parent_sha_before = sha256(CHECKPOINT)
    corpus = LanguageEpisodeCorpus.from_jsonl([EPISODES.resolve()], require_eval_splits=False)
    if corpus.sample_counts != {"train": 3762, "dev": 0, "final": 0, "retention": 0}:
        raise Refuse(f"thread-scaling corpus split differs: {corpus.sample_counts}")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    expected_digest = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    if (
        payload.get("checkpoint_digest") != expected_digest
        or payload.get("corpus_digest") != corpus.digest
    ):
        raise Refuse("warm-start zero checkpoint digest/corpus binding differs")
    if len(corpus.episodes) < TRIAL_EPISODES:
        raise Refuse("train-only corpus is too small for fixed thread probe")
    expected_updates = sum(
        len(episode.target_bytes) for episode in corpus.episodes[:TRIAL_EPISODES]
    )
    hashes = sources_before
    attempt = {
        "schema": "taiji-r2-thread-scaling-attempt-v1",
        "status": "running",
        "plan_sha256": sha256(PLAN),
        "source_hashes": hashes,
        "parent_checkpoint_sha256": parent_sha_before,
        "parent_checkpoint_digest": payload["checkpoint_digest"],
        "thread_order": list(THREAD_ORDER),
        "episodes_per_trial": TRIAL_EPISODES,
        "expected_target_byte_updates_per_trial": expected_updates,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "dev_queries": 0,
        "reference_queries": 0,
        "checkpoint_writes": 0,
        "can_promote": False,
    }
    write_new(ATTEMPT, attempt)
    started_cpu = time.process_time()
    trials: list[dict[str, Any]] = []
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    try:
        for index, thread_count in enumerate(THREAD_ORDER, 1):
            cpu_used = time.process_time() - started_cpu
            if not _trial_fits_cpu_budget(cpu_used):
                raise TimeoutError(
                    "insufficient reserved CPU for another complete trial: "
                    f"{cpu_used:.3f}+{TRIAL_CPU_RESERVE_SECONDS:.0f}/{CPU_LIMIT_SECONDS:.0f}"
                )
            torch.set_num_threads(thread_count)
            actual_threads = torch.get_num_threads()
            if actual_threads != thread_count:
                raise Refuse(
                    f"requested {thread_count} intra-op threads, runtime reports {actual_threads}"
                )
            restore_started_cpu = time.process_time()
            trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
            restore_cpu = time.process_time() - restore_started_cpu
            if trainer.checkpoint().get("checkpoint_digest") != payload["checkpoint_digest"]:
                raise Refuse(f"trial {index} did not restore the identical zero state")

            wall_started = time.perf_counter()
            cpu_started = time.process_time()
            trained = trainer.train(epochs=1, max_episodes=TRIAL_EPISODES)
            wall_seconds = time.perf_counter() - wall_started
            cpu_seconds = time.process_time() - cpu_started
            if (
                trained.get("status") != "completed"
                or trained.get("episodes") != TRIAL_EPISODES
                or trainer.global_step != expected_updates
                or trainer.episode_count != TRIAL_EPISODES
            ):
                raise Refuse(f"trial {index} did not execute the fixed train-only canary")
            row = {
                "trial": index,
                "requested_intraop_threads": thread_count,
                "actual_intraop_threads": actual_threads,
                "episodes": trainer.episode_count,
                "target_byte_updates": trainer.global_step,
                "restore_cpu_seconds": round(restore_cpu, 6),
                "training_cpu_seconds": round(cpu_seconds, 6),
                "training_wall_seconds": round(wall_seconds, 6),
                "updates_per_cpu_second": round(trainer.global_step / cpu_seconds, 3),
                "updates_per_wall_second": round(trainer.global_step / wall_seconds, 3),
            }
            trials.append(row)
            write_progress(progress, {"event": "trial_complete", **row})
            del trainer

        if time.process_time() - started_cpu > CPU_LIMIT_SECONDS:
            raise TimeoutError(f"thread benchmark exceeded {CPU_LIMIT_SECONDS:.0f} CPU seconds")
        if sha256(CHECKPOINT) != parent_sha_before or source_hashes() != sources_before:
            raise Refuse("frozen benchmark source or parent checkpoint changed")
        resource_after = {
            "cpu_percent": psutil.cpu_percent(interval=1.0),
            "available_memory_gib": round(psutil.virtual_memory().available / 1024**3, 3),
        }
        by_threads: dict[str, Any] = {}
        for count in (1, 2, 4, 8):
            rows = [row for row in trials if row["actual_intraop_threads"] == count]
            by_threads[str(count)] = {
                "repeat_count": len(rows),
                "median_updates_per_cpu_second": round(
                    statistics.median(row["updates_per_cpu_second"] for row in rows), 3
                ),
                "median_updates_per_wall_second": round(
                    statistics.median(row["updates_per_wall_second"] for row in rows), 3
                ),
                "median_cpu_seconds": round(
                    statistics.median(row["training_cpu_seconds"] for row in rows), 6
                ),
                "median_wall_seconds": round(
                    statistics.median(row["training_wall_seconds"] for row in rows), 6
                ),
            }
        result = {
            "schema": "taiji-r2-thread-scaling-result-v1",
            "status": "complete",
            "complete": True,
            "source_hashes": hashes,
            "parent_checkpoint_sha256_before": parent_sha_before,
            "parent_checkpoint_sha256_after": sha256(CHECKPOINT),
            "resource_before": resource_before,
            "episodes_per_trial": TRIAL_EPISODES,
            "expected_target_byte_updates_per_trial": expected_updates,
            "thread_order": list(THREAD_ORDER),
            "trials": trials,
            "summary_by_threads": by_threads,
            "resource_after": resource_after,
            "total_cpu_seconds": round(time.process_time() - started_cpu, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "model_queries": 0,
            "parameter_updates": len(trials) * expected_updates,
            "formal_model_updates": 0,
            "checkpoint_writes": 0,
            "dev_queries": 0,
            "reference_scoring_queries": 0,
            "can_promote": False,
        }
        write_new(RESULT, result)
        attempt.update(
            {
                "status": "complete",
                "result_sha256": sha256(RESULT),
                "total_cpu_seconds": result["total_cpu_seconds"],
            }
        )
        write_replace(ATTEMPT, attempt)
        return result
    except BaseException as exc:
        attempt.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "total_cpu_seconds": round(time.process_time() - started_cpu, 6),
            }
        )
        write_replace(ATTEMPT, attempt)
        raise
    finally:
        progress.flush()
        progress.close()


def main() -> int:
    try:
        result = execute()
    except Exception as exc:
        print(f"R2 thread-scaling diagnostic stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "trial_count": len(result["trials"]),
                "summary_by_threads": result["summary_by_threads"],
                "total_cpu_seconds": result["total_cpu_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

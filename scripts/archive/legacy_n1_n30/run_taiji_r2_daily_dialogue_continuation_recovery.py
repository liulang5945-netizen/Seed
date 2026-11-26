"""Recover the failed writeout of the completed R2 dialogue continuation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_r2_daily_dialogue_continuation as continuation
from taiji.language_alignment import LanguageAlignmentTrainer

RECOVERY_PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-CONTINUATION-RECOVERY1_20261021.md"
RECOVERY_PLAN_SHA256 = "217486709ad907f8d0082238afd73b0b8f8a59df779aae45d035197c1288a9cf"
PARTIAL_RESULT = ROOT / "reports/taiji_r2_daily_dialogue_20261021_continuation_result.json"
PARTIAL_RESULT_SHA256 = "2a2e2e7a9c7d4993d4fdf56a1b6d5ad6b9e0f3a94b75f846f2b0c0f1f11086ea"
PARTIAL_ATTEMPT = ROOT / "reports/taiji_r2_daily_dialogue_20261021_continuation_attempt.json"
PARTIAL_ATTEMPT_SHA256 = "821bf3853dd89b2066fbe72eff777290f216d0726b48bacd18d7fa806b67d116"
REPORT_ROOT = ROOT / "reports"
READINESS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_recovery1_readiness.json"
ATTEMPT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_recovery1_attempt.json"
PROGRESS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_recovery1_progress.jsonl"
GENERATIONS = (
    REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_recovery1_generations.json"
)
REVIEW = (
    REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_recovery1_generation_review.md"
)
RESULT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_recovery1_result.json"

FIRST_EPISODE_ID = "oasst2:25d50180-7062-4e4a-b1b6-69ea5b23668d"
FIRST_FAMILY_ID = "oasst2-tree:18c35ba0-0db4-4bf9-b129-f8988cd888fd"
FAILED_ATTEMPT_GENERATION_QUERIES = 6
PREVIOUS_CPU_SECONDS = 5187.078125
ENDPOINTS = (
    "parent_with_history",
    "parent_single_turn",
    "old_dialogue_context",
    "old_single_turn",
    "continued_dialogue_context",
    "continued_single_turn",
)

CONTINUATION_CHECKPOINTS = {
    "continued_dialogue_context": {
        "path": ROOT
        / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_continuation/dialogue_context_final.pt",
        "sha256": "41655eb8b5fcd7e446acb924e94e458ce9b1e966a96a95517198a5ac38df2086",
        "arm": "dialogue_context",
    },
    "continued_single_turn": {
        "path": ROOT
        / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_continuation/single_turn_final.pt",
        "sha256": "5b17b2b6e5992012bfb2e24d1593eb77abac72cf1cd0dad375640c25fdeb50d4",
        "arm": "single_turn",
    },
}


class Refuse(RuntimeError):
    """A prior artifact or recovery contract does not match."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"recovery report is write-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _write_jsonl(stream, payload: dict[str, Any]) -> None:
    json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()


def _checkpoint_specs() -> dict[str, dict[str, Any]]:
    specs = {
        "parent_with_history": {
            "path": continuation.RETRY2_CHECKPOINTS["parent_dialogue_context"]["path"],
            "sha256": continuation.RETRY2_CHECKPOINTS["parent_dialogue_context"]["sha256"],
            "arm": "dialogue_context",
            "uses_history": True,
        },
        "parent_single_turn": {
            "path": continuation.RETRY2_CHECKPOINTS["parent_single_turn"]["path"],
            "sha256": continuation.RETRY2_CHECKPOINTS["parent_single_turn"]["sha256"],
            "arm": "single_turn",
            "uses_history": False,
        },
        "old_dialogue_context": {
            "path": continuation.RETRY2_CHECKPOINTS["old_dialogue_context"]["path"],
            "sha256": continuation.RETRY2_CHECKPOINTS["old_dialogue_context"]["sha256"],
            "arm": "dialogue_context",
            "uses_history": True,
        },
        "old_single_turn": {
            "path": continuation.RETRY2_CHECKPOINTS["old_single_turn"]["path"],
            "sha256": continuation.RETRY2_CHECKPOINTS["old_single_turn"]["sha256"],
            "arm": "single_turn",
            "uses_history": False,
        },
    }
    specs.update(
        {
            name: {**spec, "uses_history": name == "continued_dialogue_context"}
            for name, spec in CONTINUATION_CHECKPOINTS.items()
        }
    )
    return specs


def _validate_partial_result(partial: dict[str, Any]) -> None:
    if (
        partial.get("status") != "partial"
        or partial.get("complete") is not False
        or partial.get("plan_sha256") != continuation.PLAN_SHA256
        or partial.get("completed_training_episodes")
        != {"dialogue_context": 3150, "single_turn": 3150}
        or partial.get("completed_response_observations")
        != {"dialogue_context": 679254, "single_turn": 679254}
        or partial.get("completed_generation_outputs") != FAILED_ATTEMPT_GENERATION_QUERIES
        or partial.get("errors") != ["KeyError: 'parent_dialogue_context'"]
        or partial.get("can_promote") is not False
    ):
        raise Refuse("the partial continuation result differs from the recovered failure")


def execute() -> dict[str, Any]:
    started = time.process_time()
    budget = continuation.CpuBudget(continuation.CPU_LIMIT_SECONDS - PREVIOUS_CPU_SECONDS)
    outputs = (READINESS, ATTEMPT, PROGRESS, GENERATIONS, REVIEW, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("continuation recovery outputs are write-once")
    if _sha256(RECOVERY_PLAN) != RECOVERY_PLAN_SHA256:
        raise Refuse("recovery plan SHA-256 changed")
    if _sha256(PARTIAL_RESULT) != PARTIAL_RESULT_SHA256:
        raise Refuse("partial continuation result SHA-256 changed")
    if _sha256(PARTIAL_ATTEMPT) != PARTIAL_ATTEMPT_SHA256:
        raise Refuse("partial continuation attempt SHA-256 changed")

    partial = continuation._read_json(PARTIAL_RESULT)
    _validate_partial_result(partial)
    base, _manifest, _prior_result, source_hashes, used_families = continuation._load_inputs()
    selected = continuation._select_remaining_final(base, used_families)
    if (
        len(selected) != 70
        or selected[0].episode_id != FIRST_EPISODE_ID
        or selected[0].family_id != FIRST_FAMILY_ID
    ):
        raise Refuse("the continuation final sample order differs from the failed attempt")

    checkpoint_specs = _checkpoint_specs()
    for name, spec in checkpoint_specs.items():
        if _sha256(spec["path"]) != spec["sha256"]:
            raise Refuse(f"recovery checkpoint changed: {name}")

    runner_path = Path(continuation.__file__).resolve()
    test_path = ROOT / "tests/taiji_native/test_r2_daily_dialogue_continuation_contract.py"
    recovery_runner_path = Path(__file__).resolve()
    source_hashes[str(PARTIAL_RESULT.resolve())] = PARTIAL_RESULT_SHA256
    source_hashes[str(PARTIAL_ATTEMPT.resolve())] = PARTIAL_ATTEMPT_SHA256
    source_hashes[str(RECOVERY_PLAN.resolve())] = RECOVERY_PLAN_SHA256
    source_hashes[str(runner_path)] = _sha256(runner_path)
    source_hashes[str(test_path.resolve())] = _sha256(test_path)
    source_hashes[str(recovery_runner_path)] = _sha256(recovery_runner_path)
    source_hashes.update(
        {str(spec["path"].resolve()): spec["sha256"] for spec in checkpoint_specs.values()}
    )

    if budget.limit <= 0:
        raise TimeoutError("the frozen continuation CPU budget has no recovery capacity")
    budget.guard(reserve=30 * len(ENDPOINTS))
    readiness = {
        "schema": "taiji-r2-daily-dialogue-continuation-recovery-readiness-v1",
        "status": "ready",
        "original_plan_sha256": continuation.PLAN_SHA256,
        "recovery_plan_sha256": RECOVERY_PLAN_SHA256,
        "partial_result_sha256": PARTIAL_RESULT_SHA256,
        "partial_attempt_sha256": PARTIAL_ATTEMPT_SHA256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "corpus_digest": base.digest,
        "split_episode_counts": base.sample_counts,
        "checkpoint_specs": {
            name: {
                "path": str(spec["path"].resolve()),
                "sha256": spec["sha256"],
                "arm": spec["arm"],
            }
            for name, spec in checkpoint_specs.items()
        },
        "unique_family_count": len(selected),
        "recovery_generation_outputs": len(selected) * len(ENDPOINTS),
        "previous_unpersisted_generation_outputs": FAILED_ATTEMPT_GENERATION_QUERIES,
        "cumulative_cpu_before_recovery_seconds": PREVIOUS_CPU_SECONDS,
        "cpu_limit_seconds": continuation.CPU_LIMIT_SECONDS,
        "remaining_cpu_limit_seconds": budget.limit - budget.used(),
        "parameter_updates": 0,
        "dev_queries": 0,
        "reference_answer_scoring_queries": 0,
        "can_promote": False,
    }
    _write_new_json(READINESS, readiness)
    attempt = {
        "schema": "taiji-r2-daily-dialogue-continuation-recovery-attempt-v1",
        "status": "started",
        "readiness_sha256": _sha256(READINESS),
        "source_hashes": dict(sorted(source_hashes.items())),
        "original_training_episodes": partial["completed_training_episodes"],
        "original_response_observations": partial["completed_response_observations"],
        "original_unpersisted_queries": FAILED_ATTEMPT_GENERATION_QUERIES,
        "replayed_episode_id": FIRST_EPISODE_ID,
        "replayed_family_id": FIRST_FAMILY_ID,
        "persisted_generation_outputs_planned": len(selected) * len(ENDPOINTS),
        "physical_generation_calls_planned": len(selected) * len(ENDPOINTS)
        + FAILED_ATTEMPT_GENERATION_QUERIES,
        "parameter_updates": 0,
        "can_promote": False,
    }
    _write_new_json(ATTEMPT, attempt)

    restore_receipts: dict[str, dict[str, Any]] = {}
    trainers: dict[str, LanguageAlignmentTrainer] = {}
    try:
        for endpoint, spec in checkpoint_specs.items():
            budget.guard(reserve=30)
            restore_receipts[endpoint] = continuation._independent_restore(
                spec["path"], spec["arm"], budget
            )
            payload = torch.load(spec["path"], map_location="cpu", weights_only=True)
            trainer = LanguageAlignmentTrainer.from_checkpoint(
                payload, continuation._arm_corpus(base, spec["arm"])
            )
            if (
                trainer.checkpoint()["checkpoint_digest"]
                != restore_receipts[endpoint]["checkpoint_digest"]
            ):
                raise Refuse(f"fresh restore and local load differ: {endpoint}")
            trainers[endpoint] = trainer

        if len(restore_receipts) != len(ENDPOINTS):
            raise Refuse("one or more endpoint checkpoint restores are absent")
        if any(
            receipt.get("complete") is not True or receipt.get("equal") is not True
            for receipt in restore_receipts.values()
        ):
            raise Refuse("a checkpoint failed its independent full-bundle restore")

        before_digests = {
            endpoint: trainer.checkpoint()["checkpoint_digest"]
            for endpoint, trainer in trainers.items()
        }
        progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
        generation_rows: list[dict[str, Any]] = []
        generation_count = 0
        try:
            for index, episode in enumerate(selected, 1):
                row: dict[str, Any] = {
                    "index": index,
                    "episode_id": episode.episode_id,
                    "family_id": episode.family_id,
                    "history_turn_count": len(episode.history),
                    "history": [[user, assistant] for user, assistant in episode.history],
                    "current_user_turn": episode.user_input,
                }
                flat = replace(episode, history=())
                for endpoint in ENDPOINTS:
                    budget.guard()
                    generation_episode = (
                        episode if checkpoint_specs[endpoint]["uses_history"] else flat
                    )
                    output = continuation._generate_readonly(trainers[endpoint], generation_episode)
                    row[endpoint] = output
                    generation_count += 1
                    _write_jsonl(
                        progress,
                        {
                            "event": "generation_output_persisted",
                            "index": index,
                            "episode_id": episode.episode_id,
                            "family_id": episode.family_id,
                            "endpoint": endpoint,
                            "output": output,
                            "reference_answer_included": False,
                            "exact_response_scored": False,
                        },
                    )
                    budget.guard()

                row["reference_answer_included"] = False
                row["exact_response_scored"] = False
                generation_rows.append(row)
                _write_jsonl(
                    progress,
                    {
                        "event": "generation_family_complete",
                        "index": index,
                        "episode_id": episode.episode_id,
                        "family_id": episode.family_id,
                        "persisted_endpoint_count": len(ENDPOINTS),
                        "reference_answer_included": False,
                    },
                )
                if index % 10 == 0:
                    print(
                        f"r2-dialogue-recovery generation={index}/{len(selected)} "
                        f"outputs={generation_count} cpu={budget.used():.1f}",
                        flush=True,
                    )
        finally:
            progress.flush()
            progress.close()

        after_digests = {
            endpoint: trainer.checkpoint()["checkpoint_digest"]
            for endpoint, trainer in trainers.items()
        }
        if before_digests != after_digests:
            raise Refuse("free generation mutated one or more checkpoint bundles")
        for path, expected in source_hashes.items():
            if not Path(path).is_file() or _sha256(Path(path)) != expected:
                raise Refuse(f"recovery source changed during generation: {path}")
        if generation_count != len(selected) * len(ENDPOINTS):
            raise Refuse("persisted output count differs from the frozen 70-by-6 face")

        all_outputs = {
            endpoint: [
                {"episode_id": row["episode_id"], "output": row[endpoint]}
                for row in generation_rows
            ]
            for endpoint in ENDPOINTS
        }
        generation_report = {
            "schema": "taiji-r2-daily-dialogue-continuation-recovery-generations-v1",
            "status": "complete",
            "original_plan_sha256": continuation.PLAN_SHA256,
            "recovery_plan_sha256": RECOVERY_PLAN_SHA256,
            "partial_result_sha256": PARTIAL_RESULT_SHA256,
            "corpus_digest": base.digest,
            "previously_consumed_tree_families": len(used_families),
            "new_final_tree_families": len({row["family_id"] for row in generation_rows}),
            "new_final_episodes": len(generation_rows),
            "input_history_strata": dict(
                Counter(
                    "with_history" if row["history_turn_count"] else "without_history"
                    for row in generation_rows
                )
            ),
            "generation_configuration": {
                "max_generation_bytes": 512,
                "constrained_decode": True,
                "sampling": False,
                "reference_answer_included": False,
                "exact_response_scored": False,
            },
            "recovery_generation_outputs": generation_count,
            "previous_unpersisted_outputs": FAILED_ATTEMPT_GENERATION_QUERIES,
            "replayed_family_id": FIRST_FAMILY_ID,
            "physical_generation_calls_including_lost_outputs": generation_count
            + FAILED_ATTEMPT_GENERATION_QUERIES,
            "metrics": {
                endpoint: continuation._summary([row[endpoint] for row in generation_rows])
                for endpoint in ENDPOINTS
            },
            "checkpoint_fresh_process_restores": restore_receipts,
            "records": generation_rows,
            "can_promote": False,
        }
        _write_new_json(GENERATIONS, generation_report)
        with REVIEW.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(continuation._continuation_review_markdown(generation_rows))
            stream.flush()

        total_cpu = PREVIOUS_CPU_SECONDS + budget.used() + (budget.started - started)
        if total_cpu > continuation.CPU_LIMIT_SECONDS:
            raise TimeoutError("cumulative continuation CPU exceeded its frozen limit")
        result = {
            "schema": "taiji-r2-daily-dialogue-continuation-recovery-result-v1",
            "status": "complete",
            "complete": True,
            "original_plan_sha256": continuation.PLAN_SHA256,
            "recovery_plan_sha256": RECOVERY_PLAN_SHA256,
            "partial_result_sha256": PARTIAL_RESULT_SHA256,
            "partial_attempt_sha256": PARTIAL_ATTEMPT_SHA256,
            "source_hashes": dict(sorted(source_hashes.items())),
            "original_training_episodes": partial["completed_training_episodes"],
            "original_response_observations": partial["completed_response_observations"],
            "original_training_final_checkpoints": {
                name: {
                    "path": str(spec["path"].resolve()),
                    "sha256": spec["sha256"],
                    "checkpoint_digest": restore_receipts[name]["checkpoint_digest"],
                    "global_step": restore_receipts[name]["global_step"],
                    "episode_count": restore_receipts[name]["episode_count"],
                }
                for name, spec in checkpoint_specs.items()
            },
            "fresh_process_restores": restore_receipts,
            "previous_unpersisted_generation_outputs": FAILED_ATTEMPT_GENERATION_QUERIES,
            "persisted_generation_outputs": generation_count,
            "physical_generation_calls_including_lost_outputs": generation_count
            + FAILED_ATTEMPT_GENERATION_QUERIES,
            "unique_final_tree_families": len({row["family_id"] for row in generation_rows}),
            "generation_report_sha256": _sha256(GENERATIONS),
            "generation_review_sha256": _sha256(REVIEW),
            "progress_sha256": _sha256(PROGRESS),
            "cpu_seconds_including_partial_attempt": round(total_cpu, 6),
            "cpu_limit_seconds": continuation.CPU_LIMIT_SECONDS,
            "parameter_updates": 0,
            "dev_queries": 0,
            "reference_answer_scoring_queries": 0,
            "can_promote": False,
        }
        _write_new_json(RESULT, result)
        return result
    except Exception as exc:
        failure = {
            "schema": "taiji-r2-daily-dialogue-continuation-recovery-result-v1",
            "status": "partial",
            "complete": False,
            "original_plan_sha256": continuation.PLAN_SHA256,
            "recovery_plan_sha256": RECOVERY_PLAN_SHA256,
            "partial_result_sha256": PARTIAL_RESULT_SHA256,
            "source_hashes": dict(sorted(source_hashes.items())),
            "completed_generation_outputs": locals().get("generation_count", 0),
            "completed_generation_families": len(locals().get("generation_rows", [])),
            "checkpoint_fresh_process_restores": restore_receipts,
            "cpu_seconds_including_partial_attempt": round(
                PREVIOUS_CPU_SECONDS + budget.used() + (budget.started - started), 6
            ),
            "cpu_limit_seconds": continuation.CPU_LIMIT_SECONDS,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "can_promote": False,
        }
        try:
            if not RESULT.exists():
                _write_new_json(RESULT, failure)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    torch.set_num_threads(1)
    try:
        report = execute()
    except Exception as exc:
        print(f"R2 continuation recovery refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print(
        json.dumps(
            {"status": report["status"], "result": str(RESULT.resolve())}, ensure_ascii=False
        )
    )

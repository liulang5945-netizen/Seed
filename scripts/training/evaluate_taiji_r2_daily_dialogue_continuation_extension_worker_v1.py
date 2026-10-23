"""Versioned extension evaluation worker (CC-1.2).

Same Job Object / IPC / restore / generate protocol as the frozen recovery-v2
worker, but every restore re-runs the extension provider so the parent and the
worker share one endpoint snapshot.  The frozen evaluator is imported as a
library; its module globals are never patched.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as recovery  # noqa: E402
from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter  # noqa: E402

MAX_WORKER_IPC_FRAME_BYTES = recovery.MAX_WORKER_IPC_FRAME_BYTES
PROCESS_START_CPU = recovery.PROCESS_START_CPU
PROCESS_START_WALL = recovery.PROCESS_START_WALL
ENDPOINT_ORDER = recovery.ENDPOINT_ORDER
Refuse = recovery.Refuse


def _handle_task(
    request: Mapping[str, Any],
    *,
    attempt_path: Path,
    ledger_path: Path,
    durability_api: Any,
    trainers: dict[str, Any],
    endpoint_specs: dict[str, dict[str, Any]],
    corpus_holder: dict[str, Any],
    expected_attempt_path: Path | None = None,
    expected_ledger_path: Path | None = None,
    job_membership_check: Callable[[], bool] | None = None,
    job_name: str | None = None,
    authorize_fn: Callable[..., Mapping[str, Any]] | None = None,
    preflight_fn: Callable[[], Mapping[str, Any]] | None = None,
    evaluation_paths_fn: Callable[[], Mapping[str, Path]] | None = None,
) -> dict[str, Any]:
    operation = request.get("operation")
    endpoint_id = request.get("endpoint_id")
    task_id = request.get("task_id")
    token = request.get("token")
    if not all(isinstance(value, str) and value for value in (operation, endpoint_id, task_id)):
        raise Refuse("worker IPC task header is invalid")
    membership_check = job_membership_check or (
        lambda: recovery._current_process_is_in_job(job_name or "")
    )
    try:
        in_job = membership_check()
    except BaseException as exc:
        raise Refuse(f"worker Job Object membership could not be verified: {exc}") from exc
    if not in_job:
        raise Refuse("worker task is refused outside the parent-supervised Job Object")
    authorize = authorize_fn or recovery._authorize_worker_task
    authorization = authorize(
        operation=operation,
        endpoint_id=endpoint_id,
        task_id=task_id,
        token=token if isinstance(token, str) else None,
        attempt_path=attempt_path,
        ledger_path=ledger_path,
        api=durability_api,
        expected_attempt_path=expected_attempt_path,
        expected_ledger_path=expected_ledger_path,
    )
    if request.get("attempt_id") != authorization["attempt"]["attempt_id"]:
        raise Refuse("worker IPC attempt identity differs from the durable parent attempt")
    task_cpu_start = time.process_time()
    task_wall_start = time.monotonic()
    if operation == "restore":
        #: The only provider-dependent part of the frozen handler: both the
        #: readiness and the endpoint map come from the extension provider.
        preflight = (preflight_fn or adapter.extension_preflight)()
        evaluation_paths = (evaluation_paths_fn or adapter.extension_evaluation_paths)()
        readiness = recovery.read_json(evaluation_paths["readiness"])
        attempt = authorization["attempt"]
        if (
            recovery._readiness_payload(preflight) != readiness
            or attempt.get("readiness") != readiness
            or attempt.get("source_hashes") != preflight.get("source_hashes")
            or attempt.get("input_report_hashes") != preflight.get("report_hashes")
        ):
            raise Refuse("worker restore metadata differs from the parent readiness")
        endpoint_specs.clear()
        endpoint_specs.update({item["endpoint_id"]: dict(item) for item in preflight["endpoints"]})
        if corpus_holder.get("assistant") is None:
            corpus_holder["assistant"] = recovery.base._load_assistant_train_corpus()
        endpoint = endpoint_specs.get(endpoint_id)
        if endpoint is None:
            raise Refuse("restore endpoint is not in the fixed four-endpoint map")
        if authorization["intent"].get("checkpoint_sha256") != endpoint["sha256"]:
            raise Refuse("restore endpoint SHA differs from the durable parent intent")
        trainer, model_digest = recovery._load_endpoint_trainer(
            endpoint, corpus_holder["assistant"]
        )
        expected_episodes = (
            0
            if endpoint_id == "parent_baseline"
            else recovery.EXPECTED_STAGE_COUNTS["assistant_sft"][0]
        )
        expected_updates = (
            0
            if endpoint_id == "parent_baseline"
            else recovery.EXPECTED_STAGE_COUNTS["assistant_sft"][1]
        )
        if trainer.episode_count != expected_episodes or trainer.global_step != expected_updates:
            raise Refuse(f"{endpoint_id} restore endpoint training count differs")
        if recovery.sha256_file(Path(endpoint["path"])) != endpoint["sha256"]:
            raise Refuse(f"{endpoint_id} checkpoint bytes changed during restore")
        trainers[endpoint_id] = trainer
        receipt = {
            "complete": True,
            "equal": True,
            "endpoint_id": endpoint_id,
            "stage": "parent_baseline" if endpoint_id == "parent_baseline" else "assistant_sft",
            "path": str(Path(endpoint["path"]).resolve()),
            "sha256": endpoint["sha256"],
            "checkpoint_digest": endpoint["checkpoint_digest"],
            "model_digest": model_digest,
            "corpus_digest": corpus_holder["assistant"].digest,
            "episode_count": expected_episodes,
            "global_step": expected_updates,
            "active_parameters": trainer.model.parameter_count(),
            "generation_queries": 0,
            "parameter_updates": 0,
            "optimizer_updates": 0,
            "can_promote": False,
        }
        receipt["cpu_seconds"] = max(0.0, time.process_time() - task_cpu_start)
        return {"status": "ok", "operation": operation, "receipt": receipt}

    if operation != "generate":
        raise Refuse("worker operation is not supported")
    if tuple(trainers) != ENDPOINT_ORDER:
        raise Refuse("worker generation cannot begin before all four restores")
    if endpoint_id not in trainers or endpoint_id not in endpoint_specs:
        raise Refuse("worker generation endpoint has no restored model")
    prompt_row = recovery._read_worker_prompt_row(
        request,
        attempt_id=authorization["attempt"]["attempt_id"],
        expected_prompt_sha256=str(authorization["intent"].get("prompt_sha256", "")),
        ledger_path=ledger_path,
    )
    if request.get("index") != authorization["intent"].get("index"):
        raise Refuse("worker prompt SHA differs from the durable query intent")
    endpoint = endpoint_specs[endpoint_id]
    checkpoint_path = Path(endpoint["path"])
    checkpoint_before = recovery.sha256_file(checkpoint_path)
    if checkpoint_before != endpoint["sha256"]:
        raise Refuse(f"{endpoint_id} checkpoint changed before generation")
    trainer = trainers[endpoint_id]
    model_before = recovery.content_digest(trainer.model.checkpoint())
    prompt = recovery.base.PromptOnlyEpisode(
        episode_id=str(prompt_row["episode_id"]),
        family_id=str(prompt_row["family_id"]),
        system=str(prompt_row["system"]),
        context=str(prompt_row["context"]),
        history=tuple((str(user), str(assistant)) for user, assistant in prompt_row["history"]),
        user_input=str(prompt_row["current_user_turn"]),
        unknown_policy=str(prompt_row["unknown_policy"]),
    )
    output = recovery._validate_generation_output(recovery.base._generate_readonly(trainer, prompt))
    model_after = recovery.content_digest(trainer.model.checkpoint())
    checkpoint_after = recovery.sha256_file(checkpoint_path)
    if model_after != model_before or checkpoint_after != checkpoint_before:
        raise Refuse(f"{endpoint_id} model/checkpoint changed during read-only generation")
    return {
        "status": "ok",
        "operation": operation,
        "output": output,
        "model_digest_before": model_before,
        "model_digest_after": model_after,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "worker_cpu_seconds": time.process_time() - task_cpu_start,
        "worker_wall_seconds": time.monotonic() - task_wall_start,
        "parameter_updates": 0,
        "optimizer_updates": 0,
        "can_promote": False,
    }


def worker_loop(attempt_path: Path, ledger_path: Path, job_name: str) -> int:
    """Private persistent worker; one IPC task at a time, no selector."""
    if os.name != "nt":
        raise Refuse("DURABILITY_UNSUPPORTED: restore/generation worker requires Windows NTFS")
    torch.set_num_threads(1)
    durability_api = recovery.WindowsDurabilityApi()
    if not attempt_path.exists() or not ledger_path.exists():
        raise Refuse("worker requires an existing parent attempt and ledger")
    attempt = recovery.read_json(attempt_path)
    if not recovery._current_process_is_in_job(job_name):
        raise Refuse("worker bootstrap is refused outside the parent-supervised Job Object")
    trainers: dict[str, Any] = {}
    endpoint_specs: dict[str, dict[str, Any]] = {}
    corpus_holder: dict[str, Any] = {"assistant": None}
    print(
        json.dumps(
            {
                "status": "ready",
                "operation": "ready",
                "task_id": "worker_ready",
                "endpoint_id": "worker",
                "attempt_id": attempt.get("attempt_id"),
                "job_name": job_name,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    while True:
        line = sys.stdin.readline(MAX_WORKER_IPC_FRAME_BYTES + 1)
        if not line:
            return 0
        request: dict[str, Any] | None = None
        try:
            if not line.endswith("\n") or len(line.encode("utf-8")) > MAX_WORKER_IPC_FRAME_BYTES:
                raise Refuse("worker IPC frame exceeds the bounded protocol size")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise Refuse("worker IPC message must be an object")
            if request.get("operation") == "shutdown":
                attempt = recovery.read_json(attempt_path)
                shutdown_events = [
                    event
                    for event in recovery._read_ledger_events(ledger_path)
                    if event.get("event") == "worker_shutdown_started"
                    and event.get("attempt_id") == attempt.get("attempt_id")
                ]
                if (
                    request.get("attempt_id") != attempt.get("attempt_id")
                    or tuple(trainers) != ENDPOINT_ORDER
                    or len(shutdown_events) != 1
                ):
                    raise Refuse("worker shutdown request is not authorized after four restores")
                print(
                    json.dumps(
                        {
                            "status": "ok",
                            "operation": "shutdown",
                            "task_id": "shutdown",
                            "endpoint_id": "worker",
                            "worker_cpu_seconds_total": (time.process_time() - PROCESS_START_CPU),
                            "worker_wall_seconds_total": (time.monotonic() - PROCESS_START_WALL),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return 0
            response = _handle_task(
                request,
                attempt_path=attempt_path,
                ledger_path=ledger_path,
                durability_api=durability_api,
                trainers=trainers,
                endpoint_specs=endpoint_specs,
                corpus_holder=corpus_holder,
                job_name=job_name,
            )
            response["task_id"] = request.get("task_id")
            response["endpoint_id"] = request.get("endpoint_id")
            response["worker_cpu_seconds_total"] = time.process_time() - PROCESS_START_CPU
            response["worker_wall_seconds_total"] = time.monotonic() - PROCESS_START_WALL
            print(
                json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
        except BaseException as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "task_id": request.get("task_id") if isinstance(request, dict) else None,
                        "endpoint_id": (
                            request.get("endpoint_id") if isinstance(request, dict) else None
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "worker_cpu_seconds_total": time.process_time() - PROCESS_START_CPU,
                        "worker_wall_seconds_total": time.monotonic() - PROCESS_START_WALL,
                        "can_promote": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            return 1
    return 0

"""Versioned recovery-v2 entry: admission, one formal face access, private worker.

``--admit`` exercises the real Windows Job/IPC/NTFS path plus four real
read-only endpoint restores in an admission namespace.  ``--validate-only``
freezes the readiness of the single unexecuted formal namespace.  ``--execute``
runs the frozen one-shot evaluation with the v2 marker gate.  ``--worker`` is the
private IPC loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.training import (
    evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1 as worker,
)
from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as frozen
from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter
from scripts.training import taiji_r2_evaluation_recovery_v2 as provider

MODULE = "scripts.training.evaluate_taiji_r2_evaluation_recovery_v2"


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def worker_command(attempt_path: Path, ledger_path: Path, context: provider.Context) -> list[str]:
    return [
        sys.executable,
        "-m",
        MODULE,
        "--worker",
        "--attempt-path",
        str(Path(attempt_path).resolve()),
        "--ledger-path",
        str(Path(ledger_path).resolve()),
        "--job-name",
        "{job_name}",
        "--namespace-kind",
        context.kind,
        "--namespace-sequence",
        str(context.sequence),
    ]


def _worker_factory(context: provider.Context) -> Any:
    def factory(**kwargs: Any) -> Any:
        command = worker_command(kwargs["attempt_path"], kwargs["ledger_path"], context)
        return frozen.SupervisedWorker(**kwargs, command=command)

    return factory


def _durability() -> tuple[Any, dict[str, Any]]:
    api = frozen.WindowsDurabilityApi()
    return api, frozen.probe_ntfs_durability(provider.MARKER.parent, api=api)


def _admission_attempt(
    context: provider.Context, preflight: dict[str, Any], readiness: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": provider.ATTEMPT_SCHEMA,
        "status": "started",
        "run": frozen.RUN_ID,
        "attempt_id": f"admission-{secrets.token_hex(12)}",
        "engineering_admission": True,
        "readiness": readiness,
        "input_report_hashes": preflight["report_hashes"],
        "source_hashes": preflight["source_hashes"],
        "endpoint_order": list(frozen.ENDPOINT_ORDER),
        "endpoint_checkpoint_sha256": {
            row["endpoint_id"]: row["sha256"] for row in preflight["endpoints"]
        },
        "parameter_updates": 0,
        "optimizer_updates": 0,
        "dev_queries": 0,
        "generation_queries": 0,
        "can_promote": False,
    }


def _restore_endpoint(
    *,
    context: provider.Context,
    budget: frozen.EvaluationBudget,
    worker_handle: Any,
    attempt_id: str,
    index: int,
    endpoint: dict[str, Any],
    ledger_path: Path,
    durability_api: Any,
) -> dict[str, Any]:
    endpoint_id = endpoint["endpoint_id"]
    token, suffix = provider.new_task_token()
    task_id = f"restore-{index}-{suffix}"
    remaining = len(frozen.ENDPOINT_ORDER) - index
    reserve = provider.ADMISSION_RESTORE_RESERVE_SECONDS
    budget.guard(
        reserve_cpu_seconds=reserve * remaining + 5.0,
        reserve_wall_seconds=reserve * remaining + 5.0,
    )
    intent = {
        "event": "restore_started",
        "attempt_id": attempt_id,
        "task_id": task_id,
        "operation": "restore",
        "endpoint_id": endpoint_id,
        "checkpoint_sha256": endpoint["sha256"],
        "worker_token_sha256": worker_token_sha256(token),
        "remaining_tasks": remaining,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "unknown_tail": True,
    }
    frozen._append_jsonl(ledger_path, intent, flush_file=durability_api.flush_file)
    payload = {
        "operation": "restore",
        "attempt_id": attempt_id,
        "task_id": task_id,
        "endpoint_id": endpoint_id,
        "token": token,
        "checkpoint_sha256": endpoint["sha256"],
    }
    response, child, _parent = worker_handle.request(
        payload,
        budget=budget,
        task_cpu_limit_seconds=frozen.MAX_SINGLE_RESTORE_CPU_SECONDS,
        task_wall_limit_seconds=frozen.MAX_SINGLE_RESTORE_WALL_SECONDS,
        future_cpu_reserve_seconds=reserve * (remaining - 1) + 5.0,
        future_wall_reserve_seconds=reserve * (remaining - 1) + 5.0,
    )
    budget.charge_child_cpu(child)
    if (
        response.get("status") != "ok"
        or response.get("operation") != "restore"
        or response.get("task_id") != task_id
        or response.get("endpoint_id") != endpoint_id
    ):
        raise frozen.Refuse("admission restore response identity/status is invalid")
    receipt = dict(frozen._require_mapping(response.get("receipt"), "admission restore receipt"))
    expected = (
        (0, 0)
        if endpoint_id == "parent_baseline"
        else frozen.EXPECTED_STAGE_COUNTS["assistant_sft"]
    )
    if (
        receipt.get("complete") is not True
        or receipt.get("equal") is not True
        or receipt.get("endpoint_id") != endpoint_id
        or receipt.get("sha256") != endpoint["sha256"]
        or receipt.get("checkpoint_digest") != endpoint["checkpoint_digest"]
        or receipt.get("episode_count") != expected[0]
        or receipt.get("global_step") != expected[1]
    ):
        raise frozen.Refuse(f"{endpoint_id} admission restore receipt differs")
    frozen._append_jsonl(
        ledger_path,
        {
            "event": "restore_completed",
            "attempt_id": attempt_id,
            "task_id": task_id,
            "operation": "restore",
            "endpoint_id": endpoint_id,
            "receipt_sha256": provider.canonical_sha(receipt),
            "child_cpu_seconds": child,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "unknown_tail": False,
        },
        flush_file=durability_api.flush_file,
    )
    return receipt


def worker_token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def run_admission(context: provider.Context) -> dict[str, Any]:
    """Four real read-only restores under the real Job; never touches the face."""
    started_cpu, started_wall = time.process_time(), time.monotonic()
    preflight = context.preflight()
    paths = context.paths()
    readiness = adapter.write_extension_readiness(paths["readiness"], preflight)
    adapter.write_endpoint_manifest(paths["endpoint_manifest"], preflight)
    attempt = _admission_attempt(context, preflight, readiness)
    durability_api, qualification = _durability()
    budget = frozen.EvaluationBudget(
        cpu_limit_seconds=provider.ADMISSION_CPU_LIMIT_SECONDS,
        wall_limit_seconds=provider.ADMISSION_WALL_LIMIT_SECONDS,
        charged_parent_cpu_seconds=time.process_time() - started_cpu,
        charged_wall_seconds=time.monotonic() - started_wall,
    )
    attempt_sha = frozen._durable_write_json(
        paths["attempt"], attempt, qualification=qualification, api=durability_api
    )
    frozen._create_jsonl(
        paths["ledger"],
        {
            "event": "admission_started",
            "attempt_id": attempt["attempt_id"],
            "attempt_sha256": attempt_sha,
            "engineering_admission": True,
            "face_access": False,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "can_promote": False,
        },
    )
    provider.append_window_event(
        context,
        {
            "event": "admission_started",
            "phase": "admission",
            "sequence": context.sequence,
            "attempt_id": attempt["attempt_id"],
            "cpu_limit_seconds": provider.ADMISSION_CPU_LIMIT_SECONDS,
            "wall_limit_seconds": provider.ADMISSION_WALL_LIMIT_SECONDS,
        },
        durability_api=durability_api,
    )
    worker_handle: Any | None = None
    receipts: dict[str, dict[str, Any]] = {}
    try:
        worker_handle = _worker_factory(context)(
            attempt_path=paths["attempt"],
            ledger_path=paths["ledger"],
            budget=budget,
            startup_future_cpu_reserve_seconds=(
                provider.ADMISSION_RESTORE_RESERVE_SECONDS * len(frozen.ENDPOINT_ORDER) + 5.0
            ),
            startup_future_wall_reserve_seconds=(
                provider.ADMISSION_RESTORE_RESERVE_SECONDS * len(frozen.ENDPOINT_ORDER) + 5.0
            ),
        )
        budget.charge_child_cpu(float(getattr(worker_handle, "initial_child_cpu_seconds", 0.0)))
        for index, endpoint in enumerate(preflight["endpoints"]):
            receipts[endpoint["endpoint_id"]] = _restore_endpoint(
                context=context,
                budget=budget,
                worker_handle=worker_handle,
                attempt_id=attempt["attempt_id"],
                index=index,
                endpoint=dict(endpoint),
                ledger_path=paths["ledger"],
                durability_api=durability_api,
            )
        if tuple(receipts) != frozen.ENDPOINT_ORDER:
            raise frozen.Refuse("admission did not restore all four endpoints in order")
        frozen._append_jsonl(
            paths["ledger"],
            {
                "event": "worker_shutdown_started",
                "attempt_id": attempt["attempt_id"],
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "unknown_tail": True,
            },
            flush_file=durability_api.flush_file,
        )
        shutdown_child, _shutdown_parent = worker_handle.shutdown(
            budget=budget, cpu_limit_seconds=5.0, wall_limit_seconds=5.0
        )
        budget.charge_child_cpu(shutdown_child)
        worker_handle = None
        result = {
            "schema": "taiji-r2-evaluation-recovery-v2-admission-result-v1",
            "status": "complete",
            "complete": True,
            "attempt_id": attempt["attempt_id"],
            "engineering_admission": True,
            "face_access": False,
            "dev_queries": 0,
            "generation_outputs": 0,
            "restore_count": len(receipts),
            "restores": receipts,
            "cpu_seconds_parent": budget.parent_cpu_seconds(),
            "cpu_seconds_children": budget.child_cpu_seconds,
            "cpu_seconds_total": budget.cpu_seconds(),
            "wall_seconds_total": budget.wall_seconds(),
            "cpu_limit_seconds": provider.ADMISSION_CPU_LIMIT_SECONDS,
            "wall_limit_seconds": provider.ADMISSION_WALL_LIMIT_SECONDS,
            "unknown_tail": False,
            "can_promote": False,
        }
        frozen._write_new_json(paths["result"], result)
        context.record_outcome(
            "admission_completed",
            cpu_seconds=budget.cpu_seconds(),
            wall_seconds=budget.wall_seconds(),
            unknown_tail=False,
            attempt_id=attempt["attempt_id"],
            restore_count=len(receipts),
        )
        return result
    except BaseException as exc:
        if worker_handle is not None:
            sample = getattr(worker_handle, "sample_tree_cpu_seconds", None)
            before = float(sample()) if callable(sample) else 0.0
            stopped = bool(worker_handle.terminate_tree())
            if callable(sample):
                budget.charge_child_cpu(max(0.0, float(sample()) - before))
        else:
            stopped = bool(getattr(exc, "worker_tree_stopped", False))
        context.record_outcome(
            "admission_failed",
            cpu_seconds=budget.cpu_seconds(),
            wall_seconds=budget.wall_seconds(),
            unknown_tail=not stopped,
            attempt_id=attempt["attempt_id"],
            error_type=type(exc).__name__,
            error=str(exc),
            worker_tree_stopped=stopped,
        )
        frozen._write_new_json(
            paths["failure"],
            {
                "schema": "taiji-r2-evaluation-recovery-v2-admission-failure-v1",
                "attempt_id": attempt["attempt_id"],
                "status": "failed",
                "engineering_admission": True,
                "face_access": False,
                "dev_queries": 0,
                "generation_outputs": 0,
                "fresh_process_restores": receipts,
                "restored_endpoints": list(receipts),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cpu_seconds_total_at_failure": budget.cpu_seconds(),
                "wall_seconds_at_failure": budget.wall_seconds(),
                "worker_tree_stopped": stopped,
                "unknown_tail": not stopped,
                "can_promote": False,
            },
        )
        raise


def _record_formal(context: provider.Context, *, complete: bool, fallback_error: str = "") -> None:
    paths = context.paths()
    if complete:
        result = frozen.read_json(paths["result"])
        context.record_outcome(
            "formal_completed",
            cpu_seconds=float(result["cpu_seconds_total_at_finalization_start"]),
            wall_seconds=float(result["wall_seconds_total_at_finalization_start"]),
            unknown_tail=False,
            attempt_id=result.get("attempt_id"),
            generation_outputs=result.get("persisted_generation_outputs"),
            query_attempts=result.get("query_attempts"),
            note="publication cost is covered by the frozen finalization reserve",
        )
        return
    if paths["failure"].is_file():
        failure = frozen.read_json(paths["failure"])
        context.record_outcome(
            "formal_failed",
            cpu_seconds=float(failure.get("cpu_seconds_total_at_failure") or 0.0),
            wall_seconds=float(failure.get("wall_seconds_at_failure") or 0.0),
            unknown_tail=bool(failure.get("unknown_tail", True)),
            attempt_id=failure.get("attempt_id"),
            query_attempts=failure.get("query_attempts"),
            generation_outputs=failure.get("persisted_generation_outputs"),
            face_access_marker=provider.read_marker(context.marker_path) is not None,
            error=str(failure.get("error", "")),
        )
        return
    context.record_outcome(
        "formal_failed",
        cpu_seconds=0.0,
        wall_seconds=0.0,
        unknown_tail=True,
        error=fallback_error,
        face_access_marker=provider.read_marker(context.marker_path) is not None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--admit", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ledger-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--job-name", help=argparse.SUPPRESS)
    parser.add_argument("--namespace-kind", choices=provider.PHASES, help=argparse.SUPPRESS)
    parser.add_argument("--namespace-sequence", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        if args.worker:
            if (
                not args.attempt_path
                or not args.ledger_path
                or not args.job_name
                or args.namespace_kind not in provider.PHASES
                or not args.namespace_sequence
            ):
                raise ValueError("private worker requires attempt, ledger, Job and namespace")
            context = provider.Context(args.namespace_kind, args.namespace_sequence)
            return worker.worker_loop(
                args.attempt_path.resolve(),
                args.ledger_path.resolve(),
                args.job_name,
                preflight_fn=context.preflight,
                evaluation_paths_fn=context.paths,
                generation_gate=context.require_face_access,
            )
        if args.attempt_path or args.ledger_path or args.job_name:
            raise ValueError("private worker paths are not public evaluation arguments")
        if args.admit:
            sequence = provider.next_sequence("admission")
            context = provider.Context("admission", sequence)
            result = run_admission(context)
            _emit(
                {
                    "status": result["status"],
                    "namespace": f"admission_{sequence:03d}",
                    "restore_count": result["restore_count"],
                    "face_access": False,
                    "can_promote": False,
                }
            )
            return 0
        prepared = provider.prepared_formal_sequence(
            reports_dir=provider.ROOT / "reports", prefix_base=provider.REPORT_PREFIX
        )
        if args.validate_only:
            sequence = prepared if prepared is not None else provider.next_sequence("formal")
            context = provider.Context("formal", sequence)
            preflight = context.preflight()
            paths = context.paths()
            adapter.write_extension_readiness(paths["readiness"], preflight)
            manifest = adapter.write_endpoint_manifest(paths["endpoint_manifest"], preflight)
            provider.append_window_event(
                context,
                {
                    "event": "formal_prepared",
                    "phase": "formal",
                    "sequence": sequence,
                    "readiness_sha256": frozen.sha256_file(paths["readiness"]),
                    "endpoint_manifest_sha256": manifest["manifest_sha256"],
                },
            )
            _emit(
                {
                    "status": "ready",
                    "namespace": f"formal_{sequence:03d}",
                    "endpoint_schema": manifest["schema"],
                    "endpoint_manifest_sha256": manifest["manifest_sha256"],
                    "endpoint_order": manifest["endpoint_order"],
                    "expected_generation_outputs": provider.GENERATION_OUTPUTS,
                    "can_promote": False,
                }
            )
            return 0
        if prepared is None:
            raise ValueError("run --validate-only and save the formal readiness before --execute")
        context = provider.Context("formal", prepared)
        preflight = context.preflight()
        if frozen.read_json(context.paths()["readiness"]) != frozen._readiness_payload(preflight):
            raise ValueError("saved formal readiness no longer matches the frozen metadata")
        provider.append_window_event(
            context,
            {
                "event": "formal_started",
                "phase": "formal",
                "sequence": prepared,
                "attempt_path": str(context.paths()["attempt"].resolve()),
                "face_access_marker": provider.read_marker(context.marker_path) is not None,
            },
        )
        try:
            result = frozen.execute(
                paths=context.paths(),
                preflight_fn=context.preflight,
                prompt_selector=context.prompt_selector(),
                worker_factory=_worker_factory(context),
                budget_factory=context.budget_factory,
            )
        except BaseException as exc:
            _record_formal(context, complete=False, fallback_error=str(exc))
            raise
        _record_formal(context, complete=True)
        _emit({"status": result.get("status"), "can_promote": False})
        return 0 if result.get("complete") is True else 1
    except Exception as exc:
        _emit(
            {
                "status": "refused",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "can_promote": False,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""One-shot paired evaluation contract for the R2 recovery-v2 endpoints.

The metadata and fixture helpers in this module are intentionally independent
of prompt selection. The real dev face is only selected by the execution path
after the versioned metadata, source, checkpoint, and restore gates pass.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from ctypes import wintypes
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROCESS_START_CPU = time.process_time()
PROCESS_START_WALL = time.monotonic()

import psutil
import torch

from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as base
from scripts.training import run_taiji_r2_daily_dialogue_recovery as recovery
from taiji import LanguageAlignmentTrainer, Taiji
from taiji.internalization import content_digest

ROOT = base.ROOT
RUN_ID = "r2_daily_dialogue_recovery_v2_20261021"
REPORT_PREFIX = ROOT / "reports" / f"taiji_{RUN_ID}_paired_eval"
ARCHITECTURE_ADDENDUM = (
    ROOT
    / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-v2-evaluation-architecture.md"
)
CONTRACT_TEST = (
    ROOT / "tests/taiji_native/test_r2_daily_dialogue_recovery_v2_evaluation_contract.py"
)
FULL_TURN_PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md"
WARMSTART_PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-WARMSTART-MATCHED_20261021.md"
RECOVERY_PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-RECOVERY_20261021.md"

ARMS = ("full_turn_1m", "warm_start", "fresh_same_capacity")
ENDPOINT_ORDER = ("parent_baseline", *ARMS)
EXPECTED_STAGE_COUNTS = {
    "pretraining": (30_096, 14_517_912, "pretraining_final.pt"),
    "assistant_sft": (525, 113_209, "assistant_sft_final.pt"),
}
ARM_CPU_LIMITS = {
    "full_turn_1m": 240_000.0,
    "warm_start": 120_000.0,
    "fresh_same_capacity": 120_000.0,
}
ARM_EPISODE_RESERVES = {"full_turn_1m": 1_320.0, "warm_start": 240.0, "fresh_same_capacity": 240.0}
EVALUATION_CPU_LIMIT_SECONDS = 14_400.0
EVALUATION_WALL_LIMIT_SECONDS = 14_400.0
MIN_FREE_MEMORY_GIB = 8.0
MAX_SYSTEM_CPU_PERCENT = 85.0
FACE_SIZE = 39
MAX_GENERATION_BYTES = 512
MIN_OUTPUT_DISK_BYTES = 512 * 1024**2
MOVEFILE_WRITE_THROUGH = 0x00000008
DRIVE_FIXED = 3
MAX_WORKER_IPC_FRAME_BYTES = 2048
RESTORE_RESERVE_CPU_SECONDS = 120.0
RESTORE_RESERVE_WALL_SECONDS = 120.0
GENERATION_RESERVE_CPU_SECONDS = 60.0
GENERATION_RESERVE_WALL_SECONDS = 60.0
MAX_SINGLE_RESTORE_CPU_SECONDS = 120.0
MAX_SINGLE_RESTORE_WALL_SECONDS = 120.0
MAX_SINGLE_GENERATION_CPU_SECONDS = 60.0
MAX_SINGLE_GENERATION_WALL_SECONDS = 60.0
FINALIZATION_RESERVE_CPU_SECONDS = 300.0
FINALIZATION_RESERVE_WALL_SECONDS = 300.0
FACE_PREPARE_RESERVE_CPU_SECONDS = 60.0
FACE_PREPARE_RESERVE_WALL_SECONDS = 60.0
SUPERVISOR_POLL_INTERVAL_SECONDS = 0.025
SUPERVISOR_TERMINATE_GRACE_SECONDS = 1.0
SUPERVISOR_CANCEL_RESERVE_SECONDS = 5.0
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
CREATE_SUSPENDED = 0x00000004
THREAD_TERMINATE = 0x0001
THREAD_SUSPEND_RESUME = 0x0002
TH32CS_SNAPTHREAD = 0x00000004


class Refuse(RuntimeError):
    """A recovery-v2 paired-evaluation precondition failed."""


class WindowsDurabilityApi:
    """Small Win32 surface for local NTFS write-through marker publication."""

    def __init__(self) -> None:
        if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
            raise Refuse("DURABILITY_UNSUPPORTED: Windows local NTFS is required")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.GetVolumePathNameW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        self.kernel32.GetVolumePathNameW.restype = wintypes.BOOL
        self.kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        self.kernel32.GetDriveTypeW.restype = wintypes.UINT
        self.kernel32.GetVolumeInformationW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        self.kernel32.GetVolumeInformationW.restype = wintypes.BOOL
        self.kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        self.kernel32.FlushFileBuffers.restype = wintypes.BOOL
        self.kernel32.MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        self.kernel32.MoveFileExW.restype = wintypes.BOOL

    def inspect(self, path: Path) -> dict[str, Any]:
        raw = str(path.resolve())
        if raw.startswith("\\\\"):
            return {"is_unc": True, "drive_type": None, "filesystem": None, "volume_root": None}
        volume_buffer = ctypes.create_unicode_buffer(32768)
        if not self.kernel32.GetVolumePathNameW(raw, volume_buffer, len(volume_buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
        volume_root = volume_buffer.value
        drive_type = int(self.kernel32.GetDriveTypeW(volume_root))
        filesystem_buffer = ctypes.create_unicode_buffer(64)
        volume_name_buffer = ctypes.create_unicode_buffer(256)
        serial = wintypes.DWORD()
        max_component = wintypes.DWORD()
        flags = wintypes.DWORD()
        if not self.kernel32.GetVolumeInformationW(
            volume_root,
            volume_name_buffer,
            len(volume_name_buffer),
            ctypes.byref(serial),
            ctypes.byref(max_component),
            ctypes.byref(flags),
            filesystem_buffer,
            len(filesystem_buffer),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "is_unc": False,
            "drive_type": drive_type,
            "filesystem": filesystem_buffer.value,
            "volume_root": volume_root,
        }

    def flush_file(self, fd: int) -> None:
        import msvcrt

        handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
        if not self.kernel32.FlushFileBuffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def move_file_write_through(self, source: Path, destination: Path, flags: int) -> None:
        if flags != MOVEFILE_WRITE_THROUGH:
            raise Refuse("MoveFileExW flags must be exactly MOVEFILE_WRITE_THROUGH")
        if not self.kernel32.MoveFileExW(str(source), str(destination), flags):
            raise ctypes.WinError(ctypes.get_last_error())


def probe_ntfs_durability(
    directory: Path,
    *,
    api: Any | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless a pre-existing local fixed NTFS directory is proven."""
    directory = Path(directory).resolve()
    platform_name = os.name if platform_name is None else platform_name
    if platform_name != "nt":
        raise Refuse("DURABILITY_UNSUPPORTED: Windows local NTFS is required")
    if not directory.exists() or not directory.is_dir():
        raise Refuse("DURABILITY_UNSUPPORTED: report directory must pre-exist")
    if str(directory).startswith("\\\\"):
        raise Refuse("DURABILITY_UNSUPPORTED: UNC/network report paths are rejected")
    api = api or WindowsDurabilityApi()
    try:
        info = api.inspect(directory)
    except BaseException as exc:
        raise Refuse(f"DURABILITY_UNSUPPORTED: filesystem probe failed: {exc}") from exc
    if (
        info.get("is_unc") is not False
        or info.get("drive_type") != DRIVE_FIXED
        or str(info.get("filesystem", "")).upper() != "NTFS"
        or not isinstance(info.get("volume_root"), str)
        or not info.get("volume_root")
    ):
        raise Refuse("DURABILITY_UNSUPPORTED: only local fixed-volume NTFS is supported")
    return {
        "directory": str(directory),
        "volume_root": str(info["volume_root"]),
        "filesystem": "NTFS",
        "drive_type": DRIVE_FIXED,
        "publication": "exclusive-temp-flush-movefileex-write-through",
    }


def _durable_publish_bytes(
    path: Path,
    data: bytes,
    *,
    qualification: Mapping[str, Any],
    api: Any,
) -> str:
    """Publish one write-once marker via same-directory write-through rename."""
    path = Path(path).resolve()
    parent = path.parent
    if not parent.is_dir():
        raise Refuse("DURABILITY_UNSUPPORTED: marker directory must pre-exist")
    if path.exists():
        raise FileExistsError(path)
    info = api.inspect(parent)
    if (
        info.get("is_unc") is not False
        or info.get("drive_type") != DRIVE_FIXED
        or str(info.get("filesystem", "")).upper() != "NTFS"
        or str(info.get("volume_root", "")).casefold()
        != str(qualification["volume_root"]).casefold()
    ):
        raise Refuse("DURABILITY_UNSUPPORTED: marker destination changed volume/filesystem")
    temporary = parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            api.flush_file(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        if temporary.parent.resolve() != path.parent.resolve():
            raise Refuse("DURABILITY_UNSUPPORTED: temp and marker are not in the same directory")
        api.move_file_write_through(temporary, path, MOVEFILE_WRITE_THROUGH)
        if not path.is_file() or path.read_bytes() != data:
            raise Refuse("durable marker did not verify after write-through publication")
        return sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _durable_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    qualification: Mapping[str, Any],
    api: Any,
) -> str:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return _durable_publish_bytes(path, data, qualification=qualification, api=api)


def _durable_write_jsonl(
    path: Path,
    payload: Mapping[str, Any],
    *,
    qualification: Mapping[str, Any],
    api: Any,
) -> str:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return _durable_publish_bytes(path, data, qualification=qualification, api=api)


def _worker_claim_path(attempt_path: Path, task_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_-]{1,96}", task_id):
        raise Refuse("worker task id contains unsupported characters")
    return attempt_path.with_name(f"{attempt_path.name}.{task_id}.claim.json")


def _read_ledger_events(path: Path) -> list[dict[str, Any]]:
    try:
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Refuse(f"cannot read worker authorization ledger: {exc}") from exc
    if any(not isinstance(event, dict) for event in events):
        raise Refuse("worker authorization ledger contains a non-object event")
    return events


def _authorize_worker_task(
    *,
    operation: str,
    endpoint_id: str,
    task_id: str,
    token: str | None,
    attempt_path: Path,
    ledger_path: Path,
    api: Any,
    qualification: Mapping[str, Any] | None = None,
    expected_attempt_path: Path | None = None,
    expected_ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Consume one parent-authored task token before corpus/checkpoint access."""
    if operation not in {"restore", "generate"} or endpoint_id not in ENDPOINT_ORDER:
        raise Refuse("worker task type or endpoint is not allowed")
    if not isinstance(token, str) or len(token) < 32:
        raise Refuse("worker task lacks a valid one-shot parent token")
    attempt_path = Path(attempt_path).resolve()
    ledger_path = Path(ledger_path).resolve()
    expected_paths = _evaluation_paths()
    expected_attempt = Path(expected_attempt_path or expected_paths["attempt"]).resolve()
    expected_ledger = Path(expected_ledger_path or expected_paths["ledger"]).resolve()
    if attempt_path != expected_attempt or ledger_path != expected_ledger:
        raise Refuse("worker attempt/ledger path is outside the fixed evaluation contract")
    attempt = read_json(attempt_path)
    if (
        attempt.get("schema") != "taiji-r2-dialogue-recovery-v2-paired-evaluation-attempt-v1"
        or attempt.get("status") != "started"
        or attempt.get("run") != RUN_ID
        or not isinstance(attempt.get("attempt_id"), str)
    ):
        raise Refuse("worker attempt identity/status is invalid")
    events = _read_ledger_events(ledger_path)
    task_starts = [event for event in events if event.get("task_id") == task_id]
    expected_event = "restore_started" if operation == "restore" else "query_started"
    if len(task_starts) != 1 or task_starts[0].get("event") != expected_event:
        raise Refuse("worker task has no unique matching durable parent intent")
    start = task_starts[0]
    token_sha = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if (
        start.get("attempt_id") != attempt.get("attempt_id")
        or start.get("operation") != operation
        or start.get("endpoint_id") != endpoint_id
        or start.get("worker_token_sha256") != token_sha
    ):
        raise Refuse("worker task token does not match the unique durable parent intent")
    if any(
        event.get("worker_token_sha256") == token_sha and event.get("task_id") != task_id
        for event in events
    ):
        raise Refuse("worker one-shot token was already bound to another task")
    terminal_events = {
        "restore_completed",
        "restore_failed",
        "restore_timeout",
        "query_output_persisted",
        "query_failed",
        "query_timeout",
    }
    if any(
        event.get("task_id") == task_id and event.get("event") in terminal_events
        for event in events
    ):
        raise Refuse("worker task token has already been consumed")
    active_tasks = {
        str(event["task_id"])
        for event in events
        if event.get("event") in {"restore_started", "query_started"}
        and isinstance(event.get("task_id"), str)
        and not any(
            later.get("task_id") == event.get("task_id") and later.get("event") in terminal_events
            for later in events
        )
    }
    if active_tasks != {task_id}:
        raise Refuse("worker must consume exactly one currently open parent task")
    claim_path = _worker_claim_path(attempt_path, task_id)
    if claim_path.exists():
        raise Refuse("worker task claim already exists; duplicate worker is forbidden")
    if qualification is None:
        qualification = probe_ntfs_durability(attempt_path.parent, api=api)
    claim = {
        "schema": "taiji-r2-dialogue-recovery-v2-worker-claim-v1",
        "attempt_id": attempt["attempt_id"],
        "task_id": task_id,
        "operation": operation,
        "endpoint_id": endpoint_id,
        "token_sha256": token_sha,
        "worker_pid": os.getpid(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _durable_write_json(claim_path, claim, qualification=qualification, api=api)
    return {"attempt": attempt, "intent": start, "claim_path": claim_path}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refuse(f"cannot read evaluation metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Refuse(f"evaluation metadata must be a JSON object: {path}")
    return payload


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Refuse(f"{label} must be an object")
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Refuse(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise Refuse(f"{label} must be a SHA-256 hex digest") from exc
    return value.lower()


def _path_inside(path_value: Any, root: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise Refuse(f"{label} path is missing")
    path = Path(path_value)
    if not path.is_absolute():
        raise Refuse(f"{label} path must be absolute")
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise Refuse(f"{label} path escapes its frozen root")
    return resolved


def verify_bound_sources(
    source_hashes: Mapping[str, Any], *, hash_file: Callable[[Path], str] = sha256_file
) -> dict[str, str]:
    """Verify all source paths carried by a training binding and return its map."""
    if not source_hashes:
        raise Refuse("source_hashes must not be empty")
    verified: dict[str, str] = {}
    for raw_path, raw_expected in sorted(source_hashes.items()):
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise Refuse("source binding contains a non-absolute path")
        expected = _require_sha(raw_expected, f"source hash for {raw_path}")
        path = Path(raw_path)
        if not path.is_file() or hash_file(path) != expected:
            raise Refuse(f"bound source is missing or changed: {raw_path}")
        verified[str(path.resolve())] = expected
    return verified


def _validate_restore_record(
    restore: Mapping[str, Any],
    *,
    stage: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    episodes: int,
    target_byte_updates: int,
) -> None:
    if restore.get("complete") is not True or restore.get("equal") is not True:
        raise Refuse(f"{stage} final checkpoint lacks a complete independent restore")
    if restore.get("stage") != stage:
        raise Refuse(f"{stage} restore receipt names a different stage")
    if Path(str(restore.get("path", ""))).resolve() != checkpoint:
        raise Refuse(f"{stage} restore path differs from its checkpoint")
    if _require_sha(restore.get("sha256"), f"{stage} restore file SHA") != checkpoint_sha256:
        raise Refuse(f"{stage} restore file SHA differs from its result")
    if not isinstance(restore.get("checkpoint_digest"), str) or not restore["checkpoint_digest"]:
        raise Refuse(f"{stage} restore receipt has no checkpoint digest")
    if (
        restore.get("episode_count") != episodes
        or restore.get("global_step") != target_byte_updates
    ):
        raise Refuse(f"{stage} restore receipt denominator differs")
    cpu_seconds = restore.get("cpu_seconds")
    if (
        not isinstance(cpu_seconds, (int, float))
        or not math.isfinite(float(cpu_seconds))
        or cpu_seconds < 0
    ):
        raise Refuse(f"{stage} restore CPU receipt is invalid")


def validate_arm_result(
    arm: str,
    readiness: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    arm_root: Path | None = None,
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    """Validate one immutable v2 result and its two final stage receipts.

    This function reads no prompt, corpus, or model payload. It only checks
    already persisted JSON metadata and hashes the named source/checkpoint files.
    """
    if arm not in ARMS:
        raise Refuse(f"unknown recovery-v2 arm: {arm}")
    readiness = _require_mapping(readiness, f"{arm} readiness")
    result = _require_mapping(result, f"{arm} result")
    if (
        readiness.get("schema") != "taiji-r2-dialogue-recovery-readiness-v1"
        or readiness.get("status") != "ready"
        or readiness.get("arm") != arm
        or readiness.get("can_promote") is not False
        or readiness.get("formal_updates") != 0
        or readiness.get("dev_queries") != 0
    ):
        raise Refuse(f"{arm} readiness is incomplete or has an unsafe capability flag")
    if (
        result.get("schema") != "taiji-r2-dialogue-recovery-result-v1"
        or result.get("status") != "complete"
        or result.get("complete") is not True
        or result.get("arm") != arm
        or result.get("restart_from_zero") is not True
        or result.get("can_promote") is not False
        or result.get("dev_queries") != 0
        or result.get("generation_queries") != 0
    ):
        raise Refuse(f"{arm} result is incomplete or contains an unsafe query/capability flag")

    readiness_bindings = _require_mapping(readiness.get("bindings"), f"{arm} readiness bindings")
    result_bindings = _require_mapping(result.get("bindings"), f"{arm} result bindings")
    if readiness_bindings != result_bindings:
        raise Refuse(f"{arm} readiness and result bindings differ")
    if (
        result_bindings.get("run") != RUN_ID
        or result_bindings.get("arm") != arm
        or not result_bindings.get("zero_sha256")
    ):
        raise Refuse(f"{arm} run/arm/zero binding differs")

    source_hashes = _require_mapping(result_bindings.get("source_hashes"), f"{arm} source hashes")
    verified_sources = verify_bound_sources(source_hashes, hash_file=hash_file)

    expected_reserve = ARM_EPISODE_RESERVES[arm]
    if (
        readiness.get("per_episode_cpu_reserve") != expected_reserve
        or result.get("per_episode_cpu_reserve") != expected_reserve
        or readiness.get("worst_episode_probe_hashes") != result.get("worst_episode_probe_hashes")
    ):
        raise Refuse(f"{arm} calibrated episode reserve/probe binding differs")
    if readiness.get("worst_episode_probe_hashes") is None:
        raise Refuse(f"{arm} readiness has no longest-episode probe binding")

    cpu_seconds = result.get("cpu_seconds")
    if (
        not isinstance(cpu_seconds, (int, float))
        or not math.isfinite(float(cpu_seconds))
        or cpu_seconds < 0
        or cpu_seconds > ARM_CPU_LIMITS[arm]
    ):
        raise Refuse(f"{arm} training CPU total exceeds its frozen arm budget")

    root = arm_root if arm_root is not None else recovery.paths(arm)["root"]
    stages = _require_mapping(result.get("stages"), f"{arm} final stages")
    final_checkpoints: dict[str, dict[str, Any]] = {}
    for stage, (
        expected_episodes,
        expected_updates,
        expected_name,
    ) in EXPECTED_STAGE_COUNTS.items():
        record = _require_mapping(stages.get(stage), f"{arm} {stage} stage")
        if (
            record.get("episodes") != expected_episodes
            or record.get("target_byte_updates") != expected_updates
        ):
            raise Refuse(f"{arm} {stage} fixed episode/update denominator differs")
        checkpoint = _path_inside(record.get("path"), root, f"{arm} {stage}")
        if checkpoint.parent != root.resolve() or checkpoint.name != expected_name:
            raise Refuse(f"{arm} {stage} checkpoint is not the frozen final file")
        if not checkpoint.is_file():
            raise Refuse(f"{arm} {stage} final checkpoint is missing")
        checkpoint_sha = _require_sha(record.get("sha256"), f"{arm} {stage} checkpoint SHA")
        if hash_file(checkpoint) != checkpoint_sha:
            raise Refuse(f"{arm} {stage} checkpoint bytes differ from the result")
        restore = _require_mapping(record.get("restore"), f"{arm} {stage} restore")
        _validate_restore_record(
            restore,
            stage=stage,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            episodes=expected_episodes,
            target_byte_updates=expected_updates,
        )
        final_checkpoints[stage] = {
            "path": checkpoint,
            "sha256": checkpoint_sha,
            "checkpoint_digest": restore["checkpoint_digest"],
            "episodes": expected_episodes,
            "target_byte_updates": expected_updates,
        }

    return {
        "arm": arm,
        "readiness": dict(readiness),
        "result": dict(result),
        "result_sha256": None,
        "source_hashes": verified_sources,
        "final_checkpoints": final_checkpoints,
        "training_cpu_seconds": float(cpu_seconds),
    }


def validate_baseline_checkpoint(
    path: Path,
    *,
    expected_sha256: str = base.BASELINE_CHECKPOINT_SHA256,
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    """Verify baseline file identity without deserializing its model payload."""
    if not path.is_file():
        raise Refuse("fixed R2 parent baseline checkpoint is missing")
    expected = _require_sha(expected_sha256, "baseline checkpoint SHA")
    actual = hash_file(path)
    if actual != expected:
        raise Refuse("fixed R2 parent baseline checkpoint bytes changed")
    return {
        "endpoint_id": "parent_baseline",
        "path": path.resolve(),
        "sha256": actual,
        "checkpoint_digest": "1a77364a42633686dc9e1e9ee6a13a7e09b4f07770e3841d873bacd9eadaeaeb",
    }


def build_endpoint_map(
    baseline: Mapping[str, Any], arm_records: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Return the frozen baseline + three assistant_sft final endpoints."""
    if set(arm_records) != set(ARMS):
        raise Refuse("paired evaluation requires exactly the three recovery-v2 arms")
    if baseline.get("endpoint_id") != "parent_baseline":
        raise Refuse("the fourth endpoint must be the fixed parent baseline")
    baseline_endpoint = {
        "endpoint_id": "parent_baseline",
        "path": Path(baseline["path"]).resolve(),
        "sha256": _require_sha(baseline.get("sha256"), "baseline SHA"),
        "checkpoint_digest": str(baseline.get("checkpoint_digest", "")),
        "source": "fixed_parent_checkpoint",
    }
    if not baseline_endpoint["checkpoint_digest"]:
        raise Refuse("baseline checkpoint digest is missing")

    endpoints = [baseline_endpoint]
    for arm in ARMS:
        record = _require_mapping(arm_records[arm], f"{arm} validated result")
        if record.get("arm") != arm:
            raise Refuse(f"validated result key/arm identity differs for {arm}")
        final = _require_mapping(record.get("final_checkpoints"), f"{arm} final checkpoints")
        assistant = _require_mapping(final.get("assistant_sft"), f"{arm} assistant_sft final")
        if assistant.get("path") is None or assistant.get("sha256") is None:
            raise Refuse(f"{arm} assistant_sft final endpoint is incomplete")
        endpoints.append(
            {
                "endpoint_id": arm,
                "path": Path(assistant["path"]).resolve(),
                "sha256": _require_sha(assistant.get("sha256"), f"{arm} assistant_sft SHA"),
                "checkpoint_digest": str(assistant.get("checkpoint_digest", "")),
                "source": "assistant_sft_final",
            }
        )
        if not endpoints[-1]["checkpoint_digest"]:
            raise Refuse(f"{arm} assistant_sft checkpoint digest is missing")
    if tuple(endpoint["endpoint_id"] for endpoint in endpoints) != ENDPOINT_ORDER:
        raise Refuse("paired evaluation endpoint order differs from its frozen contract")
    return tuple(endpoints)


def expected_generation_count(family_count: int = FACE_SIZE) -> int:
    if family_count != FACE_SIZE:
        raise Refuse(f"paired evaluation face must contain exactly {FACE_SIZE} families")
    return family_count * len(ENDPOINT_ORDER)


def freeze_prompt_face(prompts: Sequence[Any]) -> dict[str, Any]:
    """Canonicalize selected prompts without copying labels or target responses."""
    if len(prompts) != FACE_SIZE:
        raise Refuse(f"prompt face must contain exactly {FACE_SIZE} families")
    rows: list[dict[str, Any]] = []
    family_ids: set[str] = set()
    episode_ids: set[str] = set()
    prompt_hashes: list[str] = []
    face = hashlib.sha256()
    for index, prompt in enumerate(prompts, 1):
        try:
            episode_id = str(prompt.episode_id)
            family_id = str(prompt.family_id)
            system = str(prompt.system)
            context = str(prompt.context)
            history = [[str(user), str(assistant)] for user, assistant in prompt.history]
            user_input = str(prompt.user_input)
            unknown_policy = str(prompt.unknown_policy)
        except (AttributeError, TypeError) as exc:
            raise Refuse("prompt selector returned an invalid prompt-only episode") from exc
        if not episode_id or not family_id or not user_input:
            raise Refuse("prompt face has an empty episode/family/user identity")
        if family_id in family_ids or episode_id in episode_ids:
            raise Refuse("prompt face contains a repeated family or episode")
        family_ids.add(family_id)
        episode_ids.add(episode_id)
        row = {
            "index": index,
            "episode_id": episode_id,
            "family_id": family_id,
            "split": "dev",
            "system": system,
            "context": context,
            "history_turn_count": len(history),
            "history": history,
            "current_user_turn": user_input,
            "unknown_policy": unknown_policy,
        }
        canonical = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        prompt_hashes.append(digest)
        face.update(canonical)
        face.update(b"\n")
        rows.append(row)
    return {"rows": rows, "prompt_hashes": prompt_hashes, "face_sha256": face.hexdigest()}


def validate_resource_snapshot(
    cpu_percent_samples: Sequence[float], available_memory_gib: float
) -> dict[str, Any]:
    if len(cpu_percent_samples) != 2:
        raise Refuse("evaluation requires exactly two fresh system CPU samples")
    samples = [float(value) for value in cpu_percent_samples]
    if any(
        not math.isfinite(value) or value < 0 or value > MAX_SYSTEM_CPU_PERCENT for value in samples
    ):
        raise Refuse("evaluation start CPU samples exceed the frozen 85% ceiling")
    memory = float(available_memory_gib)
    if not math.isfinite(memory) or memory < MIN_FREE_MEMORY_GIB:
        raise Refuse(f"evaluation requires at least {MIN_FREE_MEMORY_GIB} GiB available memory")
    return {"cpu_percent_samples": samples, "available_memory_gib": round(memory, 3)}


def resource_snapshot() -> dict[str, Any]:
    samples = [psutil.cpu_percent(interval=1.0), psutil.cpu_percent(interval=1.0)]
    available_gib = psutil.virtual_memory().available / 1024**3
    return validate_resource_snapshot(samples, available_gib)


@dataclass
class EvaluationBudget:
    cpu_limit_seconds: float = EVALUATION_CPU_LIMIT_SECONDS
    wall_limit_seconds: float = EVALUATION_WALL_LIMIT_SECONDS
    cpu_clock: Callable[[], float] = field(default_factory=lambda: time.process_time)
    wall_clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)
    charged_parent_cpu_seconds: float = 0.0
    charged_wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.cpu_started = self.cpu_clock()
        self.wall_started = self.wall_clock()
        self.child_cpu_seconds = 0.0

    def parent_cpu_seconds(self) -> float:
        return self.charged_parent_cpu_seconds + self.cpu_clock() - self.cpu_started

    def cpu_seconds(self) -> float:
        return self.parent_cpu_seconds() + self.child_cpu_seconds

    def wall_seconds(self) -> float:
        return self.charged_wall_seconds + self.wall_clock() - self.wall_started

    def charge_child_cpu(self, seconds: float) -> None:
        value = float(seconds)
        if not math.isfinite(value) or value < 0:
            raise Refuse("child process CPU receipt is invalid")
        self.child_cpu_seconds += value

    def guard(self, *, reserve_cpu_seconds: float = 0.0, reserve_wall_seconds: float = 0.0) -> None:
        cpu = self.cpu_seconds()
        wall = self.wall_seconds()
        if (
            not math.isfinite(reserve_cpu_seconds)
            or not math.isfinite(reserve_wall_seconds)
            or reserve_cpu_seconds < 0
            or reserve_wall_seconds < 0
        ):
            raise Refuse("evaluation reserve must be finite and non-negative")
        if (
            not math.isfinite(cpu)
            or not math.isfinite(wall)
            or cpu + reserve_cpu_seconds > self.cpu_limit_seconds
            or wall + reserve_wall_seconds > self.wall_limit_seconds
        ):
            raise TimeoutError(
                "R2 recovery-v2 paired evaluation budget exceeded "
                f"(cpu={cpu:.3f}+{reserve_cpu_seconds:.3f}/{self.cpu_limit_seconds:.3f}, "
                f"wall={wall:.3f}+{reserve_wall_seconds:.3f}/{self.wall_limit_seconds:.3f})"
            )


def _evaluation_source_paths() -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                ARCHITECTURE_ADDENDUM,
                CONTRACT_TEST,
                Path(__file__).resolve(),
                Path(base.__file__).resolve(),
                Path(recovery.__file__).resolve(),
                FULL_TURN_PLAN,
                WARMSTART_PLAN,
                RECOVERY_PLAN,
                ROOT / "taiji/language_alignment.py",
                ROOT / "taiji/model.py",
                ROOT / "taiji/response_plan_target.py",
                base.ASSISTANT_EPISODES,
                base.ASSISTANT_MANIFEST,
            )
        )
    )


def evaluation_source_hashes(*, hash_file: Callable[[Path], str] = sha256_file) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in _evaluation_source_paths():
        if not path.is_file():
            raise Refuse(f"evaluation freeze source is missing: {path}")
        hashes[str(path.resolve())] = hash_file(path)
    return dict(sorted(hashes.items()))


def _evaluation_paths() -> dict[str, Path]:
    return {
        "readiness": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_readiness.json"),
        "attempt": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_attempt.json"),
        "ledger": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_ledger.jsonl"),
        "progress": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_progress.jsonl"),
        "face": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_face.json"),
        "generations": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_generations.json"),
        "result": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_result.json"),
        "cost_receipt": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_cost_receipt.json"),
        "review": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_review.md"),
        "failure": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_failure.json"),
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_new_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _create_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(
    path: Path,
    payload: Mapping[str, Any],
    *,
    flush_file: Callable[[int], None] | None = None,
) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        (flush_file or os.fsync)(stream.fileno())


def _merge_hash_maps(*maps: Mapping[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source_map in maps:
        for raw_path, raw_digest in source_map.items():
            path = Path(raw_path)
            resolved = str(path.resolve())
            digest = _require_sha(raw_digest, f"source hash for {raw_path}")
            previous = merged.get(resolved)
            if previous is not None and previous != digest:
                raise Refuse(f"source bindings disagree for {resolved}")
            merged[resolved] = digest
    return dict(sorted(merged.items()))


def metadata_preflight(
    *,
    arm_paths: Callable[[str], Mapping[str, Path]] | None = None,
    hash_file: Callable[[Path], str] = sha256_file,
    eval_hashes: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Read and validate reports, source/checkpoint bytes, and endpoint mapping.

    This is metadata-only: it never calls the prompt selector, loads a corpus,
    deserializes a checkpoint, or starts a model/restore worker.
    """
    arm_paths = arm_paths or recovery.paths
    validated_arms: dict[str, dict[str, Any]] = {}
    report_hashes: dict[str, dict[str, str]] = {}
    arm_source_maps: list[Mapping[str, Any]] = []
    for arm in ARMS:
        paths = arm_paths(arm)
        readiness_path = Path(paths["readiness"])
        result_path = Path(paths["result"])
        readiness = read_json(readiness_path)
        result = read_json(result_path)
        record = validate_arm_result(
            arm,
            readiness,
            result,
            arm_root=Path(paths["root"]),
            hash_file=hash_file,
        )
        record["result_sha256"] = hash_file(result_path)
        record["readiness_sha256"] = hash_file(readiness_path)
        validated_arms[arm] = record
        report_hashes[arm] = {
            "result_sha256": record["result_sha256"],
            "readiness_sha256": record["readiness_sha256"],
        }
        arm_source_maps.append(record["source_hashes"])

    baseline = validate_baseline_checkpoint(
        base.BASELINE_CHECKPOINT,
        expected_sha256=base.BASELINE_CHECKPOINT_SHA256,
        hash_file=hash_file,
    )
    endpoints = build_endpoint_map(baseline, validated_arms)
    freeze_hashes = (eval_hashes or evaluation_source_hashes)()
    all_source_hashes = _merge_hash_maps(*arm_source_maps, freeze_hashes)
    for raw_path, expected in all_source_hashes.items():
        source_path = Path(raw_path)
        if not source_path.is_file() or hash_file(source_path) != expected:
            raise Refuse(f"evaluation source changed during metadata preflight: {raw_path}")

    return {
        "validated_arms": validated_arms,
        "report_hashes": report_hashes,
        "baseline": baseline,
        "endpoints": endpoints,
        "source_hashes": all_source_hashes,
    }


def _readiness_payload(preflight: Mapping[str, Any]) -> dict[str, Any]:
    endpoints = [
        {
            "endpoint_id": endpoint["endpoint_id"],
            "path": str(endpoint["path"]),
            "sha256": endpoint["sha256"],
            "checkpoint_digest": endpoint["checkpoint_digest"],
            "source": endpoint["source"],
        }
        for endpoint in preflight["endpoints"]
    ]
    return {
        "schema": "taiji-r2-dialogue-recovery-v2-paired-evaluation-readiness-v1",
        "status": "ready",
        "run": RUN_ID,
        "endpoint_order": list(ENDPOINT_ORDER),
        "endpoints": endpoints,
        "input_report_hashes": preflight["report_hashes"],
        "baseline_checkpoint_sha256": preflight["baseline"]["sha256"],
        "source_hashes": preflight["source_hashes"],
        "input_families": FACE_SIZE,
        "expected_generation_outputs": expected_generation_count(),
        "parameter_updates": 0,
        "optimizer_updates": 0,
        "generation_queries": 0,
        "dev_queries": 0,
        "reference_answer_scoring_queries": 0,
        "exact_match_scoring_queries": 0,
        "can_promote": False,
    }


def _write_or_verify_readiness(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        _write_new_json(path, payload)
        return dict(payload)
    existing = read_json(path)
    if existing != dict(payload):
        raise Refuse("existing evaluation readiness differs from current frozen metadata")
    return existing


def validate_only(
    *,
    paths: Mapping[str, Path] | None = None,
    preflight_fn: Callable[[], dict[str, Any]] = metadata_preflight,
) -> dict[str, Any]:
    """Create or re-verify metadata readiness without model or prompt access."""
    output = dict(paths or _evaluation_paths())
    preflight = preflight_fn()
    readiness = _readiness_payload(preflight)
    return _write_or_verify_readiness(Path(output["readiness"]), readiness)


def _restore_cpu_reserve(remaining_endpoints: int) -> float:
    return remaining_endpoints * RESTORE_RESERVE_CPU_SECONDS


def _restore_wall_reserve(remaining_endpoints: int) -> float:
    return remaining_endpoints * RESTORE_RESERVE_WALL_SECONDS


def _generation_cpu_reserve(remaining_queries: int) -> float:
    return remaining_queries * GENERATION_RESERVE_CPU_SECONDS


def _generation_wall_reserve(remaining_queries: int) -> float:
    return remaining_queries * GENERATION_RESERVE_WALL_SECONDS


def _whole_run_reserve(
    *, endpoints: int = len(ENDPOINT_ORDER), queries: int | None = None
) -> tuple[float, float]:
    query_count = expected_generation_count() if queries is None else queries
    return (
        _restore_cpu_reserve(endpoints)
        + _generation_cpu_reserve(query_count)
        + FACE_PREPARE_RESERVE_CPU_SECONDS
        + FINALIZATION_RESERVE_CPU_SECONDS,
        _restore_wall_reserve(endpoints)
        + _generation_wall_reserve(query_count)
        + FACE_PREPARE_RESERVE_WALL_SECONDS
        + FINALIZATION_RESERVE_WALL_SECONDS,
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _json_size_upper_bound(value: Any, limit: int) -> int:
    """Bound an IPC JSON frame without constructing a potentially huge encoding."""
    if value is None:
        return 4
    if value is True:
        return 4
    if value is False:
        return 5
    if isinstance(value, str):
        if len(value) > limit:
            return limit + 1
        encoded_length = len(value.encode("utf-8"))
        extra = 2
        for char in value:
            if char in {'"', "\\"}:
                extra += 1
            elif ord(char) < 0x20:
                extra += 1 if char in "\b\f\n\r\t" else 5
            if encoded_length + extra > limit:
                return limit + 1
        return encoded_length + extra
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise Refuse("worker IPC numeric values must be finite")
        return len(str(value))
    if isinstance(value, Mapping):
        total = 2
        for key, item in value.items():
            if not isinstance(key, str):
                raise Refuse("worker IPC object keys must be strings")
            total += _json_size_upper_bound(key, limit) + 1
            total += _json_size_upper_bound(item, limit) + 1
            if total > limit:
                return limit + 1
        return max(2, total - 1)
    if isinstance(value, (list, tuple)):
        total = 2
        for item in value:
            total += _json_size_upper_bound(item, limit) + 1
            if total > limit:
                return limit + 1
        return max(2, total - 1)
    raise Refuse(f"worker IPC value is not JSON-compatible: {type(value).__name__}")


def _append_durable_jsonl(path: Path, payload: Mapping[str, Any], durability_api: Any) -> None:
    _append_jsonl(path, payload, flush_file=durability_api.flush_file)


def _task_limits(
    *,
    budget: EvaluationBudget,
    operation: str,
    remaining_tasks: int,
    future_cpu_reserve: float,
    future_wall_reserve: float,
) -> tuple[float, float]:
    if remaining_tasks < 1 or operation not in {"restore", "generate"}:
        raise Refuse("worker deadline request is invalid")
    cap_cpu = (
        MAX_SINGLE_RESTORE_CPU_SECONDS
        if operation == "restore"
        else MAX_SINGLE_GENERATION_CPU_SECONDS
    )
    cap_wall = (
        MAX_SINGLE_RESTORE_WALL_SECONDS
        if operation == "restore"
        else MAX_SINGLE_GENERATION_WALL_SECONDS
    )
    global_cpu_share = (
        EVALUATION_CPU_LIMIT_SECONDS
        - budget.cpu_seconds()
        - future_cpu_reserve
        - SUPERVISOR_CANCEL_RESERVE_SECONDS
    ) / remaining_tasks
    global_wall_share = (
        EVALUATION_WALL_LIMIT_SECONDS
        - budget.wall_seconds()
        - future_wall_reserve
        - SUPERVISOR_CANCEL_RESERVE_SECONDS
    ) / remaining_tasks
    cpu_limit = min(cap_cpu, global_cpu_share)
    wall_limit = min(cap_wall, global_wall_share)
    if (
        not math.isfinite(cpu_limit)
        or not math.isfinite(wall_limit)
        or cpu_limit <= 0
        or wall_limit <= 0
    ):
        raise TimeoutError(f"no finite global budget remains for {operation} task")
    return cpu_limit, wall_limit


def _load_endpoint_trainer(
    endpoint: Mapping[str, Any], assistant_corpus: Any
) -> tuple[LanguageAlignmentTrainer, str]:
    checkpoint = Path(endpoint["path"])
    if sha256_file(checkpoint) != endpoint["sha256"]:
        raise Refuse(f"endpoint checkpoint changed before restore: {endpoint['endpoint_id']}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    endpoint_id = str(endpoint["endpoint_id"])
    if endpoint_id == "parent_baseline":
        payload_digest = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if (
            payload.get("checkpoint_digest") != payload_digest
            or payload.get("checkpoint_digest") != endpoint["checkpoint_digest"]
        ):
            raise Refuse("parent baseline checkpoint payload digest differs")
        model = Taiji.from_checkpoint(payload["model"], device="cpu")
        config = replace(
            base.LanguageAlignmentConfig(**dict(payload["config"])),
            response_plan_target_geometry="signed_hash_span",
        )
        trainer = LanguageAlignmentTrainer(
            model,
            assistant_corpus,
            config=config,
            code_revision=base._fresh_code_revision(),
        )
        restored_digest = str(payload["checkpoint_digest"])
    else:
        trainer = LanguageAlignmentTrainer.from_checkpoint(payload, assistant_corpus)
        restored_digest = str(trainer.checkpoint().get("checkpoint_digest", ""))
        if restored_digest != endpoint["checkpoint_digest"]:
            raise Refuse(f"{endpoint_id} assistant final restore digest differs")
    if not restored_digest:
        raise Refuse(f"{endpoint_id} endpoint has no checkpoint digest")
    return trainer, content_digest(trainer.model.checkpoint())


def _prompt_row_sha256(row: Mapping[str, Any]) -> str:
    expected_keys = {
        "index",
        "episode_id",
        "family_id",
        "split",
        "system",
        "context",
        "history_turn_count",
        "history",
        "current_user_turn",
        "unknown_policy",
    }
    if set(row) != expected_keys or row.get("split") != "dev":
        raise Refuse("worker prompt payload is not the frozen prompt-only schema")
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _read_worker_prompt_row(
    request: Mapping[str, Any],
    *,
    attempt_id: str,
    expected_prompt_sha256: str,
    ledger_path: Path,
    expected_face_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve one query from the durable face, never from a large IPC body."""
    expected_face_path = Path(expected_face_path or _evaluation_paths()["face"]).resolve()
    raw_face_path = request.get("face_path")
    if not isinstance(raw_face_path, str) or Path(raw_face_path).resolve() != expected_face_path:
        raise Refuse("worker query must reference the fixed persisted prompt face")
    face = read_json(expected_face_path)
    face_events = [
        event
        for event in _read_ledger_events(ledger_path)
        if event.get("event") == "prompt_face_persisted"
        and event.get("attempt_id") == attempt_id
        and event.get("face_sha256") == face.get("face_sha256")
    ]
    rows, hashes = face.get("rows"), face.get("prompt_hashes")
    if len(face_events) != 1 or not isinstance(rows, list):
        raise Refuse("worker query face lacks one unique durable parent publication event")
    if not isinstance(hashes, list) or len(hashes) != len(rows):
        raise Refuse("worker query face prompt hash map is invalid")
    index = request.get("index")
    if type(index) is not int or not 1 <= index <= len(rows):
        raise Refuse("worker query family index is outside the persisted prompt face")
    row = dict(_require_mapping(rows[index - 1], "persisted worker prompt row"))
    prompt_sha = _prompt_row_sha256(row)
    if (
        prompt_sha != request.get("prompt_sha256")
        or prompt_sha != hashes[index - 1]
        or prompt_sha != expected_prompt_sha256
        or request.get("face_sha256") != face.get("face_sha256")
    ):
        raise Refuse("worker prompt SHA differs from the durable query intent or published face")
    return row


def _current_process_is_in_job(job_name: str) -> bool:
    if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.OpenJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = kernel32.OpenJobObjectW(0x0004, False, job_name)  # JOB_OBJECT_QUERY
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    belongs = wintypes.BOOL()
    try:
        if not kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), job, ctypes.byref(belongs)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(belongs.value)
    finally:
        kernel32.CloseHandle(job)


def _worker_handle_task(
    request: Mapping[str, Any],
    *,
    attempt_path: Path,
    ledger_path: Path,
    durability_api: Any,
    trainers: dict[str, LanguageAlignmentTrainer],
    endpoint_specs: dict[str, dict[str, Any]],
    corpus_holder: dict[str, Any],
    expected_attempt_path: Path | None = None,
    expected_ledger_path: Path | None = None,
    job_membership_check: Callable[[], bool] | None = None,
    job_name: str | None = None,
) -> dict[str, Any]:
    operation = request.get("operation")
    endpoint_id = request.get("endpoint_id")
    task_id = request.get("task_id")
    token = request.get("token")
    if not all(isinstance(value, str) and value for value in (operation, endpoint_id, task_id)):
        raise Refuse("worker IPC task header is invalid")
    membership_check = job_membership_check or (lambda: _current_process_is_in_job(job_name or ""))
    try:
        in_job = membership_check()
    except BaseException as exc:
        raise Refuse(f"worker Job Object membership could not be verified: {exc}") from exc
    if not in_job:
        raise Refuse("worker task is refused outside the parent-supervised Job Object")
    authorization = _authorize_worker_task(
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
        preflight = metadata_preflight()
        readiness = read_json(_evaluation_paths()["readiness"])
        attempt = authorization["attempt"]
        if (
            _readiness_payload(preflight) != readiness
            or attempt.get("readiness") != readiness
            or attempt.get("source_hashes") != preflight.get("source_hashes")
            or attempt.get("input_report_hashes") != preflight.get("report_hashes")
        ):
            raise Refuse("worker restore metadata differs from the parent readiness")
        endpoint_specs.clear()
        endpoint_specs.update({item["endpoint_id"]: dict(item) for item in preflight["endpoints"]})
        if corpus_holder.get("assistant") is None:
            corpus_holder["assistant"] = base._load_assistant_train_corpus()
        endpoint = endpoint_specs.get(endpoint_id)
        if endpoint is None:
            raise Refuse("restore endpoint is not in the fixed four-endpoint map")
        if authorization["intent"].get("checkpoint_sha256") != endpoint["sha256"]:
            raise Refuse("restore endpoint SHA differs from the durable parent intent")
        trainer, model_digest = _load_endpoint_trainer(endpoint, corpus_holder["assistant"])
        expected_episodes = (
            0 if endpoint_id == "parent_baseline" else EXPECTED_STAGE_COUNTS["assistant_sft"][0]
        )
        expected_updates = (
            0 if endpoint_id == "parent_baseline" else EXPECTED_STAGE_COUNTS["assistant_sft"][1]
        )
        if trainer.episode_count != expected_episodes or trainer.global_step != expected_updates:
            raise Refuse(f"{endpoint_id} restore endpoint training count differs")
        if sha256_file(Path(endpoint["path"])) != endpoint["sha256"]:
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
    prompt_row = _read_worker_prompt_row(
        request,
        attempt_id=authorization["attempt"]["attempt_id"],
        expected_prompt_sha256=str(authorization["intent"].get("prompt_sha256", "")),
        ledger_path=ledger_path,
    )
    if request.get("index") != authorization["intent"].get("index"):
        raise Refuse("worker prompt SHA differs from the durable query intent")
    endpoint = endpoint_specs[endpoint_id]
    checkpoint_path = Path(endpoint["path"])
    checkpoint_before = sha256_file(checkpoint_path)
    if checkpoint_before != endpoint["sha256"]:
        raise Refuse(f"{endpoint_id} checkpoint changed before generation")
    trainer = trainers[endpoint_id]
    model_before = content_digest(trainer.model.checkpoint())
    prompt = base.PromptOnlyEpisode(
        episode_id=str(prompt_row["episode_id"]),
        family_id=str(prompt_row["family_id"]),
        system=str(prompt_row["system"]),
        context=str(prompt_row["context"]),
        history=tuple((str(user), str(assistant)) for user, assistant in prompt_row["history"]),
        user_input=str(prompt_row["current_user_turn"]),
        unknown_policy=str(prompt_row["unknown_policy"]),
    )
    output = _validate_generation_output(base._generate_readonly(trainer, prompt))
    model_after = content_digest(trainer.model.checkpoint())
    checkpoint_after = sha256_file(checkpoint_path)
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


def _worker_loop(attempt_path: Path, ledger_path: Path, job_name: str) -> int:
    """Private persistent worker. It has no selector and handles one IPC task at a time."""
    if os.name != "nt":
        raise Refuse("DURABILITY_UNSUPPORTED: restore/generation worker requires Windows NTFS")
    durability_api = WindowsDurabilityApi()
    if not attempt_path.exists() or not ledger_path.exists():
        raise Refuse("worker requires an existing parent attempt and ledger")
    attempt = read_json(attempt_path)
    if not _current_process_is_in_job(job_name):
        raise Refuse("worker bootstrap is refused outside the parent-supervised Job Object")
    trainers: dict[str, LanguageAlignmentTrainer] = {}
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
                attempt = read_json(attempt_path)
                shutdown_events = [
                    event
                    for event in _read_ledger_events(ledger_path)
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
                            "worker_cpu_seconds_total": time.process_time() - PROCESS_START_CPU,
                            "worker_wall_seconds_total": time.monotonic() - PROCESS_START_WALL,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return 0
            response = _worker_handle_task(
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


class WorkerTaskFailure(Refuse):
    def __init__(
        self,
        message: str,
        *,
        observed_child_cpu_seconds: float = 0.0,
        timed_out: bool = False,
        worker_tree_stopped: bool = True,
        cpu_measurement_valid: bool = True,
        ipc_write_thread_stopped: bool = True,
    ) -> None:
        super().__init__(message)
        self.observed_child_cpu_seconds = observed_child_cpu_seconds
        self.timed_out = timed_out
        self.worker_tree_stopped = worker_tree_stopped
        self.cpu_measurement_valid = cpu_measurement_valid
        self.ipc_write_thread_stopped = ipc_write_thread_stopped


class _JobBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JobIoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _JobExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimitInformation),
        ("IoInfo", _JobIoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _ThreadEntry32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


def _resume_suspended_primary_thread(process_id: int) -> None:
    if os.name != "nt":
        raise Refuse("worker resume requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32)]
    kernel32.Thread32First.restype = wintypes.BOOL
    kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32)]
    kernel32.Thread32Next.restype = wintypes.BOOL
    kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or snapshot == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel32.Thread32First(snapshot, ctypes.byref(entry)):
            raise ctypes.WinError(ctypes.get_last_error())
        thread_ids: list[int] = []
        while True:
            if int(entry.th32OwnerProcessID) == int(process_id):
                thread_ids.append(int(entry.th32ThreadID))
            if not kernel32.Thread32Next(snapshot, ctypes.byref(entry)):
                break
        if len(thread_ids) != 1:
            raise Refuse(
                f"suspended worker must have exactly one primary thread, found {len(thread_ids)}"
            )
    finally:
        if not kernel32.CloseHandle(snapshot):
            raise ctypes.WinError(ctypes.get_last_error())
    thread = kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, thread_ids[0])
    if not thread:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        previous_suspend_count = int(kernel32.ResumeThread(thread))
        if previous_suspend_count == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        if previous_suspend_count < 1:
            raise Refuse("worker primary thread was not suspended at Job assignment")
    finally:
        if not kernel32.CloseHandle(thread):
            raise ctypes.WinError(ctypes.get_last_error())


def _cancel_synchronous_thread_io(thread_id: int) -> None:
    """Cancel a dedicated writer thread's pending synchronous pipe I/O."""
    if os.name != "nt":
        raise Refuse("IPC writer cancellation requires Windows")
    if type(thread_id) is not int or thread_id <= 0:
        raise Refuse("IPC writer has no Windows thread identity")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.CancelSynchronousIo.argtypes = [wintypes.HANDLE]
    kernel32.CancelSynchronousIo.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    thread = kernel32.OpenThread(THREAD_TERMINATE, False, thread_id)
    if not thread:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel32.CancelSynchronousIo(thread):
            error = ctypes.get_last_error()
            if error != 1168:  # ERROR_NOT_FOUND: the writer may have just completed.
                raise ctypes.WinError(error)
    finally:
        if not kernel32.CloseHandle(thread):
            raise ctypes.WinError(ctypes.get_last_error())


def _worker_command(attempt_path: Path, ledger_path: Path, job_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2",
        "--worker",
        "--attempt-path",
        str(Path(attempt_path).resolve()),
        "--ledger-path",
        str(Path(ledger_path).resolve()),
        "--job-name",
        job_name,
    ]


class WindowsJobObject:
    """Windows job that accounts for descendant CPU and kills the whole tree."""

    def __init__(self, process_handle: Any, *, name: str, api: Any | None = None) -> None:
        if os.name != "nt":
            raise Refuse("JOB_OBJECT_UNAVAILABLE: worker supervision requires Windows")
        self.kernel32 = api or ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self.kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel32.TerminateJobObject.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel32.CreateJobObjectW(None, name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = _JobExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.kernel32.SetInformationJobObject(
                self.handle,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel32.AssignProcessToJobObject(self.handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def accounting(self) -> Any:
        info = _JobBasicAccountingInformation()
        returned = wintypes.DWORD()
        if not self.kernel32.QueryInformationJobObject(
            self.handle,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(returned),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return info

    def cpu_seconds(self) -> float:
        info = self.accounting()
        return (int(info.TotalUserTime) + int(info.TotalKernelTime)) / 10_000_000.0

    def terminate_and_confirm(self, *, timeout_seconds: float) -> bool:
        if self.accounting().ActiveProcesses == 0:
            return True
        if self.handle and not self.kernel32.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.accounting().ActiveProcesses == 0:
                return True
            time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
        return self.accounting().ActiveProcesses == 0

    def close(self) -> None:
        if getattr(self, "handle", None):
            if not self.kernel32.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None

    def __del__(self) -> None:
        # 终结器里的清理失败必须吞掉（解释器关闭期什么都可能失败），
        # contextlib.suppress 与原 try/except-pass 语义等价。
        with contextlib.suppress(BaseException):
            self.close()


class SupervisedWorker:
    """One long-lived, parent-supervised restore/generation process tree."""

    def __init__(
        self,
        attempt_path: Path,
        ledger_path: Path,
        *,
        command: Sequence[str] | None = None,
        cwd: Path = ROOT,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        process_api: Any = psutil,
        job_factory: Callable[..., WindowsJobObject] = WindowsJobObject,
        resume_fn: Callable[[int], None] = _resume_suspended_primary_thread,
        cancel_synchronous_io_fn: Callable[[int], Any] = _cancel_synchronous_thread_io,
        budget: EvaluationBudget | None = None,
        startup_cpu_limit_seconds: float = MAX_SINGLE_RESTORE_CPU_SECONDS,
        startup_wall_limit_seconds: float = MAX_SINGLE_RESTORE_WALL_SECONDS,
        startup_future_cpu_reserve_seconds: float = 0.0,
        startup_future_wall_reserve_seconds: float = 0.0,
        poll_interval_seconds: float = SUPERVISOR_POLL_INTERVAL_SECONDS,
    ) -> None:
        self.attempt_path = Path(attempt_path).resolve()
        self.ledger_path = Path(ledger_path).resolve()
        self.popen_factory = popen_factory
        self.process_api = process_api
        self.resume_fn = resume_fn
        self.cancel_synchronous_io_fn = cancel_synchronous_io_fn
        self._closed = False
        self._active_ipc_writer: threading.Thread | None = None
        self._ipc_writer_cancel_error: str | None = None
        self._cpu_by_pid: dict[int, float] = {}
        self._last_worker_cpu_total = 0.0
        self._final_job_cpu_total: float | None = None
        if os.name != "nt":
            raise Refuse("JOB_OBJECT_UNAVAILABLE: supervised workers require Windows")
        self.poll_interval_seconds = poll_interval_seconds
        self.job_name = f"Local\\TaijiR2Eval_{secrets.token_hex(16)}"
        self.command = (
            [str(part).replace("{job_name}", self.job_name) for part in command]
            if command is not None
            else _worker_command(self.attempt_path, self.ledger_path, self.job_name)
        )
        self._bootstrap_ready_required = command is None or "--worker" in self.command
        if self._bootstrap_ready_required and Path(cwd).resolve() != ROOT.resolve():
            raise Refuse("private worker must start as a module from the repository root")
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            TAIJI_R2_EVALUATION_JOB_NAME=self.job_name,
        )
        self.process = popen_factory(
            self.command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            creationflags=CREATE_SUSPENDED,
        )
        try:
            self.process_info = process_api.Process(self.process.pid)
            startup_times = self.process_info.cpu_times()
            self.initial_child_cpu_seconds = max(
                0.0, float(startup_times.user + startup_times.system)
            )
        except BaseException as exc:
            try:
                self.process.kill()
                self.process.wait(timeout=SUPERVISOR_TERMINATE_GRACE_SECONDS)
            except BaseException:
                pass
            raise WorkerTaskFailure(
                f"worker startup CPU could not be measured before Job assignment: {exc}",
                worker_tree_stopped=self.process.poll() is not None,
                cpu_measurement_valid=False,
            ) from exc
        resume_attempted = False
        try:
            self.job = job_factory(self.process._handle, name=self.job_name)
            accounting = self.job.accounting()
            if int(accounting.ActiveProcesses) < 1 or int(accounting.TotalProcesses) < 1:
                raise Refuse("Job Object assignment did not report the worker process")
            self.job_active_processes_at_start = int(accounting.ActiveProcesses)
            self._job_cpu_baseline = float(self.job.cpu_seconds())
            resume_attempted = True
            self.resume_fn(self.process.pid)
            accounting = self.job.accounting()
            if int(accounting.ActiveProcesses) < 1:
                raise Refuse("resumed worker is no longer active in its Job Object")
        except BaseException as exc:
            tree_stopped = False
            try:
                if hasattr(self, "job"):
                    tree_stopped = bool(
                        self.job.terminate_and_confirm(
                            timeout_seconds=SUPERVISOR_TERMINATE_GRACE_SECONDS
                        )
                    )
                    tree_stopped = tree_stopped and int(self.job.accounting().ActiveProcesses) == 0
                    self.job.close()
                self.process.kill()
                self.process.wait(timeout=SUPERVISOR_TERMINATE_GRACE_SECONDS)
                if not resume_attempted:
                    tree_stopped = tree_stopped or self.process.poll() is not None
            except BaseException:
                tree_stopped = False
            raise WorkerTaskFailure(
                f"JOB_OBJECT_UNAVAILABLE: worker cannot receive authorization: {exc}",
                observed_child_cpu_seconds=self.initial_child_cpu_seconds,
                worker_tree_stopped=tree_stopped,
                cpu_measurement_valid=True,
            ) from exc
        self.responses: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        self.sample_tree_cpu_seconds()
        if self._bootstrap_ready_required:
            self._await_ready(
                budget=budget or EvaluationBudget(),
                startup_cpu_limit_seconds=startup_cpu_limit_seconds,
                startup_wall_limit_seconds=startup_wall_limit_seconds,
                future_cpu_reserve_seconds=startup_future_cpu_reserve_seconds,
                future_wall_reserve_seconds=startup_future_wall_reserve_seconds,
            )

    def _read_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            self.responses.put(None)
            return
        try:
            for line in stream:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    @property
    def ipc_write_thread_stopped(self) -> bool:
        writer = self._active_ipc_writer
        return writer is None or not writer.is_alive()

    def _cancel_and_join_ipc_writer(self) -> bool:
        writer = self._active_ipc_writer
        if writer is None:
            return True
        if writer.is_alive():
            thread_id = writer.native_id
            if thread_id is not None:
                try:
                    self.cancel_synchronous_io_fn(thread_id)
                except BaseException as exc:
                    self._ipc_writer_cancel_error = f"{type(exc).__name__}: {exc}"
            writer.join(timeout=SUPERVISOR_TERMINATE_GRACE_SECONDS)
        stopped = not writer.is_alive()
        if stopped:
            self._active_ipc_writer = None
        return stopped

    def _await_ready(
        self,
        *,
        budget: EvaluationBudget,
        startup_cpu_limit_seconds: float,
        startup_wall_limit_seconds: float,
        future_cpu_reserve_seconds: float,
        future_wall_reserve_seconds: float,
    ) -> None:
        if (
            not math.isfinite(startup_cpu_limit_seconds)
            or not math.isfinite(startup_wall_limit_seconds)
            or startup_cpu_limit_seconds <= 0
            or startup_wall_limit_seconds <= 0
        ):
            stopped = self.terminate_tree()
            raise WorkerTaskFailure(
                "worker startup has no finite deadline",
                worker_tree_stopped=stopped,
                timed_out=True,
            )
        expected_attempt_id = read_json(self.attempt_path).get("attempt_id")
        startup_child_before = self.initial_child_cpu_seconds
        start_wall = time.monotonic()
        line: str | None = None
        while line is None:
            child_total = self.sample_tree_cpu_seconds()
            child_delta = max(0.0, child_total - startup_child_before)
            wall_delta = time.monotonic() - start_wall
            if (
                child_delta > startup_cpu_limit_seconds
                or wall_delta > startup_wall_limit_seconds
                or budget.cpu_seconds() + child_delta + future_cpu_reserve_seconds
                > EVALUATION_CPU_LIMIT_SECONDS
                or budget.wall_seconds() + wall_delta + future_wall_reserve_seconds
                > EVALUATION_WALL_LIMIT_SECONDS
            ):
                stopped = self.terminate_tree()
                observed = max(0.0, self.sample_tree_cpu_seconds() - startup_child_before)
                raise WorkerTaskFailure(
                    f"worker bootstrap exceeded CPU/wall deadline (child={observed:.3f}, wall={wall_delta:.3f})",
                    observed_child_cpu_seconds=observed,
                    timed_out=True,
                    worker_tree_stopped=stopped,
                )
            try:
                line = self.responses.get(timeout=self.poll_interval_seconds)
            except queue.Empty as err:
                if self.process.poll() is not None:
                    line = None
                    stopped = self.terminate_tree()
                    observed = max(0.0, self.sample_tree_cpu_seconds() - startup_child_before)
                    raise WorkerTaskFailure(
                        "worker exited before authenticated bootstrap-ready handshake",
                        observed_child_cpu_seconds=observed,
                        worker_tree_stopped=stopped,
                    ) from err
        if line is None:
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - startup_child_before)
            raise WorkerTaskFailure(
                "worker stdout closed before bootstrap-ready handshake",
                observed_child_cpu_seconds=observed,
                worker_tree_stopped=stopped,
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            stopped = self.terminate_tree()
            raise WorkerTaskFailure(
                "worker bootstrap-ready response is malformed",
                worker_tree_stopped=stopped,
            ) from exc
        if (
            response.get("status") != "ready"
            or response.get("operation") != "ready"
            or response.get("task_id") != "worker_ready"
            or response.get("endpoint_id") != "worker"
            or response.get("attempt_id") != expected_attempt_id
            or response.get("job_name") != self.job_name
        ):
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - startup_child_before)
            raise WorkerTaskFailure(
                "worker bootstrap-ready response identity differs from the assigned Job Object",
                observed_child_cpu_seconds=observed,
                worker_tree_stopped=stopped,
            )
        ready_child_total = self.sample_tree_cpu_seconds()
        self.initial_child_cpu_seconds = ready_child_total
        self._job_cpu_baseline = float(self.job.cpu_seconds())
        self._last_worker_cpu_total = ready_child_total

    def sample_tree_cpu_seconds(self) -> float:
        if self._final_job_cpu_total is not None:
            return self._final_job_cpu_total
        try:
            job_cpu = float(self.job.cpu_seconds())
            return self.initial_child_cpu_seconds + max(0.0, job_cpu - self._job_cpu_baseline)
        except BaseException as exc:
            stopped = self.terminate_tree()
            raise WorkerTaskFailure(
                f"Windows Job Object CPU sampling failed: {exc}",
                worker_tree_stopped=stopped,
                cpu_measurement_valid=False,
                ipc_write_thread_stopped=self.ipc_write_thread_stopped,
            ) from exc

    def request(
        self,
        payload: Mapping[str, Any],
        *,
        budget: EvaluationBudget,
        task_cpu_limit_seconds: float,
        task_wall_limit_seconds: float,
        future_cpu_reserve_seconds: float,
        future_wall_reserve_seconds: float,
        cpu_now: Callable[[], float] = time.process_time,
        wall_now: Callable[[], float] = time.monotonic,
    ) -> tuple[dict[str, Any], float, float]:
        if self._closed:
            raise WorkerTaskFailure("supervised worker is not alive", worker_tree_stopped=True)
        if self.process.poll() is not None:
            try:
                stopped = self.job.accounting().ActiveProcesses == 0
            except BaseException:
                stopped = False
            if not stopped:
                stopped = self.terminate_tree()
            raise WorkerTaskFailure(
                "supervised worker exited before receiving the task",
                worker_tree_stopped=bool(stopped),
            )
        if not math.isfinite(task_cpu_limit_seconds) or task_cpu_limit_seconds <= 0:
            raise WorkerTaskFailure("worker task has no finite CPU budget", timed_out=True)
        if not math.isfinite(task_wall_limit_seconds) or task_wall_limit_seconds <= 0:
            raise WorkerTaskFailure("worker task has no finite wall deadline", timed_out=True)
        if self.process.stdin is None:
            raise WorkerTaskFailure("supervised worker stdin is unavailable")
        task_id = str(payload.get("task_id", "unknown"))
        operation = payload.get("operation")
        start_parent_cpu = cpu_now()
        start_wall = wall_now()
        child_total_before = self.sample_tree_cpu_seconds()
        try:
            frame_size_upper = _json_size_upper_bound(payload, MAX_WORKER_IPC_FRAME_BYTES)
        except BaseException as exc:
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - child_total_before)
            raise WorkerTaskFailure(
                f"worker IPC frame is invalid for {task_id}: {exc}",
                observed_child_cpu_seconds=observed,
                timed_out=True,
                worker_tree_stopped=stopped,
            ) from exc
        if frame_size_upper + 1 > MAX_WORKER_IPC_FRAME_BYTES:
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - child_total_before)
            raise WorkerTaskFailure(
                f"worker IPC frame exceeds {MAX_WORKER_IPC_FRAME_BYTES} bytes for {task_id}",
                observed_child_cpu_seconds=observed,
                timed_out=True,
                worker_tree_stopped=stopped,
            )
        request_line = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        )
        if len(request_line.encode("utf-8")) > MAX_WORKER_IPC_FRAME_BYTES:
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - child_total_before)
            raise WorkerTaskFailure(
                f"worker IPC frame exceeds {MAX_WORKER_IPC_FRAME_BYTES} bytes for {task_id}",
                observed_child_cpu_seconds=observed,
                timed_out=True,
                worker_tree_stopped=stopped,
            )
        write_done = threading.Event()
        write_errors: list[BaseException] = []

        def write_frame() -> None:
            try:
                self.process.stdin.write(request_line)
                self.process.stdin.flush()
            except BaseException as exc:
                write_errors.append(exc)
            finally:
                write_done.set()

        writer = threading.Thread(target=write_frame, name=f"r2-ipc-{task_id}", daemon=True)
        writer.start()
        self._active_ipc_writer = writer

        response_line: str | None = None
        child_total_after = child_total_before
        while response_line is None or not write_done.is_set():
            if write_done.is_set() and write_errors:
                break
            child_total_after = self.sample_tree_cpu_seconds()
            child_delta = max(0.0, child_total_after - child_total_before)
            parent_delta = max(0.0, cpu_now() - start_parent_cpu)
            wall_delta = max(0.0, wall_now() - start_wall)
            projected_cpu = budget.cpu_seconds() + child_delta
            if (
                child_delta > task_cpu_limit_seconds
                or wall_delta > task_wall_limit_seconds
                or projected_cpu + future_cpu_reserve_seconds > EVALUATION_CPU_LIMIT_SECONDS
                or budget.wall_seconds() + future_wall_reserve_seconds
                > EVALUATION_WALL_LIMIT_SECONDS
            ):
                stopped = self.terminate_tree()
                observed = max(0.0, self.sample_tree_cpu_seconds() - child_total_before)
                raise WorkerTaskFailure(
                    f"worker task {task_id} exceeded its CPU/wall deadline "
                    f"(child={observed:.3f}/{task_cpu_limit_seconds:.3f}, "
                    f"wall={wall_delta:.3f}/{task_wall_limit_seconds:.3f})",
                    observed_child_cpu_seconds=observed,
                    timed_out=True,
                    worker_tree_stopped=stopped,
                    ipc_write_thread_stopped=self.ipc_write_thread_stopped,
                )
            if response_line is None:
                try:
                    response_line = self.responses.get(timeout=self.poll_interval_seconds)
                except queue.Empty:
                    if self.process.poll() is not None:
                        response_line = None
                        break
            else:
                write_done.wait(timeout=self.poll_interval_seconds)
        if write_errors:
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - child_total_before)
            raise WorkerTaskFailure(
                f"worker IPC write failed for {task_id}: {write_errors[0]}",
                observed_child_cpu_seconds=observed,
                worker_tree_stopped=stopped,
                ipc_write_thread_stopped=self.ipc_write_thread_stopped,
            )
        if response_line is None:
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - child_total_before)
            raise WorkerTaskFailure(
                f"worker exited before responding to {task_id}",
                observed_child_cpu_seconds=observed,
                worker_tree_stopped=stopped,
                ipc_write_thread_stopped=self.ipc_write_thread_stopped,
            )
        writer.join(timeout=min(0.05, self.poll_interval_seconds * 2))
        if writer.is_alive():
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - child_total_before)
            raise WorkerTaskFailure(
                f"worker IPC writer did not stop after sending {task_id}",
                observed_child_cpu_seconds=observed,
                worker_tree_stopped=stopped,
                ipc_write_thread_stopped=self.ipc_write_thread_stopped,
            )
        self._active_ipc_writer = None
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            stopped = self.terminate_tree()
            observed = max(0.0, self.sample_tree_cpu_seconds() - child_total_before)
            raise WorkerTaskFailure(
                f"worker returned malformed JSON for {task_id}",
                observed_child_cpu_seconds=observed,
                worker_tree_stopped=stopped,
            ) from exc
        child_total_after = self.sample_tree_cpu_seconds()
        observed_child = max(0.0, child_total_after - child_total_before)
        child_delta = observed_child
        self._last_worker_cpu_total = child_total_after
        parent_delta = max(0.0, cpu_now() - start_parent_cpu)
        wall_delta = max(0.0, wall_now() - start_wall)
        if operation == "shutdown":
            expected_task_id, expected_endpoint = "shutdown", "worker"
        else:
            expected_task_id, expected_endpoint = payload.get("task_id"), payload.get("endpoint_id")
        if (
            response.get("task_id") != expected_task_id
            or response.get("endpoint_id") != expected_endpoint
        ):
            stopped = self.terminate_tree()
            raise WorkerTaskFailure(
                f"worker response identity mismatch for {task_id}",
                observed_child_cpu_seconds=child_delta,
                worker_tree_stopped=stopped,
            )
        if response.get("status") != "ok":
            try:
                tree_stopped = (
                    self.process.poll() is not None and self.job.accounting().ActiveProcesses == 0
                )
            except BaseException:
                tree_stopped = False
            if not tree_stopped:
                tree_stopped = self.terminate_tree()
            raise WorkerTaskFailure(
                f"worker task {task_id} failed: {response.get('error_type')}: {response.get('error')}",
                observed_child_cpu_seconds=child_delta,
                worker_tree_stopped=tree_stopped,
            )
        if child_delta > task_cpu_limit_seconds or wall_delta > task_wall_limit_seconds:
            stopped = self.terminate_tree()
            raise WorkerTaskFailure(
                f"worker task {task_id} crossed its deadline before response was observed",
                observed_child_cpu_seconds=child_delta,
                timed_out=True,
                worker_tree_stopped=stopped,
            )
        return response, child_delta, parent_delta

    def terminate_tree(self) -> bool:
        if self._closed:
            self._cancel_and_join_ipc_writer()
            return True
        known_pids = {self.process.pid, *self._cpu_by_pid.keys()}
        try:
            stopped = self.job.terminate_and_confirm(
                timeout_seconds=SUPERVISOR_TERMINATE_GRACE_SECONDS
            )
            self.process.wait(timeout=max(0.05, SUPERVISOR_TERMINATE_GRACE_SECONDS))
        except subprocess.TimeoutExpired:
            self._cancel_and_join_ipc_writer()
            return False
        except BaseException:
            try:
                self._final_job_cpu_total = self.initial_child_cpu_seconds + max(
                    0.0, float(self.job.cpu_seconds()) - self._job_cpu_baseline
                )
            except BaseException:
                self._final_job_cpu_total = self.initial_child_cpu_seconds
            try:
                # Last-resort tree termination still relies on the Job Object's
                # KILL_ON_JOB_CLOSE contract; it is never treated as confirmed.
                self.job.close()
                self.process.wait(timeout=max(0.05, SUPERVISOR_TERMINATE_GRACE_SECONDS))
            except BaseException:
                pass
            self._cancel_and_join_ipc_writer()
            return False
        try:
            self._final_job_cpu_total = self.initial_child_cpu_seconds + max(
                0.0, float(self.job.cpu_seconds()) - self._job_cpu_baseline
            )
            active = int(self.job.accounting().ActiveProcesses)
            survivors = [pid for pid in known_pids if self.process_api.pid_exists(pid)]
        except BaseException:
            return False
        self._closed = stopped and active == 0 and not survivors and self.process.poll() is not None
        if self._closed:
            self.job.close()
        self._cancel_and_join_ipc_writer()
        return self._closed

    def shutdown(
        self, *, budget: EvaluationBudget, cpu_limit_seconds: float, wall_limit_seconds: float
    ) -> tuple[float, float]:
        if self._closed:
            return 0.0, 0.0
        try:
            response, child_cpu, parent_cpu = self.request(
                {
                    "operation": "shutdown",
                    "task_id": "shutdown",
                    "endpoint_id": "worker",
                    "attempt_id": read_json(self.attempt_path)["attempt_id"],
                },
                budget=budget,
                task_cpu_limit_seconds=cpu_limit_seconds,
                task_wall_limit_seconds=wall_limit_seconds,
                future_cpu_reserve_seconds=0.0,
                future_wall_reserve_seconds=0.0,
            )
            if response.get("operation") != "shutdown":
                raise Refuse("worker shutdown receipt is invalid")
            self.process.wait(timeout=max(0.05, wall_limit_seconds))
            final_child_total = self.sample_tree_cpu_seconds()
            final_child_delta = max(0.0, final_child_total - self._last_worker_cpu_total)
            self._last_worker_cpu_total = final_child_total
            child_cpu += final_child_delta
            if self.process.poll() is None or self.job.accounting().ActiveProcesses != 0:
                raise Refuse("worker process tree did not fully exit after shutdown")
            self._closed = True
            self.job.close()
            return child_cpu, parent_cpu
        except BaseException as error:
            stopped = self.terminate_tree()
            if not stopped:
                # 链上原始异常（Refuse 之后不再执行 bare raise，断了链就丢根因）。
                raise Refuse("supervised worker tree could not be proven stopped") from error
            raise


def _validate_restore_receipt(endpoint: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    endpoint_id = str(endpoint["endpoint_id"])
    is_baseline = endpoint_id == "parent_baseline"
    expected_stage = "parent_baseline" if is_baseline else "assistant_sft"
    expected_episodes = 0 if is_baseline else EXPECTED_STAGE_COUNTS["assistant_sft"][0]
    expected_updates = 0 if is_baseline else EXPECTED_STAGE_COUNTS["assistant_sft"][1]
    if (
        receipt.get("complete") is not True
        or receipt.get("equal") is not True
        or receipt.get("endpoint_id") != endpoint_id
        or receipt.get("stage") != expected_stage
        or Path(str(receipt.get("path", ""))).resolve() != Path(endpoint["path"]).resolve()
        or _require_sha(receipt.get("sha256"), f"{endpoint_id} restored checkpoint SHA")
        != endpoint["sha256"]
        or receipt.get("checkpoint_digest") != endpoint["checkpoint_digest"]
        or not isinstance(receipt.get("model_digest"), str)
        or not receipt.get("model_digest")
        or receipt.get("episode_count") != expected_episodes
        or receipt.get("global_step") != expected_updates
        or receipt.get("generation_queries") != 0
        or receipt.get("parameter_updates") != 0
        or receipt.get("optimizer_updates") != 0
        or receipt.get("can_promote") is not False
    ):
        raise Refuse(f"{endpoint_id} fresh-process restore receipt differs from its endpoint")
    cpu_seconds = receipt.get("cpu_seconds")
    if (
        not isinstance(cpu_seconds, (int, float))
        or not math.isfinite(float(cpu_seconds))
        or cpu_seconds < 0
    ):
        raise Refuse(f"{endpoint_id} fresh-process restore CPU receipt is invalid")


def _disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    if usage.free < MIN_OUTPUT_DISK_BYTES:
        raise Refuse("less than 512 MiB remains for complete paired-evaluation outputs")
    return {"free_bytes": usage.free, "required_free_bytes": MIN_OUTPUT_DISK_BYTES}


def _audit_unclosed_restore(
    ledger_path: Path,
    *,
    host_cpu_fn: Callable[[], float],
    boot_time_fn: Callable[[], float],
    wall_now: Callable[[], float] = time.monotonic,
    flush_file: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Durably mark interrupted restore, query and selector work as unknown tails."""
    if not ledger_path.is_file():
        return []
    try:
        events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Refuse(f"cannot audit existing restore ledger: {exc}") from exc
    started_tasks: dict[str, Mapping[str, Any]] = {}
    closed_tasks: set[str] = set()
    selector_started: Mapping[str, Any] | None = None
    selector_closed = False
    worker_startup_started: Mapping[str, Any] | None = None
    worker_startup_closed = False
    terminal_names = {
        "restore_completed",
        "restore_failed",
        "restore_timeout",
        "restore_tail_audited",
        "query_output_persisted",
        "query_failed",
        "query_timeout",
        "query_tail_audited",
    }
    for event in events:
        if not isinstance(event, Mapping):
            raise Refuse("existing evaluation ledger contains a non-object event")
        event_name = event.get("event")
        task_id = event.get("task_id")
        if event_name in {"restore_started", "query_started"}:
            key = str(task_id or f"legacy-{event_name}-{event.get('endpoint_id', 'unknown')}")
            started_tasks[key] = event
        elif event_name in terminal_names and isinstance(task_id, str):
            closed_tasks.add(task_id)
        elif event_name == "restore_tail_audited" and isinstance(event.get("endpoint_id"), str):
            closed_tasks.add(f"legacy-restore_started-{event['endpoint_id']}")
        elif event_name == "prompt_selection_started":
            selector_started = event
        elif event_name in {"prompt_face_persisted", "prompt_selection_tail_audited"}:
            selector_closed = True
        elif event_name == "worker_startup_started":
            worker_startup_started = event
        elif event_name in {
            "worker_supervision_ready",
            "worker_startup_failed",
            "worker_startup_tail_audited",
        }:
            worker_startup_closed = True
    open_tasks = [
        (task_id, event) for task_id, event in started_tasks.items() if task_id not in closed_tasks
    ]
    if (
        not open_tasks
        and (selector_started is None or selector_closed)
        and (worker_startup_started is None or worker_startup_closed)
    ):
        return []

    audited = []
    current_boot = boot_time_fn()
    current_host_cpu = host_cpu_fn()
    current_wall = wall_now()
    for task_id, event in open_tasks:
        operation = "restore" if event.get("event") == "restore_started" else "query"
        endpoint_id = event.get("endpoint_id")
        start_boot_value = event.get("boot_identity")
        logical_cores = max(1, int(event.get("logical_cpu_upper", 1)))
        if (
            isinstance(start_boot_value, (int, float))
            and isinstance(event.get("host_cpu_seconds_before"), (int, float))
            and current_boot == start_boot_value
        ):
            start_boot: float | None = float(start_boot_value)
            delta = max(0.0, current_host_cpu - float(event["host_cpu_seconds_before"]))
            accounting = "same_boot_host_cpu_delta_upper"
        elif isinstance(start_boot_value, (int, float)):
            start_boot = float(start_boot_value)
            timestamp = datetime.fromisoformat(str(event["timestamp_utc"]))
            wall_elapsed = max(0.0, (datetime.now(UTC) - timestamp).total_seconds())
            delta = wall_elapsed * logical_cores
            accounting = "reboot_wall_times_logical_cores_upper"
        else:
            start_boot = None
            delta = None
            accounting = "unmeasured_unknown_tail"
        audit = {
            "event": f"{operation}_tail_audited",
            "task_id": task_id,
            "operation": operation,
            "endpoint_id": endpoint_id,
            "started_event_timestamp_utc": event.get("timestamp_utc"),
            "boot_identity_before": start_boot,
            "boot_identity_at_audit": current_boot,
            "host_cpu_seconds_at_audit": current_host_cpu,
            "charged_child_cpu_upper": delta,
            "accounting": accounting,
            "unknown_tail": True,
            "no_resume_or_retry": True,
            "wall_clock_monotonic_at_audit": current_wall,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        _append_jsonl(ledger_path, audit, flush_file=flush_file)
        audited.append(audit)
    if selector_started is not None and not selector_closed:
        audit = {
            "event": "prompt_selection_tail_audited",
            "attempt_id": selector_started.get("attempt_id"),
            "unknown_tail": True,
            "prompt_face_persisted": False,
            "no_resume_or_retry": True,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        _append_jsonl(ledger_path, audit, flush_file=flush_file)
        audited.append(audit)
    if worker_startup_started is not None and not worker_startup_closed:
        audit = {
            "event": "worker_startup_tail_audited",
            "attempt_id": worker_startup_started.get("attempt_id"),
            "boot_identity_before": worker_startup_started.get("boot_identity"),
            "host_cpu_seconds_before": worker_startup_started.get("host_cpu_seconds_before"),
            "boot_identity_at_audit": current_boot,
            "host_cpu_seconds_at_audit": current_host_cpu,
            "charged_child_cpu_upper": (
                max(
                    0.0, current_host_cpu - float(worker_startup_started["host_cpu_seconds_before"])
                )
                if worker_startup_started.get("boot_identity") == current_boot
                and isinstance(worker_startup_started.get("host_cpu_seconds_before"), (int, float))
                else None
            ),
            "unknown_tail": True,
            "no_resume_or_retry": True,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        _append_jsonl(ledger_path, audit, flush_file=flush_file)
        audited.append(audit)
    return audited


def _review_markdown(rows: Sequence[Mapping[str, Any]], *, sample_count: int = 12) -> str:
    labels = {
        "parent_baseline": "R2 parent基线",
        "full_turn_1m": "1M全轮预训练候选",
        "warm_start": "300K热启动候选",
        "fresh_same_capacity": "300K同规模随机初始化对照",
    }
    lines = [
        "# R2 recovery v2 四端点自然对话样例",
        "",
        "同一39-family保留面，四端点逐题并列；不读取或比较reference答案。",
        "人工判断中文可读性、贴题、历史连贯和重复。格式/停止统计不替代语义审阅。",
        "",
    ]
    for row in rows[:sample_count]:
        lines.extend(
            (
                f"## 样本 {row['index']}（family={row['family_id']}，history={row['history_turn_count']}）",
                "",
            )
        )
        for turn, (user, assistant) in enumerate(row["history"], 1):
            lines.extend((f"历史用户{turn}：{user}", f"历史助手{turn}：{assistant}", ""))
        lines.extend((f"当前用户：{row['current_user_turn']}", ""))
        for endpoint_id in ENDPOINT_ORDER:
            output = row[endpoint_id]
            text = str(output["generated_text"]).replace("```", "` ` `")
            lines.extend(
                (
                    f"**{labels[endpoint_id]}**（{output['stop_reason']}，{output['generated_bytes_length']}字节）",
                    "",
                    "```text",
                    text,
                    "```",
                    "",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


def _validate_generation_output(output: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise Refuse("generation output must be an object")
    required = {
        "generated_text",
        "raw_generated_sha256",
        "generated_bytes_hex",
        "generated_bytes_length",
        "utf8_valid",
        "no_replacement",
        "response_boundary_present",
        "stop_reason",
        "pii_pattern_redacted",
        "repeated_4gram_rate",
    }
    if not required.issubset(output):
        raise Refuse("generation output is missing frozen audit fields")
    length = output.get("generated_bytes_length")
    if type(length) is not int or length < 0 or length > MAX_GENERATION_BYTES:
        raise Refuse("generation output exceeds the frozen 512-byte limit")
    _require_sha(output.get("raw_generated_sha256"), "raw generation SHA")
    if not isinstance(output.get("generated_text"), str):
        raise Refuse("generated_text must be a string")
    hex_bytes = output.get("generated_bytes_hex")
    if hex_bytes is not None:
        if not isinstance(hex_bytes, str) or len(hex_bytes) != length * 2:
            raise Refuse("generation hex bytes differ from the recorded output length")
        try:
            raw_bytes = bytes.fromhex(hex_bytes)
        except ValueError as exc:
            raise Refuse("generation hex bytes are invalid") from exc
        if hashlib.sha256(raw_bytes).hexdigest() != output["raw_generated_sha256"].lower():
            raise Refuse("generation raw SHA differs from the persisted bytes")
    elif output.get("pii_pattern_redacted") is not True:
        raise Refuse("raw bytes may only be omitted when PII redaction occurred")
    if not isinstance(output.get("stop_reason"), str) or not output["stop_reason"]:
        raise Refuse("generation stop reason is missing")
    return dict(output)


def _write_failure(
    paths: Mapping[str, Path], *, error: BaseException, details: Mapping[str, Any]
) -> None:
    failure = {
        "schema": "taiji-r2-dialogue-recovery-v2-paired-evaluation-failure-v1",
        "status": "failed",
        "complete": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "can_promote": False,
        "reference_answer_scoring_queries": 0,
        "exact_match_scoring_queries": 0,
        **dict(details),
    }
    with contextlib.suppress(FileExistsError):
        _write_new_json(Path(paths["failure"]), failure)
    failed_result = {
        "schema": "taiji-r2-dialogue-recovery-v2-paired-evaluation-result-v1",
        "status": "failed",
        "complete": False,
        "can_promote": False,
        "reference_answer_scoring_queries": 0,
        "exact_match_scoring_queries": 0,
        **dict(details),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with contextlib.suppress(FileExistsError):
        _write_new_json(Path(paths["result"]), failed_result)


def execute(
    *,
    paths: Mapping[str, Path] | None = None,
    preflight_fn: Callable[[], dict[str, Any]] | None = None,
    resources_fn: Callable[[], dict[str, Any]] | None = None,
    disk_fn: Callable[[Path], dict[str, Any]] | None = None,
    prompt_selector: Callable[[], Sequence[Any]] | None = None,
    hash_file: Callable[[Path], str] | None = None,
    budget_factory: Callable[..., EvaluationBudget] | None = None,
    host_cpu_fn: Callable[[], float] | None = None,
    boot_time_fn: Callable[[], float] | None = None,
    cpu_now: Callable[[], float] | None = None,
    wall_now: Callable[[], float] | None = None,
    worker_factory: Callable[..., Any] | None = None,
    durability_api: Any | None = None,
    durability_probe_fn: Callable[..., dict[str, Any]] | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Run the single-use evaluation under one durable, supervised worker."""
    output = dict(paths or _evaluation_paths())
    preflight_fn = preflight_fn or metadata_preflight
    resources_fn = resources_fn or resource_snapshot
    disk_fn = disk_fn or _disk_snapshot
    prompt_selector = prompt_selector or base._select_dev_prompts
    hash_file = hash_file or sha256_file
    budget_factory = budget_factory or EvaluationBudget
    host_cpu_fn = host_cpu_fn or recovery.host_cpu_seconds
    boot_time_fn = boot_time_fn or psutil.boot_time
    cpu_now = cpu_now or time.process_time
    wall_now = wall_now or time.monotonic
    durability_api = durability_api or WindowsDurabilityApi()
    durability_probe_fn = durability_probe_fn or probe_ntfs_durability
    worker_factory = worker_factory or SupervisedWorker
    required = {
        "readiness",
        "attempt",
        "ledger",
        "progress",
        "face",
        "generations",
        "result",
        "cost_receipt",
        "review",
        "failure",
    }
    if not required.issubset(output):
        raise Refuse("evaluation output path map is incomplete")
    attempt_path, ledger_path = Path(output["attempt"]), Path(output["ledger"])
    readiness_path, progress_path = Path(output["readiness"]), Path(output["progress"])
    if not readiness_path.is_file():
        raise Refuse("run --validate-only and save the evaluation readiness before --execute")
    if attempt_path.exists():
        _audit_unclosed_restore(
            ledger_path,
            host_cpu_fn=host_cpu_fn,
            boot_time_fn=boot_time_fn,
            wall_now=wall_now,
            flush_file=durability_api.flush_file,
        )
        raise Refuse(
            "paired evaluation attempt already exists and is consumed; resume/retry is forbidden"
        )
    for key in required - {"readiness"}:
        if Path(output[key]).exists():
            raise Refuse(f"paired evaluation is write-once; existing {key} blocks execution")
    if list(attempt_path.parent.glob(f"{attempt_path.name}.*.claim.json")):
        raise Refuse("a prior worker task claim exists; one-shot attempt is consumed")

    start_cpu, start_wall = cpu_now(), wall_now()
    preflight = preflight_fn()
    ready = _readiness_payload(preflight)
    if read_json(readiness_path) != ready:
        raise Refuse(
            "saved evaluation readiness no longer matches current source/result/checkpoint metadata"
        )
    resources = resources_fn()
    resources = validate_resource_snapshot(
        resources["cpu_percent_samples"], resources["available_memory_gib"]
    )
    disk = disk_fn(Path(output["result"]).parent)
    if int(disk.get("free_bytes", -1)) < MIN_OUTPUT_DISK_BYTES:
        raise Refuse("insufficient report disk space before attempt creation")
    preflight_cpu, preflight_wall = max(0.0, cpu_now() - start_cpu), max(
        0.0, wall_now() - start_wall
    )
    budget = budget_factory(
        charged_parent_cpu_seconds=preflight_cpu, charged_wall_seconds=preflight_wall
    )
    whole_cpu, whole_wall = _whole_run_reserve()
    budget.guard(
        reserve_cpu_seconds=whole_cpu + SUPERVISOR_CANCEL_RESERVE_SECONDS,
        reserve_wall_seconds=whole_wall + SUPERVISOR_CANCEL_RESERVE_SECONDS,
    )
    if attempt_path.parent.resolve() != ledger_path.parent.resolve():
        raise Refuse("attempt and initial ledger must share one pre-existing report directory")
    qualification = durability_probe_fn(
        attempt_path.parent, api=durability_api, platform_name=platform_name
    )
    endpoints = list(preflight["endpoints"])
    if tuple(row["endpoint_id"] for row in endpoints) != ENDPOINT_ORDER:
        raise Refuse("metadata preflight endpoint order differs from the frozen four-endpoint map")
    checkpoint_shas = {row["endpoint_id"]: hash_file(Path(row["path"])) for row in endpoints}
    if any(checkpoint_shas[row["endpoint_id"]] != row["sha256"] for row in endpoints):
        raise Refuse("endpoint checkpoint changed between preflight and attempt publication")

    attempt_id = secrets.token_hex(24)
    attempt = {
        "schema": "taiji-r2-dialogue-recovery-v2-paired-evaluation-attempt-v1",
        "status": "started",
        "run": RUN_ID,
        "attempt_id": attempt_id,
        "readiness": ready,
        "input_report_hashes": preflight["report_hashes"],
        "source_hashes": preflight["source_hashes"],
        "endpoint_order": list(ENDPOINT_ORDER),
        "endpoint_checkpoint_sha256": checkpoint_shas,
        "budget": {
            "cpu_limit_seconds": EVALUATION_CPU_LIMIT_SECONDS,
            "wall_limit_seconds": EVALUATION_WALL_LIMIT_SECONDS,
            "preflight_parent_cpu_seconds": preflight_cpu,
            "preflight_wall_seconds": preflight_wall,
            "whole_run_cpu_reserve_seconds": whole_cpu,
            "whole_run_wall_reserve_seconds": whole_wall,
            "supervisor_cancel_reserve_seconds": SUPERVISOR_CANCEL_RESERVE_SECONDS,
        },
        "durability_qualification": qualification,
        "resource_samples": resources,
        "disk_sample": disk,
        "parameter_updates": 0,
        "optimizer_updates": 0,
        "dev_queries": 0,
        "generation_queries": 0,
        "reference_answer_scoring_queries": 0,
        "exact_match_scoring_queries": 0,
        "can_promote": False,
    }
    attempt_sha = _durable_write_json(
        attempt_path, attempt, qualification=qualification, api=durability_api
    )
    try:
        _durable_write_jsonl(
            ledger_path,
            {
                "event": "attempt_started",
                "attempt_id": attempt_id,
                "attempt_sha256": attempt_sha,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "cpu_seconds_charged_before_worker": budget.cpu_seconds(),
                "wall_seconds_charged_before_worker": budget.wall_seconds(),
                "durability_qualification": qualification,
                "unknown_tail": False,
            },
            qualification=qualification,
            api=durability_api,
        )
    except BaseException as exc:
        _write_failure(
            output,
            error=exc,
            details={
                "attempt_id": attempt_id,
                "attempt_consumed": True,
                "initial_ledger_published": False,
                "worker_started": False,
                "can_promote": False,
            },
        )
        raise Refuse(
            "attempt is consumed because initial ledger publication failed; worker was not started"
        ) from exc
    _create_jsonl(
        progress_path,
        {
            "event": "execution_started",
            "attempt_id": attempt_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "resource_samples": resources,
            "can_promote": False,
        },
    )

    results: dict[str, dict[str, Any]] = {}
    endpoint_cpu = {row["endpoint_id"]: 0.0 for row in endpoints}
    query_attempts = persisted = 0
    active_task: dict[str, Any] | None = None
    active_task_intent_written = False
    active_task_terminal_written = False
    selector_started = face_persisted = failure_written = False
    worker: Any | None = None

    def durable_append(path: Path, event: Mapping[str, Any]) -> None:
        _append_jsonl(path, event, flush_file=durability_api.flush_file)

    def send(payload: dict[str, Any], op: str, remaining: int, fixed_cpu: float, fixed_wall: float):
        task_cpu, task_wall = _task_limits(
            budget=budget,
            operation=op,
            remaining_tasks=remaining,
            future_cpu_reserve=fixed_cpu,
            future_wall_reserve=fixed_wall,
        )
        response, child, parent = worker.request(
            payload,
            budget=budget,
            task_cpu_limit_seconds=task_cpu,
            task_wall_limit_seconds=task_wall,
            future_cpu_reserve_seconds=task_cpu * max(0, remaining - 1)
            + fixed_cpu
            + SUPERVISOR_CANCEL_RESERVE_SECONDS,
            future_wall_reserve_seconds=task_wall * max(0, remaining - 1)
            + fixed_wall
            + SUPERVISOR_CANCEL_RESERVE_SECONDS,
            cpu_now=cpu_now,
            wall_now=wall_now,
        )
        return response, child, parent, task_cpu, task_wall

    def failed_task(
        op: str, task_id: str, endpoint_id: str, payload: Mapping[str, Any], exc: BaseException
    ) -> None:
        nonlocal active_task_terminal_written, failure_written
        child_observed = max(0.0, float(getattr(exc, "observed_child_cpu_seconds", 0.0)))
        host_after, boot_after = host_cpu_fn(), boot_time_fn()
        boot_before = payload.get("boot_identity_before")
        host_before = payload.get("host_cpu_seconds_before")
        wall_before = payload.get("wall_started_monotonic")
        wall_elapsed = (
            max(0.0, wall_now() - float(wall_before))
            if isinstance(wall_before, (int, float))
            else 0.0
        )
        if not bool(getattr(exc, "cpu_measurement_valid", False)):
            if boot_before == boot_after and isinstance(host_before, (int, float)):
                fallback_child_upper = max(0.0, host_after - float(host_before))
                cpu_accounting = "same_boot_host_cpu_delta_upper"
            else:
                fallback_child_upper = wall_elapsed * max(
                    1, int(payload.get("logical_cpu_upper", os.cpu_count() or 1))
                )
                cpu_accounting = "wall_times_logical_cores_upper"
        else:
            fallback_child_upper = child_observed
            cpu_accounting = "windows_job_object_sample"
        child = max(child_observed, fallback_child_upper)
        budget.charge_child_cpu(child)
        stopped = bool(getattr(exc, "worker_tree_stopped", False))
        ipc_write_stopped = bool(getattr(exc, "ipc_write_thread_stopped", True))
        if worker is not None and (not stopped or not ipc_write_stopped):
            stopped = bool(worker.terminate_tree())
            ipc_write_stopped = bool(getattr(worker, "ipc_write_thread_stopped", ipc_write_stopped))
        if not stopped or not ipc_write_stopped:
            if isinstance(exc, WorkerTaskFailure):
                exc.worker_tree_stopped = stopped
                exc.ipc_write_thread_stopped = ipc_write_stopped
                raise exc
            raise WorkerTaskFailure(
                f"worker tree or IPC writer termination could not be proven: {exc}",
                observed_child_cpu_seconds=child,
                timed_out=bool(getattr(exc, "timed_out", False)),
                worker_tree_stopped=stopped,
                cpu_measurement_valid=bool(getattr(exc, "cpu_measurement_valid", False)),
                ipc_write_thread_stopped=ipc_write_stopped,
            ) from exc
        budget.guard(
            reserve_cpu_seconds=FINALIZATION_RESERVE_CPU_SECONDS,
            reserve_wall_seconds=FINALIZATION_RESERVE_WALL_SECONDS,
        )
        timeout = bool(getattr(exc, "timed_out", False))
        event_name = (
            ("restore_timeout" if timeout else "restore_failed")
            if op == "restore"
            else ("query_timeout" if timeout else "query_failed")
        )
        event = {
            "event": event_name,
            "attempt_id": attempt_id,
            "task_id": task_id,
            "operation": op,
            "endpoint_id": endpoint_id,
            "index": payload.get("index"),
            "prompt_sha256": payload.get("prompt_sha256"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "child_cpu_seconds_observed": child,
            "child_cpu_seconds_charged_upper": child,
            "cpu_accounting": cpu_accounting,
            "host_cpu_delta_upper": (
                max(0.0, host_after - float(host_before))
                if isinstance(host_before, (int, float)) and boot_before == boot_after
                else None
            ),
            "wall_seconds": wall_elapsed,
            "boot_identity_after": boot_after,
            "cpu_seconds_parent": budget.parent_cpu_seconds(),
            "cpu_seconds_total": budget.cpu_seconds(),
            "wall_seconds_total": budget.wall_seconds(),
            "worker_tree_stopped": True,
            "unknown_tail": True,
            "ipc_write_thread_stopped": ipc_write_stopped,
            "ipc_writer_cancel_error": getattr(worker, "_ipc_writer_cancel_error", None),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        durable_append(ledger_path, event)
        active_task_terminal_written = True
        durable_append(progress_path, event)
        _write_failure(
            output,
            error=exc,
            details={
                "attempt_id": attempt_id,
                "last_task_id": task_id,
                "last_operation": op,
                "last_endpoint": endpoint_id,
                "query_attempts": query_attempts,
                "persisted_generation_outputs": persisted,
                "fresh_process_restores": results,
                "cpu_seconds_parent": budget.parent_cpu_seconds(),
                "cpu_seconds_children": budget.child_cpu_seconds,
                "cpu_seconds_total_at_failure": budget.cpu_seconds(),
                "wall_seconds_at_failure": budget.wall_seconds(),
                "unknown_tail": True,
                "ipc_writer_exit_unverified": not ipc_write_stopped,
                "ipc_writer_cancel_error": getattr(worker, "_ipc_writer_cancel_error", None),
                "can_promote": False,
            },
        )
        failure_written = True

    try:
        startup_host_before, startup_boot_before = host_cpu_fn(), boot_time_fn()
        durable_append(
            ledger_path,
            {
                "event": "worker_startup_started",
                "attempt_id": attempt_id,
                "host_cpu_seconds_before": startup_host_before,
                "boot_identity": startup_boot_before,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "unknown_tail": True,
            },
        )
        startup_future_cpu = (
            _restore_cpu_reserve(len(endpoints))
            + _generation_cpu_reserve(expected_generation_count())
            + FACE_PREPARE_RESERVE_CPU_SECONDS
            + FINALIZATION_RESERVE_CPU_SECONDS
            + SUPERVISOR_CANCEL_RESERVE_SECONDS
        )
        startup_future_wall = (
            _restore_wall_reserve(len(endpoints))
            + _generation_wall_reserve(expected_generation_count())
            + FACE_PREPARE_RESERVE_WALL_SECONDS
            + FINALIZATION_RESERVE_WALL_SECONDS
            + SUPERVISOR_CANCEL_RESERVE_SECONDS
        )
        worker = worker_factory(
            attempt_path=attempt_path,
            ledger_path=ledger_path,
            budget=budget,
            startup_future_cpu_reserve_seconds=startup_future_cpu,
            startup_future_wall_reserve_seconds=startup_future_wall,
        )
        # Default production factory supervises bootstrap with the active budget;
        # injected synthetic workers deliberately need no OS process bootstrap.
        startup_cpu = max(0.0, float(getattr(worker, "initial_child_cpu_seconds", 0.0)))
        budget.charge_child_cpu(startup_cpu)
        endpoint_cpu["worker_startup"] = startup_cpu
        durable_append(
            ledger_path,
            {
                "event": "worker_supervision_ready",
                "attempt_id": attempt_id,
                "worker_job_name": getattr(worker, "job_name", None),
                "worker_pid": getattr(getattr(worker, "process", None), "pid", None),
                "supervision_kind": (
                    "windows_job_object"
                    if isinstance(worker, SupervisedWorker)
                    else "synthetic_injected_worker"
                ),
                "worker_startup_cpu_seconds_charged_total": startup_cpu,
                "job_object_active_processes": getattr(
                    worker, "job_active_processes_at_start", None
                ),
                "job_object_kill_on_close": isinstance(worker, SupervisedWorker),
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "unknown_tail": False,
            },
        )
        torch.set_num_threads(1)
        for i, endpoint in enumerate(endpoints):
            endpoint_id = endpoint["endpoint_id"]
            task_id, token = f"restore-{i}-{secrets.token_hex(8)}", secrets.token_urlsafe(32)
            remaining = len(endpoints) - i
            future_cpu = (
                _generation_cpu_reserve(156)
                + FACE_PREPARE_RESERVE_CPU_SECONDS
                + FINALIZATION_RESERVE_CPU_SECONDS
            )
            future_wall = (
                _generation_wall_reserve(156)
                + FACE_PREPARE_RESERVE_WALL_SECONDS
                + FINALIZATION_RESERVE_WALL_SECONDS
            )
            budget.guard(
                reserve_cpu_seconds=_restore_cpu_reserve(remaining)
                + future_cpu
                + SUPERVISOR_CANCEL_RESERVE_SECONDS,
                reserve_wall_seconds=_restore_wall_reserve(remaining)
                + future_wall
                + SUPERVISOR_CANCEL_RESERVE_SECONDS,
            )
            restore_host_before, restore_boot_before, restore_wall_before = (
                host_cpu_fn(),
                boot_time_fn(),
                wall_now(),
            )
            payload = {
                "operation": "restore",
                "attempt_id": attempt_id,
                "task_id": task_id,
                "endpoint_id": endpoint_id,
                "token": token,
                "checkpoint_sha256": endpoint["sha256"],
                "host_cpu_seconds_before": restore_host_before,
                "boot_identity_before": restore_boot_before,
                "wall_started_monotonic": restore_wall_before,
                "logical_cpu_upper": os.cpu_count() or 1,
            }
            intent = {
                "event": "restore_started",
                "attempt_id": attempt_id,
                "task_id": task_id,
                "operation": "restore",
                "endpoint_id": endpoint_id,
                "checkpoint_sha256": endpoint["sha256"],
                "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "remaining_tasks": remaining,
                "parent_cpu_seconds_before": budget.parent_cpu_seconds(),
                "child_cpu_seconds_before": budget.child_cpu_seconds,
                "total_cpu_seconds_before": budget.cpu_seconds(),
                "host_cpu_seconds_before": restore_host_before,
                "boot_identity": restore_boot_before,
                "logical_cpu_upper": os.cpu_count() or 1,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "unknown_tail": True,
            }
            active_task_intent_written = False
            active_task_terminal_written = False
            active_task = {"operation": "restore", "task_id": task_id, "endpoint_id": endpoint_id}
            durable_append(ledger_path, intent)
            active_task_intent_written = True
            durable_append(progress_path, intent)
            child = 0.0
            child_measurement_valid = False
            try:
                response, child, parent_task_cpu, task_cpu, task_wall = send(
                    payload, "restore", remaining, future_cpu, future_wall
                )
                child_measurement_valid = True
                if (
                    response.get("status") != "ok"
                    or response.get("operation") != "restore"
                    or response.get("task_id") != task_id
                    or response.get("endpoint_id") != endpoint_id
                ):
                    raise WorkerTaskFailure("worker restore response identity/status is invalid")
                receipt = dict(_require_mapping(response.get("receipt"), "worker restore receipt"))
                receipt["cpu_seconds"] = max(float(receipt.get("cpu_seconds", 0.0)), child)
                _validate_restore_receipt(endpoint, receipt)
                budget.charge_child_cpu(child)
                endpoint_cpu[endpoint_id] += child
                results[endpoint_id] = receipt
                host_after, boot_after = host_cpu_fn(), boot_time_fn()
                event = {
                    "event": "restore_completed",
                    "attempt_id": attempt_id,
                    "task_id": task_id,
                    "operation": "restore",
                    "endpoint_id": endpoint_id,
                    "receipt": receipt,
                    "receipt_sha256": _canonical_sha256(receipt),
                    "child_cpu_seconds": child,
                    "parent_cpu_seconds": parent_task_cpu,
                    "host_cpu_delta_upper": (
                        max(0.0, host_after - restore_host_before)
                        if boot_after == restore_boot_before
                        else None
                    ),
                    "boot_identity_after": boot_after,
                    "wall_seconds": max(0.0, wall_now() - restore_wall_before),
                    "task_cpu_deadline_seconds": task_cpu,
                    "task_wall_deadline_seconds": task_wall,
                    "cpu_seconds_charged_total": budget.cpu_seconds(),
                    "unknown_tail": False,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                }
                durable_append(ledger_path, event)
                active_task_terminal_written = True
                durable_append(progress_path, event)
            except BaseException as exc:
                if not isinstance(exc, WorkerTaskFailure):
                    exc = WorkerTaskFailure(
                        str(exc),
                        observed_child_cpu_seconds=child,
                        worker_tree_stopped=False,
                        cpu_measurement_valid=child_measurement_valid,
                    )
                if active_task_terminal_written:
                    raise
                failed_task("restore", task_id, endpoint_id, payload, exc)
                active_task = None
                raise Refuse(
                    f"{endpoint_id} restore failed; this one-shot evaluation attempt is consumed"
                ) from exc
            active_task = None
            remaining = len(endpoints) - i - 1
            budget.guard(
                reserve_cpu_seconds=_restore_cpu_reserve(remaining)
                + _generation_cpu_reserve(156)
                + FACE_PREPARE_RESERVE_CPU_SECONDS
                + FINALIZATION_RESERVE_CPU_SECONDS
                + SUPERVISOR_CANCEL_RESERVE_SECONDS,
                reserve_wall_seconds=_restore_wall_reserve(remaining)
                + _generation_wall_reserve(156)
                + FACE_PREPARE_RESERVE_WALL_SECONDS
                + FINALIZATION_RESERVE_WALL_SECONDS
                + SUPERVISOR_CANCEL_RESERVE_SECONDS,
            )

        if tuple(results) != ENDPOINT_ORDER:
            raise Refuse("all four endpoint restores must complete before prompt access")
        checked = preflight_fn()
        if (
            _readiness_payload(checked) != ready
            or checked.get("source_hashes") != attempt["source_hashes"]
            or checked.get("report_hashes") != preflight["report_hashes"]
            or tuple(row["endpoint_id"] for row in checked["endpoints"]) != ENDPOINT_ORDER
        ):
            raise Refuse("source/result/readiness binding changed before prompt selection")
        after_restore_shas = {
            row["endpoint_id"]: hash_file(Path(row["path"])) for row in checked["endpoints"]
        }
        if after_restore_shas != checkpoint_shas:
            raise Refuse("checkpoint binding changed before prompt selection")
        selector_started = True
        durable_append(
            ledger_path,
            {
                "event": "prompt_selection_started",
                "attempt_id": attempt_id,
                "source_hashes": attempt["source_hashes"],
                "report_hashes": preflight["report_hashes"],
                "endpoint_checkpoint_sha256": checkpoint_shas,
                "restore_receipt_sha256": {k: _canonical_sha256(v) for k, v in results.items()},
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "unknown_tail": True,
            },
        )
        prompts = prompt_selector()
        face = freeze_prompt_face(prompts)
        _write_new_json(Path(output["face"]), face)
        face_persisted = True
        durable_append(
            ledger_path,
            {
                "event": "prompt_face_persisted",
                "attempt_id": attempt_id,
                "face_sha256": face["face_sha256"],
                "prompt_count": len(face["rows"]),
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "unknown_tail": False,
            },
        )
        generation_rows = [dict(row) for row in face["rows"]]
        for i, _row in enumerate(face["rows"], 1):
            for endpoint in endpoints:
                endpoint_id = endpoint["endpoint_id"]
                task_id, token = (
                    f"query-{query_attempts:03d}-{secrets.token_hex(8)}",
                    secrets.token_urlsafe(32),
                )
                remaining = 156 - query_attempts
                prompt_sha = face["prompt_hashes"][i - 1]
                available = psutil.virtual_memory().available / 1024**3
                if available < MIN_FREE_MEMORY_GIB:
                    raise Refuse("available memory fell below 8 GiB before generation")
                query_host_before, query_boot_before, query_wall_before = (
                    host_cpu_fn(),
                    boot_time_fn(),
                    wall_now(),
                )
                payload = {
                    "operation": "generate",
                    "attempt_id": attempt_id,
                    "task_id": task_id,
                    "endpoint_id": endpoint_id,
                    "token": token,
                    "face_path": str(Path(output["face"]).resolve()),
                    "face_sha256": face["face_sha256"],
                    "prompt_sha256": prompt_sha,
                    "index": i,
                    "host_cpu_seconds_before": query_host_before,
                    "boot_identity_before": query_boot_before,
                    "wall_started_monotonic": query_wall_before,
                    "logical_cpu_upper": os.cpu_count() or 1,
                }
                intent = {
                    "event": "query_started",
                    "attempt_id": attempt_id,
                    "task_id": task_id,
                    "operation": "generate",
                    "index": i,
                    "query_number": query_attempts + 1,
                    "endpoint_id": endpoint_id,
                    "prompt_sha256": prompt_sha,
                    "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "memory_available_gib": round(available, 3),
                    "cpu_seconds_before": budget.cpu_seconds(),
                    "host_cpu_seconds_before": query_host_before,
                    "boot_identity": query_boot_before,
                    "logical_cpu_upper": os.cpu_count() or 1,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "unknown_tail": True,
                }
                active_task_intent_written = False
                active_task_terminal_written = False
                active_task = {
                    "operation": "generate",
                    "task_id": task_id,
                    "endpoint_id": endpoint_id,
                    "index": i,
                }
                durable_append(ledger_path, intent)
                active_task_intent_written = True
                query_attempts += 1
                durable_append(progress_path, intent)
                child = 0.0
                child_measurement_valid = False
                try:
                    response, child, parent_task_cpu, task_cpu, task_wall = send(
                        payload,
                        "generate",
                        remaining,
                        FINALIZATION_RESERVE_CPU_SECONDS,
                        FINALIZATION_RESERVE_WALL_SECONDS,
                    )
                    child_measurement_valid = True
                    if (
                        response.get("status") != "ok"
                        or response.get("operation") != "generate"
                        or response.get("task_id") != task_id
                        or response.get("endpoint_id") != endpoint_id
                    ):
                        raise WorkerTaskFailure(
                            "worker generation response identity/status is invalid"
                        )
                    generated = _validate_generation_output(response.get("output"))
                    if (
                        response.get("model_digest_before") != results[endpoint_id]["model_digest"]
                        or response.get("model_digest_after") != response.get("model_digest_before")
                        or response.get("checkpoint_sha256_before") != checkpoint_shas[endpoint_id]
                        or response.get("checkpoint_sha256_after") != checkpoint_shas[endpoint_id]
                    ):
                        raise Refuse("worker generation changed model/checkpoint binding")
                    budget.charge_child_cpu(child)
                    endpoint_cpu[endpoint_id] += child
                    generation_rows[i - 1][endpoint_id] = generated
                    output_event = {
                        "event": "generation_output_persisted",
                        "attempt_id": attempt_id,
                        "task_id": task_id,
                        "index": i,
                        "query_number": query_attempts,
                        "endpoint_id": endpoint_id,
                        "prompt_sha256": prompt_sha,
                        "output": generated,
                        "worker_tree_cpu_seconds": child,
                        "task_cpu_deadline_seconds": task_cpu,
                        "task_wall_deadline_seconds": task_wall,
                        "parent_cpu_seconds": parent_task_cpu,
                        "wall_seconds": max(0.0, wall_now() - query_wall_before),
                        "can_promote": False,
                    }
                    durable_append(progress_path, output_event)
                    durable_append(
                        ledger_path,
                        {
                            "event": "query_output_persisted",
                            "attempt_id": attempt_id,
                            "task_id": task_id,
                            "index": i,
                            "query_number": query_attempts,
                            "endpoint_id": endpoint_id,
                            "prompt_sha256": prompt_sha,
                            "output_sha256": _canonical_sha256(generated),
                            "worker_tree_cpu_seconds": child,
                            "parent_cpu_seconds": parent_task_cpu,
                            "host_cpu_delta_upper": (
                                max(0.0, host_cpu_fn() - query_host_before)
                                if boot_time_fn() == query_boot_before
                                else None
                            ),
                            "wall_seconds": max(0.0, wall_now() - query_wall_before),
                            "model_digest_stable": True,
                            "checkpoint_sha_stable": True,
                            "unknown_tail": False,
                            "timestamp_utc": datetime.now(UTC).isoformat(),
                        },
                    )
                    active_task_terminal_written = True
                    persisted += 1
                except BaseException as exc:
                    if not isinstance(exc, WorkerTaskFailure):
                        exc = WorkerTaskFailure(
                            str(exc),
                            observed_child_cpu_seconds=child,
                            worker_tree_stopped=False,
                            cpu_measurement_valid=child_measurement_valid,
                        )
                    if active_task_terminal_written:
                        raise
                    failed_task("generate", task_id, endpoint_id, payload, exc)
                    active_task = None
                    raise Refuse(
                        "generation failed; this one-shot evaluation attempt is consumed"
                    ) from exc
                active_task = None

        if query_attempts != 156 or persisted != 156:
            raise Refuse("one-shot evaluation did not persist all 156 required generations")
        durable_append(
            ledger_path,
            {
                "event": "worker_shutdown_started",
                "attempt_id": attempt_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "unknown_tail": True,
            },
        )
        shutdown_cpu = min(
            SUPERVISOR_CANCEL_RESERVE_SECONDS,
            EVALUATION_CPU_LIMIT_SECONDS - budget.cpu_seconds() - FINALIZATION_RESERVE_CPU_SECONDS,
        )
        shutdown_wall = min(
            SUPERVISOR_CANCEL_RESERVE_SECONDS,
            EVALUATION_WALL_LIMIT_SECONDS
            - budget.wall_seconds()
            - FINALIZATION_RESERVE_WALL_SECONDS,
        )
        if shutdown_cpu <= 0 or shutdown_wall <= 0:
            raise TimeoutError("no bounded worker shutdown budget remains")
        child_shutdown, parent_shutdown = worker.shutdown(
            budget=budget, cpu_limit_seconds=shutdown_cpu, wall_limit_seconds=shutdown_wall
        )
        budget.charge_child_cpu(child_shutdown)
        endpoint_cpu["worker_shutdown"] = child_shutdown
        worker = None
        durable_append(
            ledger_path,
            {
                "event": "worker_shutdown_completed",
                "attempt_id": attempt_id,
                "child_cpu_seconds": child_shutdown,
                "parent_cpu_seconds": parent_shutdown,
                "worker_tree_stopped": True,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "unknown_tail": False,
            },
        )
        budget.guard(
            reserve_cpu_seconds=FINALIZATION_RESERVE_CPU_SECONDS,
            reserve_wall_seconds=FINALIZATION_RESERVE_WALL_SECONDS,
        )
        checkpoint_after = {row["endpoint_id"]: hash_file(Path(row["path"])) for row in endpoints}
        if checkpoint_after != checkpoint_shas or _readiness_payload(preflight_fn()) != ready:
            raise Refuse("source/result/checkpoint metadata changed during evaluation")
        generation_payload = {
            "schema": "taiji-r2-dialogue-recovery-v2-paired-evaluation-generations-v1",
            "status": "complete",
            "input_families": FACE_SIZE,
            "endpoint_order": list(ENDPOINT_ORDER),
            "generation_outputs": persisted,
            "max_generation_bytes": MAX_GENERATION_BYTES,
            "constrained_decode": True,
            "sampling": False,
            "metrics": {
                ep: base._summary([row[ep] for row in generation_rows]) for ep in ENDPOINT_ORDER
            },
            "reference_answer_included": False,
            "exact_response_scored": False,
            "records": generation_rows,
            "can_promote": False,
        }
        _write_new_json(Path(output["generations"]), generation_payload)
        _write_new_text(Path(output["review"]), _review_markdown(generation_rows))
        result = {
            "schema": "taiji-r2-dialogue-recovery-v2-paired-evaluation-result-v1",
            "status": "complete",
            "complete": True,
            "run": RUN_ID,
            "attempt_id": attempt_id,
            "source_hashes": preflight["source_hashes"],
            "input_report_hashes": preflight["report_hashes"],
            "baseline_checkpoint_sha256": preflight["baseline"]["sha256"],
            "endpoint_order": list(ENDPOINT_ORDER),
            "endpoint_checkpoints_before": checkpoint_shas,
            "endpoint_checkpoints_after": checkpoint_after,
            "fresh_process_restores": results,
            "model_digests_before": {k: v["model_digest"] for k, v in results.items()},
            "model_digests_after": {k: v["model_digest"] for k, v in results.items()},
            "generation_face_sha256": face["face_sha256"],
            "prompt_hashes": face["prompt_hashes"],
            "input_families": FACE_SIZE,
            "query_attempts": query_attempts,
            "generation_outputs": persisted,
            "generation_queries": persisted,
            "dev_queries": persisted,
            "persisted_generation_outputs": persisted,
            "reused_generation_outputs": 0,
            "combined_generation_outputs": persisted,
            "generations_sha256": hash_file(Path(output["generations"])),
            "review_sha256": hash_file(Path(output["review"])),
            "cost_receipt_path": str(Path(output["cost_receipt"]).resolve()),
            "parameter_updates": 0,
            "optimizer_updates": 0,
            "reference_answer_scoring_queries": 0,
            "exact_match_scoring_queries": 0,
            "cpu_seconds_parent_at_finalization_start": budget.parent_cpu_seconds(),
            "cpu_seconds_worker_tree": budget.child_cpu_seconds,
            "cpu_seconds_total_at_finalization_start": budget.cpu_seconds(),
            "wall_seconds_total_at_finalization_start": budget.wall_seconds(),
            "finalization_reserve_cpu_seconds": FINALIZATION_RESERVE_CPU_SECONDS,
            "finalization_reserve_wall_seconds": FINALIZATION_RESERVE_WALL_SECONDS,
            "per_endpoint_worker_tree_cpu_seconds": endpoint_cpu,
            "cpu_limit_seconds": EVALUATION_CPU_LIMIT_SECONDS,
            "wall_limit_seconds": EVALUATION_WALL_LIMIT_SECONDS,
            "resource_samples": resources,
            "disk_sample": disk,
            "errors": [],
            "can_promote": False,
        }
        durable_append(
            ledger_path,
            {
                "event": "evaluation_finalizing",
                "attempt_id": attempt_id,
                "query_attempts": query_attempts,
                "persisted_generation_outputs": persisted,
                "cpu_seconds_parent": budget.parent_cpu_seconds(),
                "cpu_seconds_worker_tree": budget.child_cpu_seconds,
                "cpu_seconds_total_before_cost_receipt": budget.cpu_seconds(),
                "wall_seconds_before_cost_receipt": budget.wall_seconds(),
                "can_promote": False,
            },
        )
        receipt = {
            "schema": "taiji-r2-dialogue-recovery-v2-cost-receipt-v1",
            "status": "measured_before_cost_receipt_publish",
            "attempt_id": attempt_id,
            "result_path": str(Path(output["result"]).resolve()),
            "result_schema": result["schema"],
            "source_hashes": attempt["source_hashes"],
            "input_report_hashes": preflight["report_hashes"],
            "endpoint_checkpoint_sha256": checkpoint_shas,
            "preflight_parent_cpu_seconds": preflight_cpu,
            "preflight_wall_seconds": preflight_wall,
            "cpu_seconds_parent_before_receipt_publish": budget.parent_cpu_seconds(),
            "cpu_seconds_worker_tree_before_receipt_publish": budget.child_cpu_seconds,
            "cpu_seconds_total_before_receipt_publish": budget.cpu_seconds(),
            "wall_seconds_before_receipt_publish": budget.wall_seconds(),
            "per_endpoint_worker_tree_cpu_seconds": endpoint_cpu,
            "cpu_accounting": "parent_process_time_plus_windows_job_object_tree_user_system",
            "wall_accounting": "parent_monotonic_elapsed",
            "cpu_limit_seconds": EVALUATION_CPU_LIMIT_SECONDS,
            "wall_limit_seconds": EVALUATION_WALL_LIMIT_SECONDS,
            "attempt_cpu_seconds_remaining_before_receipt": max(
                0.0, EVALUATION_CPU_LIMIT_SECONDS - budget.cpu_seconds()
            ),
            "attempt_wall_seconds_remaining_before_receipt": max(
                0.0, EVALUATION_WALL_LIMIT_SECONDS - budget.wall_seconds()
            ),
            "result_publication_excluded": True,
            "receipt_publication_excluded": True,
            "result_publication_reserve_cpu_seconds": FINALIZATION_RESERVE_CPU_SECONDS,
            "result_publication_reserve_wall_seconds": FINALIZATION_RESERVE_WALL_SECONDS,
            "query_attempts": query_attempts,
            "persisted_generation_outputs": persisted,
            "restore_receipt_count": len(results),
            "worker_tree_exit_confirmed": True,
            "parameter_updates": 0,
            "optimizer_updates": 0,
            "reference_answer_scoring_queries": 0,
            "exact_match_scoring_queries": 0,
            "unknown_tail": False,
            "can_promote": False,
        }
        receipt_sha = _durable_write_json(
            Path(output["cost_receipt"]), receipt, qualification=qualification, api=durability_api
        )
        result["cost_receipt_sha256"] = receipt_sha
        _write_new_json(Path(output["result"]), result)
        return result
    except BaseException as exc:
        task_intent_unverified = active_task is not None and not active_task_intent_written
        task_terminal_missing = (
            active_task is not None
            and active_task_intent_written
            and not active_task_terminal_written
        )
        worker_tree_exit_verified = not (
            isinstance(exc, WorkerTaskFailure) and not exc.worker_tree_stopped
        )
        ipc_writer_exit_verified = not (
            isinstance(exc, WorkerTaskFailure) and not exc.ipc_write_thread_stopped
        )
        if worker is not None:
            sample = getattr(worker, "sample_tree_cpu_seconds", None)
            before = float(sample()) if callable(sample) else 0.0
            stopped = worker.terminate_tree()
            worker_tree_exit_verified = bool(stopped)
            ipc_writer_exit_verified = bool(
                getattr(worker, "ipc_write_thread_stopped", ipc_writer_exit_verified)
            )
            if not stopped:
                exc = Refuse(f"worker process tree termination could not be proven: {exc}")
            elif callable(sample):
                budget.charge_child_cpu(max(0.0, float(sample()) - before))
        elif isinstance(exc, WorkerTaskFailure):
            budget.charge_child_cpu(max(0.0, exc.observed_child_cpu_seconds))
        can_finalize_failure = True
        if not failure_written:
            try:
                budget.guard(
                    reserve_cpu_seconds=FINALIZATION_RESERVE_CPU_SECONDS,
                    reserve_wall_seconds=FINALIZATION_RESERVE_WALL_SECONDS,
                )
            except TimeoutError:
                can_finalize_failure = False
        if not failure_written and can_finalize_failure:
            event = {
                "event": "evaluation_stopped",
                "attempt_id": attempt_id,
                "task_id": active_task.get("task_id") if active_task else None,
                "operation": active_task.get("operation") if active_task else None,
                "last_endpoint": active_task.get("endpoint_id") if active_task else None,
                "query_attempts": query_attempts,
                "persisted_generation_outputs": persisted,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cpu_seconds_parent": budget.parent_cpu_seconds(),
                "cpu_seconds_worker_tree": budget.child_cpu_seconds,
                "cpu_seconds_total": budget.cpu_seconds(),
                "wall_seconds_total": budget.wall_seconds(),
                "unknown_tail": bool(selector_started and not face_persisted)
                or task_intent_unverified
                or task_terminal_missing
                or not worker_tree_exit_verified
                or not ipc_writer_exit_verified,
                "task_terminal_missing": task_terminal_missing,
                "task_intent_unverified": task_intent_unverified,
                "worker_tree_exit_unverified": not worker_tree_exit_verified,
                "ipc_writer_exit_unverified": not ipc_writer_exit_verified,
                "ipc_writer_cancel_error": getattr(worker, "_ipc_writer_cancel_error", None),
                "cpu_measurement_valid": bool(getattr(exc, "cpu_measurement_valid", False)),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
            try:
                durable_append(ledger_path, event)
                durable_append(progress_path, event)
            except BaseException:
                pass
            _write_failure(
                output,
                error=exc,
                details={
                    "attempt_id": attempt_id,
                    "last_task": active_task,
                    "query_attempts": query_attempts,
                    "persisted_generation_outputs": persisted,
                    "fresh_process_restores": results,
                    "cpu_seconds_parent": budget.parent_cpu_seconds(),
                    "cpu_seconds_children": budget.child_cpu_seconds,
                    "cpu_seconds_total_at_failure": budget.cpu_seconds(),
                    "wall_seconds_at_failure": budget.wall_seconds(),
                    "unknown_tail": bool(selector_started and not face_persisted)
                    or task_intent_unverified
                    or task_terminal_missing
                    or not worker_tree_exit_verified
                    or not ipc_writer_exit_verified,
                    "task_terminal_missing": task_terminal_missing,
                    "task_intent_unverified": task_intent_unverified,
                    "worker_tree_exit_unverified": not worker_tree_exit_verified,
                    "ipc_writer_exit_unverified": not ipc_writer_exit_verified,
                    "ipc_writer_cancel_error": getattr(worker, "_ipc_writer_cancel_error", None),
                    "can_promote": False,
                },
            )
        raise Refuse(f"one-shot evaluation attempt is consumed: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ledger-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--job-name", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.attempt_path is None or args.ledger_path is None or not args.job_name:
            parser.error("private --worker mode requires attempt, ledger and Job Object name")
        torch.set_num_threads(1)
        return _worker_loop(args.attempt_path.resolve(), args.ledger_path.resolve(), args.job_name)
    if args.attempt_path is not None or args.ledger_path is not None or args.job_name:
        parser.error("attempt/ledger paths are only valid in private worker mode")
    result = validate_only() if args.validate_only else execute()
    print(json.dumps({"status": result.get("status"), "can_promote": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

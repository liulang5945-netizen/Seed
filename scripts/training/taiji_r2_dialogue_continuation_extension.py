"""Versioned, durable CPU accounting for the R2 continuation extension.

The runner must call :func:`initialize_ledger` before any parent restore or
training work. It then calls :func:`append_charge` after every durable CPU
snapshot and :func:`guard` before each bounded work unit. The aggregate ledger
is an append-only series of write-once JSON event files in the existing
``reports`` directory; ``paths(arm)["aggregate_ledger_path"]`` is the common
filename prefix (not a mutable JSONL file).

If initialization stopped after its contract became durable but before event
zero was published, :func:`reconcile_initialize_ledger` can verify the exact
original inputs and add that one missing initialization event. It never
replaces a contract or an event.

Parent records passed to ``initialize_ledger`` must contain
``parent_attempt_sha256``, ``parent_receipt_sha256``, ``parent_bundle_sha256``,
``charged_cpu_seconds``, ``tail_status`` (``closed`` or
``conservative_upper_bound``), and ``accounting_start``. The accounting start
must identify its UTC snapshot, boot identity, host CPU counter, and logical
CPU upper bound. Unresolved or partial tails are rejected.

``guard`` atomically reserves one bounded work unit under the aggregate lock
and returns a 32-character hexadecimal token. ``append_charge`` settles one
active token via ``details["reservation_id"]``; missing, repeated, cross-arm,
or incorrect tokens are refused. ``latest_charges`` includes outstanding
reservation uppers, and ``latest_reservations`` exposes tokens for a new
process to reconcile with a ``continuation_resumed`` settlement.

Durable writes are supported only on a local fixed NTFS volume. Files are
written to an exclusive same-directory temporary file, flushed, and published
with ``MoveFileExW(MOVEFILE_WRITE_THROUGH)``. A persistent byte-range lock
serializes event numbering across processes; the OS releases the lock if a
writer exits.

For other extension artifacts, ``qualify_storage_target(path,
expected_volume_root=None)`` returns the approved fixed-NTFS qualification.
``durable_publish(path, writer, replace=False, expected_volume_root=None)``
durably publishes arbitrary binary content and returns its verified readback
SHA-256. Passing the reports volume root pins a checkpoint or sidecar
destination to the same approved volume.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

ARMS = ("full_turn_1m", "warm_start", "fresh_same_capacity")
BASE_PER_ARM_CPU_LIMIT = 1_450_000.0
BASE_AGGREGATE_CPU_LIMIT = 4_200_000.0
PER_ARM_CPU_LIMIT = BASE_PER_ARM_CPU_LIMIT
AGGREGATE_CPU_LIMIT = BASE_AGGREGATE_CPU_LIMIT
MOVEFILE_WRITE_THROUGH = 0x00000008
MOVEFILE_REPLACE_EXISTING = 0x00000001
DRIVE_FIXED = 3
CONTRACT_SCHEMA = "taiji-r2-dialogue-continuation-extension-v1-budget-contract"
EVENT_SCHEMA = "taiji-r2-dialogue-continuation-extension-v1-budget-event"
SUMMARY_SCHEMA = "taiji-r2-dialogue-continuation-extension-v1-budget-summary"

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
PARENT_CHECKPOINT_ROOT = ROOT / "data/checkpoints/r2_daily_dialogue_recovery_v2_20261021"
EXTENSION_CHECKPOINT_ROOT = PARENT_CHECKPOINT_ROOT / "continuation_v1"
EXTENSION_PREFIX = "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1"
CONTRACT_PATH = REPORTS_DIR / f"{EXTENSION_PREFIX}_contract.json"
AGGREGATE_LEDGER_PREFIX = REPORTS_DIR / f"{EXTENSION_PREFIX}_aggregate_cpu_ledger"
AGGREGATE_LEDGER_SUMMARY = REPORTS_DIR / f"{EXTENSION_PREFIX}_aggregate_cpu_ledger.latest.json"
LOCK_PATH = REPORTS_DIR / f"{EXTENSION_PREFIX}_aggregate_cpu_ledger.lock"

_THREAD_LOCK = threading.RLock()
_EVENT_RE = re.compile(r"^(?P<prefix>.+)\.(?P<sequence>\d{8})\.json$")
_HEX_256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION_V1_1_SCHEMA = "taiji-r2-continuation-source-revision-v1.1"
_SOURCE_REVISION_V1_2_SCHEMA = "taiji-r2-continuation-source-revision-v1.2"
_SOURCE_REVISION_V1_1_NAME = f"{EXTENSION_PREFIX}_source_revision_v1_1.json"
_SOURCE_REVISION_V1_2_NAME = f"{EXTENSION_PREFIX}_source_revision_v1_2.json"
_SOURCE_REVISION_V1_1_REVIEW = (
    ROOT
    / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-source-revision-v1-1.md"
)
_SOURCE_REVISION_V1_2_REVIEW = (
    ROOT
    / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-source-revision-v1-2.md"
)
_SOURCE_REVISION_ARCHITECTURE_REPORT = (
    ROOT
    / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-extension-architecture.md"
)
_ACTIVE_BUDGET_REVISION_SHA256: dict[str, str] = {}


class ExtensionLedgerRefuse(RuntimeError):
    """An extension accounting or durability precondition failed."""


def paths(arm: str) -> dict[str, Path]:
    """Return stable parent and extension paths for one frozen arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown R2 extension arm: {arm}")
    parent_prefix = "taiji_r2_daily_dialogue_recovery_v2_20261021"
    return {
        "parent_attempt": REPORTS_DIR / f"{parent_prefix}_{arm}_attempt.json",
        "parent_readiness": REPORTS_DIR / f"{parent_prefix}_{arm}_readiness.json",
        "parent_progress": REPORTS_DIR / f"{parent_prefix}_{arm}_progress.jsonl",
        "parent_result": REPORTS_DIR / f"{parent_prefix}_{arm}_result.json",
        "parent_checkpoint_root": PARENT_CHECKPOINT_ROOT / arm,
        "extension_attempt": REPORTS_DIR / f"{EXTENSION_PREFIX}_{arm}_attempt.json",
        "extension_progress": REPORTS_DIR / f"{EXTENSION_PREFIX}_{arm}_progress.jsonl",
        "extension_result": REPORTS_DIR / f"{EXTENSION_PREFIX}_{arm}_result.json",
        "extension_checkpoint_root": EXTENSION_CHECKPOINT_ROOT / arm,
        "contract_path": CONTRACT_PATH,
        # This is a filename prefix. Immutable events append as .00000000.json,
        # .00000001.json, ... beside it in the pre-existing reports directory.
        "aggregate_ledger_path": AGGREGATE_LEDGER_PREFIX,
    }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _digest(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    payload = {key: item for key, item in value.items() if key != omit}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtensionLedgerRefuse(f"cannot read durable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtensionLedgerRefuse(f"durable JSON must contain an object: {path}")
    return value


class _WindowsDurabilityApi:
    """Minimal Win32 API for fixed local NTFS write-through publication."""

    def __init__(self) -> None:
        if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
            raise ExtensionLedgerRefuse("DURABILITY_UNSUPPORTED: Windows local NTFS is required")
        from ctypes import wintypes

        self.wintypes = wintypes
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
        raw = str(Path(path).resolve())
        if raw.startswith("\\\\"):
            return {"is_unc": True, "drive_type": None, "filesystem": None, "volume_root": None}
        volume_buffer = ctypes.create_unicode_buffer(32768)
        if not self.kernel32.GetVolumePathNameW(raw, volume_buffer, len(volume_buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
        volume_root = volume_buffer.value
        filesystem_buffer = ctypes.create_unicode_buffer(64)
        volume_name_buffer = ctypes.create_unicode_buffer(256)
        serial = self.wintypes.DWORD()
        max_component = self.wintypes.DWORD()
        flags = self.wintypes.DWORD()
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
            "drive_type": int(self.kernel32.GetDriveTypeW(volume_root)),
            "filesystem": filesystem_buffer.value,
            "volume_root": volume_root,
        }

    def flush_file(self, fd: int) -> None:
        import msvcrt

        handle = self.wintypes.HANDLE(msvcrt.get_osfhandle(fd))
        if not self.kernel32.FlushFileBuffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def move_file_write_through(self, source: Path, destination: Path, flags: int) -> None:
        allowed = {MOVEFILE_WRITE_THROUGH, MOVEFILE_WRITE_THROUGH | MOVEFILE_REPLACE_EXISTING}
        if flags not in allowed:
            raise ExtensionLedgerRefuse(
                "MoveFileExW must use write-through with optional replacement"
            )
        if not self.kernel32.MoveFileExW(str(source), str(destination), flags):
            raise ctypes.WinError(ctypes.get_last_error())


_DURABILITY_API: Any | None = None


def _durability_api() -> Any:
    global _DURABILITY_API
    if _DURABILITY_API is None:
        _DURABILITY_API = _WindowsDurabilityApi()
    return _DURABILITY_API


def _normalize_volume_root(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(value))).rstrip("\\/").casefold()


def _qualify_storage_target_with_api(
    path: str | os.PathLike[str],
    expected_volume_root: str | os.PathLike[str] | None,
    api: Any,
) -> dict[str, Any]:
    target = Path(path).resolve()
    directory = target if target.is_dir() else target.parent
    if not directory.is_dir() or str(directory).startswith("\\\\"):
        raise ExtensionLedgerRefuse(
            "DURABILITY_UNSUPPORTED: existing local target directory required"
        )
    info = api.inspect(directory)
    if (
        info.get("is_unc") is not False
        or info.get("drive_type") != DRIVE_FIXED
        or str(info.get("filesystem", "")).upper() != "NTFS"
        or not isinstance(info.get("volume_root"), str)
        or not info.get("volume_root")
    ):
        raise ExtensionLedgerRefuse(
            "DURABILITY_UNSUPPORTED: only local fixed-volume NTFS is supported"
        )
    if expected_volume_root is not None and _normalize_volume_root(
        expected_volume_root
    ) != _normalize_volume_root(info["volume_root"]):
        raise ExtensionLedgerRefuse(
            "DURABILITY_UNSUPPORTED: target volume differs from approved volume"
        )
    return {
        "directory": str(directory),
        "volume_root": info["volume_root"],
        "filesystem": "NTFS",
        "drive_type": DRIVE_FIXED,
        "publication": "exclusive-temp-flush-movefileex-write-through",
    }


def qualify_storage_target(
    path: str | os.PathLike[str],
    expected_volume_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Prove a storage target is on an allowed local fixed NTFS volume.

    ``path`` may be an existing directory or a file path whose parent exists.
    Supplying ``expected_volume_root`` pins the target to a previously approved
    volume, such as the root recorded during report-directory qualification.
    """
    return _qualify_storage_target_with_api(path, expected_volume_root, _durability_api())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _durable_publish_with_api(
    path: str | os.PathLike[str],
    writer: Callable[[BinaryIO], Any],
    *,
    replace: bool,
    expected_volume_root: str | os.PathLike[str] | None,
    api: Any,
) -> str:
    path = Path(path).resolve()
    if not callable(writer):
        raise TypeError("writer must be callable with a binary stream")
    parent = path.parent
    if not parent.is_dir():
        raise ExtensionLedgerRefuse("durable destination directory is missing")
    _qualify_storage_target_with_api(parent, expected_volume_root, api)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    try:
        with temporary.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
            api.flush_file(stream.fileno())
        temp_sha = _file_sha256(temporary)
        if path.exists() and not replace:
            raise FileExistsError(path)
        if temporary.parent.resolve() != path.parent.resolve():
            raise ExtensionLedgerRefuse(
                "DURABILITY_UNSUPPORTED: temp and destination directories differ"
            )
        flags = MOVEFILE_WRITE_THROUGH | (MOVEFILE_REPLACE_EXISTING if replace else 0)
        api.move_file_write_through(temporary, path, flags)
        if not path.is_file():
            raise ExtensionLedgerRefuse("write-through destination is missing after publication")
        readback_sha = _file_sha256(path)
        if readback_sha != temp_sha:
            raise ExtensionLedgerRefuse("write-through readback SHA-256 mismatch")
        return readback_sha
    finally:
        temporary.unlink(missing_ok=True)


def durable_publish(
    path: str | os.PathLike[str],
    writer: Callable[[BinaryIO], Any],
    replace: bool = False,
    expected_volume_root: str | os.PathLike[str] | None = None,
) -> str:
    """Publish arbitrary binary content durably and return its readback SHA-256."""
    return _durable_publish_with_api(
        path,
        writer,
        replace=replace,
        expected_volume_root=expected_volume_root,
        api=_durability_api(),
    )


def _qualify_reports_directory(api: Any) -> dict[str, Any]:
    qualification = _qualify_storage_target_with_api(REPORTS_DIR, None, api)
    qualification["publication"] = "exclusive-temp-flush-movefileex-write-through"
    return qualification


def _durable_write(
    path: Path,
    payload: Mapping[str, Any],
    qualification: Mapping[str, Any],
    api: Any,
    *,
    replace: bool = False,
) -> str:
    raw = _canonical_json(payload)
    return _durable_publish_with_api(
        path,
        lambda stream: stream.write(raw),
        replace=replace,
        expected_volume_root=qualification["volume_root"],
        api=api,
    )


@contextmanager
def _aggregate_lock(lock_path: Path, timeout_seconds: float = 20.0) -> Iterator[None]:
    """Serialize appends with an OS byte-range lock that is released on process exit."""
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("lock timeout must be finite and positive")
    with _THREAD_LOCK:
        lock_path = Path(lock_path)
        if not lock_path.parent.is_dir():
            raise ExtensionLedgerRefuse("aggregate lock directory must already exist")
        with lock_path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(0)
            deadline = time.monotonic() + timeout_seconds
            locked = False
            if os.name == "nt":
                import msvcrt

                while time.monotonic() < deadline:
                    try:
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        locked = True
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EDEADLK, errno.EAGAIN}:
                            raise
                        time.sleep(0.01)
                if not locked:
                    raise ExtensionLedgerRefuse("aggregate budget ledger writer remained busy")
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                while time.monotonic() < deadline:
                    try:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                        break
                    except BlockingIOError:
                        time.sleep(0.01)
                if not locked:
                    raise ExtensionLedgerRefuse("aggregate budget ledger writer remained busy")
                try:
                    yield
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _event_path(prefix: Path, sequence: int) -> Path:
    return prefix.with_name(f"{prefix.name}.{sequence:08d}.json")


def _event_files(prefix: Path) -> list[tuple[int, Path]]:
    prefix = Path(prefix)
    if not prefix.parent.is_dir():
        raise ExtensionLedgerRefuse("aggregate ledger directory must already exist")
    temp_pattern = f".{prefix.name}.*.tmp"
    if any(prefix.parent.glob(temp_pattern)):
        raise ExtensionLedgerRefuse("uncommitted aggregate ledger temp file requires review")
    events: list[tuple[int, Path]] = []
    for path in prefix.parent.glob(f"{prefix.name}.*.json"):
        match = _EVENT_RE.fullmatch(path.name)
        if match and match.group("prefix") == prefix.name:
            events.append((int(match.group("sequence")), path))
    events.sort(key=lambda item: item[0])
    if [sequence for sequence, _ in events] != list(range(len(events))):
        raise ExtensionLedgerRefuse("aggregate ledger sequence has a gap or duplicate")
    return events


def _validate_parent_records(
    parent_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(parent_records) != set(ARMS):
        raise ExtensionLedgerRefuse("parent records must cover exactly the three frozen arms")
    normalized: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        raw = parent_records[arm]
        if not isinstance(raw, Mapping):
            raise ExtensionLedgerRefuse(f"parent record must be an object: {arm}")
        for field in ("parent_attempt_sha256", "parent_receipt_sha256", "parent_bundle_sha256"):
            value = raw.get(field)
            if not isinstance(value, str) or not _HEX_256_RE.fullmatch(value):
                raise ExtensionLedgerRefuse(f"parent record has invalid {field}: {arm}")
        charged = raw.get("charged_cpu_seconds")
        if not isinstance(charged, (int, float)) or isinstance(charged, bool):
            raise ExtensionLedgerRefuse(f"parent charge must be numeric: {arm}")
        charged = float(charged)
        if not math.isfinite(charged) or charged < 0 or charged > BASE_PER_ARM_CPU_LIMIT:
            raise ExtensionLedgerRefuse(f"parent charge is outside the approved ceiling: {arm}")
        if raw.get("tail_status") not in {"closed", "conservative_upper_bound"}:
            raise ExtensionLedgerRefuse(f"parent tail is unresolved or partial: {arm}")
        accounting = raw.get("accounting_start")
        if not isinstance(accounting, Mapping) or not all(
            key in accounting
            for key in ("snapshot_utc", "boot_identity", "host_cpu_seconds", "logical_cpu_upper")
        ):
            raise ExtensionLedgerRefuse(
                f"parent record lacks a complete accounting snapshot: {arm}"
            )
        if not isinstance(accounting.get("snapshot_utc"), str) or not accounting["snapshot_utc"]:
            raise ExtensionLedgerRefuse(f"parent accounting snapshot has no UTC time: {arm}")
        if not isinstance(accounting.get("boot_identity"), (str, int, float)):
            raise ExtensionLedgerRefuse(f"parent accounting snapshot has no boot identity: {arm}")
        if not _finite_nonnegative(accounting.get("host_cpu_seconds")):
            raise ExtensionLedgerRefuse(f"parent accounting snapshot has invalid host CPU: {arm}")
        cores = accounting.get("logical_cpu_upper")
        if not isinstance(cores, int) or isinstance(cores, bool) or cores < 1:
            raise ExtensionLedgerRefuse(
                f"parent accounting snapshot has invalid CPU upper bound: {arm}"
            )
        try:
            json.dumps(raw, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ExtensionLedgerRefuse(f"parent record is not JSON serializable: {arm}") from exc
        normalized[arm] = dict(raw)
        normalized[arm]["charged_cpu_seconds"] = charged
    total = sum(item["charged_cpu_seconds"] for item in normalized.values())
    if total > BASE_AGGREGATE_CPU_LIMIT:
        raise ExtensionLedgerRefuse("parent charges exceed the aggregate CPU ceiling")
    return normalized


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _source_revision_path(revision: str) -> Path:
    names = {
        "v1.1": _SOURCE_REVISION_V1_1_NAME,
        "v1.2": _SOURCE_REVISION_V1_2_NAME,
        "v1.3": f"{EXTENSION_PREFIX}_source_revision_v1_3.json",
    }
    try:
        name = names[revision]
    except KeyError as exc:
        raise ValueError(f"unknown source revision: {revision}") from exc
    return REPORTS_DIR / name


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ExtensionLedgerRefuse(f"cannot read revision-bound file: {path}") from exc


def _base_contract() -> dict[str, Any]:
    contract = _read_json(CONTRACT_PATH)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("per_arm_cpu_seconds") != BASE_PER_ARM_CPU_LIMIT
        or contract.get("aggregate_cpu_seconds") != BASE_AGGREGATE_CPU_LIMIT
        or contract.get("arms") != list(ARMS)
    ):
        raise ExtensionLedgerRefuse("extension budget contract schema or caps differ")
    expected_contract_sha = contract.get("contract_sha256")
    if not isinstance(expected_contract_sha, str) or expected_contract_sha != _digest(
        contract, omit="contract_sha256"
    ):
        raise ExtensionLedgerRefuse("extension budget contract digest mismatch")
    return contract


def _expected_source_revision_files() -> dict[str, Path]:
    return {
        "runner": ROOT / "scripts/training/run_taiji_r2_daily_dialogue_continuation_extension.py",
        "budget_helper": Path(__file__).resolve(),
        "runner_contract_test": (
            ROOT / "tests/taiji_native/test_r2_dialogue_continuation_extension_runner_contract.py"
        ),
        "budget_contract_test": (
            ROOT / "tests/taiji_native/test_r2_dialogue_continuation_extension_contract.py"
        ),
        "source_validator": ROOT
        / "scripts/training/taiji_r2_dialogue_continuation_source_validation.py",
        "source_validator_contract_test": (
            ROOT / "tests/taiji_native/test_r2_dialogue_continuation_source_validation_contract.py"
        ),
        "source_revision_contract_test": (
            ROOT / "tests/taiji_native/test_r2_dialogue_continuation_source_revision_contract.py"
        ),
    }


def _validate_source_binding(binding: Any, *, verify_file_bytes: bool) -> None:
    expected_files = _expected_source_revision_files()
    expected_keys = {"runner_path", "runner_sha256", "files", "report_path", "report_sha256"}
    if not isinstance(binding, Mapping) or set(binding) != expected_keys:
        raise ExtensionLedgerRefuse("source revision continuation binding is malformed")
    files = binding.get("files")
    if not isinstance(files, Mapping) or set(files) != set(expected_files):
        raise ExtensionLedgerRefuse("source revision tracked file set differs")
    normalized: dict[str, tuple[Path, str]] = {}
    for name, expected_path in expected_files.items():
        row = files.get(name)
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ExtensionLedgerRefuse(f"source revision file binding is malformed: {name}")
        path_value = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(path_value, str)
            or not isinstance(digest, str)
            or not _HEX_256_RE.fullmatch(digest)
        ):
            raise ExtensionLedgerRefuse(f"source revision file binding is invalid: {name}")
        expected_resolved = expected_path.resolve()
        try:
            actual_resolved = Path(path_value).resolve(strict=True)
        except OSError as exc:
            raise ExtensionLedgerRefuse(f"source revision file is missing: {name}") from exc
        if actual_resolved != expected_resolved or not expected_resolved.is_file():
            raise ExtensionLedgerRefuse(f"source revision file path differs: {name}")
        if expected_resolved.is_symlink():
            raise ExtensionLedgerRefuse(f"source revision file must not be a symlink: {name}")
        if verify_file_bytes and _sha256_file(expected_resolved) != digest:
            raise ExtensionLedgerRefuse(f"source revision file hash differs: {name}")
        normalized[name] = (expected_resolved, digest)

    runner_path, runner_digest = normalized["runner"]
    if (
        binding.get("runner_path") != str(runner_path)
        or binding.get("runner_sha256") != runner_digest
    ):
        raise ExtensionLedgerRefuse("source revision runner binding differs from tracked file row")
    report_path_value = binding.get("report_path")
    report_digest = binding.get("report_sha256")
    expected_report = _SOURCE_REVISION_ARCHITECTURE_REPORT.resolve()
    if (
        report_path_value != str(expected_report)
        or not isinstance(report_digest, str)
        or not _HEX_256_RE.fullmatch(report_digest)
        or not expected_report.is_file()
        or expected_report.is_symlink()
    ):
        raise ExtensionLedgerRefuse("source revision architecture report binding is invalid")
    if verify_file_bytes and _sha256_file(expected_report) != report_digest:
        raise ExtensionLedgerRefuse("source revision architecture report hash differs")


def _validate_prior_v1_1(contract_sha256: str) -> str:
    path = _source_revision_path("v1.1")
    if path.is_symlink() or not path.is_file():
        raise ExtensionLedgerRefuse("v1.2 budget amendment requires the v1.1 source revision")
    previous = _read_json(path)
    if (
        previous.get("schema") != _SOURCE_REVISION_V1_1_SCHEMA
        or previous.get("contract_sha256") != contract_sha256
        or previous.get("approved_budget")
        != {
            "per_arm": BASE_PER_ARM_CPU_LIMIT,
            "aggregate": BASE_AGGREGATE_CPU_LIMIT,
        }
    ):
        raise ExtensionLedgerRefuse("v1.1 source revision does not bind the immutable base budget")
    _validate_source_binding(previous.get("current_continuation_sources"), verify_file_bytes=False)
    review_sha = previous.get("review_report_sha256")
    if (
        not isinstance(review_sha, str)
        or not _HEX_256_RE.fullmatch(review_sha)
        or not _SOURCE_REVISION_V1_1_REVIEW.is_file()
        or _SOURCE_REVISION_V1_1_REVIEW.is_symlink()
        or _sha256_file(_SOURCE_REVISION_V1_1_REVIEW) != review_sha
    ):
        raise ExtensionLedgerRefuse("v1.1 source revision review report binding differs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_budget_limits() -> tuple[float, float]:
    """Return the immutable v1 caps or the validated v1.2 approved caps.

    The v1.2 source revision is the sole budget-amendment artifact. Its absence
    keeps the original v1 ceiling; a present but invalid revision fails closed.
    """
    v13_path = _source_revision_path("v1.3")
    revision_path = v13_path if v13_path.exists() else _source_revision_path("v1.2")
    revision_key = str(revision_path.resolve())
    if not revision_path.exists():
        if revision_key in _ACTIVE_BUDGET_REVISION_SHA256:
            raise ExtensionLedgerRefuse("active v1.2 budget amendment disappeared")
        return BASE_PER_ARM_CPU_LIMIT, BASE_AGGREGATE_CPU_LIMIT
    if revision_path.is_symlink() or not revision_path.is_file():
        raise ExtensionLedgerRefuse("v1.2 budget amendment must be a regular write-once file")

    raw_bytes = revision_path.read_bytes()
    revision_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    previous_sha256 = _ACTIVE_BUDGET_REVISION_SHA256.get(revision_key)
    if previous_sha256 is not None and previous_sha256 != revision_sha256:
        raise ExtensionLedgerRefuse("v1.2 budget amendment changed after activation")
    try:
        revision = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtensionLedgerRefuse("v1.2 budget amendment is not valid JSON") from exc
    if not isinstance(revision, dict):
        raise ExtensionLedgerRefuse("v1.2 budget amendment must be a JSON object")

    _base_contract()
    contract_sha256 = _sha256_file(CONTRACT_PATH)
    prior_sha256 = _validate_prior_v1_1(contract_sha256)
    expected_schema = _SOURCE_REVISION_V1_2_SCHEMA
    review_path = _SOURCE_REVISION_V1_2_REVIEW
    if revision_path == v13_path:
        prior_path = _source_revision_path("v1.2")
        prior = _read_json(prior_path)
        if (
            prior.get("schema") != _SOURCE_REVISION_V1_2_SCHEMA
            or prior.get("contract_sha256") != contract_sha256
            or prior.get("previous_source_revision_sha256") != prior_sha256
            or prior.get("approved_budget") != {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0}
            or prior.get("review_report_sha256") != _sha256_file(_SOURCE_REVISION_V1_2_REVIEW)
        ):
            raise ExtensionLedgerRefuse("v1.3 predecessor differs")
        _validate_source_binding(prior.get("current_continuation_sources"), verify_file_bytes=False)
        prior_sha256 = _sha256_file(prior_path)
        expected_schema = "taiji-r2-continuation-source-revision-v1.3"
        review_path = (
            ROOT
            / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-boot-accounting-v1-3.md"
        )
        audit = (
            ROOT
            / "reports/taiji_r2_daily_dialogue_recovery_v2_20261021_boot_accounting_audit_v1.json"
        )
        expected_boot_sources = {
            str((ROOT / rel).resolve()): _sha256_file(ROOT / rel)
            for rel in (
                "scripts/training/taiji_r2_dialogue_boot_accounting.py",
                "tests/taiji_native/test_r2_dialogue_boot_accounting_contract.py",
            )
        }
        if (
            revision.get("boot_audit_path") != str(audit.resolve())
            or revision.get("boot_audit_sha256") != _sha256_file(audit)
            or revision.get("boot_accounting_sources") != expected_boot_sources
        ):
            raise ExtensionLedgerRefuse("v1.3 boot audit or source binding differs")
    expected_budget = {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0}
    budget = revision.get("approved_budget")
    if (
        revision.get("schema") != expected_schema
        or revision.get("contract_sha256") != contract_sha256
        or revision.get("previous_source_revision_sha256") != prior_sha256
        or not isinstance(budget, Mapping)
        or set(budget) != {"per_arm", "aggregate"}
        or any(not _finite_nonnegative(value) for value in budget.values())
        or dict(budget) != expected_budget
        or budget["per_arm"] <= BASE_PER_ARM_CPU_LIMIT
        or budget["aggregate"] <= BASE_AGGREGATE_CPU_LIMIT
    ):
        raise ExtensionLedgerRefuse("v1.2 budget amendment schema, lineage, or caps differ")
    _validate_source_binding(revision.get("current_continuation_sources"), verify_file_bytes=True)
    review_sha = revision.get("review_report_sha256")
    if (
        not isinstance(review_sha, str)
        or not _HEX_256_RE.fullmatch(review_sha)
        or not review_path.is_file()
        or review_path.is_symlink()
        or _sha256_file(review_path) != review_sha
    ):
        raise ExtensionLedgerRefuse("v1.2 source revision review report binding differs")
    _ACTIVE_BUDGET_REVISION_SHA256[revision_key] = revision_sha256
    return float(budget["per_arm"]), float(budget["aggregate"])


def _read_contract() -> dict[str, Any]:
    contract = _base_contract()
    effective_budget_limits()
    return contract


def _validate_charges(charges: Mapping[str, Any]) -> dict[str, float]:
    if set(charges) != set(ARMS):
        raise ExtensionLedgerRefuse("aggregate ledger arm charges are incomplete")
    normalized: dict[str, float] = {}
    for arm in ARMS:
        value = charges[arm]
        if not _finite_nonnegative(value) or float(value) > PER_ARM_CPU_LIMIT:
            raise ExtensionLedgerRefuse(f"aggregate ledger charge is invalid: {arm}")
        normalized[arm] = float(value)
    if sum(normalized.values()) > AGGREGATE_CPU_LIMIT:
        raise ExtensionLedgerRefuse("aggregate ledger charge exceeds aggregate limit")
    return normalized


def _validate_reservations(
    reservations: Mapping[str, Any], charges: Mapping[str, float]
) -> dict[str, dict[str, Any]]:
    if not isinstance(reservations, Mapping) or not set(reservations).issubset(ARMS):
        raise ExtensionLedgerRefuse("aggregate reservation map is malformed")
    normalized: dict[str, dict[str, Any]] = {}
    for arm, raw in reservations.items():
        if not isinstance(raw, Mapping):
            raise ExtensionLedgerRefuse(f"reservation record is malformed: {arm}")
        reservation_id = raw.get("reservation_id")
        reserved = raw.get("reserved_cpu_seconds")
        if not isinstance(reservation_id, str) or not re.fullmatch(r"[0-9a-f]{32}", reservation_id):
            raise ExtensionLedgerRefuse(f"reservation id is invalid: {arm}")
        if not _finite_nonnegative(reserved) or float(reserved) <= 0:
            raise ExtensionLedgerRefuse(f"reservation CPU upper bound is invalid: {arm}")
        reserved = float(reserved)
        if float(charges[arm]) + reserved >= PER_ARM_CPU_LIMIT:
            raise ExtensionLedgerRefuse(f"reserved CPU exceeds per-arm headroom: {arm}")
        normalized[arm] = {
            "reservation_id": reservation_id,
            "reserved_cpu_seconds": reserved,
        }
    if (
        sum(charges.values())
        + sum(float(item["reserved_cpu_seconds"]) for item in normalized.values())
        > AGGREGATE_CPU_LIMIT
    ):
        raise ExtensionLedgerRefuse("charges plus reservations exceed aggregate CPU limit")
    return normalized


def _summary_payload(
    contract: Mapping[str, Any],
    charges: Mapping[str, float],
    reservations: Mapping[str, Mapping[str, Any]],
    sequence: int,
    event_sha: str,
) -> dict[str, Any]:
    summary = {
        "schema": SUMMARY_SCHEMA,
        "sequence": sequence,
        "charges_by_arm": dict(charges),
        "reservations_by_arm": {arm: dict(value) for arm, value in reservations.items()},
        "latest_event_sha256": event_sha,
        "contract_sha256": contract["contract_sha256"],
    }
    summary["summary_sha256"] = _digest(summary)
    return summary


def _write_summary(
    contract: Mapping[str, Any],
    charges: Mapping[str, float],
    reservations: Mapping[str, Mapping[str, Any]],
    sequence: int,
    event_sha: str,
    qualification: Mapping[str, Any],
    api: Any,
) -> None:
    summary = _summary_payload(contract, charges, reservations, sequence, event_sha)
    _durable_write(
        AGGREGATE_LEDGER_SUMMARY,
        summary,
        qualification,
        api,
        replace=AGGREGATE_LEDGER_SUMMARY.exists(),
    )


def _read_summary_fast(
    *, api: Any | None = None
) -> tuple[dict[str, Any], dict[str, float], dict[str, dict[str, Any]], str]:
    api = api or _durability_api()
    _qualify_reports_directory(api)
    contract = _read_contract()
    if any(AGGREGATE_LEDGER_PREFIX.parent.glob(f".{AGGREGATE_LEDGER_PREFIX.name}.*.tmp")):
        raise ExtensionLedgerRefuse("uncommitted aggregate ledger temp file requires review")
    summary = _read_json(AGGREGATE_LEDGER_SUMMARY)
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("contract_sha256") != contract["contract_sha256"]
        or summary.get("summary_sha256") != _digest(summary, omit="summary_sha256")
    ):
        raise ExtensionLedgerRefuse("aggregate ledger summary is invalid")
    sequence = summary.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ExtensionLedgerRefuse("aggregate ledger summary sequence is invalid")
    event_path = _event_path(AGGREGATE_LEDGER_PREFIX, sequence)
    event = _read_json(event_path)
    event_sha = event.get("event_sha256")
    if (
        event.get("sequence") != sequence
        or event.get("schema") != EVENT_SCHEMA
        or event_sha != summary.get("latest_event_sha256")
        or event_sha != _digest(event, omit="event_sha256")
    ):
        raise ExtensionLedgerRefuse("aggregate ledger summary does not match its latest event")
    # An event newer than the summary means a writer stopped between durable
    # event publication and summary replacement. Full verification repairs it.
    if _event_path(AGGREGATE_LEDGER_PREFIX, sequence + 1).exists():
        raise ExtensionLedgerRefuse(
            "aggregate ledger summary is stale; full verification is required"
        )
    charges = _validate_charges(summary.get("charges_by_arm", {}))
    reservations = _validate_reservations(summary.get("reservations_by_arm", {}), charges)
    if sequence == 0:
        if (
            event.get("event") != "extension_budget_initialized"
            or event.get("charges_by_arm") != charges
            or reservations
        ):
            raise ExtensionLedgerRefuse("aggregate ledger initialization summary differs")
    else:
        arm = event.get("arm")
        if arm not in ARMS:
            raise ExtensionLedgerRefuse("aggregate ledger latest event arm is invalid")
        if event.get("event") == "work_unit_reserved":
            active = reservations.get(arm)
            if (
                active is None
                or active.get("reservation_id") != event.get("reservation_id")
                or active.get("reserved_cpu_seconds") != event.get("reserved_cpu_seconds")
                or charges[arm] != float(event.get("charged_cpu_seconds"))
            ):
                raise ExtensionLedgerRefuse("latest reservation event differs from summary")
        else:
            value = event.get("charged_cpu_seconds")
            if (
                value is None
                or reservations.get(arm) is not None
                or charges[arm] != float(value)
                or event.get("reservation_id") is None
            ):
                raise ExtensionLedgerRefuse("latest settlement event differs from summary")
    return contract, charges, reservations, str(event_sha)


def _validate_boot_correction(event, previous_charge, prior_files):
    from scripts.training.taiji_r2_dialogue_boot_accounting import reconciled_host_cpu_upper

    revision = _read_json(_source_revision_path("v1.3"))
    audit_path = Path(revision["boot_audit_path"])
    if _sha256_file(audit_path) != revision["boot_audit_sha256"]:
        raise ExtensionLedgerRefuse("boot audit digest differs")
    audit = _read_json(audit_path)
    row = audit["rows"][event["arm"]]
    sequence = row["basis_sequence"]
    if sequence >= len(prior_files):
        raise ExtensionLedgerRefuse("boot audit basis is not earlier in chain")
    basis = _read_json(prior_files[sequence][1])
    if (
        basis.get("event") != "continuation_prepared"
        or basis.get("arm") != event["arm"]
        or basis.get("event_sha256") != row["basis_event_sha256"]
        or basis.get("charged_cpu_seconds") != row["basis_charge"]
        or basis.get("details", {}).get("accounting_snapshot") != row["basis_snapshot"]
        or previous_charge != row["current_ledger_charge"]
    ):
        raise ExtensionLedgerRefuse("boot correction basis or previous charge differs")
    for _, path in prior_files:
        prior = _read_json(path)
        if prior.get("event") in {
            "episode_completed",
            "durable_training_commit",
            "parent_checkpoint_restored",
        }:
            raise ExtensionLedgerRefuse("boot correction refuses training history")
        if (
            prior.get("event") == "boot_accounting_upper_reconciled"
            and prior.get("arm") == event["arm"]
        ):
            raise ExtensionLedgerRefuse("duplicate boot correction")
    expected = reconciled_host_cpu_upper(
        row["basis_charge"],
        row["basis_snapshot"],
        current_boot_identity=audit["current_boot_identity"],
        current_host_cpu_seconds=audit["current_host_cpu_seconds"],
    )
    if (
        event.get("charged_cpu_seconds") != expected
        or event.get("details", {}).get("audit_sha256") != revision["boot_audit_sha256"]
    ):
        raise ExtensionLedgerRefuse("boot correction upper differs from proof")


def reconcile_boot_accounting():
    api = _durability_api()
    qualification = _qualify_reports_directory(api)
    with _aggregate_lock(LOCK_PATH):
        contract, charges, reservations, previous_sha = _read_ledger_full(api=api)
        if reservations:
            raise ExtensionLedgerRefuse("boot correction requires settled writers")
        revision = _read_json(_source_revision_path("v1.3"))
        audit = _read_json(Path(revision["boot_audit_path"]))
        for arm in ARMS:
            files = _event_files(AGGREGATE_LEDGER_PREFIX)
            if any(
                _read_json(path).get("event") == "boot_accounting_upper_reconciled"
                and _read_json(path).get("arm") == arm
                for _, path in files
            ):
                continue
            row = audit["rows"][arm]
            entry = {
                "schema": EVENT_SCHEMA,
                "sequence": len(files),
                "event": "boot_accounting_upper_reconciled",
                "extension_id": EXTENSION_PREFIX,
                "arm": arm,
                "charged_cpu_seconds": row["resolved_host_cpu_upper"],
                "parent_attempt_sha256": contract["parent_records"][arm]["parent_attempt_sha256"],
                "parent_receipt_sha256": contract["parent_records"][arm]["parent_receipt_sha256"],
                "details": {
                    "audit_sha256": revision["boot_audit_sha256"],
                    "previous_charge": charges[arm],
                    "accounting_snapshot": {
                        "charged_cpu_seconds": row["resolved_host_cpu_upper"],
                        "boot_identity": audit["current_boot_identity"],
                        "host_cpu_seconds": audit["current_host_cpu_seconds"],
                        "logical_cpu_upper": row["basis_snapshot"]["logical_cpu_upper"],
                        "snapshot_utc": audit["utc"],
                    },
                },
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "previous_event_sha256": previous_sha,
            }
            _validate_boot_correction(entry, charges[arm], files)
            entry["event_sha256"] = _digest(entry)
            _durable_write(
                _event_path(AGGREGATE_LEDGER_PREFIX, len(files)), entry, qualification, api
            )
            charges[arm] = entry["charged_cpu_seconds"]
            previous_sha = entry["event_sha256"]
            _write_summary(contract, charges, {}, len(files), previous_sha, qualification, api)
    return charges


def _read_ledger_full(
    *, api: Any | None = None
) -> tuple[dict[str, Any], dict[str, float], dict[str, dict[str, Any]], str]:
    api = api or _durability_api()
    qualification = _qualify_reports_directory(api)
    contract = _read_contract()
    files = _event_files(AGGREGATE_LEDGER_PREFIX)
    if not files:
        raise ExtensionLedgerRefuse("extension budget initialization event is missing")
    charges: dict[str, float] = {}
    reservations: dict[str, dict[str, Any]] = {}
    previous_sha: str | None = None
    charge_history: list[dict[str, float]] = []
    reservation_history: list[dict[str, dict[str, Any]]] = []
    event_hash_history: list[str] = []
    for expected_sequence, (sequence, event_path) in enumerate(files):
        if sequence != expected_sequence:
            raise ExtensionLedgerRefuse("aggregate ledger event sequence is invalid")
        event = _read_json(event_path)
        if event.get("schema") != EVENT_SCHEMA or event.get("sequence") != sequence:
            raise ExtensionLedgerRefuse("aggregate ledger event schema/sequence mismatch")
        if event.get("previous_event_sha256") != previous_sha:
            raise ExtensionLedgerRefuse("aggregate ledger hash chain is broken")
        if event.get("event_sha256") != _digest(event, omit="event_sha256"):
            raise ExtensionLedgerRefuse("aggregate ledger event digest mismatch")
        if event.get("extension_id") != EXTENSION_PREFIX:
            raise ExtensionLedgerRefuse("aggregate ledger extension identity differs")
        if sequence == 0:
            if event.get("event") != "extension_budget_initialized":
                raise ExtensionLedgerRefuse("aggregate ledger has no initialization event")
            if event.get("source_compatibility_report_sha256") != contract.get(
                "source_compatibility_report_sha256"
            ) or event.get("contract_sha256") != contract.get("contract_sha256"):
                raise ExtensionLedgerRefuse("initial aggregate event is not bound to its contract")
            initial = event.get("charges_by_arm")
            if not isinstance(initial, dict) or set(initial) != set(ARMS):
                raise ExtensionLedgerRefuse("initial aggregate charges are incomplete")
            if initial != contract.get("initial_charges_by_arm"):
                raise ExtensionLedgerRefuse("initial event charges differ from contract")
            charges = {}
            reservations = {}
            for arm in ARMS:
                value = initial[arm]
                if not _finite_nonnegative(value) or float(value) > PER_ARM_CPU_LIMIT:
                    raise ExtensionLedgerRefuse(f"initial charge is invalid: {arm}")
                charges[arm] = float(value)
        else:
            arm = event.get("arm")
            value = event.get("charged_cpu_seconds")
            if arm not in ARMS or not _finite_nonnegative(value):
                raise ExtensionLedgerRefuse("aggregate ledger charge event is malformed")
            parent = contract["parent_records"][arm]
            if event.get("parent_attempt_sha256") != parent.get(
                "parent_attempt_sha256"
            ) or event.get("parent_receipt_sha256") != parent.get("parent_receipt_sha256"):
                raise ExtensionLedgerRefuse("aggregate ledger charge lost parent lineage")
            value = float(value)
            if event.get("event") == "boot_accounting_upper_reconciled":
                if reservations:
                    raise ExtensionLedgerRefuse("boot correction requires no reservations")
                _validate_boot_correction(event, charges[arm], files[:sequence])
                charges[arm] = value
            elif event.get("event") == "work_unit_reserved":
                if arm in reservations:
                    raise ExtensionLedgerRefuse(f"arm already has an active reservation: {arm}")
                if value < charges[arm] or value > PER_ARM_CPU_LIMIT:
                    raise ExtensionLedgerRefuse(f"reservation charge is invalid: {arm}")
                charges[arm] = value
                reservations[arm] = {
                    "reservation_id": event.get("reservation_id"),
                    "reserved_cpu_seconds": event.get("reserved_cpu_seconds"),
                }
                reservations = _validate_reservations(reservations, charges)
            else:
                current = reservations.get(arm)
                if current is None or event.get("reservation_id") != current["reservation_id"]:
                    raise ExtensionLedgerRefuse(f"reservation settlement token differs: {arm}")
                if event.get("reserved_cpu_seconds") != current["reserved_cpu_seconds"]:
                    raise ExtensionLedgerRefuse(
                        f"reservation settlement upper bound differs: {arm}"
                    )
                if value < charges[arm] or value > PER_ARM_CPU_LIMIT:
                    raise ExtensionLedgerRefuse(f"settled CPU charge is invalid: {arm}")
                charges[arm] = value
                del reservations[arm]
        if (
            sum(charges.values())
            + sum(float(item["reserved_cpu_seconds"]) for item in reservations.values())
            > AGGREGATE_CPU_LIMIT
        ):
            raise ExtensionLedgerRefuse("charges plus reservations exceed aggregate limit")
        previous_sha = event["event_sha256"]
        charge_history.append(dict(charges))
        reservation_history.append({arm: dict(value) for arm, value in reservations.items()})
        event_hash_history.append(previous_sha)
    initial_event = _read_json(files[0][1])
    if (
        initial_event.get("contract_file_sha256")
        != hashlib.sha256(_canonical_json(contract)).hexdigest()
    ):
        raise ExtensionLedgerRefuse("initial budget event is not bound to the contract file")
    final_sequence = files[-1][0]
    publish_summary = True
    if AGGREGATE_LEDGER_SUMMARY.exists():
        summary = _read_json(AGGREGATE_LEDGER_SUMMARY)
        summary_sequence = summary.get("sequence")
        if (
            summary.get("schema") != SUMMARY_SCHEMA
            or summary.get("contract_sha256") != contract["contract_sha256"]
            or summary.get("summary_sha256") != _digest(summary, omit="summary_sha256")
            or not isinstance(summary_sequence, int)
            or isinstance(summary_sequence, bool)
            or summary_sequence < 0
            or summary_sequence > final_sequence
        ):
            raise ExtensionLedgerRefuse("aggregate ledger summary points beyond event history")
        if (
            summary.get("charges_by_arm") != charge_history[summary_sequence]
            or summary.get("reservations_by_arm") != reservation_history[summary_sequence]
            or summary.get("latest_event_sha256") != event_hash_history[summary_sequence]
        ):
            raise ExtensionLedgerRefuse("aggregate ledger summary disagrees with event history")
        publish_summary = summary_sequence < final_sequence
    if publish_summary:
        _write_summary(
            contract,
            charges,
            reservations,
            final_sequence,
            previous_sha or "",
            qualification,
            api,
        )
    return contract, charges, reservations, previous_sha or ""


def _read_ledger_unlocked(
    *, api: Any | None = None
) -> tuple[dict[str, Any], dict[str, float], dict[str, dict[str, Any]], str]:
    try:
        return _read_summary_fast(api=api)
    except (FileNotFoundError, ExtensionLedgerRefuse):
        return _read_ledger_full(api=api)


def _validate_initialization_inputs(
    parent_records: Mapping[str, Mapping[str, Any]],
    source_report_sha256: str,
    approved_utc: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(source_report_sha256, str) or not _HEX_256_RE.fullmatch(source_report_sha256):
        raise ExtensionLedgerRefuse("source compatibility report SHA-256 is invalid")
    if not isinstance(approved_utc, str) or not approved_utc.strip():
        raise ExtensionLedgerRefuse("approved_utc is required")
    return _validate_parent_records(parent_records)


def _initialization_event(contract: Mapping[str, Any], contract_file_sha256: str) -> dict[str, Any]:
    event = {
        "schema": EVENT_SCHEMA,
        "sequence": 0,
        "event": "extension_budget_initialized",
        "extension_id": EXTENSION_PREFIX,
        "charges_by_arm": contract["initial_charges_by_arm"],
        "source_compatibility_report_sha256": contract["source_compatibility_report_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "contract_file_sha256": contract_file_sha256,
        "created_utc": contract["created_utc"],
        "previous_event_sha256": None,
    }
    event["event_sha256"] = _digest(event)
    return event


def _initialization_artifacts() -> list[tuple[int, Path]]:
    if any(CONTRACT_PATH.parent.glob(f".{CONTRACT_PATH.name}.*.tmp")):
        raise ExtensionLedgerRefuse("uncommitted initialization contract temp requires review")
    files = _event_files(AGGREGATE_LEDGER_PREFIX)
    allowed_names = {AGGREGATE_LEDGER_SUMMARY.name, LOCK_PATH.name}
    for path in AGGREGATE_LEDGER_PREFIX.parent.glob(f"{AGGREGATE_LEDGER_PREFIX.name}.*"):
        if path.name in allowed_names:
            continue
        match = _EVENT_RE.fullmatch(path.name)
        if match is None or match.group("prefix") != AGGREGATE_LEDGER_PREFIX.name:
            raise ExtensionLedgerRefuse(
                f"unknown aggregate initialization artifact requires review: {path.name}"
            )
    return files


def initialize_ledger(
    parent_records: Mapping[str, Mapping[str, Any]],
    source_report_sha256: str,
    approved_utc: str,
) -> dict[str, Any]:
    """Write the one-time contract and initial aggregate charge event.

    Must complete before any parent checkpoint restore worker or training step.
    Repeated initialization is refused; callers must use ``latest_charges`` to
    resume the same extension window.
    """
    if _source_revision_path("v1.2").exists():
        raise ExtensionLedgerRefuse("cannot initialize the immutable v1 contract after v1.2")
    parents = _validate_initialization_inputs(parent_records, source_report_sha256, approved_utc)
    charges = {arm: float(parents[arm]["charged_cpu_seconds"]) for arm in ARMS}
    api = _durability_api()
    qualification = _qualify_reports_directory(api)
    if not CONTRACT_PATH.parent.is_dir():
        raise ExtensionLedgerRefuse("reports directory must exist before ledger initialization")
    with _aggregate_lock(LOCK_PATH):
        if CONTRACT_PATH.exists() or _event_files(AGGREGATE_LEDGER_PREFIX):
            raise FileExistsError("R2 extension budget ledger is write-once")
        contract = {
            "schema": CONTRACT_SCHEMA,
            "extension_id": EXTENSION_PREFIX,
            "arms": list(ARMS),
            "per_arm_cpu_seconds": BASE_PER_ARM_CPU_LIMIT,
            "aggregate_cpu_seconds": BASE_AGGREGATE_CPU_LIMIT,
            "approved_utc": approved_utc,
            "source_compatibility_report_sha256": source_report_sha256,
            "parent_records": parents,
            "initial_charges_by_arm": charges,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "durability": qualification,
        }
        contract["contract_sha256"] = _digest(contract)
        contract_sha = _durable_write(CONTRACT_PATH, contract, qualification, api)
        initial = _initialization_event(contract, contract_sha)
        _durable_write(_event_path(AGGREGATE_LEDGER_PREFIX, 0), initial, qualification, api)
        _write_summary(contract, charges, {}, 0, initial["event_sha256"], qualification, api)
    return {"contract": contract, "charges_by_arm": charges, "contract_file_sha256": contract_sha}


def reconcile_initialize_ledger(
    parent_records: Mapping[str, Mapping[str, Any]],
    source_report_sha256: str,
    approved_utc: str,
) -> dict[str, Any]:
    """Verify a matching initialization and repair only its missing event zero.

    This is safe to retry after an interrupted initialization. It requires the
    original durable contract, rejects unknown/partial artifacts, and never
    replaces an existing event. A complete ledger is fully validated and
    returned unchanged; a missing fast summary may be rebuilt from its verified
    immutable event chain.
    """
    parents = _validate_initialization_inputs(parent_records, source_report_sha256, approved_utc)
    api = _durability_api()
    qualification = _qualify_reports_directory(api)
    if not CONTRACT_PATH.is_file():
        raise ExtensionLedgerRefuse(
            "initialization contract is missing; reconciliation requires a durable contract"
        )
    with _aggregate_lock(LOCK_PATH):
        files = _initialization_artifacts()
        contract = _read_contract()
        if (
            contract.get("source_compatibility_report_sha256") != source_report_sha256
            or contract.get("approved_utc") != approved_utc
            or contract.get("parent_records") != parents
            or contract.get("initial_charges_by_arm")
            != {arm: float(parents[arm]["charged_cpu_seconds"]) for arm in ARMS}
        ):
            raise ExtensionLedgerRefuse("initialization inputs differ from the durable contract")
        durability = contract.get("durability")
        if (
            not isinstance(durability, Mapping)
            or not isinstance(durability.get("volume_root"), str)
            or _normalize_volume_root(durability["volume_root"])
            != _normalize_volume_root(qualification["volume_root"])
        ):
            raise ExtensionLedgerRefuse(
                "initialization contract volume differs from the current reports volume"
            )
        contract_sha = _file_sha256(CONTRACT_PATH)
        if not files:
            if AGGREGATE_LEDGER_SUMMARY.exists():
                raise ExtensionLedgerRefuse(
                    "initialization summary exists without immutable event zero"
                )
            if not isinstance(contract.get("created_utc"), str) or not contract["created_utc"]:
                raise ExtensionLedgerRefuse("initialization contract timestamp is invalid")
            initial = _initialization_event(contract, contract_sha)
            _durable_write(
                _event_path(AGGREGATE_LEDGER_PREFIX, 0),
                initial,
                qualification,
                api,
            )
            charges = {arm: float(contract["initial_charges_by_arm"][arm]) for arm in ARMS}
            _validate_charges(charges)
            _write_summary(
                contract,
                charges,
                {},
                0,
                initial["event_sha256"],
                qualification,
                api,
            )
        verified_contract, current_charges, reservations, _ = _read_ledger_full(api=api)
        if verified_contract != contract:
            raise ExtensionLedgerRefuse("verified initialization contract changed")
        return {
            "contract": contract,
            "charges_by_arm": {arm: float(contract["initial_charges_by_arm"][arm]) for arm in ARMS},
            "contract_file_sha256": contract_sha,
            "current_charges_by_arm": current_charges,
            "reservations_by_arm": reservations,
        }


def latest_charges() -> dict[str, float]:
    """Read cumulative charges, including every active reservation's upper bound."""
    with _aggregate_lock(LOCK_PATH):
        _, charges, reservations, _ = _read_ledger_full()
    effective = dict(charges)
    for arm, reservation in reservations.items():
        effective[arm] += float(reservation["reserved_cpu_seconds"])
    return effective


def latest_charges_fast() -> dict[str, float]:
    """Read the verified summary tip, replaying the full chain only if stale."""
    with _aggregate_lock(LOCK_PATH):
        _, charges, reservations, _ = _read_ledger_unlocked()
    effective = dict(charges)
    for arm, reservation in reservations.items():
        effective[arm] += float(reservation["reserved_cpu_seconds"])
    return effective


def latest_reservations() -> dict[str, dict[str, Any]]:
    """Return active per-arm reservation tokens after a full chain audit."""
    with _aggregate_lock(LOCK_PATH):
        _, _, reservations, _ = _read_ledger_full()
    return {arm: dict(value) for arm, value in reservations.items()}


def append_charge(
    arm: str,
    event: str,
    charged_cpu_seconds: float,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """Settle the current reservation with an actual cumulative CPU snapshot.

    ``details["reservation_id"]`` must match the active arm token. The settled
    charge and reservation release are one durable event.
    """
    if arm not in ARMS:
        raise ExtensionLedgerRefuse(f"unknown R2 extension arm: {arm}")
    if not isinstance(event, str) or not event or len(event) > 128:
        raise ExtensionLedgerRefuse("ledger event name is invalid")
    if not _finite_nonnegative(charged_cpu_seconds):
        raise ExtensionLedgerRefuse("charged CPU seconds must be finite and nonnegative")
    if not isinstance(details, Mapping):
        raise ExtensionLedgerRefuse("ledger event details must be an object")
    try:
        json.dumps(details, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExtensionLedgerRefuse("ledger event details are not strict JSON") from exc
    api = _durability_api()
    qualification = _qualify_reports_directory(api)
    with _aggregate_lock(LOCK_PATH):
        contract, charges, reservations, previous_sha = _read_ledger_unlocked(api=api)
        active = reservations.get(arm)
        reservation_id = details.get("reservation_id")
        if active is None:
            raise ExtensionLedgerRefuse(f"no active CPU reservation to settle: {arm}")
        if reservation_id != active["reservation_id"]:
            raise ExtensionLedgerRefuse(f"reservation token mismatch: {arm}")
        if event == "work_unit_reserved":
            raise ExtensionLedgerRefuse("work-unit reservations must be created by guard")
        charge = float(charged_cpu_seconds)
        if charge < charges[arm]:
            raise ExtensionLedgerRefuse(f"CPU charge cannot decrease for {arm}")
        if charge > PER_ARM_CPU_LIMIT:
            raise ExtensionLedgerRefuse(f"CPU charge exceeds per-arm cap for {arm}")
        candidate = dict(charges)
        candidate[arm] = charge
        candidate_reservations = {key: dict(value) for key, value in reservations.items()}
        del candidate_reservations[arm]
        if (
            sum(candidate.values())
            + sum(float(item["reserved_cpu_seconds"]) for item in candidate_reservations.values())
            > AGGREGATE_CPU_LIMIT
        ):
            raise ExtensionLedgerRefuse("CPU charge exceeds aggregate cap")
        files = _event_files(AGGREGATE_LEDGER_PREFIX)
        sequence = len(files)
        entry = {
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "event": event,
            "extension_id": EXTENSION_PREFIX,
            "arm": arm,
            "charged_cpu_seconds": charge,
            "parent_attempt_sha256": contract["parent_records"][arm]["parent_attempt_sha256"],
            "parent_receipt_sha256": contract["parent_records"][arm]["parent_receipt_sha256"],
            "reservation_id": reservation_id,
            "reserved_cpu_seconds": active["reserved_cpu_seconds"],
            "details": dict(details),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "previous_event_sha256": previous_sha,
        }
        entry["event_sha256"] = _digest(entry)
        file_sha = _durable_write(
            _event_path(AGGREGATE_LEDGER_PREFIX, sequence), entry, qualification, api
        )
        _write_summary(
            contract,
            candidate,
            candidate_reservations,
            sequence,
            entry["event_sha256"],
            qualification,
            api,
        )
    return {**entry, "event_file_sha256": file_sha}


def guard(arm: str, next_charge: float, reserve_cpu_seconds: float = 0.0) -> str:
    """Atomically reserve CPU for the next bounded work unit and return its token."""
    if arm not in ARMS:
        raise ExtensionLedgerRefuse(f"unknown R2 extension arm: {arm}")
    if (
        not _finite_nonnegative(next_charge)
        or not _finite_nonnegative(reserve_cpu_seconds)
        or float(reserve_cpu_seconds) <= 0
    ):
        raise ExtensionLedgerRefuse("budget guard requires finite CPU charge and positive reserve")
    api = _durability_api()
    qualification = _qualify_reports_directory(api)
    with _aggregate_lock(LOCK_PATH):
        contract, charges, reservations, previous_sha = _read_ledger_unlocked(api=api)
        if arm in reservations:
            raise ExtensionLedgerRefuse(
                f"prior reservation must be settled before next guard: {arm}"
            )
        charge = float(next_charge)
        reserve = float(reserve_cpu_seconds)
        if charge < charges[arm]:
            raise ExtensionLedgerRefuse(f"budget guard charge regressed for {arm}")
        if charge + reserve >= PER_ARM_CPU_LIMIT:
            raise ExtensionLedgerRefuse(f"insufficient per-arm CPU headroom for {arm}")
        candidate_charges = dict(charges)
        candidate_charges[arm] = charge
        candidate_reservations = {key: dict(value) for key, value in reservations.items()}
        reservation_id = secrets.token_hex(16)
        candidate_reservations[arm] = {
            "reservation_id": reservation_id,
            "reserved_cpu_seconds": reserve,
        }
        total_reserved = sum(candidate_charges.values()) + sum(
            float(item["reserved_cpu_seconds"]) for item in candidate_reservations.values()
        )
        if total_reserved > AGGREGATE_CPU_LIMIT:
            raise ExtensionLedgerRefuse("insufficient aggregate CPU headroom")
        sequence = len(_event_files(AGGREGATE_LEDGER_PREFIX))
        entry = {
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "event": "work_unit_reserved",
            "extension_id": EXTENSION_PREFIX,
            "arm": arm,
            "charged_cpu_seconds": charge,
            "reservation_id": reservation_id,
            "reserved_cpu_seconds": reserve,
            "parent_attempt_sha256": contract["parent_records"][arm]["parent_attempt_sha256"],
            "parent_receipt_sha256": contract["parent_records"][arm]["parent_receipt_sha256"],
            "details": {},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "previous_event_sha256": previous_sha,
        }
        entry["event_sha256"] = _digest(entry)
        _durable_write(_event_path(AGGREGATE_LEDGER_PREFIX, sequence), entry, qualification, api)
        _write_summary(
            contract,
            candidate_charges,
            candidate_reservations,
            sequence,
            entry["event_sha256"],
            qualification,
            api,
        )
    return reservation_id


# A fresh process adopts a previously published v1.2 amendment before it can
# read, reserve, reconcile, or settle any extension budget.
PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT = effective_budget_limits()


__all__ = [
    "AGGREGATE_CPU_LIMIT",
    "ARMS",
    "BASE_AGGREGATE_CPU_LIMIT",
    "BASE_PER_ARM_CPU_LIMIT",
    "CONTRACT_PATH",
    "EXTENSION_CHECKPOINT_ROOT",
    "ExtensionLedgerRefuse",
    "PER_ARM_CPU_LIMIT",
    "append_charge",
    "durable_publish",
    "effective_budget_limits",
    "guard",
    "initialize_ledger",
    "latest_charges",
    "latest_charges_fast",
    "latest_reservations",
    "paths",
    "qualify_storage_target",
    "reconcile_initialize_ledger",
]

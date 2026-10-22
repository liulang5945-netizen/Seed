"""Continue the three R2 arms from immutable parent checkpoints under a new budget.

This entry point is intentionally separate from the frozen recovery runner. It
never touches dev/reference data or the evaluator. Use ``--prepare-only`` once
to create the three-arm extension contract, then explicitly use ``--run`` or
``--resume`` for one arm.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROCESS_START_CPU = time.process_time()

import psutil
import torch

from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as base
from scripts.training import run_taiji_r2_daily_dialogue_recovery as frozen
from scripts.training import taiji_r2_dialogue_continuation_extension as continuation_budget
from scripts.training import taiji_r2_dialogue_continuation_source_validation as source_validation
from scripts.training import taiji_r2_dialogue_recovery as recovery
from scripts.training.taiji_r2_dialogue_continuation_extension import (
    AGGREGATE_CPU_LIMIT,
    EXTENSION_CHECKPOINT_ROOT,
    PER_ARM_CPU_LIMIT,
    REPORTS_DIR,
    append_charge,
    durable_publish,
    initialize_ledger,
    latest_charges,
    latest_charges_fast,
    latest_reservations,
    paths,
    qualify_storage_target,
    reconcile_initialize_ledger,
)
from scripts.training.taiji_r2_dialogue_continuation_extension import (
    guard as ledger_guard,
)
from taiji import Taiji

ARMS = ("full_turn_1m", "warm_start", "fresh_same_capacity")
SAVE_INTERVAL = 32
LEDGER_EVENT_RESERVE_SECONDS = 2.0
TERMINAL_STOP_RESERVE_SECONDS = 10.0
THROUGHPUT_WINDOW_INTERVALS = 16
THROUGHPUT_SAFETY_FACTOR = 1.15
COMPLETION_CPU_RESERVE_SECONDS = 800.0
PREPARE_EVENT_RESERVE_SECONDS = 120.0
FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS = 120.0
FINAL_PUBLICATION_RESERVE_SECONDS = 30.0
APPROVED_UTC = "2026-10-22"
APPROVAL_SOURCE = (
    "user approved cumulative continuation ceilings: 1.5M CPU seconds per arm, 4.3M aggregate"
)
V1_2_PER_ARM_CPU_LIMIT = 1_500_000.0
V1_2_AGGREGATE_CPU_LIMIT = 4_300_000.0
ARCHITECTURE_REPORT = (
    ROOT
    / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-extension-architecture.md"
)
_APPROVED_VOLUME_ROOT: str | None = None

PARENT_ATTEMPT_SHA256 = {
    "full_turn_1m": "c3557687753a0ae6b27baaa365175f73157e442db0ab3a6a4d725b8f056c9692",
    "warm_start": "fff7ef55d72aef39778b8a781b13e7d18c1161253ceee21d7d3eb78aaf38f7e0",
    "fresh_same_capacity": "db2b3e5f311fbcb35669da54107351033c7c527bea7e9e69a98fd49d744ed2d4",
}
PARENT_RECEIPT_SHA256 = {
    "full_turn_1m": "8ec115196c39f9936ea3704ac5aaa8934f8efeaf8794f58e709f285ceb3a23bb",
    "warm_start": "f8a2a72eec4d67ca362351158f4d11d19aee5bd044cec5b2ed7a399396b07583",
    "fresh_same_capacity": "cdf91736fd6ea1c837ed075886c478b6df430f2494ecaf1163f81eefa6ad067e",
}
PARENT_BUNDLE_SHA256 = {
    "full_turn_1m": "d4d5887fd44b68d5e3d5e42a2d8832a83ca6495ec1f43f8f5847d28ad004c89d",
    "warm_start": "ed74609d541f0340f0d243e9e34b496cc3061427bc572923215399dfd309c416",
    "fresh_same_capacity": "ca00b48dd28d0466a8c9b20cb224341af75b0e4de6b0625d55534c5413b0a9dc",
}
PARENT_CHECKPOINT_DIGEST = {
    "full_turn_1m": "d2a42cc7b7323c599c5b4c5ec46f34649a20d579811933679714237e1a9b747d",
    "warm_start": "579e88acbcd98f83baa37e45f979558fa9707be5fb77178ddec1e7dc8be86d7c",
    "fresh_same_capacity": "4a68dd311b31589f9494b3d3199902283b47310a9dcce1b7732411acdb352fcf",
}
PARENT_DURABLE_EPISODES = {
    "full_turn_1m": 9_472,
    "warm_start": 14_528,
    "fresh_same_capacity": 14_528,
}
PARENT_DURABLE_UPDATES = {
    "full_turn_1m": 4_549_963,
    "warm_start": 7_054_531,
    "fresh_same_capacity": 7_054_531,
}
PARENT_CURSOR = {
    "full_turn_1m": {"epoch": 2, "index": 1_948},
    "warm_start": {"epoch": 3, "index": 3_242},
    "fresh_same_capacity": {"epoch": 3, "index": 3_242},
}

# The old parent SHA remains immutable. These exact Git revisions prove the
# historical byte source; current hashes pin the reviewed compatible files.
SOURCE_COMPATIBILITY = {
    "scripts/data_prep/build_taiji_r2_daily_dialogue_full_turn_pretraining_corpus.py": {
        "parent_sha256": "d88b126ffd3caeccddfc6094d182403e052af8dbde80358871a6378034e0dfd8",
        "current_sha256": "0173946126a4fdd304b5c55f6aa73342f7455205236cdc2bd08a9544cfc3116b",
        "git_revision": "02e82d921d14782fca511276020957220910a5c6",
    },
    "scripts/training/run_taiji_r2_daily_dialogue_full_turn_pretraining.py": {
        "parent_sha256": "4065f25ed47d3c823abc0cafeeb6f6d4a055986078f93fdd107cfa16b78aebef",
        "current_sha256": "69691de36b79841c56cc4e1783cfda7c8cd2b8137524e4ef5b1d48529b7013b9",
        "git_revision": "02e82d921d14782fca511276020957220910a5c6",
    },
    "scripts/training/run_taiji_r2_daily_dialogue_recovery.py": {
        "parent_sha256": "e84e48215b72d6ad212c25d3dab63aca119654d7b28c94fde6fa2bc3d98c9e04",
        "current_sha256": "92205705d740e7949ff933366f5de964d97d40b4d0cfe9ba0447b098c15ff81c",
        "git_revision": "85bfc59c8b60708525ff1e8f8c28f1d5ead1ba48",
    },
    "tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py": {
        "parent_sha256": "fe889be52580947c1241aadcfcde117e0bc0cd604636d41026c5df6b07f5696f",
        "current_sha256": "5e93253af290fa65eece542f4de18a677b052a9ce4df53de3d88506543da5f3a",
        "git_revision": "85bfc59c8b60708525ff1e8f8c28f1d5ead1ba48",
    },
    "tests/taiji_native/test_r2_dialogue_recovery_contract.py": {
        "parent_sha256": "4fd94e699a397b6f5b569aefae53b8f8bcb01eb953dfe029186b06718cbae40e",
        "current_sha256": "abc55fecbd3cf6b62bce86d17ef68e9a7eadbcd330c63daa43ea9626f78001de",
        "git_revision": "0669123da231d19993ec5fe795187791cf0dd628",
    },
    "scripts/training/run_taiji_r2_daily_dialogue_warmstart_matched.py": {
        "parent_sha256": "266ea28b5b4af8381a7633eb7aaa2be8cc75d7b758e1ab620dc4a281b3328e44",
        "current_sha256": "2aac78a86a5509dee6baf66b80a993d297c67758ba7668a760be64c0a9e74a02",
        "git_revision": "b6bb45b52702479ad20f91502bed1cb34766d324",
    },
    "tests/taiji_native/test_r2_daily_dialogue_warmstart_matched_contract.py": {
        "parent_sha256": "b12861083031c31646df19ee3d52c5626b98d7c7ded9399ade28719e94d6d041",
        "current_sha256": "827db17d6b014c6167230f329771d46d9d156b399b7ad3a63b5f57e0a4cce1cf",
        "git_revision": "b6bb45b52702479ad20f91502bed1cb34766d324",
    },
}


class Refuse(RuntimeError):
    """Fail closed when a source, parent, budget or durability gate is unclear."""


def _normalized_volume_root(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(value))).rstrip("\\/").casefold()


def _nearest_existing_directory(path: Path) -> Path:
    candidate = Path(path).resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise Refuse(f"no existing ancestor for storage target: {path}")
        candidate = parent
    return candidate if candidate.is_dir() else candidate.parent


def _is_within(path: Path, root: Path) -> bool:
    path_value = os.path.normcase(os.path.abspath(path))
    root_value = os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath((path_value, root_value)) == root_value
    except ValueError:
        return False


def _extension_publish(
    path: Path,
    writer,
    *,
    replace: bool = False,
) -> str:
    if _APPROVED_VOLUME_ROOT is None:
        raise Refuse("approved NTFS volume is not configured for extension writes")
    target = Path(path).resolve()
    if not (
        _is_within(target, Path(REPORTS_DIR).resolve())
        or _is_within(target, Path(EXTENSION_CHECKPOINT_ROOT).resolve())
    ):
        raise Refuse(f"extension publisher refused an out-of-scope path: {target}")
    return durable_publish(
        target,
        writer,
        replace=replace,
        expected_volume_root=_APPROVED_VOLUME_ROOT,
    )


def _extension_publish_json(path: Path, payload: dict[str, Any], *, replace: bool = False) -> str:
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return _extension_publish(path, lambda stream: stream.write(raw), replace=replace)


def _extension_atomic_save_final(path: Path, payload: dict[str, Any]) -> None:
    _extension_publish(path, lambda stream: torch.save(payload, stream))


def _install_extension_publisher() -> None:
    if _APPROVED_VOLUME_ROOT is None:
        raise Refuse("approved NTFS volume is not configured for extension writes")
    # RecoveryStore resolves its publisher through this module function. Bind
    # it only in the new extension process; frozen source files stay untouched.
    recovery._publish = _extension_publish
    recovery._publish_json = _extension_publish_json
    frozen.atomic_save_final = _extension_atomic_save_final


def _qualify_storage_targets(expected_volume_root: str | None = None) -> str:
    global _APPROVED_VOLUME_ROOT
    reports = qualify_storage_target(REPORTS_DIR, expected_volume_root)
    volume_root = reports.get("volume_root")
    if not isinstance(volume_root, str) or not volume_root:
        raise Refuse("reports directory did not yield an approved NTFS volume root")
    if expected_volume_root is not None and _normalized_volume_root(
        volume_root
    ) != _normalized_volume_root(expected_volume_root):
        raise Refuse("reports directory moved off the contract-approved volume")
    for arm in ARMS:
        target = paths(arm)["extension_checkpoint_root"]
        existing = target if target.is_dir() else _nearest_existing_directory(target)
        qualification = qualify_storage_target(existing, volume_root)
        if _normalized_volume_root(qualification.get("volume_root", "")) != _normalized_volume_root(
            volume_root
        ):
            raise Refuse(f"extension checkpoint target is on a different volume: {arm}")
    _APPROVED_VOLUME_ROOT = volume_root
    _install_extension_publisher()
    return volume_root


def _configure_prepared_storage() -> dict[str, Any]:
    contract_path = paths(ARMS[0])["contract_path"]
    contract = read_json(contract_path)
    if (
        contract.get("per_arm_cpu_seconds") != continuation_budget.BASE_PER_ARM_CPU_LIMIT
        or contract.get("aggregate_cpu_seconds") != continuation_budget.BASE_AGGREGATE_CPU_LIMIT
        or continuation_budget.effective_budget_limits() != (PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT)
    ):
        raise Refuse("runner and versioned cumulative budget amendment differ")
    durability = contract.get("durability")
    if not isinstance(durability, dict) or not isinstance(durability.get("volume_root"), str):
        raise Refuse("continuation contract lacks its approved storage volume")
    _qualify_storage_targets(durability["volume_root"])
    return contract


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_oid(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Refuse(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        if required:
            raise Refuse(f"required progress file is missing: {path}")
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise Refuse(f"progress has an incomplete final record: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise Refuse(f"blank progress record at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Refuse(f"progress record is not an object at {path}:{line_number}")
        records.append(value)
    return records


def append_durable_jsonl(path: Path, record: dict[str, Any]) -> None:
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path = Path(path)
    previous = path.read_bytes() if path.exists() else b""
    if previous and not previous.endswith(b"\n"):
        raise Refuse(f"progress has an incomplete final record: {path}")
    _extension_publish(
        path,
        lambda stream: stream.write(previous + raw),
        replace=path.exists(),
    )


@contextmanager
def _reclaim_guard_recovery_lock(directory: Path):
    lock_path = Path(directory) / ".extension-reclaim-guard-recovery.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise Refuse("reclaim guard recovery lock must not be a symlink")
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise Refuse("reclaim guard recovery is already in progress") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise Refuse("reclaim guard recovery is already in progress") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _identity_is_live(identity: dict[str, Any], *, label: str) -> bool:
    pid = identity.get("pid")
    create_time = identity.get("create_time")
    if (
        type(pid) is not int
        or pid <= 0
        or type(create_time) not in (int, float)
        or create_time <= 0
    ):
        raise Refuse(f"{label} process identity is malformed")
    try:
        owner = psutil.Process(pid)
        if owner.create_time() != float(create_time):
            return False
        return owner.is_running()
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied as exc:
        raise Refuse(f"cannot verify {label} process identity") from exc


def _guard_archive_path(directory: Path, token: str) -> Path:
    return Path(directory) / "stale-locks" / f"reclaim-guard-{token}.json"


def _dead_guard_archive_exists(directory: Path) -> bool:
    archive_directory = Path(directory) / "stale-locks"
    if not archive_directory.is_dir():
        return False
    for path in archive_directory.glob("reclaim-guard-*.json"):
        try:
            record = read_json(path)
            identity = record.get("guard")
            if (
                record.get("schema") == "taiji-r2-extension-dead-reclaim-guard-v1"
                and isinstance(identity, dict)
                and not _identity_is_live(identity, label="archived reclaim guard")
            ):
                return True
        except Refuse:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise Refuse(f"reclaim guard archive is malformed: {path}") from exc
    return False


def _dead_writer_archive_exists(directory: Path) -> bool:
    archive_directory = Path(directory) / "stale-locks"
    if not archive_directory.is_dir():
        return False
    for path in archive_directory.glob("writer-lock-*.json"):
        if path.is_symlink():
            raise Refuse(f"writer lock archive must not be a symlink: {path}")
        identity = read_json(path)
        token = identity.get("token")
        if (
            not isinstance(token, str)
            or len(token) != 32
            or any(char not in "0123456789abcdef" for char in token)
            or path.name != f"writer-lock-{token}.json"
        ):
            raise Refuse(f"writer lock archive identity is malformed: {path}")
        if not _identity_is_live(identity, label="archived writer lock"):
            return True
        raise Refuse(f"archived writer lock unexpectedly has a live owner: {path}")
    return False


def _recover_dead_reclaim_guard(directory: Path) -> bool:
    directory = Path(directory)
    guard_path = directory / "reclaim.guard"
    if not guard_path.exists():
        return False
    if guard_path.is_symlink():
        raise Refuse("reclaim guard must not be a symlink")
    with _reclaim_guard_recovery_lock(directory):
        if not guard_path.exists():
            return False
        raw = guard_path.read_bytes()
        try:
            identity = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Refuse("reclaim guard JSON is invalid") from exc
        if not isinstance(identity, dict):
            raise Refuse("reclaim guard identity is malformed")
        if _identity_is_live(identity, label="reclaim guard"):
            raise Refuse("reclaim already in progress; reclaim guard owner is live")
        lock_path = directory / "writer.lock"
        if lock_path.exists():
            if lock_path.is_symlink():
                raise Refuse("writer lock must not be a symlink")
            lock_identity = read_json(lock_path)
            if _identity_is_live(lock_identity, label="writer lock"):
                raise Refuse("writer lock owner is live; stale reclaim guard cannot be recovered")
        token = identity.get("token")
        if (
            not isinstance(token, str)
            or not token
            or any(char not in "0123456789abcdef" for char in token)
        ):
            raise Refuse("reclaim guard token is malformed")
        archive_directory = directory / "stale-locks"
        archive_directory.mkdir(parents=True, exist_ok=True)
        archive_path = _guard_archive_path(directory, token)
        archived = {
            "schema": "taiji-r2-extension-dead-reclaim-guard-v1",
            "guard": identity,
            "recovered_utc": utc_now(),
            "guard_sha256": sha256_bytes(raw),
        }
        if archive_path.exists():
            existing = read_json(archive_path)
            if (
                existing.get("schema") != archived["schema"]
                or existing.get("guard") != identity
                or existing.get("guard_sha256") != archived["guard_sha256"]
            ):
                raise Refuse("reclaim guard archive conflicts with the stale guard")
        else:
            _extension_publish_json(archive_path, archived)
        if guard_path.read_bytes() != raw:
            raise Refuse("reclaim guard identity changed during stale recovery")
        guard_path.unlink()
        return True


def _create_reclaim_guard(directory: Path) -> tuple[str, dict[str, Any]]:
    guard_path = Path(directory) / "reclaim.guard"
    if guard_path.exists() or guard_path.is_symlink():
        raise Refuse("reclaim guard already exists before this writer claim")
    token = uuid.uuid4().hex
    identity = {
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
        "token": token,
    }
    _extension_publish_json(guard_path, identity)
    return token, identity


def _archive_dead_writer_lock(directory: Path, lock_path: Path) -> None:
    if lock_path.is_symlink():
        raise Refuse("writer lock must not be a symlink")
    raw = lock_path.read_bytes()
    identity = read_json(lock_path)
    if _identity_is_live(identity, label="writer lock"):
        raise Refuse("writer lock owner is live")
    token = identity.get("token")
    if (
        not isinstance(token, str)
        or len(token) != 32
        or any(char not in "0123456789abcdef" for char in token)
    ):
        raise Refuse("writer lock token is malformed")
    archive_directory = Path(directory) / "stale-locks"
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive_path = archive_directory / f"writer-lock-{token}.json"
    if archive_path.exists():
        if archive_path.is_symlink() or archive_path.read_bytes() != raw:
            raise Refuse("writer lock archive conflicts with the stale lock")
    else:
        _extension_publish(archive_path, lambda stream: stream.write(raw))
    if lock_path.read_bytes() != raw:
        raise Refuse("writer lock identity changed during stale recovery")
    lock_path.unlink()


class _ExtensionRecoveryStore(recovery.RecoveryStore):
    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._check_reclaim_guard()
        lock_path = self.directory / "writer.lock"
        identity = {
            "pid": os.getpid(),
            "create_time": psutil.Process().create_time(),
            "token": self.token,
        }
        try:
            _extension_publish_json(lock_path, identity)
        except FileExistsError:
            raise Refuse("writer lock already exists during extension claim") from None
        try:
            self._check_reclaim_guard()
        except BaseException:
            if lock_path.is_file() and not lock_path.is_symlink():
                with contextlib.suppress(Exception):
                    if read_json(lock_path).get("token") == self.token:
                        lock_path.unlink()
            raise
        self.locked = True
        return self


@contextmanager
def _open_extension_store(
    directory: Path, bindings: dict[str, Any], corpus, *, resume: bool = False
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    recovered_guard = _recover_dead_reclaim_guard(directory)
    stale_guard_archive = _dead_guard_archive_exists(directory)
    stale_writer_archive = _dead_writer_archive_exists(directory)
    lock_path = directory / "writer.lock"
    reclaim_token: str | None = None
    reclaimed_dead_writer = recovered_guard or stale_guard_archive or stale_writer_archive
    if lock_path.exists():
        if not resume:
            raise Refuse("writer lock exists; explicit resume is required")
        if lock_path.is_symlink():
            raise Refuse("writer lock must not be a symlink")
        lock_identity = read_json(lock_path)
        if _identity_is_live(lock_identity, label="writer lock"):
            raise Refuse("writer owner is still alive")
        reclaim_token, _guard_identity = _create_reclaim_guard(directory)
        try:
            _archive_dead_writer_lock(directory, lock_path)
        except BaseException:
            guard_path = directory / "reclaim.guard"
            if guard_path.is_file() and not guard_path.is_symlink():
                with contextlib.suppress(Exception):
                    if read_json(guard_path).get("token") == reclaim_token:
                        guard_path.unlink()
            raise
        reclaimed_dead_writer = True
    elif recovered_guard or stale_guard_archive or stale_writer_archive:
        reclaim_token, _guard_identity = _create_reclaim_guard(directory)

    store = _ExtensionRecoveryStore(directory, bindings, corpus, reclaim_token=reclaim_token)
    entered = False
    guard_published = reclaim_token is not None
    guard_path = directory / "reclaim.guard"
    try:
        store = store.__enter__()
        entered = True
        if guard_published:
            current_guard = read_json(guard_path)
            if current_guard.get("token") != reclaim_token:
                raise Refuse("reclaim guard changed during writer acquisition")
            guard_path.unlink()
            guard_published = False
        store.reclaimed_dead_writer = reclaimed_dead_writer
        yield store
    finally:
        if entered:
            store.__exit__(*sys.exc_info())
        if guard_published and guard_path.is_file() and not guard_path.is_symlink():
            with contextlib.suppress(Exception):
                if read_json(guard_path).get("token") == reclaim_token:
                    guard_path.unlink()


def current_accounting_snapshot(charged_cpu_seconds: float) -> dict[str, Any]:
    return {
        "charged_cpu_seconds": float(charged_cpu_seconds),
        "snapshot_utc": utc_now(),
        "boot_identity": float(psutil.boot_time()),
        "host_cpu_seconds": float(frozen.host_cpu_seconds()),
        "logical_cpu_upper": int(os.cpu_count() or 1),
    }


def tail_charge_from_snapshot(
    snapshot: dict[str, Any],
    *,
    now: datetime | None = None,
    current_boot_identity: float | None = None,
    current_host_cpu_seconds: float | None = None,
    current_logical_cpu_count: int | None = None,
) -> float:
    """Apply the frozen boot-aware unknown-tail upper bound to an intent event."""
    now = now or datetime.now(UTC)
    current_boot_identity = (
        float(psutil.boot_time()) if current_boot_identity is None else float(current_boot_identity)
    )
    current_host_cpu_seconds = (
        float(frozen.host_cpu_seconds())
        if current_host_cpu_seconds is None
        else float(current_host_cpu_seconds)
    )
    current_logical_cpu_count = (
        int(os.cpu_count() or 1)
        if current_logical_cpu_count is None
        else int(current_logical_cpu_count)
    )
    required = (
        "charged_cpu_seconds",
        "snapshot_utc",
        "boot_identity",
        "host_cpu_seconds",
        "logical_cpu_upper",
    )
    if any(key not in snapshot for key in required):
        raise Refuse("accounting snapshot is incomplete")
    saved_boot = float(snapshot["boot_identity"])
    host_delta = current_host_cpu_seconds - float(snapshot["host_cpu_seconds"])
    if abs(current_boot_identity - saved_boot) <= 0.001 and host_delta >= 0:
        lost_upper = host_delta
    else:
        elapsed = max(
            0.0,
            (now - datetime.fromisoformat(str(snapshot["snapshot_utc"]))).total_seconds(),
        )
        cores = max(int(snapshot["logical_cpu_upper"]), current_logical_cpu_count)
        lost_upper = elapsed * cores
    return float(snapshot["charged_cpu_seconds"]) + lost_upper


def _git_commit_sha(revision: str) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if process.returncode:
        raise Refuse(f"source provenance commit cannot be resolved: {revision}")
    commit = process.stdout.decode("ascii", errors="strict").strip().lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise Refuse(f"source provenance commit is malformed: {revision}")
    return commit


def _source_blob_provenance(
    revision: str,
    relative: str,
    expected_sha256: str,
    *,
    worktree_path: Path | None = None,
) -> dict[str, str]:
    commit = _git_commit_sha(revision)
    oid_process = subprocess.run(
        ["git", "rev-parse", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if oid_process.returncode:
        raise Refuse(f"source provenance blob is unavailable: {relative}@{commit}")
    blob_oid = oid_process.stdout.decode("ascii", errors="strict").strip().lower()
    if len(blob_oid) != 40 or any(char not in "0123456789abcdef" for char in blob_oid):
        raise Refuse(f"source provenance blob OID is malformed: {relative}@{commit}")
    process = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if process.returncode or _git_blob_oid(process.stdout) != blob_oid:
        raise Refuse(f"source provenance blob is unavailable or inconsistent: {relative}@{commit}")
    blob_sha = sha256_bytes(process.stdout)
    result = {
        "git_revision": commit,
        "git_commit_sha": commit,
        "git_blob_oid": blob_oid,
        "git_blob_sha256": blob_sha,
    }
    if worktree_path is not None:
        worktree_oid = subprocess.run(
            ["git", "hash-object", f"--path={relative}", str(worktree_path.resolve())],
            cwd=ROOT,
            capture_output=True,
            check=False,
            timeout=15,
        )
        if worktree_oid.returncode:
            raise Refuse(f"working source cannot be hashed through Git filters: {relative}")
        normalized_oid = worktree_oid.stdout.decode("ascii", errors="strict").strip().lower()
        if normalized_oid != blob_oid:
            raise Refuse(f"working source does not match its clean-filter HEAD blob: {relative}")
        result["git_worktree_blob_oid"] = normalized_oid
        if blob_sha == expected_sha256:
            result["git_filter_normalization"] = "byte_exact"
        else:
            result["git_filter_normalization"] = "working_tree_clean_filter"
    elif blob_sha != expected_sha256:
        raise Refuse(f"historical source blob does not match parent SHA: {relative}")
    return result


def validate_source_compatibility(
    arm: str,
    parent_source_hashes: dict[str, str],
    parent_identity: dict[str, str] | None = None,
    *,
    exact_revision: str = "HEAD",
) -> dict[str, Any]:
    if arm not in ARMS:
        raise Refuse(f"unknown arm: {arm}")
    if not isinstance(parent_source_hashes, dict) or not parent_source_hashes:
        raise Refuse(f"parent source hash map is empty or malformed: {arm}")
    rows: list[dict[str, Any]] = []
    for absolute_path, parent_sha in sorted(parent_source_hashes.items()):
        source_path = Path(absolute_path).resolve()
        try:
            relative = source_path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise Refuse(f"parent source escapes repository root: {absolute_path}") from exc
        source_bytes = source_path.read_bytes()
        current_sha = sha256_bytes(source_bytes)
        row: dict[str, Any] = {
            "path": relative,
            "parent_sha256": str(parent_sha),
            "current_sha256": current_sha,
            "compatibility": "exact_parent_bytes",
        }
        if current_sha != parent_sha:
            spec = SOURCE_COMPATIBILITY.get(relative)
            if (
                spec is None
                or spec["parent_sha256"] != parent_sha
                or spec["current_sha256"] != current_sha
            ):
                raise Refuse(f"unreviewed source drift: {relative}")
            if arm != "full_turn_1m" and relative in {
                "scripts/training/run_taiji_r2_daily_dialogue_warmstart_matched.py",
                "tests/taiji_native/test_r2_daily_dialogue_warmstart_matched_contract.py",
            }:
                affected_arms = {"warm_start", "fresh_same_capacity"}
            else:
                affected_arms = set(ARMS)
            if arm not in affected_arms:
                raise Refuse(f"source mismatch is not authorized for {arm}: {relative}")
            row.update(
                {
                    "compatibility": "reviewed_parent_blob_compatibility",
                    **_source_blob_provenance(spec["git_revision"], relative, str(parent_sha)),
                }
            )
        else:
            try:
                row.update(
                    _source_blob_provenance(
                        exact_revision,
                        relative,
                        str(parent_sha),
                        worktree_path=source_path,
                    )
                )
            except Refuse:
                if not relative.startswith("data/"):
                    raise
                if not isinstance(parent_identity, dict) or not all(
                    isinstance(parent_identity.get(key), str) and len(parent_identity[key]) == 64
                    for key in (
                        "parent_attempt_sha256",
                        "parent_receipt_sha256",
                        "parent_bundle_sha256",
                    )
                ):
                    raise Refuse(
                        f"untracked data source lacks complete parent lineage: {relative}"
                    ) from None
                row.update(
                    {
                        "compatibility": "parent_bound_external_artifact",
                        "git_provenance_applicable": False,
                        "parent_identity": dict(parent_identity),
                    }
                )
        rows.append(row)
    return {
        "report_path": str(ARCHITECTURE_REPORT.resolve()),
        "report_sha256": sha256_file(ARCHITECTURE_REPORT),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "parent_identity": dict(parent_identity or {}),
        "arm": arm,
        "files": rows,
    }


V1_RUNNER_SHA256 = "58a96d63186681e7cdaaa1034f4e0a68ee14b20c22050bad214cd12eec3ef18f"
V1_SOURCE_HASHES = {
    "runner": V1_RUNNER_SHA256,
    "budget_helper": "5bc1787499bee803a8e5310349672e40b038205ced35f55d28b0c7a7893c72d9",
    "runner_contract_test": "3b25e646e1f823c025fb51ff89de0fb3de930e396b8ab9fa8834176e025cd39a",
    "budget_contract_test": "09e67815752b23d9db1797b1ee640056d931f80168a851954f78e4da68488469",
}
SOURCE_REVISION_REPORT = (
    ROOT
    / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-source-revision-v1-1.md"
)
SOURCE_REVISION_V1_2_REPORT = (
    ROOT
    / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-source-revision-v1-2.md"
)


def _source_revision_path(revision: str | None = None) -> Path:
    names = {
        "v1.1": "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_source_revision_v1_1.json",
        "v1.2": "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_source_revision_v1_2.json",
        "v1.3": "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_source_revision_v1_3.json",
    }
    if revision is None:
        revision = (
            "v1.3"
            if (REPORTS_DIR / names["v1.3"]).exists()
            else ("v1.2" if (REPORTS_DIR / names["v1.2"]).exists() else "v1.1")
        )
    if revision not in names:
        raise Refuse(f"unknown source revision: {revision}")
    return REPORTS_DIR / names[revision]


def _continuation_source_files() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "budget_helper": Path(continuation_budget.__file__).resolve(),
        "runner_contract_test": (
            ROOT / "tests/taiji_native/test_r2_dialogue_continuation_extension_runner_contract.py"
        ).resolve(),
        "budget_contract_test": (
            ROOT / "tests/taiji_native/test_r2_dialogue_continuation_extension_contract.py"
        ).resolve(),
        "source_validator": Path(source_validation.__file__).resolve(),
        "source_validator_contract_test": (
            ROOT / "tests/taiji_native/test_r2_dialogue_continuation_source_validation_contract.py"
        ).resolve(),
        "source_revision_contract_test": (
            ROOT / "tests/taiji_native/test_r2_dialogue_continuation_source_revision_contract.py"
        ).resolve(),
    }


def _continuation_source_binding(source_report_sha256: str) -> dict[str, Any]:
    files = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in _continuation_source_files().items()
    }
    runner = files["runner"]
    return {
        "runner_path": runner["path"],
        "runner_sha256": runner["sha256"],
        "files": files,
        "report_path": str(ARCHITECTURE_REPORT.resolve()),
        "report_sha256": source_report_sha256,
    }


def _effective_attempt(
    attempt: dict[str, Any],
    source_report_sha256: str,
    *,
    candidate_revision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = _continuation_source_binding(source_report_sha256)
    if attempt.get("continuation_sources") == expected:
        return attempt
    revision_path = _source_revision_path()
    if revision_path.is_symlink() or (candidate_revision is None and not revision_path.is_file()):
        raise Refuse("continuation source binding changed without a versioned source revision")
    revision = candidate_revision if candidate_revision is not None else read_json(revision_path)
    arm = attempt.get("arm")
    if arm not in ARMS:
        raise Refuse("source revision arm is invalid")
    location = _expected_parent_paths(arm)
    old = revision.get("original_attempts", {}).get(arm, {})
    old_sources = attempt.get("continuation_sources", {})
    if (
        revision.get("contract_sha256") != sha256_file(location["contract_path"])
        or old.get("attempt_sha256") != sha256_file(location["extension_attempt"])
        or old.get("parent_readiness_sha256") != sha256_file(location["parent_readiness"])
        or old.get("parent_receipt_sha256") != PARENT_RECEIPT_SHA256[arm]
        or old_sources.get("runner_sha256") != V1_RUNNER_SHA256
        or {name: row.get("sha256") for name, row in old_sources.get("files", {}).items()}
        != V1_SOURCE_HASHES
        or old_sources.get("report_sha256") != sha256_file(ARCHITECTURE_REPORT)
        or attempt.get("source_compatibility", {}).get("runner_sha256") != V1_RUNNER_SHA256
    ):
        raise Refuse("source revision does not exactly bind original v1 lineage")

    schema = revision.get("schema")
    if schema == "taiji-r2-continuation-source-revision-v1.1":
        valid_revision = (
            revision.get("current_continuation_sources") == expected
            and revision.get("review_report_sha256") == sha256_file(SOURCE_REVISION_REPORT)
            and revision.get("approved_budget")
            == {"per_arm": 1_450_000.0, "aggregate": 4_200_000.0}
        )
    elif schema == "taiji-r2-continuation-source-revision-v1.2":
        previous_path = _source_revision_path("v1.1")
        valid_revision = (
            previous_path.is_file()
            and revision.get("previous_source_revision_sha256") == sha256_file(previous_path)
            and revision.get("current_continuation_sources") == expected
            and revision.get("review_report_sha256") == sha256_file(SOURCE_REVISION_V1_2_REPORT)
            and revision.get("approved_budget")
            == {
                "per_arm": V1_2_PER_ARM_CPU_LIMIT,
                "aggregate": V1_2_AGGREGATE_CPU_LIMIT,
            }
        )
    elif schema == "taiji-r2-continuation-source-revision-v1.3":
        previous_path = _source_revision_path("v1.2")
        report = (
            ROOT
            / "plans/reference/delivery-runs/20261021-n-mainline/artifacts/04-continuation-boot-accounting-v1-3.md"
        )
        valid_revision = (
            previous_path.is_file()
            and revision.get("previous_source_revision_sha256") == sha256_file(previous_path)
            and revision.get("current_continuation_sources") == expected
            and revision.get("review_report_sha256") == sha256_file(report)
            and revision.get("approved_budget")
            == {"per_arm": V1_2_PER_ARM_CPU_LIMIT, "aggregate": V1_2_AGGREGATE_CPU_LIMIT}
        )
    else:
        valid_revision = False
    if not valid_revision:
        raise Refuse("versioned source and cumulative budget revision binding differs")

    effective = dict(attempt)
    effective["continuation_sources"] = expected
    effective["source_compatibility"] = {
        **attempt["source_compatibility"],
        "runner_sha256": expected["runner_sha256"],
    }
    effective["source_revision_sha256"] = (
        sha256_bytes((json.dumps(revision, sort_keys=True, separators=(",", ":")) + "\n").encode())
        if candidate_revision is not None
        else sha256_file(revision_path)
    )
    return effective


@contextmanager
def _source_revision_budget_snapshot(*, allow_stopped_runtime: bool = False):
    # Hold the shared budget serialization point until amendment publication.
    with continuation_budget._aggregate_lock(continuation_budget.LOCK_PATH):
        _, charges, reservations, _ = continuation_budget._read_ledger_full()
        if reservations:
            raise Refuse("source revision requires no active CPU reservation")
        allowed_events = {
            "extension_budget_initialized",
            "work_unit_reserved",
            "continuation_prepare_tail_reconciled",
            "continuation_prepared",
            "continuation_prepare_reconciled",
        }
        if allow_stopped_runtime:
            allowed_events.update(
                {
                    "continuation_started",
                    "training_resource_preflight_passed",
                    "continuation_stopped",
                    "work_unit_settled",
                }
            )
        for _sequence, event_path in continuation_budget._event_files(
            continuation_budget.AGGREGATE_LEDGER_PREFIX
        ):
            if read_json(event_path).get("event") not in allowed_events:
                raise Refuse("source revision ledger contains execution history")
        yield dict(charges)


def _validate_stopped_untrained_progress(events: list[dict[str, Any]]) -> None:
    allowed = {
        "continuation_prepare_intent",
        "continuation_prepare_tail_reconciled",
        "continuation_prepared",
        "continuation_prepare_reconciled",
        "continuation_start_intent",
        "continuation_started",
        "training_resource_preflight_passed",
        "continuation_stopped",
    }
    if any(event.get("event") not in allowed for event in events):
        raise Refuse("budget revision refuses restore or training history")
    counts = {name: sum(event.get("event") == name for event in events) for name in allowed}
    if not counts["continuation_prepared"] or any(
        counts[name] != 1
        for name in (
            "continuation_start_intent",
            "continuation_started",
            "training_resource_preflight_passed",
            "continuation_stopped",
        )
    ):
        raise Refuse("budget revision requires one clean stopped start attempt per arm")
    stopped = next(event for event in events if event.get("event") == "continuation_stopped")
    if stopped.get("stop_reason") != "throughput_projection_exceeds_remaining_budget":
        raise Refuse("budget revision only adopts the reviewed forecast stop")


def prepare_source_revision() -> dict[str, Any]:
    # v1.2 continues the same ledger and adopts only the reviewed pre-training stops.
    _configure_prepared_storage()
    source_report_sha = sha256_file(ARCHITECTURE_REPORT)
    previous_path = _source_revision_path("v1.1")
    target_path = _source_revision_path("v1.2")
    if not previous_path.is_file() or previous_path.is_symlink():
        raise Refuse("v1.2 requires the durable v1.1 source amendment")
    previous_revision = read_json(previous_path)
    previous_sha256 = sha256_file(previous_path)
    original_attempts: dict[str, Any] = {}
    for arm in ARMS:
        context = _parent_context(arm)
        location = context["locations"]
        attempt = read_json(location["extension_attempt"])
        old_sources = attempt.get("continuation_sources", {})
        compatible = dict(context["source_compatibility"])
        compatible["runner_sha256"] = V1_RUNNER_SHA256
        if (
            previous_revision.get("schema") != "taiji-r2-continuation-source-revision-v1.1"
            or attempt.get("schema") != "taiji-r2-dialogue-continuation-extension-attempt-v1"
            or attempt.get("arm") != arm
            or attempt.get("source_compatibility_report_sha256") != source_report_sha
            or attempt.get("parent_checkpoint", {}).get("receipt_sha256")
            != context["parent_receipt_sha256"]
            or old_sources.get("runner_sha256") != V1_RUNNER_SHA256
            or {name: row.get("sha256") for name, row in old_sources.get("files", {}).items()}
            != V1_SOURCE_HASHES
            or old_sources.get("report_sha256") != source_report_sha
            or attempt.get("source_compatibility") != compatible
            or attempt.get("parent", {}).get("attempt_sha256") != context["parent_attempt_sha256"]
        ):
            raise Refuse("v1.2 requires exact original v1 prepared sources")
        events = read_jsonl(location["extension_progress"])
        _validate_stopped_untrained_progress(events)
        for directory in (
            location["extension_checkpoint_root"],
            location["extension_checkpoint_root"] / "pretraining",
            location["extension_checkpoint_root"] / "assistant_sft",
        ):
            if (directory / "writer.lock").exists() or (directory / "reclaim.guard").exists():
                raise Refuse("budget revision requires quiescent checkpoint writers")
        if (
            location["extension_result"].exists()
            or _readiness_path(location["extension_result"]).exists()
        ):
            raise Refuse("budget revision cannot adopt terminal artifacts")
        _expected, missing = _parent_copy_state(context)
        if missing:
            raise Refuse("budget revision parent fork is incomplete")
        original = previous_revision.get("original_attempts", {}).get(arm)
        if (
            not isinstance(original, dict)
            or original.get("attempt_sha256") != sha256_file(location["extension_attempt"])
            or original.get("parent_readiness_sha256") != sha256_file(location["parent_readiness"])
            or original.get("parent_receipt_sha256") != context["parent_receipt_sha256"]
        ):
            raise Refuse("v1.2 original v1 attempt lineage changed")
        original_attempts[arm] = {
            **original,
            "stopped_progress_sha256": sha256_file(location["extension_progress"]),
        }
    payload = {
        "schema": "taiji-r2-continuation-source-revision-v1.2",
        "contract_sha256": sha256_file(_expected_parent_paths(ARMS[0])["contract_path"]),
        "previous_source_revision_sha256": previous_sha256,
        "original_attempts": original_attempts,
        "current_continuation_sources": _continuation_source_binding(source_report_sha),
        "review_report_sha256": sha256_file(SOURCE_REVISION_V1_2_REPORT),
        "approved_budget": {
            "per_arm": V1_2_PER_ARM_CPU_LIMIT,
            "aggregate": V1_2_AGGREGATE_CPU_LIMIT,
        },
        "approval_source": "user approved the cumulative ceilings 1,500,000 per arm and 4,300,000 aggregate; prior charges and tails remain included",
        "reason": "the approved v1.1 run passed resources and stopped before restore/training because its projection exceeded 4.2M",
    }
    with _source_revision_budget_snapshot(allow_stopped_runtime=True) as charges:
        _validate_prepared(source_revision_candidate=payload, ledger_snapshot=charges)
        _write_json_once(target_path, payload)
    global PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT
    PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT = continuation_budget.effective_budget_limits()
    return _validate_prepared()


def _require_continuation_source_bindings(
    attempt: dict[str, Any], source_report_sha256: str
) -> dict[str, Any]:
    expected = _continuation_source_binding(source_report_sha256)
    if attempt.get("continuation_sources") != expected:
        raise Refuse("continuation source binding changed after prepare")
    return expected


def _assert_parent_writer_is_not_live(parent_root: Path) -> None:
    for directory in (Path(parent_root), Path(parent_root) / "pretraining"):
        lock = directory / "writer.lock"
        if not lock.exists():
            continue
        identity = read_json(lock)
        try:
            owner = psutil.Process(int(identity["pid"]))
            live = owner.create_time() == float(identity["create_time"])
        except psutil.NoSuchProcess:
            live = False
        except psutil.AccessDenied as exc:
            raise Refuse(f"cannot verify parent writer identity: {lock}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise Refuse(f"parent writer identity is malformed: {lock}") from exc
        if live:
            raise Refuse(f"parent checkpoint still has a live writer: {lock}")


def _expected_parent_paths(arm: str) -> dict[str, Path]:
    if arm not in ARMS:
        raise Refuse(f"unknown arm: {arm}")
    result = paths(arm)
    required = {
        "parent_attempt",
        "parent_readiness",
        "parent_progress",
        "parent_result",
        "parent_checkpoint_root",
        "extension_attempt",
        "extension_progress",
        "extension_result",
        "extension_checkpoint_root",
        "contract_path",
        "aggregate_ledger_path",
    }
    if not required.issubset(result):
        raise Refuse(f"continuation path contract missing keys: {sorted(required - set(result))}")
    return {key: Path(value) for key, value in result.items()}


def _parent_context(arm: str) -> dict[str, Any]:
    location = _expected_parent_paths(arm)
    attempt_path = location["parent_attempt"]
    if sha256_file(attempt_path) != PARENT_ATTEMPT_SHA256[arm]:
        raise Refuse(f"parent attempt changed: {attempt_path}")
    attempt = read_json(attempt_path)
    if attempt.get("arm") != arm or attempt.get("status") != "training":
        raise Refuse(f"parent attempt is not the expected partial arm: {arm}")
    if location["parent_result"].exists():
        raise Refuse(f"parent result already exists; extension must not replace it: {arm}")
    readiness = read_json(location["parent_readiness"])
    if readiness.get("status") != "ready" or readiness.get("bindings") != attempt.get("bindings"):
        raise Refuse(f"parent readiness/bindings mismatch: {arm}")
    source_compatibility = validate_source_compatibility(
        arm,
        attempt["bindings"].get("source_hashes", {}),
        {
            "parent_attempt_sha256": PARENT_ATTEMPT_SHA256[arm],
            "parent_receipt_sha256": PARENT_RECEIPT_SHA256[arm],
            "parent_bundle_sha256": PARENT_BUNDLE_SHA256[arm],
        },
        exact_revision="ad18ce7c7add64ea444f7f78c678e5d19a459cb9",
    )
    progress_records = read_jsonl(location["parent_progress"])
    durable = [
        event
        for event in progress_records
        if event.get("event") == "durable_training_commit" and event.get("stage") == "pretraining"
    ]
    if not durable:
        raise Refuse(f"parent has no durable pretraining checkpoint event: {arm}")
    last_durable = durable[-1]
    stage_root = location["parent_checkpoint_root"] / "pretraining"
    latest_path = stage_root / "latest.json"
    latest = read_json(latest_path)
    receipt_path = stage_root / str(latest.get("receipt", ""))
    if not receipt_path.is_file() or sha256_file(receipt_path) != latest.get("receipt_sha256"):
        raise Refuse(f"parent receipt is missing or corrupt: {arm}")
    receipt = read_json(receipt_path)
    bundle_path = stage_root / str(receipt.get("bundle", ""))
    if (
        not bundle_path.is_file()
        or sha256_file(bundle_path) != receipt.get("sha256")
        or receipt.get("sha256") != PARENT_BUNDLE_SHA256[arm]
        or latest.get("receipt_sha256") != PARENT_RECEIPT_SHA256[arm]
        or receipt.get("digest") != PARENT_CHECKPOINT_DIGEST[arm]
    ):
        raise Refuse(f"parent bundle/receipt binding differs: {arm}")
    if last_durable.get("receipt", {}).get("sha256") != receipt.get("sha256"):
        raise Refuse(f"parent progress does not point to the latest receipt: {arm}")
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    expected_bindings = {**attempt["bindings"], "stage": "pretraining"}
    if (
        payload.get("schema") != "taiji-r2-recovery-v1"
        or payload.get("bindings") != expected_bindings
    ):
        raise Refuse(f"parent checkpoint source binding differs: {arm}")
    if payload.get("digest") != recovery.content_digest(
        {key: value for key, value in payload.items() if key != "digest"}
    ):
        raise Refuse(f"parent checkpoint inner digest differs: {arm}")
    cursor = payload.get("cursor")
    if cursor != PARENT_CURSOR[arm]:
        raise Refuse(f"parent cursor differs from the reviewed checkpoint: {arm}")
    trainer = payload.get("trainer", {})
    if (
        int(trainer.get("episode_count", -1)) != PARENT_DURABLE_EPISODES[arm]
        or int(trainer.get("global_step", -1)) != PARENT_DURABLE_UPDATES[arm]
        or int(last_durable.get("episodes", -1)) != PARENT_DURABLE_EPISODES[arm]
        or int(last_durable.get("target_byte_updates", -1)) != PARENT_DURABLE_UPDATES[arm]
        or last_durable.get("cursor") != cursor
    ):
        raise Refuse(f"parent progress/cursor/update denominator mismatch: {arm}")
    if receipt.get("checkpoint_digest") != trainer.get("checkpoint_digest"):
        raise Refuse(f"parent trainer digest differs from its receipt: {arm}")
    _assert_parent_writer_is_not_live(location["parent_checkpoint_root"])

    known_cpu = [float(attempt.get("cpu_seconds_before_training", 0.0))]
    if readiness.get("cpu_seconds") is not None:
        known_cpu.append(float(readiness["cpu_seconds"]))
    for event in progress_records:
        if event.get("cpu_seconds") is not None:
            known_cpu.append(float(event["cpu_seconds"]))
    ledger = payload.get("ledger", {})
    if not isinstance(ledger, dict):
        raise Refuse(f"parent checkpoint ledger missing: {arm}")
    tail = float(frozen.tail_charge(payload))
    known_cpu.append(float(ledger.get("charged_cpu_seconds", 0.0)))
    initial_charge = max([tail, *known_cpu])
    old_boot = float(ledger["host_boot_time"])
    current_boot = float(psutil.boot_time())
    tail_status = (
        "closed"
        if old_boot == current_boot
        and frozen.host_cpu_seconds() >= float(ledger["host_cpu_seconds"])
        else "conservative_upper_bound"
    )
    snapshot = current_accounting_snapshot(initial_charge)
    return {
        "arm": arm,
        "locations": location,
        "parent_attempt": attempt,
        "parent_attempt_sha256": PARENT_ATTEMPT_SHA256[arm],
        "parent_readiness_sha256": sha256_file(location["parent_readiness"]),
        "parent_progress_sha256": sha256_file(location["parent_progress"]),
        "parent_attempt_path": str(attempt_path.resolve()),
        "parent_checkpoint_root": str(location["parent_checkpoint_root"].resolve()),
        "parent_stage_root": str(stage_root.resolve()),
        "parent_latest": latest,
        "parent_receipt": receipt,
        "parent_receipt_path": str(receipt_path.resolve()),
        "parent_receipt_sha256": PARENT_RECEIPT_SHA256[arm],
        "parent_bundle_path": str(bundle_path.resolve()),
        "parent_bundle_sha256": PARENT_BUNDLE_SHA256[arm],
        "parent_checkpoint_digest": PARENT_CHECKPOINT_DIGEST[arm],
        "parent_cursor": cursor,
        "parent_episodes": PARENT_DURABLE_EPISODES[arm],
        "parent_updates": PARENT_DURABLE_UPDATES[arm],
        "parent_bindings": attempt["bindings"],
        "source_compatibility": source_compatibility,
        "initial_charge": initial_charge,
        "tail_status": tail_status,
        "accounting_start": snapshot,
        "parent_envelope": payload,
    }


def _write_json_once(path: Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    if path.exists():
        raise Refuse(f"write-once extension record already exists: {path}")
    return _extension_publish_json(path, payload)


def _write_empty_once(path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise Refuse(f"write-once extension file already exists: {path}")
    _extension_publish(path, lambda stream: stream.write(b""))


def _parent_copy_state(context: dict[str, Any]) -> tuple[dict[str, bytes], set[str]]:
    arm = context["arm"]
    location = context["locations"]
    source_stage = location["parent_checkpoint_root"] / "pretraining"
    extension_root = location["extension_checkpoint_root"]
    destination_stage = location["extension_checkpoint_root"] / "pretraining"
    if extension_root.is_symlink() or (extension_root.exists() and not extension_root.is_dir()):
        raise Refuse(f"extension checkpoint root is not a safe directory: {arm}")
    if destination_stage.is_symlink() or (
        destination_stage.exists() and not destination_stage.is_dir()
    ):
        raise Refuse(f"extension parent-copy stage is not a safe directory: {arm}")
    files = (
        source_stage / "latest.json",
        source_stage / context["parent_receipt"]["bundle"].replace(".pt", ".json"),
        source_stage / context["parent_receipt"]["bundle"],
    )
    expected: dict[str, bytes] = {}
    for source in files:
        if not source.is_file() or source.is_symlink():
            raise Refuse(f"parent checkpoint source file is missing or unsafe: {arm}/{source.name}")
        expected[source.name] = source.read_bytes()
    if not destination_stage.is_dir():
        return expected, set(expected)
    unexpected = {path.name for path in destination_stage.iterdir()} - set(expected)
    if unexpected:
        raise Refuse(
            f"extension parent-copy stage has unexpected files: {arm}/{sorted(unexpected)}"
        )
    missing: set[str] = set()
    for name, raw in expected.items():
        destination = destination_stage / name
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise Refuse(f"existing parent-copy artifact is not a regular file: {arm}/{name}")
            if destination.read_bytes() != raw:
                raise Refuse(f"existing parent-copy artifact differs byte-for-byte: {arm}/{name}")
        else:
            missing.add(name)
    return expected, missing


def _copy_parent_checkpoint(context: dict[str, Any]) -> None:
    arm = context["arm"]
    extension_root = context["locations"]["extension_checkpoint_root"]
    destination_stage = extension_root / "pretraining"
    expected, missing = _parent_copy_state(context)
    if missing:
        extension_root.mkdir(parents=True, exist_ok=True)
        destination_stage.mkdir(parents=False, exist_ok=True)
    for name in sorted(missing):
        raw = expected[name]
        destination = destination_stage / name
        _extension_publish(destination, lambda stream, data=raw: stream.write(data))
        if destination.read_bytes() != raw or sha256_file(destination) != sha256_bytes(raw):
            raise Refuse(f"parent checkpoint fork copy SHA mismatch: {arm}/{name}")
    copied_latest = read_json(destination_stage / "latest.json")
    if copied_latest != context["parent_latest"]:
        raise Refuse(f"parent latest pointer changed while forking: {arm}")


def _attempt_is_reconciled(context: dict[str, Any], source_report_sha256: str) -> bool:
    path = context["locations"]["extension_attempt"]
    if not path.exists():
        return False
    existing = read_json(path)
    expected = _sidecar_attempt(context, source_report_sha256)
    expected["prepared_utc"] = existing.get("prepared_utc")
    if not isinstance(existing.get("prepared_utc"), str) or existing != expected:
        raise Refuse(
            f"existing extension attempt differs from its immutable contract: {context['arm']}"
        )
    return True


def _progress_preparation_state(
    context: dict[str, Any], source_report_sha256: str
) -> tuple[bool, list[dict[str, Any]]]:
    path = context["locations"]["extension_progress"]
    if not path.exists():
        return False, []
    events = read_jsonl(path)
    for event in events:
        if event.get("arm") != context["arm"]:
            raise Refuse(f"extension progress arm mismatch: {context['arm']}")
        if event.get("event") in {
            "continuation_started",
            "continuation_resumed",
            "continuation_start_intent",
            "continuation_completed",
        }:
            raise Refuse("prepare-only cannot reconcile an arm after execution has started")
        if event.get("parent_attempt_sha256") not in (
            None,
            context["parent_attempt_sha256"],
        ):
            raise Refuse("prepare progress parent attempt identity differs")
        if event.get("parent_receipt_sha256") not in (
            None,
            context["parent_receipt_sha256"],
        ):
            raise Refuse("prepare progress parent receipt identity differs")
        if event.get("source_report_sha256") not in (None, source_report_sha256):
            raise Refuse("prepare progress source report SHA differs")
    return any(event.get("event") == "continuation_prepared" for event in events), events


def _reconcile_prepare_progress(arm: str, progress_path: Path) -> list[dict[str, Any]]:
    if not progress_path.exists():
        return []
    events = read_jsonl(progress_path)
    active = latest_reservations().get(arm)
    intents = [item for item in events if item.get("event") == "continuation_prepare_intent"]
    for intent in intents:
        token = intent.get("reservation_id")
        if not isinstance(token, str) or not token:
            raise Refuse("prepare intent lacks its reservation token")
        settlement = _find_settlement_event(arm, token)
        if settlement is not None:
            _append_progress_from_ledger(progress_path, events, settlement)
        elif active is None or active.get("reservation_id") != token:
            raise Refuse("prepare intent has neither an active reservation nor a settlement event")
    return events


def _prepare_intent(
    arm: str,
    token: str,
    charged: float,
    context: dict[str, Any],
    source_report_sha256: str,
    progress_path: Path,
) -> None:
    if not progress_path.exists():
        _write_empty_once(progress_path)
    events = read_jsonl(progress_path)
    existing = [
        item
        for item in events
        if item.get("event") == "continuation_prepare_intent"
        and item.get("reservation_id") == token
    ]
    if existing:
        if (
            len(existing) != 1
            or existing[0].get("parent_attempt_sha256") != context["parent_attempt_sha256"]
            or existing[0].get("parent_receipt_sha256") != context["parent_receipt_sha256"]
            or existing[0].get("source_report_sha256") != source_report_sha256
        ):
            raise Refuse("prepare intent conflicts with the existing reservation")
        return
    append_durable_jsonl(
        progress_path,
        {
            "event": "continuation_prepare_intent",
            "arm": arm,
            "utc": utc_now(),
            "reservation_id": token,
            "charged_cpu_seconds": charged,
            "parent_attempt_sha256": context["parent_attempt_sha256"],
            "parent_receipt_sha256": context["parent_receipt_sha256"],
            "source_report_sha256": source_report_sha256,
        },
    )


def _prepare_artifacts_complete(context: dict[str, Any], source_report_sha256: str) -> bool:
    location = context["locations"]
    attempt_exists = _attempt_is_reconciled(context, source_report_sha256)
    prepared_event, _progress = _progress_preparation_state(context, source_report_sha256)
    _expected, missing_copy_files = _parent_copy_state(context)
    if location["extension_result"].exists():
        raise Refuse(f"extension result exists before continuation start: {context['arm']}")
    return attempt_exists and prepared_event and not missing_copy_files


def _sidecar_attempt(context: dict[str, Any], source_report_sha256: str) -> dict[str, Any]:
    continuation_sources = _continuation_source_binding(source_report_sha256)
    return {
        "schema": "taiji-r2-dialogue-continuation-extension-attempt-v1",
        "extension_id": f"r2-dialogue-continuation-v1:{context['arm']}:{context['parent_attempt_sha256']}",
        "arm": context["arm"],
        "status": "prepared",
        "approved_budget": {
            "per_arm_cpu_seconds": PER_ARM_CPU_LIMIT,
            "aggregate_cpu_seconds": AGGREGATE_CPU_LIMIT,
            "approved_date": APPROVED_UTC,
            "approved_source": APPROVAL_SOURCE,
            "semantics": "cumulative_total_including_parent_actual_and_unknown_tail",
        },
        "parent": {
            "attempt_path": context["parent_attempt_path"],
            "attempt_sha256": context["parent_attempt_sha256"],
            "attempt_status": context["parent_attempt"].get("status"),
            "readiness_sha256": context["parent_readiness_sha256"],
            "progress_sha256": context["parent_progress_sha256"],
            "checkpoint_root": context["parent_checkpoint_root"],
        },
        "parent_checkpoint": {
            "stage": "pretraining",
            "cursor": context["parent_cursor"],
            "episodes": context["parent_episodes"],
            "target_byte_updates": context["parent_updates"],
            "latest": context["parent_latest"],
            "receipt_path": context["parent_receipt_path"],
            "receipt_sha256": context["parent_receipt_sha256"],
            "bundle_path": context["parent_bundle_path"],
            "bundle_sha256": context["parent_bundle_sha256"],
            "checkpoint_digest": context["parent_checkpoint_digest"],
            "bindings": context["parent_bindings"],
        },
        "accounting_start": {
            "charged_cpu_seconds": context["initial_charge"],
            "tail_status": context["tail_status"],
            "snapshot": context["accounting_start"],
        },
        "source_compatibility": context["source_compatibility"],
        "source_compatibility_report_sha256": source_report_sha256,
        "continuation_sources": continuation_sources,
        "dev_queries": 0,
        "reference_queries": 0,
        "generation_queries": 0,
        "can_promote": False,
        "prepared_utc": utc_now(),
    }


def _has_any_extension_artifact() -> bool:
    for arm in ARMS:
        location = _expected_parent_paths(arm)
        for key in (
            "extension_attempt",
            "extension_progress",
            "extension_result",
            "extension_checkpoint_root",
        ):
            if location[key].exists():
                return True
        prefix = location["aggregate_ledger_path"]
        if list(prefix.parent.glob(prefix.name + ".*.json")):
            return True
    first = _expected_parent_paths(ARMS[0])
    return first["contract_path"].exists()


def _validate_prepared(
    *,
    allow_terminal_artifacts: bool = False,
    source_revision_candidate: dict[str, Any] | None = None,
    ledger_snapshot: dict[str, float] | None = None,
) -> dict[str, Any]:
    source_report_sha = sha256_file(ARCHITECTURE_REPORT)
    if not _expected_parent_paths(ARMS[0])["contract_path"].is_file():
        raise Refuse("global continuation contract is missing; run --prepare-only")
    _configure_prepared_storage()
    charges = latest_charges() if ledger_snapshot is None else dict(ledger_snapshot)
    if set(charges) != set(ARMS):
        raise Refuse("aggregate ledger does not contain all three prepared arms")
    records: dict[str, Any] = {}
    for arm in ARMS:
        context = _parent_context(arm)
        location = context["locations"]
        attempt = _effective_attempt(
            read_json(location["extension_attempt"]),
            source_report_sha,
            candidate_revision=source_revision_candidate,
        )
        if (
            attempt.get("schema") != "taiji-r2-dialogue-continuation-extension-attempt-v1"
            or attempt.get("arm") != arm
            or attempt.get("parent", {}).get("attempt_sha256") != context["parent_attempt_sha256"]
            or attempt.get("parent_checkpoint", {}).get("receipt_sha256")
            != context["parent_receipt_sha256"]
            or attempt.get("source_compatibility_report_sha256") != source_report_sha
            or attempt.get("source_compatibility") != context["source_compatibility"]
        ):
            raise Refuse(f"prepared sidecar no longer matches parent/source contract: {arm}")
        _require_continuation_source_bindings(attempt, source_report_sha)
        if location["extension_result"].exists() and not allow_terminal_artifacts:
            raise Refuse(f"extension result already exists; do not retrain: {arm}")
        progress = read_jsonl(location["extension_progress"])
        if not any(event.get("event") == "continuation_prepared" for event in progress):
            raise Refuse(f"prepared progress is incomplete: {arm}")
        copied = location["extension_checkpoint_root"] / "pretraining"
        for name in (
            "latest.json",
            context["parent_receipt"]["bundle"].replace(".pt", ".json"),
            context["parent_receipt"]["bundle"],
        ):
            if not (copied / name).is_file():
                raise Refuse(f"parent checkpoint fork is incomplete: {arm}/{name}")
        if (
            sha256_file(copied / context["parent_receipt"]["bundle"])
            != context["parent_bundle_sha256"]
        ):
            raise Refuse(f"parent checkpoint fork changed: {arm}")
        copied_latest = read_json(copied / "latest.json")
        copied_receipt = copied / str(copied_latest.get("receipt", ""))
        if (
            copied_latest != context["parent_latest"]
            or not copied_receipt.is_file()
            or sha256_file(copied_receipt) != context["parent_receipt_sha256"]
        ):
            raise Refuse(f"parent receipt fork changed: {arm}")
        records[arm] = context
    if any(not 0 <= float(charges[arm]) <= PER_ARM_CPU_LIMIT for arm in ARMS):
        raise Refuse("per-arm CPU ledger exceeds the approved ceiling")
    if sum(float(charges[arm]) for arm in ARMS) > AGGREGATE_CPU_LIMIT:
        raise Refuse("aggregate CPU ledger exceeds the approved ceiling")
    return {"source_report_sha256": source_report_sha, "charges": charges, "parents": records}


def _prepare_only_legacy() -> dict[str, Any]:
    if _has_any_extension_artifact():
        return _validate_prepared()
    contexts = {arm: _parent_context(arm) for arm in ARMS}
    prepared_volume_root = _qualify_storage_targets()
    source_report_sha = sha256_file(ARCHITECTURE_REPORT)
    initial = {
        arm: {
            "parent_attempt_sha256": contexts[arm]["parent_attempt_sha256"],
            "parent_receipt_sha256": contexts[arm]["parent_receipt_sha256"],
            "parent_bundle_sha256": contexts[arm]["parent_bundle_sha256"],
            "charged_cpu_seconds": contexts[arm]["initial_charge"],
            "tail_status": contexts[arm]["tail_status"],
            "accounting_start": contexts[arm]["accounting_start"],
            "parent_cursor": contexts[arm]["parent_cursor"],
            "source_compatibility": contexts[arm]["source_compatibility"],
        }
        for arm in ARMS
    }
    if any(
        initial[arm]["charged_cpu_seconds"]
        + PREPARE_EVENT_RESERVE_SECONDS
        + frozen.episode_cpu_reserve(arm)[0]
        >= PER_ARM_CPU_LIMIT
        for arm in ARMS
    ):
        raise Refuse("remaining per-arm CPU cannot cover the next episode reserve")
    if (
        sum(initial[arm]["charged_cpu_seconds"] for arm in ARMS)
        + len(ARMS) * PREPARE_EVENT_RESERVE_SECONDS
        > AGGREGATE_CPU_LIMIT
    ):
        raise Refuse("current conservative tail exceeds aggregate CPU ceiling")
    prepare_started_cpu = time.process_time()
    initialized = initialize_ledger(initial, source_report_sha, APPROVED_UTC)
    initialized_contract = initialized.get("contract") if isinstance(initialized, dict) else None
    initialized_durability = (
        initialized_contract.get("durability") if isinstance(initialized_contract, dict) else None
    )
    if not isinstance(initialized_durability, dict) or _normalized_volume_root(
        initialized_durability.get("volume_root", "")
    ) != _normalized_volume_root(prepared_volume_root):
        raise Refuse("initialized ledger volume differs from prequalified extension targets")
    reservations: dict[str, str] = {}
    for arm in ARMS:
        token = ledger_guard(
            arm,
            float(initial[arm]["charged_cpu_seconds"]),
            reserve_cpu_seconds=PREPARE_EVENT_RESERVE_SECONDS,
        )
        if not isinstance(token, str) or not token:
            raise Refuse("aggregate ledger did not return a preparation reservation token")
        reservations[arm] = token
        context = contexts[arm]
        location = context["locations"]
        attempt = _sidecar_attempt(context, source_report_sha)
        _write_json_once(location["extension_attempt"], attempt)
        _write_empty_once(location["extension_progress"])
        _copy_parent_checkpoint(context)
    preparation_cpu = max(0.0, time.process_time() - prepare_started_cpu)
    shared_prepare_cpu_per_arm = preparation_cpu / len(ARMS)
    for arm in ARMS:
        context = contexts[arm]
        location = context["locations"]
        charged = max(
            context["initial_charge"] + shared_prepare_cpu_per_arm,
            tail_charge_from_snapshot(context["accounting_start"]),
        )
        if charged > context["initial_charge"] + PREPARE_EVENT_RESERVE_SECONDS:
            raise Refuse(f"preparation exceeded reserved CPU headroom for {arm}")
        snapshot = current_accounting_snapshot(charged)
        append_charge(
            arm,
            "continuation_prepared",
            charged,
            {
                "reservation_id": reservations[arm],
                "parent_attempt_sha256": context["parent_attempt_sha256"],
                "parent_receipt_sha256": context["parent_receipt_sha256"],
                "source_report_sha256": source_report_sha,
                "shared_prepare_cpu_seconds": shared_prepare_cpu_per_arm,
                "prepared_charge_uses_reserved_upper": charged
                > context["initial_charge"] + shared_prepare_cpu_per_arm,
                "accounting_snapshot": snapshot,
            },
        )
        append_durable_jsonl(
            location["extension_progress"],
            {
                "event": "continuation_prepared",
                "arm": arm,
                "utc": utc_now(),
                "charged_cpu_seconds": charged,
                "accounting_snapshot": snapshot,
                "parent_attempt_sha256": context["parent_attempt_sha256"],
                "parent_receipt_sha256": context["parent_receipt_sha256"],
            },
        )
    return _validate_prepared()


def prepare_only() -> dict[str, Any]:
    contexts = {arm: _parent_context(arm) for arm in ARMS}
    prepared_volume_root = _qualify_storage_targets()
    source_report_sha = sha256_file(ARCHITECTURE_REPORT)
    current_initial = {
        arm: {
            "parent_attempt_sha256": contexts[arm]["parent_attempt_sha256"],
            "parent_receipt_sha256": contexts[arm]["parent_receipt_sha256"],
            "parent_bundle_sha256": contexts[arm]["parent_bundle_sha256"],
            "charged_cpu_seconds": contexts[arm]["initial_charge"],
            "tail_status": contexts[arm]["tail_status"],
            "accounting_start": contexts[arm]["accounting_start"],
            "parent_cursor": contexts[arm]["parent_cursor"],
            "source_compatibility": contexts[arm]["source_compatibility"],
        }
        for arm in ARMS
    }
    contract_path = _expected_parent_paths(ARMS[0])["contract_path"]
    if contract_path.exists():
        existing_contract = read_json(contract_path)
        parent_records = existing_contract.get("parent_records")
        if not isinstance(parent_records, dict) or set(parent_records) != set(ARMS):
            raise Refuse("durable continuation contract has incomplete parent records")
        for arm in ARMS:
            context = contexts[arm]
            saved = parent_records[arm]
            stable_values = {
                "parent_attempt_sha256": context["parent_attempt_sha256"],
                "parent_receipt_sha256": context["parent_receipt_sha256"],
                "parent_bundle_sha256": context["parent_bundle_sha256"],
                "parent_cursor": context["parent_cursor"],
                "source_compatibility": context["source_compatibility"],
            }
            if any(saved.get(key) != value for key, value in stable_values.items()):
                raise Refuse(f"durable contract parent identity/source differs: {arm}")
            context["initial_charge"] = float(saved["charged_cpu_seconds"])
            context["tail_status"] = saved["tail_status"]
            context["accounting_start"] = dict(saved["accounting_start"])
        initialized = reconcile_initialize_ledger(parent_records, source_report_sha, APPROVED_UTC)
    else:
        if _has_any_extension_artifact():
            raise Refuse("partial extension artifacts exist without their durable contract")
        if any(
            current_initial[arm]["charged_cpu_seconds"]
            + PREPARE_EVENT_RESERVE_SECONDS
            + LEDGER_EVENT_RESERVE_SECONDS
            + frozen.episode_cpu_reserve(arm)[0]
            >= PER_ARM_CPU_LIMIT
            for arm in ARMS
        ):
            raise Refuse("remaining per-arm CPU cannot cover the next episode reserve")
        if (
            sum(current_initial[arm]["charged_cpu_seconds"] for arm in ARMS)
            + len(ARMS) * (PREPARE_EVENT_RESERVE_SECONDS + LEDGER_EVENT_RESERVE_SECONDS)
            > AGGREGATE_CPU_LIMIT
        ):
            raise Refuse("current conservative tail exceeds aggregate CPU ceiling")
        initialized = initialize_ledger(current_initial, source_report_sha, APPROVED_UTC)

    initialized_contract = initialized.get("contract") if isinstance(initialized, dict) else None
    durability = (
        initialized_contract.get("durability") if isinstance(initialized_contract, dict) else None
    )
    if not isinstance(durability, dict) or _normalized_volume_root(
        durability.get("volume_root", "")
    ) != _normalized_volume_root(prepared_volume_root):
        raise Refuse("initialized ledger volume differs from prequalified extension targets")

    latest = latest_charges()
    pending = latest_reservations()
    for arm, reservation in pending.items():
        context = contexts[arm]
        progress_path = context["locations"]["extension_progress"]
        progress = read_jsonl(progress_path) if progress_path.exists() else []
        if any(
            event.get("event")
            in {"continuation_started", "continuation_resumed", "continuation_start_intent"}
            for event in progress
        ):
            raise Refuse("prepare-only cannot reconcile a reservation after execution started")
        token = str(reservation["reservation_id"])
        _prepare_intent(arm, token, latest[arm], context, source_report_sha, progress_path)
        settlement = _find_settlement_event(arm, token)
        if settlement is None:
            charged = max(
                float(latest[arm]),
                tail_charge_from_snapshot(context["accounting_start"]),
            )
            settlement = append_charge(
                arm,
                "continuation_prepare_resumed",
                charged,
                {
                    "reservation_id": token,
                    "parent_attempt_sha256": context["parent_attempt_sha256"],
                    "parent_receipt_sha256": context["parent_receipt_sha256"],
                    "source_report_sha256": source_report_sha,
                    "reconciled_reserved_upper": True,
                    "accounting_snapshot": current_accounting_snapshot(charged),
                },
            )
        _append_progress_from_ledger(progress_path, read_jsonl(progress_path), settlement)

    latest = latest_charges()
    for arm in ARMS:
        context = contexts[arm]
        location = context["locations"]
        progress_path = location["extension_progress"]
        events = _reconcile_prepare_progress(arm, progress_path)
        attempt_exists = _attempt_is_reconciled(context, source_report_sha)
        prepared_event = any(item.get("event") == "continuation_prepared" for item in events)
        _expected, missing_copy_files = _parent_copy_state(context)
        if location["extension_result"].exists():
            raise Refuse(f"extension result exists before continuation start: {arm}")
        if attempt_exists and prepared_event and not missing_copy_files:
            continue

        latest = latest_charges()
        tail = tail_charge_from_snapshot(context["accounting_start"])
        base_charge = max(float(latest[arm]), float(tail))
        if base_charge > float(latest[arm]):
            tail_token = ledger_guard(
                arm,
                float(latest[arm]),
                reserve_cpu_seconds=LEDGER_EVENT_RESERVE_SECONDS,
            )
            if not isinstance(tail_token, str) or not tail_token:
                raise Refuse("aggregate ledger did not return a tail-reconciliation token")
            _prepare_intent(arm, tail_token, base_charge, context, source_report_sha, progress_path)
            tail_record = append_charge(
                arm,
                "continuation_prepare_tail_reconciled",
                base_charge,
                {
                    "reservation_id": tail_token,
                    "parent_attempt_sha256": context["parent_attempt_sha256"],
                    "parent_receipt_sha256": context["parent_receipt_sha256"],
                    "source_report_sha256": source_report_sha,
                    "accounting_snapshot": current_accounting_snapshot(base_charge),
                },
            )
            _append_progress_from_ledger(progress_path, read_jsonl(progress_path), tail_record)

        work_started_cpu = time.process_time()
        token = ledger_guard(
            arm,
            base_charge,
            reserve_cpu_seconds=PREPARE_EVENT_RESERVE_SECONDS + LEDGER_EVENT_RESERVE_SECONDS,
        )
        if not isinstance(token, str) or not token:
            raise Refuse("aggregate ledger did not return a preparation reservation token")
        _prepare_intent(arm, token, base_charge, context, source_report_sha, progress_path)
        if not attempt_exists:
            _write_json_once(
                location["extension_attempt"], _sidecar_attempt(context, source_report_sha)
            )
        _copy_parent_checkpoint(context)
        measured = max(0.0, time.process_time() - work_started_cpu)
        charged = max(
            base_charge + measured, tail_charge_from_snapshot(context["accounting_start"])
        )
        settlement_event = (
            "continuation_prepared" if not prepared_event else "continuation_prepare_reconciled"
        )
        record = append_charge(
            arm,
            settlement_event,
            charged,
            {
                "reservation_id": token,
                "parent_attempt_sha256": context["parent_attempt_sha256"],
                "parent_receipt_sha256": context["parent_receipt_sha256"],
                "source_report_sha256": source_report_sha,
                "shared_prepare_cpu_seconds": measured,
                "prepared_charge_uses_reserved_upper": charged > base_charge + measured,
                "reserve_overrun": charged
                > base_charge + PREPARE_EVENT_RESERVE_SECONDS + LEDGER_EVENT_RESERVE_SECONDS,
                "accounting_snapshot": current_accounting_snapshot(charged),
            },
        )
        _append_progress_from_ledger(progress_path, read_jsonl(progress_path), record)
        if charged > base_charge + PREPARE_EVENT_RESERVE_SECONDS + LEDGER_EVENT_RESERVE_SECONDS:
            raise Refuse(f"preparation exceeded reserved CPU headroom for {arm}")

    return _validate_prepared()


def resource_snapshot() -> dict[str, Any]:
    samples = [psutil.cpu_percent(interval=1.0), psutil.cpu_percent(interval=1.0)]
    available_gib = psutil.virtual_memory().available / 1024**3
    if available_gib < 8.0:
        raise Refuse(f"available memory below 8 GiB: {available_gib:.2f}")
    if any(sample > 85.0 for sample in samples):
        raise Refuse(f"system CPU start samples exceed 85%: {samples}")
    return {"cpu_percent_samples": samples, "available_memory_gib": round(available_gib, 3)}


class ExtensionBudget:
    def __init__(self, arm: str, charged: float) -> None:
        self.arm = arm
        self.limit = PER_ARM_CPU_LIMIT
        self.episode_reserve, self.probe_hashes = frozen.episode_cpu_reserve(arm)
        self.started = time.process_time()
        self.charged = float(charged) + max(0.0, self.started - PROCESS_START_CPU)
        self.child_seconds = 0.0
        self.active_reservation_id: str | None = None
        self.active_reservation_upper: float | None = None
        self.pending_stop_details: dict[str, Any] = {}

    def used(self) -> float:
        return self.charged + time.process_time() - self.started + self.child_seconds

    def guard(self, reserve: float | None = None, *, terminal: bool = False) -> None:
        if self.active_reservation_id is not None:
            self.settle_reservation("work_unit_settled")
        if reserve is None:
            reserve = self.episode_reserve
        next_charge = self.used()
        reserve_with_events = float(reserve) + LEDGER_EVENT_RESERVE_SECONDS
        if not terminal:
            reserve_with_events += TERMINAL_STOP_RESERVE_SECONDS
        refusal: str | None = None
        if next_charge + reserve_with_events >= self.limit:
            refusal = "continuation CPU allocation exhausted; no implicit budget reset"
        elif psutil.virtual_memory().available < 8 * 1024**3:
            refusal = "continuation available memory below 8 GiB"
        else:
            try:
                token = ledger_guard(self.arm, next_charge, reserve_cpu_seconds=reserve_with_events)
            except (Refuse, continuation_budget.ExtensionLedgerRefuse) as exc:
                refusal = f"aggregate continuation reservation refused: {type(exc).__name__}: {exc}"
        if refusal is not None:
            self.pending_stop_details = {
                **self.pending_stop_details,
                "stop_reason": self.pending_stop_details.get("stop_reason", "work_unit_refused"),
                "refusal": refusal,
                "requested_reserve_cpu_seconds": float(reserve),
                "terminal_stop_reserve_cpu_seconds": TERMINAL_STOP_RESERVE_SECONDS,
            }
            if not terminal and next_charge + LEDGER_EVENT_RESERVE_SECONDS < self.limit:
                try:
                    stop_token = ledger_guard(
                        self.arm,
                        next_charge,
                        reserve_cpu_seconds=LEDGER_EVENT_RESERVE_SECONDS,
                    )
                except (Refuse, continuation_budget.ExtensionLedgerRefuse):
                    stop_token = None
                if isinstance(stop_token, str) and stop_token:
                    self.active_reservation_id = stop_token
                    self.active_reservation_upper = next_charge + LEDGER_EVENT_RESERVE_SECONDS
            raise Refuse(refusal)
        if not isinstance(token, str) or not token:
            raise Refuse("aggregate ledger did not return a CPU reservation token")
        self.active_reservation_id = token
        self.active_reservation_upper = next_charge + reserve_with_events

    def settle_reservation(
        self,
        event: str,
        *,
        charge: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        token = self.active_reservation_id
        if token is None:
            return None
        charged = self.used() if charge is None else float(charge)
        payload = dict(details or {})
        snapshot = payload.get("accounting_snapshot")
        if (
            not isinstance(snapshot, dict)
            or float(snapshot.get("charged_cpu_seconds", -1)) != charged
        ):
            snapshot = current_accounting_snapshot(charged)
        payload.update({"reservation_id": token, "accounting_snapshot": snapshot})
        record = append_charge(self.arm, event, charged, payload)
        if charged > self.used():
            self.charged += charged - self.used()
        self.active_reservation_id = None
        self.active_reservation_upper = None
        return record

    def ledger(self) -> dict[str, Any]:
        return {
            "charged_cpu_seconds": self.used(),
            "snapshot_utc": utc_now(),
            "host_cpu_seconds": float(frozen.host_cpu_seconds()),
            "logical_cpu_upper": int(os.cpu_count() or 1),
            "host_boot_time": float(psutil.boot_time()),
            "episode_cpu_reserve": self.episode_reserve,
            "worst_episode_probe_hashes": self.probe_hashes,
            "cpu_limit": self.limit,
            "extension_id": "r2-dialogue-continuation-extension-v1",
        }

    def resume_from(self, envelope: dict[str, Any], previous_charge: float) -> None:
        self.charged = max(
            self.used(), float(previous_charge), _extension_checkpoint_tail(envelope)
        )
        self.started = time.process_time()
        self.child_seconds = 0.0


def _fresh_process_restore(
    bundle: Path, bindings: dict[str, Any], stage: str, budget: ExtensionBudget
) -> dict[str, Any]:
    # The subprocess has a hard 120-second deadline. The normal budget guard
    # adds the durable-ledger settlement reserve on top of this work bound.
    budget.guard(reserve=FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS)
    host_before = frozen.host_cpu_seconds()
    command = [
        sys.executable,
        "-m",
        "scripts.training.taiji_r2_dialogue_recovery",
        "--verify",
        str(bundle.resolve()),
        "--bindings",
        json.dumps(bindings, sort_keys=True, separators=(",", ":")),
        "--threads",
        "1",
    ]
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS,
            check=False,
            env=dict(
                os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
            ),
        )
    except subprocess.TimeoutExpired as exc:
        budget.child_seconds += max(0.0, frozen.host_cpu_seconds() - host_before)
        raise Refuse(f"fresh-process {stage} restore timed out") from exc
    except OSError as exc:
        budget.child_seconds += max(0.0, frozen.host_cpu_seconds() - host_before)
        raise Refuse(f"fresh-process {stage} restore could not start") from exc
    if process.returncode:
        budget.child_seconds += max(0.0, frozen.host_cpu_seconds() - host_before)
        raise Refuse(f"fresh-process {stage} restore failed: {process.stderr[-1500:]}")
    try:
        receipt = json.loads(process.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        budget.child_seconds += max(0.0, frozen.host_cpu_seconds() - host_before)
        raise Refuse(f"fresh-process {stage} restore receipt is invalid") from exc
    budget.child_seconds += max(0.0, float(receipt.get("cpu_seconds", 0.0)))
    if receipt.get("digest") is None:
        raise Refuse(f"fresh-process {stage} restore receipt is incomplete")
    budget.guard(reserve=10.0)
    return receipt


def _fresh_final_restore(checkpoint: Path, stage: str, budget: ExtensionBudget) -> dict[str, Any]:
    budget.guard(reserve=FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS)
    host_before = frozen.host_cpu_seconds()
    command = [
        sys.executable,
        str(Path(base.__file__).resolve()),
        "--restore-worker",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--stage",
        stage,
    ]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        budget.child_seconds += max(0.0, frozen.host_cpu_seconds() - host_before)
        raise Refuse(f"final {stage} restore timed out") from exc
    except OSError as exc:
        budget.child_seconds += max(0.0, frozen.host_cpu_seconds() - host_before)
        raise Refuse(f"final {stage} restore could not start") from exc
    if process.returncode:
        budget.child_seconds += max(0.0, frozen.host_cpu_seconds() - host_before)
        raise Refuse(f"final {stage} restore failed: {process.stderr[-1500:]}")
    try:
        receipt = json.loads(process.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        budget.child_seconds += max(0.0, frozen.host_cpu_seconds() - host_before)
        raise Refuse(f"final {stage} restore receipt is invalid") from exc
    if (
        receipt.get("complete") is not True
        or receipt.get("equal") is not True
        or receipt.get("stage") != stage
    ):
        budget.child_seconds += max(0.0, float(receipt.get("cpu_seconds", 0.0)))
        raise Refuse(f"final {stage} restore receipt is incomplete")
    budget.child_seconds += max(0.0, float(receipt.get("cpu_seconds", 0.0)))
    budget.guard(reserve=10.0)
    return {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint), **receipt}


def _latest_extension_envelope(context: dict[str, Any]) -> tuple[str, dict[str, Any], Path]:
    location = context["locations"]
    root = location["extension_checkpoint_root"]
    for stage in ("assistant_sft", "pretraining"):
        latest_path = root / stage / "latest.json"
        if not latest_path.exists():
            continue
        latest = read_json(latest_path)
        receipt_path = latest_path.parent / latest["receipt"]
        if sha256_file(receipt_path) != latest.get("receipt_sha256"):
            raise Refuse(f"extension latest receipt SHA mismatch: {context['arm']}/{stage}")
        receipt = read_json(receipt_path)
        bundle = latest_path.parent / receipt["bundle"]
        if sha256_file(bundle) != receipt.get("sha256"):
            raise Refuse(f"extension checkpoint bundle SHA mismatch: {context['arm']}/{stage}")
        envelope = torch.load(bundle, map_location="cpu", weights_only=False)
        expected = {**context["parent_bindings"], "stage": stage}
        if envelope.get("bindings") != expected or envelope.get(
            "digest"
        ) != recovery.content_digest(
            {key: value for key, value in envelope.items() if key != "digest"}
        ):
            raise Refuse(f"extension checkpoint binding/digest mismatch: {context['arm']}/{stage}")
        return stage, envelope, bundle
    raise Refuse(f"extension checkpoint branch has no latest receipt: {context['arm']}")


def _last_accounting_snapshot(arm: str) -> dict[str, Any]:
    prefix = _expected_parent_paths(arm)["aggregate_ledger_path"]
    files = sorted(prefix.parent.glob(prefix.name + ".*.json"))
    for path in reversed(files):
        event = read_json(path)
        details = event.get("details")
        if event.get("arm") == arm and isinstance(details, dict):
            snapshot = details.get("accounting_snapshot")
            if isinstance(snapshot, dict):
                return snapshot
    raise Refuse(f"aggregate ledger lacks an accounting snapshot for {arm}")


def _extension_checkpoint_tail(envelope: dict[str, Any]) -> float:
    ledger = envelope["ledger"]
    return tail_charge_from_snapshot(
        {
            "charged_cpu_seconds": ledger["charged_cpu_seconds"],
            "snapshot_utc": ledger["snapshot_utc"],
            "boot_identity": ledger["host_boot_time"],
            "host_cpu_seconds": ledger["host_cpu_seconds"],
            "logical_cpu_upper": ledger["logical_cpu_upper"],
        }
    )


def _start_charge(context: dict[str, Any]) -> float:
    arm = context["arm"]
    charges = latest_charges()
    if arm not in charges:
        raise Refuse(f"aggregate ledger lacks arm: {arm}")
    stage, envelope, _bundle = _latest_extension_envelope(context)
    parent_receipt_sha = context["parent_receipt_sha256"]
    stage_latest = read_json(
        context["locations"]["extension_checkpoint_root"] / stage / "latest.json"
    )
    forked_parent = (
        stage == "pretraining" and stage_latest.get("receipt_sha256") == parent_receipt_sha
    )
    snapshot_tail = tail_charge_from_snapshot(_last_accounting_snapshot(arm))
    if forked_parent:
        # Preparation already carried the parent tail into a durable extension
        # snapshot. Reusing the older parent timestamp now would double charge
        # all elapsed time after that extension event.
        tail = snapshot_tail
    else:
        tail = max(float(_extension_checkpoint_tail(envelope)), snapshot_tail)
    return max(float(charges[arm]), tail)


def _append_cpu_snapshot(
    arm: str,
    event: str,
    budget: ExtensionBudget,
    details: dict[str, Any],
    progress_path: Path,
    *,
    conservative_upper: bool = False,
) -> dict[str, Any]:
    if budget.active_reservation_id is None:
        if event == "continuation_stopped":
            budget.guard(reserve=0.0, terminal=True)
        else:
            budget.guard(reserve=LEDGER_EVENT_RESERVE_SECONDS)
    charge = budget.used()
    if conservative_upper and budget.active_reservation_upper is not None:
        charge = max(charge, budget.active_reservation_upper)
    snapshot = current_accounting_snapshot(charge)
    details = {
        **details,
        "reservation_id": budget.active_reservation_id,
        "accounting_snapshot": snapshot,
    }
    record = budget.settle_reservation(event, charge=charge, details=details)
    if record is None:
        raise Refuse("CPU snapshot was not covered by an active reservation")
    append_durable_jsonl(
        progress_path,
        {
            "event": event,
            "arm": arm,
            "utc": utc_now(),
            "charged_cpu_seconds": charge,
            "accounting_snapshot": snapshot,
            "reservation_id": details.get("reservation_id"),
            "ledger_sequence": record.get("sequence"),
            "ledger_event_sha256": record.get("event_sha256"),
            **{key: value for key, value in details.items() if key != "accounting_snapshot"},
        },
    )
    if event != "continuation_stopped":
        budget.pending_stop_details = {}
    return record


def _verified_ledger_events(arm: str) -> list[dict[str, Any]]:
    # latest_charges performs the backend's full append-only hash-chain audit.
    latest_charges()
    prefix = _expected_parent_paths(arm)["aggregate_ledger_path"]
    events = [read_json(path) for path in sorted(prefix.parent.glob(prefix.name + ".*.json"))]
    return [event for event in events if event.get("arm") == arm]


def _find_settlement_event(arm: str, reservation_id: str) -> dict[str, Any] | None:
    matches = [
        event
        for event in _verified_ledger_events(arm)
        if event.get("reservation_id") == reservation_id
        and event.get("event") != "work_unit_reserved"
    ]
    if len(matches) > 1:
        raise Refuse(f"reservation has multiple settlement events: {arm}")
    return matches[0] if matches else None


def _progress_record_from_ledger(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    if not isinstance(details, dict):
        raise Refuse("settlement event details are malformed")
    snapshot = details.get("accounting_snapshot")
    if not isinstance(snapshot, dict):
        raise Refuse("settlement event lacks its accounting snapshot")
    return {
        "event": event.get("event"),
        "arm": event.get("arm"),
        "utc": event.get("created_utc"),
        "charged_cpu_seconds": event.get("charged_cpu_seconds"),
        "accounting_snapshot": snapshot,
        "reservation_id": event.get("reservation_id"),
        "ledger_sequence": event.get("sequence"),
        "ledger_event_sha256": event.get("event_sha256"),
        **{key: value for key, value in details.items() if key != "accounting_snapshot"},
    }


def _append_progress_from_ledger(
    progress_path: Path, existing: list[dict[str, Any]], event: dict[str, Any]
) -> None:
    event_sha = event.get("event_sha256")
    if event_sha is not None and any(
        item.get("ledger_event_sha256") == event_sha for item in existing
    ):
        return
    append_durable_jsonl(progress_path, _progress_record_from_ledger(event))
    existing.append(_progress_record_from_ledger(event))


def _reconcile_start_progress(arm: str, progress_path: Path) -> list[dict[str, Any]]:
    events = read_jsonl(progress_path)
    intents = [item for item in events if item.get("event") == "continuation_start_intent"]
    active = latest_reservations().get(arm)
    for intent in intents:
        token = intent.get("reservation_id")
        if not isinstance(token, str) or not token:
            raise Refuse("continuation start intent lacks its reservation token")
        settlement = _find_settlement_event(arm, token)
        if settlement is not None:
            _append_progress_from_ledger(progress_path, events, settlement)
        elif active is None or active.get("reservation_id") != token:
            raise Refuse("start intent has neither an active reservation nor a settlement event")
    return events


def _append_start_intent(
    arm: str,
    event_name: str,
    reservation_id: str,
    charged: float,
    context: dict[str, Any],
    progress_path: Path,
) -> None:
    events = read_jsonl(progress_path)
    existing = [
        item
        for item in events
        if item.get("event") == "continuation_start_intent"
        and item.get("reservation_id") == reservation_id
    ]
    if existing:
        if (
            len(existing) != 1
            or existing[0].get("requested_event")
            not in {"continuation_started", "continuation_resumed"}
            or existing[0].get("parent_attempt_sha256") != context["parent_attempt_sha256"]
            or existing[0].get("parent_receipt_sha256") != context["parent_receipt_sha256"]
        ):
            raise Refuse("continuation start intent conflicts with its reservation")
        return
    append_durable_jsonl(
        progress_path,
        {
            "event": "continuation_start_intent",
            "arm": arm,
            "utc": utc_now(),
            "requested_event": event_name,
            "reservation_id": reservation_id,
            "charged_cpu_seconds": charged,
            "parent_attempt_sha256": context["parent_attempt_sha256"],
            "parent_receipt_sha256": context["parent_receipt_sha256"],
        },
    )


def _durable_start(
    arm: str,
    *,
    resume: bool,
    allow_empty_resume: bool = False,
    context: dict[str, Any],
    attempt: dict[str, Any],
    budget: ExtensionBudget,
    progress_path: Path,
) -> dict[str, Any]:
    events = _reconcile_start_progress(arm, progress_path)
    has_settled_start = any(
        event.get("event") in {"continuation_started", "continuation_resumed"} for event in events
    )
    intents = [event for event in events if event.get("event") == "continuation_start_intent"]
    active = latest_reservations().get(arm)
    if not resume and (has_settled_start or intents or active is not None):
        raise Refuse("continuation already started or has an active intent; use --resume")
    if (
        resume
        and not (has_settled_start or intents or active is not None)
        and not allow_empty_resume
    ):
        raise Refuse("--resume requires a durable start event, start intent, or reservation")

    if active is not None:
        token = str(active["reservation_id"])
        if budget.active_reservation_id not in (None, token):
            raise Refuse("loaded start reservation token differs from the budget ledger")
        budget.active_reservation_id = token
        budget.active_reservation_upper = budget.used()
    else:
        budget.guard(reserve=LEDGER_EVENT_RESERVE_SECONDS)
        token = str(budget.active_reservation_id)

    requested_event = "continuation_resumed" if resume else "continuation_started"
    _append_start_intent(
        arm,
        requested_event,
        token,
        budget.used(),
        context,
        progress_path,
    )
    latest_stage, latest_envelope, _latest_bundle = _latest_extension_envelope(context)
    return _append_cpu_snapshot(
        arm,
        requested_event,
        budget,
        {
            "parent_attempt_sha256": context["parent_attempt_sha256"],
            "parent_receipt_sha256": context["parent_receipt_sha256"],
            "runner_sha256": attempt["continuation_sources"]["runner_sha256"],
            "reconciled_reserved_upper": active is not None,
            "stage": latest_stage,
            "episodes": int(latest_envelope["trainer"]["episode_count"]),
            "cursor": latest_envelope.get("cursor"),
        },
        progress_path,
        conservative_upper=active is not None,
    )


def _readiness_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.stem + "_readiness.json")


def _resources_at_training_start(progress_path: Path) -> dict[str, Any]:
    rows = [
        event
        for event in read_jsonl(progress_path)
        if event.get("event") == "training_resource_preflight_passed"
    ]
    if not rows or not isinstance(rows[-1].get("resources"), dict):
        raise Refuse("terminal artifacts lack a durable resource preflight record")
    return rows[-1]["resources"]


def _validate_final_readiness(
    arm: str,
    context: dict[str, Any],
    attempt: dict[str, Any],
    readiness_path: Path,
) -> dict[str, Any]:
    ready = read_json(readiness_path)
    expected_attempt_sha = sha256_file(context["locations"]["extension_attempt"])
    expected_runner_sha = attempt["continuation_sources"]["runner_sha256"]
    cumulative_cpu = ready.get("cpu_seconds_cumulative")
    if (
        ready.get("schema") != "taiji-r2-dialogue-continuation-extension-readiness-v1"
        or ready.get("status") != "ready"
        or ready.get("arm") != arm
        or ready.get("parent_attempt_sha256") != context["parent_attempt_sha256"]
        or ready.get("parent_receipt_sha256") != context["parent_receipt_sha256"]
        or ready.get("extension_attempt_sha256") != expected_attempt_sha
        or ready.get("extension_runner_sha256") != expected_runner_sha
        or ready.get("source_compatibility_report_sha256")
        != attempt["source_compatibility_report_sha256"]
        or ready.get("can_promote") is not False
        or ready.get("dev_queries") != 0
        or ready.get("reference_queries") != 0
        or ready.get("generation_queries") != 0
        or not isinstance(cumulative_cpu, (int, float))
        or isinstance(cumulative_cpu, bool)
        or not 0 <= float(cumulative_cpu) <= PER_ARM_CPU_LIMIT
        or ready.get("cpu_seconds_cumulative_is_lower_bound") is not True
        or ready.get("cpu_seconds_cumulative_boundary")
        != "measured_before_final_result_publication"
        or ready.get("completion_cpu_source") != "continuation_completed_ledger_event"
        or ready.get("source_revision_sha256") != attempt.get("source_revision_sha256")
    ):
        raise Refuse("terminal readiness identity or capability boundary is invalid")
    stages = ready.get("stages")
    if not isinstance(stages, dict):
        raise Refuse("terminal readiness stages are malformed")
    for stage_name, expected_episodes, expected_updates in (
        ("pretraining_final", 30_096, 14_517_912),
        ("assistant_sft_final", 525, 113_209),
    ):
        stage = stages.get(stage_name)
        if not isinstance(stage, dict):
            raise Refuse(f"terminal readiness lacks {stage_name}")
        checkpoint = Path(str(stage.get("path", "")))
        restore = stage.get("restore")
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != stage.get("sha256")
            or stage.get("episodes") != expected_episodes
            or stage.get("target_byte_updates") != expected_updates
            or stage.get("checkpoint_digest") is None
            or not isinstance(restore, dict)
            or restore.get("complete") is not True
            or restore.get("equal") is not True
            or restore.get("stage") != stage_name.removesuffix("_final")
            or restore.get("checkpoint_digest") != stage.get("checkpoint_digest")
        ):
            raise Refuse(f"terminal {stage_name} receipt/checkpoint is invalid")
    return ready


def _validate_final_result(
    arm: str,
    context: dict[str, Any],
    attempt: dict[str, Any],
    result_path: Path,
    readiness_path: Path,
    ready: dict[str, Any],
) -> dict[str, Any]:
    result = read_json(result_path)
    if (
        result.get("schema") != "taiji-r2-dialogue-continuation-extension-result-v1"
        or result.get("status") != "complete"
        or result.get("complete") is not True
        or result.get("arm") != arm
        or result.get("parent_attempt_sha256") != context["parent_attempt_sha256"]
        or result.get("parent_checkpoint_receipt_sha256") != context["parent_receipt_sha256"]
        or result.get("extension_attempt_sha256")
        != sha256_file(context["locations"]["extension_attempt"])
        or result.get("readiness_path") != str(readiness_path.resolve())
        or result.get("readiness_sha256") != sha256_file(readiness_path)
        or result.get("stages") != ready["stages"]
        or result.get("can_promote") is not False
        or result.get("dev_queries") != 0
        or result.get("reference_queries") != 0
        or result.get("generation_queries") != 0
        or result.get("cpu_seconds_cumulative") != ready["cpu_seconds_cumulative"]
        or result.get("cpu_seconds_cumulative_is_lower_bound") is not True
        or result.get("cpu_seconds_cumulative_boundary") != ready["cpu_seconds_cumulative_boundary"]
        or result.get("completion_cpu_source") != ready["completion_cpu_source"]
        or result.get("source_revision_sha256") != attempt.get("source_revision_sha256")
    ):
        raise Refuse("terminal result identity or readiness binding is invalid")
    return result


def _reconcile_finalization(
    arm: str,
    context: dict[str, Any],
    attempt: dict[str, Any],
    progress_path: Path,
) -> dict[str, Any] | None:
    location = context["locations"]
    result_path = location["extension_result"]
    readiness_path = _readiness_path(result_path)
    if not result_path.exists() and not readiness_path.exists():
        return None
    if not readiness_path.is_file():
        raise Refuse("terminal result exists without its readiness sidecar")
    ready = _validate_final_readiness(arm, context, attempt, readiness_path)
    completion_events = [
        event
        for event in _verified_ledger_events(arm)
        if event.get("event") == "continuation_completed"
    ]
    if len(completion_events) > 1:
        raise Refuse("terminal completion ledger contains duplicate events")
    reservation = latest_reservations().get(arm)
    if not result_path.exists() and not completion_events and not isinstance(reservation, dict):
        raise Refuse("readiness-only terminal state lacks a completion event or reservation")
    if result_path.exists():
        result = _validate_final_result(arm, context, attempt, result_path, readiness_path, ready)
    else:
        result = {
            "schema": "taiji-r2-dialogue-continuation-extension-result-v1",
            "status": "complete",
            "complete": True,
            "arm": arm,
            "parent_attempt_sha256": context["parent_attempt_sha256"],
            "parent_checkpoint_receipt_sha256": context["parent_receipt_sha256"],
            "extension_attempt_sha256": ready["extension_attempt_sha256"],
            "readiness_path": str(readiness_path.resolve()),
            "readiness_sha256": sha256_file(readiness_path),
            "source_revision_sha256": attempt.get("source_revision_sha256"),
            "stages": ready["stages"],
            "cpu_seconds_cumulative": ready["cpu_seconds_cumulative"],
            "cpu_seconds_cumulative_is_lower_bound": ready["cpu_seconds_cumulative_is_lower_bound"],
            "cpu_seconds_cumulative_boundary": ready["cpu_seconds_cumulative_boundary"],
            "completion_cpu_source": ready["completion_cpu_source"],
            "resources_at_training_start": _resources_at_training_start(progress_path),
            "dev_queries": 0,
            "reference_queries": 0,
            "generation_queries": 0,
            "can_promote": False,
        }
        _write_json_once(result_path, result)
        result = _validate_final_result(arm, context, attempt, result_path, readiness_path, ready)
    result_sha = sha256_file(result_path)
    readiness_sha = sha256_file(readiness_path)
    if completion_events:
        event = completion_events[0]
        details = event.get("details")
        if (
            not isinstance(details, dict)
            or details.get("result_sha256") != result_sha
            or details.get("readiness_sha256") != readiness_sha
            or details.get("parent_attempt_sha256") != context["parent_attempt_sha256"]
        ):
            raise Refuse("terminal completion ledger does not match published artifacts")
        _append_progress_from_ledger(progress_path, read_jsonl(progress_path), event)
        return result

    if not isinstance(reservation, dict):
        raise Refuse("published terminal artifacts lack a pending completion reservation")
    budget = ExtensionBudget(arm, _start_charge(context))
    budget.active_reservation_id = str(reservation.get("reservation_id", ""))
    budget.active_reservation_upper = float(reservation.get("reserved_cpu_seconds", -1.0))
    if not budget.active_reservation_id or budget.active_reservation_upper < 0:
        raise Refuse("terminal reservation is malformed")
    _append_cpu_snapshot(
        arm,
        "continuation_completed",
        budget,
        {
            "parent_attempt_sha256": context["parent_attempt_sha256"],
            "readiness_sha256": readiness_sha,
            "result_sha256": result_sha,
            "result_cpu_snapshot_boundary": "measured_before_final_result_publication",
            "completion_cpu_source": "continuation_completed_ledger_event",
            "completion_reconciled_after_restart": True,
        },
        progress_path,
        conservative_upper=True,
    )
    return result


def _persist_stop_event(
    arm: str,
    budget: ExtensionBudget,
    progress_path: Path,
    parent_attempt_sha256: str,
    error: BaseException,
) -> None:
    error_text = f"{type(error).__name__}: {error}"
    stop_id = sha256_bytes(
        f"{arm}:{parent_attempt_sha256}:continuation_stopped:{error_text}".encode()
    )
    ledger_events = _verified_ledger_events(arm)
    completed = [event for event in ledger_events if event.get("event") == "continuation_completed"]
    if len(completed) > 1:
        raise Refuse("duplicate durable completion events exist")
    if completed:
        location = _expected_parent_paths(arm)
        result_path = location["extension_result"]
        readiness_path = _readiness_path(result_path)
        details = completed[0].get("details")
        if (
            not result_path.is_file()
            or not readiness_path.is_file()
            or not isinstance(details, dict)
            or details.get("result_sha256") != sha256_file(result_path)
            or details.get("readiness_sha256") != sha256_file(readiness_path)
        ):
            raise Refuse("durable completion event does not match its published artifacts")
        _append_progress_from_ledger(progress_path, read_jsonl(progress_path), completed[0])
        return
    existing_ledger = [
        event
        for event in ledger_events
        if event.get("event") == "continuation_stopped"
        and isinstance(event.get("details"), dict)
        and event["details"].get("stop_id") == stop_id
    ]
    if len(existing_ledger) > 1:
        raise Refuse("duplicate durable stop events exist for the same failure")
    if existing_ledger:
        _append_progress_from_ledger(progress_path, read_jsonl(progress_path), existing_ledger[0])
        return

    try:
        _append_cpu_snapshot(
            arm,
            "continuation_stopped",
            budget,
            {
                "error": error_text,
                "stop_id": stop_id,
                "unknown_tail": True,
                "terminal_reserve_used": budget.active_reservation_id is not None,
                **budget.pending_stop_details,
            },
            progress_path,
            conservative_upper=True,
        )
        return
    except Exception as settlement_error:
        # The ledger may have settled before the progress append failed. Repair
        # that exact event before considering a progress-only refusal record.
        settled = [
            event
            for event in _verified_ledger_events(arm)
            if event.get("event") == "continuation_stopped"
            and isinstance(event.get("details"), dict)
            and event["details"].get("stop_id") == stop_id
        ]
        if len(settled) > 1:
            raise Refuse(
                "duplicate durable stop events exist for the same failure"
            ) from settlement_error
        if settled:
            _append_progress_from_ledger(progress_path, read_jsonl(progress_path), settled[0])
            return
        existing_progress = read_jsonl(progress_path)
        if any(item.get("stop_id") == stop_id for item in existing_progress):
            return
        fallback = {
            "event": "continuation_stopped",
            "arm": arm,
            "utc": utc_now(),
            "stop_id": stop_id,
            "parent_attempt_sha256": parent_attempt_sha256,
            "charged_cpu_seconds": budget.used(),
            "accounting_snapshot": current_accounting_snapshot(budget.used()),
            "error": error_text,
            "unknown_tail": True,
            **budget.pending_stop_details,
            "ledger_settlement_failed": f"{type(settlement_error).__name__}: {settlement_error}",
        }
        try:
            append_durable_jsonl(progress_path, fallback)
        except Exception as progress_error:
            raise Refuse(
                "failed to persist either ledger stop settlement or durable progress refusal"
            ) from progress_error


def _episode_cpu_rate(
    parent_events: list[dict[str, Any]],
    extension_events: list[dict[str, Any]],
    stage: str,
) -> tuple[float, int]:
    parent_points = [
        (int(event["episodes"]), float(event["cpu_seconds"]))
        for event in parent_events
        if event.get("event") == "durable_training_commit"
        and event.get("stage") == "pretraining"
        and isinstance(event.get("episodes"), int)
        and isinstance(event.get("cpu_seconds"), (int, float))
    ]
    parent_samples = [
        (later_cpu - earlier_cpu) / (later_episode - earlier_episode)
        for (earlier_episode, earlier_cpu), (later_episode, later_cpu) in zip(
            parent_points, parent_points[1:], strict=False
        )
        if later_episode > earlier_episode and later_cpu > earlier_cpu
    ]
    extension_samples: list[float] = []
    baseline: tuple[str, int, float] | None = None
    for event in extension_events:
        event_name = event.get("event")
        if event_name in {"continuation_started", "continuation_resumed"}:
            event_stage = event.get("stage")
            episode_count = event.get("episodes")
            cpu = event.get("charged_cpu_seconds")
            if (
                isinstance(event_stage, str)
                and isinstance(episode_count, int)
                and isinstance(cpu, (int, float))
            ):
                baseline = (event_stage, episode_count, float(cpu))
            continue
        event_stage = event.get("stage")
        episode_count = event.get("episodes")
        cpu = event.get("charged_cpu_seconds")
        if (
            event_stage != stage
            or event_name not in {"episode_completed", "durable_training_commit"}
            or not isinstance(episode_count, int)
            or not isinstance(cpu, (int, float))
        ):
            continue
        if baseline is not None and baseline[0] == stage:
            episode_delta = episode_count - baseline[1]
            cpu_delta = float(cpu) - baseline[2]
            if episode_delta > 0 and cpu_delta > 0:
                extension_samples.append(cpu_delta / episode_delta)
        baseline = (stage, episode_count, float(cpu))
    samples = extension_samples[-THROUGHPUT_WINDOW_INTERVALS:]
    if stage == "assistant_sft" and not extension_samples and parent_samples:
        samples = parent_samples[-THROUGHPUT_WINDOW_INTERVALS:]
        return statistics.fmean(samples) * THROUGHPUT_SAFETY_FACTOR * 1.2, len(samples)
    if len(samples) < 4:
        samples = (parent_samples[-THROUGHPUT_WINDOW_INTERVALS:] + samples)[
            -THROUGHPUT_WINDOW_INTERVALS:
        ]
    if not samples:
        raise Refuse(f"no verified CPU throughput samples for stage {stage}")
    return statistics.fmean(samples) * THROUGHPUT_SAFETY_FACTOR, len(samples)


def _throughput_projection(
    current_arm: str,
    contexts: dict[str, Any],
    *,
    current_stage: str,
    current_episodes: int,
) -> dict[str, Any]:
    charges = latest_charges_fast()
    projected_by_arm: dict[str, float] = {}
    per_arm_rates: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        context = contexts[arm]
        parent_events = read_jsonl(context["locations"]["parent_progress"])
        extension_progress = context["locations"]["extension_progress"]
        extension_events = read_jsonl(extension_progress, required=False)
        pretraining_episodes = PARENT_DURABLE_EPISODES[arm]
        assistant_episodes = 0
        for event in extension_events:
            episode_count = event.get("episodes")
            stage = event.get("stage")
            if not isinstance(episode_count, int):
                continue
            if stage == "pretraining":
                pretraining_episodes = max(pretraining_episodes, episode_count)
            elif stage == "assistant_sft":
                pretraining_episodes = 30_096
                assistant_episodes = max(assistant_episodes, episode_count)
            elif (
                event.get("event") in {"continuation_started", "continuation_resumed"}
                and stage == "assistant_sft"
            ):
                pretraining_episodes = 30_096
        if arm == current_arm:
            if current_stage == "pretraining":
                pretraining_episodes = max(pretraining_episodes, current_episodes)
            elif current_stage == "assistant_sft":
                pretraining_episodes = 30_096
                assistant_episodes = max(assistant_episodes, current_episodes)
        pre_rate, pre_samples = _episode_cpu_rate(parent_events, extension_events, "pretraining")
        try:
            sft_rate, sft_samples = _episode_cpu_rate(
                parent_events, extension_events, "assistant_sft"
            )
        except Refuse:
            sft_rate, sft_samples = pre_rate * 1.2, pre_samples
        pre_remaining = max(0, 30_096 - pretraining_episodes)
        sft_remaining = max(0, 525 - assistant_episodes)
        projected = (
            pre_remaining * pre_rate + sft_remaining * sft_rate + COMPLETION_CPU_RESERVE_SECONDS
        )
        projected_by_arm[arm] = projected
        per_arm_rates[arm] = {
            "pretraining_cpu_seconds_per_episode": pre_rate,
            "pretraining_samples": pre_samples,
            "assistant_sft_cpu_seconds_per_episode": sft_rate,
            "assistant_sft_samples": sft_samples,
            "pretraining_episodes_remaining": pre_remaining,
            "assistant_sft_episodes_remaining": sft_remaining,
        }
    arm_headroom_by_arm = {arm: PER_ARM_CPU_LIMIT - float(charges[arm]) for arm in ARMS}
    arm_headroom = arm_headroom_by_arm[current_arm]
    aggregate_headroom = AGGREGATE_CPU_LIMIT - sum(float(value) for value in charges.values())
    projected_aggregate = sum(projected_by_arm.values())
    per_arm_feasible = {arm: projected_by_arm[arm] <= arm_headroom_by_arm[arm] for arm in ARMS}
    return {
        "arm": current_arm,
        "current_stage": current_stage,
        "current_episodes": current_episodes,
        "rates": per_arm_rates,
        "projected_cpu_by_arm": projected_by_arm,
        "projected_cpu_aggregate": projected_aggregate,
        "arm_headroom_cpu_seconds": arm_headroom,
        "arm_headroom_by_arm_cpu_seconds": arm_headroom_by_arm,
        "per_arm_feasible": per_arm_feasible,
        "aggregate_headroom_cpu_seconds": aggregate_headroom,
        "feasible": all(per_arm_feasible.values()) and projected_aggregate <= aggregate_headroom,
    }


def execute_arm(arm: str, *, resume: bool) -> dict[str, Any]:
    if arm not in ARMS:
        raise Refuse(f"unknown arm: {arm}")
    prepared = _validate_prepared(allow_terminal_artifacts=True)
    context = prepared["parents"][arm]
    location = context["locations"]
    attempt = _effective_attempt(
        read_json(location["extension_attempt"]), prepared["source_report_sha256"]
    )
    if attempt.get("continuation_sources", {}).get("runner_sha256") != sha256_file(
        Path(__file__).resolve()
    ):
        raise Refuse("extension runner changed after the sidecar attempt was created")
    progress_path = location["extension_progress"]
    readiness_path = _readiness_path(location["extension_result"])
    has_terminal_artifacts = location["extension_result"].exists() or readiness_path.exists()
    if has_terminal_artifacts and not resume:
        raise Refuse("terminal artifacts already exist; use --resume to reconcile only")
    validated_training = source_validation.validate_training_sources(
        context["source_compatibility"]
    )
    pretrain = validated_training.pretrain_corpus
    assistant = validated_training.assistant_corpus
    frozen.validate_probe_face(arm, pretrain, assistant)
    root_context = _open_extension_store(
        location["extension_checkpoint_root"],
        {**context["parent_bindings"], "stage": "runner"},
        pretrain,
        resume=resume,
    )
    root_entered = False
    root_store = None
    budget: ExtensionBudget | None = None
    try:
        # A same-arm writer lock is the serialization point for all mutations.
        # In particular, no second --resume may settle the live runner's token.
        root_store = root_context.__enter__()
        root_entered = True
        if has_terminal_artifacts:
            reconciled = _reconcile_finalization(arm, context, attempt, progress_path)
            if reconciled is None:
                raise Refuse("terminal artifacts disappeared during reconciliation")
            return reconciled

        events = _reconcile_start_progress(arm, progress_path)
        started = any(
            event.get("event") in {"continuation_started", "continuation_resumed"}
            for event in events
        )
        has_intent = any(event.get("event") == "continuation_start_intent" for event in events)
        pending_reservation = latest_reservations().get(arm)
        reclaimed_dead_writer = bool(
            getattr(root_store, "reclaimed_dead_writer", False)
            or getattr(root_store, "reclaim_token", None) is not None
        )
        if (
            resume
            and not (started or has_intent or pending_reservation is not None)
            and not reclaimed_dead_writer
        ):
            raise Refuse("--resume without start state requires reclaiming a dead writer lock")
        if not resume and (started or has_intent or pending_reservation is not None):
            raise Refuse("continuation already started or has an active reservation; use --resume")

        start_charge = _start_charge(context)
        budget = ExtensionBudget(arm, start_charge)
        if start_charge >= PER_ARM_CPU_LIMIT:
            budget.pending_stop_details = {
                "stop_reason": "cumulative_cpu_ceiling_exhausted",
                "cumulative_cpu_ceiling_seconds": PER_ARM_CPU_LIMIT,
                "cumulative_cpu_at_start_seconds": start_charge,
            }
            raise Refuse("cumulative CPU ceiling exhausted before execution")
        if pending_reservation is not None:
            budget.active_reservation_id = str(pending_reservation["reservation_id"])
            budget.active_reservation_upper = start_charge
        _durable_start(
            arm,
            resume=resume,
            allow_empty_resume=resume and reclaimed_dead_writer,
            context=context,
            attempt=attempt,
            budget=budget,
            progress_path=progress_path,
        )
        resources = resource_snapshot()
        _append_cpu_snapshot(
            arm,
            "training_resource_preflight_passed",
            budget,
            {"resources": resources},
            progress_path,
        )

        stage, _latest_envelope, latest_bundle = _latest_extension_envelope(context)
        stage_bindings = {**context["parent_bindings"], "stage": stage}
        latest_pointer = read_json(location["extension_checkpoint_root"] / stage / "latest.json")
        latest_receipt = read_json(
            location["extension_checkpoint_root"] / stage / latest_pointer["receipt"]
        )
        restore_receipt = _fresh_process_restore(latest_bundle, stage_bindings, stage, budget)
        if restore_receipt.get("digest") != latest_receipt.get("digest"):
            raise Refuse("fresh-process restore digest differs from the latest receipt")

        receipts: dict[str, Any] = {}
        final_payload: dict[str, Any] | None = None
        for current_stage, corpus, epochs, expected_episodes, expected_updates in (
            ("pretraining", pretrain, 8, 30_096, 14_517_912),
            ("assistant_sft", assistant, 1, 525, 113_209),
        ):
            store_directory = location["extension_checkpoint_root"] / current_stage
            store_bindings = {**context["parent_bindings"], "stage": current_stage}
            store_context = _open_extension_store(
                store_directory, store_bindings, corpus, resume=resume
            )
            with store_context as store:
                latest_path = store_directory / "latest.json"
                if latest_path.exists():
                    trainer, cursor, envelope = store.load_latest()
                    if current_stage == "pretraining" and cursor.epoch == epochs:
                        final_payload = trainer.checkpoint()
                elif current_stage == "assistant_sft":
                    if final_payload is None:
                        raise Refuse("assistant SFT requires the completed pretraining final")
                    trainer, _encoder = base._assistant_sft_trainer(
                        Taiji.from_checkpoint(final_payload["model"], device="cpu"),
                        assistant,
                        final_payload["code_revision"],
                    )
                    cursor = recovery.Cursor()
                    receipt = frozen.commit(store, trainer, cursor, epochs, budget)
                    receipts["assistant_sft_zero"] = receipt
                    _append_cpu_snapshot(
                        arm,
                        "durable_training_commit",
                        budget,
                        {
                            "stage": current_stage,
                            "episodes": trainer.episode_count,
                            "cursor": {"epoch": cursor.epoch, "index": cursor.index},
                            "receipt": receipt,
                        },
                        progress_path,
                    )
                else:
                    raise Refuse(f"missing versioned stage checkpoint: {current_stage}")

                while cursor.epoch < epochs:
                    throughput_projection = None
                    if trainer.episode_count % SAVE_INTERVAL == 0:
                        throughput_projection = _throughput_projection(
                            arm,
                            prepared["parents"],
                            current_stage=current_stage,
                            current_episodes=trainer.episode_count,
                        )
                        if not throughput_projection["feasible"]:
                            budget.pending_stop_details = {
                                "stop_reason": "throughput_projection_exceeds_remaining_budget",
                                "throughput_projection": throughput_projection,
                            }
                            raise Refuse(
                                "measured throughput projects remaining work beyond approved CPU headroom"
                            )
                    try:
                        budget.guard()
                    except Refuse:
                        if throughput_projection is not None:
                            budget.pending_stop_details = {
                                **budget.pending_stop_details,
                                "throughput_projection": throughput_projection,
                            }
                        raise
                    _record, cursor = recovery.train_next(trainer, cursor, epochs)
                    checkpoint_due = trainer.episode_count % SAVE_INTERVAL == 0 or cursor.index == 0
                    if checkpoint_due:
                        receipt = frozen.commit(store, trainer, cursor, epochs, budget)
                        receipts[current_stage] = receipt
                        commit_details = {
                            "stage": current_stage,
                            "episodes": trainer.episode_count,
                            "target_byte_updates": trainer.global_step,
                            "cursor": {"epoch": cursor.epoch, "index": cursor.index},
                            "receipt": receipt,
                        }
                        if throughput_projection is not None:
                            commit_details["throughput_projection"] = throughput_projection
                        _append_cpu_snapshot(
                            arm,
                            "durable_training_commit",
                            budget,
                            commit_details,
                            progress_path,
                        )
                    else:
                        episode_details = {
                            "stage": current_stage,
                            "episodes": trainer.episode_count,
                            "target_byte_updates": trainer.global_step,
                            "cursor": {"epoch": cursor.epoch, "index": cursor.index},
                        }
                        if throughput_projection is not None:
                            episode_details["throughput_projection"] = throughput_projection
                        _append_cpu_snapshot(
                            arm,
                            "episode_completed",
                            budget,
                            episode_details,
                            progress_path,
                        )
                if (
                    trainer.episode_count != expected_episodes
                    or trainer.global_step != expected_updates
                ):
                    raise Refuse(f"{current_stage} terminal denominator differs from frozen plan")
                final_payload = trainer.checkpoint()
                final_path = location["extension_checkpoint_root"] / f"{current_stage}_final.pt"
                if not final_path.exists():
                    _extension_publish(
                        final_path,
                        lambda stream, payload=final_payload: torch.save(payload, stream),
                    )
                elif torch.load(final_path, map_location="cpu", weights_only=True).get(
                    "checkpoint_digest"
                ) != final_payload.get("checkpoint_digest"):
                    raise Refuse(f"existing {current_stage} final differs from restored trainer")
                restore = _fresh_final_restore(final_path, current_stage, budget)
                receipts[f"{current_stage}_final"] = {
                    "path": str(final_path.resolve()),
                    "sha256": sha256_file(final_path),
                    "checkpoint_digest": final_payload["checkpoint_digest"],
                    "restore": restore,
                    "episodes": trainer.episode_count,
                    "target_byte_updates": trainer.global_step,
                }
                _append_cpu_snapshot(
                    arm,
                    "final_stage_restored",
                    budget,
                    {"stage": current_stage, "restore": restore},
                    progress_path,
                )

        # Pin current source and parent identity again before publishing results.
        final_parent = _parent_context(arm)
        if (
            final_parent["parent_attempt_sha256"] != context["parent_attempt_sha256"]
            or final_parent["parent_receipt_sha256"] != context["parent_receipt_sha256"]
            or final_parent["source_compatibility"] != context["source_compatibility"]
        ):
            raise Refuse("parent/source compatibility changed during extension")
        budget.guard(reserve=FINAL_PUBLICATION_RESERVE_SECONDS)
        final_charge = budget.used()
        ready = {
            "schema": "taiji-r2-dialogue-continuation-extension-readiness-v1",
            "status": "ready",
            "arm": arm,
            "parent_attempt_sha256": context["parent_attempt_sha256"],
            "parent_receipt_sha256": context["parent_receipt_sha256"],
            "extension_attempt_sha256": sha256_file(location["extension_attempt"]),
            "extension_runner_sha256": attempt["continuation_sources"]["runner_sha256"],
            "source_revision_sha256": attempt.get("source_revision_sha256"),
            "source_compatibility_report_sha256": attempt["source_compatibility_report_sha256"],
            "stages": receipts,
            "cpu_seconds_cumulative": final_charge,
            "cpu_seconds_cumulative_is_lower_bound": True,
            "cpu_seconds_cumulative_boundary": "measured_before_final_result_publication",
            "completion_cpu_source": "continuation_completed_ledger_event",
            "dev_queries": 0,
            "reference_queries": 0,
            "generation_queries": 0,
            "can_promote": False,
        }
        ready_path = location["extension_result"].with_name(
            location["extension_result"].stem + "_readiness.json"
        )
        _write_json_once(ready_path, ready)
        result = {
            "schema": "taiji-r2-dialogue-continuation-extension-result-v1",
            "status": "complete",
            "complete": True,
            "arm": arm,
            "parent_attempt_sha256": context["parent_attempt_sha256"],
            "parent_checkpoint_receipt_sha256": context["parent_receipt_sha256"],
            "extension_attempt_sha256": ready["extension_attempt_sha256"],
            "readiness_path": str(ready_path.resolve()),
            "readiness_sha256": sha256_file(ready_path),
            "source_revision_sha256": attempt.get("source_revision_sha256"),
            "stages": receipts,
            "cpu_seconds_cumulative": final_charge,
            "cpu_seconds_cumulative_is_lower_bound": True,
            "cpu_seconds_cumulative_boundary": "measured_before_final_result_publication",
            "completion_cpu_source": "continuation_completed_ledger_event",
            "resources_at_training_start": resources,
            "dev_queries": 0,
            "reference_queries": 0,
            "generation_queries": 0,
            "can_promote": False,
        }
        _write_json_once(location["extension_result"], result)
        _append_cpu_snapshot(
            arm,
            "continuation_completed",
            budget,
            {
                "parent_attempt_sha256": context["parent_attempt_sha256"],
                "readiness_sha256": result["readiness_sha256"],
                "result_sha256": sha256_file(location["extension_result"]),
                "result_cpu_snapshot_boundary": "measured_before_final_result_publication",
                "completion_cpu_source": "continuation_completed_ledger_event",
            },
            progress_path,
        )
        return result
    except BaseException as exc:
        if budget is not None:
            _persist_stop_event(
                arm,
                budget,
                progress_path,
                context["parent_attempt_sha256"],
                exc,
            )
        raise
    finally:
        if root_entered and root_context is not None:
            root_context.__exit__(*sys.exc_info())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--prepare-source-revision", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.prepare_only or args.prepare_source_revision:
            if args.arm is not None:
                raise Refuse("--prepare-only is global; omit --arm")
            record = prepare_source_revision() if args.prepare_source_revision else prepare_only()
            print(
                json.dumps({"status": "prepared", "arms": list(ARMS), "charges": record["charges"]})
            )
            return 0
        if args.arm is None:
            raise Refuse("--run/--resume requires exactly one --arm")
        if args.run == args.resume:
            raise Refuse("select exactly one of --run or --resume")
        result = execute_arm(args.arm, resume=args.resume)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "arm": args.arm,
                    "result_sha256": sha256_file(paths(args.arm)["extension_result"]),
                }
            )
        )
        return 0
    except Exception as exc:
        print(f"R2 continuation extension stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

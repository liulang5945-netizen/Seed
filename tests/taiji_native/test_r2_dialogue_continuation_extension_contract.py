from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

from scripts.training import taiji_r2_dialogue_continuation_extension as extension


class FakeDurabilityApi:
    def __init__(
        self,
        volume_root: str,
        *,
        filesystem: str = "NTFS",
        drive_type: int | None = None,
        fail_flush: bool = False,
        fail_move: bool = False,
        tamper_readback: bool = False,
    ):
        self.volume_root = volume_root
        self.filesystem = filesystem
        self.drive_type = extension.DRIVE_FIXED if drive_type is None else drive_type
        self.fail_flush = fail_flush
        self.fail_move = fail_move
        self.tamper_readback = tamper_readback
        self.move_flags: list[int] = []

    def inspect(self, path: Path) -> dict[str, object]:
        return {
            "is_unc": False,
            "drive_type": self.drive_type,
            "filesystem": self.filesystem,
            "volume_root": self.volume_root,
        }

    def flush_file(self, fd: int) -> None:
        if self.fail_flush:
            raise OSError("injected FlushFileBuffers failure")
        os.fsync(fd)

    def move_file_write_through(self, source: Path, destination: Path, flags: int) -> None:
        self.move_flags.append(flags)
        if self.fail_move:
            raise OSError("injected MoveFileExW failure")
        assert flags in {
            extension.MOVEFILE_WRITE_THROUGH,
            extension.MOVEFILE_WRITE_THROUGH | extension.MOVEFILE_REPLACE_EXISTING,
        }
        if flags & extension.MOVEFILE_REPLACE_EXISTING:
            os.replace(source, destination)
        else:
            os.rename(source, destination)
        if self.tamper_readback:
            destination.write_bytes(b"tampered")


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extension, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(extension, "CONTRACT_PATH", tmp_path / "extension-contract.json")
    monkeypatch.setattr(extension, "AGGREGATE_LEDGER_PREFIX", tmp_path / "aggregate-ledger")
    monkeypatch.setattr(
        extension, "AGGREGATE_LEDGER_SUMMARY", tmp_path / "aggregate-ledger.latest.json"
    )
    monkeypatch.setattr(extension, "LOCK_PATH", tmp_path / "aggregate-ledger.lock")
    monkeypatch.setattr(extension, "_DURABILITY_API", FakeDurabilityApi(str(tmp_path.anchor)))


def _parent_records(charges: dict[str, float] | None = None) -> dict[str, dict[str, object]]:
    charges = charges or {
        "full_turn_1m": 780_628.774,
        "warm_start": 780_081.702,
        "fresh_same_capacity": 779_196.622,
    }
    result: dict[str, dict[str, object]] = {}
    for index, arm in enumerate(extension.ARMS):
        result[arm] = {
            "parent_attempt_sha256": f"{index + 1:064x}",
            "parent_receipt_sha256": f"{index + 10:064x}",
            "parent_bundle_sha256": f"{index + 20:064x}",
            "charged_cpu_seconds": charges[arm],
            "tail_status": "conservative_upper_bound",
            "accounting_start": {
                "snapshot_utc": "2026-10-22T05:30:50.602314+00:00",
                "boot_identity": "boot-before-update",
                "host_cpu_seconds": 1_582_496.5 + index,
                "logical_cpu_upper": 24,
            },
        }
    return result


def _initialize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    _configure(tmp_path, monkeypatch)
    extension.initialize_ledger(_parent_records(), "a" * 64, "2026-10-22")
    return extension.latest_charges()


def _leave_contract_without_initial_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, dict[str, object]], str, str]:
    _configure(tmp_path, monkeypatch)
    parents = _parent_records()
    source_sha = "a" * 64
    approved_utc = "2026-10-22"
    original_write = extension._durable_write
    initial_event_path = extension._event_path(extension.AGGREGATE_LEDGER_PREFIX, 0)

    def fail_initial_event(path, payload, qualification, api, *, replace=False):
        if Path(path) == initial_event_path:
            raise OSError("injected interruption after contract publication")
        return original_write(path, payload, qualification, api, replace=replace)

    monkeypatch.setattr(extension, "_durable_write", fail_initial_event)
    with pytest.raises(OSError, match="after contract publication"):
        extension.initialize_ledger(parents, source_sha, approved_utc)
    assert extension.CONTRACT_PATH.is_file()
    assert not initial_event_path.exists()
    assert not extension.AGGREGATE_LEDGER_SUMMARY.exists()
    monkeypatch.setattr(extension, "_durable_write", original_write)
    return parents, source_sha, approved_utc


def _reserve_in_child(
    report_dir: str,
    arm: str,
    aggregate_limit: float,
    barrier: Any,
    result_queue: Any,
) -> None:
    report_path = Path(report_dir)
    extension.REPORTS_DIR = report_path
    extension.CONTRACT_PATH = report_path / "extension-contract.json"
    extension.AGGREGATE_LEDGER_PREFIX = report_path / "aggregate-ledger"
    extension.AGGREGATE_LEDGER_SUMMARY = report_path / "aggregate-ledger.latest.json"
    extension.LOCK_PATH = report_path / "aggregate-ledger.lock"
    extension.AGGREGATE_CPU_LIMIT = aggregate_limit
    extension._DURABILITY_API = FakeDurabilityApi(str(report_path.anchor))
    try:
        barrier.wait(timeout=15)
        token = extension.guard(arm, 700_000.0, 250_000.0)
        result_queue.put(("reserved", arm, token))
    except Exception as exc:
        result_queue.put(("refused", arm, f"{type(exc).__name__}: {exc}"))


def _read_resume_state_in_child(
    report_dir: str, aggregate_limit: float, arm: str, result_queue: Any
) -> None:
    report_path = Path(report_dir)
    extension.REPORTS_DIR = report_path
    extension.CONTRACT_PATH = report_path / "extension-contract.json"
    extension.AGGREGATE_LEDGER_PREFIX = report_path / "aggregate-ledger"
    extension.AGGREGATE_LEDGER_SUMMARY = report_path / "aggregate-ledger.latest.json"
    extension.LOCK_PATH = report_path / "aggregate-ledger.lock"
    extension.AGGREGATE_CPU_LIMIT = aggregate_limit
    extension._DURABILITY_API = FakeDurabilityApi(str(report_path.anchor))
    result_queue.put((extension.latest_charges()[arm], extension.latest_reservations()[arm]))


def test_paths_return_stable_lineage_keys_and_reject_unknown_arm() -> None:
    result = extension.paths("warm_start")
    assert set(result) == {
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
    assert result["parent_attempt"].name.endswith("_warm_start_attempt.json")
    assert result["extension_checkpoint_root"].parts[-2:] == ("continuation_v1", "warm_start")
    with pytest.raises(ValueError, match="unknown R2 extension arm"):
        extension.paths("other")


def test_initialization_is_write_once_and_records_all_parent_charges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    charges = _initialize(tmp_path, monkeypatch)
    assert charges == {
        arm: float(_parent_records()[arm]["charged_cpu_seconds"]) for arm in extension.ARMS
    }
    assert (tmp_path / "extension-contract.json").is_file()
    assert (tmp_path / "aggregate-ledger.00000000.json").is_file()
    assert (tmp_path / "aggregate-ledger.latest.json").is_file()
    with pytest.raises(FileExistsError, match="write-once"):
        extension.initialize_ledger(_parent_records(), "a" * 64, "2026-10-22")


def test_reconcile_repairs_only_missing_initial_event_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parents, source_sha, approved_utc = _leave_contract_without_initial_event(tmp_path, monkeypatch)
    result = extension.reconcile_initialize_ledger(parents, source_sha, approved_utc)
    event_path = extension._event_path(extension.AGGREGATE_LEDGER_PREFIX, 0)
    event_bytes = event_path.read_bytes()
    expected = {arm: float(parents[arm]["charged_cpu_seconds"]) for arm in extension.ARMS}
    assert result["charges_by_arm"] == expected
    assert result["current_charges_by_arm"] == expected
    assert result["reservations_by_arm"] == {}
    assert event_path.is_file()
    assert extension.AGGREGATE_LEDGER_SUMMARY.is_file()
    assert len(extension._event_files(extension.AGGREGATE_LEDGER_PREFIX)) == 1

    repeated = extension.reconcile_initialize_ledger(parents, source_sha, approved_utc)
    assert repeated["contract"]["contract_sha256"] == result["contract"]["contract_sha256"]
    assert repeated["current_charges_by_arm"] == expected
    assert event_path.read_bytes() == event_bytes
    assert len(extension._event_files(extension.AGGREGATE_LEDGER_PREFIX)) == 1


@pytest.mark.parametrize("mismatch", ["parent", "report", "approval"])
def test_reconcile_refuses_initialization_input_mismatch_without_publishing_event(
    mismatch: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parents, source_sha, approved_utc = _leave_contract_without_initial_event(tmp_path, monkeypatch)
    if mismatch == "parent":
        parents["warm_start"]["parent_bundle_sha256"] = "f" * 64
    elif mismatch == "report":
        source_sha = "b" * 64
    else:
        approved_utc = "2026-10-23"

    with pytest.raises(extension.ExtensionLedgerRefuse, match="differ from the durable contract"):
        extension.reconcile_initialize_ledger(parents, source_sha, approved_utc)
    assert not extension._event_path(extension.AGGREGATE_LEDGER_PREFIX, 0).exists()


@pytest.mark.parametrize(
    ("artifact", "error"),
    [
        ("summary", "summary exists without immutable event zero"),
        ("sequence_gap", "sequence has a gap"),
        ("unknown", "unknown aggregate initialization artifact"),
        ("temp", "uncommitted aggregate ledger temp"),
        ("corrupt_event", "event schema/sequence mismatch"),
    ],
)
def test_reconcile_refuses_partial_unknown_or_corrupt_event_state(
    artifact: str,
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents, source_sha, approved_utc = _leave_contract_without_initial_event(tmp_path, monkeypatch)
    prefix = extension.AGGREGATE_LEDGER_PREFIX
    if artifact == "summary":
        extension.AGGREGATE_LEDGER_SUMMARY.write_text("{}", encoding="utf-8")
    elif artifact == "sequence_gap":
        extension._event_path(prefix, 1).write_text("{}", encoding="utf-8")
    elif artifact == "unknown":
        prefix.with_name(prefix.name + ".unknown.json").write_text("{}", encoding="utf-8")
    elif artifact == "temp":
        (prefix.parent / f".{prefix.name}.00000000.json.abandoned.tmp").write_bytes(b"partial")
    else:
        extension._event_path(prefix, 0).write_text(
            json.dumps({"event": "extension_budget_initialized"}), encoding="utf-8"
        )
    before = {path.name: path.read_bytes() for path in prefix.parent.iterdir() if path.is_file()}

    with pytest.raises(extension.ExtensionLedgerRefuse, match=error):
        extension.reconcile_initialize_ledger(parents, source_sha, approved_utc)
    after = {path.name: path.read_bytes() for path in prefix.parent.iterdir() if path.is_file()}
    assert after == before


def test_reconcile_validates_existing_full_event_chain_without_rewriting_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parents = _parent_records()
    source_sha = "a" * 64
    approved_utc = "2026-10-22"
    _configure(tmp_path, monkeypatch)
    extension.initialize_ledger(parents, source_sha, approved_utc)
    charge = float(parents["warm_start"]["charged_cpu_seconds"])
    token = extension.guard("warm_start", charge, reserve_cpu_seconds=10.0)
    event_files = extension._event_files(extension.AGGREGATE_LEDGER_PREFIX)
    before = {sequence: path.read_bytes() for sequence, path in event_files}

    result = extension.reconcile_initialize_ledger(parents, source_sha, approved_utc)

    assert result["reservations_by_arm"]["warm_start"]["reservation_id"] == token
    assert result["current_charges_by_arm"]["warm_start"] == charge
    assert {
        sequence: path.read_bytes()
        for sequence, path in extension._event_files(extension.AGGREGATE_LEDGER_PREFIX)
    } == before


def test_reconcile_refuses_a_broken_existing_hash_chain_without_rewriting_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parents = _parent_records()
    source_sha = "a" * 64
    approved_utc = "2026-10-22"
    _configure(tmp_path, monkeypatch)
    extension.initialize_ledger(parents, source_sha, approved_utc)
    charge = float(parents["warm_start"]["charged_cpu_seconds"])
    extension.guard("warm_start", charge, reserve_cpu_seconds=10.0)
    event_path = extension._event_path(extension.AGGREGATE_LEDGER_PREFIX, 1)
    damaged = json.loads(event_path.read_text(encoding="utf-8"))
    damaged["previous_event_sha256"] = "0" * 64
    damaged["event_sha256"] = extension._digest(damaged, omit="event_sha256")
    event_path.write_bytes(extension._canonical_json(damaged))
    before = {
        sequence: path.read_bytes()
        for sequence, path in extension._event_files(extension.AGGREGATE_LEDGER_PREFIX)
    }

    with pytest.raises(extension.ExtensionLedgerRefuse, match="hash chain is broken"):
        extension.reconcile_initialize_ledger(parents, source_sha, approved_utc)
    assert {
        sequence: path.read_bytes()
        for sequence, path in extension._event_files(extension.AGGREGATE_LEDGER_PREFIX)
    } == before


def test_guard_durably_reserves_and_settlement_requires_matching_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = _initialize(tmp_path, monkeypatch)
    arm = "warm_start"
    token = extension.guard(arm, initial[arm], reserve_cpu_seconds=20_000.0)
    assert len(token) == 32
    assert extension.latest_charges()[arm] == initial[arm] + 20_000.0
    reservation = extension.latest_reservations()[arm]
    assert reservation["reservation_id"] == token
    assert reservation["reserved_cpu_seconds"] == 20_000.0
    with pytest.raises(extension.ExtensionLedgerRefuse, match="prior reservation must be settled"):
        extension.guard(arm, initial[arm], reserve_cpu_seconds=100.0)
    with pytest.raises(extension.ExtensionLedgerRefuse, match="token mismatch"):
        extension.append_charge(
            arm, "cpu_snapshot", initial[arm] + 1_000.0, {"reservation_id": "0" * 32}
        )

    actual = initial[arm] + 8_000.0
    settled = extension.append_charge(
        arm, "continuation_started", actual, {"reservation_id": token, "restore_cpu": 8_000.0}
    )
    assert settled["reservation_id"] == token
    assert extension.latest_charges()[arm] == actual
    assert arm not in extension.latest_reservations()
    with pytest.raises(extension.ExtensionLedgerRefuse, match="no active CPU reservation"):
        extension.append_charge(arm, "duplicate_settlement", actual, {"reservation_id": token})


def test_latest_charges_fast_uses_a_verified_current_summary_tip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = _initialize(tmp_path, monkeypatch)
    arm = "warm_start"
    extension.guard(arm, initial[arm], reserve_cpu_seconds=20_000.0)
    monkeypatch.setattr(
        extension,
        "_read_ledger_full",
        lambda **_kwargs: pytest.fail("current summary tip should avoid a full chain replay"),
    )

    effective = extension.latest_charges_fast()

    assert effective[arm] == initial[arm] + 20_000.0
    assert sum(effective.values()) == sum(initial.values()) + 20_000.0


def test_unsettled_reservation_is_charged_and_reconciled_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = _initialize(tmp_path, monkeypatch)
    arm = "full_turn_1m"
    reserve = 1_200.0
    token = extension.guard(arm, initial[arm], reserve_cpu_seconds=reserve)
    effective = extension.latest_charges()[arm]
    assert effective == initial[arm] + reserve
    pending = extension.latest_reservations()[arm]
    resumed = extension.append_charge(
        arm,
        "continuation_resumed",
        effective,
        {"reservation_id": pending["reservation_id"], "reconciled_reserved_upper": True},
    )
    assert resumed["reservation_id"] == token
    assert extension.latest_charges()[arm] == effective
    assert extension.latest_reservations() == {}
    next_token = extension.guard(arm, effective, reserve_cpu_seconds=500.0)
    assert next_token != token


def test_unresolved_or_partial_parent_tail_refuses_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(tmp_path, monkeypatch)
    records = _parent_records()
    records["fresh_same_capacity"]["tail_status"] = "unresolved"
    with pytest.raises(extension.ExtensionLedgerRefuse, match="unresolved or partial"):
        extension.initialize_ledger(records, "a" * 64, "2026-10-22")
    assert not (tmp_path / "extension-contract.json").exists()

    records = _parent_records()
    del records["warm_start"]["accounting_start"]
    with pytest.raises(extension.ExtensionLedgerRefuse, match="complete accounting snapshot"):
        extension.initialize_ledger(records, "a" * 64, "2026-10-22")


def test_guard_respects_per_arm_reserve_and_actual_settlement_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    charges = {
        "full_turn_1m": extension.PER_ARM_CPU_LIMIT - 25_000.0,
        "warm_start": 500_000.0,
        "fresh_same_capacity": 500_000.0,
    }
    _configure(tmp_path, monkeypatch)
    extension.initialize_ledger(_parent_records(charges), "a" * 64, "2026-10-22")
    with pytest.raises(extension.ExtensionLedgerRefuse, match="insufficient per-arm"):
        extension.guard(
            "full_turn_1m",
            extension.PER_ARM_CPU_LIMIT - 25_000.0,
            reserve_cpu_seconds=50_000.0,
        )
    token = extension.guard(
        "full_turn_1m",
        extension.PER_ARM_CPU_LIMIT - 25_000.0,
        reserve_cpu_seconds=24_999.0,
    )
    with pytest.raises(extension.ExtensionLedgerRefuse, match="per-arm cap"):
        extension.append_charge(
            "full_turn_1m",
            "bad_settlement",
            extension.PER_ARM_CPU_LIMIT + 1.0,
            {"reservation_id": token},
        )


def test_two_processes_cannot_overreserve_shared_aggregate_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extension, "AGGREGATE_CPU_LIMIT", 2_500_000.0)
    charges = {arm: 700_000.0 for arm in extension.ARMS}
    _configure(tmp_path, monkeypatch)
    extension.initialize_ledger(_parent_records(charges), "a" * 64, "2026-10-22")

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_reserve_in_child,
            args=(str(tmp_path), arm, 2_500_000.0, barrier, result_queue),
        )
        for arm in ("full_turn_1m", "warm_start")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    results = [result_queue.get(timeout=5) for _ in processes]
    assert sorted(result[0] for result in results) == ["refused", "reserved"]
    assert len(list(tmp_path.glob("aggregate-ledger.000000*.json"))) == 2

    pending = extension.latest_reservations()
    assert len(pending) == 1
    reserved_arm, reservation = next(iter(pending.items()))
    restart_queue = context.Queue()
    restart = context.Process(
        target=_read_resume_state_in_child,
        args=(str(tmp_path), 2_500_000.0, reserved_arm, restart_queue),
    )
    restart.start()
    restart.join(timeout=20)
    assert restart.exitcode == 0
    effective_charge, restored_reservation = restart_queue.get(timeout=5)
    assert effective_charge == 950_000.0
    assert restored_reservation == reservation
    extension.append_charge(
        reserved_arm,
        "continuation_resumed",
        effective_charge,
        {"reservation_id": reservation["reservation_id"], "reconciled_reserved_upper": True},
    )
    assert extension.latest_reservations() == {}
    assert sum(extension.latest_charges().values()) == 2_350_000.0


def test_incomplete_temp_or_corrupt_chain_refuses_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _initialize(tmp_path, monkeypatch)
    (tmp_path / ".aggregate-ledger.00000001.json.abandoned.tmp").write_bytes(b"partial")
    with pytest.raises(extension.ExtensionLedgerRefuse, match="temp file"):
        extension.latest_charges()

    (tmp_path / ".aggregate-ledger.00000001.json.abandoned.tmp").unlink()
    event_path = tmp_path / "aggregate-ledger.00000000.json"
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["charges_by_arm"]["full_turn_1m"] += 1
    event_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(extension.ExtensionLedgerRefuse, match="digest mismatch"):
        extension.latest_charges()


def test_durable_publish_binary_writer_qualifies_volume_and_returns_readback_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(tmp_path, monkeypatch)
    directory = tmp_path / "checkpoint-store"
    directory.mkdir()
    api = FakeDurabilityApi(str(tmp_path.anchor))
    monkeypatch.setattr(extension, "_DURABILITY_API", api)
    target = directory / "bundle.pt"
    payload = b"\x00frozen-bundle\xff"

    qualification = extension.qualify_storage_target(
        target, expected_volume_root=str(tmp_path.anchor)
    )
    assert qualification["filesystem"] == "NTFS"
    digest = extension.durable_publish(
        target,
        lambda stream: stream.write(payload),
        expected_volume_root=qualification["volume_root"],
    )
    assert target.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert api.move_flags == [extension.MOVEFILE_WRITE_THROUGH]
    assert not list(directory.glob(".*.tmp"))


def test_durable_publish_replace_is_explicit_and_volume_mismatch_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(tmp_path, monkeypatch)
    directory = tmp_path / "reports"
    directory.mkdir()
    api = FakeDurabilityApi(str(tmp_path.anchor))
    monkeypatch.setattr(extension, "_DURABILITY_API", api)
    target = directory / "summary.json"
    extension.durable_publish(target, lambda stream: stream.write(b"old"))
    with pytest.raises(FileExistsError):
        extension.durable_publish(target, lambda stream: stream.write(b"overwrite"))
    extension.durable_publish(target, lambda stream: stream.write(b"new"), replace=True)
    assert target.read_bytes() == b"new"
    assert api.move_flags[-1] == (
        extension.MOVEFILE_WRITE_THROUGH | extension.MOVEFILE_REPLACE_EXISTING
    )

    called = False

    def writer(stream) -> None:
        nonlocal called
        called = True
        stream.write(b"must-not-publish")

    with pytest.raises(extension.ExtensionLedgerRefuse, match="differs from approved volume"):
        extension.durable_publish(
            directory / "wrong-volume.bin",
            writer,
            expected_volume_root="Z:\\",
        )
    assert not called

    unsupported = FakeDurabilityApi(str(tmp_path.anchor), filesystem="ReFS")
    monkeypatch.setattr(extension, "_DURABILITY_API", unsupported)
    with pytest.raises(extension.ExtensionLedgerRefuse, match="only local fixed-volume NTFS"):
        extension.qualify_storage_target(directory)


@pytest.mark.parametrize("failure", ["writer", "flush", "move"])
def test_durable_publish_failure_leaves_no_destination_or_temp(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(tmp_path, monkeypatch)
    directory = tmp_path / "checkpoint-store"
    directory.mkdir()
    api = FakeDurabilityApi(
        str(tmp_path.anchor), fail_flush=failure == "flush", fail_move=failure == "move"
    )
    monkeypatch.setattr(extension, "_DURABILITY_API", api)
    target = directory / "candidate.pt"

    def writer(stream) -> None:
        stream.write(b"partial-or-complete")
        if failure == "writer":
            raise RuntimeError("injected writer failure")

    expected = RuntimeError if failure == "writer" else OSError
    with pytest.raises(expected):
        extension.durable_publish(target, writer, expected_volume_root=str(tmp_path.anchor))
    assert not target.exists()
    assert not list(directory.glob(".*.tmp"))


def test_durable_publish_detects_readback_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(tmp_path, monkeypatch)
    directory = tmp_path / "checkpoint-store"
    directory.mkdir()
    api = FakeDurabilityApi(str(tmp_path.anchor), tamper_readback=True)
    monkeypatch.setattr(extension, "_DURABILITY_API", api)
    target = directory / "candidate.pt"
    with pytest.raises(extension.ExtensionLedgerRefuse, match="readback SHA-256 mismatch"):
        extension.durable_publish(
            target,
            lambda stream: stream.write(b"good-bytes"),
            expected_volume_root=str(tmp_path.anchor),
        )
    assert target.read_bytes() == b"tampered"
    assert not list(directory.glob(".*.tmp"))

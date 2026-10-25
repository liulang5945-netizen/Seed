from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as runner
from scripts.training import taiji_r2_dialogue_continuation_extension as ledger


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_oid(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fake_write_through_publish(path, writer, *, replace=False, expected_volume_root=None):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".synthetic.tmp")
    with temporary.open("wb") as stream:
        writer(stream)
        stream.flush()
    if replace:
        temporary.replace(path)
    else:
        temporary.rename(path)
    return _sha(path.read_bytes())


def _configure_synthetic_publisher(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(runner, "_APPROVED_VOLUME_ROOT", "SYNTHETIC:\\")
    monkeypatch.setattr(runner, "REPORTS_DIR", root)
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", root / "extension")
    monkeypatch.setattr(runner, "durable_publish", _fake_write_through_publish)


def test_runner_and_ledger_share_approved_budget_constants() -> None:
    # 批 B 入库 source_revision 链后，import 期有效预算=批准的 v1.2/v1.3 值
    # （owner 批准的 D1 预算链）；本测钉的是 runner 与 ledger 的共享一致性。
    assert ledger.PER_ARM_CPU_LIMIT == runner.PER_ARM_CPU_LIMIT == 1_500_000.0
    assert ledger.AGGREGATE_CPU_LIMIT == runner.AGGREGATE_CPU_LIMIT == 4_300_000.0


def test_tail_charge_uses_wall_times_cpu_upper_bound_after_boot_change() -> None:
    snapshot = {
        "charged_cpu_seconds": 100.0,
        "snapshot_utc": "2026-10-22T00:00:00+00:00",
        "boot_identity": 10.0,
        "host_cpu_seconds": 900.0,
        "logical_cpu_upper": 8,
    }
    tail = runner.tail_charge_from_snapshot(
        snapshot,
        now=datetime(2026, 10, 22, 0, 0, 10, tzinfo=UTC),
        current_boot_identity=11.0,
        current_host_cpu_seconds=999.0,
        current_logical_cpu_count=24,
    )
    assert tail == 340.0


def test_tail_charge_uses_host_counter_when_boot_is_unchanged() -> None:
    snapshot = {
        "charged_cpu_seconds": 100.0,
        "snapshot_utc": "2026-10-22T00:00:00+00:00",
        "boot_identity": 10.0,
        "host_cpu_seconds": 900.0,
        "logical_cpu_upper": 8,
    }
    tail = runner.tail_charge_from_snapshot(
        snapshot,
        now=datetime(2026, 10, 22, 1, 0, 0, tzinfo=UTC),
        current_boot_identity=10.0,
        current_host_cpu_seconds=960.0,
        current_logical_cpu_count=24,
    )
    assert tail == 160.0


def test_source_compatibility_requires_a_pinned_git_blob_and_current_sha(
    tmp_path: Path, monkeypatch
) -> None:
    relative = "training.py"
    source = tmp_path / relative
    source.write_bytes(b"current reviewed source\n")
    report = tmp_path / "architecture.md"
    report.write_text("reviewed source compatibility\n", encoding="utf-8")
    parent_bytes = b"parent source bytes\n"
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", report)
    monkeypatch.setattr(
        runner,
        "SOURCE_COMPATIBILITY",
        {
            relative: {
                "parent_sha256": _sha(parent_bytes),
                "current_sha256": _sha(source.read_bytes()),
                "git_revision": "a" * 40,
            }
        },
    )

    def run_git(args, **kwargs):
        if args[1] == "rev-parse" and args[2] == "--verify":
            stdout = b"a" * 40 + b"\n"
        elif args[1] == "rev-parse":
            stdout = _git_blob_oid(parent_bytes).encode() + b"\n"
        else:
            stdout = parent_bytes
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(runner.subprocess, "run", run_git)

    result = runner.validate_source_compatibility("full_turn_1m", {str(source): _sha(parent_bytes)})
    assert result["files"][0]["compatibility"] == "reviewed_parent_blob_compatibility"
    assert result["files"][0]["git_blob_sha256"] == _sha(parent_bytes)
    assert result["files"][0]["git_revision"] == "a" * 40
    assert result["files"][0]["git_blob_oid"] == _git_blob_oid(parent_bytes)
    assert result["report_sha256"] == _sha(report.read_bytes())

    source.write_bytes(b"unreviewed source drift\n")
    with pytest.raises(runner.Refuse, match="unreviewed source drift"):
        runner.validate_source_compatibility("full_turn_1m", {str(source): _sha(parent_bytes)})


@pytest.mark.parametrize(
    ("source_bytes", "blob_bytes"),
    [
        (b"exact parent and HEAD source\r\n", b"exact parent and HEAD source\n"),
        (b"exact parent and HEAD source\n", b"exact parent and HEAD source\r\n"),
    ],
)
def test_exact_source_compatibility_accepts_git_clean_filter_for_windows_eol(
    tmp_path: Path, monkeypatch, source_bytes: bytes, blob_bytes: bytes
) -> None:
    relative = "training.py"
    source = tmp_path / relative
    source.write_bytes(source_bytes)
    report = tmp_path / "architecture.md"
    report.write_text("reviewed source compatibility\n", encoding="utf-8")
    commit = "c" * 40
    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", report)
    monkeypatch.setattr(runner, "SOURCE_COMPATIBILITY", {})

    def run(args, **kwargs):
        commands.append(args)
        if args[1] == "rev-parse":
            stdout = (
                f"{commit}\n".encode()
                if args[2] == "--verify"
                else f"{_git_blob_oid(blob_bytes)}\n".encode()
            )
            return SimpleNamespace(returncode=0, stdout=stdout)
        if args[1] == "show":
            return SimpleNamespace(returncode=0, stdout=blob_bytes)
        assert args[1] == "hash-object"
        return SimpleNamespace(returncode=0, stdout=f"{_git_blob_oid(blob_bytes)}\n".encode())

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.validate_source_compatibility("full_turn_1m", {str(source): _sha(source_bytes)})

    row = result["files"][0]
    assert row["compatibility"] == "exact_parent_bytes"
    assert row["git_revision"] == commit
    assert row["current_sha256"] == _sha(source_bytes)
    assert row["git_blob_oid"] == _git_blob_oid(blob_bytes)
    assert row["git_blob_sha256"] == _sha(blob_bytes)
    assert row["git_filter_normalization"] == "working_tree_clean_filter"
    assert row["git_worktree_blob_oid"] == _git_blob_oid(blob_bytes)
    assert commands == [
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        ["git", "rev-parse", f"{commit}:{relative}"],
        ["git", "show", f"{commit}:{relative}"],
        ["git", "hash-object", f"--path={relative}", str(source.resolve())],
    ]


def test_exact_source_compatibility_stays_pinned_to_prepare_commit_after_head_moves(
    tmp_path: Path, monkeypatch
) -> None:
    relative = "training.py"
    source = tmp_path / relative
    parent_bytes = b"source bytes captured by the prepared attempt\n"
    source.write_bytes(parent_bytes)
    report = tmp_path / "architecture.md"
    report.write_text("reviewed source compatibility\n", encoding="utf-8")
    prepare_commit = "a" * 40
    current_head = {"sha": "b" * 40}
    blob_oid = _git_blob_oid(parent_bytes)
    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", report)
    monkeypatch.setattr(runner, "SOURCE_COMPATIBILITY", {})

    def run(args, **kwargs):
        commands.append(args)
        if args[1] == "rev-parse" and args[2] == "--verify":
            ref = args[3]
            if ref == f"{prepare_commit}^{{commit}}":
                return SimpleNamespace(returncode=0, stdout=f"{prepare_commit}\n".encode())
            if ref == "HEAD^{commit}":
                return SimpleNamespace(returncode=0, stdout=f"{current_head['sha']}\n".encode())
            pytest.fail(f"unexpected commit ref: {ref}")
        if args[1] == "rev-parse":
            revision, path = args[2].split(":", maxsplit=1)
            assert path == relative
            assert revision in {prepare_commit, current_head["sha"]}
            return SimpleNamespace(returncode=0, stdout=f"{blob_oid}\n".encode())
        if args[1] == "show":
            revision, path = args[2].split(":", maxsplit=1)
            assert path == relative
            assert revision in {prepare_commit, current_head["sha"]}
            return SimpleNamespace(returncode=0, stdout=parent_bytes)
        assert args[1] == "hash-object"
        return SimpleNamespace(returncode=0, stdout=f"{blob_oid}\n".encode())

    monkeypatch.setattr(runner.subprocess, "run", run)
    parent_hashes = {str(source): _sha(parent_bytes)}
    first = runner.validate_source_compatibility(
        "full_turn_1m", parent_hashes, exact_revision=prepare_commit
    )
    current_head["sha"] = "c" * 40
    second = runner.validate_source_compatibility(
        "full_turn_1m", parent_hashes, exact_revision=prepare_commit
    )

    assert first["files"] == second["files"]
    assert first["files"][0]["git_revision"] == prepare_commit
    assert first["files"][0]["git_blob_oid"] == blob_oid
    assert first["files"][0]["current_sha256"] == _sha(parent_bytes)
    assert all(
        command[3] == f"{prepare_commit}^{{commit}}"
        for command in commands
        if command[1] == "rev-parse" and command[2] == "--verify"
    )

    source.write_bytes(b"unreviewed source byte drift\n")
    with pytest.raises(runner.Refuse, match="unreviewed source drift"):
        runner.validate_source_compatibility(
            "full_turn_1m", parent_hashes, exact_revision=prepare_commit
        )


def test_untracked_data_source_is_bound_to_parent_lineage_without_git_blob_case_a(
    tmp_path: Path, monkeypatch
) -> None:
    relative = "data/checkpoints/dialogue_context_final.pt"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source_bytes = b"parent-bound external training artifact\n"
    source.write_bytes(source_bytes)
    report = tmp_path / "architecture.md"
    report.write_text("reviewed source compatibility\n", encoding="utf-8")
    commit = "e" * 40
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", report)
    monkeypatch.setattr(runner, "SOURCE_COMPATIBILITY", {})

    def run(args, **kwargs):
        if args[1] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=f"{commit}\n".encode())
        return SimpleNamespace(returncode=128, stdout=b"", stderr=b"path not in tree")

    monkeypatch.setattr(runner.subprocess, "run", run)
    parent_identity = {
        "parent_attempt_sha256": "a" * 64,
        "parent_receipt_sha256": "b" * 64,
        "parent_bundle_sha256": "c" * 64,
    }

    result = runner.validate_source_compatibility(
        "warm_start",
        {str(source): _sha(source_bytes)},
        parent_identity,
    )

    row = result["files"][0]
    assert row["compatibility"] == "parent_bound_external_artifact"
    assert row["git_provenance_applicable"] is False
    assert row["parent_identity"] == parent_identity
    assert result["parent_identity"] == parent_identity


def test_untracked_data_source_is_bound_to_parent_lineage_without_git_blob_case_b(
    tmp_path: Path, monkeypatch
) -> None:
    relative = "data/checkpoints/dialogue_context_final.pt"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source_bytes = b"parent-bound external training artifact\n"
    source.write_bytes(source_bytes)
    report = tmp_path / "architecture.md"
    report.write_text("reviewed source compatibility\n", encoding="utf-8")
    commit = "e" * 40
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", report)
    monkeypatch.setattr(runner, "SOURCE_COMPATIBILITY", {})

    def run(args, **kwargs):
        if args[1] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=f"{commit}\n".encode())
        return SimpleNamespace(returncode=128, stdout=b"", stderr=b"path not in tree")

    monkeypatch.setattr(runner.subprocess, "run", run)
    parent_identity = {
        "parent_attempt_sha256": "a" * 64,
        "parent_receipt_sha256": "b" * 64,
        "parent_bundle_sha256": "c" * 64,
    }

    result = runner.validate_source_compatibility(
        "warm_start",
        {str(source): _sha(source_bytes)},
        parent_identity,
    )

    row = result["files"][0]
    assert row["compatibility"] == "parent_bound_external_artifact"
    assert row["git_provenance_applicable"] is False
    assert row["parent_identity"] == parent_identity
    assert result["parent_identity"] == parent_identity


def test_continuation_attempt_binds_runner_backend_tests_and_report_sha(
    tmp_path: Path, monkeypatch
) -> None:
    source_paths = {
        name: tmp_path / f"{name}.py"
        for name in (
            "runner",
            "budget_helper",
            "runner_contract_test",
            "budget_contract_test",
        )
    }
    for name, path in source_paths.items():
        path.write_text(f"{name} source\n", encoding="utf-8")
    report = tmp_path / "architecture.md"
    report.write_text("architecture source\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", report)
    monkeypatch.setattr(runner, "_continuation_source_files", lambda: source_paths)

    binding = runner._continuation_source_binding(_sha(report.read_bytes()))

    assert set(binding["files"]) == set(source_paths)
    for name, path in source_paths.items():
        assert binding["files"][name] == {
            "path": str(path.resolve()),
            "sha256": _sha(path.read_bytes()),
        }
    attempt = {"continuation_sources": binding}
    runner._require_continuation_source_bindings(attempt, _sha(report.read_bytes()))
    source_paths["budget_contract_test"].write_text("drift\n", encoding="utf-8")
    with pytest.raises(runner.Refuse, match="continuation source binding changed"):
        runner._require_continuation_source_bindings(attempt, _sha(report.read_bytes()))


def test_ledger_start_settlement_rebuilds_missing_progress_exactly_once(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    progress = tmp_path / "extension_progress.jsonl"
    progress.write_bytes(b"")
    token = "d" * 32
    runner.append_durable_jsonl(
        progress,
        {
            "event": "continuation_start_intent",
            "arm": "warm_start",
            "reservation_id": token,
            "requested_event": "continuation_started",
            "parent_attempt_sha256": "a" * 64,
            "parent_receipt_sha256": "b" * 64,
        },
    )
    settlement = {
        "arm": "warm_start",
        "event": "continuation_started",
        "sequence": 14,
        "event_sha256": "c" * 64,
        "reservation_id": token,
        "charged_cpu_seconds": 650.0,
        "created_utc": "2026-10-22T00:00:00Z",
        "details": {
            "reservation_id": token,
            "accounting_snapshot": {"charged_cpu_seconds": 650.0},
        },
    }
    monkeypatch.setattr(runner, "latest_reservations", lambda: {})
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda arm: [settlement])

    first = runner._reconcile_start_progress("warm_start", progress)
    second = runner._reconcile_start_progress("warm_start", progress)

    settled_rows = [item for item in second if item.get("ledger_event_sha256") == "c" * 64]
    assert len(settled_rows) == 1
    assert settled_rows[0]["charged_cpu_seconds"] == 650.0
    assert len(second) == len(first)


def test_continuation_claims_writer_lock_before_durable_start(tmp_path: Path, monkeypatch) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    arm = "warm_start"
    progress = tmp_path / "extension_progress.jsonl"
    progress.write_bytes(b"")
    attempt_path = tmp_path / "extension_attempt.json"
    attempt_path.write_text("{}", encoding="utf-8")
    context = {
        "arm": arm,
        "locations": {
            "extension_attempt": attempt_path,
            "extension_progress": progress,
            "extension_result": tmp_path / "extension_result.json",
            "extension_checkpoint_root": tmp_path / "extension_root",
        },
        "parent_attempt_sha256": "a" * 64,
        "parent_receipt_sha256": "b" * 64,
        "source_compatibility": {},
        "parent_bindings": {"source_hashes": {"synthetic": "f" * 64}},
    }
    attempt = {"continuation_sources": {"runner_sha256": runner.sha256_file(Path(runner.__file__))}}
    monkeypatch.setattr(runner, "_effective_attempt", lambda attempt, _sha: attempt)
    monkeypatch.setattr(
        runner,
        "_validate_prepared",
        lambda **_kwargs: {"parents": {arm: context}, "source_report_sha256": "a" * 64},
    )

    def read_json(path):
        path = Path(path)
        if path == attempt_path:
            return attempt
        if path.name == "latest.json":
            return {"receipt": "latest-receipt.json"}
        return {"digest": "d" * 64}

    monkeypatch.setattr(runner, "read_json", read_json)
    monkeypatch.setattr(runner, "_start_charge", lambda _context: 500.0)
    monkeypatch.setattr(
        runner,
        "_latest_extension_envelope",
        lambda _context: (
            "pretraining",
            {"trainer": {"episode_count": 12}, "cursor": {"epoch": 0, "index": 12}},
            tmp_path / "synthetic-parent.pt",
        ),
    )
    monkeypatch.setattr(
        runner.source_validation,
        "validate_training_sources",
        lambda _proof: SimpleNamespace(pretrain_corpus=object(), assistant_corpus=object()),
    )
    monkeypatch.setattr(runner.frozen, "validate_probe_face", lambda *args: None)
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda _arm: (5.0, {}))
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    active: dict[str, str | None] = {"token": None}
    settlements: list[dict] = []

    def guard(_arm, _next_charge, reserve_cpu_seconds=0.0):
        assert reserve_cpu_seconds > 0
        token = f"{len(settlements) + 1:032x}"
        active["token"] = token
        return token

    def append(arm_name, event, charged, details):
        assert details["reservation_id"] == active["token"]
        record = {
            "arm": arm_name,
            "event": event,
            "charged_cpu_seconds": charged,
            "reservation_id": details["reservation_id"],
            "details": details,
            "sequence": len(settlements),
            "event_sha256": f"{len(settlements) + 1:064x}",
        }
        settlements.append(record)
        active["token"] = None
        return record

    monkeypatch.setattr(runner, "ledger_guard", guard)
    monkeypatch.setattr(runner, "append_charge", append)
    monkeypatch.setattr(runner, "latest_reservations", lambda: {})
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda _arm: [])

    def resource_snapshot():
        events = runner.read_jsonl(progress)
        names = [event["event"] for event in events]
        assert names.index("continuation_start_intent") < names.index("continuation_started")
        assert any(event["event"] == "continuation_started" for event in settlements)
        return {"synthetic": True}

    monkeypatch.setattr(runner, "resource_snapshot", resource_snapshot)

    class LockHeldUntilFailure:
        def __enter__(self):
            events = runner.read_jsonl(progress)
            assert not any(
                event["event"]
                in {"continuation_start_intent", "continuation_started", "continuation_resumed"}
                for event in events
            )
            assert not settlements
            return self

        @staticmethod
        def __exit__(*args):
            return False

    def stop_after_lock(*_args):
        raise RuntimeError("synthetic restore stop after writer claim")

    monkeypatch.setattr(
        runner, "_open_extension_store", lambda *args, **kwargs: LockHeldUntilFailure()
    )
    monkeypatch.setattr(runner, "_fresh_process_restore", stop_after_lock)
    with pytest.raises(RuntimeError, match="synthetic restore stop after writer claim"):
        runner.execute_arm(arm, resume=False)

    events = runner.read_jsonl(progress)
    names = [event["event"] for event in events]
    assert names.index("continuation_start_intent") < names.index("continuation_started")
    assert "training_resource_preflight_passed" in names


def test_live_writer_rejects_second_resume_without_settling_owner_token(
    tmp_path: Path, monkeypatch
) -> None:
    arm = "warm_start"
    progress = tmp_path / "extension_progress.jsonl"
    token = "a" * 32
    progress.write_bytes(
        _json_bytes(
            {
                "event": "continuation_start_intent",
                "arm": arm,
                "reservation_id": token,
                "requested_event": "continuation_resumed",
                "parent_attempt_sha256": "b" * 64,
                "parent_receipt_sha256": "c" * 64,
            }
        )
    )
    before_progress = progress.read_bytes()
    attempt_path = tmp_path / "extension_attempt.json"
    attempt = {"continuation_sources": {"runner_sha256": runner.sha256_file(Path(runner.__file__))}}
    attempt_path.write_bytes(_json_bytes(attempt))
    checkpoint_root = tmp_path / "extension" / "extension_root"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "writer.lock").write_bytes(
        _json_bytes(
            {
                "pid": runner.os.getpid(),
                "create_time": runner.psutil.Process().create_time(),
                "token": "writer-token",
            }
        )
    )
    context = {
        "arm": arm,
        "locations": {
            "extension_attempt": attempt_path,
            "extension_progress": progress,
            "extension_result": tmp_path / "extension_result.json",
            "extension_checkpoint_root": checkpoint_root,
        },
        "parent_attempt_sha256": "b" * 64,
        "parent_receipt_sha256": "c" * 64,
        "source_compatibility": {},
        "parent_bindings": {"source_hashes": {"synthetic": "d" * 64}},
    }
    monkeypatch.setattr(runner, "_effective_attempt", lambda attempt, _sha: attempt)
    monkeypatch.setattr(
        runner,
        "_validate_prepared",
        lambda **_kwargs: {"parents": {arm: context}, "source_report_sha256": "a" * 64},
    )
    from taiji import LanguageEpisode, LanguageEpisodeCorpus

    corpus = LanguageEpisodeCorpus(
        (
            LanguageEpisode(
                episode_id="writer-lock-episode",
                family_id="writer-lock-family",
                task_family="fact",
                split="train",
                context="",
                user_input="question",
                response="answer",
            ),
        ),
        require_eval_splits=False,
    )
    monkeypatch.setattr(
        runner.source_validation,
        "validate_training_sources",
        lambda _proof: SimpleNamespace(pretrain_corpus=corpus, assistant_corpus=object()),
    )
    monkeypatch.setattr(runner.frozen, "validate_probe_face", lambda *_args: None)
    reservation_reads: list[str] = []
    monkeypatch.setattr(
        runner,
        "latest_reservations",
        lambda: (
            reservation_reads.append("read")
            or {arm: {"reservation_id": token, "reserved_cpu_seconds": 20.0}}
        ),
    )
    writes: list[tuple] = []
    monkeypatch.setattr(
        runner, "ledger_guard", lambda *args, **kwargs: writes.append((args, kwargs))
    )
    monkeypatch.setattr(
        runner, "append_charge", lambda *args, **kwargs: writes.append((args, kwargs))
    )
    monkeypatch.setattr(
        runner,
        "_reconcile_start_progress",
        lambda *_args: pytest.fail("progress reconciliation ran before writer lock"),
    )
    monkeypatch.setattr(
        runner,
        "_start_charge",
        lambda *_args: pytest.fail("budget reconciliation ran before writer lock"),
    )

    with pytest.raises(runner.Refuse, match="writer owner is still alive"):
        runner.execute_arm(arm, resume=True)

    assert progress.read_bytes() == before_progress
    assert reservation_reads == []
    assert writes == []

    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda _arm: (5.0, {}))
    owner_settlements: list[dict] = []

    def settle_owner(owner_arm, event, charged, details):
        assert owner_arm == arm
        assert details["reservation_id"] == token
        owner_settlements.append({"event": event, "reservation_id": details["reservation_id"]})
        return {"event": event, "reservation_id": token}

    monkeypatch.setattr(runner, "append_charge", settle_owner)
    owner_budget = runner.ExtensionBudget(arm, 500.0)
    owner_budget.active_reservation_id = token
    owner_budget.active_reservation_upper = 520.0
    owner_budget.settle_reservation("episode_completed")
    assert owner_settlements == [{"event": "episode_completed", "reservation_id": token}]


@pytest.mark.parametrize("writer_lock_exists", [True, False])
def test_empty_resume_recovers_dead_reclaim_guard_after_interruption(
    tmp_path: Path, monkeypatch, writer_lock_exists: bool
) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    arm = "warm_start"
    progress = tmp_path / "extension_progress.jsonl"
    progress.write_bytes(b"")
    attempt_path = tmp_path / "extension_attempt.json"
    attempt = {"continuation_sources": {"runner_sha256": runner.sha256_file(Path(runner.__file__))}}
    attempt_path.write_bytes(_json_bytes(attempt))
    checkpoint_root = tmp_path / "extension" / "extension_root"
    (checkpoint_root / "pretraining").mkdir(parents=True)
    (checkpoint_root / "pretraining" / "latest.json").write_bytes(
        _json_bytes({"receipt": "latest-receipt.json"})
    )
    (checkpoint_root / "pretraining" / "latest-receipt.json").write_bytes(
        _json_bytes({"digest": "e" * 64})
    )
    dead_identity = {"pid": 2_147_483_647, "create_time": 1.0, "token": "d" * 32}
    if writer_lock_exists:
        (checkpoint_root / "writer.lock").write_bytes(_json_bytes(dead_identity))
    (checkpoint_root / "reclaim.guard").write_bytes(_json_bytes(dead_identity))
    context = {
        "arm": arm,
        "locations": {
            "extension_attempt": attempt_path,
            "extension_progress": progress,
            "extension_result": tmp_path / "extension_result.json",
            "extension_checkpoint_root": checkpoint_root,
        },
        "parent_attempt_sha256": "f" * 64,
        "parent_receipt_sha256": "1" * 64,
        "source_compatibility": {},
        "parent_bindings": {"source_hashes": {"synthetic": "2" * 64}},
    }
    monkeypatch.setattr(runner, "_effective_attempt", lambda attempt, _sha: attempt)
    monkeypatch.setattr(
        runner,
        "_validate_prepared",
        lambda **_kwargs: {"parents": {arm: context}, "source_report_sha256": "a" * 64},
    )
    from taiji import LanguageEpisode, LanguageEpisodeCorpus

    corpus = LanguageEpisodeCorpus(
        (
            LanguageEpisode(
                episode_id="resume-after-stale-lock",
                family_id="resume-after-stale-lock",
                task_family="fact",
                split="train",
                context="",
                user_input="question",
                response="answer",
            ),
        ),
        require_eval_splits=False,
    )
    monkeypatch.setattr(
        runner.source_validation,
        "validate_training_sources",
        lambda _proof: SimpleNamespace(pretrain_corpus=corpus, assistant_corpus=object()),
    )
    monkeypatch.setattr(runner.frozen, "validate_probe_face", lambda *_args: None)
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda _arm: (5.0, {}))
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    monkeypatch.setattr(runner, "_start_charge", lambda _context: 500.0)
    monkeypatch.setattr(
        runner,
        "_latest_extension_envelope",
        lambda _context: (
            "pretraining",
            {"trainer": {"episode_count": 12}, "cursor": {"epoch": 0, "index": 12}},
            tmp_path / "synthetic-parent.pt",
        ),
    )
    active: dict[str, str | None] = {"token": None}
    settlements: list[dict] = []

    def guard(_arm, _next_charge, reserve_cpu_seconds=0.0):
        assert reserve_cpu_seconds > 0
        token = f"{len(settlements) + 1:032x}"
        active["token"] = token
        return token

    def append(arm_name, event, charged, details):
        assert details["reservation_id"] == active["token"]
        record = {
            "arm": arm_name,
            "event": event,
            "charged_cpu_seconds": charged,
            "reservation_id": details["reservation_id"],
            "details": details,
            "sequence": len(settlements),
            "event_sha256": f"{len(settlements) + 1:064x}",
        }
        settlements.append(record)
        active["token"] = None
        return record

    monkeypatch.setattr(runner, "ledger_guard", guard)
    monkeypatch.setattr(runner, "append_charge", append)
    monkeypatch.setattr(runner, "latest_reservations", lambda: {})
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda _arm: [])
    monkeypatch.setattr(runner, "resource_snapshot", lambda: {"synthetic": True})

    def stop_after_stale_recovery(*_args):
        raise RuntimeError("synthetic restore stop after stale lock recovery")

    monkeypatch.setattr(runner, "_fresh_process_restore", stop_after_stale_recovery)
    with pytest.raises(RuntimeError, match="synthetic restore stop after stale lock recovery"):
        runner.execute_arm(arm, resume=True)

    events = runner.read_jsonl(progress)
    assert sum(event["event"] == "continuation_start_intent" for event in events) == 1
    assert sum(event["event"] == "continuation_resumed" for event in events) == 1
    assert any(event["event"] == "continuation_stopped" for event in events)
    assert not (checkpoint_root / "writer.lock").exists()
    archived = list((checkpoint_root / "stale-locks").glob("*.json"))
    assert any(path.name.startswith("reclaim-guard-") for path in archived)
    assert (len(archived) >= 2) is writer_lock_exists


def test_empty_resume_without_reclaimed_writer_lock_stays_refused(
    tmp_path: Path, monkeypatch
) -> None:
    arm = "warm_start"
    progress = tmp_path / "extension_progress.jsonl"
    progress.write_bytes(b"")
    attempt_path = tmp_path / "extension_attempt.json"
    attempt_path.write_bytes(
        _json_bytes(
            {"continuation_sources": {"runner_sha256": runner.sha256_file(Path(runner.__file__))}}
        )
    )
    context = {
        "arm": arm,
        "locations": {
            "extension_attempt": attempt_path,
            "extension_progress": progress,
            "extension_result": tmp_path / "extension_result.json",
            "extension_checkpoint_root": tmp_path / "extension_root",
        },
        "parent_attempt_sha256": "a" * 64,
        "parent_receipt_sha256": "b" * 64,
        "source_compatibility": {},
        "parent_bindings": {"source_hashes": {"synthetic": "c" * 64}},
    }
    monkeypatch.setattr(runner, "_effective_attempt", lambda attempt, _sha: attempt)
    monkeypatch.setattr(
        runner,
        "_validate_prepared",
        lambda **_kwargs: {"parents": {arm: context}, "source_report_sha256": "a" * 64},
    )
    from taiji import LanguageEpisode, LanguageEpisodeCorpus

    corpus = LanguageEpisodeCorpus(
        (
            LanguageEpisode(
                episode_id="empty-resume",
                family_id="empty-resume",
                task_family="fact",
                split="train",
                context="",
                user_input="question",
                response="answer",
            ),
        ),
        require_eval_splits=False,
    )
    monkeypatch.setattr(
        runner.source_validation,
        "validate_training_sources",
        lambda _proof: SimpleNamespace(pretrain_corpus=corpus, assistant_corpus=object()),
    )
    monkeypatch.setattr(runner.frozen, "validate_probe_face", lambda *_args: None)

    class NewWriter:
        reclaim_token = None

        def __enter__(self):
            return self

        @staticmethod
        def __exit__(*_args):
            return False

    monkeypatch.setattr(runner, "_open_extension_store", lambda *_args, **_kwargs: NewWriter())
    monkeypatch.setattr(runner, "latest_reservations", lambda: {})
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda _arm: [])
    monkeypatch.setattr(
        runner,
        "_start_charge",
        lambda *_args: pytest.fail("empty resume consumed budget without a stale lock"),
    )
    before_progress = progress.read_bytes()
    with pytest.raises(runner.Refuse, match="requires reclaiming a dead writer lock"):
        runner.execute_arm(arm, resume=True)
    assert progress.read_bytes() == before_progress


def test_reclaim_guard_publish_failure_keeps_dead_writer_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    checkpoint_root = tmp_path / "extension" / "guard-publish-retry"
    checkpoint_root.mkdir(parents=True)
    lock_path = checkpoint_root / "writer.lock"
    lock_raw = _json_bytes({"pid": 2_147_483_647, "create_time": 1.0, "token": "e" * 32})
    lock_path.write_bytes(lock_raw)
    real_publish_json = runner._extension_publish_json

    def fail_guard_publish(path, payload, *, replace=False):
        if Path(path).name == "reclaim.guard":
            raise OSError("injected atomic guard publication interruption")
        return real_publish_json(path, payload, replace=replace)

    monkeypatch.setattr(runner, "_extension_publish_json", fail_guard_publish)
    corpus = SimpleNamespace(episodes=[SimpleNamespace(split="train")])
    bindings = {"arm": "warm_start", "stage": "runner"}

    with (
        pytest.raises(OSError, match="atomic guard publication interruption"),
        runner._open_extension_store(checkpoint_root, bindings, corpus, resume=True),
    ):
        pytest.fail("writer should not open before guard publication")
    assert lock_path.read_bytes() == lock_raw
    assert not (checkpoint_root / "reclaim.guard").exists()
    assert not (checkpoint_root / "stale-locks" / f"writer-lock-{'e' * 32}.json").exists()

    monkeypatch.setattr(runner, "_extension_publish_json", real_publish_json)
    with runner._open_extension_store(checkpoint_root, bindings, corpus, resume=True) as store:
        assert store.reclaimed_dead_writer is True
        assert store.reclaim_token
        assert not (checkpoint_root / "reclaim.guard").exists()
        assert (checkpoint_root / "writer.lock").is_file()
    assert not lock_path.exists()
    assert (
        checkpoint_root / "stale-locks" / f"writer-lock-{'e' * 32}.json"
    ).read_bytes() == lock_raw


def test_writer_lock_publish_failure_is_retryable(tmp_path: Path, monkeypatch) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    checkpoint_root = tmp_path / "extension" / "writer-lock-publish-retry"
    checkpoint_root.mkdir(parents=True)
    corpus = SimpleNamespace(episodes=[SimpleNamespace(split="train")])
    bindings = {"arm": "warm_start", "stage": "runner"}
    real_publish_json = runner._extension_publish_json

    def fail_writer_publish(path, payload, *, replace=False):
        if Path(path).name == "writer.lock":
            raise OSError("injected atomic writer publication interruption")
        return real_publish_json(path, payload, replace=replace)

    monkeypatch.setattr(runner, "_extension_publish_json", fail_writer_publish)
    with (
        pytest.raises(OSError, match="atomic writer publication interruption"),
        runner._open_extension_store(checkpoint_root, bindings, corpus),
    ):
        pytest.fail("writer should not enter before durable lock publication")
    assert not (checkpoint_root / "writer.lock").exists()
    assert not (checkpoint_root / "reclaim.guard").exists()

    monkeypatch.setattr(runner, "_extension_publish_json", real_publish_json)
    with runner._open_extension_store(checkpoint_root, bindings, corpus) as store:
        assert store.locked is True
        assert store.reclaimed_dead_writer is False
        lock = runner.read_json(checkpoint_root / "writer.lock")
        assert lock["pid"] == runner.os.getpid()
        assert lock["token"] == store.token
    assert not (checkpoint_root / "writer.lock").exists()


def test_dead_writer_archive_survives_new_lock_publish_failure(tmp_path: Path, monkeypatch) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    checkpoint_root = tmp_path / "extension" / "reclaim-writer-publish-retry"
    checkpoint_root.mkdir(parents=True)
    token = "a" * 32
    lock_raw = _json_bytes({"pid": 2_147_483_647, "create_time": 1.0, "token": token})
    (checkpoint_root / "writer.lock").write_bytes(lock_raw)
    real_publish_json = runner._extension_publish_json

    def fail_writer_publish(path, payload, *, replace=False):
        if Path(path).name == "writer.lock":
            raise OSError("injected reclaimed writer publication interruption")
        return real_publish_json(path, payload, replace=replace)

    corpus = SimpleNamespace(episodes=[SimpleNamespace(split="train")])
    bindings = {"arm": "warm_start", "stage": "runner"}
    monkeypatch.setattr(runner, "_extension_publish_json", fail_writer_publish)
    with (
        pytest.raises(OSError, match="reclaimed writer publication interruption"),
        runner._open_extension_store(checkpoint_root, bindings, corpus, resume=True),
    ):
        pytest.fail("writer must not enter before durable publication")
    archive = checkpoint_root / "stale-locks" / f"writer-lock-{token}.json"
    assert archive.read_bytes() == lock_raw
    assert not (checkpoint_root / "writer.lock").exists()
    assert not (checkpoint_root / "reclaim.guard").exists()
    monkeypatch.setattr(runner, "_extension_publish_json", real_publish_json)
    with runner._open_extension_store(checkpoint_root, bindings, corpus, resume=True) as store:
        assert store.reclaimed_dead_writer is True
        assert store.locked is True
    assert archive.read_bytes() == lock_raw


def test_dead_writer_archive_replay_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    checkpoint_root = tmp_path / "extension" / "writer-archive-replay"
    checkpoint_root.mkdir(parents=True)
    token = "f" * 32
    lock_raw = _json_bytes({"pid": 2_147_483_647, "create_time": 1.0, "token": token})
    (checkpoint_root / "writer.lock").write_bytes(lock_raw)
    archive_directory = checkpoint_root / "stale-locks"
    archive_directory.mkdir()
    archive_path = archive_directory / f"writer-lock-{token}.json"
    archive_path.write_bytes(lock_raw)
    corpus = SimpleNamespace(episodes=[SimpleNamespace(split="train")])
    bindings = {"arm": "warm_start", "stage": "runner"}

    with runner._open_extension_store(checkpoint_root, bindings, corpus, resume=True) as store:
        assert store.reclaimed_dead_writer is True
        assert runner.read_json(checkpoint_root / "writer.lock")["token"] == store.token

    assert archive_path.read_bytes() == lock_raw
    assert not (checkpoint_root / "writer.lock").exists()


def test_resume_after_dead_guard_archive_and_unlink_window(tmp_path: Path, monkeypatch) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    checkpoint_root = tmp_path / "extension" / "guard-archive-replay"
    archive_directory = checkpoint_root / "stale-locks"
    archive_directory.mkdir(parents=True)
    identity = {"pid": 2_147_483_647, "create_time": 1.0, "token": "1" * 32}
    raw = _json_bytes(identity)
    archive_path = archive_directory / f"reclaim-guard-{identity['token']}.json"
    archive_path.write_bytes(
        _json_bytes(
            {
                "schema": "taiji-r2-extension-dead-reclaim-guard-v1",
                "guard": identity,
                "recovered_utc": runner.utc_now(),
                "guard_sha256": _sha(raw),
            }
        )
    )
    corpus = SimpleNamespace(episodes=[SimpleNamespace(split="train")])
    bindings = {"arm": "warm_start", "stage": "runner"}

    with runner._open_extension_store(checkpoint_root, bindings, corpus, resume=True) as store:
        assert store.reclaimed_dead_writer is True
        assert runner.read_json(checkpoint_root / "writer.lock")["token"] == store.token
        assert not (checkpoint_root / "reclaim.guard").exists()

    assert archive_path.is_file()
    assert not (checkpoint_root / "writer.lock").exists()


def test_reclaim_guard_recovery_lock_is_exclusive_and_released_after_process_death(
    tmp_path: Path,
) -> None:
    ready_path = tmp_path / "recovery-lock-ready"
    script = """
from pathlib import Path
import sys
import time
from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as runner
with runner._reclaim_guard_recovery_lock(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text("ready", encoding="utf-8")
    time.sleep(60)
"""
    process = subprocess.Popen(
        [runner.sys.executable, "-c", script, str(tmp_path), str(ready_path)],
        cwd=runner.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = runner.time.monotonic() + 20.0
        while (
            not ready_path.exists()
            and process.poll() is None
            and runner.time.monotonic() < deadline
        ):
            runner.time.sleep(0.01)
        assert ready_path.is_file(), "child did not acquire the OS recovery lock"
        with (
            pytest.raises(runner.Refuse, match="reclaim guard recovery is already in progress"),
            runner._reclaim_guard_recovery_lock(tmp_path),
        ):
            pytest.fail("second process acquired the active recovery lock")
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10.0)

    with runner._reclaim_guard_recovery_lock(tmp_path):
        pass


@pytest.mark.parametrize(
    "start_charge", [runner.PER_ARM_CPU_LIMIT, runner.PER_ARM_CPU_LIMIT + 100.0]
)
def test_start_ceiling_refusal_is_durably_recorded_before_training(
    tmp_path: Path, monkeypatch, start_charge: float
) -> None:
    _configure_synthetic_publisher(monkeypatch, tmp_path)
    arm = "warm_start"
    progress = tmp_path / "extension_progress.jsonl"
    progress.write_bytes(b"")
    attempt_path = tmp_path / "extension_attempt.json"
    attempt_path.write_bytes(
        _json_bytes(
            {"continuation_sources": {"runner_sha256": runner.sha256_file(Path(runner.__file__))}}
        )
    )
    context = {
        "arm": arm,
        "locations": {
            "extension_attempt": attempt_path,
            "extension_progress": progress,
            "extension_result": tmp_path / "extension_result.json",
            "extension_checkpoint_root": tmp_path / "extension" / "root",
        },
        "parent_attempt_sha256": "a" * 64,
        "parent_receipt_sha256": "b" * 64,
        "source_compatibility": {},
        "parent_bindings": {"source_hashes": {"synthetic": "c" * 64}},
    }
    monkeypatch.setattr(runner, "_effective_attempt", lambda attempt, _sha: attempt)
    monkeypatch.setattr(
        runner,
        "_validate_prepared",
        lambda **_kwargs: {"parents": {arm: context}, "source_report_sha256": "a" * 64},
    )
    from taiji import LanguageEpisode, LanguageEpisodeCorpus

    corpus = LanguageEpisodeCorpus(
        (
            LanguageEpisode(
                episode_id="budget-refusal",
                family_id="budget-refusal",
                task_family="fact",
                split="train",
                context="",
                user_input="question",
                response="answer",
            ),
        ),
        require_eval_splits=False,
    )
    monkeypatch.setattr(
        runner.source_validation,
        "validate_training_sources",
        lambda _proof: SimpleNamespace(pretrain_corpus=corpus, assistant_corpus=object()),
    )
    monkeypatch.setattr(runner.frozen, "validate_probe_face", lambda *_args: None)
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda _arm: (5.0, {}))
    monkeypatch.setattr(runner, "_reconcile_start_progress", lambda *_args: [])
    monkeypatch.setattr(runner, "latest_reservations", lambda: {})
    monkeypatch.setattr(runner, "_start_charge", lambda _context: start_charge)
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda _arm: [])
    monkeypatch.setattr(
        runner,
        "ledger_guard",
        lambda *_args, **_kwargs: pytest.fail("ceiling refusal must not reserve beyond cap"),
    )
    monkeypatch.setattr(
        runner,
        "append_charge",
        lambda *_args, **_kwargs: pytest.fail("ceiling refusal has no ledger reservation"),
    )

    class WriterClaimed:
        def __enter__(self):
            return SimpleNamespace(reclaimed_dead_writer=False, reclaim_token=None)

        @staticmethod
        def __exit__(*args):
            return False

    monkeypatch.setattr(runner, "_open_extension_store", lambda *_args, **_kwargs: WriterClaimed())

    with pytest.raises(runner.Refuse, match="cumulative CPU ceiling exhausted before execution"):
        runner.execute_arm(arm, resume=False)

    events = runner.read_jsonl(progress)
    assert [event["event"] for event in events] == ["continuation_stopped"]
    assert events[0]["stop_reason"] == "cumulative_cpu_ceiling_exhausted"
    assert events[0]["cumulative_cpu_ceiling_seconds"] == runner.PER_ARM_CPU_LIMIT
    assert events[0]["cumulative_cpu_at_start_seconds"] >= start_charge


def test_progress_with_partial_final_record_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    path.write_bytes(b'{"event":"continuation_started"}')
    with pytest.raises(runner.Refuse, match="incomplete final record"):
        runner.read_jsonl(path)


def test_parent_checkpoint_fork_is_byte_exact_and_leaves_parent_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    parent_stage = tmp_path / "parent" / "pretraining"
    parent_stage.mkdir(parents=True)
    bundle = b"synthetic parent checkpoint bundle\x00\x01"
    receipt = {"bundle": "parent.pt", "sha256": _sha(bundle), "digest": "d" * 64}
    receipt_bytes = _json_bytes(receipt)
    latest = {
        "receipt": "parent.json",
        "receipt_sha256": _sha(receipt_bytes),
    }
    latest_bytes = _json_bytes(latest)
    (parent_stage / "parent.pt").write_bytes(bundle)
    (parent_stage / "parent.json").write_bytes(receipt_bytes)
    (parent_stage / "latest.json").write_bytes(latest_bytes)
    before = {
        name: (parent_stage / name).read_bytes()
        for name in ("parent.pt", "parent.json", "latest.json")
    }
    extension_root = tmp_path / "extension" / "full_turn_1m"
    volume_root = "SYNTHETIC:\\"
    monkeypatch.setattr(runner, "_APPROVED_VOLUME_ROOT", volume_root)
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", tmp_path / "extension")
    monkeypatch.setattr(runner, "durable_publish", _fake_write_through_publish)
    context = {
        "arm": "full_turn_1m",
        "locations": {
            "parent_checkpoint_root": parent_stage.parent,
            "extension_checkpoint_root": extension_root,
        },
        "parent_latest": latest,
        "parent_receipt": receipt,
    }

    runner._copy_parent_checkpoint(context)

    extension_stage = extension_root / "pretraining"
    assert (extension_stage / "parent.pt").read_bytes() == bundle
    assert (extension_stage / "parent.json").read_bytes() == receipt_bytes
    assert (extension_stage / "latest.json").read_bytes() == latest_bytes
    assert {
        name: (parent_stage / name).read_bytes()
        for name in ("parent.pt", "parent.json", "latest.json")
    } == before


def test_parent_checkpoint_fork_repairs_only_missing_bytes_and_rejects_drift_case_a(
    tmp_path: Path, monkeypatch
) -> None:
    parent_stage = tmp_path / "parent" / "pretraining"
    parent_stage.mkdir(parents=True)
    bundle = b"synthetic parent checkpoint bundle"
    receipt = {"bundle": "parent.pt", "sha256": _sha(bundle)}
    receipt_bytes = _json_bytes(receipt)
    latest = {"receipt": "parent.json", "receipt_sha256": _sha(receipt_bytes)}
    files = {
        "latest.json": _json_bytes(latest),
        "parent.json": receipt_bytes,
        "parent.pt": bundle,
    }
    for name, raw in files.items():
        (parent_stage / name).write_bytes(raw)
    extension_root = tmp_path / "extension" / "warm_start"
    extension_stage = extension_root / "pretraining"
    extension_stage.mkdir(parents=True)
    (extension_stage / "parent.pt").write_bytes(files["parent.pt"])
    publisher_calls: list[str] = []
    monkeypatch.setattr(runner, "_APPROVED_VOLUME_ROOT", "SYNTHETIC:\\")
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", tmp_path / "extension")

    def publish(path, writer, *, replace=False, expected_volume_root=None):
        publisher_calls.append(Path(path).name)
        return _fake_write_through_publish(
            path,
            writer,
            replace=replace,
            expected_volume_root=expected_volume_root,
        )

    monkeypatch.setattr(runner, "durable_publish", publish)
    context = {
        "arm": "warm_start",
        "locations": {
            "parent_checkpoint_root": parent_stage.parent,
            "extension_checkpoint_root": extension_root,
        },
        "parent_latest": latest,
        "parent_receipt": receipt,
    }

    runner._copy_parent_checkpoint(context)

    assert publisher_calls == ["latest.json", "parent.json"]
    assert {path.name: path.read_bytes() for path in extension_stage.iterdir()} == files
    (extension_stage / "parent.json").write_bytes(b"corrupt")
    with pytest.raises(runner.Refuse, match="differs byte-for-byte"):
        runner._copy_parent_checkpoint(context)
    assert (extension_stage / "parent.json").read_bytes() == b"corrupt"


def test_parent_checkpoint_fork_repairs_only_missing_bytes_and_rejects_drift_case_b(
    tmp_path: Path, monkeypatch
) -> None:
    parent_stage = tmp_path / "parent" / "pretraining"
    parent_stage.mkdir(parents=True)
    bundle = b"synthetic parent checkpoint bundle"
    receipt = {"bundle": "parent.pt", "sha256": _sha(bundle)}
    receipt_bytes = _json_bytes(receipt)
    latest = {"receipt": "parent.json", "receipt_sha256": _sha(receipt_bytes)}
    files = {
        "latest.json": _json_bytes(latest),
        "parent.json": receipt_bytes,
        "parent.pt": bundle,
    }
    for name, raw in files.items():
        (parent_stage / name).write_bytes(raw)
    extension_root = tmp_path / "extension" / "warm_start"
    extension_stage = extension_root / "pretraining"
    extension_stage.mkdir(parents=True)
    (extension_stage / "parent.pt").write_bytes(files["parent.pt"])
    publisher_calls: list[str] = []
    monkeypatch.setattr(runner, "_APPROVED_VOLUME_ROOT", "SYNTHETIC:\\")
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", tmp_path / "extension")

    def publish(path, writer, *, replace=False, expected_volume_root=None):
        publisher_calls.append(Path(path).name)
        return _fake_write_through_publish(
            path,
            writer,
            replace=replace,
            expected_volume_root=expected_volume_root,
        )

    monkeypatch.setattr(runner, "durable_publish", publish)
    context = {
        "arm": "warm_start",
        "locations": {
            "parent_checkpoint_root": parent_stage.parent,
            "extension_checkpoint_root": extension_root,
        },
        "parent_latest": latest,
        "parent_receipt": receipt,
    }

    runner._copy_parent_checkpoint(context)

    assert publisher_calls == ["latest.json", "parent.json"]
    assert {path.name: path.read_bytes() for path in extension_stage.iterdir()} == files
    (extension_stage / "parent.json").write_bytes(b"corrupt")
    with pytest.raises(runner.Refuse, match="differs byte-for-byte"):
        runner._copy_parent_checkpoint(context)
    assert (extension_stage / "parent.json").read_bytes() == b"corrupt"


def test_budget_guard_preserves_cumulative_charge_and_reserve(monkeypatch) -> None:
    calls: list[tuple] = []
    settlements: list[tuple] = []
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda arm: (5.0, {}))
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    monkeypatch.setattr(
        runner,
        "ledger_guard",
        lambda arm, next_charge, reserve_cpu_seconds=0.0: (
            calls.append((arm, next_charge, reserve_cpu_seconds)) or f"reservation-{len(calls)}"
        ),
    )
    monkeypatch.setattr(
        runner,
        "append_charge",
        lambda arm, event, charged, details: (
            settlements.append((arm, event, charged, details)) or {"arm": arm, "event": event}
        ),
    )
    budget = runner.ExtensionBudget("full_turn_1m", 500.0)
    budget.guard(reserve=5.0)
    assert calls and calls[-1][0] == "full_turn_1m"
    assert calls[-1][1] >= 500.0
    assert calls[-1][2] == (
        5.0 + runner.LEDGER_EVENT_RESERVE_SECONDS + runner.TERMINAL_STOP_RESERVE_SECONDS
    )
    assert budget.active_reservation_id == "reservation-1"
    budget.guard(reserve=5.0)
    assert settlements[-1][1] == "work_unit_settled"
    assert settlements[-1][3]["reservation_id"] == "reservation-1"
    assert budget.active_reservation_id == "reservation-2"

    exhausted = runner.ExtensionBudget("full_turn_1m", runner.PER_ARM_CPU_LIMIT - 1.0)
    with pytest.raises(runner.Refuse, match="allocation exhausted"):
        exhausted.guard(reserve=5.0)


def test_budget_edge_reserves_and_persists_terminal_stop_event(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda _arm: (5.0, {}))
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    calls: list[float] = []
    settlements: list[tuple] = []

    def reserve(_arm, _charge, reserve_cpu_seconds=0.0):
        calls.append(reserve_cpu_seconds)
        if reserve_cpu_seconds > runner.LEDGER_EVENT_RESERVE_SECONDS:
            raise runner.Refuse("aggregate headroom cannot cover next work unit")
        return "terminal-stop-token"

    monkeypatch.setattr(runner, "ledger_guard", reserve)
    monkeypatch.setattr(
        runner,
        "append_charge",
        lambda arm, event, charged, details: (
            settlements.append((arm, event, charged, dict(details))) or {"arm": arm, "event": event}
        ),
    )
    monkeypatch.setattr(runner, "current_accounting_snapshot", lambda charged: {"charged": charged})
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda _arm: [])
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", tmp_path / "extension")
    monkeypatch.setattr(runner, "_APPROVED_VOLUME_ROOT", "SYNTHETIC:\\")
    monkeypatch.setattr(runner, "durable_publish", _fake_write_through_publish)
    monkeypatch.setattr(runner.time, "process_time", lambda: 0.0)
    progress = tmp_path / "stop.jsonl"
    progress.write_bytes(b"")
    budget = runner.ExtensionBudget("warm_start", runner.PER_ARM_CPU_LIMIT - 15.0)

    with pytest.raises(runner.Refuse, match="allocation exhausted") as caught:
        budget.guard()
    assert budget.active_reservation_id == "terminal-stop-token"
    runner._persist_stop_event("warm_start", budget, progress, "a" * 64, caught.value)

    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["continuation_stopped"]
    assert rows[0]["stop_id"]
    assert settlements[0][1] == "continuation_stopped"
    assert calls == [runner.LEDGER_EVENT_RESERVE_SECONDS]


def test_aggregate_edge_persists_progress_refusal_when_no_ledger_reserve_fits(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda _arm: (5.0, {}))
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    calls: list[float] = []

    def exhausted(_arm, _charge, reserve_cpu_seconds=0.0):
        calls.append(reserve_cpu_seconds)
        raise ledger.ExtensionLedgerRefuse("aggregate ceiling has no remaining headroom")

    monkeypatch.setattr(runner, "ledger_guard", exhausted)
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda _arm: [])
    monkeypatch.setattr(
        runner, "append_charge", lambda *_args, **_kwargs: pytest.fail("must not overcharge")
    )
    monkeypatch.setattr(runner, "current_accounting_snapshot", lambda charged: {"charged": charged})
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", tmp_path / "extension")
    monkeypatch.setattr(runner, "_APPROVED_VOLUME_ROOT", "SYNTHETIC:\\")
    monkeypatch.setattr(runner, "durable_publish", _fake_write_through_publish)
    monkeypatch.setattr(runner.time, "process_time", lambda: 0.0)
    progress = tmp_path / "aggregate-stop.jsonl"
    progress.write_bytes(b"")
    budget = runner.ExtensionBudget("fresh_same_capacity", 500.0)

    with pytest.raises(runner.Refuse, match="aggregate continuation reservation refused") as caught:
        budget.guard()
    assert budget.pending_stop_details["stop_reason"] == "work_unit_refused"
    assert budget.pending_stop_details["refusal"].endswith(
        "ExtensionLedgerRefuse: aggregate ceiling has no remaining headroom"
    )
    runner._persist_stop_event("fresh_same_capacity", budget, progress, "b" * 64, caught.value)

    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "continuation_stopped"
    assert rows[0]["ledger_settlement_failed"]
    assert len(calls) >= 3


def test_resume_snapshot_settles_pending_reservation_token(tmp_path: Path, monkeypatch) -> None:
    settled: list[tuple] = []
    monkeypatch.setattr(
        runner,
        "append_charge",
        lambda arm, event, charged, details: (
            settled.append((arm, event, charged, dict(details))) or {"arm": arm, "event": event}
        ),
    )
    budget = runner.ExtensionBudget("warm_start", 1_000.0)
    budget.active_reservation_id = "a" * 32
    budget.active_reservation_upper = 1_050.0
    progress = tmp_path / "extension_progress.jsonl"
    progress.write_bytes(b"")
    monkeypatch.setattr(runner, "_APPROVED_VOLUME_ROOT", "SYNTHETIC:\\")
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", tmp_path / "extension")
    monkeypatch.setattr(runner, "durable_publish", _fake_write_through_publish)

    record = runner._append_cpu_snapshot(
        "warm_start",
        "continuation_resumed",
        budget,
        {"reconciled_reserved_upper": True},
        progress,
    )

    assert record["event"] == "continuation_resumed"
    assert settled[0][3]["reservation_id"] == "a" * 32
    assert settled[0][3]["reconciled_reserved_upper"] is True
    assert settled[0][2] >= 1_000.0
    assert budget.active_reservation_id is None
    assert json.loads(progress.read_text(encoding="utf-8"))["event"] == "continuation_resumed"


def test_prepare_only_creates_one_complete_synthetic_extension_without_restore_or_training(
    tmp_path: Path, monkeypatch
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    volume_root = "SYNTHETIC:\\"
    architecture = reports / "architecture.md"
    architecture.write_text("synthetic reviewed contract\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", architecture)
    monkeypatch.setattr(runner, "REPORTS_DIR", reports)
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", tmp_path / "data" / "extension")
    publisher_calls: list[tuple[str, bool, str | None]] = []
    fail_parent_receipt_once = {"value": True}

    def record_publish(path, writer, *, replace=False, expected_volume_root=None):
        publisher_calls.append((str(path), replace, expected_volume_root))
        if (
            fail_parent_receipt_once["value"]
            and "warm_start" in str(path)
            and Path(path).name == "parent.json"
        ):
            fail_parent_receipt_once["value"] = False
            raise OSError("synthetic interruption after parent latest copy")
        return _fake_write_through_publish(
            path,
            writer,
            replace=replace,
            expected_volume_root=expected_volume_root,
        )

    monkeypatch.setattr(runner, "durable_publish", record_publish)
    monkeypatch.setattr(
        runner,
        "qualify_storage_target",
        lambda path, expected_volume_root=None: {
            "volume_root": volume_root,
            "filesystem": "NTFS",
            "drive_type": ledger.DRIVE_FIXED,
        },
    )
    # _install_extension_publisher mutates process-local imports for the new
    # entry point; keep this synthetic test isolated from later tests.
    monkeypatch.setattr(runner.recovery, "_publish", runner.recovery._publish)
    monkeypatch.setattr(runner.recovery, "_publish_json", runner.recovery._publish_json)
    monkeypatch.setattr(runner.frozen, "atomic_save_final", runner.frozen.atomic_save_final)

    path_map: dict[str, dict[str, Path]] = {}
    contexts: dict[str, dict] = {}
    charges = {arm: 0.0 for arm in runner.ARMS}
    initialize_calls: list[dict] = []
    append_calls: list[tuple] = []
    active_reservations: dict[str, dict[str, Any]] = {}
    settlement_records: list[dict[str, Any]] = []

    for index, arm in enumerate(runner.ARMS):
        parent_root = tmp_path / "data" / "parent" / arm
        parent_stage = parent_root / "pretraining"
        parent_stage.mkdir(parents=True)
        source_attempt = reports / f"{arm}_parent_attempt.json"
        source_progress = reports / f"{arm}_parent_progress.jsonl"
        source_attempt.write_bytes(f"immutable-{arm}-attempt".encode())
        source_progress.write_bytes(f"immutable-{arm}-progress\n".encode())
        bundle = f"synthetic-{arm}-bundle".encode()
        receipt = {"bundle": "parent.pt", "sha256": _sha(bundle), "digest": "e" * 64}
        receipt_bytes = _json_bytes(receipt)
        latest = {"receipt": "parent.json", "receipt_sha256": _sha(receipt_bytes)}
        (parent_stage / "parent.pt").write_bytes(bundle)
        (parent_stage / "parent.json").write_bytes(receipt_bytes)
        (parent_stage / "latest.json").write_bytes(_json_bytes(latest))

        location = {
            "parent_attempt": source_attempt,
            "parent_readiness": reports / f"{arm}_readiness.json",
            "parent_progress": source_progress,
            "parent_result": reports / f"{arm}_parent_result.json",
            "parent_checkpoint_root": parent_root,
            "extension_attempt": reports / f"{arm}_extension_attempt.json",
            "extension_progress": reports / f"{arm}_extension_progress.jsonl",
            "extension_result": reports / f"{arm}_extension_result.json",
            "extension_checkpoint_root": tmp_path / "data" / "extension" / arm,
            "contract_path": reports / "global_contract.json",
            "aggregate_ledger_path": reports / "budget_event",
        }
        path_map[arm] = location
        source_compatibility = {
            "arm": arm,
            "report_path": str(architecture),
            "report_sha256": _sha(architecture.read_bytes()),
            "runner_sha256": runner.sha256_file(Path(runner.__file__)),
            "files": [],
        }
        contexts[arm] = {
            "arm": arm,
            "locations": location,
            "parent_attempt": {"arm": arm, "status": "training"},
            "parent_attempt_sha256": f"{index + 1:064x}",
            "parent_attempt_path": str(source_attempt),
            "parent_readiness_sha256": "a" * 64,
            "parent_progress_sha256": _sha(source_progress.read_bytes()),
            "parent_checkpoint_root": str(parent_root),
            "parent_stage_root": str(parent_stage),
            "parent_latest": latest,
            "parent_receipt": receipt,
            "parent_receipt_path": str(parent_stage / "parent.json"),
            "parent_receipt_sha256": latest["receipt_sha256"],
            "parent_bundle_path": str(parent_stage / "parent.pt"),
            "parent_bundle_sha256": receipt["sha256"],
            "parent_checkpoint_digest": receipt["digest"],
            "parent_cursor": {"epoch": 0, "index": 0},
            "parent_episodes": 0,
            "parent_updates": 0,
            "parent_bindings": {"source_hashes": {"synthetic": "f" * 64}},
            "source_compatibility": source_compatibility,
            "initial_charge": 100.0 + index,
            "tail_status": "conservative_upper_bound",
            "accounting_start": runner.current_accounting_snapshot(100.0 + index),
        }

    monkeypatch.setattr(runner, "paths", lambda arm: path_map[arm])
    monkeypatch.setattr(runner, "_parent_context", lambda arm: contexts[arm])
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda arm: (1.0, {}))
    monkeypatch.setattr(
        runner,
        "tail_charge_from_snapshot",
        lambda snapshot: float(snapshot["charged_cpu_seconds"]),
    )

    def initialize(parent_records, source_report_sha256, approved_utc):
        initialize_calls.append(dict(parent_records))
        assert set(parent_records) == set(runner.ARMS)
        assert source_report_sha256 == _sha(architecture.read_bytes())
        assert approved_utc == runner.APPROVED_UTC
        contract = {
            # durable contract 记录不可变 v1 预算（BASE）；有效预算由修订链承载。
            "per_arm_cpu_seconds": ledger.BASE_PER_ARM_CPU_LIMIT,
            "aggregate_cpu_seconds": ledger.BASE_AGGREGATE_CPU_LIMIT,
            "durability": {"volume_root": volume_root},
            "parent_records": parent_records,
            "source_compatibility_report_sha256": source_report_sha256,
            "approved_utc": approved_utc,
        }
        runner._extension_publish(
            path_map[runner.ARMS[0]]["contract_path"],
            lambda stream: stream.write(_json_bytes(contract)),
        )
        for arm in runner.ARMS:
            charges[arm] = float(parent_records[arm]["charged_cpu_seconds"])
        return {"contract": contract}

    def reconcile(parent_records, source_report_sha256, approved_utc):
        contract = json.loads(path_map[runner.ARMS[0]]["contract_path"].read_text(encoding="utf-8"))
        assert contract["parent_records"] == parent_records
        assert contract["source_compatibility_report_sha256"] == source_report_sha256
        assert contract["approved_utc"] == approved_utc
        return {"contract": contract}

    def append(arm, event, charged_cpu_seconds, details):
        append_calls.append((arm, event, charged_cpu_seconds, details))
        charges[arm] = max(charges[arm], float(charged_cpu_seconds))
        active_reservations.pop(arm, None)
        record = {
            "arm": arm,
            "event": event,
            "charged_cpu_seconds": charged_cpu_seconds,
            "reservation_id": details["reservation_id"],
            "details": details,
            "sequence": len(append_calls),
            "event_sha256": f"{len(append_calls):064x}",
            "created_utc": "2026-10-22T00:00:00Z",
        }
        settlement_records.append(record)
        return record

    def effective_charges():
        effective = dict(charges)
        for arm, reservation in active_reservations.items():
            effective[arm] += float(reservation["reserved_cpu_seconds"])
        return effective

    def find_settlement(arm, token):
        matches = [
            event
            for event in settlement_records
            if event["arm"] == arm and event["reservation_id"] == token
        ]
        if len(matches) > 1:
            raise AssertionError("synthetic reservation settled more than once")
        return matches[0] if matches else None

    monkeypatch.setattr(runner, "initialize_ledger", initialize)
    guard_calls: list[tuple] = []

    def guard(arm, next_charge, reserve_cpu_seconds=0.0):
        guard_calls.append((arm, next_charge, reserve_cpu_seconds))
        token = f"{len(guard_calls):032x}"
        active_reservations[arm] = {
            "reservation_id": token,
            "reserved_cpu_seconds": reserve_cpu_seconds,
        }
        return token

    monkeypatch.setattr(runner, "ledger_guard", guard)
    monkeypatch.setattr(runner, "reconcile_initialize_ledger", reconcile)
    monkeypatch.setattr(runner, "latest_charges", effective_charges)
    monkeypatch.setattr(runner, "latest_reservations", lambda: dict(active_reservations))
    monkeypatch.setattr(runner, "_find_settlement_event", find_settlement)
    monkeypatch.setattr(runner, "append_charge", append)
    monkeypatch.setattr(
        runner,
        "_fresh_process_restore",
        lambda *args, **kwargs: pytest.fail("prepare-only must not restore a model"),
    )
    monkeypatch.setattr(
        runner,
        "execute_arm",
        lambda *args, **kwargs: pytest.fail("prepare-only must not train"),
    )

    with pytest.raises(OSError, match="after parent latest copy"):
        runner.prepare_only()
    warm_copy = path_map["warm_start"]["extension_checkpoint_root"] / "pretraining"
    assert (warm_copy / "latest.json").is_file()
    assert not (warm_copy / "parent.json").exists()

    prepared = runner.prepare_only()
    assert set(prepared["charges"]) == set(runner.ARMS)
    assert len(initialize_calls) == 1
    assert fail_parent_receipt_once["value"] is False
    assert not active_reservations
    assert all(
        call[2] == runner.PREPARE_EVENT_RESERVE_SECONDS + runner.LEDGER_EVENT_RESERVE_SECONDS
        for call in guard_calls
    )
    assert sum("parent.pt" in path for path, _, _ in publisher_calls) == 3
    assert sum("parent.json" in path for path, _, _ in publisher_calls) == 4
    assert sum("latest.json" in path for path, _, _ in publisher_calls) == 3
    assert all(root == volume_root for _, _, root in publisher_calls)
    assert any(replace for _, replace, _ in publisher_calls)
    assert len(append_calls) >= 3
    for arm in runner.ARMS:
        assert path_map[arm]["extension_attempt"].is_file()
        assert path_map[arm]["extension_progress"].is_file()
        assert not path_map[arm]["extension_result"].exists()
        source_attempt_bytes = path_map[arm]["parent_attempt"].read_bytes()
        assert source_attempt_bytes.startswith(b"immutable-")
    published_count = len(publisher_calls)
    runner.prepare_only()
    assert len(initialize_calls) == 1
    assert len(publisher_calls) == published_count


def test_prepare_write_failure_stops_before_restore_or_training(
    tmp_path: Path, monkeypatch
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    architecture = reports / "architecture.md"
    architecture.write_text("synthetic reviewed contract\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    volume_root = "SYNTHETIC:\\"
    location_map = {}
    contexts = {}
    for index, arm in enumerate(runner.ARMS):
        location_map[arm] = {
            "parent_attempt": reports / f"{arm}_parent_attempt.json",
            "parent_readiness": reports / f"{arm}_parent_readiness.json",
            "parent_progress": reports / f"{arm}_parent_progress.jsonl",
            "parent_result": reports / f"{arm}_parent_result.json",
            "parent_checkpoint_root": data_root / "parents" / arm,
            "extension_attempt": reports / f"{arm}_extension_attempt.json",
            "extension_progress": reports / f"{arm}_extension_progress.jsonl",
            "extension_result": reports / f"{arm}_extension_result.json",
            "extension_checkpoint_root": data_root / "extension" / arm,
            "contract_path": reports / "contract.json",
            "aggregate_ledger_path": reports / "ledger",
        }
        contexts[arm] = {
            "arm": arm,
            "locations": location_map[arm],
            "parent_attempt_sha256": f"{index + 1:064x}",
            "parent_receipt_sha256": "b" * 64,
            "parent_bundle_sha256": "c" * 64,
            "initial_charge": 10.0,
            "tail_status": "closed",
            "accounting_start": runner.current_accounting_snapshot(10.0),
            "parent_cursor": {"epoch": 0, "index": 0},
            "source_compatibility": {},
        }

    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", architecture)
    monkeypatch.setattr(runner, "REPORTS_DIR", reports)
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", data_root / "extension")
    monkeypatch.setattr(runner, "paths", lambda arm: location_map[arm])
    monkeypatch.setattr(runner, "_parent_context", lambda arm: contexts[arm])
    monkeypatch.setattr(runner, "_has_any_extension_artifact", lambda: False)
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda arm: (1.0, {}))
    monkeypatch.setattr(
        runner,
        "qualify_storage_target",
        lambda path, expected_volume_root=None: {
            "volume_root": volume_root,
            "filesystem": "NTFS",
            "drive_type": ledger.DRIVE_FIXED,
        },
    )
    monkeypatch.setattr(
        runner, "_sidecar_attempt", lambda context, report_sha: {"arm": context["arm"]}
    )
    monkeypatch.setattr(
        runner,
        "initialize_ledger",
        lambda *args: {
            "contract": {
                "durability": {"volume_root": volume_root},
                "per_arm_cpu_seconds": runner.PER_ARM_CPU_LIMIT,
                "aggregate_cpu_seconds": runner.AGGREGATE_CPU_LIMIT,
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "latest_charges",
        lambda: {arm: 10.0 for arm in runner.ARMS},
    )
    monkeypatch.setattr(runner, "latest_reservations", lambda: {})
    monkeypatch.setattr(
        runner, "_parent_copy_state", lambda context: ({"parent.pt": b""}, {"parent.pt"})
    )
    monkeypatch.setattr(
        runner,
        "append_charge",
        lambda arm, event, charged, details: {
            "arm": arm,
            "event": event,
            "charged_cpu_seconds": charged,
            "reservation_id": details["reservation_id"],
            "details": details,
            "sequence": 0,
            "event_sha256": "d" * 64,
            "created_utc": "2026-10-22T00:00:00Z",
        },
    )
    monkeypatch.setattr(runner, "ledger_guard", lambda *args, **kwargs: "a" * 32)

    def fail_publish(*args, **kwargs):
        raise OSError("synthetic MoveFileExW write-through failure")

    monkeypatch.setattr(runner, "durable_publish", fail_publish)
    monkeypatch.setattr(runner.recovery, "_publish", runner.recovery._publish)
    monkeypatch.setattr(runner.recovery, "_publish_json", runner.recovery._publish_json)
    monkeypatch.setattr(runner.frozen, "atomic_save_final", runner.frozen.atomic_save_final)
    monkeypatch.setattr(
        runner,
        "_fresh_process_restore",
        lambda *args, **kwargs: pytest.fail("failed prepare must not restore a model"),
    )
    monkeypatch.setattr(
        runner,
        "execute_arm",
        lambda *args, **kwargs: pytest.fail("failed prepare must not train"),
    )

    with pytest.raises(OSError, match="write-through failure"):
        runner.prepare_only()
    assert not location_map[runner.ARMS[0]]["extension_attempt"].exists()


def test_prepare_refuses_checkpoint_volume_mismatch_before_ledger_initialization(
    tmp_path: Path, monkeypatch
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    extension_root = tmp_path / "other-volume" / "extension"
    extension_root.mkdir(parents=True)
    location_map = {
        arm: {
            "extension_checkpoint_root": extension_root / arm,
        }
        for arm in runner.ARMS
    }
    monkeypatch.setattr(runner, "REPORTS_DIR", reports)
    monkeypatch.setattr(runner, "paths", lambda arm: location_map[arm])

    def qualify(path, expected_volume_root=None):
        if Path(path).is_relative_to(extension_root):
            return {"volume_root": "OTHER:\\"}
        return {"volume_root": "REPORTS:\\"}

    monkeypatch.setattr(runner, "qualify_storage_target", qualify)
    with pytest.raises(runner.Refuse, match="different volume"):
        runner._qualify_storage_targets()


def test_start_charge_uses_latest_extension_snapshot_monotonically(
    tmp_path: Path, monkeypatch
) -> None:
    prefix = tmp_path / "aggregate_cpu_ledger"
    snapshot = {
        "charged_cpu_seconds": 800.0,
        "snapshot_utc": "2026-10-22T00:00:00+00:00",
        "boot_identity": 20.0,
        "host_cpu_seconds": 100.0,
        "logical_cpu_upper": 8,
    }
    (prefix.parent / f"{prefix.name}.00000001.json").write_text(
        json.dumps(
            {
                "arm": "full_turn_1m",
                "details": {"accounting_snapshot": snapshot},
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "continuation_v1" / "full_turn_1m"
    stage_root = root / "pretraining"
    stage_root.mkdir(parents=True)
    context = {
        "arm": "full_turn_1m",
        "parent_receipt_sha256": "a" * 64,
        "locations": {
            "aggregate_ledger_path": prefix,
            "extension_checkpoint_root": root,
        },
    }
    monkeypatch.setattr(
        runner,
        "_expected_parent_paths",
        lambda arm: {"aggregate_ledger_path": prefix},
    )
    monkeypatch.setattr(runner, "latest_charges", lambda: {"full_turn_1m": 810.0})
    monkeypatch.setattr(
        runner,
        "_latest_extension_envelope",
        lambda context: (
            "pretraining",
            {"synthetic": True},
            stage_root / "parent.pt",
        ),
    )
    (stage_root / "latest.json").write_text(
        json.dumps({"receipt_sha256": "a" * 64}), encoding="utf-8"
    )
    monkeypatch.setattr(runner.psutil, "boot_time", lambda: 21.0)
    monkeypatch.setattr(runner.frozen, "host_cpu_seconds", lambda: 200.0)
    now = datetime(2026, 10, 22, 0, 0, 10, tzinfo=UTC)
    monkeypatch.setattr(runner, "utc_now", lambda: now.isoformat())
    monkeypatch.setattr(
        runner,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: now,
            fromisoformat=datetime.fromisoformat,
        ),
    )

    # Re-anchor from the most recent extension event, not the old parent
    # checkpoint timestamp; max() keeps the cumulative charge monotonic.
    assert runner._start_charge(context) == 1_040.0


def test_fresh_process_restore_failure_fails_closed(tmp_path: Path, monkeypatch) -> None:
    class FakeBudget:
        child_seconds = 0.0
        reserves: list[float] = []

        def guard(self, reserve=None):
            self.reserves.append(reserve)

    host_values = iter((10.0, 60.0))
    monkeypatch.setattr(runner.frozen, "host_cpu_seconds", lambda: next(host_values))
    calls: list[dict] = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="synthetic failure")

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        run,
    )
    budget = FakeBudget()
    with pytest.raises(runner.Refuse, match="fresh-process pretraining restore failed"):
        runner._fresh_process_restore(
            tmp_path / "synthetic-parent.pt",
            {"arm": "full_turn_1m", "stage": "pretraining"},
            "pretraining",
            budget,
        )
    assert budget.reserves == [runner.FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS]
    assert calls[0]["timeout"] == runner.FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS
    assert budget.child_seconds == 50.0


def test_fresh_process_restore_timeout_keeps_restore_reservation_unsettled(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeBudget:
        child_seconds = 0.0
        reserves: list[float] = []

        def guard(self, reserve=None):
            self.reserves.append(reserve)

    host_values = iter((25.0, 145.0))
    monkeypatch.setattr(runner.frozen, "host_cpu_seconds", lambda: next(host_values))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        ),
    )
    budget = FakeBudget()
    with pytest.raises(runner.Refuse, match="fresh-process assistant_sft restore timed out"):
        runner._fresh_process_restore(
            tmp_path / "synthetic-parent.pt",
            {"arm": "warm_start", "stage": "assistant_sft"},
            "assistant_sft",
            budget,
        )
    assert budget.reserves == [runner.FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS]
    assert budget.child_seconds == 120.0


def test_final_restore_uses_bounded_worker_and_accounts_cpu(tmp_path: Path, monkeypatch) -> None:
    class FakeBudget:
        child_seconds = 0.0

        def __init__(self):
            self.reserves = []

        def guard(self, reserve=None):
            self.reserves.append(reserve)

    checkpoint = tmp_path / "synthetic-final.pt"
    checkpoint.write_bytes(b"synthetic final checkpoint")
    host_values = iter((10.0,))
    monkeypatch.setattr(runner.frozen, "host_cpu_seconds", lambda: next(host_values))
    calls: list[tuple] = []
    receipt = {
        "complete": True,
        "equal": True,
        "stage": "assistant_sft",
        "cpu_seconds": 7.5,
    }

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(receipt),
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    budget = FakeBudget()

    result = runner._fresh_final_restore(checkpoint, "assistant_sft", budget)

    assert calls[0][0][1] == str(Path(runner.base.__file__).resolve())
    assert calls[0][0][2:4] == ["--restore-worker", "--checkpoint"]
    assert calls[0][1]["timeout"] == runner.FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "1"
    assert budget.reserves == [runner.FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS, 10.0]
    assert budget.child_seconds == 7.5
    assert result["stage"] == "assistant_sft"
    assert result["sha256"] == _sha(checkpoint.read_bytes())


def test_final_restore_timeout_is_bounded_and_charges_host_delta(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeBudget:
        child_seconds = 0.0

        def __init__(self):
            self.reserves = []

        def guard(self, reserve=None):
            self.reserves.append(reserve)

    host_values = iter((40.0, 160.0))
    monkeypatch.setattr(runner.frozen, "host_cpu_seconds", lambda: next(host_values))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        ),
    )
    budget = FakeBudget()

    with pytest.raises(runner.Refuse, match="final pretraining restore timed out"):
        runner._fresh_final_restore(tmp_path / "synthetic-final.pt", "pretraining", budget)

    assert budget.reserves == [runner.FRESH_PROCESS_RESTORE_TIMEOUT_SECONDS]
    assert budget.child_seconds == 120.0


def _terminal_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    arm = "warm_start"
    parent_attempt_sha = "a" * 64
    parent_receipt_sha = "b" * 64
    attempt_path = tmp_path / f"{arm}_attempt.json"
    attempt = {
        "schema": "taiji-r2-dialogue-continuation-extension-attempt-v1",
        "arm": arm,
        "continuation_sources": {"runner_sha256": "c" * 64},
        "source_compatibility_report_sha256": "d" * 64,
    }
    attempt_path.write_bytes(_json_bytes(attempt))
    result_path = tmp_path / f"{arm}_result.json"
    progress_path = tmp_path / f"{arm}_progress.jsonl"
    resource_event = {
        "event": "training_resource_preflight_passed",
        "arm": arm,
        "resources": {"available_memory_gib": 16.0, "cpu_percent_samples": [10.0, 11.0]},
    }
    progress_path.write_bytes(_json_bytes(resource_event))
    stages = {}
    for stage, episodes, updates, letter in (
        ("pretraining", 30_096, 14_517_912, "e"),
        ("assistant_sft", 525, 113_209, "f"),
    ):
        checkpoint = tmp_path / f"{arm}_{stage}_final.pt"
        checkpoint.write_bytes(f"synthetic-{stage}".encode())
        checkpoint_digest = letter * 64
        stages[f"{stage}_final"] = {
            "path": str(checkpoint.resolve()),
            "sha256": _sha(checkpoint.read_bytes()),
            "checkpoint_digest": checkpoint_digest,
            "episodes": episodes,
            "target_byte_updates": updates,
            "restore": {
                "complete": True,
                "equal": True,
                "stage": stage,
                "checkpoint_digest": checkpoint_digest,
            },
        }
    readiness_path = result_path.with_name(result_path.stem + "_readiness.json")
    ready = {
        "schema": "taiji-r2-dialogue-continuation-extension-readiness-v1",
        "status": "ready",
        "arm": arm,
        "parent_attempt_sha256": parent_attempt_sha,
        "parent_receipt_sha256": parent_receipt_sha,
        "extension_attempt_sha256": _sha(attempt_path.read_bytes()),
        "extension_runner_sha256": "c" * 64,
        "source_compatibility_report_sha256": "d" * 64,
        "stages": stages,
        "cpu_seconds_cumulative": 500.0,
        "cpu_seconds_cumulative_is_lower_bound": True,
        "cpu_seconds_cumulative_boundary": "measured_before_final_result_publication",
        "completion_cpu_source": "continuation_completed_ledger_event",
        "dev_queries": 0,
        "reference_queries": 0,
        "generation_queries": 0,
        "can_promote": False,
    }
    readiness_path.write_bytes(_json_bytes(ready))
    context = {
        "arm": arm,
        "parent_attempt_sha256": parent_attempt_sha,
        "parent_receipt_sha256": parent_receipt_sha,
        "locations": {
            "extension_attempt": attempt_path,
            "extension_result": result_path,
            "extension_checkpoint_root": tmp_path / "checkpoints",
        },
        "parent_bindings": {"arm": arm},
    }
    return context, attempt, progress_path


def test_finalization_recovers_readiness_only_and_completion_progress_gap(
    tmp_path: Path, monkeypatch
) -> None:
    context, attempt, progress_path = _terminal_fixture(tmp_path)
    arm = context["arm"]
    result_path = context["locations"]["extension_result"]
    reservation = {"reservation_id": "r" * 32, "reserved_cpu_seconds": 540.0}
    ledger_events: list[dict[str, Any]] = []
    completion_settlements: list[dict[str, Any]] = []

    monkeypatch.setattr(runner, "_start_charge", lambda _context: 500.0)
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda _arm: (10.0, {}))
    monkeypatch.setattr(runner, "latest_reservations", lambda: {arm: reservation})
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda _arm: list(ledger_events))
    monkeypatch.setattr(
        runner,
        "_write_json_once",
        lambda path, payload: path.write_bytes(_json_bytes(payload)) or "sha",
    )

    def settle(_arm, event, budget, details, _progress, *, conservative_upper=False):
        assert event == "continuation_completed"
        assert conservative_upper is True
        record = {
            "arm": arm,
            "event": event,
            "event_sha256": "9" * 64,
            "reservation_id": reservation["reservation_id"],
            "details": dict(details),
        }
        ledger_events.append(record)
        completion_settlements.append(record)
        return record

    monkeypatch.setattr(runner, "_append_cpu_snapshot", settle)
    monkeypatch.setattr(
        runner,
        "_append_progress_from_ledger",
        lambda _path, _existing, event: completion_settlements.append(event),
    )
    monkeypatch.setattr(
        runner.recovery,
        "train_next",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not retrain")),
    )

    result = runner._reconcile_finalization(arm, context, attempt, progress_path)
    assert result is not None and result["status"] == "complete"
    assert result["cpu_seconds_cumulative_is_lower_bound"] is True
    assert result["completion_cpu_source"] == "continuation_completed_ledger_event"
    assert result_path.is_file()
    assert len(completion_settlements) == 1
    reservation = {}
    again = runner._reconcile_finalization(arm, context, attempt, progress_path)
    assert again == result
    assert len(completion_settlements) == 2  # one ledger event plus one progress repair


def test_finalization_rejects_result_without_readiness(tmp_path: Path) -> None:
    context, attempt, progress_path = _terminal_fixture(tmp_path)
    readiness = runner._readiness_path(context["locations"]["extension_result"])
    readiness.unlink()
    context["locations"]["extension_result"].write_bytes(_json_bytes({"status": "complete"}))
    with pytest.raises(runner.Refuse, match="without its readiness sidecar"):
        runner._reconcile_finalization(context["arm"], context, attempt, progress_path)


def test_finalization_does_not_publish_result_without_reservation_or_completion(
    tmp_path: Path, monkeypatch
) -> None:
    context, attempt, progress_path = _terminal_fixture(tmp_path)
    arm = context["arm"]
    monkeypatch.setattr(runner, "_verified_ledger_events", lambda _arm: [])
    monkeypatch.setattr(runner, "latest_reservations", lambda: {})
    with pytest.raises(runner.Refuse, match="lacks a completion event or reservation"):
        runner._reconcile_finalization(arm, context, attempt, progress_path)
    assert not context["locations"]["extension_result"].exists()


def test_extension_cursor_rng_and_32_episode_commit_resume_are_equivalent(
    tmp_path: Path,
) -> None:
    import random

    import torch

    from scripts.training.taiji_r2_dialogue_recovery import Cursor, RecoveryStore, train_next
    from taiji import LanguageAlignmentTrainer, LanguageEpisode, LanguageEpisodeCorpus, Taiji

    torch.set_num_threads(1)
    corpus = LanguageEpisodeCorpus(
        tuple(
            LanguageEpisode(
                episode_id=f"episode-{index}",
                family_id=f"family-{index}",
                task_family="fact",
                split="train",
                context="",
                user_input=f"question {index}",
                response="answer",
            )
            for index in range(3_762)
        ),
        require_eval_splits=False,
    )
    template = LanguageAlignmentTrainer(
        Taiji(episode_id="extension-cursor-resume-test"),
        corpus,
        code_revision="synthetic-extension-resume",
    )
    parent_episode_count = runner.PARENT_DURABLE_EPISODES["full_turn_1m"]
    parent_updates = runner.PARENT_DURABLE_UPDATES["full_turn_1m"]
    parent_cursor = Cursor(**runner.PARENT_CURSOR["full_turn_1m"])
    template.episode_count = parent_episode_count
    template.global_step = parent_updates
    template.history = [
        {
            "epoch": index // len(corpus.episodes),
            "episode_id": corpus.episodes[index % len(corpus.episodes)].episode_id,
            "family_id": corpus.episodes[index % len(corpus.episodes)].family_id,
            "observations": 1,
        }
        for index in range(parent_episode_count)
    ]
    template.history[0]["observations"] += parent_updates - parent_episode_count
    template.developmental_history = [{"epoch": index} for index in range(parent_cursor.epoch)]
    parent_payload = template.checkpoint()
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()

    uninterrupted = LanguageAlignmentTrainer.from_checkpoint(parent_payload, corpus)
    random.setstate(python_rng)
    torch.set_rng_state(torch_rng)
    uninterrupted_cursor = parent_cursor
    uninterrupted_records = []
    for _ in range(runner.SAVE_INTERVAL + 1):
        record, uninterrupted_cursor = train_next(uninterrupted, uninterrupted_cursor, 8)
        uninterrupted_records.append(record)

    candidate = LanguageAlignmentTrainer.from_checkpoint(parent_payload, corpus)
    random.setstate(python_rng)
    torch.set_rng_state(torch_rng)
    cursor = parent_cursor
    checkpoint_root = tmp_path / "extension-pretraining"
    bindings = {"arm": "synthetic", "stage": "pretraining"}
    with RecoveryStore(checkpoint_root, bindings, corpus) as store:
        for _ in range(runner.SAVE_INTERVAL - 1):
            _, cursor = train_next(candidate, cursor, 8)
        assert candidate.episode_count == parent_episode_count + runner.SAVE_INTERVAL - 1
        assert candidate.episode_count % runner.SAVE_INTERVAL != 0
        store.commit(candidate, cursor, 8, {"synthetic_parent": True})
        candidate, cursor, _ = store.load_latest()
        next_record, cursor = train_next(candidate, cursor, 8)
        assert next_record == uninterrupted_records[runner.SAVE_INTERVAL - 1]
        assert candidate.episode_count == parent_episode_count + runner.SAVE_INTERVAL
        assert cursor == Cursor(parent_cursor.epoch, parent_cursor.index + runner.SAVE_INTERVAL)
        assert candidate.episode_count % runner.SAVE_INTERVAL == 0
        store.commit(candidate, cursor, 8, {"synthetic_boundary": True})

    # Simulate a crash after latest.json is durable but before the runner's
    # progress/CPU event append. Resume must load the committed 32-episode
    # state and must not perform the same training update again.
    with RecoveryStore(checkpoint_root, bindings, corpus) as store:
        resumed, cursor, payload = store.load_latest()
        assert resumed.episode_count == parent_episode_count + runner.SAVE_INTERVAL
        assert len(resumed.history) == parent_episode_count + runner.SAVE_INTERVAL
        assert cursor == Cursor(parent_cursor.epoch, parent_cursor.index + runner.SAVE_INTERVAL)
        assert payload["ledger"] == {"synthetic_boundary": True}
        next_record, cursor = train_next(resumed, cursor, 8)
        assert next_record == uninterrupted_records[-1]
    assert (
        resumed.checkpoint()["checkpoint_digest"] == uninterrupted.checkpoint()["checkpoint_digest"]
    )


def test_throughput_projection_uses_recent_extension_rate_and_shared_budget(
    tmp_path: Path, monkeypatch
) -> None:
    parent_events = [
        {
            "event": "durable_training_commit",
            "stage": "pretraining",
            "episodes": 100,
            "cpu_seconds": 1000.0,
        },
        {
            "event": "durable_training_commit",
            "stage": "pretraining",
            "episodes": 132,
            "cpu_seconds": 1128.0,
        },
        {
            "event": "durable_training_commit",
            "stage": "pretraining",
            "episodes": 164,
            "cpu_seconds": 1288.0,
        },
    ]
    extension_events = [
        {
            "event": "continuation_started",
            "stage": "pretraining",
            "episodes": 164,
            "charged_cpu_seconds": 100_000.0,
        }
    ]
    for index in range(1, 5):
        extension_events.append(
            {
                "event": "episode_completed",
                "stage": "pretraining",
                "episodes": 164 + index,
                "charged_cpu_seconds": 100_000.0 + 3.0 * index,
            }
        )
    rate, samples = runner._episode_cpu_rate(parent_events, extension_events, "pretraining")
    assert samples == 4
    assert rate == pytest.approx(3.0 * runner.THROUGHPUT_SAFETY_FACTOR)

    contexts = {}
    for arm in runner.ARMS:
        parent_progress = tmp_path / f"{arm}_parent_progress.jsonl"
        parent_progress.write_bytes(b"".join(_json_bytes(event) for event in parent_events))
        extension_progress = tmp_path / f"{arm}_extension_progress.jsonl"
        extension_progress.write_bytes(b"")
        contexts[arm] = {
            "locations": {
                "parent_progress": parent_progress,
                "extension_progress": extension_progress,
            }
        }
    monkeypatch.setattr(
        runner, "latest_charges_fast", lambda: {arm: 1_000_000.0 for arm in runner.ARMS}
    )
    allowed = runner._throughput_projection(
        "full_turn_1m", contexts, current_stage="pretraining", current_episodes=9_472
    )
    assert allowed["feasible"] is True
    assert allowed["aggregate_headroom_cpu_seconds"] == runner.AGGREGATE_CPU_LIMIT - 3_000_000.0
    assert allowed["rates"]["full_turn_1m"]["pretraining_samples"] == 2

    monkeypatch.setattr(
        runner,
        "latest_charges_fast",
        lambda: {arm: runner.PER_ARM_CPU_LIMIT - 110_000.0 for arm in runner.ARMS},
    )
    refused = runner._throughput_projection(
        "full_turn_1m", contexts, current_stage="pretraining", current_episodes=9_472
    )
    assert refused["feasible"] is False
    assert refused["arm_headroom_cpu_seconds"] == 110_000.0


def test_throughput_projection_checks_every_arm_ceiling(tmp_path: Path, monkeypatch) -> None:
    parent_events = [
        {
            "event": "durable_training_commit",
            "stage": "pretraining",
            "episodes": 100,
            "cpu_seconds": 1000.0,
        },
        {
            "event": "durable_training_commit",
            "stage": "pretraining",
            "episodes": 132,
            "cpu_seconds": 1128.0,
        },
        {
            "event": "durable_training_commit",
            "stage": "pretraining",
            "episodes": 164,
            "cpu_seconds": 1288.0,
        },
    ]
    contexts = {}
    for arm in runner.ARMS:
        parent_progress = tmp_path / f"{arm}_parent_progress.jsonl"
        parent_progress.write_bytes(b"".join(_json_bytes(event) for event in parent_events))
        extension_progress = tmp_path / f"{arm}_extension_progress.jsonl"
        extension_progress.write_bytes(b"")
        contexts[arm] = {
            "locations": {
                "parent_progress": parent_progress,
                "extension_progress": extension_progress,
            }
        }
    monkeypatch.setattr(
        runner,
        "latest_charges_fast",
        lambda: {
            "full_turn_1m": 1_000_000.0,
            "warm_start": runner.PER_ARM_CPU_LIMIT - 10_000.0,
            "fresh_same_capacity": 1_000_000.0,
        },
    )

    projection = runner._throughput_projection(
        "full_turn_1m", contexts, current_stage="assistant_sft", current_episodes=525
    )

    assert projection["aggregate_headroom_cpu_seconds"] > projection["projected_cpu_aggregate"]
    assert (
        projection["projected_cpu_by_arm"]["warm_start"]
        > projection["arm_headroom_by_arm_cpu_seconds"]["warm_start"]
    )
    assert projection["per_arm_feasible"]["warm_start"] is False
    assert projection["feasible"] is False


def test_failed_work_settlement_uses_unsettled_reservation_upper(
    tmp_path: Path, monkeypatch
) -> None:
    settlements: list[tuple] = []
    monkeypatch.setattr(runner.frozen, "episode_cpu_reserve", lambda arm: (1.0, {}))
    monkeypatch.setattr(runner.time, "process_time", lambda: 10.0)
    monkeypatch.setattr(
        runner,
        "append_charge",
        lambda arm, event, charged, details: (
            settlements.append((arm, event, charged, dict(details))) or {"arm": arm, "event": event}
        ),
    )
    monkeypatch.setattr(
        runner,
        "current_accounting_snapshot",
        lambda charged: {"charged_cpu_seconds": charged},
    )
    progress = tmp_path / "stopped.jsonl"
    progress.write_bytes(b"")
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "EXTENSION_CHECKPOINT_ROOT", tmp_path / "extension")
    monkeypatch.setattr(runner, "_APPROVED_VOLUME_ROOT", "SYNTHETIC:\\")
    monkeypatch.setattr(runner, "durable_publish", _fake_write_through_publish)
    budget = runner.ExtensionBudget("warm_start", 500.0)
    budget.active_reservation_id = "b" * 32
    budget.active_reservation_upper = 622.0
    budget.child_seconds = 50.0

    runner._append_cpu_snapshot(
        "warm_start",
        "continuation_stopped",
        budget,
        {"unknown_tail": True},
        progress,
        conservative_upper=True,
    )

    assert settlements[0][1] == "continuation_stopped"
    assert settlements[0][2] == 622.0
    assert settlements[0][3]["reservation_id"] == "b" * 32
    assert json.loads(progress.read_text(encoding="utf-8"))["charged_cpu_seconds"] == 622.0

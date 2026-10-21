"""Offline contracts for the restart runner; no checkpoint or dev data loads."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import torch

from scripts.training import run_taiji_r2_daily_dialogue_recovery as recovery


def test_recovery_arms_have_disjoint_write_once_paths() -> None:
    paths = [recovery.paths(arm) for arm in recovery.ARMS]
    assert len({row["root"] for row in paths}) == 3
    assert len({row["readiness"] for row in paths}) == 3
    assert len({row["result"] for row in paths}) == 3
    assert all(row["root"].is_relative_to(recovery.CHECKPOINT_ROOT) for row in paths)


def test_interruption_tail_uses_monotonic_host_cpu_delta(monkeypatch) -> None:
    snapshot = datetime.now(UTC) - timedelta(seconds=10)
    monkeypatch.setattr(recovery, "host_cpu_seconds", lambda: 500.0)
    monkeypatch.setattr(recovery.psutil, "boot_time", lambda: 1_000.0)
    charge = recovery.tail_charge(
        {
            "ledger": {
                "charged_cpu_seconds": 100.0,
                "host_cpu_seconds": 450.0,
                "host_boot_time": 1_000.0,
                "snapshot_utc": snapshot.isoformat(),
                "logical_cpu_upper": 24,
            }
        }
    )
    assert charge == 150


def test_interruption_tail_uses_wall_bound_after_host_cpu_counter_reset(monkeypatch) -> None:
    snapshot = datetime.now(UTC) - timedelta(seconds=10)
    monkeypatch.setattr(recovery, "host_cpu_seconds", lambda: 10.0)
    monkeypatch.setattr(recovery.psutil, "boot_time", lambda: 2_000.0)
    charge = recovery.tail_charge(
        {
            "ledger": {
                "charged_cpu_seconds": 100,
                "host_cpu_seconds": 450,
                "host_boot_time": 1_000.0,
                "snapshot_utc": snapshot.isoformat(),
                "logical_cpu_upper": 24,
            }
        }
    )
    assert charge >= 100 + 9 * max(24, recovery.os.cpu_count() or 1)


def test_tail_accounting_refuses_missing_capacity_evidence_after_reboot(monkeypatch) -> None:
    monkeypatch.setattr(recovery.psutil, "boot_time", lambda: 2_000.0)
    with pytest.raises(KeyError):
        recovery.tail_charge(
            {
                "ledger": {
                    "charged_cpu_seconds": 100,
                    "host_cpu_seconds": 500,
                    "host_boot_time": 1_000.0,
                    "snapshot_utc": datetime.now(UTC).isoformat(),
                }
            }
        )


def test_formal_recovery_budget_does_not_inherit_old_attempt_as_zero() -> None:
    assert recovery.Budget("full_turn_1m").limit == 240_000
    assert recovery.Budget("warm_start").limit == 120_000
    assert recovery.Budget("fresh_same_capacity").limit == 120_000
    assert recovery.ARMS == ("full_turn_1m", "warm_start", "fresh_same_capacity")


def test_start_resource_guard_requires_two_cpu_samples_and_memory(monkeypatch) -> None:
    samples = iter((30.0, 80.0))
    monkeypatch.setattr(recovery.psutil, "cpu_percent", lambda *, interval: next(samples))
    monkeypatch.setattr(recovery.psutil, "virtual_memory", lambda: type("Mem", (), {"available": 16 * 1024**3})())
    result = recovery.resource_snapshot()
    assert result["cpu_percent_samples"] == [30.0, 80.0]
    assert result["available_memory_gib"] == 16


def test_start_resource_guard_rejects_overloaded_cpu(monkeypatch) -> None:
    samples = iter((20.0, 86.0))
    monkeypatch.setattr(recovery.psutil, "cpu_percent", lambda *, interval: next(samples))
    monkeypatch.setattr(recovery.psutil, "virtual_memory", lambda: type("Mem", (), {"available": 16 * 1024**3})())
    with pytest.raises(recovery.base.Refuse, match="CPU start samples"):
        recovery.resource_snapshot()


def test_final_export_is_fsync_published_and_write_once(tmp_path) -> None:
    path = tmp_path / "final.pt"
    checkpoint = {"probe": torch.tensor([1, 2, 3])}
    recovery.atomic_save_final(path, checkpoint)
    loaded = torch.load(path, weights_only=True)
    assert torch.equal(loaded["probe"], checkpoint["probe"])
    with pytest.raises(FileExistsError):
        recovery.atomic_save_final(path, checkpoint)


def test_resume_uses_latest_training_stage_only(tmp_path) -> None:
    with pytest.raises(recovery.base.Refuse, match="no committed"):
        recovery.latest_resume_stage(tmp_path)


def test_resume_prefers_assistant_stage_after_transition(tmp_path) -> None:
    (tmp_path / "pretraining").mkdir()
    (tmp_path / "pretraining" / "latest.json").touch()
    assert recovery.latest_resume_stage(tmp_path) == "pretraining"
    (tmp_path / "assistant_sft").mkdir()
    (tmp_path / "assistant_sft" / "latest.json").touch()
    assert recovery.latest_resume_stage(tmp_path) == "assistant_sft"


def test_failed_commit_charges_host_cpu_delta(monkeypatch) -> None:
    values = iter((100.0, 110.0, 140.0))
    monkeypatch.setattr(recovery, "host_cpu_seconds", lambda: next(values))

    class FailedStore:
        def commit(self, *_args, **_kwargs):
            raise TimeoutError("restore worker timed out")

    budget = recovery.Budget("warm_start")
    with pytest.raises(TimeoutError):
        recovery.commit(FailedStore(), object(), recovery.Cursor(), 1, budget)
    assert budget.child_seconds == 40.0


def test_failed_final_restore_charges_host_cpu_delta(monkeypatch, tmp_path) -> None:
    values = iter((200.0, 225.0))
    monkeypatch.setattr(recovery, "host_cpu_seconds", lambda: next(values))

    def fail_restore(*_args, **_kwargs):
        raise RuntimeError("restore receipt invalid")

    monkeypatch.setattr(recovery.base, "_independent_restore", fail_restore)
    budget = recovery.Budget("warm_start")
    with pytest.raises(RuntimeError, match="receipt invalid"):
        recovery.restore_final(tmp_path / "candidate.pt", "pretraining", budget)
    assert budget.child_seconds == 25.0


def _preparation_output(tmp_path):
    return {
        "root": tmp_path / "checkpoints",
        "prepare_attempt": tmp_path / "prepare_attempt.json",
        "prepare_events": tmp_path / "events",
        "readiness": tmp_path / "readiness.json",
        "result": tmp_path / "result.json",
    }


def _write_attempt(output, monkeypatch):
    import json
    from pathlib import Path

    ledger = {
        "charged_cpu_seconds": 10.0,
        "host_cpu_seconds": 100.0,
        "host_boot_time": 1.0,
        "snapshot_utc": "2026-10-21T00:00:00+00:00",
        "logical_cpu_upper": 24,
    }
    output["prepare_attempt"].write_text(
        json.dumps(
            {
                "runner_sha256": recovery.base.sha256(Path(recovery.__file__)),
                "parent_zero_sha256": "parent",
                "ledger": ledger,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(recovery, "host_cpu_seconds", lambda: 130.0)
    monkeypatch.setattr(recovery.psutil, "boot_time", lambda: 1.0)
    return ledger


def test_prepare_interruption_without_checkpoint_keeps_initial_tail(tmp_path, monkeypatch):
    output = _preparation_output(tmp_path)
    _write_attempt(output, monkeypatch)
    assert recovery.preparation_resume_charge(output, {"zero_sha256": "parent"}, None, None) == 40.0


def test_prepare_resume_prefers_verified_zero_envelope(tmp_path, monkeypatch):
    output = _preparation_output(tmp_path)
    ledger = _write_attempt(output, monkeypatch)
    directory = output["root"] / "pretraining"
    directory.mkdir(parents=True)
    (directory / "latest.json").touch()
    parent = output["root"] / "old-parent.pt"
    parent.write_bytes(b"immutable-zero-parent")
    parent_sha = recovery.base.sha256(parent)

    class VerifiedStore:
        def __init__(self, *_args):
            pass

        def load_latest(self, **_kwargs):
            return (
                None,
                recovery.Cursor(),
                {
                    "trainer": {"global_step": 0},
                    "ledger": {
                        **ledger,
                        "snapshot_utc": "2026-10-21T00:00:01+00:00",
                        "charged_cpu_seconds": 15.0,
                        "host_cpu_seconds": 120.0,
                    },
                },
            )

    monkeypatch.setattr(recovery, "RecoveryStore", VerifiedStore)
    assert recovery.preparation_resume_charge(output, {"zero_sha256": "parent"}, None, None) == 25.0
    assert recovery.base.sha256(parent) == parent_sha


def test_prepare_ready_and_implicit_retry_refused_before_sources(tmp_path, monkeypatch):
    output = _preparation_output(tmp_path)
    monkeypatch.setattr(recovery, "paths", lambda _arm: output)
    output["readiness"].touch()
    with pytest.raises(recovery.base.Refuse, match="readiness/result"):
        recovery.prepare("warm_start", resume=True)
    output["readiness"].unlink()
    output["prepare_attempt"].touch()
    with pytest.raises(recovery.base.Refuse, match="prepare-resume"):
        recovery.prepare("warm_start")


def test_prepared_zero_reuses_receipt_without_parent_write(tmp_path, monkeypatch):
    import json

    (tmp_path / "latest.json").write_text(json.dumps({"receipt": "receipt.json"}), encoding="utf-8")
    (tmp_path / "receipt.json").write_text(json.dumps({"sha256": "unchanged"}), encoding="utf-8")

    class Zero:
        def checkpoint(self):
            return {"checkpoint_digest": "same"}

    class Store:
        directory = tmp_path

        def load_latest(self, **_kwargs):
            return Zero(), recovery.Cursor(), {}

        def commit(self, *_args):
            raise AssertionError("must not re-commit an existing zero")

    before = recovery.base.sha256(tmp_path / "receipt.json")
    assert recovery.prepared_zero(Store(), Zero(), 8, object()) == {"sha256": "unchanged"}
    assert recovery.base.sha256(tmp_path / "receipt.json") == before


def test_preparation_event_carries_complete_cpu_ledger(tmp_path, monkeypatch):
    output = _preparation_output(tmp_path)
    monkeypatch.setattr(recovery, "host_cpu_seconds", lambda: 123.0)
    monkeypatch.setattr(recovery.psutil, "boot_time", lambda: 456.0)
    event = recovery.preparation_event(
        output, "prepare_failed", recovery.Budget("warm_start"), {"arm": "a"}
    )
    assert event["ledger"]["host_cpu_seconds"] == 123.0
    assert event["ledger"]["host_boot_time"] == 456.0
    assert event["formal_updates"] == 0
    assert len(list(output["prepare_events"].glob("*.json"))) == 1


def test_prepare_repeated_source_failures_never_reset_prior_event_charge(tmp_path, monkeypatch):
    output = _preparation_output(tmp_path)
    ledger = _write_attempt(output, monkeypatch)
    import json

    initial = recovery.base._read_json(output["prepare_attempt"])
    initial["arm"] = "warm_start"
    output["prepare_attempt"].write_text(json.dumps(initial), encoding="utf-8")

    class FailedSessionBudget:
        def ledger(self):
            return {
                **ledger,
                "charged_cpu_seconds": 500.0,
                "host_cpu_seconds": 120.0,
                "snapshot_utc": datetime.now(UTC).isoformat(),
            }

    recovery.preparation_event(
        output,
        "prepare_failed",
        FailedSessionBudget(),
        {"arm": "warm_start", "zero_sha256": "parent"},
    )
    monkeypatch.setattr(recovery, "paths", lambda _arm: output)

    def fail_sources(_arm):
        raise recovery.base.Refuse("synthetic source validation failure")

    monkeypatch.setattr(recovery, "source_bindings", fail_sources)
    previous = 510.0
    for _ in range(3):
        with pytest.raises(recovery.base.Refuse, match="synthetic source"):
            recovery.prepare("warm_start", resume=True)
        events = [
            recovery.base._read_json(path) for path in output["prepare_events"].glob("*.json")
        ]
        latest = max(events, key=lambda event: event["ledger"]["snapshot_utc"])
        assert latest["ledger"]["charged_cpu_seconds"] >= previous
        assert latest["arm"] == "warm_start"
        assert latest["runner_sha256"] == initial["runner_sha256"]
        previous = latest["ledger"]["charged_cpu_seconds"]
    assert len(events) == 4

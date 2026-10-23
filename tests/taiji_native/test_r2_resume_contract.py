"""Exercise recovery cost, tamper rejection, and consistent runtime guards."""

from types import SimpleNamespace

import pytest

from scripts.training import run_taiji_r2_resume as resume


def event(token="token", sequence=7):
    row = {
        "arm": "warm_start",
        "event": "continuation_resumed",
        "reservation_id": token,
        "sequence": sequence,
    }
    row["event_sha256"] = resume.ledger._digest(row)
    return row


def setup_progress(monkeypatch, rows, active=None):
    monkeypatch.setattr(resume.runner, "read_jsonl", lambda _: rows.copy())
    monkeypatch.setattr(
        resume.runner,
        "latest_reservations",
        lambda: {} if active is None else {"warm_start": active},
    )


def test_settled_start_reads_one_addressed_event_without_history_scan(monkeypatch):
    settled = event()
    rows = [
        {"event": "continuation_start_intent", "reservation_id": "token"},
        {
            "event": "continuation_resumed",
            "reservation_id": "token",
            "ledger_sequence": 7,
            "ledger_event_sha256": settled["event_sha256"],
        },
    ]
    setup_progress(monkeypatch, rows)
    reads = []
    monkeypatch.setattr(resume.runner, "read_json", lambda path: reads.append(path) or settled)
    monkeypatch.setattr(
        resume.runner,
        "_verified_ledger_events",
        lambda _: pytest.fail("unnecessary full history scan"),
    )
    assert resume.reconcile_start_progress("warm_start", "progress") == rows
    assert len(reads) == 1


def test_recorded_settlement_tampering_is_refused(monkeypatch):
    settled = event()
    rows = [
        {"event": "continuation_start_intent", "reservation_id": "token"},
        {
            "event": "continuation_resumed",
            "reservation_id": "token",
            "ledger_sequence": 7,
            "ledger_event_sha256": settled["event_sha256"],
        },
    ]
    setup_progress(monkeypatch, rows)
    settled["charged_cpu_seconds"] = 0
    monkeypatch.setattr(resume.runner, "read_json", lambda _: settled)
    with pytest.raises(resume.runner.Refuse, match="differs"):
        resume.reconcile_start_progress("warm_start", "progress")


def test_pending_active_start_needs_no_scan(monkeypatch):
    rows = [{"event": "continuation_start_intent", "reservation_id": "token"}]
    setup_progress(monkeypatch, rows, {"reservation_id": "token"})
    monkeypatch.setattr(
        resume.runner,
        "_verified_ledger_events",
        lambda _: pytest.fail("active reservation already verified"),
    )
    assert resume.reconcile_start_progress("warm_start", "progress") == rows


def test_multiple_legacy_intents_share_one_scan(monkeypatch):
    rows = [
        {"event": "continuation_start_intent", "reservation_id": token} for token in ("one", "two")
    ]
    setup_progress(monkeypatch, rows)
    scans = []
    monkeypatch.setattr(
        resume.runner,
        "_verified_ledger_events",
        lambda _: scans.append(1) or [event("one"), event("two", 8)],
    )
    monkeypatch.setattr(
        resume.runner, "_append_progress_from_ledger", lambda _p, _rows, _event: None
    )
    resume.reconcile_start_progress("warm_start", "progress")
    assert len(scans) == 1


@pytest.mark.parametrize("memory,passes", [(7.1, True), (6.9, False)])
def test_same_memory_floor_at_start_and_work_unit(monkeypatch, memory, passes):
    monkeypatch.setattr(
        resume.runner.psutil, "virtual_memory", lambda: SimpleNamespace(available=memory * 1024**3)
    )
    calls = []
    monkeypatch.setattr(
        resume.runner, "ledger_guard", lambda *args, **kw: calls.append(kw) or "reservation"
    )
    budget = object.__new__(resume.ResumeBudget)
    budget.active_reservation_id = None
    budget.episode_reserve = 10
    budget.arm = "warm_start"
    budget.limit = 1000
    budget.pending_stop_details = {}
    budget.used = lambda: 100
    if passes:
        resume.resource_snapshot()
        budget.guard()
        assert budget.active_reservation_id == "reservation"
        assert calls[0]["reserve_cpu_seconds"] > 10
    else:
        with pytest.raises(resume.runner.Refuse):
            resume.resource_snapshot()
        with pytest.raises(resume.runner.Refuse):
            budget.guard()
        assert budget.active_reservation_id == "reservation"  # stop settlement reserve


def test_periodic_budget_metadata_does_not_read_training_histories(monkeypatch):
    monkeypatch.setattr(
        resume, "current_fast_charges", lambda: {arm: 100 for arm in resume.runner.ARMS}
    )
    monkeypatch.setattr(
        resume.runner, "read_jsonl", lambda *_args, **_kw: pytest.fail("repeated history read")
    )
    assert resume.budget_status("warm_start", {})["policy"].startswith("actual_cpu_guard")


def test_existing_amendment_is_adopted_before_reading_over_old_cap(monkeypatch):
    from contextlib import nullcontext

    record = {
        "approved_budget": {"per_arm": 1550000, "aggregate": 4400000},
        "contract_object_sha256": "contract",
    }
    monkeypatch.setattr(resume, "read_validated_amendment", lambda: record)
    monkeypatch.setattr(resume.ledger, "PER_ARM_CPU_LIMIT", 1500000)
    monkeypatch.setattr(resume.ledger, "AGGREGATE_CPU_LIMIT", 4300000)
    monkeypatch.setattr(resume.ledger, "effective_budget_limits", lambda: (1500000, 4300000))
    monkeypatch.setattr(resume.ledger, "_aggregate_lock", lambda _: nullcontext())

    def read_summary_after_adoption():
        assert resume.ledger.PER_ARM_CPU_LIMIT == 1550000
        assert resume.ledger.AGGREGATE_CPU_LIMIT == 4400000
        return {"contract_sha256": "contract"}, {}, {}, "tip"

    monkeypatch.setattr(resume.ledger, "_read_summary_fast", read_summary_after_adoption)
    assert resume.activate_budget() == record

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as runner


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_once(path: Path, payload: dict[str, Any]) -> str:
    raw = _json_bytes(payload)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha(raw)


def _revision_fixture(tmp_path: Path, monkeypatch):
    arm = "warm_start"
    source_report = tmp_path / "architecture.md"
    source_report.write_text("frozen v1 architecture\n", encoding="utf-8")
    source_report_sha = _sha(source_report.read_bytes())
    review_report = tmp_path / "source-revision-review.md"
    review_report.write_text("reviewed v1.1 source adoption\n", encoding="utf-8")
    contract_path = tmp_path / "extension-contract.json"
    contract_path.write_bytes(_json_bytes({"schema": "v1", "budget": "approved"}))
    readiness_path = tmp_path / "parent-readiness.json"
    readiness_path.write_bytes(_json_bytes({"status": "ready"}))
    attempt_path = tmp_path / "extension-attempt.json"
    progress_path = tmp_path / "extension-progress.jsonl"
    progress_path.write_bytes(
        (json.dumps({"event": "continuation_prepared", "arm": arm}) + "\n").encode()
    )
    revision_path = tmp_path / "source-revision.json"
    expected_current = {
        "runner_path": str((tmp_path / "current-runner.py").resolve()),
        "runner_sha256": "a" * 64,
        "files": {"runner": {"path": "current-runner.py", "sha256": "a" * 64}},
        "report_path": str(source_report.resolve()),
        "report_sha256": source_report_sha,
    }
    old_compatibility = {
        "runner_sha256": runner.V1_RUNNER_SHA256,
        "report_sha256": source_report_sha,
    }
    old_sources = {
        "runner_path": "v1-runner.py",
        "runner_sha256": runner.V1_RUNNER_SHA256,
        "files": {name: {"sha256": digest} for name, digest in runner.V1_SOURCE_HASHES.items()},
        "report_path": str(source_report.resolve()),
        "report_sha256": source_report_sha,
    }
    attempt = {
        "schema": "taiji-r2-dialogue-continuation-extension-attempt-v1",
        "arm": arm,
        "status": "prepared",
        "parent": {"attempt_sha256": "b" * 64},
        "source_compatibility_report_sha256": source_report_sha,
        "continuation_sources": old_sources,
        "source_compatibility": old_compatibility,
    }
    attempt_path.write_bytes(_json_bytes(attempt))
    parent_receipt_sha = "c" * 64
    parent_attempt_sha = "b" * 64
    parent_copy_root = tmp_path / "parent-copy"
    parent_copy_root.mkdir()
    extension_root = tmp_path / "extension-checkpoints"
    (extension_root / "pretraining").mkdir(parents=True)
    (extension_root / "pretraining" / "latest.json").write_bytes(
        _json_bytes({"receipt": "parent.json"})
    )
    (extension_root / "pretraining" / "parent.json").write_bytes(_json_bytes({"sha256": "d" * 64}))
    (extension_root / "pretraining" / "parent.pt").write_bytes(b"parent checkpoint")
    context = {
        "arm": arm,
        "parent_attempt_sha256": parent_attempt_sha,
        "parent_receipt_sha256": parent_receipt_sha,
        "parent_bindings": {"source_hashes": {"synthetic": "e" * 64}},
        "source_compatibility": {
            **old_compatibility,
            "runner_sha256": expected_current["runner_sha256"],
        },
        "parent_receipt": {"bundle": "parent.pt"},
        "locations": {
            "extension_attempt": attempt_path,
            "extension_progress": progress_path,
            "extension_result": tmp_path / "extension-result.json",
            "extension_checkpoint_root": extension_root,
            "parent_readiness": readiness_path,
            "parent_checkpoint_root": parent_copy_root,
        },
    }
    monkeypatch.setattr(runner, "SOURCE_REVISION_REPORT", review_report)
    monkeypatch.setattr(runner, "_source_revision_path", lambda: revision_path)
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", source_report)
    monkeypatch.setattr(runner, "_continuation_source_binding", lambda _sha: expected_current)
    monkeypatch.setattr(
        runner,
        "_expected_parent_paths",
        lambda _arm: {
            "contract_path": contract_path,
            "extension_attempt": attempt_path,
            "parent_readiness": readiness_path,
        },
    )
    monkeypatch.setitem(runner.PARENT_RECEIPT_SHA256, arm, parent_receipt_sha)
    return {
        "arm": arm,
        "attempt": attempt,
        "attempt_path": attempt_path,
        "context": context,
        "contract_path": contract_path,
        "expected_current": expected_current,
        "old_sources": old_sources,
        "progress_path": progress_path,
        "readiness_path": readiness_path,
        "review_report": review_report,
        "revision_path": revision_path,
        "source_report_sha": source_report_sha,
    }


def _revision_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "taiji-r2-continuation-source-revision-v1.1",
        "contract_sha256": _sha(case["contract_path"].read_bytes()),
        "original_attempts": {
            case["arm"]: {
                "attempt_sha256": _sha(case["attempt_path"].read_bytes()),
                "parent_readiness_sha256": _sha(case["readiness_path"].read_bytes()),
                "parent_receipt_sha256": "c" * 64,
                "prepared_progress_sha256": _sha(case["progress_path"].read_bytes()),
            }
        },
        "current_continuation_sources": case["expected_current"],
        "review_report_sha256": _sha(case["review_report"].read_bytes()),
        "approved_budget": {
            "per_arm": runner.PER_ARM_CPU_LIMIT,
            "aggregate": runner.AGGREGATE_CPU_LIMIT,
        },
    }


def test_effective_attempt_adopts_revision_in_memory_without_mutating_v1_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    case = _revision_fixture(tmp_path, monkeypatch)
    _write_once(case["revision_path"], _revision_payload(case))
    original_bytes = case["attempt_path"].read_bytes()
    original = json.loads(original_bytes)

    effective = runner._effective_attempt(original, case["source_report_sha"])

    assert effective is not original
    assert effective["continuation_sources"] == case["expected_current"]
    assert (
        effective["source_compatibility"]["runner_sha256"]
        == case["expected_current"]["runner_sha256"]
    )
    assert effective["source_compatibility"]["report_sha256"] == case["source_report_sha"]
    assert effective["source_revision_sha256"] == _sha(case["revision_path"].read_bytes())
    assert original["continuation_sources"] == case["old_sources"]
    assert original["source_compatibility"]["runner_sha256"] == runner.V1_RUNNER_SHA256
    assert case["attempt_path"].read_bytes() == original_bytes


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("old_attempt", "source revision does not exactly bind"),
        ("readiness", "source revision does not exactly bind"),
        ("contract", "source revision does not exactly bind"),
        ("review_report", "source revision does not exactly bind"),
        ("budget", "source revision does not exactly bind"),
        ("old_sources", "source revision does not exactly bind"),
        ("current_binding", "source revision does not exactly bind"),
        ("parent_receipt", "source revision does not exactly bind"),
    ],
)
def test_effective_attempt_rejects_revision_or_v1_lineage_drift(
    tmp_path: Path, monkeypatch, mutation: str, message: str
) -> None:
    case = _revision_fixture(tmp_path, monkeypatch)
    payload = _revision_payload(case)
    if mutation == "old_attempt":
        case["attempt_path"].write_bytes(_json_bytes({**case["attempt"], "extra": True}))
    elif mutation == "readiness":
        case["readiness_path"].write_bytes(_json_bytes({"status": "changed"}))
    elif mutation == "contract":
        case["contract_path"].write_bytes(_json_bytes({"schema": "changed"}))
    elif mutation == "review_report":
        case["review_report"].write_text("changed review\n", encoding="utf-8")
    elif mutation == "budget":
        payload["approved_budget"]["aggregate"] += 1
    elif mutation == "parent_receipt":
        payload["original_attempts"][case["arm"]]["parent_receipt_sha256"] = "f" * 64
    _write_once(case["revision_path"], payload)
    attempt = json.loads(case["attempt_path"].read_bytes())
    if mutation == "old_sources":
        attempt["continuation_sources"]["files"]["runner"]["sha256"] = "9" * 64
    if mutation == "current_binding":
        monkeypatch.setattr(
            runner,
            "_continuation_source_binding",
            lambda _sha: {**case["expected_current"], "runner_sha256": "8" * 64},
        )

    with pytest.raises(runner.Refuse, match=message):
        runner._effective_attempt(attempt, case["source_report_sha"])


def test_effective_attempt_refuses_missing_revision_for_changed_v1_sources(
    tmp_path: Path, monkeypatch
) -> None:
    case = _revision_fixture(tmp_path, monkeypatch)
    with pytest.raises(runner.Refuse, match="without a versioned source revision"):
        runner._effective_attempt(case["attempt"], case["source_report_sha"])


def _prepared_arms_fixture(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    source_report = tmp_path / "architecture.md"
    source_report.write_text("frozen v1 architecture\n", encoding="utf-8")
    source_report_sha = _sha(source_report.read_bytes())
    review_report = tmp_path / "source-revision-review.md"
    review_report.write_text("reviewed v1.1 source adoption\n", encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(_json_bytes({"schema": "v1", "ledger": "same"}))
    expected_current = {
        "runner_path": str((tmp_path / "runner.py").resolve()),
        "runner_sha256": "a" * 64,
        "files": {"runner": {"path": "runner.py", "sha256": "a" * 64}},
        "report_path": str(source_report.resolve()),
        "report_sha256": source_report_sha,
    }
    contexts: dict[str, dict[str, Any]] = {}
    for arm in runner.ARMS:
        old_compat = {"runner_sha256": runner.V1_RUNNER_SHA256, "proof": "v1"}
        receipt_index = runner.ARMS.index(arm) + 1
        parent_attempt_sha = f"{receipt_index}" * 64
        parent_bundle = f"{arm}-parent.pt"
        bundle_bytes = f"synthetic parent bundle for {arm}".encode()
        parent_receipt = {
            "bundle": parent_bundle,
            "sha256": _sha(bundle_bytes),
            "digest": f"{receipt_index + 3}" * 64,
        }
        receipt_name = parent_bundle.replace(".pt", ".json")
        receipt_bytes = _json_bytes(parent_receipt)
        parent_latest = {"receipt": receipt_name, "receipt_sha256": _sha(receipt_bytes)}
        attempt = {
            "schema": "taiji-r2-dialogue-continuation-extension-attempt-v1",
            "arm": arm,
            "status": "prepared",
            "parent": {"attempt_sha256": parent_attempt_sha},
            "parent_checkpoint": {"receipt_sha256": _sha(receipt_bytes)},
            "continuation_sources": {
                "runner_sha256": runner.V1_RUNNER_SHA256,
                "files": {
                    name: {"sha256": digest} for name, digest in runner.V1_SOURCE_HASHES.items()
                },
                "report_sha256": source_report_sha,
            },
            "source_compatibility_report_sha256": source_report_sha,
            "source_compatibility": old_compat,
        }
        attempt_path = tmp_path / f"{arm}-attempt.json"
        attempt_path.write_bytes(_json_bytes(attempt))
        progress_path = tmp_path / f"{arm}-progress.jsonl"
        progress_path.write_bytes(
            (json.dumps({"event": "continuation_prepared", "arm": arm}) + "\n").encode()
        )
        root = tmp_path / arm / "checkpoint"
        for directory in (root, root / "pretraining", root / "assistant_sft"):
            directory.mkdir(parents=True, exist_ok=True)
        (root / "pretraining" / "latest.json").write_bytes(_json_bytes(parent_latest))
        (root / "pretraining" / receipt_name).write_bytes(receipt_bytes)
        (root / "pretraining" / parent_bundle).write_bytes(bundle_bytes)
        locations = {
            "extension_attempt": attempt_path,
            "extension_progress": progress_path,
            "extension_result": tmp_path / f"{arm}-result.json",
            "extension_checkpoint_root": root,
            "parent_readiness": tmp_path / f"{arm}-readiness.json",
            "parent_checkpoint_root": tmp_path / f"{arm}-parent-root",
        }
        locations["parent_readiness"].write_bytes(_json_bytes({"status": "ready"}))
        receipt_sha = _sha(receipt_bytes)
        context = {
            "arm": arm,
            "parent_attempt_sha256": parent_attempt_sha,
            "parent_receipt_sha256": receipt_sha,
            "parent_bundle_sha256": _sha(bundle_bytes),
            "parent_latest": parent_latest,
            "parent_bindings": {"source_hashes": {"synthetic": "e" * 64}},
            "source_compatibility": {
                **old_compat,
                "runner_sha256": expected_current["runner_sha256"],
            },
            "parent_receipt": parent_receipt,
            "locations": locations,
        }
        contexts[arm] = context
        monkeypatch.setitem(runner.PARENT_RECEIPT_SHA256, arm, receipt_sha)
    monkeypatch.setattr(runner, "ARCHITECTURE_REPORT", source_report)
    monkeypatch.setattr(runner, "SOURCE_REVISION_REPORT", review_report)
    monkeypatch.setattr(runner, "_continuation_source_binding", lambda _sha: expected_current)
    monkeypatch.setattr(runner, "_source_revision_path", lambda: tmp_path / "source-revision.json")
    monkeypatch.setattr(
        runner,
        "_expected_parent_paths",
        lambda arm: {
            "contract_path": contract_path,
            "extension_attempt": contexts[arm]["locations"]["extension_attempt"],
            "parent_readiness": contexts[arm]["locations"]["parent_readiness"],
        },
    )
    monkeypatch.setattr(runner, "_configure_prepared_storage", lambda: {})
    monkeypatch.setattr(runner, "latest_reservations", lambda: {})
    default_charges = {arm: 1_317_000.0 + index for index, arm in enumerate(runner.ARMS)}
    monkeypatch.setattr(runner, "latest_charges", lambda: dict(default_charges))

    @contextmanager
    def mock_source_revision_budget_snapshot():
        yield runner.latest_charges()

    monkeypatch.setattr(
        runner, "_source_revision_budget_snapshot", mock_source_revision_budget_snapshot
    )
    monkeypatch.setattr(runner, "_parent_context", lambda arm: contexts[arm])
    monkeypatch.setattr(runner, "_parent_copy_state", lambda _context: ({}, []))
    monkeypatch.setattr(runner, "_write_json_once", _write_once)
    return {
        "contexts": contexts,
        "contract_path": contract_path,
        "source_report_sha": source_report_sha,
    }


def test_prepare_source_revision_is_append_only_and_does_not_reset_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _prepared_arms_fixture(tmp_path, monkeypatch)
    attempt_bytes = {
        arm: ctx["locations"]["extension_attempt"].read_bytes()
        for arm, ctx in fixture["contexts"].items()
    }
    progress_bytes = {
        arm: ctx["locations"]["extension_progress"].read_bytes()
        for arm, ctx in fixture["contexts"].items()
    }
    ledger_state = {arm: 1_317_000.0 + index for index, arm in enumerate(runner.ARMS)}
    monkeypatch.setattr(runner, "latest_charges", lambda: dict(ledger_state))
    monkeypatch.setattr(
        runner,
        "initialize_ledger",
        lambda *_args, **_kwargs: pytest.fail("must not reinitialize ledger"),
    )
    monkeypatch.setattr(
        runner,
        "append_charge",
        lambda *_args, **_kwargs: pytest.fail("must not reset or alter charges"),
    )

    prepared = runner.prepare_source_revision()
    revision = runner.read_json(runner._source_revision_path())

    assert revision["schema"] == "taiji-r2-continuation-source-revision-v1.1"
    assert revision["approved_budget"] == {
        "per_arm": runner.PER_ARM_CPU_LIMIT,
        "aggregate": runner.AGGREGATE_CPU_LIMIT,
    }
    assert set(prepared["parents"]) == set(runner.ARMS)
    assert runner.latest_charges() == ledger_state
    for arm, context in fixture["contexts"].items():
        assert context["locations"]["extension_attempt"].read_bytes() == attempt_bytes[arm]
        assert context["locations"]["extension_progress"].read_bytes() == progress_bytes[arm]
        assert revision["original_attempts"][arm]["prepared_progress_sha256"] == _sha(
            progress_bytes[arm]
        )


@pytest.mark.parametrize(
    "charges, message",
    [
        ({"warm_start": 1_317_000.0}, "does not contain all three prepared arms"),
        (
            {
                "full_turn_1m": 0.0,
                "warm_start": runner.PER_ARM_CPU_LIMIT + 1.0,
                "fresh_same_capacity": 0.0,
            },
            "per-arm CPU ledger exceeds",
        ),
        (
            {
                "full_turn_1m": runner.AGGREGATE_CPU_LIMIT / 3 + 1.0,
                "warm_start": runner.AGGREGATE_CPU_LIMIT / 3 + 1.0,
                "fresh_same_capacity": runner.AGGREGATE_CPU_LIMIT / 3 + 1.0,
            },
            "aggregate CPU ledger exceeds",
        ),
    ],
)
def test_prepare_source_revision_rejects_invalid_ledger_before_publishing(
    tmp_path: Path, monkeypatch, charges: dict[str, float], message: str
) -> None:
    _prepared_arms_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "latest_charges", lambda: dict(charges))

    with pytest.raises(runner.Refuse, match=message):
        runner.prepare_source_revision()

    assert not runner._source_revision_path().exists()


def test_prepare_source_revision_rejects_invalid_candidate_checkpoint_before_publishing(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _prepared_arms_fixture(tmp_path, monkeypatch)
    copied_latest = (
        fixture["contexts"]["warm_start"]["locations"]["extension_checkpoint_root"]
        / "pretraining"
        / "latest.json"
    )
    copied_latest.write_bytes(_json_bytes({"receipt": "drifted.json"}))

    with pytest.raises(runner.Refuse, match="parent receipt fork changed"):
        runner.prepare_source_revision()

    assert not runner._source_revision_path().exists()


def _mock_source_revision_ledger_history(
    tmp_path: Path, monkeypatch, event_names: list[str]
) -> tuple[dict[str, Any], dict[str, float]]:
    charges = {arm: 1_317_000.0 + index for index, arm in enumerate(runner.ARMS)}
    event_files = []
    for sequence, event_name in enumerate(event_names):
        event_path = tmp_path / f"ledger.{sequence:08d}.json"
        event_path.write_bytes(_json_bytes({"event": event_name, "sequence": sequence}))
        event_files.append((sequence, event_path))

    lock_state: dict[str, Any] = {"held": False, "ledger_reads": 0, "event_reads": 0}

    @contextmanager
    def mock_aggregate_lock(path: Path):
        assert path == runner.continuation_budget.LOCK_PATH
        assert not lock_state["held"]
        lock_state["held"] = True
        try:
            yield
        finally:
            lock_state["held"] = False

    def mock_read_ledger_full():
        assert lock_state["held"], "ledger must be replayed under aggregate lock"
        lock_state["ledger_reads"] += 1
        return {}, dict(charges), {}, "ledger-event-chain-sha"

    def mock_event_files(prefix: Path):
        assert lock_state["held"], "ledger events must be checked under aggregate lock"
        assert prefix == runner.continuation_budget.AGGREGATE_LEDGER_PREFIX
        lock_state["event_reads"] += 1
        return list(event_files)

    monkeypatch.setattr(runner.continuation_budget, "_aggregate_lock", mock_aggregate_lock)
    monkeypatch.setattr(runner.continuation_budget, "_read_ledger_full", mock_read_ledger_full)
    monkeypatch.setattr(runner.continuation_budget, "_event_files", mock_event_files)
    return lock_state, charges


@pytest.mark.parametrize(
    "settled_runtime_event",
    ["continuation_started", "continuation_stopped"],
)
def test_source_revision_budget_gate_rejects_settled_runtime_history(
    tmp_path: Path, monkeypatch, settled_runtime_event: str
) -> None:
    lock_state, _charges = _mock_source_revision_ledger_history(
        tmp_path,
        monkeypatch,
        ["extension_budget_initialized", "work_unit_reserved", settled_runtime_event],
    )

    with (
        pytest.raises(runner.Refuse, match="ledger contains execution history"),
        runner._source_revision_budget_snapshot(),
    ):
        pytest.fail("runtime history must be rejected before yielding a budget snapshot")

    assert not lock_state["held"]
    assert lock_state["ledger_reads"] == 1
    assert lock_state["event_reads"] == 1


def test_source_revision_budget_gate_allows_settled_prepare_only_history(
    tmp_path: Path, monkeypatch
) -> None:
    lock_state, charges = _mock_source_revision_ledger_history(
        tmp_path,
        monkeypatch,
        [
            "extension_budget_initialized",
            "work_unit_reserved",
            "continuation_prepare_tail_reconciled",
            "work_unit_reserved",
            "continuation_prepared",
        ],
    )

    with runner._source_revision_budget_snapshot() as snapshot:
        assert snapshot == charges
        assert lock_state["held"]

    assert not lock_state["held"]
    assert lock_state["ledger_reads"] == 1
    assert lock_state["event_reads"] == 1


@pytest.mark.parametrize(
    "blocked_state",
    [
        "pending",
        "start_event",
        "restore_event",
        "training_event",
        "root_lock",
        "stage_lock",
        "terminal_result",
        "terminal_readiness",
        "incomplete_parent_fork",
    ],
)
def test_prepare_source_revision_refuses_any_nonprepared_state(
    tmp_path: Path, monkeypatch, blocked_state: str
) -> None:
    fixture = _prepared_arms_fixture(tmp_path, monkeypatch)
    target = fixture["contexts"]["warm_start"]["locations"]
    if blocked_state == "pending":
        monkeypatch.setattr(
            runner, "latest_reservations", lambda: {"warm_start": {"reservation_id": "a" * 32}}
        )
    elif blocked_state in {"start_event", "restore_event", "training_event"}:
        event = {
            "start_event": "continuation_started",
            "restore_event": "parent_checkpoint_restored",
            "training_event": "durable_training_commit",
        }[blocked_state]
        with target["extension_progress"].open("ab") as stream:
            stream.write((json.dumps({"event": event}) + "\n").encode())
    elif blocked_state == "root_lock":
        (target["extension_checkpoint_root"] / "writer.lock").write_bytes(_json_bytes({"pid": 1}))
    elif blocked_state == "stage_lock":
        (target["extension_checkpoint_root"] / "pretraining" / "writer.lock").write_bytes(
            _json_bytes({"pid": 1})
        )
    elif blocked_state == "terminal_result":
        target["extension_result"].write_bytes(_json_bytes({"status": "complete"}))
    elif blocked_state == "terminal_readiness":
        readiness = runner._readiness_path(target["extension_result"])
        readiness.write_bytes(_json_bytes({"status": "ready"}))
    elif blocked_state == "incomplete_parent_fork":
        monkeypatch.setattr(runner, "_parent_copy_state", lambda _context: ({}, ["parent.pt"]))
    monkeypatch.setattr(
        runner,
        "_write_json_once",
        lambda *_args, **_kwargs: pytest.fail("must not publish amendment"),
    )

    with pytest.raises(runner.Refuse):
        runner.prepare_source_revision()

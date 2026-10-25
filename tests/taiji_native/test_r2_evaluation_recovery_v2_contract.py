"""Recovery v2: admission is separate from one durable formal face access."""

import json
import os
import sys
from pathlib import Path

import pytest

from scripts.training import evaluate_taiji_r2_evaluation_recovery_v2 as entry
from scripts.training import taiji_r2_evaluation_recovery_v2 as provider


class FakeDurability:
    def flush_file(self, fd: int) -> None:
        os.fsync(fd)


def _hex(char: str) -> str:
    return char * 64


OLD_EVIDENCE = {
    "cc1": {
        "attempt_id": "cc1-failed",
        "raw_sha256": {key: _hex("1") for key in ("attempt", "failure", "result")},
        "ledger_sha256": _hex("2"),
        "stopped_before_selection_in_this_attempt": True,
    },
    "v1": {
        "attempt_id": "v1-failed",
        "raw_sha256": {key: _hex("3") for key in ("attempt", "failure", "result")},
        "ledger_sha256": _hex("4"),
        "stopped_before_selection_in_this_attempt": True,
    },
}

ENDPOINT_SHA = {
    name: _hex(char) for name, char in zip(provider.frozen.ENDPOINT_ORDER, "abcd", strict=True)
}


def _contract_payload() -> dict:
    return {
        "recovery_id": provider.RECOVERY_ID,
        "status": "adopted",
        "adopts_consumption_boundary_v2": True,
        "authorizes_engineering_admission": True,
        "authorizes_one_formal_face_consumption": True,
        "preserves_v1_failure": True,
        "preserves_cc1_failure": True,
        "same_dev_face_reuse_approved": True,
        "historical_tail_reserve_accepted": True,
        "cpu_limit_seconds": provider.ADMISSION_CPU_LIMIT_SECONDS,
        "wall_limit_seconds": provider.ADMISSION_WALL_LIMIT_SECONDS,
        "formal_cpu_limit_seconds": provider.FORMAL_CPU_LIMIT_SECONDS,
        "formal_wall_limit_seconds": provider.FORMAL_WALL_LIMIT_SECONDS,
        "historical_cpu_reserve_seconds": provider.HISTORICAL_CPU_RESERVE_SECONDS,
        "historical_wall_reserve_seconds": provider.HISTORICAL_WALL_RESERVE_SECONDS,
        "generation_outputs": provider.GENERATION_OUTPUTS,
        "review_answers": provider.REVIEW_ANSWERS,
        "endpoint_checkpoint_sha256": dict(ENDPOINT_SHA),
        "old_cc1": dict(OLD_EVIDENCE["cc1"]),
        "old_v1": dict(OLD_EVIDENCE["v1"]),
        "approved_source_sha256": {
            name: provider._sha(provider.ROOT / name) for name in provider.REQUIRED_SOURCES
        },
        "can_promote": False,
    }


def _write_contract(tmp_path: Path, **overrides: object) -> Path:
    payload = _contract_payload()
    payload.update(overrides)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _extension_preflight() -> dict:
    return {
        "endpoints": [
            {"endpoint_id": name, "sha256": ENDPOINT_SHA[name]}
            for name in provider.frozen.ENDPOINT_ORDER
        ],
        "source_hashes": {},
        "report_hashes": {},
    }


def _context(
    tmp_path: Path,
    *,
    kind: str = "formal",
    sequence: int = 1,
    extension_preflight=None,
    old_evidence=None,
) -> provider.Context:
    contract = _write_contract(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    return provider.Context(
        kind,
        sequence,
        reports_dir=reports,
        prefix_base="test_recovery_v2",
        contract_path=contract,
        marker_path=reports / "face_access.json",
        ledger_path=reports / "window.jsonl",
        extension_preflight_fn=extension_preflight or _extension_preflight,
        old_evidence_fn=old_evidence or (lambda: json.loads(json.dumps(OLD_EVIDENCE))),
        durability_api=FakeDurability(),
    )


def _write_attempt(paths: dict[str, Path], attempt_id: str = "formal-attempt") -> dict:
    attempt = {
        "schema": provider.ATTEMPT_SCHEMA,
        "status": "started",
        "attempt_id": attempt_id,
        "readiness": {"ready": True},
        "source_hashes": {"source": "digest"},
        "endpoint_checkpoint_sha256": dict(ENDPOINT_SHA),
    }
    paths["attempt"].parent.mkdir(parents=True, exist_ok=True)
    paths["attempt"].write_text(json.dumps(attempt), encoding="utf-8")
    return attempt


def _write_marker(context: provider.Context, attempt_id: str = "formal-attempt") -> None:
    payload = {
        "schema": provider.MARKER_SCHEMA,
        "recovery_id": provider.RECOVERY_ID,
        "phase": "formal",
        "sequence": context.sequence,
        "attempt_id": attempt_id,
        "attempt_sha256": _hex("5"),
        "readiness_sha256": _hex("6"),
        "endpoint_checkpoint_sha256": dict(ENDPOINT_SHA),
        "source_hashes_sha256": _hex("7"),
        "contract_sha256": provider._sha(context.contract_path),
        "window_ledger_sha256": None,
        "selector_pid": 1,
        "published_at_utc": "2026-10-25T00:00:00+00:00",
        "can_promote": False,
    }
    context.marker_path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "flag",
    [
        "adopts_consumption_boundary_v2",
        "authorizes_engineering_admission",
        "authorizes_one_formal_face_consumption",
        "preserves_v1_failure",
        "preserves_cc1_failure",
    ],
)
def test_each_unadopted_protocol_field_refuses(flag: str) -> None:
    value = _contract_payload()
    value[flag] = False
    with pytest.raises(ValueError, match="explicit protocol decision"):
        provider.require_authorization(value)


@pytest.mark.parametrize(
    "field",
    [
        "cpu_limit_seconds",
        "wall_limit_seconds",
        "formal_cpu_limit_seconds",
        "formal_wall_limit_seconds",
        "historical_cpu_reserve_seconds",
        "generation_outputs",
    ],
)
def test_limits_and_denominators_cannot_drift(field: str) -> None:
    value = _contract_payload()
    value[field] = value[field] + 1
    with pytest.raises(ValueError):
        provider.require_authorization(value)


def test_window_allocation_leaves_a_positive_margin() -> None:
    cpu = (
        provider.HISTORICAL_CPU_RESERVE_SECONDS
        + provider.V1_MEASURED_CPU_SECONDS
        + provider.ADMISSION_CPU_LIMIT_SECONDS
        + provider.FORMAL_CPU_LIMIT_SECONDS
    )
    wall = (
        provider.HISTORICAL_WALL_RESERVE_SECONDS
        + provider.V1_MEASURED_WALL_SECONDS
        + provider.ADMISSION_WALL_LIMIT_SECONDS
        + provider.FORMAL_WALL_LIMIT_SECONDS
    )
    assert cpu <= provider.WINDOW_CPU_LIMIT_SECONDS
    assert wall <= provider.WINDOW_WALL_LIMIT_SECONDS
    assert provider.WINDOW_CPU_LIMIT_SECONDS - cpu == pytest.approx(85.515625)


def test_empty_source_snapshot_cannot_authorize() -> None:
    value = _contract_payload()
    value["approved_source_sha256"] = {}
    with pytest.raises(ValueError, match="snapshot is incomplete"):
        provider.require_authorization(value)


def test_contract_is_checked_before_endpoint_metadata(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        extension_preflight=lambda: pytest.fail("endpoint metadata opened before adoption"),
    )
    _write_contract(tmp_path, authorizes_engineering_admission=False)
    with pytest.raises(ValueError, match="authorizes_engineering_admission"):
        provider.preflight(context)


def test_old_evidence_drift_refuses(tmp_path: Path) -> None:
    drifted = json.loads(json.dumps(OLD_EVIDENCE))
    drifted["v1"]["ledger_sha256"] = _hex("f")
    context = _context(tmp_path, old_evidence=lambda: drifted)
    with pytest.raises(ValueError, match="old_v1 ledger changed"):
        provider.preflight(context)


def test_marker_of_another_attempt_blocks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.marker_path.write_text(
        json.dumps(
            {
                "schema": provider.MARKER_SCHEMA,
                "attempt_id": "previous-attempt",
                "endpoint_checkpoint_sha256": dict(ENDPOINT_SHA),
                "source_hashes_sha256": _hex("e"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(provider.frozen.Refuse, match="already consumed"):
        provider.preflight(context)


def test_unreadable_marker_is_treated_as_consumed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.marker_path.write_text("{half-written", encoding="utf-8")
    with pytest.raises(provider.frozen.Refuse, match="already consumed"):
        provider.preflight(context)


def test_marker_bound_to_live_attempt_allows_final_recheck(tmp_path: Path) -> None:
    context = _context(tmp_path)
    paths = context.paths()
    _write_attempt(paths)
    _write_marker(context)
    assert provider.preflight(context)["source_hashes"]


def test_selector_publishes_marker_before_reading_the_face(tmp_path: Path) -> None:
    context = _context(tmp_path)
    paths = context.paths()
    _write_attempt(paths)
    calls: list[str] = []

    def publish(payload: dict) -> str:
        calls.append("publish")
        context.marker_path.write_text(json.dumps(payload), encoding="utf-8")
        return provider.canonical_sha(payload)

    def selector() -> tuple[str, ...]:
        assert provider.read_marker(context.marker_path) is not None
        calls.append("select")
        return ("prompt",)

    assert context.prompt_selector(publish_fn=publish, selector_fn=selector)() == ("prompt",)
    assert calls == ["publish", "select"]
    events = provider.read_window_events(context.ledger_path)
    assert [event["event"] for event in events] == ["face_access_marker_published"]


def test_selector_refuses_second_access(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_attempt(context.paths())
    context.marker_path.write_text(
        json.dumps(
            {
                "schema": provider.MARKER_SCHEMA,
                "attempt_id": "formal-attempt",
                "endpoint_checkpoint_sha256": dict(ENDPOINT_SHA),
                "source_hashes_sha256": _hex("e"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(provider.frozen.Refuse, match="already consumed"):
        context.prompt_selector(selector_fn=lambda: pytest.fail("face read twice"))()


def test_publication_failure_is_a_durable_claim(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_attempt(context.paths())

    def broken(payload: dict) -> str:
        raise OSError("durable publish failed")

    with pytest.raises(OSError):
        context.prompt_selector(publish_fn=broken, selector_fn=lambda: "face")()
    events = provider.read_window_events(context.ledger_path)
    assert events[-1]["event"] == "face_access_marker_publication_failed"
    assert provider.read_marker(context.marker_path) is None
    with pytest.raises(provider.frozen.Refuse, match="already consumed"):
        provider.preflight(context)


def test_insufficient_window_balance_blocks_admission(tmp_path: Path) -> None:
    context = _context(tmp_path, kind="admission")
    provider.append_window_event(
        context,
        {
            "event": "admission_completed",
            "phase": "admission",
            "sequence": 1,
            "cpu_seconds_measured": 10_900.0,
            "wall_seconds_measured": 100.0,
            "unknown_tail": False,
        },
    )
    with pytest.raises(ValueError, match="cannot cover the next phase"):
        provider.preflight(context)


def test_window_ledger_accumulates_across_namespaces(tmp_path: Path) -> None:
    context = _context(tmp_path)
    provider.append_window_event(
        context,
        {
            "event": "formal_failed",
            "phase": "formal",
            "sequence": 1,
            "cpu_seconds_measured": 250.0,
            "wall_seconds_measured": 40.0,
            "unknown_tail": True,
        },
    )
    summary = provider.window_summary(context.ledger_path)
    assert summary["v2_measured_cpu_seconds"] == 250.0
    assert summary["unknown_tail_events"] == 1
    assert provider.next_sequence("formal", context.ledger_path) == 2
    assert summary["remaining_cpu_seconds"] == pytest.approx(
        provider.WINDOW_CPU_LIMIT_SECONDS
        - provider.HISTORICAL_CPU_RESERVE_SECONDS
        - provider.V1_MEASURED_CPU_SECONDS
        - 250.0
    )


def test_prepared_namespace_is_reused_until_an_attempt_exists(tmp_path: Path) -> None:
    context = _context(tmp_path)
    paths = context.paths()
    paths["readiness"].write_text("{}", encoding="utf-8")
    provider.append_window_event(
        context, {"event": "formal_prepared", "phase": "formal", "sequence": 1}
    )
    assert (
        provider.prepared_formal_sequence(
            reports_dir=context.reports_dir,
            prefix_base=context.prefix_base,
            ledger_path=context.ledger_path,
        )
        == 1
    )
    paths["attempt"].write_text("{}", encoding="utf-8")
    assert (
        provider.prepared_formal_sequence(
            reports_dir=context.reports_dir,
            prefix_base=context.prefix_base,
            ledger_path=context.ledger_path,
        )
        is None
    )


def test_generation_gate_requires_the_formal_marker(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(provider.frozen.Refuse, match="before the formal face marker"):
        context.require_face_access()
    paths = context.paths()
    _write_attempt(paths)
    _write_marker(context)
    paths["face"].write_text("{}", encoding="utf-8")
    context.require_face_access()


def test_generation_gate_rejects_a_marker_from_another_attempt(tmp_path: Path) -> None:
    context = _context(tmp_path)
    paths = context.paths()
    _write_attempt(paths)
    _write_marker(context, attempt_id="another-attempt")
    paths["face"].write_text("{}", encoding="utf-8")
    with pytest.raises(provider.frozen.Refuse, match="another attempt"):
        context.require_face_access()


def test_worker_command_carries_the_namespace(tmp_path: Path) -> None:
    context = _context(tmp_path, sequence=2)
    command = entry.worker_command(Path("attempt.json"), Path("ledger.jsonl"), context)
    assert command[command.index("--namespace-kind") + 1] == "formal"
    assert command[command.index("--namespace-sequence") + 1] == "2"
    assert command[command.index("-m") + 1] == entry.MODULE


def test_private_worker_uses_provider_hooks(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    def loop(*args, **kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(entry.worker, "worker_loop", loop)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recovery-v2",
            "--worker",
            "--attempt-path",
            str(tmp_path / "attempt.json"),
            "--ledger-path",
            str(tmp_path / "ledger.jsonl"),
            "--job-name",
            "synthetic-job",
            "--namespace-kind",
            "formal",
            "--namespace-sequence",
            "1",
        ],
    )
    assert entry.main() == 0
    assert callable(observed["preflight_fn"])
    assert callable(observed["evaluation_paths_fn"])
    assert callable(observed["generation_gate"])


def test_namespaces_never_reuse_the_old_prefixes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    paths = context.paths()
    assert "recovery_v2_formal_001" in paths["attempt"].name
    assert paths["ledger"].suffix == ".jsonl"
    assert paths["progress"].suffix == ".jsonl"
    assert set(paths) >= {"readiness", "face", "generations", "review", "cost_receipt"}

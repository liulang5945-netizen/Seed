"""Recovery preparation must distinguish zero queries from exposure permission."""

import json
from pathlib import Path

import pytest

from scripts.training import check_taiji_r2_evaluation_recovery as preparation


def failed_attempt(tmp_path: Path) -> Path:
    prefix = tmp_path / "failed"
    attempt = {"attempt_id": "original-failed-attempt", "source_hashes": {}}
    failure = {
        "attempt_id": attempt["attempt_id"],
        "status": "failed",
        "query_attempts": 0,
        "persisted_generation_outputs": 0,
        "fresh_process_restores": {},
        "cpu_seconds_total_at_failure": 60.78125,
        "wall_seconds_at_failure": 63.953,
    }
    for suffix, value in (("attempt", attempt), ("failure", failure), ("result", failure)):
        prefix.with_name(f"failed_{suffix}.json").write_text(json.dumps(value), encoding="utf-8")
    events = [
        {"event": "attempt_started"},
        {"event": "worker_startup_started"},
        {"event": "evaluation_stopped", "unknown_tail": False},
        {
            "event": "worker_startup_tail_audited",
            "unknown_tail": True,
            "charged_child_cpu_upper": None,
            "boot_identity_before": 1000.0,
            "boot_identity_at_audit": 1000.0000000000001,
        },
    ]
    prefix.with_name("failed_ledger.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    return prefix


def test_audit_preserves_later_unknown_tail_and_original_bytes(tmp_path: Path) -> None:
    prefix = failed_attempt(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    report = preparation.audit_failed_attempt(prefix)
    assert report["stopped_before_selection_in_this_attempt"] is True
    assert report["latest_startup_tail_unknown"] is True
    assert report["latest_startup_tail_upper"] is None
    assert report["no_retry_authorized"] is True
    assert "not a machine-wide access audit" in report["exposure_scope"]
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


@pytest.mark.parametrize(
    "event", ["prompt_selection_started", "prompt_face_persisted", "query_started"]
)
def test_zero_outputs_do_not_override_selector_or_query_event(tmp_path: Path, event: str) -> None:
    prefix = failed_attempt(tmp_path)
    with prefix.with_name("failed_ledger.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event}) + "\n")
    assert (
        preparation.audit_failed_attempt(prefix)["stopped_before_selection_in_this_attempt"]
        is False
    )


def test_existing_face_is_exposure_signal_even_with_zero_queries(tmp_path: Path) -> None:
    prefix = failed_attempt(tmp_path)
    prefix.with_name("failed_face.json").write_text("{}", encoding="utf-8")
    assert (
        preparation.audit_failed_attempt(prefix)["stopped_before_selection_in_this_attempt"]
        is False
    )


def test_mixed_attempt_identity_is_rejected(tmp_path: Path) -> None:
    prefix = failed_attempt(tmp_path)
    path = prefix.with_name("failed_failure.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["attempt_id"] = "another-attempt"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="identities differ"):
        preparation.audit_failed_attempt(prefix)


def test_checkpoint_sources_are_not_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = failed_attempt(tmp_path)
    path = prefix.with_name("failed_attempt.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source_hashes"][str(preparation.ROOT / "taiji_data" / "not-opened.pt")] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    original = preparation.sha256

    def source_hash(source: Path) -> str:
        assert source.suffix != ".pt", "preparation must not open a checkpoint"
        return original(source)

    monkeypatch.setattr(preparation, "sha256", source_hash)
    assert preparation.audit_failed_attempt(prefix)["source_mismatches_after_fix"] == []

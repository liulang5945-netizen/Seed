"""Synthetic contract for the continuation-extension evaluation adapter (CC-1.2).

These tests never read a prompt, model or checkpoint: they prove the metadata
seam only.  The adapter must map extension endpoints onto the frozen evaluator's
four-endpoint envelope, reject a forged `restart_from_zero`, and keep the
versioned worker entry separate from the frozen recovery-v2 entry.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.training import evaluate_taiji_r2_daily_dialogue_continuation_extension_v1 as entry
from scripts.training import (
    evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1 as worker,
)
from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as recovery
from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter

REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/training/evaluate_taiji_r2_daily_dialogue_continuation_extension_v1.py"


# 2026-10-25 owner 裁定：data/ 下 checkpoints/语料不入 git——产物在位才跑，
# CI 上跳过（本机仍全量守护）。准入出处：roadmap/12 批 C。
_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl"
)
_ARTIFACT_PRESENT = _ARTIFACT.is_file()
_ARTIFACT_REASON = (
    "consumes data/ artifacts that are not git-stored "
    "(owner 2026-10-25: checkpoints/corpus stay out of git)"
)


def _sha(character: str) -> str:
    return character * 64


def _stage(path: Path, name: str, episodes: int, updates: int) -> dict:
    return {
        "path": str(path),
        "sha256": _sha("a"),
        "checkpoint_digest": _sha("b"),
        "episodes": episodes,
        "target_byte_updates": updates,
        "restore": {
            "complete": True,
            "equal": True,
            "stage": name.removesuffix("_final"),
            "checkpoint_digest": _sha("b"),
        },
    }


def _manifest(tmp_path: Path) -> dict:
    launch_source = tmp_path / "launch_manifest.json"
    launch_source.write_text("{}\n", encoding="utf-8")
    historical_source = tmp_path / "historical_launch.json"
    historical_source.write_text("{}\n", encoding="utf-8")
    records = {}
    for arm in adapter.ARMS:
        records[arm] = {
            "arm": arm,
            "endpoint_kind": "continuation_extension",
            "can_promote": False,
            "final_checkpoints": {
                adapter.PRETRAIN_STAGE: _stage(
                    tmp_path / f"{arm}_pretraining_final.pt",
                    adapter.PRETRAIN_STAGE,
                    30096,
                    14517912,
                ),
                adapter.FINAL_STAGE: _stage(
                    tmp_path / f"{arm}_assistant_sft_final.pt",
                    adapter.FINAL_STAGE,
                    525,
                    113209,
                ),
            },
        }
    launches = {
        arm: [
            {
                "path": f"launch-{arm}.json",
                "sha256": _sha("1"),
                "start_receipt_sha256": _sha("2"),
                "start_bundle_sha256": _sha("3"),
            }
        ]
        for arm in adapter.ARMS
    }
    return {
        "schema": adapter.MANIFEST_SCHEMA,
        "endpoints": records,
        "source_revision_sha256": _sha("c"),
        "launch_provenance": {
            "launches": launches,
            "verified_sources": {str(launch_source): _hash_file(launch_source)},
            "historical_sources": {str(historical_source): _hash_file(historical_source)},
        },
        "verified_ledger_tip_sha256": _sha("7"),
        "ledger_charges": {arm: 1_400_000.0 for arm in adapter.ARMS},
        "approved_budget": {"per_arm": 1_550_000.0, "aggregate": 4_400_000.0},
    }


def _baseline(tmp_path: Path) -> dict:
    return {
        "endpoint_id": "parent_baseline",
        "path": tmp_path / "parent_baseline.pt",
        "sha256": _sha("e"),
        "checkpoint_digest": _sha("f"),
    }


def _hash_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arm_paths(tmp_path: Path):
    def build(arm: str) -> dict[str, Path]:
        result = tmp_path / f"{arm}_result.json"
        readiness = tmp_path / f"{arm}_result_readiness.json"
        attempt = tmp_path / f"{arm}_attempt.json"
        for path in (result, readiness, attempt):
            path.write_text("{}\n", encoding="utf-8")
        return {
            "extension_result": result,
            "extension_attempt": attempt,
        }

    return build


def _preflight(tmp_path: Path, manifest: dict) -> dict:
    return adapter.extension_preflight(
        manifest,
        arm_paths=_arm_paths(tmp_path),
        hash_file=_hash_file,
        baseline_fn=lambda: _baseline(tmp_path),
    )


def test_extension_map_uses_assistant_sft_final_in_frozen_order(tmp_path: Path) -> None:
    rows, baseline, _ = adapter.extension_endpoint_map(
        _manifest(tmp_path), baseline_fn=lambda: _baseline(tmp_path)
    )
    assert [row["endpoint_id"] for row in rows] == list(adapter.ENDPOINT_ORDER)
    assert baseline["endpoint_id"] == "parent_baseline"
    for arm in adapter.ARMS:
        row = next(item for item in rows if item["endpoint_id"] == arm)
        assert row["path"].name == f"{arm}_assistant_sft_final.pt"
        assert row["source"] == adapter.FINAL_STAGE


def test_extension_map_rejects_a_wrong_final_denominator(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["endpoints"][adapter.ARMS[0]]["final_checkpoints"][adapter.FINAL_STAGE][
        "episodes"
    ] = 524
    with pytest.raises(adapter.AdapterRefuse, match="denominator"):
        adapter.extension_endpoint_map(manifest, baseline_fn=lambda: _baseline(tmp_path))


def test_extension_map_rejects_a_forged_restart_from_zero(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["endpoints"][adapter.ARMS[1]]["restart_from_zero"] = True
    with pytest.raises(adapter.AdapterRefuse, match="restart_from_zero"):
        adapter.extension_endpoint_map(manifest, baseline_fn=lambda: _baseline(tmp_path))


def test_extension_map_requires_the_exact_three_arms_and_schema(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["schema"] = "something-else"
    with pytest.raises(adapter.AdapterRefuse, match="schema"):
        adapter.extension_endpoint_map(manifest, baseline_fn=lambda: _baseline(tmp_path))

    manifest = _manifest(tmp_path)
    del manifest["endpoints"][adapter.ARMS[2]]
    with pytest.raises(adapter.AdapterRefuse, match="three arms"):
        adapter.extension_endpoint_map(manifest, baseline_fn=lambda: _baseline(tmp_path))


def test_worker_command_is_versioned_not_the_frozen_evaluator() -> None:
    command = adapter.extension_worker_command(Path("attempt.json"), Path("ledger.jsonl"))
    assert adapter.WORKER_MODULE in command
    assert "{job_name}" in command
    assert "scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2" not in command


def test_worker_command_module_has_a_runnable_entry_point() -> None:
    """Regression: the worker library module has no ``__main__`` block.

    The one-shot W5b of 2026-10-25 died before the bootstrap-ready handshake
    because ``-m`` pointed at the library module, which exits silently.  The
    documented private entry is the versioned entry script with ``--worker``.
    """
    command = adapter.extension_worker_command(Path("attempt.json"), Path("ledger.jsonl"))
    module = command[command.index("-m") + 1]
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage" in completed.stdout.lower()


def test_preflight_keeps_the_frozen_readiness_envelope(tmp_path: Path) -> None:
    preflight = adapter.extension_preflight(
        _manifest(tmp_path),
        arm_paths=_arm_paths(tmp_path),
        hash_file=_hash_file,
        baseline_fn=lambda: _baseline(tmp_path),
        eval_hashes=lambda: {"extra_source": _sha("9")},
    )
    assert len(preflight["endpoints"]) == 4
    assert set(preflight["report_hashes"]) == set(adapter.ARMS)
    assert preflight["source_hashes"]["extra_source"] == _sha("9")
    assert preflight["endpoint_schema"] == adapter.ENDPOINT_SCHEMA

    readiness = adapter.extension_readiness(preflight)
    assert readiness == recovery._readiness_payload(preflight)
    assert readiness["expected_generation_outputs"] == 156
    assert readiness["input_families"] == 39
    manifest = adapter.endpoint_manifest_payload(preflight)
    assert manifest["schema"] == adapter.ENDPOINT_SCHEMA
    assert manifest["can_promote"] is False
    assert [row["endpoint_id"] for row in manifest["endpoints"]] == list(adapter.ENDPOINT_ORDER)


@pytest.mark.skipif(not _ARTIFACT_PRESENT, reason=_ARTIFACT_REASON)
def test_preflight_binds_frozen_and_extension_sources_by_default(tmp_path: Path) -> None:
    preflight = adapter.extension_preflight(
        _manifest(tmp_path),
        arm_paths=_arm_paths(tmp_path),
        hash_file=_hash_file,
        baseline_fn=lambda: _baseline(tmp_path),
    )
    bound = set(preflight["source_hashes"])
    expected = {
        str(Path(adapter.__file__).resolve()),
        str(Path(worker.__file__).resolve()),
        str(Path(entry.__file__).resolve()),
        str(Path(recovery.__file__).resolve()),
        str(Path(__file__).resolve()),
    }
    assert expected <= bound


def test_extension_map_rejects_a_restore_receipt_from_another_stage(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["endpoints"][adapter.ARMS[1]]["final_checkpoints"][adapter.FINAL_STAGE]["restore"][
        "stage"
    ] = "pretraining"
    with pytest.raises(adapter.AdapterRefuse, match="restore stage"):
        adapter.extension_endpoint_map(manifest, baseline_fn=lambda: _baseline(tmp_path))

    manifest = _manifest(tmp_path)
    manifest["endpoints"][adapter.ARMS[2]]["final_checkpoints"][adapter.PRETRAIN_STAGE]["restore"][
        "checkpoint_digest"
    ] = _sha("d")
    with pytest.raises(adapter.AdapterRefuse, match="restore digest"):
        adapter.extension_endpoint_map(manifest, baseline_fn=lambda: _baseline(tmp_path))


@pytest.mark.skipif(not _ARTIFACT_PRESENT, reason=_ARTIFACT_REASON)
def test_preflight_binds_launch_provenance_ledger_and_budget(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    preflight = _preflight(tmp_path, manifest)
    launch = preflight["launch"]
    assert launch["source_revision_sha256"] == _sha("c")
    assert launch["approved_budget"] == {"per_arm": 1_550_000.0, "aggregate": 4_400_000.0}
    assert set(launch["ledger_charges"]) == set(adapter.ARMS)
    source = next(iter(manifest["launch_provenance"]["verified_sources"]))
    assert str(Path(source).resolve()) in preflight["source_hashes"]

    payload = adapter.endpoint_manifest_payload(preflight)
    assert payload["launch_provenance_sha256"] == launch["launch_provenance_sha256"]
    assert payload["verified_ledger_tip_sha256"] == _sha("7")
    assert payload["approved_budget"] == launch["approved_budget"]
    assert [row["endpoint_kind"] for row in payload["endpoints"]] == [
        "parent_baseline",
        *("continuation_extension" for _ in adapter.ARMS),
    ]


def test_preflight_rejects_launch_source_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    source = Path(next(iter(manifest["launch_provenance"]["verified_sources"])))
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(adapter.AdapterRefuse, match="launch source bytes differ"):
        _preflight(tmp_path, manifest)


@pytest.mark.skipif(not _ARTIFACT_PRESENT, reason=_ARTIFACT_REASON)
def test_preflight_records_historical_sources_without_byte_binding(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    historical = Path(next(iter(manifest["launch_provenance"]["historical_sources"])))
    preflight = _preflight(tmp_path, manifest)
    resolved = str(historical.resolve())
    assert resolved in preflight["launch"]["historical_sources"]
    assert resolved not in preflight["source_hashes"]
    assert resolved in adapter.endpoint_manifest_payload(preflight)["historical_sources"]


def test_preflight_rejects_missing_launch_arm_or_budget_overrun(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    del manifest["launch_provenance"]["launches"][adapter.ARMS[0]]
    with pytest.raises(adapter.AdapterRefuse, match="exactly the three arms"):
        _preflight(tmp_path, manifest)

    manifest = _manifest(tmp_path)
    manifest["ledger_charges"][adapter.ARMS[0]] = 1_600_000.0
    with pytest.raises(adapter.AdapterRefuse, match="exceed the approved launch budget"):
        _preflight(tmp_path, manifest)


def test_entry_refuses_execute_and_worker_with_stdout_json() -> None:
    for flag in ("--execute", "--worker"):
        proc = subprocess.run([sys.executable, str(ENTRY), flag], capture_output=True, check=False)
        assert proc.returncode == 2, (flag, proc.returncode, proc.stdout, proc.stderr)
        assert proc.stderr.decode("utf-8", errors="replace") == ""
        payload = json.loads(proc.stdout.decode("utf-8"))
        assert payload["status"] == "refused"
        assert payload["can_promote"] is False


def test_recovery_consumption_guard_blocks_a_second_face(tmp_path: Path) -> None:
    old_attempt = tmp_path / "old_attempt.json"
    old_attempt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(adapter.AdapterRefuse, match="already consumed"):
        adapter.refuse_if_recovery_evaluation_consumed(paths_fn=lambda: {"attempt": old_attempt})


def test_readiness_alone_does_not_block_the_extension(tmp_path: Path) -> None:
    """`--validate-only` preparation is not consumption; only run signals block."""
    readiness = tmp_path / "old_readiness.json"
    readiness.write_text("{}\n", encoding="utf-8")
    adapter.refuse_if_recovery_evaluation_consumed(paths_fn=lambda: {"readiness": readiness})


def _request(operation: str) -> dict[str, str]:
    return {
        "operation": operation,
        "endpoint_id": "parent_baseline",
        "task_id": "task-1",
        "token": "token",
        "attempt_id": "attempt-1",
    }


def _authorize(**_: object) -> dict[str, object]:
    return {"attempt": {"attempt_id": "attempt-1"}, "intent": {}}


def test_worker_refuses_restore_when_readiness_differs(tmp_path: Path) -> None:
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text('{"status":"other"}\n', encoding="utf-8")
    preflight = {
        "endpoints": [],
        "report_hashes": {},
        "source_hashes": {},
        "baseline": {"sha256": _sha("a")},
    }
    with pytest.raises(worker.Refuse, match="metadata differs"):
        worker._handle_task(
            _request("restore"),
            attempt_path=tmp_path / "attempt.json",
            ledger_path=tmp_path / "ledger.jsonl",
            durability_api=object(),
            trainers={},
            endpoint_specs={},
            corpus_holder={"assistant": object()},
            job_membership_check=lambda: True,
            authorize_fn=_authorize,
            preflight_fn=lambda: preflight,
            evaluation_paths_fn=lambda: {"readiness": readiness_path},
        )


def test_worker_refuses_restore_when_endpoint_is_not_in_the_fixed_map(
    tmp_path: Path,
) -> None:
    preflight = {
        "endpoints": [],
        "report_hashes": {},
        "source_hashes": {},
        "baseline": {"sha256": _sha("a")},
    }
    readiness = recovery._readiness_payload(preflight)
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    attempt = {
        "attempt_id": "attempt-1",
        "readiness": readiness,
        "source_hashes": {},
        "input_report_hashes": {},
    }
    with pytest.raises(worker.Refuse, match="fixed four-endpoint map"):
        worker._handle_task(
            _request("restore"),
            attempt_path=tmp_path / "attempt.json",
            ledger_path=tmp_path / "ledger.jsonl",
            durability_api=object(),
            trainers={},
            endpoint_specs={},
            corpus_holder={"assistant": object()},
            job_membership_check=lambda: True,
            authorize_fn=lambda **_: {"attempt": attempt, "intent": {"checkpoint_sha256": "x"}},
            preflight_fn=lambda: preflight,
            evaluation_paths_fn=lambda: {"readiness": readiness_path},
        )


def test_worker_refuses_generate_before_four_restores(tmp_path: Path) -> None:
    with pytest.raises(worker.Refuse, match="before all four restores"):
        worker._handle_task(
            _request("generate"),
            attempt_path=tmp_path / "attempt.json",
            ledger_path=tmp_path / "ledger.jsonl",
            durability_api=object(),
            trainers={},
            endpoint_specs={},
            corpus_holder={},
            job_membership_check=lambda: True,
            authorize_fn=_authorize,
        )


def test_entry_worker_factory_uses_the_versioned_command(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_worker(**kwargs: object) -> str:
        captured.update(kwargs)
        return "worker"

    monkeypatch.setattr(recovery, "SupervisedWorker", fake_worker)
    result = entry._worker_factory(
        attempt_path=Path("attempt.json"),
        ledger_path=Path("ledger.jsonl"),
        budget=None,
    )
    assert result == "worker"
    command = captured["command"]
    assert isinstance(command, list)
    assert adapter.WORKER_MODULE in command
    assert "scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2" not in command

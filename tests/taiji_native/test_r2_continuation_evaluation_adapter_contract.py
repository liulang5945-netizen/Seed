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

from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as recovery
from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter

REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/training/evaluate_taiji_r2_daily_dialogue_continuation_extension_v1.py"


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
    return {
        "schema": adapter.MANIFEST_SCHEMA,
        "endpoints": records,
        "source_revision_sha256": _sha("c"),
    }


def _baseline(tmp_path: Path) -> dict:
    return {
        "endpoint_id": "parent_baseline",
        "path": tmp_path / "parent_baseline.pt",
        "sha256": _sha("e"),
        "checkpoint_digest": _sha("f"),
    }


def _hash_file(path: Path) -> str:
    return hashlib.sha256(Path(path).name.encode("utf-8")).hexdigest()


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
    manifest = adapter.endpoint_manifest_payload(preflight)
    assert manifest["schema"] == adapter.ENDPOINT_SCHEMA
    assert manifest["can_promote"] is False
    assert [row["endpoint_id"] for row in manifest["endpoints"]] == list(adapter.ENDPOINT_ORDER)


def test_entry_refuses_execute_and_worker_with_stdout_json() -> None:
    for flag in ("--execute", "--worker"):
        proc = subprocess.run([sys.executable, str(ENTRY), flag], capture_output=True, check=False)
        assert proc.returncode == 2, (flag, proc.returncode, proc.stdout, proc.stderr)
        assert proc.stderr.decode("utf-8", errors="replace") == ""
        payload = json.loads(proc.stdout.decode("utf-8"))
        assert payload["status"] == "refused"
        assert payload["can_promote"] is False

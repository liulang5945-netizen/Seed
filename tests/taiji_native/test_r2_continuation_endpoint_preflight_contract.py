import copy
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.training import prepare_taiji_r2_continuation_endpoints as m
from scripts.training.validate_taiji_r2_continuation_endpoint import STAGES, EndpointRefuse


def fixture(tmp_path):
    lineage = {
        key: str(index) * 64
        for index, key in enumerate(
            (
                "extension_attempt_sha256",
                "parent_attempt_sha256",
                "source_revision_sha256",
                "parent_receipt_sha256",
                "runner_sha256",
            ),
            1,
        )
    }
    stages = {}
    for name, (episodes, updates) in STAGES.items():
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        digest = "a" * 64
        stages[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "episodes": episodes,
            "target_byte_updates": updates,
            "checkpoint_digest": digest,
            "restore": {
                "complete": True,
                "equal": True,
                "stage": name.removesuffix("_final"),
                "checkpoint_digest": digest,
            },
        }
    common = {
        "arm": "warm_start",
        "can_promote": False,
        "dev_queries": 0,
        "reference_queries": 0,
        "generation_queries": 0,
        "extension_attempt_sha256": lineage["extension_attempt_sha256"],
        "parent_attempt_sha256": lineage["parent_attempt_sha256"],
        "source_revision_sha256": lineage["source_revision_sha256"],
        "stages": stages,
        "cpu_seconds_cumulative": 1_400_000.0,
        "cpu_seconds_cumulative_is_lower_bound": True,
        "cpu_seconds_cumulative_boundary": "measured_before_final_result_publication",
        "completion_cpu_source": "continuation_completed_ledger_event",
    }
    ready = {
        **common,
        "schema": "taiji-r2-dialogue-continuation-extension-readiness-v1",
        "status": "ready",
        "parent_receipt_sha256": lineage["parent_receipt_sha256"],
        "extension_runner_sha256": lineage["runner_sha256"],
    }
    result = {
        **copy.deepcopy(common),
        "schema": "taiji-r2-dialogue-continuation-extension-result-v1",
        "status": "complete",
        "complete": True,
        "parent_checkpoint_receipt_sha256": lineage["parent_receipt_sha256"],
        "readiness_sha256": "b" * 64,
    }
    event = {
        "event": "continuation_completed",
        "arm": "warm_start",
        "extension_id": "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1",
        "parent_attempt_sha256": lineage["parent_attempt_sha256"],
        "charged_cpu_seconds": 1_400_005.0,
        "event_sha256": "d" * 64,
        "details": {"result_sha256": "c" * 64, "readiness_sha256": "b" * 64},
    }
    kwargs = {
        "lineage": lineage,
        "root": tmp_path,
        "result_sha256": "c" * 64,
        "readiness_sha256": "b" * 64,
        "per_arm_cpu_limit": 1_500_000.0,
    }
    return ready, result, event, kwargs


def complete_backend(tmp_path):
    arms = ("full_turn_1m", "warm_start", "fresh_same_capacity")
    paths = {}
    originals = {}
    fixtures = {}
    event_paths = []
    for arm in arms:
        root = tmp_path / arm
        root.mkdir()
        ready, result, event, kwargs = fixture(root)
        for record in (ready, result, event):
            record["arm"] = arm
        parent_attempt = root / "parent-attempt.json"
        parent_attempt.write_text("{}")
        parent_ready = root / "parent-ready.json"
        parent_ready.write_text("{}")
        receipt = root / "parent-receipt.json"
        receipt.write_text("{}")
        bundle = root / "parent.pt"
        bundle.write_bytes(b"parent")
        attempt = {
            "parent": {"attempt_sha256": m.sha(parent_attempt)},
            "parent_checkpoint": {
                "receipt_path": str(receipt),
                "bundle_path": str(bundle),
                "bundle_sha256": m.sha(bundle),
            },
            "source_compatibility": {"files": []},
        }
        attempt_path = root / "attempt.json"
        attempt_path.write_text(json.dumps(attempt))
        result_path = root / "result.json"
        paths[arm] = {
            "extension_result": result_path,
            "extension_attempt": attempt_path,
            "extension_checkpoint_root": root,
            "parent_attempt": parent_attempt,
            "parent_readiness": parent_ready,
        }
        originals[arm] = {
            "attempt_sha256": m.sha(attempt_path),
            "parent_readiness_sha256": m.sha(parent_ready),
            "parent_receipt_sha256": m.sha(receipt),
        }
        fixtures[arm] = ready, result, event
    revision = tmp_path / "fixture_source_revision_v1_3.json"
    revision.write_text(
        json.dumps(
            {
                "original_attempts": originals,
                "current_continuation_sources": {"runner_sha256": "5" * 64},
            }
        )
    )
    for arm in arms:
        ready, result, event = fixtures[arm]
        ready["source_revision_sha256"] = result["source_revision_sha256"] = m.sha(revision)
        ready["extension_attempt_sha256"] = result["extension_attempt_sha256"] = originals[arm][
            "attempt_sha256"
        ]
        ready["parent_attempt_sha256"] = result["parent_attempt_sha256"] = event[
            "parent_attempt_sha256"
        ] = m.sha(paths[arm]["parent_attempt"])
        ready["parent_receipt_sha256"] = result["parent_checkpoint_receipt_sha256"] = originals[
            arm
        ]["parent_receipt_sha256"]
        result_path = paths[arm]["extension_result"]
        ready_path = m.readiness_path(result_path)
        ready_path.write_text(json.dumps(ready))
        result["readiness_sha256"] = m.sha(ready_path)
        result_path.write_text(json.dumps(result))
        event["details"] = {
            "result_sha256": m.sha(result_path),
            "readiness_sha256": m.sha(ready_path),
        }
        event_path = tmp_path / f"{arm}-completion.json"
        event_path.write_text(json.dumps(event))
        event_paths.append((len(event_paths), event_path))
    backend = SimpleNamespace(
        ARMS=arms,
        paths=lambda arm: paths[arm],
        REPORTS_DIR=tmp_path,
        EXTENSION_PREFIX="fixture",
        LOCK_PATH=tmp_path / "ledger.lock",
        AGGREGATE_LEDGER_PREFIX=tmp_path / "ledger",
        PER_ARM_CPU_LIMIT=1_500_000.0,
        AGGREGATE_CPU_LIMIT=4_300_000.0,
        _aggregate_lock=lambda _path: nullcontext(),
        _read_ledger_full=lambda: ({}, {arm: 1_400_005.0 for arm in arms}, {}, "a" * 64),
        _event_files=lambda _prefix: event_paths,
    )
    return backend, event_paths


def test_all_three_metadata_endpoints_require_no_model_or_prompt_access(tmp_path):
    backend, _ = complete_backend(tmp_path)
    manifest = m.metadata_preflight(backend)
    assert set(manifest["endpoints"]) == set(backend.ARMS)
    assert manifest["can_query_dev"] is False
    assert manifest["can_promote"] is False


@pytest.mark.parametrize("drift", ["incomplete", "duplicate_completion", "writer", "parent"])
def test_incomplete_or_unbound_metadata_is_refused(tmp_path, drift):
    backend, event_paths = complete_backend(tmp_path)
    if drift == "incomplete":
        backend.paths("warm_start")["extension_result"].unlink()
        backend._read_ledger_full = lambda: pytest.fail(
            "incomplete endpoints must not enter ledger preflight"
        )
    elif drift == "duplicate_completion":
        event_paths.append(event_paths[0])
    elif drift == "writer":
        (backend.paths("warm_start")["extension_checkpoint_root"] / "writer.lock").write_text(
            "live"
        )
    else:
        backend.paths("warm_start")["parent_attempt"].write_text("changed")
    with pytest.raises(EndpointRefuse):
        m.metadata_preflight(backend)

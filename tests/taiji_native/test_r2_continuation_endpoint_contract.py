import copy
import hashlib

import pytest

from scripts.training.validate_taiji_r2_continuation_endpoint import (
    STAGES,
    EndpointRefuse,
    validate_extension_endpoint,
)


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


def test_completed_extension_is_bound_without_old_schema_translation(tmp_path):
    ready, result, event, kwargs = fixture(tmp_path)
    endpoint = validate_extension_endpoint("warm_start", ready, result, event, **kwargs)
    assert endpoint["endpoint_kind"] == "continuation_extension"
    assert endpoint["can_promote"] is False
    assert endpoint["cpu_seconds_cumulative_lower_bound"] == 1_400_005.0


@pytest.mark.parametrize(
    "drift", ["partial", "queries", "denominator", "restore", "ledger", "checkpoint", "identity"]
)
def test_partial_or_unbound_endpoint_is_rejected(tmp_path, drift):
    ready, result, event, kwargs = fixture(tmp_path)
    if drift == "partial":
        result["complete"] = False
    elif drift == "queries":
        result["dev_queries"] = 1
    elif drift == "denominator":
        result["stages"]["assistant_sft_final"]["episodes"] = 524
    elif drift == "restore":
        result["stages"]["assistant_sft_final"]["restore"]["equal"] = False
    elif drift == "ledger":
        event["details"]["result_sha256"] = "f" * 64
    elif drift == "checkpoint":
        (tmp_path / "assistant_sft_final.pt").write_bytes(b"changed")
    else:
        kwargs["lineage"]["source_revision_sha256"] = None
    if drift in {"denominator", "restore"}:
        ready["stages"] = copy.deepcopy(result["stages"])
    with pytest.raises(EndpointRefuse):
        validate_extension_endpoint("warm_start", ready, result, event, **kwargs)

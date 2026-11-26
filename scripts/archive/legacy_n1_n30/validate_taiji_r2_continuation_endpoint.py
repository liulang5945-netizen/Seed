"""Metadata-only endpoint gate for completed continuation-extension arms.

Callers must first verify the full aggregate ledger and supply its unique
completion event. No model, corpus, prompt selector or evaluator is imported.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARMS = ("full_turn_1m", "warm_start", "fresh_same_capacity")
STAGES = {"pretraining_final": (30_096, 14_517_912), "assistant_sft_final": (525, 113_209)}


class EndpointRefuse(ValueError):
    """The endpoint is incomplete or its metadata cannot be bound."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_extension_endpoint(
    arm: str,
    readiness: Mapping[str, Any],
    result: Mapping[str, Any],
    verified_completion_event: Mapping[str, Any],
    *,
    lineage: Mapping[str, str],
    root: Path,
    result_sha256: str,
    readiness_sha256: str,
    per_arm_cpu_limit: float,
) -> dict[str, Any]:
    """Bind both final files and restore evidence to one verified completion.

    This creates an extension endpoint record; it never relabels it as an old
    restart-from-zero recovery result. CPU totals remain explicit lower bounds.
    """
    if arm not in ARMS:
        raise EndpointRefuse("unknown arm")
    required_hashes = [
        result_sha256,
        readiness_sha256,
        verified_completion_event.get("event_sha256"),
    ]
    required_hashes.extend(
        lineage.get(key)
        for key in (
            "extension_attempt_sha256",
            "parent_attempt_sha256",
            "source_revision_sha256",
            "parent_receipt_sha256",
            "runner_sha256",
        )
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        for value in required_hashes
    ):
        raise EndpointRefuse("endpoint identity SHA is missing or malformed")
    if root.is_symlink():
        raise EndpointRefuse("endpoint root must not be a symlink")
    for record, schema, status in (
        (readiness, "taiji-r2-dialogue-continuation-extension-readiness-v1", "ready"),
        (result, "taiji-r2-dialogue-continuation-extension-result-v1", "complete"),
    ):
        if (
            record.get("schema") != schema
            or record.get("status") != status
            or record.get("arm") != arm
        ):
            raise EndpointRefuse("endpoint schema, status or arm differs")
        if record.get("can_promote") is not False or any(
            record.get(key) != 0
            for key in ("dev_queries", "reference_queries", "generation_queries")
        ):
            raise EndpointRefuse("endpoint query or promotion boundary differs")
        if (
            record.get("extension_attempt_sha256") != lineage["extension_attempt_sha256"]
            or record.get("parent_attempt_sha256") != lineage["parent_attempt_sha256"]
            or record.get("source_revision_sha256") != lineage["source_revision_sha256"]
        ):
            raise EndpointRefuse("endpoint lineage differs")
        if (
            record.get("cpu_seconds_cumulative_is_lower_bound") is not True
            or record.get("cpu_seconds_cumulative_boundary")
            != "measured_before_final_result_publication"
            or record.get("completion_cpu_source") != "continuation_completed_ledger_event"
        ):
            raise EndpointRefuse("CPU evidence boundary differs")
    if result.get("complete") is not True or result.get("readiness_sha256") != readiness_sha256:
        raise EndpointRefuse("result does not bind ready endpoint")
    if (
        readiness.get("parent_receipt_sha256") != lineage["parent_receipt_sha256"]
        or result.get("parent_checkpoint_receipt_sha256") != lineage["parent_receipt_sha256"]
        or readiness.get("extension_runner_sha256") != lineage["runner_sha256"]
    ):
        raise EndpointRefuse("parent receipt or runner binding differs")
    event = verified_completion_event
    if (
        event.get("event") != "continuation_completed"
        or event.get("arm") != arm
        or event.get("parent_attempt_sha256") != lineage["parent_attempt_sha256"]
        or event.get("extension_id") != "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1"
    ):
        raise EndpointRefuse("verified ledger event is not this endpoint completion")
    details = event.get("details", {})
    if (
        details.get("result_sha256") != result_sha256
        or details.get("readiness_sha256") != readiness_sha256
    ):
        raise EndpointRefuse("completion event binds different result or readiness bytes")
    lower = result.get("cpu_seconds_cumulative")
    charged = event.get("charged_cpu_seconds")
    if (
        any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (lower, charged)
        )
        or not 0 <= lower <= charged <= per_arm_cpu_limit
        or readiness.get("cpu_seconds_cumulative") != lower
    ):
        raise EndpointRefuse("cumulative CPU evidence is invalid")
    if result.get("stages") != readiness.get("stages"):
        raise EndpointRefuse("final stage mappings differ")
    verified = {}
    for name, (episodes, updates) in STAGES.items():
        stage = result.get("stages", {}).get(name, {})
        path = Path(stage.get("path", ""))
        expected = root.resolve() / f"{name}.pt"
        if (
            path.is_symlink()
            or path.resolve() != expected
            or not path.is_file()
            or _sha(path) != stage.get("sha256")
        ):
            raise EndpointRefuse("final checkpoint path or SHA differs")
        restore = stage.get("restore", {})
        if (
            stage.get("episodes") != episodes
            or stage.get("target_byte_updates") != updates
            or restore.get("complete") is not True
            or restore.get("equal") is not True
            or restore.get("stage") != name.removesuffix("_final")
            or restore.get("checkpoint_digest") != stage.get("checkpoint_digest")
            or not stage.get("checkpoint_digest")
        ):
            raise EndpointRefuse("final denominator or independent restore differs")
        verified[name] = dict(stage)
    return {
        "arm": arm,
        "endpoint_kind": "continuation_extension",
        "final_checkpoints": verified,
        "source_revision_sha256": lineage["source_revision_sha256"],
        "completion_event_sha256": event.get("event_sha256"),
        "cpu_seconds_cumulative_lower_bound": charged,
        "can_promote": False,
    }

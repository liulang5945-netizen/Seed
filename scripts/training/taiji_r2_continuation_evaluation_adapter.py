"""Provider seam for the continuation-extension paired evaluation (CC-1.2).

The recovery-v2 evaluator is frozen: its ``metadata_preflight()`` and
``_evaluation_paths()`` resolve the recovery-v2 arms, and its validator requires
``restart_from_zero=true``.  The extension arms have the opposite lineage
(``origin=resume_from_verified_parent``, never ``restart_from_zero``).  This
module adapts the extension endpoint manifest into the same four-endpoint
envelope the frozen evaluator consumes, without editing it or monkeypatching its
globals.

Metadata-only: no prompt, no model, no checkpoint deserialization.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as recovery
from scripts.training import prepare_taiji_r2_continuation_endpoints as endpoints
from scripts.training import taiji_r2_dialogue_continuation_extension as extension

ARMS = extension.ARMS
ENDPOINT_ORDER = ("parent_baseline", *ARMS)
ENDPOINT_SCHEMA = "taiji-r2-continuation-evaluation-endpoints-v1"
MANIFEST_SCHEMA = "taiji-r2-continuation-endpoint-manifest-v1"
WORKER_MODULE = "scripts.training.evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1"
FINAL_STAGE = "assistant_sft_final"
PRETRAIN_STAGE = "pretraining_final"
EVAL_STAGE = "assistant_sft"
EXPECTED_FINAL = {"episodes": 525, "target_byte_updates": 113209}
EXPECTED_PRETRAIN = {"episodes": 30096, "target_byte_updates": 14517912}
ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "scripts/training/evaluate_taiji_r2_daily_dialogue_continuation_extension_v1.py",
    ROOT / "scripts/training/evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1.py",
    ROOT / "scripts/training/taiji_r2_dialogue_continuation_extension.py",
    ROOT / "scripts/training/prepare_taiji_r2_continuation_endpoints.py",
    ROOT / "scripts/training/validate_taiji_r2_continuation_endpoint.py",
    ROOT / "tests/taiji_native/test_r2_continuation_evaluation_adapter_contract.py",
)


class AdapterRefuse(ValueError):
    """An extension endpoint manifest cannot enter the frozen evaluation envelope."""


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def snapshot_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def extension_source_hashes(
    *, hash_file: Callable[[Path], str] = recovery.sha256_file
) -> dict[str, str]:
    """Bind the frozen evaluator sources plus every extension-source seam.

    The frozen evaluator hashes its own source closure.  The extension runs the
    same protocol through three extra modules, so those files must travel in the
    readiness/attempt ``source_hashes`` too; otherwise a mid-run edit could not
    be detected by the parent or the supervised worker.
    """
    paths = dict.fromkeys((*recovery._evaluation_source_paths(), *EXTENSION_SOURCE_PATHS))
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise AdapterRefuse(f"extension evaluation source is missing: {path}")
        hashes[str(path.resolve())] = hash_file(path)
    return dict(sorted(hashes.items()))


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise AdapterRefuse(f"{label} must be a finite non-negative number")
    return float(value)


def launch_evidence(
    manifest: Mapping[str, Any], *, hash_file: Callable[[Path], str] = recovery.sha256_file
) -> dict[str, Any]:
    """Bind actual launch lineage, ledger tip/charges and approved budget.

    ``prepare_taiji_r2_continuation_endpoints.metadata_preflight`` produces these
    fields from the verified launch manifests, source revision, boot audit and
    aggregate ledger.  They must enter the evaluation snapshot: otherwise the
    endpoint map would be frozen while the launches and their accounting could
    drift unnoticed.
    """
    manifest = _mapping(manifest, "extension endpoint manifest")
    provenance = _mapping(manifest.get("launch_provenance"), "launch provenance")
    launches = _mapping(provenance.get("launches"), "launch provenance arms")
    if set(launches) != set(ARMS):
        raise AdapterRefuse("launch provenance must cover exactly the three arms")
    for arm in ARMS:
        rows = launches[arm]
        if not isinstance(rows, (list, tuple)) or not rows:
            raise AdapterRefuse(f"{arm} launch provenance is empty")
        for row in rows:
            row = _mapping(row, f"{arm} launch provenance row")
            _sha(row.get("sha256"), f"{arm} launch manifest sha256")
            _sha(row.get("start_receipt_sha256"), f"{arm} launch start receipt sha256")
            _sha(row.get("start_bundle_sha256"), f"{arm} launch start bundle sha256")
    verified = _mapping(provenance.get("verified_sources"), "verified launch sources")
    if not verified:
        raise AdapterRefuse("launch provenance has no verified source bytes")
    historical = provenance.get("historical_sources", {})
    if not isinstance(historical, Mapping):
        raise AdapterRefuse("historical launch sources must be an object")
    historical_sources: dict[str, str] = {}
    for raw_path, digest in historical.items():
        expected = _sha(digest, f"historical launch source {raw_path}")
        path = Path(str(raw_path))
        absolute = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        if not absolute.is_file():
            raise AdapterRefuse(f"historical launch source is missing: {raw_path}")
        historical_sources[str(absolute)] = expected
    source_hashes: dict[str, str] = {}
    for raw_path, digest in verified.items():
        expected = _sha(digest, f"launch source {raw_path}")
        path = Path(str(raw_path))
        absolute = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        if not absolute.is_file():
            raise AdapterRefuse(f"launch source is missing: {raw_path}")
        actual = hash_file(absolute)
        if actual != expected:
            raise AdapterRefuse(f"launch source bytes differ: {raw_path}")
        source_hashes[str(absolute)] = actual
    tip = _sha(manifest.get("verified_ledger_tip_sha256"), "verified ledger tip")
    raw_charges = _mapping(manifest.get("ledger_charges"), "ledger charges")
    if set(raw_charges) != set(ARMS):
        raise AdapterRefuse("ledger charges must cover exactly the three arms")
    charges = {arm: _number(raw_charges[arm], f"{arm} charged CPU") for arm in ARMS}
    raw_budget = _mapping(manifest.get("approved_budget"), "approved budget")
    per_arm = _number(raw_budget.get("per_arm"), "approved per-arm budget")
    aggregate = _number(raw_budget.get("aggregate"), "approved aggregate budget")
    if any(charge > per_arm for charge in charges.values()) or sum(charges.values()) > aggregate:
        raise AdapterRefuse("ledger charges exceed the approved launch budget")
    return {
        "source_revision_sha256": _sha(
            manifest.get("source_revision_sha256"), "source revision sha256"
        ),
        "launch_provenance_sha256": snapshot_sha256(provenance),
        "verified_ledger_tip_sha256": tip,
        "ledger_charges": charges,
        "approved_budget": {"per_arm": per_arm, "aggregate": aggregate},
        "source_hashes": dict(sorted(source_hashes.items())),
        "historical_sources": dict(sorted(historical_sources.items())),
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterRefuse(f"{label} must be an object")
    return value


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdapterRefuse(f"{label} must be a lowercase sha256")
    return value


def _reject_restart_from_zero(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        if value.get("restart_from_zero") is True:
            raise AdapterRefuse(f"{label} forges restart_from_zero=true")
        for key, item in value.items():
            _reject_restart_from_zero(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_restart_from_zero(item, f"{label}[{index}]")


def _stage(
    record: Mapping[str, Any], name: str, expected: Mapping[str, int], label: str
) -> Mapping[str, Any]:
    final = _mapping(record.get("final_checkpoints"), f"{label} final_checkpoints")
    stage = _mapping(final.get(name), f"{label} {name}")
    if (
        stage.get("episodes") != expected["episodes"]
        or stage.get("target_byte_updates") != expected["target_byte_updates"]
    ):
        raise AdapterRefuse(f"{label} {name} denominator differs")
    _sha(stage.get("sha256"), f"{label} {name} sha256")
    if not stage.get("checkpoint_digest"):
        raise AdapterRefuse(f"{label} {name} checkpoint digest is missing")
    if not Path(str(stage.get("path", ""))).is_absolute():
        raise AdapterRefuse(f"{label} {name} path must be absolute")
    restore = _mapping(stage.get("restore"), f"{label} {name} restore")
    if restore.get("complete") is not True or restore.get("equal") is not True:
        raise AdapterRefuse(f"{label} {name} independent restore is incomplete")
    if restore.get("stage") != name.removesuffix("_final"):
        raise AdapterRefuse(f"{label} {name} restore stage differs from its checkpoint stage")
    if restore.get("checkpoint_digest") != stage.get("checkpoint_digest"):
        raise AdapterRefuse(f"{label} {name} restore digest differs from the checkpoint digest")
    return stage


def extension_endpoint_map(
    manifest: Mapping[str, Any],
    *,
    baseline_fn: Callable[[], dict[str, Any]] = recovery.validate_baseline_checkpoint,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], dict[str, dict[str, Any]]]:
    """Map the extension manifest onto the frozen ``parent + three arms`` order.

    The frozen evaluator restores the ``assistant_sft`` final checkpoint.  The
    extension manifest names the same checkpoint ``assistant_sft_final``; the
    rename is explicit here so the two stages can never be silently swapped.
    """

    manifest = _mapping(manifest, "extension endpoint manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise AdapterRefuse("extension endpoint manifest schema differs")
    _reject_restart_from_zero(manifest, "manifest")
    records = _mapping(manifest.get("endpoints"), "extension endpoint records")
    if set(records) != set(ARMS):
        raise AdapterRefuse("extension endpoint manifest must contain exactly the three arms")
    arm_records: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        record = _mapping(records[arm], f"{arm} endpoint")
        if record.get("arm") != arm or record.get("endpoint_kind") != "continuation_extension":
            raise AdapterRefuse(f"{arm} endpoint identity differs")
        if record.get("can_promote") is not False:
            raise AdapterRefuse(f"{arm} endpoint must not be promotable")
        _stage(record, PRETRAIN_STAGE, EXPECTED_PRETRAIN, arm)
        assistant = _stage(record, FINAL_STAGE, EXPECTED_FINAL, arm)
        arm_records[arm] = {
            "arm": arm,
            "final_checkpoints": {
                EVAL_STAGE: {
                    "path": assistant["path"],
                    "sha256": assistant["sha256"],
                    "checkpoint_digest": assistant["checkpoint_digest"],
                }
            },
        }
    baseline = dict(baseline_fn())
    endpoint_rows = recovery.build_endpoint_map(baseline, arm_records)
    return endpoint_rows, baseline, arm_records


def extension_evaluation_paths() -> dict[str, Path]:
    """Independent output prefix; never the recovery-v2 paired-evaluation prefix."""
    prefix = extension.REPORTS_DIR / f"{extension.EXTENSION_PREFIX}_paired_eval"
    suffixes = {
        "readiness": "readiness",
        "attempt": "attempt",
        "ledger": "ledger",
        "progress": "progress",
        "face": "face",
        "generations": "generations",
        "result": "result",
        "cost_receipt": "cost_receipt",
        "review": "review",
        "failure": "failure",
        "endpoint_manifest": "endpoints",
    }
    return {
        key: (
            prefix.with_name(f"{prefix.name}_{suffix}.json")
            if key != "ledger" and key != "progress"
            else prefix.with_name(f"{prefix.name}_{suffix}.jsonl")
        )
        for key, suffix in suffixes.items()
    }


def extension_worker_command(attempt_path: Path, ledger_path: Path) -> list[str]:
    """Versioned worker entry; ``{job_name}`` is substituted by SupervisedWorker."""
    return [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--worker",
        "--attempt-path",
        str(Path(attempt_path).resolve()),
        "--ledger-path",
        str(Path(ledger_path).resolve()),
        "--job-name",
        "{job_name}",
    ]


CONSUMPTION_KEYS = (
    "attempt",
    "ledger",
    "progress",
    "face",
    "generations",
    "result",
    "cost_receipt",
    "review",
    "failure",
)


def refuse_if_recovery_evaluation_consumed(
    *,
    paths_fn: Callable[[], Mapping[str, Path]] = recovery._evaluation_paths,
) -> None:
    """The frozen recovery-v2 evaluation and this extension share one 39-family face.

    Only consumption signals block the extension.  A readiness written by
    ``--validate-only`` is preparation, not consumption, so it must not block.
    The old entry is frozen and cannot check the extension prefix; running the
    old entry after the extension is a protocol violation recorded in 03.
    """
    paths = paths_fn()
    for key in CONSUMPTION_KEYS:
        path = paths.get(key)
        if path is not None and Path(path).exists():
            raise AdapterRefuse(
                f"recovery-v2 evaluation is already consumed ({key}); "
                "the shared 39-family face cannot be queried twice"
            )


def extension_preflight(
    manifest: Mapping[str, Any] | None = None,
    *,
    arm_paths: Callable[[str], Mapping[str, Path]] = extension.paths,
    hash_file: Callable[[Path], str] = recovery.sha256_file,
    baseline_fn: Callable[[], dict[str, Any]] = recovery.validate_baseline_checkpoint,
    eval_hashes: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the frozen evaluator's envelope from extension endpoints and evidence."""
    refuse_if_recovery_evaluation_consumed()
    if manifest is None:
        manifest = endpoints.metadata_preflight()
    endpoint_rows, baseline, arm_records = extension_endpoint_map(manifest, baseline_fn=baseline_fn)
    launch = launch_evidence(manifest, hash_file=hash_file)
    lineage_snapshot = launch["launch_provenance_sha256"]
    endpoint_rows = tuple(
        dict(
            row,
            endpoint_kind=(
                "parent_baseline"
                if row["endpoint_id"] == "parent_baseline"
                else "continuation_extension"
            ),
            lineage_snapshot_sha256=lineage_snapshot,
        )
        for row in endpoint_rows
    )
    report_hashes: dict[str, dict[str, str]] = {}
    source_hashes: dict[str, str] = dict(launch["source_hashes"])
    for arm in ARMS:
        paths = arm_paths(arm)
        result = Path(paths["extension_result"])
        readiness = result.with_name(f"{result.stem}_readiness.json")
        if not result.is_file() or not readiness.is_file():
            raise AdapterRefuse(f"{arm} extension result/readiness is missing")
        report_hashes[arm] = {
            "result_sha256": hash_file(result),
            "readiness_sha256": hash_file(readiness),
        }
        for path in (result, readiness, Path(paths["extension_attempt"])):
            source_hashes[str(path.resolve())] = hash_file(path)
    for row in endpoint_rows:
        source_hashes[str(Path(row["path"]).resolve())] = row["sha256"]
    source_hashes[str(Path(baseline["path"]).resolve())] = baseline["sha256"]
    frozen_or_injected = eval_hashes or (lambda: extension_source_hashes(hash_file=hash_file))
    source_hashes.update(frozen_or_injected())
    return {
        "validated_arms": arm_records,
        "report_hashes": report_hashes,
        "baseline": baseline,
        "endpoints": list(endpoint_rows),
        "source_hashes": dict(sorted(source_hashes.items())),
        "endpoint_schema": ENDPOINT_SCHEMA,
        "endpoint_manifest_sha256": snapshot_sha256(manifest),
        "launch": launch,
    }


def extension_readiness(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen evaluator's readiness envelope, driven by extension metadata.

    The schema stays the frozen evaluator's envelope because ``execute()``
    compares it byte-for-byte.  Extension identity travels in the separate
    endpoint manifest below, not by mutating the frozen envelope.
    """
    return recovery._readiness_payload(preflight)


def write_extension_readiness(path: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    payload = extension_readiness(preflight)
    recovery._write_or_verify_readiness(Path(path), payload)
    return payload


def endpoint_manifest_payload(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Independent extension schema; never consumed by the frozen evaluator."""
    launch = preflight["launch"]
    return {
        "schema": ENDPOINT_SCHEMA,
        "manifest_sha256": preflight["endpoint_manifest_sha256"],
        "endpoint_order": list(ENDPOINT_ORDER),
        "endpoints": [
            {
                "endpoint_id": row["endpoint_id"],
                "endpoint_kind": row["endpoint_kind"],
                "path": str(row["path"]),
                "sha256": row["sha256"],
                "checkpoint_digest": row["checkpoint_digest"],
                "source": row["source"],
                "lineage_snapshot_sha256": row["lineage_snapshot_sha256"],
            }
            for row in preflight["endpoints"]
        ],
        "source_revision_sha256": launch["source_revision_sha256"],
        "launch_provenance_sha256": launch["launch_provenance_sha256"],
        "verified_ledger_tip_sha256": launch["verified_ledger_tip_sha256"],
        "ledger_charges": launch["ledger_charges"],
        "approved_budget": launch["approved_budget"],
        "source_hashes": preflight["source_hashes"],
        "historical_sources": launch["historical_sources"],
        "can_promote": False,
    }


def write_endpoint_manifest(path: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    payload = endpoint_manifest_payload(preflight)
    recovery._write_or_verify_readiness(Path(path), payload)
    return payload

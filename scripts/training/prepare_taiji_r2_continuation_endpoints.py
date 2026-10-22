"""Prepare extension endpoint metadata after all arms finish; no evaluation or prompt access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import taiji_r2_dialogue_continuation_extension as budget
from scripts.training.validate_taiji_r2_continuation_endpoint import (
    EndpointRefuse,
    validate_extension_endpoint,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readiness_path(result):
    return result.with_name(result.stem + "_readiness.json")


def pending_arms(backend=budget):
    return [
        arm
        for arm in backend.ARMS
        if any(
            not p.is_file()
            for p in (
                backend.paths(arm)["extension_result"],
                readiness_path(backend.paths(arm)["extension_result"]),
            )
        )
    ]


def metadata_preflight(backend=budget):
    pending = pending_arms(backend)
    if pending:
        raise EndpointRefuse(f"training endpoints are incomplete: {pending}")
    revision_path = backend.REPORTS_DIR / f"{backend.EXTENSION_PREFIX}_source_revision_v1_3.json"
    revision = read_json(revision_path)
    revision_sha = sha(revision_path)
    with backend._aggregate_lock(backend.LOCK_PATH):
        _, charges, reservations, tip = backend._read_ledger_full()
        if reservations:
            raise EndpointRefuse("endpoint writers still have CPU reservations")
        completions = {arm: [] for arm in backend.ARMS}
        for _, path in backend._event_files(backend.AGGREGATE_LEDGER_PREFIX):
            event = read_json(path)
            if event.get("event") == "continuation_completed" and event.get("arm") in completions:
                completions[event["arm"]].append(event)
        if any(len(events) != 1 for events in completions.values()):
            raise EndpointRefuse("each endpoint requires exactly one verified completion event")
    endpoints = {}
    for arm in backend.ARMS:
        paths = backend.paths(arm)
        root = paths["extension_checkpoint_root"]
        if any(
            (directory / name).exists()
            for directory in (root, root / "pretraining", root / "assistant_sft")
            for name in ("writer.lock", "reclaim.guard")
        ):
            raise EndpointRefuse("endpoint checkpoint writer is not closed")
        attempt_path = paths["extension_attempt"]
        attempt = read_json(attempt_path)
        original = revision["original_attempts"][arm]
        checkpoint = attempt["parent_checkpoint"]
        if (
            sha(attempt_path) != original["attempt_sha256"]
            or sha(paths["parent_attempt"]) != attempt["parent"]["attempt_sha256"]
            or sha(paths["parent_readiness"]) != original["parent_readiness_sha256"]
            or sha(Path(checkpoint["receipt_path"])) != original["parent_receipt_sha256"]
            or sha(Path(checkpoint["bundle_path"])) != checkpoint["bundle_sha256"]
        ):
            raise EndpointRefuse("original parent lineage differs")
        for row in attempt["source_compatibility"]["files"]:
            source = ROOT / row["path"]
            if source.is_symlink() or sha(source) != row["current_sha256"]:
                raise EndpointRefuse("bound training input or compatible source differs")
        result_path = paths["extension_result"]
        ready_path = readiness_path(result_path)
        lineage = {
            "extension_attempt_sha256": sha(attempt_path),
            "parent_attempt_sha256": attempt["parent"]["attempt_sha256"],
            "parent_receipt_sha256": original["parent_receipt_sha256"],
            "runner_sha256": revision["current_continuation_sources"]["runner_sha256"],
            "source_revision_sha256": revision_sha,
        }
        endpoints[arm] = validate_extension_endpoint(
            arm,
            read_json(ready_path),
            read_json(result_path),
            completions[arm][0],
            lineage=lineage,
            root=root,
            result_sha256=sha(result_path),
            readiness_sha256=sha(ready_path),
            per_arm_cpu_limit=backend.PER_ARM_CPU_LIMIT,
        )
    return {
        "schema": "taiji-r2-continuation-endpoint-manifest-v1",
        "endpoints": endpoints,
        "source_revision_sha256": revision_sha,
        "verified_ledger_tip_sha256": tip,
        "ledger_charges": charges,
        "approved_budget": {
            "per_arm": backend.PER_ARM_CPU_LIMIT,
            "aggregate": backend.AGGREGATE_CPU_LIMIT,
        },
        "can_query_dev": False,
        "can_promote": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare", action="store_true", help="write metadata once after all endpoints validate"
    )
    args = parser.parse_args()
    pending = pending_arms()
    if pending and not args.prepare:
        print(json.dumps({"status": "training", "missing_final_arms": pending, "dev_queries": 0}))
        return
    manifest = metadata_preflight()
    if args.prepare:
        output = budget.REPORTS_DIR / f"{budget.EXTENSION_PREFIX}_endpoint_manifest_v1.json"
        raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if output.exists():
            if output.read_bytes() != raw:
                raise EndpointRefuse("existing endpoint manifest differs")
        else:
            budget.durable_publish(output, lambda stream: stream.write(raw))
    print(json.dumps({"status": "endpoints_validated", "dev_queries": 0, "can_query_dev": False}))


if __name__ == "__main__":
    main()

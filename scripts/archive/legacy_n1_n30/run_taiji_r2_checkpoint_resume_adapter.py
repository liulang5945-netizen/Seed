"""Resume a trained continuation via verified receipt ancestry without changing live runner sources."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as r


def verify_receipt_ancestry(directory: Path, parent_receipt_sha256: str) -> None:
    latest = r.read_json(directory / "latest.json")
    receipt_name = latest.get("receipt")
    if not isinstance(receipt_name, str) or Path(receipt_name).name != receipt_name:
        raise r.Refuse("latest receipt path is invalid")
    current_path = directory / receipt_name
    if not current_path.is_file() or r.sha256_file(current_path) != latest.get("receipt_sha256"):
        raise r.Refuse("latest receipt SHA differs")
    by_hash = {r.sha256_file(p): p for p in directory.glob("commit-*.json") if not p.is_symlink()}
    current = latest["receipt_sha256"]
    seen = set()
    while current != parent_receipt_sha256:
        if current in seen or current not in by_hash:
            raise r.Refuse("receipt chain does not reach immutable parent")
        seen.add(current)
        receipt = r.read_json(by_hash[current])
        bundle_name = receipt.get("bundle")
        if not isinstance(bundle_name, str) or Path(bundle_name).name != bundle_name:
            raise r.Refuse("receipt bundle path is invalid")
        bundle = directory / bundle_name
        if (
            bundle.is_symlink()
            or not bundle.is_file()
            or r.sha256_file(bundle) != receipt.get("sha256")
        ):
            raise r.Refuse("receipt bundle SHA differs")
        current = receipt.get("previous_receipt_sha256")
    if current not in by_hash:
        raise r.Refuse("immutable parent receipt is absent")


def validate_prepared(
    *,
    allow_terminal_artifacts: bool = False,
    source_revision_candidate: dict[str, Any] | None = None,
    ledger_snapshot: dict[str, float] | None = None,
) -> dict[str, Any]:
    source_report_sha = r.sha256_file(r.ARCHITECTURE_REPORT)
    if not r._expected_parent_paths(r.ARMS[0])["contract_path"].is_file():
        raise r.Refuse("global continuation contract is missing; run --prepare-only")
    r._configure_prepared_storage()
    charges = r.latest_charges() if ledger_snapshot is None else dict(ledger_snapshot)
    if set(charges) != set(r.ARMS):
        raise r.Refuse("aggregate ledger does not contain all three prepared arms")
    records: dict[str, Any] = {}
    for arm in r.ARMS:
        context = r._parent_context(arm)
        location = context["locations"]
        attempt = r._effective_attempt(
            r.read_json(location["extension_attempt"]),
            source_report_sha,
            candidate_revision=source_revision_candidate,
        )
        if (
            attempt.get("schema") != "taiji-r2-dialogue-continuation-extension-attempt-v1"
            or attempt.get("arm") != arm
            or attempt.get("parent", {}).get("attempt_sha256") != context["parent_attempt_sha256"]
            or attempt.get("parent_checkpoint", {}).get("receipt_sha256")
            != context["parent_receipt_sha256"]
            or attempt.get("source_compatibility_report_sha256") != source_report_sha
            or attempt.get("source_compatibility") != context["source_compatibility"]
        ):
            raise r.Refuse(f"prepared sidecar no longer matches parent/source contract: {arm}")
        r._require_continuation_source_bindings(attempt, source_report_sha)
        if location["extension_result"].exists() and not allow_terminal_artifacts:
            raise r.Refuse(f"extension result already exists; do not retrain: {arm}")
        progress = r.read_jsonl(location["extension_progress"])
        if not any(event.get("event") == "continuation_prepared" for event in progress):
            raise r.Refuse(f"prepared progress is incomplete: {arm}")
        copied = location["extension_checkpoint_root"] / "pretraining"
        for name in (
            "latest.json",
            context["parent_receipt"]["bundle"].replace(".pt", ".json"),
            context["parent_receipt"]["bundle"],
        ):
            if not (copied / name).is_file():
                raise r.Refuse(f"parent checkpoint fork is incomplete: {arm}/{name}")
        if (
            r.sha256_file(copied / context["parent_receipt"]["bundle"])
            != context["parent_bundle_sha256"]
        ):
            raise r.Refuse(f"parent checkpoint fork changed: {arm}")
        verify_receipt_ancestry(copied, context["parent_receipt_sha256"])
        records[arm] = context
    if any(not 0 <= float(charges[arm]) <= r.PER_ARM_CPU_LIMIT for arm in r.ARMS):
        raise r.Refuse("per-arm CPU ledger exceeds the approved ceiling")
    if sum(float(charges[arm]) for arm in r.ARMS) > r.AGGREGATE_CPU_LIMIT:
        raise r.Refuse("aggregate CPU ledger exceeds the approved ceiling")
    return {"source_report_sha256": source_report_sha, "charges": charges, "parents": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=r.ARMS, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    r.torch.set_num_threads(1)
    validate_prepared(allow_terminal_artifacts=True)
    if args.check_only:
        print("checkpoint ancestry and unchanged source/budget gates passed")
        return
    source_revision = r._source_revision_path()
    manifest = {
        "schema": "taiji-r2-checkpoint-validation-adapter-v1",
        "arm": args.arm,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": r.sha256_file(Path(__file__)),
        "adapter_test_sha256": r.sha256_file(
            r.ROOT / "tests/taiji_native/test_r2_dialogue_checkpoint_resume_adapter_contract.py"
        ),
        "source_revision_sha256": r.sha256_file(source_revision),
        "extension_attempt_sha256": r.sha256_file(r.paths(args.arm)["extension_attempt"]),
        "latest_pointer": r.read_json(
            r.paths(args.arm)["extension_checkpoint_root"] / "pretraining/latest.json"
        ),
        "approved_budget": {"per_arm": r.PER_ARM_CPU_LIMIT, "aggregate": r.AGGREGATE_CPU_LIMIT},
        "utc": r.utc_now(),
        "purpose": "accept an intact receipt chain descended from parent; preserve all source, data, CPU, writer and fresh-restore gates",
    }
    r._write_json_once(
        r.REPORTS_DIR / f"taiji_r2_checkpoint_adapter_{args.arm}_{uuid.uuid4().hex}.json", manifest
    )
    # An explicit, manifest-bound validation adapter; the training implementation is unchanged.
    r._validate_prepared = validate_prepared
    result = r.execute_arm(args.arm, resume=True)
    print(result["status"])


if __name__ == "__main__":
    main()

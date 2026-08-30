"""Run the deterministic R5B-S1 capability-registry canary."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.capability_registry import (  # noqa: E402
    CapabilityBundle,
    CapabilityRegistry,
)
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5b-s1-capability-registry-v1"


def _read_v2_bundle() -> CapabilityBundle:
    return CapabilityBundle(
        capability_id="workspace.read",
        schema={"type": "object", "properties": {"path": {"type": "string"}}},
        permissions=("workspace.read",),
        executor_id="workspace.read.v2",
        executor_version="2.0.0",
    )


def evaluate_registry(workspace_path: Path | None = None) -> dict[str, object]:
    workspace_context = (
        nullcontext(workspace_path)
        if workspace_path is not None
        else TemporaryDirectory(prefix="seed-r5b-s1-")
    )
    with workspace_context as raw_workspace:
        workspace = Path(raw_workspace)
        with (workspace / "README.md").open("w", encoding="utf-8", newline="") as handle:
            handle.write("registry canary\n")
        environment = WorkbenchEnvironment(workspace)
        registry = environment.capability_registry
        enabled_ids = {
            item.capability_id
            for item in environment.capability_snapshot.capabilities
            if item.enabled
        }
        registered_ids = {item.capability_id for item in registry.snapshot.bundles}
        read_bundle = registry.resolve("workspace.read")
        dispatch = environment.execute_tool("workspace.read", {"path": "README.md"})
        dispatch_payload = dict(environment.last_result)
        parent_snapshot = registry.snapshot_id
        candidate = _read_v2_bundle()
        registry.register(candidate)
        registry.shadow(candidate.bundle_digest)
        replaced = registry.replace(
            read_bundle.bundle_digest,
            candidate.bundle_digest,
            approval_id="approval:r5b-s1",
            expected_snapshot_id=parent_snapshot,
        )
        environment._capability_executors[candidate.executor_id] = environment._read_workspace
        stale = environment.execute_tool(
            "workspace.read",
            {"path": "README.md"},
            capability_registry_snapshot_id=parent_snapshot,
        )
        stale_payload = dict(environment.last_result)
        checkpoint = registry.checkpoint()
        restored = CapabilityRegistry.from_checkpoint(checkpoint)
        checkpoint_roundtrip = restored.snapshot.to_payload() == registry.snapshot.to_payload()
        rolled_back = restored.rollback(
            candidate.bundle_digest,
            expected_snapshot_id=restored.snapshot_id,
        )
        restored_bundle = restored.resolve("workspace.read")
        passed = (
            registered_ids == enabled_ids
            and dispatch.success
            and dispatch_payload.get("content") == "registry canary\n"
            and replaced.status == "active"
            and not stale.success
            and stale_payload.get("error_code") == "stale_capability_registry"
            and checkpoint_roundtrip
            and rolled_back.status == "rolled_back"
            and restored_bundle.bundle_digest == read_bundle.bundle_digest
        )
        return {
            "format": REPORT_FORMAT,
            "registry": {
                "snapshot_id": parent_snapshot,
                "snapshot_revision": registry.snapshot.revision,
                "enabled_capabilities": sorted(enabled_ids),
                "registered_capabilities": sorted(registered_ids),
            },
            "dispatch": {
                "executor_id": read_bundle.executor_id,
                "success": dispatch.success,
                "result_digest": dispatch_payload.get("digest", ""),
                "stale_error_code": stale_payload.get("error_code", ""),
            },
            "replacement": {
                "candidate_bundle_digest": candidate.bundle_digest,
                "status_after_replace": replaced.status,
                "status_after_rollback": rolled_back.status,
                "restored_executor_id": restored_bundle.executor_id,
            },
            "checkpoint": {
                "roundtrip": checkpoint_roundtrip,
                "checkpoint_digest": checkpoint["checkpoint_digest"],
            },
            "gate": {
                "passed": passed,
                "criterion": "registry-backed dispatch covers every enabled Workbench capability, rejects stale bindings, and restores the parent executor after replacement rollback",
            },
            "boundary": "R5B-S1 registry canary only; no cognition ownership, provider selection, training, CUDA, or physical deletion",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5b_s1_capability_registry_20260830.json",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Use a pre-created workspace directory when the host temp ACL is restricted.",
    )
    args = parser.parse_args()
    report = evaluate_registry(args.workspace)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit("R5B-S1 capability registry canary failed")


if __name__ == "__main__":
    main()

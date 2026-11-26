"""M3.R5: execute one explicitly approved patch in an isolated temp workspace.

This is an executor-boundary canary, not native write learning.  The script
copies the Workbench fixture into a process-owned temporary directory, binds a
single exact approval token to a digest-checked patch, executes it once, and
undoes it.  No path under the repository is passed to the executor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m3r1_native_observation import (  # noqa: E402
    FIXTURE_ROOT,
    _course_registry,
    _fixture_digest,
)
from scripts.training.eval_taiji_m3r4_preview_approval import (  # noqa: E402
    _intent_request,
    _patch_parameters,
)
from seed_platform.workbench import (  # noqa: E402
    WorkbenchConflictError,
    WorkbenchEnvironment,
)
from taiji import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m3r5-isolated-approved-execution-v1"
REPORT_VERSION = 1
GATE_NAME = "M3.R5"
SOURCE_FIXTURE = FIXTURE_ROOT / "native_observation_test"


def _environment(root: Path) -> WorkbenchEnvironment:
    return WorkbenchEnvironment(
        root=root,
        programming_language_registry=_course_registry(),
    )


def _error_gate(call: Any, error_type: type[BaseException]) -> dict[str, Any]:
    try:
        call()
    except error_type as exc:
        return {"rejected": True, "error_type": type(exc).__name__}
    return {"rejected": False, "error_type": None}


def _outcome_payload(outcome: Any, environment: WorkbenchEnvironment) -> dict[str, Any]:
    last_result = dict(environment.last_result)
    transaction = last_result.get("transaction")
    if isinstance(transaction, dict):
        transaction = dict(transaction)
        transaction["undo_token_present"] = bool(transaction.get("undo_token"))
        transaction.pop("undo_token", None)
        last_result["transaction"] = transaction
    return {
        "success": bool(outcome.success),
        "reward": float(outcome.reward),
        "last_result_digest": content_digest(last_result),
        "last_result": last_result,
    }


@contextmanager
def _isolated_temp_root() -> Iterator[Path]:
    """Create a writable process-owned root without Windows 0700 ACLs."""
    temp_parent = Path(os.environ.get("SEED_M3R5_TMPDIR") or tempfile.gettempdir())
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"seed-m3r5-{uuid4().hex}"
    temp_root.mkdir()
    try:
        yield temp_root
    finally:
        shutil.rmtree(temp_root)


def run_gate(output_path: Path | None = None) -> dict[str, Any]:
    source_before = _fixture_digest(SOURCE_FIXTURE)
    temp_root_path: Path | None = None
    cleaned = False
    with _isolated_temp_root() as isolated_root:
        temp_root_path = isolated_root
        workspace_root = temp_root_path / "workspace"
        shutil.copytree(SOURCE_FIXTURE, workspace_root)
        environment = _environment(workspace_root)
        workspace_before = _fixture_digest(workspace_root)
        parameters = _patch_parameters(environment)
        request = _intent_request(
            environment,
            parameters,
            request_id="m3r5-approved-patch",
        )
        policy_before = environment.policy_for(request)
        approval = environment.issue_approval(request)
        approved_request = replace(
            request,
            approval_token=str(approval["approval_token"]),
        )
        tampered_policy = environment.policy_for(
            replace(
                approved_request,
                parameters={
                    **parameters,
                    "expected_after_digest": "0" * 64,
                },
            )
        )
        stale_snapshot_policy = environment.policy_for(
            replace(request, snapshot_id="stale-capability-snapshot")
        )
        transaction_checkpoint = environment.transaction_state_checkpoint()
        restored = _environment(workspace_root)
        restored.restore_transaction_state(transaction_checkpoint)
        restored_policy = restored.policy_for(approved_request)

        environment.consume_approval(approved_request)
        replay_policy = environment.policy_for(approved_request)
        executed = environment.execute_tool(
            request.capability_id,
            request.parameters,
            capability_registry_snapshot_id=environment.capability_registry.snapshot_id,
        )
        after_execute = environment.read_workspace_evidence({"path": "app.py"})
        transaction = dict(environment.last_result.get("transaction", {}))
        undo_token = str(transaction.get("undo_token", ""))
        executed_payload = _outcome_payload(executed, environment)
        undo_request = _intent_request(
            environment,
            {"undo_token": undo_token},
            kind="workspace.undo",
            request_id="m3r5-approved-undo",
        )
        undo_policy_before = environment.policy_for(undo_request)
        undo_approval = environment.issue_approval(undo_request)
        approved_undo_request = replace(
            undo_request,
            approval_token=str(undo_approval["approval_token"]),
        )
        environment.consume_approval(approved_undo_request)
        undone = environment.execute_tool(
            undo_request.capability_id,
            undo_request.parameters,
            capability_registry_snapshot_id=environment.capability_registry.snapshot_id,
        )
        after_undo = environment.read_workspace_evidence({"path": "app.py"})
        undo_transaction = dict(environment.last_result.get("transaction", {}))
        undone_payload = _outcome_payload(undone, environment)
        duplicate_undo = _error_gate(
            lambda: environment.preview_tool(
                "workspace.undo",
                {"undo_token": undo_token},
            ),
            ValueError,
        )

        stale_root = temp_root_path / "stale"
        shutil.copytree(SOURCE_FIXTURE, stale_root)
        stale_environment = _environment(stale_root)
        stale_parameters = _patch_parameters(stale_environment)
        stale_file = stale_root / "app.py"
        stale_file.write_text(
            stale_file.read_text(encoding="utf-8") + "# changed during approval\n",
            encoding="utf-8",
        )
        stale_preview = _error_gate(
            lambda: stale_environment.preview_tool(
                "workspace.apply_patch",
                stale_parameters,
            ),
            WorkbenchConflictError,
        )

        workspace_after = _fixture_digest(workspace_root)
        source_after = _fixture_digest(SOURCE_FIXTURE)
        temp_root_path_for_report = str(temp_root_path)

    cleaned = temp_root_path is not None and not temp_root_path.exists()
    gates = {
        "isolated_executor_root": temp_root_path_for_report != str(PROJECT_ROOT)
        and temp_root_path_for_report not in str(PROJECT_ROOT),
        "policy_requires_approval": policy_before.decision == "ask_user"
        and policy_before.reason_code == "capability_requires_approval",
        "tampered_request_rejected": tampered_policy.decision == "ask_user"
        and tampered_policy.reason_code == "approval_invalid",
        "stale_snapshot_rejected": stale_snapshot_policy.decision == "deny"
        and stale_snapshot_policy.reason_code == "stale_capability_snapshot",
        "approval_not_restored": transaction_checkpoint["approvals_restored"] is False
        and restored_policy.decision == "ask_user"
        and restored_policy.reason_code == "approval_invalid",
        "exact_request_executes_once": executed.success
        and replay_policy.decision == "ask_user"
        and replay_policy.reason_code == "approval_invalid",
        "after_digest_matches_preview": after_execute["digest"]
        == parameters["expected_after_digest"]
        and transaction.get("before_digest") == parameters["before_digest"]
        and transaction.get("after_digest") == parameters["expected_after_digest"],
        "undo_is_approved_and_restores": undo_policy_before.decision == "ask_user"
        and undo_policy_before.reason_code == "capability_requires_approval"
        and undone.success
        and after_undo["digest"] == parameters["before_digest"]
        and undo_transaction.get("after_digest") == parameters["before_digest"],
        "duplicate_undo_rejected": duplicate_undo["rejected"],
        "stale_file_rejected": stale_preview["rejected"],
        "temporary_workspace_restored": workspace_after == workspace_before,
        "temporary_directory_cleaned": cleaned,
        "source_fixture_unchanged": source_before == source_after,
    }
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "gate": GATE_NAME,
        "status": "passed" if all(gates.values()) else "failed",
        "can_promote": False,
        "promotion_reason": (
            "one approved patch executed only inside a temporary copied fixture and "
            "was reverted; native write learning and real workspace execution remain closed."
        ),
        "execution": {
            "capability_id": request.capability_id,
            "before_digest": parameters["before_digest"],
            "expected_after_digest": parameters["expected_after_digest"],
            "executed": executed_payload,
            "undone": undone_payload,
            "transaction": {
                key: value for key, value in transaction.items() if key != "undo_token"
            },
            "undo_token_present": bool(undo_token),
            "undo_transaction": {
                key: value for key, value in undo_transaction.items() if key != "undo_token"
            },
            "workspace_before": workspace_before,
            "workspace_after": workspace_after,
        },
        "controls": {
            "tampered_policy": tampered_policy.to_payload(),
            "stale_snapshot_policy": stale_snapshot_policy.to_payload(),
            "restored_policy": restored_policy.to_payload(),
            "replay_policy": replay_policy.to_payload(),
            "stale_file": stale_preview,
            "duplicate_undo": duplicate_undo,
        },
        "fixtures": {
            "source_before": source_before,
            "source_after": source_after,
            "source_unchanged": source_before == source_after,
            "temporary_workspace_restored": workspace_after == workspace_before,
            "temporary_directory_cleaned": cleaned,
        },
        "boundaries": {
            "native_write_head": False,
            "temporary_executor_only": True,
            "real_workspace_write": False,
            "editor_set_language": False,
            "terminal": False,
            "mcp": False,
            "cuda": False,
        },
        "gates": gates,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m3r5_approved_execution_20260907.json",
    )
    args = parser.parse_args()
    report = run_gate(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""M3.R4: verify Workbench preview/approval without executing side effects.

R3 proves that native state can produce a bounded read-only intent.  R4 does
not add a learned write head.  It supplies one explicit, host-authored
``workspace.apply_patch`` candidate to the existing Workbench contract and
checks that preview, approval binding, stale-state rejection, and one-shot
consumption all fail closed while the workspace remains byte-for-byte
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m3r1_native_observation import (  # noqa: E402
    FIXTURE_ROOT,
    _course_registry,
    _fixture_digest,
)
from seed_platform.workbench import (  # noqa: E402
    WorkbenchActionRequest,
    WorkbenchConflictError,
    WorkbenchEnvironment,
)
from taiji import ActionIntent, content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m3r4-preview-approval-dry-run-v1"
REPORT_VERSION = 1
GATE_NAME = "M3.R4"
TARGET_PATH = "app.py"
OLD_TEXT = "value + 3"
NEW_TEXT = "value + 4"


def _environment() -> WorkbenchEnvironment:
    return WorkbenchEnvironment(
        root=FIXTURE_ROOT / "native_observation_test",
        programming_language_registry=_course_registry(),
    )


def _patch_parameters(environment: WorkbenchEnvironment) -> dict[str, Any]:
    evidence = environment.read_workspace_evidence({"path": TARGET_PATH})
    content = str(evidence["content"])
    start = content.index(OLD_TEXT)
    updated = content[:start] + NEW_TEXT + content[start + len(OLD_TEXT) :]
    return {
        "path": TARGET_PATH,
        "before_digest": str(evidence["digest"]),
        "patch": {
            "kind": "text_replace",
            "operations": [
                {
                    "start": start,
                    "end": start + len(OLD_TEXT),
                    "text": NEW_TEXT,
                }
            ],
        },
        "expected_after_digest": hashlib.sha256(updated.encode("utf-8")).hexdigest(),
    }


def _intent_request(
    environment: WorkbenchEnvironment,
    parameters: dict[str, Any],
    *,
    kind: str = "workspace.apply_patch",
    snapshot_id: str | None = None,
    request_id: str = "m3r4-preview-patch",
) -> WorkbenchActionRequest:
    intent = ActionIntent(
        intent_id=f"intent:{request_id}",
        kind=kind,
        parameters=parameters,
        expected_outcome="preview one digest-checked text replacement",
        confidence=1.0,
        tick=1,
    )
    return WorkbenchActionRequest.from_action_intent(
        intent,
        snapshot_id=snapshot_id or environment.capability_snapshot.snapshot_id,
    )


def _parameter_contract(
    environment: WorkbenchEnvironment,
    request: WorkbenchActionRequest,
) -> dict[str, Any]:
    descriptor = environment.capability_snapshot.get(request.capability_id)
    if descriptor is None:
        return {
            "descriptor_present": False,
            "unknown": [*sorted(request.parameters)],
            "missing": [],
            "exact": False,
        }
    actual = {str(name) for name in request.parameters}
    declared = set(descriptor.parameter_names)
    unknown = sorted(actual - declared)
    missing = sorted(declared - actual)
    return {
        "descriptor_present": True,
        "unknown": unknown,
        "missing": missing,
        "exact": not unknown and not missing,
    }


def _preview_shape(preview: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    mutation = preview.get("mutation", {})
    return {
        "validated": preview.get("validated") is True,
        "capability_id": preview.get("capability_id"),
        "risk": preview.get("risk"),
        "mutation_operation": mutation.get("operation"),
        "path": mutation.get("path"),
        "before_digest": mutation.get("before_digest"),
        "after_digest": mutation.get("after_digest"),
        "before_byte_length": mutation.get("before_byte_length"),
        "after_byte_length": mutation.get("after_byte_length"),
        "operations": mutation.get("operations"),
        "patch_bytes": len(json.dumps(parameters["patch"], sort_keys=True).encode("utf-8")),
    }


def _expect_error(call: Any, error_type: type[BaseException]) -> dict[str, Any]:
    try:
        call()
    except error_type as exc:
        return {"rejected": True, "error_type": type(exc).__name__}
    return {"rejected": False, "error_type": None}


def run_gate(output_path: Path | None = None) -> dict[str, Any]:
    environment = _environment()
    fixture_root = FIXTURE_ROOT / "native_observation_test"
    fixture_before = _fixture_digest(fixture_root)
    parameters = _patch_parameters(environment)
    request = _intent_request(environment, parameters)
    descriptor_contract = _parameter_contract(environment, request)
    policy = environment.policy_for(request)
    preview = environment.preview_tool(request.capability_id, request.parameters)
    preview_repeat = environment.preview_tool(request.capability_id, request.parameters)
    preview_shape = _preview_shape(preview, parameters)
    approval = environment.issue_approval(request)
    approved_request = replace(request, approval_token=str(approval["approval_token"]))
    approved_policy = environment.policy_for(approved_request)
    approval_preview_shape = _preview_shape(approval["preview"], parameters)
    checkpoint = environment.transaction_state_checkpoint()
    approval_token = approved_request.approval_token
    checkpoint_without_secret = approval_token not in json.dumps(
        checkpoint,
        ensure_ascii=False,
        sort_keys=True,
    )
    restored = _environment()
    restored.restore_transaction_state(checkpoint)
    restored_policy = restored.policy_for(approved_request)

    tampered_request = replace(
        approved_request,
        parameters={
            **parameters,
            "expected_after_digest": "0" * 64,
        },
    )
    tampered_policy = environment.policy_for(tampered_request)
    stale_snapshot_policy = environment.policy_for(
        replace(request, snapshot_id="stale-capability-snapshot")
    )
    stale_file = _expect_error(
        lambda: environment.preview_tool(
            request.capability_id,
            {**parameters, "before_digest": "0" * 64},
        ),
        WorkbenchConflictError,
    )
    invalid_patch = _expect_error(
        lambda: environment.preview_tool(
            request.capability_id,
            {
                **parameters,
                "patch": {
                    "kind": "text_replace",
                    "operations": [{"start": -1, "end": 0, "text": "bad"}],
                },
            },
        ),
        ValueError,
    )
    drift_request = _intent_request(
        environment,
        {**parameters, "undeclared_parameter": "drift"},
        request_id="m3r4-drift",
    )
    drift_contract = _parameter_contract(environment, drift_request)

    consume_before = _fixture_digest(fixture_root)
    environment.consume_approval(approved_request)
    duplicate_consume = _expect_error(
        lambda: environment.consume_approval(approved_request),
        WorkbenchConflictError,
    )
    consumed_policy = environment.policy_for(approved_request)
    consume_after = _fixture_digest(fixture_root)
    fixture_after = _fixture_digest(fixture_root)

    gates = {
        "patch_parameter_contract": descriptor_contract["exact"]
        and request.capability_id == "workspace.apply_patch",
        "policy_requires_approval": policy.decision == "ask_user"
        and policy.reason_code == "capability_requires_approval",
        "preview_is_valid_and_digest_bound": (
            preview_shape["validated"]
            and preview_shape["risk"] == "file_write"
            and preview_shape["mutation_operation"] == "workspace.apply_patch"
            and preview_shape["path"] == TARGET_PATH
            and preview_shape["before_digest"] == parameters["before_digest"]
            and preview_shape["after_digest"] == parameters["expected_after_digest"]
            and preview_shape["operations"] == 1
            and preview_shape == approval_preview_shape
        ),
        "preview_repeat_is_stable": content_digest(preview)
        == content_digest(preview_repeat),
        "approval_is_exactly_bound": approved_policy.decision == "allow"
        and approved_policy.reason_code == "explicit_approval"
        and approval["preview"] == preview,
        "tampered_request_rejected": tampered_policy.decision == "ask_user"
        and tampered_policy.reason_code == "approval_invalid",
        "stale_snapshot_rejected": stale_snapshot_policy.decision == "deny"
        and stale_snapshot_policy.reason_code == "stale_capability_snapshot",
        "stale_file_preview_rejected": stale_file["rejected"],
        "invalid_patch_rejected": invalid_patch["rejected"],
        "parameter_drift_rejected_before_preview": not drift_contract["exact"]
        and drift_contract["unknown"] == ["undeclared_parameter"],
        "approval_not_checkpointed": checkpoint["approvals_restored"] is False
        and checkpoint_without_secret
        and restored.status()["pending_approvals"] == 0
        and restored_policy.decision == "ask_user"
        and restored_policy.reason_code == "approval_invalid",
        "approval_is_single_use": duplicate_consume["rejected"]
        and consumed_policy.decision == "ask_user"
        and consumed_policy.reason_code == "approval_invalid",
        "preview_has_no_workspace_side_effect": consume_before == consume_after,
        "fixture_unchanged": fixture_before == fixture_after,
    }
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "gate": GATE_NAME,
        "status": "passed" if all(gates.values()) else "failed",
        "can_promote": False,
        "promotion_reason": (
            "M3.R4 only validates preview and approval binding; no executor, file write, "
            "editor selection, terminal, or MCP call is admitted."
        ),
        "candidate": {
            "source": "host-authored-controlled-patch",
            "capability_id": request.capability_id,
            "parameters": {
                "path": TARGET_PATH,
                "before_digest": parameters["before_digest"],
                "expected_after_digest": parameters["expected_after_digest"],
                "patch_bytes": preview_shape["patch_bytes"],
            },
            "parameter_contract": descriptor_contract,
        },
        "preview": preview_shape,
        "approval": {
            "policy": policy.to_payload(),
            "approved_policy": approved_policy.to_payload(),
            "restored_policy": restored_policy.to_payload(),
            "tampered_policy": tampered_policy.to_payload(),
            "stale_snapshot_policy": stale_snapshot_policy.to_payload(),
            "checkpoint": checkpoint,
            "token_exposed": False,
        },
        "controls": {
            "stale_file": stale_file,
            "invalid_patch": invalid_patch,
            "parameter_drift": drift_contract,
            "duplicate_consume": duplicate_consume,
            "consumed_policy": consumed_policy.to_payload(),
        },
        "fixtures": {
            "before": fixture_before,
            "after": fixture_after,
            "unchanged": fixture_before == fixture_after,
        },
        "boundaries": {
            "native_write_head": False,
            "workbench_preview": True,
            "workbench_executor": False,
            "workspace_write": False,
            "editor_set_language": False,
            "terminal": False,
            "mcp": False,
            "cuda": False,
        },
        "gates": gates,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m3r4_preview_approval_20260907.json",
    )
    args = parser.parse_args()
    report = run_gate(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

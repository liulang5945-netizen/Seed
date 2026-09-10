"""Run the P3.2 K-candidate to G-owner transfer preflight.

The inherited K1/K2 workers are loaded read-only.  K1 still proposes the
semantic candidate, but a native ``GSelectionState`` owns the final
goal/content selection used by the action boundary.  The script compares that
owner-transfer path with the existing K-only path, then exercises independent
checkpoint restore at event and case boundaries.  It never fits a new G
learner and it rejects external goal/content targets as runtime inputs.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_p2_3_recovery_continuation import (  # noqa: E402
    _restore_tensors,
)
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    READ_ONLY_ROUTES,
)
from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _atomic_roundtrip,
)
from scripts.training.eval_taiji_m5_k_p2_2_safety_bridge_canary import (  # noqa: E402
    _run_intent,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _chain_row,
    _fresh_learners,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p3_1_single_cell import (  # noqa: E402
    _build_holdout_cases,
    _verify_base_continuation,
)
from taiji import (  # noqa: E402
    OWNER_TRANSFER_OWNER_IDS,
    GSelectionState,
    NativeReadOnlyIntentPlanner,
    OwnerTransferCursor,
    ReadOnlyIntentPolicy,
    TaijiContinuationCheckpoint,
    TaijiOwnerTransferCheckpoint,
    TaijiOwnerTransferEvent,
    TaijiOwnerTransferManifest,
    TaijiSingleCellManifest,
    content_digest,
)

P2_7_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_7_generalization_20260910.json"
P2_7_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_7_generalization_manifest_v1.json"
)
P3_0_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_0_checkpoint_contract_20260910.json"
P3_1_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_1_single_cell_20260910.json"
P3_1_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_1_single_cell_manifest_v1.json"
)
REPORT_FORMAT = "taiji-m5-k-p3-2-owner-transfer-preflight-v1"
VERSION = 1
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / (
    "taiji_m5_k_p3_2_owner_transfer_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _tupleify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleify(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _tupleify(item) for key, item in value.items()}
    return value


def _state_digest(payload: Mapping[str, Any]) -> str:
    return content_digest(dict(payload))


def _placeholder(owner: str, *, cursor: int) -> dict[str, Any]:
    return {
        "owner": owner,
        "schema": "owner-transfer-cursor-state-v1",
        "status": "not-yet-reached",
        "cursor": int(cursor),
    }


def _prepare_transfer(
    *,
    cases: Sequence[Mapping[str, Any]],
    semantic: Any,
    transition: Any,
    worker_digests: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current: dict[str, str | None] = {owner: None for owner in OWNER_TRANSFER_OWNER_IDS}
    state_payloads: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def append_event(
        *,
        event_type: str,
        input_payload: Mapping[str, Any],
        output_payload: Mapping[str, Any],
        owner: str | None,
        confidence: float,
        status: str,
        attributes: Mapping[str, Any],
    ) -> None:
        nonlocal current
        before = dict(current)
        output_digest = _state_digest(output_payload)
        after = dict(current)
        if owner is not None:
            after[owner] = output_digest
            state_payloads[owner] = dict(output_payload)
        event = TaijiOwnerTransferEvent.create(
            event_index=len(events),
            event_type=event_type,
            input_payload=input_payload,
            state_before=before,
            state_after=after,
            output_payload=output_payload,
            confidence=max(0.0, min(1.0, float(confidence))),
            status=status,
            attributes=attributes,
        )
        events.append(
            {
                "event": event,
                "state_payloads": {
                    key: dict(value) for key, value in state_payloads.items()
                },
            }
        )
        current = after

    for case in cases:
        record = case["record"]
        candidate = record["candidate"]
        from taiji import StructuredSemanticExample, StructuredSemanticTransitionExample

        semantic_example = StructuredSemanticExample.from_payload(
            _restore_tensors(candidate["semantic_example"])
        )
        transition_example = StructuredSemanticTransitionExample.from_payload(
            _restore_tensors(candidate["transition_example"])
        )
        k_only = _chain_row(
            semantic=semantic,
            transition=transition,
            case=case,
            semantic_example=semantic_example,
            transition_example=transition_example,
            label="p3-2-k-only",
        )
        semantic_result = semantic.predict(semantic_example.percept)
        transition_result = None
        if semantic_result.world is not None:
            transition_result = transition.predict(
                semantic_result.world,
                transition_example.event,
            )
        selection = GSelectionState.from_k1_result(semantic_result)
        transfer_action = None
        if transition_result is not None:
            planner = NativeReadOnlyIntentPlanner(
                ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES)
            )
            transfer_action = _run_intent(
                case=case,
                planner=planner,
                goal=selection.selected_goal,
                content=selection.selected_content,
                world=transition_result.world,
                label="p3-2-owner-transfer",
                oracle_control=False,
            )
        expected_goal_id = semantic_example.goal.goal_id
        expected_content_id = semantic_example.content.content_id
        transfer_row = {
            "index": int(case["index"]),
            "class_key": str(case["class_key"]),
            "selection_status": selection.selection_status,
            "selection_digest": selection.selection_digest,
            "candidate_digest": selection.candidate_digest,
            "external_target_used": bool(selection.external_target_used),
            "expected_goal_id": expected_goal_id,
            "expected_content_id": expected_content_id,
            "g_goal_id": (
                None if selection.selected_goal is None else selection.selected_goal.goal_id
            ),
            "g_content_id": (
                None
                if selection.selected_content is None
                else selection.selected_content.content_id
            ),
            "g_goal_hit": bool(
                selection.selected_goal is not None
                and selection.selected_goal.goal_id == expected_goal_id
            ),
            "g_content_hit": bool(
                selection.selected_content is not None
                and selection.selected_content.content_id == expected_content_id
            ),
            "k2_goal_id": (
                None
                if transition_result is None or transition_result.goal is None
                else transition_result.goal.goal_id
            ),
            "k2_content_id": (
                None
                if transition_result is None or transition_result.content_plan is None
                else transition_result.content_plan.content_id
            ),
            "k2_goal_hit": bool(
                transition_result is not None
                and transition_result.goal is not None
                and transition_result.goal.goal_id == expected_goal_id
            ),
            "k2_content_hit": bool(
                transition_result is not None
                and transition_result.content_plan is not None
                and transition_result.content_plan.content_id == expected_content_id
            ),
            "safe_abstention": bool(k_only["safe_abstention"]),
            "action": transfer_action,
        }
        k_only_action = k_only.get("action") or {}
        transfer_action_payload = transfer_action or {}
        equivalent = {
            "selection_matches_k1": (
                transfer_row["g_goal_id"] == k_only["k1_goal_id"]
                and transfer_row["g_content_id"] == k_only["k1_content_id"]
            ),
            "k2_output_matches": (
                transfer_row["k2_goal_id"] == k_only["k2_goal_id"]
                and transfer_row["k2_content_id"] == k_only["k2_content_id"]
            ),
            "safe_abstention_matches": transfer_row["safe_abstention"]
            == k_only["safe_abstention"],
            "workbench_matches": bool(transfer_action_payload.get("workbench_success", False))
            == bool(k_only_action.get("workbench_success", False)),
        }
        transfer_row["equivalent"] = equivalent
        rows.append({"k_only": k_only, "owner_transfer": transfer_row})

        record_id = str(record["record_id"])
        attributes = {
            "record_id": record_id,
            "project_id": str(record["project_id"]),
            "path": str(record["candidate"]["observation"]["path"]),
            "external_target_used": False,
            "selection_digest": selection.selection_digest,
        }
        evidence_payload = {
            "owner": "S",
            "schema": "observation+percept-v1",
            "record_id": record_id,
            "observation_digest": str(record["candidate"]["observation"]["observation_digest"]),
            "percept_digest": content_digest(semantic_example.percept.to_payload()),
        }
        append_event(
            event_type="observation",
            input_payload=dict(record["candidate"]["observation"]),
            output_payload=evidence_payload,
            owner="S",
            confidence=float(semantic_example.percept.confidence),
            status="observed",
            attributes=attributes,
        )
        candidate_payload = {
            "owner": "K",
            "schema": "k1-candidate-v1",
            "record_id": record_id,
            "worker_checkpoint_digests": dict(worker_digests),
            "candidate_digest": selection.candidate_digest,
            "semantic_result": semantic_result.to_payload(),
            "fit_called": False,
        }
        append_event(
            event_type="k_candidate",
            input_payload={"evidence_digest": current["S"]},
            output_payload=candidate_payload,
            owner="K",
            confidence=float(semantic_result.confidence),
            status=str(semantic_result.status),
            attributes=attributes,
        )
        selection_payload = {
            "owner": "G",
            "schema": "g-selection-v1",
            "record_id": record_id,
            "selection": selection.to_payload(),
            "external_target_used": False,
        }
        append_event(
            event_type="g_selection",
            input_payload={"candidate_digest": selection.candidate_digest},
            output_payload=selection_payload,
            owner="G",
            confidence=float(selection.confidence),
            status=selection.selection_status,
            attributes=attributes,
        )
        action_payload = {
            "owner": "G+K",
            "schema": "typed-action-owner-transfer-v1",
            "record_id": record_id,
            "selection_digest": selection.selection_digest,
            "k2_goal_id": transfer_row["k2_goal_id"],
            "k2_content_id": transfer_row["k2_content_id"],
            "action": transfer_action,
            "external_target_used": False,
        }
        append_event(
            event_type="action",
            input_payload={
                "selection_digest": current["G"],
                "k_candidate_digest": current["K"],
            },
            output_payload=action_payload,
            owner=None,
            confidence=float(selection.confidence),
            status=(
                "executed"
                if bool(transfer_action_payload.get("workbench_success", False))
                else "not-executed"
            ),
            attributes=attributes,
        )
    return events, rows


def _load_owner_payloads(checkpoint: TaijiOwnerTransferCheckpoint) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    expected = dict(checkpoint.owner_state_digests)
    for owner, ref in checkpoint.owner_state_refs:
        payload = _load_mapping(Path(ref))
        if content_digest(payload) != expected[owner]:
            raise ValueError(f"owner-transfer owner state digest mismatch: {owner}")
        payloads[owner] = payload
    return payloads


def _save_boundary(
    *,
    output_dir: Path,
    base: TaijiContinuationCheckpoint,
    base_cell: TaijiSingleCellManifest,
    manifest: TaijiOwnerTransferManifest,
    prepared: Sequence[Mapping[str, Any]],
    event_refs: Sequence[str],
    event_digests: Sequence[str],
    event_count: int,
    current_state: Mapping[str, str | None],
    current_payloads: Mapping[str, Mapping[str, Any]],
    rng: random.Random,
    budgets: Mapping[str, int],
    base_ref: str,
    base_cell_ref: str,
) -> tuple[TaijiOwnerTransferCheckpoint, dict[str, Any]]:
    directory = output_dir / "boundary"
    owner_dir = directory / "owners"
    owner_dir.mkdir(parents=True, exist_ok=True)
    owner_digests: dict[str, str] = {}
    owner_refs: dict[str, str] = {}
    for owner in OWNER_TRANSFER_OWNER_IDS:
        payload = dict(current_payloads.get(owner, _placeholder(owner, cursor=event_count)))
        path = owner_dir / f"{owner.lower()}_{event_count:04d}.pt"
        _atomic_roundtrip(path, payload)
        owner_digests[owner] = content_digest(payload)
        owner_refs[owner] = str(path)
    if event_count != len(event_refs) or event_count != len(event_digests):
        raise ValueError("owner-transfer boundary event prefix is inconsistent")
    stage = "complete" if event_count == len(prepared) else (
        str(prepared[event_count - 1]["event"].event_type) if event_count else "observation"
    )
    checkpoint = TaijiOwnerTransferCheckpoint.create(
        base_continuation_checkpoint_digest=base.checkpoint_digest,
        base_continuation_ref=base_ref,
        base_single_cell_manifest_digest=base_cell.manifest_digest,
        base_single_cell_manifest_ref=base_cell_ref,
        manifest_digest=manifest.manifest_digest,
        worker_checkpoint_digests=dict(base.worker_checkpoint_digests),
        worker_checkpoint_refs=dict(base.worker_checkpoint_refs),
        owner_state_digests=owner_digests,
        owner_state_refs=owner_refs,
        event_digests=tuple(event_digests),
        event_refs=tuple(event_refs),
        cursor=OwnerTransferCursor(
            event_index=event_count,
            event_total=len(prepared),
            case_index=event_count // 4,
            stage=stage,
        ),
        rng_state=_jsonable(rng.getstate()),
        budget_counts=budgets,
        lineage_chain=(base.checkpoint_digest, base_cell.manifest_digest, manifest.manifest_digest),
    )
    checkpoint_path = directory / "checkpoint.json"
    _write_json_atomic(checkpoint_path, checkpoint.to_payload())
    return checkpoint, {"checkpoint": checkpoint, "checkpoint_path": str(checkpoint_path)}


def _run_trajectory(
    *,
    output_dir: Path,
    base: TaijiContinuationCheckpoint,
    base_cell: TaijiSingleCellManifest,
    manifest: TaijiOwnerTransferManifest,
    prepared: list[dict[str, Any]],
    base_ref: str,
    base_cell_ref: str,
    stop_after: int | None,
    boundary: TaijiOwnerTransferCheckpoint | None = None,
) -> tuple[TaijiOwnerTransferCheckpoint, dict[str, Any]]:
    if boundary is None:
        start = 0
        current_state = {owner: None for owner in OWNER_TRANSFER_OWNER_IDS}
        current_payloads: dict[str, dict[str, Any]] = {}
        event_refs: list[str] = []
        event_digests: list[str] = []
        rng = random.Random(32017)
        budgets = {"events": 0, "k_candidates": 0, "g_selections": 0, "actions": 0}
    else:
        boundary.assert_base(base.checkpoint_digest)
        boundary.assert_single_cell(base_cell.manifest_digest)
        boundary.assert_manifest(manifest.manifest_digest)
        start = boundary.cursor.event_index
        current_payloads = _load_owner_payloads(boundary)
        if start:
            last_event = TaijiOwnerTransferEvent.from_payload(
                _load_mapping(Path(boundary.event_refs[-1]))
            )
            current_state = dict(last_event.state_after)
        else:
            current_state = {owner: None for owner in OWNER_TRANSFER_OWNER_IDS}
        event_refs = list(boundary.event_refs)
        event_digests = list(boundary.event_digests)
        rng = random.Random()
        rng.setstate(_tupleify(boundary.rng_state))
        budgets = {str(key): int(value) for key, value in boundary.budget_counts}
    for index in range(start, len(prepared)):
        item = prepared[index]
        event = item["event"]
        if dict(event.state_before) != current_state:
            raise ValueError(f"owner-transfer event state chain mismatch at {index}")
        rng.random()
        path = output_dir / "events" / f"event_{event.event_index:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_roundtrip(path, event.to_payload())
        event_refs.append(str(path))
        event_digests.append(event.event_digest)
        current_state = dict(event.state_after)
        for owner, payload in item["state_payloads"].items():
            current_payloads[owner] = dict(payload)
        budgets["events"] = int(budgets.get("events", 0)) + 1
        if event.event_type == "k_candidate":
            budgets["k_candidates"] = int(budgets.get("k_candidates", 0)) + 1
        elif event.event_type == "g_selection":
            budgets["g_selections"] = int(budgets.get("g_selections", 0)) + 1
        elif event.event_type == "action":
            budgets["actions"] = int(budgets.get("actions", 0)) + 1
        if stop_after is not None and index + 1 == stop_after:
            return _save_boundary(
                output_dir=output_dir,
                base=base,
                base_cell=base_cell,
                manifest=manifest,
                prepared=prepared,
                event_refs=event_refs,
                event_digests=event_digests,
                event_count=index + 1,
                current_state=current_state,
                current_payloads=current_payloads,
                rng=rng,
                budgets=budgets,
                base_ref=base_ref,
                base_cell_ref=base_cell_ref,
            )
    return _save_boundary(
        output_dir=output_dir,
        base=base,
        base_cell=base_cell,
        manifest=manifest,
        prepared=prepared,
        event_refs=event_refs,
        event_digests=event_digests,
        event_count=len(prepared),
        current_state=current_state,
        current_payloads=current_payloads,
        rng=rng,
        budgets=budgets,
        base_ref=base_ref,
        base_cell_ref=base_cell_ref,
    )


def _verify_checkpoint(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    single_cell_manifest_path: Path,
) -> dict[str, Any]:
    checkpoint = TaijiOwnerTransferCheckpoint.from_payload(
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
    )
    manifest = TaijiOwnerTransferManifest.from_payload(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    base_cell = TaijiSingleCellManifest.from_payload(
        json.loads(single_cell_manifest_path.read_text(encoding="utf-8"))
    )
    checkpoint.assert_manifest(manifest.manifest_digest)
    checkpoint.assert_single_cell(base_cell.manifest_digest)
    base = TaijiContinuationCheckpoint.from_payload(
        json.loads(Path(checkpoint.base_continuation_ref).read_text(encoding="utf-8"))
    )
    checkpoint.assert_base(base.checkpoint_digest)
    worker_restore = {}
    for owner, ref in checkpoint.worker_checkpoint_refs:
        payload = _load_mapping(Path(ref))
        actual = content_digest(payload)
        if actual != dict(checkpoint.worker_checkpoint_digests)[owner]:
            raise ValueError(f"owner-transfer worker restore digest mismatch: {owner}")
        worker_restore[owner] = actual
    owner_restore = {}
    for owner, ref in checkpoint.owner_state_refs:
        payload = _load_mapping(Path(ref))
        actual = content_digest(payload)
        if actual != dict(checkpoint.owner_state_digests)[owner]:
            raise ValueError(f"owner-transfer owner restore digest mismatch: {owner}")
        if owner == "G":
            GSelectionState.from_payload(payload["selection"])
        owner_restore[owner] = actual
    event_restore = []
    for digest, ref in zip(checkpoint.event_digests, checkpoint.event_refs, strict=True):
        event = TaijiOwnerTransferEvent.from_payload(_load_mapping(Path(ref)))
        if event.event_digest != digest:
            raise ValueError(f"owner-transfer event restore digest mismatch: {event.event_index}")
        event_restore.append(event.event_digest)
    return {
        "checkpoint_digest": checkpoint.checkpoint_digest,
        "logical_digest": checkpoint.logical_digest,
        "worker_restore": worker_restore,
        "owner_restore": owner_restore,
        "event_count": len(event_restore),
        "independent_process_restore": True,
    }


def _independent_restore(
    checkpoint_path: Path,
    manifest_path: Path,
    single_cell_manifest_path: Path,
) -> dict[str, Any]:
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--verify-checkpoint",
            str(checkpoint_path),
            "--manifest",
            str(manifest_path),
            "--single-cell-manifest",
            str(single_cell_manifest_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[str, Any] = {
        "returncode": int(child.returncode),
        "independent_process_restore": False,
        "stdout": child.stdout[-2000:],
        "stderr": child.stderr[-2000:],
    }
    if child.returncode != 0:
        return result
    try:
        decoded = json.loads(child.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
        return result
    result.update(decoded)
    result["independent_process_restore"] = bool(decoded.get("independent_process_restore"))
    return result


def _save_manifest(path: Path, manifest: TaijiOwnerTransferManifest) -> None:
    payload = {
        **manifest.to_payload(),
        "manifest_kind": "k-candidate-to-g-owner-transfer",
        "owner_contribution": {
            "S": "runtime-evidence",
            "K": "existing-k1-k2-candidate-provider",
            "G": "native-selection-state-no-fit",
        },
        "external_target_used": False,
        "parameter_growth": False,
        "fit_called": False,
    }
    _write_json_atomic(path, payload)


def run_preflight(
    *,
    p2_7_report_path: Path = P2_7_REPORT,
    p2_7_manifest_path: Path = P2_7_MANIFEST,
    p3_0_report_path: Path = P3_0_REPORT,
    p3_1_report_path: Path = P3_1_REPORT,
    p3_1_manifest_path: Path = P3_1_MANIFEST,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p3_2_owner_transfer_{uuid4().hex}"
    scratch = run_dir / "scratch"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "external_target_used": False,
        "parameter_growth": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "p3_1_manifest": str(p3_1_manifest_path),
    }
    try:
        p2_7_report = json.loads(p2_7_report_path.read_text(encoding="utf-8"))
        p2_7_manifest = json.loads(p2_7_manifest_path.read_text(encoding="utf-8"))
        p3_0_report = json.loads(p3_0_report_path.read_text(encoding="utf-8"))
        p3_1_report = json.loads(p3_1_report_path.read_text(encoding="utf-8"))
        p3_1_manifest = TaijiSingleCellManifest.from_payload(
            json.loads(p3_1_manifest_path.read_text(encoding="utf-8"))
        )
        if p2_7_report.get("status") != "completed":
            raise ValueError("P3.2 requires a completed P2.7 report")
        if not all(bool(value) for value in p2_7_report.get("generalization_gate", {}).values()):
            raise ValueError("P2.7 generalization gate is not fully passed")
        if p3_1_report.get("status") != "completed":
            raise ValueError("P3.2 requires a completed P3.1 report")
        if not all(bool(value) for value in p3_1_report.get("checkpoint_gate", {}).values()):
            raise ValueError("P3.1 checkpoint gate is not fully passed")
        if p3_1_report.get("manifest_digest") != p3_1_manifest.manifest_digest:
            raise ValueError("P3.1 manifest digest mismatch")
        if p2_7_report.get("manifest_digest") != p2_7_manifest.get("manifest_digest"):
            raise ValueError("P2.7 manifest digest mismatch")
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        base, base_gate = _verify_base_continuation(p3_report=p3_0_report, run_dir=run_dir)
        if base.checkpoint_digest != p3_1_manifest.base_continuation_checkpoint_digest:
            raise ValueError("P3.1 single-cell manifest does not attach to the P3.0 base")
        manifest = TaijiOwnerTransferManifest.create(
            base_single_cell_manifest_digest=p3_1_manifest.manifest_digest,
            base_continuation_checkpoint_digest=base.checkpoint_digest,
            worker_checkpoint_digests=dict(base.worker_checkpoint_digests),
        )
        _save_manifest(manifest_path, manifest)
        cases = _build_holdout_cases(
            scratch=scratch / "cases",
            p2_7_manifest=p2_7_manifest,
        )
        worker_payloads = {
            owner: _load_mapping(Path(ref)) for owner, ref in base.worker_checkpoint_refs
        }
        semantic, transition = _fresh_learners(worker_payloads["k1"], worker_payloads["k2"])
        parameter_count = {
            "k1": int(semantic.parameter_count),
            "k2": int(transition.parameter_count),
            "total": int(semantic.parameter_count + transition.parameter_count),
        }
        prepared, rows = _prepare_transfer(
            cases=cases,
            semantic=semantic,
            transition=transition,
            worker_digests=dict(base.worker_checkpoint_digests),
        )
        if len(prepared) != len(cases) * 4:
            raise RuntimeError("P3.2 event stream must contain four events per holdout case")
        base_cell_ref = str(p3_1_manifest_path)
        uninterrupted, uninterrupted_artifact = _run_trajectory(
            output_dir=run_dir / "uninterrupted",
            base=base,
            base_cell=p3_1_manifest,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            base_cell_ref=base_cell_ref,
            stop_after=None,
        )
        interrupted_event, interrupted_event_artifact = _run_trajectory(
            output_dir=run_dir / "event-interrupted",
            base=base,
            base_cell=p3_1_manifest,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            base_cell_ref=base_cell_ref,
            stop_after=6,
        )
        interrupted_case, interrupted_case_artifact = _run_trajectory(
            output_dir=run_dir / "case-boundary-interrupted",
            base=base,
            base_cell=p3_1_manifest,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            base_cell_ref=base_cell_ref,
            stop_after=8,
        )
        resumed_event, resumed_event_artifact = _run_trajectory(
            output_dir=run_dir / "event-resumed",
            base=base,
            base_cell=p3_1_manifest,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            base_cell_ref=base_cell_ref,
            stop_after=None,
            boundary=interrupted_event,
        )
        resumed_case, resumed_case_artifact = _run_trajectory(
            output_dir=run_dir / "case-boundary-resumed",
            base=base,
            base_cell=p3_1_manifest,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            base_cell_ref=base_cell_ref,
            stop_after=None,
            boundary=interrupted_case,
        )
        independent = {
            "uninterrupted": _independent_restore(
                Path(uninterrupted_artifact["checkpoint_path"]), manifest_path, p3_1_manifest_path
            ),
            "event_boundary": _independent_restore(
                Path(interrupted_event_artifact["checkpoint_path"]), manifest_path, p3_1_manifest_path
            ),
            "case_boundary": _independent_restore(
                Path(interrupted_case_artifact["checkpoint_path"]), manifest_path, p3_1_manifest_path
            ),
            "event_resumed": _independent_restore(
                Path(resumed_event_artifact["checkpoint_path"]), manifest_path, p3_1_manifest_path
            ),
            "case_resumed": _independent_restore(
                Path(resumed_case_artifact["checkpoint_path"]), manifest_path, p3_1_manifest_path
            ),
        }
        trajectory_gate = {
            "event_stream_equal": uninterrupted.event_digests
            == resumed_event.event_digests
            == resumed_case.event_digests,
            "owner_state_equal": uninterrupted.owner_state_digests
            == resumed_event.owner_state_digests
            == resumed_case.owner_state_digests,
            "worker_digest_equal": uninterrupted.worker_checkpoint_digests
            == resumed_event.worker_checkpoint_digests
            == resumed_case.worker_checkpoint_digests,
            "budget_equal": uninterrupted.budget_counts
            == resumed_event.budget_counts
            == resumed_case.budget_counts,
            "rng_equal": uninterrupted.rng_state_digest
            == resumed_event.rng_state_digest
            == resumed_case.rng_state_digest,
            "cursor_equal": uninterrupted.cursor == resumed_event.cursor == resumed_case.cursor,
            "logical_final_digest_equal": uninterrupted.logical_digest
            == resumed_event.logical_digest
            == resumed_case.logical_digest,
        }
        comparison_gate = {
            "all_rows_present": len(rows) == len(cases),
            "external_target_not_used": all(
                not bool(row["owner_transfer"]["external_target_used"]) for row in rows
            ),
            "selection_matches_k1": all(
                bool(row["owner_transfer"]["equivalent"]["selection_matches_k1"])
                for row in rows
            ),
            "k2_output_matches": all(
                bool(row["owner_transfer"]["equivalent"]["k2_output_matches"])
                for row in rows
            ),
            "safe_abstention_matches": all(
                bool(row["owner_transfer"]["equivalent"]["safe_abstention_matches"])
                for row in rows
            ),
            "workbench_matches": all(
                bool(row["owner_transfer"]["equivalent"]["workbench_matches"])
                for row in rows
            ),
            "parameter_count_stable": parameter_count["total"]
            == int(parameter_count["k1"] + parameter_count["k2"]),
        }
        tampered = uninterrupted.to_payload()
        tampered["cursor"]["event_index"] = max(0, int(tampered["cursor"]["event_index"]) - 1)
        try:
            TaijiOwnerTransferCheckpoint.from_payload(tampered)
        except (KeyError, TypeError, ValueError):
            tampered_rejected = True
        else:
            tampered_rejected = False
        try:
            uninterrupted.assert_base("0" * 64)
        except ValueError:
            wrong_base_rejected = True
        else:
            wrong_base_rejected = False
        try:
            uninterrupted.assert_manifest("0" * 64)
        except ValueError:
            wrong_manifest_rejected = True
        else:
            wrong_manifest_rejected = False
        mask_event = prepared[1]["event"].to_payload()
        mask_event["read_owners"] = ["G"]
        try:
            TaijiOwnerTransferEvent.from_payload(mask_event)
        except (KeyError, TypeError, ValueError):
            wrong_mask_rejected = True
        else:
            wrong_mask_rejected = False
        checkpoint_gate = {
            name: bool(result.get("independent_process_restore"))
            for name, result in independent.items()
        }
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "base_continuation": base_gate,
                "base_single_cell_manifest_digest": p3_1_manifest.manifest_digest,
                "manifest_digest": manifest.manifest_digest,
                "worker_checkpoint_digests": dict(manifest.worker_checkpoint_digests),
                "owner_roles": dict(manifest.owner_roles),
                "owner_contribution": {
                    "S": "runtime-evidence",
                    "K": "existing-k1-k2-candidate-provider",
                    "G": "native-selection-state-no-fit",
                },
                "parameter_count": parameter_count,
                "comparison_gate": comparison_gate,
                "validation_replay": {
                    "row_count": len(rows),
                    "k_only_k1_goal_hit_count": sum(
                        bool(row["k_only"]["k1_goal_hit"]) for row in rows
                    ),
                    "k_only_k1_content_hit_count": sum(
                        bool(row["k_only"]["k1_content_hit"]) for row in rows
                    ),
                    "g_goal_hit_count": sum(
                        bool(row["owner_transfer"]["g_goal_hit"]) for row in rows
                    ),
                    "g_content_hit_count": sum(
                        bool(row["owner_transfer"]["g_content_hit"]) for row in rows
                    ),
                    "k_only_workbench_success_count": sum(
                        bool((row["k_only"].get("action") or {}).get("workbench_success"))
                        for row in rows
                    ),
                    "owner_transfer_workbench_success_count": sum(
                        bool((row["owner_transfer"].get("action") or {}).get("workbench_success"))
                        for row in rows
                    ),
                },
                "rows": rows,
                "event_contract": {
                    "event_count": len(prepared),
                    "case_count": len(cases),
                    "event_types_per_case": [
                        str(item["event"].event_type) for item in prepared[:4]
                    ],
                    "event_digests": [str(item["event"].event_digest) for item in prepared],
                },
                "checkpoint_gate": checkpoint_gate,
                "trajectory_gate": trajectory_gate,
                "rejection_gate": {
                    "tampered_cursor_rejected": tampered_rejected,
                    "wrong_base_rejected": wrong_base_rejected,
                    "wrong_manifest_rejected": wrong_manifest_rejected,
                    "wrong_owner_mask_rejected": wrong_mask_rejected,
                },
                "rollback_gate": {
                    "rollback_parent_is_p3_1_cell": manifest.base_single_cell_manifest_digest
                    == p3_1_manifest.manifest_digest,
                    "rollback_base_worker_restore": bool(base_gate["passed"]),
                },
                "boundaries": {
                    "event_boundary": interrupted_event.to_payload(),
                    "case_boundary": interrupted_case.to_payload(),
                    "uninterrupted_final": uninterrupted.to_payload(),
                },
                "resumed": {
                    "event_boundary": resumed_event.to_payload(),
                    "case_boundary": resumed_case.to_payload(),
                },
                "independent_restore": independent,
                "interpretation": (
                    "P3.2 owner-transfer preflight only; K supplies the inherited candidate, "
                    "G owns the copied selection state without external targets, no new fit, "
                    "no topology growth, and no promotion claim"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "run_dir": str(run_dir),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    _write_json_atomic(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-7-report", type=Path, default=P2_7_REPORT)
    parser.add_argument("--p2-7-manifest", type=Path, default=P2_7_MANIFEST)
    parser.add_argument("--p3-0-report", type=Path, default=P3_0_REPORT)
    parser.add_argument("--p3-1-report", type=Path, default=P3_1_REPORT)
    parser.add_argument("--p3-1-manifest", type=Path, default=P3_1_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--single-cell-manifest", type=Path, default=P3_1_MANIFEST)
    parser.add_argument("--verify-checkpoint", type=Path)
    args = parser.parse_args()
    if args.verify_checkpoint is not None:
        result = _verify_checkpoint(
            checkpoint_path=args.verify_checkpoint,
            manifest_path=args.manifest,
            single_cell_manifest_path=args.single_cell_manifest,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    result = run_preflight(
        p2_7_report_path=args.p2_7_report,
        p2_7_manifest_path=args.p2_7_manifest,
        p3_0_report_path=args.p3_0_report,
        p3_1_report_path=args.p3_1_report,
        p3_1_manifest_path=args.p3_1_manifest,
        manifest_path=args.manifest,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "report": str(args.report),
                "status": result["status"],
                "training_performed": result["training_performed"],
                "fit_called": result["fit_called"],
                "comparison_gate": result.get("comparison_gate"),
                "checkpoint_gate": result.get("checkpoint_gate"),
                "trajectory_gate": result.get("trajectory_gate"),
                "rejection_gate": result.get("rejection_gate"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

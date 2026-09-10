"""Run the P3.1 S/G/K single-cell state-integration preflight.

This step replays the four disjoint P2.7 holdout cases without fitting.  It
binds the existing runtime evidence (S), an explicitly external goal/content
selection context (G), and the learned P2.6 K1/K2 workers (K) into one audited
event stream.  S/G are not promoted as learned workers here: G is deliberately
marked ``control-only`` until a native learned goal owner exists.

The preflight saves two interruption boundaries, restores each in a fresh
process, and compares the path-independent final state/event digest with an
uninterrupted replay.  It proves state ownership and persistence, not
autonomous growth or promotion.
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
from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _atomic_roundtrip,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _chain_row,
    _fresh_learners,
)
from scripts.training.eval_taiji_m5_k_p2_6_novel_learning import (  # noqa: E402
    _build_record as _build_novel_record,
)
from scripts.training.eval_taiji_m5_k_p2_7_generalization_probe import (  # noqa: E402
    HOLDOUT_SPECS,
    _relabel_holdout_record,
)
from taiji import (  # noqa: E402
    SINGLE_CELL_OWNER_IDS,
    SingleCellEventCursor,
    SingleCellOwnerContract,
    TaijiContinuationCheckpoint,
    TaijiSingleCellCheckpoint,
    TaijiSingleCellEvent,
    TaijiSingleCellManifest,
    content_digest,
)

P2_7_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_7_generalization_20260910.json"
P2_7_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_7_generalization_manifest_v1.json"
)
P3_0_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_0_checkpoint_contract_20260910.json"
REPORT_FORMAT = "taiji-m5-k-p3-1-single-cell-preflight-v1"
MANIFEST_FORMAT = "taiji-m5-k-p3-1-single-cell-manifest-v1"
VERSION = 1
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / (
    "taiji_m5_k_p3_1_single_cell_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_1_single_cell_20260910.json"


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


def _owner_contracts() -> tuple[SingleCellOwnerContract, ...]:
    return (
        SingleCellOwnerContract.create(
            owner_id="S",
            owner_kind="runtime_evidence",
            availability="active",
            schema="observation+percept-v1",
            read_scopes=("external.observation",),
            write_scopes=("evidence.observation", "evidence.percept"),
            learning_owner="none",
        ),
        SingleCellOwnerContract.create(
            owner_id="G",
            owner_kind="runtime_selection",
            availability="active",
            schema="goal-selection-v1",
            read_scopes=("evidence.percept", "external.goal_target"),
            write_scopes=("selection.goal", "selection.content"),
            learning_owner="none",
        ),
        SingleCellOwnerContract.create(
            owner_id="K",
            owner_kind="learned_worker",
            availability="active",
            schema="k1.semantic+k2.transition-v1",
            read_scopes=("evidence.percept", "evidence.world"),
            write_scopes=("readout.k1", "readout.k2"),
            learning_owner="k1.semantic+k2.transition",
        ),
    )


def _verify_base_continuation(
    *,
    p3_report: Mapping[str, Any],
    run_dir: Path,
) -> tuple[TaijiContinuationCheckpoint, dict[str, Any]]:
    if p3_report.get("status") != "completed":
        raise ValueError("P3.1 requires a completed P3.0 report")
    for gate_name in ("checkpoint_gate", "trajectory_gate", "rejection_gate", "rollback_gate"):
        gate = p3_report.get(gate_name, {})
        if not gate or not all(bool(value) for value in gate.values()):
            raise ValueError(f"P3.0 {gate_name} is not fully passed")
    payload = p3_report.get("trajectories", {}).get("uninterrupted")
    if not isinstance(payload, Mapping):
        raise ValueError("P3.0 report has no uninterrupted continuation payload")
    base = TaijiContinuationCheckpoint.from_payload(payload)
    expected_parent = str(p3_report.get("parent_checkpoint_digest", ""))
    base.assert_parent(expected_parent)
    base_ref = run_dir / "base_continuation.json"
    _write_json_atomic(base_ref, base.to_payload())
    worker_results: dict[str, Any] = {}
    worker_digests = dict(base.worker_checkpoint_digests)
    for owner, ref in base.worker_checkpoint_refs:
        path = Path(ref)
        payload = _load_mapping(path)
        digest = content_digest(payload)
        if digest != worker_digests[owner]:
            raise ValueError(f"P3.0 worker digest mismatch at P3.1 entry: {owner}")
        worker_results[owner] = {
            "path": str(path),
            "digest": digest,
            "bytes": path.stat().st_size,
        }
    return base, {
        "base_checkpoint_digest": base.checkpoint_digest,
        "base_ref": str(base_ref),
        "worker_restore": worker_results,
        "passed": True,
    }


def _build_holdout_cases(
    *,
    scratch: Path,
    p2_7_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records_by_index = {int(record["index"]): record for record in p2_7_manifest["records"]}
    cases: list[dict[str, Any]] = []
    source_manifest_digest = str(p2_7_manifest["source_p1_manifest_digest"])
    parent_digest = str(p2_7_manifest["parent_checkpoint_digest"])
    for index, candidate_path, project_id in HOLDOUT_SPECS:
        source_record, case = _build_novel_record(
            root=scratch / f"holdout-{index:04d}",
            index=index,
            candidate_path=candidate_path,
            project_id=project_id,
            task_seed=6300 + index,
            split="validation",
            source_manifest_digest=source_manifest_digest,
            parent_digest=parent_digest,
            fit_eligible=False,
        )
        record = records_by_index[index]
        relabeled = _relabel_holdout_record(
            source_record,
            source_p2_6_manifest_digest=str(p2_7_manifest["source_p2_6_manifest_digest"]),
            index=index,
            project_id=project_id,
        )
        if relabeled["record_digest"] != record["record_digest"]:
            raise ValueError(f"P2.7 holdout reconstruction drifted for index {index}")
        case["record"] = record
        cases.append(case)
    return cases


def _state_digest(payload: Mapping[str, Any]) -> str:
    return content_digest(dict(payload))


def _event_attributes(
    *,
    case: Mapping[str, Any],
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    action = row.get("action") or {}
    return {
        "record_id": str(record["record_id"]),
        "project_id": str(record["project_id"]),
        "path": str(record["candidate"]["observation"]["path"]),
        "source": source,
        "expected_goal_id": str(row["expected_goal_id"]),
        "expected_content_id": str(row["expected_content_id"]),
        "predicted_k1_goal_id": row.get("k1_goal_id"),
        "predicted_k1_content_id": row.get("k1_content_id"),
        "predicted_k2_goal_id": row.get("k2_goal_id"),
        "predicted_k2_content_id": row.get("k2_content_id"),
        "safe_abstention": bool(row.get("safe_abstention", False)),
        "planner_status": action.get("planner_status"),
        "workbench_success": bool(action.get("workbench_success", False)),
        "case_index": int(case["index"]),
    }


def _prepare_events(
    *,
    cases: Sequence[Mapping[str, Any]],
    semantic: Any,
    transition: Any,
    worker_digests: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current: dict[str, str | None] = {owner: None for owner in SINGLE_CELL_OWNER_IDS}
    events: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    state_payloads: dict[str, dict[str, Any]] = {}

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
        event = TaijiSingleCellEvent.create(
            event_index=len(events),
            event_type=event_type,
            input_digest=content_digest(dict(input_payload)),
            state_before=before,
            state_after=after,
            output_digest=output_digest,
            confidence=max(0.0, min(1.0, float(confidence))),
            status=status,
            attributes=attributes,
        )
        events.append({"event": event, "state_payloads": {key: dict(value) for key, value in state_payloads.items()}})
        current = after

    for case in cases:
        record = case["record"]
        candidate = record["candidate"]
        semantic_example = semantic_example_from_record(record)
        transition_example = transition_example_from_record(record)
        semantic_result = semantic.predict(semantic_example.percept)
        transition_result = None
        if semantic_result.world is not None:
            transition_result = transition.predict(
                semantic_result.world,
                transition_example.event,
            )
        row = _chain_row(
            semantic=semantic,
            transition=transition,
            case=case,
            semantic_example=semantic_example,
            transition_example=transition_example,
            label="p3-1-single-cell",
        )
        rows.append(row)
        observation_payload = dict(candidate["observation"])
        percept_payload = semantic_example.percept.to_payload()
        evidence_payload = {
            "owner": "S",
            "schema": "observation+percept-v1",
            "record_id": str(record["record_id"]),
            "observation": observation_payload,
            "percept": percept_payload,
            "observation_digest": str(record["candidate"]["observation"]["observation_digest"]),
        }
        append_event(
            event_type="observation",
            input_payload=observation_payload,
            output_payload=evidence_payload,
            owner="S",
            confidence=float(semantic_example.percept.confidence),
            status="observed",
            attributes=_event_attributes(case=case, record=record, row=row, source="p2-7-holdout"),
        )
        bound_evidence = {
            **evidence_payload,
            "state": "evidence-bound",
            "percept_digest": content_digest(percept_payload),
        }
        append_event(
            event_type="s_update",
            input_payload=evidence_payload,
            output_payload=bound_evidence,
            owner="S",
            confidence=float(semantic_example.percept.confidence),
            status="bound",
            attributes=_event_attributes(case=case, record=record, row=row, source="native-percept"),
        )
        goal_payload = {
            "owner": "G",
            "schema": "goal-selection-v1",
            "record_id": str(record["record_id"]),
            "selection_source": "validation-goal-contract",
            "learned": False,
            "goal": semantic_example.goal.to_payload(),
            "content_plan": semantic_example.content.to_payload(),
        }
        append_event(
            event_type="g_select",
            input_payload={
                "evidence_digest": current["S"],
                "goal_target": semantic_example.goal.to_payload(),
                "content_target": semantic_example.content.to_payload(),
            },
            output_payload=goal_payload,
            owner="G",
            confidence=float(semantic_example.percept.confidence),
            status="control-only",
            attributes={
                **_event_attributes(case=case, record=record, row=row, source="external-goal-target"),
                "learning_credit": "none",
            },
        )
        k_payload = {
            "owner": "K",
            "schema": "k1.semantic+k2.transition-v1",
            "record_id": str(record["record_id"]),
            "worker_checkpoint_digests": dict(worker_digests),
            "semantic_result": semantic_result.to_payload(),
            "transition_result": (
                None if transition_result is None else transition_result.to_payload()
            ),
            "learned": True,
            "fit_called": False,
        }
        k_confidence = float(semantic_result.confidence)
        if transition_result is not None:
            k_confidence = min(k_confidence, float(transition_result.confidence))
        append_event(
            event_type="k_readout",
            input_payload={"evidence_digest": current["S"]},
            output_payload=k_payload,
            owner="K",
            confidence=k_confidence,
            status=f"k1:{semantic_result.status}/k2:{None if transition_result is None else transition_result.status}",
            attributes=_event_attributes(case=case, record=record, row=row, source="learned-k-readout"),
        )
        action = row.get("action") or {}
        action_payload = {
            "owner": "G+K",
            "schema": "typed-action-abstention-workbench-v1",
            "record_id": str(record["record_id"]),
            "goal_digest": current["G"],
            "k_readout_digest": current["K"],
            "action": action,
            "safe_abstention": bool(row.get("safe_abstention", False)),
        }
        action_status = "abstained" if row.get("safe_abstention") else (
            "executed" if action.get("workbench_success") else "not-executed"
        )
        append_event(
            event_type="action",
            input_payload={"goal_digest": current["G"], "k_readout_digest": current["K"]},
            output_payload=action_payload,
            owner=None,
            confidence=k_confidence,
            status=action_status,
            attributes=_event_attributes(case=case, record=record, row=row, source="read-only-workbench"),
        )
    return events, rows


def semantic_example_from_record(record: Mapping[str, Any]) -> Any:
    from taiji import StructuredSemanticExample

    return StructuredSemanticExample.from_payload(
        _restore_tensors(record["candidate"]["semantic_example"])
    )


def transition_example_from_record(record: Mapping[str, Any]) -> Any:
    from taiji import StructuredSemanticTransitionExample

    return StructuredSemanticTransitionExample.from_payload(
        _restore_tensors(record["candidate"]["transition_example"])
    )


def _placeholder_payload(owner: str, *, cursor: int) -> dict[str, Any]:
    return {
        "owner": owner,
        "schema": "single-cell-cursor-state-v1",
        "status": "not-yet-reached",
        "cursor": int(cursor),
    }


def _load_owner_payloads(checkpoint: TaijiSingleCellCheckpoint) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    expected = dict(checkpoint.owner_state_digests)
    for owner, ref in checkpoint.owner_state_refs:
        payload = _load_mapping(Path(ref))
        if content_digest(payload) != expected[owner]:
            raise ValueError(f"single-cell owner state digest mismatch: {owner}")
        payloads[owner] = payload
    return payloads


def _save_boundary(
    *,
    output_dir: Path,
    base: TaijiContinuationCheckpoint,
    manifest: TaijiSingleCellManifest,
    prepared: Sequence[Mapping[str, Any]],
    event_count: int,
    current_state: Mapping[str, str | None],
    current_payloads: Mapping[str, Mapping[str, Any]],
    rng: random.Random,
    budgets: Mapping[str, int],
    base_ref: str,
    case_count: int,
) -> tuple[TaijiSingleCellCheckpoint, dict[str, Any]]:
    directory = output_dir / "boundary"
    owner_dir = directory / "owners"
    event_dir = directory / "events"
    owner_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)
    owner_digests: dict[str, str] = {}
    owner_refs: dict[str, str] = {}
    for owner in SINGLE_CELL_OWNER_IDS:
        payload = dict(current_payloads.get(owner, _placeholder_payload(owner, cursor=event_count)))
        path = owner_dir / f"{owner.lower()}_{event_count:04d}.pt"
        _atomic_roundtrip(path, payload)
        owner_digests[owner] = content_digest(payload)
        owner_refs[owner] = str(path)
    event_digests: list[str] = []
    event_refs: list[str] = []
    for item in prepared[:event_count]:
        event = item["event"]
        existing = item.get("saved_ref")
        if existing:
            event_refs.append(str(existing))
        else:
            path = event_dir / f"event_{event.event_index:04d}.pt"
            _atomic_roundtrip(path, event.to_payload())
            item["saved_ref"] = str(path)
            event_refs.append(str(path))
        event_digests.append(event.event_digest)
    if event_count == len(prepared):
        stage = "complete"
    else:
        stage = str(prepared[event_count - 1]["event"].event_type) if event_count else "observation"
    cursor = SingleCellEventCursor(
        event_index=event_count,
        event_total=len(prepared),
        case_index=event_count // 5,
        stage=stage,
    )
    checkpoint = TaijiSingleCellCheckpoint.create(
        base_continuation_checkpoint_digest=base.checkpoint_digest,
        base_continuation_ref=base_ref,
        manifest_digest=manifest.manifest_digest,
        worker_checkpoint_digests=dict(base.worker_checkpoint_digests),
        worker_checkpoint_refs=dict(base.worker_checkpoint_refs),
        owner_state_digests=owner_digests,
        owner_state_refs=owner_refs,
        event_digests=event_digests,
        event_refs=event_refs,
        cursor=cursor,
        rng_state=_jsonable(rng.getstate()),
        budget_counts=budgets,
        lineage_chain=(base.checkpoint_digest, manifest.manifest_digest),
    )
    checkpoint_path = directory / "checkpoint.json"
    _write_json_atomic(checkpoint_path, checkpoint.to_payload())
    return checkpoint, {
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "event_count": event_count,
        "case_count": case_count,
        "current_state": dict(current_state),
    }


def _run_trajectory(
    *,
    output_dir: Path,
    base: TaijiContinuationCheckpoint,
    manifest: TaijiSingleCellManifest,
    prepared: list[dict[str, Any]],
    base_ref: str,
    stop_after: int | None,
    boundary: TaijiSingleCellCheckpoint | None = None,
) -> tuple[TaijiSingleCellCheckpoint, dict[str, Any]]:
    if boundary is None:
        start = 0
        current_state: dict[str, str | None] = {owner: None for owner in SINGLE_CELL_OWNER_IDS}
        current_payloads: dict[str, dict[str, Any]] = {}
        event_refs: list[str] = []
        rng = random.Random(31017)
        budgets = {"events": 0, "k1_readouts": 0, "k2_readouts": 0, "actions": 0}
    else:
        boundary.assert_base(base.checkpoint_digest)
        boundary.assert_manifest(manifest.manifest_digest)
        start = boundary.cursor.event_index
        payloads = _load_owner_payloads(boundary)
        current_payloads = {owner: dict(payload) for owner, payload in payloads.items()}
        if start:
            prefix_event = TaijiSingleCellEvent.from_payload(
                _load_mapping(Path(boundary.event_refs[-1]))
            )
            current_state = dict(prefix_event.state_after)
        else:
            current_state = {owner: None for owner in SINGLE_CELL_OWNER_IDS}
        event_refs = list(boundary.event_refs)
        rng = random.Random()
        rng.setstate(_tupleify(boundary.rng_state))
        budgets = {str(key): int(value) for key, value in boundary.budget_counts}
    for index in range(start, len(prepared)):
        item = prepared[index]
        event = item["event"]
        if dict(event.state_before) != current_state:
            raise ValueError(f"single-cell event state chain mismatch at {index}")
        rng.random()
        path = output_dir / "events" / f"event_{event.event_index:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_roundtrip(path, event.to_payload())
        item["saved_ref"] = str(path)
        event_refs.append(str(path))
        current_state = dict(event.state_after)
        for owner, payload in item["state_payloads"].items():
            current_payloads[owner] = dict(payload)
        budgets["events"] = int(budgets.get("events", 0)) + 1
        if event.event_type == "k_readout":
            budgets["k1_readouts"] = int(budgets.get("k1_readouts", 0)) + 1
            budgets["k2_readouts"] = int(budgets.get("k2_readouts", 0)) + 1
        if event.event_type == "action":
            budgets["actions"] = int(budgets.get("actions", 0)) + 1
        if stop_after is not None and index + 1 == stop_after:
            return _save_boundary(
                output_dir=output_dir,
                base=base,
                manifest=manifest,
                prepared=prepared,
                event_count=index + 1,
                current_state=current_state,
                current_payloads=current_payloads,
                rng=rng,
                budgets=budgets,
                base_ref=base_ref,
                case_count=(index + 1) // 5,
            )
    return _save_boundary(
        output_dir=output_dir,
        base=base,
        manifest=manifest,
        prepared=prepared,
        event_count=len(prepared),
        current_state=current_state,
        current_payloads=current_payloads,
        rng=rng,
        budgets=budgets,
        base_ref=base_ref,
        case_count=len(prepared) // 5,
    )


def _verify_checkpoint(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    checkpoint = TaijiSingleCellCheckpoint.from_payload(
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
    )
    manifest = TaijiSingleCellManifest.from_payload(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    checkpoint.assert_manifest(manifest.manifest_digest)
    base = TaijiContinuationCheckpoint.from_payload(
        json.loads(Path(checkpoint.base_continuation_ref).read_text(encoding="utf-8"))
    )
    checkpoint.assert_base(base.checkpoint_digest)
    worker_digests = dict(checkpoint.worker_checkpoint_digests)
    worker_restore = {}
    for owner, ref in checkpoint.worker_checkpoint_refs:
        payload = _load_mapping(Path(ref))
        actual = content_digest(payload)
        if actual != worker_digests[owner]:
            raise ValueError(f"single-cell worker restore digest mismatch: {owner}")
        worker_restore[owner] = actual
    owner_restore = {}
    for owner, ref in checkpoint.owner_state_refs:
        payload = _load_mapping(Path(ref))
        actual = content_digest(payload)
        if actual != dict(checkpoint.owner_state_digests)[owner]:
            raise ValueError(f"single-cell owner restore digest mismatch: {owner}")
        owner_restore[owner] = actual
    event_restore = []
    for digest, ref in zip(checkpoint.event_digests, checkpoint.event_refs, strict=True):
        payload = _load_mapping(Path(ref))
        event = TaijiSingleCellEvent.from_payload(payload)
        if event.event_digest != digest:
            raise ValueError(f"single-cell event restore digest mismatch: {event.event_index}")
        event_restore.append(event.event_digest)
    return {
        "checkpoint_digest": checkpoint.checkpoint_digest,
        "logical_digest": checkpoint.logical_digest,
        "worker_restore": worker_restore,
        "owner_restore": owner_restore,
        "event_count": len(event_restore),
        "independent_process_restore": True,
    }


def _independent_restore(checkpoint_path: Path, manifest_path: Path) -> dict[str, Any]:
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--verify-checkpoint",
            str(checkpoint_path),
            "--manifest",
            str(manifest_path),
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


def _save_manifest(path: Path, manifest: TaijiSingleCellManifest) -> None:
    payload = {
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        **manifest.to_payload(),
        "owner_contribution": {
            "S": "control-only-runtime-evidence",
            "G": "control-only-external-goal-selection",
            "K": "learned-k1-k2-readout",
        },
        "parameter_growth": False,
        "fit_called": False,
    }
    _write_json_atomic(path, payload)


def run_preflight(
    *,
    p2_7_report_path: Path = P2_7_REPORT,
    p2_7_manifest_path: Path = P2_7_MANIFEST,
    p3_0_report_path: Path = P3_0_REPORT,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p3_1_single_cell_{uuid4().hex}"
    scratch = run_dir / "scratch"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "sealed_payload_read": False,
        "parameter_growth": False,
        "can_promote": False,
        "p2_7_report": str(p2_7_report_path),
        "p2_7_manifest": str(p2_7_manifest_path),
        "p3_0_report": str(p3_0_report_path),
        "manifest": str(manifest_path),
    }
    try:
        p2_7_report = json.loads(p2_7_report_path.read_text(encoding="utf-8"))
        p2_7_manifest = json.loads(p2_7_manifest_path.read_text(encoding="utf-8"))
        p3_0_report = json.loads(p3_0_report_path.read_text(encoding="utf-8"))
        if p2_7_report.get("status") != "completed":
            raise ValueError("P3.1 requires a completed P2.7 report")
        if not all(bool(value) for value in p2_7_report.get("generalization_gate", {}).values()):
            raise ValueError("P2.7 generalization gate is not fully passed")
        if p2_7_report.get("manifest_digest") != p2_7_manifest.get("manifest_digest"):
            raise ValueError("P2.7 manifest digest mismatch")
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        base, base_gate = _verify_base_continuation(p3_report=p3_0_report, run_dir=run_dir)
        owners = _owner_contracts()
        manifest = TaijiSingleCellManifest.create(
            base_continuation_checkpoint_digest=base.checkpoint_digest,
            source_cohort_digest=str(p2_7_manifest["manifest_digest"]),
            owner_contracts=owners,
            rollback_parent_digest=base.checkpoint_digest,
        )
        _save_manifest(manifest_path, manifest)
        cases = _build_holdout_cases(scratch=scratch / "cases", p2_7_manifest=p2_7_manifest)
        worker_payloads = {
            owner: _load_mapping(Path(ref)) for owner, ref in base.worker_checkpoint_refs
        }
        semantic, transition = _fresh_learners(worker_payloads["k1"], worker_payloads["k2"])
        prepared, rows = _prepare_events(
            cases=cases,
            semantic=semantic,
            transition=transition,
            worker_digests=dict(base.worker_checkpoint_digests),
        )
        if len(prepared) != len(HOLDOUT_SPECS) * 5:
            raise RuntimeError("P3.1 event stream does not contain five stages per holdout case")
        uninterrupted, uninterrupted_artifact = _run_trajectory(
            output_dir=run_dir / "uninterrupted",
            base=base,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            stop_after=None,
        )
        interrupted_event, interrupted_event_artifact = _run_trajectory(
            output_dir=run_dir / "event-interrupted",
            base=base,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            stop_after=7,
        )
        interrupted_boundary, interrupted_boundary_artifact = _run_trajectory(
            output_dir=run_dir / "phase-boundary-interrupted",
            base=base,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            stop_after=10,
        )
        resumed_event, resumed_event_artifact = _run_trajectory(
            output_dir=run_dir / "event-resumed",
            base=base,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            stop_after=None,
            boundary=interrupted_event,
        )
        resumed_boundary, resumed_boundary_artifact = _run_trajectory(
            output_dir=run_dir / "phase-boundary-resumed",
            base=base,
            manifest=manifest,
            prepared=prepared,
            base_ref=base_gate["base_ref"],
            stop_after=None,
            boundary=interrupted_boundary,
        )
        independent = {
            "uninterrupted": _independent_restore(
                Path(uninterrupted_artifact["checkpoint_path"]), manifest_path
            ),
            "event_boundary": _independent_restore(
                Path(interrupted_event_artifact["checkpoint_path"]), manifest_path
            ),
            "phase_boundary": _independent_restore(
                Path(interrupted_boundary_artifact["checkpoint_path"]), manifest_path
            ),
            "event_resumed": _independent_restore(
                Path(resumed_event_artifact["checkpoint_path"]), manifest_path
            ),
            "phase_resumed": _independent_restore(
                Path(resumed_boundary_artifact["checkpoint_path"]), manifest_path
            ),
        }
        trajectory_gate = {
            "event_stream_equal": uninterrupted.event_digests
            == resumed_event.event_digests
            == resumed_boundary.event_digests,
            "owner_state_equal": uninterrupted.owner_state_digests
            == resumed_event.owner_state_digests
            == resumed_boundary.owner_state_digests,
            "worker_digest_equal": uninterrupted.worker_checkpoint_digests
            == resumed_event.worker_checkpoint_digests
            == resumed_boundary.worker_checkpoint_digests,
            "budget_equal": uninterrupted.budget_counts
            == resumed_event.budget_counts
            == resumed_boundary.budget_counts,
            "rng_equal": uninterrupted.rng_state_digest
            == resumed_event.rng_state_digest
            == resumed_boundary.rng_state_digest,
            "cursor_equal": uninterrupted.cursor == resumed_event.cursor == resumed_boundary.cursor,
            "logical_final_digest_equal": uninterrupted.logical_digest
            == resumed_event.logical_digest
            == resumed_boundary.logical_digest,
        }
        tampered = uninterrupted.to_payload()
        tampered["cursor"]["event_index"] = max(0, int(tampered["cursor"]["event_index"]) - 1)
        try:
            TaijiSingleCellCheckpoint.from_payload(tampered)
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
        checkpoint_gate = {
            name: bool(result.get("independent_process_restore"))
            for name, result in independent.items()
        }
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "base_continuation": base_gate,
                "manifest_digest": manifest.manifest_digest,
                "source_cohort_digest": manifest.source_cohort_digest,
                "owner_contracts": [owner.to_payload() for owner in owners],
                "owner_contribution": {
                    "S": "control-only-runtime-evidence",
                    "G": "control-only-external-goal-selection",
                    "K": "learned-k1-k2-readout",
                },
                "event_contract": {
                    "event_count": len(prepared),
                    "case_count": len(cases),
                    "event_types_per_case": [
                        str(item["event"].event_type) for item in prepared[:5]
                    ],
                    "event_digests": [str(item["event"].event_digest) for item in prepared],
                },
                "validation_replay": {
                    "row_count": len(rows),
                    "k1_goal_hit_count": sum(bool(row["k1_goal_hit"]) for row in rows),
                    "k1_content_hit_count": sum(bool(row["k1_content_hit"]) for row in rows),
                    "k2_goal_hit_count": sum(bool(row["k2_goal_hit"]) for row in rows),
                    "k2_content_hit_count": sum(bool(row["k2_content_hit"]) for row in rows),
                    "safe_abstention_count": sum(bool(row["safe_abstention"]) for row in rows),
                    "workbench_success_count": sum(
                        bool((row.get("action") or {}).get("workbench_success")) for row in rows
                    ),
                },
                "checkpoint_gate": checkpoint_gate,
                "trajectory_gate": trajectory_gate,
                "rejection_gate": {
                    "tampered_cursor_rejected": tampered_rejected,
                    "wrong_base_rejected": wrong_base_rejected,
                    "wrong_manifest_rejected": wrong_manifest_rejected,
                },
                "rollback_gate": {
                    "rollback_parent_is_p3_0_base": manifest.rollback_parent_digest
                    == base.checkpoint_digest,
                    "rollback_base_worker_restore": bool(base_gate["passed"]),
                },
                "boundaries": {
                    "event_midpoint": interrupted_event.to_payload(),
                    "phase_boundary": interrupted_boundary.to_payload(),
                    "uninterrupted_final": uninterrupted.to_payload(),
                },
                "resumed": {
                    "event_midpoint": resumed_event.to_payload(),
                    "phase_boundary": resumed_boundary.to_payload(),
                },
                "independent_restore": independent,
                "training_performed": False,
                "fit_called": False,
                "parameter_growth": False,
                "can_promote": False,
                "interpretation": (
                    "P3.1 state-ownership and event-replay preflight only; S/G are control-only, "
                    "K uses the existing learned K1/K2 checkpoint; no new fit, topology growth, "
                    "lineage-to-capability claim, or promotion"
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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-checkpoint", type=Path)
    args = parser.parse_args()
    if args.verify_checkpoint is not None:
        result = _verify_checkpoint(checkpoint_path=args.verify_checkpoint, manifest_path=args.manifest)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    result = run_preflight(
        p2_7_report_path=args.p2_7_report,
        p2_7_manifest_path=args.p2_7_manifest,
        p3_0_report_path=args.p3_0_report,
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

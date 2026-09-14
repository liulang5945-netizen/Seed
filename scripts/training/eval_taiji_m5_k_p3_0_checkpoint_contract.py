"""Run the P3.0 resumable-continuation checkpoint contract.

This is the first P3 step after P2.7 local generalization.  It does not add
S/G workers or grow topology.  It starts from the saved P2.6 learned K arm,
then compares one uninterrupted K1/K2 continuation with two fresh-process
resumptions: a wake midpoint and a replay-phase boundary.  The checkpoint
contract carries phase cursor, ordered experience stream, RNG state, budgets,
worker digests, parent lineage, and rollback evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _fit_direct,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _candidate_examples,
    _fresh_learners,
    _independent_restore,
    _load_mapping,
    _save_arm,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    PILOT_PER_CLASS,
    WORKER_ROOT,
    _context,
    _rebuild_and_verify_manifest,
    _select_balanced_wake,
)
from taiji import (  # noqa: E402
    CONTINUATION_PHASES,
    ContinuationPhaseCursor,
    TaijiContinuationCheckpoint,
    content_digest,
)

P2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
P2_6_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_6_novel_learning_20260910.json"
P2_6_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_6_novel_learning_manifest_v1.json"
)
REPORT_FORMAT = "taiji-m5-k-p3-0-checkpoint-contract-v1"
MANIFEST_FORMAT = "taiji-m5-k-p3-0-checkpoint-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_0_checkpoint_contract_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_0_checkpoint_contract_20260910.json"


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


def _load_worker_payloads(checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    files = checkpoint.get("files")
    digests = checkpoint.get("checkpoint_digests")
    if not isinstance(files, Mapping) or not isinstance(digests, Mapping):
        raise ValueError("worker checkpoint is missing content-addressed files")
    payloads: dict[str, Any] = {}
    for owner in ("k1", "k2"):
        path = Path(str(files[owner]["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"worker checkpoint file is missing: {path}")
        payload = _load_mapping(path)
        if content_digest(payload) != str(digests[owner]):
            raise ValueError(f"worker checkpoint digest mismatch: {owner}")
        payloads[owner] = payload
    return payloads["k1"], payloads["k2"]


def _phase_stream(
    *,
    p1_manifest: Mapping[str, Any],
    p2_report: Mapping[str, Any],
    p2_6_manifest: Mapping[str, Any],
    scratch: Path,
    parent_digest: str,
    bundle: Any,
    projector: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    (
        experiences,
        metadata,
        _validation,
        _validation_metadata,
        verification,
    ) = _rebuild_and_verify_manifest(
        scratch=scratch / "p1",
        manifest=p1_manifest,
        parent_digest=parent_digest,
        bundle=bundle,
        projector=projector,
    )
    if not verification["passed"]:
        raise RuntimeError("P1 manifest reconstruction failed for P3.0")
    wake, wake_metadata = _select_balanced_wake(experiences, metadata)
    expected_wake = p2_report["contract"]["wake_experience_digests"]
    if [str(item.experience_digest) for item in wake] != [str(item) for item in expected_wake]:
        raise RuntimeError("P2 rehearsal stream drifted before P3.0")
    if any(
        sum(item["class_key"] == class_key for item in wake_metadata) != PILOT_PER_CLASS
        for class_key in ("A", "B", "C", "D", "R")
    ):
        raise RuntimeError("P2 rehearsal class balance drifted before P3.0")
    novel_semantic, novel_transition = _candidate_examples(p2_6_manifest["train_records"])
    novel_records = p2_6_manifest["train_records"]
    phases = {
        "wake": [
            {
                "experience_digest": str(record["record_digest"]),
                "semantic_example": semantic,
                "transition_example": transition,
            }
            for record, semantic, transition in zip(
                novel_records[:2], novel_semantic[:2], novel_transition[:2], strict=True
            )
        ],
        "replay": [
            {
                "experience_digest": str(item.experience_digest),
                "semantic_example": item.semantic_example,
                "transition_example": item.transition_example,
            }
            for item in wake[:2]
        ],
        "consolidate": [
            {
                "experience_digest": str(item.experience_digest),
                "semantic_example": item.semantic_example,
                "transition_example": item.transition_example,
            }
            for item in wake[2:4]
        ],
    }
    stream_rows = [
        {
            "phase": phase,
            "index": index,
            "experience_digest": item["experience_digest"],
        }
        for phase in CONTINUATION_PHASES
        for index, item in enumerate(phases[phase])
    ]
    return phases, {
        "phase_rows": stream_rows,
        "stream_digest": content_digest(stream_rows),
        "experience_digests": [str(item["experience_digest"]) for item in stream_rows],
        "class_counts": {
            key: sum(item["class_key"] == key for item in wake_metadata)
            for key in ("A", "B", "C", "D", "R")
        },
    }


def _save_continuation_boundary(
    *,
    directory: Path,
    semantic: Any,
    transition: Any,
    phase: str,
    index: int,
    total: int,
    phases: Mapping[str, Sequence[Mapping[str, Any]]],
    stream_contract: Mapping[str, Any],
    rng: random.Random,
    budget_counts: Mapping[str, int],
    parent_bundle_digest: str,
) -> tuple[TaijiContinuationCheckpoint, dict[str, Any]]:
    worker = _save_arm(directory / "workers", semantic, transition)
    worker_digests = {owner: str(worker["checkpoint_digests"][owner]) for owner in ("k1", "k2")}
    worker_refs = {owner: str(worker["files"][owner]["path"]) for owner in ("k1", "k2")}
    checkpoint = TaijiContinuationCheckpoint.create(
        parent_checkpoint_digest=parent_bundle_digest,
        worker_checkpoint_digests=worker_digests,
        worker_checkpoint_refs=worker_refs,
        phase_cursor=ContinuationPhaseCursor(
            phase=phase,
            index=index,
            total=total,
        ),
        experience_digests=tuple(stream_contract["experience_digests"]),
        stream_digest=str(stream_contract["stream_digest"]),
        rng_state=_jsonable(rng.getstate()),
        budget_counts=budget_counts,
        origin_parent_digest=parent_bundle_digest,
        attached_parent_digest=parent_bundle_digest,
        lineage_chain=(parent_bundle_digest,),
    )
    checkpoint_path = directory / "continuation.json"
    _write_json_atomic(checkpoint_path, checkpoint.to_payload())
    return checkpoint, {
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "worker": worker,
        "phase_lengths": {phase_name: len(items) for phase_name, items in phases.items()},
    }


def _run_until(
    *,
    semantic: Any,
    transition: Any,
    phases: Mapping[str, Sequence[Mapping[str, Any]]],
    stream_contract: Mapping[str, Any],
    output_dir: Path,
    parent_bundle_digest: str,
    stop_after: tuple[str, int] | None,
    rng: random.Random | None = None,
    start_phase: str = "wake",
    start_index: int = 0,
    budget_counts: Mapping[str, int] | None = None,
) -> tuple[TaijiContinuationCheckpoint, dict[str, Any]]:
    active_rng = rng or random.Random(17)
    counts = {"k1": 0, "k2": 0, "stream_items": 0}
    if budget_counts is not None:
        counts.update({str(key): int(value) for key, value in budget_counts.items()})
    phase_position = CONTINUATION_PHASES.index(start_phase)
    for position, phase in enumerate(CONTINUATION_PHASES):
        if position < phase_position:
            continue
        phase_start = start_index if position == phase_position else 0
        items = phases[phase]
        for index in range(phase_start, len(items)):
            active_rng.random()
            item = items[index]
            _fit_direct(
                semantic,
                transition,
                type(
                    "P30Experience",
                    (),
                    {
                        "semantic_example": item["semantic_example"],
                        "transition_example": item["transition_example"],
                    },
                )(),
            )
            counts["k1"] += 1
            counts["k2"] += 1
            counts["stream_items"] += 1
            if stop_after == (phase, index + 1):
                return _save_continuation_boundary(
                    directory=output_dir / f"boundary-{phase}-{index + 1}",
                    semantic=semantic,
                    transition=transition,
                    phase=phase,
                    index=index + 1,
                    total=len(items),
                    phases=phases,
                    stream_contract=stream_contract,
                    rng=active_rng,
                    budget_counts=counts,
                    parent_bundle_digest=parent_bundle_digest,
                )
    return _save_continuation_boundary(
        directory=output_dir / "final",
        semantic=semantic,
        transition=transition,
        phase="consolidate",
        index=len(phases["consolidate"]),
        total=len(phases["consolidate"]),
        phases=phases,
        stream_contract=stream_contract,
        rng=active_rng,
        budget_counts=counts,
        parent_bundle_digest=parent_bundle_digest,
    )


def _resume_from_boundary(
    *,
    boundary: TaijiContinuationCheckpoint,
    output_dir: Path,
    phases: Mapping[str, Sequence[Mapping[str, Any]]],
    stream_contract: Mapping[str, Any],
    parent_bundle_digest: str,
) -> tuple[TaijiContinuationCheckpoint, dict[str, Any]]:
    boundary.assert_parent(parent_bundle_digest)
    payloads = _load_worker_payloads(
        {
            "files": {owner: {"path": path} for owner, path in boundary.worker_checkpoint_refs},
            "checkpoint_digests": dict(boundary.worker_checkpoint_digests),
        }
    )
    semantic, transition = _fresh_learners(payloads[0], payloads[1])
    rng = random.Random()
    rng.setstate(_tupleify(boundary.rng_state))
    position = CONTINUATION_PHASES.index(boundary.phase_cursor.phase)
    if boundary.phase_cursor.index >= boundary.phase_cursor.total:
        position += 1
        start_index = 0
    else:
        start_index = boundary.phase_cursor.index
    if position >= len(CONTINUATION_PHASES):
        raise ValueError("P3.0 boundary already points beyond the phase stream")
    return _run_until(
        semantic=semantic,
        transition=transition,
        phases=phases,
        stream_contract=stream_contract,
        output_dir=output_dir,
        parent_bundle_digest=parent_bundle_digest,
        stop_after=None,
        rng=rng,
        start_phase=CONTINUATION_PHASES[position],
        start_index=start_index,
        budget_counts=dict(boundary.budget_counts),
    )


def _boundary_payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rejection_checks(
    checkpoint: TaijiContinuationCheckpoint, checkpoint_path: str
) -> dict[str, bool]:
    tampered = _boundary_payload(checkpoint_path)
    tampered["phase_cursor"]["index"] = int(tampered["phase_cursor"]["index"]) + 1
    try:
        TaijiContinuationCheckpoint.from_payload(tampered)
    except (KeyError, TypeError, ValueError):
        tamper_rejected = True
    else:
        tamper_rejected = False
    try:
        checkpoint.assert_parent("9" * 64)
    except ValueError:
        wrong_parent_rejected = True
    else:
        wrong_parent_rejected = False
    missing_lineage = _boundary_payload(checkpoint_path)
    missing_lineage.pop("lineage_chain", None)
    try:
        TaijiContinuationCheckpoint.from_payload(missing_lineage)
    except (KeyError, TypeError, ValueError):
        missing_lineage_rejected = True
    else:
        missing_lineage_rejected = False
    return {
        "tampered_cursor_rejected": tamper_rejected,
        "wrong_parent_rejected": wrong_parent_rejected,
        "missing_lineage_rejected": missing_lineage_rejected,
    }


def _trajectory_summary(
    uninterrupted: TaijiContinuationCheckpoint,
    resumed_wake: TaijiContinuationCheckpoint,
    resumed_replay: TaijiContinuationCheckpoint,
) -> dict[str, bool]:
    def worker_digest_map(checkpoint: TaijiContinuationCheckpoint) -> dict[str, str]:
        return dict(checkpoint.worker_checkpoint_digests)

    return {
        "wake_midpoint_final_workers_equal": worker_digest_map(uninterrupted)
        == worker_digest_map(resumed_wake),
        "replay_boundary_final_workers_equal": worker_digest_map(uninterrupted)
        == worker_digest_map(resumed_replay),
        "wake_midpoint_budget_equal": uninterrupted.budget_counts == resumed_wake.budget_counts,
        "replay_boundary_budget_equal": uninterrupted.budget_counts == resumed_replay.budget_counts,
        "wake_midpoint_rng_equal": uninterrupted.rng_state_digest == resumed_wake.rng_state_digest,
        "replay_boundary_rng_equal": uninterrupted.rng_state_digest
        == resumed_replay.rng_state_digest,
        "stream_digest_equal": (
            uninterrupted.stream_digest
            == resumed_wake.stream_digest
            == resumed_replay.stream_digest
        ),
        "final_cursor_equal": (
            uninterrupted.phase_cursor == resumed_wake.phase_cursor == resumed_replay.phase_cursor
        ),
    }


def run_contract(
    *,
    p1_manifest_path: Path = P1_MANIFEST,
    p2_report_path: Path = P2_REPORT,
    p2_6_report_path: Path = P2_6_REPORT,
    p2_6_manifest_path: Path = P2_6_MANIFEST,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p3_0_checkpoint_{uuid4().hex}"
    scratch = run_dir / "scratch"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "model_seed": MODEL_SEED,
        "training_performed": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "p1_manifest": str(p1_manifest_path),
        "p2_report": str(p2_report_path),
        "p2_6_report": str(p2_6_report_path),
        "p2_6_manifest": str(p2_6_manifest_path),
        "manifest": str(manifest_path),
    }
    try:
        p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
        p2_report = json.loads(p2_report_path.read_text(encoding="utf-8"))
        p2_6_report = json.loads(p2_6_report_path.read_text(encoding="utf-8"))
        p2_6_manifest = json.loads(p2_6_manifest_path.read_text(encoding="utf-8"))
        if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P3.0 requires the P1 v2 manifest")
        if p2_report.get("status") != "completed" or p2_report.get("sealed_payload_read"):
            raise ValueError("P3.0 requires a completed, unsealed P2 report")
        if p2_6_report.get("status") != "completed" or p2_6_report.get("sealed_payload_read"):
            raise ValueError("P3.0 requires a completed, unsealed P2.6 report")
        if not all(bool(value) for value in p2_6_report.get("retention_gate", {}).values()):
            raise ValueError("P2.6 retention gate is not fully passed")
        if p2_6_report.get("manifest_digest") != p2_6_manifest.get("manifest_digest"):
            raise ValueError("P2.6 manifest digest mismatch")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2 parent checkpoint digest drifted")
        learned_checkpoint = p2_6_report["arms"]["interleaved-rehearsal-novel"]["checkpoint"]
        if not learned_checkpoint.get("passed"):
            raise ValueError("P2.6 learned checkpoint failed its own Gate")
        learned_dir = Path(str(learned_checkpoint["files"]["k1"]["path"])).parent
        source_restore = _independent_restore(learned_dir)
        if not source_restore.get("independent_process_restore"):
            raise RuntimeError("P2.6 learned source checkpoint cannot be restored")
        learned_k1, learned_k2 = _load_worker_payloads(learned_checkpoint)
        parent_bundle_digest = content_digest(
            {
                "k1": str(learned_checkpoint["checkpoint_digests"]["k1"]),
                "k2": str(learned_checkpoint["checkpoint_digests"]["k2"]),
            }
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        phases, stream_contract = _phase_stream(
            p1_manifest=p1_manifest,
            p2_report=p2_report,
            p2_6_manifest=p2_6_manifest,
            scratch=scratch / "stream",
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p2_6_manifest_digest": str(p2_6_manifest["manifest_digest"]),
            "source_p2_6_worker_checkpoint_digests": learned_checkpoint["checkpoint_digests"],
            "parent_checkpoint_digest": parent_bundle_digest,
            "phase_order": list(CONTINUATION_PHASES),
            "phase_rows": stream_contract["phase_rows"],
            "stream_digest": stream_contract["stream_digest"],
            "experience_digests": stream_contract["experience_digests"],
            "contract": {
                "phase_cursor": True,
                "rng_state": True,
                "budget_counts": True,
                "origin_attached_lineage": True,
                "worker_payload_digest": True,
                "independent_restore": True,
                "parameter_growth": False,
            },
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        uninterrupted_semantic, uninterrupted_transition = _fresh_learners(learned_k1, learned_k2)
        wake_semantic, wake_transition = _fresh_learners(learned_k1, learned_k2)
        replay_semantic, replay_transition = _fresh_learners(learned_k1, learned_k2)
        uninterrupted, uninterrupted_artifact = _run_until(
            semantic=uninterrupted_semantic,
            transition=uninterrupted_transition,
            phases=phases,
            stream_contract=stream_contract,
            output_dir=run_dir / "uninterrupted",
            parent_bundle_digest=parent_bundle_digest,
            stop_after=None,
        )
        wake_boundary, wake_boundary_artifact = _run_until(
            semantic=wake_semantic,
            transition=wake_transition,
            phases=phases,
            stream_contract=stream_contract,
            output_dir=run_dir / "wake-interrupted",
            parent_bundle_digest=parent_bundle_digest,
            stop_after=("wake", 1),
        )
        replay_boundary, replay_boundary_artifact = _run_until(
            semantic=replay_semantic,
            transition=replay_transition,
            phases=phases,
            stream_contract=stream_contract,
            output_dir=run_dir / "replay-interrupted",
            parent_bundle_digest=parent_bundle_digest,
            stop_after=("replay", len(phases["replay"])),
        )
        resumed_wake, resumed_wake_artifact = _resume_from_boundary(
            boundary=wake_boundary,
            output_dir=run_dir / "wake-resumed",
            phases=phases,
            stream_contract=stream_contract,
            parent_bundle_digest=parent_bundle_digest,
        )
        resumed_replay, resumed_replay_artifact = _resume_from_boundary(
            boundary=replay_boundary,
            output_dir=run_dir / "replay-resumed",
            phases=phases,
            stream_contract=stream_contract,
            parent_bundle_digest=parent_bundle_digest,
        )
        rollback_dir = run_dir / "rollback-parent"
        rollback_worker = _save_arm(rollback_dir, *_fresh_learners(learned_k1, learned_k2))
        rollback_gate = {
            "rollback_k1_digest_matches_source": rollback_worker["checkpoint_digests"]["k1"]
            == learned_checkpoint["checkpoint_digests"]["k1"],
            "rollback_k2_digest_matches_source": rollback_worker["checkpoint_digests"]["k2"]
            == learned_checkpoint["checkpoint_digests"]["k2"],
            "rollback_independent_restore": bool(rollback_worker["passed"]),
        }
        rejection_gate = _rejection_checks(
            wake_boundary,
            str(wake_boundary_artifact["checkpoint_path"]),
        )
        trajectory_gate = _trajectory_summary(uninterrupted, resumed_wake, resumed_replay)
        checkpoint_gate = {
            "source_independent_restore": bool(source_restore["independent_process_restore"]),
            "uninterrupted_saved_restore": bool(uninterrupted_artifact["worker"]["passed"]),
            "wake_boundary_saved_restore": bool(wake_boundary_artifact["worker"]["passed"]),
            "replay_boundary_saved_restore": bool(replay_boundary_artifact["worker"]["passed"]),
            "resumed_wake_saved_restore": bool(resumed_wake_artifact["worker"]["passed"]),
            "resumed_replay_saved_restore": bool(resumed_replay_artifact["worker"]["passed"]),
        }
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "parent_checkpoint_digest": parent_bundle_digest,
                "source_checkpoint_preflight": source_restore,
                "phase_stream": stream_contract,
                "phase_boundaries": {
                    "wake_midpoint": wake_boundary.to_payload(),
                    "replay_boundary": replay_boundary.to_payload(),
                    "uninterrupted_final": uninterrupted.to_payload(),
                },
                "trajectories": {
                    "uninterrupted": uninterrupted.to_payload(),
                    "wake_midpoint_resumed": resumed_wake.to_payload(),
                    "replay_boundary_resumed": resumed_replay.to_payload(),
                },
                "checkpoint_gate": checkpoint_gate,
                "trajectory_gate": trajectory_gate,
                "rejection_gate": rejection_gate,
                "rollback_gate": rollback_gate,
                "parameter_growth": False,
                "can_promote": False,
                "interpretation": (
                    "P3.0 checkpoint/interrupt-resume contract only; no S/G worker, no topology growth, "
                    "no lineage-to-capability claim, no promotion"
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
    parser.add_argument("--p1-manifest", type=Path, default=P1_MANIFEST)
    parser.add_argument("--p2-report", type=Path, default=P2_REPORT)
    parser.add_argument("--p2-6-report", type=Path, default=P2_6_REPORT)
    parser.add_argument("--p2-6-manifest", type=Path, default=P2_6_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_contract(
        p1_manifest_path=args.p1_manifest,
        p2_report_path=args.p2_report,
        p2_6_report_path=args.p2_6_report,
        p2_6_manifest_path=args.p2_6_manifest,
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
                "can_promote": result["can_promote"],
                "checkpoint_gate": result.get("checkpoint_gate"),
                "trajectory_gate": result.get("trajectory_gate"),
                "rejection_gate": result.get("rejection_gate"),
                "rollback_gate": result.get("rollback_gate"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

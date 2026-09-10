"""Run the P2.4 retention-preserving objective canary.

P2.3 showed that fitting only six recovery candidates did not add validation
capability and interfered with old high-evidence readouts.  This canary
rebuilds the exact 50-example balanced P2 rehearsal stream and compares it
with a deterministic interleaving of the same rehearsal plus the six P2.3
continuation candidates.

The canary is still a bounded CPU experiment.  It does not add parameters,
read sealed payloads, lower the confidence floor, or claim promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_p2_3_recovery_continuation import (  # noqa: E402
    DEFAULT_MANIFEST as P2_3_MANIFEST,
)
from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _fit_direct,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _candidate_examples,
    _evaluate_arm,
    _fresh_learners,
    _load_arm,
    _save_arm,
)
from scripts.training.eval_taiji_m5_k_p2_output_action_diagnostic import (  # noqa: E402
    _build_validation_cases,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    PILOT_PER_CLASS,
    WORKER_ROOT,
    _checkpoint_preflight,
    _context,
    _rebuild_and_verify_manifest,
    _select_balanced_wake,
)
from taiji import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m5-k-p2-4-retention-canary-v1"
VERSION = 1
P2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
P2_3_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p2_3_recovery_continuation_contract_20260910.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p2_4_retention_canary_20260910.json"
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _interleaved_stream(
    wake: list[Any],
    candidate_semantic: list[Any],
    candidate_transition: list[Any],
) -> list[tuple[str, Any, Any]]:
    if len(candidate_semantic) != len(candidate_transition):
        raise ValueError("continuation semantic/transition stream lengths differ")
    insertion_positions = tuple(
        round(index * (len(wake) - 1) / max(1, len(candidate_semantic) - 1))
        for index in range(len(candidate_semantic))
    )
    insertion_by_position = dict(zip(insertion_positions, range(len(candidate_semantic)), strict=True))
    stream: list[tuple[str, Any, Any]] = []
    for index, experience in enumerate(wake):
        candidate_index = insertion_by_position.get(index)
        if candidate_index is not None:
            stream.append(
                (
                    "continuation",
                    candidate_semantic[candidate_index],
                    candidate_transition[candidate_index],
                )
            )
        stream.append(("rehearsal", experience.semantic_example, experience.transition_example))
    return stream


def _fit_stream(
    semantic: Any,
    transition: Any,
    stream: list[tuple[str, Any, Any]],
) -> dict[str, Any]:
    counts = {"rehearsal": 0, "continuation": 0}
    for kind, semantic_example, transition_example in stream:
        _fit_direct(
            semantic,
            transition,
            type(
                "P24Experience",
                (),
                {"semantic_example": semantic_example, "transition_example": transition_example},
            )(),
        )
        counts[kind] += 1
    return counts


def run_canary(
    *,
    p2_report_path: Path = P2_REPORT,
    p2_3_report_path: Path = P2_3_REPORT,
    p2_3_manifest_path: Path = P2_3_MANIFEST,
    p1_manifest_path: Path = P1_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p2_4_retention_{uuid4().hex}"
    scratch = run_dir / "scratch"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "model_seed": MODEL_SEED,
        "training_performed": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "p2_report": str(p2_report_path),
        "p2_3_report": str(p2_3_report_path),
        "p2_3_manifest": str(p2_3_manifest_path),
    }
    try:
        p2_report = json.loads(p2_report_path.read_text(encoding="utf-8"))
        p2_3_report = json.loads(p2_3_report_path.read_text(encoding="utf-8"))
        p2_3_manifest = json.loads(p2_3_manifest_path.read_text(encoding="utf-8"))
        p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
        if p2_report.get("status") != "completed" or p2_report.get("sealed_payload_read"):
            raise ValueError("P2.4 requires a completed, unsealed P2 report")
        if p2_3_report.get("status") != "completed" or p2_3_report.get("training_performed"):
            raise ValueError("P2.4 requires the completed P2.3 data contract report")
        if not all(bool(value) for value in p2_3_report.get("checks", {}).values()):
            raise ValueError("P2.3 data contract checks are not all passed")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2.4 parent checkpoint digest drifted")
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        semantic_parent_payload = copy.deepcopy(_artifacts["k1.semantic"]["checkpoint"])
        transition_parent_payload = copy.deepcopy(_artifacts["k2.transition"]["checkpoint"])
        (
            train_experiences,
            train_metadata,
            validation,
            validation_metadata,
            manifest_verification,
        ) = _rebuild_and_verify_manifest(
            scratch=scratch / "p1",
            manifest=p1_manifest,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        if not manifest_verification["passed"]:
            raise RuntimeError("P1 manifest reconstruction failed")
        p1_cases, p1_case_mismatches = _build_validation_cases(
            scratch=scratch / "p1-cases",
            validation=validation,
            validation_metadata=validation_metadata,
        )
        if p1_case_mismatches:
            raise RuntimeError(f"P1 validation mismatch: {p1_case_mismatches[:3]}")
        wake, wake_metadata = _select_balanced_wake(train_experiences, train_metadata)
        expected_wake = p2_report["contract"]["wake_experience_digests"]
        actual_wake = [str(item.experience_digest) for item in wake]
        if actual_wake != [str(item) for item in expected_wake]:
            raise RuntimeError("P2 rehearsal experience digest/order drifted")
        if any(
            sum(item["class_key"] == class_key for item in wake_metadata) != PILOT_PER_CLASS
            for class_key in ("A", "B", "C", "D", "R")
        ):
            raise RuntimeError("P2 rehearsal class balance drifted")
        candidate_semantic, candidate_transition = _candidate_examples(
            p2_3_manifest["train_records"]
        )
        candidate_validation_records = p2_3_manifest["validation_records"]
        from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (
            _build_candidate_cases,
        )

        candidate_cases = _build_candidate_cases(
            records=candidate_validation_records,
            scratch=scratch / "p2-3-candidates",
        )
        parent_preflight = _checkpoint_preflight(
            output_dir=run_dir / "preflight-parent",
            semantic_parent=semantic_parent_payload,
            transition_parent=transition_parent_payload,
        )
        if not parent_preflight["passed"]:
            raise RuntimeError("P2.4 parent checkpoint preflight failed")
        interleaved_stream = _interleaved_stream(
            wake,
            candidate_semantic,
            candidate_transition,
        )
        stream_contract = [
            {
                "kind": kind,
                "semantic_input_digest": str(semantic_example.input_digest),
                "transition_input_digest": str(transition_example.input_digest),
            }
            for kind, semantic_example, transition_example in interleaved_stream
        ]
        stream_contract_digest = content_digest(stream_contract)
        arm_data: dict[str, dict[str, Any]] = {}
        parent_semantic, parent_transition = _fresh_learners(
            semantic_parent_payload,
            transition_parent_payload,
        )
        rehearsal_semantic, rehearsal_transition = _fresh_learners(
            semantic_parent_payload,
            transition_parent_payload,
        )
        interleaved_semantic, interleaved_transition = _fresh_learners(
            semantic_parent_payload,
            transition_parent_payload,
        )
        rehearsal_stream = [
            ("rehearsal", experience.semantic_example, experience.transition_example)
            for experience in wake
        ]
        rehearsal_counts = _fit_stream(rehearsal_semantic, rehearsal_transition, rehearsal_stream)
        interleaved_counts = _fit_stream(
            interleaved_semantic,
            interleaved_transition,
            interleaved_stream,
        )
        reference_semantic, reference_transition = _load_arm(
            p2_report["arms"]["wake-only"]["checkpoint"]
        )
        arm_data.update(
            {
                "parent-frozen": {
                    "semantic": parent_semantic,
                    "transition": parent_transition,
                    "fit_counts": {"rehearsal": 0, "continuation": 0},
                },
                "rehearsal-only": {
                    "semantic": rehearsal_semantic,
                    "transition": rehearsal_transition,
                    "fit_counts": rehearsal_counts,
                },
                "interleaved-rehearsal-continuation": {
                    "semantic": interleaved_semantic,
                    "transition": interleaved_transition,
                    "fit_counts": interleaved_counts,
                },
                "p2-wake-only-reference": {
                    "semantic": reference_semantic,
                    "transition": reference_transition,
                    "fit_counts": {"rehearsal": 0, "continuation": 0},
                },
            }
        )
        arm_payload: dict[str, Any] = {}
        for arm_name, arm in arm_data.items():
            checkpoint = _save_arm(run_dir / "arms" / arm_name, arm["semantic"], arm["transition"])
            if not checkpoint["passed"]:
                raise RuntimeError(f"P2.4 independent restore failed for {arm_name}")
            scores = _evaluate_arm(
                semantic=arm["semantic"],
                transition=arm["transition"],
                p1_cases=p1_cases,
                candidate_cases=candidate_cases,
                candidate_validation=candidate_validation_records,
            )
            arm_payload[arm_name] = {
                "checkpoint": checkpoint,
                "fit_counts": arm["fit_counts"],
                "training_steps_total": {
                    "k1": int(arm["semantic"].training_steps),
                    "k2": int(arm["transition"].training_steps),
                },
                "scores": scores,
            }
        parent_scores = arm_payload["parent-frozen"]["scores"]
        interleaved_scores = arm_payload["interleaved-rehearsal-continuation"]["scores"]
        retention_gate = {
            "high_evidence_k1_goal_non_decreasing": interleaved_scores["p1_validation"][
                "k1_goal_hit_count"
            ]
            >= parent_scores["p1_validation"]["k1_goal_hit_count"],
            "high_evidence_k2_goal_non_decreasing": interleaved_scores["p1_validation"][
                "k2_goal_hit_count"
            ]
            >= parent_scores["p1_validation"]["k2_goal_hit_count"],
            "safe_abstention_non_decreasing": interleaved_scores["p1_validation"][
                "safe_abstention_count"
            ]
            >= parent_scores["p1_validation"]["safe_abstention_count"],
            "workbench_success_non_decreasing": interleaved_scores["p1_validation"][
                "workbench_success_count"
            ]
            >= parent_scores["p1_validation"]["workbench_success_count"],
            "continuation_workbench_success": interleaved_scores[
                "p2_3_continuation_validation"
            ]["workbench_success_count"]
            >= parent_scores["p2_3_continuation_validation"]["workbench_success_count"],
        }
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "run_dir": str(run_dir),
                "manifest_reconstruction": manifest_verification,
                "p1_validation_case_count": len(p1_cases),
                "p2_rehearsal": {
                    "count": len(wake),
                    "class_counts": {
                        key: sum(item["class_key"] == key for item in wake_metadata)
                        for key in ("A", "B", "C", "D", "R")
                    },
                    "experience_digest_match": True,
                    "stream_digest": stream_contract_digest,
                },
                "training_contract": {
                    "rehearsal_count": len(wake),
                    "continuation_train_count": len(candidate_semantic),
                    "interleaved_stream_count": len(interleaved_stream),
                    "validation_used_for_fit": False,
                    "parameter_growth": False,
                    "confidence_floor": 0.55,
                    "replay_source_count": 0,
                },
                "checkpoint_preflight": {
                    "parent_before_fit": parent_preflight,
                    "all_saved_arms_after_fit": True,
                },
                "arms": arm_payload,
                "retention_gate": retention_gate,
                "can_promote": False,
                "interpretation": (
                    "retention-objective-canary-only; no promotion claim; host recovery is not model credit"
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
    parser.add_argument("--p2-report", type=Path, default=P2_REPORT)
    parser.add_argument("--p2-3-report", type=Path, default=P2_3_REPORT)
    parser.add_argument("--p2-3-manifest", type=Path, default=P2_3_MANIFEST)
    parser.add_argument("--p1-manifest", type=Path, default=P1_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run_canary(
        p2_report_path=args.p2_report,
        p2_3_report_path=args.p2_3_report,
        p2_3_manifest_path=args.p2_3_manifest,
        p1_manifest_path=args.p1_manifest,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": payload["status"],
                "training_performed": payload["training_performed"],
                "can_promote": payload["can_promote"],
                "retention_gate": payload.get("retention_gate"),
                "error": payload.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

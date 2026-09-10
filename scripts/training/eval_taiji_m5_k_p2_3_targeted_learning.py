"""Run the bounded P2.3 recovery-continuation learning pilot.

The pilot starts from the same P2 parent checkpoint and compares:

* ``parent-frozen``: no update;
* ``continuation-targeted``: K1/K2 fit only on the six train candidate
  examples from the P2.3 continuation manifest;
* ``p2-wake-only-reference``: the already saved P2 wake-only artifact, loaded
  without additional fitting.

The missing-target step remains host policy and is never fit.  Original P1 v2
validation and the two P2.3 candidate validation rows are read-only scoring
inputs.  The pilot performs checkpoint save/independent-restore preflight
before fitting and repeats it for every saved arm.  It never reads sealed
payloads and never promotes a candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_p1_data import (  # noqa: E402
    _prepare_workspace,
    _registry_for_state,
)
from scripts.training.build_taiji_m5_k_p2_3_recovery_continuation import (  # noqa: E402
    DEFAULT_MANIFEST as P2_3_MANIFEST,
)
from scripts.training.build_taiji_m5_k_p2_3_recovery_continuation import (
    _restore_tensors,
)
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    READ_ONLY_ROUTES,
)
from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _atomic_roundtrip,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p2_2_safety_bridge_canary import (  # noqa: E402
    _run_intent,
)
from scripts.training.eval_taiji_m5_k_p2_output_action_diagnostic import (  # noqa: E402
    _build_validation_cases,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _checkpoint_preflight,
    _context,
    _rebuild_and_verify_manifest,
)
from taiji import (  # noqa: E402
    NativeReadOnlyIntentPlanner,
    ReadOnlyIntentPolicy,
    StructuredSemanticExample,
    StructuredSemanticLearner,
    StructuredSemanticTransitionExample,
    StructuredSemanticTransitionLearner,
    WorkbenchObservation,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p2-3-targeted-learning-pilot-v1"
VERSION = 1
CONFIDENCE_FLOOR = 0.55
SEMANTIC_EPOCHS = 160
SEMANTIC_LR = 2.0
TRANSITION_EPOCHS = 220
TRANSITION_LR = 0.5
P2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
P2_3_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p2_3_recovery_continuation_contract_20260910.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p2_3_targeted_learning_pilot_20260910.json"
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fresh_learners(
    semantic_payload: Mapping[str, Any], transition_payload: Mapping[str, Any]
) -> tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]:
    return (
        StructuredSemanticLearner.from_checkpoint(copy.deepcopy(semantic_payload), device="cpu"),
        StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(transition_payload), device="cpu"
        ),
    )


def _independent_restore(directory: Path) -> dict[str, Any]:
    verifier = Path(__file__).resolve().with_name("eval_taiji_m5_k_p2_validation_pilot.py")
    child = subprocess.run(
        [sys.executable, str(verifier), "--verify-only", str(directory)],
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
    result["independent_process_restore"] = bool(decoded.get("passed"))
    return result


def _save_arm(
    directory: Path,
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    k1_path = directory / "birth_k1_semantic.pt"
    k2_path = directory / "birth_k2_transition.pt"
    semantic_payload = semantic.checkpoint()
    transition_payload = transition.checkpoint()
    _atomic_roundtrip(k1_path, semantic_payload)
    _atomic_roundtrip(k2_path, transition_payload)
    restore = _independent_restore(directory)
    return {
        "files": {
            "k1": {"path": str(k1_path), "bytes": k1_path.stat().st_size},
            "k2": {"path": str(k2_path), "bytes": k2_path.stat().st_size},
        },
        "checkpoint_digests": {
            "k1": content_digest(semantic_payload),
            "k2": content_digest(transition_payload),
        },
        "restore": restore,
        "passed": bool(restore.get("independent_process_restore")),
    }


def _load_arm(checkpoint: Mapping[str, Any]) -> tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]:
    return _fresh_learners(
        _load_mapping(Path(str(checkpoint["k1"]["path"]))),
        _load_mapping(Path(str(checkpoint["k2"]["path"]))),
    )


def _candidate_examples(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[StructuredSemanticExample], list[StructuredSemanticTransitionExample]]:
    semantics: list[StructuredSemanticExample] = []
    transitions: list[StructuredSemanticTransitionExample] = []
    for record in records:
        candidate = record["candidate"]
        semantics.append(
            StructuredSemanticExample.from_payload(
                _restore_tensors(candidate["semantic_example"])
            )
        )
        transitions.append(
            StructuredSemanticTransitionExample.from_payload(
                _restore_tensors(candidate["transition_example"])
            )
        )
    return semantics, transitions


def _build_candidate_cases(
    *,
    records: Sequence[Mapping[str, Any]],
    scratch: Path,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for record in records:
        index = int(record["index"])
        candidate_payload = record["candidate"]["observation"]
        observation = WorkbenchObservation.from_payload(candidate_payload)
        root = scratch / f"candidate-validation-{index:04d}"
        root.mkdir(parents=True, exist_ok=False)
        _prepare_workspace(
            root,
            task_seed=4100 + index,
            first_path="missing_validation.txt",
            state_profile="recovery-no-selection",
        )
        cases.append(
            {
                "index": index,
                "class_key": "R-continuation",
                "state_profile": "recovered-candidate",
                "root": root,
                "registry": _registry_for_state(observation.path, "resolved-language"),
                "observation": observation,
                "record": record,
            }
        )
    return cases


def _chain_row(
    *,
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    case: Mapping[str, Any],
    semantic_example: StructuredSemanticExample,
    transition_example: StructuredSemanticTransitionExample,
    label: str,
) -> dict[str, Any]:
    semantic_result = semantic.predict(semantic_example.percept)
    transition_result = None
    if semantic_result.world is not None:
        transition_result = transition.predict(
            semantic_result.world,
            transition_example.event,
        )
    expected_goal_id = semantic_example.goal.goal_id
    expected_content_id = semantic_example.content.content_id
    row: dict[str, Any] = {
        "index": int(case["index"]),
        "class_key": str(case["class_key"]),
        "label": label,
        "input_confidence": float(semantic_example.percept.confidence),
        "expected_goal_id": expected_goal_id,
        "expected_content_id": expected_content_id,
        "k1_status": semantic_result.status,
        "k1_goal_id": None if semantic_result.goal is None else semantic_result.goal.goal_id,
        "k1_content_id": (
            None
            if semantic_result.content_plan is None
            else semantic_result.content_plan.content_id
        ),
        "k1_goal_hit": bool(
            semantic_result.goal is not None
            and semantic_result.goal.goal_id == expected_goal_id
        ),
        "k1_content_hit": bool(
            semantic_result.content_plan is not None
            and semantic_result.content_plan.content_id == expected_content_id
        ),
        "k2_status": None if transition_result is None else transition_result.status,
        "k2_goal_id": (
            None if transition_result is None or transition_result.goal is None else transition_result.goal.goal_id
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
        "safe_abstention": bool(
            semantic_example.percept.confidence < CONFIDENCE_FLOOR
            and semantic_result.status == "unknown"
            and semantic_result.goal is None
            and semantic_result.content_plan is None
        ),
        "action": None,
    }
    if transition_result is not None:
        planner = NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES))
        row["action"] = _run_intent(
            case=case,
            planner=planner,
            goal=semantic_result.goal,
            content=semantic_result.content_plan,
            world=transition_result.world,
            label=label,
            oracle_control=False,
        )
    return row


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "rows": 0,
            "k1_goal_hits": 0,
            "k1_content_hits": 0,
            "k2_goal_hits": 0,
            "k2_content_hits": 0,
            "safe_abstentions": 0,
            "workbench_successes": 0,
        }
    )
    for row in rows:
        bucket = by_class[str(row["class_key"])]
        bucket["rows"] += 1
        bucket["k1_goal_hits"] += int(bool(row["k1_goal_hit"]))
        bucket["k1_content_hits"] += int(bool(row["k1_content_hit"]))
        bucket["k2_goal_hits"] += int(bool(row["k2_goal_hit"]))
        bucket["k2_content_hits"] += int(bool(row["k2_content_hit"]))
        bucket["safe_abstentions"] += int(bool(row["safe_abstention"]))
        action = row.get("action") or {}
        bucket["workbench_successes"] += int(bool(action.get("workbench_success")))
    return {
        "row_count": len(rows),
        "k1_goal_hit_count": sum(int(bool(row["k1_goal_hit"])) for row in rows),
        "k1_content_hit_count": sum(int(bool(row["k1_content_hit"])) for row in rows),
        "k2_goal_hit_count": sum(int(bool(row["k2_goal_hit"])) for row in rows),
        "k2_content_hit_count": sum(int(bool(row["k2_content_hit"])) for row in rows),
        "safe_abstention_count": sum(int(bool(row["safe_abstention"])) for row in rows),
        "workbench_success_count": sum(
            int(bool((row.get("action") or {}).get("workbench_success"))) for row in rows
        ),
        "by_class": {key: by_class[key] for key in sorted(by_class)},
        "rows": list(rows),
    }


def _evaluate_arm(
    *,
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    p1_cases: Sequence[Mapping[str, Any]],
    candidate_cases: Sequence[Mapping[str, Any]],
    candidate_validation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    p1_rows: list[dict[str, Any]] = []
    for case in p1_cases:
        experience = case["experience"]
        p1_rows.append(
            _chain_row(
                semantic=semantic,
                transition=transition,
                case=case,
                semantic_example=experience.semantic_example,
                transition_example=experience.transition_example,
                label="p1-validation",
            )
        )
    candidate_rows: list[dict[str, Any]] = []
    for case, record in zip(candidate_cases, candidate_validation, strict=True):
        candidate = record["candidate"]
        semantic_example = StructuredSemanticExample.from_payload(
            _restore_tensors(candidate["semantic_example"])
        )
        transition_example = StructuredSemanticTransitionExample.from_payload(
            _restore_tensors(candidate["transition_example"])
        )
        candidate_rows.append(
            _chain_row(
                semantic=semantic,
                transition=transition,
                case=case,
                semantic_example=semantic_example,
                transition_example=transition_example,
                label="p2-3-continuation-validation",
            )
        )
    return {
        "parameter_count": {
            "k1": int(semantic.parameter_count),
            "k2": int(transition.parameter_count),
            "total": int(semantic.parameter_count + transition.parameter_count),
        },
        "p1_validation": _summarize(p1_rows),
        "p2_3_continuation_validation": _summarize(candidate_rows),
    }


def run_pilot(
    *,
    p2_report_path: Path = P2_REPORT,
    p2_3_report_path: Path = P2_3_REPORT,
    p2_3_manifest_path: Path = P2_3_MANIFEST,
    p1_manifest_path: Path = P1_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p2_3_targeted_learning_{uuid4().hex}"
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
            raise ValueError("P2.3 requires a completed, unsealed P2 report")
        if p2_3_report.get("status") != "completed" or p2_3_report.get(
            "training_performed"
        ):
            raise ValueError("P2.3 targeted pilot requires the completed data-contract report")
        if not all(bool(value) for value in p2_3_report.get("checks", {}).values()):
            raise ValueError("P2.3 data-contract checks are not all passed")
        if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P2.3 requires the P1 v2 manifest")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2 parent digest does not match the current worker artifact")
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        semantic_parent = copy.deepcopy(_artifacts["k1.semantic"]["checkpoint"])
        transition_parent = copy.deepcopy(_artifacts["k2.transition"]["checkpoint"])
        (
            _train_experiences,
            _train_metadata,
            p1_validation,
            p1_validation_metadata,
            p1_verification,
        ) = _rebuild_and_verify_manifest(
            scratch=scratch / "p1",
            manifest=p1_manifest,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        if not p1_verification["passed"]:
            raise RuntimeError("P1 validation reconstruction failed before P2.3 fit")
        p1_cases, p1_case_mismatches = _build_validation_cases(
            scratch=scratch / "p1-cases",
            validation=p1_validation,
            validation_metadata=p1_validation_metadata,
        )
        if p1_case_mismatches:
            raise RuntimeError(f"P1 validation case mismatch: {p1_case_mismatches[:3]}")
        train_records = p2_3_manifest["train_records"]
        validation_records = p2_3_manifest["validation_records"]
        train_semantic, train_transition = _candidate_examples(train_records)
        candidate_cases = _build_candidate_cases(
            records=validation_records,
            scratch=scratch / "p2-3-candidates",
        )
        parent_preflight = _checkpoint_preflight(
            output_dir=run_dir / "preflight-parent",
            semantic_parent=semantic_parent,
            transition_parent=transition_parent,
        )
        if not parent_preflight["passed"]:
            raise RuntimeError("P2.3 checkpoint save/independent-restore preflight failed")

        parent_semantic, parent_transition = _fresh_learners(
            semantic_parent,
            transition_parent,
        )
        targeted_semantic, targeted_transition = _fresh_learners(
            semantic_parent,
            transition_parent,
        )
        targeted_losses = {
            "k1": targeted_semantic.fit(
                train_semantic,
                epochs=SEMANTIC_EPOCHS,
                learning_rate=SEMANTIC_LR,
            ),
            "k2": targeted_transition.fit(
                train_transition,
                epochs=TRANSITION_EPOCHS,
                learning_rate=TRANSITION_LR,
            ),
        }
        reference_semantic, reference_transition = _load_arm(p2_report["arms"]["wake-only"]["checkpoint"])

        arm_data: dict[str, Any] = {
            "parent-frozen": {
                "semantic": parent_semantic,
                "transition": parent_transition,
                "fit_calls": {"k1": 0, "k2": 0},
                "losses": None,
            },
            "continuation-targeted": {
                "semantic": targeted_semantic,
                "transition": targeted_transition,
                "fit_calls": {"k1": 1, "k2": 1},
                "losses": targeted_losses,
            },
            "p2-wake-only-reference": {
                "semantic": reference_semantic,
                "transition": reference_transition,
                "fit_calls": {"k1": 0, "k2": 0},
                "losses": None,
            },
        }
        arm_payload: dict[str, Any] = {}
        for arm_name, arm in arm_data.items():
            checkpoint = _save_arm(run_dir / "arms" / arm_name, arm["semantic"], arm["transition"])
            if not checkpoint["passed"]:
                raise RuntimeError(f"independent restore failed for arm {arm_name}")
            scored = _evaluate_arm(
                semantic=arm["semantic"],
                transition=arm["transition"],
                p1_cases=p1_cases,
                candidate_cases=candidate_cases,
                candidate_validation=validation_records,
            )
            arm_payload[arm_name] = {
                "checkpoint": checkpoint,
                "fit_calls": arm["fit_calls"],
                "training_steps_total": {
                    "k1": int(arm["semantic"].training_steps),
                    "k2": int(arm["transition"].training_steps),
                },
                "new_training_steps": {
                    "k1": (
                        int(arm["semantic"].training_steps)
                        - int(parent_semantic.training_steps)
                        if arm["fit_calls"]["k1"]
                        else 0
                    ),
                    "k2": (
                        int(arm["transition"].training_steps)
                        - int(parent_transition.training_steps)
                        if arm["fit_calls"]["k2"]
                        else 0
                    ),
                },
                "losses": arm["losses"],
                "scores": scored,
            }
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "run_dir": str(run_dir),
                "manifest_reconstruction": p1_verification,
                "p1_validation_case_count": len(p1_cases),
                "p2_3_train_candidate_count": len(train_records),
                "p2_3_validation_candidate_count": len(validation_records),
                "training_contract": {
                    "semantic_epochs": SEMANTIC_EPOCHS,
                    "semantic_learning_rate": SEMANTIC_LR,
                    "transition_epochs": TRANSITION_EPOCHS,
                    "transition_learning_rate": TRANSITION_LR,
                    "fit_record_ids": [str(record["record_id"]) for record in train_records],
                    "validation_used_for_fit": False,
                    "replay_added": False,
                    "parameter_growth": False,
                    "confidence_floor": CONFIDENCE_FLOOR,
                },
                "checkpoint_preflight": {
                    "parent_before_fit": parent_preflight,
                    "all_saved_arms_after_fit": True,
                },
                "arms": arm_payload,
                "interpretation": (
                    "targeted-learning-pilot-only; host workspace.list is not model credit; "
                    "P2.3 and P1 validation are read-only scores; no promotion claim"
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
    payload = run_pilot(
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
                "error": payload.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

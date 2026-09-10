"""Run the no-training P2.2 safety and recovery bridge canary.

The canary uses the same ten P1 v2 validation cases and the immutable P2
candidate checkpoints.  It compares four explicitly separated paths:

* the original learned K1 -> K2 -> read-only planner path;
* a typed non-executable abstention for low evidence;
* a recovery route for a missing target, using ``workspace.list`` at ``.``;
* an oracle world-alignment control that isolates K2 world drift from the
  planner and Workbench boundary.

The recovery and alignment controls are engineering canaries, not model
accuracy claims: they use the validation target or ground-truth world and are
marked ``oracle_control`` in the report.  This script never calls ``fit``,
reads sealed payloads, changes readout thresholds, or promotes a candidate.
"""

from __future__ import annotations

import argparse
import json
import shutil
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

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    READ_ONLY_ROUTES,
    _world,
)
from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p2_output_action_diagnostic import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    _build_validation_cases,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _context,
    _rebuild_and_verify_manifest,
)
from seed import Seed  # noqa: E402
from seed.config import SeedConfig  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    NativeReadOnlyIntentPlanner,
    ReadOnlyAbstention,
    ReadOnlyIntentPolicy,
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    TaijiConfig,
)

REPORT_FORMAT = "taiji-m5-k-p2-2-safety-bridge-canary-v1"
VERSION = 1
DEFAULT_P2_1_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p2_output_action_diagnostic_20260910.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_2_safety_bridge_canary_20260910.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _recovery_policy() -> ReadOnlyIntentPolicy:
    return ReadOnlyIntentPolicy(
        routes=tuple((*READ_ONLY_ROUTES, ("content:recover-target", "workspace.list"))),
        route_parameters=(("content:recover-target", (("path", "."),)),),
    )


def _run_intent(
    *,
    case: Mapping[str, Any],
    planner: NativeReadOnlyIntentPlanner,
    goal: Any,
    content: Any,
    world: Any,
    label: str,
    oracle_control: bool,
) -> dict[str, Any]:
    observation = case["observation"]
    root = case["root"]
    registry = case["registry"]
    result: dict[str, Any] = {
        "label": label,
        "oracle_control": oracle_control,
        "planner_status": "not_attempted",
        "reason_code": None,
        "accepted": False,
        "snapshot_match": False,
        "parameter_drift": None,
        "intent_kind": None,
        "intent_parameters": None,
        "workbench_status": "not_attempted",
        "workbench_success": False,
        "workbench_error_code": None,
        "failure_recovery_status": "not_triggered",
    }
    if goal is None or content is None or world is None:
        result["planner_status"] = "model_output_missing"
        return result

    import seed_platform.workbench as workbench_module

    original_get_setting = workbench_module.get_setting
    workbench_module.get_setting = lambda key, default=None: (
        str(root) if key == "workspace_path" else default
    )
    try:
        runtime = SeedRuntime(
            Seed(
                SeedConfig(taiji=TaijiConfig(seed=MODEL_SEED)),
                episode_id=f"taiji-m5-k-p2-2-{label}-{case['class_key']}-{case['index']}",
            )
        )
        runtime._workbench_environment = WorkbenchEnvironment(
            root=root,
            programming_language_registry=registry,
        )
        snapshot = runtime.workbench_environment.capability_snapshot
        result["snapshot_match"] = bool(
            observation.capability_snapshot_id == snapshot.snapshot_id
            and observation.capability_revision == snapshot.revision
        )
        decision = planner.propose(
            observation=observation,
            world=world,
            goal=goal,
            content=content,
            capability_snapshot=snapshot,
            tick=1,
        )
        result["planner_status"] = "accepted" if decision.accepted else "rejected"
        result["reason_code"] = decision.reason_code
        result["accepted"] = bool(decision.accepted)
        if decision.action_intent is None:
            return result
        result["intent_kind"] = decision.action_intent.kind
        result["intent_parameters"] = dict(decision.action_intent.parameters)
        expected_parameters = planner.policy.parameters_for(content.content_id)
        result["parameter_drift"] = {
            "declared_overrides": expected_parameters,
            "actual": dict(decision.action_intent.parameters),
            "matches_contract": (
                all(
                    decision.action_intent.parameters.get(key) == value
                    for key, value in expected_parameters.items()
                )
            ),
        }
        outcome = runtime.execute_workbench_intent(
            decision.action_intent,
            snapshot_id=snapshot.snapshot_id,
            learn=False,
        )
        outcome_payload = dict(outcome.get("outcome") or {})
        result["workbench_status"] = str(outcome_payload.get("status", "unknown"))
        result["workbench_success"] = bool(outcome_payload.get("success", False))
        result["workbench_error_code"] = outcome_payload.get("error_code")
        if not result["workbench_success"]:
            result["failure_recovery_status"] = "not_implemented_in_read_only_probe"
        return result
    except (TypeError, ValueError, RuntimeError, OSError) as exc:
        result["planner_status"] = "diagnostic_error"
        result["reason_code"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        workbench_module.get_setting = original_get_setting


def _abstention(
    *,
    case: Mapping[str, Any],
    planner: NativeReadOnlyIntentPlanner,
    reason_code: str,
    next_step: str,
    confidence: float,
) -> dict[str, Any]:
    import seed_platform.workbench as workbench_module

    original_get_setting = workbench_module.get_setting
    workbench_module.get_setting = lambda key, default=None: (
        str(case["root"]) if key == "workspace_path" else default
    )
    try:
        runtime = SeedRuntime(
            Seed(
                SeedConfig(taiji=TaijiConfig(seed=MODEL_SEED)),
                episode_id=f"taiji-m5-k-p2-2-abstention-{case['class_key']}-{case['index']}",
            )
        )
        runtime._workbench_environment = WorkbenchEnvironment(
            root=case["root"],
            programming_language_registry=case["registry"],
        )
        snapshot = runtime.workbench_environment.capability_snapshot
        abstention = planner.abstain(
            observation=case["observation"],
            capability_snapshot=snapshot,
            reason_code=reason_code,
            next_step=next_step,
            confidence=confidence,
        )
        restored = ReadOnlyAbstention.from_payload(abstention.to_payload())
        payload = abstention.to_payload()
        return {
            "status": "typed_abstention",
            "reason_code": abstention.reason_code,
            "next_step": abstention.next_step,
            "confidence": abstention.confidence,
            "roundtrip": restored.to_payload() == payload,
            "action_intent_is_none": payload.get("action_intent") is None,
            "snapshot_match": (
                abstention.snapshot_id == case["observation"].capability_snapshot_id
                and abstention.capability_revision == case["observation"].capability_revision
            ),
        }
    finally:
        workbench_module.get_setting = original_get_setting


def _arm_canary(
    *,
    arm_name: str,
    arm_payload: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint = arm_payload["checkpoint"]
    semantic = StructuredSemanticLearner.from_checkpoint(
        _load_mapping(Path(str(checkpoint["k1"]["path"]))), device="cpu"
    )
    transition = StructuredSemanticTransitionLearner.from_checkpoint(
        _load_mapping(Path(str(checkpoint["k2"]["path"]))), device="cpu"
    )
    original_planner = NativeReadOnlyIntentPlanner(
        ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES)
    )
    recovery_planner = NativeReadOnlyIntentPlanner(_recovery_policy())
    rows: list[dict[str, Any]] = []
    for case in cases:
        experience = case["experience"]
        semantic_result = semantic.predict(experience.semantic_example.percept)
        if semantic_result.world is None:
            raise RuntimeError(f"P2.2 K1 world missing for row {case['index']}")
        transition_result = transition.predict(
            semantic_result.world,
            experience.transition_example.event,
        )
        baseline = _run_intent(
            case=case,
            planner=original_planner,
            goal=semantic_result.goal,
            content=semantic_result.content_plan,
            world=transition_result.world,
            label="original",
            oracle_control=False,
        )
        alignment = _run_intent(
            case=case,
            planner=original_planner,
            goal=semantic_result.goal,
            content=semantic_result.content_plan,
            world=_world(case["observation"], tick=1),
            label="world-alignment-control",
            oracle_control=True,
        )
        needs_abstention = semantic_result.content_plan is None or semantic_result.status in {
            "unknown",
            "ambiguous",
        }
        abstention = None
        if needs_abstention:
            missing_target = not case["observation"].read_success
            abstention = _abstention(
                case=case,
                planner=original_planner,
                reason_code=(
                    "missing_target" if missing_target else "input_confidence_below_floor"
                ),
                next_step="workspace.list" if missing_target else "request_clarification",
                confidence=float(experience.semantic_example.percept.confidence),
            )
        recovery = None
        if not case["observation"].read_success:
            recovery = _run_intent(
                case=case,
                planner=recovery_planner,
                goal=experience.semantic_example.goal,
                content=experience.semantic_example.content,
                world=_world(case["observation"], tick=1),
                label="recover-target",
                oracle_control=True,
            )
        rows.append(
            {
                "index": int(case["index"]),
                "class_key": str(case["class_key"]),
                "state_profile": str(case["state_profile"]),
                "input_confidence": float(experience.semantic_example.percept.confidence),
                "k1_status": str(semantic_result.status),
                "k2_status": str(transition_result.status),
                "baseline": baseline,
                "typed_abstention": abstention,
                "world_alignment_control": alignment,
                "recovery_route_control": recovery,
            }
        )

    def count(path: tuple[str, ...], predicate) -> int:
        return sum(bool(predicate(row)) for row in rows)

    planner_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        reason = row["baseline"].get("reason_code")
        if reason:
            planner_reasons[str(reason)] += 1
    abstention_rows = [row for row in rows if row["typed_abstention"] is not None]
    recovery_rows = [row for row in rows if row["recovery_route_control"] is not None]
    alignment_rows = [row for row in rows if row["world_alignment_control"] is not None]
    return {
        "parameter_count": {
            "k1": int(semantic.parameter_count),
            "k2": int(transition.parameter_count),
            "total": int(semantic.parameter_count + transition.parameter_count),
        },
        "baseline": {
            "planner_accept_count": count(("baseline",), lambda row: row["baseline"]["accepted"]),
            "workbench_success_count": count(
                ("baseline",), lambda row: row["baseline"]["workbench_success"]
            ),
            "planner_reason_counts": dict(sorted(planner_reasons.items())),
        },
        "typed_abstention": {
            "count": len(abstention_rows),
            "all_roundtrip": all(
                row["typed_abstention"]["roundtrip"] for row in abstention_rows
            ),
            "all_non_executable": all(
                row["typed_abstention"]["action_intent_is_none"] for row in abstention_rows
            ),
            "next_step_counts": dict(
                sorted(
                    {
                        step: sum(
                            row["typed_abstention"]["next_step"] == step
                            for row in abstention_rows
                        )
                        for step in {
                            row["typed_abstention"]["next_step"] for row in abstention_rows
                        }
                    }.items()
                )
            ),
        },
        "world_alignment_control": {
            "count": len(alignment_rows),
            "planner_accept_count": sum(
                row["world_alignment_control"]["accepted"] for row in alignment_rows
            ),
            "workbench_success_count": sum(
                row["world_alignment_control"]["workbench_success"] for row in alignment_rows
            ),
            "oracle_control": True,
        },
        "recovery_route_control": {
            "count": len(recovery_rows),
            "planner_accept_count": sum(
                row["recovery_route_control"]["accepted"] for row in recovery_rows
            ),
            "workbench_success_count": sum(
                row["recovery_route_control"]["workbench_success"] for row in recovery_rows
            ),
            "all_root_scoped": all(
                row["recovery_route_control"]["intent_parameters"] == {"path": "."}
                for row in recovery_rows
                if row["recovery_route_control"]["accepted"]
            ),
            "oracle_control": True,
        },
        "rows": rows,
    }


def run_canary(
    *,
    p2_1_report: Path = DEFAULT_P2_1_REPORT,
    manifest_path: Path = P1_MANIFEST,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    source = json.loads(p2_1_report.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "p2_1_report": str(p2_1_report),
        "manifest": str(manifest_path),
    }
    scratch = DEFAULT_OUTPUT_ROOT / f"_taiji_m5_k_p2_2_scratch_{uuid4().hex}"
    try:
        if source.get("status") != "completed" or source.get("training_performed"):
            raise ValueError("P2.2 requires the completed read-only P2.1 report")
        if source.get("sealed_payload_read"):
            raise ValueError("P2.2 refuses a report that read sealed payload")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT, model_seed=MODEL_SEED
        )
        scratch.mkdir(parents=True, exist_ok=False)
        (
            _train_experiences,
            _train_metadata,
            validation,
            validation_metadata,
            manifest_verification,
        ) = _rebuild_and_verify_manifest(
            scratch=scratch / "manifest",
            manifest=manifest,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        if not manifest_verification["passed"]:
            raise RuntimeError("P1 v2 manifest reconstruction failed during P2.2")
        cases, mismatches = _build_validation_cases(
            scratch=scratch / "cases",
            validation=validation,
            validation_metadata=validation_metadata,
        )
        if mismatches:
            raise RuntimeError(f"validation observation reconstruction mismatch: {mismatches[:3]}")
        arm_results: dict[str, Any] = {}
        for arm_name in ("frozen", "wake-only", "wake-replay"):
            arm_payload = source.get("arms", {}).get(arm_name)
            if not isinstance(arm_payload, Mapping):
                raise ValueError(f"P2.1 is missing arm: {arm_name}")
            arm_results[arm_name] = _arm_canary(
                arm_name=arm_name,
                arm_payload=arm_payload,
                cases=cases,
            )
        payload.update(
            {
                "status": "completed",
                "manifest_reconstruction": manifest_verification,
                "validation_case_count": len(cases),
                "recovery_policy": _recovery_policy().to_payload(),
                "arms": arm_results,
                "interpretation": (
                    "P2.2 safety bridge canary only; recovery and world alignment controls "
                    "are oracle-labeled and do not establish model capability"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    _write_json_atomic(report, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-1-report", type=Path, default=DEFAULT_P2_1_REPORT)
    parser.add_argument("--manifest", type=Path, default=P1_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run_canary(
        p2_1_report=args.p2_1_report,
        manifest_path=args.manifest,
        report=args.report,
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": payload["status"],
                "training_performed": payload["training_performed"],
                "sealed_payload_read": payload["sealed_payload_read"],
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

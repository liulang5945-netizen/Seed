"""Diagnose P2 output readout and the minimal read-only action chain.

This is the first P2.1 diagnostic.  It consumes the immutable artifacts from
the completed P2 pilot and the passed P1 v2 manifest.  It does not call
``fit``, read sealed payloads, alter thresholds, or promote an arm.

The report separates four layers that the P2 MSE pilot intentionally kept
apart:

1. K1/K2 continuous scores and discrete structured outputs;
2. the existing native read-only intent planner;
3. isolated Workbench execution;
4. failure recovery (only recorded when an admitted action actually fails).

The distinction is important: a lower regression loss is not evidence that a
usable intent was produced or that the Workbench could execute it.
"""

from __future__ import annotations

import argparse
import json
import os
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
from scripts.training.build_taiji_m5_k_p1_data import (  # noqa: E402
    _observations,
    _prepare_workspace,
    _registry_for_state,
)
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    READ_ONLY_ROUTES,
    _schema,
    _world,
)
from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _context,
    _loss_score,
    _rebuild_and_verify_manifest,
)
from seed import Seed  # noqa: E402
from seed.config import SeedConfig  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    NativeReadOnlyIntentPlanner,
    ReadOnlyIntentPolicy,
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    TaijiConfig,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p2-output-action-diagnostic-v1"
VERSION = 1
DEFAULT_PILOT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_output_action_diagnostic_20260910.json"
ROUTE_CONTENT_IDS = {content_id for content_id, _ in READ_ONLY_ROUTES}


def _jsonable_scores(values: Mapping[str, float]) -> dict[str, float]:
    return {str(key): float(value) for key, value in sorted(values.items())}


def _rank(scores: Mapping[str, float]) -> dict[str, Any]:
    ordered = sorted(
        ((str(key), float(value)) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not ordered:
        return {
            "argmax_id": None,
            "argmax_confidence": None,
            "second_confidence": None,
            "margin": None,
        }
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    return {
        "argmax_id": ordered[0][0],
        "argmax_confidence": ordered[0][1],
        "second_confidence": second,
        "margin": ordered[0][1] - second,
    }


def _output_payload(
    result: Any,
    *,
    target_goal_id: str,
    target_content_id: str,
    score_fields: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    goal_scores = score_fields["goal"]
    content_scores = score_fields["content"]
    goal_rank = _rank(goal_scores)
    content_rank = _rank(content_scores)
    returned_goal_id = None if result.goal is None else str(result.goal.goal_id)
    returned_content_id = (
        None if result.content_plan is None else str(result.content_plan.content_id)
    )
    return {
        "status": str(result.status),
        "confidence": float(result.confidence),
        "ambiguity": float(result.ambiguity),
        "fact_scores": _jsonable_scores(score_fields["fact"]),
        "goal_scores": _jsonable_scores(goal_scores),
        "content_scores": _jsonable_scores(content_scores),
        "goal_rank": goal_rank,
        "content_rank": content_rank,
        "target_goal_id": target_goal_id,
        "target_content_id": target_content_id,
        "returned_goal_id": returned_goal_id,
        "returned_content_id": returned_content_id,
        "argmax_goal_hit": goal_rank["argmax_id"] == target_goal_id,
        "argmax_content_hit": content_rank["argmax_id"] == target_content_id,
        "returned_goal_hit": returned_goal_id == target_goal_id,
        "returned_content_hit": returned_content_id == target_content_id,
        "none_fields": {
            "goal": returned_goal_id is None,
            "content": returned_content_id is None,
        },
        "world_relation_count": (None if result.world is None else len(result.world.relations)),
        "world_uncertainty": (None if result.world is None else float(result.world.uncertainty)),
    }


def _resource_rss() -> tuple[int | None, str]:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss), "psutil_process_rss"
    except Exception:
        return None, "unavailable"


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get(key, False)) for row in rows) / len(rows)


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    transition_status_counts: dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    planner_reason_counts: dict[str, int] = defaultdict(int)
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        status_counts[str(row["semantic"]["status"])] += 1
        transition_status_counts[str(row["transition"]["status"])] += 1
        action_counts[str(row["action_chain"]["planner_status"])] += 1
        reason = row["action_chain"].get("planner_reason_code")
        if reason:
            planner_reason_counts[str(reason)] += 1
        by_class[str(row["class_key"])].append(row)

    def class_summary(class_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(class_rows),
            "semantic_argmax_goal_hit_rate": _rate(
                [row["semantic"] for row in class_rows], "argmax_goal_hit"
            ),
            "semantic_argmax_content_hit_rate": _rate(
                [row["semantic"] for row in class_rows], "argmax_content_hit"
            ),
            "semantic_returned_goal_hit_rate": _rate(
                [row["semantic"] for row in class_rows], "returned_goal_hit"
            ),
            "semantic_returned_content_hit_rate": _rate(
                [row["semantic"] for row in class_rows], "returned_content_hit"
            ),
            "transition_argmax_goal_hit_rate": _rate(
                [row["transition"] for row in class_rows], "argmax_goal_hit"
            ),
            "transition_argmax_content_hit_rate": _rate(
                [row["transition"] for row in class_rows], "argmax_content_hit"
            ),
            "transition_returned_goal_hit_rate": _rate(
                [row["transition"] for row in class_rows], "returned_goal_hit"
            ),
            "transition_returned_content_hit_rate": _rate(
                [row["transition"] for row in class_rows], "returned_content_hit"
            ),
            "planner_accept_rate": _rate(
                [row["action_chain"] for row in class_rows], "planner_accepted"
            ),
            "workbench_success_rate": _rate(
                [row["action_chain"] for row in class_rows], "workbench_success"
            ),
        }

    return {
        "row_count": len(rows),
        "semantic_status_counts": dict(sorted(status_counts.items())),
        "transition_status_counts": dict(sorted(transition_status_counts.items())),
        "planner_status_counts": dict(sorted(action_counts.items())),
        "planner_reason_counts": dict(sorted(planner_reason_counts.items())),
        "input_confidence_below_k1_floor_rate": _rate(rows, "input_confidence_below_k1_floor"),
        "input_confidence_below_k2_floor_rate": _rate(rows, "input_confidence_below_k2_floor"),
        "semantic_argmax_goal_hit_rate": _rate(
            [row["semantic"] for row in rows], "argmax_goal_hit"
        ),
        "semantic_argmax_content_hit_rate": _rate(
            [row["semantic"] for row in rows], "argmax_content_hit"
        ),
        "semantic_returned_goal_hit_rate": _rate(
            [row["semantic"] for row in rows], "returned_goal_hit"
        ),
        "semantic_returned_content_hit_rate": _rate(
            [row["semantic"] for row in rows], "returned_content_hit"
        ),
        "transition_argmax_goal_hit_rate": _rate(
            [row["transition"] for row in rows], "argmax_goal_hit"
        ),
        "transition_argmax_content_hit_rate": _rate(
            [row["transition"] for row in rows], "argmax_content_hit"
        ),
        "transition_returned_goal_hit_rate": _rate(
            [row["transition"] for row in rows], "returned_goal_hit"
        ),
        "transition_returned_content_hit_rate": _rate(
            [row["transition"] for row in rows], "returned_content_hit"
        ),
        "semantic_goal_none_rate": _rate([row["semantic"] for row in rows], "none_goal"),
        "semantic_content_none_rate": _rate([row["semantic"] for row in rows], "none_content"),
        "transition_goal_none_rate": _rate([row["transition"] for row in rows], "none_goal"),
        "transition_content_none_rate": _rate([row["transition"] for row in rows], "none_content"),
        "model_output_correct_rate": _rate(rows, "model_output_correct"),
        "planner_accept_rate": _rate([row["action_chain"] for row in rows], "planner_accepted"),
        "workbench_success_rate": _rate([row["action_chain"] for row in rows], "workbench_success"),
        "failure_recovery_status_counts": dict(
            sorted(
                {
                    status: sum(
                        1
                        for row in rows
                        if row["action_chain"]["failure_recovery_status"] == status
                    )
                    for status in {row["action_chain"]["failure_recovery_status"] for row in rows}
                }.items()
            )
        ),
        "by_class": {key: class_summary(by_class[key]) for key in sorted(by_class)},
    }


def _build_validation_cases(
    *,
    scratch: Path,
    validation: Sequence[Any],
    validation_metadata: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    schema = _schema()
    for experience, metadata in zip(validation, validation_metadata, strict=True):
        index = int(metadata["index"])
        class_key = str(metadata["class_key"])
        variant_paths = tuple(str(item) for item in metadata["variant_paths"])
        state_profile = str(metadata["state_profile"])
        root = scratch / f"validation-{index:04d}"
        root.mkdir(parents=True, exist_ok=False)
        _prepare_workspace(
            root,
            task_seed=2000 + index,
            first_path=variant_paths[0],
            state_profile=state_profile,
        )
        observations = _observations(
            root,
            paths=variant_paths,
            split=f"p1-validation-{index:04d}",
            project_id="p1-validation-project",
            state_profile=state_profile,
            schema=schema,
        )
        anchor = observations[0]
        current = observations[1]
        expected_percept = experience.semantic_example.percept
        expected_event = experience.transition_example.event
        if current.observation_digest != experience.observation_digest:
            mismatches.append(
                {
                    "index": index,
                    "field": "observation_digest",
                    "expected": experience.observation_digest,
                    "actual": current.observation_digest,
                }
            )
        if content_digest(current.to_percept_event(tick=1).to_payload()) != content_digest(
            expected_percept.to_payload()
        ):
            mismatches.append({"index": index, "field": "semantic_percept"})
        if content_digest(current.to_percept_event(tick=1).to_payload()) != content_digest(
            expected_event.to_payload()
        ):
            mismatches.append({"index": index, "field": "transition_event"})
        cases.append(
            {
                "index": index,
                "class_key": class_key,
                "state_profile": state_profile,
                "variant_paths": list(variant_paths),
                "root": root,
                "registry": _registry_for_state(variant_paths[0], state_profile),
                "observation": current,
                "before_world": _world(anchor, tick=0),
                "experience": experience,
                "target_goal_id": str(experience.semantic_example.goal.goal_id),
                "target_content_id": str(experience.semantic_example.content.content_id),
            }
        )
    return cases, mismatches


def _action_chain(
    *,
    case: Mapping[str, Any],
    semantic_result: Any,
    transition_result: Any,
    planner: NativeReadOnlyIntentPlanner,
) -> dict[str, Any]:
    observation = case["observation"]
    root = case["root"]
    registry = case["registry"]
    payload: dict[str, Any] = {
        "k1_output_present": semantic_result.goal is not None
        and semantic_result.content_plan is not None,
        "k2_output_present": transition_result.goal is not None
        and transition_result.content_plan is not None,
        "world_matches_observation": False,
        "planner_status": "not_attempted",
        "planner_reason_code": None,
        "planner_accepted": False,
        "workbench_status": "not_attempted",
        "workbench_success": False,
        "workbench_error_code": None,
        "failure_recovery_status": "not_triggered",
    }
    if transition_result.world is None:
        payload["planner_status"] = "k2_world_missing"
        return payload
    payload["world_matches_observation"] = bool(
        planner._world_matches_observation(transition_result.world, observation)
    )
    if semantic_result.goal is None or semantic_result.content_plan is None:
        payload["planner_status"] = "k1_output_missing"
        return payload
    if transition_result.goal is None or transition_result.content_plan is None:
        payload["planner_status"] = "k2_output_missing"
        return payload

    import seed_platform.workbench as workbench_module

    original_get_setting = workbench_module.get_setting
    workbench_module.get_setting = lambda key, default=None: (
        str(root) if key == "workspace_path" else default
    )
    try:
        runtime = SeedRuntime(
            Seed(
                SeedConfig(taiji=TaijiConfig(seed=MODEL_SEED)),
                episode_id=f"taiji-m5-k-p2-1-{case['class_key']}-{case['index']}",
            )
        )
        runtime._workbench_environment = WorkbenchEnvironment(
            root=root,
            programming_language_registry=registry,
        )
        snapshot = runtime.workbench_environment.capability_snapshot
        payload["snapshot_match"] = (
            observation.capability_snapshot_id == snapshot.snapshot_id
            and observation.capability_revision == snapshot.revision
        )
        decision = planner.propose(
            observation=observation,
            world=transition_result.world,
            goal=semantic_result.goal,
            content=semantic_result.content_plan,
            capability_snapshot=snapshot,
            tick=1,
        )
        payload["planner_status"] = "accepted" if decision.accepted else "rejected"
        payload["planner_reason_code"] = decision.reason_code
        payload["planner_accepted"] = bool(decision.accepted)
        if not decision.accepted or decision.action_intent is None:
            return payload
        outcome = runtime.execute_workbench_intent(
            decision.action_intent,
            snapshot_id=snapshot.snapshot_id,
            learn=False,
        )
        outcome_payload = dict(outcome.get("outcome") or {})
        payload["workbench_status"] = str(outcome_payload.get("status", "unknown"))
        payload["workbench_success"] = bool(outcome_payload.get("success", False))
        payload["workbench_error_code"] = outcome_payload.get("error_code")
        if not payload["workbench_success"]:
            payload["failure_recovery_status"] = "not_implemented_in_read_only_probe"
        return payload
    except (TypeError, ValueError, RuntimeError, OSError) as exc:
        payload["planner_status"] = "diagnostic_error"
        payload["planner_reason_code"] = f"{type(exc).__name__}: {exc}"
        return payload
    finally:
        workbench_module.get_setting = original_get_setting


def _diagnose_arm(
    *,
    arm_name: str,
    arm_payload: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint = arm_payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"P2 arm has no checkpoint block: {arm_name}")
    k1_path = Path(str(checkpoint["k1"]["path"]))
    k2_path = Path(str(checkpoint["k2"]["path"]))
    k1_payload = _load_mapping(k1_path)
    k2_payload = _load_mapping(k2_path)
    semantic = StructuredSemanticLearner.from_checkpoint(k1_payload, device="cpu")
    transition = StructuredSemanticTransitionLearner.from_checkpoint(k2_payload, device="cpu")
    restore = {
        "k1_digest_equal": content_digest(semantic.checkpoint()) == content_digest(k1_payload),
        "k2_digest_equal": content_digest(transition.checkpoint()) == content_digest(k2_payload),
    }
    restore["passed"] = bool(restore["k1_digest_equal"] and restore["k2_digest_equal"])
    planner = NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES))
    rows: list[dict[str, Any]] = []
    rss_before, rss_method = _resource_rss()
    process_start = time.process_time()
    started = time.perf_counter()
    for case in cases:
        experience = case["experience"]
        semantic_result = semantic.predict(experience.semantic_example.percept)
        teacher_forced_transition_result = transition.predict(
            experience.transition_example.before,
            experience.transition_example.event,
        )
        if semantic_result.world is None:
            raise RuntimeError(f"K1 did not produce a world for validation row {case['index']}")
        transition_result = transition.predict(
            semantic_result.world,
            experience.transition_example.event,
        )
        score = _loss_score(semantic, transition, (experience,))
        semantic_payload = _output_payload(
            semantic_result,
            target_goal_id=case["target_goal_id"],
            target_content_id=case["target_content_id"],
            score_fields={
                "fact": semantic_result.fact_scores,
                "goal": semantic_result.goal_scores,
                "content": semantic_result.content_scores,
            },
        )
        transition_payload = _output_payload(
            transition_result,
            target_goal_id=str(experience.transition_example.goal.goal_id),
            target_content_id=str(experience.transition_example.content.content_id),
            score_fields={
                "fact": transition_result.fact_scores,
                "goal": transition_result.goal_scores,
                "content": transition_result.content_scores,
            },
        )
        teacher_forced_transition_payload = _output_payload(
            teacher_forced_transition_result,
            target_goal_id=str(experience.transition_example.goal.goal_id),
            target_content_id=str(experience.transition_example.content.content_id),
            score_fields={
                "fact": teacher_forced_transition_result.fact_scores,
                "goal": teacher_forced_transition_result.goal_scores,
                "content": teacher_forced_transition_result.content_scores,
            },
        )
        action = _action_chain(
            case=case,
            semantic_result=semantic_result,
            transition_result=transition_result,
            planner=planner,
        )
        row = {
            "index": int(case["index"]),
            "class_key": str(case["class_key"]),
            "state_profile": str(case["state_profile"]),
            "variant_paths": list(case["variant_paths"]),
            "combined_mse": float(score["combined_mse"]),
            "input_confidence": float(experience.semantic_example.percept.confidence),
            "input_prediction_error": float(experience.semantic_example.percept.prediction_error),
            "input_confidence_below_k1_floor": bool(
                experience.semantic_example.percept.confidence < semantic.confidence_floor
            ),
            "input_confidence_below_k2_floor": bool(
                experience.transition_example.event.confidence < transition.confidence_floor
            ),
            "semantic_mse": float(
                (score["k1.fact_mse"] + score["k1.goal_mse"] + score["k1.content_mse"]) / 3.0
            ),
            "transition_mse": float(
                (score["k2.transition_mse"] + score["k2.goal_mse"] + score["k2.content_mse"]) / 3.0
            ),
            "semantic": semantic_payload,
            "transition": transition_payload,
            "transition_teacher_forced": teacher_forced_transition_payload,
            "model_output_correct": bool(
                semantic_payload["returned_goal_hit"]
                and semantic_payload["returned_content_hit"]
                and transition_payload["returned_goal_hit"]
                and transition_payload["returned_content_hit"]
            ),
            "action_chain": action,
        }
        row["semantic"]["none_goal"] = bool(row["semantic"]["none_fields"]["goal"])
        row["semantic"]["none_content"] = bool(row["semantic"]["none_fields"]["content"])
        row["transition"]["none_goal"] = bool(row["transition"]["none_fields"]["goal"])
        row["transition"]["none_content"] = bool(row["transition"]["none_fields"]["content"])
        rows.append(row)
    rss_after, _ = _resource_rss()
    elapsed = time.perf_counter() - started
    return {
        "checkpoint": {
            "k1": {"path": str(k1_path), "bytes": k1_path.stat().st_size},
            "k2": {"path": str(k2_path), "bytes": k2_path.stat().st_size},
        },
        "parameter_count": {
            "k1": int(semantic.parameter_count),
            "k2": int(transition.parameter_count),
            "total": int(semantic.parameter_count + transition.parameter_count),
        },
        "readout_contract": {
            "k1_confidence_floor": float(semantic.confidence_floor),
            "k1_fact_threshold": float(semantic.fact_threshold),
            "k1_ambiguity_ceiling": float(semantic.ambiguity_ceiling),
            "k2_confidence_floor": float(transition.confidence_floor),
            "k2_fact_threshold": float(transition.fact_threshold),
            "k2_ambiguity_ceiling": float(transition.ambiguity_ceiling),
        },
        "restore": restore,
        "resources": {
            "inference_rows": len(rows),
            "elapsed_seconds": elapsed,
            "process_cpu_seconds": time.process_time() - process_start,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": (
                None if rss_before is None or rss_after is None else rss_after - rss_before
            ),
            "rss_method": rss_method,
        },
        "output_diagnostics": _aggregate_rows(rows),
        "rows": rows,
    }


def _diagnosis(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    evidence: list[str] = []
    route_gaps: set[str] = set()
    for arm_name, arm in arms.items():
        rows = arm["rows"]
        for row in rows:
            target_content = str(row["semantic"]["target_content_id"])
            if target_content not in ROUTE_CONTENT_IDS:
                route_gaps.add(target_content)
        summary = arm["output_diagnostics"]
        low_input_count = sum(1 for row in rows if row["input_confidence_below_k1_floor"])
        if low_input_count:
            evidence.append(
                f"{arm_name}: {low_input_count}/{len(rows)} rows are below the native K1 "
                "confidence floor and therefore abstain before goal/content readout"
            )
        for reason, count in arm["output_diagnostics"]["planner_reason_counts"].items():
            if reason == "stale_world_observation":
                evidence.append(
                    f"{arm_name}: {count} planner rejection(s) are caused by K2 world/observation drift"
                )
        if (
            summary["semantic_argmax_content_hit_rate"]
            > summary["semantic_returned_content_hit_rate"]
            and summary["semantic_content_none_rate"] > 0.0
        ):
            evidence.append(
                f"{arm_name}: K1 content argmax exceeds returned content hit while None output exists"
            )
        if summary["planner_accept_rate"] == 0.0 and summary["model_output_correct_rate"] > 0.0:
            evidence.append(
                f"{arm_name}: at least one fully correct K1/K2 row still has zero planner acceptance"
            )
    if route_gaps:
        evidence.append(
            "existing read-only policy has no route for " + ", ".join(sorted(route_gaps))
        )
    if not evidence:
        evidence.append("no single readout/action cause was isolated by this bounded probe")
    return {
        "route_content_ids": sorted(ROUTE_CONTENT_IDS),
        "missing_route_content_ids": sorted(route_gaps),
        "evidence": evidence,
        "interpretation": (
            "diagnostic-only; continuous fit cannot be promoted until discrete output, "
            "planner admission, Workbench success, retention, and resource gates are measured"
        ),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_diagnostic(
    *,
    pilot_report: Path = DEFAULT_PILOT_REPORT,
    manifest_path: Path = P1_MANIFEST,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    pilot = json.loads(pilot_report.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "pilot_report": str(pilot_report),
        "manifest": str(manifest_path),
    }
    scratch = DEFAULT_OUTPUT_ROOT / f"_taiji_m5_k_p2_1_scratch_{uuid4().hex}"
    try:
        if pilot.get("status") != "completed":
            raise ValueError("P2.1 requires a completed P2 pilot report")
        if pilot.get("sealed_payload_read"):
            raise ValueError("P2.1 refuses a pilot report that read sealed payload")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts, parent_digest, bundle, projector = _context(
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
            raise RuntimeError("P1 v2 manifest reconstruction failed during P2.1")
        cases, case_mismatches = _build_validation_cases(
            scratch=scratch / "cases",
            validation=validation,
            validation_metadata=validation_metadata,
        )
        if case_mismatches:
            raise RuntimeError(
                f"validation observation reconstruction mismatch: {case_mismatches[:3]}"
            )

        arm_results: dict[str, Any] = {}
        for arm_name in ("frozen", "wake-only", "wake-replay"):
            arm_payload = pilot.get("arms", {}).get(arm_name)
            if not isinstance(arm_payload, Mapping):
                raise ValueError(f"P2 pilot is missing arm: {arm_name}")
            arm_results[arm_name] = _diagnose_arm(
                arm_name=arm_name,
                arm_payload=arm_payload,
                cases=cases,
            )
        payload.update(
            {
                "status": "completed",
                "training_performed": False,
                "manifest_reconstruction": manifest_verification,
                "validation_case_count": len(cases),
                "validation_case_classes": [str(case["class_key"]) for case in cases],
                "read_only_routes": [list(item) for item in READ_ONLY_ROUTES],
                "arms": arm_results,
                "diagnosis": _diagnosis(arm_results),
                "interpretation": (
                    "P2.1 read-only diagnostic; no fit, no sealed payload, no promotion"
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
    parser.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT_REPORT)
    parser.add_argument("--manifest", type=Path, default=P1_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run_diagnostic(
        pilot_report=args.pilot_report,
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

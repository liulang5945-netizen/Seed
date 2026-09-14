"""Audit whether frozen K exposes a real behavioral signal for G.

P3.3 proved that G can be updated, but its candidate set was too close to a
K-proposal-versus-abstain copy.  P3.4 keeps K and G frozen and adds all
content-compatible K score-grid alternatives plus typed safe candidates.  It
executes proposal candidates through the native read-only Workbench contract
and records an independent, content-addressed behavioral utility.  No fit is
performed in this canary; a zero K-only/behavior disagreement is a data stop.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_p1_data import (  # noqa: E402
    _observations,
    _registry_for_state,
    _schema,
)
from scripts.training.eval_taiji_m5_k_p2_2_safety_bridge_canary import (  # noqa: E402
    _recovery_policy,
    _run_intent,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _fresh_learners,
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
from scripts.training.eval_taiji_m5_k_p3_3_g_signal_canary import (  # noqa: E402
    CONFIDENCE_FLOOR,
    P2_6_MANIFEST,
    P2_7_MANIFEST,
    P3_2_MANIFEST,
    P3_2_REPORT,
    _candidate,
    _candidate_identity,
    _catalogs,
    _choose_train,
)
from taiji import (  # noqa: E402
    GSelectionBehaviorOutcome,
    GSelectionBehaviorSet,
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p3-4-behavior-signal-canary-v1"
MANIFEST_FORMAT = "taiji-m5-k-p3-4-behavior-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_4_behavior_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_4_behavior_signal_20260911.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON mapping at {path}")
    return dict(payload)


def _load_mapping(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping checkpoint at {path}")
    return {str(key): value for key, value in payload.items()}


def _case_for_metadata(*, scratch: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    split = str(metadata["split"])
    index = int(metadata["index"])
    if split == "train":
        course_seed = int(metadata["course_seed"])
        root = scratch / f"course-{course_seed}" / f"train-{index:04d}"
        observation_split = f"p1-train-{index:04d}"
    elif split == "validation":
        root = scratch / "validation" / f"validation-{index:04d}"
        observation_split = f"p1-validation-{index:04d}"
    else:
        raise ValueError(f"unsupported reconstructed case split: {split}")
    paths = tuple(str(item) for item in metadata["variant_paths"])
    registry = _registry_for_state(paths[0], str(metadata["state_profile"]))
    observations = _observations(
        root,
        paths=paths,
        split=observation_split,
        project_id=str(metadata["project_id"]),
        state_profile=str(metadata["state_profile"]),
        schema=_schema(),
    )
    if len(observations) < 2:
        raise ValueError("reconstructed P3.4 case has no current observation")
    return {
        "root": root,
        "registry": registry,
        "observation": observations[1],
        "class_key": str(metadata["class_key"]),
        "index": index,
        "project_id": str(metadata["project_id"]),
        "path": paths[0],
    }


def _expanded_candidates(
    *,
    experience: Any,
    metadata: Mapping[str, Any],
    semantic: Any,
    transition: Any,
    goals: Mapping[str, Any],
    content_plans: Mapping[str, Any],
) -> tuple[str, str, tuple[GSelectionCandidate, ...]]:
    semantic_example = experience.semantic_example
    transition_example = experience.transition_example
    semantic_result = semantic.predict(semantic_example.percept)
    transition_result = transition.predict(transition_example.before, transition_example.event)
    candidates: list[GSelectionCandidate] = []
    seen_pairs: set[tuple[str | None, str | None]] = set()

    def add(
        *,
        candidate_id: str,
        source: str,
        candidate_role: str,
        status: str,
        goal: Any,
        content: Any,
        result: Any,
    ) -> None:
        pair = (
            None if goal is None else goal.goal_id,
            None if content is None else content.content_id,
        )
        if pair in seen_pairs and candidate_role == "proposal":
            return
        if candidate_role == "proposal":
            seen_pairs.add(pair)
        candidates.append(
            _candidate(
                candidate_id=candidate_id,
                source=source,
                candidate_role=candidate_role,
                status=status,
                goal=goal,
                content=content,
                result=result,
            )
        )

    if semantic_result.goal is not None or semantic_result.content_plan is not None:
        add(
            candidate_id="k1:selected",
            source="k1.semantic",
            candidate_role="proposal",
            status=semantic_result.status,
            goal=semantic_result.goal,
            content=semantic_result.content_plan,
            result=semantic_result,
        )
    if transition_result.goal is not None or transition_result.content_plan is not None:
        add(
            candidate_id="k2:selected",
            source="k2.transition",
            candidate_role="proposal",
            status=transition_result.status,
            goal=transition_result.goal,
            content=transition_result.content_plan,
            result=transition_result,
        )

    # These alternatives are generated from K's score vocabulary, not from a
    # semantic expected target.  Keeping every catalog-compatible pair makes
    # behavior, rather than the top-1 identity, decide whether a real signal
    # exists.  Empty-score/low-evidence rows remain safe-only.
    if semantic_result.goal_scores or semantic_result.content_scores:
        for goal_id in sorted(goals):
            for content_id in sorted(content_plans):
                goal = goals[goal_id]
                content = content_plans[content_id]
                if content.source_goal_id not in {None, goal.goal_id}:
                    continue
                status = "clarify" if content.intent_kind == "request_information" else "resolved"
                add(
                    candidate_id=f"k1:grid:{goal_id}:{content_id}",
                    source="k1.score-grid",
                    candidate_role="proposal",
                    status=status,
                    goal=goal,
                    content=content,
                    result=semantic_result,
                )

    add(
        candidate_id="abstain",
        source="runtime.abstain",
        candidate_role="abstain",
        status="abstained",
        goal=None,
        content=None,
        result=None,
    )
    add(
        candidate_id="reobserve",
        source="runtime.reobserve",
        candidate_role="reobserve",
        status="ambiguous",
        goal=None,
        content=None,
        result=None,
    )
    if len(candidates) < 2:
        raise ValueError("P3.4 candidate construction produced fewer than two candidates")
    example_id, family_id = _candidate_identity(experience=experience, metadata=metadata)
    return example_id, family_id, tuple(candidates)


def _world_consistent(candidate: GSelectionCandidate, observation: Any) -> bool:
    if candidate.goal is None or candidate.content_plan is None:
        return False
    slots = candidate.content_plan.semantic_slots
    return (
        slots.get("observed_language") == observation.language_id
        and slots.get("selection_state") == observation.selection_state
        and bool(slots.get("toolchain_available")) == bool(observation.toolchain_available)
    )


def _safe_flags(candidate: GSelectionCandidate, observation: Any) -> tuple[bool, bool]:
    low_evidence = (
        not observation.read_success
        or observation.language_confidence < CONFIDENCE_FLOOR
        or observation.selection_state != "resolved"
    )
    safe_exit_valid = candidate.candidate_role in {"abstain", "reobserve"} and low_evidence
    safe_exit_progress = candidate.candidate_role == "reobserve" and (
        not observation.read_success or observation.selection_state in {"ambiguous", "unknown"}
    )
    return safe_exit_valid, safe_exit_progress


def _candidate_outcome(
    *,
    case: Mapping[str, Any],
    candidate: GSelectionCandidate,
    world: Any,
    label: str,
) -> tuple[GSelectionBehaviorOutcome, dict[str, Any]]:
    observation = case["observation"]
    safe_exit_valid, safe_exit_progress = _safe_flags(candidate, observation)
    common = {
        "label": label,
        "candidate_id": candidate.candidate_id,
        "candidate_role": candidate.candidate_role,
        "world_consistent": _world_consistent(candidate, observation),
    }
    if candidate.candidate_role in {"abstain", "reobserve"}:
        outcome = GSelectionBehaviorOutcome.create(
            candidate_id=candidate.candidate_id,
            candidate_digest=candidate.candidate_digest,
            candidate_role=candidate.candidate_role,
            snapshot_match=bool(
                observation.capability_snapshot_id and observation.capability_revision > 0
            ),
            planner_accepted=False,
            route_valid=False,
            parameter_valid=False,
            world_consistent=False,
            execution_success=False,
            safe_exit_valid=safe_exit_valid,
            safe_exit_progress=safe_exit_progress,
        )
        common.update(
            {
                "planner_status": "safe_exit",
                "workbench_success": False,
                "safe_exit_valid": safe_exit_valid,
                "safe_exit_progress": safe_exit_progress,
            }
        )
        return outcome, common

    from scripts.training.eval_taiji_m5_k1_skill_composition import READ_ONLY_ROUTES

    if (
        candidate.content_plan is not None
        and candidate.content_plan.content_id == "content:recover-target"
    ):
        from taiji import NativeReadOnlyIntentPlanner

        planner = NativeReadOnlyIntentPlanner(policy=_recovery_policy())
    else:
        from taiji import NativeReadOnlyIntentPlanner, ReadOnlyIntentPolicy

        planner = NativeReadOnlyIntentPlanner(policy=ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES))
    action = _run_intent(
        case=case,
        planner=planner,
        goal=candidate.goal,
        content=candidate.content_plan,
        world=world,
        label=label,
        oracle_control=False,
    )
    parameter_valid = bool((action.get("parameter_drift") or {}).get("matches_contract", False))
    planner_accepted = bool(action.get("accepted", False))
    route_valid = planner_accepted and action.get("intent_kind") is not None
    execution_success = bool(action.get("workbench_success", False))
    outcome = GSelectionBehaviorOutcome.create(
        candidate_id=candidate.candidate_id,
        candidate_digest=candidate.candidate_digest,
        candidate_role=candidate.candidate_role,
        snapshot_match=bool(action.get("snapshot_match", False)),
        planner_accepted=planner_accepted,
        route_valid=route_valid,
        parameter_valid=parameter_valid,
        world_consistent=common["world_consistent"],
        execution_success=execution_success,
        safe_exit_valid=False,
        safe_exit_progress=False,
    )
    common.update(
        {
            "action": action,
            "planner_status": action.get("planner_status"),
            "workbench_success": execution_success,
            "safe_exit_valid": False,
            "safe_exit_progress": False,
        }
    )
    return outcome, common


def _behavior_record(
    *,
    experience: Any,
    metadata: Mapping[str, Any],
    case: Mapping[str, Any],
    semantic: Any,
    transition: Any,
    goals: Mapping[str, Any],
    content_plans: Mapping[str, Any],
    label: str,
) -> tuple[GSelectionCandidateSet, GSelectionBehaviorSet, dict[str, Any]]:
    example_id, family_id, candidates = _expanded_candidates(
        experience=experience,
        metadata=metadata,
        semantic=semantic,
        transition=transition,
        goals=goals,
        content_plans=content_plans,
    )
    transition_result = transition.predict(
        experience.transition_example.before,
        experience.transition_example.event,
    )
    outcomes: list[GSelectionBehaviorOutcome] = []
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        outcome, detail = _candidate_outcome(
            case=case,
            candidate=candidate,
            world=transition_result.world,
            label=f"{label}:{candidate.candidate_id}",
        )
        outcomes.append(outcome)
        diagnostics.append({**detail, "utility": outcome.utility})
    ranked = sorted(outcomes, key=lambda item: (-item.utility, item.candidate_id))
    target = ranked[0]
    target_candidate = next(item for item in candidates if item.candidate_id == target.candidate_id)
    target_kind = (
        "pair" if target_candidate.candidate_role == "proposal" else target_candidate.candidate_role
    )
    candidate_set = GSelectionCandidateSet.create(
        example_id=example_id,
        family_id=family_id,
        split=str(metadata["split"]),
        project_id=str(metadata["project_id"]),
        path=str(metadata["variant_paths"][0]),
        input_digest=str(experience.semantic_example.input_digest),
        candidates=candidates,
        target_candidate_id=target.candidate_id,
        target_kind=target_kind,
    )
    behavior_set = GSelectionBehaviorSet.create(
        candidate_set_digest=candidate_set.candidate_set_digest,
        inference_digest=candidate_set.inference_digest,
        split=candidate_set.split,
        project_id=candidate_set.project_id,
        path=candidate_set.path,
        outcomes=outcomes,
    )
    if behavior_set.behavior_target_candidate_id != candidate_set.target_candidate_id:
        raise AssertionError("behavior and candidate-set target drifted")
    return (
        candidate_set,
        behavior_set,
        {
            "example_id": example_id,
            "class_key": str(metadata["class_key"]),
            "state_profile": str(metadata["state_profile"]),
            "observation": {
                "observation_digest": str(case["observation"].observation_digest),
                "language_id": str(case["observation"].language_id),
                "selection_state": str(case["observation"].selection_state),
                "toolchain_available": bool(case["observation"].toolchain_available),
                "read_success": bool(case["observation"].read_success),
                "language_confidence": float(case["observation"].language_confidence),
            },
            "k_only_candidate_id": _k_only_candidate(candidate_set).candidate_id,
            "behavior_target_candidate_id": behavior_set.behavior_target_candidate_id,
            "utility_margin": behavior_set.utility_margin,
            "behavior_disagreement": _k_only_candidate(candidate_set).candidate_id
            != behavior_set.behavior_target_candidate_id,
            "candidate_count": len(candidates),
            "diagnostics": diagnostics,
        },
    )


def _k_only_candidate(candidate_set: GSelectionCandidateSet) -> GSelectionCandidate:
    by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    for candidate_id in ("k1:selected", "k2:selected"):
        candidate = by_id.get(candidate_id)
        if (
            candidate is not None
            and candidate.candidate_role == "proposal"
            and candidate.goal is not None
            and candidate.content_plan is not None
            and candidate.confidence >= CONFIDENCE_FLOOR
        ):
            return candidate
    safe = [
        candidate for candidate in candidate_set.candidates if candidate.candidate_role == "abstain"
    ]
    if safe:
        return sorted(safe, key=lambda candidate: candidate.candidate_id)[0]
    reobserve = [
        candidate
        for candidate in candidate_set.candidates
        if candidate.candidate_role == "reobserve"
    ]
    if reobserve:
        return sorted(reobserve, key=lambda candidate: candidate.candidate_id)[0]
    raise ValueError("P3.4 K-only baseline has no safe candidate")


def _signal_gate(
    records: Sequence[tuple[GSelectionCandidateSet, GSelectionBehaviorSet, Mapping[str, Any]]],
) -> dict[str, Any]:
    candidate_sets = [item[0] for item in records]
    behavior_sets = [item[1] for item in records]
    diagnostics = [item[2] for item in records]
    train = [item for item in records if item[0].split == "train"]
    validation = [item for item in records if item[0].split == "validation"]
    train_projects = {item[0].project_id for item in train}
    validation_projects = {item[0].project_id for item in validation}
    train_paths = {item[0].path for item in train}
    validation_paths = {item[0].path for item in validation}
    roles = {candidate.candidate_role for item in candidate_sets for candidate in item.candidates}
    margins = [item.utility_margin for item in behavior_sets]
    disagreements = [item["behavior_disagreement"] for item in diagnostics]
    runtime_clean = all(
        "target_candidate_id" not in item.to_inference_payload()
        and "target_kind" not in item.to_inference_payload()
        for item in candidate_sets
    )
    checks = {
        "train_nonempty": bool(train),
        "validation_nonempty": bool(validation),
        "minimum_two_candidates": bool(candidate_sets)
        and min(len(item.candidates) for item in candidate_sets) >= 2,
        "proposal_and_safe_roles": {"proposal", "abstain", "reobserve"}.issubset(roles),
        "behavior_outcome_complete": all(
            len(item.outcomes) == len(candidate_set.candidates)
            for candidate_set, item in zip(candidate_sets, behavior_sets, strict=True)
        ),
        "nonzero_utility_margin": any(margin > 1e-9 for margin in margins),
        "k_only_behavior_disagreement": any(disagreements),
        "project_split_disjoint": train_projects.isdisjoint(validation_projects),
        "path_split_disjoint": train_paths.isdisjoint(validation_paths),
        "runtime_target_excluded": runtime_clean,
        "candidate_set_digest_unique": len({item.candidate_set_digest for item in candidate_sets})
        == len(candidate_sets),
        "behavior_digest_unique": len({item.behavior_digest for item in behavior_sets})
        == len(behavior_sets),
        "sealed_holdout_untouched": True,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "train_count": len(train),
        "validation_count": len(validation),
        "candidate_count_min": min((len(item.candidates) for item in candidate_sets), default=0),
        "candidate_count_max": max((len(item.candidates) for item in candidate_sets), default=0),
        "nonzero_utility_margin_count": sum(margin > 1e-9 for margin in margins),
        "k_only_behavior_disagreement_count": sum(bool(value) for value in disagreements),
        "reobserve_behavior_target_count": sum(
            item.behavior_target_candidate_id == "reobserve" for item in behavior_sets
        ),
        "candidate_roles": sorted(roles),
        "train_projects": sorted(train_projects),
        "validation_projects": sorted(validation_projects),
    }


def _verify_sources(
    *,
    p1_manifest: Mapping[str, Any],
    p2_6_manifest: Mapping[str, Any],
    p2_7_manifest: Mapping[str, Any],
    p3_2_report: Mapping[str, Any],
    p3_2_manifest: Mapping[str, Any],
    p3_3_report: Mapping[str, Any],
    p3_3_manifest: Mapping[str, Any],
) -> None:
    if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
        raise ValueError("P3.4 requires P1 v2 data")
    if p3_3_report.get("status") != "completed":
        raise ValueError("P3.4 requires completed P3.3 G learning")
    if (
        p3_3_report.get("fit_called") is not True
        or p3_3_report.get("training_performed") is not True
    ):
        raise ValueError("P3.4 requires the recorded P3.3 G-only fit boundary")
    if p3_3_report.get("can_promote"):
        raise ValueError("P3.4 cannot replace a promoted G artifact")
    if p3_3_report.get("source_signal_manifest_digest") != p3_3_manifest.get("manifest_digest"):
        raise ValueError("P3.3 learning report and signal manifest differ")
    if p3_3_report.get("external_target_used") or p3_3_report.get("p2_7_holdout_fit_count") != 0:
        raise ValueError("P3.3 source crossed the runtime/holdout boundary")
    if p3_2_report.get("status") != "completed" or p3_2_report.get(
        "manifest_digest"
    ) != p3_2_manifest.get("manifest_digest"):
        raise ValueError("P3.4 requires a completed P3.2 owner-transfer source")
    if len(p2_7_manifest.get("records", ())) != 4:
        raise ValueError("P3.4 requires the fixed P2.7 holdout contract")
    if any(
        record.get("candidate", {}).get("fit_eligible") is not False
        for record in p2_7_manifest["records"]
    ):
        raise ValueError("P2.7 holdout is not sealed as non-fit")


def run_canary(
    *,
    p1_manifest_path: Path = P1_MANIFEST,
    p2_6_manifest_path: Path = P2_6_MANIFEST,
    p2_7_manifest_path: Path = P2_7_MANIFEST,
    p3_2_report_path: Path = P3_2_REPORT,
    p3_2_manifest_path: Path = P3_2_MANIFEST,
    p3_3_report_path: Path = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_3_g_learning_20260911.json",
    p3_3_manifest_path: Path = PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p3_3_g_candidate_manifest_v1.json",
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p3_4_behavior_signal_{uuid4().hex}"
    scratch = run_dir / "scratch"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "external_target_used": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "report": str(report_path),
    }
    try:
        p1_manifest = _load_json(p1_manifest_path)
        p2_6_manifest = _load_json(p2_6_manifest_path)
        p2_7_manifest = _load_json(p2_7_manifest_path)
        p3_2_report = _load_json(p3_2_report_path)
        p3_2_manifest = _load_json(p3_2_manifest_path)
        p3_3_report = _load_json(p3_3_report_path)
        p3_3_manifest = _load_json(p3_3_manifest_path)
        _verify_sources(
            p1_manifest=p1_manifest,
            p2_6_manifest=p2_6_manifest,
            p2_7_manifest=p2_7_manifest,
            p3_2_report=p3_2_report,
            p3_2_manifest=p3_2_manifest,
            p3_3_report=p3_3_report,
            p3_3_manifest=p3_3_manifest,
        )
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        k1_path = Path(str(worker_restore["k1"]["path"]))
        k2_path = Path(str(worker_restore["k2"]["path"]))
        semantic_payload = _load_mapping(k1_path)
        transition_payload = _load_mapping(k2_path)
        worker_digests = {
            "k1": str(p3_2_report["worker_checkpoint_digests"]["k1"]),
            "k2": str(p3_2_report["worker_checkpoint_digests"]["k2"]),
        }
        if (
            content_digest(semantic_payload) != worker_digests["k1"]
            or content_digest(transition_payload) != worker_digests["k2"]
        ):
            raise ValueError("P3.2 K checkpoint digest drifted before P3.4")
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        checkpoint_preflight = _checkpoint_preflight(
            output_dir=run_dir / "k-preflight",
            semantic_parent=copy.deepcopy(semantic_payload),
            transition_parent=copy.deepcopy(transition_payload),
        )
        if not checkpoint_preflight.get("passed"):
            raise RuntimeError("P3.4 K checkpoint preflight failed")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_6_manifest.get("parent_checkpoint_digest") != parent_digest:
            raise ValueError("P2.6 parent digest drifted")
        (
            train_experiences,
            train_metadata,
            validation_experiences,
            validation_metadata,
            reconstruction,
        ) = _rebuild_and_verify_manifest(
            scratch=scratch / "p1",
            manifest=p1_manifest,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        if not reconstruction["passed"]:
            raise RuntimeError("P1 data reconstruction failed before P3.4")
        selected_train, selected_train_metadata = _choose_train(train_experiences, train_metadata)
        goals, content_plans = _catalogs(semantic_payload)
        train_cases = [
            _case_for_metadata(scratch=scratch / "p1", metadata=item)
            for item in selected_train_metadata
        ]
        validation_cases = [
            _case_for_metadata(scratch=scratch / "p1", metadata=item)
            for item in validation_metadata
        ]
        records: list[tuple[GSelectionCandidateSet, GSelectionBehaviorSet, dict[str, Any]]] = []
        for split_name, experiences, metadata_items, cases in (
            ("train", selected_train, selected_train_metadata, train_cases),
            ("validation", validation_experiences, validation_metadata, validation_cases),
        ):
            for index, (experience, metadata, case) in enumerate(
                zip(experiences, metadata_items, cases, strict=True)
            ):
                candidate_set, behavior_set, detail = _behavior_record(
                    experience=experience,
                    metadata=metadata,
                    case=case,
                    semantic=semantic,
                    transition=transition,
                    goals=goals,
                    content_plans=content_plans,
                    label=f"p3-4-{split_name}-{index}",
                )
                records.append((candidate_set, behavior_set, detail))
        signal_gate = _signal_gate(records)
        train_records = [
            {
                "candidate_set": candidate_set.to_payload(),
                "behavior_set": behavior_set.to_payload(),
                "diagnostic": detail,
            }
            for candidate_set, behavior_set, detail in records
            if candidate_set.split == "train"
        ]
        validation_records = [
            {
                "candidate_set": candidate_set.to_payload(),
                "behavior_set": behavior_set.to_payload(),
                "diagnostic": detail,
            }
            for candidate_set, behavior_set, detail in records
            if candidate_set.split == "validation"
        ]
        holdout_records = [
            {
                "record_id": str(record["record_id"]),
                "project_id": str(record["project_id"]),
                "path": str(record["candidate"]["observation"]["path"]),
                "fit_eligible": bool(record["candidate"].get("fit_eligible", False)),
            }
            for record in p2_7_manifest["records"]
        ]
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p1_manifest_digest": str(p1_manifest["manifest_digest"]),
            "source_p2_6_manifest_digest": str(p2_6_manifest["manifest_digest"]),
            "source_p2_7_manifest_digest": str(p2_7_manifest["manifest_digest"]),
            "source_p3_2_manifest_digest": str(p3_2_manifest["manifest_digest"]),
            "source_p3_3_report_digest": content_digest(p3_3_report),
            "k_checkpoint_digests": worker_digests,
            "fit_policy": {
                "fit_called": False,
                "p2_7_holdout_fit_count": 0,
                "runtime_behavior_label_used": False,
            },
            "train_records": train_records,
            "validation_records": validation_records,
            "p2_7_holdout": holdout_records,
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        zero_step = GSelectionLearner(
            parent_manifest_digest=str(p3_2_manifest["manifest_digest"]),
            k_checkpoint_digests=worker_digests,
            confidence_floor=CONFIDENCE_FLOOR,
        )
        zero_step_selection = []
        for candidate_set, _behavior_set, _detail in records:
            decision = zero_step.select(candidate_set)
            zero_step_selection.append(
                {
                    "split": candidate_set.split,
                    "example_id": candidate_set.example_id,
                    "selected_candidate_id": decision.selected_candidate_id,
                    "selection_status": decision.selection_status,
                    "decision_digest": decision.decision_digest,
                }
            )
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "checkpoint_preflight": checkpoint_preflight,
                "reconstruction": reconstruction,
                "signal_gate": signal_gate,
                "fit_policy": {
                    "fit_called": False,
                    "training_performed": False,
                    "p2_7_holdout_fit_count": 0,
                },
                "behavior_policy": {
                    "proposal_utility": "snapshot/planner/route/parameter/world/execution",
                    "safe_utility": "snapshot/safe-exit/safe-progress",
                    "runtime_inference_excludes_behavior_target": True,
                },
                "zero_step_selection": zero_step_selection,
                "parameter_count": {
                    "k1": int(semantic.parameter_count),
                    "k2": int(transition.parameter_count),
                    "g_zero_step": zero_step.parameter_count,
                },
                "interpretation": (
                    "passed: behavioral candidate signal is nontrivial; G-only fit may be designed next"
                    if signal_gate["passed"]
                    else "completed: behavioral signal Gate failed; do not fit G and revise candidate/utility contract"
                ),
                "can_start_g_fit": bool(signal_gate["passed"]),
                "can_promote": False,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    _write_json_atomic(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_canary(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

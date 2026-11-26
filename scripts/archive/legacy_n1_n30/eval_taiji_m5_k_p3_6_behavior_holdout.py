"""Run the independent P3.6 behavior-generalization and retention Gate.

P3.5 demonstrated that G-only fitting can change behavior on a non-zero-margin
cohort.  P3.6 deliberately performs no fit: it restores the P3.5 zero-step and
trained-G checkpoints, builds new project/path identities, recomputes behavior
utility through the real read-only Workbench boundary, and checks that the
trained choice generalizes without weakening old classes or safe exits.
"""

from __future__ import annotations

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
    ANCHOR_PATH,
    _observations,
    _prepare_workspace,
    _registry_for_state,
    _schema,
    _template_id,
)
from scripts.training.eval_taiji_m4v2_b3_k_single_step import _build_experience  # noqa: E402
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    WORKER_ROOT,
    _checkpoint_preflight,
    _context,
)
from scripts.training.eval_taiji_m5_k_p3_3_g_learning import (  # noqa: E402
    CONFIDENCE_FLOOR,
    P2_7_MANIFEST,
    P3_2_MANIFEST,
    P3_2_REPORT,
    _fresh_learners,
    _holdout_actions,
    _holdout_candidate_sets,
)
from scripts.training.eval_taiji_m5_k_p3_4_behavior_signal_canary import (  # noqa: E402
    P2_6_MANIFEST,
    _behavior_record,
)
from scripts.training.eval_taiji_m5_k_p3_5_g_learning import (  # noqa: E402
    P3_4_MANIFEST,
    P3_4_REPORT,
    _independent_g_restore,
    _load_json,
    _load_mapping,
    _runtime_snapshot,
    _select_row,
    _summarize,
)
from scripts.training.eval_taiji_m5_k_p3_5_g_learning import (
    _k_only_candidate as _p35_k_only_candidate,
)
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidateSet,
    GSelectionDecision,
    GSelectionLearner,
    NativeReadOnlyIntentPlanner,
    ReadOnlyAbstention,
    ReadOnlyIntentPolicy,
    WorkbenchObservationSchema,
    content_digest,
    project_g_decision,
)

REPORT_FORMAT = "taiji-m5-k-p3-6-behavior-holdout-v1"
MANIFEST_FORMAT = "taiji-m5-k-p3-6-behavior-holdout-manifest-v1"
VERSION = 1
UTILITY_MARGIN_EPSILON = 1e-9
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_6_behavior_holdout_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_6_behavior_holdout_20260911.json"
P3_5_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_5_g_learning_manifest_v1.json"
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"

# These names are intentionally outside every earlier P1/P2/P3 artifact.  The
# first path of each row is the scored observation; the remaining paths make
# the transition episode concrete without reusing an earlier path identity.
NEW_HOLDOUT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "index": 0,
        "class_key": "A",
        "project_id": "p3-6-holdout-project-a",
        "variant_paths": ("p36_a_python.py", "p36_a_rust.rs", "p36_a_python_b.py"),
        "state_profile": "resolved-language",
        "task_seed": 3600,
    },
    {
        "index": 1,
        "class_key": "R",
        "project_id": "p3-6-holdout-project-a",
        "variant_paths": ("p36_a_missing.txt", "p36_a_python_c.py", "p36_a_rust_b.rs"),
        "state_profile": "recovery-no-selection",
        "task_seed": 3601,
    },
    {
        "index": 2,
        "class_key": "C",
        "project_id": "p3-6-holdout-project-b",
        "variant_paths": ("p36_b_typescript.ts", "p36_b_python.py", "p36_b_rust.rs"),
        "state_profile": "ambiguous-language",
        "task_seed": 3602,
    },
    {
        "index": 3,
        "class_key": "D",
        "project_id": "p3-6-holdout-project-b",
        "variant_paths": ("p36_b_header.h", "p36_b_python_d.py", "p36_b_rust_d.rs"),
        "state_profile": "ambiguous-header",
        "task_seed": 3603,
    },
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _projection_observation_schema() -> WorkbenchObservationSchema:
    return WorkbenchObservationSchema(
        language_ids=("cpp", "python", "rust", "typescript", "unknown"),
        selection_states=("ambiguous", "resolved", "unknown"),
        task_kinds=("inspect-language",),
        extensions=("<none>", ".cpp", ".h", ".py", ".rs", ".ts"),
    )


def _materialize_new_case(
    *,
    scratch: Path,
    spec: Mapping[str, Any],
    parent_digest: str,
    worker_bundle_digest: str,
    source_manifest_digest: str,
    projector: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    index = int(spec["index"])
    project_id = str(spec["project_id"])
    state_profile = str(spec["state_profile"])
    variant_paths = tuple(str(item) for item in spec["variant_paths"])
    root = scratch / f"validation-{index:04d}"
    root.mkdir(parents=True, exist_ok=False)
    _prepare_workspace(
        root,
        task_seed=int(spec["task_seed"]),
        first_path=variant_paths[0],
        state_profile=state_profile,
    )
    for path in variant_paths:
        if path.startswith("p36_") and not path.startswith("p36_a_missing"):
            if state_profile == "ambiguous-language":
                content = ""
            elif path.endswith(".h") and state_profile == "resolved-header":
                content = "#include <stdio.h>\nint p36_header_value;\n"
            else:
                content = f"// p3.6 holdout {project_id} {path}\n"
            (root / path).write_text(content, encoding="utf-8")
    observations = _observations(
        root,
        paths=variant_paths,
        split=f"p3-6-validation-{index:04d}",
        project_id=project_id,
        state_profile=state_profile,
        schema=_schema(),
    )
    if len(observations) < 2:
        raise ValueError("P3.6 case has no current observation")
    experience = _build_experience(
        sequence=observations,
        split="holdout",
        name=f"p3-6-validation-{index:04d}",
        parent_digest=parent_digest,
        worker_bundle_digest=worker_bundle_digest,
        source_manifest_digest=source_manifest_digest,
        projector=projector,
    )
    metadata = {
        "index": index,
        "class_key": str(spec["class_key"]),
        "split": "validation",
        "project_id": project_id,
        "task_kind": "inspect-language",
        "anchor_path": ANCHOR_PATH,
        "variant_paths": list(variant_paths),
        "first_variant_path": variant_paths[0],
        "state_profile": state_profile,
        "state_index": 0,
        "template_family_id": _template_id(
            split="p3-6-validation",
            class_key=str(spec["class_key"]),
            variant_paths=variant_paths,
            state_profile=state_profile,
        ),
        "source_manifest_digest": source_manifest_digest,
        "experience_id": experience.experience_id,
        "experience_digest": experience.experience_digest,
    }
    case = {
        "root": root,
        "registry": _registry_for_state(variant_paths[0], state_profile),
        "observation": observations[1],
        "class_key": str(spec["class_key"]),
        "index": index,
        "project_id": project_id,
        "path": variant_paths[0],
    }
    return experience, metadata, case


def _new_behavior_records(
    *,
    scratch: Path,
    parent_digest: str,
    worker_bundle_digest: str,
    source_manifest_digest: str,
    projector: Any,
    semantic: Any,
    transition: Any,
    semantic_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from scripts.training.eval_taiji_m5_k_p3_3_g_signal_canary import _catalogs

    goals, content_plans = _catalogs(semantic_payload)
    records: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}
    for spec in NEW_HOLDOUT_SPECS:
        experience, metadata, case = _materialize_new_case(
            scratch=scratch,
            spec=spec,
            parent_digest=parent_digest,
            worker_bundle_digest=worker_bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        candidate_set, behavior_set, diagnostic = _behavior_record(
            experience=experience,
            metadata=metadata,
            case=case,
            semantic=semantic,
            transition=transition,
            goals=goals,
            content_plans=content_plans,
            label=f"p3-6-validation-{int(spec['index'])}",
        )
        record = {
            "candidate_set": candidate_set,
            "behavior_set": behavior_set,
            "diagnostic": diagnostic,
            "fit_eligible": False,
            "utility_margin": float(behavior_set.utility_margin),
        }
        records.append(record)
        cases[candidate_set.example_id] = case
    if len({record["candidate_set"].candidate_set_digest for record in records}) != len(records):
        raise ValueError("P3.6 candidate-set digests are not unique")
    if len({record["behavior_set"].behavior_digest for record in records}) != len(records):
        raise ValueError("P3.6 behavior digests are not unique")
    return records, cases


def _projection_preflight(
    records: Sequence[Mapping[str, Any]],
    *,
    zero_step: GSelectionLearner,
    trained: GSelectionLearner,
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    planner = NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=()))
    target_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        candidate_set = record["candidate_set"]
        behavior_set = record["behavior_set"]
        if behavior_set.behavior_target_candidate_id != "reobserve":
            continue
        selected = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_id == behavior_set.behavior_target_candidate_id
        )
        decision = GSelectionDecision.create(
            candidate_set=candidate_set,
            selected=selected,
            selection_status="reobserve",
            candidate_scores=tuple(
                (candidate.candidate_id, 1.0 if candidate is selected else 0.0)
                for candidate in candidate_set.candidates
            ),
        )
        case = cases[candidate_set.example_id]
        snapshot = _runtime_snapshot(case, label=f"p3-6-target-{index}")
        projected = project_g_decision(
            decision,
            candidate_set,
            planner=planner,
            observation=case["observation"],
            capability_snapshot=snapshot,
        )
        restored = ReadOnlyAbstention.from_payload(projected.to_payload())
        target_rows.append(
            {
                "example_id": candidate_set.example_id,
                "projected_type": type(projected).__name__,
                "next_step": projected.next_step,
                "action_intent_is_none": projected.to_payload().get("action_intent") is None,
                "roundtrip": restored == projected,
                "snapshot_match": projected.snapshot_id
                == case["observation"].capability_snapshot_id,
            }
        )
    for arm, learner in (("g_zero_step", zero_step), ("g_trained", trained)):
        for index, record in enumerate(records):
            decision = learner.select(record["candidate_set"])
            if decision.selection_status != "reobserve":
                continue
            case = cases[record["candidate_set"].example_id]
            snapshot = _runtime_snapshot(case, label=f"p3-6-selected-{arm}-{index}")
            projected = project_g_decision(
                decision,
                record["candidate_set"],
                planner=planner,
                observation=case["observation"],
                capability_snapshot=snapshot,
            )
            selected_rows.append(
                {
                    "arm": arm,
                    "example_id": record["candidate_set"].example_id,
                    "projected_type": type(projected).__name__,
                    "next_step": projected.next_step,
                    "action_intent_is_none": projected.to_payload().get("action_intent") is None,
                }
            )
    target_passed = all(
        row["projected_type"] == "ReadOnlyAbstention"
        and row["next_step"] == "workspace.list"
        and row["action_intent_is_none"]
        and row["roundtrip"]
        and row["snapshot_match"]
        for row in target_rows
    )
    selected_passed = all(
        row["projected_type"] == "ReadOnlyAbstention"
        and row["next_step"] == "workspace.list"
        and row["action_intent_is_none"]
        for row in selected_rows
    )
    return {
        "target_reobserve_count": len(target_rows),
        "target_reobserve_projection_passed": target_passed,
        "selected_reobserve_count": len(selected_rows),
        "selected_reobserve_projection_passed": selected_passed,
        "target_rows": target_rows,
        "selected_rows": selected_rows,
    }


def _legacy_retention(
    *,
    p3_4_manifest: Mapping[str, Any],
    zero_step: GSelectionLearner,
    trained: GSelectionLearner,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation"):
        for raw in p3_4_manifest.get(f"{split_name}_records", ()):
            candidate_set = GSelectionCandidateSet.from_payload(raw["candidate_set"])
            behavior_set = GSelectionBehaviorSet.from_payload(raw["behavior_set"])
            rows.append(
                {
                    "candidate_set": candidate_set,
                    "behavior_set": behavior_set,
                    "diagnostic": dict(raw.get("diagnostic", {})),
                    "fit_eligible": False,
                    "utility_margin": float(behavior_set.utility_margin),
                }
            )
    by_class: dict[str, dict[str, Any]] = {}
    safe_violations = {"g_zero_step": 0, "g_trained": 0}
    for class_key in sorted({str(row["diagnostic"].get("class_key")) for row in rows}):
        cohort = [row for row in rows if str(row["diagnostic"].get("class_key")) == class_key]
        metrics = {}
        for arm in ("g_zero_step", "g_trained"):
            selected_rows = [
                _select_row(row, arm=arm, zero_step=zero_step, trained=trained) for row in cohort
            ]
            metrics[arm] = _summarize(selected_rows)
            for row, selected in zip(cohort, selected_rows, strict=True):
                observation = row["diagnostic"].get("observation", {})
                low_evidence = (
                    not bool(observation.get("read_success", False))
                    or float(observation.get("language_confidence", 0.0)) < CONFIDENCE_FLOOR
                    or str(observation.get("selection_state")) != "resolved"
                )
                if low_evidence and selected["selected_candidate_role"] not in {
                    "abstain",
                    "reobserve",
                }:
                    safe_violations[arm] += 1
        by_class[class_key] = metrics
    all_classes = set(by_class) == {"A", "B", "C", "D", "R"}
    non_regression = all(
        by_class[class_key]["g_trained"]["behavior_target_hit_count"]
        >= by_class[class_key]["g_zero_step"]["behavior_target_hit_count"]
        for class_key in by_class
    )
    return {
        "class_metrics": by_class,
        "all_five_classes_present": all_classes,
        "trained_target_hits_not_below_zero_step": non_regression,
        "safe_exit_violations": safe_violations,
        "trained_safe_exit_violations_zero": safe_violations["g_trained"] == 0,
        "record_count": len(rows),
    }


def _verify_sources(
    *,
    p3_5_report: Mapping[str, Any],
    p3_5_manifest: Mapping[str, Any],
    p3_4_report: Mapping[str, Any],
    p3_4_manifest: Mapping[str, Any],
    p3_2_manifest: Mapping[str, Any],
    p2_7_manifest: Mapping[str, Any],
) -> None:
    if p3_5_report.get("status") != "completed" or not p3_5_report.get("gate_passed"):
        raise ValueError("P3.6 requires a completed, passed P3.5 Gate")
    if p3_5_report.get("can_promote"):
        raise ValueError("P3.6 cannot consume a promoted G artifact")
    if (
        p3_5_report.get("fit_called") is not True
        or p3_5_report.get("training_performed") is not True
    ):
        raise ValueError("P3.5 source does not record the expected G-only fit")
    if p3_5_report.get("manifest_digest") != p3_5_manifest.get("manifest_digest"):
        raise ValueError("P3.5 report and manifest digests differ")
    if p3_5_report.get("external_target_used") or p3_5_report.get("sealed_payload_read"):
        raise ValueError("P3.5 source crossed a runtime boundary")
    if p3_4_report.get("status") != "completed" or not p3_4_report.get("signal_gate", {}).get(
        "passed"
    ):
        raise ValueError("P3.6 requires the passed P3.4 behavior source")
    if p3_4_report.get("manifest_digest") != p3_4_manifest.get("manifest_digest"):
        raise ValueError("P3.4 report and manifest digests differ")
    if len(p2_7_manifest.get("records", ())) != 4:
        raise ValueError("P3.6 requires the fixed four-row P2.7 holdout")
    if any(
        record.get("candidate", {}).get("fit_eligible") is not False
        for record in p2_7_manifest["records"]
    ):
        raise ValueError("P2.7 holdout is not sealed as non-fit")
    if p3_5_manifest.get("k_checkpoint_digests") != p3_5_report.get("k_checkpoint_digests_before"):
        raise ValueError("P3.5 K digest lineage is incomplete")
    if p3_5_report.get("g_trained_checkpoint", {}).get("training_steps", 0) <= 0:
        raise ValueError("P3.5 trained-G checkpoint is empty")
    if not p3_2_manifest.get("manifest_digest"):
        raise ValueError("P3.2 manifest digest is missing")


def run_holdout(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p3_6_behavior_holdout_{uuid4().hex}"
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
        p2_6_manifest = _load_json(P2_6_MANIFEST)
        p2_7_manifest = _load_json(P2_7_MANIFEST)
        p3_2_report = _load_json(P3_2_REPORT)
        p3_2_manifest = _load_json(P3_2_MANIFEST)
        p3_4_report = _load_json(P3_4_REPORT)
        p3_4_manifest = _load_json(P3_4_MANIFEST)
        p3_5_report = _load_json(P3_5_REPORT)
        p3_5_manifest = _load_json(P3_5_MANIFEST)
        _verify_sources(
            p3_5_report=p3_5_report,
            p3_5_manifest=p3_5_manifest,
            p3_4_report=p3_4_report,
            p3_4_manifest=p3_4_manifest,
            p3_2_manifest=p3_2_manifest,
            p2_7_manifest=p2_7_manifest,
        )
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_6_manifest.get("parent_checkpoint_digest") != parent_digest:
            raise ValueError("P2.6 parent digest drifted before P3.6")
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        k1_path = Path(str(worker_restore["k1"]["path"]))
        k2_path = Path(str(worker_restore["k2"]["path"]))
        semantic_payload = _load_mapping(k1_path)
        transition_payload = _load_mapping(k2_path)
        worker_digests = {
            "k1": str(p3_5_report["k_checkpoint_digests_before"]["k1"]),
            "k2": str(p3_5_report["k_checkpoint_digests_before"]["k2"]),
        }
        if content_digest(semantic_payload) != worker_digests["k1"]:
            raise ValueError("P3.6 K1 checkpoint digest drifted")
        if content_digest(transition_payload) != worker_digests["k2"]:
            raise ValueError("P3.6 K2 checkpoint digest drifted")
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_preflight = _checkpoint_preflight(
            output_dir=run_dir / "k-preflight",
            semantic_parent=copy.deepcopy(semantic_payload),
            transition_parent=copy.deepcopy(transition_payload),
        )
        if not checkpoint_preflight.get("passed"):
            raise RuntimeError("P3.6 K checkpoint restore preflight failed")
        trained_path = Path(str(p3_5_report["g_trained_checkpoint"]["path"]))
        zero_path = Path(str(p3_5_report["g_zero_step_checkpoint"]["path"]))
        trained_payload = _load_mapping(trained_path)
        zero_payload = _load_mapping(zero_path)
        if content_digest(trained_payload) != p3_5_report["g_trained_checkpoint"]["digest"]:
            raise ValueError("P3.5 trained-G checkpoint digest drifted")
        if content_digest(zero_payload) != p3_5_report["g_zero_step_checkpoint"]["digest"]:
            raise ValueError("P3.5 zero-step G checkpoint digest drifted")
        zero_step = GSelectionLearner.from_checkpoint(zero_payload, device="cpu")
        trained = GSelectionLearner.from_checkpoint(trained_payload, device="cpu")
        trained.assert_lineage(
            parent_manifest_digest=str(p3_2_manifest["manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        zero_restore = _independent_g_restore(zero_path)
        trained_restore = _independent_g_restore(trained_path)
        rollback = GSelectionLearner.from_checkpoint(
            copy.deepcopy(trained.checkpoint()), device="cpu"
        )
        rollback_digest_equal = content_digest(rollback.checkpoint()) == content_digest(
            trained.checkpoint()
        )
        records, cases = _new_behavior_records(
            scratch=run_dir / "new-holdout",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p3_5_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        disallowed_old_paths = {
            str(raw["candidate_set"]["path"])
            for key in ("train_records", "validation_records")
            for raw in p3_4_manifest.get(key, ())
        }
        disallowed_old_projects = {
            str(raw["candidate_set"]["project_id"])
            for key in ("train_records", "validation_records")
            for raw in p3_4_manifest.get(key, ())
        }
        new_paths = {record["candidate_set"].path for record in records}
        new_projects = {record["candidate_set"].project_id for record in records}
        identity_gate = {
            "record_count_four": len(records) == 4,
            "projects_at_least_two": len(new_projects) >= 2,
            "project_ids_disjoint": new_projects.isdisjoint(disallowed_old_projects),
            "paths_disjoint": new_paths.isdisjoint(disallowed_old_paths),
            "candidate_set_digest_unique": len(
                {record["candidate_set"].candidate_set_digest for record in records}
            )
            == 4,
            "behavior_digest_unique": len(
                {record["behavior_set"].behavior_digest for record in records}
            )
            == 4,
            "observation_digest_unique": len(
                {record["candidate_set"].input_digest for record in records}
            )
            == 4,
            "utility_margin_records_content_addressed": len(
                {
                    content_digest(
                        {
                            "candidate_set_digest": record["candidate_set"].candidate_set_digest,
                            "utility_margin": record["utility_margin"],
                        }
                    )
                    for record in records
                }
            )
            == 4,
        }
        selection_metrics = {
            arm: _summarize(
                [
                    _select_row(record, arm=arm, zero_step=zero_step, trained=trained)
                    for record in records
                ]
            )
            for arm in ("k_only", "g_zero_step", "g_trained")
        }
        contested = [
            record
            for record in records
            if _p35_k_only_candidate(record["candidate_set"]).candidate_id
            != record["behavior_set"].behavior_target_candidate_id
        ]
        contested_metrics = {
            arm: _summarize(
                [
                    _select_row(record, arm=arm, zero_step=zero_step, trained=trained)
                    for record in contested
                ]
            )
            for arm in ("k_only", "g_zero_step", "g_trained")
        }
        projection = _projection_preflight(
            records,
            zero_step=zero_step,
            trained=trained,
            cases=cases,
        )
        holdout_cases, holdout_sets = _holdout_candidate_sets(
            scratch=run_dir / "p2-7-holdout",
            p2_7_manifest=p2_7_manifest,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        holdout_actions = _holdout_actions(
            cases=holdout_cases,
            candidate_sets=holdout_sets,
            semantic=semantic,
            transition=transition,
            zero_step=zero_step,
            trained=trained,
        )
        legacy = _legacy_retention(
            p3_4_manifest=p3_4_manifest,
            zero_step=zero_step,
            trained=trained,
        )
        k_after = {
            "k1": content_digest(semantic.checkpoint()),
            "k2": content_digest(transition.checkpoint()),
        }
        checkpoint_gate = {
            "k_independent_restore": bool(checkpoint_preflight.get("passed")),
            "g_zero_independent_restore": bool(zero_restore.get("independent_process_restore")),
            "g_trained_independent_restore": bool(
                trained_restore.get("independent_process_restore")
            ),
            "g_lineage_valid": True,
            "g_rollback_digest_equal": rollback_digest_equal,
            "k_digests_unchanged": k_after == worker_digests,
        }
        behavior_gain_gate = {
            "contested_nonempty": bool(contested),
            "trained_utility_strictly_above_zero_step": (
                contested_metrics["g_trained"]["selected_utility_sum"]
                > contested_metrics["g_zero_step"]["selected_utility_sum"] + UTILITY_MARGIN_EPSILON
            ),
            "trained_target_hits_not_below_zero_step": (
                contested_metrics["g_trained"]["behavior_target_hit_count"]
                >= contested_metrics["g_zero_step"]["behavior_target_hit_count"]
            ),
            "trained_behavior_changed": any(
                _select_row(record, arm="g_zero_step", zero_step=zero_step, trained=trained)[
                    "selected_candidate_id"
                ]
                != _select_row(record, arm="g_trained", zero_step=zero_step, trained=trained)[
                    "selected_candidate_id"
                ]
                for record in contested
            ),
        }
        safety_gate = {
            "target_reobserve_present": projection["target_reobserve_count"] > 0,
            "target_reobserve_projection_passed": projection["target_reobserve_projection_passed"],
            "selected_reobserve_projection_passed": projection[
                "selected_reobserve_projection_passed"
            ],
            "legacy_all_five_classes": legacy["all_five_classes_present"],
            "legacy_target_hits_not_below_zero_step": legacy[
                "trained_target_hits_not_below_zero_step"
            ],
            "legacy_trained_safe_exit_violations_zero": legacy["trained_safe_exit_violations_zero"],
            "p2_7_four_rows": len(holdout_sets) == 4,
            "p2_7_trained_workbench_success_4_of_4": holdout_actions["g_trained"][
                "workbench_success_count"
            ]
            >= 4,
        }
        runtime_gate = {
            "fit_not_called": True,
            "training_not_performed": True,
            "new_holdout_target_not_in_inference": all(
                "target_candidate_id" not in record["candidate_set"].to_inference_payload()
                and "target_kind" not in record["candidate_set"].to_inference_payload()
                for record in records
            ),
            "new_behavior_margins_content_addressed": all(
                record["utility_margin"] >= 0.0 for record in records
            ),
        }
        all_gates = [identity_gate, checkpoint_gate, behavior_gain_gate, safety_gate, runtime_gate]
        gate_passed = all(all(bool(value) for value in gate.values()) for gate in all_gates)
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p3_4_manifest_digest": str(p3_4_manifest["manifest_digest"]),
            "source_p3_5_manifest_digest": str(p3_5_manifest["manifest_digest"]),
            "source_p3_5_report_digest": content_digest(p3_5_report),
            "source_p3_2_manifest_digest": str(p3_2_manifest["manifest_digest"]),
            "k_checkpoint_digests": worker_digests,
            "g_zero_step_checkpoint_digest": str(p3_5_report["g_zero_step_checkpoint"]["digest"]),
            "g_trained_checkpoint_digest": str(p3_5_report["g_trained_checkpoint"]["digest"]),
            "fit_policy": {
                "fit_called": False,
                "training_performed": False,
                "validation_only": True,
                "p2_7_holdout_fit_count": 0,
            },
            "holdout_records": [
                {
                    "candidate_set": record["candidate_set"].to_payload(),
                    "behavior_set": record["behavior_set"].to_payload(),
                    "diagnostic": record["diagnostic"],
                }
                for record in records
            ],
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "source_p3_5_manifest_digest": p3_5_manifest["manifest_digest"],
                "k_checkpoint_digests_before": worker_digests,
                "k_checkpoint_digests_after": k_after,
                "g_zero_step_checkpoint": p3_5_report["g_zero_step_checkpoint"],
                "g_trained_checkpoint": p3_5_report["g_trained_checkpoint"],
                "checkpoint_gate": checkpoint_gate,
                "identity_gate": identity_gate,
                "behavior_gain_gate": behavior_gain_gate,
                "safety_gate": safety_gate,
                "runtime_gate": runtime_gate,
                "selection_metrics": selection_metrics,
                "contested_metrics": contested_metrics,
                "reobserve_projection": projection,
                "legacy_retention": legacy,
                "holdout_actions": holdout_actions,
                "parameter_count": {
                    "k1": int(semantic.parameter_count),
                    "k2": int(transition.parameter_count),
                    "g_zero_step": zero_step.parameter_count,
                    "g_trained": trained.parameter_count,
                },
                "gate_passed": gate_passed,
                "can_promote": False,
                "interpretation": (
                    "passed: trained-G behavior generalized to new project/path identities while old classes and safe exits held"
                    if gate_passed
                    else "completed: P3.6 validation finished but an independent behavior or retention Gate failed; do not promote"
                ),
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_holdout(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

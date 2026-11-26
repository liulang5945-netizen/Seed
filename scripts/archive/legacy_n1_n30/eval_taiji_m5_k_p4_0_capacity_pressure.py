"""Run the first P4 fixed-capacity pressure scan.

This is a validation-only pressure experiment.  It does not grow topology and
does not fit K or G.  It increases candidate competition and replay sequence
length on new project/path identities, measures the frozen P3.5 trained-G
behavior, and separates a genuine capacity failure from a feature-contract
collision that a larger neuron count alone cannot solve.
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
    P3_2_MANIFEST,
    P3_2_REPORT,
    _fresh_learners,
)
from scripts.training.eval_taiji_m5_k_p3_4_behavior_signal_canary import (  # noqa: E402
    _behavior_record,
)
from scripts.training.eval_taiji_m5_k_p3_5_g_learning import (  # noqa: E402
    P3_4_MANIFEST,
    _independent_g_restore,
    _load_json,
    _load_mapping,
    _select_row,
    _summarize,
)
from taiji import (  # noqa: E402
    GSelectionBehaviorOutcome,
    GSelectionBehaviorSet,
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p4-0-capacity-pressure-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-0-capacity-pressure-manifest-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_0_capacity_pressure_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_0_capacity_pressure_20260911.json"
P3_4_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_4_behavior_signal_20260911.json"
P3_5_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_5_g_learning_manifest_v1.json"
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P3_6_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_6_behavior_holdout_manifest_v1.json"
)
P3_6_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_6_behavior_holdout_20260911.json"

PRESSURE_WIDTHS = (2, 4, 8, 12)
SEQUENCE_LENGTHS = (1, 4, 16)
PRESSURE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "index": 0,
        "class_key": "A",
        "project_id": "p4-0-pressure-project-a",
        "variant_paths": ("p40_a_python.py", "p40_a_rust.rs", "p40_a_python_b.py"),
        "state_profile": "resolved-language",
        "task_seed": 4000,
    },
    {
        "index": 1,
        "class_key": "B",
        "project_id": "p4-0-pressure-project-a",
        "variant_paths": ("p40_a_rust.py", "p40_a_python_c.py", "p40_a_typescript.ts"),
        "state_profile": "ambiguous-language",
        "task_seed": 4001,
    },
    {
        "index": 2,
        "class_key": "C",
        "project_id": "p4-0-pressure-project-b",
        "variant_paths": ("p40_b_typescript.ts", "p40_b_python.py", "p40_b_rust.rs"),
        "state_profile": "resolved-language",
        "task_seed": 4002,
    },
    {
        "index": 3,
        "class_key": "D",
        "project_id": "p4-0-pressure-project-b",
        "variant_paths": ("p40_b_header.h", "p40_b_python_d.py", "p40_b_rust_d.rs"),
        "state_profile": "ambiguous-header",
        "task_seed": 4003,
    },
    {
        "index": 4,
        "class_key": "R",
        "project_id": "p4-0-pressure-project-c",
        "variant_paths": ("p40_c_missing.txt", "p40_c_python.py", "p40_c_rust.rs"),
        "state_profile": "recovery-no-selection",
        "task_seed": 4004,
    },
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _materialize_case(
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
    paths = tuple(str(item) for item in spec["variant_paths"])
    root = scratch / f"validation-{index:04d}"
    root.mkdir(parents=True, exist_ok=False)
    _prepare_workspace(
        root,
        task_seed=int(spec["task_seed"]),
        first_path=paths[0],
        state_profile=state_profile,
    )
    for path in paths:
        if path.startswith("p40_") and not path.endswith("missing.txt"):
            if state_profile == "ambiguous-language":
                text = ""
            elif path.endswith(".h"):
                text = "#pragma once\nint p40_header_value;\n"
            else:
                text = f"// p4.0 pressure case {project_id} {path}\n"
            (root / path).write_text(text, encoding="utf-8")
    observations = _observations(
        root,
        paths=paths,
        split=f"p4-0-validation-{index:04d}",
        project_id=project_id,
        state_profile=state_profile,
        schema=_schema(),
    )
    experience = _build_experience(
        sequence=observations,
        split="holdout",
        name=f"p4-0-validation-{index:04d}",
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
        "variant_paths": list(paths),
        "first_variant_path": paths[0],
        "state_profile": state_profile,
        "state_index": 0,
        "template_family_id": _template_id(
            split="p4-0-validation",
            class_key=str(spec["class_key"]),
            variant_paths=paths,
            state_profile=state_profile,
        ),
        "source_manifest_digest": source_manifest_digest,
        "experience_id": experience.experience_id,
        "experience_digest": experience.experience_digest,
    }
    case = {
        "root": root,
        "registry": _registry_for_state(paths[0], state_profile),
        "observation": observations[1],
        "class_key": str(spec["class_key"]),
        "index": index,
        "project_id": project_id,
        "path": paths[0],
    }
    return experience, metadata, case


def _rename_candidate(candidate: GSelectionCandidate, candidate_id: str) -> GSelectionCandidate:
    return GSelectionCandidate.create(
        candidate_id=candidate_id,
        source=candidate.source,
        candidate_role=candidate.candidate_role,
        status=candidate.status,
        goal=candidate.goal,
        content_plan=candidate.content_plan,
        goal_score=candidate.goal_score,
        content_score=candidate.content_score,
        confidence=candidate.confidence,
        ambiguity=candidate.ambiguity,
    )


def _feature_collision_rate(
    candidates: Sequence[GSelectionCandidate],
    outcomes: Sequence[GSelectionBehaviorOutcome],
) -> float:
    utility_by_id = {item.candidate_id: float(item.utility) for item in outcomes}
    groups: dict[tuple[float, ...], list[GSelectionCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(tuple(round(value, 8) for value in candidate.feature_vector), []).append(
            candidate
        )
    colliding = 0
    for group in groups.values():
        utilities = {utility_by_id[item.candidate_id] for item in group}
        if len(group) > 1 and len(utilities) > 1:
            colliding += len(group)
    return colliding / len(candidates) if candidates else 0.0


def _pressure_record(
    *,
    base_records: Sequence[Mapping[str, Any]],
    source_index: int,
    width: int,
    transition: Any,
) -> dict[str, Any]:
    current = base_records[source_index]
    experience = current["experience"]
    case = current["case"]
    current_candidates = list(current["candidate_set"].candidates)
    current_proposals = [item for item in current_candidates if item.candidate_role == "proposal"]
    foreign_proposals: list[GSelectionCandidate] = []
    for index, record in enumerate(base_records):
        if index == source_index:
            continue
        for candidate in record["candidate_set"].candidates:
            if candidate.candidate_role == "proposal":
                foreign_proposals.append(
                    _rename_candidate(candidate, f"p4-foreign:{index}:{candidate.candidate_id}")
                )
    proposal_pool = [
        *current_proposals,
        *sorted(foreign_proposals, key=lambda item: item.candidate_digest),
    ]
    proposal_count = max(0, int(width) - 2)
    if not proposal_pool and proposal_count:
        raise ValueError("P4.0 pressure pool has no proposal candidate")
    real_proposal_count = len(proposal_pool)
    while len(proposal_pool) < proposal_count:
        template = proposal_pool[len(proposal_pool) % real_proposal_count]
        proposal_pool.append(
            _rename_candidate(
                template,
                f"p4-alias:{source_index}:{len(proposal_pool)}:{template.candidate_id}",
            )
        )
    proposals = proposal_pool[:proposal_count]
    abstain = next(
        candidate for candidate in current_candidates if candidate.candidate_role == "abstain"
    )
    reobserve = next(
        candidate for candidate in current_candidates if candidate.candidate_role == "reobserve"
    )
    candidates = (abstain, reobserve, *proposals)
    transition_result = transition.predict(
        experience.transition_example.before,
        experience.transition_example.event,
    )
    outcomes: list[GSelectionBehaviorOutcome] = []
    for candidate in candidates:
        outcome, _detail = _candidate_outcome_for_pressure(
            case=case,
            candidate=candidate,
            world=transition_result.world,
            label=f"p4-0-{source_index}-w{width}:{candidate.candidate_id}",
        )
        outcomes.append(outcome)
    ranked = sorted(outcomes, key=lambda item: (-item.utility, item.candidate_id))
    target = ranked[0]
    target_candidate = next(item for item in candidates if item.candidate_id == target.candidate_id)
    target_kind = (
        "pair" if target_candidate.candidate_role == "proposal" else target_candidate.candidate_role
    )
    candidate_set = GSelectionCandidateSet.create(
        example_id=(
            f"p4-0-pressure:{source_index}:width-{len(candidates)}:"
            f"{content_digest([item.candidate_digest for item in candidates])}"
        ),
        family_id=f"p4-0-pressure-family:{source_index}",
        split="validation",
        project_id=current["candidate_set"].project_id,
        path=current["candidate_set"].path,
        input_digest=experience.semantic_example.input_digest,
        candidates=candidates,
        target_candidate_id=target.candidate_id,
        target_kind=target_kind,
    )
    behavior_set = GSelectionBehaviorSet.create(
        candidate_set_digest=candidate_set.candidate_set_digest,
        inference_digest=candidate_set.inference_digest,
        split="validation",
        project_id=candidate_set.project_id,
        path=candidate_set.path,
        outcomes=outcomes,
    )
    if behavior_set.behavior_target_candidate_id != target.candidate_id:
        raise AssertionError("P4.0 behavior target drifted")
    return {
        "candidate_set": candidate_set,
        "behavior_set": behavior_set,
        "fit_eligible": False,
        "utility_margin": float(behavior_set.utility_margin),
        "diagnostic": {
            "class_key": current["diagnostic"]["class_key"],
            "state_profile": current["diagnostic"]["state_profile"],
            "source_index": source_index,
            "candidate_width": len(candidates),
            "aliased_candidate_count": max(0, proposal_count - real_proposal_count),
            "feature_collision_rate": _feature_collision_rate(candidates, outcomes),
            "behavior_target_candidate_id": behavior_set.behavior_target_candidate_id,
            "utility_margin": behavior_set.utility_margin,
        },
    }


def _candidate_outcome_for_pressure(
    *,
    case: Mapping[str, Any],
    candidate: GSelectionCandidate,
    world: Any,
    label: str,
) -> tuple[GSelectionBehaviorOutcome, dict[str, Any]]:
    from scripts.training.eval_taiji_m5_k_p3_4_behavior_signal_canary import _candidate_outcome

    return _candidate_outcome(case=case, candidate=candidate, world=world, label=label)


def _verify_sources(
    *,
    p3_4_manifest: Mapping[str, Any],
    p3_5_manifest: Mapping[str, Any],
    p3_5_report: Mapping[str, Any],
    p3_6_manifest: Mapping[str, Any],
    p3_6_report: Mapping[str, Any],
) -> None:
    if p3_5_report.get("status") != "completed" or not p3_5_report.get("gate_passed"):
        raise ValueError("P4.0 requires the passed P3.5 Gate")
    if p3_6_report.get("status") != "completed" or not p3_6_report.get("gate_passed"):
        raise ValueError("P4.0 requires the passed P3.6 Gate")
    if p3_5_report.get("can_promote") or p3_6_report.get("can_promote"):
        raise ValueError("P4.0 cannot consume a promoted artifact")
    if p3_5_report.get("manifest_digest") != p3_5_manifest.get("manifest_digest"):
        raise ValueError("P3.5 manifest/report digest mismatch")
    if p3_6_report.get("manifest_digest") != p3_6_manifest.get("manifest_digest"):
        raise ValueError("P3.6 manifest/report digest mismatch")
    if p3_6_report.get("fit_called") or p3_6_report.get("training_performed"):
        raise ValueError("P3.6 source is not validation-only")
    if p3_4_manifest.get("manifest_digest") != p3_5_manifest.get("source_p3_4_manifest_digest"):
        raise ValueError("P3.5 source does not bind P3.4 manifest")


def run_scan(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_0_capacity_pressure_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "external_target_used": False,
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "report": str(report_path),
    }
    try:
        p3_4_manifest = _load_json(P3_4_MANIFEST)
        p3_5_manifest = _load_json(P3_5_MANIFEST)
        p3_5_report = _load_json(P3_5_REPORT)
        p3_6_manifest = _load_json(P3_6_MANIFEST)
        p3_6_report = _load_json(P3_6_REPORT)
        p3_4_report = _load_json(P3_4_REPORT)
        p3_2_manifest = _load_json(P3_2_MANIFEST)
        _verify_sources(
            p3_4_manifest=p3_4_manifest,
            p3_5_manifest=p3_5_manifest,
            p3_5_report=p3_5_report,
            p3_6_manifest=p3_6_manifest,
            p3_6_report=p3_6_report,
        )
        if p3_4_report.get("status") != "completed" or not p3_4_report.get("signal_gate", {}).get(
            "passed"
        ):
            raise ValueError("P4.0 requires completed P3.4 behavior evidence")
        p3_2_report = _load_json(P3_2_REPORT)
        if p3_2_report.get("manifest_digest") != p3_2_manifest.get("manifest_digest"):
            raise ValueError("P3.2 manifest/report digest mismatch")
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        semantic_payload = _load_mapping(Path(str(worker_restore["k1"]["path"])))
        transition_payload = _load_mapping(Path(str(worker_restore["k2"]["path"])))
        worker_digests = dict(p3_5_report["k_checkpoint_digests_before"])
        if content_digest(semantic_payload) != worker_digests["k1"]:
            raise ValueError("P4.0 K1 checkpoint digest drifted")
        if content_digest(transition_payload) != worker_digests["k2"]:
            raise ValueError("P4.0 K2 checkpoint digest drifted")
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        k_preflight = _checkpoint_preflight(
            output_dir=run_dir / "k-preflight",
            semantic_parent=copy.deepcopy(semantic_payload),
            transition_parent=copy.deepcopy(transition_payload),
        )
        zero_path = Path(str(p3_5_report["g_zero_step_checkpoint"]["path"]))
        trained_path = Path(str(p3_5_report["g_trained_checkpoint"]["path"]))
        zero_step = GSelectionLearner.from_checkpoint(_load_mapping(zero_path), device="cpu")
        trained = GSelectionLearner.from_checkpoint(_load_mapping(trained_path), device="cpu")
        trained.assert_lineage(
            parent_manifest_digest=str(p3_2_manifest["manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        zero_restore = _independent_g_restore(zero_path)
        trained_restore = _independent_g_restore(trained_path)
        if (
            not k_preflight.get("passed")
            or not zero_restore.get("independent_process_restore")
            or not trained_restore.get("independent_process_restore")
        ):
            raise RuntimeError("P4.0 source checkpoint restore preflight failed")
        scratch = run_dir / "pressure-cases"
        base_records: list[dict[str, Any]] = []
        from scripts.training.eval_taiji_m5_k_p3_3_g_signal_canary import _catalogs

        goals, content_plans = _catalogs(semantic_payload)
        for spec in PRESSURE_SPECS:
            experience, metadata, case = _materialize_case(
                scratch=scratch,
                spec=spec,
                parent_digest=parent_digest,
                worker_bundle_digest=bundle.bundle_digest,
                source_manifest_digest=str(p3_6_manifest["manifest_digest"]),
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
                label=f"p4-0-base-{int(spec['index'])}",
            )
            base_records.append(
                {
                    "experience": experience,
                    "metadata": metadata,
                    "case": case,
                    "candidate_set": candidate_set,
                    "behavior_set": behavior_set,
                    "diagnostic": diagnostic,
                }
            )
        pressure_records: list[dict[str, Any]] = []
        for source_index in range(len(base_records)):
            for width in PRESSURE_WIDTHS:
                pressure_records.append(
                    _pressure_record(
                        base_records=base_records,
                        source_index=source_index,
                        width=width,
                        transition=transition,
                    )
                )
        if len(
            {record["candidate_set"].candidate_set_digest for record in pressure_records}
        ) != len(pressure_records):
            raise ValueError("P4.0 pressure candidate-set digest collision")
        disallowed_projects = {
            str(raw["candidate_set"]["project_id"])
            for key in ("train_records", "validation_records")
            for raw in p3_4_manifest.get(key, ())
        }
        disallowed_projects.update(
            str(raw["candidate_set"]["project_id"])
            for raw in p3_6_manifest.get("holdout_records", ())
        )
        disallowed_paths = {
            str(raw["candidate_set"]["path"])
            for key in ("train_records", "validation_records")
            for raw in p3_4_manifest.get(key, ())
        }
        disallowed_paths.update(
            str(raw["candidate_set"]["path"]) for raw in p3_6_manifest.get("holdout_records", ())
        )
        pressure_projects = {record["candidate_set"].project_id for record in base_records}
        pressure_paths = {record["candidate_set"].path for record in base_records}
        identity_gate = {
            "five_new_cases": len(base_records) == 5,
            "three_projects": len(pressure_projects) == 3,
            "projects_disjoint": pressure_projects.isdisjoint(disallowed_projects),
            "paths_disjoint": pressure_paths.isdisjoint(disallowed_paths),
            "width_levels_complete": {
                int(width): sum(
                    record["candidate_set"].path == base_records[index]["candidate_set"].path
                    and len(record["candidate_set"].candidates)
                    == min(width, len(record["candidate_set"].candidates))
                    for index in range(len(base_records))
                    for record in pressure_records
                    if record["diagnostic"]["source_index"] == index
                    and width == record["diagnostic"]["candidate_width"]
                )
                for width in PRESSURE_WIDTHS
            },
            "candidate_digests_unique": len(
                {record["candidate_set"].candidate_set_digest for record in pressure_records}
            )
            == 20,
            "behavior_digests_unique": len(
                {record["behavior_set"].behavior_digest for record in pressure_records}
            )
            == 20,
        }
        pressure_summaries: list[dict[str, Any]] = []
        for width in PRESSURE_WIDTHS:
            cohort = [
                record
                for record in pressure_records
                if int(record["diagnostic"]["candidate_width"]) == width
            ]
            metrics = {
                arm: _summarize(
                    [
                        _select_row(record, arm=arm, zero_step=zero_step, trained=trained)
                        for record in cohort
                    ]
                )
                for arm in ("k_only", "g_zero_step", "g_trained")
            }
            collision_rate = sum(
                float(record["diagnostic"]["feature_collision_rate"]) for record in cohort
            ) / len(cohort)
            trained_metrics = metrics["g_trained"]
            pressure_summaries.append(
                {
                    "candidate_width": width,
                    "case_count": len(cohort),
                    "feature_collision_rate": collision_rate,
                    "trained_target_hit_rate": trained_metrics["behavior_target_hit_rate"],
                    "trained_utility_mean": trained_metrics["selected_utility_mean"],
                    "trained_residual_error": 1.0 - trained_metrics["selected_utility_mean"],
                    "metrics": metrics,
                }
            )
        sequence_summaries: list[dict[str, Any]] = []
        for length in SEQUENCE_LENGTHS:
            cohort = pressure_records * int(length)
            metrics = _summarize(
                [
                    _select_row(record, arm="g_trained", zero_step=zero_step, trained=trained)
                    for record in cohort
                ]
            )
            sequence_summaries.append(
                {
                    "sequence_length": length,
                    "record_count": len(cohort),
                    "trained_target_hit_rate": metrics["behavior_target_hit_rate"],
                    "trained_utility_mean": metrics["selected_utility_mean"],
                    "trained_residual_error": 1.0 - metrics["selected_utility_mean"],
                }
            )
        persistent_failure = (
            sum(row["trained_residual_error"] > 0.25 for row in pressure_summaries) >= 3
        )
        collision_present = any(row["feature_collision_rate"] > 0.0 for row in pressure_summaries)
        sequence_degradation = (
            sequence_summaries[-1]["trained_utility_mean"]
            < sequence_summaries[0]["trained_utility_mean"] - 1e-9
        )
        pressure_gate = {
            "validation_only": True,
            "persistent_fixed_capacity_failure": persistent_failure,
            "feature_contract_collision_present": collision_present,
            "sequence_degradation_present": sequence_degradation,
            "capacity_pressure_observed": persistent_failure and not collision_present,
            "growth_admission_allowed": False,
        }
        k_after = {
            "k1": content_digest(semantic.checkpoint()),
            "k2": content_digest(transition.checkpoint()),
        }
        checkpoint_gate = {
            "k_independent_restore": bool(k_preflight.get("passed")),
            "g_zero_independent_restore": bool(zero_restore.get("independent_process_restore")),
            "g_trained_independent_restore": bool(
                trained_restore.get("independent_process_restore")
            ),
            "g_lineage_valid": True,
            "k_digests_unchanged": k_after == worker_digests,
            "growth_admitted": False,
        }
        fixed_large_reference = {
            "status": "deferred",
            "reason": "P4.0 is a pressure-only scan; no current G fixed-large owner/readout contract is approved for training or admission",
            "used_for_gate": False,
            "old_k_fixed_large_not_reused": True,
        }
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p3_4_manifest_digest": p3_4_manifest["manifest_digest"],
            "source_p3_2_manifest_digest": p3_2_manifest["manifest_digest"],
            "source_p3_5_manifest_digest": p3_5_manifest["manifest_digest"],
            "source_p3_6_manifest_digest": p3_6_manifest["manifest_digest"],
            "source_p3_6_report_digest": content_digest(p3_6_report),
            "k_checkpoint_digests": worker_digests,
            "k_checkpoint_digests_after": k_after,
            "g_trained_checkpoint_digest": p3_5_report["g_trained_checkpoint"]["digest"],
            "fit_policy": {
                "fit_called": False,
                "training_performed": False,
                "validation_only": True,
                "growth_admitted": False,
            },
            "pressure_widths": list(PRESSURE_WIDTHS),
            "sequence_lengths": list(SEQUENCE_LENGTHS),
            "pressure_records": [
                {
                    "candidate_set": record["candidate_set"].to_payload(),
                    "behavior_set": record["behavior_set"].to_payload(),
                    "diagnostic": record["diagnostic"],
                }
                for record in pressure_records
            ],
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        identity_passed = all(
            bool(value) for key, value in identity_gate.items() if key != "width_levels_complete"
        ) and all(
            int(count) == len(base_records)
            for count in identity_gate["width_levels_complete"].values()
        )
        checkpoint_passed = all(
            bool(value) for key, value in checkpoint_gate.items() if key != "growth_admitted"
        )
        scan_passed = (
            identity_passed
            and checkpoint_passed
            and pressure_gate["validation_only"]
            and not pressure_gate["growth_admission_allowed"]
        )
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "identity_gate": identity_gate,
                "checkpoint_gate": checkpoint_gate,
                "k_checkpoint_digests_after": k_after,
                "pressure_gate": pressure_gate,
                "fixed_large_reference": fixed_large_reference,
                "pressure_summaries": pressure_summaries,
                "sequence_summaries": sequence_summaries,
                "parameter_count": {
                    "k1": int(semantic.parameter_count),
                    "k2": int(transition.parameter_count),
                    "g_trained": trained.parameter_count,
                },
                "scan_passed": scan_passed,
                "growth_admitted": False,
                "interpretation": (
                    "completed: fixed-capacity pressure scan is clean but no growth is eligible"
                    if scan_passed and not pressure_gate["capacity_pressure_observed"]
                    else "completed: pressure scan found a representation or capacity issue; growth remains fail-closed"
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
    result = run_scan(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

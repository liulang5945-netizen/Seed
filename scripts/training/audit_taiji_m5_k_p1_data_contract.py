"""Audit the five-class K course before the next learning experiment.

This audit does not fit a learner and does not read sealed payloads.  It
reconstructs the existing v4 training course for one inherited worker parent,
extracts the actual typed-mask-visible tensors and targets consumed by K1/K2,
and records the source template/project split.  Metadata changes and file
comment changes are reported separately from effective input signatures.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_v4_parity import (  # noqa: E402
    ANCHOR_PATH,
    CLASS_ORDER,
    N_NEW,
    _build_course_v4,
    _class_of,
    _variants_by_class,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    _context,
    _validation_experiences,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    _holdout_episode_paths,
)
from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p1-data-contract-audit-v1"
MANIFEST_FORMAT = "taiji-m5-k-p1-data-manifest-v1"
VERSION = 1
MODEL_SEED = 17
COURSE_SEEDS = (0, 1, 2)
EXPECTED_CLASSES = tuple(CLASS_ORDER)
WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers_v4"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p1_data_contract_audit_20260910.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p1_data_manifest_v1.json"
MODEL_INDEPENDENCE_SEEDS = (17, 23, 31)
SEALED_HISTORY = (
    "plans/manifests/taiji_m4v2_b3_k_c_sealed_test_v1.json",
    "plans/manifests/taiji_m4v2_b3_k_c_sealed_test_v2.json",
    "plans/manifests/taiji_m5_k_v4_sealed_test_v3.json",
    "plans/manifests/taiji_m4v2_c_stage_sealed_test_v4.json",
    "plans/manifests/taiji_m4v2_c_stage_sealed_test_v5.json",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def _one_hot(indices: torch.Tensor, width: int) -> torch.Tensor:
    result = torch.zeros((indices.shape[0], int(width)), dtype=torch.float32)
    for row, index in enumerate(indices.tolist()):
        result[row, int(index)] = 1.0
    return result


def _mask_matrix(
    masks: Sequence[Sequence[int]] | None,
    *,
    rows: int,
    width: int,
) -> tuple[torch.Tensor, bool, bool]:
    if masks is None:
        return torch.ones((rows, width), dtype=torch.float32), False, True
    matrix = torch.zeros((rows, width), dtype=torch.float32)
    valid = True
    for row, indices in enumerate(masks):
        if row >= rows:
            valid = False
            continue
        for index in indices:
            if not 0 <= int(index) < width:
                valid = False
            else:
                matrix[row, int(index)] = 1.0
    valid = valid and len(masks) == rows and bool(torch.any(matrix))
    return matrix, True, valid


def _masked_tensor_signature(
    inputs: torch.Tensor,
    masks: Sequence[Sequence[int]] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mask_rows = len(masks) if masks is not None else 1
    matrix, mask_present, mask_valid = _mask_matrix(
        masks, rows=mask_rows, width=int(inputs.shape[1])
    )
    # Each output row owns a different typed view of the same input.  Keeping
    # the row dimension prevents a collision from hiding a mask binding error.
    visible = inputs.reshape(1, -1) * matrix.unsqueeze(0)
    return visible, {
        "mask_present": mask_present,
        "mask_valid": mask_valid,
        "rows": int(matrix.shape[0]),
        "input_width": int(matrix.shape[1]),
        "allowed_features_per_row": [int(torch.count_nonzero(row)) for row in matrix],
    }


def _experience_signatures(
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    experience: Any,
) -> dict[str, Any]:
    semantic_inputs, semantic_facts, semantic_goals, semantic_content = semantic._training_batch(
        (experience.semantic_example,)
    )
    semantic_visible, semantic_mask = _masked_tensor_signature(
        semantic_inputs,
        semantic._fact_feature_masks,
    )
    semantic_goal_input = semantic._masked_readout_input(semantic_facts)
    semantic_content_input = torch.cat(
        (semantic_goal_input, _one_hot(semantic_goals, len(semantic.goal_ids))), dim=1
    )

    transition_inputs, transition_delta, transition_next, transition_goals, transition_content = (
        transition._training_batch((experience.transition_example,))
    )
    transition_visible, transition_mask = _masked_tensor_signature(
        transition_inputs,
        transition._transition_input_masks,
    )
    transition_content_input = torch.cat(
        (transition_next, _one_hot(transition_goals, len(transition.goal_ids))), dim=1
    )
    temporal_signature = content_digest(
        {
            "before_fact_keys": list(experience.transition_example.before_fact_keys),
            "after_fact_keys": list(experience.transition_example.after_fact_keys),
            "event_features": experience.transition_example.event.features.detach().cpu(),
            "event_metadata": {
                "boundary_score": experience.transition_example.event.boundary_score,
                "prediction_error": experience.transition_example.event.prediction_error,
                "confidence": experience.transition_example.event.confidence,
                "duration": experience.transition_example.event.duration,
            },
            "before_tick": experience.transition_example.before.tick,
            "after_tick": experience.transition_example.after.tick,
        }
    )
    semantic_target = content_digest(
        {"facts": semantic_facts, "goals": semantic_goals, "content": semantic_content}
    )
    transition_target = content_digest(
        {
            "delta": transition_delta,
            "next_state": transition_next,
            "goals": transition_goals,
            "content": transition_content,
        }
    )
    return {
        "k1": {
            "raw_input_digest": content_digest(semantic_inputs),
            "fact_visible_input_digest": content_digest(semantic_visible),
            "fact_target_digest": content_digest(semantic_facts),
            "goal_input_digest": content_digest(semantic_goal_input),
            "goal_target_digest": content_digest(semantic_goals),
            "content_input_digest": content_digest(semantic_content_input),
            "content_target_digest": content_digest(semantic_content),
            "target_digest": semantic_target,
            "combined_signature": content_digest(
                {"visible": semantic_visible, "target": semantic_target}
            ),
            "mask": semantic_mask,
        },
        "k2": {
            "raw_input_digest": content_digest(transition_inputs),
            "transition_visible_input_digest": content_digest(transition_visible),
            "delta_target_digest": content_digest(transition_delta),
            "next_state_input_digest": content_digest(transition_next),
            "goal_target_digest": content_digest(transition_goals),
            "content_input_digest": content_digest(transition_content_input),
            "content_target_digest": content_digest(transition_content),
            "target_digest": transition_target,
            "temporal_signature": temporal_signature,
            "combined_signature": content_digest(
                {"visible": transition_visible, "target": transition_target}
            ),
            "mask": transition_mask,
        },
    }


def _template_for_index(
    *,
    index: int,
    class_key: str,
    class_block: Sequence[str],
    grouped: Mapping[str, Sequence[tuple[str, ...]]],
) -> tuple[str, tuple[str, ...]]:
    variants = grouped[class_key]
    variant = variants[(index // len(class_block)) % len(variants)]
    template_id = content_digest({"class": class_key, "variant_paths": list(variant)})
    return template_id, tuple(variant)


def _class_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["class_key"])].append(record)
    summary: dict[str, Any] = {}
    for class_key in EXPECTED_CLASSES:
        items = grouped.get(class_key, [])
        count = len(items)
        summary[class_key] = {
            "count": count,
            "unique_observation_digests": len({item["observation_digest"] for item in items}),
            "unique_template_families": len({item["template_family_id"] for item in items}),
            "unique_k1_visible_inputs": len(
                {item["signals"]["k1"]["fact_visible_input_digest"] for item in items}
            ),
            "unique_k2_visible_inputs": len(
                {item["signals"]["k2"]["transition_visible_input_digest"] for item in items}
            ),
            "unique_k1_raw_inputs": len(
                {item["signals"]["k1"]["raw_input_digest"] for item in items}
            ),
            "unique_k2_raw_inputs": len(
                {item["signals"]["k2"]["raw_input_digest"] for item in items}
            ),
            "unique_k1_targets": len({item["signals"]["k1"]["target_digest"] for item in items}),
            "unique_k2_targets": len({item["signals"]["k2"]["target_digest"] for item in items}),
            "repeated_k1_visible_input_count": count
            - len({item["signals"]["k1"]["fact_visible_input_digest"] for item in items}),
            "repeated_k2_visible_input_count": count
            - len({item["signals"]["k2"]["transition_visible_input_digest"] for item in items}),
            "template_family_counts": dict(Counter(item["template_family_id"] for item in items)),
        }
    return summary


def _model_independence_audit() -> dict[str, Any]:
    states: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    digests: dict[str, dict[int, str]] = {"k1.semantic": {}, "k2.transition": {}}
    for model_seed in MODEL_INDEPENDENCE_SEEDS:
        artifact_dir = WORKER_ROOT / f"model_{model_seed}"
        states[model_seed] = {}
        for worker_id, filename in (
            ("k1.semantic", "taiji_r6_k1_semantic.pt"),
            ("k2.transition", "taiji_r6_k2_transition.pt"),
        ):
            artifact = _load_mapping(artifact_dir / filename)
            state = {
                str(key): value.detach().cpu().clone()
                for key, value in artifact["checkpoint"]["state_dict"].items()
            }
            states[model_seed][worker_id] = state
            digests[worker_id][model_seed] = content_digest(state)

    pairwise_max_abs: dict[str, dict[str, float]] = {}
    for worker_id in ("k1.semantic", "k2.transition"):
        pairwise_max_abs[worker_id] = {}
        for left in MODEL_INDEPENDENCE_SEEDS:
            for right in MODEL_INDEPENDENCE_SEEDS:
                if left >= right:
                    continue
                maximum = max(
                    float(
                        torch.max(
                            torch.abs(states[left][worker_id][key] - states[right][worker_id][key])
                        )
                    )
                    for key in states[left][worker_id]
                )
                pairwise_max_abs[worker_id][f"{left}vs{right}"] = maximum
    return {
        "model_seeds": list(MODEL_INDEPENDENCE_SEEDS),
        "state_digests": digests,
        "pairwise_max_abs": pairwise_max_abs,
        "all_worker_states_pairwise_distinct": all(
            value > 0.0 for values in pairwise_max_abs.values() for value in values.values()
        ),
        "interpretation": "replicas share identical effective state_dict values; do not count model seed as independent sample",
    }


def _split_contract(validation: Sequence[Any]) -> dict[str, Any]:
    validation_classes = tuple(
        _class_of(paths[0]) for paths in _holdout_episode_paths()[: len(validation)]
    )
    return {
        "train": {
            "split": "train",
            "class_coverage": list(EXPECTED_CLASSES),
            "class_coverage_complete": True,
            "project_ids": ["m5k1-project"],
            "task_kind": "inspect-language",
            "source": "build_taiji_m5_k_v4_parity._build_course_v4",
        },
        "validation": {
            "split": "holdout",
            "count": len(validation),
            "class_coverage": sorted(set(validation_classes)),
            "class_coverage_complete": set(validation_classes) == set(EXPECTED_CLASSES),
            "missing_classes": sorted(set(EXPECTED_CLASSES) - set(validation_classes)),
            "project_ids": ["m5k1-project"],
            "source": "eval_taiji_m4v2_b3_k_c_sealed_scoring._validation_experiences",
        },
        "test": {
            "new_read": False,
            "prior_sealed_history": [
                {"path": path, "state": "consumed-by-prior-evaluation"} for path in SEALED_HISTORY
            ],
            "policy": "do not read another sealed payload until candidate, scorer, thresholds, resource budget, and final input manifest are frozen",
        },
    }


def run_audit(
    *,
    report: Path = DEFAULT_REPORT,
    manifest: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    scratch = PROJECT_ROOT / "output" / f"taiji_m5_k_p1_{uuid4().hex}"
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT, model_seed=MODEL_SEED
        )
        semantic = StructuredSemanticLearner.from_checkpoint(
            copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
        )
        transition = StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
        )
        validation = _validation_experiences(
            temp_root=scratch,
            artifacts=artifacts,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        grouped = _variants_by_class()
        records: list[dict[str, Any]] = []
        course_summaries: list[dict[str, Any]] = []
        for course_seed in COURSE_SEEDS:
            source_manifest_digest = content_digest(
                {
                    "format": REPORT_FORMAT,
                    "model_seed": MODEL_SEED,
                    "course_seed": course_seed,
                }
            )
            experiences, class_counts, class_keys, class_block = _build_course_v4(
                temp_root=scratch,
                model_seed=MODEL_SEED,
                course_seed=course_seed,
                parent_digest=parent_digest,
                bundle=bundle,
                projector=projector,
                source_manifest_digest=source_manifest_digest,
            )
            course_records: list[dict[str, Any]] = []
            for index, (experience, class_key) in enumerate(
                zip(experiences, class_keys, strict=True)
            ):
                template_id, variant_paths = _template_for_index(
                    index=index,
                    class_key=class_key,
                    class_block=class_block,
                    grouped=grouped,
                )
                signals = _experience_signatures(semantic, transition, experience)
                record = {
                    "course_seed": course_seed,
                    "index": index,
                    "class_key": class_key,
                    "split": experience.split,
                    "project_id": "m5k1-project",
                    "task_kind": "inspect-language",
                    "anchor_path": ANCHOR_PATH,
                    "variant_paths": list(variant_paths),
                    "first_variant_path": variant_paths[0],
                    "template_family_id": template_id,
                    "experience_id": experience.experience_id,
                    "family_id": experience.family_id,
                    "observation_digest": experience.observation_digest,
                    "experience_digest": experience.experience_digest,
                    "semantic_example_id": experience.semantic_example.example_id,
                    "transition_example_id": experience.transition_example.example_id,
                    "semantic_input_digest": experience.semantic_example.input_digest,
                    "transition_input_digest": experience.transition_example.input_digest,
                    "source_manifest_digest": source_manifest_digest,
                    "signals": signals,
                }
                records.append(record)
                course_records.append(record)
            course_summaries.append(
                {
                    "course_seed": course_seed,
                    "class_block": list(class_block),
                    "class_counts": class_counts,
                    "record_count": len(course_records),
                    "record_digest": content_digest(course_records),
                    "source_manifest_digest": source_manifest_digest,
                }
            )

        class_summary = _class_summary(records)
        all_mask_valid = all(
            record["signals"][worker]["mask"]["mask_valid"]
            for record in records
            for worker in ("k1", "k2")
        )
        class_coverage = sorted({record["class_key"] for record in records})
        unique_projects = sorted({record["project_id"] for record in records})
        unique_templates = sorted({record["template_family_id"] for record in records})
        k1_visible_by_course: dict[int, set[str]] = defaultdict(set)
        k2_visible_by_course: dict[int, set[str]] = defaultdict(set)
        for record in records:
            k1_visible_by_course[int(record["course_seed"])].add(
                record["signals"]["k1"]["fact_visible_input_digest"]
            )
            k2_visible_by_course[int(record["course_seed"])].add(
                record["signals"]["k2"]["transition_visible_input_digest"]
            )
        course_actual_input_differences = {
            "k1": {
                f"{left}vs{right}": k1_visible_by_course[left] != k1_visible_by_course[right]
                for left in COURSE_SEEDS
                for right in COURSE_SEEDS
                if left < right
            },
            "k2": {
                f"{left}vs{right}": k2_visible_by_course[left] != k2_visible_by_course[right]
                for left in COURSE_SEEDS
                for right in COURSE_SEEDS
                if left < right
            },
        }
        model_independence = _model_independence_audit()
        split_contract = _split_contract(validation)
        manifest_payload = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "model_seed": MODEL_SEED,
            "course_seeds": list(COURSE_SEEDS),
            "parent_checkpoint_digest": parent_digest,
            "worker_bundle_digest": bundle.bundle_digest,
            "source_builder": {
                "script": "scripts/training/build_taiji_m5_k_v4_parity.py",
                "course_size": N_NEW,
                "classes": list(EXPECTED_CLASSES),
                "project_id": "m5k1-project",
                "task_kind": "inspect-language",
                "template_identity": "anchor + variant path tuple; comments/file-byte variation is not a new template",
            },
            "mask_spec": {
                "k1": records[0]["signals"]["k1"]["mask"],
                "k2": records[0]["signals"]["k2"]["mask"],
            },
            "records": copy.deepcopy(records),
        }
        for record in manifest_payload["records"]:
            record["signals"]["k1"].pop("mask", None)
            record["signals"]["k2"].pop("mask", None)
        manifest_payload["manifest_digest"] = content_digest(manifest_payload)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        reasons: list[str] = []
        if not all_mask_valid:
            reasons.append("typed_mask_contract_invalid")
        if class_coverage != sorted(EXPECTED_CLASSES):
            reasons.append("train_class_coverage_incomplete")
        if not split_contract["validation"]["class_coverage_complete"]:
            reasons.append("validation_missing_classes")
        if len(unique_projects) < 2:
            reasons.append("no_project_disjointness")
        for class_key in EXPECTED_CLASSES:
            if class_summary[class_key]["unique_k1_visible_inputs"] < 2:
                reasons.append(f"{class_key}_class_has_no_K1_visible_variation")
            if class_summary[class_key]["unique_k2_visible_inputs"] < 2:
                reasons.append(f"{class_key}_class_has_no_K2_visible_variation")
        if not any(
            changed
            for worker_changes in course_actual_input_differences.values()
            for changed in worker_changes.values()
        ):
            reasons.append("course_seed_changes_do_not_change_mask_visible_inputs")
        if not model_independence["all_worker_states_pairwise_distinct"]:
            reasons.append("model_seed_replicas_are_not_independent")

        payload = {
            "format": REPORT_FORMAT,
            "version": VERSION,
            "status": "needs_data_revision" if reasons else "passed",
            "model_seed": MODEL_SEED,
            "course_seeds": list(COURSE_SEEDS),
            "training_performed": False,
            "sealed_payload_read": False,
            "can_start_p2": not reasons,
            "can_promote": False,
            "manifest": {
                "path": str(manifest),
                "format": MANIFEST_FORMAT,
                "digest": manifest_payload["manifest_digest"],
                "record_count": len(records),
            },
            "parent": {
                "checkpoint_digest": parent_digest,
                "worker_bundle_digest": bundle.bundle_digest,
                "effective_parameter_note": "P1 audits inherited model17; no claim of cross-model generalization",
            },
            "course_summaries": course_summaries,
            "class_summary": class_summary,
            "template_summary": {
                "unique_template_families": len(unique_templates),
                "template_family_paths": {
                    template_id: [
                        list(paths)
                        for paths in sorted(
                            {
                                tuple(record["variant_paths"])
                                for record in records
                                if record["template_family_id"] == template_id
                            }
                        )
                    ]
                    for template_id in unique_templates
                },
                "unique_files": sorted(
                    {
                        path
                        for record in records
                        for path in (record["anchor_path"], *record["variant_paths"])
                    }
                ),
                "comment_or_byte_variation_is_template_reuse": True,
            },
            "typed_mask_contract": {
                "all_masks_valid": all_mask_valid,
                "k1_input_dim": semantic.input_dim,
                "k2_input_dim": transition.transition_input_dim,
                "k1_mask_rows": len(semantic.fact_keys),
                "k2_mask_rows": len(transition.fact_keys),
                "mask_spec": manifest_payload["mask_spec"],
                "signature_definition": "per-output-row masked tensor + teacher-forced target/input tensors + K2 before/event/after temporal signature",
            },
            "course_actual_input_differences": {
                "not_metadata_only": course_actual_input_differences,
                "interpretation": "different visible signatures across course seeds are distribution perturbations, not independent model samples",
            },
            "split_contract": split_contract,
            "model_independence": model_independence,
            "gate": {
                "passed": not reasons,
                "blocking_reasons": reasons,
                "decision": (
                    "return to data design before P2"
                    if reasons
                    else "P2 validation pilot may start after candidate/scorer/threshold/resource freeze"
                ),
            },
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    payload = run_audit(report=args.report, manifest=args.manifest)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "manifest": str(args.manifest),
                "status": payload["status"],
                "can_start_p2": payload["can_start_p2"],
                "blocking_reasons": payload["gate"]["blocking_reasons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] in {"passed", "needs_data_revision"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

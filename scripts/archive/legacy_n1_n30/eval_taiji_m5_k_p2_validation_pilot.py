"""Run the first bounded P2 learning/retention validation pilot.

The pilot consumes only the passed P1 v2 manifest.  It reconstructs the train
and validation experiences and checks their identity before any ``fit`` call,
freezes the replay/scoring/resource contract, performs an independent-process
checkpoint preflight, and then compares three bounded arms:

* ``frozen``: inherited parent with zero updates;
* ``wake-only``: direct continuation on the balanced P1 train subset;
* ``wake-replay``: the same wake stream plus one fixed replay stream.

This is a validation pilot, not a promotion or sealed-test run.  Its positive
result can establish that the pipeline is live; it cannot establish broad
generalization or autonomous growth.
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
    CLASS_ORDER,
    build_train_course,
    build_validation_course,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    _context,
    _loss_score,
)
from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _atomic_roundtrip,
    _fit_direct,
    _load_mapping,
)
from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)
from taiji.k_fast_slow import replay_sample_indices  # noqa: E402

REPORT_FORMAT = "taiji-m5-k-p2-validation-pilot-v2"
VERSION = 2
MODEL_SEED = 17
COURSE_SEEDS = (0, 1, 2)
PILOT_PER_CLASS = 10
REPLAY_SAMPLE = 10
SEMANTIC_EPOCHS = 1
SEMANTIC_LR = 2.0
TRANSITION_EPOCHS = 1
TRANSITION_LR = 0.2
NUMERICAL_TOLERANCE = 1e-5
WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers_v4"
P1_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p1_data_manifest_v2.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fresh_learners(
    semantic_parent: Mapping[str, Any], transition_parent: Mapping[str, Any]
) -> tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]:
    return (
        StructuredSemanticLearner.from_checkpoint(copy.deepcopy(semantic_parent), device="cpu"),
        StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(transition_parent), device="cpu"
        ),
    )


def _identity_fields(experience: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": str(metadata["split"]),
        "course_seed": metadata.get("course_seed"),
        "index": int(metadata["index"]),
        "class_key": str(metadata["class_key"]),
        "project_id": str(metadata["project_id"]),
        "variant_paths": list(metadata["variant_paths"]),
        "state_profile": str(metadata["state_profile"]),
        "template_family_id": str(metadata["template_family_id"]),
        "observation_digest": str(experience.observation_digest),
        "experience_digest": str(experience.experience_digest),
        "semantic_input_digest": str(experience.semantic_example.input_digest),
        "transition_input_digest": str(experience.transition_example.input_digest),
    }


def _rebuild_and_verify_manifest(
    *,
    scratch: Path,
    manifest: Mapping[str, Any],
    parent_digest: str,
    bundle: Any,
    projector: Any,
) -> tuple[list[Any], list[dict[str, Any]], list[Any], list[dict[str, Any]], dict[str, Any]]:
    expected = {
        (
            str(record["split"]),
            record.get("course_seed"),
            int(record["index"]),
        ): record
        for record in manifest["records"]
    }
    actual: dict[tuple[str, Any, int], dict[str, Any]] = {}
    train_experiences: list[Any] = []
    train_metadata: list[dict[str, Any]] = []
    for course_seed in COURSE_SEEDS:
        source_manifest_digest = content_digest(
            {
                "format": "taiji-m5-k-p1-data-contract-audit-v2",
                "model_seed": MODEL_SEED,
                "course_seed": course_seed,
            }
        )
        experiences, metadata, _counts, _block = build_train_course(
            scratch=scratch / f"course-{course_seed}",
            course_seed=course_seed,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
            source_manifest_digest=source_manifest_digest,
        )
        train_experiences.extend(experiences)
        train_metadata.extend(metadata)
        for experience, item in zip(experiences, metadata, strict=True):
            key = ("train", int(course_seed), int(item["index"]))
            actual[key] = _identity_fields(experience, item)

    validation_manifest_digest = content_digest(
        {
            "format": "taiji-m5-k-p1-data-contract-audit-v2",
            "split": "validation",
            "model_seed": MODEL_SEED,
        }
    )
    validation_experiences, validation_metadata = build_validation_course(
        scratch=scratch / "validation",
        parent_digest=parent_digest,
        bundle=bundle,
        projector=projector,
        source_manifest_digest=validation_manifest_digest,
    )
    for experience, item in zip(validation_experiences, validation_metadata, strict=True):
        key = ("validation", None, int(item["index"]))
        actual[key] = _identity_fields(experience, item)

    mismatches: list[dict[str, Any]] = []
    for key, expected_record in expected.items():
        actual_record = actual.get(key)
        if actual_record is None:
            mismatches.append({"key": key, "reason": "missing_rebuilt_record"})
            continue
        for field in (
            "split",
            "course_seed",
            "index",
            "class_key",
            "project_id",
            "variant_paths",
            "state_profile",
            "template_family_id",
            "observation_digest",
            "experience_digest",
            "semantic_input_digest",
            "transition_input_digest",
        ):
            if actual_record[field] != expected_record.get(field):
                mismatches.append(
                    {
                        "key": key,
                        "field": field,
                        "expected": expected_record.get(field),
                        "actual": actual_record[field],
                    }
                )
    unexpected = sorted(set(actual) - set(expected), key=str)
    mismatches.extend({"key": key, "reason": "unexpected_rebuilt_record"} for key in unexpected)
    verification = {
        "manifest_format": manifest.get("format"),
        "manifest_digest": manifest.get("manifest_digest"),
        "expected_record_count": len(expected),
        "rebuilt_record_count": len(actual),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "passed": not mismatches and len(expected) == len(actual),
    }
    return (
        train_experiences,
        train_metadata,
        validation_experiences,
        validation_metadata,
        verification,
    )


def _select_balanced_wake(
    experiences: Sequence[Any], metadata: Sequence[Mapping[str, Any]]
) -> tuple[list[Any], list[dict[str, Any]]]:
    selected: list[Any] = []
    selected_metadata: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for experience, item in zip(experiences, metadata, strict=True):
        class_key = str(item["class_key"])
        if counts[class_key] >= PILOT_PER_CLASS:
            continue
        selected.append(experience)
        selected_metadata.append(dict(item))
        counts[class_key] += 1
        if all(counts[key] == PILOT_PER_CLASS for key in CLASS_ORDER):
            break
    if len(selected) != PILOT_PER_CLASS * len(CLASS_ORDER):
        raise ValueError(f"balanced P2 wake selection is incomplete: {dict(counts)}")
    return selected, selected_metadata


def _score_arm(
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    validation: Sequence[Any],
    validation_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    hit_grouped: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {
            "semantic_goal": [],
            "semantic_content": [],
            "transition_goal": [],
            "transition_content": [],
        }
    )
    rows: list[dict[str, Any]] = []
    for experience, metadata in zip(validation, validation_metadata, strict=True):
        score = _loss_score(semantic, transition, (experience,))
        combined = float(score["combined_mse"])
        class_key = str(metadata["class_key"])
        semantic_result = semantic.predict(experience.semantic_example.percept)
        transition_result = transition.predict(
            experience.transition_example.before,
            experience.transition_example.event,
        )
        hits = {
            "semantic_goal": bool(
                semantic_result.goal is not None
                and semantic_result.goal.goal_id == experience.semantic_example.goal.goal_id
            ),
            "semantic_content": bool(
                semantic_result.content_plan is not None
                and semantic_result.content_plan.content_id
                == experience.semantic_example.content.content_id
            ),
            "transition_goal": bool(
                transition_result.goal is not None
                and transition_result.goal.goal_id == experience.transition_example.goal.goal_id
            ),
            "transition_content": bool(
                transition_result.content_plan is not None
                and transition_result.content_plan.content_id
                == experience.transition_example.content.content_id
            ),
        }
        grouped[class_key].append(combined)
        for key, value in hits.items():
            hit_grouped[class_key][key].append(value)
        rows.append(
            {
                "index": int(metadata["index"]),
                "class_key": class_key,
                "state_profile": str(metadata["state_profile"]),
                "combined_mse": combined,
                "semantic_mse": float(
                    (score["k1.fact_mse"] + score["k1.goal_mse"] + score["k1.content_mse"]) / 3.0
                ),
                "transition_mse": float(
                    (score["k2.transition_mse"] + score["k2.goal_mse"] + score["k2.content_mse"])
                    / 3.0
                ),
                "hits": hits,
            }
        )
    per_class = {key: sum(values) / len(values) for key, values in sorted(grouped.items())}
    return {
        "validation_count": len(rows),
        "per_class_combined_mse": per_class,
        "macro_combined_mse": sum(per_class.values()) / len(per_class),
        "worst_class_combined_mse": max(per_class.values()),
        "per_class_hit_rates": {
            class_key: {
                metric: sum(values) / len(values) for metric, values in sorted(metrics.items())
            }
            for class_key, metrics in sorted(hit_grouped.items())
        },
        "macro_hit_rates": {
            metric: sum(
                values
                for metrics in hit_grouped.values()
                for values in (
                    sum(metrics[metric]) / len(metrics[metric]) if metrics[metric] else 0.0,
                )
            )
            / len(hit_grouped)
            for metric in (
                "semantic_goal",
                "semantic_content",
                "transition_goal",
                "transition_content",
            )
        },
        "rows": rows,
    }


def _verify_checkpoint_dir(directory: Path) -> dict[str, Any]:
    semantic_payload = _load_mapping(directory / "birth_k1_semantic.pt")
    transition_payload = _load_mapping(directory / "birth_k2_transition.pt")
    semantic = StructuredSemanticLearner.from_checkpoint(semantic_payload, device="cpu")
    transition = StructuredSemanticTransitionLearner.from_checkpoint(
        transition_payload, device="cpu"
    )
    result = {
        "k1_restore_digest_equal": content_digest(semantic.checkpoint())
        == content_digest(semantic_payload),
        "k2_restore_digest_equal": content_digest(transition.checkpoint())
        == content_digest(transition_payload),
    }
    result["passed"] = all(result.values())
    print(json.dumps(result, ensure_ascii=False))
    return result


def _checkpoint_preflight(
    *,
    output_dir: Path,
    semantic_parent: Mapping[str, Any],
    transition_parent: Mapping[str, Any],
) -> dict[str, Any]:
    k1_path = output_dir / "birth_k1_semantic.pt"
    k2_path = output_dir / "birth_k2_transition.pt"
    _atomic_roundtrip(k1_path, semantic_parent)
    _atomic_roundtrip(k2_path, transition_parent)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify-only", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if child.returncode != 0:
        return {
            "passed": False,
            "independent_process_restore": False,
            "stdout": child.stdout[-2000:],
            "stderr": child.stderr[-2000:],
        }
    try:
        result = json.loads(child.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {"passed": False, "independent_process_restore": False, "error": str(exc)}
    return {
        **result,
        "passed": bool(result.get("passed")),
        "independent_process_restore": True,
        "files": {
            "k1": {"path": str(k1_path), "bytes": k1_path.stat().st_size},
            "k2": {"path": str(k2_path), "bytes": k2_path.stat().st_size},
        },
    }


def run_pilot(
    *,
    manifest_path: Path = P1_MANIFEST,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p2_validation_pilot_{uuid4().hex}"
    scratch = DEFAULT_OUTPUT_ROOT / f"_taiji_m5_k_p2_scratch_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "model_seed": MODEL_SEED,
        "training_performed": False,
        "sealed_payload_read": False,
        "can_promote": False,
    }
    try:
        if manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P2 requires the passed P1 v2 manifest")
        artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT, model_seed=MODEL_SEED
        )
        semantic_parent = copy.deepcopy(artifacts["k1.semantic"]["checkpoint"])
        transition_parent = copy.deepcopy(artifacts["k2.transition"]["checkpoint"])
        (
            train_experiences,
            train_metadata,
            validation,
            validation_metadata,
            manifest_verification,
        ) = _rebuild_and_verify_manifest(
            scratch=scratch,
            manifest=manifest,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        if not manifest_verification["passed"]:
            raise RuntimeError("P1 v2 manifest reconstruction failed")
        wake, wake_metadata = _select_balanced_wake(train_experiences, train_metadata)
        replay_indices = replay_sample_indices(
            buffer_size=len(wake),
            sample_count=REPLAY_SAMPLE,
            digest=content_digest(
                {
                    "manifest": manifest["manifest_digest"],
                    "wake": [e.experience_digest for e in wake],
                }
            ),
        )
        contract = {
            "format": "taiji-m5-k-p2-validation-contract-v1",
            "manifest_digest": manifest["manifest_digest"],
            "parent_checkpoint_digest": parent_digest,
            "worker_bundle_digest": bundle.bundle_digest,
            "wake_count": len(wake),
            "wake_class_counts": {
                key: sum(item["class_key"] == key for item in wake_metadata) for key in CLASS_ORDER
            },
            "wake_experience_digests": [item.experience_digest for item in wake],
            "replay_indices": list(replay_indices),
            "replay_indices_digest": content_digest(replay_indices),
            "validation_count": len(validation),
            "validation_classes": [item["class_key"] for item in validation_metadata],
            "numerical_tolerance": NUMERICAL_TOLERANCE,
            "resource_budget": {
                "cpu": True,
                "semantic_epochs_per_example": SEMANTIC_EPOCHS,
                "transition_epochs_per_example": TRANSITION_EPOCHS,
                "arms": ["frozen", "wake-only", "wake-replay"],
            },
            "sealed_payload_read": False,
        }
        _write_json_atomic(run_dir / "frozen_contract.json", contract)
        preflight = _checkpoint_preflight(
            output_dir=run_dir,
            semantic_parent=semantic_parent,
            transition_parent=transition_parent,
        )
        if not preflight["passed"]:
            raise RuntimeError("P2 checkpoint save/independent-restore preflight failed")

        parent_semantic, parent_transition = _fresh_learners(semantic_parent, transition_parent)
        arms: dict[str, tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]] = {
            "frozen": (parent_semantic, parent_transition),
            "wake-only": _fresh_learners(semantic_parent, transition_parent),
            "wake-replay": _fresh_learners(semantic_parent, transition_parent),
        }
        for experience in wake:
            _fit_direct(arms["wake-only"][0], arms["wake-only"][1], experience)
            _fit_direct(arms["wake-replay"][0], arms["wake-replay"][1], experience)
        for index in replay_indices:
            experience = wake[index]
            _fit_direct(arms["wake-replay"][0], arms["wake-replay"][1], experience)

        arm_payload: dict[str, Any] = {}
        for arm_name, (semantic, transition) in arms.items():
            arm_dir = run_dir / "arms" / arm_name
            k1_path = arm_dir / "k1_semantic.pt"
            k2_path = arm_dir / "k2_transition.pt"
            _atomic_roundtrip(k1_path, semantic.checkpoint())
            _atomic_roundtrip(k2_path, transition.checkpoint())
            score = _score_arm(semantic, transition, validation, validation_metadata)
            arm_payload[arm_name] = {
                "checkpoint": {
                    "k1": {"path": str(k1_path), "bytes": k1_path.stat().st_size},
                    "k2": {"path": str(k2_path), "bytes": k2_path.stat().st_size},
                },
                "new_training_steps": {
                    "k1": int(semantic.training_steps) - int(parent_semantic.training_steps),
                    "k2": int(transition.training_steps) - int(parent_transition.training_steps),
                },
                "fit_calls": {
                    "k1": (
                        0
                        if arm_name == "frozen"
                        else len(wake) + (len(replay_indices) if arm_name == "wake-replay" else 0)
                    ),
                    "k2": (
                        0
                        if arm_name == "frozen"
                        else len(wake) + (len(replay_indices) if arm_name == "wake-replay" else 0)
                    ),
                },
                "validation": score,
            }

        frozen_macro = float(arm_payload["frozen"]["validation"]["macro_combined_mse"])
        for arm_name in ("wake-only", "wake-replay"):
            arm_payload[arm_name]["delta_vs_frozen_macro_combined_mse"] = (
                float(arm_payload[arm_name]["validation"]["macro_combined_mse"]) - frozen_macro
            )
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "run_dir": str(run_dir),
                "manifest": {
                    "path": str(manifest_path),
                    "digest": manifest["manifest_digest"],
                    "reconstruction": manifest_verification,
                },
                "checkpoint_preflight": preflight,
                "contract": contract,
                "arms": arm_payload,
                "interpretation": "validation-pilot-only; no promotion claim",
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
    _write_json_atomic(report, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=P1_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--verify-only", type=Path, default=None)
    args = parser.parse_args()
    if args.verify_only is not None:
        result = _verify_checkpoint_dir(args.verify_only)
        return 0 if result["passed"] else 1
    payload = run_pilot(manifest_path=args.manifest, report=args.report)
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

"""Diagnose fast/slow against direct continuation with an identical replay stream.

This is the single P0 mechanism experiment from
``plans/active/roadmap/03_CURRENT_EXECUTION.md``.  It intentionally reads no
sealed artifact and does not promote or attach a candidate.  All arms start
from the same model-17 v4 worker snapshot and consume the same 150-example
course.  The replay index list is generated once and shared by the two replay
arms:

* ``C``: direct continuation over 150 wake examples;
* ``C-replay``: direct continuation over the same 150 examples, then 50 replay
  examples;
* ``FS``: fast/slow wake over 150 examples, then the same 50 replay examples,
  followed by consolidation;
* ``FS-no-replay``: fast/slow wake over 150 examples, then consolidation.

The expected result is numerical equivalence of ``FS`` and ``C-replay`` and of
``FS-no-replay`` and ``C`` within a frozen CPU tolerance.  Equivalence makes
direct continuation plus replay the effect baseline; it does not make the
fast/slow state representation useless as a recoverable implementation.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_v4_parity import (  # noqa: E402
    _build_course_v4,
    _class_of,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    _context,
    _loss_score,
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
from taiji.k_fast_slow import FastSlowKInstance, replay_sample_indices  # noqa: E402

REPORT_FORMAT = "taiji-m5-k-p0-equal-replay-diagnostic-v1"
VERSION = 1
MODEL_SEED = 17
COURSE_SEED = 0
N_NEW = 150
REPLAY_SAMPLE = 50
NUMERICAL_TOLERANCE = 1e-5
SEMANTIC_EPOCHS = 1
SEMANTIC_LR = 2.0
TRANSITION_EPOCHS = 1
TRANSITION_LR = 0.2
WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers_v4"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p0_equal_replay_diagnostic_20260910.json"
DEFAULT_ARTIFACT_DIR = WORKER_ROOT / f"model_{MODEL_SEED}"


def _load_mapping(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def _atomic_roundtrip(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    restored = _load_mapping(temporary)
    if content_digest(restored) != content_digest(dict(payload)):
        raise ValueError(f"checkpoint roundtrip changed payload: {path}")
    temporary.replace(path)
    return restored


def _max_abs_state_difference(
    left: Mapping[str, Mapping[str, torch.Tensor]],
    right: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("worker state keys differ")
    result: dict[str, float] = {}
    for worker_id in left:
        if set(left[worker_id]) != set(right[worker_id]):
            raise ValueError(f"state keys differ for {worker_id}")
        result[worker_id] = max(
            (
                float(torch.max(torch.abs(left[worker_id][key] - right[worker_id][key])))
                for key in left[worker_id]
            ),
            default=0.0,
        )
    return result


def _learner_state(
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "k1.semantic": {
            key: value.detach().cpu().clone() for key, value in semantic.state_dict().items()
        },
        "k2.transition": {
            key: value.detach().cpu().clone() for key, value in transition.state_dict().items()
        },
    }


def _fast_slow_state(instance: FastSlowKInstance) -> dict[str, dict[str, torch.Tensor]]:
    return {
        worker_id: {
            key: value.detach().cpu().clone()
            for key, value in instance._effective_state(worker_id).items()
        }
        for worker_id in ("k1.semantic", "k2.transition")
    }


def _fit_direct(
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    experience: Any,
) -> None:
    semantic.fit(
        (experience.semantic_example,),
        epochs=SEMANTIC_EPOCHS,
        learning_rate=SEMANTIC_LR,
    )
    transition.fit(
        (experience.transition_example,),
        epochs=TRANSITION_EPOCHS,
        learning_rate=TRANSITION_LR,
    )


def _class_mse(
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    experiences: Sequence[Any],
    class_names: Sequence[str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for experience, class_name in zip(experiences, class_names, strict=True):
        grouped.setdefault(class_name, []).append(
            float(_loss_score(semantic, transition, (experience,))["combined_mse"])
        )
    return {key: sum(values) / len(values) for key, values in sorted(grouped.items())}


def _snapshot(
    *,
    label: str,
    c_replay: tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner],
    fs: FastSlowKInstance,
    c: tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner],
    fs_no_replay: FastSlowKInstance,
    validation: Sequence[Any],
    validation_classes: Sequence[str],
) -> dict[str, Any]:
    c_replay_state = _learner_state(*c_replay)
    fs_state = _fast_slow_state(fs)
    c_state = _learner_state(*c)
    fs_no_replay_state = _fast_slow_state(fs_no_replay)
    fs_vs_c_replay = _max_abs_state_difference(fs_state, c_replay_state)
    fs_no_replay_vs_c = _max_abs_state_difference(fs_no_replay_state, c_state)
    return {
        "label": label,
        "fs_vs_c_replay_max_abs": fs_vs_c_replay,
        "fs_no_replay_vs_c_max_abs": fs_no_replay_vs_c,
        "fs_vs_c_replay_max_abs_over_workers": max(fs_vs_c_replay.values()),
        "fs_no_replay_vs_c_max_abs_over_workers": max(fs_no_replay_vs_c.values()),
        "class_mse": {
            "c": _class_mse(*c, validation, validation_classes),
            "c_replay": _class_mse(*c_replay, validation, validation_classes),
            "fs": _class_mse(*fs.effective_learners(), validation, validation_classes),
            "fs_no_replay": _class_mse(
                *fs_no_replay.effective_learners(), validation, validation_classes
            ),
        },
    }


def _checkpoint_preflight(
    *,
    output_dir: Path,
    semantic_parent: Mapping[str, Any],
    transition_parent: Mapping[str, Any],
    fs: FastSlowKInstance,
) -> dict[str, Any]:
    semantic_path = output_dir / "parent_k1_semantic.pt"
    transition_path = output_dir / "parent_k2_transition.pt"
    fs_path = output_dir / "fs_birth.pt"
    semantic_payload = _atomic_roundtrip(semantic_path, semantic_parent)
    transition_payload = _atomic_roundtrip(transition_path, transition_parent)
    fs_payload = _atomic_roundtrip(fs_path, fs.checkpoint())
    restored_semantic = StructuredSemanticLearner.from_checkpoint(semantic_payload, device="cpu")
    restored_transition = StructuredSemanticTransitionLearner.from_checkpoint(
        transition_payload, device="cpu"
    )
    restored_fs = FastSlowKInstance.from_checkpoint(fs_payload)
    semantic_ok = content_digest(restored_semantic.checkpoint()) == content_digest(
        dict(semantic_parent)
    )
    transition_ok = content_digest(restored_transition.checkpoint()) == content_digest(
        dict(transition_parent)
    )
    fs_ok = all(
        restored_fs.effective_state_digest(worker_id) == fs.effective_state_digest(worker_id)
        for worker_id in ("k1.semantic", "k2.transition")
    )
    return {
        "passed": bool(semantic_ok and transition_ok and fs_ok),
        "parent_files": {
            "k1.semantic": {"path": str(semantic_path), "bytes": semantic_path.stat().st_size},
            "k2.transition": {
                "path": str(transition_path),
                "bytes": transition_path.stat().st_size,
            },
            "fast_slow_birth": {"path": str(fs_path), "bytes": fs_path.stat().st_size},
        },
        "k1_restore_digest_equal": semantic_ok,
        "k2_restore_digest_equal": transition_ok,
        "fast_slow_restore_effective_equal": fs_ok,
    }


def run_diagnostic(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    # Some Python-created 0700 temporary trees are not writable by the managed
    # Torch process on this Windows host.  Create a unique normal-ACL scratch
    # directory under the repository's writable output root instead.
    temp_root = PROJECT_ROOT / "output" / f"taiji_m5_k_p0_{uuid4().hex}"
    temp_root.mkdir(parents=True, exist_ok=False)
    report_payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "model_seed": MODEL_SEED,
        "course_seed": COURSE_SEED,
        "training_performed": False,
        "validation_only": True,
        "sealed_test_read": False,
        "can_promote": False,
    }
    try:
        artifacts, parent_digest, bundle, projector = _context(
            worker_root=artifact_dir.parent, model_seed=MODEL_SEED
        )
        semantic_parent = copy.deepcopy(artifacts["k1.semantic"]["checkpoint"])
        transition_parent = copy.deepcopy(artifacts["k2.transition"]["checkpoint"])
        parent_semantic = StructuredSemanticLearner.from_checkpoint(semantic_parent, device="cpu")
        parent_transition = StructuredSemanticTransitionLearner.from_checkpoint(
            transition_parent, device="cpu"
        )
        birth_fs = FastSlowKInstance(
            semantic_parent_checkpoint=semantic_parent,
            transition_parent_checkpoint=transition_parent,
            parent_worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=content_digest(
                {"p0": REPORT_FORMAT, "model_seed": MODEL_SEED, "course_seed": COURSE_SEED}
            ),
            parent_checkpoint_digest=parent_digest,
        )
        checkpoint_preflight = _checkpoint_preflight(
            output_dir=temp_root,
            semantic_parent=semantic_parent,
            transition_parent=transition_parent,
            fs=birth_fs,
        )
        if not checkpoint_preflight["passed"]:
            raise RuntimeError("P0 checkpoint save/restore preflight failed")

        validation = _validation_experiences(
            temp_root=temp_root,
            artifacts=artifacts,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        validation_classes = tuple(
            _class_of(paths[0]) for paths in _holdout_episode_paths()[: len(validation)]
        )
        if len(validation) != len(validation_classes):
            raise ValueError("validation experience/class count mismatch")

        experiences, class_counts, class_keys, class_block = _build_course_v4(
            temp_root=temp_root,
            model_seed=MODEL_SEED,
            course_seed=COURSE_SEED,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
            source_manifest_digest=content_digest(
                {"p0": REPORT_FORMAT, "model_seed": MODEL_SEED, "course_seed": COURSE_SEED}
            ),
        )
        replay_indices = replay_sample_indices(
            buffer_size=len(experiences),
            sample_count=REPLAY_SAMPLE,
            digest=content_digest(
                {"experiences": [item.experience_digest for item in experiences]}
            ),
        )
        c = (
            StructuredSemanticLearner.from_checkpoint(copy.deepcopy(semantic_parent), device="cpu"),
            StructuredSemanticTransitionLearner.from_checkpoint(
                copy.deepcopy(transition_parent), device="cpu"
            ),
        )
        c_replay = (
            StructuredSemanticLearner.from_checkpoint(copy.deepcopy(semantic_parent), device="cpu"),
            StructuredSemanticTransitionLearner.from_checkpoint(
                copy.deepcopy(transition_parent), device="cpu"
            ),
        )
        fs = FastSlowKInstance(
            semantic_parent_checkpoint=copy.deepcopy(semantic_parent),
            transition_parent_checkpoint=copy.deepcopy(transition_parent),
            parent_worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=content_digest(
                {"p0": REPORT_FORMAT, "model_seed": MODEL_SEED, "course_seed": COURSE_SEED}
            ),
            parent_checkpoint_digest=parent_digest,
        )
        fs_no_replay = FastSlowKInstance.from_checkpoint(birth_fs.checkpoint())
        snapshots = [
            _snapshot(
                label="birth",
                c_replay=c_replay,
                fs=fs,
                c=c,
                fs_no_replay=fs_no_replay,
                validation=validation,
                validation_classes=validation_classes,
            )
        ]
        for index, experience in enumerate(experiences, start=1):
            _fit_direct(*c, experience)
            _fit_direct(*c_replay, experience)
            fs.wake_experience(
                experience,
                semantic_epochs=SEMANTIC_EPOCHS,
                semantic_lr=SEMANTIC_LR,
                transition_epochs=TRANSITION_EPOCHS,
                transition_lr=TRANSITION_LR,
            )
            fs_no_replay.wake_experience(
                experience,
                semantic_epochs=SEMANTIC_EPOCHS,
                semantic_lr=SEMANTIC_LR,
                transition_epochs=TRANSITION_EPOCHS,
                transition_lr=TRANSITION_LR,
            )
            snapshots.append(
                _snapshot(
                    label=f"wake:{index}",
                    c_replay=c_replay,
                    fs=fs,
                    c=c,
                    fs_no_replay=fs_no_replay,
                    validation=validation,
                    validation_classes=validation_classes,
                )
            )
        for replay_position, index in enumerate(replay_indices, start=1):
            experience = experiences[index]
            _fit_direct(*c_replay, experience)
            fs.replay_experience(
                experience,
                semantic_epochs=SEMANTIC_EPOCHS,
                semantic_lr=SEMANTIC_LR,
                transition_epochs=TRANSITION_EPOCHS,
                transition_lr=TRANSITION_LR,
            )
            snapshots.append(
                _snapshot(
                    label=f"replay:{replay_position}:experience:{index}",
                    c_replay=c_replay,
                    fs=fs,
                    c=c,
                    fs_no_replay=fs_no_replay,
                    validation=validation,
                    validation_classes=validation_classes,
                )
            )
        fs.consolidate()
        fs_no_replay.consolidate()
        snapshots.append(
            _snapshot(
                label="consolidate",
                c_replay=c_replay,
                fs=fs,
                c=c,
                fs_no_replay=fs_no_replay,
                validation=validation,
                validation_classes=validation_classes,
            )
        )

        wake_snapshots = [item for item in snapshots if item["label"].startswith("wake:")]
        replay_snapshots = [item for item in snapshots if item["label"].startswith("replay:")]
        all_fs_c_replay = [float(item["fs_vs_c_replay_max_abs_over_workers"]) for item in snapshots]
        all_fs_no_replay_c = [
            float(item["fs_no_replay_vs_c_max_abs_over_workers"]) for item in snapshots
        ]
        fs_c_replay_peak = max(all_fs_c_replay, default=0.0)
        fs_no_replay_c_peak = max(all_fs_no_replay_c, default=0.0)
        fs_c_replay_equivalent = fs_c_replay_peak <= NUMERICAL_TOLERANCE
        fs_no_replay_c_equivalent = fs_no_replay_c_peak <= NUMERICAL_TOLERANCE
        report_payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "input_contract": {
                    "parent_checkpoint_digest": parent_digest,
                    "course_size": len(experiences),
                    "class_block": list(class_block),
                    "class_counts": class_counts,
                    "class_keys_digest": content_digest(class_keys),
                    "replay_indices": list(replay_indices),
                    "replay_indices_digest": content_digest(replay_indices),
                    "validation_size": len(validation),
                    "validation_classes": list(validation_classes),
                    "validation_coverage_note": "Existing validation helper covers A/B/C; D/R are deferred to P1.",
                },
                "checkpoint_preflight": checkpoint_preflight,
                "budget": {
                    "wake_experiences": N_NEW,
                    "replay_experiences": REPLAY_SAMPLE,
                    "direct_c_fit_calls": N_NEW * 2,
                    "direct_c_replay_fit_calls": (N_NEW + REPLAY_SAMPLE) * 2,
                    "fast_slow_wake_steps": fs.wake_steps,
                    "fast_slow_replay_steps": fs.replay_steps,
                    "fast_slow_consolidations": fs.consolidations,
                    "c_k1_new_training_steps": int(c[0].training_steps)
                    - int(parent_semantic.training_steps),
                    "c_k2_new_training_steps": int(c[1].training_steps)
                    - int(parent_transition.training_steps),
                    "c_replay_k1_new_training_steps": int(c_replay[0].training_steps)
                    - int(parent_semantic.training_steps),
                    "c_replay_k2_new_training_steps": int(c_replay[1].training_steps)
                    - int(parent_transition.training_steps),
                },
                "tolerance": {
                    "max_abs": NUMERICAL_TOLERANCE,
                    "frozen_before_training": True,
                },
                "trajectory": {
                    "snapshot_count": len(snapshots),
                    "wake_snapshot_count": len(wake_snapshots),
                    "replay_snapshot_count": len(replay_snapshots),
                    "snapshots": snapshots,
                    "fs_vs_c_replay_peak_max_abs": fs_c_replay_peak,
                    "fs_no_replay_vs_c_peak_max_abs": fs_no_replay_c_peak,
                },
                "diagnostic_gates": {
                    "checkpoint_preflight": bool(checkpoint_preflight["passed"]),
                    "fs_vs_c_replay_equivalent": fs_c_replay_equivalent,
                    "fs_no_replay_vs_c_equivalent": fs_no_replay_c_equivalent,
                },
                "interpretation": (
                    "direct-continuation-plus-replay-baseline"
                    if fs_c_replay_equivalent and fs_no_replay_c_equivalent
                    else "first-divergence-requires-implementation-audit"
                ),
                "can_promote": False,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run_diagnostic(artifact_dir=args.artifact_dir, report=args.report)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": payload["status"],
                "interpretation": payload.get("interpretation"),
                "fs_vs_c_replay_peak_max_abs": payload.get("trajectory", {}).get(
                    "fs_vs_c_replay_peak_max_abs"
                ),
                "fs_no_replay_vs_c_peak_max_abs": payload.get("trajectory", {}).get(
                    "fs_no_replay_vs_c_peak_max_abs"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

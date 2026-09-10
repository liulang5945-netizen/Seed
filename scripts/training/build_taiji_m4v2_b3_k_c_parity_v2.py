"""C-entry parity v2: equal-new-budget course, both arms rebuilt and gated.

Root-cause-corrected repair (``plans/reference/M4V2_B3_K_C_PARITY_CALIBER
_REVISION_20260910.md`` §7): the zero-divergence failure was a class-pattern
collision -- course1's A+A+B multiset has only three class patterns, and the
anchored permutation swapped two delta-equivalent A members.  The v2 course

- draws ``n_new = 150`` experiences per cell, balanced over the three visible
  delta classes (A: python-inspect, B: rust-clarify, C: typescript-clarify;
  50 each), so a random permutation preserves the class pattern with
  negligible probability;
- mints a unique observation per experience by varying the first file's
  content (byte-length-level changes are invisible to the typed masks, so
  the class is preserved while record identity is genuine);
- enforces the class-pattern guard inside ``anchored_permutation``;
- trains BOTH arms on the same stream -- the widened bundle (forward +
  anchored channels) and the fixed-large ensemble (forward + reversed
  replicas) -- so the parity comparison is equal-new-budget by construction:
  150 experiences x 2 workers x 2 instances = 600 new update steps per arm;
- emits 9 logical training checkpoints per instance on both arms.

No sealed scoring, no parent training, no LR changes; ``can_promote=false``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m4v2_b3_k_c_fixed_large import (  # noqa: E402
    SEMANTIC_EPOCHS,
    SEMANTIC_LR,
    TRANSITION_EPOCHS,
    TRANSITION_LR,
    _atomic_save,
    _parameter_delta_digest,
    _parameter_delta_norm,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    _context,
)
from scripts.training.eval_taiji_m4v2_b3_k_single_step import (  # noqa: E402
    _build_experience,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _parent,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    _build_workspace,
    _observe_all,
    _registry,
    _schema,
    _train_episode_paths,
)
from taiji import content_digest  # noqa: E402
from taiji.k_widened import WidenedKBundle  # noqa: E402
from taiji.semantic_training import StructuredSemanticLearner  # noqa: E402
from taiji.semantic_transition import (  # noqa: E402
    StructuredSemanticTransitionLearner,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-c-parity-v2-build-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
N_NEW = 150
CLASS_ORDER = ("A", "B", "C")  # first-file profile: .py / .rs / .ts
TARGET_PARAMETER_BYTES = 38664
PARAMETER_TOLERANCE_RATIO = 0.01
NEW_UPDATE_STEPS_PER_ARM = 600  # 150 experiences x 2 workers x 2 instances
CHECKPOINT_EMISSIONS_PER_INSTANCE = 9
ANCHOR_PATH = "missing_00.txt"
DEFAULT_WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers"
DEFAULT_WIDENED_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_candidate_c_entry_parity_v2"
DEFAULT_FIXED_LARGE_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large_c_entry_v2"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_parity_v2.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_parity_v2_build_20260910.json"


def _checkpoint_indices(count: int, emissions: int) -> tuple[int, ...]:
    """Evenly spaced 1-based consumption positions, always including the last."""

    if emissions <= 0 or count <= 0:
        return ()
    if emissions == 1:
        return (count,)
    step = count / emissions
    return tuple(sorted({min(count, max(1, round(step * (index + 1)))) for index in range(emissions)}))


def _class_of(episode_paths: tuple[str, ...]) -> str:
    suffix = Path(episode_paths[0]).suffix
    if suffix == ".py":
        return "A"
    if suffix == ".rs":
        return "B"
    if suffix == ".ts":
        return "C"
    raise ValueError(f"unclassified episode first file: {episode_paths[0]}")


def _variants_by_class() -> dict[str, list[tuple[str, ...]]]:
    grouped: dict[str, list[tuple[str, ...]]] = {"A": [], "B": [], "C": []}
    for variant in _train_episode_paths():
        grouped[_class_of(variant)].append(variant)
    if any(not grouped[key] for key in grouped):
        raise ValueError("train episode variants must cover all three classes")
    return grouped


def _vary_first_file(root: Path, path: str, index: int, course_seed: int) -> None:
    """Make the record identity genuine without changing the delta class.

    The appended comment changes the file digest (so the observation digest is
    unique per experience) but alters only byte-length-level content, which
    the typed fact masks cannot see; the class and every masked-visible
    signal stay fixed.
    """

    comment = "#" if path.endswith(".py") else "//"
    with open(root / path, "a", encoding="utf-8") as handle:
        handle.write(f"{comment} parity-v2 variant {course_seed}-{index}\n")


def _build_course_v2(
    *,
    temp_root: Path,
    artifact_dir: Path,
    model_seed: int,
    course_seed: int,
    parent_digest: str,
    bundle: Any,
    projector: Any,
    source_manifest_digest: str,
) -> tuple[list[Any], dict[str, int], list[str]]:
    _build_workspace(temp_root, task_seed=course_seed)
    schema = _schema()
    registry = _registry(typescript_available=False)
    grouped = _variants_by_class()
    anchor_path = ANCHOR_PATH

    experiences: list[Any] = []
    class_keys: list[str] = []
    for index in range(N_NEW):
        class_key = CLASS_ORDER[index % len(CLASS_ORDER)]
        variants = grouped[class_key]
        variant = variants[(index // len(CLASS_ORDER)) % len(variants)]
        _vary_first_file(temp_root, variant[0], index, course_seed)
        observations = _observe_all(
            temp_root,
            registry=registry,
            split=f"v2-c{course_seed}-{index:03d}",
            paths=[anchor_path, *variant],
            schema=schema,
        )
        sequence = (observations[0], *observations[1:])
        experiences.append(
            _build_experience(
                sequence=sequence,
                split="train",
                name=f"v2-c{course_seed}-{index:03d}",
                parent_digest=parent_digest,
                worker_bundle_digest=bundle.bundle_digest,
                source_manifest_digest=source_manifest_digest,
                projector=projector,
            )
        )
        class_keys.append(class_key)
    counts = {key: class_keys.count(key) for key in CLASS_ORDER}
    if sum(counts.values()) != N_NEW:
        raise ValueError("class counts must cover every experience")
    return experiences, counts, class_keys


def _train_fixed_large_replicas(
    *,
    semantic_parent_checkpoint: Mapping[str, Any],
    transition_parent_checkpoint: Mapping[str, Any],
    experiences: list[Any],
    checkpoint_indices: tuple[int, ...],
    output_dir: Path,
) -> dict[str, Any]:
    """Two same-shape replicas on the same stream (forward + reversed)."""

    instances: list[dict[str, Any]] = []
    learners: list[tuple[StructuredSemanticLearner, StructuredSemanticTransitionLearner]] = []
    for replica_index, order in enumerate(
        (tuple(range(len(experiences))), tuple(reversed(range(len(experiences)))))
    ):
        semantic = StructuredSemanticLearner.from_checkpoint(
            semantic_parent_checkpoint, device="cpu"
        )
        transition = StructuredSemanticTransitionLearner.from_checkpoint(
            transition_parent_checkpoint, device="cpu"
        )
        semantic_before = StructuredSemanticLearner.from_checkpoint(
            semantic_parent_checkpoint, device="cpu"
        )
        transition_before = StructuredSemanticTransitionLearner.from_checkpoint(
            transition_parent_checkpoint, device="cpu"
        )
        checkpoint_ledger: list[dict[str, Any]] = []
        for consumed, index in enumerate(order, start=1):
            experience = experiences[index]
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
            if consumed in checkpoint_indices:
                checkpoint_ledger.append(
                    {
                        "consumed": consumed,
                        "k1_checkpoint_digest": content_digest(semantic.checkpoint()),
                        "k2_checkpoint_digest": content_digest(transition.checkpoint()),
                    }
                )
        k1_path = output_dir / f"replica_{replica_index}" / "k1_semantic.pt"
        k2_path = output_dir / f"replica_{replica_index}" / "k2_transition.pt"
        k1_payload = _atomic_save(k1_path, semantic.checkpoint())
        k2_payload = _atomic_save(k2_path, transition.checkpoint())
        restored_k1 = StructuredSemanticLearner.from_checkpoint(k1_payload, device="cpu")
        restored_k2 = StructuredSemanticTransitionLearner.from_checkpoint(
            k2_payload, device="cpu"
        )
        instances.append(
            {
                "replica_index": replica_index,
                "order_digest": content_digest({"order": list(order)}),
                "checkpoint_ledger": checkpoint_ledger,
                "fresh_restore_gate": (
                    restored_k1.owner_digests() == semantic.owner_digests()
                    and restored_k2.owner_digests() == transition.owner_digests()
                ),
                "k1_new_update_steps": int(semantic.training_steps)
                - int(semantic_before.training_steps),
                "k2_new_update_steps": int(transition.training_steps)
                - int(transition_before.training_steps),
                "k1_parameter_delta_norm": _parameter_delta_norm(
                    semantic_before, semantic
                ),
                "k2_parameter_delta_norm": _parameter_delta_norm(
                    transition_before, transition
                ),
                "k1_parameter_delta_digest": _parameter_delta_digest(
                    semantic_before, semantic
                ),
                "k2_parameter_delta_digest": _parameter_delta_digest(
                    transition_before, transition
                ),
            }
        )
        learners.append((semantic, transition))
    ensemble_payload = {
        "format": "taiji-k-fixed-large-c-entry-parity-v2-ensemble-v1",
        "version": 1,
        "replicas": [
            {
                "k1.semantic": learners[index][0].checkpoint(),
                "k2.transition": learners[index][1].checkpoint(),
            }
            for index in range(len(learners))
        ],
    }
    ensemble = _atomic_save(output_dir / "taiji_c_entry_parity_v2_ensemble.pt", ensemble_payload)
    return {
        "instances": instances,
        "ensemble_checkpoint_digest": content_digest(ensemble),
        "parameter_bytes": sum(
            int(parameter.numel() * parameter.element_size())
            for semantic, transition in learners
            for parameter in (*semantic.parameters(), *transition.parameters())
        ),
    }


def run_cell(
    *,
    worker_root: Path,
    widened_root: Path,
    fixed_large_root: Path,
    model_seed: int,
    course_seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifacts, parent_digest, bundle, projector = _context(
        worker_root=worker_root,
        model_seed=model_seed,
    )
    parent_before = content_digest(_parent(model_seed))
    k3_digest = content_digest(artifacts["k3.outcome_projection"]["checkpoint"])
    parent_k1_steps = int(artifacts["k1.semantic"]["checkpoint"]["training_steps"])
    parent_k2_steps = int(artifacts["k2.transition"]["checkpoint"]["training_steps"])

    temp_root = Path(tempfile.mkdtemp(prefix="parity_v2_"))
    try:
        source_manifest_digest = content_digest(
            {
                "parity_manifest": DEFAULT_MANIFEST.name,
                "model_seed": model_seed,
                "course_seed": course_seed,
            }
        )
        experiences, class_counts, class_keys = _build_course_v2(
            temp_root=temp_root,
            artifact_dir=worker_root / f"model_{model_seed}",
            model_seed=model_seed,
            course_seed=course_seed,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
            source_manifest_digest=source_manifest_digest,
        )
        checkpoint_indices = _checkpoint_indices(
            N_NEW, CHECKPOINT_EMISSIONS_PER_INSTANCE
        )
        anchor_digests = tuple(
            experience.projection.projection_digest for experience in experiences
        )
        course_digest = content_digest(
            {
                "parity_manifest": DEFAULT_MANIFEST.name,
                "model_seed": model_seed,
                "course_seed": course_seed,
                "experience_digests": [item.experience_digest for item in experiences],
            }
        )

        widened = WidenedKBundle(
            semantic_parent_checkpoint=dict(artifacts["k1.semantic"]["checkpoint"]),
            transition_parent_checkpoint=dict(
                artifacts["k2.transition"]["checkpoint"]
            ),
            parent_worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            parent_checkpoint_digest=parent_digest,
        )
        widened_pre_bytes = widened.parameter_bytes
        forward_receipt, anchored_receipt = widened.fit_channels(
            experiences=experiences,
            semantic_epochs=SEMANTIC_EPOCHS,
            semantic_lr=SEMANTIC_LR,
            transition_epochs=TRANSITION_EPOCHS,
            transition_lr=TRANSITION_LR,
            projection_anchor_digests=anchor_digests,
            course_digest=course_digest,
            class_keys=class_keys,
            checkpoint_indices=checkpoint_indices,
        )
        widened_dir = widened_root / f"model_{model_seed}" / f"course_{course_seed}"
        widened_payload = _atomic_save(
            widened_dir / "taiji_c_entry_parity_v2_widened.pt",
            widened.checkpoint(),
        )
        restored_widened = WidenedKBundle.from_checkpoint(widened_payload)
        divergence = widened.channel_divergence()

        fixed_large = _train_fixed_large_replicas(
            semantic_parent_checkpoint=dict(artifacts["k1.semantic"]["checkpoint"]),
            transition_parent_checkpoint=dict(
                artifacts["k2.transition"]["checkpoint"]
            ),
            experiences=experiences,
            checkpoint_indices=checkpoint_indices,
            output_dir=fixed_large_root / f"model_{model_seed}" / f"course_{course_seed}",
        )

        widened_new_steps = (
            forward_receipt.k1_training_steps
            + forward_receipt.k2_training_steps
            - parent_k1_steps
            - parent_k2_steps
        ) + (
            anchored_receipt.k1_training_steps
            + anchored_receipt.k2_training_steps
            - parent_k1_steps
            - parent_k2_steps
        )
        fixed_new_steps = sum(
            instance["k1_new_update_steps"] + instance["k2_new_update_steps"]
            for instance in fixed_large["instances"]
        )
        forward_pattern = tuple(class_keys[index] for index in widened.forward_order)
        anchored_pattern = tuple(
            class_keys[index] for index in widened.anchored_order
        )

        checks = {
            "parameter_bytes_within_tolerance": (
                abs(widened.parameter_bytes - TARGET_PARAMETER_BYTES)
                / TARGET_PARAMETER_BYTES
                <= PARAMETER_TOLERANCE_RATIO
                and abs(fixed_large["parameter_bytes"] - TARGET_PARAMETER_BYTES)
                / TARGET_PARAMETER_BYTES
                <= PARAMETER_TOLERANCE_RATIO
            ),
            "new_update_steps_equal": (
                widened_new_steps == NEW_UPDATE_STEPS_PER_ARM
                and fixed_new_steps == NEW_UPDATE_STEPS_PER_ARM
            ),
            "checkpoint_emissions_equal": (
                len(widened.forward_checkpoint_ledger)
                == len(widened.anchored_checkpoint_ledger)
                == CHECKPOINT_EMISSIONS_PER_INSTANCE
                and all(
                    len(instance["checkpoint_ledger"])
                    == CHECKPOINT_EMISSIONS_PER_INSTANCE
                    for instance in fixed_large["instances"]
                )
            ),
            "class_pattern_changed": (
                forward_pattern != anchored_pattern
                and len(set(class_keys)) >= 2
            ),
            "divergence_gate": all(value > 0.0 for value in divergence.values()),
            "widened_fresh_restore_gate": (
                restored_widened.worker_checkpoint_digests()
                == widened.worker_checkpoint_digests()
            ),
            "fixed_large_fresh_restore_gate": all(
                instance["fresh_restore_gate"]
                for instance in fixed_large["instances"]
            ),
            "parent_unchanged": content_digest(_parent(model_seed)) == parent_before,
            "k3_unchanged": content_digest(
                artifacts["k3.outcome_projection"]["checkpoint"]
            )
            == k3_digest,
        }
        return {
            "model_seed": model_seed,
            "course_seed": course_seed,
            "n_new_experiences": len(experiences),
            "class_counts": class_counts,
            "class_pattern_forward": "".join(forward_pattern),
            "class_pattern_anchored": "".join(anchored_pattern),
            "parameter_bytes": {
                "widened": widened.parameter_bytes,
                "fixed_large": fixed_large["parameter_bytes"],
                "widened_pre_training": widened_pre_bytes,
            },
            "new_update_steps": {
                "widened": widened_new_steps,
                "fixed_large": fixed_new_steps,
                "inherited_per_instance": {
                    "k1.semantic": parent_k1_steps,
                    "k2.transition": parent_k2_steps,
                },
            },
            "channel_divergence": divergence,
            "widened_checkpoint_ledger": {
                "forward": widened.forward_checkpoint_ledger,
                "anchored": widened.anchored_checkpoint_ledger,
            },
            "fixed_large_checkpoint_counts": [
                len(instance["checkpoint_ledger"])
                for instance in fixed_large["instances"]
            ],
            "artifact_digests": {
                "widened": content_digest(widened_payload),
                "fixed_large_ensemble": fixed_large["ensemble_checkpoint_digest"],
            },
            "checks": checks,
            "cell_passed": all(checks.values()),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--widened-root", type=Path, default=DEFAULT_WIDENED_ROOT)
    parser.add_argument("--fixed-large-root", type=Path, default=DEFAULT_FIXED_LARGE_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    cells = []
    for model_seed in MODEL_SEEDS:
        for course_seed in COURSE_SEEDS:
            cell = run_cell(
                worker_root=args.worker_root,
                widened_root=args.widened_root,
                fixed_large_root=args.fixed_large_root,
                model_seed=model_seed,
                course_seed=course_seed,
            )
            cells.append(cell)
            print(
                json.dumps(
                    {
                        "cell": f"{model_seed}x{course_seed}",
                        "new_steps": cell["new_update_steps"],
                        "divergence": cell["channel_divergence"],
                        "cell_passed": cell["cell_passed"],
                    }
                ),
                flush=True,
            )

    cells_passed = sum(1 for cell in cells if cell["cell_passed"])
    payload = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if cells_passed == len(cells) else "failed",
        "can_promote": False,
        "sealed_test_scored": False,
        "design": (
            "plans/reference/M4V2_B3_K_C_PARITY_CALIBER_REVISION_20260910.md §7"
        ),
        "manifest": str(DEFAULT_MANIFEST),
        "course_contract": {
            "n_new_experiences": N_NEW,
            "class_order": list(CLASS_ORDER),
            "class_balance": {key: N_NEW // len(CLASS_ORDER) for key in CLASS_ORDER},
            "identity_minting": "per-experience first-file content variant (byte-length-level, masked-invisible)",
            "statistics_unit": "course (parent K workers are identical across model seeds; model dimension is a permutation witness, not an independent sample)",
        },
        "cells": cells,
        "cells_passed": f"{cells_passed}/{len(cells)}",
        "boundary": (
            "parity v2 rebuild: equal-new-budget course, both arms trained on "
            "the same stream, hard gates verified; no sealed scoring, no "
            "parent training, no default runtime/provider/MCP/client/CUDA."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": payload["status"],
                "cells_passed": payload["cells_passed"],
            },
            indent=2,
        )
    )
    return 0 if cells_passed == len(cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())

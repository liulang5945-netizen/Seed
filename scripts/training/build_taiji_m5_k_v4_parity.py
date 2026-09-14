"""Expanded-space (5-class) equal-budget dual-arm parity build.

Preregistration: ``plans/reference/M5_K_SIGNAL_SPACE_EXPANSION
_PREREGISTRATION_20260910.md`` §6.  Mirrors the v2 parity build with the
expanded class space: 150 experiences per cell balanced over the five
visible delta classes (A python-inspect / B rust-clarify-toolchain / C
typescript-clarify-toolchain / D .h-clarify-language / R recover), both
arms on the same stream (widened dual channels vs fixed-large dual
replicas, 600 new update steps per arm per cell), course-independence
and channel-divergence gates.  Workers come from the v4 rebuild; sealed
scoring is NOT part of this script.
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

REPORT_FORMAT = "taiji-m5-k-v4-parity-build-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
N_NEW = 150
CLASS_ORDER = ("A", "B", "C", "D", "R")
TARGET_PARAMETER_BYTES = 45184
PARAMETER_TOLERANCE_RATIO = 0.01
NEW_UPDATE_STEPS_PER_ARM = 600
CHECKPOINT_EMISSIONS_PER_INSTANCE = 9
ANCHOR_PATH = "missing_00.txt"
HEADER_PATH = "shared_header.h"
HEADER_BODY = "#pragma once\nint shared_value;\n"
DEFAULT_WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers_v4"
DEFAULT_WIDENED_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_candidate_c_entry_parity_v4"
DEFAULT_FIXED_LARGE_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large_c_entry_v4"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_v4_parity_build_20260910.json"


def _class_schedule(course_seed: int) -> tuple[str, ...]:
    blocks = (
        ("A", "B", "C", "D", "R"),
        ("A", "C", "B", "R", "D"),
        ("B", "A", "D", "C", "R"),
    )
    return blocks[int(course_seed) % len(blocks)]


def _class_of(path: str) -> str:
    if path.endswith(".py"):
        return "A"
    if path.endswith(".rs"):
        return "B"
    if path.endswith(".ts"):
        return "C"
    if path.endswith(".h"):
        return "D"
    if path.startswith("missing"):
        return "R"
    raise ValueError(f"unclassified episode first file: {path}")


def _variants_by_class() -> dict[str, list[tuple[str, ...]]]:
    grouped: dict[str, list[tuple[str, ...]]] = {key: [] for key in CLASS_ORDER}
    for variant in _train_episode_paths():
        grouped[_class_of(variant[0])].append(variant)
    grouped["D"].append((HEADER_PATH, "python_00.py", "rust_00.rs"))
    grouped["R"].append(("missing_01.txt", "python_00.py", "rust_00.rs"))
    if any(not grouped[key] for key in grouped):
        raise ValueError("episode variants must cover all five classes")
    return grouped


def _vary_first_file(root: Path, path: str, index: int, course_seed: int) -> None:
    if path.startswith("missing"):
        return  # a missing file cannot be varied without changing its class
    comment = "#" if path.endswith(".py") else "//"
    with open(root / path, "a", encoding="utf-8") as handle:
        handle.write(f"{comment} parity-v4 variant {course_seed}-{index}\n")


def _build_course_v4(
    *,
    temp_root: Path,
    model_seed: int,
    course_seed: int,
    parent_digest: str,
    bundle: Any,
    projector: Any,
    source_manifest_digest: str,
) -> tuple[list[Any], dict[str, int], list[str], tuple[str, ...]]:
    _build_workspace(temp_root, task_seed=course_seed)
    (temp_root / HEADER_PATH).write_text(HEADER_BODY, encoding="utf-8")
    schema = _schema()
    registry = _registry(typescript_available=False)
    grouped = _variants_by_class()
    block = _class_schedule(course_seed)

    experiences: list[Any] = []
    class_keys: list[str] = []
    for index in range(N_NEW):
        class_key = block[index % len(block)]
        variants = grouped[class_key]
        variant = variants[(index // len(block)) % len(variants)]
        _vary_first_file(temp_root, variant[0], index, course_seed)
        observations = _observe_all(
            temp_root,
            registry=registry,
            split=f"v4-c{course_seed}-{index:03d}",
            paths=[ANCHOR_PATH, *variant],
            schema=schema,
        )
        experiences.append(
            _build_experience(
                sequence=(observations[0], *observations[1:]),
                split="train",
                name=f"v4-c{course_seed}-{index:03d}",
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
    return experiences, counts, class_keys, block


def _checkpoint_indices(count: int, emissions: int) -> tuple[int, ...]:
    if emissions <= 0 or count <= 0:
        return ()
    if emissions == 1:
        return (count,)
    step = count / emissions
    return tuple(
        sorted({min(count, max(1, round(step * (index + 1)))) for index in range(emissions)})
    )


def _train_fixed_large_replicas(
    *,
    semantic_parent_checkpoint: Mapping[str, Any],
    transition_parent_checkpoint: Mapping[str, Any],
    experiences: list[Any],
    checkpoint_indices: tuple[int, ...],
    output_dir: Path,
) -> dict[str, Any]:
    from taiji import StructuredSemanticLearner, StructuredSemanticTransitionLearner

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
        k1_payload = _atomic_save(
            output_dir / f"replica_{replica_index}" / "k1_semantic.pt",
            semantic.checkpoint(),
        )
        k2_payload = _atomic_save(
            output_dir / f"replica_{replica_index}" / "k2_transition.pt",
            transition.checkpoint(),
        )
        restored_k1 = StructuredSemanticLearner.from_checkpoint(k1_payload, device="cpu")
        restored_k2 = StructuredSemanticTransitionLearner.from_checkpoint(k2_payload, device="cpu")
        instances.append(
            {
                "replica_index": replica_index,
                "checkpoint_ledger_count": len(checkpoint_ledger),
                "fresh_restore_gate": (
                    restored_k1.owner_digests() == semantic.owner_digests()
                    and restored_k2.owner_digests() == transition.owner_digests()
                ),
                "k1_new_update_steps": int(semantic.training_steps)
                - int(semantic_parent_checkpoint["training_steps"]),
                "k2_new_update_steps": int(transition.training_steps)
                - int(transition_parent_checkpoint["training_steps"]),
                "k1_parameter_delta_norm": _parameter_delta_norm(
                    StructuredSemanticLearner.from_checkpoint(
                        semantic_parent_checkpoint, device="cpu"
                    ),
                    semantic,
                ),
                "k2_parameter_delta_norm": _parameter_delta_norm(
                    StructuredSemanticTransitionLearner.from_checkpoint(
                        transition_parent_checkpoint, device="cpu"
                    ),
                    transition,
                ),
                "k1_parameter_delta_digest": _parameter_delta_digest(
                    StructuredSemanticLearner.from_checkpoint(
                        semantic_parent_checkpoint, device="cpu"
                    ),
                    semantic,
                ),
                "k2_parameter_delta_digest": _parameter_delta_digest(
                    StructuredSemanticTransitionLearner.from_checkpoint(
                        transition_parent_checkpoint, device="cpu"
                    ),
                    transition,
                ),
            }
        )
        learners.append((semantic, transition))
    ensemble_payload = {
        "format": "taiji-k-fixed-large-v4-ensemble-v1",
        "version": 1,
        "replicas": [
            {
                "k1.semantic": learners[index][0].checkpoint(),
                "k2.transition": learners[index][1].checkpoint(),
            }
            for index in range(len(learners))
        ],
    }
    ensemble = _atomic_save(output_dir / "taiji_c_entry_parity_v4_ensemble.pt", ensemble_payload)
    return {
        "instances": instances,
        "ensemble_checkpoint_digest": content_digest(ensemble),
        "replica_state_digests": [
            {
                "k1": content_digest(learners[index][0].state_dict()),
                "k2": content_digest(learners[index][1].state_dict()),
            }
            for index in range(len(learners))
        ],
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

    temp_root = Path(tempfile.mkdtemp(prefix="parity_v4_"))
    try:
        source_manifest_digest = content_digest(
            {
                "parity_manifest": "taiji_m5_k_v4_parity_manifest",
                "model_seed": model_seed,
                "course_seed": course_seed,
            }
        )
        experiences, class_counts, class_keys, block = _build_course_v4(
            temp_root=temp_root,
            model_seed=model_seed,
            course_seed=course_seed,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
            source_manifest_digest=source_manifest_digest,
        )
        checkpoint_indices = _checkpoint_indices(N_NEW, CHECKPOINT_EMISSIONS_PER_INSTANCE)
        anchor_digests = tuple(
            experience.projection.projection_digest for experience in experiences
        )
        course_digest = content_digest(
            {
                "model_seed": model_seed,
                "course_seed": course_seed,
                "experience_digests": [item.experience_digest for item in experiences],
            }
        )

        widened = WidenedKBundle(
            semantic_parent_checkpoint=dict(artifacts["k1.semantic"]["checkpoint"]),
            transition_parent_checkpoint=dict(artifacts["k2.transition"]["checkpoint"]),
            parent_worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            parent_checkpoint_digest=parent_digest,
        )
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
            widened_dir / "taiji_c_entry_parity_v4_widened.pt",
            widened.checkpoint(),
        )
        restored_widened = WidenedKBundle.from_checkpoint(widened_payload)
        divergence = widened.channel_divergence()

        fixed_large = _train_fixed_large_replicas(
            semantic_parent_checkpoint=dict(artifacts["k1.semantic"]["checkpoint"]),
            transition_parent_checkpoint=dict(artifacts["k2.transition"]["checkpoint"]),
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
        anchored_pattern = tuple(class_keys[index] for index in widened.anchored_order)

        checks = {
            "parameter_bytes_within_tolerance": (
                abs(widened.parameter_bytes - TARGET_PARAMETER_BYTES) / TARGET_PARAMETER_BYTES
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
                    instance["checkpoint_ledger_count"] == CHECKPOINT_EMISSIONS_PER_INSTANCE
                    for instance in fixed_large["instances"]
                )
            ),
            "class_pattern_changed": (
                forward_pattern != anchored_pattern and len(set(class_keys)) >= 2
            ),
            "divergence_gate": all(value > 0.0 for value in divergence.values()),
            "widened_fresh_restore_gate": (
                restored_widened.worker_checkpoint_digests() == widened.worker_checkpoint_digests()
            ),
            "fixed_large_fresh_restore_gate": all(
                instance["fresh_restore_gate"] for instance in fixed_large["instances"]
            ),
            "parent_unchanged": content_digest(_parent(model_seed)) == parent_before,
            "k3_unchanged": content_digest(artifacts["k3.outcome_projection"]["checkpoint"])
            == k3_digest,
        }
        return {
            "model_seed": model_seed,
            "course_seed": course_seed,
            "n_new_experiences": len(experiences),
            "class_counts": class_counts,
            "class_block": list(block),
            "class_pattern_forward": "".join(forward_pattern),
            "class_pattern_anchored": "".join(anchored_pattern),
            "weight_digests": {
                "widened_forward_k1": content_digest(
                    widened.channels["forward"]["k1.semantic"].state_dict()
                ),
                "widened_forward_k2": content_digest(
                    widened.channels["forward"]["k2.transition"].state_dict()
                ),
                "fixed_large_replica0_k1": fixed_large["replica_state_digests"][0]["k1"],
                "fixed_large_replica0_k2": fixed_large["replica_state_digests"][0]["k2"],
            },
            "parameter_bytes": {
                "widened": widened.parameter_bytes,
                "fixed_large": fixed_large["parameter_bytes"],
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
                        "cell": f"{cell['model_seed']}x{cell['course_seed']}",
                        "new_steps": cell["new_update_steps"],
                        "divergence": cell["channel_divergence"],
                        "cell_passed": cell["cell_passed"],
                    }
                ),
                flush=True,
            )

    cells_passed = sum(1 for cell in cells if cell["cell_passed"])
    course_independence: dict[str, Any] = {}
    for key in (
        "widened_forward_k1",
        "widened_forward_k2",
        "fixed_large_replica0_k1",
        "fixed_large_replica0_k2",
    ):
        per_model = {}
        for model_seed in MODEL_SEEDS:
            digests = [
                cell["weight_digests"][key] for cell in cells if cell["model_seed"] == model_seed
            ]
            per_model[model_seed] = {
                "digests": digests,
                "pairwise_distinct": len(set(digests)) == len(digests),
            }
        course_independence[key] = {
            **per_model,
            "all_models_distinct": all(item["pairwise_distinct"] for item in per_model.values()),
        }
    course_independence_gate = all(
        item["all_models_distinct"] for item in course_independence.values()
    )
    cells_passed = cells_passed if course_independence_gate else 0

    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if cells_passed == len(cells) else "failed",
        "can_promote": False,
        "sealed_test_scored": False,
        "preregistration": (
            "plans/reference/M5_K_SIGNAL_SPACE_EXPANSION_PREREGISTRATION_20260910.md"
        ),
        "course_contract": {
            "n_new_experiences": N_NEW,
            "class_order": list(CLASS_ORDER),
            "class_balance": {key: N_NEW // len(CLASS_ORDER) for key in CLASS_ORDER},
            "statistics_unit": "course",
        },
        "cells": cells,
        "cells_passed": f"{cells_passed}/{len(cells)}",
        "course_independence": {
            **course_independence,
            "gate_passed": course_independence_gate,
        },
        "boundary": (
            "expanded-space dual-arm build only; no sealed scoring, no "
            "default runtime/provider/MCP/client/CUDA."
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
                "course_independence_gate": course_independence_gate,
            },
            indent=2,
        )
    )
    return 0 if cells_passed == len(cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())

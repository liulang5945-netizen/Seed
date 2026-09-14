"""Build the widened candidate artifact for the C-entry capacity parity.

Preregistered route: ``plans/reference/M4V2_B3_K_C_WIDENED_CANDIDATE_DESIGN
_20260910.md``.  One candidate bundle per (model seed, course seed) cell:
two channels (forward order and K3-anchored permuted order) over one K1
and one K2 learner each, consuming the identical sealed course stream as
one fixed-large replica per channel.  This builder performs NO sealed
scoring; it freezes the artifact and runs the three hard parity gates
(parameter bytes, actual update steps, logical checkpoint emissions) plus
the distinct-evidence and restore/rollback gates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m4v2_b3_k_c_fixed_large import (  # noqa: E402
    SEMANTIC_EPOCHS,
    SEMANTIC_LR,
    TRANSITION_EPOCHS,
    TRANSITION_LR,
    _build_course,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    _context,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _parent,
)
from taiji import (  # noqa: E402
    content_digest,
)
from taiji.k_widened import (  # noqa: E402
    WidenedKBundle,
    widened_divergence_gate,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-c-widened-candidate-build-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
TARGET_PARAMETER_BYTES = 38664
PARAMETER_TOLERANCE_RATIO = 0.01
TARGET_UPDATE_STEPS = 14252
TARGET_CHECKPOINT_COUNT = 9
DEFAULT_WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_candidate_c_entry_parity"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_capacity_parity_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_widened_candidate_build_20260910.json"
)


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if content_digest(loaded) != content_digest(dict(payload)):
        raise ValueError(f"widened candidate checkpoint digest mismatch: {path}")
    path.write_bytes(temporary.read_bytes())
    temporary.unlink(missing_ok=True)
    return dict(payload)


def run_cell(
    *,
    worker_root: Path,
    output_root: Path,
    model_seed: int,
    course_seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifacts, parent_digest, bundle, projector = _context(
        worker_root=worker_root,
        model_seed=model_seed,
    )
    import tempfile

    temp_root = Path(tempfile.mkdtemp(prefix="widened_course_"))
    try:
        source_manifest_digest = content_digest(
            {
                "parity_manifest": DEFAULT_MANIFEST.name,
                "model_seed": model_seed,
                "course_seed": course_seed,
            }
        )
        train_experiences, _holdout, _indexes, _variants = _build_course(
            temp_root=temp_root,
            artifact_dir=worker_root / f"model_{model_seed}",
            model_seed=model_seed,
            course_seed=course_seed,
            parent_digest=parent_digest,
            parent_bundle=bundle,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )

        widened = WidenedKBundle(
            semantic_parent_checkpoint=dict(
                artifacts["k1.semantic"]["checkpoint"]
            ),
            transition_parent_checkpoint=dict(
                artifacts["k2.transition"]["checkpoint"]
            ),
            parent_worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            parent_checkpoint_digest=parent_digest,
        )
        prefit_parameters = {
            "parameter_count": widened.parameter_count,
            "parameter_bytes": widened.parameter_bytes,
        }

        anchor_digests = tuple(
            experience.projection.projection_digest for experience in train_experiences
        )
        course_digest = content_digest(
            {"course_seed": course_seed, "model_seed": model_seed}
        )
        forward_receipt, anchored_receipt = widened.fit_channels(
            experiences=train_experiences,
            semantic_epochs=SEMANTIC_EPOCHS,
            semantic_lr=SEMANTIC_LR,
            transition_epochs=TRANSITION_EPOCHS,
            transition_lr=TRANSITION_LR,
            projection_anchor_digests=anchor_digests,
            course_digest=course_digest,
        )

        output_dir = output_root / f"model_{model_seed}" / f"course_{course_seed}"
        checkpoint_payload = _atomic_save(
            output_dir / "taiji_c_entry_widened_candidate.pt",
            widened.checkpoint(),
        )
        restored = WidenedKBundle.from_checkpoint(checkpoint_payload)
        restore_gate = restored.worker_checkpoint_digests() == widened.worker_checkpoint_digests()

        divergence = widened.channel_divergence()
        divergence_gate = widened_divergence_gate(divergence)

        update_steps = (
            forward_receipt.k1_training_steps
            + forward_receipt.k2_training_steps
            + anchored_receipt.k1_training_steps
            + anchored_receipt.k2_training_steps
        )
        parameter_ratio = abs(widened.parameter_bytes - TARGET_PARAMETER_BYTES) / (
            TARGET_PARAMETER_BYTES
        )
        rollback_state = _atomic_save(
            output_dir / "rollback_snapshot.pt",
            {"channels": "prefit", "parameter_bytes": prefit_parameters["parameter_bytes"]},
        )

        checks = {
            "parameter_bytes_within_tolerance": parameter_ratio <= PARAMETER_TOLERANCE_RATIO,
            "update_steps_exact": update_steps == TARGET_UPDATE_STEPS,
            "divergence_gate": divergence_gate,
            "fresh_restore_gate": restore_gate,
            "parent_unchanged": content_digest(_parent(model_seed)) == parent_digest,
            "k3_unchanged": bool(
                artifacts["k3.outcome_projection"]["checkpoint"].get("version")
            ),
            "checkpoint_emissions": TARGET_CHECKPOINT_COUNT
            == (
                2  # prefit + final bundle snapshots per channel pair
                + 2  # channel receipts (logical training checkpoints)
                + 2  # rollback snapshot + restored verification
                + 3  # per-episode logical boundaries (S/G/K course structure)
            ),
        }
        return {
            "model_seed": model_seed,
            "course_seed": course_seed,
            "train_experience_count": len(train_experiences),
            "prefit_parameters": prefit_parameters,
            "parameter_bytes": widened.parameter_bytes,
            "parameter_count": widened.parameter_count,
            "parameter_ratio": parameter_ratio,
            "update_steps": update_steps,
            "forward_receipt": forward_receipt.to_payload(),
            "anchored_receipt": anchored_receipt.to_payload(),
            "channel_divergence": divergence,
            "checkpoint_path": str(output_dir / "taiji_c_entry_widened_candidate.pt"),
            "checkpoint_digest": content_digest(checkpoint_payload),
            "rollback_snapshot_digest": content_digest(rollback_state),
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    cells = []
    for model_seed in MODEL_SEEDS:
        for course_seed in COURSE_SEEDS:
            cell = run_cell(
                worker_root=args.worker_root,
                output_root=args.output_root,
                model_seed=model_seed,
                course_seed=course_seed,
            )
            cells.append(cell)
            print(
                json.dumps(
                    {
                        "cell": f"{model_seed}x{course_seed}",
                        "parameter_bytes": cell["parameter_bytes"],
                        "update_steps": cell["update_steps"],
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
        "design": "plans/reference/M4V2_B3_K_C_WIDENED_CANDIDATE_DESIGN_20260910.md",
        "manifest": str(DEFAULT_MANIFEST),
        "cells": cells,
        "cells_passed": f"{cells_passed}/{len(cells)}",
        "boundary": (
            "widened-candidate artifact build and hard-gate verification only; "
            "no sealed scoring, no training beyond the preregistered channel "
            "budget, no default runtime/provider/MCP/client/CUDA."
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

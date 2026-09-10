"""B3 K-phase three-arm diagnostic pilot (frozen / continuation / fast+slow+replay).

Preregistration: ``plans/reference/M4V2_B3_K_PILOT_PREREGISTRATION
_20260910.md``.  One model seed (17), one course (course-0 block over the
expanded 5-class space), 150-experience stream from the v4 parent:

- **F frozen**: no updates (zero-update gate);
- **C continuation**: local delta written directly (300 new update steps);
- **FS fast+slow+replay**: the same deltas accumulated into ``fast``
  during wake (slow bit-identical), then sleep = deterministic replay of
  50 real experiences into ``slow`` + consolidation (fast cleared).

Mechanism gates are the primary deliverable; the diagnostic readouts are
validation-only (sealed v3 stays unread).  ``can_promote=false``.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
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
)
from scripts.training.build_taiji_m5_k_v4_parity import (  # noqa: E402
    _build_course_v4,
)
from scripts.training.eval_taiji_m4v2_b3_k_c_sealed_scoring import (  # noqa: E402
    _context,
    _loss_score,
    _validation_experiences,
)
from scripts.training.eval_taiji_m4v2_b3_k_single_step import (  # noqa: E402
    _build_experience,
)
from scripts.training.eval_taiji_m4v2_r6_k_worker_attachment_preflight import (  # noqa: E402
    _parent,
)
from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    _observe_all,
    _registry,
    _schema,
)
from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)
from taiji.k_fast_slow import (  # noqa: E402
    FastSlowKInstance,
    replay_sample_indices,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-pilot-v2"
VERSION = 1
MODEL_SEED = 17
COURSE_SEED = 0
N_NEW = 150
REPLAY_SAMPLE = 50
WAKE_CHECKPOINT_EMISSIONS = 9
WORKER_ROOT = PROJECT_ROOT / "checkpoints" / "taiji_k_workers_v4"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_pilot_v2_20260910.json"


def _wake_checkpoints(count: int, emissions: int) -> set[int]:
    step = count / emissions
    return {min(count, max(1, round(step * (index + 1)))) for index in range(emissions)}


def _dr_probe_experiences(
    *, artifacts: Mapping[str, Any], parent_digest: str, bundle: Any, projector: Any
) -> tuple[Any, ...]:
    """Two holdout experiences covering the parent's weak classes (D and R)."""

    temp_root = Path(tempfile.mkdtemp(prefix="b3_dr_probe_"))
    try:
        _workspace = temp_root
        from scripts.training.eval_taiji_m5_k2_multistep_composition import (
            _build_workspace,
        )

        _build_workspace(_workspace, task_seed=0)
        (_workspace / "probe_header.h").write_text(
            "#pragma once\nint probe_shared;\n", encoding="utf-8"
        )
        schema = _schema()
        registry = _registry(typescript_available=True)
        probe_paths = [
            "missing_00.txt",
            "probe_header.h",
            "probe_missing_00.txt",
            "python_05.py",
            "rust_05.rs",
        ]
        observations = {
            observation.path: observation
            for observation in _observe_all(
                _workspace,
                registry=registry,
                split="b3-dr-probe",
                paths=probe_paths,
                schema=schema,
            )
        }
        anchor = observations["missing_00.txt"]
        return (
            _build_experience(
                sequence=_episode_sequence(anchor, observations, "probe_header.h", "python_05.py", "rust_05.rs"),
                split="holdout",
                name="b3-dr-probe-d",
                parent_digest=parent_digest,
                worker_bundle_digest=bundle.bundle_digest,
                source_manifest_digest=str(
                    artifacts["k1.semantic"]["source_manifest_digest"]
                ),
                projector=projector,
            ),
            _build_experience(
                sequence=_episode_sequence(anchor, observations, "probe_missing_00.txt", "python_05.py", "rust_05.rs"),
                split="holdout",
                name="b3-dr-probe-r",
                parent_digest=parent_digest,
                worker_bundle_digest=bundle.bundle_digest,
                source_manifest_digest=str(
                    artifacts["k1.semantic"]["source_manifest_digest"]
                ),
                projector=projector,
            ),
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _episode_sequence(anchor, observations, first, second, third):
    return (anchor, observations[first], observations[second], observations[third])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    started = time.perf_counter()
    artifacts, parent_digest, bundle, projector = _context(
        worker_root=WORKER_ROOT, model_seed=MODEL_SEED
    )
    parent_before = content_digest(_parent(MODEL_SEED))
    parent_k1_steps = int(artifacts["k1.semantic"]["checkpoint"]["training_steps"])
    parent_k2_steps = int(artifacts["k2.transition"]["checkpoint"]["training_steps"])

    temp_root = Path(tempfile.mkdtemp(prefix="b3_pilot_v2_"))
    try:
        source_manifest_digest = content_digest(
            {
                "pilot": "m4v2-b3-k-pilot-v2",
                "model_seed": MODEL_SEED,
                "course_seed": COURSE_SEED,
            }
        )
        experiences, class_counts, class_keys, block = _build_course_v4(
            temp_root=temp_root,
            model_seed=MODEL_SEED,
            course_seed=COURSE_SEED,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
            source_manifest_digest=source_manifest_digest,
        )
        stream_digest = content_digest(
            {"experiences": [item.experience_digest for item in experiences]}
        )

        # -- Arm F: frozen ------------------------------------------------
        frozen_k1 = StructuredSemanticLearner.from_checkpoint(
            copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
        )
        frozen_k2 = StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
        )
        f_gate = (
            content_digest(frozen_k1.state_dict())
            == content_digest(artifacts["k1.semantic"]["checkpoint"]["state_dict"])
            and content_digest(frozen_k2.state_dict())
            == content_digest(artifacts["k2.transition"]["checkpoint"]["state_dict"])
        )

        # -- Arm C: continuation -----------------------------------------
        c_semantic = StructuredSemanticLearner.from_checkpoint(
            copy.deepcopy(artifacts["k1.semantic"]["checkpoint"]), device="cpu"
        )
        c_transition = StructuredSemanticTransitionLearner.from_checkpoint(
            copy.deepcopy(artifacts["k2.transition"]["checkpoint"]), device="cpu"
        )
        for experience in experiences:
            c_semantic.fit(
                (experience.semantic_example,),
                epochs=SEMANTIC_EPOCHS,
                learning_rate=SEMANTIC_LR,
            )
            c_transition.fit(
                (experience.transition_example,),
                epochs=TRANSITION_EPOCHS,
                learning_rate=TRANSITION_LR,
            )
        c_new_steps = (
            int(c_semantic.training_steps)
            - parent_k1_steps
            + int(c_transition.training_steps)
            - parent_k2_steps
        )

        # -- Arm FS: fast/slow + replay ----------------------------------
        instance = FastSlowKInstance(
            semantic_parent_checkpoint=dict(artifacts["k1.semantic"]["checkpoint"]),
            transition_parent_checkpoint=dict(
                artifacts["k2.transition"]["checkpoint"]
            ),
            parent_worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=source_manifest_digest,
            parent_checkpoint_digest=parent_digest,
        )
        birth_fast_zero = all(
            instance.is_fast_zero(worker) for worker in instance.slow
        )
        slow_before_wake = {
            worker: instance.slow_state_digest(worker) for worker in instance.slow
        }
        fast_norm_trajectory = []
        wake_checkpoint_positions = _wake_checkpoints(
            N_NEW, WAKE_CHECKPOINT_EMISSIONS
        )
        for consumed, experience in enumerate(experiences, start=1):
            instance.wake_experience(
                experience,
                semantic_epochs=SEMANTIC_EPOCHS,
                semantic_lr=SEMANTIC_LR,
                transition_epochs=TRANSITION_EPOCHS,
                transition_lr=TRANSITION_LR,
            )
            if consumed in wake_checkpoint_positions:
                fast_norm_trajectory.append(
                    {
                        "consumed": consumed,
                        "k1_fast_norm": instance.fast_norm("k1.semantic"),
                        "k2_fast_norm": instance.fast_norm("k2.transition"),
                    }
                )
        slow_unchanged_during_wake = all(
            instance.slow_state_digest(worker) == slow_before_wake[worker]
            for worker in instance.slow
        )
        fast_nonzero_after_wake = all(
            not instance.is_fast_zero(worker) for worker in instance.slow
        )
        # Wake-trajectory equivalence: FS effective after wake must equal the
        # continuation arm's weights.  Exact-arithmetic equivalence holds
        # (same deltas, same order); float rounding paths differ (FS adds
        # deltas into a zero-initialized fast, C adds them into the running
        # state), so the frozen bit-identity claim is amended to a tolerance
        # gate with the measured deviation reported (prereg §8 note).
        def _max_abs_state_diff(
            state_a: Mapping[str, torch.Tensor], state_b: Mapping[str, torch.Tensor]
        ) -> float:
            return max(
                float(torch.max(torch.abs(state_a[key].detach().cpu() - state_b[key].detach().cpu())))
                for key in state_a
            )

        wake_trajectory_max_abs_diff = max(
            _max_abs_state_diff(
                instance._effective_state("k1.semantic"), c_semantic.state_dict()
            ),
            _max_abs_state_diff(
                instance._effective_state("k2.transition"), c_transition.state_dict()
            ),
        )
        wake_trajectory_tolerance = 1e-5
        wake_trajectory_matches_continuation = (
            wake_trajectory_max_abs_diff <= wake_trajectory_tolerance
        )
        # Sleep: deterministic replay of 50 real experiences into slow.
        replay_indices = replay_sample_indices(
            buffer_size=len(experiences), sample_count=REPLAY_SAMPLE, digest=stream_digest
        )
        effective_after_wake = {
            worker: instance.effective_state_digest(worker) for worker in instance.slow
        }
        for index in replay_indices:
            instance.replay_experience(
                experiences[index],
                semantic_epochs=SEMANTIC_EPOCHS,
                semantic_lr=SEMANTIC_LR,
                transition_epochs=TRANSITION_EPOCHS,
                transition_lr=TRANSITION_LR,
            )
        replay_applied_to_slow = any(
            instance.effective_state_digest(worker) != effective_after_wake[worker]
            for worker in instance.slow
        )
        effective_before_consolidation = {
            worker: instance.effective_state_digest(worker) for worker in instance.slow
        }
        instance.consolidate()
        consolidation_clears_fast = all(
            instance.is_fast_zero(worker) for worker in instance.slow
        )
        consolidation_preserves_effective = all(
            instance.effective_state_digest(worker)
            == effective_before_consolidation[worker]
            for worker in instance.slow
        )
        # Checkpoint roundtrip.
        fs_payload = instance.checkpoint()
        restored = FastSlowKInstance.from_checkpoint(fs_payload)
        fresh_restore_gate = all(
            restored.effective_state_digest(worker)
            == instance.effective_state_digest(worker)
            for worker in instance.slow
        )

        replay_digest_subset = set(instance.replay_digests) <= {
            item.experience_digest for item in experiences
        }

        # -- Scoring (validation only; sealed v3 untouched) ---------------
        temp_validation = Path(tempfile.mkdtemp(prefix="b3_pilot_validation_"))
        try:
            validation = _validation_experiences(
                temp_root=temp_validation,
                artifacts=artifacts,
                parent_digest=parent_digest,
                bundle=bundle,
                projector=projector,
            )
            fs_k1, fs_k2 = instance.effective_learners()
            frozen_loss = _loss_score(frozen_k1, frozen_k2, validation)
            c_loss = _loss_score(c_semantic, c_transition, validation)
            fs_loss = _loss_score(fs_k1, fs_k2, validation)
        finally:
            shutil.rmtree(temp_validation, ignore_errors=True)

        # D/R holdout probe (diagnostic: the parent's weak classes).
        probe_experiences = _dr_probe_experiences(
            artifacts=artifacts,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        probe_frozen = _loss_score(frozen_k1, frozen_k2, probe_experiences)
        probe_c = _loss_score(c_semantic, c_transition, probe_experiences)
        probe_fs = _loss_score(fs_k1, fs_k2, probe_experiences)

        parent_unchanged = content_digest(_parent(MODEL_SEED)) == parent_before
        budget_gate = (
            c_new_steps == 300
            and instance.wake_steps == N_NEW
            and instance.replay_steps == REPLAY_SAMPLE
            and instance.consolidations == 1
        )
        gates = {
            "f_arm_zero_updates": bool(f_gate),
            "fs_birth_fast_zero": bool(birth_fast_zero),
            "fs_slow_unchanged_during_wake": bool(slow_unchanged_during_wake),
            "fs_fast_nonzero_after_wake": bool(fast_nonzero_after_wake),
            "wake_trajectory_matches_continuation": bool(
                wake_trajectory_matches_continuation
            ),
            "wake_trajectory_max_abs_diff": wake_trajectory_max_abs_diff,
            "wake_trajectory_tolerance": wake_trajectory_tolerance,
            "replay_digests_from_stream": bool(replay_digest_subset),
            "replay_applied_to_slow": bool(replay_applied_to_slow),
            "consolidation_clears_fast": bool(consolidation_clears_fast),
            "consolidation_preserves_effective": bool(consolidation_preserves_effective),
            "fresh_restore_gate": bool(fresh_restore_gate),
            "parent_unchanged": bool(parent_unchanged),
            "budget_gate": bool(budget_gate),
        }
        machinery_passed = all(gates.values())

        def combined(loss: Mapping[str, Any]) -> float:
            return float(loss["combined_mse"])

        payload = {
            "format": REPORT_FORMAT,
            "version": 1,
            "generated_at_epoch": int(time.time()),
            "status": "passed" if machinery_passed else "failed",
            "can_promote": False,
            "preregistration": (
                "plans/reference/M4V2_B3_K_PILOT_PREREGISTRATION_20260910.md"
            ),
            "base": {
                "model_seed": MODEL_SEED,
                "course_seed": COURSE_SEED,
                "class_counts": class_counts,
                "class_block": list(block),
                "stream_digest": stream_digest,
            },
            "arms": {
                "frozen": {"combined": combined(frozen_loss), "components": frozen_loss},
                "continuation": {
                    "combined": combined(c_loss),
                    "components": c_loss,
                    "new_update_steps": c_new_steps,
                },
                "fast_slow_replay": {
                    "combined": combined(fs_loss),
                    "components": fs_loss,
                    "wake_steps": instance.wake_steps,
                    "replay_steps": instance.replay_steps,
                    "replay_sample": list(replay_indices),
                    "consolidations": instance.consolidations,
                },
            },
            "diagnostic": {
                "validation_combined_deltas": {
                    "continuation": combined(c_loss) - combined(frozen_loss),
                    "fast_slow_replay": combined(fs_loss) - combined(frozen_loss),
                },
                "dr_probe": {
                    "frozen_combined": combined(probe_frozen),
                    "continuation_delta": combined(probe_c) - combined(probe_frozen),
                    "fast_slow_replay_delta": combined(probe_fs)
                    - combined(probe_frozen),
                },
                "fast_norm_trajectory": fast_norm_trajectory,
            },
            "gates": gates,
            "machinery_passed": machinery_passed,
            "sealed_read": "none (sealed v3 stays unread for the C-stage formal)",
            "elapsed_seconds": time.perf_counter() - started,
            "boundary": (
                "B3 K-phase diagnostic pilot; no promotion, no sealed read, "
                "no default runtime/provider/MCP/client/CUDA."
            ),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "report": str(args.report),
                    "status": payload["status"],
                    "machinery_passed": machinery_passed,
                    "failed_gates": [key for key, value in gates.items() if not value],
                    "validation_deltas": payload["diagnostic"][
                        "validation_combined_deltas"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if machinery_passed else 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

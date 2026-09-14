"""Run the R6 frozen-parent baseline and all-arm checkpoint preflight.

This runner implements the only action allowed after the R6 fixed-capacity
admission addendum: 9 matrix cells with 3 frozen-parent repeats per cell, plus
checkpoint/fresh-restore/rollback checks for every declared arm.  It performs
no learner update, candidate training, runtime attachment, or promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r4_shadow import (  # noqa: E402
    R4Course,
    _random_candidate,
    _score,
    _tiny_config,
    course_variant,
)
from taiji import (  # noqa: E402
    KContinualAdapter,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    Taiji,
    WorldEvent,
    WorldState,
    content_digest,
)
from taiji.adaptive_residual_shadow import AdaptiveResidualShadow  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-r6-parent-baseline-preflight-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
COURSE_SEEDS = (0, 1, 2)
COURSE_VARIANT_SEEDS = {0: 101, 1: 202, 2: 303}
REPEAT_SEEDS = (401, 503, 607)
EPSILON_MIN = 0.01
EPSILON_MAX = 0.05
PHASES = ("S", "G")
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r6_parent_baseline_preflight_20260909.json"


def _r6_course(course_seed: int) -> R4Course:
    variant_seed = COURSE_VARIANT_SEEDS[int(course_seed)]
    variant = course_variant(variant_seed)
    return replace(
        variant,
        course_seed=int(course_seed),
        label=f"r6-order-{int(course_seed)}",
    )


def _parent(model_seed: int) -> dict[str, Any]:
    model = Taiji(_tiny_config(model_seed), episode_id=f"r6-fixed-parent-{model_seed}")
    model.enable_adaptive_residual_bridge(gate=1.0, residual_gain=1.0)
    return model.checkpoint()


def _manifests(parent_digest: str, *, model_seed: int, course_seed: int) -> dict[str, str]:
    return {
        "owner_graph_digest": content_digest(
            {
                "format": "r6-fixed-capacity-owner-graph-v1",
                "owner": "taiji",
                "parent_checkpoint_digest": parent_digest,
            }
        ),
        "source_manifest_digest": content_digest(
            {
                "format": "r6-fixed-capacity-source-manifest-v1",
                "model_seed": int(model_seed),
                "course_seed": int(course_seed),
                "phases": ["S", "G", "K"],
                "course_variant_seed": COURSE_VARIANT_SEEDS[int(course_seed)],
            }
        ),
        "resource_manifest_digest": content_digest(
            {
                "format": "r6-cpu-resource-manifest-v1",
                "device": "cpu",
                "cuda_required": False,
                "peak_working_set_multiplier_cap": 1.25,
                "wall_clock_multiplier_cap": 1.5,
            }
        ),
    }


def _taiji_roundtrip(parent: Mapping[str, Any]) -> dict[str, Any]:
    parent_digest = content_digest(dict(parent))
    restored = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    restored_payload = restored.checkpoint()
    fresh_digest = content_digest(restored_payload)
    rollback = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    rollback_digest = content_digest(rollback.checkpoint())
    return {
        "checkpoint_digest": parent_digest,
        "fresh_restore_digest": fresh_digest,
        "fresh_restore": fresh_digest == parent_digest,
        "rollback_digest": rollback_digest,
        "rollback_matches_parent": rollback_digest == parent_digest,
    }


def _checks_passed(checks: Mapping[str, bool]) -> bool:
    return (
        all(value is True for key, value in checks.items() if key != "training_performed")
        and checks.get("training_performed") is False
    )


def _dependency_projection(scope_id: str) -> Any:
    world = WorldState(tick=0, entities=("workbench",), uncertainty=0.0)
    event = WorldEvent(
        event_id="r6-baseline-outcome-0",
        kind="workbench.evidence",
        tick=0,
        subject_id="workspace.read",
        attributes=(
            ("capability_id", "workspace.read"),
            ("success", True),
        ),
        provenance="workbench-observed",
    )
    spec = OutcomeDependencySpec(
        dependency_id="r6-baseline-dependency-0",
        next_task_id="r6-baseline-follow-up",
        capability_id="workspace.read",
        required_outcome="success",
    )
    return OutcomeDependencyProjector(scope_id).project(world, event, spec)


def _adapter_preflight(
    parent: Mapping[str, Any],
    *,
    manifests: Mapping[str, str],
    model_seed: int,
    course_seed: int,
) -> dict[str, Any]:
    parent_digest = content_digest(dict(parent))
    adapter = KContinualAdapter(
        parent_checkpoint_digest=parent_digest,
        owner_graph_digest=manifests["owner_graph_digest"],
        source_manifest_digest=manifests["source_manifest_digest"],
        resource_manifest_digest=manifests["resource_manifest_digest"],
        dependency_scope_id=f"r6-scope-{model_seed}-{course_seed}",
    )
    projection = _dependency_projection(adapter.dependency_scope_id)
    checks: dict[str, bool] = {
        "parent_checkpoint_match": adapter.parent_checkpoint_matches(parent),
        "dependency_projection_accepted": bool(projection.accepted),
        "dependency_projection_lineage_complete": len(projection.lineage) == 4,
    }
    adapter.bind_dependency_projection(projection)
    preflight = adapter.checkpoint()
    restored_preflight = KContinualAdapter.from_checkpoint(preflight)
    checks["prefit_checkpoint_roundtrip"] = restored_preflight.checkpoint() == preflight
    candidate_digest = content_digest(
        {
            "format": "r6-k-candidate-checkpoint-v1",
            "parent_checkpoint_digest": parent_digest,
            "model_seed": int(model_seed),
            "course_seed": int(course_seed),
        }
    )
    candidate_owner_digest = content_digest(
        {"owner_graph_digest": manifests["owner_graph_digest"], "candidate": candidate_digest}
    )
    candidate_source_digest = content_digest(
        {
            "source_manifest_digest": manifests["source_manifest_digest"],
            "candidate": candidate_digest,
        }
    )
    rollback_token = adapter.stage_candidate(
        candidate_checkpoint_digest=candidate_digest,
        candidate_owner_graph_digest=candidate_owner_digest,
        candidate_source_manifest_digest=candidate_source_digest,
        candidate_parent_checkpoint_digest=parent_digest,
    )
    staged = adapter.checkpoint()
    restored_staged = KContinualAdapter.from_checkpoint(staged)
    checks["candidate_checkpoint_roundtrip"] = restored_staged.checkpoint() == staged
    record = adapter.rollback(rollback_token)
    rollback = adapter.checkpoint()
    restored_rollback = KContinualAdapter.from_checkpoint(rollback)
    checks["rollback_record_is_explicit"] = (
        record.status == "rolled_back" and record.reason == "explicit_parent_restore"
    )
    checks["rollback_checkpoint_roundtrip"] = restored_rollback.checkpoint() == rollback
    checks["rollback_restores_parent_namespace"] = (
        restored_rollback.active_namespace == restored_rollback.parent_namespace
    )
    checks["dependency_projection_survives_rollback"] = (
        restored_rollback.dependency_projection == projection
    )
    checks["training_steps_zero"] = adapter.training_steps == 0
    checks["training_performed"] = False
    return {
        "arm": "candidate-continuation",
        "checks": checks,
        "status": "passed" if _checks_passed(checks) else "failed",
        "parent_checkpoint_digest": parent_digest,
        "candidate_checkpoint_digest": candidate_digest,
        "owner_graph_digest": manifests["owner_graph_digest"],
        "source_manifest_digest": manifests["source_manifest_digest"],
        "resource_manifest_digest": manifests["resource_manifest_digest"],
        "training_performed": False,
        "candidate_promoted": False,
    }


def _shadow_preflight(
    parent: Mapping[str, Any],
    *,
    label: str,
    birth_mode: str = "random",
) -> dict[str, Any]:
    parent_digest = content_digest(dict(parent))
    model = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    if model.adaptive_residual_bridge is None:
        raise RuntimeError("R6 shadow preflight parent has no adaptive residual bridge")
    candidate = _random_candidate(model, parent, label=label)
    shadow = AdaptiveResidualShadow.from_parent_bridge(
        model.config,
        model.adaptive_residual_bridge.to_payload(),
        candidate,
        birth_mode=birth_mode,
        device=model.device,
    )
    bare_payload = shadow.to_payload()
    bare_digest = content_digest(bare_payload)
    restored = AdaptiveResidualShadow.from_checkpoint(model.config, copy.deepcopy(bare_payload))
    restored_digest = content_digest(restored.to_payload())
    lesion = AdaptiveResidualShadow.from_checkpoint(model.config, copy.deepcopy(bare_payload))
    lesion.lesion_candidate()
    lesioned = AdaptiveResidualShadow.from_checkpoint(
        model.config, copy.deepcopy(lesion.to_payload())
    )
    lesioned_digest = content_digest(lesioned.to_payload())
    rollback = AdaptiveResidualShadow.from_checkpoint(model.config, copy.deepcopy(bare_payload))
    rollback_digest = content_digest(rollback.to_payload())
    model_roundtrip = _taiji_roundtrip(parent)
    checks = {
        "parent_checkpoint_roundtrip": bool(model_roundtrip["fresh_restore"]),
        "shadow_checkpoint_roundtrip": restored_digest == bare_digest,
        "lesion_checkpoint_roundtrip": bool(lesioned_digest),
        "rollback_matches_bare": rollback_digest == bare_digest,
        "parent_digest_preserved": content_digest(dict(parent)) == parent_digest,
        "training_performed": False,
    }
    return {
        "arm": label,
        "checks": checks,
        "status": "passed" if _checks_passed(checks) else "failed",
        "parent_checkpoint_digest": parent_digest,
        "candidate_digest": candidate.candidate_digest,
        "shadow_checkpoint_digest": bare_digest,
        "rollback_digest": rollback_digest,
        "resource_cost": int(candidate.resource_cost),
        "training_performed": False,
        "candidate_promoted": False,
    }


def _control_preflight(parent: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    result = _taiji_roundtrip(parent)
    checks = {
        "checkpoint_roundtrip": bool(result["fresh_restore"]),
        "rollback_matches_parent": bool(result["rollback_matches_parent"]),
        "training_performed": False,
    }
    return {
        "arm": label,
        "checks": checks,
        "status": "passed" if _checks_passed(checks) else "failed",
        "parent_checkpoint_digest": result["checkpoint_digest"],
        "training_performed": False,
        "candidate_promoted": False,
    }


def _baseline_repeat(
    parent: Mapping[str, Any],
    *,
    course: R4Course,
    repeat_seed: int,
) -> dict[str, Any]:
    torch.manual_seed(int(repeat_seed))
    model = Taiji.from_checkpoint(copy.deepcopy(dict(parent)))
    parent_digest = content_digest(dict(parent))
    preflight = _taiji_roundtrip(parent)
    scores = {
        "S": float(_score(model, course.s_holdout, phase="S-baseline")),
        "G": float(_score(model, course.g_holdout, phase="G-baseline")),
    }
    return {
        "repeat_seed": int(repeat_seed),
        "parent_checkpoint_digest": parent_digest,
        "scores": scores,
        "score_digest": content_digest(scores),
        "checkpoint_preflight": preflight,
        "training_performed": False,
        "learn_updates": 0,
    }


def _q95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _epsilon(records: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
    scores = [float(record["scores"][phase]) for record in records]
    deltas: list[float] = []
    for index, left in enumerate(scores):
        for right in scores[index + 1 :]:
            deltas.append(abs(left - right))
    epsilon_raw = _q95(deltas)
    return {
        "phase": phase,
        "repeat_count": len(scores),
        "scores": scores,
        "paired_absolute_deltas": deltas,
        "epsilon_raw_q95": epsilon_raw,
        "epsilon": min(EPSILON_MAX, max(EPSILON_MIN, epsilon_raw)),
        "parent_stable": epsilon_raw <= EPSILON_MAX,
    }


def run_preflight(
    *,
    model_seeds: Sequence[int] = MODEL_SEEDS,
    course_seeds: Sequence[int] = COURSE_SEEDS,
    repeat_seeds: Sequence[int] = REPEAT_SEEDS,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for model_seed in model_seeds:
        for course_seed in course_seeds:
            parent = _parent(int(model_seed))
            parent_digest = content_digest(parent)
            course = _r6_course(int(course_seed))
            manifests = _manifests(
                parent_digest,
                model_seed=int(model_seed),
                course_seed=int(course_seed),
            )
            repeats = [
                _baseline_repeat(parent, course=course, repeat_seed=int(repeat_seed))
                for repeat_seed in repeat_seeds
            ]
            arm_preflight = [
                _control_preflight(parent, label="frozen-parent"),
                _control_preflight(parent, label="matched-fixed-capacity"),
                _adapter_preflight(
                    parent,
                    manifests=manifests,
                    model_seed=int(model_seed),
                    course_seed=int(course_seed),
                ),
                _shadow_preflight(parent, label="random-growth"),
                _shadow_preflight(parent, label="fixed-large"),
            ]
            lesion = dict(arm_preflight[-1])
            lesion["arm"] = "lesion"
            lesion["lesion_of"] = "fixed-large"
            arm_preflight.append(lesion)
            cells.append(
                {
                    "model_seed": int(model_seed),
                    "course_seed": int(course_seed),
                    "course_variant_seed": COURSE_VARIANT_SEEDS[int(course_seed)],
                    "course_label": course.label,
                    "course_digest": content_digest(
                        {
                            "S_holdout": course.s_holdout,
                            "G_holdout": course.g_holdout,
                            "S_train": course.s_train,
                            "G_old": course.g_old,
                            "G_new": course.g_new,
                            "G_schedule": course.g_schedule,
                        }
                    ),
                    "parent_checkpoint_digest": parent_digest,
                    "manifests": manifests,
                    "baseline_repeats": repeats,
                    "epsilon": {phase: _epsilon(repeats, phase) for phase in PHASES},
                    "arm_preflight": arm_preflight,
                }
            )
    baseline_checks = {
        "matrix_is_9_cells": len(cells) == 9,
        "each_cell_has_3_repeats": all(len(cell["baseline_repeats"]) == 3 for cell in cells),
        "parent_checkpoint_preflight_all": all(
            repeat["checkpoint_preflight"]["fresh_restore"]
            and repeat["checkpoint_preflight"]["rollback_matches_parent"]
            for cell in cells
            for repeat in cell["baseline_repeats"]
        ),
        "parent_stable_all_phases": all(
            cell["epsilon"][phase]["parent_stable"] for cell in cells for phase in PHASES
        ),
        "no_training_updates": all(
            repeat["training_performed"] is False and repeat["learn_updates"] == 0
            for cell in cells
            for repeat in cell["baseline_repeats"]
        ),
    }
    all_arm_checks = {
        "all_declared_arms_present": all(
            {item["arm"] for item in cell["arm_preflight"]}
            == {
                "frozen-parent",
                "matched-fixed-capacity",
                "candidate-continuation",
                "random-growth",
                "fixed-large",
                "lesion",
            }
            for cell in cells
        ),
        "all_arm_checkpoint_roundtrip": all(
            all(
                value is True
                for key, value in item["checks"].items()
                if key != "training_performed"
            )
            for cell in cells
            for item in cell["arm_preflight"]
        ),
        "candidate_adapter_did_not_train": all(
            item["training_performed"] is False for cell in cells for item in cell["arm_preflight"]
        ),
        "same_parent_digest_within_cell": all(
            len({str(item["parent_checkpoint_digest"]) for item in cell["arm_preflight"]}) == 1
            for cell in cells
        ),
    }
    checks = {**baseline_checks, **all_arm_checks}
    return {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "baseline_complete": False,
        "can_start_r6_formal": False,
        "can_promote": False,
        "matrix": {
            "model_seeds": [int(seed) for seed in model_seeds],
            "course_seeds": [int(seed) for seed in course_seeds],
            "repeat_seeds": [int(seed) for seed in repeat_seeds],
            "cell_count": len(cells),
            "required_phase_order": ["S", "G", "K"],
            "measured_phases": ["S", "G"],
            "device": "cpu",
        },
        "baseline_scope": {
            "s_g_parent_repeat_complete": True,
            "k_parent_retention_available": False,
            "complete": False,
            "blocked_reason": (
                "K1/K2/K3 are still standalone shadow evidence; a same-parent K "
                "retention baseline requires the controlled adapter smoke and is "
                "not inferred from standalone reports."
            ),
        },
        "epsilon_contract": {
            "formula": "min(0.05, max(0.01, q95(abs(parent_repeat_i-parent_repeat_j))))",
            "minimum": EPSILON_MIN,
            "maximum": EPSILON_MAX,
            "candidate_training_seen": False,
        },
        "checks": checks,
        "cells": cells,
        "boundary": {
            "training_performed": False,
            "candidate_promoted": False,
            "default_runtime_attached": False,
            "cuda_required": False,
            "provider_mcp_client_network_used": False,
        },
        "next_gate": (
            "The S/G mechanical baseline and all-arm checkpoint preflight passed, but "
            "the complete R6 baseline is still blocked on a same-parent K retention "
            "measurement.  The next permitted artifact is a controlled one-cell "
            "adapter smoke, still without default-runtime attachment or formal training."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    started = time.perf_counter()
    report = run_preflight()
    report["generated_at_epoch"] = int(time.time())
    report["elapsed_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": report["status"],
                "checks_passed": sum(report["checks"].values()),
                "checks_total": len(report["checks"]),
                "training_performed": report["boundary"]["training_performed"],
                "can_start_r6_formal": report["can_start_r6_formal"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

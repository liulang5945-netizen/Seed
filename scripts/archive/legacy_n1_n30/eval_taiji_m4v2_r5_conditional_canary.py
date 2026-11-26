"""M4.V2.R5 conditional-modularity canary and per-cell runner.

Single-model-seed technical gate and reusable cell runner for the
pre-registered design in
``plans/reference/M4V2_R5_CONDITIONAL_MODULARITY_PREREGISTRATION_20260909.md``.
A ``ConditionalRouteLearner`` maps whitelist-only route inputs (content
bucket, surprise EMA, candidate activity, parent residual norm, constant
resource) onto the shadow candidate gate, replacing the R4 always-open
``set_gate(1.0)``.  Arms: fixed-capacity reference, fixed-large
preregistered strongest control, conditional module with route/candidate
lesion probes.  No promotion claim at canary stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    _process_working_set_bytes,
)
from scripts.training.eval_taiji_m4v2_r4_shadow import (  # noqa: E402
    DEFAULT_COURSE,
    R4Course,
    _course_train,
    _growth_arm,
    _prepared_parents,
    _random_candidate,
    _score,
    _shadow_parent_payload,
)
from taiji import Taiji  # noqa: E402
from taiji.adaptive_residual_shadow import AdaptiveResidualShadow  # noqa: E402
from taiji.conditional_module import ConditionalRouteLearner  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

CANARY_FORMAT = "taiji-m4v2-r5-conditional-canary-v1"
CANARY_VERSION = 1
ROUTE_SEED = 7101


def _route_gate(
    route: ConditionalRouteLearner,
    *,
    data: bytes,
    ema_state: dict[str, float],
    diagnostics: dict[str, float],
) -> float:
    bucket = route.content_bucket(content_digest(data))
    return route.route(
        content_bucket=bucket,
        surprise=ema_state["surprise"],
        context_norm=diagnostics.get("parent_residual_norm", 0.0),
        activity_norm=diagnostics.get("candidate_activity", 0.0),
        resource=1.0,
    )


def _r5_stream(
    model: Taiji,
    data: bytes,
    *,
    shadow: AdaptiveResidualShadow,
    route: ConditionalRouteLearner,
    learn: bool,
    freeze_parent: bool = True,
    phase: str = "unspecified",
    trace: list[dict[str, Any]] | None = None,
) -> float:
    """R4 stream with the route learner replacing the always-open gate."""
    model.reset_dynamics(episode_id="r5-conditional-stream")
    shadow.reset_dynamics()
    ema_surprise = 0.0
    total_surprise = 0.0
    count = 0
    for symbol in (model.config.boundary_symbol, *data):
        diagnostics = shadow.diagnostics
        gate = _route_gate(
            route,
            data=data,
            ema_state={"surprise": ema_surprise},
            diagnostics=diagnostics,
        )
        shadow.set_gate(gate)
        step = model.observe(
            symbol,
            learn=learn,
            learn_fabric=False,
            learn_predictive_context=False,
            learn_predictive_readout=False,
            learn_adaptive_residual_bridge=learn,
            readout="predictive",
            use_memory=False,
            use_identity=False,
            _adaptive_residual_shadow=shadow,
            _learn_adaptive_residual_shadow=bool(learn),
            _adaptive_residual_shadow_freeze_parent=freeze_parent,
        )
        if step.prior_probability is not None:
            surprise_tick = -math.log(max(float(step.prior_probability), 1e-12))
            ema_surprise += 0.1 * (surprise_tick - ema_surprise)
            total_surprise += surprise_tick
            count += 1
            if learn:
                route.learn(surprise_tick)
        if trace is not None:
            trace.append(
                {
                    "phase": phase,
                    "symbol": int(symbol),
                    "route_gate": gate,
                    **shadow.diagnostics,
                }
            )
    return total_surprise / max(1, count)


def _r5_course_train(
    model: Taiji,
    *,
    shadow: AdaptiveResidualShadow,
    route: ConditionalRouteLearner,
    course: Any = DEFAULT_COURSE,
    freeze_parent: bool = False,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    _r5_stream(
        model,
        course.s_train,
        shadow=shadow,
        route=route,
        learn=True,
        freeze_parent=freeze_parent,
        phase="S",
        trace=trace,
    )
    for old_count, new_count in course.g_schedule:
        blended = course.g_old * old_count + course.g_new * new_count
        _r5_stream(
            model,
            blended,
            shadow=shadow,
            route=route,
            learn=True,
            freeze_parent=freeze_parent,
            phase="G",
            trace=trace,
        )


def run_cell(
    *,
    model_seed: int,
    course: R4Course = DEFAULT_COURSE,
) -> dict[str, Any]:
    """Run one (model_seed, course) cell: three arms plus lesion probes."""
    cell_started = time.perf_counter()
    r3_parent, pressure_parent, checkpoint_preflight = _prepared_parents(model_seed)
    if not checkpoint_preflight:
        raise RuntimeError(f"cell {model_seed} parent checkpoint preflight failed")

    arms: dict[str, Any] = {}

    # A: fixed-capacity unconditional bridge (always-open gate, no candidate).
    fixed = Taiji.from_checkpoint(copy.deepcopy(r3_parent))
    working_set_before = _process_working_set_bytes()
    arm_started = time.perf_counter()
    _course_train(fixed, learn_bridge=True, course=course)
    arms["fixed_capacity"] = {
        "scores": {
            "S": _score(fixed, course.s_holdout),
            "G": _score(fixed, course.g_holdout),
        },
        "resources": {
            "train_seconds": time.perf_counter() - arm_started,
            "working_set_bytes_after": _process_working_set_bytes(),
            "working_set_bytes_before": working_set_before,
        },
    }

    # B: fixed-large preregistered control (random birth, always-open gate,
    # same final parameter count as the conditional candidate).
    fixed_large_started = time.perf_counter()
    fixed_large = _growth_arm(
        pressure_parent,
        candidate=_random_candidate(
            Taiji.from_checkpoint(copy.deepcopy(pressure_parent)),
            pressure_parent,
            label="fixed-large",
        ),
        freeze_parent=False,
        course=course,
        label="fixed-large",
    )
    working_set_after_fixed_large = _process_working_set_bytes()
    arms["fixed_large"] = {
        "scores": fixed_large["scores"],
        "candidate_lesion_delta": fixed_large["candidate_lesion_delta"],
        "shadow_fresh_restore": fixed_large["shadow_fresh_restore"],
        "resources": {
            "train_seconds": time.perf_counter() - fixed_large_started,
        },
    }

    # C: conditional module with the dynamic route gate.  freeze_parent=False
    # aligns the learning boundary with the fixed-large arm.
    model = Taiji.from_checkpoint(copy.deepcopy(pressure_parent))
    candidate = model.adaptive_residual_growth_candidate
    if candidate is None:
        raise RuntimeError("pressure parent did not carry a growth candidate")
    shadow = AdaptiveResidualShadow.from_parent_bridge(
        model.config,
        model.adaptive_residual_bridge.to_payload(),
        candidate,
        birth_mode="pressure_mixture",
        device=model.device,
    )
    parent_substrate_digest = content_digest(_shadow_parent_payload(shadow))
    mature_context = content_digest(model.predictive_context.to_payload())
    mature_readout = content_digest(model.predictive_readout.to_payload())

    route = ConditionalRouteLearner(generator=torch.Generator().manual_seed(ROUTE_SEED))
    training_trace: list[dict[str, Any]] = []
    working_set_before = _process_working_set_bytes()
    arm_started = time.perf_counter()
    _r5_course_train(
        model,
        shadow=shadow,
        route=route,
        course=course,
        freeze_parent=False,
        trace=training_trace,
    )
    train_seconds = time.perf_counter() - arm_started

    route_gate_values = [float(item["route_gate"]) for item in training_trace]
    route_gate_std = (
        float(torch.tensor(route_gate_values).std(unbiased=False).item())
        if len(route_gate_values) >= 2
        else 0.0
    )

    trained_shadow_payload = shadow.to_payload()
    parent_substrate_unchanged = (
        content_digest(_shadow_parent_payload(shadow)) == parent_substrate_digest
    )
    owner_unchanged = (
        content_digest(model.predictive_context.to_payload()) == mature_context
        and content_digest(model.predictive_readout.to_payload()) == mature_readout
    )
    restored_shadow = AdaptiveResidualShadow.from_checkpoint(model.config, trained_shadow_payload)
    shadow_round_trip = content_digest(restored_shadow.to_payload()) == content_digest(
        trained_shadow_payload
    )

    route_payload = route.to_payload()
    restored_route = ConditionalRouteLearner()
    restored_route.from_payload(copy.deepcopy(route_payload))
    route_round_trip = content_digest(route_payload) == content_digest(restored_route.to_payload())

    # All conditional-arm scoring goes through the route-gated stream: the
    # R4 ``_score`` helper would re-open the gate every tick and erase the
    # route condition.
    conditional_scores = {
        "S": _r5_stream(
            model,
            course.s_holdout,
            shadow=shadow,
            route=route,
            learn=False,
            phase="S-holdout",
        ),
        "G": _r5_stream(
            model,
            course.g_holdout,
            shadow=shadow,
            route=route,
            learn=False,
            phase="G-holdout",
        ),
    }

    # Route lesion: the pre-registered causal probe.  A lesioned route
    # forces the gate to zero, so the candidate no longer injects anything.
    route.lesion()
    route_lesion_scores = {
        "S": _r5_stream(
            model,
            course.s_holdout,
            shadow=shadow,
            route=route,
            learn=False,
            phase="S-holdout-route-lesion",
        ),
        "G": _r5_stream(
            model,
            course.g_holdout,
            shadow=shadow,
            route=route,
            learn=False,
            phase="G-holdout-route-lesion",
        ),
    }
    route_lesion_delta = {
        phase: float(route_lesion_scores[phase] - conditional_scores[phase])
        for phase in conditional_scores
    }

    # Candidate lesion isolates the candidate unit under a live route.
    route.unlesion()
    restored_for_candidate_lesion = AdaptiveResidualShadow.from_checkpoint(
        model.config, trained_shadow_payload
    )
    restored_for_candidate_lesion.lesion_candidate()
    candidate_lesion_scores = {
        "S": _r5_stream(
            model,
            course.s_holdout,
            shadow=restored_for_candidate_lesion,
            route=route,
            learn=False,
            phase="S-holdout-candidate-lesion",
        ),
        "G": _r5_stream(
            model,
            course.g_holdout,
            shadow=restored_for_candidate_lesion,
            route=route,
            learn=False,
            phase="G-holdout-candidate-lesion",
        ),
    }
    candidate_lesion_delta = {
        phase: float(candidate_lesion_scores[phase] - conditional_scores[phase])
        for phase in conditional_scores
    }

    working_set_after = _process_working_set_bytes()
    checks = {
        "checkpoint_preflight": bool(checkpoint_preflight),
        # freeze_parent=False means the parent bridge legitimately learns, so
        # the substrate is expected to change (R4 formal semantics); the
        # mature F1 owners are the protected surface and must not move.
        "parent_substrate_learns_as_expected": parent_substrate_unchanged is False,
        "mature_owners_unchanged": bool(owner_unchanged),
        "shadow_round_trip": bool(shadow_round_trip),
        "route_round_trip": bool(route_round_trip),
        "route_gates_non_constant": route_gate_std > 1e-4,
        "route_gates_in_unit_range": all(0.0 <= v <= 1.0 for v in route_gate_values),
        "route_lesion_observable_g": route_lesion_delta["G"] > 0.0,
        "route_lesion_observable_s": route_lesion_delta["S"] > 0.0,
        "candidate_lesion_observable": (
            candidate_lesion_delta["G"] > 0.0 or candidate_lesion_delta["S"] > 0.0
        ),
        "resource_records_complete": bool(
            route.active_parameter_bytes() > 0 and len(training_trace) > 0
        ),
    }

    return {
        "arms": arms,
        "conditional_module": {
            "scores": conditional_scores,
            "route_lesion_scores": route_lesion_scores,
            "route_lesion_delta": route_lesion_delta,
            "candidate_lesion_scores": candidate_lesion_scores,
            "candidate_lesion_delta": candidate_lesion_delta,
        },
        "checks": checks,
        "technical_gate_all_passed": all(bool(v) for v in checks.values()),
        "resource_records": {
            "route_parameter_bytes": route.active_parameter_bytes(),
            "training_ticks": len(training_trace),
            "route_gate_mean": sum(route_gate_values) / max(1, len(route_gate_values)),
            "route_gate_std": route_gate_std,
            "conditional_train_seconds": train_seconds,
            "working_set_bytes_after_conditional": working_set_after,
            "working_set_bytes_after_fixed_capacity": arms["fixed_capacity"]["resources"][
                "working_set_bytes_after"
            ],
            "working_set_bytes_after_fixed_large": working_set_after_fixed_large,
        },
        "route_inputs": {
            "whitelist": [
                "content_bucket",
                "surprise_ema",
                "parent_residual_norm",
                "candidate_activity",
                "resource_scalar_constant_1.0_in_canary",
            ],
            "forbidden": ["evaluator task id", "phase label", "file name"],
        },
        "cell_elapsed_seconds": time.perf_counter() - cell_started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=71)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cell = run_cell(model_seed=args.model_seed)
    technical_gate_all_passed = cell["technical_gate_all_passed"]

    report = {
        "format": CANARY_FORMAT,
        "version": CANARY_VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if technical_gate_all_passed else "failed",
        "can_promote": False,
        "model_seed": int(args.model_seed),
        "course": {
            "label": DEFAULT_COURSE.label,
            "course_seed": DEFAULT_COURSE.course_seed,
        },
        "cell": cell,
        "checks": cell["checks"],
        "technical_gate_all_passed": technical_gate_all_passed,
        "resources": {"total_elapsed_seconds": time.perf_counter() - started},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "technical_gate_all_passed": technical_gate_all_passed,
                "failed_checks": [k for k, v in cell["checks"].items() if not v],
                "route_lesion_delta": cell["conditional_module"]["route_lesion_delta"],
                "candidate_lesion_delta": cell["conditional_module"]["candidate_lesion_delta"],
            },
            indent=2,
        )
    )
    return 0 if technical_gate_all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the bounded M4.V2.R4 shadow-growth canary.

This is a short CPU comparison, not an A8 promotion experiment.  It keeps a
single prepared R3 parent and the same S/G course across five arms.  Growth
arms use the native shadow projection and receive only the causal predictive
error through ``Taiji.observe``; mature F1 owners and the live parent bridge
remain frozen.  The report always keeps ``can_promote=false``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    AdaptiveResidualGrowthCandidate,
    AdaptiveResidualGrowthPolicy,
    AdaptiveResidualShadow,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest  # noqa: E402

CANARY_FORMAT = "taiji-m4v2-r4-shadow-canary-v1"
CANARY_VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r4_shadow_canary_20260909.json"

S_TRAIN = b"abba-caba-abba-caba"
S_HOLDOUT = b"caba-bbaa-caba"
G_OLD = b"abba-caba"
G_NEW = b"xyz-zyx-uvw"
G_HOLDOUT = b"zyx-uvw-xyz"
G_SCHEDULE = ((8, 2), (6, 4), (4, 6), (2, 8))


def _tiny_config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24,),
        synapse_fan_in=6,
        motor_fan_in=12,
        predictive_context_fan_in=6,
        memory_units=24,
        memory_fan_in=6,
        memory_meta_dim=16,
        memory_readout_fan_in=12,
        identity_organ_enabled=False,
        seed=71,
    )


def _growth_policy() -> AdaptiveResidualGrowthPolicy:
    return AdaptiveResidualGrowthPolicy(
        ema_rate=1.0,
        minimum_pressure=0.0,
        minimum_residual_error=0.0,
        minimum_fast_slow_conflict=0.0,
        minimum_activity_saturation=0.0,
        minimum_utility_gap=0.0,
        minimum_resource_state=0.0,
        required_pressure_steps=1,
        growth_resource_cost=1,
    )


def _observe_pressure(model: Taiji) -> None:
    model.reset_dynamics(episode_id="r4-shadow-pressure")
    for symbol in (97, 98):
        model.observe(
            symbol,
            learn=True,
            learn_fabric=False,
            learn_predictive_context=False,
            learn_predictive_readout=False,
            learn_adaptive_residual_bridge=True,
            readout="predictive",
        )


def _prepared_parents() -> tuple[dict[str, Any], dict[str, Any], bool]:
    model = Taiji(_tiny_config(), episode_id="r4-shadow-parent")
    model.enable_adaptive_residual_bridge(gate=1.0, residual_gain=1.0)
    r3_parent = model.checkpoint()
    r3_restored = Taiji.from_checkpoint(copy.deepcopy(r3_parent))
    checkpoint_preflight = content_digest(r3_restored.checkpoint()) == content_digest(r3_parent)

    model.enable_adaptive_residual_growth(policy=_growth_policy())
    _observe_pressure(model)
    decision = model.adaptive_residual_growth_decision
    if decision is None or not decision.should_propose:
        raise RuntimeError("R4 canary parent did not emit a native growth decision")
    model.propose_adaptive_residual_growth_candidate()
    pressure_parent = model.checkpoint()
    pressure_restored = Taiji.from_checkpoint(copy.deepcopy(pressure_parent))
    checkpoint_preflight = checkpoint_preflight and (
        content_digest(pressure_restored.checkpoint()) == content_digest(pressure_parent)
    )
    return r3_parent, pressure_parent, checkpoint_preflight


def _blended_train(old_count: int, new_count: int) -> bytes:
    return G_OLD * old_count + G_NEW * new_count


def _stream(
    model: Taiji,
    data: bytes,
    *,
    shadow: AdaptiveResidualShadow | None = None,
    learn: bool,
    learn_bridge: bool = False,
    trace: list[dict[str, Any]] | None = None,
) -> float:
    model.reset_dynamics(episode_id="r4-shadow-stream")
    if shadow is not None:
        shadow.reset_dynamics()
        shadow.set_gate(1.0)
    total_surprise = 0.0
    count = 0
    for symbol in (model.config.boundary_symbol, *data):
        step = model.observe(
            symbol,
            learn=learn,
            learn_fabric=False,
            learn_predictive_context=False,
            learn_predictive_readout=False,
            learn_adaptive_residual_bridge=learn_bridge,
            readout="predictive",
            use_memory=False,
            use_identity=False,
            _adaptive_residual_shadow=shadow,
            _learn_adaptive_residual_shadow=bool(learn and shadow is not None),
        )
        if shadow is not None and trace is not None:
            trace.append({"symbol": int(symbol), **shadow.diagnostics})
        if step.prior_probability is not None:
            total_surprise += -math.log(max(float(step.prior_probability), 1e-12))
            count += 1
    return total_surprise / max(1, count)


def _course_train(
    model: Taiji,
    *,
    shadow: AdaptiveResidualShadow | None = None,
    learn_bridge: bool = False,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    _stream(
        model,
        S_TRAIN,
        shadow=shadow,
        learn=True,
        learn_bridge=learn_bridge,
        trace=trace,
    )
    for old_count, new_count in G_SCHEDULE:
        _stream(
            model,
            _blended_train(old_count, new_count),
            shadow=shadow,
            learn=True,
            learn_bridge=learn_bridge,
            trace=trace,
        )


def _score(model: Taiji, data: bytes, *, shadow: AdaptiveResidualShadow | None = None) -> float:
    return _stream(model, data, shadow=shadow, learn=False)


def _counts(model: Taiji, shadow: AdaptiveResidualShadow | None) -> dict[str, int]:
    return {
        "mature_model_parameter_count": int(model.parameter_count()),
        "shadow_parameter_count": (
            0
            if shadow is None
            else int(sum(tensor.numel() for tensor in shadow.parameter_tensors()))
        ),
        "shadow_unit_count": 0 if shadow is None else int(shadow.unit_count),
        "shadow_edge_count": 0 if shadow is None else int(shadow.edge_count),
    }


def _random_candidate(
    model: Taiji,
    parent_checkpoint: Mapping[str, Any],
    *,
    label: str,
) -> AdaptiveResidualGrowthCandidate:
    bridge = model.adaptive_residual_bridge
    if bridge is None:
        raise RuntimeError("random growth baseline requires the R3 bridge")
    evidence_ids = (f"{label}:{content_digest(parent_checkpoint)}",)
    proposal = bridge.region.propose_unit_add(
        unit_id=f"{bridge.region.region_id}.u{bridge.unit_count}",
        evidence_ids=evidence_ids,
        source_region_id=bridge.region.input_source_id,
        parent_checkpoint_id=f"{label}-parent:{content_digest(parent_checkpoint)}",
        resource_cost=1,
    )
    parent_edge_count = bridge.edge_count
    incoming_edge_delta = bridge.region.incoming.row_fan_in
    recurrent_edge_delta = (
        0 if bridge.region.recurrent is None else bridge.region.recurrent.row_fan_in
    )
    edge_delta = incoming_edge_delta + recurrent_edge_delta
    return AdaptiveResidualGrowthCandidate.create(
        bridge_id=bridge.region.region_id,
        unit_id=str(dict(proposal.specification)["unit_id"]),
        parent_checkpoint_digest=content_digest(parent_checkpoint),
        source_checkpoint_digest=content_digest(parent_checkpoint),
        decision_digest=f"{label}-decision:{content_digest(parent_checkpoint)}",
        evidence_ids=evidence_ids,
        parent_unit_count=bridge.unit_count,
        proposed_unit_count=bridge.unit_count + 1,
        parent_edge_count=parent_edge_count,
        proposed_edge_count=parent_edge_count + edge_delta,
        topology_diff=(
            ("unit_count", 1),
            ("incoming_edges", incoming_edge_delta),
            ("recurrent_edges", recurrent_edge_delta),
            ("edge_count", edge_delta),
            ("parameter_scalars", edge_delta),
        ),
        resource_cost=1,
        structural_budget=int(model.config.development_structural_budget),
        proposal=proposal,
    )


def _growth_arm(
    parent_checkpoint: dict[str, Any],
    *,
    candidate: AdaptiveResidualGrowthCandidate | None = None,
    train_from_parent: bool = True,
    label: str,
) -> dict[str, Any]:
    model = Taiji.from_checkpoint(copy.deepcopy(parent_checkpoint))
    if candidate is None:
        candidate = model.adaptive_residual_growth_candidate
        if candidate is None:
            candidate = _random_candidate(model, parent_checkpoint, label=label)
    shadow = AdaptiveResidualShadow.from_parent_bridge(
        model.config,
        model.adaptive_residual_bridge.to_payload(),
        candidate,
        device=model.device,
    )
    bare_shadow = shadow.to_payload()
    bare_shadow_digest = content_digest(bare_shadow)
    mature_context = content_digest(model.predictive_context.to_payload())
    mature_readout = content_digest(model.predictive_readout.to_payload())
    training_trace: list[dict[str, Any]] = []
    if train_from_parent:
        _course_train(model, shadow=shadow, trace=training_trace)
    active_trace = [
        item for item in training_trace if item["candidate_activity"] > 1e-8
    ]
    residual_trace = [
        item for item in training_trace if item["candidate_residual_norm"] > 1e-8
    ]
    credit_trace = [
        item for item in training_trace if item["candidate_credit_norm"] > 1e-8
    ]
    residual_ratios = [
        item["candidate_residual_norm"] / item["parent_residual_norm"]
        for item in training_trace
        if item["parent_residual_norm"] > 1e-8
    ]
    scores = {
        "S": _score(model, S_HOLDOUT, shadow=shadow),
        "G": _score(model, G_HOLDOUT, shadow=shadow),
    }
    trained_shadow = shadow.to_payload()
    restored_shadow = AdaptiveResidualShadow.from_checkpoint(model.config, trained_shadow)
    fresh_restore = content_digest(restored_shadow.to_payload()) == content_digest(trained_shadow)
    lesioned = AdaptiveResidualShadow.from_checkpoint(model.config, trained_shadow)
    lesioned.lesion_candidate()
    lesion_scores = {
        "S": _score(model, S_HOLDOUT, shadow=lesioned),
        "G": _score(model, G_HOLDOUT, shadow=lesioned),
    }
    rollback = AdaptiveResidualShadow.from_checkpoint(model.config, bare_shadow)
    rollback_matches_bare = content_digest(rollback.to_payload()) == bare_shadow_digest
    return {
        "arm": label,
        "scores": scores,
        "candidate_lesion_scores": lesion_scores,
        "candidate_lesion_delta": {
            phase: float(lesion_scores[phase] - scores[phase]) for phase in scores
        },
        "candidate_digest": candidate.candidate_digest,
        "candidate_unit_id": candidate.unit_id,
        "candidate_activity": float(shadow.candidate_activity),
        "training_diagnostics": {
            "trace_digest": content_digest(training_trace),
            "ticks": len(training_trace),
            "active_ticks": len(active_trace),
            "residual_ticks": len(residual_trace),
            "credit_ticks": len(credit_trace),
            "max_activity": max(
                (item["candidate_activity"] for item in training_trace), default=0.0
            ),
            "max_eligibility_norm": max(
                (item["candidate_eligibility_norm"] for item in training_trace),
                default=0.0,
            ),
            "max_candidate_residual_norm": max(
                (item["candidate_residual_norm"] for item in training_trace), default=0.0
            ),
            "mean_candidate_residual_ratio": (
                sum(residual_ratios) / len(residual_ratios)
                if residual_ratios
                else 0.0
            ),
            "max_candidate_residual_ratio": max(residual_ratios, default=0.0),
            "max_candidate_credit_norm": max(
                (item["candidate_credit_norm"] for item in training_trace), default=0.0
            ),
            "max_projection_update_norm": max(
                (item["candidate_projection_update_norm"] for item in training_trace),
                default=0.0,
            ),
            "trace": training_trace,
        },
        "candidate_only_training_changed": content_digest(bare_shadow) != content_digest(trained_shadow),
        "shadow_fresh_restore": fresh_restore,
        "shadow_rollback_matches_bare": rollback_matches_bare,
        "mature_f1_owners_unchanged": (
            mature_context == content_digest(model.predictive_context.to_payload())
            and mature_readout == content_digest(model.predictive_readout.to_payload())
        ),
        "counts": _counts(model, shadow),
        "checkpoint_digest": content_digest(model.checkpoint()),
        "shadow_digest": content_digest(trained_shadow),
    }


def run_canary() -> dict[str, Any]:
    r3_parent, pressure_parent, checkpoint_preflight = _prepared_parents()
    frozen = Taiji.from_checkpoint(copy.deepcopy(r3_parent))
    frozen_scores = {
        "S": _score(frozen, S_HOLDOUT),
        "G": _score(frozen, G_HOLDOUT),
    }

    fixed_capacity = Taiji.from_checkpoint(copy.deepcopy(r3_parent))
    _course_train(fixed_capacity, learn_bridge=True)
    fixed_capacity_scores = {
        "S": _score(fixed_capacity, S_HOLDOUT),
        "G": _score(fixed_capacity, G_HOLDOUT),
    }

    pressure_model = Taiji.from_checkpoint(copy.deepcopy(pressure_parent))
    pressure_candidate = pressure_model.adaptive_residual_growth_candidate
    if pressure_candidate is None:
        raise RuntimeError("prepared pressure parent has no candidate artifact")
    pressure = _growth_arm(
        pressure_parent,
        candidate=pressure_candidate,
        label="pressure-driven-growth",
    )

    random_model = Taiji.from_checkpoint(copy.deepcopy(pressure_parent))
    random_candidate = _random_candidate(
        random_model,
        pressure_parent,
        label="random-growth",
    )
    random_growth = _growth_arm(
        pressure_parent,
        candidate=random_candidate,
        label="random-growth",
    )

    fixed_large_parent = Taiji.from_checkpoint(copy.deepcopy(r3_parent))
    fixed_large_candidate = _random_candidate(
        fixed_large_parent,
        r3_parent,
        label="fixed-large",
    )
    fixed_large = _growth_arm(
        r3_parent,
        candidate=fixed_large_candidate,
        label="fixed-large",
    )

    arms = {
        "frozen-parent": {"arm": "frozen-parent", "scores": frozen_scores, "counts": _counts(frozen, None)},
        "r3-fixed-capacity": {
            "arm": "r3-fixed-capacity",
            "scores": fixed_capacity_scores,
            "counts": _counts(fixed_capacity, None),
            "checkpoint_digest": content_digest(fixed_capacity.checkpoint()),
        },
        "pressure-driven-growth": pressure,
        "random-growth": random_growth,
        "fixed-large": fixed_large,
    }
    growth_counts = {
        name: tuple(value["counts"].items())
        for name, value in arms.items()
        if name in {"pressure-driven-growth", "random-growth", "fixed-large"}
    }
    matched_capacity = len(set(growth_counts.values())) == 1
    technical_gates = {
        "checkpoint_preflight": checkpoint_preflight,
        "pressure_candidate_present": pressure_candidate is not None,
        "pressure_shadow_training_changed": pressure["candidate_only_training_changed"],
        "pressure_shadow_fresh_restore": pressure["shadow_fresh_restore"],
        "pressure_shadow_rollback": pressure["shadow_rollback_matches_bare"],
        "pressure_mature_f1_owners_unchanged": pressure["mature_f1_owners_unchanged"],
        "random_shadow_fresh_restore": random_growth["shadow_fresh_restore"],
        "fixed_large_shadow_fresh_restore": fixed_large["shadow_fresh_restore"],
        "matched_growth_capacity": matched_capacity,
    }
    return {
        "format": CANARY_FORMAT,
        "version": CANARY_VERSION,
        "status": "passed" if all(technical_gates.values()) else "failed",
        "can_promote": False,
        "promotion_reason": "R4 is a short CPU shadow canary; holdout and matched-capacity results do not promote structure",
        "course": {
            "S_train_digest": content_digest(S_TRAIN),
            "S_holdout_digest": content_digest(S_HOLDOUT),
            "G_holdout_digest": content_digest(G_HOLDOUT),
            "G_schedule": [list(item) for item in G_SCHEDULE],
        },
        "parents": {
            "r3_checkpoint_digest": content_digest(r3_parent),
            "pressure_checkpoint_digest": content_digest(pressure_parent),
        },
        "arms": arms,
        "technical_gates": technical_gates,
        "interpretation": {
            "lower_mean_surprise_is_better": True,
            "candidate_lesion_positive_delta_means_candidate_helped": True,
            "matched_capacity_is_required_before_growth_claim": True,
            "short_course_is_not_A8": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_canary()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "format": report["format"],
                "status": report["status"],
                "can_promote": report["can_promote"],
                "technical_gates": report["technical_gates"],
                "arms": {
                    name: {
                        "scores": arm["scores"],
                        "candidate_lesion_delta": arm.get("candidate_lesion_delta"),
                        "training_diagnostics": {
                            key: value
                            for key, value in arm.get("training_diagnostics", {}).items()
                            if key != "trace"
                        },
                    }
                    for name, arm in report["arms"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

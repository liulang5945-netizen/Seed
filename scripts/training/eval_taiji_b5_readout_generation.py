"""Prototype generation-isolated predictive readout routing for B5.

This evaluator is deliberately not a production Taiji default.  It keeps a
protected old predictive readout and trains a copied active readout on phase B.
An input-only familiarity router chooses a generation from a compact centroid
and threshold learned from phase-A contexts; it never stores an answer table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_foundation_baseline import (  # noqa: E402
    _b5_phase_b_stream,
    _load_joint_child,
    _score_loaded_model,
)
from taiji import FoundationTrainingDataset, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402


def _owner_digests(model: Taiji) -> dict[str, str]:
    values = {
        "fabric": content_digest(model.fabric.to_payload()),
        "motor": content_digest(model.motor.to_payload()),
        "memory": content_digest(model.memory.to_payload()),
        "predictive_context": content_digest(model.predictive_context.to_payload()),
        "predictive_readout": content_digest(model.predictive_readout.to_payload()),
    }
    if model.identity_organ is not None:
        values["identity"] = content_digest(
            model.identity_organ.to_payload(parent_checkpoint_digest="b5-readout-generation")
        )
    return values


def _phase_a_context_profile(model: Taiji, data: bytes) -> dict[str, Any]:
    """Build an input-only familiarity profile from protected phase-A traffic."""

    checkpoint = model.checkpoint()
    model.reset_dynamics(episode_id="b5-readout-generation-profile")
    contexts: list[torch.Tensor] = []
    for symbol in model.sensor.symbols(data, include_boundary=True):
        model.observe(
            symbol,
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
        )
        contexts.append(model.snapshot().motor_context.detach().clone())
    model.restore(checkpoint)
    if not contexts:
        raise ValueError("phase-A profile needs at least one context")
    stack = torch.stack(contexts)
    centroid = stack.mean(dim=0)
    distances = torch.linalg.vector_norm(stack - centroid, dim=1)
    threshold = float(torch.quantile(distances, 0.99).item())
    return {
        "centroid": centroid,
        "threshold": threshold,
        "distance_max": float(distances.max().item()),
        "distance_mean": float(distances.mean().item()),
        "sample_count": len(contexts),
        "centroid_digest": content_digest(centroid.detach().cpu()),
    }


def _phase_a_surprise_profile(model: Taiji, data: bytes) -> dict[str, float | int]:
    """Profile old-generation surprise without using target bytes for routing."""

    checkpoint = model.checkpoint()
    model.reset_dynamics(episode_id="b5-readout-generation-surprise-profile")
    surprises: list[float] = []
    for symbol in model.sensor.symbols(data, include_boundary=True):
        step = model.observe(
            symbol,
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
        )
        if step.prior_probability is not None:
            surprises.append(-torch.log(torch.tensor(step.prior_probability)).item())
    model.restore(checkpoint)
    if not surprises:
        raise ValueError("phase-A surprise profile needs at least one prediction")
    values = torch.tensor(surprises)
    return {
        "threshold": float(torch.quantile(values, 0.99).item()),
        "surprise_mean": float(values.mean().item()),
        "surprise_max": float(values.max().item()),
        "sample_count": len(surprises),
    }


def _route_score(
    protected_model: Taiji,
    active_model: Taiji,
    data: bytes,
    *,
    centroid: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    """Score two generations while routing only from the prior input context."""

    protected_checkpoint = protected_model.checkpoint()
    active_checkpoint = active_model.checkpoint()
    protected_model.reset_dynamics(episode_id="b5-readout-generation-score-old")
    active_model.reset_dynamics(episode_id="b5-readout-generation-score-active")
    observations = 0
    correct = 0
    surprise_sum = 0.0
    active_routes = 0
    for symbol in protected_model.sensor.symbols(data, include_boundary=True):
        has_prior = protected_model.snapshot().last_symbol is not None
        route_active = False
        if has_prior:
            context = protected_model.snapshot().motor_context
            route_active = float(torch.linalg.vector_norm(context - centroid).item()) > threshold
        protected_step = protected_model.observe(
            symbol,
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
        )
        active_step = active_model.observe(
            symbol,
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
        )
        selected = active_step if route_active else protected_step
        active_routes += int(route_active)
        if selected.prior_prediction is not None:
            observations += 1
            correct += int(selected.prior_prediction == symbol)
            surprise_sum += float(selected.surprise or 0.0)
    protected_model.restore(protected_checkpoint)
    active_model.restore(active_checkpoint)
    return {
        "bpb": (surprise_sum / max(1, observations)) / torch.log(torch.tensor(2.0)).item(),
        "accuracy": correct / max(1, observations),
        "observations": observations,
        "active_routes": active_routes,
        "active_route_ratio": active_routes / max(1, observations),
    }


def _online_novelty_score(
    protected_model: Taiji,
    active_model: Taiji,
    data: bytes,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Route tick t+1 from old-generation surprise observed at tick t."""

    protected_checkpoint = protected_model.checkpoint()
    active_checkpoint = active_model.checkpoint()
    protected_model.reset_dynamics(episode_id="b5-readout-generation-online-old")
    active_model.reset_dynamics(episode_id="b5-readout-generation-online-active")
    observations = 0
    correct = 0
    surprise_sum = 0.0
    active_routes = 0
    route_active_next = False
    for symbol in protected_model.sensor.symbols(data, include_boundary=True):
        route_active = route_active_next
        protected_step = protected_model.observe(
            symbol,
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
        )
        active_step = active_model.observe(
            symbol,
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
        )
        selected = active_step if route_active else protected_step
        active_routes += int(route_active)
        if selected.prior_prediction is not None:
            observations += 1
            correct += int(selected.prior_prediction == symbol)
            surprise_sum += float(selected.surprise or 0.0)
        # This surprise uses the just-observed symbol only to set the route for
        # the next prediction.  It never changes which head scored this tick.
        if protected_step.prior_probability is None:
            route_active_next = False
        else:
            observed_surprise = -torch.log(torch.tensor(protected_step.prior_probability)).item()
            route_active_next = observed_surprise > threshold
    protected_model.restore(protected_checkpoint)
    active_model.restore(active_checkpoint)
    return {
        "bpb": (surprise_sum / max(1, observations)) / torch.log(torch.tensor(2.0)).item(),
        "accuracy": correct / max(1, observations),
        "observations": observations,
        "active_routes": active_routes,
        "active_route_ratio": active_routes / max(1, observations),
    }


def run_diagnosis(
    checkpoint: Path,
    *,
    seed: int,
    protected_corpus: Path,
    protected_partition_seed: int,
    phase_b_seed: int,
) -> dict[str, Any]:
    payload, source_model, _parent = _load_joint_child(checkpoint, expected_seed=seed)
    protected = FoundationTrainingDataset.from_jsonl(
        [protected_corpus],
        profile="foundation",
        partition_seed=protected_partition_seed,
    )
    phase_a_train = protected.train[:4_096]
    phase_a_holdout = protected.holdout[:200]
    phase_b_train = _b5_phase_b_stream(seed, length=4_096, offset=0)
    phase_b_holdout = _b5_phase_b_stream(seed, length=200, offset=1)
    retention = protected.retention[:200]

    profile_model = Taiji.from_checkpoint(source_model.checkpoint())
    profile = _phase_a_context_profile(profile_model, phase_a_train)
    centroid = profile["centroid"]
    threshold = float(profile["threshold"])

    protected_model = Taiji.from_checkpoint(source_model.checkpoint())
    active_model = Taiji.from_checkpoint(source_model.checkpoint())
    active_before = _owner_digests(active_model)
    active_model.learn_bytes(
        phase_b_train,
        epochs=1,
        learn_fabric=False,
        learn_predictive_context=False,
        learn_predictive_readout=True,
    )
    active_after = _owner_digests(active_model)
    if active_before["predictive_readout"] == active_after["predictive_readout"]:
        raise RuntimeError("active readout generation did not learn phase B")
    if active_before["predictive_context"] != active_after["predictive_context"]:
        raise RuntimeError("readout-generation prototype changed private context")

    before_protected = Taiji.from_checkpoint(source_model.checkpoint())
    before_active = Taiji.from_checkpoint(source_model.checkpoint())
    old_before = _route_score(
        before_protected,
        before_active,
        phase_a_holdout,
        centroid=centroid,
        threshold=threshold,
    )
    old_after = _route_score(
        protected_model,
        active_model,
        phase_a_holdout,
        centroid=centroid,
        threshold=threshold,
    )
    new_after = _route_score(
        protected_model,
        active_model,
        phase_b_holdout,
        centroid=centroid,
        threshold=threshold,
    )
    retention_after = _route_score(
        protected_model,
        active_model,
        retention,
        centroid=centroid,
        threshold=threshold,
    )
    surprise_profile_model = Taiji.from_checkpoint(source_model.checkpoint())
    surprise_profile = _phase_a_surprise_profile(surprise_profile_model, phase_a_train)
    surprise_threshold = float(surprise_profile["threshold"])
    online_before_protected = Taiji.from_checkpoint(source_model.checkpoint())
    online_before_active = Taiji.from_checkpoint(source_model.checkpoint())
    online_old_before = _online_novelty_score(
        online_before_protected,
        online_before_active,
        phase_a_holdout,
        threshold=surprise_threshold,
    )
    online_old_after = _online_novelty_score(
        protected_model,
        active_model,
        phase_a_holdout,
        threshold=surprise_threshold,
    )
    online_new_after = _online_novelty_score(
        protected_model,
        active_model,
        phase_b_holdout,
        threshold=surprise_threshold,
    )
    online_retention_after = _online_novelty_score(
        protected_model,
        active_model,
        retention,
        threshold=surprise_threshold,
    )
    active_only_new = _score_loaded_model(active_model, phase_b_holdout)
    protected_only_old = _score_loaded_model(protected_model, phase_a_holdout)
    return {
        "format": "taiji-foundation-b5-readout-generation-v1",
        "ability_id": "b5_continual_learning",
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": str(payload["checkpoint_digest"]),
        "protected_partition_seed": int(protected_partition_seed),
        "phase_b_seed": int(phase_b_seed),
        "generation_contract": {
            "old_generation": "protected_readout_v1",
            "active_generation": "phase_b_readout_v1",
            "old_generation_mutable": False,
            "active_generation_learns_readout_only": True,
            "answer_table": False,
            "unconditional_average": False,
        },
        "router": {
            "kind": "phase_a_context_familiarity_centroid_v1",
            "centroid_digest": profile["centroid_digest"],
            "threshold": threshold,
            "phase_a_profile_samples": profile["sample_count"],
            "phase_a_distance_mean": profile["distance_mean"],
            "phase_a_distance_max": profile["distance_max"],
        },
        "online_novelty_router": {
            "kind": "one_step_lagged_old_prediction_surprise_v1",
            "phase_a_threshold": surprise_threshold,
            "phase_a_profile_samples": surprise_profile["sample_count"],
            "phase_a_surprise_mean": surprise_profile["surprise_mean"],
            "phase_a_surprise_max": surprise_profile["surprise_max"],
            "uses_current_target_for_current_route": False,
        },
        "sample_counts": {
            "phase_a_train": len(phase_a_train),
            "phase_a_holdout": len(phase_a_holdout),
            "phase_b_train": len(phase_b_train),
            "phase_b_holdout": len(phase_b_holdout),
            "retention": len(retention),
        },
        "metrics": {
            "old_before": old_before,
            "old_after": old_after,
            "new_after": new_after,
            "retention_after": retention_after,
            "backward_transfer": old_before["bpb"] - old_after["bpb"],
            "active_only_new_bpb": active_only_new,
            "protected_only_old_bpb": protected_only_old,
        },
        "online_novelty_metrics": {
            "old_before": online_old_before,
            "old_after": online_old_after,
            "new_after": online_new_after,
            "retention_after": online_retention_after,
            "backward_transfer": online_old_before["bpb"] - online_old_after["bpb"],
        },
        "owner_audit": {
            "protected_generation": _owner_digests(protected_model),
            "active_before": active_before,
            "active_after": active_after,
            "active_owner_changes": {
                key: active_before[key] != active_after[key] for key in active_before
            },
            "shared_owners_preserved": all(
                active_before[key] == active_after[key]
                for key in ("fabric", "motor", "memory", "identity")
                if key in active_before
            ),
            "private_context_preserved": (
                active_before["predictive_context"] == active_after["predictive_context"]
            ),
            "score_checkpoint_read_only": True,
        },
        "round_trip": {
            "source_checkpoint_load_verified": True,
            "active_generation_is_evaluator_owned": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--protected-corpus", type=Path, required=True)
    parser.add_argument("--protected-partition-seed", type=int, required=True)
    parser.add_argument("--phase-b-seed", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = run_diagnosis(
        args.checkpoint,
        seed=args.seed,
        protected_corpus=args.protected_corpus,
        protected_partition_seed=args.protected_partition_seed,
        phase_b_seed=args.phase_b_seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result["report_written"] = True
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    json.loads(args.report.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report": str(args.report),
                "backward_transfer": result["metrics"]["backward_transfer"],
                "old_after_bpb": result["metrics"]["old_after"]["bpb"],
                "new_after_bpb": result["metrics"]["new_after"]["bpb"],
                "retention_after_bpb": result["metrics"]["retention_after"]["bpb"],
                "active_route_ratios": {
                    label: result["metrics"][label]["active_route_ratio"]
                    for label in ("old_after", "new_after", "retention_after")
                },
                "online_novelty": {
                    "backward_transfer": result["online_novelty_metrics"]["backward_transfer"],
                    "new_after_bpb": result["online_novelty_metrics"]["new_after"]["bpb"],
                    "retention_after_bpb": result["online_novelty_metrics"]["retention_after"][
                        "bpb"
                    ],
                    "active_route_ratios": {
                        label: result["online_novelty_metrics"][label]["active_route_ratio"]
                        for label in ("old_after", "new_after", "retention_after")
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["report_written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

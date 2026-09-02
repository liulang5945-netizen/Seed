"""Measure cue-population collisions behind the cue-selective B5 prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import CueBindingBank, Taiji, TaijiConfig  # noqa: E402
from taiji.sparse import SparseSynapses, bound_norm  # noqa: E402

FORMAT = "taiji-native-b5-cue-collision-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_b12_binding_diagnostic_canary.json"


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator.item()) <= 1e-8:
        return 0.0
    return float(torch.dot(left, right).item() / denominator.item())


def _write_episode(
    model: Taiji,
    cue: int,
    action: int,
    outcome: int,
    episode_id: str,
) -> torch.Tensor:
    model.reset_dynamics(episode_id=episode_id)
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
    model.observe(cue, learn=False, learn_motor=False)
    context = model.fabric.cortical_context(model.snapshot().regions)
    cue_pattern = model.memory._cue_pattern(context, model.snapshot().memory.threshold)
    model.act((action,), sample=False)
    model.settle_action(1.0, learn=False, learn_memory=True)
    model.observe(outcome, learn=False, learn_motor=False)
    return cue_pattern.detach().clone()


def _query_pattern(model: Taiji, cue: int, query_id: str) -> torch.Tensor:
    model.reset_dynamics(episode_id=query_id)
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
    model.observe(cue, learn=False, learn_motor=False)
    context = model.fabric.cortical_context(model.snapshot().regions)
    return model.memory._cue_pattern(context, model.snapshot().memory.threshold).detach().clone()


def _pairwise(values: list[torch.Tensor]) -> list[float]:
    return [
        _cosine(values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]


def _competitive_binding(
    pattern: torch.Tensor,
    projection: SparseSynapses,
    *,
    top_k: int,
) -> torch.Tensor:
    drive = torch.relu(projection.forward(pattern))
    values, indices = torch.topk(drive, k=top_k)
    code = torch.zeros_like(drive)
    code.scatter_(0, indices, values)
    return bound_norm(code, 1.0)


def _seed_record(seed: int, train_count: int, holdout_count: int) -> dict[str, object]:
    base = _memory_config(seed).to_dict()
    base["memory_action_decoder"] = "cue_selective"
    config = TaijiConfig.from_dict(base)
    corpus = build_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
        retention_count=holdout_count,
    )
    model = Taiji(config, episode_id=f"m1-b11-collision-{seed}")
    phase_a_patterns = {
        episode.cue: _write_episode(
            model,
            episode.cue,
            episode.action,
            episode.outcome,
            episode.memory_id,
        )
        for episode in corpus.phase_a_train
    }
    phase_b_patterns = {
        episode.cue: _write_episode(
            model,
            episode.cue,
            episode.action,
            episode.outcome,
            episode.memory_id,
        )
        for episode in corpus.phase_b_train
    }
    phase_a_query_cosines = [
        _cosine(
            phase_a_patterns[query.cue],
            _query_pattern(model, query.cue, query.query_id),
        )
        for query in corpus.phase_a_holdout
    ]
    phase_b_query_cosines = [
        _cosine(
            phase_b_patterns[query.cue],
            _query_pattern(model, query.cue, query.query_id),
        )
        for query in corpus.phase_b_holdout
    ]
    binding = CueBindingBank(
        config.memory_units,
        config.memory_units,
        match_threshold=0.85,
        update_rate=0.10,
        device=model.device,
    )
    phase_a_slots = {
        cue: binding.route(pattern, learn=True).slot_index
        for cue, pattern in phase_a_patterns.items()
    }
    phase_b_slots = {
        cue: binding.route(pattern, learn=True).slot_index
        for cue, pattern in phase_b_patterns.items()
    }
    restored_binding = CueBindingBank(
        config.memory_units,
        config.memory_units,
        match_threshold=0.85,
        update_rate=0.10,
        device=model.device,
    )
    restored_binding.load_payload(binding.to_payload())
    query_slot_mismatches = 0
    for query in (*corpus.phase_a_holdout, *corpus.phase_b_holdout):
        routed = restored_binding.route(
            _query_pattern(model, query.cue, query.query_id),
            learn=False,
        )
        expected_slot = phase_a_slots.get(query.cue, phase_b_slots.get(query.cue))
        query_slot_mismatches += int(routed.slot_index != expected_slot)
    occupied_slots = set(slot for slot in (*phase_a_slots.values(), *phase_b_slots.values()) if slot is not None)
    cross_phase_collisions = len(set(phase_a_slots.values()) & set(phase_b_slots.values()))
    binding_generator = torch.Generator(device="cpu")
    binding_generator.manual_seed(int(seed) + 100_003)
    binding_projection = SparseSynapses(
        config.memory_units,
        config.memory_units,
        config.memory_fan_in,
        generator=binding_generator,
        init_scale=config.weight_init_scale,
        max_weight_norm=config.max_weight_norm,
        device=model.device,
        allow_self=False,
    )
    binding_top_k = max(1, min(config.memory_units, round(config.target_activity * config.memory_units)))
    phase_a_binding = [
        _competitive_binding(pattern, binding_projection, top_k=binding_top_k)
        for pattern in phase_a_patterns.values()
    ]
    phase_b_binding = [
        _competitive_binding(pattern, binding_projection, top_k=binding_top_k)
        for pattern in phase_b_patterns.values()
    ]
    action_support = {}
    action_support = {}
    for action in sorted({episode.action for episode in (*corpus.phase_a_train, *corpus.phase_b_train)}):
        support = set(int(value) for value in model.memory.local_action_readout.pre_index[action])
        action_support[str(action)] = {
            "fan_in": len(support),
            "phase_a_active_support_min": min(
                int(torch.count_nonzero(pattern[list(support)]).item())
                for pattern in phase_a_patterns.values()
            ),
            "phase_b_active_support_min": min(
                int(torch.count_nonzero(pattern[list(support)]).item())
                for pattern in phase_b_patterns.values()
            ),
        }
    within_a = _pairwise(list(phase_a_patterns.values()))
    within_b = _pairwise(list(phase_b_patterns.values()))
    cross = [
        _cosine(left, right)
        for left in phase_a_patterns.values()
        for right in phase_b_patterns.values()
    ]
    return {
        "seed": seed,
        "phase_a_pairwise_cosine": {
            "max": max(within_a),
            "mean": sum(within_a) / len(within_a),
        },
        "phase_b_pairwise_cosine": {
            "max": max(within_b),
            "mean": sum(within_b) / len(within_b),
        },
        "cross_phase_pairwise_cosine": {
            "max": max(cross),
            "mean": sum(cross) / len(cross),
        },
        "write_to_query_cosine": {
            "phase_a_min": min(phase_a_query_cosines),
            "phase_a_mean": sum(phase_a_query_cosines) / len(phase_a_query_cosines),
            "phase_b_min": min(phase_b_query_cosines),
            "phase_b_mean": sum(phase_b_query_cosines) / len(phase_b_query_cosines),
        },
        "competitive_binding": {
            "units": config.memory_units,
            "top_k": binding_top_k,
            "phase_a_pairwise_cosine": {
                "max": max(_pairwise(phase_a_binding)),
                "mean": sum(_pairwise(phase_a_binding)) / len(_pairwise(phase_a_binding)),
            },
            "phase_b_pairwise_cosine": {
                "max": max(_pairwise(phase_b_binding)),
                "mean": sum(_pairwise(phase_b_binding)) / len(_pairwise(phase_b_binding)),
            },
            "cross_phase_pairwise_cosine": {
                "max": max(
                    _cosine(left, right)
                    for left in phase_a_binding
                    for right in phase_b_binding
                ),
                "mean": sum(
                    _cosine(left, right)
                    for left in phase_a_binding
                    for right in phase_b_binding
                )
                / (len(phase_a_binding) * len(phase_b_binding)),
            },
        },
        "slot_binding": {
            "capacity": binding.capacity,
            "match_threshold": binding.match_threshold,
            "occupied_count": binding.occupied_count,
            "occupied_slot_count_from_corpus": len(occupied_slots),
            "phase_a_slots": phase_a_slots,
            "phase_b_slots": phase_b_slots,
            "cross_phase_collisions": cross_phase_collisions,
            "query_slot_mismatches_after_checkpoint_roundtrip": query_slot_mismatches,
            "allocation_count": binding.allocation_count,
            "match_count": binding.match_count,
            "replacement_count": binding.replacement_count,
        },
        "action_support": action_support,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    records = [
        _seed_record(int(seed), args.train_count, args.holdout_count) for seed in args.seeds
    ]
    result = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "can_promote": False,
        "decoder": "cue_selective",
        "train_count": args.train_count,
        "holdout_count": args.holdout_count,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(args.report)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

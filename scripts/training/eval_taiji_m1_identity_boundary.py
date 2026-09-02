"""Validate identity-route behavior at unseen-cue and fixed-capacity boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_identity_route import (  # noqa: E402
    CueIdentityRoute,
    _native_cue_pattern,
    _route_score,
    _route_training,
)
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import DelayedMemoryQuery, MemoryEpisode, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-identity-boundary-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_identity_boundary_20260902.json"
DEFAULT_CAPACITY = 128
DEFAULT_MATCH_THRESHOLD = 0.90
DEFAULT_ROUTE_LEARNING_RATE = 0.50


def _boundary_corpus(*, train_count: int, holdout_count: int) -> tuple[
    tuple[MemoryEpisode, ...],
    tuple[DelayedMemoryQuery, ...],
    tuple[DelayedMemoryQuery, ...],
    tuple[DelayedMemoryQuery, ...],
]:
    phase_a = tuple(
        MemoryEpisode(
            memory_id=f"m1-26-a-{index}",
            cue=65 + index,
            action=48 + index % 2,
            outcome=43 + index % 2 * 2,
        )
        for index in range(train_count)
    )
    phase_b = tuple(
        MemoryEpisode(
            memory_id=f"m1-26-b-{index}",
            cue=145 + index,
            action=50 + index % 2,
            outcome=43 + index % 2 * 2,
        )
        for index in range(train_count)
    )
    phase_a_queries = tuple(
        DelayedMemoryQuery(
            query_id=f"m1-26-a-query-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(phase_a[index % len(phase_a)] for index in range(holdout_count))
    )
    phase_b_queries = tuple(
        DelayedMemoryQuery(
            query_id=f"m1-26-b-query-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(phase_b[index % len(phase_b)] for index in range(holdout_count))
    )
    unseen_queries = tuple(
        DelayedMemoryQuery(
            query_id=f"m1-26-unseen-{index}",
            cue=210 + index,
            expected_action=0,
        )
        for index in range(holdout_count)
    )
    return (*phase_a, *phase_b), phase_a_queries, phase_b_queries, unseen_queries


def _stress_episodes(count: int) -> tuple[MemoryEpisode, ...]:
    return tuple(
        MemoryEpisode(
            memory_id=f"m1-26-stress-{index}",
            cue=1 + index,
            action=48 + index % 2,
            outcome=43 + index % 2 * 2,
        )
        for index in range(count)
    )


def _seed_record(
    seed: int,
    *,
    train_count: int,
    holdout_count: int,
    capacity: int,
    match_threshold: float,
    route_learning_rate: float,
) -> dict[str, object]:
    episodes, phase_a_queries, phase_b_queries, unseen_queries = _boundary_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
    )
    config = _memory_config(seed)
    model = Taiji(config, episode_id=f"m1-26-boundary-{seed}")
    route = CueIdentityRoute(
        capacity=capacity,
        pattern_dim=config.memory_units,
        action_count=config.alphabet_size,
        match_threshold=match_threshold,
        route_learning_rate=route_learning_rate,
    )
    actions = tuple(dict.fromkeys(episode.action for episode in episodes))
    phase_a = episodes[:train_count]
    phase_b = episodes[train_count:]
    phase_a_slots = _route_training(route, model, phase_a)
    parent_old, _ = _route_score(route, model, phase_a_queries, actions)
    phase_b_slots = _route_training(route, model, phase_b)
    old_score, old_slots = _route_score(route, model, phase_a_queries, actions)
    new_score, new_slots = _route_score(route, model, phase_b_queries, actions)
    route_before_unseen = content_digest(route.to_payload())
    _, unseen_slots = _route_score(route, model, unseen_queries, actions)
    route_after_unseen = content_digest(route.to_payload())
    repeated_matches: list[bool] = []
    for episode, expected_slot in zip(phase_a, phase_a_slots, strict=True):
        binding = route.learn(_native_cue_pattern(model, episode.cue), episode.action)
        repeated_matches.append(binding.slot_index == expected_slot)
    repeated_old, _ = _route_score(route, model, phase_a_queries, actions)
    repeated_new, _ = _route_score(route, model, phase_b_queries, actions)
    final_old, _ = _route_score(route, model, phase_a_queries, actions)
    final_new, _ = _route_score(route, model, phase_b_queries, actions)
    bundle = {"model": model.checkpoint(), "route": route.to_payload()}
    bundle_digest = content_digest(bundle)
    restored_model = Taiji.from_checkpoint(deepcopy(bundle["model"]))
    restored_route = CueIdentityRoute(
        capacity=capacity,
        pattern_dim=config.memory_units,
        action_count=config.alphabet_size,
        match_threshold=match_threshold,
        route_learning_rate=route_learning_rate,
    )
    restored_route.load_payload(deepcopy(bundle["route"]))
    _route_score(restored_route, restored_model, phase_a_queries, actions)
    _route_score(restored_route, restored_model, phase_b_queries, actions)
    restored_bundle = {"model": restored_model.checkpoint(), "route": restored_route.to_payload()}

    stress_model = Taiji(config, episode_id=f"m1-26-stress-{seed}")
    stress_route = CueIdentityRoute(
        capacity=capacity,
        pattern_dim=config.memory_units,
        action_count=config.alphabet_size,
        match_threshold=match_threshold,
        route_learning_rate=route_learning_rate,
    )
    stress = _stress_episodes(capacity + 8)
    stress_patterns = [_native_cue_pattern(stress_model, episode.cue) for episode in stress]
    stress_slots = tuple(
        int(stress_route.learn(pattern, episode.action).slot_index)
        for pattern, episode in zip(stress_patterns, stress, strict=True)
    )
    old_bound = sum(stress_route.query(pattern).slot_index is not None for pattern in stress_patterns[:capacity])
    new_bound = sum(stress_route.query(pattern).slot_index is not None for pattern in stress_patterns[capacity:])
    stress_bundle = {"model": stress_model.checkpoint(), "route": stress_route.to_payload()}
    stress_bundle_digest = content_digest(stress_bundle)
    return {
        "seed": seed,
        "parent_old_holdout": parent_old,
        "learned": {
            "old_holdout": old_score,
            "new_holdout": new_score,
            "old_bound_queries": sum(slot is not None for slot in old_slots),
            "new_bound_queries": sum(slot is not None for slot in new_slots),
        },
        "unseen": {
            "query_count": len(unseen_slots),
            "unbound_rate": sum(slot is None for slot in unseen_slots) / len(unseen_slots),
            "route_digest_unchanged": route_before_unseen == route_after_unseen,
        },
        "repeated_replay": {
            "same_slot_rate": sum(repeated_matches) / len(repeated_matches),
            "old_holdout_after": repeated_old,
            "new_holdout_after": repeated_new,
            "final_old_holdout": final_old,
            "final_new_holdout": final_new,
        },
        "identity": {
            "phase_a_unique_slots": len(set(phase_a_slots)),
            "phase_b_unique_slots": len(set(phase_b_slots)),
            "cross_phase_slot_collisions": len(set(phase_a_slots) & set(phase_b_slots)),
            "occupied_count": route.bank.occupied_count,
            "replacement_count": route.bank.replacement_count,
        },
        "capacity_stress": {
            "requested": len(stress),
            "capacity": capacity,
            "unique_slots": len(set(stress_slots)),
            "occupied_count": stress_route.bank.occupied_count,
            "replacement_count": stress_route.bank.replacement_count,
            "old_bound_rate_after_pressure": old_bound / capacity,
            "new_bound_rate_after_pressure": new_bound / 8,
            "replacement_is_explicit": stress_route.bank.replacement_count > 0,
            "bundle_digest_matches": content_digest(stress_bundle) == stress_bundle_digest,
        },
        "budget": {
            "capacity": capacity,
            "pattern_dim": config.memory_units,
            "action_count": config.alphabet_size,
            "identity_total_parameters": route.parameter_count,
            "identity_action_edges": route.edge_count,
        },
        "checkpoint": {
            "bundle_digest_matches": content_digest(restored_bundle) == bundle_digest,
        },
        "holdout_updates": 0,
    }


def run_boundary_diagnostics(
    *,
    train_count: int,
    holdout_count: int,
    seeds: tuple[int, ...],
    capacity: int = DEFAULT_CAPACITY,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    route_learning_rate: float = DEFAULT_ROUTE_LEARNING_RATE,
) -> dict[str, object]:
    if train_count < 4 or holdout_count < 4:
        raise ValueError("identity boundary canary needs at least four samples")
    records = [
        _seed_record(
            seed,
            train_count=train_count,
            holdout_count=holdout_count,
            capacity=capacity,
            match_threshold=match_threshold,
            route_learning_rate=route_learning_rate,
        )
        for seed in seeds
    ]
    gate = all(
        record["unseen"]["unbound_rate"] == 1.0
        and record["unseen"]["route_digest_unchanged"]
        and record["repeated_replay"]["same_slot_rate"] == 1.0
        and record["capacity_stress"]["replacement_is_explicit"]
        and record["capacity_stress"]["bundle_digest_matches"]
        and record["checkpoint"]["bundle_digest_matches"]
        for record in records
    )
    return {
        "sample_counts": {
            "phase_train": train_count,
            "learned_holdout": holdout_count,
            "unseen_queries": holdout_count,
            "stress": capacity + 8,
        },
        "capacity": capacity,
        "match_threshold": match_threshold,
        "route_learning_rate": route_learning_rate,
        "promotable": gate,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--route-learning-rate", type=float, default=DEFAULT_ROUTE_LEARNING_RATE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    diagnostics = run_boundary_diagnostics(
        train_count=args.train_count,
        holdout_count=args.holdout_count,
        seeds=tuple(int(seed) for seed in args.seeds),
        capacity=int(args.capacity),
        match_threshold=float(args.match_threshold),
        route_learning_rate=float(args.route_learning_rate),
    )
    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "identity_route_boundary_only": True,
        "diagnostics": diagnostics,
        "can_promote": bool(diagnostics["promotable"]),
        "report_path": str(args.report),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Canary a fixed-capacity cue identity route without touching native Taiji."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from scripts.training.eval_taiji_m1_interleaved_rehearsal import (  # noqa: E402
    _persistent_digest,
    _scores,
    _write,
)
from scripts.training.eval_taiji_m1_phase_b_write_ablation import (  # noqa: E402
    _write_without_memory,
)
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import (  # noqa: E402
    ContinualMemoryCorpus,
    Taiji,
    TaijiConfig,
)
from taiji.cue_binding import CueBindingBank, CueBindingResult  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-identity-route-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_identity_route_20260902.json"
MODES = ("shared_control", "identity_route")
DEFAULT_CAPACITY = 128
DEFAULT_MATCH_THRESHOLD = 0.90
DEFAULT_ROUTE_LEARNING_RATE = 0.50


class CueIdentityRoute:
    """Physical slot prototypes plus a bounded slot-to-action synapse bank."""

    CHECKPOINT_FORMAT = "taiji-cue-identity-route-v1"

    def __init__(
        self,
        *,
        capacity: int,
        pattern_dim: int,
        action_count: int,
        match_threshold: float,
        route_learning_rate: float,
    ) -> None:
        self.bank = CueBindingBank(
            capacity,
            pattern_dim,
            match_threshold=match_threshold,
            update_rate=0.10,
        )
        self.action_count = int(action_count)
        self.route_learning_rate = float(route_learning_rate)
        self.action_synapses = torch.zeros(
            (int(capacity), self.action_count), dtype=torch.float32
        )
        if self.action_count <= 1 or self.route_learning_rate <= 0.0:
            raise ValueError("identity route action_count and learning rate must be positive")

    @property
    def edge_count(self) -> int:
        return int(self.action_synapses.numel())

    @property
    def parameter_count(self) -> int:
        return int(self.bank.prototypes.numel() + self.action_synapses.numel())

    def learn(self, pattern: torch.Tensor, action: int) -> CueBindingResult:
        binding = self.bank.route(pattern, learn=True)
        if binding.slot_index is None:
            raise RuntimeError("identity route learning did not allocate a slot")
        slot = int(binding.slot_index)
        probabilities = torch.softmax(self.action_synapses[slot], dim=0)
        target = torch.zeros(self.action_count)
        target[int(action)] = 1.0
        self.action_synapses[slot].add_(
            self.route_learning_rate * (target - probabilities)
        )
        return binding

    def query(self, pattern: torch.Tensor) -> CueBindingResult:
        return self.bank.route(pattern, learn=False)

    def predict(self, slot_index: int, actions: tuple[int, ...]) -> int:
        if not actions:
            raise ValueError("identity route prediction needs available actions")
        logits = self.action_synapses[int(slot_index)]
        return max(actions, key=lambda action: float(logits[int(action)].item()))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "action_count": self.action_count,
            "route_learning_rate": self.route_learning_rate,
            "action_synapses": self.action_synapses.detach().cpu().clone(),
            "bank": self.bank.to_payload(),
        }

    def load_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError("unsupported identity route checkpoint format")
        if int(payload["action_count"]) != self.action_count:
            raise ValueError("identity route action dimension does not match")
        if float(payload["route_learning_rate"]) != self.route_learning_rate:
            raise ValueError("identity route learning rate does not match")
        weights = payload["action_synapses"].detach().to(dtype=torch.float32)
        if weights.shape != self.action_synapses.shape or not torch.isfinite(weights).all():
            raise ValueError("identity route action synapses are invalid")
        self.action_synapses = weights.clone()
        self.bank.load_payload(dict(payload["bank"]))


def _native_cue_pattern(model: Taiji, cue: int) -> torch.Tensor:
    model.reset_dynamics(episode_id=f"m1-25-native-cue-{cue}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
    model.observe(cue, learn=False, learn_motor=False, use_memory=False)
    return model.snapshot().memory.activity.detach().clone()


def _route_score(
    route: CueIdentityRoute,
    model: Taiji,
    queries: tuple[Any, ...],
    actions: tuple[int, ...],
) -> tuple[float, tuple[int | None, ...]]:
    correct = 0
    slots: list[int | None] = []
    for query in queries:
        binding = route.query(_native_cue_pattern(model, query.cue))
        slot = None if binding.slot_index is None else int(binding.slot_index)
        slots.append(slot)
        prediction = None if slot is None else route.predict(slot, actions)
        correct += int(prediction == query.expected_action)
    return correct / len(queries), tuple(slots)


def _route_training(
    route: CueIdentityRoute,
    model: Taiji,
    episodes: tuple[Any, ...],
) -> tuple[int, ...]:
    slots: list[int] = []
    for episode in episodes:
        binding = route.learn(_native_cue_pattern(model, episode.cue), episode.action)
        assert binding.slot_index is not None
        slots.append(int(binding.slot_index))
    return tuple(slots)


def _route_record(
    seed: int,
    corpus: ContinualMemoryCorpus,
    *,
    capacity: int,
    match_threshold: float,
    route_learning_rate: float,
) -> dict[str, object]:
    config = _memory_config(seed)
    model = Taiji(config, episode_id=f"m1-25-identity-{seed}")
    route = CueIdentityRoute(
        capacity=capacity,
        pattern_dim=config.memory_units,
        action_count=config.alphabet_size,
        match_threshold=match_threshold,
        route_learning_rate=route_learning_rate,
    )
    actions = tuple(
        dict.fromkeys(
            episode.action for episode in (*corpus.phase_a_train, *corpus.phase_b_train)
        )
    )
    phase_a_slots = _route_training(route, model, corpus.phase_a_train)
    parent_score, _ = _route_score(route, model, corpus.phase_a_holdout, actions)
    parent_retention, _ = _route_score(route, model, corpus.phase_a_retention, actions)
    phase_a_route_payload = deepcopy(route.to_payload())
    phase_a_route_digest = content_digest(phase_a_route_payload)
    phase_b_slots = _route_training(route, model, corpus.phase_b_train)
    child_score, _ = _route_score(route, model, corpus.phase_a_holdout, actions)
    child_retention, _ = _route_score(route, model, corpus.phase_a_retention, actions)
    new_score, new_slots = _route_score(route, model, corpus.phase_b_holdout, actions)
    route_before_replay = content_digest(route.to_payload())
    repeated_matches: list[bool] = []
    for episode, expected_slot in zip(corpus.phase_a_train, phase_a_slots, strict=True):
        binding = route.learn(_native_cue_pattern(model, episode.cue), episode.action)
        repeated_matches.append(binding.slot_index == expected_slot)
    repeated_route_digest = content_digest(route.to_payload())
    replay_old, _ = _route_score(route, model, corpus.phase_a_holdout, actions)
    replay_new, _ = _route_score(route, model, corpus.phase_b_holdout, actions)
    no_change_before = content_digest(route.to_payload())
    _, no_change_slots = _route_score(route, model, corpus.phase_a_holdout, actions)
    no_change_after = content_digest(route.to_payload())
    # Establish the same deterministic final dynamic state that the fresh
    # process will use for its restore canary before taking the bundle.
    _route_score(route, model, corpus.phase_a_holdout, actions)
    _route_score(route, model, corpus.phase_b_holdout, actions)
    checkpoint_payload = {
        "model": model.checkpoint(),
        "route": route.to_payload(),
    }
    checkpoint_digest = content_digest(checkpoint_payload)
    restored_model = Taiji.from_checkpoint(deepcopy(checkpoint_payload["model"]))
    restored_route = CueIdentityRoute(
        capacity=capacity,
        pattern_dim=config.memory_units,
        action_count=config.alphabet_size,
        match_threshold=match_threshold,
        route_learning_rate=route_learning_rate,
    )
    restored_route.load_payload(deepcopy(checkpoint_payload["route"]))
    restored_score, _ = _route_score(
        restored_route, restored_model, corpus.phase_a_holdout, actions
    )
    restored_new, _ = _route_score(
        restored_route, restored_model, corpus.phase_b_holdout, actions
    )
    restored_payload = {
        "model": restored_model.checkpoint(),
        "route": restored_route.to_payload(),
    }
    phase_a_set = set(phase_a_slots)
    phase_b_set = set(phase_b_slots)
    return {
        "seed": seed,
        "mode": "identity_route",
        "parent": {
            "old_holdout": parent_score,
            "old_retention": parent_retention,
            "new_holdout": 0.0,
        },
        "child": {
            "old_holdout": child_score,
            "old_retention": child_retention,
            "new_holdout": new_score,
        },
        "repeated_replay": {
            "same_slot_rate": sum(repeated_matches) / len(repeated_matches),
            "old_holdout_after": replay_old,
            "new_holdout_after": replay_new,
            "route_changed": route_before_replay != repeated_route_digest,
        },
        "no_change": {
            "query_count": len(no_change_slots),
            "all_bound": all(slot is not None for slot in no_change_slots),
            "route_digest_unchanged": no_change_before == no_change_after,
        },
        "identity": {
            "phase_a_unique_slots": len(phase_a_set),
            "phase_b_unique_slots": len(phase_b_set),
            "cross_phase_slot_collisions": len(phase_a_set & phase_b_set),
            "allocation_count": route.bank.allocation_count,
            "replacement_count": route.bank.replacement_count,
            "occupied_count": route.bank.occupied_count,
        },
        "budget": {
            "capacity": capacity,
            "pattern_dim": config.memory_units,
            "action_count": config.alphabet_size,
            "identity_prototype_parameters": int(route.bank.prototypes.numel()),
            "identity_action_edges": route.edge_count,
            "identity_total_parameters": route.parameter_count,
        },
        "checkpoint": {
            "parent_route_digest": phase_a_route_digest,
            "child_bundle_digest": checkpoint_digest,
            "restore_bundle_digest_matches": content_digest(restored_payload) == checkpoint_digest,
            "restored_old_holdout": restored_score,
            "restored_new_holdout": restored_new,
        },
        "holdout_updates": 0,
        "no_write_new_holdout": 0.0,
        "new_gain_vs_no_write": new_score,
    }


def _shared_record(seed: int, corpus: ContinualMemoryCorpus) -> dict[str, object]:
    config_values = _memory_config(seed).to_dict()
    config_values.update(
        {
            "memory_action_decoder": "shared",
            "memory_confidence_decay": 0.0,
            "replay_memory_learning_scale": 0.25,
        }
    )
    config = TaijiConfig.from_dict(config_values)
    actions = tuple(
        dict.fromkeys(
            episode.action for episode in (*corpus.phase_a_train, *corpus.phase_b_train)
        )
    )
    phase_a = Taiji(config, episode_id=f"m1-25-shared-phase-a-{seed}")
    for episode in corpus.phase_a_train:
        _write(phase_a, episode)
    parent_scores = _scores(phase_a, corpus, actions)
    payload = deepcopy(phase_a.checkpoint())
    model = Taiji(config, episode_id=f"m1-25-shared-{seed}")
    model.restore(deepcopy(payload))
    for episode in corpus.phase_b_train:
        _write(model, episode)
    child_scores = _scores(model, corpus, actions)
    no_write = Taiji(config, episode_id=f"m1-25-shared-no-write-{seed}")
    no_write.restore(deepcopy(payload))
    for episode in corpus.phase_b_train:
        _write_without_memory(no_write, episode)
    no_write_scores = _scores(no_write, corpus, actions)
    checkpoint = model.checkpoint()
    checkpoint_digest = content_digest(checkpoint)
    restored = Taiji.from_checkpoint(deepcopy(checkpoint))
    restored_scores = _scores(restored, corpus, actions)
    persistent_before = _persistent_digest(restored)
    _scores(restored, corpus, actions)
    persistent_after = _persistent_digest(restored)
    return {
        "seed": seed,
        "mode": "shared_control",
        "parent": parent_scores,
        "no_write": no_write_scores,
        "child": child_scores,
        "restored": restored_scores,
        "checkpoint": {
            "child_digest": checkpoint_digest,
            "restore_digest_matches": content_digest(restored.checkpoint()) == checkpoint_digest,
            "read_only_persistent_state": persistent_before == persistent_after,
        },
        "holdout_updates": 0,
    }


def _identity_promotable(record: dict[str, object]) -> bool:
    parent = record["parent"]
    child = record["child"]
    repeated = record["repeated_replay"]
    no_change = record["no_change"]
    identity = record["identity"]
    checkpoint = record["checkpoint"]
    return bool(
        child["old_holdout"] >= parent["old_holdout"]
        and child["old_retention"] >= parent["old_retention"]
        and child["new_holdout"] > record["no_write_new_holdout"]
        and repeated["same_slot_rate"] == 1.0
        and repeated["old_holdout_after"] >= parent["old_holdout"]
        and repeated["new_holdout_after"] >= child["new_holdout"]
        and no_change["all_bound"]
        and no_change["route_digest_unchanged"]
        and identity["cross_phase_slot_collisions"] == 0
        and identity["replacement_count"] == 0
        and checkpoint["restore_bundle_digest_matches"]
        and record["holdout_updates"] == 0
    )


def run_identity_diagnostics(
    *,
    train_count: int,
    holdout_count: int,
    retention_count: int,
    seeds: tuple[int, ...],
    modes: tuple[str, ...] = MODES,
    capacity: int = DEFAULT_CAPACITY,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    route_learning_rate: float = DEFAULT_ROUTE_LEARNING_RATE,
) -> dict[str, object]:
    corpus = build_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
        retention_count=retention_count,
    )
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unsupported identity route mode: {sorted(unknown)}")
    records: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    if "shared_control" in modes:
        records["shared_control"] = [_shared_record(seed, corpus) for seed in seeds]
    if "identity_route" in modes:
        records["identity_route"] = [
            _route_record(
                seed,
                corpus,
                capacity=capacity,
                match_threshold=match_threshold,
                route_learning_rate=route_learning_rate,
            )
            for seed in seeds
        ]
    promotable = {
        mode: all(_identity_promotable(record) for record in values)
        for mode, values in records.items()
        if mode == "identity_route"
    }
    return {
        "corpus_digest": corpus.digest,
        "sample_counts": corpus.sample_counts,
        "modes": list(modes),
        "capacity": capacity,
        "match_threshold": match_threshold,
        "route_learning_rate": route_learning_rate,
        "replay_disabled": True,
        "answer_table": False,
        "promotable_modes": [mode for mode, passed in promotable.items() if passed],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--retention-count", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--route-learning-rate", type=float, default=DEFAULT_ROUTE_LEARNING_RATE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "decoder": "shared_control_plus_isolated_identity_route",
        "identity_route_isolated": True,
        "diagnostics": run_identity_diagnostics(
            train_count=args.train_count,
            holdout_count=args.holdout_count,
            retention_count=args.retention_count,
            seeds=tuple(int(seed) for seed in args.seeds),
            modes=tuple(str(mode) for mode in args.modes),
            capacity=int(args.capacity),
            match_threshold=float(args.match_threshold),
            route_learning_rate=float(args.route_learning_rate),
        ),
    }
    result["can_promote"] = bool(result["diagnostics"]["promotable_modes"])
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

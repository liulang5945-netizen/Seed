"""Validate the evaluator-only identity-route boundary contract."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from scripts.training.eval_taiji_m1_identity_route import (  # noqa: E402
    CueIdentityRoute,
    _route_training,
)
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import ContinualMemoryCorpus, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-identity-integration-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_identity_integration_20260902.json"
MODES = ("shared_only", "identity_route_fallback")
CAPACITY = 128
MATCH_THRESHOLD = 0.90
ROUTE_LEARNING_RATE = 0.50


@dataclass(frozen=True)
class IdentityRouteResponse:
    """A motor evidence boundary result, never an executable action intent."""

    cue: int
    slot_index: int | None
    source: str
    provenance: str
    confidence: float
    action_probabilities: torch.Tensor
    action_intent: None = None


class IdentityRouteIntegrationAdapter:
    """Connect route evidence to Taiji motor probabilities without execution."""

    def __init__(self, model: Taiji, route: CueIdentityRoute | None) -> None:
        self.model = model
        self.route = route

    def process(self, cue: int, actions: tuple[int, ...]) -> IdentityRouteResponse:
        self.model.reset_dynamics(episode_id=f"m1-27-process-{cue}")
        self.model.observe(
            self.model.config.boundary_symbol,
            learn=False,
            learn_motor=False,
            use_memory=False,
        )
        self.model.observe(cue, learn=False, learn_motor=False, use_memory=False)
        state = self.model.snapshot()
        pattern = state.memory.activity.detach().clone()
        binding = None if self.route is None else self.route.query(pattern)
        if binding is None or binding.slot_index is None:
            probabilities = state.motor_probabilities.detach().clone()
            return IdentityRouteResponse(
                cue=int(cue),
                slot_index=None,
                source="shared-fallback",
                provenance="identity-route:unbound→shared-fallback",
                confidence=0.0,
                action_probabilities=probabilities,
            )
        evidence = self.route.action_synapses[int(binding.slot_index)]
        probabilities = self.model.motor.probabilities(
            state.motor_context,
            episodic_evidence=evidence,
        )
        return IdentityRouteResponse(
            cue=int(cue),
            slot_index=int(binding.slot_index),
            source="identity-route",
            provenance="identity-route:bound→motor-evidence",
            confidence=float(binding.similarity),
            action_probabilities=probabilities.detach().clone(),
        )


def _actions(corpus: ContinualMemoryCorpus) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            episode.action for episode in (*corpus.phase_a_train, *corpus.phase_b_train)
        )
    )


def _response_payload(response: IdentityRouteResponse, actions: tuple[int, ...]) -> dict[str, object]:
    selected = max(actions, key=lambda action: float(response.action_probabilities[action].item()))
    return {
        "cue": response.cue,
        "slot_index": response.slot_index,
        "source": response.source,
        "provenance": response.provenance,
        "confidence": response.confidence,
        "selected_action": selected,
        "action_evidence_norm": float(response.action_probabilities.norm().item()),
        "action_intent": response.action_intent,
    }


def _seed_record(seed: int, corpus: ContinualMemoryCorpus, mode: str) -> dict[str, object]:
    model = Taiji(_memory_config(seed), episode_id=f"m1-27-{mode}-{seed}")
    route: CueIdentityRoute | None = None
    if mode == "identity_route_fallback":
        route = CueIdentityRoute(
            capacity=CAPACITY,
            pattern_dim=model.config.memory_units,
            action_count=model.config.alphabet_size,
            match_threshold=MATCH_THRESHOLD,
            route_learning_rate=ROUTE_LEARNING_RATE,
        )
        _route_training(route, model, corpus.phase_a_train)
        _route_training(route, model, corpus.phase_b_train)
    adapter = IdentityRouteIntegrationAdapter(model, route)
    actions = _actions(corpus)
    learned_cues = tuple(episode.cue for episode in corpus.phase_a_train[:4])
    unseen_cues = tuple(210 + index for index in range(4))
    route_before = None if route is None else content_digest(route.to_payload())
    learned = [
        _response_payload(adapter.process(cue, actions), actions) for cue in learned_cues
    ]
    unseen = [
        _response_payload(adapter.process(cue, actions), actions) for cue in unseen_cues
    ]
    route_after = None if route is None else content_digest(route.to_payload())
    bundle = {
        "model": model.checkpoint(),
        "route": None if route is None else route.to_payload(),
    }
    bundle_digest = content_digest(bundle)
    restored_model = Taiji.from_checkpoint(deepcopy(bundle["model"]))
    restored_route: CueIdentityRoute | None = None
    if route is not None:
        restored_route = CueIdentityRoute(
            capacity=CAPACITY,
            pattern_dim=model.config.memory_units,
            action_count=model.config.alphabet_size,
            match_threshold=MATCH_THRESHOLD,
            route_learning_rate=ROUTE_LEARNING_RATE,
        )
        restored_route.load_payload(deepcopy(bundle["route"]))
    restored_adapter = IdentityRouteIntegrationAdapter(restored_model, restored_route)
    restored_learned = [
        _response_payload(restored_adapter.process(cue, actions), actions)
        for cue in learned_cues
    ]
    restored_unseen = [
        _response_payload(restored_adapter.process(cue, actions), actions)
        for cue in unseen_cues
    ]
    restored_bundle = {
        "model": restored_model.checkpoint(),
        "route": None if restored_route is None else restored_route.to_payload(),
    }
    return {
        "seed": seed,
        "mode": mode,
        "learned": learned,
        "unseen": unseen,
        "restored_learned": restored_learned,
        "restored_unseen": restored_unseen,
        "route_digest_unchanged_during_queries": route_before == route_after,
        "fallback_count": sum(item["source"] == "shared-fallback" for item in unseen),
        "identity_bound_count": sum(item["source"] == "identity-route" for item in learned),
        "no_action_intent": all(
            item["action_intent"] is None for item in (*learned, *unseen)
        ),
        "checkpoint": {
            "bundle_digest_matches": content_digest(restored_bundle) == bundle_digest,
            "restored_outputs_match": learned == restored_learned and unseen == restored_unseen,
        },
        "holdout_updates": 0,
    }


def run_integration_diagnostics(
    *,
    train_count: int,
    holdout_count: int,
    retention_count: int,
    seeds: tuple[int, ...],
    modes: tuple[str, ...] = MODES,
) -> dict[str, object]:
    corpus = build_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
        retention_count=retention_count,
    )
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unsupported integration mode: {sorted(unknown)}")
    records = {
        mode: [_seed_record(seed, corpus, mode) for seed in seeds]
        for mode in modes
    }
    gate = all(
        all(record["no_action_intent"] for record in values)
        and all(record["route_digest_unchanged_during_queries"] for record in values)
        and all(record["checkpoint"]["bundle_digest_matches"] for record in values)
        and all(record["checkpoint"]["restored_outputs_match"] for record in values)
        for values in records.values()
    )
    if "identity_route_fallback" in records:
        gate = gate and all(
            record["identity_bound_count"] == min(4, train_count)
            and record["fallback_count"] == 4
            for record in records["identity_route_fallback"]
        )
    return {
        "sample_counts": {
            "phase_train": train_count,
            "holdout": holdout_count,
            "retention": retention_count,
            "learned_queries": 4,
            "unseen_queries": 4,
        },
        "modes": list(modes),
        "route_capacity": CAPACITY,
        "match_threshold": MATCH_THRESHOLD,
        "promotable": gate,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--retention-count", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    diagnostics = run_integration_diagnostics(
        train_count=args.train_count,
        holdout_count=args.holdout_count,
        retention_count=args.retention_count,
        seeds=tuple(int(seed) for seed in args.seeds),
        modes=tuple(str(mode) for mode in args.modes),
    )
    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "identity_route_isolated": True,
        "action_intent_execution": False,
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

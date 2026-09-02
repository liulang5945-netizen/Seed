"""Compare cue-pattern and event-pattern local action binding substrates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_36_cue_curriculum import (  # noqa: E402
    _config,
    _curriculum,
)
from taiji import DelayedMemoryTask, Taiji, TaijiConfig  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-40-binding-substrate-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_40_binding_substrate_20260902.json"
SEEDS = (11, 29, 47)
DECODERS = ("shared", "local", "cue_selective")
REPLAY_PROVENANCE = "experienced"
REPLAY_SCALE = 1.0
REPLAY_TARGETS = "all"


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _config_for_decoder(seed: int, decoder: str) -> TaijiConfig:
    if decoder not in DECODERS:
        raise ValueError(f"unsupported binding substrate: {decoder}")
    values = _config(seed).to_dict()
    values["memory_action_decoder"] = decoder
    return TaijiConfig.from_dict(values)


def _probe(model: Taiji, queries: tuple[Any, ...], actions: tuple[int, ...]) -> float:
    correct = 0
    for query in queries:
        model.reset_dynamics(episode_id=f"m1-40-probe-{query.query_id}")
        model.observe(
            model.config.boundary_symbol,
            learn=False,
            learn_motor=False,
            use_memory=True,
        )
        step = model.observe(
            query.cue,
            learn=False,
            learn_motor=False,
            use_memory=True,
        )
        prediction = max(
            actions,
            key=lambda action: float(step.probabilities[action].item()),
        )
        correct += int(prediction == query.expected_action)
    return correct / len(queries)


def _binding_route_digest(model: Taiji) -> str:
    memory = model.memory
    route: dict[str, Any] = {
        "decoder": model.config.memory_action_decoder,
        "local_action_readout_pre_index": memory.local_action_readout.pre_index.detach().cpu(),
    }
    if model.config.memory_action_decoder == "shared":
        route.update(
            {
                "presynaptic_route": "receptor_context",
                "receptor_channel": memory.readout_receptors.channel.detach().cpu(),
                "receptor_polarity": memory.readout_receptors.polarity.detach().cpu(),
                "action_readout_pre_index": memory.action_readout.pre_index.detach().cpu(),
            }
        )
    elif model.config.memory_action_decoder == "local":
        route["presynaptic_route"] = "event_pattern"
    else:
        route.update(
            {
                "presynaptic_route": "cue_pattern",
                "cue_encoder_pre_index": memory.cue_encoder.pre_index.detach().cpu(),
            }
        )
    return content_digest(route)


def _write(model: Taiji, episode: Any) -> None:
    DelayedMemoryTask._write_episode(
        model,
        episode,
        provenance=REPLAY_PROVENANCE,
        memory_learning_scale=REPLAY_SCALE,
        memory_learning_targets=REPLAY_TARGETS,
    )


def _condition_record(
    decoder: str,
    corpus: Any,
    event_set_digest: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    actions = _actions(*corpus.phase_a_train, *corpus.phase_b_train)
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        config = _config_for_decoder(seed, decoder)
        phase_a = Taiji(config, episode_id=f"m1-40-phase-a-{decoder}-{seed}")
        for episode in corpus.phase_a_train:
            _write(phase_a, episode)
        old_before = _probe(phase_a, corpus.phase_a_holdout, actions)
        old_retention_before = _probe(phase_a, corpus.phase_a_retention, actions)
        phase_a_checkpoint = deepcopy(phase_a.checkpoint())
        phase_a_digest = content_digest(phase_a_checkpoint)
        route_digest = _binding_route_digest(phase_a)

        no_replay = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        for episode in corpus.phase_b_train:
            _write(no_replay, episode)
        no_replay_old = _probe(no_replay, corpus.phase_a_holdout, actions)
        no_replay_new = _probe(no_replay, corpus.phase_b_holdout, actions)

        replay = Taiji.from_checkpoint(deepcopy(phase_a_checkpoint))
        for episode in corpus.phase_b_train:
            _write(replay, episode)
        for episode in corpus.replay_train:
            _write(replay, episode)
        replay_old = _probe(replay, corpus.phase_a_holdout, actions)
        replay_retention = _probe(replay, corpus.phase_a_retention, actions)
        replay_new = _probe(replay, corpus.phase_b_holdout, actions)
        replay_checkpoint = deepcopy(replay.checkpoint())
        replay_digest = content_digest(replay_checkpoint)
        restored = Taiji.from_checkpoint(deepcopy(replay_checkpoint))
        restored_digest = content_digest(restored.checkpoint())
        records.append(
            {
                "seed": seed,
                "decoder": decoder,
                "presynaptic_route": (
                    "receptor_context"
                    if decoder == "shared"
                    else "event_pattern" if decoder == "local" else "cue_pattern"
                ),
                "binding_route_digest": route_digest,
                "old_before": old_before,
                "old_retention_before": old_retention_before,
                "no_replay_old_after": no_replay_old,
                "no_replay_new_after": no_replay_new,
                "replay_old_after": replay_old,
                "replay_retention_after": replay_retention,
                "replay_new_after": replay_new,
                "replay_backward_transfer": replay_old - old_before,
                "replay_causal_gain": replay_old - no_replay_old,
                "replay_new_delta_vs_no_replay": replay_new - no_replay_new,
                "phase_a_checkpoint_digest": phase_a_digest,
                "replay_checkpoint_digest": replay_digest,
                "continued_from_phase_a": phase_a_digest != replay_digest,
                "checkpoint_roundtrip_exact": restored_digest == replay_digest,
                "active_parameter_count": replay.parameter_count(),
                "planned_active_parameter_count": replay.config.planned_active_parameter_count,
                "parameter_count_matches_plan": (
                    replay.parameter_count() == replay.config.planned_active_parameter_count
                ),
                "holdout_updates": 0,
            }
        )
    return {
        "decoder": decoder,
        "presynaptic_route": (
            "receptor_context"
            if decoder == "shared"
            else "event_pattern" if decoder == "local" else "cue_pattern"
        ),
        "event_set_digest": event_set_digest,
        "records": records,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def _condition_passed(condition: dict[str, Any]) -> bool:
    return all(
        record["continued_from_phase_a"]
        and record["checkpoint_roundtrip_exact"]
        and record["parameter_count_matches_plan"]
        and record["holdout_updates"] == 0
        and record["replay_backward_transfer"] >= 0.0
        and record["replay_retention_after"] >= record["old_retention_before"]
        and record["replay_new_after"] + 0.05 >= record["no_replay_new_after"]
        and record["replay_causal_gain"] > 0.0
        for record in condition["records"]
    )


def run_diagnosis() -> dict[str, Any]:
    corpus = _curriculum(phase_a_start=0, phase_b_start=192)
    event_set_digest = content_digest(
        {
            "phase_a_train": [
                [item.memory_id, item.cue, item.action, item.outcome]
                for item in corpus.phase_a_train
            ],
            "phase_b_train": [
                [item.memory_id, item.cue, item.action, item.outcome]
                for item in corpus.phase_b_train
            ],
            "replay_train": [
                [item.memory_id, item.cue, item.action, item.outcome]
                for item in corpus.replay_train
            ],
            "provenance": REPLAY_PROVENANCE,
            "scale": REPLAY_SCALE,
            "targets": REPLAY_TARGETS,
        }
    )
    conditions = [
        _condition_record(decoder, corpus, event_set_digest)
        for decoder in DECODERS
    ]
    for condition in conditions:
        condition["condition_gate_passed"] = _condition_passed(condition)
    shared, local, cue_selective = conditions
    shared_budget = {
        record["seed"]: record["active_parameter_count"]
        for record in shared["records"]
    }
    local_budgets = {
        decoder: {
            record["seed"]: record["active_parameter_count"]
            for record in condition["records"]
        }
        for decoder, condition in (("local", local), ("cue_selective", cue_selective))
    }
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed"
        if local["condition_gate_passed"] or cue_selective["condition_gate_passed"]
        else "failed",
        "variable_changed": "episodic action-readout presynaptic substrate only",
        "baseline_decoder": "shared",
        "candidate_decoders": ["local", "cue_selective"],
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "replay_provenance": REPLAY_PROVENANCE,
        "memory_units": _config(SEEDS[0]).memory_units,
        "identity_organ_enabled": False,
        "replay_scale": REPLAY_SCALE,
        "replay_targets": REPLAY_TARGETS,
        "corpus_digest": corpus.digest,
        "event_set_digest": event_set_digest,
        "parameter_budgets_equal": all(
            shared_budget == candidate_budget
            for candidate_budget in local_budgets.values()
        ),
        "conditions": conditions,
        "conclusion": {
            "local_binding_passed": local["condition_gate_passed"],
            "cue_selective_binding_passed": cue_selective["condition_gate_passed"],
            "binding_substrate_is_sufficient_explanation": (
                (local["condition_gate_passed"] or cue_selective["condition_gate_passed"])
                and not shared["condition_gate_passed"]
            ),
            "next_boundary": (
                "a local binding substrate passed; hold for M1-41 structure review"
                if local["condition_gate_passed"] or cue_selective["condition_gate_passed"]
                else "neither local binding substrate passed B5; preserve shared baseline and split association formation from action write"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_diagnosis()
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

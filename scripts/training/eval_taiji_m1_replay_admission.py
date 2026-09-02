"""Diagnose cue-aware admission for native memory replay without changing Taiji."""

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
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import ContinualMemoryCorpus, Taiji, TaijiConfig  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-replay-admission-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_replay_admission_20260902.json"
POLICIES = ("no_gate", "familiarity_gate", "conflict_reject_gate")
FAMILIARITY_THRESHOLD = 0.90
CONFLICT_THRESHOLD = 0.82


def _probe_cue(model: Taiji, cue: int) -> tuple[torch.Tensor, float]:
    """Read native cue activity and confidence without learning or answer labels."""

    model.reset_dynamics(episode_id=f"m1-21-probe-{cue}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=True)
    model.observe(cue, learn=False, learn_motor=False, use_memory=True)
    state = model.snapshot()
    return state.memory.activity.detach().clone(), float(state.memory.last_confidence)


def _candidate_signal(
    model: Taiji,
    episode: Any,
    phase_b_pattern: torch.Tensor,
) -> dict[str, float | str]:
    activity, familiarity = _probe_cue(model, episode.cue)
    similarity = torch.nn.functional.cosine_similarity(
        activity.unsqueeze(0), phase_b_pattern.unsqueeze(0), dim=1
    )
    return {
        "memory_id": str(episode.memory_id),
        "cue_familiarity": familiarity,
        "phase_b_conflict": float(similarity.max().item()),
    }


def _decision(
    policy: str,
    signal: dict[str, float | str],
    *,
    replay_scale: float,
) -> tuple[bool, float]:
    familiarity = float(signal["cue_familiarity"])
    conflict = float(signal["phase_b_conflict"])
    if policy == "no_gate":
        return True, replay_scale
    if policy == "familiarity_gate":
        return familiarity >= FAMILIARITY_THRESHOLD, replay_scale * max(
            0.05, min(1.0, familiarity)
        )
    if policy == "conflict_reject_gate":
        admitted = conflict < CONFLICT_THRESHOLD
        local_scale = replay_scale * max(0.05, 1.0 - conflict)
        return admitted, local_scale
    raise ValueError(f"unsupported replay admission policy: {policy}")


def _train_phase_b(model: Taiji, corpus: ContinualMemoryCorpus) -> None:
    for episode in corpus.phase_b_train:
        _write(model, episode)


def _apply_policy(
    model: Taiji,
    corpus: ContinualMemoryCorpus,
    policy: str,
) -> dict[str, object]:
    replay_scale = float(model.config.replay_memory_learning_scale)
    decisions: list[dict[str, object]] = []
    for index, phase_b_episode in enumerate(corpus.phase_b_train):
        _write(model, phase_b_episode)
        phase_b_pattern, _ = _probe_cue(model, phase_b_episode.cue)
        replay_episode = corpus.replay_train[index % len(corpus.replay_train)]
        signal = _candidate_signal(model, replay_episode, phase_b_pattern)
        admitted, local_scale = _decision(policy, signal, replay_scale=replay_scale)
        decisions.append(
            {
                **signal,
                "phase_b_memory_id": str(phase_b_episode.memory_id),
                "admitted": admitted,
                "learning_scale": local_scale,
            }
        )
        if admitted:
            _write(
                model,
                replay_episode,
                provenance="replayed",
                memory_learning_scale=local_scale,
            )
    return {
        "replay_considered": len(decisions),
        "replay_admitted": sum(bool(item["admitted"]) for item in decisions),
        "replay_rejected": sum(not bool(item["admitted"]) for item in decisions),
        "decisions": decisions,
    }


def _promotable(record: dict[str, object]) -> bool:
    parent = record["parent"]
    child = record["child"]
    restored = record["restored"]
    checkpoint = record["checkpoint"]
    no_replay = record["no_replay"]
    return bool(
        child["old_holdout"] >= parent["old_holdout"]
        and child["old_retention"] >= parent["old_retention"]
        and child["new_holdout"] >= no_replay["new_holdout"]
        and restored == child
        and checkpoint["restore_digest_matches"]
        and checkpoint["read_only_persistent_state"]
        and record["holdout_updates"] == 0
    )


def _seed_record(
    seed: int,
    corpus: ContinualMemoryCorpus,
    policy: str,
) -> dict[str, object]:
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
    phase_a = Taiji(_memory_config(seed), episode_id=f"m1-21-phase-a-{seed}")
    for episode in corpus.phase_a_train:
        _write(phase_a, episode)
    parent_scores = _scores(phase_a, corpus, actions)
    phase_a_payload = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_payload)

    no_replay = Taiji(config, episode_id=f"m1-21-no-replay-{seed}")
    no_replay.restore(deepcopy(phase_a_payload))
    _train_phase_b(no_replay, corpus)
    no_replay_scores = _scores(no_replay, corpus, actions)

    model = Taiji(config, episode_id=f"m1-21-{policy}-{seed}")
    model.restore(deepcopy(phase_a_payload))
    _train_phase_b(model, corpus)
    replay = _apply_policy(model, corpus, policy)
    child_scores = _scores(model, corpus, actions)
    checkpoint_payload = model.checkpoint()
    checkpoint_digest = content_digest(checkpoint_payload)
    restored = Taiji(config, episode_id=f"m1-21-restored-{policy}-{seed}")
    restored.restore(deepcopy(checkpoint_payload))
    restored_scores = _scores(restored, corpus, actions)
    persistent_before = _persistent_digest(restored)
    persistent_after_scores = _scores(restored, corpus, actions)
    persistent_after = _persistent_digest(restored)
    decisions = replay["decisions"]
    familiarity = [float(item["cue_familiarity"]) for item in decisions]
    conflict = [float(item["phase_b_conflict"]) for item in decisions]
    return {
        "seed": seed,
        "policy": policy,
        "parent": parent_scores,
        "no_replay": no_replay_scores,
        "child": child_scores,
        "restored": restored_scores,
        "replay": {
            **replay,
            "base_learning_scale": float(config.replay_memory_learning_scale),
            "cue_familiarity_min": min(familiarity),
            "cue_familiarity_max": max(familiarity),
            "phase_b_conflict_min": min(conflict),
            "phase_b_conflict_max": max(conflict),
        },
        "replay_causal_gain_vs_parent": child_scores["old_holdout"]
        - parent_scores["old_holdout"],
        "new_gain_vs_no_replay": child_scores["new_holdout"]
        - no_replay_scores["new_holdout"],
        "checkpoint": {
            "parent_digest": phase_a_digest,
            "child_digest": checkpoint_digest,
            "restore_digest_matches": content_digest(restored.checkpoint()) == checkpoint_digest,
            "read_only_persistent_state": persistent_before == persistent_after,
        },
        "holdout_updates": 0,
        "restored_scores_match": restored_scores == persistent_after_scores,
    }


def run_replay_admission_diagnostics(
    *,
    train_count: int,
    holdout_count: int,
    retention_count: int,
    seeds: tuple[int, ...],
    policies: tuple[str, ...] = POLICIES,
) -> dict[str, object]:
    corpus = build_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
        retention_count=retention_count,
    )
    unknown = set(policies) - set(POLICIES)
    if unknown:
        raise ValueError(f"unsupported replay admission policy: {sorted(unknown)}")
    records = {
        policy: [_seed_record(seed, corpus, policy) for seed in seeds]
        for policy in policies
    }
    promotable = {
        policy: all(_promotable(record) for record in values)
        for policy, values in records.items()
    }
    return {
        "corpus_digest": corpus.digest,
        "sample_counts": corpus.sample_counts,
        "policies": list(policies),
        "thresholds": {
            "familiarity": FAMILIARITY_THRESHOLD,
            "phase_b_conflict": CONFLICT_THRESHOLD,
        },
        "promotable_policies": [policy for policy, passed in promotable.items() if passed],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--retention-count", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "decoder": "shared",
        "admission_only": True,
        "interleaved_after_phase_b": True,
        "answer_table": False,
        "diagnostics": run_replay_admission_diagnostics(
            train_count=args.train_count,
            holdout_count=args.holdout_count,
            retention_count=args.retention_count,
            seeds=tuple(int(seed) for seed in args.seeds),
            policies=tuple(str(policy) for policy in args.policies),
        ),
    }
    result["can_promote"] = bool(result["diagnostics"]["promotable_policies"])
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

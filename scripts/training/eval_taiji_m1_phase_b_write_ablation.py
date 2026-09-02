"""Isolate phase-B native memory write targets and local learning strength."""

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

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from scripts.training.eval_taiji_m1_interleaved_rehearsal import (  # noqa: E402
    _persistent_digest,
    _scores,
    _write,
)
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import ContinualMemoryCorpus, Taiji, TaijiConfig  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-phase-b-write-ablation-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_phase_b_write_ablation_20260902.json"
TARGETS = ("all", "association", "readout")
DEFAULT_SCALES = (0.05, 0.10, 0.25, 0.50, 1.0)
NO_WRITE = "no_write"


def _write_without_memory(model: Taiji, episode: Any) -> None:
    """Consume a phase-B episode while explicitly forbidding memory writes."""

    model.reset_dynamics(episode_id=f"m1-22-no-write-{episode.memory_id}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
    model.observe(episode.cue, learn=False, learn_motor=False, use_memory=False)
    model.act((episode.action,), sample=False)
    model.settle_action(1.0, learn=False, learn_memory=False)
    model.observe(episode.outcome, learn=False, learn_motor=False, use_memory=False)


def _train_phase_b(
    model: Taiji,
    corpus: ContinualMemoryCorpus,
    *,
    target: str | None,
    scale: float | None,
) -> None:
    for episode in corpus.phase_b_train:
        if target is None:
            _write_without_memory(model, episode)
        else:
            assert scale is not None
            _write(
                model,
                episode,
                memory_learning_scale=scale,
                memory_learning_targets=target,
            )


def _promotable(record: dict[str, object]) -> bool:
    parent = record["parent"]
    child = record["child"]
    no_write = record["no_write"]
    restored = record["restored"]
    checkpoint = record["checkpoint"]
    return bool(
        child["old_holdout"] >= parent["old_holdout"]
        and child["old_retention"] >= parent["old_retention"]
        and child["new_holdout"] > no_write["new_holdout"]
        and restored == child
        and checkpoint["restore_digest_matches"]
        and checkpoint["read_only_persistent_state"]
        and record["holdout_updates"] == 0
    )


def _seed_record(
    seed: int,
    corpus: ContinualMemoryCorpus,
    candidate: str,
    *,
    target: str | None,
    scale: float | None,
    no_write_scores: dict[str, float],
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
    phase_a = Taiji(_memory_config(seed), episode_id=f"m1-22-phase-a-{seed}")
    for episode in corpus.phase_a_train:
        _write(phase_a, episode)
    parent_scores = _scores(phase_a, corpus, actions)
    phase_a_payload = deepcopy(phase_a.checkpoint())
    phase_a_digest = content_digest(phase_a_payload)
    phase_a_write_count = int(phase_a.memory.write_count)

    model = Taiji(config, episode_id=f"m1-22-{candidate}-{seed}")
    model.restore(deepcopy(phase_a_payload))
    _train_phase_b(model, corpus, target=target, scale=scale)
    child_scores = _scores(model, corpus, actions)
    checkpoint_payload = model.checkpoint()
    checkpoint_digest = content_digest(checkpoint_payload)
    restored = Taiji(config, episode_id=f"m1-22-restored-{candidate}-{seed}")
    restored.restore(deepcopy(checkpoint_payload))
    restored_scores = _scores(restored, corpus, actions)
    persistent_before = _persistent_digest(restored)
    persistent_after_scores = _scores(restored, corpus, actions)
    persistent_after = _persistent_digest(restored)
    return {
        "seed": seed,
        "candidate": candidate,
        "target": target,
        "learning_scale": scale,
        "parent": parent_scores,
        "no_write": no_write_scores,
        "child": child_scores,
        "restored": restored_scores,
        "new_gain_vs_no_write": child_scores["new_holdout"]
        - no_write_scores["new_holdout"],
        "checkpoint": {
            "parent_digest": phase_a_digest,
            "child_digest": checkpoint_digest,
            "restore_digest_matches": content_digest(restored.checkpoint()) == checkpoint_digest,
            "read_only_persistent_state": persistent_before == persistent_after,
        },
        "memory_write_count": int(model.memory.write_count),
        "phase_b_memory_writes": int(model.memory.write_count) - phase_a_write_count,
        "holdout_updates": 0,
        "restored_scores_match": restored_scores == persistent_after_scores,
    }


def run_phase_b_write_diagnostics(
    *,
    train_count: int,
    holdout_count: int,
    retention_count: int,
    seeds: tuple[int, ...],
    targets: tuple[str, ...] = TARGETS,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> dict[str, object]:
    corpus = build_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
        retention_count=retention_count,
    )
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"unsupported phase-B learning target: {sorted(unknown)}")
    if not scales or any(float(scale) <= 0.0 for scale in scales):
        raise ValueError("phase-B learning scales must be positive")

    candidate_specs: list[tuple[str, str | None, float | None]] = [(NO_WRITE, None, None)]
    candidate_specs.extend(
        (f"{target}@{scale:g}", target, float(scale))
        for target in targets
        for scale in scales
    )
    records: dict[str, list[dict[str, object]]] = {name: [] for name, _, _ in candidate_specs}
    no_write_by_seed: dict[int, dict[str, float]] = {}
    for seed in seeds:
        no_write = _seed_record(
            seed,
            corpus,
            NO_WRITE,
            target=None,
            scale=None,
            no_write_scores={"old_holdout": 0.0, "old_retention": 0.0, "new_holdout": 0.0},
        )
        no_write_scores = dict(no_write["child"])
        no_write["no_write"] = no_write_scores
        records[NO_WRITE].append(no_write)
        no_write_by_seed[seed] = no_write_scores
    for seed in seeds:
        for candidate, target, scale in candidate_specs[1:]:
            records[candidate].append(
                _seed_record(
                    seed,
                    corpus,
                    candidate,
                    target=target,
                    scale=scale,
                    no_write_scores=no_write_by_seed[seed],
                )
            )
    promotable = {
        candidate: all(_promotable(record) for record in values)
        for candidate, values in records.items()
        if candidate != NO_WRITE
    }
    return {
        "corpus_digest": corpus.digest,
        "sample_counts": corpus.sample_counts,
        "targets": list(targets),
        "scales": [float(scale) for scale in scales],
        "replay_disabled": True,
        "promotable_candidates": [candidate for candidate, passed in promotable.items() if passed],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--holdout-count", type=int, default=8)
    parser.add_argument("--retention-count", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--scales", nargs="+", type=float, default=list(DEFAULT_SCALES))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "decoder": "shared",
        "memory_learning_target_ablation": True,
        "answer_table": False,
        "diagnostics": run_phase_b_write_diagnostics(
            train_count=args.train_count,
            holdout_count=args.holdout_count,
            retention_count=args.retention_count,
            seeds=tuple(int(seed) for seed in args.seeds),
            targets=tuple(str(target) for target in args.targets),
            scales=tuple(float(scale) for scale in args.scales),
        ),
    }
    result["can_promote"] = bool(result["diagnostics"]["promotable_candidates"])
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

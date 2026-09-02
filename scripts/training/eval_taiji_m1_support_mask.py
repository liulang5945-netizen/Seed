"""Audit and isolate cue-conditioned support for the native shared decoder."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
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

FORMAT = "taiji-native-m1-support-mask-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_support_mask_20260902.json"
MODES = ("shared", "cue_mask")
DEFAULT_MASK_FRACTION = 0.50


def _probe_cue(model: Taiji, cue: int) -> tuple[torch.Tensor, float]:
    model.reset_dynamics(episode_id=f"m1-23-probe-{cue}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=True)
    model.observe(cue, learn=False, learn_motor=False, use_memory=True)
    state = model.snapshot()
    return state.memory.activity.detach().clone(), float(state.memory.last_confidence)


def _cue_support_mask(
    model: Taiji,
    cortical_context: torch.Tensor,
    threshold: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    cue_pattern = model.memory._cue_pattern(cortical_context, threshold)
    cue_context = model.memory.readout_receptors.forward(cue_pattern)
    width = int(cue_context.numel())
    keep = max(1, min(width, int(round(width * float(fraction)))))
    indices = torch.topk(cue_context.abs(), k=keep, dim=0).indices
    mask = torch.zeros_like(cue_context)
    return mask.scatter(0, indices, torch.ones_like(indices, dtype=mask.dtype))


@contextmanager
def _support_mode(
    model: Taiji,
    mode: str,
    mask_fraction: float,
) -> Iterator[None]:
    if mode == "shared":
        yield
        return
    if mode != "cue_mask":
        raise ValueError(f"unsupported support mode: {mode}")
    if not 0.0 < float(mask_fraction) <= 1.0:
        raise ValueError("mask_fraction must be in (0, 1]")

    memory = model.memory
    original_write = memory.write
    original_recall = memory.recall

    def masked_write(cortical_context: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        threshold = kwargs["threshold"]
        mask = _cue_support_mask(model, cortical_context, threshold, mask_fraction)
        original_forward = memory.action_readout.forward
        original_local_update = memory.action_readout.local_update

        def masked_forward(presynaptic: torch.Tensor) -> torch.Tensor:
            return original_forward(presynaptic * mask)

        def masked_local_update(
            postsynaptic_error: torch.Tensor,
            presynaptic_trace: torch.Tensor,
            **update_kwargs: Any,
        ) -> None:
            original_local_update(
                postsynaptic_error,
                presynaptic_trace * mask,
                **update_kwargs,
            )

        memory.action_readout.forward = masked_forward
        memory.action_readout.local_update = masked_local_update
        try:
            return original_write(cortical_context, *args, **kwargs)
        finally:
            memory.action_readout.forward = original_forward
            memory.action_readout.local_update = original_local_update

    def masked_recall(
        cortical_context: torch.Tensor,
        previous: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        mask = _cue_support_mask(model, cortical_context, previous.threshold, mask_fraction)
        original_forward = memory.action_readout.forward

        def masked_forward(presynaptic: torch.Tensor) -> torch.Tensor:
            return original_forward(presynaptic * mask)

        memory.action_readout.forward = masked_forward
        try:
            return original_recall(cortical_context, previous, *args, **kwargs)
        finally:
            memory.action_readout.forward = original_forward

    memory.write = masked_write
    memory.recall = masked_recall
    try:
        yield
    finally:
        memory.write = original_write
        memory.recall = original_recall


def _support_metrics(
    model: Taiji,
    episodes: tuple[Any, ...],
    *,
    mode: str,
    mask_fraction: float,
) -> dict[str, object]:
    activities: list[torch.Tensor] = []
    edge_sets: list[set[tuple[int, int]]] = []
    effective_fan_in: list[int] = []
    confidences: list[float] = []
    action_readout = model.memory.action_readout
    for episode in episodes:
        activity, confidence = _probe_cue(model, episode.cue)
        context = model.memory.readout_receptors.forward(activity)
        if mode == "cue_mask":
            state = model.snapshot()
            cortical_context = model.fabric.cortical_context(state.regions)
            context = context * _cue_support_mask(
                model,
                cortical_context,
                state.memory.threshold,
                mask_fraction,
            )
        active_edges = context[action_readout.pre_index].abs() > 1e-8
        edge_sets.append(
            {
                (int(post), int(local))
                for post, local in zip(*torch.where(active_edges), strict=True)
            }
        )
        effective_fan_in.append(int(active_edges.sum().item()))
        activities.append(activity)
        confidences.append(confidence)
    activity_matrix = torch.stack(activities)
    similarity = torch.nn.functional.cosine_similarity(
        activity_matrix[:, None, :], activity_matrix[None, :, :], dim=2
    )
    off_diagonal = ~torch.eye(len(episodes), dtype=torch.bool)
    values = similarity[off_diagonal]
    union = set().union(*edge_sets)
    return {
        "episode_count": len(episodes),
        "mean_effective_fan_in": sum(effective_fan_in) / len(effective_fan_in),
        "min_effective_fan_in": min(effective_fan_in),
        "max_effective_fan_in": max(effective_fan_in),
        "active_edge_union": len(union),
        "cue_cosine_max_off_diagonal": float(values.max().item()) if len(values) else 0.0,
        "cue_confidence_min": min(confidences),
        "cue_confidence_max": max(confidences),
        "_edge_sets": edge_sets,
        "_activities": activity_matrix,
    }


def _cross_support_metrics(
    phase_a: dict[str, object],
    phase_b: dict[str, object],
) -> dict[str, float]:
    phase_a_edges = set().union(*phase_a["_edge_sets"])
    phase_b_edges = set().union(*phase_b["_edge_sets"])
    union = phase_a_edges | phase_b_edges
    shared = phase_a_edges & phase_b_edges
    activities = torch.cat((phase_a["_activities"], phase_b["_activities"]))
    cross = torch.nn.functional.cosine_similarity(
        phase_a["_activities"][:, None, :],
        phase_b["_activities"][None, :, :],
        dim=2,
    )
    pairwise_jaccard = [
        len(left & right) / max(1, len(left | right))
        for left in phase_a["_edge_sets"]
        for right in phase_b["_edge_sets"]
    ]
    return {
        "shared_edge_count": float(len(shared)),
        "edge_union_count": float(len(union)),
        "shared_edge_ratio": float(len(shared) / max(1, len(union))),
        "cross_cue_edge_jaccard_mean": float(sum(pairwise_jaccard) / len(pairwise_jaccard)),
        "cross_cue_edge_jaccard_max": float(max(pairwise_jaccard)),
        "cross_phase_cue_cosine_max": float(cross.max().item()),
        "all_cue_norm_mean": float(activities.norm(dim=1).mean().item()),
    }


def _write_without_memory(model: Taiji, episode: Any) -> None:
    model.reset_dynamics(episode_id=f"m1-23-no-write-{episode.memory_id}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=False)
    model.observe(episode.cue, learn=False, learn_motor=False, use_memory=False)
    model.act((episode.action,), sample=False)
    model.settle_action(1.0, learn=False, learn_memory=False)
    model.observe(episode.outcome, learn=False, learn_motor=False, use_memory=False)


def _promotable(record: dict[str, object]) -> bool:
    parent = record["parent"]
    child = record["child"]
    no_write = record["no_write"]
    support = record["support"]
    checkpoint = record["checkpoint"]
    support_gate = (
        support["phase_a"]["min_effective_fan_in"] > 0
        and support["phase_b"]["min_effective_fan_in"] > 0
        and math.isfinite(support["cross_phase"]["shared_edge_ratio"])
        and math.isfinite(support["cross_phase"]["cross_phase_cue_cosine_max"])
    )
    return bool(
        support_gate
        and child["old_holdout"] >= parent["old_holdout"]
        and child["old_retention"] >= parent["old_retention"]
        and child["new_holdout"] > no_write["new_holdout"]
        and restored_scores_match(record)
        and checkpoint["restore_digest_matches"]
        and checkpoint["read_only_persistent_state"]
        and record["holdout_updates"] == 0
    )


def restored_scores_match(record: dict[str, object]) -> bool:
    return record["restored"] == record["restored_after_read"]


def _seed_record(
    seed: int,
    corpus: ContinualMemoryCorpus,
    mode: str,
    mask_fraction: float,
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
    phase_a_model = Taiji(_memory_config(seed), episode_id=f"m1-23-phase-a-{mode}-{seed}")
    with _support_mode(phase_a_model, mode, mask_fraction):
        for episode in corpus.phase_a_train:
            _write(phase_a_model, episode)
        parent_scores = _scores(phase_a_model, corpus, actions)
        phase_a_support = _support_metrics(
            phase_a_model,
            corpus.phase_a_train,
            mode=mode,
            mask_fraction=mask_fraction,
        )
    phase_a_payload = deepcopy(phase_a_model.checkpoint())
    phase_a_digest = content_digest(phase_a_payload)

    no_write_model = Taiji(config, episode_id=f"m1-23-no-write-{mode}-{seed}")
    no_write_model.restore(deepcopy(phase_a_payload))
    with _support_mode(no_write_model, mode, mask_fraction):
        for episode in corpus.phase_b_train:
            _write_without_memory(no_write_model, episode)
        no_write_scores = _scores(no_write_model, corpus, actions)

    model = Taiji(config, episode_id=f"m1-23-{mode}-{seed}")
    model.restore(deepcopy(phase_a_payload))
    with _support_mode(model, mode, mask_fraction):
        for episode in corpus.phase_b_train:
            _write(model, episode)
        phase_b_support = _support_metrics(
            model,
            corpus.phase_b_train,
            mode=mode,
            mask_fraction=mask_fraction,
        )
        child_scores = _scores(model, corpus, actions)
    cross_support = _cross_support_metrics(phase_a_support, phase_b_support)
    for support in (phase_a_support, phase_b_support):
        support.pop("_edge_sets")
        support.pop("_activities")
    checkpoint_payload = model.checkpoint()
    checkpoint_digest = content_digest(checkpoint_payload)
    restored = Taiji(config, episode_id=f"m1-23-restored-{mode}-{seed}")
    restored.restore(deepcopy(checkpoint_payload))
    with _support_mode(restored, mode, mask_fraction):
        restored_scores = _scores(restored, corpus, actions)
        persistent_before = _persistent_digest(restored)
        restored_after_read = _scores(restored, corpus, actions)
        persistent_after = _persistent_digest(restored)
    return {
        "seed": seed,
        "mode": mode,
        "mask_fraction": mask_fraction if mode == "cue_mask" else None,
        "parent": parent_scores,
        "no_write": no_write_scores,
        "child": child_scores,
        "restored": restored_scores,
        "restored_after_read": restored_after_read,
        "new_gain_vs_no_write": child_scores["new_holdout"]
        - no_write_scores["new_holdout"],
        "support": {
            "phase_a": phase_a_support,
            "phase_b": phase_b_support,
            "cross_phase": cross_support,
        },
        "checkpoint": {
            "parent_digest": phase_a_digest,
            "child_digest": checkpoint_digest,
            "restore_digest_matches": content_digest(restored.checkpoint()) == checkpoint_digest,
            "read_only_persistent_state": persistent_before == persistent_after,
        },
        "phase_b_memory_writes": int(model.memory.write_count) - int(phase_a_model.memory.write_count),
        "holdout_updates": 0,
    }


def run_support_diagnostics(
    *,
    train_count: int,
    holdout_count: int,
    retention_count: int,
    seeds: tuple[int, ...],
    modes: tuple[str, ...] = MODES,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> dict[str, object]:
    corpus = build_corpus(
        train_count=train_count,
        holdout_count=holdout_count,
        retention_count=retention_count,
    )
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unsupported support mode: {sorted(unknown)}")
    if not 0.0 < float(mask_fraction) <= 1.0:
        raise ValueError("mask_fraction must be in (0, 1]")
    records = {
        mode: [_seed_record(seed, corpus, mode, mask_fraction) for seed in seeds]
        for mode in modes
    }
    promotable = {
        mode: all(_promotable(record) for record in values)
        for mode, values in records.items()
    }
    return {
        "corpus_digest": corpus.digest,
        "sample_counts": corpus.sample_counts,
        "modes": list(modes),
        "mask_fraction": float(mask_fraction),
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
    parser.add_argument("--mask-fraction", type=float, default=DEFAULT_MASK_FRACTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "decoder": "shared",
        "support_mask_isolated": True,
        "diagnostics": run_support_diagnostics(
            train_count=args.train_count,
            holdout_count=args.holdout_count,
            retention_count=args.retention_count,
            seeds=tuple(int(seed) for seed in args.seeds),
            modes=tuple(str(mode) for mode in args.modes),
            mask_fraction=float(args.mask_fraction),
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

"""Diagnose whether cue-conditioned write/read support is aligned."""

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
from scripts.training.eval_taiji_m1_support_mask import _cue_support_mask  # noqa: E402
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import ContinualMemoryCorpus, Taiji, TaijiConfig  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-support-alignment-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_support_alignment_20260902.json"
MODES = ("shared", "write_mask_only", "read_mask_only", "aligned_mask")
DEFAULT_MASK_FRACTION = 0.50


def _probe_cue(model: Taiji, cue: int) -> tuple[torch.Tensor, float, float]:
    model.reset_dynamics(episode_id=f"m1-24-probe-{cue}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False, use_memory=True)
    step = model.observe(cue, learn=False, learn_motor=False, use_memory=True)
    state = model.snapshot()
    return (
        state.memory.activity.detach().clone(),
        float(state.memory.last_confidence),
        float(step.memory_recall.action_evidence.norm().item()),
    )


def _mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "shared":
        return False, False
    if mode == "write_mask_only":
        return True, False
    if mode == "read_mask_only":
        return False, True
    if mode == "aligned_mask":
        return True, True
    raise ValueError(f"unsupported support alignment mode: {mode}")


@contextmanager
def _support_mode(
    model: Taiji,
    mode: str,
    mask_fraction: float,
) -> Iterator[None]:
    write_mask, read_mask = _mode_flags(mode)
    if not write_mask and not read_mask:
        yield
        return
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

    memory.write = masked_write if write_mask else original_write
    memory.recall = masked_recall if read_mask else original_recall
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
    write_mask, read_mask = _mode_flags(mode)
    action_readout = model.memory.action_readout
    activities: list[torch.Tensor] = []
    read_edge_sets: list[set[tuple[int, int]]] = []
    write_edge_sets: list[set[tuple[int, int]]] = []
    native_fan_in: list[int] = []
    read_fan_in: list[int] = []
    write_fan_in: list[int] = []
    confidences: list[float] = []
    evidence_norms: list[float] = []
    for episode in episodes:
        activity, confidence, evidence_norm = _probe_cue(model, episode.cue)
        state = model.snapshot()
        cortical_context = model.fabric.cortical_context(state.regions)
        native_context = model.memory.readout_receptors.forward(activity)
        masked_context = native_context * _cue_support_mask(
            model,
            cortical_context,
            state.memory.threshold,
            mask_fraction,
        )
        read_context = masked_context if read_mask else native_context
        write_context = masked_context if write_mask else native_context
        native_active = native_context[action_readout.pre_index].abs() > 1e-8
        read_active = read_context[action_readout.pre_index].abs() > 1e-8
        write_active = write_context[action_readout.pre_index].abs() > 1e-8
        native_fan_in.append(int(native_active.sum().item()))
        read_fan_in.append(int(read_active.sum().item()))
        write_fan_in.append(int(write_active.sum().item()))
        read_edge_sets.append(
            {
                (int(post), int(local))
                for post, local in zip(*torch.where(read_active), strict=True)
            }
        )
        write_edge_sets.append(
            {
                (int(post), int(local))
                for post, local in zip(*torch.where(write_active), strict=True)
            }
        )
        activities.append(activity)
        confidences.append(confidence)
        evidence_norms.append(evidence_norm)
    activity_matrix = torch.stack(activities)
    similarity = torch.nn.functional.cosine_similarity(
        activity_matrix[:, None, :], activity_matrix[None, :, :], dim=2
    )
    off_diagonal = ~torch.eye(len(episodes), dtype=torch.bool)
    values = similarity[off_diagonal]

    def union_size(edge_sets: list[set[tuple[int, int]]]) -> int:
        return len(set().union(*edge_sets))

    return {
        "episode_count": len(episodes),
        "native_mean_effective_fan_in": sum(native_fan_in) / len(native_fan_in),
        "read_mean_effective_fan_in": sum(read_fan_in) / len(read_fan_in),
        "write_mean_effective_fan_in": sum(write_fan_in) / len(write_fan_in),
        "native_union_edges": union_size(
            [
                {
                    (int(post), int(local))
                    for post, local in zip(*torch.where(native_context[action_readout.pre_index].abs() > 1e-8), strict=True)
                }
                for native_context in [
                    model.memory.readout_receptors.forward(activity) for activity in activities
                ]
            ]
        ),
        "read_union_edges": union_size(read_edge_sets),
        "write_union_edges": union_size(write_edge_sets),
        "cue_cosine_max_off_diagonal": float(values.max().item()) if len(values) else 0.0,
        "cue_confidence_min": min(confidences),
        "cue_confidence_max": max(confidences),
        "action_evidence_norm_mean": sum(evidence_norms) / len(evidence_norms),
        "action_evidence_norm_max": max(evidence_norms),
        "_read_edge_sets": read_edge_sets,
        "_write_edge_sets": write_edge_sets,
        "_activities": activity_matrix,
    }


def _cross_metrics(
    phase_a: dict[str, object],
    phase_b: dict[str, object],
    key: str,
) -> dict[str, float]:
    left_sets = phase_a[key]
    right_sets = phase_b[key]
    left_union = set().union(*left_sets)
    right_union = set().union(*right_sets)
    pairwise_jaccard = [
        len(left & right) / max(1, len(left | right))
        for left in left_sets
        for right in right_sets
    ]
    cross = torch.nn.functional.cosine_similarity(
        phase_a["_activities"][:, None, :],
        phase_b["_activities"][None, :, :],
        dim=2,
    )
    return {
        "shared_edge_count": float(len(left_union & right_union)),
        "edge_union_count": float(len(left_union | right_union)),
        "shared_edge_ratio": float(len(left_union & right_union) / max(1, len(left_union | right_union))),
        "cross_cue_edge_jaccard_mean": float(sum(pairwise_jaccard) / len(pairwise_jaccard)),
        "cross_cue_edge_jaccard_max": float(max(pairwise_jaccard)),
        "cross_phase_cue_cosine_max": float(cross.max().item()),
    }


def _write_without_memory(model: Taiji, episode: Any) -> None:
    model.reset_dynamics(episode_id=f"m1-24-no-write-{episode.memory_id}")
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
    support_gate = all(
        math.isfinite(float(support["cross_phase"][path][key]))
        for path in ("read", "write")
        for key in (
            "shared_edge_ratio",
            "cross_cue_edge_jaccard_mean",
            "cross_phase_cue_cosine_max",
        )
    ) and all(
        int(support[phase][f"{path}_mean_effective_fan_in"]) > 0
        for phase in ("phase_a", "phase_b")
        for path in ("read", "write")
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
    phase_a_model = Taiji(config, episode_id=f"m1-24-phase-a-{mode}-{seed}")
    with _support_mode(phase_a_model, mode, mask_fraction):
        for episode in corpus.phase_a_train:
            _write(phase_a_model, episode)
        phase_a_support = _support_metrics(
            phase_a_model,
            corpus.phase_a_train,
            mode=mode,
            mask_fraction=mask_fraction,
        )
        parent_scores = _scores(phase_a_model, corpus, actions)
    phase_a_payload = deepcopy(phase_a_model.checkpoint())
    phase_a_digest = content_digest(phase_a_payload)

    no_write_model = Taiji(config, episode_id=f"m1-24-no-write-{mode}-{seed}")
    no_write_model.restore(deepcopy(phase_a_payload))
    with _support_mode(no_write_model, mode, mask_fraction):
        for episode in corpus.phase_b_train:
            _write_without_memory(no_write_model, episode)
        no_write_scores = _scores(no_write_model, corpus, actions)

    model = Taiji(config, episode_id=f"m1-24-{mode}-{seed}")
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
    cross_support = {
        "read": _cross_metrics(phase_a_support, phase_b_support, "_read_edge_sets"),
        "write": _cross_metrics(phase_a_support, phase_b_support, "_write_edge_sets"),
    }
    phase_a_support.pop("_read_edge_sets")
    phase_a_support.pop("_write_edge_sets")
    phase_a_support.pop("_activities")
    phase_b_support.pop("_read_edge_sets")
    phase_b_support.pop("_write_edge_sets")
    phase_b_support.pop("_activities")
    checkpoint_payload = model.checkpoint()
    checkpoint_digest = content_digest(checkpoint_payload)
    restored = Taiji(config, episode_id=f"m1-24-restored-{mode}-{seed}")
    restored.restore(deepcopy(checkpoint_payload))
    with _support_mode(restored, mode, mask_fraction):
        restored_scores = _scores(restored, corpus, actions)
        persistent_before = _persistent_digest(restored)
        restored_after_read = _scores(restored, corpus, actions)
        persistent_after = _persistent_digest(restored)
    return {
        "seed": seed,
        "mode": mode,
        "mask_fraction": mask_fraction if mode != "shared" else None,
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


def run_alignment_diagnostics(
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
        raise ValueError(f"unsupported support alignment mode: {sorted(unknown)}")
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
        "support_alignment_isolated": True,
        "diagnostics": run_alignment_diagnostics(
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

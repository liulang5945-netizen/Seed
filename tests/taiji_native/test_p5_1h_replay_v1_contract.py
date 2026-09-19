"""P5.1h recipe-machinery gates: default-off experience replay (contract §6).

The P5.1h package's single allowed recipe variable is experience replay on the
procedural consolidation path (P5.1g left replay_digest empty and ranking
pairs at zero).  These gates pin: bitwise-identical behaviour when replay is
off, gradient-scale semantics when it is on, readout-vocabulary protection,
and the behavioural core -- replay preserves an old skill against
parameter-drift forgetting better than the same consolidation without replay.
"""

from __future__ import annotations

import pytest
import torch

from taiji.contracts import ActionIntent, EpisodicMemoryRecord
from taiji.procedural_memory import ProceduralSequenceLearner

CUE_DIM = 8
KIND_A, KIND_B = "editor.open", "mcp.list"


def _cue(pattern: int, noise: float, seed: int) -> torch.Tensor:
    """Four disjoint cue directions (2 dims each) with per-record noise."""

    generator = torch.Generator().manual_seed(seed)
    cue = torch.zeros(CUE_DIM)
    cue[pattern * 2 : pattern * 2 + 2] = 1.0
    return cue + noise * torch.rand(CUE_DIM, generator=generator)


def _record(memory_id: str, episode_id: str, tick: int, cue: torch.Tensor, kind: str):
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id=episode_id,
        tick=tick,
        cue=cue,
        action_intent=ActionIntent(intent_id=f"intent-{memory_id}", kind=kind, tick=tick),
    )


def _task_episodes(task_prefix: str, patterns: tuple[int, int], kinds: tuple[str, str], noise: float, seed: int):
    """Two episodes, one per (pattern, kind) association."""

    return (
        _record(f"{task_prefix}-p0-1", f"{task_prefix}-ep-p0", 1, _cue(patterns[0], noise, seed), kinds[0]),
        _record(f"{task_prefix}-p0-2", f"{task_prefix}-ep-p0", 2, _cue(patterns[0], noise, seed + 1), kinds[0]),
        _record(f"{task_prefix}-p1-1", f"{task_prefix}-ep-p1", 1, _cue(patterns[1], noise, seed + 2), kinds[1]),
        _record(f"{task_prefix}-p1-2", f"{task_prefix}-ep-p1", 2, _cue(patterns[1], noise, seed + 3), kinds[1]),
    )


def _accuracy(learner: ProceduralSequenceLearner, records) -> float:
    hits = 0
    with torch.no_grad():
        for record in records:
            kind = learner.predict_episode([record.cue.detach().to(dtype=torch.float32)])[0]
            hits += int(kind == record.action_intent.kind)
    return hits / len(records)


def _fresh() -> ProceduralSequenceLearner:
    return ProceduralSequenceLearner(CUE_DIM, hidden_dim=16, seed=7)


# --------------------------------------------------------------------------- #
# Gate 1: default-off is bitwise identical (P5.1g frozen path untouched)
# --------------------------------------------------------------------------- #


def test_gate1_default_off_bitwise_identical() -> None:
    train = _task_episodes("train", (0, 1), (KIND_A, KIND_B), 0.05, 11)
    replay = _task_episodes("old", (0, 1), (KIND_A, KIND_B), 0.05, 12)
    baseline = _fresh()
    baseline.consolidate(train, epochs=6, learning_rate=0.05)
    neutral = _fresh()
    neutral.consolidate(train, epochs=6, learning_rate=0.05, replay_source=replay, replay_weight=0.0)
    for (name, parameter), (_, parameter2) in zip(
        baseline.named_parameters(), neutral.named_parameters()
    ):
        assert torch.equal(parameter, parameter2), name


# --------------------------------------------------------------------------- #
# Gate 2: replay semantics (contract section 6)
# --------------------------------------------------------------------------- #


def test_gate2_positive_weight_changes_training_and_unknown_kind_raises() -> None:
    train = _task_episodes("train", (0, 1), (KIND_A, KIND_B), 0.05, 21)
    replay = _task_episodes("old", (0, 1), (KIND_A, KIND_B), 0.05, 22)
    baseline = _fresh()
    baseline.consolidate(train, epochs=6, learning_rate=0.05)
    with_replay = _fresh()
    with_replay.consolidate(train, epochs=6, learning_rate=0.05, replay_source=replay, replay_weight=1.0)
    changed = any(
        not torch.equal(p1, p2)
        for (_, p1), (_, p2) in zip(baseline.named_parameters(), with_replay.named_parameters())
    )
    assert changed
    stranger = [_record("x1", "x-ep", 1, _cue(0, 0.0, 1), "unknown.kind")]
    with pytest.raises(ValueError, match="outside the readout"):
        with_replay.consolidate(train, epochs=1, replay_source=stranger, replay_weight=1.0)
    with pytest.raises(ValueError, match="replay_weight"):
        with_replay.consolidate(train, epochs=1, replay_weight=-1.0)


# --------------------------------------------------------------------------- #
# Gate 3: behavioural core -- replay preserves the old skill (contract §6)
# --------------------------------------------------------------------------- #


def test_gate3_replay_preserves_old_skill_against_interference() -> None:
    """The second task re-maps the SAME cue regions to the opposite kinds:
    without replay the flip overwrites the old association (1.0 -> 0.5);
    with replay the old episodes keep being trained and the old skill
    survives (1.0 stays 1.0)."""

    old_task = _task_episodes("old", (0, 1), (KIND_A, KIND_B), 0.10, 41)
    new_task = _task_episodes("new", (0, 1), (KIND_B, KIND_A), 0.10, 42)

    without_replay = _fresh()
    without_replay.consolidate(old_task, epochs=10, learning_rate=0.05)
    without_replay.consolidate(new_task, epochs=10, learning_rate=0.05)

    with_replay = _fresh()
    with_replay.consolidate(old_task, epochs=10, learning_rate=0.05)
    with_replay.consolidate(
        new_task, epochs=10, learning_rate=0.05, replay_source=old_task, replay_weight=3.0
    )

    old_records = list(old_task)
    acc_without = _accuracy(without_replay, old_records)
    acc_with = _accuracy(with_replay, old_records)
    assert acc_without < 1.0, acc_without
    assert acc_with == 1.0, acc_with
    assert acc_with > acc_without, (acc_with, acc_without)

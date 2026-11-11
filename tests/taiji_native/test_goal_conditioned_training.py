"""10.7 oracles: the sample set is well formed and the training loop is wired."""

from __future__ import annotations

import torch

from taiji.goal_conditioned_training import accuracy, all_states, free_place, samples, train
from taiji.living_core import LivingCoreConfig, LivingCoreReference
from taiji.native_goal_world import legal_moves, step

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)


def subject() -> LivingCoreReference:
    torch.manual_seed(20261101)
    return LivingCoreReference(CONFIG).double()


def test_samples_cover_six_states_and_twelve_labeled_pairs():
    rows = samples()
    assert len(all_states()) == 6
    assert len(rows) == 12
    assert {tuple(row["state"].where) for row in rows} == {
        tuple(state.where) for state in all_states()
    }


def test_each_sample_has_one_legal_correct_action_and_one_legal_other():
    for row in samples():
        state = row["state"]
        free = free_place(state)
        moves = legal_moves(state, occupy=True)
        assert row["correct"] in moves
        assert row["other"] in moves
        assert row["correct"] != row["other"]
        filled = step(state, row["correct"], occupy=True)
        assert filled.place_of(row["correct"][0]) == free
        assert row["goal"] == f"把{row['correct'][0]}放到{free}"
        assert row["other"][0] != row["correct"][0]


def test_free_place_is_the_only_unoccupied_one():
    for state in all_states():
        taken = {place for _, place in state.where}
        free = free_place(state)
        assert free not in taken
        assert len(taken) == 2


def test_training_returns_one_loss_per_step():
    model = subject()
    result = train(model, samples()[:2], lr=1e-5, steps=2)
    assert len(result["losses"]) == 2
    assert all(value == value for value in result["losses"])
    assert result["cpu_seconds"] >= 0.0


def test_accuracy_reports_a_full_denominator():
    model = subject()
    held = samples()[:4]
    report = accuracy(model, held)
    assert report["total"] == len(held)
    assert 0 <= report["hits"] <= report["total"]
    assert len(report["rows"]) == len(held)

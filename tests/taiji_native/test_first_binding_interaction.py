"""Mixed answer odds cancel additive shortcuts and preserve interaction credit."""

import pytest
import torch

from scripts.training.paired_world_22_5 import build_partition
from scripts.training.probe_first_binding_22_5 import interaction, select_cells


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_additive_scores_have_exactly_zero_interaction_and_credit(dtype):
    parameters = torch.tensor([0.5, 0.25, 0.125], dtype=dtype, requires_grad=True)
    x = torch.tensor([-1, 1], dtype=dtype)
    scores = parameters[0] + parameters[1] * x[:, None] + parameters[2] * x[None, :]
    delta = interaction(scores)
    assert float(delta.detach()) == 0
    assert torch.equal(torch.autograd.grad(delta, parameters)[0], torch.zeros_like(parameters))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bilinear_scores_have_task_interaction_and_correct_credit(dtype):
    coefficient = torch.tensor(0.25, dtype=dtype, requires_grad=True)
    x = torch.tensor([-1, 1], dtype=dtype)
    scores = coefficient * x[:, None] * x[None, :]
    delta = interaction(scores)
    assert float(delta.detach()) == 1
    assert float(torch.autograd.grad(delta, coefficient)[0]) == 4
    assert float(interaction(scores.flip(0)).detach()) == -1


def test_diagnostic_cells_use_the_same_requests_and_opposite_gold_targets():
    partition = build_partition("train")
    for position in range(3):
        cells, targets = select_cells(partition, position, [0, 1])
        assert targets[0] != targets[1]
        assert cells[0][0]["targets"][0] == cells[1][1]["targets"][0] == targets[0]
        assert cells[0][1]["targets"][0] == cells[1][0]["targets"][0] == targets[1]

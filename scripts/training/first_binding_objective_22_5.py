"""Supervision-side mixed world/request credit with actual first-answer history."""

from __future__ import annotations

import torch
from torch.nn import functional

from scripts.training.paired_episode_22_5 import next_request, start_episode
from scripts.training.probe_first_binding_22_5 import interaction, select_cells
from scripts.training.task_history_loss_22_5 import generate_first, score_answer


def quad_terms(model, partition, position, colors):
    """Regroup four existing episodes, sharing their gold first-answer CE graphs."""
    cells, targets = select_cells(partition, position, colors)
    advantages, episode_losses, histories = [], [], []
    for row_index, cell_row in enumerate(cells):
        row = []
        for column_index, episode in enumerate(cell_row):
            state = start_episode(model, partition, episode)
            present_ce, _ = score_answer(model, state, targets[0])
            absent_ce, _ = score_answer(model, state, targets[1])
            first_ce = present_ce if row_index == column_index else absent_ce
            actual, content, ended = generate_first(model, state)
            if ended:
                second = next_request(model, actual, episode, 1)
                second_ce, _ = score_answer(model, second, episode["targets"][1])
                blocked = first_ce.new_zeros(())
            else:
                second_ce = first_ce.new_zeros(())
                blocked = first_ce.new_ones(())
            episode_losses.append(0.5 * (first_ce + second_ce + blocked))
            row.append(7 * (absent_ce - present_ce))
            histories.append({"first_hex": content.hex(), "ended": ended})
        advantages.append(torch.stack(row))
    odds = torch.stack(advantages)
    delta = interaction(odds)
    return {
        "ce": torch.stack(episode_losses).mean(),
        "mixed": functional.softplus(-delta),
        "delta": delta,
        "odds": odds,
        "histories": histories,
    }


def add_gradients(first, second):
    """Preserve unused-parameter ownership when combining two objective gradients."""
    combined: list[torch.Tensor | None] = []
    for left, right in zip(first, second, strict=True):
        if left is None:
            combined.append(None if right is None else right.clone())
        elif right is None:
            combined.append(left.clone())
        else:
            combined.append(left + right)
    return combined

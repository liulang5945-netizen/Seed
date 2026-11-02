"""09.6.4 consensus birth-direction rule over disjoint fit sub-windows.

The rule reads only the fit journal: it splits it into disjoint windows, takes the
top P-gradient directions of each window separately, and returns the directions
that the windows agree on. No evaluation set, no gold and no holdout input is
accepted anywhere in this module; the verifier additionally checks that at the
source level.
"""

from __future__ import annotations

import torch

from taiji.living_core import LivingCoreReference
from taiji.state_migration import CareerJournal, career_step_specs, resume_career

RULE_NAME = "consensus"


def window_edges(journal: CareerJournal, parts: int) -> list[tuple[int, int]]:
    """Split the supervised rounds, not the raw steps.

    The career starts with prompt patches that carry no supervised byte, so a
    step-based split can hand a whole window to the prompt and leave it unscoreable.
    """
    if parts < 2:
        raise ValueError("consensus needs at least two windows")
    specs = career_step_specs(journal)
    rounds = [index for index, spec in enumerate(specs) if spec["kind"] == "round"]
    if len(rounds) < parts:
        raise ValueError("fewer supervised rounds than declared windows")
    edges = [round(index * len(rounds) / parts) for index in range(parts + 1)]
    return [(rounds[edges[index]], rounds[edges[index + 1] - 1] + 1) for index in range(parts)]


def window_nll(
    model: LivingCoreReference,
    journal: CareerJournal,
    start: int,
    end: int,
    provider=None,
) -> torch.Tensor:
    _, steps = resume_career(
        model,
        model.initial_state(journal.subject_id),
        journal,
        0,
        provider,
        limit=end,
        retain_outputs=False,
    )
    terms = [probabilities[byte] for probabilities, byte in steps[start:end] if byte is not None]
    if not terms:
        raise ValueError("declared window has no supervised rounds")
    return -torch.stack(terms).log().mean()


def window_directions(
    parent: LivingCoreReference,
    journal: CareerJournal,
    parts: int,
    count: int,
    provider=None,
) -> list[list[torch.Tensor]]:
    """Per window, per group: the top `count` left singular vectors of dNLL/dP."""
    collected = []
    for start, end in window_edges(journal, parts):
        parent.zero_grad(set_to_none=True)
        loss = window_nll(parent, journal, start, end, provider)
        loss.backward()
        collected.append(
            [parent.P.grad[group].detach().clone() for group in range(parent.config.groups)]
        )
    parent.zero_grad(set_to_none=True)
    return collected


def consensus_directions(
    parent: LivingCoreReference,
    journal: CareerJournal,
    *,
    parts: int = 4,
    count: int = 2,
    provider=None,
) -> dict:
    """Directions the fit windows agree on, one list per group."""
    per_window = window_directions(parent, journal, parts, count, provider)
    directions = {}
    for group in range(parent.config.groups):
        size = per_window[0][group].shape[0]
        agreement = torch.zeros(size, size, dtype=per_window[0][group].dtype)
        for gradients in per_window:
            left, _, _ = torch.linalg.svd(gradients[group], full_matrices=False)
            for index in range(min(count, left.shape[1])):
                vector = left[:, index]
                agreement += torch.outer(vector, vector)
        values, vectors = torch.linalg.eigh(agreement)
        order = torch.argsort(values, descending=True)
        directions[(RULE_NAME, group)] = [
            vectors[:, order[index]].detach().clone() for index in range(count)
        ]
    return directions

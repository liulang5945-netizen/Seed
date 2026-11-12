"""Episode-local reuse of immutable request encodings with full autograd credit."""

from __future__ import annotations

import torch

from scripts.training.first_a_support_22_5 import accept, output
from scripts.training.paired_episode_22_5 import next_request, start_episode
from taiji.u2_subject import U2Subject


def episode_loss_reusing_request(model, partition, episode):
    state = start_episode(model, partition, episode)
    terms = []
    for turn, target in enumerate(episode["targets"]):
        if turn:
            state = next_request(model, state, episode, turn)
        request = model.language_goal(state) if isinstance(model, U2Subject) else None
        for symbol in [*target, 256]:
            if isinstance(model, U2Subject):
                probability, _, details = model.output(state, request, state["prediction_action"])
            else:
                probability, _, details = output(model, state)
            terms.append(-probability[symbol])
            state = accept(model, state, symbol, details)
    return torch.stack(terms).mean()

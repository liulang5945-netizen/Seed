"""Episode-local reuse of immutable request encodings with full autograd credit."""

from __future__ import annotations

import torch

from scripts.training.first_a_support_22_5 import accept, output
from scripts.training.paired_episode_22_5 import next_request, start_episode
from taiji.u2_history import detached_tree
from taiji.u2_subject import U2Subject


def shadow_optimizer(model, optimizer):
    """Clone parameter and moment ownership for measured real-Adam operations."""
    originals = list(model.parameters())
    temporary = [torch.nn.Parameter(p.detach().clone()) for p in originals]
    shadow = torch.optim.Adam(temporary, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    shadow.load_state_dict(detached_tree(optimizer.state_dict()))
    storage = {p.data_ptr() for p in originals}
    storage.update(
        value.data_ptr()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    assert all(p.data_ptr() not in storage for p in temporary)
    assert all(
        value.data_ptr() not in storage
        for state in shadow.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    return temporary, shadow


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

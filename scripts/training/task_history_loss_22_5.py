"""Teacher versus actual-self history, with explicit conditional-gradient boundaries."""

from __future__ import annotations

import torch

from scripts.training.first_a_support_22_5 import accept, output
from scripts.training.paired_episode_22_5 import next_request, start_episode


def score_answer(model, state, target):
    terms = []
    for symbol in [*target, 256]:
        logp, _, details = output(model, state)
        terms.append(-logp[symbol])
        state = accept(model, state, symbol, details)
    return torch.stack(terms).mean(), state


def generate_first(model, state):
    content = bytearray()
    for index in range(65):
        logp, _, details = output(model, state)
        symbol = int(logp.detach().argmax())
        if index == 64 and symbol != 256:
            break
        state = accept(model, state, symbol, details)
        if symbol == 256:
            return state, bytes(content), True
        content.append(symbol)
    return state, bytes(content), False


def history_loss(model, partition, episode, source):
    if source not in {"teacher", "actual-self"}:
        raise ValueError("unsupported_task_history_source")
    initial = start_episode(model, partition, episode)
    first_loss, teacher_state = score_answer(model, initial, episode["targets"][0])
    if source == "teacher":
        history, content, ended = teacher_state, episode["targets"][0], True
    else:
        history, content, ended = generate_first(model, initial)
    if ended:
        second = next_request(model, history, episode, 1)
        second_loss, _ = score_answer(model, second, episode["targets"][1])
        blocked = first_loss.new_zeros(())
    else:
        second_loss = first_loss.new_zeros(())
        blocked = first_loss.new_ones(())
    return {
        "loss": 0.5 * (first_loss + second_loss + blocked),
        "first_ce": first_loss,
        "second_ce": second_loss,
        "first_history_bytes": content,
        "first_ended": ended,
        "blocked_outcome": blocked,
        "history_source": source,
        "selection_credit": "conditional_state_only; no argmax distribution gradient",
    }

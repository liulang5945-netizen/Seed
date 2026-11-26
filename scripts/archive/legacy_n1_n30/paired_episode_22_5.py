"""Two requests share one observed subject state; labels stay on the supervision side."""

from __future__ import annotations

import torch

from scripts.training.first_a_support_22_5 import accept, output, prepare
from scripts.training.paired_world_22_5 import public_example
from taiji.u2_subject import U2Subject


def start_episode(model, partition, episode):
    example = public_example(partition, episode)
    return prepare(model, example["public"], example["id"])


def next_request(model, state, episode, turn):
    raw = episode["requests"][turn]
    if isinstance(model, U2Subject):
        state = model.set_request(state, f"paired-request-{turn}", raw)
        return model.begin_utterance(state)
    return model.set_request(state, raw)


def episode_loss(model, partition, episode):
    state = start_episode(model, partition, episode)
    terms = []
    for turn, target in enumerate(episode["targets"]):
        if turn:
            state = next_request(model, state, episode, turn)
        for symbol in [*target, 256]:
            probability, _, details = output(model, state)
            terms.append(-probability[symbol])
            state = accept(model, state, symbol, details)
    return torch.stack(terms).mean()


def rollout_episode(model, partition, episode, budget_guard=None):
    rows, blocked = [], False
    with torch.no_grad():
        state = start_episode(model, partition, episode)
        for turn, target in enumerate(episode["targets"]):
            if blocked:
                rows.append(
                    {
                        "turn": turn,
                        "target": target.decode(),
                        "output_hex": "",
                        "eos": False,
                        "valid_utf8": False,
                        "correct": False,
                        "status": "blocked_by_prior_round",
                    }
                )
                continue
            if turn:
                state = next_request(model, state, episode, turn)
            content, eos = bytearray(), False
            for index in range(65):
                if budget_guard is not None:
                    budget_guard()
                probability, _, details = output(model, state)
                symbol = int(probability.detach().argmax())
                if index == 64 and symbol != 256:
                    break
                state = accept(model, state, symbol, details)
                if symbol == 256:
                    eos = True
                    break
                content.append(symbol)
            try:
                bytes(content).decode("utf-8", errors="strict")
                valid = True
            except UnicodeDecodeError:
                valid = False
            rows.append(
                {
                    "turn": turn,
                    "target": target.decode(),
                    "output_hex": content.hex(),
                    "eos": eos,
                    "valid_utf8": valid,
                    "correct": bool(eos and valid and bytes(content) == target),
                    "status": "answered" if eos else "byte_budget_exhausted",
                }
            )
            blocked = not eos
    return {
        "correct": sum(row["correct"] for row in rows),
        "denominator": 2,
        "rows": rows,
        "state": state,
    }

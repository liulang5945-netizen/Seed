"""10.7 bounded training: teach the subject to answer the action verdict question.

Samples are (state, goal) pairs from the 10.1b world where exactly one legal action
satisfies the goal. The label is 「是」 for the correct action's prefix and 「否」 for
the other one; nothing else is supervised. Weights are saved under output/ only.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from pathlib import Path

import torch

from taiji.action_verdict_scorer import NO, YES, answer_nll, prefix
from taiji.constrained_learning import career_nll
from taiji.native_goal_world import OBJECTS, PLACES, WorldState, legal_moves, state_of
from taiji.state_migration import CareerJournal


def all_states() -> list[WorldState]:
    return [
        state_of(((OBJECTS[0], first), (OBJECTS[1], second)))
        for first, second in itertools.permutations(PLACES, 2)
    ]


def free_place(state: WorldState) -> str:
    taken = {place for _, place in state.where}
    return next(place for place in PLACES if place not in taken)


def samples() -> list[dict]:
    rows = []
    for state in all_states():
        free = free_place(state)
        moves = legal_moves(state, occupy=True)
        for obj in OBJECTS:
            correct = (obj, free)
            other = next(move for move in moves if move != correct)
            rows.append(
                {
                    "state": state,
                    "goal": f"把{obj}放到{free}",
                    "correct": correct,
                    "other": other,
                }
            )
    return rows


def sample_loss(subject, sample: dict, provider=None) -> torch.Tensor:
    yes_prefix = prefix(
        sample["state"], sample["goal"], f"把{sample['correct'][0]}放到{sample['correct'][1]}"
    )
    no_prefix = prefix(
        sample["state"], sample["goal"], f"把{sample['other'][0]}放到{sample['other'][1]}"
    )
    return _nll(subject, yes_prefix, YES, provider) + _nll(subject, no_prefix, NO, provider)


def _nll(subject, text: str, answer: str, provider) -> torch.Tensor:
    journal = CareerJournal(
        subject_id="t" + hashlib.sha256(text.encode()).hexdigest()[:12],
        prompt=text.encode(),
        target=answer.encode(),
        goal=text.encode(),
        records=(),
        raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3),
        known_nu=(True, True, False, True, True, True, False, True),
        now=0,
    )
    return career_nll(subject, journal, provider)


def train(subject, train_samples, *, lr: float, steps: int, provider=None) -> dict:
    parameters = [parameter for parameter in subject.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    start = time.process_time()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [sample_loss(subject, sample, provider) for sample in train_samples]
        ).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"losses": losses, "cpu_seconds": time.process_time() - start}


def accuracy(subject, held_samples, provider=None) -> dict:
    hits, rows = 0, []
    for sample in held_samples:
        correct_verdict = _verdict(subject, sample, sample["correct"], provider)
        other_verdict = _verdict(subject, sample, sample["other"], provider)
        hit = correct_verdict > other_verdict
        hits += int(hit)
        rows.append(
            {
                "goal": sample["goal"],
                "state": [list(pair) for pair in sample["state"].where],
                "correct": list(sample["correct"]),
                "correct_verdict": correct_verdict,
                "other_verdict": other_verdict,
                "hit": hit,
            }
        )
    return {"hits": hits, "total": len(held_samples), "rows": rows}


def _verdict(subject, sample: dict, move: tuple[str, str], provider) -> float:
    text = prefix(sample["state"], sample["goal"], f"把{move[0]}放到{move[1]}")
    return answer_nll(subject, text, NO, provider) - answer_nll(subject, text, YES, provider)


def save(subject, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": subject.state_dict()}, path)

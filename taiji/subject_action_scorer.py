"""10.4: let the subject itself score candidate actions, with no action head.

Each legal move is rendered as one Chinese sentence. The subject scores that
sentence as a continuation of the goal text using its own conditional NLL, and the
lowest-NLL action is selected. Nothing here reads the terminal state or the score.
"""

from __future__ import annotations

import hashlib

from taiji.constrained_learning import career_nll
from taiji.state_migration import CareerJournal, career_step_specs

NU = (0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3)
KNOWN = (True, True, False, True, True, True, False, True)


def render(move: tuple[str, str]) -> str:
    obj, place = move
    return f"把{obj}放到{place}"


def action_journal(goal_text: str, sentence: str, subject_id: str) -> CareerJournal:
    return CareerJournal(
        subject_id=subject_id,
        prompt=goal_text.encode(),
        target=sentence.encode(),
        goal=goal_text.encode(),
        records=(),
        raw_nu=NU,
        known_nu=KNOWN,
        now=0,
    )


def score_actions(subject, goal_text: str, moves, *, provider=None) -> list[dict]:
    rows = []
    for move in moves:
        sentence = render(move)
        journal = action_journal(goal_text, sentence, f"score-{subject_id_safe(sentence)}")
        loss = float(career_nll(subject, journal, provider).detach())
        rows.append({"move": move, "sentence": sentence, "nll": loss})
    return rows


def subject_id_safe(sentence: str) -> str:
    """Stable across processes: ``hash()`` is salted per interpreter."""
    return "a" + hashlib.sha256(sentence.encode()).hexdigest()[:12]


def select_action(subject, goal_text: str, moves, *, provider=None) -> dict:
    rows = score_actions(subject, goal_text, moves, provider=provider)
    best = min(rows, key=lambda row: (row["nll"], row["move"]))
    return {
        "action": best["move"],
        "sentence": best["sentence"],
        "nll": best["nll"],
        "scored": len(rows),
        "table": rows,
    }


def step_count(goal_text: str, sentence: str) -> int:
    return len(career_step_specs(action_journal(goal_text, sentence, "count")))

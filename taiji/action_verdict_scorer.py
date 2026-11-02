"""10.6 action-specific verdict scoring: ask the subject whether an action meets a goal.

The action sits inside the prefix, so every candidate has its own context; the score
compares how much more likely 「是」 is than 「否」. Both answer tokens share one
prefix and therefore one initial state, which is what makes the comparison meaningful.
"""

from __future__ import annotations

import hashlib

from taiji.constrained_learning import career_nll
from taiji.state_migration import CareerJournal

YES = "是"
NO = "否"
NU = (0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3)
KNOWN = (True, True, False, True, True, True, False, True)


def state_text(state) -> str:
    return "，".join(f"{obj}在{place}" for obj, place in state.where) + "。"


def prefix(state, goal_text: str, sentence: str) -> str:
    return f"{state_text(state)}目标：{goal_text}。执行：{sentence}。是否达成？"


def _subject_id(prefix_text: str) -> str:
    return "v" + hashlib.sha256(prefix_text.encode()).hexdigest()[:12]


def answer_nll(subject, prefix_text: str, answer: str, provider=None) -> float:
    journal = CareerJournal(
        subject_id=_subject_id(prefix_text),
        prompt=prefix_text.encode(),
        target=answer.encode(),
        goal=prefix_text.encode(),
        records=(),
        raw_nu=NU,
        known_nu=KNOWN,
        now=0,
    )
    return float(career_nll(subject, journal, provider).detach())


def score_actions(subject, state, goal_text: str, moves, *, provider=None) -> list[dict]:
    rows = []
    for move in moves:
        sentence = f"把{move[0]}放到{move[1]}"
        prefix_text = prefix(state, goal_text, sentence)
        yes = answer_nll(subject, prefix_text, YES, provider)
        no = answer_nll(subject, prefix_text, NO, provider)
        rows.append(
            {
                "move": move,
                "sentence": sentence,
                "prefix": prefix_text,
                "nll_yes": yes,
                "nll_no": no,
                "answer_gap": abs(no - yes),
                "verdict": no - yes,
            }
        )
    return rows


def choose(subject, state, goal_text: str, moves, *, provider=None) -> dict:
    rows = score_actions(subject, state, goal_text, moves, provider=provider)
    best = max(rows, key=lambda row: (row["verdict"], row["move"]))
    return {"action": best["move"], "verdict": best["verdict"], "scored": len(rows), "table": rows}

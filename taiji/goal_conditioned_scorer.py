"""10.5 goal-conditioned scoring: contrast between a matched and a mismatched goal.

The 10.4 interface compared absolute sentence likelihoods, so a lexical prior could
win without reading the goal. This one compares the same sentence under two goals,
where that prior cancels; the score that matters is the contrast, not the level.
"""

from __future__ import annotations

from taiji.constrained_learning import career_nll
from taiji.subject_action_scorer import action_journal, render, subject_id_safe


def _nll(subject, goal_text: str, sentence: str, provider) -> float:
    journal = action_journal(goal_text, sentence, subject_id_safe(sentence))
    return float(career_nll(subject, journal, provider).detach())


def contrast_rows(
    subject, matched_goal: str, mismatched_goal: str, moves, *, provider=None
) -> list[dict]:
    """Per action: how much the matching goal beats the mismatched one."""
    rows = []
    for move in moves:
        sentence = render(move)
        matched = _nll(subject, matched_goal, sentence, provider)
        mismatched = _nll(subject, mismatched_goal, sentence, provider)
        rows.append(
            {
                "move": move,
                "sentence": sentence,
                "nll_matched": matched,
                "nll_mismatched": mismatched,
                "contrast": mismatched - matched,
            }
        )
    return rows


def choose(subject, matched_goal: str, mismatched_goal: str, moves, *, provider=None) -> dict:
    rows = contrast_rows(subject, matched_goal, mismatched_goal, moves, provider=provider)
    best = max(rows, key=lambda row: (row["contrast"], row["move"]))
    return {
        "action": best["move"],
        "sentence": best["sentence"],
        "contrast": best["contrast"],
        "scored": len(rows),
        "table": rows,
    }


def margin(rows: list[dict], correct: tuple[str, str], other: tuple[str, str]) -> dict:
    """Contrast gap and direction between the correct action and the other one."""
    by_move = {tuple(row["move"]): row for row in rows}
    if correct not in by_move or other not in by_move:
        raise ValueError("both actions must be present in the contrast table")
    correct_contrast = by_move[correct]["contrast"]
    other_contrast = by_move[other]["contrast"]
    return {
        "correct": correct_contrast,
        "other": other_contrast,
        "gap": abs(correct_contrast - other_contrast),
        "correct_higher": correct_contrast > other_contrast,
    }

"""Goal paraphrases that keep the entity names but drop the action template.

The 10.7 fixture renders the goal as exactly the correct action sentence, so a copy
mechanism wins for free. These three styles name the same object and place without
using "把…放到…":

    "{obj}该待在{place}。"
    "我希望{obj}出现在{place}。"
    "目标状态：{obj}在{place}。"

They exist to be measured, not to be assumed better: if the names still carry the
answer, a character-overlap matcher will keep solving the task.
"""

from __future__ import annotations

STYLES = ("needs", "hope", "state")


def paraphrase_goal(obj: str, place: str, style: str) -> str:
    if style == "needs":
        return f"{obj}该待在{place}。"
    if style == "hope":
        return f"我希望{obj}出现在{place}。"
    if style == "state":
        return f"目标状态：{obj}在{place}。"
    raise ValueError(f"unknown style {style!r}")


def action_sentence(obj: str, place: str) -> str:
    return f"把{obj}放到{place}"

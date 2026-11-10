"""10.2 goal binding: a declared rule baseline over the 10.1 world's vocabulary.

This is a reference baseline, not the model. It reads only the Chinese goal text
and the observable state; it never sees a goal id, a gold terminal state or a
score. Anything it cannot bind unambiguously it must refuse, with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from taiji.native_goal_world import OBJECTS, PLACES, WorldState

REASONS = ("missing_referent", "ambiguous", "unsupported")
CONJUNCTIONS = ("，并且", "并且", "，且", "且")
AMBIGUITY_MARKERS = ("或", "还是")
UNKNOWN_REFERENTS = ("它", "那个", "这个")
PLACE_SUFFIXES = ("上面", "上", "里")


@dataclass(frozen=True)
class Binding:
    conditions: tuple[tuple[str, str, int], ...]
    abstain: bool = False
    reason: str | None = None
    unknowns: tuple[str, ...] = ()

    def satisfied(self, state: WorldState) -> bool:
        for obj, place, polarity in self.conditions:
            here = state.place_of(obj) == place
            if polarity > 0 and not here:
                return False
            if polarity < 0 and here:
                return False
        return not self.abstain and bool(self.conditions)


def _strip_place(token: str) -> str:
    for suffix in PLACE_SUFFIXES:
        if token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _find(token: str, vocabulary: tuple[str, ...]) -> str | None:
    for name in vocabulary:
        if name in token:
            return name
    return None


def _clause_condition(clause: str) -> tuple[tuple[str, str, int] | None, str | None]:
    """One clause → (condition, refusal reason)."""
    for object_token, place_token in (
        ("把", "放到"),
        ("把", "移到"),
        ("", "不要留在"),
        ("", "应该出现在"),
        ("", "要在"),
    ):
        if place_token in clause:
            head, _, tail = clause.partition(place_token)
            if object_token:
                head = head.removeprefix(object_token)
            obj = _find(head, OBJECTS)
            place = _find(_strip_place(tail), PLACES)
            if obj is None:
                return None, "missing_referent"
            if place is None:
                return None, "unsupported"
            polarity = -1 if "不要" in place_token else 1
            return (obj, place, polarity), None
    return None, "unsupported"


def bind(goal_text: str, state: WorldState) -> Binding:
    """Bind a Chinese goal to conditions, or refuse with a declared reason."""
    del state  # the observable state is accepted for parity with the model arm
    text = goal_text.strip().rstrip("。")
    if not text:
        return Binding((), abstain=True, reason="unsupported")
    if any(marker in text for marker in AMBIGUITY_MARKERS):
        return Binding((), abstain=True, reason="ambiguous", unknowns=("multiple_targets",))
    if any(marker in text for marker in UNKNOWN_REFERENTS) and not _find(text, OBJECTS):
        return Binding((), abstain=True, reason="missing_referent", unknowns=("object",))
    clauses = [text]
    for conjunction in CONJUNCTIONS:
        expanded = []
        for clause in clauses:
            expanded.extend(clause.split(conjunction))
        clauses = expanded
    conditions = []
    for clause in clauses:
        if not clause.strip():
            continue
        condition, reason = _clause_condition(clause.strip())
        if reason is not None:
            unknowns = ("object",) if reason == "missing_referent" else ()
            return Binding((), abstain=True, reason=reason, unknowns=unknowns)
        conditions.append(condition)
    if not conditions:
        return Binding((), abstain=True, reason="unsupported")
    return Binding(tuple(conditions))

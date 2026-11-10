"""10.3 prediction, bounded rollout and selection: reference planner baseline.

The planner keeps three outcomes apart on purpose: ``predicted`` is its one-step
model of the chosen action, ``imagined`` is the rollout it used to choose, and
``actual`` is what the world did. The explanation is for reading only; nothing in
the scoring path accepts text.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from taiji.goal_binding import Binding
from taiji.native_goal_world import (
    OBJECTS,
    PLACES,
    STEP_COST,
    WorldState,
    legal_moves,
    step,
)

DEPTH = 2


def transition(
    state: WorldState, move: tuple[str, str], *, perturbed: bool = False, occupy: bool = False
) -> WorldState:
    """Reference model. ``perturbed`` swaps the target place to make it wrong."""
    if not perturbed:
        return step(state, move, occupy=occupy)
    obj, place = move
    current = state.place_of(obj)
    others = [candidate for candidate in PLACES if candidate not in (place, current)]
    return step(state, (obj, others[0] if others else place), occupy=occupy)


def state_score(state: WorldState, binding: Binding) -> float:
    return (1.0 if binding.satisfied(state) else 0.0) - STEP_COST * state.steps


@dataclass(frozen=True)
class ActionIntent:
    action: tuple[str, str]
    predicted: WorldState
    imagined: tuple[WorldState, ...]
    expected_score: float
    explanation: str
    ledger: dict = field(default_factory=dict)


def _best_within(
    binding: Binding, state: WorldState, depth: int, model, occupy: bool = False
) -> tuple[float, tuple[WorldState, ...]]:
    """Best score reachable in at most ``depth`` moves, with one shortest trail.

    The search stops expanding a branch as soon as the binding is satisfied: a plan
    has no reason to keep moving after the goal holds, and charging those extra
    steps would make the planner prefer slower solutions.
    """
    best_score = state_score(state, binding)
    best_trail: tuple[WorldState, ...] = ()
    stack = [(state, ())]
    while stack:
        current, trail = stack.pop()
        if len(trail) >= depth or binding.satisfied(current):
            continue
        for move in legal_moves(current, occupy=occupy):
            reached = model(current, move)
            new_trail = trail + (reached,)
            score_now = state_score(reached, binding)
            better = score_now > best_score + 1e-12
            shorter = abs(score_now - best_score) <= 1e-12 and len(new_trail) < len(best_trail)
            if (
                better
                or (shorter and not best_trail)
                or (better and len(new_trail) < len(best_trail))
            ):
                best_score, best_trail = score_now, new_trail
            stack.append((reached, new_trail))
    return best_score, best_trail


def _intent(binding, state, chosen, predicted, trail, value) -> ActionIntent:
    obj, place = chosen
    return ActionIntent(
        action=chosen,
        predicted=predicted,
        imagined=tuple(trail),
        expected_score=value,
        explanation=f"把{obj}放到{place}",
        ledger={
            "step": state.steps,
            "action": list(chosen),
            "predicted": list(predicted.where),
            "imagined": [list(item.where) for item in trail],
            "actual": None,
            "predicted_matches_actual": None,
        },
    )


def _greedy_move(binding: Binding, state: WorldState, moves) -> tuple[str, str]:
    """Simple heuristic: move the bound object one place closer to its target."""
    del state
    targets = {obj: place for obj, place, polarity in binding.conditions if polarity > 0}
    return min(moves, key=lambda move: _greedy_rank(move, targets))


def _greedy_rank(move: tuple[str, str], targets: dict) -> tuple:
    obj, place = move
    target = targets.get(obj)
    distance = len(PLACES) if target is None else abs(PLACES.index(place) - PLACES.index(target))
    return (distance, OBJECTS.index(obj), PLACES.index(place))


def plan(
    binding: Binding,
    state: WorldState,
    *,
    depth: int = DEPTH,
    model=transition,
    mode: str = "planner",
    seed: int = 20261101,
    occupy: bool = False,
) -> ActionIntent:
    """Choose an action. ``mode`` selects the declared contrast arm."""
    moves = legal_moves(state, occupy=occupy)
    if mode == "no_prediction":
        chosen = random.Random(seed * 1000 + state.steps).choice(list(moves))
        predicted = model(state, chosen)
        return _intent(binding, state, chosen, predicted, (), state_score(predicted, binding))
    if mode == "greedy":
        chosen = _greedy_move(binding, state, moves)
        predicted = model(state, chosen)
        return _intent(binding, state, chosen, predicted, (), state_score(predicted, binding))
    scored = []
    for move in moves:
        head = model(state, move)
        tail_value, tail = _best_within(binding, head, max(depth - 1, 0), model, occupy=occupy)
        value = max(state_score(head, binding), tail_value)
        scored.append((value, move, head, (head,) + tail))
    value, chosen, predicted, trail = max(
        scored, key=lambda item: (item[0], -len(item[3]), item[1])
    )
    return _intent(binding, state, chosen, predicted, trail[:depth], value)


def execute(intent: ActionIntent, state: WorldState, *, occupy: bool = False):
    """Execute on the real world and close the ledger's actual/compare columns."""
    actual = step(state, intent.action, occupy=occupy)
    ledger = dict(intent.ledger)
    ledger["actual"] = list(actual.where)
    ledger["predicted_matches_actual"] = bool(
        tuple(intent.predicted.where) == tuple(actual.where)
        and intent.predicted.steps == actual.steps
    )
    return actual, ledger


def run_arm(
    binding: Binding,
    state: WorldState,
    *,
    mode: str = "planner",
    depth: int = DEPTH,
    perturbed: bool = False,
    limit: int = 4,
    seed: int = 20261101,
    occupy: bool = False,
) -> dict:
    """Play one fixture with a declared arm; returns the terminal score and ledger."""

    def model(current, move):
        return transition(current, move, perturbed=perturbed)

    rows = []
    for _ in range(limit):
        if binding.satisfied(state):
            break
        intent = plan(binding, state, depth=depth, model=model, mode=mode, seed=seed, occupy=occupy)
        state, ledger = execute(intent, state, occupy=occupy)
        rows.append(ledger)
    return {"score": state_score(state, binding), "state": state, "ledger": rows}

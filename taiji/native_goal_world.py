"""10.1 T3 task environment: natural Chinese goals over a finite move world.

The host exposes only the observable state (where each object is, plus the step
count) and a shuffled action list. Goals, correctness and scoring stay host-side.
Action ids carry no correctness: the same (object, place) gets a different id on
every step for a declared seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import cast

OBJECTS = ("蓝色杯子", "红色印章")
PLACES = ("木桌", "窗台", "左盒")
STEP_COST = 0.01
MAX_STEPS = 4
ACTION_PREFIX = "a"


@dataclass(frozen=True)
class WorldState:
    where: tuple[tuple[str, str], ...]
    steps: int = 0

    def place_of(self, obj: str) -> str:
        for name, place in self.where:
            if name == obj:
                return place
        raise KeyError(obj)


@dataclass(frozen=True)
class Goal:
    goal_id: str
    text: str
    conditions: tuple[tuple[str, str], ...]

    def satisfied(self, state: WorldState) -> bool:
        return all(state.place_of(obj) == place for obj, place in self.conditions)


def state_of(placements: tuple[tuple[str, str], ...], steps: int = 0) -> WorldState:
    return WorldState(where=tuple(sorted(placements)), steps=steps)


def occupied_places(state: WorldState) -> tuple[str, ...]:
    return tuple(place for _, place in state.where)


def legal_moves(state: WorldState, *, occupy: bool = False) -> tuple[tuple[str, str], ...]:
    """Moves for each object to every other place.

    ``occupy=True`` adds the 10.1b capacity rule: a place holds one object, so a
    move into an occupied place is illegal until the occupant leaves. Default
    ``False`` keeps the frozen 10.1 behaviour byte-identical.
    """
    taken = occupied_places(state) if occupy else ()
    moves = []
    for obj in OBJECTS:
        current = state.place_of(obj)
        for place in PLACES:
            if place == current:
                continue
            if occupy and place in taken:
                continue
            moves.append((obj, place))
    return tuple(moves)


def encode_actions(
    moves: tuple[tuple[str, str], ...], *, seed: int, step_index: int
) -> dict[str, dict[str, str]]:
    """Shuffle the move list into opaque ids; the id order carries no meaning."""
    order = list(moves)
    random.Random(seed * 1000 + step_index).shuffle(order)
    return {
        f"{ACTION_PREFIX}{index}": {"object": obj, "place": place}
        for index, (obj, place) in enumerate(order)
    }


def step(state: WorldState, move: tuple[str, str], *, occupy: bool = False) -> WorldState:
    obj, place = move
    if (obj, place) not in legal_moves(state, occupy=occupy):
        raise ValueError(f"illegal move {move}")
    where = tuple((name, place if name == obj else current) for name, current in state.where)
    return WorldState(where=tuple(sorted(where)), steps=state.steps + 1)


def score(state: WorldState, goal: Goal) -> float:
    return (1.0 if goal.satisfied(state) else 0.0) - STEP_COST * state.steps


def optimal(state: WorldState, goal: Goal, limit: int = MAX_STEPS, *, occupy: bool = False) -> dict:
    """Exhaustive bread-first search; the state space is tiny, so this is exact."""
    frontier: list[tuple[WorldState, tuple[tuple[str, str], ...]]] = [(state, ())]
    best = {"score": score(state, goal), "moves": (), "steps": 0}
    seen = {state.where}
    for _depth in range(limit):
        next_frontier = []
        for current, path in frontier:
            for move in legal_moves(current, occupy=occupy):
                nxt = step(current, move, occupy=occupy)
                path_next = path + (move,)
                candidate = score(nxt, goal)
                if candidate > cast(float, best["score"]):
                    best = {"score": candidate, "moves": path_next, "steps": nxt.steps}
                if nxt.where not in seen:
                    seen.add(nxt.where)
                    next_frontier.append((nxt, path_next))
        frontier = next_frontier
    return best


def run_policy(
    policy,
    state: WorldState,
    goal: Goal,
    *,
    seed: int,
    limit: int = MAX_STEPS,
    occupy: bool = False,
) -> dict:
    """Execute a declared shortcut policy; it may stop early once the goal holds."""
    history: list[str] = []
    for step_index in range(limit):
        if goal.satisfied(state):
            break
        actions = encode_actions(
            legal_moves(state, occupy=occupy), seed=seed, step_index=step_index
        )
        action_id = policy(actions, state, history)
        if action_id not in actions:
            raise ValueError(f"policy returned unknown action {action_id}")
        move = (actions[action_id]["object"], actions[action_id]["place"])
        state = step(state, move, occupy=occupy)
        history.append(action_id)
    return {"score": score(state, goal), "actions": history, "state": state}


def first_legal(actions: dict[str, dict[str, str]], state: WorldState, history: list[str]) -> str:
    """Canonical order, ignoring the shuffled ids; distinct from lowest_id."""
    return min(actions, key=lambda key: (actions[key]["object"], actions[key]["place"]))


def lowest_id(actions: dict[str, dict[str, str]], state: WorldState, history: list[str]) -> str:
    return min(actions, key=lambda key: int(key[len(ACTION_PREFIX) :]))


def fixed_object(actions: dict[str, dict[str, str]], state: WorldState, history: list[str]) -> str:
    for action_id, payload in actions.items():
        if payload["object"] == OBJECTS[0]:
            return action_id
    return sorted(actions)[0]


def cycle(actions: dict[str, dict[str, str]], state: WorldState, history: list[str]) -> str:
    wanted = (OBJECTS[0], PLACES[1]) if len(history) % 2 == 0 else (OBJECTS[0], PLACES[0])
    for action_id, payload in actions.items():
        if (payload["object"], payload["place"]) == wanted:
            return action_id
    return sorted(actions)[0]


SHORTCUT_POLICIES = {
    "first_legal": first_legal,
    "lowest_id": lowest_id,
    "fixed_object": fixed_object,
    "cycle": cycle,
}


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    initial: WorldState
    goal: Goal


def build_fixtures() -> tuple[Fixture, ...]:
    common = state_of(((OBJECTS[0], PLACES[0]), (OBJECTS[1], PLACES[2])))
    other = state_of(((OBJECTS[0], PLACES[1]), (OBJECTS[1], PLACES[0])))
    spare = state_of(((OBJECTS[0], PLACES[2]), (OBJECTS[1], PLACES[1])))
    cup_to_sill = Goal("cup-to-sill", "把蓝色杯子放到窗台", ((OBJECTS[0], PLACES[1]),))
    seal_to_table = Goal("seal-to-table", "把红色印章放到木桌", ((OBJECTS[1], PLACES[0]),))
    blocked = state_of(((OBJECTS[0], PLACES[0]), (OBJECTS[1], PLACES[1])))
    blocked_swap = state_of(((OBJECTS[0], PLACES[1]), (OBJECTS[1], PLACES[2])))
    return (
        Fixture("f1-cup-to-sill-from-table", common, cup_to_sill),
        Fixture("f2-seal-to-table-from-box", common, seal_to_table),
        Fixture("f3-cup-already-on-sill", other, cup_to_sill),
        Fixture("f4-seal-to-table-from-sill", spare, seal_to_table),
        Fixture("g1-sill-blocked-by-seal", blocked, cup_to_sill),
        Fixture("g2-box-blocked-by-seal", blocked_swap, cup_to_sill),
    )

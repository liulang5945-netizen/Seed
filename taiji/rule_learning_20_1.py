"""20.1 fixture: three-session rule learning with counterexamples.

Everything here is data-side.  The response view handed to the graphs never
contains the rule, its signature, the truth table or the solver: those live in
``scoring`` only, which the runner reads after the forward.  A static leak probe
asserts that the graph modules do not import this module.

World contract (public task definition, not sealed data):

* four binary attributes -> 16 combinations;
* rule ``y = b xor xor_{i in S} x_i`` with ``1 <= |S| <= 2`` -> 20 rules;
* five anchors ``0000`` and the four basis vectors label the rule uniquely;
* the action is one of two UTF-8 strings, both 15 bytes.

Turns: 51 answer turns in three sessions (10 / 24 / 17), two atomic feedback
batches (5 lines each) released after turn 5 and turn 17, and 22 post-feedback
first answers (5 v1 + 11 current-v2 + 6 previous-v1).
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Any

SEED = 20261105
ATTRIBUTES = (
    ("表面", ("细", "粗")),
    ("材质", ("软", "硬")),
    ("光泽", ("哑", "亮")),
    ("结构", ("空", "实")),
)
ACTIONS = ("应送左箱。", "应送右箱。")
ANSWER_BYTES = len(ACTIONS[0].encode("utf-8"))
ANCHORS = ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
SYLLABLES = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉"
SESSION_SIZES = (10, 24, 17)
TOTAL_TURNS = sum(SESSION_SIZES)
CORRECTION_TURNS = 5 + 11 + 6
FEEDBACK_RELEASE_TURNS = (5, 17)


@dataclass(frozen=True)
class Rule:
    bias: int
    selected: tuple[int, ...]

    def label(self, bits: tuple[int, ...]) -> int:
        value = self.bias
        for index in self.selected:
            value ^= bits[index]
        return value

    def signature(self) -> str:
        return f"b{self.bias}:s{'-'.join(str(i) for i in self.selected)}"


def enumerate_rules() -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for size in (1, 2):
        for selected in itertools.combinations(range(4), size):
            for bias in (0, 1):
                rules.append(Rule(bias, tuple(selected)))
    return tuple(rules)


RULES = enumerate_rules()


def combinations() -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.product((0, 1), repeat=4))


def bits_key(bits: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in bits)


def action_of(label: int) -> str:
    return ACTIONS[label]


def _attribute_text(bits: tuple[int, ...], order: tuple[int, ...]) -> str:
    return "，".join(
        f"{ATTRIBUTES[index][0]}{ATTRIBUTES[index][1][bits[index]]}" for index in order
    )


def render_query(name: str, bits: tuple[int, ...], order: tuple[int, ...], style: int) -> str:
    body = _attribute_text(bits, order)
    if style == 1:
        return f"{name}：{body}。"
    if style == 2:
        return f"请判断{name}的表述：{body}。"
    return f"（{name}）{body}，应如何分拣？"


def render_feedback(name: str, bits: tuple[int, ...], order: tuple[int, ...], label: int) -> str:
    values = "".join(ATTRIBUTES[index][1][bits[index]] for index in order)
    return f"{name}：{values}⇒{ACTIONS[label]}"


def _new_names(rng: random.Random, count: int) -> list[str]:
    names: list[str] = []
    while len(names) < count:
        candidate = rng.choice(SYLLABLES) + rng.choice(SYLLABLES)
        if candidate not in names:
            names.append(candidate)
    return names


def _shuffled(order: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    rows = list(order)
    rng.shuffle(rows)
    return tuple(rows)


def build_world(seed: int, index: int) -> dict[str, Any]:
    """One world: response turns plus a scoring-only truth block."""
    rule_rng = random.Random(seed * 100003 + index)
    name_rng = random.Random(seed * 7919 + index * 104729 + 41)
    order_rng = random.Random(seed * 6151 + index * 7717 + 5)
    v1_index = rule_rng.randrange(len(RULES))
    v2_index = (v1_index + 1 + rule_rng.randrange(len(RULES) - 1)) % len(RULES)
    v1, v2 = RULES[v1_index], RULES[v2_index]
    names = _new_names(name_rng, 3)
    order_v1 = _shuffled((0, 1, 2, 3), order_rng)
    order_v2 = _shuffled((0, 1, 2, 3), order_rng)
    order_v3 = _shuffled((0, 1, 2, 3), order_rng)
    anchor_v1 = _shuffled(tuple(range(5)), order_rng)
    anchor_v2 = _shuffled(tuple(range(5)), order_rng)
    others = [bits for bits in combinations() if bits not in ANCHORS]
    first_five = _shuffled(tuple(range(len(others))), order_rng)[:5]
    remaining_six = tuple(index for index in range(len(others)) if index not in first_five)
    repeat_two = _shuffled(tuple(range(5)), order_rng)[:2]
    asked_five = [others[index] for index in first_five]
    previous_six = [others[index] for index in remaining_six]
    current_eleven = others
    turns: list[dict[str, Any]] = []
    scoring_turns: dict[str, dict[str, Any]] = {}

    def add(
        session: int,
        kind: str,
        name: str,
        bits: tuple[int, ...],
        version: int,
        order: tuple[int, ...],
        style: int,
        first_answer: bool,
        correction_surface: bool = False,
        repeat_of: int | None = None,
    ) -> int:
        turn = len(turns) + 1
        query = render_query(name, bits, order, style)
        turns.append(
            {
                "turn": turn,
                "session": session,
                "kind": kind,
                "entity": name,
                "bits": bits_key(bits),
                "version": version,
                "query": query,
                "first_answer": first_answer,
                "correction_surface": correction_surface,
                "repeat_of": repeat_of,
            }
        )
        expected = (v2 if version == 2 else v1).label(bits)
        scoring_turns[str(turn)] = {
            "expected": expected,
            "expected_action": action_of(expected),
            "version": version,
            "bits": bits_key(bits),
        }
        return turn

    # session 1: blind anchors -> v1 feedback -> five non-anchor answers.
    anchor_turns = []
    for position in anchor_v1:
        bits = ANCHORS[position]
        anchor_turns.append(add(1, "anchor_blind", names[0], bits, 1, order_v1, 1, True))
    batch_v1 = [
        {
            "source_id": f"{index}-fb-v1-{position}",
            "turn": anchor_turns[position],
            "query": turns[anchor_turns[position] - 1]["query"],
            "feedback": render_feedback(
                names[0], ANCHORS[position], order_v1, v1.label(ANCHORS[position])
            ),
            "version": 1,
            "label": v1.label(ANCHORS[position]),
        }
        for position in range(5)
    ]
    for bits in asked_five:
        add(1, "query_v1", names[0], bits, 1, order_v1, 1, True, correction_surface=True)

    # session 2: two re-answers, v2 blind anchors, v2 feedback, 11 + 6 queries.
    for position in repeat_two:
        source = turns[5 + position]
        add(
            2,
            "repeat_answered",
            source["entity"],
            tuple(int(c) for c in source["bits"]),
            1,
            order_v1,
            1,
            False,
            repeat_of=source["turn"],
        )
    anchor_v2_turns = []
    for position in anchor_v2:
        bits = ANCHORS[position]
        anchor_v2_turns.append(add(2, "anchor_blind_v2", names[1], bits, 2, order_v2, 1, True))
    batch_v2 = [
        {
            "source_id": f"{index}-fb-v2-{position}",
            "turn": anchor_v2_turns[position],
            "query": turns[anchor_v2_turns[position] - 1]["query"],
            "feedback": render_feedback(
                names[1], ANCHORS[position], order_v2, v2.label(ANCHORS[position])
            ),
            "version": 2,
            "label": v2.label(ANCHORS[position]),
        }
        for position in range(5)
    ]
    for bits in current_eleven:
        add(2, "current_v2", names[1], bits, 2, order_v2, 1, True, correction_surface=True)
    for bits in previous_six:
        add(2, "previous_v1", names[1], bits, 1, order_v1, 1, True, correction_surface=True)

    # session 3: new names and phrasing, same semantic scope.
    for bits in current_eleven:
        add(3, "repeat_v2", names[2], bits, 2, order_v3, 2, False)
    for bits in previous_six:
        add(3, "repeat_previous_v1", names[2], bits, 1, order_v3, 2, False)

    changed = [
        turn["turn"]
        for turn in turns
        if turn["kind"] == "current_v2"
        and v2.label(tuple(int(c) for c in turn["bits"]))
        != v1.label(tuple(int(c) for c in turn["bits"]))
    ]
    counterexamples = [
        item
        for item in batch_v2
        if item["label"] != v1.label(ANCHORS[int(item["source_id"].rsplit("-", 1)[1])])
    ]
    releases = [
        {"after_turn": FEEDBACK_RELEASE_TURNS[0], "version": 1, "items": batch_v1},
        {"after_turn": FEEDBACK_RELEASE_TURNS[1], "version": 2, "items": batch_v2},
    ]
    return {
        "world_id": f"w{index:03d}",
        "index": index,
        "seed": seed,
        "response": {
            "world_id": f"w{index:03d}",
            "names": names,
            "turns": turns,
            "releases": releases,
            "sessions": [
                {"session": 1, "start": 1, "end": 10},
                {"session": 2, "start": 11, "end": 34},
                {"session": 3, "start": 35, "end": 51},
            ],
        },
        "scoring": {
            "v1_rule": v1.signature(),
            "v2_rule": v2.signature(),
            "v1_rule_id": v1_index,
            "v2_rule_id": v2_index,
            "rule_version_change": v2_index != v1_index,
            "turns": scoring_turns,
            "changed_query": changed,
            "feedback_counterexamples": [item["source_id"] for item in counterexamples],
            "labels": {
                "v1": {bits_key(bits): v1.label(bits) for bits in combinations()},
                "v2": {bits_key(bits): v2.label(bits) for bits in combinations()},
            },
        },
    }


def response_only(world: dict[str, Any]) -> dict[str, Any]:
    """The exact view a graph may read; scoring truth is dropped."""
    return world["response"]


def shuffled_truth(world: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Same response, scrambled scoring block: constructive leak control."""
    scoring = dict(world["scoring"])
    scoring["v1_rule"] = rng.choice(RULES).signature()
    scoring["v2_rule"] = rng.choice(RULES).signature()
    scoring["rule_version_change"] = rng.choice([True, False])
    scoring["turns"] = {
        key: {
            "expected": rng.randrange(2),
            "expected_action": ACTIONS[rng.randrange(2)],
            "version": value["version"],
            "bits": value["bits"],
        }
        for key, value in world["scoring"]["turns"].items()
    }
    scoring["labels"] = {
        version: {bits_key(bits): rng.randrange(2) for bits in combinations()}
        for version in ("v1", "v2")
    }
    return {
        "world_id": world["world_id"],
        "index": world["index"],
        "seed": world["seed"],
        "response": world["response"],
        "scoring": scoring,
    }


def record_bytes(world: dict[str, Any]) -> int:
    return sum(
        len(item["feedback"].encode("utf-8"))
        for release in world["response"]["releases"]
        for item in release["items"]
    )


def example_stats(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {"v1": {"lines": 0, "chars": 0, "bytes": 0}, "v2": {"lines": 0, "chars": 0, "bytes": 0}}
    for world in worlds:
        for release in world["response"]["releases"]:
            key = "v1" if release["version"] == 1 else "v2"
            for item in release["items"]:
                rows[key]["lines"] += 1
                rows[key]["chars"] += len(item["feedback"])
                rows[key]["bytes"] += len(item["feedback"].encode("utf-8"))
    total = {
        "lines": sum(row["lines"] for row in rows.values()),
        "chars": sum(row["chars"] for row in rows.values()),
        "bytes": sum(row["bytes"] for row in rows.values()),
    }
    return {"examples": rows, "total": total, "answer_bytes": ANSWER_BYTES}


def turn_byte_stats(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for world in worlds:
        for turn in world["response"]["turns"]:
            key = turn["kind"]
            row = rows.setdefault(key, {"turns": 0, "bytes": 0})
            row["turns"] += 1
            row["bytes"] += len(turn["query"].encode("utf-8"))
    return rows


def fixture_digest(worlds: list[dict[str, Any]]) -> str:
    payload = "|".join(
        f"{world['world_id']}:{world['scoring']['v1_rule']}->{world['scoring']['v2_rule']}"
        for world in worlds
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

"""R5 evaluation helpers: holdout replay, mechanism variants, receipts, bootstrap.

This module is pure evaluation logic.  It does not open any manifest; the
runner supplies episodes and the frozen R5 run checkpoints.  The fixed-denominator
scorer lives in :mod:`taiji.native_content_evaluator`.

Definitions recorded before the first holdout read:

* predictions are produced by the object's declared answer API with a 48-byte
  generation allowance and the time intent mapped from the turn category;
* ``selected_value`` is the first non-empty step selection mapped back to the
  stored record value (the native core and the corrected baseline); the
  ordinary baseline has no state readout;
* the receipt ratio counts correct turns whose selected value appears verbatim
  in the expected answer (the declared exact-copy channel, not a new copy task);
* relevant swap replaces the selected fact's value and expects the sentence
  with the swapped value; unrelated swap replaces a distractor value and
  expects the original answer; cut skips the selected fact; rescue is the
  unmodified baseline replay;
* swaps stay inside the fact's attribute group (colour/place/owner), so the
  unrelated control is not an out-of-domain perturbation;
* mechanism cases are sampled with a fixed seed from baseline-correct turns
  only, because cut/rescue are defined conditional on a correct baseline.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .corrected_old_baseline import VALUES
from .native_content_evaluator import (
    ExpectedTurnV1,
    MechanismCaseV1,
    TurnPredictionV1,
)

TIME_FOR_CATEGORY = {
    "current_query": "current",
    "previous_query": "previous",
    "unknown_query": "unknown",
}
FACT_ROLES = frozenset({"fact_v0", "correction", "distractor"})
PREDICTION_BYTES = 48
VALUE_GROUPS = {
    "颜色": ("红色", "蓝色", "绿色", "黄色"),
    "位置": ("桌上", "抽屉", "柜子", "包里"),
    "归属": ("我的", "你的", "他的", "她的"),
}
OWNER_PRONOUN = {"我的": "我", "你的": "你", "他的": "他", "她的": "她"}


def episode_session(episode: Mapping[str, Any]) -> str:
    return str(episode["episode_id"]).replace(":", "_")


@dataclass(frozen=True)
class ReplayResultV1:
    predictions: dict[tuple[str, int], TurnPredictionV1]
    selected_content: dict[tuple[str, int], tuple[str, str, int] | None]
    target_bytes: int


@dataclass(frozen=True)
class MutationV1:
    """One fact-turn change: skip the fact or replace its value."""

    turn_index: int
    kind: str
    value: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"skip", "swap"}:
            raise ValueError("mutation kind is outside the declared set")
        if self.kind == "swap" and not self.value:
            raise ValueError("swap mutation requires a replacement value")


def _fact_revision(turn: Mapping[str, Any]) -> int:
    return 0 if turn["role"] in {"fact_v0", "distractor"} else 1


def _swapped_text(turn: Mapping[str, Any], value: str) -> str | None:
    text = str(turn["text"])
    old = turn.get("value")
    if not old:
        return None
    old = str(old)
    if old in text:
        return text.replace(old, value, 1)
    pronoun = OWNER_PRONOUN.get(old)
    new_pronoun = OWNER_PRONOUN.get(value, value)
    if pronoun and pronoun in text:
        return text.replace(pronoun, new_pronoun, 1)
    return None


def _ingest_native(
    bundle: Any,
    episode: Mapping[str, Any],
    turn: Mapping[str, Any],
    text: str,
    value: str,
    suffix: str,
) -> None:
    raw = text.encode("utf-8")
    bundle.ingest(
        raw_span=raw,
        source_input_id=f"{episode['episode_id']}:{turn['index']}:{suffix}",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        owner="eval",
        scope="eval",
        session=episode_session(episode),
        turn_index=int(turn["index"]),
        revision=_fact_revision(turn),
        entity=turn.get("entity"),
        attribute=turn.get("attribute"),
        value=value,
        status="resolved",
    )


def _ingest_corrected(
    bundle: Any,
    episode: Mapping[str, Any],
    turn: Mapping[str, Any],
    text: str,
    value: str,
    suffix: str,
) -> None:
    bundle.ingest(
        source_text=text,
        entity=turn.get("entity"),
        attribute=turn.get("attribute"),
        value=value,
        source_input_id=f"{episode['episode_id']}:{turn['index']}:{suffix}",
        owner="eval",
        scope="eval",
        session=episode_session(episode),
        turn_index=int(turn["index"]),
        revision=_fact_revision(turn),
    )


def replay_episode(
    adapter: Any,
    model: Any,
    episode: Mapping[str, Any],
    *,
    mutations: Mapping[int, MutationV1] | None = None,
) -> ReplayResultV1:
    """Replay one episode with the declared timeline and answer every scored turn."""

    mutations = mutations or {}
    session = episode_session(episode)
    bundle = adapter.fresh_bundle(model)
    predictions: dict[tuple[str, int], TurnPredictionV1] = {}
    selected_content: dict[tuple[str, int], tuple[str, str, int] | None] = {}
    target_bytes = 0
    for turn in episode["turns"]:
        index = int(turn["index"])
        mutation = mutations.get(index)
        if turn["role"] in FACT_ROLES:
            if mutation is not None and mutation.kind == "skip":
                continue
            text = str(turn["text"])
            value = str(turn.get("value"))
            suffix = "base"
            if mutation is not None and mutation.kind == "swap":
                swapped = _swapped_text(turn, str(mutation.value))
                if swapped is None:
                    raise ValueError("swap mutation cannot rewrite the fact text")
                text, value, suffix = swapped, str(mutation.value), "swap"
            if adapter.name == "new_core_v1":
                _ingest_native(bundle, episode, turn, text, value, suffix)
            elif adapter.name == "corrected_old_v1":
                _ingest_corrected(bundle, episode, turn, text, value, suffix)
            else:
                bundle.observe(text)
        if not turn["scored"]:
            if adapter.name == "ordinary_v1":
                bundle.observe(str(turn["text"]))
            continue
        query = str(turn["text"]).encode("utf-8")
        time_intent = TIME_FOR_CATEGORY[turn["category"]]
        key = (str(episode["episode_id"]), index)
        if adapter.name == "new_core_v1":
            answer = bundle.answer(
                query=query,
                time_intent=time_intent,
                owner="eval",
                scope="eval",
                session=session,
                max_new_bytes=PREDICTION_BYTES,
            )
            selection = _native_selection(bundle, answer)
            prediction = TurnPredictionV1(
                episode_id=key[0],
                turn_index=key[1],
                produced=answer.generated,
                parse_ok=True,
                selected_value=None if selection is None else selection[1],
            )
            selected_content[key] = selection
        elif adapter.name == "corrected_old_v1":
            answer = bundle.answer(
                query=query,
                time_intent=time_intent,
                owner="eval",
                scope="eval",
                session=session,
                max_new_bytes=PREDICTION_BYTES,
            )
            prediction = TurnPredictionV1(
                episode_id=key[0],
                turn_index=key[1],
                produced=answer.generated,
                parse_ok=answer.selection.parse.parsed,
                selected_value=answer.selection.value,
            )
            selected_content[key] = None
        else:
            produced = bundle.answer(
                query=query,
                time_intent=time_intent,
                max_new_bytes=PREDICTION_BYTES,
            )
            prediction = TurnPredictionV1(
                episode_id=key[0],
                turn_index=key[1],
                produced=produced,
                parse_ok=True,
                selected_value=None,
            )
            selected_content[key] = None
        predictions[key] = prediction
        target_bytes += len(bytes(turn["target"]["target_bytes_utf8"]))
        if adapter.name == "ordinary_v1":
            bundle.observe(str(turn["text"]))
    return ReplayResultV1(
        predictions=predictions,
        selected_content=selected_content,
        target_bytes=target_bytes,
    )


def _native_selection(bundle: Any, answer: Any) -> tuple[str, str, int] | None:
    ids = tuple(item for item in answer.selected_ids if item)
    if not ids:
        return None
    record = next((item for item in bundle.store.records if item.content_id == ids[0]), None)
    if record is None or record.value is None:
        return None
    return (record.content_id, str(record.value), int(record.turn_index))


def expected_turns(episodes: Sequence[Mapping[str, Any]]) -> list[ExpectedTurnV1]:
    rows: list[ExpectedTurnV1] = []
    for episode in episodes:
        for turn in episode["turns"]:
            if not turn["scored"]:
                continue
            rows.append(
                ExpectedTurnV1(
                    episode_id=str(episode["episode_id"]),
                    turn_index=int(turn["index"]),
                    category=str(turn["category"]),
                    expected=bytes(turn["target"]["target_bytes_utf8"]),
                )
            )
    return rows


def receipt_ratio(
    expected: Sequence[ExpectedTurnV1],
    predictions: Mapping[tuple[str, int], TurnPredictionV1],
) -> tuple[int, int]:
    """Correct turns whose selected value appears verbatim in the expected bytes."""

    cases = exact = 0
    for turn in expected:
        prediction = predictions.get(turn.key)
        if prediction is None or prediction.selected_value is None:
            continue
        if prediction.produced != turn.expected:
            continue
        cases += 1
        if str(prediction.selected_value).encode("utf-8") in turn.expected:
            exact += 1
    return exact, cases


def _alternative_value(attribute: str | None, original: str, *, seed: int) -> str:
    group = VALUE_GROUPS.get(str(attribute), VALUES)
    candidates = [value for value in group if value != original] or [
        value for value in VALUES if value != original
    ]
    return candidates[random.Random(seed).randrange(len(candidates))]


def mechanism_mutations(
    episode: Mapping[str, Any],
    turn_index: int,
    selected: tuple[str, str, int],
    *,
    value_seed: int,
) -> dict[str, Any]:
    """Cut / relevant swap / unrelated swap for one baseline-correct turn."""

    _, selected_value, source_turn = selected
    expected_turn = next(
        (turn for turn in episode["turns"] if int(turn["index"]) == turn_index), None
    )
    if expected_turn is None:
        raise ValueError("mechanism turn is not in the episode")
    expected = bytes(expected_turn["target"]["target_bytes_utf8"])
    if selected_value.encode("utf-8") not in expected:
        return {"cut": None, "relevant_swap": None, "unrelated_swap": None, "expected": None}
    fact_turn = next((turn for turn in episode["turns"] if int(turn["index"]) == source_turn), None)
    if fact_turn is None:
        return {"cut": None, "relevant_swap": None, "unrelated_swap": None, "expected": None}
    replacement = _alternative_value(fact_turn.get("attribute"), selected_value, seed=value_seed)
    relevant_expected = expected.replace(
        selected_value.encode("utf-8"), replacement.encode("utf-8")
    )
    unrelated_turn = next(
        (turn for turn in episode["turns"] if turn["role"] == "distractor" and turn.get("value")),
        None,
    )
    unrelated_mutation = None
    if unrelated_turn is not None:
        alternative = _alternative_value(
            unrelated_turn.get("attribute"), str(unrelated_turn["value"]), seed=value_seed + 1
        )
        if _swapped_text(unrelated_turn, alternative) is not None:
            unrelated_mutation = MutationV1(int(unrelated_turn["index"]), "swap", alternative)
    return {
        "cut": MutationV1(source_turn, "skip"),
        "relevant_swap": MutationV1(source_turn, "swap", replacement),
        "unrelated_swap": unrelated_mutation,
        "expected": relevant_expected,
    }


def mechanism_cases(
    episodes: Sequence[Mapping[str, Any]],
    baseline: Mapping[tuple[str, int], ReplayResultV1],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample baseline-correct cases with a declared value and a source fact turn."""

    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for turn in episode["turns"]:
            if not turn["scored"]:
                continue
            key = (str(episode["episode_id"]), int(turn["index"]))
            result = baseline[key]
            prediction = result.predictions[key]
            expected = bytes(turn["target"]["target_bytes_utf8"])
            if prediction.produced != expected:
                continue
            selected = result.selected_content.get(key)
            if selected is None or selected[1].encode("utf-8") not in expected:
                continue
            mutations = mechanism_mutations(
                episode, int(turn["index"]), selected, value_seed=seed + len(rows)
            )
            if mutations["cut"] is None or mutations["relevant_swap"] is None:
                continue
            rows.append(
                {
                    "episode": episode,
                    "turn_index": int(turn["index"]),
                    "expected": expected,
                    "selected": selected,
                    "mutations": mutations,
                }
            )
    if len(rows) > limit:
        chooser = random.Random(seed)
        rows = [rows[index] for index in sorted(chooser.sample(range(len(rows)), limit))]
    return rows


def build_mechanism_case(
    adapter: Any,
    model: Any,
    row: Mapping[str, Any],
    *,
    baseline_prediction: TurnPredictionV1,
) -> MechanismCaseV1:
    episode = row["episode"]
    turn_index = int(row["turn_index"])
    mutations = row["mutations"]
    key = (str(episode["episode_id"]), turn_index)
    cut = replay_episode(
        adapter,
        model,
        episode,
        mutations={int(mutations["cut"].turn_index): mutations["cut"]},
    )
    relevant = replay_episode(
        adapter,
        model,
        episode,
        mutations={int(mutations["relevant_swap"].turn_index): mutations["relevant_swap"]},
    )
    unrelated = (
        None
        if mutations["unrelated_swap"] is None
        else replay_episode(
            adapter,
            model,
            episode,
            mutations={int(mutations["unrelated_swap"].turn_index): mutations["unrelated_swap"]},
        )
    )
    return MechanismCaseV1(
        episode_id=key[0],
        turn_index=key[1],
        expected=bytes(row["expected"]),
        baseline=baseline_prediction,
        cut=cut.predictions[key],
        rescue=baseline_prediction,
        relevant_swap=relevant.predictions[key],
        relevant_expected=bytes(mutations["expected"]),
        unrelated_swap=None if unrelated is None else unrelated.predictions[key],
    )


def change_rate(
    baseline: Mapping[tuple[str, int], TurnPredictionV1],
    variant: Mapping[tuple[str, int], TurnPredictionV1],
    keys: Sequence[tuple[str, int]],
) -> float | None:
    if not keys:
        return None
    changed = sum(
        1
        for key in keys
        if key in baseline and key in variant and baseline[key].produced != variant[key].produced
    )
    return changed / len(keys)


def bootstrap_diff(
    left: Sequence[float],
    right: Sequence[float],
    *,
    draws: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired-free bootstrap of mean(left) - mean(right)."""

    if not left or not right:
        return {"draws": draws, "mean_diff": None, "lower_95": None, "upper_95": None}
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(draws):
        left_mean = sum(rng.choices(left, k=len(left))) / len(left)
        right_mean = sum(rng.choices(right, k=len(right))) / len(right)
        diffs.append(left_mean - right_mean)
    diffs.sort()
    return {
        "draws": draws,
        "mean_left": sum(left) / len(left),
        "mean_right": sum(right) / len(right),
        "mean_diff": sum(diffs) / draws,
        "lower_95": diffs[int(0.025 * draws)],
        "upper_95": diffs[int(0.975 * draws) - 1],
    }


def exact_vector(
    expected: Sequence[ExpectedTurnV1],
    predictions: Mapping[tuple[str, int], TurnPredictionV1],
) -> list[float]:
    rows: list[float] = []
    for turn in expected:
        prediction = predictions.get(turn.key)
        rows.append(
            1.0
            if prediction is not None
            and prediction.parse_ok
            and prediction.produced == turn.expected
            else 0.0
        )
    return rows

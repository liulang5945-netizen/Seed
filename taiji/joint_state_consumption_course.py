"""N177 E1 closed-world state-dependent query course.

Every family fixes one entity and one current colour. The same query text then
appears with different answers in different families, so a decoder that ignores
the persistent content state can only guess among eight colours. Only the train
split is materialized here; query and holdout families stay as identities until
the post-training scorer is allowed to open them.

The module reuses the N177.5 turn/episode/target dataclasses so the existing
source, partition, tag and stop-byte contracts stay identical.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .input_boundary import InputFrame
from .internalization import content_digest
from .joint_input_producer import ACT_NAMES, TAG_NAMES, JointSemanticTrainingTargetV1
from .joint_t1_course import (
    JOINT_T1_STOP_BYTES,
    JointT1AnswerTrainingTargetV1,
    JointT1EpisodeV1,
    JointT1TurnV1,
)

E1_COURSE_FORMAT = "taiji-n177-e1-state-consumption-course-v1"
E1_EPISODE_WIRE_FORMAT = "taiji-n177-e1-episode-wire-v1"
E1_COURSE_SEED = 1771029
E1_PARTITIONS = ("train", "query", "holdout")
E1_ENTITIES = ("杯子", "盒子", "钥匙", "书", "球", "笔", "手机", "眼镜")
E1_COLORS = ("红色", "蓝色", "绿色", "黄色", "黑色", "白色", "紫色", "橙色")
E1_TRAIN_VALUES_PER_ENTITY = 4
E1_QUERY_VALUES_PER_ENTITY = 2
E1_HOLDOUT_VALUES_PER_ENTITY = 2
E1_EPISODES_PER_FAMILY = 4
E1_TURNS_PER_EPISODE = 8
E1_ATTRIBUTE = "颜色"
E1_LEVEL = "F1"

_ASSERTION_TEMPLATES = (
    "{entity}的{attribute}是{value}。",
    "记一下：{entity}的{attribute}为{value}。",
    "我想告诉你，{entity}的{attribute}是{value}。",
    "现在记录，{entity}的{attribute}是{value}。",
)
_QUERY_TEMPLATES = (
    "{entity}的{attribute}是什么？",
    "请告诉我{entity}的{attribute}。",
    "你记得{entity}的{attribute}吗？",
    "现在回答：{entity}的{attribute}是什么？",
)
_CORRECTION_TEMPLATES = (
    "更正：{entity}的{attribute}改为{value}。",
    "刚才说错了，{entity}的{attribute}现在是{value}。",
    "请更新事实：{entity}的{attribute}为{value}。",
    "我要纠正一下，{entity}的{attribute}应为{value}。",
)
_DISTRACTOR_TEMPLATES = (
    "另一个事实：{entity}的{attribute}是{value}。",
    "再记一件事，{entity}的{attribute}为{value}。",
    "还有，{entity}的{attribute}是{value}。",
    "请单独记录：{entity}的{attribute}为{value}。",
)
_OLD_QUERY_TEMPLATES = (
    "{entity}之前是什么颜色？",
    "在更正之前，{entity}的颜色是什么？",
    "更正前，{entity}的颜色是什么？",
    "请回答：{entity}原来的颜色是什么？",
)
_AMBIGUOUS_TEMPLATES = (
    "{entity}的颜色不确定。",
    "我不确定{entity}的颜色。",
    "关于{entity}的颜色，我还不确定。",
    "{entity}的颜色现在不确定。",
)
_DELAY_TEMPLATES = (
    ("今天先聊到这里。", "好的。"),
    ("我们先停一下。", "好的。"),
    ("稍后再继续。", "好。"),
    ("先说到这儿。", "明白。"),
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_key(kind: str, value: Any) -> str:
    return content_digest({"seed": E1_COURSE_SEED, "kind": kind, "value": value})


@dataclass(frozen=True)
class E1FamilyV1:
    family_id: str
    entity: str
    value: str
    partition: str
    old_value: str
    token: str

    def __post_init__(self) -> None:
        if self.partition not in E1_PARTITIONS:
            raise ValueError("E1 family partition is unsupported")
        if self.entity not in E1_ENTITIES or self.value not in E1_COLORS:
            raise ValueError("E1 family entity/value is outside the frozen vocabulary")
        if self.old_value not in E1_COLORS or self.old_value == self.value:
            raise ValueError("E1 family old value must be a different legal colour")
        if not self.family_id.startswith("e1-family:") or not self.token:
            raise ValueError("E1 family identity is malformed")


def _ordered_colours(entity: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            E1_COLORS,
            key=lambda colour: _stable_key("entity-colour", f"{entity}|{colour}"),
        )
    )


def _old_value(family_id: str, value: str) -> str:
    candidates = tuple(colour for colour in E1_COLORS if colour != value)
    return tuple(
        sorted(candidates, key=lambda colour: _stable_key("old-value", f"{family_id}|{colour}"))
    )[0]


def _make_family(entity: str, value: str, partition: str) -> E1FamilyV1:
    digest = content_digest({"seed": E1_COURSE_SEED, "entity": entity, "value": value})
    family_id = f"e1-family:{digest[:20]}"
    return E1FamilyV1(
        family_id=family_id,
        entity=entity,
        value=value,
        partition=partition,
        old_value=_old_value(family_id, value),
        token=digest[:12],
    )


def e1_families() -> dict[str, tuple[E1FamilyV1, ...]]:
    """Return the fixed 32/16/16 family assignment without materializing text."""

    by_partition: dict[str, list[E1FamilyV1]] = {partition: [] for partition in E1_PARTITIONS}
    for entity in E1_ENTITIES:
        ordered = _ordered_colours(entity)
        train_values = ordered[:E1_TRAIN_VALUES_PER_ENTITY]
        query_values = ordered[
            E1_TRAIN_VALUES_PER_ENTITY : E1_TRAIN_VALUES_PER_ENTITY + E1_QUERY_VALUES_PER_ENTITY
        ]
        holdout_values = ordered[E1_TRAIN_VALUES_PER_ENTITY + E1_QUERY_VALUES_PER_ENTITY :]
        if (
            len(train_values) != E1_TRAIN_VALUES_PER_ENTITY
            or len(query_values) != E1_QUERY_VALUES_PER_ENTITY
            or len(holdout_values) != E1_HOLDOUT_VALUES_PER_ENTITY
        ):
            raise ValueError("E1 colour assignment has an unexpected shape")
        for partition, values in (
            ("train", train_values),
            ("query", query_values),
            ("holdout", holdout_values),
        ):
            by_partition[partition].extend(
                _make_family(entity, value, partition) for value in values
            )
    return {
        partition: tuple(
            sorted(rows, key=lambda family: _stable_key("family-order", family.family_id))
        )
        for partition, rows in by_partition.items()
    }


def _tagged_bytes(text: str, spans: Sequence[tuple[str, str]]) -> tuple[bytes, tuple[int, ...]]:
    raw = text.encode("utf-8")
    tags = [0] * len(raw)
    for role, fragment in spans:
        if role not in {"ENTITY", "ATTRIBUTE", "VALUE", "NEGATION"}:
            raise ValueError("unsupported E1 span role")
        start_char = text.index(fragment)
        if text.find(fragment, start_char + len(fragment)) >= 0:
            raise ValueError("E1 span fragment must occur exactly once")
        start_byte = len(text[:start_char].encode("utf-8"))
        span = fragment.encode("utf-8")
        tags[start_byte] = TAG_NAMES.index(f"B_{role}")
        for index in range(start_byte + 1, start_byte + len(span)):
            tags[index] = TAG_NAMES.index(f"I_{role}")
    return raw, tuple(tags)


def _make_turn(
    *,
    family: E1FamilyV1,
    episode_id: str,
    variant: int,
    turn_index: int,
    text: str,
    answer: str,
    act: str,
    spans: Sequence[tuple[str, str]] = (),
) -> JointT1TurnV1:
    raw, tag_ids = _tagged_bytes(text, spans)
    input_id = f"n177-e1:{family.token}:{family.partition}:{variant}:{turn_index}"
    frame = InputFrame(
        input_id=input_id,
        modality="text",
        payload=raw,
        source="n177.e1.state-consumption-course.v1",
        timestamp=turn_index,
        provenance="synthetic-e1-state-consumption",
    )
    source_sha = _sha256(raw)
    semantic_target = JointSemanticTrainingTargetV1.create(
        input_id=input_id,
        source_sha256=source_sha,
        family_id=family.family_id,
        partition=family.partition,
        tag_ids=tag_ids,
        act_id=ACT_NAMES.index(act),
    )
    answer_target = JointT1AnswerTrainingTargetV1.create(
        input_id=input_id,
        source_sha256=source_sha,
        family_id=family.family_id,
        partition=family.partition,
        target_bytes=answer.encode("utf-8") + JOINT_T1_STOP_BYTES,
    )
    return JointT1TurnV1(
        frame=frame,
        family_id=family.family_id,
        episode_id=episode_id,
        level=E1_LEVEL,
        partition=family.partition,
        turn_index=turn_index,
        semantic_target=semantic_target,
        answer_target=answer_target,
    )


def _distractor(family: E1FamilyV1, variant: int) -> tuple[str, str]:
    entities = tuple(entity for entity in E1_ENTITIES if entity != family.entity)
    other_entity = tuple(
        sorted(
            entities,
            key=lambda entity: _stable_key(
                "distractor-entity", f"{family.family_id}|{variant}|{entity}"
            ),
        )
    )[0]
    colours = tuple(
        colour for colour in E1_COLORS if colour not in {family.value, family.old_value}
    )
    other_value = tuple(
        sorted(
            colours,
            key=lambda colour: _stable_key(
                "distractor-colour", f"{family.family_id}|{variant}|{colour}"
            ),
        )
    )[0]
    return other_entity, other_value


def build_episode(family: E1FamilyV1, variant: int) -> JointT1EpisodeV1:
    if not 0 <= variant < E1_EPISODES_PER_FAMILY:
        raise ValueError("E1 episode variant is outside the fixed range")
    episode_id = f"n177-e1:{family.token}:{family.partition}:{variant}"
    query_text = _QUERY_TEMPLATES[variant].format(entity=family.entity, attribute=E1_ATTRIBUTE)
    other_entity, other_value = _distractor(family, variant)
    delay_prompt, delay_answer = _DELAY_TEMPLATES[variant]
    turns = (
        _make_turn(
            family=family,
            episode_id=episode_id,
            variant=variant,
            turn_index=0,
            text=_ASSERTION_TEMPLATES[variant].format(
                entity=family.entity, attribute=E1_ATTRIBUTE, value=family.old_value
            ),
            answer=f"我记下了，{family.entity}的{E1_ATTRIBUTE}是{family.old_value}。",
            act="assertion",
            spans=(
                ("ENTITY", family.entity),
                ("ATTRIBUTE", E1_ATTRIBUTE),
                ("VALUE", family.old_value),
            ),
        ),
        _make_turn(
            family=family,
            episode_id=episode_id,
            variant=variant,
            turn_index=1,
            text=delay_prompt,
            answer=delay_answer,
            act="ambiguous",
        ),
        _make_turn(
            family=family,
            episode_id=episode_id,
            variant=variant,
            turn_index=2,
            text=query_text,
            answer=f"{family.entity}的{E1_ATTRIBUTE}是{family.old_value}。",
            act="query",
            spans=(("ENTITY", family.entity), ("ATTRIBUTE", E1_ATTRIBUTE)),
        ),
        _make_turn(
            family=family,
            episode_id=episode_id,
            variant=variant,
            turn_index=3,
            text=_CORRECTION_TEMPLATES[variant].format(
                entity=family.entity, attribute=E1_ATTRIBUTE, value=family.value
            ),
            answer=f"收到，更正为{family.value}。",
            act="correction",
            spans=(
                ("ENTITY", family.entity),
                ("ATTRIBUTE", E1_ATTRIBUTE),
                ("VALUE", family.value),
            ),
        ),
        _make_turn(
            family=family,
            episode_id=episode_id,
            variant=variant,
            turn_index=4,
            text=query_text,
            answer=f"{family.entity}的{E1_ATTRIBUTE}是{family.value}。",
            act="query",
            spans=(("ENTITY", family.entity), ("ATTRIBUTE", E1_ATTRIBUTE)),
        ),
        _make_turn(
            family=family,
            episode_id=episode_id,
            variant=variant,
            turn_index=5,
            text=_DISTRACTOR_TEMPLATES[variant].format(
                entity=other_entity, attribute=E1_ATTRIBUTE, value=other_value
            ),
            answer=f"我记下了，{other_entity}的{E1_ATTRIBUTE}是{other_value}。",
            act="assertion",
            spans=(
                ("ENTITY", other_entity),
                ("ATTRIBUTE", E1_ATTRIBUTE),
                ("VALUE", other_value),
            ),
        ),
        _make_turn(
            family=family,
            episode_id=episode_id,
            variant=variant,
            turn_index=6,
            text=_OLD_QUERY_TEMPLATES[variant].format(entity=family.entity, attribute=E1_ATTRIBUTE),
            answer=f"之前是{family.old_value}。",
            act="query",
            spans=(("ENTITY", family.entity), ("ATTRIBUTE", E1_ATTRIBUTE)),
        ),
        _make_turn(
            family=family,
            episode_id=episode_id,
            variant=variant,
            turn_index=7,
            text=_AMBIGUOUS_TEMPLATES[variant].format(entity=family.entity),
            answer="不确定，我先不改事实。",
            act="ambiguous",
            spans=(
                ("ENTITY", family.entity),
                ("ATTRIBUTE", E1_ATTRIBUTE),
                ("NEGATION", "不确定"),
            ),
        ),
    )
    return JointT1EpisodeV1(
        episode_id=episode_id,
        family_id=family.family_id,
        level=E1_LEVEL,
        partition=family.partition,
        turns=turns,
    )


def build_train_episodes() -> tuple[JointT1EpisodeV1, ...]:
    """Materialize only the train split; sealed splits remain identities."""

    families = e1_families()["train"]
    return tuple(
        build_episode(family, variant)
        for family in families
        for variant in range(E1_EPISODES_PER_FAMILY)
    )


def build_fixture_episodes() -> tuple[JointT1EpisodeV1, ...]:
    """Public synthetic fixture drawn from the first two train families."""

    families = e1_families()["train"][:2]
    return tuple(
        build_episode(family, variant)
        for family in families
        for variant in range(E1_EPISODES_PER_FAMILY)
    )


def build_sealed_split(partition: str) -> tuple[JointT1EpisodeV1, ...]:
    """Refuse to materialize query/holdout before the post-training scorer."""

    if partition in {"query", "holdout"}:
        raise ValueError(
            f"E1 {partition} text is sealed until the post-training scorer is admitted"
        )
    if partition != "train":
        raise ValueError("unsupported E1 split")
    return build_train_episodes()


def _family_pair_key(family: E1FamilyV1) -> tuple[str, str]:
    return family.entity, family.value


def audit_course() -> dict[str, Any]:
    families = e1_families()
    train_ids = {family.family_id for family in families["train"]}
    query_ids = {family.family_id for family in families["query"]}
    holdout_ids = {family.family_id for family in families["holdout"]}
    all_ids = train_ids | query_ids | holdout_ids
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "family_counts",
        all(
            len(families[partition]) == expected
            for partition, expected in (
                ("train", 32),
                ("query", 16),
                ("holdout", 16),
            )
        ),
        {partition: len(families[partition]) for partition in E1_PARTITIONS},
    )
    check("family_ids_unique", len(all_ids) == 64, {"unique": len(all_ids)})
    check(
        "splits_disjoint",
        not (train_ids & query_ids)
        and not (train_ids & holdout_ids)
        and not (query_ids & holdout_ids),
        {"train": len(train_ids), "query": len(query_ids), "holdout": len(holdout_ids)},
    )

    per_entity: dict[str, dict[str, tuple[str, ...]]] = {}
    for entity in E1_ENTITIES:
        per_entity[entity] = {
            partition: tuple(
                sorted(family.value for family in families[partition] if family.entity == entity)
            )
            for partition in E1_PARTITIONS
        }
    check(
        "every_entity_in_every_split",
        all(
            len(per_entity[entity][partition]) == expected
            for entity in E1_ENTITIES
            for partition, expected in (
                ("train", E1_TRAIN_VALUES_PER_ENTITY),
                ("query", E1_QUERY_VALUES_PER_ENTITY),
                ("holdout", E1_HOLDOUT_VALUES_PER_ENTITY),
            )
        ),
        {
            entity: {partition: len(values) for partition, values in per_entity[entity].items()}
            for entity in E1_ENTITIES
        },
    )
    check(
        "entity_values_disjoint_across_splits",
        all(
            not (set(per_entity[entity]["train"]) & set(per_entity[entity]["query"]))
            and not (set(per_entity[entity]["train"]) & set(per_entity[entity]["holdout"]))
            and not (set(per_entity[entity]["query"]) & set(per_entity[entity]["holdout"]))
            for entity in E1_ENTITIES
        ),
        {entity: per_entity[entity] for entity in E1_ENTITIES},
    )

    pair_keys = {
        partition: {_family_pair_key(family) for family in families[partition]}
        for partition in E1_PARTITIONS
    }
    check(
        "complete_entity_value_pairs_disjoint",
        not (pair_keys["train"] & pair_keys["query"])
        and not (pair_keys["train"] & pair_keys["holdout"])
        and not (pair_keys["query"] & pair_keys["holdout"]),
        {partition: len(pair_keys[partition]) for partition in E1_PARTITIONS},
    )
    check(
        "old_value_differs_from_current",
        all(family.old_value != family.value for rows in families.values() for family in rows),
        {"checked": 64},
    )
    return {"ok": all(item["passed"] for item in checks), "checks": checks}


def e1_course_contract() -> dict[str, Any]:
    families = e1_families()
    payload = {
        "format": E1_COURSE_FORMAT,
        "version": 1,
        "seed": E1_COURSE_SEED,
        "partitions": list(E1_PARTITIONS),
        "entities": list(E1_ENTITIES),
        "colours": list(E1_COLORS),
        "attribute": E1_ATTRIBUTE,
        "families_per_split": {
            "train": len(families["train"]),
            "query": len(families["query"]),
            "holdout": len(families["holdout"]),
        },
        "episodes_per_family": E1_EPISODES_PER_FAMILY,
        "turns_per_episode": E1_TURNS_PER_EPISODE,
        "sealed_materialized": False,
        "train_family_ids": [family.family_id for family in families["train"]],
        "query_family_ids": [family.family_id for family in families["query"]],
        "holdout_family_ids": [family.family_id for family in families["holdout"]],
        "family_assignments": {
            entity: {
                partition: sorted(
                    family.value for family in families[partition] if family.entity == entity
                )
                for partition in E1_PARTITIONS
            }
            for entity in E1_ENTITIES
        },
    }
    return {**payload, "course_sha256": content_digest(payload)}


def e1_split_digest(episodes: Sequence[JointT1EpisodeV1]) -> str:
    if not episodes:
        raise ValueError("E1 split digest requires a nonempty split")
    return content_digest(
        {
            "course_sha256": e1_course_contract()["course_sha256"],
            "partition": episodes[0].partition,
            "episode_digests": [content_digest(episode.to_payload()) for episode in episodes],
        }
    )


def episode_to_json(episode: JointT1EpisodeV1) -> dict[str, Any]:
    """Serialize one episode with explicit UTF-8 payload hex, never raw bytes."""

    payload = episode.to_payload()
    for turn in payload["turns"]:
        frame = turn["frame"]
        raw = bytes(frame.pop("payload"))
        frame["payload_hex"] = raw.hex()
        frame["payload_encoding"] = "hex"
    return {"wire_format": E1_EPISODE_WIRE_FORMAT, "episode": payload}


def episode_from_json(wire: Mapping[str, Any]) -> JointT1EpisodeV1:
    """Read the explicit hex wire form and reject any other encoding."""

    if wire.get("wire_format") != E1_EPISODE_WIRE_FORMAT:
        raise ValueError("E1 episode wire format differs")
    payload = dict(wire["episode"])
    turns = []
    for turn in payload["turns"]:
        turn = dict(turn)
        frame = dict(turn["frame"])
        if frame.get("payload_encoding") != "hex" or "payload_hex" not in frame:
            raise ValueError("E1 episode frame lacks explicit hex payload")
        frame["payload"] = bytes.fromhex(str(frame.pop("payload_hex")))
        frame.pop("payload_encoding")
        turn["frame"] = frame
        turns.append(turn)
    payload["turns"] = turns
    return JointT1EpisodeV1.from_payload(payload)


def e1_turn_text(turn: JointT1TurnV1) -> tuple[str, str]:
    """Decode one turn's input and target for public train/fixture inspection."""

    return (
        bytes(turn.frame.payload).decode("utf-8"),
        turn.answer_target.target_bytes[: -len(JOINT_T1_STOP_BYTES)].decode("utf-8"),
    )

"""Deterministic, family-isolated starter course for the N177 T1 slice.

The builder exposes one split at a time. Training callers request only
``train``; query and holdout targets are materialized by the post-run scorer.
All examples are synthetic, short Chinese exchanges and carry byte-aligned
semantic targets tied to their exact input bytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from .input_boundary import InputFrame
from .internalization import content_digest
from .joint_input_producer import (
    ACT_NAMES,
    TAG_NAMES,
    JointSemanticTrainingTargetV1,
)

JOINT_T1_COURSE_FORMAT = "taiji-joint-t1-course-v1"
JOINT_T1_ANSWER_TARGET_FORMAT = "taiji-joint-t1-answer-target-v1"
JOINT_T1_TURN_FORMAT = "taiji-joint-t1-turn-v1"
JOINT_T1_EPISODE_FORMAT = "taiji-joint-t1-episode-v1"
JOINT_T1_COURSE_SEED = 17751027
JOINT_T1_UPDATE_COUNT = 2560
JOINT_T1_BATCH_SIZE = 4
JOINT_T1_F0_ONLY_UPDATES = 640
JOINT_T1_LEVELS = ("F0", "F1")
JOINT_T1_PARTITIONS = ("train", "query", "holdout")
JOINT_T1_STOP_BYTES = b"<|end|>"

_TRAIN_EPISODES_PER_FAMILY = 5
_EVAL_EPISODES_PER_FAMILY = 4
_TRAIN_FAMILIES_PER_LEVEL = 64
_EVAL_FAMILIES_PER_SPLIT = 32

_NAMES = (
    "小林",
    "小周",
    "小陈",
    "小李",
    "小吴",
    "小郑",
    "小王",
    "小赵",
)
_F0_ITEMS = (
    "雨伞",
    "水杯",
    "书本",
    "钥匙",
    "背包",
    "帽子",
    "毛巾",
    "玩具",
    "雨衣",
    "水壶",
    "笔记本",
    "围巾",
    "手套",
    "饭盒",
    "球拍",
    "眼镜",
)
_F1_ENTITIES = (
    "杯子",
    "苹果",
    "书本",
    "雨伞",
    "钥匙",
    "背包",
    "毛巾",
    "玩具",
    "水壶",
    "帽子",
    "饭盒",
    "花瓶",
)
_F1_ATTRIBUTE_VALUES = {
    "颜色": ("红色", "蓝色", "绿色", "黄色"),
    "位置": ("桌上", "柜子里", "架子上", "门边"),
    "状态": ("打开", "关闭", "干净", "湿了"),
}

_F0_TURNS: tuple[tuple[tuple[str, str], ...], ...] = (
    (
        ("你好，我叫{name}，今天带了{item}。", "你好，{name}！"),
        ("嗨，我是{name}，这是我的{item}。", "嗨，{name}！"),
        ("见到你真好，我是{name}，拿着{item}。", "你好，{name}，很高兴见到你。"),
        ("我叫{name}，手边有{item}，先和你打个招呼。", "你好，{name}！"),
        ("你好呀，我是{name}，带来的东西是{item}。", "你好，{name}，我听到了。"),
    ),
    (
        ("我是{name}，想和你聊聊{item}。", "好呀，{name}，我们聊聊{item}。"),
        ("{name}想说说{item}，你愿意听吗？", "愿意，我们可以聊聊{item}。"),
        ("我叫{name}，今天想谈一谈{item}。", "好的，{name}，我们谈谈{item}。"),
        ("{name}在这里，想聊聊带来的{item}。", "好，{name}，我们聊聊它。"),
        ("可以陪{name}聊一会儿{item}吗？", "当然可以，我们聊聊{item}。"),
    ),
    (
        ("请简短回应：{name}带来了{item}。", "好的，{name}带来了{item}。"),
        ("请用一句短话告诉{name}：{item}很实用。", "{name}，{item}很实用。"),
        ("{name}拿着{item}，请简单说一句。", "好的，{name}拿着{item}。"),
        ("帮{name}确认一下：东西是{item}。", "确认了，东西是{item}。"),
        ("请简短回答{name}：你听到了什么东西？{item}。", "我听到的是{item}。"),
    ),
    (
        ("我叫{name}，今天带来的东西是{item}。请用短句回答是什么。", "你带来的是{item}。"),
        ("{name}带着{item}，请用一个短句重复东西名称。", "东西名称是{item}。"),
        ("请回答{name}带来的物品：{item}。", "物品是{item}。"),
        ("我是{name}，我手边有{item}。请简短说出物品。", "你手边有{item}。"),
        ("{name}想确认物品名称，刚才提到的是{item}。", "刚才提到的是{item}。"),
    ),
    (
        ("今天先聊到这里。再见，我是{name}，带着{item}。", "再见，{name}！下次再聊{item}。"),
        ("{name}要结束聊天了，{item}也带好了。", "好的，再见，{name}！"),
        ("我要走了，我叫{name}，我的东西是{item}。", "再见，{name}，祝你顺利。"),
        ("今天的聊天结束吧。{name}拿着{item}，先说再见。", "再见，{name}！"),
        ("谢谢你陪{name}聊{item}，现在说再见。", "不客气，再见，{name}！"),
    ),
)

_F1_ASSERTION_TEMPLATES = (
    "{entity}的{attribute}是{value}。",
    "记一下：{entity}的{attribute}为{value}。",
    "我想告诉你，{entity}的{attribute}是{value}。",
    "请记住这个事实：{entity}的{attribute}为{value}。",
    "现在记录，{entity}的{attribute}是{value}。",
)
_F1_QUERY_TEMPLATES = (
    "{entity}的{attribute}是什么？",
    "请告诉我{entity}的{attribute}。",
    "你记得{entity}的{attribute}吗？",
    "我想确认一下，{entity}的{attribute}是什么？",
    "现在回答：{entity}的{attribute}是什么？",
)
_F1_CORRECTION_TEMPLATES = (
    "更正：{entity}的{attribute}改为{value}。",
    "刚才说错了，{entity}的{attribute}现在是{value}。",
    "请更新事实：{entity}的{attribute}为{value}。",
    "我要纠正一下，{entity}的{attribute}应为{value}。",
    "新信息：{entity}的{attribute}是{value}，替换旧值。",
)
_F1_UNRELATED_TEMPLATES = (
    "另一个事实：{entity}的{attribute}是{value}。",
    "再记一件事，{entity}的{attribute}为{value}。",
    "还有，{entity}的{attribute}是{value}。",
    "请单独记录：{entity}的{attribute}为{value}。",
    "补充一个事实，{entity}的{attribute}是{value}。",
)
_F1_NEGATIVE_TEMPLATES = (
    "{entity}的{attribute}不是{value}。",
    "请注意，{entity}的{attribute}不是{value}。",
    "我想说明，{entity}的{attribute}不是{value}。",
    "{entity}的{attribute}可不是{value}。",
    "不要记成肯定事实：{entity}的{attribute}不是{value}。",
)
_F1_AMBIGUOUS_TEMPLATES = (
    "也许{entity}的{attribute}是{value}。",
    "我不确定，{entity}的{attribute}可能是{value}。",
    "有人猜{entity}的{attribute}为{value}，但还不确定。",
    "{entity}的{attribute}似乎是{value}。",
    "暂时不确定{entity}的{attribute}是不是{value}。",
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_key(kind: str, value: Any) -> tuple[str, str]:
    encoded = content_digest({"seed": JOINT_T1_COURSE_SEED, "kind": kind, "value": value})
    return encoded, str(value)


def _split_family_ids(family_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
    ordered = tuple(sorted(family_ids, key=lambda item: _stable_key("family", item)))
    expected = _TRAIN_FAMILIES_PER_LEVEL + 2 * _EVAL_FAMILIES_PER_SPLIT
    if len(ordered) != expected or len(set(ordered)) != expected:
        raise ValueError("N177 T1 course requires exactly 128 unique families per level")
    return {
        "train": ordered[:_TRAIN_FAMILIES_PER_LEVEL],
        "query": ordered[
            _TRAIN_FAMILIES_PER_LEVEL : _TRAIN_FAMILIES_PER_LEVEL + _EVAL_FAMILIES_PER_SPLIT
        ],
        "holdout": ordered[_TRAIN_FAMILIES_PER_LEVEL + _EVAL_FAMILIES_PER_SPLIT :],
    }


@dataclass(frozen=True)
class JointT1AnswerTrainingTargetV1:
    """Answer bytes supervised only for the exact training input that produced them."""

    input_id: str
    source_sha256: str
    family_id: str
    partition: str
    target_bytes: bytes
    target_sha256: str

    FORMAT = JOINT_T1_ANSWER_TARGET_FORMAT

    def __post_init__(self) -> None:
        for name in ("input_id", "source_sha256", "family_id", "target_sha256"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"answer target {name} must be nonempty")
        if self.partition not in JOINT_T1_PARTITIONS:
            raise ValueError("answer target partition is unsupported")
        if not isinstance(self.target_bytes, bytes) or not self.target_bytes:
            raise ValueError("answer target bytes must be nonempty")
        try:
            self.target_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("answer target must be valid UTF-8") from error
        if not self.target_bytes.endswith(JOINT_T1_STOP_BYTES):
            raise ValueError("answer target must end with the fixed V4 stop string")
        if self.target_sha256 != content_digest(self._payload_without_digest()):
            raise ValueError("answer target digest differs")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "input_id": self.input_id,
            "source_sha256": self.source_sha256,
            "family_id": self.family_id,
            "partition": self.partition,
            "target_hex": self.target_bytes.hex(),
        }

    @classmethod
    def create(
        cls,
        *,
        input_id: str,
        source_sha256: str,
        family_id: str,
        partition: str,
        target_bytes: bytes,
    ) -> JointT1AnswerTrainingTargetV1:
        shell = {
            "format": cls.FORMAT,
            "version": 1,
            "input_id": input_id,
            "source_sha256": source_sha256,
            "family_id": family_id,
            "partition": partition,
            "target_hex": target_bytes.hex(),
        }
        return cls(
            input_id=input_id,
            source_sha256=source_sha256,
            family_id=family_id,
            partition=partition,
            target_bytes=target_bytes,
            target_sha256=content_digest(shell),
        )

    def validate_for_fit(
        self,
        *,
        input_id: str,
        input_bytes: bytes,
        family_id: str,
    ) -> None:
        if self.partition != "train":
            raise ValueError("query/holdout answer targets cannot reach the fit API")
        if (self.input_id, self.source_sha256, self.family_id) != (
            input_id,
            _sha256(input_bytes),
            family_id,
        ):
            raise ValueError("answer target is not bound to the exact training source/family")
        if self.target_sha256 != content_digest(self._payload_without_digest()):
            raise ValueError("answer target changed after binding")

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "target_sha256": self.target_sha256}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointT1AnswerTrainingTargetV1:
        required = {
            "format",
            "version",
            "input_id",
            "source_sha256",
            "family_id",
            "partition",
            "target_hex",
            "target_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("answer target payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported answer target version")
        try:
            target_bytes = bytes.fromhex(str(payload["target_hex"]))
        except ValueError as error:
            raise ValueError("answer target hex encoding is invalid") from error
        return cls(
            input_id=str(payload["input_id"]),
            source_sha256=str(payload["source_sha256"]),
            family_id=str(payload["family_id"]),
            partition=str(payload["partition"]),
            target_bytes=target_bytes,
            target_sha256=str(payload["target_sha256"]),
        )


@dataclass(frozen=True)
class JointT1TurnV1:
    frame: InputFrame
    family_id: str
    episode_id: str
    level: str
    partition: str
    turn_index: int
    semantic_target: JointSemanticTrainingTargetV1
    answer_target: JointT1AnswerTrainingTargetV1

    FORMAT = JOINT_T1_TURN_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.frame, InputFrame):
            raise TypeError("T1 turn requires an InputFrame")
        if self.frame.modality not in {"text", "text-utf8"}:
            raise ValueError("T1 turn must carry raw UTF-8 text")
        if self.level not in JOINT_T1_LEVELS or self.partition not in JOINT_T1_PARTITIONS:
            raise ValueError("T1 turn level or partition is unsupported")
        if self.turn_index < 0:
            raise ValueError("T1 turn index cannot be negative")
        source_sha = _sha256(bytes(self.frame.payload))
        if (
            self.semantic_target.input_id,
            self.semantic_target.source_sha256,
            self.semantic_target.family_id,
            self.semantic_target.partition,
        ) != (self.frame.input_id, source_sha, self.family_id, self.partition):
            raise ValueError("semantic target differs from its source turn")
        if (
            self.answer_target.input_id,
            self.answer_target.source_sha256,
            self.answer_target.family_id,
            self.answer_target.partition,
        ) != (self.frame.input_id, source_sha, self.family_id, self.partition):
            raise ValueError("answer target differs from its source turn")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "frame": self.frame.to_payload(),
            "family_id": self.family_id,
            "episode_id": self.episode_id,
            "level": self.level,
            "partition": self.partition,
            "turn_index": self.turn_index,
            "semantic_target": self.semantic_target.to_payload(),
            "answer_target": self.answer_target.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointT1TurnV1:
        required = {
            "format",
            "version",
            "frame",
            "family_id",
            "episode_id",
            "level",
            "partition",
            "turn_index",
            "semantic_target",
            "answer_target",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("T1 turn payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported T1 turn version")
        return cls(
            frame=InputFrame.from_payload(payload["frame"]),
            family_id=str(payload["family_id"]),
            episode_id=str(payload["episode_id"]),
            level=str(payload["level"]),
            partition=str(payload["partition"]),
            turn_index=int(payload["turn_index"]),
            semantic_target=JointSemanticTrainingTargetV1.from_payload(payload["semantic_target"]),
            answer_target=JointT1AnswerTrainingTargetV1.from_payload(payload["answer_target"]),
        )


@dataclass(frozen=True)
class JointT1EpisodeV1:
    episode_id: str
    family_id: str
    level: str
    partition: str
    turns: tuple[JointT1TurnV1, ...]

    FORMAT = JOINT_T1_EPISODE_FORMAT

    def __post_init__(self) -> None:
        if not self.episode_id or not self.family_id:
            raise ValueError("T1 episode and family IDs must be nonempty")
        if self.level not in JOINT_T1_LEVELS or self.partition not in JOINT_T1_PARTITIONS:
            raise ValueError("T1 episode level or partition is unsupported")
        expected_turns = 5 if self.level == "F0" else 8
        if len(self.turns) != expected_turns:
            raise ValueError("T1 episode has the wrong fixed turn count")
        if any(
            (turn.episode_id, turn.family_id, turn.level, turn.partition, turn.turn_index)
            != (self.episode_id, self.family_id, self.level, self.partition, index)
            for index, turn in enumerate(self.turns)
        ):
            raise ValueError("T1 episode turn identity/order differs")
        input_ids = tuple(turn.frame.input_id for turn in self.turns)
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("T1 episode repeats an input ID")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "episode_id": self.episode_id,
            "family_id": self.family_id,
            "level": self.level,
            "partition": self.partition,
            "turns": [turn.to_payload() for turn in self.turns],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointT1EpisodeV1:
        required = {"format", "version", "episode_id", "family_id", "level", "partition", "turns"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("T1 episode payload schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported T1 episode version")
        return cls(
            episode_id=str(payload["episode_id"]),
            family_id=str(payload["family_id"]),
            level=str(payload["level"]),
            partition=str(payload["partition"]),
            turns=tuple(JointT1TurnV1.from_payload(item) for item in payload["turns"]),
        )


def _family_digest(level: str, values: Sequence[str]) -> str:
    digest = content_digest({"level": level, "values": list(values)})
    return f"{level.lower()}-family:{digest[:20]}"


def _f0_families() -> dict[str, tuple[str, ...]]:
    combinations = tuple(product(_NAMES, _F0_ITEMS))
    family_ids = tuple(_family_digest("F0", pair) for pair in combinations)
    splits = _split_family_ids(family_ids)
    return {partition: splits[partition] for partition in splits}


def _f1_primary_combinations() -> tuple[tuple[str, str, str, str], ...]:
    combinations = []
    for entity in _F1_ENTITIES:
        for attribute, values in _F1_ATTRIBUTE_VALUES.items():
            for old_value in values:
                for new_value in values:
                    if old_value != new_value:
                        combinations.append((entity, attribute, old_value, new_value))
    return tuple(sorted(combinations, key=lambda item: _stable_key("F1-primary", item)))


def _f1_distractor_combinations() -> tuple[tuple[str, str, str], ...]:
    combinations = tuple(
        (entity, attribute, value)
        for entity in _F1_ENTITIES
        for attribute, values in _F1_ATTRIBUTE_VALUES.items()
        for value in values
    )
    return tuple(sorted(combinations, key=lambda item: _stable_key("F1-distractor", item)))


def _assign_f1_distractors(
    primaries: Sequence[tuple[str, str, str, str]],
) -> tuple[tuple[str, str, str], ...]:
    candidates = _f1_distractor_combinations()
    owner_by_candidate: dict[tuple[str, str, str], int] = {}
    assigned: dict[int, tuple[str, str, str]] = {}

    def place(primary_index: int, seen: set[tuple[str, str, str]]) -> bool:
        primary_entity = primaries[primary_index][0]
        for candidate in candidates:
            if candidate[0] == primary_entity or candidate in seen:
                continue
            seen.add(candidate)
            previous = owner_by_candidate.get(candidate)
            if previous is None or place(previous, seen):
                owner_by_candidate[candidate] = primary_index
                assigned[primary_index] = candidate
                return True
        return False

    for index in range(len(primaries)):
        if not place(index, set()):
            raise ValueError("could not assign unique unrelated facts to F1 families")
    return tuple(assigned[index] for index in range(len(primaries)))


def _f1_families() -> dict[str, tuple[tuple[str, str, str, str, str, str, str], ...]]:
    primary = _f1_primary_combinations()[:128]
    distractors = _assign_f1_distractors(primary)
    family_rows = tuple(
        (
            *primary_row,
            *distractor,
        )
        for primary_row, distractor in zip(primary, distractors, strict=True)
    )
    family_ids = tuple(_family_digest("F1", row) for row in family_rows)
    splits = _split_family_ids(family_ids)
    by_id = {_family_digest("F1", row): row for row in family_rows}
    return {partition: tuple(by_id[item] for item in splits[partition]) for partition in splits}


def _tagged_bytes(text: str, spans: Sequence[tuple[str, str]]) -> tuple[bytes, tuple[int, ...]]:
    raw = text.encode("utf-8")
    tags = [0] * len(raw)
    for role, fragment in spans:
        if role not in {"ENTITY", "ATTRIBUTE", "VALUE", "NEGATION"}:
            raise ValueError("unsupported T1 span role")
        start_char = text.index(fragment)
        if text.find(fragment, start_char + len(fragment)) >= 0:
            raise ValueError("T1 span fragment must occur exactly once in its source text")
        start_byte = len(text[:start_char].encode("utf-8"))
        span = fragment.encode("utf-8")
        begin = TAG_NAMES.index(f"B_{role}")
        inside = TAG_NAMES.index(f"I_{role}")
        tags[start_byte] = begin
        for index in range(start_byte + 1, start_byte + len(span)):
            tags[index] = inside
    return raw, tuple(tags)


def _make_turn(
    *,
    family_id: str,
    episode_id: str,
    level: str,
    partition: str,
    turn_index: int,
    text: str,
    answer: str,
    act: str,
    spans: Sequence[tuple[str, str]] = (),
) -> JointT1TurnV1:
    raw, tag_ids = _tagged_bytes(text, spans)
    input_id = f"n177-t1:{episode_id}:{turn_index}"
    frame = InputFrame(
        input_id=input_id,
        modality="text",
        payload=raw,
        source="n177.t1.synthetic-course.v1",
        timestamp=turn_index,
        provenance="synthetic-course",
    )
    source_sha = _sha256(raw)
    semantic_target = JointSemanticTrainingTargetV1.create(
        input_id=input_id,
        source_sha256=source_sha,
        family_id=family_id,
        partition=partition,
        tag_ids=tag_ids,
        act_id=ACT_NAMES.index(act),
    )
    answer_target = JointT1AnswerTrainingTargetV1.create(
        input_id=input_id,
        source_sha256=source_sha,
        family_id=family_id,
        partition=partition,
        target_bytes=answer.encode("utf-8") + JOINT_T1_STOP_BYTES,
    )
    return JointT1TurnV1(
        frame=frame,
        family_id=family_id,
        episode_id=episode_id,
        level=level,
        partition=partition,
        turn_index=turn_index,
        semantic_target=semantic_target,
        answer_target=answer_target,
    )


def _build_f0_episode(
    *,
    family_id: str,
    partition: str,
    variant: int,
    name: str,
    item: str,
) -> JointT1EpisodeV1:
    episode_id = f"n177-t1:f0:{family_id.rsplit(':', 1)[-1]}:{partition}:{variant}"
    turns = []
    for turn_index, templates in enumerate(_F0_TURNS):
        prompt_template, answer_template = templates[(variant + turn_index) % len(templates)]
        turns.append(
            _make_turn(
                family_id=family_id,
                episode_id=episode_id,
                level="F0",
                partition=partition,
                turn_index=turn_index,
                text=prompt_template.format(name=name, item=item),
                answer=answer_template.format(name=name, item=item),
                act="ambiguous",
            )
        )
    return JointT1EpisodeV1(episode_id, family_id, "F0", partition, tuple(turns))


def _build_f1_episode(
    *,
    family_id: str,
    partition: str,
    variant: int,
    entity: str,
    attribute: str,
    old_value: str,
    new_value: str,
    distractor_entity: str,
    distractor_attribute: str,
    distractor_value: str,
) -> JointT1EpisodeV1:
    episode_id = f"n177-t1:f1:{family_id.rsplit(':', 1)[-1]}:{partition}:{variant}"
    values = _F1_ATTRIBUTE_VALUES[attribute]
    alternatives = tuple(value for value in values if value not in {old_value, new_value})
    if len(alternatives) != 2:
        raise ValueError("F1 family must leave two distinct negative/ambiguous values")
    negative_value, ambiguous_value = alternatives
    index = variant % len(_F1_ASSERTION_TEMPLATES)
    assert_text = _F1_ASSERTION_TEMPLATES[index].format(
        entity=entity, attribute=attribute, value=old_value
    )
    query_old = _F1_QUERY_TEMPLATES[index].format(entity=entity, attribute=attribute)
    correction_text = _F1_CORRECTION_TEMPLATES[index].format(
        entity=entity, attribute=attribute, value=new_value
    )
    unrelated_text = _F1_UNRELATED_TEMPLATES[index].format(
        entity=distractor_entity,
        attribute=distractor_attribute,
        value=distractor_value,
    )
    negative_text = _F1_NEGATIVE_TEMPLATES[index].format(
        entity=entity, attribute=attribute, value=negative_value
    )
    ambiguous_text = _F1_AMBIGUOUS_TEMPLATES[index].format(
        entity=entity, attribute=attribute, value=ambiguous_value
    )
    prompt_value = f"{entity}的{attribute}是"
    turns = (
        _make_turn(
            family_id=family_id,
            episode_id=episode_id,
            level="F1",
            partition=partition,
            turn_index=0,
            text=assert_text,
            answer=f"我记下了，{entity}的{attribute}是{old_value}。",
            act="assertion",
            spans=(("ENTITY", entity), ("ATTRIBUTE", attribute), ("VALUE", old_value)),
        ),
        _make_turn(
            family_id=family_id,
            episode_id=episode_id,
            level="F1",
            partition=partition,
            turn_index=1,
            text=query_old,
            answer=f"{prompt_value}{old_value}。",
            act="query",
            spans=(("ENTITY", entity), ("ATTRIBUTE", attribute)),
        ),
        _make_turn(
            family_id=family_id,
            episode_id=episode_id,
            level="F1",
            partition=partition,
            turn_index=2,
            text=correction_text,
            answer=f"收到，更正为{new_value}。",
            act="correction",
            spans=(("ENTITY", entity), ("ATTRIBUTE", attribute), ("VALUE", new_value)),
        ),
        _make_turn(
            family_id=family_id,
            episode_id=episode_id,
            level="F1",
            partition=partition,
            turn_index=3,
            text=_F1_QUERY_TEMPLATES[(index + 1) % 5].format(entity=entity, attribute=attribute),
            answer=f"{prompt_value}{new_value}。",
            act="query",
            spans=(("ENTITY", entity), ("ATTRIBUTE", attribute)),
        ),
        _make_turn(
            family_id=family_id,
            episode_id=episode_id,
            level="F1",
            partition=partition,
            turn_index=4,
            text=unrelated_text,
            answer=f"我记下了，{distractor_entity}的{distractor_attribute}是{distractor_value}。",
            act="assertion",
            spans=(
                ("ENTITY", distractor_entity),
                ("ATTRIBUTE", distractor_attribute),
                ("VALUE", distractor_value),
            ),
        ),
        _make_turn(
            family_id=family_id,
            episode_id=episode_id,
            level="F1",
            partition=partition,
            turn_index=5,
            text=_F1_QUERY_TEMPLATES[(index + 2) % 5].format(entity=entity, attribute=attribute),
            answer=f"{prompt_value}{new_value}。",
            act="query",
            spans=(("ENTITY", entity), ("ATTRIBUTE", attribute)),
        ),
        _make_turn(
            family_id=family_id,
            episode_id=episode_id,
            level="F1",
            partition=partition,
            turn_index=6,
            text=negative_text,
            answer="这是否定说法，我先不把它记成新事实。",
            act="ambiguous",
            spans=(("ENTITY", entity), ("ATTRIBUTE", attribute), ("NEGATION", "不是")),
        ),
        _make_turn(
            family_id=family_id,
            episode_id=episode_id,
            level="F1",
            partition=partition,
            turn_index=7,
            text=ambiguous_text,
            answer="这个说法还不确定，我先不改事实。",
            act="ambiguous",
            spans=(("ENTITY", entity), ("ATTRIBUTE", attribute), ("VALUE", ambiguous_value)),
        ),
    )
    return JointT1EpisodeV1(episode_id, family_id, "F1", partition, turns)


def build_joint_t1_split(partition: str) -> tuple[JointT1EpisodeV1, ...]:
    """Materialize exactly one split; training code must call this with ``train``."""

    if partition not in JOINT_T1_PARTITIONS:
        raise ValueError("unsupported N177 T1 split")
    f0_families = _f0_families()[partition]
    episode_count = (
        _TRAIN_EPISODES_PER_FAMILY if partition == "train" else _EVAL_EPISODES_PER_FAMILY
    )
    episodes: list[JointT1EpisodeV1] = []
    f0_by_id = {_family_digest("F0", pair): pair for pair in product(_NAMES, _F0_ITEMS)}
    for family_id in f0_families:
        name, item = f0_by_id[family_id]
        for variant in range(episode_count):
            episodes.append(
                _build_f0_episode(
                    family_id=family_id,
                    partition=partition,
                    variant=variant,
                    name=name,
                    item=item,
                )
            )

    f1_primary = _f1_primary_combinations()[:128]
    f1_distractors = _assign_f1_distractors(f1_primary)
    f1_rows = tuple(
        (*primary, *distractor)
        for primary, distractor in zip(f1_primary, f1_distractors, strict=True)
    )
    f1_ids = tuple(_family_digest("F1", row) for row in f1_rows)
    f1_split_ids = _split_family_ids(f1_ids)[partition]
    f1_by_id = {_family_digest("F1", row): row for row in f1_rows}
    for family_id in f1_split_ids:
        row = f1_by_id[family_id]
        for variant in range(episode_count):
            episodes.append(
                _build_f1_episode(
                    family_id=family_id,
                    partition=partition,
                    variant=variant,
                    entity=row[0],
                    attribute=row[1],
                    old_value=row[2],
                    new_value=row[3],
                    distractor_entity=row[4],
                    distractor_attribute=row[5],
                    distractor_value=row[6],
                )
            )
    return tuple(episodes)


def build_joint_t1_update_order(
    train_episodes: Sequence[JointT1EpisodeV1],
) -> tuple[tuple[str, ...], ...]:
    """Return the frozen 2,560-step episode order for all five training modes."""

    if any(episode.partition != "train" for episode in train_episodes):
        raise ValueError("update order accepts only the train partition")
    by_id = {episode.episode_id: episode for episode in train_episodes}
    if len(by_id) != len(train_episodes):
        raise ValueError("training set contains duplicate episode IDs")
    f0_ids = tuple(sorted((key for key, row in by_id.items() if row.level == "F0")))
    f1_ids = tuple(sorted((key for key, row in by_id.items() if row.level == "F1")))
    expected = _TRAIN_FAMILIES_PER_LEVEL * _TRAIN_EPISODES_PER_FAMILY
    if len(f0_ids) != expected or len(f1_ids) != expected:
        raise ValueError("training set must contain 320 F0 and 320 F1 episodes")

    batches: list[tuple[str, ...]] = []
    for epoch in range(8):
        order = tuple(sorted(f0_ids, key=lambda item: _stable_key(f"F0-epoch-{epoch}", item)))
        for offset in range(0, len(order), JOINT_T1_BATCH_SIZE):
            batches.append(order[offset : offset + JOINT_T1_BATCH_SIZE])

    for epoch in range(12):
        f0_order = tuple(sorted(f0_ids, key=lambda item: _stable_key(f"F0-mixed-{epoch}", item)))
        f1_order = tuple(sorted(f1_ids, key=lambda item: _stable_key(f"F1-mixed-{epoch}", item)))
        for offset in range(0, len(f0_order), 2):
            batches.append(
                (
                    f0_order[offset],
                    f1_order[offset],
                    f0_order[offset + 1],
                    f1_order[offset + 1],
                )
            )
    if len(batches) != JOINT_T1_UPDATE_COUNT or any(
        len(batch) != JOINT_T1_BATCH_SIZE for batch in batches
    ):
        raise RuntimeError("frozen N177 T1 update schedule has an unexpected shape")
    return tuple(batches)


def parser_permuted_target(
    target: JointSemanticTrainingTargetV1,
) -> JointSemanticTrainingTargetV1:
    """Apply the fixed F1 entity/attribute confusion control to train labels only."""

    if target.partition != "train":
        raise ValueError("parser-permuted labels are available only for train targets")
    swap = {
        TAG_NAMES.index("B_ENTITY"): TAG_NAMES.index("B_ATTRIBUTE"),
        TAG_NAMES.index("I_ENTITY"): TAG_NAMES.index("I_ATTRIBUTE"),
        TAG_NAMES.index("B_ATTRIBUTE"): TAG_NAMES.index("B_ENTITY"),
        TAG_NAMES.index("I_ATTRIBUTE"): TAG_NAMES.index("I_ENTITY"),
    }
    return JointSemanticTrainingTargetV1.create(
        input_id=target.input_id,
        source_sha256=target.source_sha256,
        family_id=target.family_id,
        partition=target.partition,
        tag_ids=tuple(swap.get(tag, tag) for tag in target.tag_ids),
        act_id=target.act_id,
    )


def joint_t1_course_contract() -> dict[str, Any]:
    """Return public course metadata without materializing any split labels."""

    split_counts = {"train": 64, "query": 32, "holdout": 32}
    payload = {
        "format": JOINT_T1_COURSE_FORMAT,
        "version": 1,
        "seed": JOINT_T1_COURSE_SEED,
        "levels": list(JOINT_T1_LEVELS),
        "partitions": list(JOINT_T1_PARTITIONS),
        "families_per_level": split_counts,
        "episodes_per_family": {
            "train": _TRAIN_EPISODES_PER_FAMILY,
            "query": _EVAL_EPISODES_PER_FAMILY,
            "holdout": _EVAL_EPISODES_PER_FAMILY,
        },
        "turns_per_episode": {"F0": 5, "F1": 8},
        "update_count": JOINT_T1_UPDATE_COUNT,
        "batch_size": JOINT_T1_BATCH_SIZE,
        "f0_only_updates": JOINT_T1_F0_ONLY_UPDATES,
        "stop_bytes_hex": JOINT_T1_STOP_BYTES.hex(),
        "parser_permutation": "swap_BI_ENTITY_with_BI_ATTRIBUTE_on_F1_train_only",
        "family_generation_sha256": content_digest(
            {
                "names": list(_NAMES),
                "f0_items": list(_F0_ITEMS),
                "f1_entities": list(_F1_ENTITIES),
                "f1_attribute_values": {
                    key: list(value) for key, value in _F1_ATTRIBUTE_VALUES.items()
                },
                "f0_turn_templates": [
                    [[prompt, answer] for prompt, answer in turn] for turn in _F0_TURNS
                ],
                "f1_templates": {
                    "assertion": list(_F1_ASSERTION_TEMPLATES),
                    "query": list(_F1_QUERY_TEMPLATES),
                    "correction": list(_F1_CORRECTION_TEMPLATES),
                    "unrelated": list(_F1_UNRELATED_TEMPLATES),
                    "negative": list(_F1_NEGATIVE_TEMPLATES),
                    "ambiguous": list(_F1_AMBIGUOUS_TEMPLATES),
                },
            }
        ),
    }
    return {**payload, "course_sha256": content_digest(payload)}


def joint_t1_split_digest(episodes: Sequence[JointT1EpisodeV1]) -> str:
    """Digest one materialized split without combining its contents with another split."""

    return content_digest(
        {
            "course_sha256": joint_t1_course_contract()["course_sha256"],
            "partition": episodes[0].partition if episodes else "empty",
            "episode_digests": [content_digest(item.to_payload()) for item in episodes],
        }
    )


def joint_t1_update_order_sha256(
    update_order: Sequence[Sequence[str]],
) -> str:
    """Return the identity of the fixed update-to-episode mapping."""

    if len(update_order) != JOINT_T1_UPDATE_COUNT or any(
        len(batch) != JOINT_T1_BATCH_SIZE for batch in update_order
    ):
        raise ValueError("update order does not match the frozen N177 T1 schedule")
    return content_digest(
        {
            "course_sha256": joint_t1_course_contract()["course_sha256"],
            "update_order": [list(batch) for batch in update_order],
        }
    )

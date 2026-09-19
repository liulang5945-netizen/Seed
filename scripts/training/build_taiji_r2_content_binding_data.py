"""Build the R2 content-binding static data package (contract draft v1, section 3).

Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md

Namespace ``r2_content_binding_v1`` -- physically separate from every D1/D8
fixture.  One group is exactly two items that share one prototype (the members
are the prototype and its counterfactual rewrite); a group lives in exactly one
split.  Seven group classes:

- ``fact_flip``            事实值翻转: same question, deciding value flips.
- ``object_swap``          提问对象交换: same material, queried object swaps.
- ``relation_flip``        关系同异翻转: same/different relation flips.
- ``negation_scope``       否定作用域变化: 是/否 flips with the closed facts.
- ``distractor_invariant`` 无关干扰不变: irrelevant distractor must not move the answer.
- ``missing_to_filled``    缺失信息→补足信息: 未知 -> complete value.
- ``unknown_preserved``    已知信息/未知边界: two phrasings, both 未知 (保持组).

Splits and quotas (groups per class): train 256 (3584 items), calibration 32
(448 items), sealed-confirmation 64 (896 items).  Group-level holdout keys
(entity words, complete value words, template family) are disjoint across
splits; characters are allowed to overlap.  The sealed split must carry two
copyable-answer slices: unseen complete values and values containing
train-unseen characters (each >= 64).

Layout rules that make the content-blind shortcut policies fail pairwise on the
three flip categories (fact/object/relation) by construction:

- fact_flip materials carry the target fact first in member a and second in
  member b (target ordinal alternates within the group), so first/last/second
  positional readings always lose one member.
- object_swap members share one material and query different objects with
  different values, so every identity-blind policy emits one constant answer
  against two different gold answers.
- missing_to_filled materials always contain distractor values, so any
  value-emitting policy answers the 未知 member wrongly.

The module also hosts the deterministic fixed-policy battery scorer inputs and
the independent reference solver (it parses the rendered text and never reads
the structured slots), asserted to agree with every generated label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NAMESPACE = "r2_content_binding_v1"
DATA_FORMAT = "r2-content-binding-data-v1"
DATA_VERSION = 1
CONTRACT = "plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md"

FIXTURES = {
    "train": Path("tests/fixtures/r2_content_binding_v1_train.jsonl"),
    "calibration": Path("tests/fixtures/r2_content_binding_v1_calibration.jsonl"),
    "sealed": Path("tests/fixtures/r2_content_binding_v1_sealed.jsonl"),
}
DATA_REPORT = Path("reports/r2_content_binding_v1/data_contract_v1_20260919.json")

GROUP_CLASSES = (
    "fact_flip",
    "object_swap",
    "relation_flip",
    "negation_scope",
    "distractor_invariant",
    "missing_to_filled",
    "unknown_preserved",
)
#: The three categories the contract's fixed-policy gate restricts (§3.2).
FLIP_GATE_CLASSES = ("fact_flip", "object_swap", "relation_flip")
#: Classes whose groups contain visible values (value-length balance applies).
VALUE_CLASSES = (
    "fact_flip",
    "object_swap",
    "relation_flip",
    "negation_scope",
    "distractor_invariant",
    "missing_to_filled",
)
GROUPS_PER_CLASS = {"train": 256, "calibration": 32, "sealed": 64}

#: Closed-class answers; they never appear inside a material body.
ANSWER_SAME, ANSWER_DIFFERENT = "相同", "不同"
ANSWER_YES, ANSWER_NO = "是", "否"
ANSWER_UNKNOWN = "未知"

MAX_PREFIX_CHARS = 256
MAX_RESPONSE_CHARS = 32
MAX_MATERIAL_OBJECTS = 3

# --------------------------------------------------------------------------- #
# Split-disjoint pools (words; characters may overlap across splits).
# --------------------------------------------------------------------------- #

POOLS: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "objects": (
            "山峰", "河流", "湖泊", "森林", "草原", "沙漠", "海湾", "岛屿",
            "峡谷", "平原", "火山", "冰川", "沼泽", "丘陵", "瀑布", "溪流",
            "洞穴", "田野", "星辰", "云层", "海洋", "草地", "岩石", "泉水",
        ),
        "values_single": ("红", "蓝", "绿", "黄", "紫", "灰", "黑", "白"),
        "values_double": ("琥珀", "珊瑚", "翡翠", "玛瑙", "琉璃", "玳瑁", "鎏金", "霜缟"),
    },
    "calibration": {
        "objects": (
            "苔原", "峡湾", "荷塘", "竹林", "盐湖", "芦荡",
            "陡崖", "荒漠", "桦林", "雪原", "礁群", "溶洞",
        ),
        "values_single": ("青", "棕", "绯", "黛"),
        "values_double": ("缥缈", "磬玉", "琅玕", "翠微"),
    },
    "sealed": {
        "objects": (
            "冰原", "花海", "松林", "断崖", "海沟", "绿洲", "星云", "幽谷",
            "枫林", "沙洲", "翠竹", "玉兰", "陡坡", "荷叶", "泉潭", "岚峰",
        ),
        # 墨/银/朱/碧 and one character of 绛霞/黛蓝/绯雪/素缟 are absent from
        # the train text: they feed the sealed unseen-character slice, while
        # 霜瑙/金珀/玳璃/琥缟 reuse only train characters as unseen words.
        "values_single": ("墨", "银", "朱", "碧"),
        "values_double": ("霜瑙", "金珀", "玳璃", "琥缟", "绛霞", "黛蓝", "绯雪", "素缟"),
    },
}

# --------------------------------------------------------------------------- #
# Template families (one per split; wording differs beyond punctuation).
# --------------------------------------------------------------------------- #

MARKERS = {
    "train": {"head": "问：", "bg": "背景：", "tail": "答："},
    "calibration": {"head": "求：", "bg": "材料：", "tail": "回："},
    "sealed": {"head": "查：", "bg": "已知：", "tail": "出："},
}

FACT_STEMS = {
    "train": "问：{X}是什么颜色？",
    "calibration": "求：{X}的颜色是什么？",
    "sealed": "查：{X}呈现什么颜色？",
}
RELATION_STEMS = {
    "train": "问：{X}和{Y}颜色相同吗？",
    "calibration": "求：{X}与{Y}的颜色一致吗？",
    "sealed": "查：{X}、{Y}两者颜色是否相同。",
}
NEGATION_STEMS = {
    "train": "问：{X}是{V}的吗？",
    "calibration": "求：{X}是否为{V}色的？",
    "sealed": "查：{X}属于{V}色吗。",
}
#: Unknown-class question stems: the split's own fact stem plus five shared
#: variants rendered under the split head (six stems x four phrasings per
#: object = 24 prefixes, enough for the group quota with pair discipline).
_SHARED_UNKNOWN_TAILS = ("{X}的颜色？", "请说出{X}的颜色。", "{X}的颜色是什么？", "{X}呈现什么颜色？", "请回答{X}的颜色。")
UNKNOWN_STEMS = {
    split: (FACT_STEMS[split],) + tuple(f"{MARKERS[split]['head']}{tail}" for tail in _SHARED_UNKNOWN_TAILS)
    for split in MARKERS
}
MISSING_CLAUSE = "没有关于{X}的信息"
UNKNOWN_PHRASINGS = (
    "没有关于{X}的信息",
    "{X}的信息缺失",
    "{X}的资料为空",
    "缺少关于{X}的记载",
)

#: Expected in-group answer relation per class (the counterfactual contract).
EXPECTED_IN_GROUP = {
    "fact_flip": "differ",
    "object_swap": "differ",
    "relation_flip": "differ",
    "negation_scope": "differ",
    "distractor_invariant": "equal",
    "missing_to_filled": "differ",
    "unknown_preserved": "equal",
}
#: Copy mask per class: whether the response is a copyable complete value.
CLASS_VALUE_RESPONSE = {
    "fact_flip": (True, True),
    "object_swap": (True, True),
    "relation_flip": (False, False),
    "negation_scope": (False, False),
    "distractor_invariant": (True, True),
    "missing_to_filled": (False, True),
    "unknown_preserved": (False, False),
}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_question(split: str, stem: str, slots: dict[str, str]) -> str:
    return stem.format(**slots)


def render_material(split: str, body: str) -> str:
    marker = MARKERS[split]
    return f"{marker['bg']}{body}。{marker['tail']}"


def make_record(
    split: str,
    group_class: str,
    group_index: int,
    member: str,
    question: str,
    material_body: str,
    response: str,
    copyable: bool,
) -> dict[str, Any]:
    material = render_material(split, material_body)
    return {
        "id": f"{split}:{group_class}:{group_index:04d}:{member}",
        "split": split,
        "group_id": f"{split}:{group_class}:{group_index:04d}",
        "group_class": group_class,
        "member": member,
        "question": question,
        "material": material,
        "prefix": question + material,
        "response": response,
        "copy_mask": [copyable] * len(response),
        "answer_kind": "value" if copyable else "closed",
        "copyable": copyable,
    }


# --------------------------------------------------------------------------- #
# Per-class group generation (deterministic rejection sampling)
# --------------------------------------------------------------------------- #


def _rng(split: str, cls: str, half: str) -> random.Random:
    return random.Random(f"{NAMESPACE}:{split}:{cls}:{half}")


def _distinct_values(rng: random.Random, values: tuple[str, ...], count: int) -> list[str]:
    return rng.sample(list(values), count)


def _objects_pair(rng: random.Random, objects: tuple[str, ...]) -> tuple[str, str]:
    x, y = rng.sample(list(objects), 2)
    return x, y


def _gen_fact_flip(
    split: str, cls: str, values: tuple[str, ...], count: int, used: set[str], half: str = "main"
) -> list[list[dict[str, Any]]]:
    """Target ordinal alternates inside the group: member a first, member b second."""

    objects = POOLS[split]["objects"]
    rng = _rng(split, cls, half)
    groups: list[list[dict[str, Any]]] = []
    attempts = 0
    while len(groups) < count:
        attempts += 1
        if attempts > 500 * count:
            raise RuntimeError(f"combo pool exhausted for {split}/{cls}")
        x, y = _objects_pair(rng, objects)
        v_a, v_b, d = _distinct_values(rng, values, 3)
        q_a = render_question(split, FACT_STEMS[split], {"X": x})
        q_b = render_question(split, FACT_STEMS[split], {"X": x})
        pre_a = q_a + render_material(split, f"{x}是{v_a}，{y}是{d}")
        pre_b = q_b + render_material(split, f"{y}是{d}，{x}是{v_b}")
        if pre_a in used or pre_b in used:
            continue
        used.update((pre_a, pre_b))
        index = len(groups)
        groups.append(
            [
                make_record(split, cls, index, "a", q_a, f"{x}是{v_a}，{y}是{d}", v_a, True),
                make_record(split, cls, index, "b", q_b, f"{y}是{d}，{x}是{v_b}", v_b, True),
            ]
        )
    return groups


def _gen_object_swap(
    split: str, cls: str, values: tuple[str, ...], count: int, used: set[str], half: str = "main"
) -> list[list[dict[str, Any]]]:
    """One material, two queried objects with different values."""

    objects = POOLS[split]["objects"]
    rng = _rng(split, cls, half)
    groups: list[list[dict[str, Any]]] = []
    attempts = 0
    while len(groups) < count:
        attempts += 1
        if attempts > 500 * count:
            raise RuntimeError(f"combo pool exhausted for {split}/{cls}")
        x, y = _objects_pair(rng, objects)
        v1, v2 = _distinct_values(rng, values, 2)
        body = f"{x}是{v1}，{y}是{v2}"
        q_a = render_question(split, FACT_STEMS[split], {"X": x})
        q_b = render_question(split, FACT_STEMS[split], {"X": y})
        pre_a, pre_b = q_a + render_material(split, body), q_b + render_material(split, body)
        if pre_a in used or pre_b in used:
            continue
        used.update((pre_a, pre_b))
        index = len(groups)
        groups.append(
            [
                make_record(split, cls, index, "a", q_a, body, v1, True),
                make_record(split, cls, index, "b", q_b, body, v2, True),
            ]
        )
    return groups


def _gen_relation_flip(
    split: str, cls: str, values: tuple[str, ...], count: int, used: set[str], half: str = "main"
) -> list[list[dict[str, Any]]]:
    objects = POOLS[split]["objects"]
    rng = _rng(split, cls, half)
    groups: list[list[dict[str, Any]]] = []
    attempts = 0
    while len(groups) < count:
        attempts += 1
        if attempts > 500 * count:
            raise RuntimeError(f"combo pool exhausted for {split}/{cls}")
        x, y = _objects_pair(rng, objects)
        v, w = _distinct_values(rng, values, 2)
        q = render_question(split, RELATION_STEMS[split], {"X": x, "Y": y})
        body_a = f"{x}是{v}，{y}是{v}"
        body_b = f"{x}是{v}，{y}是{w}"
        pre_a, pre_b = q + render_material(split, body_a), q + render_material(split, body_b)
        if pre_a in used or pre_b in used:
            continue
        used.update((pre_a, pre_b))
        index = len(groups)
        groups.append(
            [
                make_record(split, cls, index, "a", q, body_a, ANSWER_SAME, False),
                make_record(split, cls, index, "b", q, body_b, ANSWER_DIFFERENT, False),
            ]
        )
    return groups


def _gen_negation_scope(
    split: str, cls: str, values: tuple[str, ...], count: int, used: set[str], half: str = "main"
) -> list[list[dict[str, Any]]]:
    """Closed facts flip which object holds the value; the 是/否 answer flips.

    The queried object alternates between first-mentioned and second-mentioned
    across groups (exact halves via the two-phase fill below).
    """

    objects = POOLS[split]["objects"]
    groups: list[list[dict[str, Any]]] = []
    for queried_first in (True, False):
        rng = _rng(split, cls, f"{half}:" + ("queried_first" if queried_first else "queried_second"))
        quota = count // 2
        made = 0
        attempts = 0
        while made < quota:
            attempts += 1
            if attempts > 500 * count:
                raise RuntimeError(f"combo pool exhausted for {split}/{cls}")
            x, y = _objects_pair(rng, objects)
            (v,) = _distinct_values(rng, values, 1)
            q_target = x if queried_first else y
            q = render_question(split, NEGATION_STEMS[split], {"X": q_target, "V": v})
            body_a = f"{x}不是{v}，{y}是{v}"
            body_b = f"{x}是{v}，{y}不是{v}"
            pre_a, pre_b = q + render_material(split, body_a), q + render_material(split, body_b)
            if pre_a in used or pre_b in used:
                continue
            used.update((pre_a, pre_b))
            answer_a = ANSWER_NO if q_target == x else ANSWER_YES
            answer_b = ANSWER_YES if q_target == x else ANSWER_NO
            index = len(groups)
            groups.append(
                [
                    make_record(split, cls, index, "a", q, body_a, answer_a, False),
                    make_record(split, cls, index, "b", q, body_b, answer_b, False),
                ]
            )
            made += 1
    return groups


def _gen_distractor_invariant(
    split: str, cls: str, values: tuple[str, ...], count: int, used: set[str], half: str = "main"
) -> list[list[dict[str, Any]]]:
    """Distractor position balanced: half of the groups place it first."""

    objects = POOLS[split]["objects"]
    groups: list[list[dict[str, Any]]] = []
    for distractor_first in (False, True):
        rng = _rng(split, cls, f"{half}:" + ("distractor_after" if not distractor_first else "distractor_before"))
        quota = count // 2
        made = 0
        attempts = 0
        while made < quota:
            attempts += 1
            if attempts > 500 * count:
                raise RuntimeError(f"combo pool exhausted for {split}/{cls}")
            x, y = _objects_pair(rng, objects)
            v, d = _distinct_values(rng, values, 2)
            q = render_question(split, FACT_STEMS[split], {"X": x})
            body_a = f"{x}是{v}"
            body_b = f"{y}是{d}，{x}是{v}" if distractor_first else f"{x}是{v}，{y}是{d}"
            pre_a, pre_b = q + render_material(split, body_a), q + render_material(split, body_b)
            if pre_a in used or pre_b in used:
                continue
            used.update((pre_a, pre_b))
            index = len(groups)
            groups.append(
                [
                    make_record(split, cls, index, "a", q, body_a, v, True),
                    make_record(split, cls, index, "b", q, body_b, v, True),
                ]
            )
            made += 1
    return groups


def _gen_missing_to_filled(
    split: str, cls: str, values: tuple[str, ...], count: int, used: set[str], half: str = "main"
) -> list[list[dict[str, Any]]]:
    """Two distractor facts are shared by both members; only the target clause
    changes (missing clause vs. supplied fact), so value-emitting policies lose
    the 未知 member by construction."""

    objects = POOLS[split]["objects"]
    rng = _rng(split, cls, half)
    groups: list[list[dict[str, Any]]] = []
    attempts = 0
    while len(groups) < count:
        attempts += 1
        if attempts > 500 * count:
            raise RuntimeError(f"combo pool exhausted for {split}/{cls}")
        x, y, z = rng.sample(list(objects), 3)
        v, d1, d2 = _distinct_values(rng, values, 3)
        q = render_question(split, FACT_STEMS[split], {"X": x})
        shared = f"{y}是{d1}，{z}是{d2}；"
        body_a = f"{shared}{MISSING_CLAUSE.format(X=x)}"
        body_b = f"{shared}{x}是{v}"
        pre_a, pre_b = q + render_material(split, body_a), q + render_material(split, body_b)
        if pre_a in used or pre_b in used:
            continue
        used.update((pre_a, pre_b))
        index = len(groups)
        groups.append(
            [
                make_record(split, cls, index, "a", q, body_a, ANSWER_UNKNOWN, False),
                make_record(split, cls, index, "b", q, body_b, v, True),
            ]
        )
    return groups


def _gen_unknown_preserved(
    split: str, cls: str, count: int, used: set[str], half: str = "main"
) -> list[list[dict[str, Any]]]:
    """Two different phrasings, both answers 未知 (保持组).

    Exact pairing: each object owns stems x phrasings prefixes; the seeded
    shuffle pairs them two-by-two inside the object, so the fill is exact and
    both members of a group keep the same queried object.
    """

    objects = POOLS[split]["objects"]
    stems = UNKNOWN_STEMS[split]
    variants = [
        (stem_index, phrase_index)
        for stem_index in range(len(stems))
        for phrase_index in range(len(UNKNOWN_PHRASINGS))
    ]
    rng = _rng(split, cls, half)
    groups: list[list[dict[str, Any]]] = []
    for x in objects:
        local = list(variants)
        rng.shuffle(local)
        for pair_at in range(0, len(local) - 1, 2):
            if len(groups) >= count:
                return groups
            var_a, var_b = local[pair_at], local[pair_at + 1]
            q_a = render_question(split, stems[var_a[0]], {"X": x})
            q_b = render_question(split, stems[var_b[0]], {"X": x})
            body_a = UNKNOWN_PHRASINGS[var_a[1]].format(X=x)
            body_b = UNKNOWN_PHRASINGS[var_b[1]].format(X=x)
            pre_a = q_a + render_material(split, body_a)
            pre_b = q_b + render_material(split, body_b)
            if pre_a in used or pre_b in used:
                continue
            used.update((pre_a, pre_b))
            index = len(groups)
            groups.append(
                [
                    make_record(split, cls, index, "a", q_a, body_a, ANSWER_UNKNOWN, False),
                    make_record(split, cls, index, "b", q_b, body_b, ANSWER_UNKNOWN, False),
                ]
            )
    if len(groups) < count:
        raise RuntimeError(f"combo pool exhausted for {split}/{cls}")
    return groups


def _generate_split(split: str) -> list[dict[str, Any]]:
    """All groups of one split, class-major, with global prefix uniqueness."""

    pools = POOLS[split]
    used: set[str] = set()
    records: list[dict[str, Any]] = []
    count = GROUPS_PER_CLASS[split]
    for cls in GROUP_CLASSES:
        if cls == "unknown_preserved":
            halves = {"single": _gen_unknown_preserved(split, cls, count, used), "double": []}
        else:
            halves = {}
            for half_name, values in (
                ("single", pools["values_single"]),
                ("double", pools["values_double"]),
            ):
                generator: Callable[..., list[list[dict[str, Any]]]] = {
                    "fact_flip": _gen_fact_flip,
                    "object_swap": _gen_object_swap,
                    "relation_flip": _gen_relation_flip,
                    "negation_scope": _gen_negation_scope,
                    "distractor_invariant": _gen_distractor_invariant,
                    "missing_to_filled": _gen_missing_to_filled,
                }[cls]
                halves[half_name] = generator(
                    split, cls, values, count // 2, used, half=half_name
                )
        # Interleave the value-length halves so single/double groups alternate
        # along the file; valueless classes emit their single list directly.
        single, double = halves["single"], halves["double"]
        for position in range(count):
            if double:
                group = single[position // 2] if position % 2 == 0 else double[position // 2]
            else:
                group = single[position]
            for member_record in group:
                records.append(member_record)
    # Renumber group ids sequentially per class after the interleaving.
    counters: dict[str, int] = {}
    for record in records:
        cls = record["group_class"]
        counters[cls] = counters.get(cls, 0) + 1
        number = (counters[cls] + 1) // 2
        record["group_id"] = f"{split}:{cls}:{number:04d}"
        record["id"] = f"{record['group_id']}:{record['member']}"
    return records


# --------------------------------------------------------------------------- #
# Independent reference solver (parses rendered text; never reads slots)
# --------------------------------------------------------------------------- #

STATEMENT_PATTERN = re.compile(r"^(.+?)(不是|是)(.+)$")
PART_SPLIT = re.compile(r"[，。；]")
#: Ordered fact-stem patterns (most specific first); shared by every split
#: because the head marker is stripped before matching.
_FACT_PATTERNS = (
    r"^(.+?)的颜色是什么$",
    r"^(.+?)呈现什么颜色$",
    r"^(.+?)是什么颜色$",
    r"^请说出(.+?)的颜色$",
    r"^请回答(.+?)的颜色$",
    r"^(.+?)的颜色$",
)
_MENTION_PATTERNS = (
    re.compile(r"没有关于(.+?)的信息"),
    re.compile(r"(.+?)的信息缺失"),
    re.compile(r"(.+?)的资料为空"),
    re.compile(r"缺少关于(.+?)的记载"),
)


def _mentioned_objects(split: str, material: str) -> set[str]:
    """Every object a material mentions: statement subjects plus missing/unknown
    mentions (they carry no statement but still count toward the 1-3 bound)."""

    body = split_material_body(split, material)
    words = set(parse_statements(body))
    for pattern in _MENTION_PATTERNS:
        match = pattern.search(body)
        if match:
            words.add(match.group(1))
    return words


def split_question_head(split: str, question: str) -> str:
    head = MARKERS[split]["head"]
    if not question.startswith(head):
        raise RuntimeError(f"question missing head marker {head!r}: {question}")
    return question[len(head) :]


def split_material_body(split: str, material: str) -> str:
    marker = MARKERS[split]
    if not material.startswith(marker["bg"]) or not material.endswith(marker["tail"]):
        raise RuntimeError(f"material missing markers: {material}")
    return material[len(marker["bg"]) : -len(marker["tail"])].rstrip("。")


def parse_statements(material_body: str) -> dict[str, tuple[str, str]]:
    """object -> (polarity, value); duplicate objects or non-值 bodies skipped."""

    facts: dict[str, tuple[str, str]] = {}
    for part in PART_SPLIT.split(material_body):
        part = part.strip()
        if not part:
            continue
        match = STATEMENT_PATTERN.match(part)
        if match is None:
            continue
        obj, polarity, value = match.group(1), match.group(2), match.group(3)
        if obj in facts:
            raise RuntimeError(f"conflicting statements for {obj}: {material_body}")
        facts[obj] = (polarity, value)
    return facts


def reference_answer(record: dict[str, Any]) -> str:
    """Deterministic text reader: the Q5-style self-check reference key."""

    split = record["split"]
    body_q = split_question_head(split, record["question"])
    facts = parse_statements(split_material_body(split, record["material"]))

    if "相同" in body_q or "一致" in body_q:
        patterns = {
            "train": r"^(.+?)和(.+?)颜色相同吗",
            "calibration": r"^(.+?)与(.+?)的颜色一致吗",
            "sealed": r"^(.+?)、(.+?)两者颜色是否相同",
        }[split]
        match = re.match(patterns, body_q)
        if not match:
            raise RuntimeError(f"cannot parse relation stem: {record['question']}")
        x, y = match.group(1), match.group(2)
        if x not in facts or y not in facts:
            raise RuntimeError(f"relation material lacks a queried fact: {record['material']}")
        px, vx = facts[x]
        py, vy = facts[y]
        if px != "是" or py != "是":
            raise RuntimeError(f"relation material carries negation: {record['material']}")
        return ANSWER_SAME if vx == vy else ANSWER_DIFFERENT

    negation_patterns = {
        "train": r"^(.+?)是(.+?)的吗",
        "calibration": r"^(.+?)是否为(.+?)色的",
        "sealed": r"^(.+?)属于(.+?)色吗",
    }
    match = re.match(negation_patterns[split], body_q)
    if match:
        queried, value = match.group(1), match.group(2)
        if queried not in facts:
            raise RuntimeError(f"negation material lacks the queried fact: {record['material']}")
        polarity, stated = facts[queried]
        if stated != value:
            raise RuntimeError(f"negation material states another value: {record['material']}")
        return ANSWER_YES if polarity == "是" else ANSWER_NO

    queried = _queried_object(split, record["question"])
    if queried is None:
        raise RuntimeError(f"cannot parse question stem: {record['question']}")
    if queried not in facts:
        return ANSWER_UNKNOWN
    polarity, value = facts[queried]
    if polarity != "是":
        raise RuntimeError(f"fact material carries negation: {record['material']}")
    return value


# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #


def _material_objects(material_body: str) -> list[str]:
    return sorted(parse_statements(material_body))


def _value_words_of_group(group: list[dict[str, Any]]) -> frozenset[str] | None:
    words: set[str] = set()
    for record in group:
        facts = parse_statements(split_material_body(record["split"], record["material"]))
        for _obj, (_polarity, value) in facts.items():
            words.add(value)
    return frozenset(words) if words else None


def _run_checks(all_records: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[str]]:
    splits = ("train", "calibration", "sealed")
    checks: dict[str, Any] = {}
    failed: list[str] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        entry: dict[str, Any] = {"passed": bool(passed)}
        if detail is not None:
            entry["detail"] = detail
        checks[name] = entry
        if not passed:
            failed.append(name)

    # quotas and pair structure
    for split in splits:
        records = all_records[split]
        expected_groups = GROUPS_PER_CLASS[split] * len(GROUP_CLASSES)
        group_ids = {r["group_id"] for r in records}
        by_group: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_group.setdefault(record["group_id"], []).append(record)
        pair_ok = all(
            len(members) == 2 and {m["member"] for m in members} == {"a", "b"}
            for members in by_group.values()
        )
        class_counts: dict[str, int] = {
            cls: len({r["group_id"] for r in records if r["group_class"] == cls})
            for cls in GROUP_CLASSES
        }
        check(
            f"quota_{split}",
            len(records) == expected_groups * 2 and len(group_ids) == expected_groups,
            {"items": len(records), "groups": len(group_ids), "per_class": class_counts},
        )
        check(f"pair_structure_{split}", pair_ok)

    # uniqueness within split and word-level disjointness across splits
    for split in splits:
        records = all_records[split]
        prefixes = [r["prefix"] for r in records]
        pairs = [(r["prefix"], r["response"]) for r in records]
        check(f"unique_prefixes_{split}", len(set(prefixes)) == len(prefixes))
        check(f"unique_pairs_{split}", len(set(pairs)) == len(pairs))
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            pa = {r["prefix"] for r in all_records[a]}
            pb = {r["prefix"] for r in all_records[b]}
            ga = {r["group_id"] for r in all_records[a]}
            gb = {r["group_id"] for r in all_records[b]}
            check(f"prefixes_disjoint_{a}_{b}", not (pa & pb))
            check(f"groups_disjoint_{a}_{b}", not (ga & gb))

    # group-level holdout keys: entity words / value words / template family
    def entity_key(group: list[dict[str, Any]]) -> frozenset[str]:
        words: set[str] = set()
        for record in group:
            words.update(_mentioned_objects(record["split"], record["material"]))
        return frozenset(words)

    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            entity_a = {entity_key(g) for g in _groups(all_records[a])}
            entity_b = {entity_key(g) for g in _groups(all_records[b])}
            check(f"entity_keys_disjoint_{a}_{b}", not (entity_a & entity_b))
            value_a = {k for k in (_value_words_of_group(g) for g in _groups(all_records[a])) if k}
            value_b = {k for k in (_value_words_of_group(g) for g in _groups(all_records[b])) if k}
            check(f"value_keys_disjoint_{a}_{b}", not (value_a & value_b))
    markers_distinct = all(
        MARKERS[a][field] != MARKERS[b][field]
        for i, a in enumerate(splits)
        for b in splits[i + 1 :]
        for field in ("head", "bg", "tail")
    )
    check("template_family_markers_disjoint", markers_distinct)

    # word-surface cleanliness: no value word may appear inside another value
    # word, an object word, or any template/marker string of its own split
    # (keeps the text scanners and the reference parser unambiguous).
    template_text = {
        split: "".join(MARKERS[split].values())
        + FACT_STEMS[split]
        + RELATION_STEMS[split]
        + NEGATION_STEMS[split]
        + "".join(UNKNOWN_STEMS[split])
        + "".join(UNKNOWN_PHRASINGS)
        + MISSING_CLAUSE
        for split in splits
    }
    for split in splits:
        values = list(POOLS[split]["values_single"]) + list(POOLS[split]["values_double"])
        objects = list(POOLS[split]["objects"])
        clean = True
        detail: list[str] = []
        for i, v in enumerate(values):
            if v in template_text[split]:
                clean = False
                detail.append(f"value {v!r} appears in template text")
            for j, other in enumerate(values):
                if i != j and v in other:
                    clean = False
                    detail.append(f"value {v!r} inside value {other!r}")
            for o in objects:
                if v in o or o in values:
                    clean = False
                    detail.append(f"value {v!r} and object {o!r} overlap")
        for o in objects:
            if o in template_text[split]:
                clean = False
                detail.append(f"object {o!r} appears in template text")
        check(f"word_surface_clean_{split}", clean, detail[:10])

    # in-group answer relation, copy masks, closed answers, conflict-free facts
    closed_answers = {ANSWER_SAME, ANSWER_DIFFERENT, ANSWER_YES, ANSWER_NO, ANSWER_UNKNOWN}
    for split in splits:
        relation_ok = True
        mask_ok = True
        closed_ok = True
        conflict_ok = True
        for group in _groups(all_records[split]):
            a, b = group
            expected = EXPECTED_IN_GROUP[a["group_class"]]
            same = a["response"] == b["response"]
            relation_ok &= (same and expected == "equal") or (
                not same and expected == "differ"
            )
            for record, copyable in zip(group, CLASS_VALUE_RESPONSE[a["group_class"]], strict=True):
                mask_ok &= record["copyable"] == copyable
                mask_ok &= len(record["copy_mask"]) == len(record["response"])
                if copyable:
                    # every copyable position must be present in the material
                    material_text = record["material"]
                    mask_ok &= all(ch in material_text for ch in record["response"])
                else:
                    mask_ok &= not any(record["copy_mask"])
                    closed_ok &= record["response"] in closed_answers
            facts = parse_statements(split_material_body(a["split"], a["material"]))
            conflict_ok &= len(facts) == len(set(facts))
        check(f"in_group_relation_{split}", relation_ok)
        check(f"copy_mask_and_closed_{split}", mask_ok and closed_ok)
        check(f"no_conflicting_facts_{split}", conflict_ok)

    # value-length balance per value class; position balances
    for split in splits:
        balance: dict[str, Any] = {}
        for cls in VALUE_CLASSES:
            groups = [g for g in _groups(all_records[split]) if g[0]["group_class"] == cls]
            singles = set(POOLS[split]["values_single"])
            doubles = set(POOLS[split]["values_double"])
            # length class is a property of the VALUES IN THE MATERIAL, not of
            # the response string (relation/negation answers are closed-class).
            def _is_single(group: list[dict[str, Any]]) -> bool:
                words = _value_words_of_group(group) or frozenset()
                return bool(words & singles)
            single = sum(1 for g in groups if _is_single(g))
            double = sum(
                1 for g in groups if (g2 := _value_words_of_group(g)) and (set(g2) & doubles)
            )
            balance[cls] = {"single": single, "double": double}
            check(
                f"value_length_balance_{split}_{cls}",
                single == double and single + double == len(groups),
                balance[cls],
            )
            check(
                f"value_length_balance_{split}_{cls}",
                single == len(groups) - single,
                balance[cls],
            )
        negation_groups = [
            g for g in _groups(all_records[split]) if g[0]["group_class"] == "negation_scope"
        ]
        first_queried = sum(1 for g in negation_groups if _negation_queried_first(g))
        check(
            f"negation_queried_balance_{split}",
            first_queried == len(negation_groups) - first_queried,
            {"first": first_queried, "second": len(negation_groups) - first_queried},
        )
        distractor_groups = [
            g for g in _groups(all_records[split]) if g[0]["group_class"] == "distractor_invariant"
        ]
        before = sum(1 for g in distractor_groups if _distractor_first(g))
        check(
            f"distractor_position_balance_{split}",
            before == len(distractor_groups) - before,
            {"before": before, "after": len(distractor_groups) - before},
        )

    # object-count bounds and length bounds
    for split in splits:
        obj_counts: dict[int, int] = {}
        max_prefix = 0
        max_response = 0
        bounds_ok = True
        for record in all_records[split]:
            mentioned = _mentioned_objects(split, record["material"])
            obj_counts[len(mentioned)] = obj_counts.get(len(mentioned), 0) + 1
            max_prefix = max(max_prefix, len(record["prefix"]))
            max_response = max(max_response, len(record["response"]))
            bounds_ok &= len(record["prefix"]) <= MAX_PREFIX_CHARS
            bounds_ok &= len(record["response"]) <= MAX_RESPONSE_CHARS
            bounds_ok &= 1 <= len(mentioned) <= MAX_MATERIAL_OBJECTS
        check(
            f"length_and_object_bounds_{split}",
            bounds_ok,
            {"max_prefix": max_prefix, "max_response": max_response, "object_counts": obj_counts},
        )

    # sealed slice denominators (contract §3.1)
    train_text = "".join(
        r["question"] + r["material"] + r["response"] for r in all_records["train"]
    )
    train_value_words = set(POOLS["train"]["values_single"]) | set(POOLS["train"]["values_double"])
    for split in ("calibration", "sealed"):
        copyable = [r for r in all_records[split] if r["copyable"]]
        unseen_value = [
            r for r in copyable if r["response"] not in train_value_words
        ]
        unseen_char = [
            r for r in copyable if any(ch not in train_text for ch in r["response"])
        ]
        check(
            f"copyable_slice_denominators_{split}",
            len(copyable) > 0 and len(unseen_value) >= 64 and len(unseen_char) >= 64,
            {
                "copyable": len(copyable),
                "unseen_complete_value": len(unseen_value),
                "unseen_char": len(unseen_char),
            },
        )

    # independent reference solver agrees everywhere
    mismatches = [
        {"id": r["id"], "expected": r["response"], "reference": reference_answer(r)}
        for split in splits
        for r in all_records[split]
        if reference_answer(r) != r["response"]
    ]
    check("reference_self_check", not mismatches, {"mismatches": mismatches[:10]})

    return checks, failed


def _groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_group.setdefault(record["group_id"], []).append(record)
    return [by_group[key] for key in sorted(by_group)]


def _negation_queried_first(group: list[dict[str, Any]]) -> bool:
    record = group[0]
    split = record["split"]
    body_q = split_question_head(split, record["question"])
    pattern = {
        "train": r"^(.+?)是(.+?)的吗",
        "calibration": r"^(.+?)是否为(.+?)色的",
        "sealed": r"^(.+?)属于(.+?)色吗",
    }[split]
    queried = re.match(pattern, body_q).group(1)  # type: ignore[union-attr]
    facts = parse_statements(split_material_body(split, record["material"]))
    return bool(facts) and next(iter(facts)) == queried


def _distractor_first(group: list[dict[str, Any]]) -> bool:
    record = group[1]
    split = record["split"]
    facts = parse_statements(split_material_body(split, record["material"]))
    queried = _queried_object(split, group[0]["question"])
    return bool(facts) and next(iter(facts)) != queried


def _queried_object(split: str, question: str) -> str | None:
    stem = split_question_head(split, question).rstrip("？?。")
    for pattern in _FACT_PATTERNS:
        match = re.match(pattern, stem)
        if match:
            return match.group(1)
    return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DATA_REPORT)
    args = parser.parse_args()

    all_records: dict[str, list[dict[str, Any]]] = {
        split: _generate_split(split) for split in ("train", "calibration", "sealed")
    }
    checks, failed = _run_checks(all_records)

    from taiji.internalization import content_digest

    ordered = [r for split in ("train", "calibration", "sealed") for r in all_records[split]]
    digest = content_digest(ordered)

    file_sha: dict[str, str | None] = {}
    if not failed:
        for split, relative in FIXTURES.items():
            path = PROJECT_ROOT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in all_records[split]:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            file_sha[split] = hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "format": DATA_FORMAT,
        "version": DATA_VERSION,
        "namespace": NAMESPACE,
        "contract": CONTRACT,
        "generator": "scripts/training/build_taiji_r2_content_binding_data.py",
        "fixtures": {
            split: {"path": str(relative), "sha256": file_sha.get(split)}
            for split, relative in FIXTURES.items()
        },
        "corpus_digest": digest,
        "pools": {
            split: {
                "objects": list(POOLS[split]["objects"]),
                "values_single": list(POOLS[split]["values_single"]),
                "values_double": list(POOLS[split]["values_double"]),
            }
            for split in POOLS
        },
        "group_classes": list(GROUP_CLASSES),
        "flip_gate_classes": list(FLIP_GATE_CLASSES),
        "splits": {
            split: {
                "items": len(all_records[split]),
                "groups": len({r["group_id"] for r in all_records[split]}),
                "groups_per_class": {
                    cls: len(
                        {r["group_id"] for r in all_records[split] if r["group_class"] == cls}
                    )
                    for cls in GROUP_CLASSES
                },
            }
            for split in ("train", "calibration", "sealed")
        },
        "checks": checks,
        "outcome": "passed" if not failed else "failed",
        "failed_checks": failed,
    }
    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "failed": failed,
                "digest": digest[:12],
                "train_items": len(all_records["train"]),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

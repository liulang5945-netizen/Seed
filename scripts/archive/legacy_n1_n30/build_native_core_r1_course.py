"""Build the native content core R1 world course (train/query/holdout).

R1 fixed the semantic contract and a 360-history sample plan, but the exact
split and the actual episode text were still open.  This builder materialises
them without calling a model:

* 36 entity/attribute slots, each with 6 counterfactual pairs of different-value
  two-version histories (12 possible histories per slot);
* per slot, 3 pairs go to train, 1 to query, 1 to holdout, 1 is unused, so every
  slot appears in every split and a pair never crosses splits;
* 4 episodes per history, 8 turns each, following the R1 fixed protocol;
* fixed answer denominator = the four answer turns per episode (current before
  correction, current after correction, previous, unknown);
* train/query/holdout text are written to separate files so a training process
  only needs the train file.

No optimizer, no model call, no sealed read.  Query/holdout text is public
synthetic data but marked ``training_read=false`` in the split manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.internalization import content_digest  # noqa: E402

ENTITIES = (
    "雨伞",
    "背包",
    "水杯",
    "耳机",
    "遥控器",
    "台灯",
    "剪刀",
    "尺子",
    "闹钟",
    "钱包",
    "帽子",
    "围巾",
)
ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "颜色": ("红色", "蓝色", "绿色", "黄色"),
    "位置": ("桌上", "抽屉", "柜子", "包里"),
    "归属": ("我的", "你的", "他的", "她的"),
}
OWNER_PRONOUN = {"我的": "我", "你的": "你", "他的": "他", "她的": "她"}
END_MARKER = "<|end|>"
SPLIT_SALT = "native-core-r1-split-v1-20261030"
EPISODES_PER_HISTORY = 4
TURNS_PER_EPISODE = 8
ANSWER_TURNS = (2, 4, 6, 7)  # zero-based indexes of scored turns

SPLIT_MANIFEST = ROOT / "plans/manifests/native_core_r1_split_v1.json"
COURSE_FILES = {
    "train": ROOT / "plans/manifests/native_core_r1_train_v1.json",
    "query": ROOT / "plans/manifests/native_core_r1_query_v1.json",
    "holdout": ROOT / "plans/manifests/native_core_r1_holdout_v1.json",
}
REPORT = ROOT / "reports/taiji_native_core_r1_course_audit_20261030.json"


def _target(text: str) -> dict[str, Any]:
    raw = (text + END_MARKER).encode("utf-8")
    return {
        "target_text": text,
        "target_bytes_utf8": list(raw),
        "end_marker": END_MARKER,
        "first_byte": raw[0],
        "supervision": "target_t_at_step_t",
        "forbidden": "target[1:]",
    }


def _history_id(entity: str, attribute: str, v0: str, v1: str) -> str:
    return (
        "r1h:" + content_digest({"entity": entity, "attribute": attribute, "v0": v0, "v1": v1})[:20]
    )


def _pairs(values: tuple[str, ...]) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    leftovers: list[tuple[str, str]] = []
    for v0 in values:
        others = [value for value in values if value != v0]
        pairs.append(((v0, others[0]), (v0, others[1])))
        leftovers.append((v0, others[2]))
    pairs.append((leftovers[0], leftovers[1]))
    pairs.append((leftovers[2], leftovers[3]))
    return pairs


def _split_histories() -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for entity in ENTITIES:
        for attribute, values in ATTRIBUTES.items():
            slot = f"{entity}|{attribute}"
            pairs = _pairs(values)
            ranked = sorted(
                enumerate(pairs),
                key=lambda item: content_digest(
                    {"slot": slot, "pair": item[0], "salt": SPLIT_SALT}
                ),
            )
            assignment = ("train", "train", "train", "query", "holdout", "unused")
            for rank, (pair_index, pair) in enumerate(ranked):
                split = assignment[rank]
                for v0, v1 in pair:
                    histories.append(
                        {
                            "history_id": _history_id(entity, attribute, v0, v1),
                            "entity": entity,
                            "attribute": attribute,
                            "v0": v0,
                            "v1": v1,
                            "slot": slot,
                            "pair_id": f"{slot}|pair{pair_index}",
                            "pair_rank": rank,
                            "split": split,
                        }
                    )
    return histories


def _fact_text(entity: str, attribute: str, value: str, variant: int) -> str:
    if attribute == "颜色":
        templates = (
            "{e}的颜色是{v}。",
            "{e}是{v}的。",
            "{e}的颜色现在是{v}。",
            "{e}的颜色为{v}。",
        )
        return templates[variant].format(e=entity, v=value)
    if attribute == "位置":
        templates = ("{e}在{v}。", "{e}现在在{v}。", "{e}放在{v}。", "{e}位于{v}。")
        return templates[variant].format(e=entity, v=value)
    templates_owner = ("{e}是{v}。", "{e}现在归{p}。", "{e}由{p}保管。", "{e}属于{p}。")
    return templates_owner[variant].format(e=entity, v=value, p=OWNER_PRONOUN[value])


def _query_text(entity: str, attribute: str, variant: int) -> str:
    if attribute == "颜色":
        templates = (
            "{e}是什么颜色？",
            "{e}的颜色是什么？",
            "{e}现在是什么颜色？",
            "{e}的颜色呢？",
        )
    elif attribute == "位置":
        templates = ("{e}在哪里？", "{e}的位置在哪里？", "{e}现在在哪里？", "{e}放哪儿了？")
    else:
        templates = ("{e}是谁的？", "{e}归谁？", "{e}现在是谁的？", "{e}是谁的东西？")
    return templates[variant].format(e=entity)


def _answer_text(entity: str, attribute: str, value: str) -> str:
    if attribute == "颜色":
        return f"{entity}的颜色是{value}。"
    if attribute == "位置":
        return f"{entity}的位置是{value}。"
    return f"{entity}是{value}。"


def _previous_answer(attribute: str, value: str) -> str:
    if attribute == "位置":
        return f"之前在{value}。"
    return f"之前是{value}。"


def _previous_query(entity: str, attribute: str, variant: int) -> str:
    if attribute == "颜色":
        templates = (
            "{e}之前是什么颜色？",
            "{e}原来是什么颜色？",
            "{e}之前的颜色？",
            "{e}上次是什么颜色？",
        )
    elif attribute == "位置":
        templates = (
            "{e}之前在哪里？",
            "{e}原来在哪里？",
            "{e}之前的位置？",
            "{e}上次在哪里？",
        )
    else:
        templates = (
            "{e}之前是谁的？",
            "{e}原来归谁？",
            "{e}之前的归属？",
            "{e}上次是谁的？",
        )
    return templates[variant].format(e=entity)


def _episode(history: dict[str, Any], variant: int) -> dict[str, Any]:
    entity, attribute = history["entity"], history["attribute"]
    v0, v1 = history["v0"], history["v1"]
    entity_index = ENTITIES.index(entity)
    attribute_names = tuple(ATTRIBUTES)
    attribute_index = attribute_names.index(attribute)
    values = ATTRIBUTES[attribute]
    distractor_entity = ENTITIES[(entity_index + variant + 1) % len(ENTITIES)]
    distractor_value = values[(values.index(v1) + variant + 1) % len(values)]
    unknown_attribute = attribute_names[
        (attribute_index + 1 + (variant % 2)) % len(attribute_names)
    ]

    fact0 = _fact_text(entity, attribute, v0, variant)
    fact1 = _fact_text(entity, attribute, v1, 0)
    current_query = _query_text(entity, attribute, 0)
    delay = ("我们等会儿再说。", "先记一下，稍后继续。", "好，我先记着。", "等一下再确认。")[
        variant
    ]
    correction = (
        f"更正：{fact1}",
        f"说错了，{fact1}",
        f"{fact1}（刚才说错了）",
        f"改一下，{fact1}",
    )[variant]
    distractor = (
        f"顺便说一下，{_fact_text(distractor_entity, attribute, distractor_value, 0)}",
        f"另外，{_fact_text(distractor_entity, attribute, distractor_value, 0)}",
        f"补充一句，{_fact_text(distractor_entity, attribute, distractor_value, 0)}",
        f"对了，{_fact_text(distractor_entity, attribute, distractor_value, 0)}",
    )[variant]
    previous_query = _previous_query(entity, attribute, variant)
    unknown_query = _query_text(entity, unknown_attribute, variant)

    turns = [
        {
            "index": 0,
            "role": "fact_v0",
            "text": fact0,
            "entity": entity,
            "attribute": attribute,
            "value": v0,
            "category": None,
            "scored": False,
        },
        {
            "index": 1,
            "role": "delay",
            "text": delay,
            "entity": None,
            "attribute": None,
            "value": None,
            "category": None,
            "scored": False,
        },
        {
            "index": 2,
            "role": "query_current",
            "text": current_query,
            "entity": entity,
            "attribute": attribute,
            "value": None,
            "category": "current_query",
            "scored": True,
            "target": _target(_answer_text(entity, attribute, v0)),
            "expected_value": v0,
        },
        {
            "index": 3,
            "role": "correction",
            "text": correction,
            "entity": entity,
            "attribute": attribute,
            "value": v1,
            "category": None,
            "scored": False,
        },
        {
            "index": 4,
            "role": "query_current",
            "text": current_query,
            "entity": entity,
            "attribute": attribute,
            "value": None,
            "category": "current_query",
            "scored": True,
            "target": _target(_answer_text(entity, attribute, v1)),
            "expected_value": v1,
        },
        {
            "index": 5,
            "role": "distractor",
            "text": distractor,
            "entity": distractor_entity,
            "attribute": attribute,
            "value": distractor_value,
            "category": None,
            "scored": False,
        },
        {
            "index": 6,
            "role": "query_previous",
            "text": previous_query,
            "entity": entity,
            "attribute": attribute,
            "value": None,
            "category": "previous_query",
            "scored": True,
            "target": _target(_previous_answer(attribute, v0)),
            "expected_value": v0,
        },
        {
            "index": 7,
            "role": "query_unknown",
            "text": unknown_query,
            "entity": entity,
            "attribute": unknown_attribute,
            "value": None,
            "category": "unknown_query",
            "scored": True,
            "target": _target("我不知道。"),
            "expected_value": None,
        },
    ]
    return {
        "episode_id": f"{history['history_id']}:v{variant}",
        "history_id": history["history_id"],
        "split": history["split"],
        "variant": variant,
        "turns": turns,
    }


def _coverage(histories: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts: dict[str, int] = defaultdict(int)
    slot_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    entity_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    attribute_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    value_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pair_splits: dict[str, set[str]] = defaultdict(set)
    for history in histories:
        if history["split"] == "unused":
            continue
        split = history["split"]
        split_counts[split] += 1
        slot_counts[history["slot"]][split] += 1
        entity_counts[history["entity"]][split] += 1
        attribute_counts[history["attribute"]][split] += 1
        value_counts[history["v0"]][split] += 1
        value_counts[history["v1"]][split] += 1
        pair_splits[history["pair_id"]].add(split)
    return {
        "split_counts": dict(sorted(split_counts.items())),
        "slot_counts": {
            slot: dict(sorted(counts.items())) for slot, counts in sorted(slot_counts.items())
        },
        "entity_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(entity_counts.items())
        },
        "attribute_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(attribute_counts.items())
        },
        "value_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(value_counts.items())
        },
        "pairs_crossing_splits": sorted(
            pair for pair, splits in pair_splits.items() if len(splits) > 1
        ),
    }


def _bytes(episodes: list[dict[str, Any]]) -> dict[str, int]:
    prompt = target = 0
    for episode in episodes:
        for turn in episode["turns"]:
            prompt += len(turn["text"].encode("utf-8"))
            if turn.get("target"):
                target += len(turn["target"]["target_bytes_utf8"])
    return {
        "turn_text_bytes": prompt,
        "target_bytes": target,
        "total_bytes": prompt + target,
        "answer_turns": sum(
            1 for episode in episodes for turn in episode["turns"] if turn["scored"]
        ),
    }


def build_course() -> dict[str, Any]:
    histories = _split_histories()
    episodes = [
        _episode(history, variant)
        for history in histories
        if history["split"] != "unused"
        for variant in range(EPISODES_PER_HISTORY)
    ]
    coverage = _coverage(histories)
    sizes = _bytes(episodes)
    return {
        "format": "taiji-native-core-r1-course-v1",
        "version": 1,
        "world": {
            "entities": list(ENTITIES),
            "attributes": {name: list(values) for name, values in ATTRIBUTES.items()},
            "slots": len(ENTITIES) * len(ATTRIBUTES),
            "possible_histories_per_slot": 12,
            "sealed": False,
        },
        "protocol": {
            "episodes_per_history": EPISODES_PER_HISTORY,
            "turns_per_episode": TURNS_PER_EPISODE,
            "answer_turns_per_episode": len(ANSWER_TURNS),
            "answer_turn_indexes": list(ANSWER_TURNS),
            "end_marker": END_MARKER,
            "supervision": "target_t_at_step_t",
        },
        "histories": histories,
        "episodes": episodes,
        "coverage": coverage,
        "bytes": sizes,
        "split_policy": {
            "allowed_splits": ["train", "query", "holdout"],
            "train_histories": coverage["split_counts"].get("train", 0),
            "query_histories": coverage["split_counts"].get("query", 0),
            "holdout_histories": coverage["split_counts"].get("holdout", 0),
            "unused_histories": sum(1 for item in histories if item["split"] == "unused"),
            "unit": "history chain; counterfactual pairs stay in one split",
            "training_read": {"train": True, "query": False, "holdout": False},
        },
        "model_calls": 0,
        "optimizer_updates": 0,
        "sealed_text_read": 0,
        "can_promote": False,
    }


def audit_course(course: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    checks: list[str] = []

    def check(condition: bool, label: str, detail: str = "") -> None:
        if condition:
            checks.append(label)
        else:
            failures.append(label if not detail else f"{label}: {detail}")

    histories = course["histories"]
    episodes = course["episodes"]
    coverage = course["coverage"]
    check(len(histories) == 432, "36 slots x 6 pairs x 2 histories", str(len(histories)))
    check(coverage["split_counts"].get("train") == 216, "train histories = 216")
    check(coverage["split_counts"].get("query") == 72, "query histories = 72")
    check(coverage["split_counts"].get("holdout") == 72, "holdout histories = 72")
    check(
        sum(1 for item in histories if item["split"] == "unused") == 72,
        "unused histories = 72",
    )
    check(not coverage["pairs_crossing_splits"], "no counterfactual pair crosses splits")
    expected_slot = {"train": 6, "query": 2, "holdout": 2}
    bad_slots = {
        slot: counts
        for slot, counts in coverage["slot_counts"].items()
        if any(counts.get(split, 0) != value for split, value in expected_slot.items())
    }
    check(not bad_slots, "every slot appears 6/2/2 across splits", str(bad_slots))
    expected_entity = {"train": 18, "query": 6, "holdout": 6}
    bad_entities = {
        entity: counts
        for entity, counts in coverage["entity_counts"].items()
        if any(counts.get(split, 0) != value for split, value in expected_entity.items())
    }
    check(not bad_entities, "every entity appears 18/6/6 across splits", str(bad_entities))
    check(len(episodes) == 1440, "4 episodes x 360 histories", str(len(episodes)))
    check(
        all(len(episode["turns"]) == TURNS_PER_EPISODE for episode in episodes),
        "every episode has 8 turns",
    )
    check(
        all(
            sum(1 for turn in episode["turns"] if turn["scored"]) == len(ANSWER_TURNS)
            for episode in episodes
        ),
        "every episode has 4 scored answer turns",
    )

    wrong_targets = []
    for episode in episodes:
        v0 = episode["turns"][0]["value"]
        v1 = episode["turns"][3]["value"]
        if episode["turns"][2]["target"]["target_text"] != _answer_text(
            episode["turns"][0]["entity"], episode["turns"][0]["attribute"], v0
        ):
            wrong_targets.append(episode["episode_id"])
        if episode["turns"][4]["target"]["target_text"] != _answer_text(
            episode["turns"][0]["entity"], episode["turns"][0]["attribute"], v1
        ):
            wrong_targets.append(episode["episode_id"])
        if episode["turns"][6]["target"]["target_text"] != _previous_answer(
            episode["turns"][0]["attribute"], v0
        ):
            wrong_targets.append(episode["episode_id"])
        if episode["turns"][7]["target"]["target_text"] != "我不知道。":
            wrong_targets.append(episode["episode_id"])
    check(not wrong_targets, "answer targets match current/corrected/previous/unknown")

    sizes = course["bytes"]
    check(sizes["answer_turns"] == 5760, "scored answer turns = 5760", str(sizes["answer_turns"]))
    check(sizes["total_bytes"] > 0, "actual text bytes are non-zero")
    check(
        sizes["total_bytes"] <= 1113934,
        "actual text bytes stay under the R1 upper estimate",
        str(sizes["total_bytes"]),
    )
    check(course["world"]["sealed"] is False, "course is public synthetic and not sealed")
    check(course["model_calls"] == 0, "no model call")
    check(course["optimizer_updates"] == 0, "no optimizer update")

    return {
        "schema": "taiji-native-core-r1-course-audit-v1",
        "status": "course_built" if not failures else "course_failed",
        "checks": checks,
        "failures": failures,
        "coverage": coverage,
        "bytes": sizes,
        "byte_estimate_note": (
            "R1 projected 795,667-1,113,934 bytes from old E1 per-turn bytes; "
            "the materialised templates are shorter, so only the upper bound is gated."
        ),
        "digest": content_digest(course),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def write_course(course: dict[str, Any], audit: dict[str, Any]) -> None:
    for path in (SPLIT_MANIFEST, *COURSE_FILES.values(), REPORT):
        if path.exists():
            raise SystemExit(f"manifest/report already exists (write-once): {path}")
    split_payload = {
        "format": "taiji-native-core-r1-split-v1",
        "version": 1,
        "salt": SPLIT_SALT,
        "split_policy": course["split_policy"],
        "histories": [
            {
                "history_id": item["history_id"],
                "entity": item["entity"],
                "attribute": item["attribute"],
                "v0": item["v0"],
                "v1": item["v1"],
                "slot": item["slot"],
                "pair_id": item["pair_id"],
                "split": item["split"],
            }
            for item in course["histories"]
        ],
    }
    _write_json(SPLIT_MANIFEST, split_payload)
    for split, path in COURSE_FILES.items():
        episodes = [item for item in course["episodes"] if item["split"] == split]
        _write_json(
            path,
            {
                "format": f"taiji-native-core-r1-{split}-v1",
                "version": 1,
                "world": course["world"],
                "protocol": course["protocol"],
                "split": split,
                "training_read": split == "train",
                "episodes": episodes,
                "bytes": _bytes(episodes),
                "model_calls": 0,
                "sealed_text_read": 0,
            },
        )
    files = {
        str(path.relative_to(ROOT))
        .replace("\\", "/"): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in (SPLIT_MANIFEST, *COURSE_FILES.values())
    }
    audit["files"] = files
    _write_json(REPORT, audit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only", action="store_true", help="build and audit without writing"
    )
    args = parser.parse_args()
    started = time.process_time()
    course = build_course()
    audit = audit_course(course)
    audit["cpu_seconds"] = time.process_time() - started
    if args.report_only:
        print(json.dumps(audit, ensure_ascii=False))
    else:
        write_course(course, audit)
        print(
            json.dumps(
                {
                    "status": audit["status"],
                    "train": audit["coverage"]["split_counts"].get("train"),
                    "query": audit["coverage"]["split_counts"].get("query"),
                    "holdout": audit["coverage"]["split_counts"].get("holdout"),
                    "bytes": audit["bytes"]["total_bytes"],
                    "failures": audit["failures"],
                },
                ensure_ascii=False,
            )
        )
    return 0 if audit["status"] == "course_built" else 1


if __name__ == "__main__":
    raise SystemExit(main())

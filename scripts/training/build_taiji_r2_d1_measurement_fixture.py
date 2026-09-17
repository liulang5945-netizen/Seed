"""Build the R2-D1 distinguishable measurement fixture (frozen contract D1).

Contract: plans/reference/M5_R2_D1_DEV_MEASUREMENT_CONTRACT_FROZEN_20260917.md

Deterministic three-split corpus (train/dev/final) with disjoint entities,
templates and combinations, seven shapes, balanced counterfactual flip pairs
whose members share the question stem and differ in exactly one deciding fact,
and fact-preserving invariance pairs carrying an irrelevant distractor.
Unknown items are emitted once per object (never per color), and every
(prefix, response) is unique.  The module also hosts the reference solver used
by the Q5 self-check: it parses the background material deterministically and
never reads model outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
DATA_REPORT = Path("reports/r2_d1_data_contract_20260917.json")
FIXTURE_FORMAT = "r2-d1-measurement-v1"
ORDER_SEED = 0xD1

SHAPES = (
    "fact",
    "negation",
    "unknown",
    "same_opening_fact",
    "same_opening_unknown",
    "combination_same",
    "combination_different",
)
CONTENT_INDEPENDENT = {"unknown", "same_opening_unknown"}
FLIP_SHAPES = ("fact", "combination_same", "combination_different")

# Disjoint entity/color pools.  Words are taken from the earlier sweep word
# banks but the slices themselves are this fixture's own identity.
POOLS: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "objects": (
            "天空",
            "草",
            "树叶",
            "海洋",
            "山川",
            "城市",
            "森林",
            "河流",
        ),
        "colors": ("蓝", "绿", "红", "黄", "青", "紫"),
    },
    "dev": {
        "objects": ("雪", "太阳", "湖水", "云层", "礁石", "晨雾"),
        "colors": ("白", "灰白", "乳白", "米色"),
    },
    "final": {
        "objects": ("夜晚", "大海", "火焰", "冰川", "星辰", "银河"),
        "colors": ("黑", "墨黑", "漆黑", "暗红"),
    },
}

# Distinct phrasing per split, same information structure.
TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "fact": {
        "train": {"prefix": "问：{o}是什么颜色？{bg}{o}是{c}。{tail}", "answer": "{c}"},
        "dev": {"prefix": "提问：{o}的颜色？{bg}{o}是{c}。{tail}", "answer": "{c}"},
        "final": {"prefix": "查询：{o}的颜色。{bg}{o}是{c}。{tail}", "answer": "{c}"},
    },
    "negation": {
        "train": {"prefix": "问：{o}是{c}吗？{bg}不是{c}。{tail}", "answer": "不是{c}"},
        "dev": {"prefix": "提问：{o}是否为{c}？{bg}{o}不是{c}。{tail}", "answer": "不是{c}"},
        "final": {"prefix": "查询：{o}属于{c}吗。{bg}{o}不是{c}。{tail}", "answer": "不是{c}"},
    },
    "unknown": {
        "train": {"prefix": "问：{o}是什么颜色？{bg}没有关于{o}的信息。{tail}", "answer": "未知"},
        "dev": {"prefix": "提问：{o}的颜色？{bg}{o}的信息缺失。{tail}", "answer": "未知"},
        "final": {"prefix": "查询：{o}的颜色。{bg}{o}的资料为空。{tail}", "answer": "未知"},
    },
    "same_opening_fact": {
        "train": {"prefix": "问：请判断{o}的颜色。{bg}{o}是{c}。{tail}", "answer": "{c}"},
        "dev": {"prefix": "提问：请判断{o}的颜色。{bg}{o}是{c}。{tail}", "answer": "{c}"},
        "final": {"prefix": "查询：请判断{o}的颜色。{bg}{o}是{c}。{tail}", "answer": "{c}"},
    },
    "same_opening_unknown": {
        "train": {"prefix": "问：请判断{o}的颜色。{bg}没有关于{o}的信息。{tail}", "answer": "未知"},
        "dev": {"prefix": "提问：请判断{o}的颜色。{bg}{o}的信息缺失。{tail}", "answer": "未知"},
        "final": {"prefix": "查询：请判断{o}的颜色。{bg}{o}的资料为空。{tail}", "answer": "未知"},
    },
    "combination_same": {
        "train": {
            "prefix": "问：{o1}和{o2}颜色相同吗？{bg}{o1}是{c}，{o2}是{c}。{tail}",
            "answer": "相同",
        },
        "dev": {
            "prefix": "提问：{o1}与{o2}的颜色一致吗？{bg}{o1}是{c}，{o2}是{c}。{tail}",
            "answer": "相同",
        },
        "final": {
            "prefix": "查询：{o1}、{o2}颜色是否相同。{bg}{o1}是{c}，{o2}是{c}。{tail}",
            "answer": "相同",
        },
    },
    "combination_different": {
        "train": {
            "prefix": "问：{o1}和{o2}颜色相同吗？{bg}{o1}是{c1}，{o2}是{c2}。{tail}",
            "answer": "不同",
        },
        "dev": {
            "prefix": "提问：{o1}与{o2}的颜色一致吗？{bg}{o1}是{c1}，{o2}是{c2}。{tail}",
            "answer": "不同",
        },
        "final": {
            "prefix": "查询：{o1}、{o2}颜色是否相同。{bg}{o1}是{c1}，{o2}是{c2}。{tail}",
            "answer": "不同",
        },
    },
}

CLAUSES = {
    "train": {"bg": "背景：", "tail": "答："},
    "dev": {"bg": "线索：", "tail": "回答："},
    "final": {"bg": "已知：", "tail": "输出："},
}
# dev-only: distractor wording inside the material clause (invariance pairs).
DISTRACTOR = {
    "dev": "另有资料称{other}是{dc}；",
}


def _records_for_split(split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    objects = POOLS[split]["objects"]
    colors = POOLS[split]["colors"]
    clauses = CLAUSES[split]
    records: list[dict[str, Any]] = []
    counters = {shape: 0 for shape in SHAPES}
    pair_index: dict[str, dict[str, str]] = {}

    def add(
        shape: str,
        template_slots: dict[str, str],
        *,
        pair_id: str | None = None,
        pair_role: str | None = None,
        pair_type: str | None = None,
    ) -> str:
        counters[shape] += 1
        item_id = f"{split}:{shape}:{counters[shape]:03d}"
        variant = TEMPLATES[shape][split]
        slots = dict(clauses)
        slots.update(template_slots)
        prefix = variant["prefix"].format(**slots)
        response = variant["answer"].format(**slots)
        record: dict[str, Any] = {
            "id": item_id,
            "split": split,
            "shape": shape,
            "source_group": shape,
            "content_dependent": shape not in CONTENT_INDEPENDENT,
            "prefix": prefix,
            "response": response,
            "pair_id": pair_id,
            "pair_role": pair_role,
            "pair_type": pair_type,
        }
        records.append(record)
        if pair_id is not None:
            pair_index.setdefault(pair_id, {})[str(pair_role)] = item_id
        return item_id

    # Fact / negation / same-opening families.  Unknown shapes are emitted once
    # per object -- the duplication the design audit found is removed here.
    fact_ids: dict[tuple[str, str], str] = {}
    for obj in objects:
        for color in colors:
            fact_ids[(obj, color)] = add("fact", {"o": obj, "c": color})
            add("negation", {"o": obj, "c": color})
            add("same_opening_fact", {"o": obj, "c": color})
        add("unknown", {"o": obj})
        add("same_opening_unknown", {"o": obj})

    # fact_flip pairs: links between two existing fact items (colors 0 vs 1).
    for obj in objects:
        add_link = fact_ids[(obj, colors[0])]
        b_id = fact_ids[(obj, colors[1])]
        pid = f"{split}:fact_flip:{obj}"
        for linked in (add_link, b_id):
            record = next(item for item in records if item["id"] == linked)
            record["pair_id"] = pid
            record["pair_type"] = "fact_flip"
        records[next(i for i, item in enumerate(records) if item["id"] == add_link)][
            "pair_role"
        ] = "a"
        records[next(i for i, item in enumerate(records) if item["id"] == b_id)]["pair_role"] = "b"

    # combination items: adjacent object pairs (n objects -> n-1 pairs, no
    # wrap-around), aligned same/different members sharing one question stem.
    combo_count = len(objects) - 1
    for i in range(combo_count):
        o1 = objects[i]
        o2 = objects[(i + 1) % len(objects)]
        c1 = colors[i % len(colors)]
        c2 = colors[(i + 1) % len(colors)]
        pid = f"{split}:combo_flip:{i:02d}"
        add(
            "combination_same",
            {"o1": o1, "o2": o2, "c": c1},
            pair_id=pid,
            pair_role="a",
            pair_type="combo_flip",
        )
        add(
            "combination_different",
            {"o1": o1, "o2": o2, "c1": c1, "c2": c2},
            pair_id=pid,
            pair_role="b",
            pair_type="combo_flip",
        )

    # invariance pairs (dev only): an existing fact item and its distractor
    # twin.  Originals must be ordinary fact items, so pick colors outside the
    # fact_flip band (flip pairs consume colors[0]/colors[1] per object); a
    # record carries a single pair link and must never serve two pairs.
    invariance_pairs = 0
    if split == "dev":
        for k in range(4):
            obj = objects[k]
            color = colors[2 + (k % 2)]
            other = objects[k + 2]
            distractor_color = colors[3] if color == colors[2] else colors[2]
            original_id = fact_ids[(obj, color)]
            pid = f"{split}:invariance:{invariance_pairs:02d}"
            original = next(item for item in records if item["id"] == original_id)
            original["pair_id"] = pid
            original["pair_role"] = "a"
            original["pair_type"] = "invariance"
            distractor = DISTRACTOR["dev"].format(other=other, dc=distractor_color)
            add(
                "fact",
                {"o": obj, "c": color, "bg": clauses["bg"] + distractor},
                pair_id=pid,
                pair_role="b",
                pair_type="invariance",
            )
            # the distractor row stays shape=fact but is marked for stats
            records[-1]["invariance_distractor"] = True
            invariance_pairs += 1

    metadata = {
        "fact_flip_pairs": len(objects),
        "combo_flip_pairs": combo_count,
        "invariance_pairs": invariance_pairs,
    }
    return records, metadata


def _deterministic_order(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Deterministically search for a permutation with no adjacent pair members.

    The seed is part of the generator identity and is reported in metadata.
    """

    for seed in range(10_000):
        order = list(range(len(records)))
        random.Random(ORDER_SEED + seed).shuffle(order)
        arranged = [records[index] for index in order]
        positions: dict[str, list[int]] = {}
        for index, record in enumerate(arranged):
            if record["pair_id"]:
                positions.setdefault(record["pair_id"], []).append(index)
        if all(len(p) != 2 or abs(p[0] - p[1]) > 1 for p in positions.values()):
            return arranged, ORDER_SEED + seed
    raise RuntimeError("no deterministic ordering satisfies pair non-adjacency")


MATERIAL_CLAUSE = re.compile(r"背景：|线索：|已知：")


def _split_clauses(prefix: str) -> tuple[str, str]:
    """Return (question stem, material clause) for a rendered prefix."""

    parts = MATERIAL_CLAUSE.split(prefix, maxsplit=1)
    if len(parts) != 2:
        raise RuntimeError(f"cannot locate material clause: {prefix}")
    return parts[0], parts[1]


def _stem_head(stem: str) -> str:
    """Drop the speech-act prefix and the 请判断 marker from a question stem."""

    head = stem.split("：", 1)[1] if "：" in stem else stem
    return head.replace("请判断", "")


#: Split-specific answer-tail markers (R2-D2 contract section 4.2: the
#: no-context model baseline keeps the question stem and the answer cue).
TAIL_MARKERS = {"背景：": "答：", "线索：": "回答：", "已知：": "输出："}


def remove_material_clause(prefix: str) -> str:
    """Delete the background material while keeping the question stem and cue.

    Used by the context-removed model baseline: same checkpoint, no retrain,
    evaluated on stem-only prefixes.  The material marker identifies the
    split, so the full tail marker is chosen explicitly (``答：`` must not
    match inside dev's ``回答：``).
    """

    match = MATERIAL_CLAUSE.search(prefix)
    if match is None:
        raise RuntimeError(f"cannot locate material clause: {prefix}")
    stem, rest = prefix[: match.start()], prefix[match.end() :]
    bg_marker = match.group(0)
    tail = TAIL_MARKERS[bg_marker]
    if tail not in rest:
        raise RuntimeError(f"cannot locate answer tail marker {tail}: {prefix}")
    return stem + tail


def reference_answer(record: dict[str, Any]) -> str:
    """Deterministic material reader for the Q5 self-check.  Parses background."""

    shape = record["shape"]
    prefix = record["prefix"]
    if shape in ("unknown", "same_opening_unknown"):
        return "未知"
    stem, material = _split_clauses(prefix)
    if shape == "negation":
        # the queried color lives in the question stem; the three splits use
        # 是{c}吗 / 是否为{c} / 属于{c}吗 wording.
        head = _stem_head(stem)
        if "是否为" in head:
            asked = head.split("是否为", 1)[1]
        elif "属于" in head:
            asked = head.split("属于", 1)[1]
        else:
            match = re.search(r"是(.+?)[吗？。?]", head)
            if not match:
                raise RuntimeError(f"cannot parse negation stem: {prefix}")
            asked = match.group(1)
        asked = asked.rstrip("吗？。?")
        return f"不是{asked}"
    if shape in ("combination_same", "combination_different"):
        # parse the material clause only; the final-split stem itself reads
        # "是否相同" and would otherwise leak a spurious 是-token
        colors_found = [token for token in re.findall(r"是([^，。；;]+)", material) if token]
        if len(colors_found) < 2:
            raise RuntimeError(f"cannot parse combo material: {prefix}")
        return "相同" if colors_found[-2] == colors_found[-1] else "不同"
    # fact / same_opening_fact (including the distractor twin): the answer is
    # the color in the statement about the queried object, ignoring other facts.
    head = _stem_head(stem)
    obj_match = re.match(r"(.+?)(?:的颜色|是什么颜色)", head)
    if not obj_match:
        raise RuntimeError(f"cannot parse fact object: {prefix}")
    obj = obj_match.group(1)
    statement = re.search(re.escape(obj) + r"是([^。；;]+)", material)
    if not statement:
        raise RuntimeError(f"cannot parse fact statement: {prefix}")
    return statement.group(1)


def _verify(
    all_records: dict[str, list[dict[str, Any]]], metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    splits = sorted(all_records)
    prefixes = {s: {r["prefix"] for r in all_records[s]} for s in splits}
    pairs_set = {s: {(r["prefix"], r["response"]) for r in all_records[s]} for s in splits}
    entities = {s: {e for e in POOLS[s]["objects"] + POOLS[s]["colors"]} for s in splits}
    unknown_counts = {
        s: sum(1 for r in all_records[s] if r["shape"] in ("unknown", "same_opening_unknown"))
        for s in splits
    }
    objects_n = {s: len(POOLS[s]["objects"]) for s in splits}
    checks: dict[str, Any] = {
        "unique_prefixes_within_split": all(
            len(prefixes[s]) == len(all_records[s]) for s in splits
        ),
        "prefixes_disjoint_across_splits": all(
            not (prefixes[a] & prefixes[b]) for i, a in enumerate(splits) for b in splits[i + 1 :]
        ),
        "pairs_disjoint_across_splits": all(
            not (pairs_set[a] & pairs_set[b]) for i, a in enumerate(splits) for b in splits[i + 1 :]
        ),
        "entities_disjoint_across_splits": all(
            not (entities[a] & entities[b]) for i, a in enumerate(splits) for b in splits[i + 1 :]
        ),
        "all_shapes_in_every_split": all(
            {r["shape"] for r in all_records[s]} == set(SHAPES) for s in splits
        ),
        "unknown_once_per_object": all(unknown_counts[s] == 2 * objects_n[s] for s in splits),
        # Frozen counts (contract section 2): dev must carry 6 fact_flip +
        # 5 combo_flip = 11 flip pairs; the other splits follow one fact pair
        # per object and one combo pair per adjacent object pair (n objects - 1).
        "flip_pair_counts": (
            metadata["dev"]["fact_flip_pairs"] == 6
            and metadata["dev"]["combo_flip_pairs"] == 5
            and metadata["train"]["fact_flip_pairs"] == objects_n["train"]
            and metadata["train"]["combo_flip_pairs"] == objects_n["train"] - 1
            and metadata["final"]["fact_flip_pairs"] == objects_n["final"]
            and metadata["final"]["combo_flip_pairs"] == objects_n["final"] - 1
        ),
        "invariance_pairs_dev": metadata["dev"]["invariance_pairs"] == 4,
        "pair_members_non_adjacent": all(_non_adjacent(all_records[s]) for s in splits),
        "pair_structure": all(_pair_structure(all_records[s]) for s in splits),
        "reference_self_check": all(
            reference_answer(record) == record["response"]
            for s in splits
            for record in all_records[s]
        ),
    }
    return checks


def _non_adjacent(records: list[dict[str, Any]]) -> bool:
    positions: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record["pair_id"]:
            positions.setdefault(record["pair_id"], []).append(index)
    return all(len(p) != 2 or abs(p[0] - p[1]) > 1 for p in positions.values())


def _pair_structure(records: list[dict[str, Any]]) -> bool:
    """Each link has exactly two members sharing one question stem; flip pairs
    require differing answers and invariance pairs require identical answers."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["pair_id"]:
            groups.setdefault(record["pair_id"], []).append(record)
    for members in groups.values():
        if len(members) != 2:
            return False
        a, b = members
        if a["pair_type"] != b["pair_type"]:
            return False
        stem_a = MATERIAL_CLAUSE.split(a["prefix"], maxsplit=1)[0]
        stem_b = MATERIAL_CLAUSE.split(b["prefix"], maxsplit=1)[0]
        if stem_a != stem_b:
            return False
        if a["pair_type"] == "invariance":
            if a["response"] != b["response"]:
                return False
        elif a["response"] == b["response"]:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--report", type=Path, default=DATA_REPORT)
    args = parser.parse_args()

    raw: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev", "final"):
        records, meta = _records_for_split(split)
        ordered, order_seed = _deterministic_order(records)
        raw[split] = ordered
        metadata[split] = {**meta, "order_seed": order_seed}
    checks = _verify(raw, metadata)
    failed = [name for name, passed in checks.items() if passed is not True and passed is not None]

    if checks["reference_self_check"] is not True:
        mismatches = [
            {"id": record["id"], "expected": record["response"], "got": reference_answer(record)}
            for split in ("train", "dev", "final")
            for record in raw[split]
            if reference_answer(record) != record["response"]
        ]
        print(json.dumps({"reference_mismatches": mismatches[:10]}, ensure_ascii=False))

    from taiji.internalization import content_digest

    digest = content_digest([r for s in ("train", "dev", "final") for r in raw[s]])

    import hashlib

    fixture_path = PROJECT_ROOT / args.fixture
    file_sha: str | None = None
    if not failed:
        # contract section 1.2: the fixture is written only after every gate
        # passes; a failed generator run must never overwrite a sealed fixture.
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        with fixture_path.open("w", encoding="utf-8", newline="\n") as handle:
            for split in ("train", "dev", "final"):
                for record in raw[split]:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        file_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()

    report = {
        "format": FIXTURE_FORMAT,
        "version": 1,
        "generator": "scripts/training/build_taiji_r2_d1_measurement_fixture.py",
        "fixture": str(args.fixture),
        "fixture_sha256": file_sha,
        "corpus_digest": digest,
        "splits": {
            split: {
                "episodes": len(raw[split]),
                "unique_pairs": len({(r["prefix"], r["response"]) for r in raw[split]}),
                "shapes": {
                    shape: sum(1 for r in raw[split] if r["shape"] == shape) for shape in SHAPES
                },
                "pairs": metadata[split],
            }
            for split in ("train", "dev", "final")
        },
        "checks": checks,
        "outcome": "passed" if not failed else "failed",
        "failed_checks": failed,
    }
    report_path = PROJECT_ROOT / args.report
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"outcome": report["outcome"], "failed": failed, "digest": digest[:12]}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build and verify the R2-H3.8 joint-sequence corpus (data contract step).

Contract: plans/reference/M5_R2_H3_8_ISOLATED_PROTOTYPE_CONTRACT_20260917.md
section 3.  Deterministic generation with template / entity / combination
separation across train, dev, and final:

* train / dev / final use disjoint entity pools and disjoint template phrasings;
* no (template, entity) combination repeats across splits and every split
  contains all five adversarial shapes (fact replacement, negation, unknown,
  combination constraint, same-opening-different-content);
* prefixes are unique within a split and never overlap across splits;
* the generator records the corpus digest and writes the fixture; generation
  rules are never exposed to any model.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))

from taiji.internalization import content_digest  # noqa: E402

FIXTURE = Path("tests/fixtures/r2_h3_8_joint_sequence_v1.jsonl")
REPORT = Path("reports/r2_h3_8_data_contract_20260917.json")
FORMAT = "taiji-r2-h3-8-joint-sequence-corpus-v1"
GENERATION_SEED = 20260917

ENTITY_POOLS = {
    "train": {"objects": ("天空", "草", "树叶"), "colors": ("蓝", "绿")},
    "dev": {"objects": ("雪", "太阳"), "colors": ("白", "黄")},
    "final": {"objects": ("夜晚", "大海"), "colors": ("黑", "橙")},
}

TEMPLATE_VARIANTS: dict[str, dict[str, dict[str, str]]] = {
    "fact": {
        "train": {"prefix": "问：{o}是什么颜色？背景：{o}是{c}。答：", "answer": "{c}"},
        "dev": {"prefix": "提问：{o}的颜色？线索：{o}是{c}。回答：", "answer": "{c}"},
        "final": {"prefix": "查询：{o}的颜色。已知：{o}是{c}。输出：", "answer": "{c}"},
    },
    "negation": {
        "train": {"prefix": "问：{o}是{c}吗？背景：不是{c}。答：", "answer": "不是{c}"},
        "dev": {"prefix": "提问：{o}是否为{c}？线索：{o}不是{c}。回答：", "answer": "不是{c}"},
        "final": {"prefix": "查询：{o}属于{c}吗。已知：{o}不是{c}。输出：", "answer": "不是{c}"},
    },
    "unknown": {
        "train": {"prefix": "问：{o}是什么颜色？背景：没有关于{o}的信息。答：", "answer": "未知"},
        "dev": {"prefix": "提问：{o}的颜色？线索：{o}的信息缺失。回答：", "answer": "未知"},
        "final": {"prefix": "查询：{o}的颜色。已知：{o}的资料为空。输出：", "answer": "未知"},
    },
    "combination_same": {
        "train": {
            "prefix": "问：{o1}和{o2}颜色相同吗？背景：{o1}是{c}，{o2}是{c}。答：",
            "answer": "相同",
        },
        "dev": {
            "prefix": "提问：{o1}与{o2}的颜色一致吗？线索：{o1}是{c}，{o2}是{c}。回答：",
            "answer": "相同",
        },
        "final": {
            "prefix": "查询：{o1}、{o2}颜色是否相同。已知：{o1}是{c}，{o2}是{c}。输出：",
            "answer": "相同",
        },
    },
    "combination_different": {
        "train": {
            "prefix": "问：{o1}和{o2}颜色相同吗？背景：{o1}是{c1}，{o2}是{c2}。答：",
            "answer": "不同",
        },
        "dev": {
            "prefix": "提问：{o1}与{o2}的颜色一致吗？线索：{o1}是{c1}，{o2}是{c2}。回答：",
            "answer": "不同",
        },
        "final": {
            "prefix": "查询：{o1}、{o2}颜色是否相同。已知：{o1}是{c1}，{o2}是{c2}。输出：",
            "answer": "不同",
        },
    },
    # same opening, different content: both variants share the opening phrase
    "same_opening_fact": {
        "train": {"prefix": "问：请判断{o}的颜色。背景：{o}是{c}。答：", "answer": "{c}"},
        "dev": {"prefix": "提问：请判断{o}的颜色。线索：{o}是{c}。回答：", "answer": "{c}"},
        "final": {"prefix": "查询：请判断{o}的颜色。已知：{o}是{c}。输出：", "answer": "{c}"},
    },
    "same_opening_unknown": {
        "train": {"prefix": "问：请判断{o}的颜色。背景：没有关于{o}的信息。答：", "answer": "未知"},
        "dev": {"prefix": "提问：请判断{o}的颜色。线索：{o}的信息缺失。回答：", "answer": "未知"},
        "final": {"prefix": "查询：请判断{o}的颜色。已知：{o}的资料为空。输出：", "answer": "未知"},
    },
}

SHAPES = {
    "fact": "fact_replacement",
    "negation": "negation",
    "unknown": "unknown_information",
    "combination_same": "combination_constraint",
    "combination_different": "combination_constraint",
    "same_opening_fact": "same_opening_different_content",
    "same_opening_unknown": "same_opening_different_content",
}


def _records_for_split(split: str) -> list[dict[str, Any]]:
    pool = ENTITY_POOLS[split]
    objects = pool["objects"]
    colors = pool["colors"]
    records: list[dict[str, Any]] = []

    def add(
        template_id: str,
        shape: str,
        prefix: str,
        answer: str,
        entities: tuple[str, ...],
    ) -> None:
        records.append(
            {
                "split": split,
                "template_id": f"{split}:{template_id}",
                "shape": shape,
                "entities": list(entities),
                "prefix": prefix,
                "response": answer,
            }
        )

    for obj in objects:
        for color in colors:
            variant = TEMPLATE_VARIANTS["fact"][split]
            add(
                "fact",
                SHAPES["fact"],
                variant["prefix"].format(o=obj, c=color),
                variant["answer"].format(o=obj, c=color),
                (obj, color),
            )
            variant = TEMPLATE_VARIANTS["negation"][split]
            add(
                "negation",
                SHAPES["negation"],
                variant["prefix"].format(o=obj, c=color),
                variant["answer"].format(o=obj, c=color),
                (obj, color),
            )
        variant = TEMPLATE_VARIANTS["unknown"][split]
        add(
            "unknown",
            SHAPES["unknown"],
            variant["prefix"].format(o=obj),
            variant["answer"],
            (obj,),
        )
        variant = TEMPLATE_VARIANTS["same_opening_fact"][split]
        add(
            "same_opening_fact",
            SHAPES["same_opening_fact"],
            variant["prefix"].format(o=obj, c=colors[0]),
            variant["answer"].format(o=obj, c=colors[0]),
            (obj, colors[0]),
        )
        variant = TEMPLATE_VARIANTS["same_opening_unknown"][split]
        add(
            "same_opening_unknown",
            SHAPES["same_opening_unknown"],
            variant["prefix"].format(o=obj),
            variant["answer"],
            (obj,),
        )
    for first, second in list(itertools.combinations(objects, 2))[:2]:
        variant = TEMPLATE_VARIANTS["combination_same"][split]
        add(
            "combination_same",
            SHAPES["combination_same"],
            variant["prefix"].format(o1=first, o2=second, c=colors[0]),
            variant["answer"],
            (first, second, colors[0]),
        )
        variant = TEMPLATE_VARIANTS["combination_different"][split]
        add(
            "combination_different",
            SHAPES["combination_different"],
            variant["prefix"].format(o1=first, o2=second, c1=colors[0], c2=colors[1]),
            variant["answer"],
            (first, second, colors[0], colors[1]),
        )
    return records


def _verify(by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    prefixes = {split: {r["prefix"] for r in records} for split, records in by_split.items()}
    templates = {split: {r["template_id"] for r in records} for split, records in by_split.items()}
    entities = {
        split: {entity for r in records for entity in r["entities"]}
        for split, records in by_split.items()
    }
    combos = {
        split: {(r["template_id"], tuple(r["entities"])) for r in records}
        for split, records in by_split.items()
    }
    splits = sorted(by_split)
    checks: dict[str, Any] = {
        "unique_prefixes_within_split": all(
            len(prefixes[split]) == len(by_split[split]) for split in splits
        ),
        "prefixes_disjoint_across_splits": all(
            not (prefixes[a] & prefixes[b])
            for index, a in enumerate(splits)
            for b in splits[index + 1 :]
        ),
        "templates_disjoint_across_splits": all(
            not (templates[a] & templates[b])
            for index, a in enumerate(splits)
            for b in splits[index + 1 :]
        ),
        "entities_disjoint_across_splits": all(
            not (entities[a] & entities[b])
            for index, a in enumerate(splits)
            for b in splits[index + 1 :]
        ),
        "combinations_disjoint_across_splits": all(
            not (combos[a] & combos[b])
            for index, a in enumerate(splits)
            for b in splits[index + 1 :]
        ),
        "all_shapes_in_every_split": all(
            {r["shape"] for r in by_split[split]}
            == {
                "fact_replacement",
                "negation",
                "unknown_information",
                "combination_constraint",
                "same_opening_different_content",
            }
            for split in splits
        ),
        "same_opening_pairs_share_opening": _same_opening_ok(by_split),
    }
    return checks


def _same_opening_ok(by_split: dict[str, list[dict[str, Any]]]) -> bool:
    for records in by_split.values():
        for record in records:
            if record["shape"] != "same_opening_different_content":
                continue
            partner_openings = [
                other["prefix"][:9]
                for other in records
                if other["shape"] == "same_opening_different_content" and other is not record
            ]
            if record["prefix"][:9] not in partner_openings:
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    by_split = {split: _records_for_split(split) for split in ("train", "dev", "final")}
    checks = _verify(by_split)
    failed = [name for name, passed in checks.items() if not passed]
    digest = content_digest([record for split in sorted(by_split) for record in by_split[split]])

    fixture_path = PROJECT_ROOT / args.fixture
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    with fixture_path.open("w", encoding="utf-8", newline="\n") as handle:
        for split in sorted(by_split):
            for record in by_split[split]:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "format": FORMAT,
        "version": 1,
        "generation_seed": GENERATION_SEED,
        "generator": "scripts/training/build_taiji_r2_h3_8_corpus.py",
        "fixture": str(args.fixture),
        "corpus_digest": digest,
        "splits": {
            split: {
                "episodes": len(records),
                "shapes": sorted({r["shape"] for r in records}),
                "entities": sorted({entity for r in records for entity in r["entities"]}),
            }
            for split, records in sorted(by_split.items())
        },
        "checks": checks,
        "outcome": "passed" if not failed else "failed",
        "failed_checks": failed,
        "policy": (
            "generation rules are never exposed to any model; the old H3.x 12/8/4 "
            "fixture stays engineering regression only"
        ),
        "growth_admitted": False,
        "can_promote": False,
    }
    report_path = PROJECT_ROOT / args.report
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"outcome": report["outcome"], "digest": digest, "failed": failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

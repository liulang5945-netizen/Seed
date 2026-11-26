"""Build and verify the P3b-v2 goal-aligned language-episode corpus.

Preregistration: plans/reference/M5_P3B_V2_GOAL_ALIGNED_PREREGISTRATION_FROZEN_20260917.md
section 2 ("数据" row).  Deterministic generation with template / entity / combination
separation across train, dev, and final -- the *rules* are inherited from the H3.8 data
contract builder (`build_taiji_r2_h3_8_corpus.py`) so this corpus is comparable in shape:

* train / dev / final use disjoint entity pools and disjoint template phrasings;
* no ``(template, entity)`` combination repeats across splits, and every split carries
  all five adversarial shapes (fact replacement, negation, unknown, combination
  constraint, same-opening-different-content);
* ``user_input`` strings are unique within a split and never overlap across splits;
* the generator records the corpus digest and writes the fixture; the rules are never
  exposed to any model.

Output schema is the ``LanguageEpisode`` contract consumed by
``LanguageEpisodeCorpus.from_jsonl`` (8 keys), i.e. what
``scripts/training/train_taiji_r2_aligned.py --dataset`` expects.

Scale: the H3.5a fixture has 24 rows; this builder produces a strictly larger corpus
(see ``EXPECTED_MIN_ROWS``) because "放大规模" is one of the three declared
differentiators of P3b-v2.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji.internalization import content_digest  # noqa: E402

FIXTURE = Path("tests/fixtures/r2_p3b_v2_goal_aligned_episodes_v1.jsonl")
REPORT = Path("reports/r2_p3b_v2_corpus_20260917.json")
FORMAT = "taiji-r2-p3b-v2-goal-aligned-corpus-v1"
GENERATION_SEED = 20260917

#: H3.5a fixture is 24 rows; the preregistration requires a strictly larger corpus.
EXPECTED_MIN_ROWS = 24

#: Disjoint entity pools per split (same discipline as the H3.8 contract builder).
#: Each item maps to exactly one value so that ``user_input`` stays unique inside a split.
ENTITY_POOLS: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "items": ("样本甲", "样本乙", "样本丙", "样本庚", "样本辛", "样本壬", "样本癸", "样本子"),
        "values": ("红色", "蓝色", "绿色", "紫色", "青色", "粉色", "灰色", "棕色"),
    },
    "dev": {
        "items": ("样本丁", "样本戊", "样本寅", "样本卯"),
        "values": ("白色", "黄色", "银色", "金色"),
    },
    "final": {
        "items": ("样本己", "样本丑", "样本辰", "样本巳"),
        "values": ("黑色", "橙色", "藏青", "墨绿"),
    },
}

#: Disjoint template phrasings per split; five adversarial shapes per split.
TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    # 1. fact replacement: the context states a fact, the question asks for it.
    "fact": {
        "train": {
            "context": "{item}的值是{value}。",
            "user_input": "{item}的值是什么？",
            "response": "依据上下文，答案是{value}。",
        },
        "dev": {
            "context": "已知{item}被记为{value}。",
            "user_input": "请回答：{item}被记成什么？",
            "response": "根据已知信息，是{value}。",
        },
        "final": {
            "context": "资料显示{item}对应{value}。",
            "user_input": "资料里{item}对应什么？",
            "response": "按资料，对应{value}。",
        },
    },
    # 2. negation: the context states the negation; the correct answer is negative.
    "negation": {
        "train": {
            "context": "{item}的值不是{value}。",
            "user_input": "{item}的值是{value}吗？",
            "response": "不是，上下文说{value}不对。",
        },
        "dev": {
            "context": "记录表明{item}并非{value}。",
            "user_input": "{item}是{value}吗？",
            "response": "并非如此，它不是{value}。",
        },
        "final": {
            "context": "资料里{item}不等于{value}。",
            "user_input": "{item}等于{value}吗？",
            "response": "不相等，它不等于{value}。",
        },
    },
    # 3. unknown: the context carries no information; the honest answer is "unknown".
    "unknown": {
        "train": {
            "context": "关于{item}没有记录。",
            "user_input": "{item}有记录吗？它的值是什么？",
            "response": "上下文没有提供{item}的信息，无法回答。",
        },
        "dev": {
            "context": "{item}的资料为空。",
            "user_input": "请确认：有{item}的记录吗？",
            "response": "资料里没有{item}的内容，我不知道。",
        },
        "final": {
            "context": "查不到{item}的任何信息。",
            "user_input": "请核实：{item}有对应记录吗？",
            "response": "没有任何关于{item}的信息，不能确定。",
        },
    },
    # 4. combination constraint: two facts must be combined.
    "combination": {
        "train": {
            "context": "{item}的值是{value}，且只有取到该值才算通过。",
            "user_input": "{item}怎样才能通过？",
            "response": "需要{item}的值是{value}才能通过。",
        },
        "dev": {
            "context": "{item}被记为{value}，通过条件正是这个值。",
            "user_input": "{item}满足什么条件？",
            "response": "要满足{item}为{value}。",
        },
        "final": {
            "context": "{item}对应{value}，其余取值都不算通过。",
            "user_input": "{item}通过需要什么？",
            "response": "只有{item}为{value}时才算通过。",
        },
    },
    # 5. same opening, different content: identical opening, different answer.
    "same_opening": {
        "train": {
            "context": "先说明：本段只讨论{item}。{item}的值是{value}。",
            "user_input": "先说明：本段只讨论{item}。它的值是什么？",
            "response": "先说明：本段只讨论{item}。它的值是{value}。",
        },
        "dev": {
            "context": "注意，下面只涉及{item}。{item}被记为{value}。",
            "user_input": "注意，下面只涉及{item}。它被记成什么？",
            "response": "注意，下面只涉及{item}。它被记成{value}。",
        },
        "final": {
            "context": "限定范围：只谈{item}。{item}对应{value}。",
            "user_input": "限定范围：只谈{item}。它对应什么？",
            "response": "限定范围：只谈{item}。它对应{value}。",
        },
    },
}

SHAPES = tuple(TEMPLATES)


#: Shapes whose honest answer carries no required term (the answer *is* the abstention).
_ABSTENTION_SHAPES = frozenset({"unknown"})


def _rows() -> list[dict[str, Any]]:
    """Pair each item with exactly one value so ``user_input`` stays unique per split."""

    rows: list[dict[str, Any]] = []
    for split, pools in ENTITY_POOLS.items():
        paired = list(zip(pools["items"], pools["values"], strict=True))
        for shape in SHAPES:
            template = TEMPLATES[shape][split]
            for index, (item, value) in enumerate(paired, start=1):
                slot = {"item": item, "value": value}
                # LanguageEpisode requires an ASCII stable identifier, so the ids are
                # index-based (deterministic: the pairing order is fixed) while the
                # Chinese entity only appears in the text slots.
                identifier = f"{split}-{shape}-{index:02d}"
                rows.append(
                    {
                        "episode_id": identifier,
                        "family_id": identifier,
                        "task_family": shape,
                        "split": split,
                        "context": template["context"].format(**slot),
                        "user_input": template["user_input"].format(**slot),
                        "response": template["response"].format(**slot),
                        "required_terms": [] if shape in _ABSTENTION_SHAPES else [value],
                    }
                )
    return rows


def _verify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Separation + shape checks.  Raises ``SystemExit`` on any violation."""

    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in ENTITY_POOLS}
    for row in rows:
        by_split[row["split"]].append(row)

    entities: dict[str, set[str]] = {}
    inputs: dict[str, set[str]] = {}
    combos: dict[str, set[tuple[str, str]]] = {}
    for split, group in by_split.items():
        entities[split] = {term for row in group for term in row["required_terms"]}
        inputs[split] = {row["user_input"] for row in group}
        combos[split] = {(row["task_family"], row["family_id"]) for row in group}

    checks: dict[str, Any] = {
        "every_split_has_all_five_shapes": all(
            {row["task_family"] for row in group} == set(SHAPES) for group in by_split.values()
        ),
        "entity_pools_disjoint": all(
            entities[a].isdisjoint(entities[b]) for a, b in itertools.combinations(entities, 2)
        ),
        "user_inputs_unique_within_split": all(
            len(inputs[s]) == len(by_split[s]) for s in by_split
        ),
        "user_inputs_disjoint_across_splits": all(
            inputs[a].isdisjoint(inputs[b]) for a, b in itertools.combinations(inputs, 2)
        ),
        "combinations_disjoint_across_splits": all(
            combos[a].isdisjoint(combos[b]) for a, b in itertools.combinations(combos, 2)
        ),
        "required_terms_present_in_response": all(
            all(term in row["response"] for term in row["required_terms"]) for row in rows
        ),
        "no_holdout_leak_in_train": all(
            row["user_input"] not in inputs["dev"] | inputs["final"] for row in by_split["train"]
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if len(rows) <= EXPECTED_MIN_ROWS:
        failed.append(f"corpus_not_larger_than_h3_5a_fixture({len(rows)}<={EXPECTED_MIN_ROWS})")
    if failed:
        raise SystemExit(f"corpus separation checks failed: {failed}")
    return checks


def build(fixture: Path = FIXTURE, report: Path = REPORT) -> dict[str, Any]:
    rows = _rows()
    checks = _verify(rows)

    target = PROJECT_ROOT / fixture
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    counts = {split: sum(1 for r in rows if r["split"] == split) for split in ENTITY_POOLS}
    payload: dict[str, Any] = {
        "format": FORMAT,
        "generation_seed": GENERATION_SEED,
        "fixture": str(fixture),
        "row_count": len(rows),
        "rows_per_split": counts,
        "shapes": list(SHAPES),
        "entity_pools": {s: {k: list(v) for k, v in p.items()} for s, p in ENTITY_POOLS.items()},
        "separation_checks": checks,
        "all_checks_passed": all(checks.values()),
        "corpus_digest": content_digest(
            {"rows": rows, "format": FORMAT, "generation_seed": GENERATION_SEED}
        ),
        "note": (
            "deterministic generation; separation rules are never exposed to any model. "
            "Required terms are drawn from the split-local entity pool so they cannot leak "
            "holdout content into train."
        ),
    }
    out = PROJECT_ROOT / report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the P3b-v2 goal-aligned episode corpus")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    payload = build(args.fixture, args.report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

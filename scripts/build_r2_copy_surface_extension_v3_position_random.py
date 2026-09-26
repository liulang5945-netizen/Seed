"""造 `v3`：把"答案告知排第几"这一维**随机化**的位置随机题集（SPEC-A-22 §18 的后继判据）。

为什么必须有它：v1 与 v2 的机检都规定"含答案词的告知必须是第一轮"（这样"选对事件"才有定义），
后果是一条纯位置规则——"永远挑最旧"——在两份题集上都满分。A2.5 的实测把这件事钉死了：
学习式选择头在 v1 上比"只锁定"多对 5 题（+5/+5 且两电路同向过线），
但在**逐题配对换位**的 `v2-permuted` 上是 0／0（`SPEC-A-22` §16/§18）。
⇒ 只要位置与答案相关，v1/v2 就量不出"提问条件化"。

本件按**逐题确定的种子**决定答案告知落在第 1 还是第 2 轮（对半），提问与判分词一字不动，
使"挑最旧／挑最新"两条纯位置规则的期望得分**都退回 chance**——于是答题率只能由
"提问与哪条告知内容相关"来支撑。机检（任一不过即拒绝出件）：

1. 题数与 id 集合与母本 v2 相同 ⇒ 与 v2 逐题配对；
2. 答案告知落在第 1 轮与第 2 轮**各占一半**，且四个 kind 内部也都两种都有
   （否则某个家族仍被位置单独决定）；
3. 第三轮（提问）与 `expected_contains` 逐题不变；
4. 每题恰好 3 轮，且"较早入库"与"答案告知"两件事在题集上不相关（报出相关系数）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v2.json"
DEFAULT_OUT = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
SEED = 20260926


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, out: Path, seed: int = SEED) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    items = payload["dimensions"]["X"]["items"]
    rng = random.Random(seed)
    rows = []
    placement: list[str] = []
    #: **均衡牌堆**而不是逐题独立抛硬币：实测 iid 抽样给出 54/50，"位置与答案无关"就退化成
    #: 一次抽样巧合。这里钉 52/52 严格对半，再由种子决定**哪** 52 题落前 —— 可复现、且构造保证不相关。
    if len(items) % 2:
        raise ValueError("position randomisation needs an even item count for a balanced deck")
    deck = ["first"] * (len(items) // 2) + ["second"] * (len(items) // 2)
    rng.shuffle(deck)
    for item, side in zip(items, deck):
        #: `side` 是字符串标签（"first"/"second"），非布尔——上一版把它直接当条件用，
        #: 于是 "second" 也为真、104 题全落第一轮，被"严格对半"那条机检拦下。
        first = side == "first"
        turns = list(item["turns"])
        expected = list(item["expected_contains"])
        if len(turns) != 3 or not any(token in turns[0] for token in expected):
            raise ValueError(f"{item['id']}: 母本不满足『恰好 3 轮且答案告知在第一轮』的前提")
        told = [turns[0], turns[1]] if first else [turns[1], turns[0]]
        placement.append("first" if first else "second")
        row = {**item, "turns": [*told, turns[2]], "answer_tell_position": 0 if first else 1}
        answer_index = 0 if first else 1
        if not any(token in row["turns"][answer_index] for token in expected):
            raise ValueError(f"{item['id']}: 落位标记与内容不一致")
        if any(token in row["turns"][1 - answer_index] for token in expected):
            raise ValueError(f"{item['id']}: 两条告知都含答案词，位置随机化无效")
        if row["turns"][2] != turns[2] or row["expected_contains"] != expected:
            raise ValueError(f"{item['id']}: 提问或判分词漂移")
        rows.append(row)

    if {row["id"] for row in rows} != {item["id"] for item in items}:
        raise ValueError("id 集合与母本不符")
    counts = Counter(placement)
    if counts["first"] != counts["second"]:
        raise ValueError(f"落位不是严格对半：{dict(counts)}")
    by_kind: dict[str, Counter] = {}
    for row, side in zip(rows, placement):
        by_kind.setdefault(str(row["kind"]), Counter())[side] += 1
    for kind, counter in by_kind.items():
        if not counter["first"] or not counter["second"]:
            raise ValueError(f"kind={kind} 只由位置决定（{dict(counter)}）——该家族仍可走位置捷径")
    report = {
        "format": "r2-copy-surface-extension-v3",
        "frozen_on": "2026-09-26",
        "frozen_before_any_candidate_score": True,
        "purpose": (
            "**位置随机化题集**：v1/v2 机检规定『含答案词的告知必为第一轮』，"
            "于是『挑最旧』这条纯位置规则两边满分（SPEC-A-22 §16 实测：配对换位后从 36 塌到 2）。"
            "本件把答案告知的落位逐题随机成对半，使两条位置规则期望都退回 chance，"
            "答题率只能由『提问与哪条告知内容相关』支撑 ⇒ 这才是提问条件化的判据。"
        ),
        "prereg": "plans/reference/SPEC-A-22_r2_a2_5_query_conditioned_selector_prereg_20260926.md §18",
        "generator": "scripts/build_r2_copy_surface_extension_v3_position_random.py",
        "derived_from": SOURCE.relative_to(PROJECT_ROOT).as_posix(),
        "derived_from_sha256": _sha256(source),
        "seed": seed,
        "count": len(rows),
        "placement_counts": dict(counts),
        "placement_by_kind": {kind: dict(counter) for kind, counter in sorted(by_kind.items())},
        "construction_checks": {
            "id_set_matches_parent": True,
            "exactly_three_turns": True,
            "answer_tell_position_randomised_50_50": True,
            "every_kind_carries_both_positions": True,
            "question_and_scoring_tokens_unchanged": True,
            "only_one_tell_carries_the_answer": True,
        },
        "reading_rule": (
            "与 v2 逐题配对着读：同一臂在 v2 与 v3 上的差就是『位置红利』被剥掉多少。"
            "只在 v2 高、v3 塌 ⇒ 位置先验；两集同高 ⇒ 提问条件化。"
            "per-kind 与 per-落位 分层必报（不合并总分）。"
        ),
        "disjointness_note": payload.get("disjointness_note"),
        "dimensions": {"X": {**payload["dimensions"]["X"], "count": len(rows), "items": rows}},
    }
    if out.exists():
        raise ValueError(f"refusing to overwrite a frozen manifest: {out}")
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        #: `--out` 可以指到仓库外（自检/临时件），别假设它一定在仓库根下面——
        #: 第一版直接 `relative_to` 把跑进 tmp 的自检打挂了。
        "out": (
            out.relative_to(PROJECT_ROOT).as_posix()
            if out.is_relative_to(PROJECT_ROOT)
            else str(out)
        ),
        "count": len(rows),
        "placement": dict(counts),
        "by_kind": {kind: dict(counter) for kind, counter in sorted(by_kind.items())},
        "sha256": _sha256(out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    source = Path(args.source) if args.source else SOURCE
    out = Path(args.out) if args.out else DEFAULT_OUT
    print(json.dumps(build(source, out), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

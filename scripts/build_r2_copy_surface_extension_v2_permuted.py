"""造 `v2-permuted` 题集（SPEC-A-22 §12 的 12 号探针）：把"答案告知排第一"这一维打掉。

为什么需要它：v1 与 v2 的机检都**规定**含答案词的告知必须是第一轮（这样"选对事件"才有定义），
后果是一条纯位置规则——"永远挑最旧那条"——在两个题集上都满分。于是"提问条件化"与
"位置先验"在这两份题集上**结构上不可分**（`SPEC-A-22` §10/§12）。

本件只做一件事：对每题交换前两轮（答案告知变成**较新**那条），提问与判分词一字不动，
并机检"交换确实发生了"。零训练、零新参数。

写出前机检（任一不过即拒绝出件）：
1. 每题恰好 3 轮，且 `expected_contains` 中至少一词逐字出现在**交换前**的第一轮；
2. 交换后答案告知**不再是**第一轮（探针有效性本身）；
3. 第三轮（提问）与 `expected_contains` 逐题不变；
4. 题数与 id 集合与母本相同 ⇒ 与 v2 逐题配对。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

#: 本件在 `scripts/` 下（不是 `scripts/training/`），所以 parents 只上跳一层——
#: 抄 `scripts/training/*` 那批的 `[2]` 会把仓库根算成盘符（实测造出 `E:\plans\...`）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v2.json"
DEFAULT_OUT = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v2_permuted.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, out: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    items = payload["dimensions"]["X"]["items"]
    permuted = []
    for item in items:
        turns = list(item["turns"])
        expected = list(item["expected_contains"])
        if len(turns) != 3:
            raise ValueError(f"{item['id']}: probe needs exactly 3 turns, got {len(turns)}")
        if not any(token in turns[0] for token in expected):
            raise ValueError(f"{item['id']}: answer tell is not turn 0 in the母本")
        row = {**item, "turns": [turns[1], turns[0], turns[2]]}
        if any(token in row["turns"][0] for token in expected):
            raise ValueError(f"{item['id']}: swap did not move the answer tell off turn 0")
        if row["turns"][2] != turns[2] or row["expected_contains"] != expected:
            raise ValueError(f"{item['id']}: question or scoring tokens drifted")
        permuted.append(row)

    if {row["id"] for row in permuted} != {row["id"] for row in items}:
        raise ValueError("item id set drifted from the母本")
    report = {
        "format": "r2-copy-surface-extension-v2-permuted",
        "frozen_on": "2026-09-26",
        "frozen_before_any_candidate_score": True,
        "purpose": (
            "**判别探针**（不是能力题集）：v1/v2 都按机检规定「含答案词的告知排第一」，"
            "于是「永远挑最旧」这条纯位置规则在两者上都满分 ⇒ 提问条件化与位置先验在那两份题集上"
            "结构上不可分。本件只把前两轮交换，提问与判分词一字不动。"
        ),
        "prereg": "plans/reference/SPEC-A-22_r2_a2_5_query_conditioned_selector_prereg_20260926.md §12",
        "generator": "scripts/build_r2_copy_surface_extension_v2_permuted.py",
        "derived_from": SOURCE.relative_to(PROJECT_ROOT).as_posix(),
        "derived_from_sha256": _sha256(source),
        "count": len(permuted),
        "construction_checks": {
            "all_three_turn": True,
            "answer_tell_moved_off_turn_zero": True,
            "question_and_tokens_unchanged": True,
            "id_set_matches_parent": True,
            "answer_tell_is_turn_two_by_construction": True,
        },
        "reading_rule": (
            "只用**学习式选择器**的训练臂（SPEC-A-22 丙）与**未训 cue／只锁定**臂（乙）在同一份题集上比："
            "若 丙 在这里塌回 乙/对照水平 ⇒ 它用的是位置先验，「提问条件化」不成立；"
            "若仍守住 ⇒ 才是「由提问决定」的第一份正面证据。"
        ),
        "disjointness_note": payload.get("disjointness_note"),
        "dimensions": {
            "X": {**payload["dimensions"]["X"], "count": len(permuted), "items": permuted}
        },
    }
    if out.exists():
        raise ValueError(f"refusing to overwrite a frozen manifest: {out}")
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "out": out.relative_to(PROJECT_ROOT).as_posix(),
        "count": len(permuted),
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

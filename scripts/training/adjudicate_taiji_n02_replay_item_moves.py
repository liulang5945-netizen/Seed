"""PLAN-N-02 保持侧代价的逐条判读器：把复述面"跌的那几条"点到题，并分清"答不出"还是"不成句"。

为什么要有这台（DEBT-G174 买来的就是这几列）：

* 配对件只出版聚合列（`replay_strict_hits:1.0` 6→5、`replay_well_formed:2.0` 25→23），
  而 cap0 那面已把"普遍变短/普遍答不好"两支否证（㊵-1125：cap0 净涨 1 项、翻转项字节逐位相同）。
  剩下的问题只能是**具体哪一题**、以及它是**没命中**还是**不成句**——这只有逐条列能答。
* 判读只从本函数出，词表封闭四值，且每个值都要有能为假的一支（合同测钉）。
* 先证聚合列与逐条列**同分母**：`sum(per_item.hit)` 必须等于该档 `strict_hits`、
  `sum(per_item.formed_texts)` 必须等于 `well_formed_texts`，两侧各自核。不对就 rc=2，
  因为一旦不同分母，"这题从命中变不命中"就可能是切片错位而不是行为变化。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

VERDICTS = ("no_moves", "hit_only", "formed_only", "hit_with_formed")
PENALTY_KEY = "repetition_penalty"


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不出件。"""


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise Refuse(f"artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _arms_by_penalty(report: dict, label: str) -> dict[float, dict]:
    arms = report.get("arms")
    if not isinstance(arms, list) or not arms:
        raise Refuse(f"{label} has no arms column")
    if any("per_item" not in arm for arm in arms):
        raise Refuse(
            f"{label} was produced without --emit-per-item; per_item rows are absent, "
            "so the cost cannot be named per item"
        )
    keyed: dict[float, dict] = {}
    for arm in arms:
        penalty = arm.get(PENALTY_KEY)
        if not isinstance(penalty, (int, float)) or isinstance(penalty, bool):
            raise Refuse(f"{label} has an arm without a numeric repetition_penalty")
        if float(penalty) in keyed:
            raise Refuse(f"{label} repeats penalty {penalty}")
        keyed[float(penalty)] = arm
    return keyed


def _rows_by_id(arm: dict, penalty: float, label: str) -> dict[str, dict]:
    rows = arm.get("per_item")
    if not isinstance(rows, list) or not rows:
        raise Refuse(f"{label} penalty {penalty} has empty per_item")
    by_id: dict[str, dict] = {}
    for row in rows:
        item_id = str(row.get("id") or "")
        if not item_id or item_id in by_id:
            raise Refuse(f"{label} penalty {penalty} has a missing or duplicate row id")
        by_id[item_id] = row
    return by_id


def _check_same_denominator(arm: dict, rows: dict[str, dict], penalty: float, label: str) -> None:
    """逐条列要和聚合列同分母，否则"哪一题掉了"这句话没有依据。"""

    hit_sum = sum(1 for row in rows.values() if row.get("hit"))
    formed_sum = sum(int(row.get("formed_texts") or 0) for row in rows.values())
    texts_sum = sum(int(row.get("texts") or 0) for row in rows.values())
    problems = []
    if hit_sum != arm.get("strict_hits"):
        problems.append(f"strict_hits {arm.get('strict_hits')} != per-item hits {hit_sum}")
    if formed_sum != arm.get("well_formed_texts"):
        problems.append(
            f"well_formed {arm.get('well_formed_texts')} != per-item formed {formed_sum}"
        )
    if texts_sum != arm.get("texts"):
        problems.append(f"texts {arm.get('texts')} != per-item texts {texts_sum}")
    if problems:
        raise Refuse(f"{label} penalty {penalty} is not same-denominator: " + "; ".join(problems))


def compare_penalty(penalty: float, before: dict, after: dict) -> dict[str, Any]:
    rows_before = _rows_by_id(before, penalty, "before")
    rows_after = _rows_by_id(after, penalty, "after")
    _check_same_denominator(before, rows_before, penalty, "before")
    _check_same_denominator(after, rows_after, penalty, "after")
    missing = sorted(set(rows_before) ^ set(rows_after))
    if missing:
        raise Refuse(f"penalty {penalty} item id sets differ: {missing}")

    hit_lost: list[dict[str, Any]] = []
    hit_gained: list[dict[str, Any]] = []
    formed_down: list[dict[str, Any]] = []
    formed_up: list[dict[str, Any]] = []
    for item_id in sorted(rows_before):
        left, right = rows_before[item_id], rows_after[item_id]
        moved = {
            "id": item_id,
            "hit": [bool(left.get("hit")), bool(right.get("hit"))],
            "formed_texts": [
                int(left.get("formed_texts") or 0),
                int(right.get("formed_texts") or 0),
            ],
            "texts": [int(left.get("texts") or 0), int(right.get("texts") or 0)],
            "cyclic_tail_texts": [
                int(left.get("cyclic_tail_texts") or 0),
                int(right.get("cyclic_tail_texts") or 0),
            ],
            "expected_contains": right.get("expected_contains") or left.get("expected_contains"),
        }
        if moved["hit"] == [True, False]:
            hit_lost.append(moved)
        elif moved["hit"] == [False, True]:
            hit_gained.append(moved)
        if moved["formed_texts"][1] < moved["formed_texts"][0]:
            formed_down.append(moved)
        elif moved["formed_texts"][1] > moved["formed_texts"][0]:
            formed_up.append(moved)
    return {
        "repetition_penalty": penalty,
        "items": len(rows_before),
        "hit_lost": hit_lost,
        "hit_gained": hit_gained,
        "formed_down": formed_down,
        "formed_up": formed_up,
        "aggregate": {
            "strict_hits": [before.get("strict_hits"), after.get("strict_hits")],
            "well_formed_texts": [before.get("well_formed_texts"), after.get("well_formed_texts")],
            "cyclic_texts": [before.get("cyclic_texts"), after.get("cyclic_texts")],
        },
    }


def adjudicate(per_penalty: list[dict[str, Any]]) -> str:
    """封闭四值，只看**变差**那一侧（这台仪器答的是"代价"），按合取写死、不自选。

    互斥且穷尽：无变差 ⇒ `no_moves`；有变差且有题既掉命中又掉成句 ⇒ `hit_with_formed`；
    否则有掉命中 ⇒ `hit_only`；否则只剩成句变差 ⇒ `formed_only`。
    只用聚合列的"净差"会漏掉这一层：一题掉一题涨在两列上都可能是 0。
    """

    hit_lost = [item for row in per_penalty for item in row["hit_lost"]]
    formed_down = [item for row in per_penalty for item in row["formed_down"]]
    if not hit_lost and not formed_down:
        return "no_moves"
    overlap = any(item["formed_texts"][1] < item["formed_texts"][0] for item in hit_lost)
    if hit_lost and overlap:
        return "hit_with_formed"
    if hit_lost:
        return "hit_only"
    return "formed_only"


def build_payload(before_path: Path, after_path: Path) -> dict:
    before, after = read_json(before_path), read_json(after_path)
    if before.get("format") != after.get("format"):
        raise Refuse("the two readouts have different formats")
    if before.get("items_sha256") != after.get("items_sha256"):
        raise Refuse("the two readouts are not on the same item subset (items_sha256 differs)")
    if before.get("chain") != after.get("chain"):
        raise Refuse("the two readouts used different chains")
    if before.get("checkpoint") == after.get("checkpoint"):
        raise Refuse("both readouts name the same checkpoint; that is not a before/after pair")
    for key in ("base_sha256_unchanged",):
        if before.get(key) is not True or after.get(key) is not True:
            raise Refuse(f"{key} is not true on both sides; the readouts would have moved weights")

    arms_before = _arms_by_penalty(before, "before")
    arms_after = _arms_by_penalty(after, "after")
    if set(arms_before) != set(arms_after):
        raise Refuse(f"penalty columns differ: {sorted(set(arms_before) ^ set(arms_after))}")
    per_penalty = [
        compare_penalty(penalty, arms_before[penalty], arms_after[penalty])
        for penalty in sorted(arms_before)
    ]
    verdict = adjudicate(per_penalty)
    return {
        "format": "taiji-n02-replay-item-moves-v1",
        "prereg": "plans/reference/PLAN-N-02_c4_net_gain_package_prereg_20261017.md",
        "debt": "DEBT-G174",
        "pair": [str(before_path), str(after_path)],
        "before_checkpoint": before.get("checkpoint"),
        "after_checkpoint": after.get("checkpoint"),
        "items_sha256": before.get("items_sha256"),
        "per_penalty": per_penalty,
        "verdict": verdict,
        "verdict_vocabulary": list(VERDICTS),
        "verdict_meanings": {
            "no_moves": "没有任何一题**变差**（变好仍单独出版在 hit_gained／formed_up 两列）⇒ 复述面的聚合跌不是题级损失，须回到配对件查切片",
            "hit_only": "有题从命中变不命中，但那些题的成句条数没动 ⇒ 代价是'答不出'那一半",
            "formed_only": "没有命中翻转，只有成句条数变化 ⇒ 代价是'不成句'那一半",
            "hit_with_formed": "同一题既掉命中又掉成句 ⇒ 两句要一起写，不许只引一个",
        },
        "reading_limit": (
            "本题面只覆盖 X 维那 24 题 × 每档罚分，且 `hit` 是按**最后一轮**回答判的（生成器原样）；"
            "它不判 cap0 面，也不判'剂量该不该减'——那句属 owner 裁的射程变更。"
        ),
    }


def refuse_overwrite(path: Path) -> None:
    if path.is_file():
        raise Refuse(f"refusing to overwrite a published artifact: {path}; use a new --out-report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        refuse_overwrite(arguments.out_report)
        payload = build_payload(arguments.before, arguments.after)
    except Refuse as exc:
        print(f"REFUSE: {exc}")
        return 2
    arguments.out_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.out_report.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"verdict={payload['verdict']} penalties={len(payload['per_penalty'])} "
        f"hit_lost={sum(len(r['hit_lost']) for r in payload['per_penalty'])} "
        f"formed_down={sum(len(r['formed_down']) for r in payload['per_penalty'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

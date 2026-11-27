"""PLAN-N-02 保持侧代价的只读形态普查：把 cap0 前后两张件的逐条输出按字节量一遍，回答"掉分是形态坏了还是没答"。

为什么要这台仪器（不是"再多看一眼"）：

* 保持侧的列跌在 `replay_strict_hits:1.0/2.0` 与 `replay_well_formed:2.0`，而 `measure_taiji_a30_repetition_penalty.py`
  的公开件在写盘时**剥掉了 `per_item`**（`arms` 只留聚合列）⇒ 现有件里没有可点名的那一题。
* cap0 前后两张件里有逐条 `verdict_text` ＋ `score`，所以先用**零算力**把这一面量干净：
  如果 cap0 侧形态与分数都不动，那"代价"就是**复述面特有**的，而不是全局输出退化——这把下一趟机时买的东西定下来了。
* 判读只从本函数出：`verdict` 用封闭词表，且每个取值都要有能为假的那一支（合同测钉住）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

CENSUS_FORMAT = "taiji-n02-retention-output-shape-v1"
DIMENSIONS = ("B", "C", "D", "E", "G")
REPLACEMENT_CHAR = "\ufffd"
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
#: 循环判据：任一 1–3 字单元连续重复 ≥8 次。与 a30 那台的 ngram 判据不同源，只作形态分档用。
CYCLIC = re.compile(r"(.{1,3})\1{7,}")

#: 封闭词表：每个取值都必须由下面的数算出来，不许手填。面不符走 rc=2 拒绝，不占判读值。
VERDICTS = ("identical_face", "shape_moved_scores_stable", "scores_moved")


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不出件。"""


def display_name(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise Refuse(f"artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_of(reading: dict, label: str) -> Path:
    """档身份走件自述的 `checkpoint`——`identity.checkpoint_sha256` 是**检查点**的摘要，不是读数件文件自己的摘要。"""

    raw = str(reading.get("checkpoint") or "")
    if not raw:
        raise Refuse(f"{label} readout does not self-report its checkpoint")
    return (ROOT / raw.replace("\\", "/")).resolve()


def admit_pair(before: dict, after: dict, before_path: Path, after_path: Path) -> dict:
    """同一张面才许对量：题集摘要、链、模式、母/候选档身份都要逐字核。"""

    problems: list[str] = []
    if before.get("format") != after.get("format"):
        problems.append("format")
    if before.get("identity", {}).get("eval_set_sha256") != after.get("identity", {}).get(
        "eval_set_sha256"
    ):
        problems.append("eval_set_sha256")
    if before.get("chain") != after.get("chain"):
        problems.append("chain")
    if before.get("declared_mode") != after.get("declared_mode"):
        problems.append("declared_mode")
    if (
        before.get("trained_during_eval") is not False
        or after.get("trained_during_eval") is not False
    ):
        problems.append("trained_during_eval")
    if before.get("checkpoint") == after.get("checkpoint"):
        problems.append("same_checkpoint_path")
    before_checkpoint = checkpoint_of(before, "before")
    after_checkpoint = checkpoint_of(after, "after")
    if not before_checkpoint.is_file() or not after_checkpoint.is_file():
        raise Refuse(
            "checkpoint named by the readouts is not on disk: "
            f"{before_checkpoint} / {after_checkpoint}"
        )
    before_file_sha, after_file_sha = file_sha(before_checkpoint), file_sha(after_checkpoint)
    lineage = {
        "before_checkpoint": display_name(before_checkpoint),
        "after_checkpoint": display_name(after_checkpoint),
        "before_checkpoint_sha256": before_file_sha,
        "after_checkpoint_sha256": after_file_sha,
        "before_sha_matches_self_report": before_file_sha
        == before.get("identity", {}).get("checkpoint_sha256"),
        "after_sha_matches_self_report": after_file_sha
        == after.get("identity", {}).get("checkpoint_sha256"),
    }
    if (
        not lineage["before_sha_matches_self_report"]
        or not lineage["after_sha_matches_self_report"]
    ):
        problems.append("checkpoint_sha_self_report")
    if problems:
        raise Refuse("the two readouts are not the same face: " + ", ".join(problems))
    return lineage


def shape_of(text: str) -> dict[str, Any]:
    encoded = (text or "").encode("utf-8")
    runs = [len(m.group(0)) for m in CJK.finditer(text or "")]
    return {
        "chars": len(text or ""),
        "bytes": len(encoded),
        "replacement_chars": (text or "").count(REPLACEMENT_CHAR),
        "cjk_runs": len(runs),
        "cjk_longest_run": max(runs) if runs else 0,
        "cyclic_like": bool(CYCLIC.search(text or "")),
    }


def items_by_id(dimension_block: dict) -> dict[str, dict]:
    rows = dimension_block.get("items") or []
    return {str(row.get("id")): row for row in rows if isinstance(row, dict)}


def census_dimension(before_block: dict, after_block: dict, name: str) -> dict:
    before_items, after_items = items_by_id(before_block), items_by_id(after_block)
    shared = sorted(set(before_items) & set(after_items))
    gained: list[str] = []
    lost: list[str] = []
    text_changed: list[str] = []
    aggregates: dict[str, dict[str, int]] = {}
    for side, lookup in (("before", before_items), ("after", after_items)):
        totals = {
            "replacement_chars": 0,
            "cjk_longest_run_sum": 0,
            "cyclic_like": 0,
            "bytes_sum": 0,
        }
        for item_id in shared:
            shape = shape_of(lookup[item_id].get("verdict_text") or "")
            totals["replacement_chars"] += shape["replacement_chars"]
            totals["cjk_longest_run_sum"] += shape["cjk_longest_run"]
            totals["cyclic_like"] += 1 if shape["cyclic_like"] else 0
            totals["bytes_sum"] += shape["bytes"]
        aggregates[side] = totals
    for item_id in shared:
        score_before = before_items[item_id].get("score")
        score_after = after_items[item_id].get("score")
        if isinstance(score_before, int) and isinstance(score_after, int):
            if score_after > score_before:
                gained.append(item_id)
            elif score_after < score_before:
                lost.append(item_id)
        if (before_items[item_id].get("verdict_text") or "") != (
            after_items[item_id].get("verdict_text") or ""
        ):
            text_changed.append(item_id)
    #: 翻转项连字节数一起出——"掉分是因为变短"这一句只能用这一列判，不能靠聚合列推。
    flip_detail = []
    for item_id in sorted(gained + lost):
        before_text = before_items[item_id].get("verdict_text") or ""
        after_text = after_items[item_id].get("verdict_text") or ""
        flip_detail.append(
            {
                "id": item_id,
                "score": [before_items[item_id].get("score"), after_items[item_id].get("score")],
                "bytes": [len(before_text.encode("utf-8")), len(after_text.encode("utf-8"))],
                "expected_contains": before_items[item_id].get("expected_contains"),
                "matched_before": before_items[item_id].get("matched"),
                "matched_after": after_items[item_id].get("matched"),
            }
        )
    moved = any(
        aggregates["before"][key] != aggregates["after"][key]
        for key in ("replacement_chars", "cyclic_like", "cjk_longest_run_sum", "bytes_sum")
    )
    return {
        "dimension": name,
        "items_compared": len(shared),
        "items_only_before": sorted(set(before_items) - set(after_items)),
        "items_only_after": sorted(set(after_items) - set(before_items)),
        "score_gained": gained,
        "score_lost": lost,
        "flip_detail": flip_detail,
        "texts_changed": len(text_changed),
        "machine_scored": {
            "before": (before_block.get("tally") or {}).get("machine_scored_correct"),
            "after": (after_block.get("tally") or {}).get("machine_scored_correct"),
        },
        "shape_columns_moved": bool(moved),
        "aggregate_before": aggregates["before"],
        "aggregate_after": aggregates["after"],
    }


def build_payload(before_path: Path, after_path: Path) -> dict:
    before, after = read_json(before_path), read_json(after_path)
    lineage = admit_pair(before, after, before_path, after_path)
    per_dimension = [
        census_dimension(before["dimensions"][name], after["dimensions"][name], name)
        for name in DIMENSIONS
        if name in before.get("dimensions", {}) and name in after.get("dimensions", {})
    ]
    if len(per_dimension) != len(DIMENSIONS):
        present = {row["dimension"] for row in per_dimension}
        missing = [name for name in DIMENSIONS if name not in present]
        raise Refuse(f"expected {len(DIMENSIONS)} scored dimensions, missing: {', '.join(missing)}")
    scores_lost = sum(len(row["score_lost"]) for row in per_dimension)
    scores_gained = sum(len(row["score_gained"]) for row in per_dimension)
    shape_moved = any(row["shape_columns_moved"] for row in per_dimension)
    texts_changed = sum(row["texts_changed"] for row in per_dimension)
    items_compared = sum(row["items_compared"] for row in per_dimension)
    if items_compared == 0:
        raise Refuse("no item id matched between the two readouts; nothing was compared")

    if scores_lost or scores_gained:
        verdict = "scores_moved"
    elif shape_moved or texts_changed:
        verdict = "shape_moved_scores_stable"
    else:
        verdict = "identical_face"

    return {
        "format": CENSUS_FORMAT,
        "prereg": "plans/reference/PLAN-N-02_c4_net_gain_package_prereg_20261017.md",
        "pair": [display_name(before_path), display_name(after_path)],
        "lineage": lineage,
        "items_compared": items_compared,
        "per_dimension": per_dimension,
        "totals": {
            "score_lost": scores_lost,
            "score_gained": scores_gained,
            "texts_changed": texts_changed,
            "replacement_chars_before": sum(
                r["aggregate_before"]["replacement_chars"] for r in per_dimension
            ),
            "replacement_chars_after": sum(
                r["aggregate_after"]["replacement_chars"] for r in per_dimension
            ),
            "cyclic_like_before": sum(r["aggregate_before"]["cyclic_like"] for r in per_dimension),
            "cyclic_like_after": sum(r["aggregate_after"]["cyclic_like"] for r in per_dimension),
        },
        "verdict": verdict,
        "verdict_vocabulary": list(VERDICTS),
        "verdict_meanings": {
            "identical_face": "两侧逐条输出与分数都相同 ⇒ cap0 这一面测不到代价，代价是复述面特有",
            "shape_moved_scores_stable": "输出内容或形态列动了、机器分没动 ⇒ 形态差异没转化成 cap0 侧掉分",
            "scores_moved": "有项分数翻转 ⇒ 代价落在这一面，点名到 item id",
        },
        "reading_limit": (
            "本件只回答'cap0 这一面上分数与形态各自动没动'。保持侧那三列跌住在复述面（a30 那台的件写盘时剥掉 "
            "per_item），所以本件**不能**指认那几题是哪一题；那要先给复述仪补一档逐条出版再跑。"
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
    totals = payload["totals"]
    print(
        f"verdict={payload['verdict']} items={payload['items_compared']} "
        f"score_lost={totals['score_lost']} score_gained={totals['score_gained']} "
        f"texts_changed={totals['texts_changed']} cyclic={totals['cyclic_like_before']}->{totals['cyclic_like_after']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

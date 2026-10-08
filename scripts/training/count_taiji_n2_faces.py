"""N2 通电面判读器（PLAN-N2-01 §4ter 的冻结命令产出的那几件）。

三条硬规矩写死在这里：
* **缺件即响亮失败**（rc=2，并点名缺哪一枚），不静默把"没取到"判成"没跌破"；
* 逐位同只比**读数内容**——`seconds`/时间戳一类易变字段先剥掉再比，剥了哪些必须自述；
* 每条判据都要能红：CAP-0 的"前后是否全同"同时当**动态范围**检验（全同＝这把尺量不动，
  判 `ruler_usable=false`，不许拿它当"保持住了"的证据）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 易变字段：比对逐位同之前一律剥掉，件里自述剥了哪些键名。
VOLATILE_KEYS = frozenset(
    {"seconds", "elapsed", "duration_ms", "started_utc", "started_at", "finished_at"}
)

DRIVEN_DIMENSIONS = ("B", "C", "D", "E", "G")


def _load(path: Path, missing: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        missing.append(str(path.relative_to(PROJECT_ROOT)))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_volatile(node: Any) -> Any:
    """递归剥掉易变字段；返回的是可比对的那半。"""

    if isinstance(node, dict):
        return {
            key: _strip_volatile(value) for key, value in node.items() if key not in VOLATILE_KEYS
        }
    if isinstance(node, list):
        return [_strip_volatile(item) for item in node]
    return node


def _cap0_tally(report: dict[str, Any]) -> dict[str, Any]:
    """CAP-0 的严格命中分子＝件内自带的 `machine_scored_correct`，逐维分账。"""

    dimensions = report.get("dimensions") or {}
    per_dimension: dict[str, dict[str, int]] = {}
    total_correct = 0
    total_scored = 0
    total_pending = 0
    for key in DRIVEN_DIMENSIONS:
        block = dimensions.get(key) or {}
        tally = block.get("tally") or {}
        correct = int(tally.get("machine_scored_correct") or 0)
        scored = int(tally.get("machine_scored_items") or 0)
        pending = int(tally.get("pending_human_review_items") or 0)
        per_dimension[key] = {
            "correct": correct,
            "scored": scored,
            "pending_human_review": pending,
            "item_count": int(block.get("item_count") or 0),
        }
        total_correct += correct
        total_scored += scored
        total_pending += pending
    return {
        "checkpoint": str(report.get("checkpoint") or ""),
        "eval_set": str(report.get("eval_set") or ""),
        "per_dimension": per_dimension,
        "total_correct": total_correct,
        "total_scored": total_scored,
        "pending_human_review": total_pending,
    }


def _replay_arms(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """复述面按惩罚档分列的绝对计数（`verdict` 是档内对基线的差，不能当前后对照的分子）。"""

    arms: dict[str, dict[str, Any]] = {}
    for arm in report.get("arms") or []:
        key = str(arm.get("repetition_penalty"))
        arms[key] = {
            "items": arm.get("items"),
            "texts": arm.get("texts"),
            "strict_hits": arm.get("strict_hits"),
            "well_formed_texts": arm.get("well_formed_texts"),
            "cyclic_rate": arm.get("cyclic_rate"),
        }
    return arms


def judge(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    missing: list[str] = []
    repo = PROJECT_ROOT
    cap0_before = _load(repo / args.cap0_before, missing)
    cap0_after = _load(repo / args.cap0_after, missing)
    cap0_rollback = _load(repo / args.cap0_rollback, missing)
    replay_before = _load(repo / args.replay_before, missing)
    replay_after = _load(repo / args.replay_after, missing)
    powerup = _load(repo / args.powerup, missing)

    if missing:
        return (
            {
                "verdict": "not_judged_missing_faces",
                "missing": missing,
                "note": "缺件＝不判；不许把没取到的面读成没跌破（§4ter 分档结项口径）",
            },
            2,
        )

    assert cap0_before is not None and cap0_after is not None
    assert cap0_rollback is not None and powerup is not None
    assert replay_before is not None and replay_after is not None

    before_tally = _cap0_tally(cap0_before)
    after_tally = _cap0_tally(cap0_after)
    rollback_tally = _cap0_tally(cap0_rollback)

    #: G-N2-1 的实证：回退面与巩固前逐位同（剥掉易变字段之后）。
    comparable_before = json.dumps(_strip_volatile(cap0_before), sort_keys=True)
    comparable_rollback = json.dumps(_strip_volatile(cap0_rollback), sort_keys=True)
    g_n2_1_identical = comparable_before == comparable_rollback

    #: 动态范围：巩固前后如果整张面逐位同，说明这张量不动权重变化——不是"保持住了"。
    comparable_after = json.dumps(_strip_volatile(cap0_after), sort_keys=True)
    ruler_usable = comparable_before != comparable_after

    j_n2a_cap0 = after_tally["total_correct"] >= before_tally["total_correct"] - 1

    arms_before = _replay_arms(replay_before)
    arms_after = _replay_arms(replay_after)
    shared = sorted(set(arms_before) & set(arms_after))
    replay_rows: dict[str, Any] = {}
    hits_dropped = 0
    formed_dropped = 0
    for key in shared:
        b, a = arms_before[key], arms_after[key]
        hits_delta = int(a["strict_hits"] or 0) - int(b["strict_hits"] or 0)
        formed_delta = int(a["well_formed_texts"] or 0) - int(b["well_formed_texts"] or 0)
        hits_dropped += int(hits_delta < 0)
        formed_dropped += int(formed_delta < 0)
        replay_rows[key] = {
            "hits_before": b["strict_hits"],
            "hits_after": a["strict_hits"],
            "hits_delta": hits_delta,
            "formed_before": b["well_formed_texts"],
            "formed_after": a["well_formed_texts"],
            "formed_delta": formed_delta,
        }
    unpaired = sorted(set(arms_before) ^ set(arms_after))
    j_n2a_replay = bool(shared) and hits_dropped == 0 and formed_dropped == 0

    night = powerup.get("night_selection") or {}
    payload = {
        "format": "taiji-n2-face-verdict-v1",
        "prereg": "plans/reference/PLAN-N2-01_consolidation_powerup_prereg_20261007.md#4ter",
        "cap0_before": before_tally,
        "cap0_after": after_tally,
        "cap0_rollback": rollback_tally,
        "g_n2_1_rollback_identical": g_n2_1_identical,
        "volatile_keys_stripped": sorted(VOLATILE_KEYS),
        "cap0_ruler_usable": ruler_usable,
        "j_n2a_cap0": j_n2a_cap0,
        "replay_arms_compared": replay_rows,
        "replay_arms_unpaired": unpaired,
        "replay_arms_dropped_hits": hits_dropped,
        "replay_arms_dropped_formed": formed_dropped,
        "j_n2a_replay": j_n2a_replay,
        "stop_face": {
            "status": "unverified",
            "reason": "×24 停摆面的主列由配套计数仪产出，本判读器未把它接进来；"
            "未接入的面不得代答成没跌破（§4ter 更正三第②条）",
        },
        "powerup": {
            "weight_changed": powerup.get("weight_changed_by_powerup"),
            "restore_matches_mother": powerup.get("restore_matches_mother"),
            "rollback_disk_matches_mother": powerup.get("rollback_disk_matches_mother"),
            "fail_closed_pass": powerup.get("fail_closed_pass"),
            "night_pool_size": night.get("pool_size"),
            "night_k": night.get("k"),
            "scheduler_agrees_with_weighted_sort": night.get("scheduler_agrees_with_weighted_sort"),
            "material_treated": powerup.get("material_treated"),
            "material_control": powerup.get("material_control"),
            "j_n2b": powerup.get("j_n2b"),
        },
    }

    closed = bool(
        payload["powerup"]["fail_closed_pass"]
        and g_n2_1_identical
        and ruler_usable
        and j_n2a_cap0
        and j_n2a_replay
        and payload["powerup"]["j_n2b"]
    )
    payload["j_n2a"] = bool(j_n2a_cap0 and j_n2a_replay)
    payload["unit_closed"] = closed
    payload["blocking_terms"] = [
        name
        for name, value in (
            ("g_n2_1_rollback_identical", g_n2_1_identical),
            ("cap0_ruler_usable", ruler_usable),
            ("j_n2a_cap0", j_n2a_cap0),
            ("j_n2a_replay", j_n2a_replay),
            ("j_n2b", bool(payload["powerup"]["j_n2b"])),
            ("stop_face_unverified", True),
        )
        if not value
    ]
    payload["verdict"] = "acceleration_point_1_holds" if closed else "not_closed"
    return payload, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N2 通电面判读器")
    parser.add_argument("--cap0-before", default="reports/taiji_n2_cap0_before_20261008.json")
    parser.add_argument("--cap0-after", default="reports/taiji_n2_cap0_after_20261008.json")
    parser.add_argument("--cap0-rollback", default="reports/taiji_n2_cap0_rollback_20261008.json")
    parser.add_argument("--replay-before", default="reports/taiji_n2_replay24_before_20261008.json")
    parser.add_argument("--replay-after", default="reports/taiji_n2_replay24_after_20261008.json")
    parser.add_argument("--powerup", default="reports/taiji_n2_powerup_20261008.json")
    parser.add_argument("--out", default="reports/taiji_n2_face_verdict_20261008.json")
    args = parser.parse_args(argv)

    payload, rc = judge(args)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    #: Windows 控制台是 GBK，判读正文含中文 ⇒ 只印 ASCII 行，中文留给件里。
    print(
        json.dumps({k: payload[k] for k in ("verdict", "j_n2a") if k in payload}, ensure_ascii=True)
    )
    print("blocking=" + json.dumps(payload.get("blocking_terms", []), ensure_ascii=True))
    print(f"out -> {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

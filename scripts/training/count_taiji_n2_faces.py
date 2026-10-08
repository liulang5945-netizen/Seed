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

#: 出处字段记的是"这张面由哪条命令在哪个提交上取哪一枚档"，不是读数。
#: ㊵-487 实测：回退面与巩固前**只剩这三条**不同（档路径、档 sha256、git_head——两趟之间
#: 提交了一次 530b9a37），100 项回答逐位同。⇒ 逐位同比读数，出处差**单独列出来**给人核对，
#: 既不让它冒充"能力变了"，也不许当作不存在。
PROVENANCE_KEYS = frozenset({"checkpoint", "identity"})

DRIVEN_DIMENSIONS = ("B", "C", "D", "E", "G")


def _rel(path: Path) -> str:
    """缺件点名用；仓外绝对路径调 `relative_to` 会抛 ValueError，那会把响亮拒绝伪装成崩溃
    （㊵-484⑤ 同一类缺陷：拒绝路径本身不许再炸）。"""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load(path: Path, missing: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        missing.append(_rel(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _strip_volatile(node: Any, top: bool = False) -> Any:
    """递归剥掉易变字段（顶层再剥出处字段）；返回的是可比对的那半。"""

    if isinstance(node, dict):
        drop = VOLATILE_KEYS | PROVENANCE_KEYS if top else VOLATILE_KEYS
        return {key: _strip_volatile(value) for key, value in node.items() if key not in drop}
    if isinstance(node, list):
        return [_strip_volatile(item) for item in node]
    return node


def _provenance(report: dict[str, Any]) -> dict[str, Any]:
    """把被排除的出处字段原样列出，供件里核对"这两张面读的是同一枚模型"。"""

    return {key: report.get(key) for key in sorted(PROVENANCE_KEYS) if key in report}


def _cap0_tally(report: dict[str, Any]) -> dict[str, Any]:
    """CAP-0 的严格命中分子＝件内自带的 `machine_scored_correct`，逐维分账。

    **必须先数装载失败**：一项 `load_ok` 不为真的项根本没被模型答过，它的 correct 恒 0，
    拿这种件读成"能力＝0"就是将仪器缺陷报成结论（㊵-487 候选档不可 load 那次差点踩中）。
    """

    dimensions = report.get("dimensions") or {}
    per_dimension: dict[str, dict[str, int]] = {}
    total_correct = 0
    total_scored = 0
    total_pending = 0
    load_not_ok = 0
    items_seen = 0
    for key in DRIVEN_DIMENSIONS:
        block = dimensions.get(key) or {}
        tally = block.get("tally") or {}
        correct = int(tally.get("machine_scored_correct") or 0)
        scored = int(tally.get("machine_scored_items") or 0)
        pending = int(tally.get("pending_human_review_items") or 0)
        for item in block.get("items") or []:
            items_seen += 1
            if item.get("load_ok") is not True:
                load_not_ok += 1
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
        "items_seen": items_seen,
        "load_not_ok": load_not_ok,
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


def _main_column_entries(payload: Any) -> dict[str, dict[str, Any]]:
    """把计数仪读数里的每枚面摊平成 `{源路径: 主列读数}`。

    只认带 `never_lf_eaters_counted` 的条目——主列的判定权属于这台仪器，不属于本判读器；
    取不到就返回空表，由调用方记 `unverified`，**不许**从停止面件里自己数一遍（那是重抄生成链）。
    """

    found: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "never_lf_eaters_counted" in node:
                source = str(node.get("report") or node.get("path") or node.get("name") or "")
                found[source] = {
                    "never_lf_eaters_counted": node.get("never_lf_eaters_counted"),
                    "generation_rows_seen": node.get("generation_rows_seen"),
                    "items_sha256": node.get("items_sha256"),
                    "checkpoint_sha256": node.get("checkpoint_sha256"),
                    "status": node.get("status"),
                    "meets_frozen_line": node.get("meets_frozen_line"),
                }
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def judge(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    missing: list[str] = []
    repo = PROJECT_ROOT
    cap0_before = _load(repo / args.cap0_before, missing)
    cap0_after = _load(repo / args.cap0_after, missing)
    cap0_rollback = _load(repo / args.cap0_rollback, missing)
    replay_before = _load(repo / args.replay_before, missing)
    replay_after = _load(repo / args.replay_after, missing)
    postcheck = _load(repo / args.postcheck, missing)
    align = _load(repo / args.align, missing)
    material = _load(repo / args.material, missing)
    #: ×96 主列附件是**可选**输入：没跑完时判 `unverified`，跑完了就接进来——但它不参与
    #: "缺件即不判"那支，因为 §4ter 的分档结项口径本来就允许先交已取到的档。
    stop_main = _load(repo / args.stop_main, []) if args.stop_main else None

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
    assert cap0_rollback is not None
    assert replay_before is not None and replay_after is not None
    assert postcheck is not None and align is not None and material is not None

    before_tally = _cap0_tally(cap0_before)
    after_tally = _cap0_tally(cap0_after)
    rollback_tally = _cap0_tally(cap0_rollback)

    #: 面有效性优先于判据：任何一张 CAP-0 面有项没被模型答过（load 失败），这张面就是废的，
    #: 拿它的 0 去判"能力保持/归零"都是把仪器缺陷报成结论。
    invalid = {
        label: tally["load_not_ok"]
        for label, tally in (
            ("cap0_before", before_tally),
            ("cap0_after", after_tally),
            ("cap0_rollback", rollback_tally),
        )
        if tally["load_not_ok"]
    }
    if invalid:
        return (
            {
                "verdict": "not_judged_face_invalid",
                "cap0_load_failures_by_face": invalid,
                "note": "有项未被模型作答⇒该面不可用；先修装载，再谈判据",
                "cap0_before": before_tally,
                "cap0_after": after_tally,
                "cap0_rollback": rollback_tally,
            },
            2,
        )

    #: G-N2-1 的实证：回退面与巩固前**读数**逐位同（剥掉易变字段与出处字段之后）。
    comparable_before = json.dumps(_strip_volatile(cap0_before, top=True), sort_keys=True)
    comparable_rollback = json.dumps(_strip_volatile(cap0_rollback, top=True), sort_keys=True)
    g_n2_1_identical = comparable_before == comparable_rollback

    #: 动态范围：巩固前后如果整张面读数逐位同，说明这把尺量不动权重变化——不是"保持住了"。
    comparable_after = json.dumps(_strip_volatile(cap0_after, top=True), sort_keys=True)
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

    files = postcheck.get("files") or {}
    tensor = postcheck.get("tensor_delta") or {}
    mother = files.get("mother") or {}
    candidate = files.get("candidate") or {}
    rollback = files.get("rollback") or {}
    #: 通电的"改了权重"与"能回退"都改由磁盘三档实测（v1 的内存摘要位随那次崩溃一起丢了）。
    persistence = {
        "mother_disk_load_ok": mother.get("disk_load_ok"),
        "candidate_disk_load_ok": candidate.get("disk_load_ok"),
        "rollback_disk_load_ok": rollback.get("disk_load_ok"),
        "candidate_refusal": candidate.get("disk_load_error"),
        "rollback_digest_equals_mother": rollback.get("digest") == mother.get("digest"),
        "rollback_tensors_changed": tensor.get("rollback", {}).get("tensors_changed"),
        "candidate_tensors_changed": tensor.get("candidate", {}).get("tensors_changed"),
        "align_pass": align.get("align_pass"),
        "aligned_tensors_differing": (align.get("tensor_check") or {}).get("tensors_differing"),
        "treated_faces_read_on": "seed_n2_candidate_aligned_20261008.pt",
    }
    stop_entries = _main_column_entries(stop_main) if stop_main else {}
    before_mc = next((v for k, v in stop_entries.items() if "before" in k), None)
    after_mc = next((v for k, v in stop_entries.items() if "after" in k), None)
    stop_measured = bool(before_mc and after_mc)
    stop_face = {
        "status": "measured_criterion_silent" if stop_measured else "unverified",
        "before": before_mc,
        "after": after_mc,
        "delta": (
            int(after_mc["never_lf_eaters_counted"]) - int(before_mc["never_lf_eaters_counted"])
            if stop_measured
            else None
        ),
        "denominator": before_mc.get("generation_rows_seen") if stop_measured else None,
        "items_sha_match": bool(
            stop_measured and before_mc.get("items_sha256") == after_mc.get("items_sha256")
        ),
        "direction_defined": False,
        "reason": (
            "主列数已由现成计数仪取到，但 §2 那句\u201c无一项跌破巩固前同面读数\u201d**没有为\u201c拖写行数\u201d"
            "这一列指定哪个方向算跌破**（越少越好是历史解释，不是冻结判据）⇒ 只报数与差，"
            "不冒充判据判定（这是 DEBT-G46 的第三类实例）"
            if stop_measured
            else "×96 主列附件件不在 ⇒ 记 unverified；未取到的面不得代答成没跌破（§4ter 更正三第②条）"
        ),
    }
    payload = {
        "format": "taiji-n2-face-verdict-v2",
        "prereg": "plans/reference/PLAN-N2-01_consolidation_powerup_prereg_20261007.md#4ter",
        "cap0_before": before_tally,
        "cap0_after": after_tally,
        "cap0_rollback": rollback_tally,
        "g_n2_1_rollback_identical": g_n2_1_identical,
        "volatile_keys_stripped": sorted(VOLATILE_KEYS),
        "provenance_keys_excluded": sorted(PROVENANCE_KEYS),
        "provenance_before": _provenance(cap0_before),
        "provenance_rollback": _provenance(cap0_rollback),
        "provenance_after": _provenance(cap0_after),
        "cap0_ruler_usable": ruler_usable,
        "j_n2a_cap0": j_n2a_cap0,
        "replay_arms_compared": replay_rows,
        "replay_arms_unpaired": unpaired,
        "replay_arms_dropped_hits": hits_dropped,
        "replay_arms_dropped_formed": formed_dropped,
        "j_n2a_replay": j_n2a_replay,
        "persistence": persistence,
        "stop_face": stop_face,
        "material": {
            "night_pool_size": material.get("night_pool_size"),
            "night_k": material.get("night_k"),
            "scheduler_agrees_with_weighted_sort": material.get(
                "scheduler_agrees_with_weighted_sort"
            ),
            "material_treated": material.get("material_treated"),
            "material_control": material.get("material_control"),
            "j_n2b": material.get("j_n2b"),
            "treated_digest": material.get("treated_digest"),
            "control_digest": material.get("control_digest"),
        },
    }

    g_n2_1_disk = bool(
        persistence["rollback_disk_load_ok"]
        and persistence["rollback_digest_equals_mother"]
        and persistence["rollback_tensors_changed"] == 0
    )
    weight_moved = bool(
        (persistence["candidate_tensors_changed"] or 0) > 0
        and persistence["align_pass"] is True
        and persistence["aligned_tensors_differing"] == 0
    )
    j_n2b = bool(material.get("j_n2b") is True)
    closed = bool(
        g_n2_1_disk
        and weight_moved
        and g_n2_1_identical
        and ruler_usable
        and j_n2a_cap0
        and j_n2a_replay
        and j_n2b
    )
    payload["g_n2_1_disk_identity"] = g_n2_1_disk
    payload["weight_moved_by_powerup"] = weight_moved
    payload["j_n2a"] = bool(j_n2a_cap0 and j_n2a_replay)
    payload["j_n2b"] = j_n2b
    payload["unit_closed"] = closed
    payload["blocking_terms"] = [
        name
        for name, value in (
            ("g_n2_1_disk_identity", g_n2_1_disk),
            ("weight_moved_by_powerup", weight_moved),
            ("g_n2_1_rollback_face_identical", g_n2_1_identical),
            ("cap0_ruler_usable", ruler_usable),
            ("j_n2a_cap0", j_n2a_cap0),
            ("j_n2a_replay", j_n2a_replay),
            ("j_n2b", j_n2b),
            ("stop_face_direction_defined", bool(stop_face["direction_defined"])),
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
    parser.add_argument("--postcheck", default="reports/taiji_n2_postcheck_20261008.json")
    parser.add_argument("--align", default="reports/taiji_n2_align_20261008.json")
    parser.add_argument("--material", default="reports/taiji_n2_material_20261008.json")
    parser.add_argument(
        "--stop-main",
        default="reports/taiji_n2_stop96_main_20261008.json",
        help="×96 停摆面主列读数（`count_taiji_a30_eater_p_boundary_floor.py` 产出；缺件记 unverified）",
    )
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

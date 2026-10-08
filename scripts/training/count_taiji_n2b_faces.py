"""N2 第二次通电的面判读器（PLAN-N2-02 §2 的三条合取＋§3 的五条守卫）。

与第一次的判读器（`count_taiji_n2_faces.py`）的**分工**：那台钉 PLAN-N2-01 的旧判据文本，
本件判据形状不同（§0 的两处只有改形状才能修的地方），所以**另立一台**、只**复用**它的取数
函数（`_cap0_tally`/`_replay_arms`/`_main_column_entries`/`_strip_volatile`/`_provenance`）——
取法必须同源，判级各自留在自己的件里（一档只住一处）。

三条硬规矩：
* 缺件／面无效／两档不可比 ⇒ rc=2 响亮拒绝，不静默把"没取到"读成"没跌破"；
* J-N2a' 是**逐列**单值条件（§2 写死了，不再有"恰好跌破 1 项"那种中间格），主列方向预先指定；
* 每个布尔都要能为 false，且 J-N2b' 的结论**取自通电件**，本件只核符号与结论是否互相矛盾。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: 取数函数与比对口径**必须与第一次同源**——同一把尺的两档才可比。
from scripts.training.count_taiji_n2_faces import (  # noqa: E402  # noqa: E402
    _cap0_tally,
    _main_column_entries,
    _provenance,
    _replay_arms,
    _strip_volatile,
)

SHA64 = re.compile(r"^[0-9a-f]{64}$")


def _rel(path: Path) -> str:
    """缺件点名用；仓外绝对路径调 `relative_to` 会抛 ValueError，那会把响亮拒绝伪装成崩溃。"""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load(path: Path, missing: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        missing.append(_rel(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_columns(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    direction: str,
    label: str,
) -> dict[str, Any]:
    """§2 J-N2a' 的单值式：`direction="bigger"` ⇒ 后−前 ≥ 0；`"smaller"` ⇒ 后−前 ≤ 0。

    任一档缺键 ⇒ 该列记 `unverified` 而不是 0——把"没这一列"读成"跌到 0"是假结论。
    """

    rows: dict[str, dict[str, Any]] = {}
    fails: list[str] = []
    unverified: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key not in before or key not in after:
            unverified.append(key)
            continue
        b, a = before[key], after[key]
        if not isinstance(b, int) or not isinstance(a, int):
            unverified.append(key)
            continue
        delta = a - b
        held = delta <= 0 if direction == "smaller" else delta >= 0
        rows[key] = {"before": b, "after": a, "delta": delta, "held": held}
        if not held:
            fails.append(key)
    return {
        "column_group": label,
        "direction": direction,
        "columns": rows,
        "dropped": fails,
        "unverified": unverified,
        "all_held": bool(rows) and not fails and not unverified,
    }


def judge(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    missing: list[str] = []
    repo = PROJECT_ROOT
    powerup = _load(repo / args.powerup, missing)
    guard = _load(repo / args.guard_default, missing)
    cap0_before = _load(repo / args.cap0_before, missing)
    cap0_after = _load(repo / args.cap0_after, missing)
    cap0_rollback = _load(repo / args.cap0_rollback, missing)
    replay_before = _load(repo / args.replay_before, missing)
    replay_after = _load(repo / args.replay_after, missing)
    main_column = _load(repo / args.main_column, missing)

    if missing:
        return (
            {
                "verdict": "not_judged_missing_faces",
                "missing": missing,
                "note": "缺件＝不判；§2 的逐列保持判定必须有分子分母两侧",
            },
            2,
        )

    assert powerup is not None and guard is not None
    assert cap0_before is not None and cap0_after is not None and cap0_rollback is not None
    assert replay_before is not None and replay_after is not None and main_column is not None

    before_tally = _cap0_tally(cap0_before)
    after_tally = _cap0_tally(cap0_after)
    rollback_tally = _cap0_tally(cap0_rollback)

    #: 面有效性先于判据（第一次的教训：100 项全 `load_ok=false` 会读成"能力＝0"）。
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
                "note": "有项未被模型作答⇒该面不可用；§3 G-N2-5 的装载性正是本件的新守卫，先修装载再谈判据",
                "cap0_before": before_tally,
                "cap0_after": after_tally,
                "cap0_rollback": rollback_tally,
            },
            2,
        )

    #: 可比性：三张 CAP-0 必须同一份冻结题集；主列两档必须同一 `items_sha256`。
    #: 不同源的两档放在一起作差就是换尺（㊵-484 那批"跨链搬 τ"的同类缺陷，这里是跨题集版）。
    eval_sets = {
        label: str(tally["eval_set"])
        for label, tally in (
            ("cap0_before", before_tally),
            ("cap0_after", after_tally),
            ("cap0_rollback", rollback_tally),
        )
    }
    entries = _main_column_entries(main_column)
    item_shas = {str(e["items_sha256"]) for e in entries.values()}
    not_comparable: list[str] = []
    if len(set(eval_sets.values())) != 1:
        not_comparable.append(f"cap0 eval_set 不同源：{eval_sets!r}")
    if len(item_shas) != 1:
        not_comparable.append(f"主列两档 items_sha256 不同：{sorted(item_shas)!r}")
    if len(entries) < 2:
        not_comparable.append(f"主列件里取到的档数 <2：{sorted(entries)!r}")
    if not_comparable:
        return (
            {
                "verdict": "not_judged_faces_not_comparable",
                "reasons": not_comparable,
                "note": "题集／行数不同源⇒前后差不可解释；先钉面，再谈判据",
            },
            2,
        )

    #: §3 G-N2-1 的面级实证：回退面与巩固前**读数**逐位同（剥易变＋剥出处，出处差另列出来核对）。
    comparable_before = json.dumps(_strip_volatile(cap0_before, top=True), sort_keys=True)
    comparable_rollback = json.dumps(_strip_volatile(cap0_rollback, top=True), sort_keys=True)
    comparable_after = json.dumps(_strip_volatile(cap0_after, top=True), sort_keys=True)
    g_n2_1_face_identical = comparable_before == comparable_rollback
    #: 动态范围：这把尺量不动权重变化时，"保持住了"不成立（第一次实测 `ruler_usable=true`）。
    ruler_usable = comparable_before != comparable_after

    #: ---- J-N2a'（逐列单值）----
    cap0_cols = _compare_columns(
        {k: v["correct"] for k, v in before_tally["per_dimension"].items()},
        {k: v["correct"] for k, v in after_tally["per_dimension"].items()},
        direction="bigger",
        label="cap0_per_dimension_machine_scored_correct",
    )
    arms_before = _replay_arms(replay_before)
    arms_after = _replay_arms(replay_after)
    replay_hits = _compare_columns(
        {k: v["strict_hits"] for k, v in arms_before.items()},
        {k: v["strict_hits"] for k, v in arms_after.items()},
        direction="bigger",
        label="replay24_strict_hits",
    )
    replay_formed = _compare_columns(
        {k: v["well_formed_texts"] for k, v in arms_before.items()},
        {k: v["well_formed_texts"] for k, v in arms_after.items()},
        direction="bigger",
        label="replay24_well_formed_texts",
    )
    #: 主列方向**预先指定**＝`never_lf_eaters_counted` 越小越好（PLAN-N2-02 §2 原文），故 smaller。
    #: 前后两档靠文件名认（这台仪器吃的是计数仪的成对件）；认不出一对 ⇒ `unverified`，不判保持。
    before_names = sorted(n for n in entries if "before" in n)
    after_names = sorted(n for n in entries if "after" in n)
    if len(before_names) == 1 and len(after_names) == 1:
        #: 配对键必须是**同一个列名**，不是文件名——文件名两档天然不同，
        #: 拿它当列名会让 `_compare_columns` 把两侧都记成"对面缺这一列"（本机自测踩到）。
        main_cols = _compare_columns(
            {"never_lf_eaters_counted": entries[before_names[0]]["never_lf_eaters_counted"]},
            {"never_lf_eaters_counted": entries[after_names[0]]["never_lf_eaters_counted"]},
            direction="smaller",
            label="stop24_main_column_never_lf_eaters_counted",
        )
        main_cols["faces"] = {"before": before_names[0], "after": after_names[0]}
        main_cols["items_sha256_pair"] = [
            entries[before_names[0]]["items_sha256"],
            entries[after_names[0]]["items_sha256"],
        ]
    else:
        main_cols = {
            "column_group": "stop24_main_column_never_lf_eaters_counted",
            "direction": "smaller",
            "columns": {},
            "dropped": [],
            "unverified": sorted(entries),
            "all_held": False,
        }

    groups = [cap0_cols, replay_hits, replay_formed, main_cols]
    j_n2a_prime = all(g["all_held"] for g in groups)
    dropped_map = {g["column_group"]: g["dropped"] for g in groups if g["dropped"]}
    unverified_map = {g["column_group"]: g["unverified"] for g in groups if g.get("unverified")}

    #: ---- J-持久化／J-N2b'：结论住在通电件里，本件只搬运＋核对不自相矛盾 ----
    j_persistence = bool(powerup.get("j_persistence"))
    j_n2b = str(powerup.get("j_n2b_prime") or "")
    d_quality = powerup.get("j_n2b_prime_delta_quality")
    d_accuracy = powerup.get("j_n2b_prime_delta_accuracy")
    derived = None
    if isinstance(d_quality, (int, float)) and isinstance(d_accuracy, (int, float)):
        derived = (
            "holds"
            if (d_quality > 0 and d_accuracy >= 0)
            else ("not_holds" if d_quality <= 0 else "not_resolved")
        )
    j_n2b_consistent = derived == j_n2b

    #: ---- 守卫五条 ----
    organs = powerup.get("organs") or {}
    surprise = powerup.get("surprise_modulation") or {}
    #: "剂量同窗"本身要机检：读数窗长必须等于本 pass 自述的经验预算——第一次是整篇对 64，
    #: 那一格不为真时 J-N2b' 的分子就不是这一轮吃进去的剂量，所以它是判据的前置而非装饰。
    dose_window_aligned = powerup.get("window_bytes") == organs.get("max_symbols")
    guard_checks = guard.get("checks") or {}
    #: G-N2-2 不吃件里的 `guard_pass` 一个布尔就算数：`ran=false` 与"摘要未变"各自独立现算，
    #: 缺键一律 fail-closed（默认值不能是 True，否则"没这一格"会读成"守住了"）。
    g_n2_2 = (
        (guard.get("organs") or {}).get("ran") is False
        and guard_checks.get("organs_ran_false") is True
        and guard_checks.get("weight_digest_unchanged") is True
        and str(guard.get("digest_before") or "") == str(guard.get("digest_after") or "")
        and bool(SHA64.match(str(guard.get("digest_before") or "")))
    )
    g_n2_1_digest = bool(powerup.get("restore_matches_mother")) and bool(
        powerup.get("rollback_disk_matches_mother")
    )
    g_n2_4 = bool(surprise.get("selected_before_modulation")) and bool(
        surprise.get("selected_after_modulation")
    )
    g_n2_5 = j_persistence and bool(powerup.get("g_n2_5_mother_roundtrip_identical"))
    #: G-N2-6：本件禁止用对齐件代答装载性。机检＝第二次通电名下**不许存在** align 产物，
    #: 且通电件里没有 `--phase align` 带进来的字段。这条能为 false（第一次就有那枚对齐件）。
    align_files = sorted(
        str(p.relative_to(PROJECT_ROOT))
        for p in (PROJECT_ROOT / "reports").glob("taiji_n2b_align*")
    )
    g_n2_6 = not align_files and not any(k.startswith("aligned_") for k in powerup)

    failed = [
        name
        for name, ok in (
            ("J-持久化", j_persistence),
            ("J-N2a'", j_n2a_prime),
            ("J-N2b'", j_n2b == "holds"),
        )
        if not ok
    ]
    conjunction = not failed
    verdict = "holds" if conjunction else "does_not_hold"

    payload: dict[str, Any] = {
        "format": "taiji-n2b-face-verdict-v1",
        "prereg": "plans/reference/PLAN-N2-02_second_powerup_dosewindow_prereg_20261008.md#2",
        "powerup_report": args.powerup,
        "verdict": verdict,
        "conjunction_holds": conjunction,
        "failed_criteria": failed,
        "criteria": {
            "j_persistence": j_persistence,
            "j_n2a_prime": j_n2a_prime,
            "j_n2b_prime": j_n2b,
            "j_n2b_prime_delta_quality": d_quality,
            "j_n2b_prime_delta_accuracy": d_accuracy,
            "j_n2b_prime_whole_text_delta_quality_corroboration": powerup.get(
                "whole_text_delta_quality"
            ),
            "j_n2b_recomputed_from_deltas": derived,
            "j_n2b_reader_agrees_with_driver": j_n2b_consistent,
            "dose_window_equals_experience_budget": dose_window_aligned,
            "window_bytes": powerup.get("window_bytes"),
            "organs_max_symbols": organs.get("max_symbols"),
        },
        "j_n2a_prime_columns": {
            "cap0_per_dimension": cap0_cols,
            "replay24_strict_hits": replay_hits,
            "replay24_well_formed_texts": replay_formed,
            "stop24_main_column": main_cols,
            "dropped_by_group": dropped_map,
            "unverified_by_group": unverified_map,
            "note": "每列后−前 单值判定；主列方向＝越小越好（§2 原文），unverified 不参与 all_held",
        },
        "guards": {
            "g_n2_1_rollback_digest_matches_mother": g_n2_1_digest,
            "g_n2_1_rollback_face_identical_to_before": g_n2_1_face_identical,
            "g_n2_2_default_position_untouched": g_n2_2,
            "g_n2_4_surprise_modulation_disclosed": g_n2_4,
            "g_n2_5_candidate_and_mother_roundtrip": g_n2_5,
            "g_n2_6_no_alignment_artifact": g_n2_6,
            "g_n2_6_align_files_found": align_files,
        },
        "ruler": {
            "cap0_ruler_usable": ruler_usable,
            "cap0_total_correct_before_after_rollback": [
                before_tally["total_correct"],
                after_tally["total_correct"],
                rollback_tally["total_correct"],
            ],
            "note": "CAP-0 逐维分子是 1 项量级的尺；1 项翻转是否算代价由 §2 的单值式判，不由本件改口",
        },
        "cap0_before": before_tally,
        "cap0_after": after_tally,
        "cap0_rollback": rollback_tally,
        "provenance_before": _provenance(cap0_before),
        "provenance_after": _provenance(cap0_after),
        "provenance_rollback": _provenance(cap0_rollback),
        "main_column": entries,
        "digests_carried_from_powerup": {
            "mother_digest": powerup.get("mother_digest"),
            "treated_digest": powerup.get("treated_digest"),
            "note": "原样搬运自通电件；本件不复算摘要，摘要级证据住在 G-N2-1/G-N2-5 那一层",
        },
        "stage_of_powerup": powerup.get("stage"),
        "powerup_errors": powerup.get("errors"),
    }
    payload["fail_closed_pass"] = bool(
        g_n2_1_face_identical
        and g_n2_1_digest
        and g_n2_2
        and g_n2_4
        and g_n2_5
        and g_n2_6
        and dose_window_aligned
        and j_n2b_consistent
        and powerup.get("stage") == "done"
        and not (powerup.get("errors") or [])
    )
    rc = 0 if payload["fail_closed_pass"] else 2
    return payload, rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N2 第二次通电面判读（PLAN-N2-02 §2/§3）")
    parser.add_argument("--powerup", default="reports/taiji_n2b_powerup_20261008.json")
    parser.add_argument("--guard-default", default="reports/taiji_n2b_guard_default_20261008.json")
    parser.add_argument("--cap0-before", default="reports/taiji_n2_cap0_before_20261008.json")
    parser.add_argument("--cap0-after", default="reports/taiji_n2b_cap0_after_20261008.json")
    parser.add_argument("--cap0-rollback", default="reports/taiji_n2b_cap0_rollback_20261008.json")
    parser.add_argument("--replay-before", default="reports/taiji_n2_replay24_before_20261008.json")
    parser.add_argument("--replay-after", default="reports/taiji_n2b_replay24_after_20261008.json")
    parser.add_argument("--main-column", default="reports/taiji_n2b_stop24_main_20261008.json")
    parser.add_argument("--out", default="reports/taiji_n2b_face_verdict_20261008.json")
    args = parser.parse_args(argv)

    payload, rc = judge(args)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    #: stdout 一律 ASCII（GBK 控制台会把带"⇒"的中文拒绝行炸成 UnicodeEncodeError，
    #: 那会把 rc=2 的响亮拒绝伪装成 rc=1 的崩溃——㊵-484⑤ 的教训）。
    criteria = payload.get("criteria") or {}
    columns = payload.get("j_n2a_prime_columns") or {}
    print(f"verdict={payload['verdict']}")
    print(f"failed={json.dumps(payload.get('failed_criteria', []), ensure_ascii=True)}")
    print(f"j_persistence={criteria.get('j_persistence')}")
    print(f"j_n2a_prime={criteria.get('j_n2a_prime')}")
    print(f"j_n2b_prime={criteria.get('j_n2b_prime')}")
    print(f"dose_window_aligned={criteria.get('dose_window_equals_experience_budget')}")
    print(
        f"whole_text_delta_quality={criteria.get('j_n2b_prime_whole_text_delta_quality_corroboration')}"
    )
    print(f"dropped={json.dumps(columns.get('dropped_by_group', {}))}")
    print(f"unverified={json.dumps(columns.get('unverified_by_group', {}))}")
    print(f"guards={json.dumps(payload.get('guards', {}))}")
    print(f"ruler={json.dumps(payload.get('ruler', {}))}")
    print(f"missing={json.dumps(payload.get('missing', []), ensure_ascii=True)}")
    print(f"reasons={json.dumps(payload.get('reasons', []), ensure_ascii=True)}")
    print(f"fail_closed_pass={payload.get('fail_closed_pass')}")
    print(f"report -> {out_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

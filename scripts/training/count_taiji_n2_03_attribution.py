"""PLAN-N2-03 的归因判读器：run-2 那 7 列跌破，代价归**收束**还是归**巩固的权重更新**？

口径全在件里（`plans/reference/PLAN-N2-03_reset_attribution_prereg_20261008.md` §2/§3ter），本脚本只是执行者。
取数函数**从 run-1/run-2 那两台判读器导入**（同一把尺），判级层另立：

* 逐列 `attrib_R = (R − before) / (after − before)`；分母为 0 的列**剔除并计数**（不静默当 0）；
* 三档互斥（`≥5` 收束致损／`2..4` 两族共因／`≤1` 权重致损）——中间带**有名字**，不留含糊出口（DEBT-G46 那一类）；
* 每条守卫都能为假：臂档参数族必须零差、必须可装载、三张面必须与基线同题集、CAP-0 面不许有未作答项。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.count_taiji_n2_faces import (  # noqa: E402 - 同一把尺，不重抄取数
    _cap0_tally,
    _main_column_entries,
    _replay_arms,
)

#: run-2 判为跌破的 7 列（§0 的来源件；写死在这里，跑完不许回头挑列）。
DROP_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cap0", "E"),
    ("replay_strict_hits", "1.0"),
    ("replay_strict_hits", "2.0"),
    ("replay_well_formed", "0.0"),
    ("replay_well_formed", "0.5"),
    ("replay_well_formed", "1.0"),
    ("replay_well_formed", "2.0"),
)

#: §2 的三档线（先冻）：attrib_R 达到 0.5 记"这一列被收束复现"；列数 ≥5 / 2..4 / ≤1。
ATTRIB_REPRODUCED = 0.5
N_STRONG = 5
N_WEAK = 2


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load(path: Path, missing: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        missing.append(_rel(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def judge(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    missing: list[str] = []
    repo = PROJECT_ROOT
    names = {
        "--arm-report": args.arm_report,
        "--cap0-before": args.cap0_before,
        "--cap0-after": args.cap0_after,
        "--cap0-arm": args.cap0_arm,
        "--replay-before": args.replay_before,
        "--replay-after": args.replay_after,
        "--replay-arm": args.replay_arm,
        #: 主列是旁证面（§2 末句：不入归因分数），但它同样是件——缺件也走 rc=2 的响亮拒绝，
        #: 不许在判读器里裸 `json.loads` 把"件没落"伪装成崩溃。
        "--main-column": args.main_column,
        "--main-column-baseline": args.main_column_baseline,
    }
    loaded = {k: _load(repo / v, missing) for k, v in names.items()}
    if missing:
        return (
            {"verdict": "not_judged_missing_faces", "missing": missing, "inputs": names},
            2,
        )
    arm = loaded["--arm-report"]
    assert arm is not None
    cap0 = {key: loaded[key] for key in ("--cap0-before", "--cap0-after", "--cap0-arm")}
    replay = {key: loaded[key] for key in ("--replay-before", "--replay-after", "--replay-arm")}
    assert all(v is not None for v in cap0.values()) and all(v is not None for v in replay.values())

    #: 守卫一／二：这一臂必须"只收束"（参数族零差）且产物可载。任一不成立 ⇒ 整件不判。
    arm_guards = {
        "g_n6_1_parameter_family_unchanged": arm.get("g_n6_1_parameter_family_unchanged") is True,
        "g_n6_2_arm_loadable_and_params_identical": arm.get(
            "g_n6_2_arm_loadable_and_params_identical"
        )
        is True,
    }
    #: 参数族"零差"还要看驱动自述的家族计数（摘要之外再多一道：条目数与 changed 计数）。
    fam = (arm.get("family_diff") or {}).get("parameter_family") or {}
    arm_guards["g_n6_1b_parameter_entries_stable"] = (
        fam.get("changed") == 0 and fam.get("dropped") == 0 and fam.get("added") == 0
    )
    if not all(arm_guards.values()):
        return (
            {"verdict": "not_judged_arm_not_clean", "arm_guards": arm_guards, "family_diff": fam},
            2,
        )

    #: 面有效性：CAP-0 任一面有项未被作答 ⇒ 那张面作废（不许读成"能力归零"）。
    tallies = {key: _cap0_tally(cap0[key]) for key in cap0}
    invalid = {key: t["load_not_ok"] for key, t in tallies.items() if t["load_not_ok"]}
    if invalid:
        return (
            {
                "verdict": "not_judged_face_invalid",
                "cap0_load_failures_by_face": invalid,
                "note": "有项未被模型作答⇒该面不可用；先修装载再谈归因",
            },
            2,
        )
    #: 面同源（G-N6-3）：CAP-0 三张面同一份题集；复述三张面同一 `items`／`item_offset`／`first_item`。
    eval_sets = {key: str(t["eval_set"]) for key, t in tallies.items()}
    replay_identities = {
        key: {f: replay[key].get(f) for f in ("items", "item_offset", "first_item")}
        for key in replay
    }
    mismatched: dict[str, Any] = {}
    if len(set(eval_sets.values())) != 1:
        mismatched["cap0_eval_set"] = eval_sets
    if len({json.dumps(v, sort_keys=True) for v in replay_identities.values()}) != 1:
        mismatched["replay_identity"] = replay_identities
    if mismatched:
        return (
            {"verdict": "not_judged_faces_not_comparable", "mismatch": mismatched},
            2,
        )

    #: 主列旁证（§2 末句：不入归因分数）＋G-N6-3 的主列那一半：题集 sha 必须与基线同一份。
    #: 守卫要能为假——两边都取不到 sha（None／缺键）时**不算通过**，否则"没记 sha"会伪装成"sha 相同"。
    main_arm_entries = _main_column_entries(loaded["--main-column"])
    main_baseline_entries = _main_column_entries(loaded["--main-column-baseline"])
    main_arm_shas = {v["items_sha256"] for v in main_arm_entries.values()}
    main_baseline_shas = {v["items_sha256"] for v in main_baseline_entries.values()}
    shas_ok = (
        all(isinstance(sha, str) and sha for sha in main_arm_shas | main_baseline_shas)
        and bool(main_arm_shas)
        and bool(main_baseline_shas)
        and main_arm_shas == main_baseline_shas
    )
    if not shas_ok:
        mismatched["main_column_items_sha256"] = {
            "arm": sorted(str(s) for s in main_arm_shas),
            "baseline": sorted(str(s) for s in main_baseline_shas),
            "arm_entries": len(main_arm_entries),
            "baseline_entries": len(main_baseline_entries),
        }
        return (
            {"verdict": "not_judged_faces_not_comparable", "mismatch": mismatched},
            2,
        )

    replay_arms = {key: _replay_arms(replay[key]) for key in replay}
    sources = {
        "cap0": {
            key: {d: v["correct"] for d, v in tallies[key]["per_dimension"].items()}
            for key in tallies
        },
        "replay_strict_hits": {
            key: {a: v["strict_hits"] for a, v in replay_arms[key].items()} for key in replay_arms
        },
        "replay_well_formed": {
            key: {a: v["well_formed_texts"] for a, v in replay_arms[key].items()}
            for key in replay_arms
        },
    }

    rows: dict[str, Any] = {}
    excluded: list[str] = []
    n_reproduced = 0
    #: 面族 → 读数来源（键名写成对，避免"面"与"列"两套口径混在一处）
    family_sources = {
        "cap0": ("--cap0-before", "--cap0-after", "--cap0-arm"),
        "replay_strict_hits": ("--replay-before", "--replay-after", "--replay-arm"),
        "replay_well_formed": ("--replay-before", "--replay-after", "--replay-arm"),
    }
    for face, column in DROP_COLUMNS:
        src = sources[face]
        k_before, k_after, k_arm = family_sources[face]
        before, after, arm_r = (
            src[k_before].get(column),
            src[k_after].get(column),
            src[k_arm].get(column),
        )
        key = f"{face}:{column}"
        if None in (before, after, arm_r):
            excluded.append(key)
            continue
        denom = after - before
        if denom == 0:
            #: run-2 判这列跌过 ⇒ 分母为 0 说明面不同源或读数漂移，不能当"0 归因"。
            excluded.append(key)
            continue
        attrib = (arm_r - before) / denom
        reproduced = attrib >= ATTRIB_REPRODUCED
        n_reproduced += int(reproduced)
        rows[key] = {
            "before": before,
            "after_run2": after,
            "arm_R": arm_r,
            "delta_run2": denom,
            "delta_armR": arm_r - before,
            "attrib_R": round(attrib, 4),
            "reproduced": reproduced,
        }

    considered = len(DROP_COLUMNS) - len(excluded)
    if n_reproduced >= N_STRONG:
        verdict = "cost_from_wake_reset"
    elif n_reproduced >= N_WEAK:
        verdict = "partially_resolved"
    else:
        verdict = "cost_from_weight_update"

    return (
        {
            "format": "taiji-n2-03-attribution-v1",
            "prereg": "plans/reference/PLAN-N2-03_reset_attribution_prereg_20261008.md#2",
            "verdict": verdict,
            "criteria": {
                "columns_frozen": [f"{f}:{c}" for f, c in DROP_COLUMNS],
                "attrib_reproduced_line": ATTRIB_REPRODUCED,
                "n_reproduced": n_reproduced,
                "n_strong_line": N_STRONG,
                "n_weak_line": N_WEAK,
                "columns_considered": considered,
                "columns_excluded": excluded,
                "note": "三档互斥；被剔除的列只减分母不减证据，全部点名（不留中间格）",
            },
            "per_column": rows,
            "guards": arm_guards,
            "face_identity": {
                "cap0_eval_set": eval_sets,
                "replay_identity": replay_identities,
                "main_column_items_sha256": sorted(str(s) for s in main_arm_shas),
            },
            "arm_self_report": {
                "wake_episode_id": arm.get("wake_episode_id"),
                "params_digest_before": arm.get("params_digest_before"),
                "params_digest_after": arm.get("params_digest_after"),
                "family_diff": arm.get("family_diff"),
                "excluded_changed_paths": arm.get("excluded_changed_paths"),
                "tick_before_after": [arm.get("mother_tick"), arm.get("tick_after_reset")],
            },
            "main_column_corroboration_not_judged": {
                "arm": main_arm_entries,
                "baseline": main_baseline_entries,
                "note": "§2 末句：主列 run-2 是改善不是跌破，方向为越小越好 ⇒ 只报数、不入归因分数",
            },
            "inputs": names,
        },
        0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N2-03 归因判读（收束 vs 权重更新）")
    parser.add_argument("--arm-report", default="reports/taiji_n2b_resetarm_20261008.json")
    parser.add_argument("--cap0-before", default="reports/taiji_n2_cap0_before_20261008.json")
    parser.add_argument("--cap0-after", default="reports/taiji_n2b_cap0_after_20261008.json")
    parser.add_argument("--cap0-arm", default="reports/taiji_n2b_resetarm_cap0_20261008.json")
    parser.add_argument("--replay-before", default="reports/taiji_n2_replay24_before_20261008.json")
    parser.add_argument("--replay-after", default="reports/taiji_n2b_replay24_after_20261008.json")
    parser.add_argument("--replay-arm", default="reports/taiji_n2b_resetarm_replay24_20261008.json")
    parser.add_argument(
        "--main-column", default="reports/taiji_n2b_resetarm_stop24_main_20261008.json"
    )
    parser.add_argument(
        "--main-column-baseline",
        default="reports/taiji_n2b_stop24_main_20261008.json",
        help="run-2 的主列计数件（题集 sha 与臂档必须同一份，G-N6-3）",
    )
    parser.add_argument("--out", default="reports/taiji_n2_03_attribution_20261008.json")
    args = parser.parse_args(argv)

    payload, rc = judge(args)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"verdict={payload['verdict']}")
    crit = payload.get("criteria") or {}
    print(f"n_reproduced={crit.get('n_reproduced')} considered={crit.get('columns_considered')}")
    print(f"excluded={json.dumps(crit.get('columns_excluded', []))}")
    print(f"guards={json.dumps(payload.get('guards', {}))}")
    if "per_column" in payload:
        print(
            "attrib_R=" + json.dumps({k: v["attrib_R"] for k, v in payload["per_column"].items()})
        )
    print(f"missing={json.dumps(payload.get('missing', []))}")
    print(f"report -> {out_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

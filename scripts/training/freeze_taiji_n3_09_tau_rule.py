"""PLAN-N3-09 的 τ 规则升版判读器：冻"取法与验收"，不冻一个够不到的数。

口径全在件里（`plans/reference/PLAN-N3-09_tau_rule_upgrade_prereg_20261008.md` §2/§3/§4），本脚本只是执行者。
四条硬规矩：

* **只用面内自述**：母量＝每行的 `decision_pressure`（＝产品的 `pressure_ema`），六道阈与
  `required_pressure_steps` 全取 face 头的 `policy` 段——**一个数都不回落到产品默认**（回落会静默重建 DEBT-G53）；
* **每行都与产品自述对表**：我按 §2 重算的"六道合取"必须与该行 `decision_reasons` 里
  有没有 `pressure_below_threshold` 逐行一致，任一行不一致 ⇒ rc=2（这条把"我另算一遍"钉成"产品算的那一次"）；
* **动态范围是验收不是流程**：候选 τ 若不能让六道合取在同面连续为真 ≥ `required_pressure_steps` 步 ⇒
  判 `not_freezable_at_this_grid` 并出版那条"要可解必须 ≤ X"的上界；
* **格宽 0.05 的旧答案同报**（对照用、不参与判级），因为本件主张的就是"0.05 粗到把链差吞掉"。

控制台输出走 ASCII 转义：拒绝理由里有"⇒"，GBK 终端上 `print` 会在判完之后抛 UnicodeEncodeError，
把 rc=2 的响亮拒绝伪装成 rc=1 的崩溃（同一类缺陷已犯过两次）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.count_taiji_n3_pressure_thresholds import (
    MIN_OBSERVATIONS,
    _percentile,
)  # noqa: E402

#: §2.2 的 warmup 剔除条数（推导 `log(0.01)/log(0.75) ≈ 16.0078` 上取整 ⇒ 17；**本件推导值，非实测**）。
WARMUP_DROP = 17
#: §2.3 的新格宽与旧格宽（旧的那把只出对照）。
GRID_NEW = 0.01
GRID_OLD = 0.05
FACE_FORMAT = "taiji-n3-pressure-face-v2"

#: 六道闸：(行内的 EMA 键, 头里的阈键)。
GATES = (
    ("decision_pressure", "minimum_pressure"),
    ("decision_residual_error_ema", "minimum_residual_error"),
    ("decision_fast_slow_conflict_ema", "minimum_fast_slow_conflict"),
    ("decision_activity_saturation_ema", "minimum_activity_saturation"),
    ("decision_utility_gap_ema", "minimum_utility_gap"),
    ("decision_resource_state_ema", "minimum_resource_state"),
)
#: 五道分项闸（算"要可解必须 ≤ X"那条上界时要摘掉合成量）。
SUB_GATES = GATES[1:]
REQUIRED_POLICY_KEYS = tuple(field for _, field in GATES) + (
    "required_pressure_steps",
    "growth_resource_cost",
    "ema_rate",
)


def _on_grid(value: float, grid: float) -> float:
    return round(min(1.0, math.ceil(value / grid) * grid), 4)


def _longest_true_run(flags: list[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _load_face(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        kind = record.get("kind")
        if kind == "face":
            if header is not None:
                raise ValueError(f"{path.name} 里有两条 face 行 ⇒ 面不唯一，不判")
            header = record
        elif kind == "pressure":
            rows.append(record)
        elif kind == "tail":
            continue
        else:
            raise ValueError(f"{path.name} 出现不认识的行 kind={kind!r} ⇒ 不判")
    if header is None:
        raise ValueError(f"{path.name} 缺 face 头 ⇒ 这张面没有自述，不判")
    if str(header.get("format")) != FACE_FORMAT:
        raise ValueError(
            f"{path.name} 的 format={header.get('format')!r} 不是 {FACE_FORMAT} ⇒ "
            "本件的规则只认面内自述口径，v1 面缺六道阈与 EMA ⇒ 不判（要走豁免须按 PLAN-N3-07 §2 重放＋双锚点）"
        )
    policy = header.get("policy") or {}
    missing = [key for key in REQUIRED_POLICY_KEYS if key not in policy]
    if missing:
        raise ValueError(f"{path.name} 的 face 头缺自述 {missing} ⇒ 不许回落产品默认，不判")
    if not (header.get("ema_initial") or {}):
        raise ValueError(f"{path.name} 缺 ema_initial 段 ⇒ warmup 那条处置无从核对，不判")
    for index, row in enumerate(rows):
        for key, _ in GATES:
            if key not in row:
                raise ValueError(f"{path.name} 第 {index} 行缺 {key!r} ⇒ 不判")
        if "decision_reasons" not in row:
            raise ValueError(f"{path.name} 第 {index} 行缺 decision_reasons ⇒ 对表做不了，不判")
    if len(rows) - WARMUP_DROP < MIN_OBSERVATIONS:
        raise ValueError(
            f"{path.name} 剔除 warmup 后只剩 {len(rows) - WARMUP_DROP} 条，"
            f"低于样本下限 {MIN_OBSERVATIONS} ⇒ 不判"
        )
    return header, rows


def judge_face(path: Path) -> dict[str, Any]:
    header, rows = _load_face(path)
    policy = header["policy"]
    required_steps = int(policy["required_pressure_steps"])
    composite_key, composite_field = GATES[0]

    #: 逐行与产品自述对表：我算的"六道合取"必须等于 `reasons` 里没有 `pressure_below_threshold`。
    mismatch = 0
    first_mismatch: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        meets = all(
            float(row[key]) >= float(policy[field]) for key, field in GATES if key != composite_key
        ) and float(row[composite_key]) >= float(policy[composite_field])
        claimed = "pressure_below_threshold" not in list(row["decision_reasons"])
        if meets != claimed:
            mismatch += 1
            if first_mismatch is None:
                first_mismatch = {"row": index, "derived": meets, "product": claimed}
    if mismatch:
        raise ValueError(
            f"A 表失败：{path.name} 有 {mismatch} 行合取与产品自述不一致，"
            f"首个在第 {first_mismatch['row']} 行（推导 {first_mismatch['derived']} 对 "
            f"产品 {first_mismatch['product']}）⇒ 本件的口径与闸不同源，不判"
        )

    kept = rows[WARMUP_DROP:]
    series = [float(row[composite_key]) for row in kept]
    ordered = sorted(series)
    p90 = _percentile(ordered, 0.90)
    candidate = _on_grid(p90, GRID_NEW)

    #: 动态范围硬验收：在候选 τ 下，六道合取为真的最长连续段。
    flags = [
        all(float(row[key]) >= float(policy[field]) for key, field in SUB_GATES)
        and float(row[composite_key]) >= candidate
        for row in kept
    ]
    true_steps = sum(1 for flag in flags if flag)
    longest = _longest_true_run(flags)
    freezable = longest >= required_steps

    #: 要可解必须 ≤ 的那个上界：只在"五道分项闸同时为真"的窗口里看合成量能到多高。
    window = [
        float(row[composite_key])
        for row in kept
        if all(float(row[key]) >= float(policy[field]) for key, field in SUB_GATES)
    ]
    ceiling = round(max(window), 6) if window else None

    verdict = "freezable" if freezable else "not_freezable_at_this_grid"
    return {
        "face_dir": path.parent.name,
        "format": FACE_FORMAT,
        "observations": len(rows),
        "warmup_dropped": WARMUP_DROP,
        "warmup_share": round(WARMUP_DROP / len(rows), 6),
        "policy_self_reported": {key: policy[key] for key in REQUIRED_POLICY_KEYS},
        "ema_initial": header["ema_initial"],
        "gate_table_rows_checked": len(rows),
        "gate_table_mismatches": mismatch,
        "pressure_ema_kept": {
            "p50": round(_percentile(ordered, 0.50), 6),
            "p90": round(p90, 6),
            "p99": round(_percentile(ordered, 0.99), 6),
            "max": round(ordered[-1], 6),
            "mean": round(sum(ordered) / len(ordered), 6),
        },
        "tau_candidate_v2": candidate,
        "grid_005_comparison": _on_grid(p90, GRID_OLD),
        "dynamic_range_at_candidate": {
            "required_pressure_steps": required_steps,
            "six_gate_true_steps": true_steps,
            "six_gate_longest_run": longest,
            "passes": freezable,
        },
        "five_gate_window_composite_ceiling": ceiling,
        "must_be_at_most_to_be_solvable": ceiling,
        "proposals_in_face": sum(1 for row in rows if row.get("decision_should_propose")),
        "counterfactual_valid_up_to_first_proposal": True,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N3-09 τ 规则升版判读（面内自述口径）")
    parser.add_argument("--face", action="append", required=True, help="v2 压强面 JSONL")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    judged: list[dict[str, Any]] = []
    refused: list[dict[str, str]] = []
    for raw in args.face:
        path = Path(raw)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        try:
            judged.append(judge_face(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            refused.append({"face": str(path), "error": str(error)[:400]})

    candidates = {f["face_dir"]: f["tau_candidate_v2"] for f in judged}
    #: J-N3e-跨链分辨力：同 gate 档上 beta 与 circuit 的候选值是否不同（相同 ⇒ 升格主张被否证）。
    discrimination: dict[str, Any] = {}
    for suffix in ("gate025", "gate100"):
        beta = candidates.get(f"beta_{suffix}_v2")
        circuit = candidates.get(f"circuit_{suffix}_v2")
        if beta is None or circuit is None:
            discrimination[suffix] = {"status": "unverified_missing_face"}
        else:
            discrimination[suffix] = {
                "beta": beta,
                "circuit": circuit,
                "resolvable": beta != circuit,
            }

    payload = {
        "format": "taiji-n3-09-tau-rule-v1",
        "prereg": "plans/reference/PLAN-N3-09_tau_rule_upgrade_prereg_20261008.md",
        "status": "ok" if judged and not refused else ("refused" if not judged else "partial"),
        "rule": {
            "measure": "decision_pressure（＝产品的 pressure_ema，五个 EMA 的加权和 :388-395）",
            "warmup_drop": WARMUP_DROP,
            "grid_new": GRID_NEW,
            "grid_old_reported_for_comparison": GRID_OLD,
            "acceptance": "候选 τ 下六道合取的最长连续段必须 ≥ required_pressure_steps，否则判不可冻",
        },
        "faces": judged,
        "refused": refused,
        "faces_judged": len(judged),
        "faces_refused": len(refused),
        "verdict_by_face": {f["face_dir"]: f["verdict"] for f in judged},
        "j_n3e_cross_chain_resolution": discrimination,
    }
    rc = 0 if judged and not refused else 2
    if args.out:
        out = Path(args.out)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())

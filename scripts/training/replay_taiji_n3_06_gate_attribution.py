"""PLAN-N3-06：把已在盘的压强面**逐 tick 重放**过产品自己的生长闸，定位六道分项闸里谁在拦。

判据与验收式全在 `plans/reference/PLAN-N3-06_gate_attribution_replay_prereg_20261008.md` §2/§3，本脚本只是执行者。
三条硬规矩：

* **零重抄算式**：观测用 `AdaptiveResidualGrowthPressure.create(...)` 构造（加权和与 `pressure_digest` 由产品自己算），
  决策用同一支 `AdaptiveResidualGrowthTrigger.observe` 产出；仪器只做"读面→构造→observe→摊平"。
* **锚点先行**：重放出的 `should_propose` 与 `pressure_digest` 必须与面里逐行相同（A-1/A-2）才出版；
  任一行不等 ⇒ rc=2 并点名首个失配行——这条把"外部重放"钉成"被产品自述核对过的重放"。
* **假设要自述**：面头只记了三道阈（`minimum_pressure`／`required_pressure_steps`／`growth_resource_cost`），
  另外四道 `minimum_*` 与 `ema_rate` 取自产品默认 ⇒ 件里逐条标 `assumed_from_product_defaults`（＝DEBT-G53）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.count_taiji_n3_pressure_thresholds import (  # noqa: E402 - 同一把尺
    MIN_OBSERVATIONS,
    _freeze,
    _percentile,
)
from taiji.adaptive_residual_growth import (  # noqa: E402
    AdaptiveResidualGrowthPolicy,
    AdaptiveResidualGrowthPressure,
    AdaptiveResidualGrowthTrigger,
)

#: 六道闸（`adaptive_residual_growth.py:437-444` 的一条 `and` 链）：
#: **(读数标签, decision 上的属性名, policy 里的阈字段名, 面行里的原始字段名)**。
#: 第一道是**合成量**——decision 的属性叫 `pressure`，但它装的是 `pressure_ema`（:388-395 的加权和、:475 的取值）。
GATES = (
    ("pressure_ema", "pressure", "minimum_pressure", "pressure"),
    ("residual_error_ema", "residual_error_ema", "minimum_residual_error", "residual_error"),
    (
        "fast_slow_conflict_ema",
        "fast_slow_conflict_ema",
        "minimum_fast_slow_conflict",
        "fast_slow_conflict",
    ),
    (
        "activity_saturation_ema",
        "activity_saturation_ema",
        "minimum_activity_saturation",
        "activity_saturation",
    ),
    ("utility_gap_ema", "utility_gap_ema", "minimum_utility_gap", "utility_gap"),
    ("resource_state_ema", "resource_state_ema", "minimum_resource_state", "resource_state"),
)

#: 读侧接受的 face 版本集；写侧的字面量由契约测钉为"必须落进这个集合"（G-N3c-4）。
FACE_FORMATS = ("taiji-n3-pressure-face-v1", "taiji-n3-pressure-face-v2")

#: 闸的九个 policy 字段——面头里有几个就用几个，缺的才回落产品默认（并逐条披露）。
POLICY_FIELDS = (
    "minimum_pressure",
    "minimum_residual_error",
    "minimum_fast_slow_conflict",
    "minimum_activity_saturation",
    "minimum_utility_gap",
    "minimum_resource_state",
    "required_pressure_steps",
    "growth_resource_cost",
    "ema_rate",
)

#: 面头缺自述、只能取产品默认的字段（如实披露，不假装面里有过）——v1 面缺这六条，v2 面应为空。
ASSUMED_FIELDS = POLICY_FIELDS[1:6] + ("ema_rate",)

#: 产品在 :461-467 只会发这四条原因。出现表外原因 ⇒ 说明决策侧被动过，本件的指认不再成立 ⇒ 拒判。
KNOWN_REASONS = (
    "pressure_below_threshold",
    "pressure_persistence_below_threshold",
    "structural_budget_insufficient",
    "persistent_native_pressure",
)


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
    #: 版本集机检（G-N3c-4 的读侧一半）：表外版本 ⇒ 响亮拒绝，不按"大概是 v1"猜着读。
    if str(header.get("format")) not in FACE_FORMATS:
        raise ValueError(
            f"{path.name} 的 format={header.get('format')!r} 不在读侧接受集 {FACE_FORMATS} ⇒ 不判"
        )
    if len(rows) < MIN_OBSERVATIONS:
        raise ValueError(
            f"{path.name} 观测只有 {len(rows)} 条，低于样本下限 {MIN_OBSERVATIONS} ⇒ 不判"
        )
    for index, row in enumerate(rows):
        for _, _, _, raw_field in GATES:
            if raw_field not in row:
                raise ValueError(
                    f"{path.name} 第 {index} 行缺原始字段 {raw_field!r} ⇒ 重放做不了，不判"
                )
        for field in ("pressure", "decision_should_propose", "evidence_id", "tick"):
            if field not in row:
                raise ValueError(f"{path.name} 第 {index} 行缺 {field!r} ⇒ 锚点核对做不了，不判")
    return header, rows


def _policy_from_header(header: dict[str, Any]) -> tuple[Any, list[str]]:
    """用面头自述的阈覆盖产品默认，**缺哪条回落哪条并逐条披露**。

    v1 面只自述三条 ⇒ `assumed_from_product_defaults` 非空；v2 面九条齐 ⇒ 空。
    回落不是错（旧面读不了就作废更糟），但**必须被点名**——静默回落会把"仪器没记"读成"产品默认"。
    """

    recorded = header.get("policy") or {}
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("face 头里没有 policy 段 ⇒ 连阈都取不到，不判")
    if "minimum_pressure" not in recorded:
        raise ValueError("face 头缺 minimum_pressure 自述 ⇒ 合成量那道闸读不出，不判")
    policy = AdaptiveResidualGrowthPolicy()
    for field, value in recorded.items():
        if field not in POLICY_FIELDS:
            raise ValueError(f"face 头里出现表外 policy 字段 {field!r} ⇒ 不判")
        policy = dataclasses.replace(policy, **{field: value})
    return policy, [field for field in ASSUMED_FIELDS if field not in recorded]


def _longest_true_run(flags: list[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def replay(path: Path) -> dict[str, Any]:
    header, rows = _load_face(path)
    policy, assumed = _policy_from_header(header)
    bridge_id = rows[0].get("bridge_id") or (header.get("growth") or {}).get("bridge_id")
    if not bridge_id:
        raise ValueError(f"{path.name} 取不到 bridge_id ⇒ trigger 构造不出来，不判")
    parent = rows[0].get("parent_checkpoint_digest") or ""
    trigger = AdaptiveResidualGrowthTrigger(
        bridge_id=str(bridge_id), policy=policy, parent_checkpoint_digest=str(parent) or None
    )
    #: 预算不是状态量（`taiji/model.py:1084`），且闸对它只有 `>= cost` 一种用法 ⇒ 取 cost 与原始跑等价；
    #: 这条等价**不由论证生效**，由下面 A-1 的逐行 `should_propose` 全等生效。
    budget = int(policy.growth_resource_cost)

    ema_series: dict[str, list[float]] = {label: [] for label, _, _, _ in GATES}
    pressure_ema_series: list[float] = []
    meets_flags: list[bool] = []
    replay_proposes: list[bool] = []
    recorded_proposes: list[bool] = []
    reason_counts: dict[str, int] = {}
    streak_max = 0
    digest_mismatch: dict[str, Any] | None = None
    faithfulness_mismatch: dict[str, Any] | None = None
    digest_compared = digest_mismatch_count = faithfulness_mismatch_count = 0

    for index, row in enumerate(rows):
        observation = AdaptiveResidualGrowthPressure.create(
            bridge_id=str(row["bridge_id"]),
            tick=int(row["tick"]),
            residual_error=float(row["residual_error"]),
            fast_slow_conflict=float(row["fast_slow_conflict"]),
            activity_saturation=float(row["activity_saturation"]),
            utility_gap=float(row["utility_gap"]),
            resource_state=float(row["resource_state"]),
            evidence_id=str(row["evidence_id"]),
            parent_checkpoint_digest=str(row.get("parent_checkpoint_digest") or ""),
        )
        payload = observation.to_payload()
        if "pressure_digest" in row:
            digest_compared += 1
            if str(payload.get("pressure_digest")) != str(row["pressure_digest"]):
                digest_mismatch_count += 1
                if digest_mismatch is None:
                    digest_mismatch = {
                        "row": index,
                        "replay": str(payload.get("pressure_digest")),
                        "face": str(row["pressure_digest"]),
                    }
        decision = trigger.observe(observation, structural_budget=budget)
        replay_flag = bool(decision.should_propose)
        recorded_flag = bool(row["decision_should_propose"])
        replay_proposes.append(replay_flag)
        recorded_proposes.append(recorded_flag)
        if replay_flag != recorded_flag:
            faithfulness_mismatch_count += 1
            if faithfulness_mismatch is None:
                faithfulness_mismatch = {"row": index, "replay": replay_flag, "face": recorded_flag}
        meets = "pressure_below_threshold" not in list(decision.reasons)
        meets_flags.append(meets)
        streak_max = max(streak_max, int(decision.consecutive_pressure_steps))
        for label, attr, _, _ in GATES:
            ema_series[label].append(float(getattr(decision, attr)))
        pressure_ema_series.append(float(decision.pressure))
        for reason in decision.reasons:
            if str(reason) not in KNOWN_REASONS:
                raise ValueError(
                    f"第 {index} 行出现表外原因 {str(reason)!r} ⇒ 决策侧词表与件 §1 锚点不符，不判"
                )
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1

    if digest_mismatch is not None:
        raise ValueError(
            f"A-2 观测自洽失败：第 {digest_mismatch['row']} 行重放摘要 "
            f"{digest_mismatch['replay'][:16]}… 与面里 {digest_mismatch['face'][:16]}… 不同 ⇒ 不判"
        )
    if faithfulness_mismatch is not None:
        raise ValueError(
            f"A-1 重放忠实性失败：第 {faithfulness_mismatch['row']} 行重放 should_propose="
            f"{faithfulness_mismatch['replay']} 对 面里 {faithfulness_mismatch['face']} ⇒ 不判"
        )

    per_gate: dict[str, Any] = {}
    dead_gates: list[str] = []
    #: 六道闸逐道判"过没过"，再与产品自己那条 `and` 链的结果对齐（:437-444）。
    #: 这条等式是本件"指认哪一道"的合法性来源：对不上就说明我的读数口径与门不同源 ⇒ 拒判。
    derived_meets = [
        all(
            ema_series[label][step] >= float(getattr(policy, threshold_field))
            for label, _, threshold_field, _ in GATES
        )
        for step in range(len(rows))
    ]
    derived_meets_steps = sum(1 for flag in derived_meets if flag)
    product_meets_steps = sum(1 for flag in meets_flags if flag)
    if derived_meets_steps != product_meets_steps:
        raise ValueError(
            f"六道闸合取与产品自述不一致：推导 {derived_meets_steps} 步对 "
            f"`reasons` 推得 {product_meets_steps} 步 ⇒ 指认不成立，不判"
        )
    for label, _, threshold_field, raw_field in GATES:
        values = sorted(ema_series[label])
        threshold = float(getattr(policy, threshold_field))
        flags = [value >= threshold for value in ema_series[label]]
        true_steps = sum(1 for flag in flags if flag)
        if true_steps == 0:
            dead_gates.append(label)
        raw = sorted(float(row[raw_field]) for row in rows)
        per_gate[label] = {
            "threshold": threshold,
            "threshold_self_reported_in_face": threshold_field in (header.get("policy") or {}),
            "ema_last": round(values[-1], 6),
            "ema_p90": round(_percentile(values, 0.90), 6),
            "ema_max": round(values[-1], 6),
            "raw_p90": round(_percentile(raw, 0.90), 6),
            "raw_max": round(raw[-1], 6),
            "true_steps": true_steps,
            "true_share": round(true_steps / len(rows), 6),
            "longest_true_run": _longest_true_run(flags),
            "gap_to_threshold_at_p90": round(threshold - _percentile(values, 0.90), 6),
        }

    #: 反事实定价（不参与 J-N3c 判级，只给"下调 τ 能不能触发"一个下界）：
    #: 把合成量那道闸摘掉，其余五道**同时**为真的步数与最长连续段。
    other_gate_flags = [
        all(
            ema_series[label][step] >= float(getattr(policy, threshold_field))
            for label, _, threshold_field, _ in GATES
            if label != "pressure_ema"
        )
        for step in range(len(rows))
    ]
    five_gate_steps = sum(1 for flag in other_gate_flags if flag)
    five_gate_longest = _longest_true_run(other_gate_flags)
    #: τ 的**可触发上界**：只在其余五道同时为真的步里看合成量能到多高——τ 高于这个数就算术不可能触发。
    composite_in_five_windows = [
        ema_series["pressure_ema"][step] for step, flag in enumerate(other_gate_flags) if flag
    ]
    tau_fire_ceiling = (
        round(max(composite_in_five_windows), 6) if composite_in_five_windows else None
    )
    #: 把既有冻结规则（p90 向上取整到 0.05 格）**原样**套到 EMA 口径上，再看六道合取能不能连续到
    #: `required_pressure_steps`——这条是"τ 该定在哪"的可执行定价，仍不参与 J-N3c 判级。
    tau_candidate = _freeze(_percentile(sorted(ema_series["pressure_ema"]), 0.90))
    candidate_flags = [
        other_gate_flags[step] and ema_series["pressure_ema"][step] >= tau_candidate
        for step in range(len(rows))
    ]
    candidate_steps = sum(1 for flag in candidate_flags if flag)
    candidate_longest = _longest_true_run(candidate_flags)
    would_trigger = candidate_longest >= int(policy.required_pressure_steps)

    if len(dead_gates) == 1:
        verdict = "single_gate_blocking"
    elif len(dead_gates) >= 2:
        verdict = "multiple_gates_blocking"
    else:
        verdict = "not_the_six_subgates"

    return {
        "face": path.name,
        "face_dir": path.parent.name,
        "observations": len(rows),
        "g_n3c_a1_rows_compared": len(rows),
        "g_n3c_a1_should_propose_mismatches": faithfulness_mismatch_count,
        "g_n3c_a2_digests_compared": digest_compared,
        "g_n3c_a2_digest_mismatches": digest_mismatch_count,
        "policy_from_face_header": header.get("policy"),
        "assumed_from_product_defaults": assumed,
        "structural_budget_used": budget,
        "should_propose_replay_total": sum(1 for flag in replay_proposes if flag),
        "should_propose_face_total": sum(1 for flag in recorded_proposes if flag),
        "meets_pressure_true_steps": sum(1 for flag in meets_flags if flag),
        "g_n3c_meets_pressure_crosscheck_derived_steps": derived_meets_steps,
        "g_n3c_meets_pressure_crosscheck_equal": derived_meets_steps == product_meets_steps,
        "meets_pressure_longest_run": _longest_true_run(meets_flags),
        "consecutive_pressure_steps_max": streak_max,
        "required_pressure_steps": int(policy.required_pressure_steps),
        "pressure_ema": {
            "p50": round(_percentile(sorted(pressure_ema_series), 0.50), 6),
            "p90": round(_percentile(sorted(pressure_ema_series), 0.90), 6),
            "max": round(max(pressure_ema_series), 6),
            "last": round(pressure_ema_series[-1], 6),
            "minimum_pressure_threshold": float(policy.minimum_pressure),
        },
        "reasons_histogram": dict(sorted(reason_counts.items())),
        "dead_gates": dead_gates,
        "per_gate": per_gate,
        "counterfactual_not_judged": {
            "five_gate_all_pass_steps": five_gate_steps,
            "five_gate_all_pass_share": round(five_gate_steps / len(rows), 6),
            "five_gate_longest_run": five_gate_longest,
            "required_pressure_steps": int(policy.required_pressure_steps),
            "tau_fire_ceiling_within_five_gate_windows": tau_fire_ceiling,
            "frozen_rule_reapplied_on_ema_caliber": {
                "candidate_tau": tau_candidate,
                "six_gate_true_steps": candidate_steps,
                "six_gate_longest_run": candidate_longest,
                "required_pressure_steps": int(policy.required_pressure_steps),
                "would_have_triggered": would_trigger,
            },
            "note": "这条只给'下调 τ 能不能触发'的算术上界，不参与 J-N3c 判级；"
            "ceiling 为 None ⇒ 其余五道从未同时为真过，τ 调到多少都不会触发",
        },
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N3-06 六道分项闸定位（零算力重放）")
    parser.add_argument(
        "--face", action="append", required=True, help="`--pressure-record` 的 JSONL"
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args(argv)

    faces: list[dict[str, Any]] = []
    refused: list[dict[str, str]] = []
    for raw in args.face:
        path = Path(raw)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        try:
            faces.append(replay(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            refused.append({"face": str(path), "error": str(error)[:300]})

    verdicts = [face["verdict"] for face in faces]
    payload = {
        "format": "taiji-n3-06-gate-attribution-v1",
        "prereg": "plans/reference/PLAN-N3-06_gate_attribution_replay_prereg_20261008.md",
        "status": "ok" if faces and not refused else ("refused" if not faces else "partial"),
        "faces": faces,
        "refused": refused,
        "faces_judged": len(faces),
        "faces_refused": len(refused),
        "verdict_by_face": {face["face_dir"]: face["verdict"] for face in faces},
        "verdict_all_agree": len(set(verdicts)) == 1 if verdicts else False,
    }
    rc = 0 if faces and not refused else 2
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_report:
        out = Path(args.out_report)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8", newline="\n")
    #: 控制台走 ASCII 转义：拒绝理由含"⇒"，GBK 终端上会在判完之后抛 UnicodeEncodeError，
    #: 把 rc=2 的响亮拒绝伪装成 rc=1 的崩溃（同一类缺陷已在 N3-01 判读器上犯过一次）。
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())

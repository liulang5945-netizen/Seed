"""N3 甲（×10 单点 scaling 探针）的判读器——PLAN-N3-02 §2/§4quater 的落地件（只读，不跑）。

它要回答的只有一句：`×16.8 容量在等符号暴露下有没有把斜率抬出它自己的噪声带`，
而这条面上一切"看起来像读数"的东西都得先被处置干净：

* **DEBT-G63 当场生效**：进度面最后一行的 `window_ticks == 0`，而 `online_accuracy`／`mean_surprise`
  出版的是 0.0（0/0 被 `max(1, …)` 护栏抹平）⇒ **算段均值/斜率之前必须按 `window_ticks > 0` 筛**，
  并披露剔除了几行。筛不掉就等于把"没有窗口样本"读成"精度为零"。
* **噪声带自取**（PLAN-N3-02 §0 的形状）：按 `ticks` 五等分取段均值，带＝相邻段最大跳幅，不引外部阈值。
* **§8.7 五项报告缺任一项 ⇒ 整档只算"跑了"不算"测了"**（rc=2 响亮拒判，但数值全部照出版）。
* **读数边界 `interpretation_limit` 必须原样在件里**（PLAN-N3-10 §3：缺字段 ⇒ 判读器 rc=2）。

用法：`--arm-a/--arm-b` 各指一支的 `progress.jsonl`，`--exit-a/--exit-b` 指对应收尾件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: PLAN-N3-02 §0 的对照锚（在库件 `a31_onpolicy` 末两段段均值），点名不许换成别的数。
BASELINE_LAST_SEGMENT = 0.594120
#: §2 的合取线：末段均值要高出基线较大者 0.02；J-N3a-保 用新列 `holdout_surprise_v2`（PLAN-N3-05）。
MIN_LAST_SEGMENT_MARGIN = 0.02
HOLDOUT_DEGRADATION_LIMIT = 0.02
SEGMENTS = 5
#: §8.7 要求的五项，逐项点名它的来源列；取不到就是"跑了不算测了"。
REQUIRED_REPORT_FIELDS = {
    "corpus_available": ("exit", "corpus_fingerprint"),
    "unique_episodes": ("exit", "unique_documents"),
    "actual_updates": ("line", "ticks"),
    "sequence_length": ("exit", "sequence_length"),
    "independent_test_coverage": ("line", "holdout_surprise_v2"),
}

INTERPRETATION_LIMIT = (
    "形状 B（等符号暴露、唯一篇数封顶）只允许回答“×16.8 容量在等符号暴露下是否把斜率抬出噪声带”，"
    "不允许回答“封顶在容量还是数据”——后者需要 PLAN-N3-10 §4 的 C 形状（等数据量变参数量）。"
)


def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / raw


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _exit_record(path: Path) -> dict[str, Any]:
    """收尾件是 `json.dumps(..., indent=2)` 的**多行对象**（`exit_record_path()` 的写法），
    而进度面是 JSONL ⇒ 两种形状都要认；按行读会把多行对象读成半截 JSON 而炸在 parse 期。
    """

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(s) for s in text.splitlines() if s.strip()]
        return rows[0] if rows else {}
    return payload if isinstance(payload, dict) else {}


def _segment_means(rows: list[dict[str, Any]], key: str) -> list[float]:
    """按 `ticks` 五等分取段均值（段内不足则合并到非空段，带因此可能变宽——照实出版段宽）。"""

    if not rows:
        return []
    edges = [rows[i * len(rows) // SEGMENTS]["ticks"] for i in range(SEGMENTS)] + [
        rows[-1]["ticks"] + 1
    ]
    means: list[float] = []
    for index in range(SEGMENTS):
        lo, hi = edges[index], edges[index + 1]
        bucket = [float(row[key]) for row in rows if lo <= row["ticks"] < hi]
        if bucket:
            means.append(sum(bucket) / len(bucket))
    return means


def _band(means: list[float]) -> float | None:
    if len(means) < 2:
        return None
    return max(abs(b - a) for a, b in zip(means[1:], means[:-1], strict=True))


def _sequence_length(exit_record: dict[str, Any], seq_path: Path | None) -> tuple[Any, str]:
    """§8.7 第四项的取值面：收尾件自述优先，其次只认守卫齐了的同源复算件。"""

    value = exit_record.get("sequence_length")
    if value is not None:
        return value, "exit_record"
    if seq_path is None:
        return None, "absent"
    if not seq_path.is_file():
        return None, "missing_sidecar"
    payload = json.loads(seq_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        #: 三条守卫不齐的复算件**不算在场**——它证不了"走的确实是当时那条流"。
        return None, f"sidecar_guard_failed:{payload.get('status')}"
    sidecar_value = payload.get("sequence_length")
    if sidecar_value is None:
        return None, "sidecar_empty"
    return sidecar_value, "recomputed_same_stream"


def judge_arm(
    progress: Path, exit_path: Path | None, seq_path: Path | None = None
) -> dict[str, Any]:
    """读一支臂的两张面并出 §8.7 在场性；`seq_path` 是**同源复算件**（PLAN-N3-13）。

    §8.7 第四项 `sequence_length` 优先认**收尾件自述**（㊵-545 之后的跑法都有）；旧产物没有这一列
    时，只接受 `status=="ok"` 的复算件（三条守卫：同指纹／同预算／重放篇数＝自述篇数），
    并把取值面写进 `sequence_length_source` 如实披露——**缺件或守卫不齐都不算在场**。
    """
    rows = _lines(progress)
    #: **DEBT-G63 的处置**：窗口为零的行不参与 acc／mean_surprise 统计。
    usable = [row for row in rows if int(row.get("window_ticks", 0)) > 0]
    dropped = len(rows) - len(usable)
    acc_means = _segment_means(usable, "online_accuracy")
    surprise_means = _segment_means(usable, "mean_surprise")
    holdout_first = usable[0]["holdout_surprise_v2"] if usable else None
    holdout_last = usable[-1]["holdout_surprise_v2"] if usable else None
    exit_record = _exit_record(exit_path) if exit_path and exit_path.is_file() else {}
    completeness = {
        name: (
            exit_record.get(column)
            if source == "exit"
            else (usable[-1].get(column) if usable else None)
        )
        is not None
        for name, (source, column) in REQUIRED_REPORT_FIELDS.items()
    }
    #: 第四项走"自述优先，其次齐守卫的复算件"这条取值面（PLAN-N3-13）。
    sequence_value, sequence_source = _sequence_length(exit_record, seq_path)
    completeness["sequence_length"] = sequence_value is not None
    slope = (acc_means[-1] - acc_means[0]) if len(acc_means) >= 2 else None
    band = _band(acc_means)
    return {
        "progress_path": str(progress),
        "progress_lines": len(rows),
        "lines_dropped_window_zero": dropped,
        "acc_segment_means": [round(m, 6) for m in acc_means],
        "acc_slope_first_to_last": None if slope is None else round(slope, 6),
        "acc_noise_band_adjacent_max": None if band is None else round(band, 6),
        "acc_last_segment_mean": None if not acc_means else round(acc_means[-1], 6),
        "mean_surprise_segment_means": [round(m, 6) for m in surprise_means],
        "holdout_v2_first": holdout_first,
        "holdout_v2_last": holdout_last,
        "holdout_v2_delta": (
            None
            if holdout_first is None or holdout_last is None
            else round(float(holdout_last) - float(holdout_first), 6)
        ),
        "exit_record": {
            key: exit_record.get(key)
            for key in (
                "exit_reason",
                "reached_budget",
                "budget_max_symbols",
                "ticks_at_exit",
                "unique_documents",
                "document_visits",
                "mean_revisits",
                "sequence_length",
                "elapsed_seconds",
                "checkpoint_sha256",
                "corpus_fingerprint",
            )
        },
        "section_8_7_completeness": completeness,
        "sequence_length": sequence_value,
        "sequence_length_source": sequence_source,
        "measurement_complete": all(completeness.values()),
    }


#: PLAN-N3-12 §1 第二合取项的门槛。与旧锚那条（`MIN_LAST_SEGMENT_MARGIN`）同值但**语义不同**：
#: 这一条比的是"同语料、同 seed、同预算的 ×2 对照臂"。两个数值分名分开钉——改一个不许静默挪走
#: 另一个（`shared-default-read-write-constant` 那条教训：同一个常量既当读默认又当写默认时测不开）。
CONTROL_MARGIN = 0.02


def adjudicate_prime(
    arm_a_judged: dict[str, Any],
    arm_a_raw: dict[str, Any],
    control_raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """把 J-N3a′（PLAN-N3-12 §1 三条）从"人算末段均值"变成机算——㊵-548 ⑦ 欠的那台仪器。

    三条逐字照升版件：甲臂斜率 > 自身噪声带 ∧ 甲臂末段均值 **>** 对照臂末段均值 + 0.02
    ∧ `holdout_surprise_v2` 不劣化。任何一侧 §8.7 五项不齐 ⇒ 判 `ran_not_measured`
    （PLAN-N3-02 §2 原话"跑了不算测了"）；对照臂缺席或与甲臂不同源 ⇒ `not_judged` 并点名原因，
    **不许**退回旧锚 `0.594120`（那条正是 DEBT-G64 点名的跨链/跨语料值）。
    """

    if control_raw is None:
        return {
            "J_N3a_prime": "not_judged",
            "reason": "对照臂缺席 ⇒ PLAN-N3-12 §1 的第二合取项没有值",
            "forbid_fallback_to_old_anchor": True,
        }
    arm_last = arm_a_raw.get("acc_last_segment_mean")
    control_last = control_raw.get("acc_last_segment_mean")
    sides_complete = bool(arm_a_raw.get("measurement_complete")) and bool(
        control_raw.get("measurement_complete")
    )
    out: dict[str, Any] = {
        "margin": CONTROL_MARGIN,
        "arm_a_last_segment_mean": arm_last,
        "control_last_segment_mean": control_last,
        "required_strictly_greater_than": (
            None if control_last is None else round(float(control_last) + CONTROL_MARGIN, 6)
        ),
        "slope_gt_own_band": arm_a_judged.get("slope_gt_own_band"),
        "holdout_v2_within_limit": arm_a_judged.get("holdout_v2_within_limit"),
        "gt_control_plus_margin": (
            None
            if arm_last is None or control_last is None
            else bool(float(arm_last) > float(control_last) + CONTROL_MARGIN)
        ),
        "both_sides_measurement_complete": sides_complete,
    }
    if not sides_complete:
        out["J_N3a_prime"] = "ran_not_measured"
        return out
    if out["gt_control_plus_margin"] is None:
        out["J_N3a_prime"] = "not_judged"
        return out
    out["J_N3a_prime"] = (
        "holds"
        if out["slope_gt_own_band"]
        and out["gt_control_plus_margin"]
        and out["holdout_v2_within_limit"]
        else "not_holds"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N3 甲 ×10 单点探针判读（只读、零跑）")
    parser.add_argument("--arm-a", required=True, help="甲臂 progress.jsonl")
    parser.add_argument("--arm-b", help="乙臂 progress.jsonl（缺 ⇒ 否证对照记 unverified）")
    parser.add_argument("--exit-a", required=True, help="甲臂 progress_exit.json")
    parser.add_argument("--exit-b", help="乙臂 progress_exit.json")
    parser.add_argument(
        "--seq-a",
        help="甲臂 §8.7 第四项的同源复算件（PLAN-N3-13；收尾件已自述这一列时不必给）",
    )
    parser.add_argument(
        "--seq-b",
        help="乙臂的同源复算件（守卫不齐的件不算在场）",
    )
    parser.add_argument(
        "--control-progress",
        help="×2 同语料对照臂 progress.jsonl（PLAN-N3-12 §1 新锚的读数面）",
    )
    parser.add_argument(
        "--control-exit",
        help="对照臂 progress_exit.json（与 --control-progress 配对给）",
    )
    parser.add_argument(
        "--control-seq",
        help="对照臂的同源复算件（它已自述 sequence_length 那一列时不必给）",
    )
    parser.add_argument(
        "--baseline-last-segment",
        type=float,
        default=BASELINE_LAST_SEGMENT,
        help="§0 的在库对照锚（默认 0.594120＝a31_onpolicy 末两段段均值）",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    for raw in filter(None, (args.arm_a, args.arm_b, args.exit_a, args.exit_b)):
        if not _resolve(str(raw)).is_file():
            print("REJECT missing_face", str(raw).encode("ascii", "replace").decode("ascii"))
            return 2
    #: 复算件给了就必须存在——"给了路径却读不到"不许静默降级成"这一项不在场"。
    for raw in filter(None, (args.seq_a, args.seq_b)):
        if not _resolve(str(raw)).is_file():
            print("REJECT missing_sidecar", str(raw).encode("ascii", "replace").decode("ascii"))
            return 2

    arms: dict[str, Any] = {
        "arm_A": judge_arm(
            _resolve(args.arm_a),
            _resolve(args.exit_a),
            _resolve(args.seq_a) if args.seq_a else None,
        ),
    }
    rc = 0
    if args.arm_b:
        arms["arm_B"] = judge_arm(
            _resolve(args.arm_b),
            _resolve(args.exit_b),
            _resolve(args.seq_b) if args.seq_b else None,
        )
    else:
        arms["arm_B"] = {
            "status": "absent",
            "verdict_note": "PLAN-N3-02 §2 的否证对照记 unverified",
        }

    payload: dict[str, Any] = {
        "format": "taiji-n3a-scaling-probe-adjudication-v1",
        "prereg": "plans/reference/PLAN-N3-02_scaling_probe_prereg_20261007.md §2/§4quater ＋ "
        "PLAN-N3-10 §2/§3（形状 B 与读数边界）",
        "interpretation_limit": INTERPRETATION_LIMIT,
        "baseline_last_segment": args.baseline_last_segment,
        "arms": arms,
    }

    judged: dict[str, Any] = {}
    for name, arm in arms.items():
        if "acc_slope_first_to_last" not in arm or arm["acc_noise_band_adjacent_max"] is None:
            judged[name] = {"J_N3a": "not_judged"}
            continue
        slope = arm["acc_slope_first_to_last"]
        band = arm["acc_noise_band_adjacent_max"]
        last_mean = arm["acc_last_segment_mean"]
        judged[name] = {
            "slope_gt_own_band": bool(slope is not None and slope > band),
            "last_mean_gt_baseline_plus_margin": bool(
                last_mean is not None
                and last_mean > args.baseline_last_segment + MIN_LAST_SEGMENT_MARGIN
            ),
            "holdout_v2_within_limit": bool(
                arm["holdout_v2_delta"] is not None
                and float(arm["holdout_v2_delta"]) <= HOLDOUT_DEGRADATION_LIMIT
            ),
            "measurement_complete": arm["measurement_complete"],
        }
        judged[name]["J_N3a"] = (
            "holds"
            if judged[name]["slope_gt_own_band"]
            and judged[name]["last_mean_gt_baseline_plus_margin"]
            and judged[name]["holdout_v2_within_limit"]
            else (
                "degradation_blocks" if not judged[name]["holdout_v2_within_limit"] else "not_holds"
            )
        )
        if not arm["measurement_complete"]:
            #: §2 原话：缺任一项 ⇒ 这档只算"跑了"，不算"测了"。
            judged[name]["J_N3a"] = "ran_not_measured"
            rc = 2

    payload["judgement"] = judged
    payload["single_variable_check"] = {
        "both_arms_present": bool(args.arm_b),
        "unique_documents_differ": (
            None
            if "arm_B" not in arms or "exit_record" not in arms["arm_B"]
            else arms["arm_A"]["exit_record"]["unique_documents"]
            != arms["arm_B"]["exit_record"]["unique_documents"]
        ),
        "mean_revisits_published": (
            None
            if "arm_B" not in arms or "exit_record" not in arms["arm_B"]
            else [
                arms["arm_A"]["exit_record"]["mean_revisits"],
                arms["arm_B"]["exit_record"]["mean_revisits"],
            ]
        ),
        "instrument_broken_if_revisits_equal": (
            None
            if "arm_B" not in arms or "exit_record" not in arms["arm_B"]
            else arms["arm_A"]["exit_record"]["mean_revisits"]
            == arms["arm_B"]["exit_record"]["mean_revisits"]
        ),
        "ticks_equal": (
            None
            if "arm_B" not in arms or "exit_record" not in arms["arm_B"]
            else arms["arm_A"]["exit_record"]["ticks_at_exit"]
            == arms["arm_B"]["exit_record"]["ticks_at_exit"]
        ),
    }
    payload["single_variable_check"]["corpus_fingerprints"] = (
        None
        if "exit_record" not in arms.get("arm_B", {})
        else [
            arms["arm_A"]["exit_record"]["corpus_fingerprint"],
            arms["arm_B"]["exit_record"]["corpus_fingerprint"],
        ]
    )
    payload["single_variable_check"]["corpus_fingerprints_equal"] = (
        None
        if "exit_record" not in arms.get("arm_B", {})
        else arms["arm_A"]["exit_record"]["corpus_fingerprint"]
        == arms["arm_B"]["exit_record"]["corpus_fingerprint"]
    )
    #: **G-N3f-3／G-N3g-1 的同源必检**：两臂吃的不是同一份材料，整件对照作废——
    #: 这条不许由人读命令比对，必须由件里出版。
    if payload["single_variable_check"]["corpus_fingerprints_equal"] is False:
        payload["verdict"] = "arms_not_same_source"
        for name in ("arm_A", "arm_B"):
            if name in payload["judgement"]:
                payload["judgement"][name]["J_N3a"] = "not_judged"
        rc = 2
    #: **J-N3a′ 的新锚**（PLAN-N3-12 §1）：对照臂给就机械判，不给就 `not_judged`——不许退回旧锚代答。
    control_raw: dict[str, Any] | None = None
    if args.control_progress:
        if not args.control_exit:
            print("REJECT control_arm_incomplete 给了 --control-progress 必须配 --control-exit")
            return 2
        control_path = _resolve(args.control_progress)
        if not control_path.is_file():
            print(
                "REJECT missing_control_face",
                str(control_path).encode("ascii", "replace").decode("ascii"),
            )
            return 2
        control_raw = judge_arm(
            control_path,
            _resolve(args.control_exit),
            _resolve(args.control_seq) if args.control_seq else None,
        )
        payload["control_arm"] = control_raw

    prime = adjudicate_prime(judged.get("arm_A", {}), arms["arm_A"], control_raw)
    if control_raw is not None:
        prime["control_corpus_fingerprint_matches_arm_a"] = bool(
            arms["arm_A"]["exit_record"].get("corpus_fingerprint")
            == control_raw["exit_record"].get("corpus_fingerprint")
        )
        if prime["control_corpus_fingerprint_matches_arm_a"] is False:
            #: G-N3g-1 的新锚版本：不同源的对照臂没有点亮 `J-N3a′` 的资格。
            prime["J_N3a_prime"] = "not_judged"
            prime["reason"] = "对照臂与甲臂不同源（G-N3g-1）⇒ 新锚作废"
    if prime["J_N3a_prime"] == "ran_not_measured":
        rc = 2
    payload["J_N3a_prime"] = prime

    target = _resolve(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("RC", rc, "JUDGEMENT", json.dumps(judged, ensure_ascii=False, sort_keys=True)[:220])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

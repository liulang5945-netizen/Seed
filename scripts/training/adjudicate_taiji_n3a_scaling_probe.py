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


def judge_arm(progress: Path, exit_path: Path | None) -> dict[str, Any]:
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
                "elapsed_seconds",
                "checkpoint_sha256",
                "corpus_fingerprint",
            )
        },
        "section_8_7_completeness": completeness,
        "measurement_complete": all(completeness.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N3 甲 ×10 单点探针判读（只读、零跑）")
    parser.add_argument("--arm-a", required=True, help="甲臂 progress.jsonl")
    parser.add_argument("--arm-b", help="乙臂 progress.jsonl（缺 ⇒ 否证对照记 unverified）")
    parser.add_argument("--exit-a", required=True, help="甲臂 progress_exit.json")
    parser.add_argument("--exit-b", help="乙臂 progress_exit.json")
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

    arms: dict[str, Any] = {
        "arm_A": judge_arm(_resolve(args.arm_a), _resolve(args.exit_a)),
    }
    rc = 0
    if args.arm_b:
        arms["arm_B"] = judge_arm(_resolve(args.arm_b), _resolve(args.exit_b))
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
    target = _resolve(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("RC", rc, "JUDGEMENT", json.dumps(judged, ensure_ascii=False, sort_keys=True)[:220])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

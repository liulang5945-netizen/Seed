"""N3 甲标定读数器：把两臂标定的**实测**耗时与磁盘取成一份入库件，并外推出正式档的预算。

为什么要有这台仪器（缺陷本体）：09 §2 N3 甲要"参数×10 单点"，但 `--scale` 乘的是尺寸不是参数量
（region 与 fan_in 同乘 ⇒ 边数近似二次），在库 progress 的两个吞吐锚点又相差 190 倍
（首行 45,170 tick/s 对末段 238.6 tick/s）⇒ **拿在库数外推就是编数**。
owner 2026-10-08 裁"标定跑批，机器空出来后各跑一支"，本件只读标定产物做外推，不跑任何训练。

三条硬规矩：
* 缺档或缺末行 ⇒ rc=2 响亮拒绝，并把缺哪一枚点名（"没取到"不等于"很快"）；
* 外推只认**同一装配**的实测吞吐（scale 10 用 scale 10 的 tick/s），跨档搬吞吐即拒；
* 每个外推值都自带出处（哪一枚文件、第几行、什么命令），件里另列"未测到的东西"。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 标定跑的冻结命令（跑在**无并发**的机器上；owner 2026-10-08 裁"机器空出来后各跑一支"）。
CALIBRATION_COMMANDS = {
    "scale10": (
        "python scripts/training/train_seed_corpus.py --scale 10 --seed 20260822 --epochs 1 "
        "--max-symbols 4000 --progress-every 200000 "
        "--checkpoint output/seed_n3a_calib_scale10.pt "
        "--progress reports/taiji_n3a_calib_scale10_progress.jsonl"
    ),
    "budget770": (
        "python scripts/training/train_seed_corpus.py --parameter-budget 7699510 --seed 20260822 "
        "--epochs 1 --max-symbols 4000 --progress-every 200000 "
        "--checkpoint output/seed_n3a_calib_budget770.pt "
        "--progress reports/taiji_n3a_calib_budget770_progress.jsonl"
    ),
}

#: 2M 基线那档（`--scale 2`）的末段吞吐，取自入库件 `output/a31_chunked_self/progress.jsonl`
#: 的两行差：(4,000,101 − 2,010,316) ticks 对 (8383.000766 − 44.522187) 秒。
BASELINE_SCALE2_TAIL_TICKS_PER_SECOND = 238.6

#: 正式档的符号数上限候选（owner 批的是"由 owner 给上限"，这里只按实测吞吐把它们换算成时长）。
CANDIDATE_BUDGET_TICKS = (250_000, 1_000_000, 4_000_000)


def _read_last_row(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[-1] if rows else None


def _arm(arm: str, progress: Path, checkpoint: Path) -> dict[str, Any]:
    row = _read_last_row(progress)
    if row is None:
        raise RuntimeError(f"标定臂 {arm} 没有可解析的 progress 末行：{progress}")
    elapsed = float(row["elapsed_seconds"])
    ticks = int(row["ticks"])
    reached = bool(row.get("reached_budget"))
    if not reached or elapsed <= 0:
        raise RuntimeError(f"标定臂 {arm} 自述未达预算或耗时非正：reached_budget={reached}")
    return {
        "arm": arm,
        "command": CALIBRATION_COMMANDS[arm],
        "progress_file": str(progress.relative_to(PROJECT_ROOT)),
        "checkpoint_file": str(checkpoint.relative_to(PROJECT_ROOT)),
        "ticks": ticks,
        "elapsed_seconds": elapsed,
        "ticks_per_second": round(ticks / elapsed, 3),
        "parameters_reported": None,
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "online_accuracy_at_calibration": row.get("online_accuracy"),
        "corpus_fingerprint": row.get("corpus_fingerprint"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N3 甲标定读数器（零训练、只读标定产物）")
    parser.add_argument("--out-report", default="reports/taiji_n3a_calibration_cost_20261008.json")
    args = parser.parse_args(argv)

    arms: dict[str, Any] = {}
    errors: list[str] = []
    for arm in CALIBRATION_COMMANDS:
        progress = PROJECT_ROOT / f"reports/taiji_n3a_calib_{arm}_progress.jsonl"
        checkpoint = PROJECT_ROOT / f"output/seed_n3a_calib_{arm}.pt"
        try:
            arms[arm] = _arm(arm, progress, checkpoint)
        except Exception as exc:  # noqa: BLE001 - 缺档是响亮拒绝，不是零
            errors.append(f"{arm}: {type(exc).__name__}: {exc}")

    scale10 = arms.get("scale10")
    payload: dict[str, Any] = {
        "format": "taiji-n3a-calibration-cost-v1",
        "prereg": "plans/reference/PLAN-N3-02_scaling_probe_prereg_20261007.md#4bis",
        "machine_load_note": "两臂**依次**跑在只有它们自己的机器上（owner 裁"
        "机器空出来后各跑一支"
        "）⇒ 这里的 tick/s 可当标定用；并发取到的耗时按纪律不得当标定。",
        "arms": arms,
        "baseline_scale2_tail_ticks_per_second": BASELINE_SCALE2_TAIL_TICKS_PER_SECOND,
        "errors": errors,
    }
    if scale10:
        ratio = BASELINE_SCALE2_TAIL_TICKS_PER_SECOND / scale10["ticks_per_second"]
        payload["scale10_vs_scale2"] = {
            "throughput_ratio_slowdown": round(ratio, 3),
            "note": "scale 10 相对 2M 基线那档慢约这么多倍；参数量比是 ×16.8（不是 ×10），"
            "所以吞吐不是按参数量线性缩放——这条比值就是「容量代价」的实测形状。",
        }
        per_arm: dict[str, Any] = {}
        for ticks in CANDIDATE_BUDGET_TICKS:
            hours = ticks / scale10["ticks_per_second"] / 3600.0
            per_arm[f"{ticks}_ticks"] = {
                "single_arm_wallclock_hours": round(hours, 2),
                "two_arms_wallclock_hours": round(2 * hours, 2),
                "checkpoints_bytes_if_one_save_each": 2 * int(scale10["checkpoint_bytes"] or 0),
            }
        payload["extrapolation_from_measured_scale10"] = per_arm
    payload["not_measured_here"] = [
        "holdout/泛化侧随规模的走向（标定只跑 4000 符号，够不着平台判断）",
        "checkpoint-every 触发的多次存盘与 history 保留的真实磁盘放大（这里只测了一枚档的尺寸）",
        "断点续训（--resume）在 scale 10 上的耗时占比",
    ]

    rc = 2 if errors or not arms else 0
    payload["rc"] = rc
    out = PROJECT_ROOT / args.out_report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "rc": rc,
        "arms": {
            name: {
                "ticks_per_second": row["ticks_per_second"],
                "checkpoint_bytes": row["checkpoint_bytes"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            for name, row in arms.items()
        },
        "errors": errors,
    }
    #: Windows 控制台是 GBK ⇒ 只打 ASCII 转义，中文留给件里（否则 rc 会被编码崩溃伪装成崩溃）。
    print(json.dumps(summary, ensure_ascii=True))
    print(f"out -> {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

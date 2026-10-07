"""N3 甲线的第 0 点锚：在库两条 2M run 的 `progress.jsonl` 复算"平台"这件事本身（零训练、零新跑）。

PLAN-N3-02 花参数×10 的算力之前，先要确认要解除的那块平台真实存在。本脚本只读
`output/a31_chunked_self/progress.jsonl` 与 `output/a31_onpolicy_20261003/progress.jsonl`，
把每条 run 按 **ticks 五等分**取 `online_accuracy` 与 `holdout_surprise` 的段均值，并给出
末段−首段、末段−次末段两条斜率——"平台"的判据形态＝末段−首段的 acc 斜率落在噪声带内，
而噪声带用该 run 自身的段间最大跳幅来定（不引外部阈值、不猜）。

口径纪律（09 §2 N3 报告纪律 §8.7 的可用部分）：这里只报**行数、ticks 跨度、段均值与斜率**，
不报"学到多少"——语料可用量、唯一 episode、独立测试覆盖不在本件的分母里，缺的项在 PLAN-N3-02 §0 点名。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 在库两条 run 的进度件（路径写死，避免"面不唯一"；文件缺失即响亮失败）。
RUNS = {
    "corpus_a31_chunked_self": "output/a31_chunked_self/progress.jsonl",
    "onpolicy_a31_onpolicy_20261003": "output/a31_onpolicy_20261003/progress.jsonl",
}

QUINTILES = 5


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        for field in ("ticks", "online_accuracy", "holdout_surprise"):
            if field not in record:
                raise ValueError(f"{path.name}: 进度行缺字段 {field} ⇒ 不判")
        rows.append(record)
    if len(rows) < QUINTILES * 2:
        raise ValueError(f"{path.name}: 进度行 {len(rows)} 条不足以分 {QUINTILES} 段 ⇒ 不判")
    rows.sort(key=lambda record: int(record["ticks"]))
    return rows


def _segment_means(rows: list[dict[str, Any]], key: str) -> list[float]:
    ticks = [int(record["ticks"]) for record in rows]
    start, end = ticks[0], ticks[-1]
    span = end - start
    if span <= 0:
        raise ValueError("ticks 跨度为零 ⇒ 无法分段")
    means: list[float] = []
    for index in range(QUINTILES):
        low = start + span * index / QUINTILES
        high = start + span * (index + 1) / QUINTILES
        window = [
            float(record[key])
            for tick, record in zip(ticks, rows, strict=True)
            if low <= tick < high or (index == QUINTILES - 1 and tick == high)
        ]
        if not window:
            raise ValueError(f"段 {index} 为空（{low:.0f}-{high:.0f}）⇒ 分母破了，不判")
        means.append(sum(window) / len(window))
    return means


def summarize(name: str, path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    ticks = [int(record["ticks"]) for record in rows]
    acc = _segment_means(rows, "online_accuracy")
    hold = _segment_means(rows, "holdout_surprise")
    jumps = [abs(acc[i + 1] - acc[i]) for i in range(len(acc) - 1)]
    last = rows[-1]
    return {
        "run": name,
        "progress_file": str(path),
        "rows": len(rows),
        "ticks_first": ticks[0],
        "ticks_last": ticks[-1],
        "line_first_online_accuracy": round(float(rows[0]["online_accuracy"]), 6),
        "line_last_online_accuracy": round(float(last["online_accuracy"]), 6),
        "line_first_holdout_surprise": round(float(rows[0]["holdout_surprise"]), 6),
        "line_last_holdout_surprise": round(float(last["holdout_surprise"]), 6),
        "acc_quintile_means": [round(value, 6) for value in acc],
        "holdout_quintile_means": [round(value, 6) for value in hold],
        "acc_slope_last_minus_first": round(acc[-1] - acc[0], 6),
        "acc_slope_last_minus_prev": round(acc[-1] - acc[-2], 6),
        "holdout_slope_last_minus_first": round(hold[-1] - hold[0], 6),
        "max_adjacent_quintile_acc_jump": round(max(jumps), 6),
        "exit_reason": last.get("exit_reason"),
        "reached_budget": last.get("reached_budget"),
        "budget_max_symbols": last.get("budget_max_symbols"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-report", default=None, help="可选：把 JSON 读数写到该路径")
    args = parser.parse_args(argv)

    results: list[dict[str, Any]] = []
    rc = 0
    for name, relative in RUNS.items():
        path = PROJECT_ROOT / relative
        try:
            results.append(summarize(name, path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            results.append({"run": name, "status": "refused", "error": str(error)[:300]})
            rc = 2
    payload: dict[str, Any] = {
        "format": "taiji-n3-plateau-baseline-v1",
        "results": results,
        "rc": rc,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_report:
        out = Path(args.out_report)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    print(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())

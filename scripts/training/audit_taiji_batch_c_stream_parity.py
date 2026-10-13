"""批次 C 两臂进度流的**同序同界**仪（`PLAN-N5-21` 的 J-N5t-1 载体；判据冻在取数之前，㊵-769）。

要答的只有一句：两臂是不是真的吃同一段流。判据＝两臂 `progress.jsonl` 的行数相等，且逐行按行序对齐时
`symbols_consumed` 与 `absolute_tick` 同值；任一臂内 `session_id` 只许一枚值（多枚＝跨进程追加过，
历史上"按相邻行差算速率会算出负数"就是这么来的）。

`window_symbols`／`session_symbols` 与一切计时量**只出版、不判**：源码里
`window_symbols = seen - mark[0]`，而 `seen` 只在 `prior_prediction is not None` 时才涨
（`scripts/training/train_taiji_r2_readout_retrain.py`），所以它是**模型依赖量**——拿它当混杂去判，
得到的是一支必然为假的判据。这一面同时给出逐行的 `cumulative_mean_surprise`／
`cumulative_online_accuracy` 并排曲线：㊵-758 那两个点在这里升格成整条曲线，代价为零（数据已在盘上）。

同流≠同实现：浮点累加次序与线程不在本件覆盖范围，`reading_limit` 里把这句一起出版。
只读、零算力、不动 `taiji/`；缺件一律 rc=2 点名（"文件不在"不许被读成"差为空"）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FORMAT = "taiji-batch-c-stream-parity-v1"
#: 契约册（`test_n5_34`）按文件路径加载本件、也被起成真子进程，脚本自己那层目录先上 `sys.path`，
#: 统一落盘助手才解析得到（同㊵-790／802 那条例外约定）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

PREREG = "plans/reference/PLAN-N5-21_batch_c_attribution_prereg_20261013.md \u00a72"
CRITERION = "J-N5t-1"
BOUNDARY_KEYS = ("symbols_consumed", "absolute_tick")
METRIC_KEYS = ("cumulative_mean_surprise", "cumulative_online_accuracy", "window_symbols")
NAME_CAP = 8

READING_LIMIT = (
    "本件只证「两臂吃同一段流」与「逐行边界同值」这一件事：同流不等于同实现"
    "（浮点累加次序、线程数、设备都不在本件的覆盖范围）。`window_face` 与 `elapsed_*` 是出版面不是判据面"
    "——`window_symbols` 在源码里就是模型依赖量，两臂不等是**结果**而不是混杂。曲线取的是两臂各自的"
    "`cumulative_*` 累计口径，不是逐窗读数，所以曲线上相邻两行的差不等于那一窗的差。"
)


def load_stream(path: Path) -> list[dict[str, Any]]:
    """按行读进度流；空面与坏行都响亮失败，不许出版一个漂亮的 0。"""

    if not path.is_file():
        raise FileNotFoundError(f"progress stream missing: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:  # noqa: BLE001 - 坏行必须点名到行号
            raise ValueError(f"{path}:{number} is not JSON: {error}") from error
    if not rows:
        raise ValueError(f"progress stream is empty: {path}")
    return rows


def _sessions(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("session_id")) for row in rows})


def _mismatch_names(
    treated: list[dict[str, Any]], control: list[dict[str, Any]]
) -> tuple[list[str], list[int]]:
    """逐行比边界键，返回（哪些键不齐、在第几行不齐）。行序对齐是判据本身，不做模糊匹配。"""

    mismatched: set[str] = set()
    rows_bad: list[int] = []
    for index in range(min(len(treated), len(control))):
        bad_here = [
            key for key in BOUNDARY_KEYS if treated[index].get(key) != control[index].get(key)
        ]
        if bad_here:
            rows_bad.append(index)
            mismatched.update(bad_here)
    return sorted(mismatched), rows_bad


def _window_face(treated: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    """两臂的模型依赖计量面：只并排出版（含逐行差的极值），不参与 verdict。"""

    deltas: list[int] = []
    unequal_rows: list[int] = []
    for index in range(min(len(treated), len(control))):
        left = treated[index].get("window_symbols")
        right = control[index].get("window_symbols")
        if isinstance(left, int) and isinstance(right, int):
            deltas.append(left - right)
            if left != right:
                unequal_rows.append(index)
    return {
        "rows_compared": len(deltas),
        "unequal_rows": len(unequal_rows),
        "unequal_rows_listed": unequal_rows[:NAME_CAP],
        "treated_minus_control": {"min": min(deltas), "max": max(deltas)} if deltas else None,
        "note": "\u6a21\u578b\u4f9d\u8d56\u91cf\uff1a\u53d6\u51b3\u4e8e\u88ab\u6253\u5206\u6837\u672c\u6570\uff0c\u4e0d\u662f\u6d41\u7684\u8fb9\u754c",
    }


def _metric_curve(
    treated: list[dict[str, Any]], control: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """逐对齐行并排出版累计口径的两臂读数与差。"""

    curve: list[dict[str, Any]] = []
    for index in range(min(len(treated), len(control))):
        row: dict[str, Any] = {
            "row": index,
            "symbols_consumed": treated[index].get("symbols_consumed"),
        }
        for key in METRIC_KEYS:
            left = treated[index].get(key)
            right = control[index].get(key)
            row[f"treated_{key}"] = left
            row[f"control_{key}"] = right
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                row[f"delta_{key}"] = round(left - right, 12)
        curve.append(row)
    return curve


def compare(treated: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    mismatched, rows_bad = _mismatch_names(treated, control)
    treated_sessions = _sessions(treated)
    control_sessions = _sessions(control)
    verdicts = {
        "row_counts_equal": len(treated) == len(control),
        "boundary_keys_identical_on_aligned_rows": not mismatched
        and not rows_bad
        and len(treated) == len(control),
        "single_session_per_arm": len(treated_sessions) == 1 and len(control_sessions) == 1,
    }
    return {
        "rows": {"treated": len(treated), "control": len(control)},
        "boundary_keys": list(BOUNDARY_KEYS),
        "mismatched_boundary_keys": mismatched,
        "first_mismatch_row": rows_bad[0] if rows_bad else None,
        "mismatch_row_count": len(rows_bad),
        "mismatch_rows_listed": rows_bad[:NAME_CAP],
        "sessions": {"treated": treated_sessions, "control": control_sessions},
        "window_face": _window_face(treated, control),
        "metric_curve": _metric_curve(treated, control),
        "verdicts": verdicts,
        "failing_verdicts": sorted(name for name, ok in verdicts.items() if not ok),
        "verdict": "holds" if all(verdicts.values()) else "rejected",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批次 C 两臂进度流的同序同界取证（J-N5t-1）")
    parser.add_argument("--treated-progress", required=True, help="治疗臂 progress.jsonl")
    parser.add_argument("--control-progress", required=True, help="对照臂 progress.jsonl")
    parser.add_argument(
        "--out-report", required=True, help="落盘路径（件已存在则拒覆盖 rc=2，除非 --overwrite）"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    treated_path = Path(args.treated_progress)
    control_path = Path(args.control_progress)
    if not treated_path.is_absolute():
        treated_path = PROJECT_ROOT / treated_path
    if not control_path.is_absolute():
        control_path = PROJECT_ROOT / control_path
    try:
        treated = load_stream(treated_path)
        control = load_stream(control_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"REJECT {error}", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "prereg": PREREG,
        "criterion": CRITERION,
        "faces": {
            "treated": treated_path.resolve().as_posix(),
            "control": control_path.resolve().as_posix(),
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reading_limit": READING_LIMIT,
        **compare(treated, control),
    }
    report["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 3)

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)

    summary = {
        "verdict": report["verdict"],
        "failing_verdicts": report["failing_verdicts"],
        "rows": report["rows"],
        "mismatched_boundary_keys": report["mismatched_boundary_keys"],
        "first_mismatch_row": report["first_mismatch_row"],
        "curve_rows": len(report["metric_curve"]),
        "window_unequal_rows": report["window_face"]["unequal_rows"],
        "out": out.as_posix(),
    }
    sys.stdout.buffer.write(json.dumps(summary, ensure_ascii=True).encode("ascii") + b"\n")
    return 0 if report["verdict"] == "holds" else 1


if __name__ == "__main__":
    raise SystemExit(main())

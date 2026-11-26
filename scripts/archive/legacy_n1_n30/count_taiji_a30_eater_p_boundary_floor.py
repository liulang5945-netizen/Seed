"""Count the A-2 cell of §102's frozen mechanism criterion.

§102 froze A as two conjuncts, and the second one — `拖写代中 p_boundary_max ≥ 0.10 的代数` moving from
the recorded 0 to ≥ 20 — is the only frozen criterion on this line that **no in-repo instrument can
evaluate**: the four existing pure-Python ones (counterfactual / bound / pairing / judge) read other
columns. §101's "现值 0" was an ad-hoc read, so today the number that decides A cannot be re-taken by
anything checked in. This closes that gap.

Design rule taken from the judge (`judge_taiji_a30_peak_repeat_run.py`): the counter must not invent a
second definition of "拖写代". It classifies each generation row itself, then refuses to publish unless
its own tally equals the instrument's declared `generations_eating_full_budget`. Two independent
readings that disagree stop the run rather than producing a plausible number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: §102 froze both of these; they are not knobs for the reader.
FROZEN_FLOOR = 0.10
FROZEN_LINE = 20


def _declared_eaters(report: dict[str, Any]) -> int | None:
    guard = report.get("instrument_guard")
    if not isinstance(guard, dict):
        return None
    value = guard.get("generations_eating_full_budget")
    return int(value) if isinstance(value, (int, float)) else None


def _declared_never_lf(report: dict[str, Any]) -> int | None:
    summary = report.get("peak_run_summary_v37")
    if not isinstance(summary, dict):
        return None
    value = summary.get("eaters_never_lf_n")
    return int(value) if isinstance(value, (int, float)) else None


def _lf_step_count(generation: dict[str, Any], item_id: str) -> int:
    """Read this generation's own LF emission count; absence is a refusal, not a zero."""

    trace = generation.get("lf_trace_v29")
    if not isinstance(trace, dict) or "lf_step_count" not in trace:
        raise ValueError(
            f"生成行缺 lf_trace_v29.lf_step_count（题 {item_id}）⇒ 无法定义「从不发 LF」，不判"
        )
    return int(trace["lf_step_count"])


def count_eaters(report: dict[str, Any], *, floor: float) -> dict[str, Any]:
    """Walk every generation row, classify eaters by the report's own budget, and count those whose
    peak boundary probability reaches `floor`.

    Two counts are reported because §102 froze A-2 on a **narrower population than "all eaters"**: the
    group that never emits LF (its measured ceiling is 0.0478, so 0.10 is a real test there, while
    LF-emitting eaters already reach ≥0.10 on today's shipped base). `meets_frozen_line` is the
    never-LF one; the all-eater number is disclosed so nobody reads it as the frozen quantity.
    """

    max_length = report.get("max_length")
    if not isinstance(max_length, (int, float)):
        raise ValueError("件里没有 max_length ⇒ 无法定义「吃满预算」，不判")

    rows_seen = 0
    eaters = 0
    never_lf_eaters = 0
    above_floor_all = 0
    above_floor_never_lf = 0
    missing_column = 0
    for item in report.get("per_item", []):
        for generation in item.get("endstep_probe_v22", []):
            rows_seen += 1
            steps = generation.get("generation_steps")
            if not isinstance(steps, (int, float)):
                raise ValueError(f"生成行缺 generation_steps（题 {item.get('id')}）⇒ 不判")
            if steps < max_length:
                continue
            eaters += 1
            never_lf = _lf_step_count(generation, str(item.get("id"))) == 0
            never_lf_eaters += int(never_lf)
            if "p_boundary_max" not in generation:
                missing_column += 1
                continue
            reaches = float(generation["p_boundary_max"]) >= floor
            above_floor_all += int(reaches)
            above_floor_never_lf += int(reaches and never_lf)

    declared = _declared_eaters(report)
    declared_never_lf = _declared_never_lf(report)
    return {
        "generation_rows_seen": rows_seen,
        "eaters_counted": eaters,
        "eaters_declared_by_instrument": declared,
        "coherent_with_declaration": declared is not None and declared == eaters,
        "never_lf_eaters_counted": never_lf_eaters,
        "never_lf_eaters_declared": declared_never_lf,
        "coherent_with_never_lf_declaration": (
            declared_never_lf is not None and declared_never_lf == never_lf_eaters
        ),
        "rows_without_p_boundary_max_column": missing_column,
        "eaters_above_floor": above_floor_never_lf,
        "eaters_above_floor_all_eaters": above_floor_all,
        "share_of_never_lf_eaters": (
            round(above_floor_never_lf / never_lf_eaters, 6) if never_lf_eaters else None
        ),
    }


def read_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    report = json.loads(text)
    if "per_item" not in report:
        raise ValueError(f"{path.name} 没有 per_item ⇒ 不是 v3x 停止面件，不判")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="v3x 停止面件路径（可重复；每枚各自出一份读数）",
    )
    parser.add_argument(
        "--expect-items-sha", default=None, help="题面指纹：给了就必须逐枚对上，对不上按不可判处理"
    )
    parser.add_argument("--out-report", default=None, help="落盘路径；不给只打到 stdout")
    args = parser.parse_args(argv)

    results: list[dict[str, Any]] = []
    rc = 0
    for raw in args.report:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        entry: dict[str, Any] = {
            "report": path.name,
            "frozen_floor": FROZEN_FLOOR,
            "frozen_line": FROZEN_LINE,
        }
        try:
            report = read_report(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            entry["status"] = "unreadable"
            entry["error"] = str(error)[:200]
            results.append(entry)
            rc = 2
            continue

        entry["format"] = report.get("format")
        entry["checkpoint_sha256"] = report.get("checkpoint_sha256")
        entry["items_sha256"] = report.get("items_sha256")
        entry["base_sha256_unchanged"] = (report.get("instrument_guard") or {}).get(
            "base_sha256_unchanged"
        )
        if args.expect_items_sha and report.get("items_sha256") != args.expect_items_sha:
            entry["status"] = "items_sha_mismatch"
            results.append(entry)
            rc = 2
            continue

        try:
            entry.update(count_eaters(report, floor=FROZEN_FLOOR))
        except ValueError as error:
            entry["status"] = "refused"
            entry["error"] = str(error)[:200]
            results.append(entry)
            rc = 2
            continue

        # fail-closed: a moving artifact, a disagreeing tally, or a row missing the column is not a zero
        if not entry["coherent_with_declaration"]:
            entry["status"] = "incoherent_eater_count"
            rc = 2
        elif not entry["coherent_with_never_lf_declaration"]:
            entry["status"] = "incoherent_never_lf_count"
            rc = 2
        elif entry["rows_without_p_boundary_max_column"]:
            entry["status"] = "column_missing_in_rows"
            rc = 2
        elif entry["base_sha256_unchanged"] is not True:
            entry["status"] = "artifact_changed_during_probe"
            rc = 2
        else:
            entry["status"] = "ok"
            entry["meets_frozen_line"] = bool(entry["eaters_above_floor"] >= FROZEN_LINE)
        results.append(entry)

    payload = {"format": "taiji-a30-eater-p-boundary-floor-v1", "results": results, "rc": rc}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_report:
        out = Path(args.out_report)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

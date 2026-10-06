"""§110 那条**字级**判读线的仓内判读器：把"从不发 LF 的拖写代"那群的 `r` 落成甲／乙／丙。

来历（2026-10-03，PLAN-A-30 第一百一十次停靠）：§106 那格原本用"峰值那一步的字节＝上一步的字节"当
selector，而中文一个字三字节 ⇒ 相邻字节永远不相等，那个占比是多字节文本的**结构下界**不是事实
（§109 已把 §106 就地记为"不判·selector 无效"）。v37 把仪器本来就在算的**字级** `in_run`
接到峰值步上，这一格问的是同一件事：**最接近停的那一格正处在同字重复段内吗**。

判读线在数落地之前冻死（`甲 ≥0.50`／`乙 <0.20`／`丙` 其余），这里只认件里已有的读数：
`peak_run_summary_v37` 的 `eaters_never_lf_peak_in_run_{count,share}` 是占比的唯一住处，
本器不重算第二条式子，只做**一致性核对**（`count/n` 与 `share` 在四位舍入内必须对得上）。
缺列 ⇒ `rc=2` 响亮失败，**不许**被读成 `r=0`（那是仪器缺列，不是"环无关"）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: §110 冻的三条线（互斥且穷尽；改动即换题，不在数落地后动）。
JIA_LINE = 0.50
YI_LINE = 0.20

#: §110 前置①：该群分母的下界（小分母只报数不判，§五十一 那条规矩）。
MIN_DENOMINATOR = 100

#: selector 住在件里哪一处（唯一住处）。§106 那台 v36 的字节相邻列**故意不认**——它已被判为无效 selector。
SUMMARY_KEY = "peak_run_summary_v37"
GROUP = "eaters_never_lf"


def verdict_for(r: float | None) -> str:
    """§110 的三分支：`甲 ≥0.50`、`乙 <0.20`、其余 `丙`。`None` 不是零，是不可判。"""

    if r is None:
        return "not_judged_no_reading"
    if r >= JIA_LINE:
        return "甲"
    if r < YI_LINE:
        return "乙"
    return "丙"


def preconditions(report: dict[str, Any], expect_items_sha: str | None) -> dict[str, Any]:
    """§110 前置①②③④逐条独立读数，任何一条不成立 ⇒ 整格不可判。"""

    summary = report.get(SUMMARY_KEY) or {}
    guard = report.get("instrument_guard") or {}
    terminal = report.get("terminal_decision_summary_v27") or {}
    n = summary.get(f"{GROUP}_n")
    count = summary.get(f"{GROUP}_peak_in_run_count")
    share = summary.get(f"{GROUP}_peak_in_run_share")
    items_sha = report.get("items_sha256")
    checks: dict[str, Any] = {
        "column_present": bool(summary),
        "group_n_at_least_100": (n >= MIN_DENOMINATOR) if isinstance(n, int) else None,
        "self_check_zero": summary.get("incoherent_zero_run_but_peak_in_run") == 0,
        "pairing_ok": terminal.get("pairing_ok") is True,
        "terminal_rank_not_one_count_zero": terminal.get("terminal_rank_not_one_count") == 0,
        "argmax_mismatch_steps_zero": guard.get("total_argmax_mismatch_steps") == 0,
        "items_sha_declared_and_matches": (
            None if expect_items_sha is None else items_sha == expect_items_sha
        ),
        "share_coherent_with_count_over_n": (
            None
            if not (
                isinstance(n, int) and isinstance(count, int) and isinstance(share, float) and n
            )
            else abs(round(count / n, 4) - share) <= 1e-9
        ),
    }
    failed = [name for name, ok in checks.items() if ok is not True]
    return {
        "checks": checks,
        "failed": failed,
        "ok": not failed,
        "n": n,
        "count": count,
        "share": share,
    }


def summarize(report: dict[str, Any], expect_items_sha: str | None = None) -> dict[str, Any]:
    summary = report.get(SUMMARY_KEY) or {}
    pre = preconditions(report, expect_items_sha)
    r = summary.get(f"{GROUP}_peak_in_run_share")
    judged = pre["ok"] and r is not None
    return {
        "format": report.get("format"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "items_sha256": report.get("items_sha256"),
        "product_window_steps": report.get("product_window_steps"),
        "selector": f"{GROUP}_peak_in_run_share",
        "r": r if judged else None,
        "r_reported_in_file": r,
        "verdict": verdict_for(r) if judged else "not_judged_preconditions",
        "lines": {"甲": f">={JIA_LINE}", "乙": f"<{YI_LINE}", "丙": f"[{YI_LINE},{JIA_LINE})"},
        "preconditions": pre,
        "distribution": {
            key: summary[key]
            for key in sorted(summary)
            if key.startswith(("stoppers_", "eaters_with_lf_", "eaters_never_lf_", "incoherent_"))
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="append", required=True, help="v37 探针件路径（可重复）")
    parser.add_argument(
        "--expect-items-sha",
        default=None,
        help="§110 前置③：期望的题面指纹（不给就按'未声明'判为不可判）",
    )
    args = parser.parse_args(argv)
    worst = 0
    for path in args.report:
        payload = summarize(
            json.loads(Path(path).read_text(encoding="utf-8")), args.expect_items_sha
        )
        payload["report"] = path
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if not payload["verdict"].startswith(("甲", "乙", "丙")):
            worst = 2
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""§110 判读器的守卫：**线要钉住、缺列要响亮失败、四条前置每条都得能单独把整格否掉**。

这一格没有已知-good 锚点（数还没落地），所以这里的义务是**结构性的**：
`r` 不许被伪造成 0、分支互斥穷尽、前置一条不成立就不发表（`guard-must-be-able-to-fail`）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from judge_taiji_a30_peak_repeat_run import (  # noqa: E402
    GROUP,
    JIA_LINE,
    MIN_DENOMINATOR,
    SUMMARY_KEY,
    YI_LINE,
    main,
    preconditions,
    summarize,
    verdict_for,
)


def _report(
    r: float | None,
    n: int = 236,
    *,
    self_check: int = 0,
    pairing: bool = True,
    rank_not_one: int = 0,
    mismatch: int = 0,
    sha: str | None = "0541a3f4568a9c5b",
    count: int | None = None,
    share: float | None = None,
) -> dict[str, Any]:
    """按件里真实的嵌套路径造一份"前置全绿"的读数；被破坏的那一条由参数控制。

    `share` 缺省时由 `count/n` 现推（四位舍入）——件里那三个数默认必须自洽；
    要造"自相矛盾"的那一格就显式传 `share`（前置④的一致性核对就是抓这个）。
    """

    numerator = count if count is not None else round(r * n) if r is not None else 0
    derived = round(numerator / n, 4) if n else None
    summary: dict[str, Any] = {
        "incoherent_zero_run_but_peak_in_run": self_check,
        "stoppers_n": 41,
        "stoppers_peak_in_run_count": 3,
        "stoppers_peak_in_run_share": round(3 / 41, 4),
        "eaters_with_lf_n": 11,
        "eaters_with_lf_peak_in_run_count": 2,
        "eaters_with_lf_peak_in_run_share": round(2 / 11, 4),
        f"{GROUP}_n": n,
        f"{GROUP}_peak_in_run_count": numerator,
        f"{GROUP}_peak_in_run_share": derived if share is None else share,
    }
    if r is None:
        summary[f"{GROUP}_peak_in_run_share"] = None
    payload: dict[str, Any] = {
        "format": "taiji-a30-stop-failure-v37",
        "checkpoint_sha256": "ca2628077b21bc4c",
        "product_window_steps": None,
        "peak_run_summary_v37": summary,
        "terminal_decision_summary_v27": {
            "pairing_ok": pairing,
            "terminal_rank_not_one_count": rank_not_one,
        },
        "instrument_guard": {"total_argmax_mismatch_steps": mismatch},
    }
    if sha is not None:
        payload["items_sha256"] = sha
    return payload


def test_the_lines_are_the_frozen_ones() -> None:
    assert (JIA_LINE, YI_LINE, MIN_DENOMINATOR) == (0.50, 0.20, 100)


def test_the_three_branches_partition_the_unit_interval_and_are_exclusive() -> None:
    seen: dict[str, list[float]] = {"甲": [], "乙": [], "丙": []}
    steps = [i / 100.0 for i in range(101)]
    for r in steps:
        label = verdict_for(r)
        assert label in seen, f"{r} 落到三分支之外：{label}"
        seen[label].append(r)
        assert label == ("甲" if r >= 0.50 else "乙" if r < 0.20 else "丙")
    assert min(seen["乙"]) == 0.0 and max(seen["乙"]) == 0.19
    assert min(seen["丙"]) == 0.20 and max(seen["丙"]) == 0.49
    assert min(seen["甲"]) == 0.50 and max(seen["甲"]) == 1.0


def test_boundary_values_follow_the_frozen_inequalities() -> None:
    assert verdict_for(0.50) == "甲" and verdict_for(0.4999) == "丙"
    assert verdict_for(0.20) == "丙" and verdict_for(0.1999) == "乙"


def test_missing_column_is_not_read_as_zero() -> None:
    """v36 那批件没有字级列 ⇒ 只能"不发表"，不能被读成 r=0（那会被当成"乙＝环不是主犯"）。"""

    legacy = {
        "format": "taiji-a30-stop-failure-v36",
        "peak_step_summary_v36": {"eaters_never_lf_same_byte_at_peak_share": 0.0135},
    }
    payload = summarize(legacy, "0541a3f4568a9c5b")
    assert payload["r"] is None and payload["r_reported_in_file"] is None
    assert payload["verdict"] == "not_judged_preconditions", payload
    assert "column_present" in payload["preconditions"]["failed"]
    assert payload["distribution"] == {}


def test_a_clean_file_is_judged_and_exits_zero(tmp_path: Path) -> None:
    good = _report(0.62)
    payload = summarize(good, "0541a3f4568a9c5b")
    assert payload["r"] == 0.6186 and payload["verdict"] == "甲"  # 146/236，件里那三位小数就是它
    assert payload["preconditions"]["ok"] is True and payload["preconditions"]["failed"] == []
    path = tmp_path / "v37.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert main(["--report", str(path), "--expect-items-sha", "0541a3f4568a9c5b"]) == 0
    assert "eaters_never_lf_peak_in_run_share" in json.dumps(payload["distribution"])


def test_incoherent_self_check_kills_the_cell_even_at_a_decisive_r(tmp_path: Path) -> None:
    broken = _report(0.90, self_check=3)
    assert "self_check_zero" in preconditions(broken, "0541a3f4568a9c5b")["failed"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    assert main(["--report", str(path), "--expect-items-sha", "0541a3f4568a9c5b"]) == 2


def test_small_denominator_is_reported_but_not_judged() -> None:
    payload = summarize(_report(0.90, n=24), "0541a3f4568a9c5b")
    assert payload["verdict"] == "not_judged_preconditions"
    assert payload["r"] is None and payload["r_reported_in_file"] == 0.9167  # 22/24，如实报但不判
    assert "group_n_at_least_100" in payload["preconditions"]["failed"]


def test_each_other_precondition_can_fail_on_its_own() -> None:
    cases = {
        "pairing_ok": _report(0.62, pairing=False),
        "terminal_rank_not_one_count_zero": _report(0.62, rank_not_one=1),
        "argmax_mismatch_steps_zero": _report(0.62, mismatch=7),
    }
    for name, report in cases.items():
        checks = preconditions(report, "0541a3f4568a9c5b")
        assert name in checks["failed"], (name, checks["failed"])
        assert checks["ok"] is False


def test_items_sha_is_required_and_a_mismatch_is_loud() -> None:
    report = _report(0.62)
    assert "items_sha_declared_and_matches" in preconditions(report, None)["failed"]
    assert "items_sha_declared_and_matches" in preconditions(report, "deadbeef")["failed"]
    assert preconditions(report, "0541a3f4568a9c5b")["ok"] is True
    missing = dict(report)
    missing.pop("items_sha256")
    assert "items_sha_declared_and_matches" in preconditions(missing, "0541a3f4568a9c5b")["failed"]


def test_share_must_be_reproducible_from_count_over_n() -> None:
    """前置④的一致性核对：件里那三个数不能互相矛盾（列接错/手改都会在这里红）。"""

    report = _report(0.62, count=5, share=0.62)  # 5/236 绝不是 0.62
    checks = preconditions(report, "0541a3f4568a9c5b")
    assert "share_coherent_with_count_over_n" in checks["failed"]
    assert summarize(report, "0541a3f4568a9c5b")["verdict"] == "not_judged_preconditions"
    coherent = _report(0.62)
    assert (
        "share_coherent_with_count_over_n"
        not in preconditions(coherent, "0541a3f4568a9c5b")["failed"]
    )


def test_the_deprecated_byte_adjacency_selector_is_not_wired_in() -> None:
    """§106 那条字节相邻的列**不许**被这台判读器当 selector（§109 已判它无效）。"""

    source = (
        PROJECT_ROOT / "scripts" / "training" / "judge_taiji_a30_peak_repeat_run.py"
    ).read_text(encoding="utf-8")
    assert "same_byte" not in source
    assert "peak_step_summary_v36" not in source
    assert SUMMARY_KEY == "peak_run_summary_v37"

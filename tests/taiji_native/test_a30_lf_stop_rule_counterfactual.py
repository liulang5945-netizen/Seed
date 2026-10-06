"""§91 反事实计算器的守卫：这条式子必须**既能开火也能不开火**，且前置不齐时整档不发表。

写它的理由（都是本仓踩过的）：
① 判读器只验 fail-closed 分支等于没验——所以这里两条路（开火／不开火、rc=0／rc=2）都实测；
② "反事实自停率 − 实测自停率"最容易犯的错是**把已经真停的代重复计成新停**，
   所以专门钉"真停的那一代在两个计数里各算一次、净增为 0"；
③ `per_step` 只存前 6 次 ⇒ 这一格必须**低报**，所以"第 7 次以后的机会"要能被计数列抓到；
④ 代价面在对照情形下可能整体为空 ⇒ 空样本不许抛异常（按 §91 记 `n: 0`）。
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

from counterfactual_taiji_a30_lf_stop_rule import (  # noqa: E402
    R_GRID,
    R_GRID_REASON,
    counterfactual_stop,
    iter_generations,
    main,
    precondition_failures,
    run,
    summarize,
)


def _gen(
    margins: list[dict[str, Any]],
    generation_steps: int,
    stopped: bool = False,
    lf_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "generation_steps": generation_steps,
        "ate_full_budget": not stopped,
        "terminal_decision": {"step": generation_steps} if stopped else None,
        "lf_margins_v34": {
            "lf_steps": lf_steps if lf_steps is not None else len(margins),
            "per_step": margins,
        },
    }


def _row(step: int, ratio: float | None) -> dict[str, Any]:
    return {
        "step": step,
        "ratio_best_over_boundary": ratio,
        "boundary_rank_in_legal": 2,
        "p_boundary": 0.1,
    }


def test_the_rule_fires_only_when_a_post_lf_margin_is_within_r() -> None:
    """能为假：比值全都 >R ⇒ 该代仍算未停；有一条 ≤R ⇒ 停在**最早**那一步。"""

    generation = _gen([_row(50, 3.0), _row(200, 1.1), _row(120, 1.15)], 256)
    assert counterfactual_stop(generation, 1.05) is None
    assert counterfactual_stop(generation, 1.20) == 120
    assert counterfactual_stop(generation, 2.00) == 120
    #: 比值缺失（None）不能被当成"≤R"——它是不明，不是零。
    assert counterfactual_stop(_gen([_row(70, None)], 256), 2.00) is None


def test_a_real_stop_is_counted_in_both_sides_so_net_gain_stays_zero() -> None:
    """已真停的那一代：终止步恰等于 `generation_steps` ⇒ 反事实也停在同一步，净增必须为 0。"""

    stopper = _gen([_row(244, 1.0)], 244, stopped=True)
    summary = summarize([stopper], 1.05)
    assert summary["actual_self_stop_count"] == 1
    assert summary["counterfactual_stop_count"] == 1
    assert summary["net_new_stops"] == 0
    assert summary["counterfactual_stops_that_are_already_real"] == 1
    assert summary["delta_pp"] == 0.0
    assert summary["retained_share_of_cut"]["median"] == 1.0


def test_a_drag_writing_generation_that_would_have_stopped_adds_to_the_gap() -> None:
    """开火分支：拖写代在第 200/256 步本可停 ⇒ Δpp 为正、保留比例 0.7812、示例带题面身份。"""

    generation = _gen([_row(200, 1.1)], 256)
    generation["item_id"] = "V007"
    summary = summarize([generation, _gen([_row(30, 9.0)], 256)], 1.20)
    assert summary["denominator_generations"] == 2
    assert summary["counterfactual_stop_count"] == 1
    assert summary["net_new_stops"] == 1
    assert summary["delta_pp"] == 50.0
    assert summary["retained_share_of_cut"]["median"] == 0.7812
    assert summary["examples"][0]["id"] == "V007"


def test_the_seventh_onward_chance_is_counted_as_a_low_report_not_a_zero() -> None:
    """`per_step` 只存前 6 次：`lf_steps=9` 而六点全超阈值 ⇒ 不算开火，但低报来源必须被计数抓到。"""

    generation = _gen([_row(10 * i, 5.0) for i in range(1, 7)], 256, lf_steps=9)
    summary = summarize([generation], 2.00)
    assert summary["counterfactual_stop_count"] == 0
    assert summary["truncated_generations_lf_steps_gt_6"] == 1


def test_the_cost_face_reports_an_empty_sample_instead_of_raising() -> None:
    summary = summarize([_gen([_row(10, 8.0)], 256)], 1.05)
    assert summary["retained_share_of_cut"] == {"n": 0, "median": None, "p10": None}
    assert summary["counterfactual_stop_rate"] == 0.0


def _report(generations: list[dict[str, Any]], usable: bool = True) -> dict[str, Any]:
    return {
        "format": "taiji-a30-stop-failure-v34",
        "checkpoint_sha256": "deadbeef",
        "product_window_steps": None,
        "per_item": [{"id": "V001", "argmax_mismatch_steps": 0, "endstep_probe_v22": generations}],
        "lf_margin_summary_v34": {
            "eaters_with_lf_next": 25 if usable else 3,
            "eaters_dynamic_range": {"usable": usable, "distinct": 9, "n_values": 25},
        },
        "terminal_decision_summary_v27": {"pairing_ok": True, "terminal_rank_not_one_count": 0},
    }


def test_the_precondition_gate_refuses_a_degenerate_ruler(tmp_path: Path) -> None:
    """前置不齐 ⇒ `published` False **且 rc=2**；齐 ⇒ rc=0（fail-closed 的选取要两条路都跑过）。"""

    generations = [_gen([_row(200, 1.1)], 256)]
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps(_report(generations), ensure_ascii=False), encoding="utf-8")
    bad.write_text(
        json.dumps(_report(generations, usable=False), ensure_ascii=False), encoding="utf-8"
    )
    assert run(good)["published"] is True
    assert main(["--report", str(good)]) == 0
    payload = run(bad)
    assert payload["published"] is False
    assert "eaters_ruler_not_usable" in payload["precondition_failures"]
    assert payload["grid"] == []
    assert main(["--report", str(bad)]) == 2
    #: 尺子退化时**不许**给出任何阈值上的数字——整档不发表是这条仪器的职责。
    assert precondition_failures({"format": "taiji-a30-stop-failure-v33"}) == [
        "format_is_not_v34",
        "lf_margin_summary_v34_absent",
    ]


def test_a_missing_margin_readout_refuses_to_publish(tmp_path: Path) -> None:
    """§91 第二条前置：`per_step` 里比值或步序为空 ⇒ 不发表（缺读数不能被当成"没开火"）。"""

    broken = _gen([_row(200, None)], 256)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(_report([broken]), ensure_ascii=False), encoding="utf-8")
    payload = run(path)
    assert payload["published"] is False
    assert "per_step_has_missing_readout" in payload["precondition_failures"]
    assert payload["grid"] == []
    assert main(["--report", str(path)]) == 2


def test_the_grid_is_the_one_frozen_before_the_numbers(tmp_path: Path) -> None:
    """网格必须先于数写死：四个阈值、顺序与理由都在案，且件里逐条报出（不许四选一）。"""

    assert R_GRID == (1.05, 1.20, 1.50, 2.00)
    assert len(R_GRID_REASON) == 4
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(_report([_gen([_row(200, 1.1)], 256)]), ensure_ascii=False), encoding="utf-8"
    )
    payload = run(good)
    assert [item["R"] for item in payload["grid"]] == list(R_GRID)
    #: iter_generations 把题面身份挂到逐代上（逐代字典本身没有 `item_id`）。
    assert iter_generations(_report([_gen([], 256)]))[0]["item_id"] == "V001"
